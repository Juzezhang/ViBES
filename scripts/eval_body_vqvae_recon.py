"""
Evaluate body VQ-VAE reconstruction metrics (MPJPE, PA-MPJPE, Accel Error)
for two checkpoints on AMASS and BEAT2 test sets.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m scripts.eval_body_vqvae_recon
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from multimodal_tokenizers.config import get_obj_from_str
from multimodal_tokenizers.utils.rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)

# ─── Constants ──────────────────────────────────────────────────────────────
GENMO_LOWER_JOINT_INDICES = [0, 1, 3, 4, 6, 7, 9, 10]
GENMO_UPPER_JOINT_INDICES = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

AMASS_ROOT = "/path/to/AMASS"
AMASS_GENMO_DIR = os.path.join(AMASS_ROOT, "amass_genmo_25")
AMASS_TEST_SPLIT = os.path.join(AMASS_ROOT, "test.txt")

BEAT2_ROOT = "/path/to/BEAT2/beat_english_v2.0.0"
BEAT2_GENMO_DIR = os.path.join(BEAT2_ROOT, "beat2_genmo_25", "smplxflame_25")
BEAT2_SPLIT_CSV = os.path.join(BEAT2_ROOT, "train_test_split.csv")

SMPLX_MODEL_DIR = str(ROOT_DIR / "model_files" / "smplx_models")

MAX_SAMPLES_PER_DATASET = 50
SEQ_LEN = 128  # match config pose_length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Helpers ────────────────────────────────────────────────────────────────

def load_test_file_ids_amass():
    with open(AMASS_TEST_SPLIT) as f:
        return [line.strip() for line in f if line.strip()]


def load_test_file_ids_beat2():
    ids = []
    with open(BEAT2_SPLIT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "test":
                ids.append(row["id"])
    return ids


def load_genmo_samples(dataset_name, max_samples=MAX_SAMPLES_PER_DATASET, min_len=SEQ_LEN):
    """Load GENMO 145D motion vectors from test split."""
    if dataset_name == "AMASS":
        file_ids = load_test_file_ids_amass()
        base_dir = AMASS_GENMO_DIR
        get_path = lambda fid: os.path.join(base_dir, f"{fid}.npz")
    elif dataset_name == "BEAT2":
        file_ids = load_test_file_ids_beat2()
        base_dir = BEAT2_GENMO_DIR
        get_path = lambda fid: os.path.join(base_dir, f"{fid}.npz")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    samples = []
    for fid in file_ids:
        if len(samples) >= max_samples:
            break
        path = get_path(fid)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        mv = data["motion_vector"].astype(np.float32)
        if mv.shape[0] < min_len:
            continue
        # Take first SEQ_LEN frames
        samples.append(torch.from_numpy(mv[:min_len]))
    return samples


def genmo145_to_joints(motion_145, smplx_model):
    """
    Convert GENMO 145D to 22 SMPL-X joint positions via FK.
    motion_145: (T, 145) tensor on DEVICE
    Returns: (T, 22, 3) joint positions
    """
    T = motion_145.shape[0]
    body_r6d = motion_145[:, :126].reshape(T, 21, 6)
    betas = motion_145[:, 126:136]
    global_r6d = motion_145[:, 136:142]
    local_vel = motion_145[:, 142:145]

    # Convert rotations
    body_R = rotation_6d_to_matrix(body_r6d.reshape(-1, 6)).reshape(T, 21, 3, 3)
    body_aa = matrix_to_axis_angle(body_R.reshape(-1, 3, 3)).reshape(T, 63)
    global_R = rotation_6d_to_matrix(global_r6d)
    global_aa = matrix_to_axis_angle(global_R)

    # Compute translation from local velocity
    world_vel = torch.einsum("tij,tj->ti", global_R, local_vel)
    transl = torch.cumsum(world_vel, dim=0)

    betas_single = betas[0:1].expand(T, -1)

    output = smplx_model(
        body_pose=body_aa,
        global_orient=global_aa,
        transl=transl,
        betas=betas_single,
    )
    return output.joints[:, :22, :]


def procrustes_align(predicted, target):
    """Align predicted to target using Procrustes (SVD-based). numpy arrays (J,3)."""
    mu_pred = predicted.mean(axis=0, keepdims=True)
    mu_tgt = target.mean(axis=0, keepdims=True)
    pred_c = predicted - mu_pred
    tgt_c = target - mu_tgt
    H = pred_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    scale = np.sum(S) / (np.sum(pred_c ** 2) + 1e-10)
    aligned = scale * (pred_c @ R) + mu_tgt
    return aligned


def compute_metrics(gt_joints, pred_joints):
    """
    gt_joints, pred_joints: (T, 22, 3) numpy arrays in meters.
    Returns dict with MPJPE (mm), PA-MPJPE (mm), Accel Error (mm/s^2).
    MPJPE is computed root-relative (subtract pelvis/root joint) to avoid
    translation drift dominating the metric.
    """
    T, J, _ = gt_joints.shape

    # Root-relative joints (subtract root = joint 0)
    gt_rel = gt_joints - gt_joints[:, :1, :]
    pred_rel = pred_joints - pred_joints[:, :1, :]

    # MPJPE (root-relative)
    mpjpe_per_frame = np.sqrt(((gt_rel - pred_rel) ** 2).sum(axis=-1)).mean(axis=-1)  # (T,)
    mpjpe = mpjpe_per_frame.mean() * 1000  # to mm

    # PA-MPJPE (on global joints, aligned per-frame)
    pa_errors = []
    for t in range(T):
        aligned = procrustes_align(pred_joints[t], gt_joints[t])
        err = np.sqrt(((gt_joints[t] - aligned) ** 2).sum(axis=-1)).mean()
        pa_errors.append(err)
    pa_mpjpe = np.mean(pa_errors) * 1000  # to mm

    # Acceleration error (root-relative)
    if T >= 3:
        gt_acc = gt_rel[2:] + gt_rel[:-2] - 2 * gt_rel[1:-1]
        pred_acc = pred_rel[2:] + pred_rel[:-2] - 2 * pred_rel[1:-1]
        accel_err = np.sqrt(((gt_acc - pred_acc) ** 2).sum(axis=-1)).mean() * 1000  # mm
    else:
        accel_err = 0.0

    return {"MPJPE": mpjpe, "PA-MPJPE": pa_mpjpe, "Accel": accel_err}


# ─── Model builders ────────────────────────────────────────────────────────

def build_smplx_model():
    from utils.genmo.body_model import BodyModelSMPLX
    model = BodyModelSMPLX(
        model_path=SMPLX_MODEL_DIR,
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
    ).to(DEVICE).eval()
    return model


def build_vae(target_cls, params):
    """Instantiate a VAE from config target + params."""
    cls = get_obj_from_str(target_cls)
    model = cls(**params).to(DEVICE).eval()
    return model


def load_state_dict_for_prefix(ckpt_path, prefix):
    """Load checkpoint, extract keys with given prefix, strip prefix."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Extract state_dict
    sd = checkpoint
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model_state" in sd:
            sd = sd["model_state"]
    result = {}
    for k, v in sd.items():
        if k.startswith(prefix + "."):
            result[k[len(prefix) + 1:]] = v
    return result


# ─── Checkpoint A: Hybrid (Normal Upper + Genmo Lower) ─────────────────────

def build_checkpoint_a():
    """Build and load Checkpoint A models (vae_body=lower 61D, vae_upper=78D)."""
    base = ROOT_DIR / "model_files" / "pretrained_cpt" / "VQVAE_0318_NormalUpper_GenmoLower"

    # Lower body VAE (61D input)
    vae_lower = build_vae(
        "multimodal_tokenizers.archs.motiongpt_vq.MotionGPTVQVaeAdapter",
        dict(nfeats=61, quantizer="ema_reset", code_num=256, code_dim=128,
             output_emb_width=128, down_t=2, stride_t=2, width=512, depth=3,
             dilation_growth_rate=3, norm="GN", activation="relu", mu=0.95),
    )
    # Load weights - checkpoint has vae_lower prefix, mapped to vae_body in load_checkpoint
    # but the actual ckpt file stores with vae_body or vae_lower prefix
    ckpt_lower = str(base / "vqvar_genmo_lower_global_last.ckpt")
    sd = load_state_dict_for_prefix(ckpt_lower, "vae_body")
    if not sd:
        sd = load_state_dict_for_prefix(ckpt_lower, "vae_lower")
    vae_lower.load_state_dict(sd, strict=True)
    print(f"  Loaded lower VAE from {ckpt_lower} ({len(sd)} params)")

    # Upper body VAE (78D input)
    vae_upper = build_vae(
        "multimodal_tokenizers.archs.motiongpt_vq.MotionGPTVQVaeAdapter",
        dict(nfeats=78, quantizer="ema_reset", code_num=128, code_dim=128,
             output_emb_width=128, down_t=2, stride_t=2, width=256, depth=3,
             dilation_growth_rate=3, norm=None, activation="relu", mu=0.95),
    )
    ckpt_upper = str(base / "vqvae_normal_upper_last.ckpt")
    sd = load_state_dict_for_prefix(ckpt_upper, "vae_upper")
    vae_upper.load_state_dict(sd, strict=True)
    print(f"  Loaded upper VAE from {ckpt_upper} ({len(sd)} params)")

    return vae_lower, vae_upper


def reconstruct_checkpoint_a(motion_145, vae_lower, vae_upper):
    """
    Reconstruct 145D GENMO through hybrid upper+lower VAEs.
    motion_145: (T, 145) tensor on device
    Returns: (T, 145) reconstructed
    """
    T = motion_145.shape[0]
    body_r6d = motion_145[:, :126].reshape(T, 21, 6)
    betas = motion_145[:, 126:136]
    global_r6d = motion_145[:, 136:142]
    local_vel = motion_145[:, 142:145]

    # Split into lower (61D) and upper (78D)
    lower_joints = body_r6d[:, GENMO_LOWER_JOINT_INDICES].reshape(T, -1)  # (T, 48)
    upper_joints = body_r6d[:, GENMO_UPPER_JOINT_INDICES].reshape(T, -1)  # (T, 78)

    # Lower: 48D joints + 10D betas + 3D local_vel = 61D (no global orient)
    lower_input = torch.cat([lower_joints, betas, local_vel], dim=-1)  # (T, 61)

    # Encode/decode through VAEs
    with torch.no_grad():
        lower_out = vae_lower(lower_input.unsqueeze(0))["rec_pose"].squeeze(0)  # (T, 61)
        upper_out = vae_upper(upper_joints.unsqueeze(0))["rec_pose"].squeeze(0)  # (T, 78)

    # Merge back to 145D
    rec_lower_joints = lower_out[:, :48].reshape(T, 8, 6)
    rec_betas = lower_out[:, 48:58]
    rec_local_vel = lower_out[:, 58:61]
    rec_upper_joints = upper_out.reshape(T, 13, 6)

    rec_body_r6d = torch.zeros(T, 21, 6, device=motion_145.device, dtype=motion_145.dtype)
    for i, idx in enumerate(GENMO_LOWER_JOINT_INDICES):
        rec_body_r6d[:, idx] = rec_lower_joints[:, i]
    for i, idx in enumerate(GENMO_UPPER_JOINT_INDICES):
        rec_body_r6d[:, idx] = rec_upper_joints[:, i]

    # Use GT global_orient (not reconstructed by either VAE)
    return torch.cat([
        rec_body_r6d.reshape(T, 126), rec_betas, global_r6d, rec_local_vel
    ], dim=-1)


# ─── Checkpoint B: GenmoFull (135D, no betas) ──────────────────────────────

def build_checkpoint_b():
    """Build and load Checkpoint B model (vae_body=135D)."""
    base = ROOT_DIR / "model_files" / "pretrained_cpt" / "VQVAE_0320_GenmoFull"
    vae_body = build_vae(
        "multimodal_tokenizers.archs.motiongpt_vq.MotionGPTVQVaeAdapter",
        dict(nfeats=135, quantizer="ema_reset", code_num=256, code_dim=256,
             output_emb_width=256, down_t=2, stride_t=2, width=512, depth=3,
             dilation_growth_rate=3, norm=None, activation="relu", mu=0.99),
    )
    ckpt_path = str(base / "last.ckpt")
    sd = load_state_dict_for_prefix(ckpt_path, "vae_body")
    vae_body.load_state_dict(sd, strict=True)
    print(f"  Loaded body VAE from {ckpt_path} ({len(sd)} params)")
    return vae_body


def reconstruct_checkpoint_b(motion_145, vae_body):
    """
    Reconstruct 145D GENMO through full-body VAE (135D, no betas).
    motion_145: (T, 145) tensor on device
    Returns: (T, 145) reconstructed
    """
    T = motion_145.shape[0]
    # 135D = body_r6d(126) + global_r6d(6) + local_vel(3), skip betas
    input_135 = torch.cat([motion_145[:, :126], motion_145[:, 136:145]], dim=-1)

    with torch.no_grad():
        rec_135 = vae_body(input_135.unsqueeze(0))["rec_pose"].squeeze(0)  # (T, 135)

    # Re-insert GT betas at position 126
    gt_betas = motion_145[:, 126:136]
    return torch.cat([rec_135[:, :126], gt_betas, rec_135[:, 126:]], dim=-1)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Body VQ-VAE Reconstruction Evaluation")
    print("=" * 70)

    # Build SMPL-X model for FK
    print("\nBuilding SMPL-X model...")
    smplx_model = build_smplx_model()

    # Build VAEs
    print("\nLoading Checkpoint A (Hybrid: NormalUpper + GenmoLower)...")
    vae_lower_a, vae_upper_a = build_checkpoint_a()

    print("\nLoading Checkpoint B (GenmoFull 135D)...")
    vae_body_b = build_checkpoint_b()

    # Results accumulator
    all_results = {}

    for dataset_name in ["AMASS", "BEAT2"]:
        print(f"\n{'─' * 50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─' * 50}")

        samples = load_genmo_samples(dataset_name, max_samples=MAX_SAMPLES_PER_DATASET)
        print(f"Loaded {len(samples)} test samples (seq_len={SEQ_LEN})")

        if len(samples) == 0:
            print(f"  No samples found, skipping.")
            continue

        metrics_a = {"MPJPE": [], "PA-MPJPE": [], "Accel": []}
        metrics_b = {"MPJPE": [], "PA-MPJPE": [], "Accel": []}

        for i, gt_motion in enumerate(tqdm(samples, desc=f"  Evaluating {dataset_name}")):
            gt_motion = gt_motion.to(DEVICE)

            # GT joints
            with torch.no_grad():
                gt_joints = genmo145_to_joints(gt_motion, smplx_model).cpu().numpy()

            # Checkpoint A reconstruction
            rec_a = reconstruct_checkpoint_a(gt_motion, vae_lower_a, vae_upper_a)
            with torch.no_grad():
                rec_joints_a = genmo145_to_joints(rec_a, smplx_model).cpu().numpy()
            m_a = compute_metrics(gt_joints, rec_joints_a)
            for k in metrics_a:
                metrics_a[k].append(m_a[k])

            # Checkpoint B reconstruction
            rec_b = reconstruct_checkpoint_b(gt_motion, vae_body_b)
            with torch.no_grad():
                rec_joints_b = genmo145_to_joints(rec_b, smplx_model).cpu().numpy()
            m_b = compute_metrics(gt_joints, rec_joints_b)
            for k in metrics_b:
                metrics_b[k].append(m_b[k])

        all_results[dataset_name] = {
            "A": {k: np.mean(v) for k, v in metrics_a.items()},
            "B": {k: np.mean(v) for k, v in metrics_b.items()},
            "n_samples": len(samples),
        }

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Checkpoint':<45} {'MPJPE':>8} {'PA-MPJPE':>10} {'Accel':>8}")
    print(f"{'':12} {'':45} {'(mm)':>8} {'(mm)':>10} {'(mm)':>8}")
    print("-" * 95)

    for ds in ["AMASS", "BEAT2"]:
        if ds not in all_results:
            continue
        r = all_results[ds]
        n = r["n_samples"]
        ma = r["A"]
        mb = r["B"]
        print(f"{ds:<12} {'A: Hybrid (NormalUpper+GenmoLower)':<45} {ma['MPJPE']:8.2f} {ma['PA-MPJPE']:10.2f} {ma['Accel']:8.2f}")
        print(f"{'(n=' + str(n) + ')':<12} {'B: GenmoFull (135D)':<45} {mb['MPJPE']:8.2f} {mb['PA-MPJPE']:10.2f} {mb['Accel']:8.2f}")
        print("-" * 95)

    # Combined averages
    if len(all_results) == 2:
        print(f"\n{'Combined (weighted by n_samples)':}")
        total_n = sum(r["n_samples"] for r in all_results.values())
        for ckpt in ["A", "B"]:
            avg = {}
            for metric in ["MPJPE", "PA-MPJPE", "Accel"]:
                avg[metric] = sum(
                    r[ckpt][metric] * r["n_samples"] for r in all_results.values()
                ) / total_n
            label = "A: Hybrid (NormalUpper+GenmoLower)" if ckpt == "A" else "B: GenmoFull (135D)"
            print(f"{'Combined':<12} {label:<45} {avg['MPJPE']:8.2f} {avg['PA-MPJPE']:10.2f} {avg['Accel']:8.2f}")


if __name__ == "__main__":
    main()
