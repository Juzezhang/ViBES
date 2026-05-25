"""
Preprocess SMPL-X data to GENMO 145D format and generate motion tokens
using two VAE tokenizers (fullbody_genmo and lower_genmo).

Modes:
  preprocess     - Convert SMPL-X rotation data to GENMO 145D motion vectors.
  fullbody_genmo - Tokenize GENMO data through the GenmoFull VAE (135D, no betas).
  lower_genmo    - Tokenize GENMO data through the hybrid lower VAE (61D).
  verify         - Check token shapes and downsample ratios.
  visualize      - Decode tokens back through VAE and inspect.

Usage:
    python -m scripts.get_genmo_motion_code --mode preprocess
    python -m scripts.get_genmo_motion_code --mode fullbody_genmo
    python -m scripts.get_genmo_motion_code --mode lower_genmo
    python -m scripts.get_genmo_motion_code --mode verify
    python -m scripts.get_genmo_motion_code --mode fullbody_genmo --datasets AMASS_talking BEAT2 --max_samples 10
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from multimodal_tokenizers.archs.motiongpt_vq import MotionGPTVQVaeAdapter
from multimodal_tokenizers.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)


# =============================================================================
# Constants
# =============================================================================

# SMPL-X 22-joint ordering (pelvis + 21 body joints):
#   0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee,
#   6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3, 10: L_Foot, 11: R_Foot,
#   12: Neck, 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shoulder,
#   17: R_Shoulder, 18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist

# Lower body: 9 joints (pelvis + hips + knees + ankles + feet)
LOWER_JOINT_INDICES_22 = [0, 1, 2, 4, 5, 7, 8, 10, 11]

# Dataset paths
DATASET_PATHS = {
    "AMASS_talking": {
        "source_rotation": "/path/to/AMASS_talking/amass_data_align_25_audios_rotation",
        "genmo_output": "/path/to/AMASS_talking/amass_genmo_25",
        "foot_contacts": "/path/to/AMASS_talking/foot_contacts_25",
        "token_output": "/path/to/AMASS_talking/TOKENS_AGENT_25",
        "subdirs": [None],  # No subdirectory structure
    },
    "BEAT2": {
        "source_rotation_base": "/path/to/BEAT2/beat_english_v2.0.0",
        "genmo_output_base": "/path/to/BEAT2/beat_english_v2.0.0/beat2_genmo_25",
        "foot_contacts": "/path/to/BEAT2/beat_english_v2.0.0/foot_contacts_25",
        "token_output": "/path/to/BEAT2/beat_english_v2.0.0/TOKENS_AGENT_25",
        "subdirs": ["smplxflame_25", "smplxflame_25_mirror"],
    },
}

# VAE checkpoint paths (relative to ROOT_DIR)
FULLBODY_GENMO_CKPT = ROOT_DIR / "model_files" / "pretrained_cpt" / "VQVAE_0320_GenmoFull" / "last.ckpt"
LOWER_GENMO_CKPT = (
    ROOT_DIR / "model_files" / "pretrained_cpt"
    / "VQVAE_0318_NormalUpper_GenmoLower" / "vqvar_genmo_lower_global_last.ckpt"
)


# =============================================================================
# SMPL-X to GENMO 145D Conversion
# =============================================================================

def convert_smplx_to_genmo(poses, trans, betas):
    """
    Convert SMPL-X rotation data to GENMO 145D motion vector format.

    GENMO 145D layout:
      [0:126]   body_pose_r6d   - 21 body joints x 6D rotation
      [126:136] betas           - 10D shape parameters
      [136:142] global_orient_r6d - root rotation in 6D
      [142:145] local_vel       - translation velocity in root-local frame

    Data is already 25fps and Y-up; no coordinate conversion or resampling.

    Args:
        poses:  (T, 165) SMPL-X pose parameters (axis-angle)
        trans:  (T, 3) world translation
        betas:  (N,) or (T, N) shape parameters

    Returns:
        dict with keys: motion_vector, num_frames, fps, gender,
                        body_pose, global_orient, trans, betas
    """
    T = poses.shape[0]

    global_orient_aa = torch.from_numpy(poses[:, :3].astype(np.float32))       # (T, 3)
    body_pose_aa = torch.from_numpy(poses[:, 3:66].astype(np.float32))         # (T, 63)

    # Body pose: 21 joints, axis-angle -> rotation matrix -> 6D
    body_pose_reshaped = body_pose_aa.reshape(T, 21, 3)
    body_pose_mat = axis_angle_to_matrix(body_pose_reshaped.reshape(-1, 3))    # (T*21, 3, 3)
    body_pose_r6d = matrix_to_rotation_6d(body_pose_mat).reshape(T, 21, 6)    # (T, 21, 6)
    body_pose_r6d_flat = body_pose_r6d.reshape(T, 126)                         # (T, 126)

    # Global orient: axis-angle -> rotation matrix -> 6D
    global_orient_mat = axis_angle_to_matrix(global_orient_aa)                 # (T, 3, 3)
    global_orient_r6d = matrix_to_rotation_6d(global_orient_mat)               # (T, 6)

    # Betas: broadcast to (T, 10)
    if len(betas.shape) == 1:
        betas_10 = np.tile(betas[:10], (T, 1))
    else:
        betas_10 = np.broadcast_to(betas[:, :10], (T, 10)).copy()
    betas_t = torch.from_numpy(betas_10.astype(np.float32))                   # (T, 10)

    # Local velocity: world velocity transformed to root-local frame
    trans_t = torch.from_numpy(trans.astype(np.float32))                       # (T, 3)
    vel = trans_t[1:] - trans_t[:-1]                                           # (T-1, 3)
    R = axis_angle_to_matrix(global_orient_aa)                                 # (T, 3, 3)
    local_vel = torch.bmm(R[:-1].transpose(-1, -2), vel.unsqueeze(-1))         # (T-1, 3, 1)
    local_vel = local_vel.squeeze(-1)                                          # (T-1, 3)
    local_vel = torch.cat([local_vel, torch.zeros(1, 3)], dim=0)               # (T, 3)

    # Assemble 145D motion vector
    motion_vector = torch.cat([
        body_pose_r6d_flat,   # 126
        betas_t,              # 10
        global_orient_r6d,    # 6
        local_vel,            # 3
    ], dim=-1)                # = 145

    return {
        "motion_vector": motion_vector.numpy(),
        "num_frames": T,
        "fps": 25.0,
        "gender": "neutral",
        "body_pose": poses[:, 3:66],            # (T, 63) axis-angle
        "global_orient": poses[:, :3],           # (T, 3)  axis-angle
        "trans": trans,                          # (T, 3)
        "betas": betas_10,                       # (T, 10)
    }


# =============================================================================
# Preprocessing Mode
# =============================================================================

def preprocess_amass_talking(max_samples=None):
    """Convert AMASS_talking SMPL-X rotation npz files to GENMO 145D format."""
    input_dir = Path(DATASET_PATHS["AMASS_talking"]["source_rotation"])
    output_dir = Path(DATASET_PATHS["AMASS_talking"]["genmo_output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    if max_samples is not None:
        npz_files = npz_files[:max_samples]

    stats = {"processed": 0, "skipped_existing": 0, "skipped_error": 0}

    for npz_path in tqdm(npz_files, desc="AMASS_talking preprocess"):
        out_path = output_dir / npz_path.name
        if out_path.exists():
            stats["skipped_existing"] += 1
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            poses = data["poses"]     # (T, 165)
            trans = data["trans"]     # (T, 3)
            betas = data["betas"]     # (16,)

            if poses.shape[0] < 2:
                stats["skipped_error"] += 1
                continue

            result = convert_smplx_to_genmo(poses, trans, betas)
            np.savez_compressed(out_path, **result)
            stats["processed"] += 1
        except Exception as e:
            stats["skipped_error"] += 1
            tqdm.write(f"Error processing {npz_path.name}: {e}")

    print(f"AMASS_talking: {stats['processed']} processed, "
          f"{stats['skipped_existing']} existing, {stats['skipped_error']} errors")
    return stats


def preprocess_beat2(max_samples=None):
    """Convert BEAT2 SMPL-X rotation npz files to GENMO 145D format."""
    base_input = Path(DATASET_PATHS["BEAT2"]["source_rotation_base"])
    base_output = Path(DATASET_PATHS["BEAT2"]["genmo_output_base"])

    total_stats = {"processed": 0, "skipped_existing": 0, "skipped_error": 0}

    for subdir in DATASET_PATHS["BEAT2"]["subdirs"]:
        input_dir = base_input / subdir
        output_dir = base_output / subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            print(f"Skipping {input_dir}: directory not found")
            continue

        npz_files = sorted(input_dir.glob("*.npz"))
        if max_samples is not None:
            npz_files = npz_files[:max_samples]

        for npz_path in tqdm(npz_files, desc=f"BEAT2/{subdir} preprocess"):
            out_path = output_dir / npz_path.name
            if out_path.exists():
                total_stats["skipped_existing"] += 1
                continue

            try:
                data = np.load(npz_path, allow_pickle=True)
                poses = data["poses"]     # (T, 165)
                trans = data["trans"]     # (T, 3)
                betas = data["betas"]     # (300,)

                if poses.shape[0] < 2:
                    total_stats["skipped_error"] += 1
                    continue

                result = convert_smplx_to_genmo(poses, trans, betas)
                np.savez_compressed(out_path, **result)
                total_stats["processed"] += 1
            except Exception as e:
                total_stats["skipped_error"] += 1
                tqdm.write(f"Error processing {npz_path.name}: {e}")

    print(f"BEAT2: {total_stats['processed']} processed, "
          f"{total_stats['skipped_existing']} existing, {total_stats['skipped_error']} errors")
    return total_stats


def run_preprocess(datasets, max_samples=None):
    """Run GENMO preprocessing for specified datasets."""
    print("=" * 60)
    print("Mode: preprocess (SMPL-X -> GENMO 145D)")
    print("=" * 60)

    if "AMASS_talking" in datasets:
        preprocess_amass_talking(max_samples=max_samples)
    if "BEAT2" in datasets:
        preprocess_beat2(max_samples=max_samples)


# =============================================================================
# VAE Loading Utilities
# =============================================================================

def load_vae_from_checkpoint(vae, ckpt_path, key_prefix, device="cuda"):
    """
    Load VAE weights from a Lightning checkpoint, extracting keys by prefix.

    Args:
        vae: MotionGPTVQVaeAdapter instance
        ckpt_path: Path to .ckpt file
        key_prefix: e.g. "vae_body" or "vae_lower"
        device: target device
    """
    ckpt_path = str(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state dict from Lightning checkpoint
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"Unexpected checkpoint format from {ckpt_path}")

    # Strip key prefix
    filtered = OrderedDict()
    prefix_dot = key_prefix + "."
    for k, v in state_dict.items():
        if k.startswith(prefix_dot):
            filtered[k[len(prefix_dot):]] = v

    if not filtered:
        raise RuntimeError(
            f"No keys with prefix '{key_prefix}' found in {ckpt_path}. "
            f"Available prefixes: {set(k.split('.')[0] for k in state_dict.keys())}"
        )

    vae.load_state_dict(filtered, strict=True)
    vae = vae.to(device).eval()
    print(f"Loaded VAE from {ckpt_path} (prefix='{key_prefix}', {len(filtered)} params)")
    return vae


def build_fullbody_genmo_vae(device="cuda"):
    """Build and load the GenmoFull VAE (135D input, no betas)."""
    vae = MotionGPTVQVaeAdapter(
        nfeats=135,
        code_dim=256,
        code_num=256,
        mu=0.99,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        width=512,
        output_emb_width=256,
    )
    vae = load_vae_from_checkpoint(vae, FULLBODY_GENMO_CKPT, "vae_body", device)
    return vae


def build_lower_genmo_vae(device="cuda"):
    """Build and load the hybrid lower VAE (61D input)."""
    vae = MotionGPTVQVaeAdapter(
        nfeats=61,
        code_dim=128,
        code_num=256,
        mu=0.95,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        width=512,
        output_emb_width=128,
        norm="GN",
    )
    vae = load_vae_from_checkpoint(vae, LOWER_GENMO_CKPT, "vae_lower", device)
    return vae


# =============================================================================
# Fullbody GENMO Tokenization
# =============================================================================

def collect_genmo_files(dataset_name):
    """
    Collect (genmo_npz_path, seq_name, token_output_dir) tuples for a dataset.

    Returns:
        list of (Path, str, Path) tuples
    """
    entries = []

    if dataset_name == "AMASS_talking":
        genmo_dir = Path(DATASET_PATHS["AMASS_talking"]["genmo_output"])
        token_base = Path(DATASET_PATHS["AMASS_talking"]["token_output"])
        if not genmo_dir.exists():
            print(f"Warning: GENMO directory not found: {genmo_dir}")
            return entries
        for npz_path in sorted(genmo_dir.glob("*.npz")):
            seq_name = npz_path.stem
            entries.append((npz_path, seq_name, token_base))

    elif dataset_name == "BEAT2":
        genmo_base = Path(DATASET_PATHS["BEAT2"]["genmo_output_base"])
        token_base = Path(DATASET_PATHS["BEAT2"]["token_output"])
        for subdir in DATASET_PATHS["BEAT2"]["subdirs"]:
            genmo_dir = genmo_base / subdir
            if not genmo_dir.exists():
                print(f"Warning: GENMO directory not found: {genmo_dir}")
                continue
            for npz_path in sorted(genmo_dir.glob("*.npz")):
                seq_name = npz_path.stem
                entries.append((npz_path, seq_name, token_base))

    return entries


def run_fullbody_genmo(datasets, device="cuda", max_samples=None):
    """
    Tokenize GENMO 145D data through the GenmoFull VAE (135D, strips betas).
    Saves token indices to {dataset_root}/TOKENS_AGENT_25/fullbody_genmo/{seq_name}.npy
    """
    print("=" * 60)
    print("Mode: fullbody_genmo (GENMO 145D -> GenmoFull tokens)")
    print("=" * 60)

    vae = build_fullbody_genmo_vae(device)

    for dataset_name in datasets:
        entries = collect_genmo_files(dataset_name)
        if not entries:
            print(f"No GENMO files found for {dataset_name}, skipping.")
            continue

        if max_samples is not None:
            entries = entries[:max_samples]

        stats = {"processed": 0, "skipped_existing": 0, "skipped_error": 0}

        for genmo_path, seq_name, token_base in tqdm(entries, desc=f"{dataset_name} fullbody_genmo"):
            out_dir = token_base / "fullbody_genmo"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{seq_name}.npy"

            if out_path.exists():
                stats["skipped_existing"] += 1
                continue

            try:
                data = np.load(genmo_path, allow_pickle=True)
                mv = data["motion_vector"]  # (T, 145)

                # Strip betas to get 135D: body_r6d(126) + global_orient_r6d(6) + local_vel(3)
                m135 = np.concatenate([mv[:, :126], mv[:, 136:145]], axis=-1)  # (T, 135)
                m135_t = torch.from_numpy(m135.astype(np.float32)).to(device)

                with torch.no_grad():
                    tokens = vae.map2index(m135_t.unsqueeze(0))  # (1, T//4)

                np.save(out_path, tokens.cpu().numpy())
                stats["processed"] += 1
            except Exception as e:
                stats["skipped_error"] += 1
                tqdm.write(f"Error tokenizing {seq_name}: {e}")

        print(f"{dataset_name}: {stats['processed']} processed, "
              f"{stats['skipped_existing']} existing, {stats['skipped_error']} errors")


# =============================================================================
# Lower GENMO Tokenization
# =============================================================================

def collect_source_files(dataset_name):
    """
    Collect (source_npz_path, genmo_npz_path, seq_name, foot_contact_path, token_output_dir)
    tuples for a dataset. Source npz has SMPL-X params; genmo npz has motion_vector for local_vel.

    Returns:
        list of tuples
    """
    entries = []

    if dataset_name == "AMASS_talking":
        source_dir = Path(DATASET_PATHS["AMASS_talking"]["source_rotation"])
        genmo_dir = Path(DATASET_PATHS["AMASS_talking"]["genmo_output"])
        fc_dir = Path(DATASET_PATHS["AMASS_talking"]["foot_contacts"])
        token_base = Path(DATASET_PATHS["AMASS_talking"]["token_output"])

        if not source_dir.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            return entries
        for npz_path in sorted(source_dir.glob("*.npz")):
            seq_name = npz_path.stem
            genmo_path = genmo_dir / f"{seq_name}.npz"
            fc_path = fc_dir / f"{seq_name}.npy"
            entries.append((npz_path, genmo_path, seq_name, fc_path, token_base))

    elif dataset_name == "BEAT2":
        base_input = Path(DATASET_PATHS["BEAT2"]["source_rotation_base"])
        genmo_base = Path(DATASET_PATHS["BEAT2"]["genmo_output_base"])
        fc_dir = Path(DATASET_PATHS["BEAT2"]["foot_contacts"])
        token_base = Path(DATASET_PATHS["BEAT2"]["token_output"])

        for subdir in DATASET_PATHS["BEAT2"]["subdirs"]:
            source_dir = base_input / subdir
            genmo_dir = genmo_base / subdir
            is_mirror = "mirror" in subdir

            if not source_dir.exists():
                print(f"Warning: Source directory not found: {source_dir}")
                continue

            for npz_path in sorted(source_dir.glob("*.npz")):
                seq_name = npz_path.stem
                genmo_path = genmo_dir / f"{seq_name}.npz"

                # Mirror files use M_ prefix for foot contacts
                if is_mirror:
                    fc_path = fc_dir / f"M_{seq_name}.npy"
                else:
                    fc_path = fc_dir / f"{seq_name}.npy"

                entries.append((npz_path, genmo_path, seq_name, fc_path, token_base))

    return entries


def build_lower_61d(source_npz_path, genmo_npz_path, foot_contact_path):
    """
    Build the 61D lower body input for the hybrid lower VAE.

    61D layout:
      [0:54]  - 9 lower joints x 6D rotation = 54D
      [54:57] - local velocity 3D (from genmo motion_vector)
      [57:61] - foot contact 4D

    Args:
        source_npz_path: Path to SMPL-X rotation npz
        genmo_npz_path:  Path to GENMO 145D npz (for local_vel)
        foot_contact_path: Path to foot contact npy

    Returns:
        numpy array (T, 61)
    """
    source = np.load(source_npz_path, allow_pickle=True)
    genmo = np.load(genmo_npz_path, allow_pickle=True)
    mv = genmo["motion_vector"]  # (T, 145)
    T = mv.shape[0]

    poses = source["poses"]  # (T, 165)
    global_orient_aa = torch.from_numpy(poses[:T, :3].astype(np.float32))   # (T, 3)
    body_pose_aa = torch.from_numpy(poses[:T, 3:66].astype(np.float32))     # (T, 63)

    # Build 22-joint 6D: concatenate global_orient + body_pose
    full22_aa = torch.cat([
        global_orient_aa.unsqueeze(1),       # (T, 1, 3) - joint 0 (pelvis)
        body_pose_aa.reshape(T, 21, 3),      # (T, 21, 3) - joints 1-21
    ], dim=1)                                # (T, 22, 3)

    full22_mat = axis_angle_to_matrix(full22_aa.reshape(-1, 3)).reshape(T, 22, 3, 3)
    full22_r6d = matrix_to_rotation_6d(full22_mat.reshape(-1, 3, 3)).reshape(T, 22, 6)

    # Select 9 lower joints
    lower_r6d = full22_r6d[:, LOWER_JOINT_INDICES_22]   # (T, 9, 6)
    lower_r6d_flat = lower_r6d.reshape(T, 54)           # (T, 54)

    # Local velocity from genmo motion_vector
    local_vel = mv[:, 142:145]                           # (T, 3)

    # Foot contact
    if foot_contact_path.exists():
        foot_contact = np.load(foot_contact_path)        # (T, 4)
        # Ensure length matches
        if foot_contact.shape[0] < T:
            pad = np.zeros((T - foot_contact.shape[0], 4))
            foot_contact = np.concatenate([foot_contact, pad], axis=0)
        foot_contact = foot_contact[:T]
    else:
        tqdm.write(f"Warning: foot contact not found at {foot_contact_path}, using zeros")
        foot_contact = np.zeros((T, 4))

    # Concatenate: 54 + 3 + 4 = 61D
    lower_61 = np.concatenate([
        lower_r6d_flat.numpy(),       # 54
        local_vel.astype(np.float32), # 3
        foot_contact.astype(np.float32),  # 4
    ], axis=-1)                       # (T, 61)

    return lower_61


def run_lower_genmo(datasets, device="cuda", max_samples=None):
    """
    Tokenize motion data through the hybrid lower VAE (61D input).
    Saves token indices to {dataset_root}/TOKENS_AGENT_25/lower_genmo/{seq_name}.npy
    """
    print("=" * 60)
    print("Mode: lower_genmo (SMPL-X -> 61D -> lower tokens)")
    print("=" * 60)

    vae = build_lower_genmo_vae(device)

    for dataset_name in datasets:
        entries = collect_source_files(dataset_name)
        if not entries:
            print(f"No source files found for {dataset_name}, skipping.")
            continue

        if max_samples is not None:
            entries = entries[:max_samples]

        stats = {"processed": 0, "skipped_existing": 0, "skipped_missing": 0, "skipped_error": 0}

        for source_path, genmo_path, seq_name, fc_path, token_base in tqdm(
            entries, desc=f"{dataset_name} lower_genmo"
        ):
            out_dir = token_base / "lower_genmo"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{seq_name}.npy"

            if out_path.exists():
                stats["skipped_existing"] += 1
                continue

            if not genmo_path.exists():
                stats["skipped_missing"] += 1
                continue

            try:
                lower_61 = build_lower_61d(source_path, genmo_path, fc_path)
                lower_t = torch.from_numpy(lower_61.astype(np.float32)).to(device)

                with torch.no_grad():
                    tokens = vae.map2index(lower_t.unsqueeze(0))  # (1, T//4)

                np.save(out_path, tokens.cpu().numpy())
                stats["processed"] += 1
            except Exception as e:
                stats["skipped_error"] += 1
                tqdm.write(f"Error tokenizing {seq_name}: {e}")

        print(f"{dataset_name}: {stats['processed']} processed, "
              f"{stats['skipped_existing']} existing, "
              f"{stats['skipped_missing']} missing genmo, "
              f"{stats['skipped_error']} errors")


# =============================================================================
# Verify Mode
# =============================================================================

def run_verify(datasets, device="cuda"):
    """
    Verify token shapes and downsample ratios for a few samples.
    Compares with source data to confirm expected temporal compression.
    """
    print("=" * 60)
    print("Mode: verify")
    print("=" * 60)

    for dataset_name in datasets:
        print(f"\n--- {dataset_name} ---")

        entries = collect_genmo_files(dataset_name)
        if not entries:
            print(f"No GENMO files found for {dataset_name}")
            continue

        # Check a few samples
        check_entries = entries[:5]

        for genmo_path, seq_name, token_base in check_entries:
            data = np.load(genmo_path, allow_pickle=True)
            mv = data["motion_vector"]
            T = mv.shape[0]
            print(f"\n  Sequence: {seq_name}")
            print(f"    Source frames (T): {T}")
            print(f"    Motion vector shape: {mv.shape}")

            # Check fullbody tokens
            fb_path = token_base / "fullbody_genmo" / f"{seq_name}.npy"
            if fb_path.exists():
                fb_tokens = np.load(fb_path)
                print(f"    Fullbody tokens shape: {fb_tokens.shape}")
                expected_len = T // 4  # stride_t=2, down_t=2 => 4x downsample
                print(f"    Expected token len (T//4): {expected_len}")
                print(f"    Downsample ratio: {T / fb_tokens.shape[-1]:.1f}x")
            else:
                print(f"    Fullbody tokens: NOT FOUND")

            # Check lower tokens
            lo_path = token_base / "lower_genmo" / f"{seq_name}.npy"
            if lo_path.exists():
                lo_tokens = np.load(lo_path)
                print(f"    Lower tokens shape: {lo_tokens.shape}")
                print(f"    Downsample ratio: {T / lo_tokens.shape[-1]:.1f}x")
            else:
                print(f"    Lower tokens: NOT FOUND")


# =============================================================================
# Visualize Mode
# =============================================================================

def run_visualize(datasets, device="cuda", num_samples=3):
    """
    Decode tokens back through VAE and print reconstruction stats.
    Useful for sanity-checking that encode/decode round-trips are reasonable.
    """
    print("=" * 60)
    print("Mode: visualize (decode tokens and check reconstruction)")
    print("=" * 60)

    vae_full = build_fullbody_genmo_vae(device)
    vae_lower = build_lower_genmo_vae(device)

    for dataset_name in datasets:
        print(f"\n--- {dataset_name} ---")
        entries = collect_genmo_files(dataset_name)
        if not entries:
            print(f"No GENMO files found for {dataset_name}")
            continue

        check_entries = entries[:num_samples]

        for genmo_path, seq_name, token_base in check_entries:
            print(f"\n  Sequence: {seq_name}")

            data = np.load(genmo_path, allow_pickle=True)
            mv = data["motion_vector"]
            T = mv.shape[0]

            # Fullbody: encode -> decode round-trip
            m135 = np.concatenate([mv[:, :126], mv[:, 136:145]], axis=-1)
            m135_t = torch.from_numpy(m135.astype(np.float32)).to(device)
            with torch.no_grad():
                tokens = vae_full.map2index(m135_t.unsqueeze(0))
                rec_135 = vae_full.decode(tokens.int()).squeeze(0)  # (T', 135)

            rec_T = rec_135.shape[0]
            overlap = min(T, rec_T)
            gt_135 = m135_t[:overlap]
            rec_overlap = rec_135[:overlap]
            mse_full = ((gt_135 - rec_overlap) ** 2).mean().item()
            print(f"    Fullbody: T={T}, tokens={tokens.shape[-1]}, "
                  f"rec_T={rec_T}, MSE={mse_full:.6f}")

            # Lower: check token file existence
            fb_path = token_base / "fullbody_genmo" / f"{seq_name}.npy"
            lo_path = token_base / "lower_genmo" / f"{seq_name}.npy"

            if lo_path.exists():
                lo_tokens = np.load(lo_path)
                lo_tokens_t = torch.from_numpy(lo_tokens).to(device)
                with torch.no_grad():
                    rec_61 = vae_lower.decode(lo_tokens_t.int()).squeeze(0)
                print(f"    Lower: tokens={lo_tokens.shape[-1]}, rec shape={rec_61.shape}")
            else:
                print(f"    Lower tokens not found, skipping decode check")


# =============================================================================
# CLI
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess SMPL-X to GENMO and generate motion tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "fullbody_genmo", "lower_genmo", "verify", "visualize"],
        help="Processing mode.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["AMASS_talking", "BEAT2"],
        help="Datasets to process (default: AMASS_talking BEAT2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for VAE inference (default: cuda).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples for visualize mode (default: 3).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit total processing count for testing (default: None = all).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    if args.mode == "preprocess":
        run_preprocess(args.datasets, max_samples=args.max_samples)
    elif args.mode == "fullbody_genmo":
        run_fullbody_genmo(args.datasets, device=args.device, max_samples=args.max_samples)
    elif args.mode == "lower_genmo":
        run_lower_genmo(args.datasets, device=args.device, max_samples=args.max_samples)
    elif args.mode == "verify":
        run_verify(args.datasets, device=args.device)
    elif args.mode == "visualize":
        run_visualize(args.datasets, device=args.device, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
