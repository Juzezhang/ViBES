"""
Evaluate face VQ-VAE tokenizer reconstruction quality.

Metrics (computed on FLAME vertices in mm):
  - LVE   : Lip Vertex Error (per-frame max L2 over lip verts, averaged over frames)
  - FFD   : Upper-face Dynamics Deviation (CodeTalker definition)
  - MPVPE : Mean Per-Vertex Position Error (Euclidean, averaged over verts and frames)
  - MOD   : Mouth Opening Difference (per-frame mean L2 over lip verts, averaged over frames)

Additionally reports parameter-space metrics:
  - Recons L1 (6D rotation + expression)
  - Expression L1
  - Jaw 6D L1

Usage:
    python -m inference.test_face_tokenizer \
        --cfg configs/config_mixed_stage1_face.yaml \
        --checkpoint ./model_files/pretrained_cpt/face/epoch=29.ckpt
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich import get_console
from rich.table import Table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodal_tokenizers.config import parse_args, instantiate_from_config
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle


# ---------------------------------------------------------------------------
# FLAME wrapper (same initialisation as MultimodalTokenizer)
# ---------------------------------------------------------------------------
def build_flame(flame_dir, batch_size=1024):
    """Build a FLAME model from the given directory."""
    from smplx import FLAME
    return FLAME(
        str(flame_dir),
        num_expression_coeffs=100,
        ext="pkl",
        batch_size=batch_size,
    ).eval()


def forward_flame(flame_model, exps, jaw_aa, batch_size=1024):
    """
    Run FLAME forward pass in batches.
    Args:
        exps:   (N, 100) expression coefficients
        jaw_aa: (N, 3)   jaw axis-angle
    Returns:
        vertices: (N, 5023, 3)
    """
    device = exps.device
    if flame_model.J_regressor.device != device:
        flame_model.to(device)

    N = exps.shape[0]
    all_verts = []
    s, r = N // batch_size, N % batch_size

    for i in range(s):
        b, e = i * batch_size, (i + 1) * batch_size
        out = flame_model(
            global_orient=torch.zeros((batch_size, 3), device=device),
            expression=exps[b:e],
            jaw_pose=jaw_aa[b:e],
            shape=torch.zeros((batch_size, 10), device=device),
        )
        all_verts.append(out.vertices)

    if r > 0:
        start = s * batch_size
        pad = batch_size - r
        out = flame_model(
            global_orient=torch.zeros((batch_size, 3), device=device),
            expression=torch.cat([exps[start:], torch.zeros((pad, 100), device=device)]),
            jaw_pose=torch.cat([jaw_aa[start:], torch.zeros((pad, 3), device=device)]),
            shape=torch.zeros((batch_size, 10), device=device),
        )
        all_verts.append(out.vertices[:r])

    return torch.cat(all_verts, dim=0)


# ---------------------------------------------------------------------------
# Metric helpers (mirrors FaceMetrics but standalone)
# ---------------------------------------------------------------------------
def compute_lve(pred_v, gt_v, lip_idx):
    """Per-frame max L2 over lip vertices, summed over frames."""
    idx = torch.as_tensor(lip_idx, device=pred_v.device, dtype=torch.long)
    diffs = pred_v.index_select(1, idx) - gt_v.index_select(1, idx)
    per_vert_l2 = torch.linalg.norm(diffs, dim=2)
    return per_vert_l2.max(dim=1).values.sum()


def compute_fdd(pred_v, gt_v, upper_face_idx, template):
    """Upper-face Dynamics Deviation (CodeTalker definition)."""
    if template.dim() == 2:
        template = template.unsqueeze(0)
    template = template.to(pred_v.device, dtype=pred_v.dtype)
    motion_pred = pred_v - template
    motion_gt = gt_v - template
    pred_msq = (motion_pred[:, upper_face_idx, :] ** 2).sum(dim=2)
    gt_msq = (motion_gt[:, upper_face_idx, :] ** 2).sum(dim=2)
    return gt_msq.std(dim=0, unbiased=False).mean() - pred_msq.std(dim=0, unbiased=False).mean()


def compute_mpvpe(pred_v, gt_v):
    """Mean Per-Vertex Position Error."""
    return torch.norm(pred_v - gt_v, p=2, dim=-1).mean(-1).sum()


def compute_mod(pred_v, gt_v, lip_idx):
    """Per-frame mean L2 over lip vertices, summed over frames."""
    idx = torch.as_tensor(lip_idx, device=pred_v.device, dtype=torch.long)
    diffs = pred_v.index_select(1, idx) - gt_v.index_select(1, idx)
    per_vert_l2 = torch.linalg.norm(diffs, dim=2)
    return per_vert_l2.mean(dim=1).sum()


# ---------------------------------------------------------------------------
# Build VQ-VAE model (face only)
# ---------------------------------------------------------------------------
def build_face_vae(cfg):
    """Instantiate just the face VQ-VAE from config."""
    vae_cfg = cfg.model.params.modality_tokenizer.vae_face
    from omegaconf import OmegaConf
    vae_config = OmegaConf.to_container(vae_cfg, resolve=True)
    return instantiate_from_config(vae_config)


def _remap_checkpoint(checkpoint_path):
    """Load a checkpoint that may reference the old 'conver_agent' module namespace."""
    import pickle
    import io

    class _RemapUnpickler(pickle.Unpickler):
        """Redirect old module namespaces → multimodal_tokenizers.* during unpickling."""
        _PREFIXES = ("conver_agent", "lom")

        def find_class(self, module, name):
            for prefix in self._PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    module = "multimodal_tokenizers" + module[len(prefix):]
                    break
            return super().find_class(module, name)

    # torch.load with custom unpickler via pickle_module
    class _PickleModule:
        Unpickler = _RemapUnpickler
        # Forward everything else to pickle
        def __getattr__(self, name):
            return getattr(pickle, name)

    pickle_module = _PickleModule()
    return torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
        pickle_module=pickle_module,
    )


def load_face_checkpoint(vae, checkpoint_path):
    """Load checkpoint weights into the face VQ-VAE."""
    ckpt = _remap_checkpoint(checkpoint_path)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    # Extract vae_face prefix
    vae_state = {
        k.replace("vae_face.", ""): v
        for k, v in state_dict.items()
        if k.startswith("vae_face.")
    }
    if not vae_state:
        # Fallback: try direct key match
        model_keys = set(vae.state_dict().keys())
        vae_state = {k: v for k, v in state_dict.items() if k in model_keys}

    if not vae_state:
        raise RuntimeError(f"No matching keys found in checkpoint: {checkpoint_path}")

    info = vae.load_state_dict(vae_state, strict=False)
    if info.missing_keys:
        print(f"  Warning: {len(info.missing_keys)} missing keys")
    if info.unexpected_keys:
        print(f"  Warning: {len(info.unexpected_keys)} unexpected keys")
    print(f"Loaded face VAE from {checkpoint_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_table(title, metrics):
    table = Table(title=title)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
        else:
            table.add_row(key, str(value))
    console = get_console()
    console.print(table, justify="center")


def main():
    parser = argparse.ArgumentParser(description="Face VQ-VAE tokenizer evaluation")
    parser.add_argument("--cfg", type=str, default="configs/config_mixed_stage1_face.yaml")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./model_files/pretrained_cpt/face/epoch=29.ckpt",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--vae_layer", type=int, default=0, help="Override vae_layer (0 = use config)")
    parser.add_argument("--code_num", type=int, default=0, help="Override code_num (0 = use config)")
    parser.add_argument("--codebook_size", type=int, default=0, help="Override codebook_size (0 = use config)")
    parser.add_argument("--vae_target", type=str, default=None,
                        help="Override VAE class target (e.g. multimodal_tokenizers.archs.lom_vq.VQVAEConvZeroDSUS_PaperVersion)")
    parser.add_argument("--test_datasets", type=str, nargs="+", default=None,
                        help="Override test datasets (e.g. --test_datasets TFHP)")
    args, remaining = parser.parse_known_args()

    # Filter out empty strings from remaining (can occur with shell line continuations)
    remaining = [r for r in remaining if r]
    sys.argv = [sys.argv[0], "--cfg", args.cfg, "--nodebug"] + remaining
    cfg = parse_args(phase="test")

    # Override VAE architecture params if specified
    vae_cfg = cfg.model.params.modality_tokenizer.vae_face
    if args.vae_target:
        vae_cfg.target = args.vae_target
    if args.vae_layer > 0:
        vae_cfg.params.vae_layer = args.vae_layer
    if args.code_num > 0:
        vae_cfg.params.code_num = args.code_num
    if args.codebook_size > 0:
        vae_cfg.params.codebook_size = args.codebook_size

    # Override test datasets (FaceVQDataset uses cfg.DATASET.datasets for all splits)
    if args.test_datasets:
        from omegaconf import OmegaConf
        cfg.DATASET.datasets = OmegaConf.create(
            [{"name": d} for d in args.test_datasets]
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Build face VQ-VAE ----
    print("Building face VQ-VAE...")
    vae = build_face_vae(cfg)
    load_face_checkpoint(vae, args.checkpoint)
    vae = vae.to(device).eval()

    face_dim = cfg.model.params.modality_tokenizer.vae_face.params.vae_test_dim
    print(f"Face dim: {face_dim}")

    # ---- Build FLAME ----
    print("Building FLAME model...")
    flame_dir = cfg.DATASET.FLAME_MODEL_DIR
    flame = build_flame(flame_dir)
    flame = flame.to(device).eval()

    # ---- Load region masks and template ----
    masks_path = os.path.join(str(flame_dir), "FLAME_masks.pkl")
    with open(masks_path, "rb") as f:
        masks = pickle.load(f, encoding="latin1")
    lip_idx = masks["lips"]
    upper_face_idx = np.concatenate([masks["eye_region"], masks["forehead"]])

    # Load template from FLAME model
    flame_model_path = os.path.join(str(flame_dir), "FLAME_NEUTRAL.pkl")
    with open(flame_model_path, "rb") as f:
        flame_data = pickle.load(f, encoding="latin1")
    template = torch.from_numpy(np.array(flame_data["v_template"])).float()

    # ---- Build dataloader ----
    print("Building test dataloader...")
    datamodule = build_data(cfg, phase="test")
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()
    print(f"Test dataset size: {len(test_loader.dataset)}")

    # ---- Run evaluation ----
    # Accumulators
    total_lve = 0.0
    total_fdd = 0.0
    total_mpvpe = 0.0
    total_mod = 0.0
    total_frames = 0

    total_recons_l1 = 0.0
    total_exp_l1 = 0.0
    total_jaw_l1 = 0.0
    total_param_frames = 0

    n_samples = 0
    n_valid_samples = 0  # samples with non-zero face data (for FFD averaging)

    # Determine face data layout.
    # 112D = head_6d(6) + jaw_6d(6) + expr(100)  (face_with_head)
    # 106D = jaw_6d(6) + expr(100)                (face without head)
    has_head = (face_dim == 112)
    jaw_start = 6 if has_head else 0
    jaw_end = 12 if has_head else 6
    expr_start = 12 if has_head else 6
    expr_end = face_dim  # 112 or 106
    print(f"Face layout: has_head={has_head}, jaw=[{jaw_start}:{jaw_end}], expr=[{expr_start}:{expr_end}]")

    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if args.max_samples > 0 and n_samples >= args.max_samples:
                break

            tar_face = batch["face"].to(device)  # (B, T, face_dim)
            bs, T = tar_face.shape[0], tar_face.shape[1]

            # VQ-VAE encode -> decode
            net_out = vae(tar_face[..., :face_dim])
            rec_face = net_out["rec_pose"]

            n = min(T, rec_face.shape[1])
            rec_face = rec_face[:, :n]
            tar_face = tar_face[:, :n]

            # --- Parameter-space metrics ---
            # Mask out zero-padded samples
            nonzero_mask = tar_face.abs().sum(dim=-1).sum(dim=-1) > 0  # (B,)
            if nonzero_mask.any():
                rec_valid = rec_face[nonzero_mask]
                tar_valid = tar_face[nonzero_mask]
                b_valid = rec_valid.shape[0]
                n_frames_param = b_valid * n

                total_recons_l1 += F.l1_loss(rec_valid, tar_valid, reduction="sum").item()
                total_exp_l1 += F.l1_loss(
                    rec_valid[..., expr_start:expr_end],
                    tar_valid[..., expr_start:expr_end],
                    reduction="sum",
                ).item()
                total_jaw_l1 += F.l1_loss(
                    rec_valid[..., jaw_start:jaw_end],
                    tar_valid[..., jaw_start:jaw_end],
                    reduction="sum",
                ).item()
                total_param_frames += n_frames_param
            else:
                b_valid = 0

            # --- Vertex-space metrics (via FLAME) ---
            for bi in range(bs):
                face_sum = tar_face[bi].abs().sum()
                if face_sum < 1e-6:
                    continue

                rec_i = rec_face[bi, :n]  # (n, face_dim)
                tar_i = tar_face[bi, :n]

                # Face data layout:
                #   112D: [head_6d(6), jaw_6d(6), expr(100)]
                #   106D: [jaw_6d(6), expr(100)]
                rec_jaw_aa = rotation_6d_to_axis_angle(
                    rec_i[:, jaw_start:jaw_end].reshape(-1, 6)
                ).reshape(-1, 3)
                tar_jaw_aa = rotation_6d_to_axis_angle(
                    tar_i[:, jaw_start:jaw_end].reshape(-1, 6)
                ).reshape(-1, 3)
                rec_exps = rec_i[:, expr_start:expr_end]
                tar_exps = tar_i[:, expr_start:expr_end]

                # Pad or trim expressions to exactly 100D
                if rec_exps.shape[-1] < 100:
                    rec_exps = F.pad(rec_exps, (0, 100 - rec_exps.shape[-1]))
                    tar_exps = F.pad(tar_exps, (0, 100 - tar_exps.shape[-1]))
                elif rec_exps.shape[-1] > 100:
                    rec_exps = rec_exps[..., :100]
                    tar_exps = tar_exps[..., :100]

                # FLAME forward
                rec_verts = forward_flame(flame, rec_exps, rec_jaw_aa)  # (n, 5023, 3)
                tar_verts = forward_flame(flame, tar_exps, tar_jaw_aa)  # (n, 5023, 3)

                # Scale to mm
                rec_verts_mm = rec_verts * 1000.0
                tar_verts_mm = tar_verts * 1000.0

                total_lve += compute_lve(rec_verts_mm, tar_verts_mm, lip_idx).item()
                total_fdd += compute_fdd(rec_verts_mm, tar_verts_mm, upper_face_idx, template * 1000.0).item()
                total_mpvpe += compute_mpvpe(rec_verts_mm, tar_verts_mm).item()
                total_mod += compute_mod(rec_verts_mm, tar_verts_mm, lip_idx).item()
                total_frames += n
                n_valid_samples += 1

            n_samples += bs

    # ---- Compute final metrics ----
    results = {}
    if total_frames > 0:
        results["LVE (mm)"] = total_lve / total_frames
        results["FFD (mm^2)"] = total_fdd / max(n_valid_samples, 1)  # FFD is per-sample
        results["MPVPE (mm)"] = total_mpvpe / total_frames
        results["MOD (mm)"] = total_mod / total_frames

    if total_param_frames > 0:
        n_elements_recons = total_param_frames * face_dim
        n_elements_exp = total_param_frames * (expr_end - expr_start)
        n_elements_jaw = total_param_frames * 6
        results["Recons L1"] = total_recons_l1 / n_elements_recons
        results["Expression L1"] = total_exp_l1 / n_elements_exp
        results["Jaw 6D L1"] = total_jaw_l1 / n_elements_jaw

    results["Total samples"] = n_samples
    results["Total valid samples"] = n_valid_samples
    results["Total frames"] = total_frames

    print_table("Face VQ-VAE Reconstruction Metrics", results)

    # Save to JSON
    import json
    out_dir = Path("results") / "face_tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = Path(args.checkpoint).stem
    out_path = out_dir / f"metrics_{ckpt_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
