"""Quantitative evaluation of the Global VAE (``VAE_Global_from_Lower54``).

The Global VAE recovers **global translation from the lower-body pose alone**: it predicts the
root local velocity from the 54-dim lower-body 6D rotations (non-rotation channels zeroed), then
integrates that velocity (using the pelvis global orientation) into world-frame translation. This
script runs it over the test set and compares the recovered translation against the ground truth.

Metrics (translation is in metres; reported in mm):
  * trans_err_mm      — mean per-frame ‖rec_trans − gt_trans‖₂, averaged over frames then samples
  * final_drift_mm    — endpoint error ‖rec_trans[-1] − gt_trans[-1]‖₂, averaged over samples
  * trans_err_axis_mm — same per-frame error split into x / y / z
  * local_vel_mae     — mean |rec_local_vel − gt_local_vel| (the directly-supervised quantity)

Runtime overrides via env vars (keeps internal/scrubbed paths out of the repo):
  VIBES_GLOBAL_CKPT      global VAE checkpoint (default: TEST.CHECKPOINTS from the config)
  VIBES_BEAT2_ROOT       real BEAT2 root  (overrides the assets.yaml placeholder)
  VIBES_AMASS_ROOT       real AMASS root
  VIBES_EVAL_MODALITIES  comma list to restrict datasets, e.g. "BEAT2"
  VIBES_EVAL_MAX_SAMPLES cap on number of test samples (0 = all)

Usage:
  python -m scripts.eval_global_vae_translation --cfg configs/config_mixed_stage1_vae_global_wo_mesh_lr1e-4.yaml
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from multimodal_tokenizers.config import parse_args
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.data.utils import conversation_collate
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.load_checkpoint import _extract_state_dict


def _load_vae_global(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, ckpt_path)
    sd_global = {k.replace("vae_global.", ""): v for k, v in state_dict.items()
                 if k.startswith("vae_global.")}
    if not sd_global:
        sd_global = state_dict
    model.vae_global.load_state_dict(sd_global, strict=True)


def _foot_refine_transl(model, endec, lower, rec_trans, rec_contacts_logits, betas10):
    """V3 foot-contact refine of the velocity-integrated translation.

    Build the SMPL-X pose from the lower body (legs real, rest identity), then use the regressed
    4 foot contacts (joints 7,8,10,11) to anchor stationary feet via pp_static_joint, which cancels
    contact-foot displacement (de-skating) and pins the ground (fixes vertical drift).
    """
    from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle
    from multimodal_tokenizers.data.mixed_dataset.data_tools import JOINT_MASK_LOWER_6D
    from multimodal_tokenizers.utils.genmo.postprocess import pp_static_joint

    B, n = lower.shape[0], lower.shape[1]
    dev = lower.device
    lower_rot6d = lower[:, :, :54].reshape(B * n, 9 * 6)
    full6d = model.inverse_selection_tensor_full_body_6D_from_lower(
        lower_rot6d, JOINT_MASK_LOWER_6D, B * n).reshape(B * n, 55, 6)
    zmask = torch.isclose(full6d.abs().sum(-1), torch.zeros(1, device=dev))
    if zmask.any():
        full6d[zmask] = torch.tensor([1., 0., 0., 0., 1., 0.], device=dev)
    pose_aa = rotation_6d_to_axis_angle(full6d).reshape(B, n, 55 * 3)
    global_orient, body_pose = pose_aa[:, :, :3], pose_aa[:, :, 3:66]
    betas = betas10.reshape(1, 1, -1).expand(B, n, 10).contiguous()
    # remap our 4 foot contacts [J7,J8,J10,J11] into pp_static_joint's 6-slot [7,10,8,11,20,21]; wrists off
    c = rec_contacts_logits
    conf6 = torch.full((B, n, 6), -1e6, device=dev)
    conf6[..., 0], conf6[..., 1], conf6[..., 2], conf6[..., 3] = c[..., 0], c[..., 2], c[..., 1], c[..., 3]
    outputs = {
        "pred_smpl_params_global": {"body_pose": body_pose, "global_orient": global_orient,
                                    "transl": rec_trans, "betas": betas},
        "static_conf_logits": conf6,
    }
    return pp_static_joint(outputs, endec)


def main():
    cfg = parse_args(phase="test")
    if cfg.Selected_part != "global":
        raise ValueError("eval_global_vae_translation expects Selected_part == 'global'")

    # --- runtime overrides (env) ---
    if os.environ.get("VIBES_BEAT2_ROOT"):
        cfg.DATASET.BEAT2.ROOT = os.environ["VIBES_BEAT2_ROOT"]
    if os.environ.get("VIBES_AMASS_ROOT"):
        cfg.DATASET.AMASS.ROOT = os.environ["VIBES_AMASS_ROOT"]
    # Load whatever datasets the config defines (combined), then FILTER per-sample by
    # dataset_name in the loop. Restricting cfg.DATASET.{MODALITIES,datasets} corrupts the
    # data representation, so we don't touch them — we just skip non-target samples.
    target_ds = None
    if os.environ.get("VIBES_EVAL_MODALITIES"):
        target_ds = {m.strip().lower() for m in os.environ["VIBES_EVAL_MODALITIES"].split(",") if m.strip()}
    max_samples = int(os.environ.get("VIBES_EVAL_MAX_SAMPLES", "0")) or None

    ckpt_path = os.environ.get("VIBES_GLOBAL_CKPT") or OmegaConf.select(cfg, "TEST.CHECKPOINTS")
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise ValueError(f"Global VAE checkpoint not found: {ckpt_path} "
                         "(set VIBES_GLOBAL_CKPT or TEST.CHECKPOINTS).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    _load_vae_global(model, ckpt_path)

    cfg.TRAIN.NUM_WORKERS = 0
    cfg.EVAL.NUM_WORKERS = 0
    cfg.TEST.NUM_WORKERS = 0
    datamodule = build_data(cfg)
    cfg.TEST.SPLIT = "test"
    datamodule._test_dataset = None
    dataset = datamodule.test_dataset
    n_total = len(dataset)
    n_eval = n_total if max_samples is None else min(max_samples, n_total)
    print(f"Global VAE eval | ckpt={ckpt_path} | test samples={n_eval}/{n_total}")

    global_dim = cfg.model.params.modality_tokenizer.vae_global.params.vae_test_dim

    do_refine = bool(os.environ.get("VIBES_EVAL_REFINE"))
    endec = None
    if do_refine:
        from multimodal_tokenizers.utils.genmo.endecoder import EnDecoder
        endec = EnDecoder().to(device).eval()
        print("Foot-contact refine: ON (V3)")

    trans_err, final_drift, axis_err, vel_mae = [], [], [], []
    r_trans_err, r_final_drift, r_axis_err = [], [], []
    skipped = 0
    for idx in tqdm(range(n_eval), desc="eval"):
        try:
            batch = conversation_collate([dataset[idx]])
            if target_ds is not None:
                dsname = str(batch.get("dataset_name", ["?"])[0]).lower()
                if not any(t in dsname for t in target_ds):
                    continue
            lower = batch["lower"].to(device)
            n = lower.shape[1]
            lengths = batch.get("motion_len", None)
            if lengths is not None:
                n = int(lengths[0])
            if n < 2:
                skipped += 1
                continue
            lower = lower[:, :n]

            tar_local_vel = lower[:, :, 54:57]
            tar_go6d = lower[:, :, :6]
            tar_trans, _ = model._integrate_local_velocity(tar_local_vel, tar_go6d)

            to_global = lower[:, :, :global_dim].clone()
            if to_global.shape[2] > 54:
                to_global[:, :, 54:] = 0.0
            with torch.no_grad():
                rec_pose = model.vae_global(to_global)["rec_pose"]
            rec_local_vel = rec_pose[:, :, 54:57]
            rec_trans, _ = model._integrate_local_velocity(
                rec_local_vel, tar_go6d, init_pos=tar_trans[:, 0, :])

            d = (rec_trans - tar_trans)[0]                      # (n, 3) metres
            trans_err.append(d.norm(dim=-1).mean().item() * 1000.0)
            final_drift.append(d[-1].norm().item() * 1000.0)
            axis_err.append((d.abs().mean(dim=0) * 1000.0).tolist())
            vel_mae.append((rec_local_vel - tar_local_vel)[0].abs().mean().item())

            if do_refine:
                betas10 = (batch["shape"][0, 0, :10].to(device) if "shape" in batch
                           else lower[0, 0, -10:])
                with torch.no_grad():
                    refined = _foot_refine_transl(model, endec, lower, rec_trans,
                                                  rec_pose[:, :, 57:61], betas10)
                # Re-anchor to GT first frame: pp_static_joint's ground_y step re-references the
                # vertical origin (feet→y=0), which differs from GT's origin by a constant. Align
                # starts so the metric reflects drift, not the coordinate-origin offset.
                refined = refined - (refined[:, :1, :] - tar_trans[:, :1, :])
                rd = (refined - tar_trans)[0]
                r_trans_err.append(rd.norm(dim=-1).mean().item() * 1000.0)
                r_final_drift.append(rd[-1].norm().item() * 1000.0)
                r_axis_err.append((rd.abs().mean(dim=0) * 1000.0).tolist())

            if os.environ.get("VIBES_EVAL_DEBUG") and len(trans_err) <= 3:
                print(f"[dbg] n={n} lower={tuple(lower.shape)} "
                      f"tar_vel(min/max/mean)={tar_local_vel.min():.4f}/{tar_local_vel.max():.4f}/{tar_local_vel.mean():.4f} "
                      f"rec_vel(min/max/mean)={rec_local_vel.min():.4f}/{rec_local_vel.max():.4f}/{rec_local_vel.mean():.4f} "
                      f"tar_trans(min/max)={tar_trans.min():.3f}/{tar_trans.max():.3f} "
                      f"rec_trans(min/max)={rec_trans.min():.3f}/{rec_trans.max():.3f} "
                      f"d_axis_m={[round(v,3) for v in d.abs().mean(0).tolist()]}", flush=True)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            if skipped <= 5:
                print(f"  skip idx={idx}: {e}")

    if not trans_err:
        raise RuntimeError("No samples evaluated — check data roots / modalities.")

    axis = np.array(axis_err)  # (N, 3)
    res = {
        "label": os.environ.get("VIBES_EVAL_LABEL", os.path.basename(str(ckpt_path))),
        "dataset": os.environ.get("VIBES_EVAL_MODALITIES", ""),
        "checkpoint": str(ckpt_path),
        "n_evaluated": len(trans_err),
        "n_skipped": skipped,
        "trans_err_mm_mean": float(np.mean(trans_err)),
        "trans_err_mm_std": float(np.std(trans_err)),
        "trans_err_mm_median": float(np.median(trans_err)),
        "final_drift_mm_mean": float(np.mean(final_drift)),
        "final_drift_mm_median": float(np.median(final_drift)),
        "trans_err_axis_mm": {"x": float(axis[:, 0].mean()),
                              "y": float(axis[:, 1].mean()),
                              "z": float(axis[:, 2].mean())},
        "local_vel_mae_mean": float(np.mean(vel_mae)),
    }
    if do_refine and r_trans_err:
        raxis = np.array(r_axis_err)
        res["refined"] = {
            "trans_err_mm_mean": float(np.mean(r_trans_err)),
            "trans_err_mm_median": float(np.median(r_trans_err)),
            "final_drift_mm_mean": float(np.mean(r_final_drift)),
            "trans_err_axis_mm": {"x": float(raxis[:, 0].mean()),
                                  "y": float(raxis[:, 1].mean()),
                                  "z": float(raxis[:, 2].mean())},
        }
    print("\n===== Global VAE translation eval =====")
    print(json.dumps(res, indent=2))

    if os.environ.get("VIBES_EVAL_OUT"):
        out_path = Path(os.environ["VIBES_EVAL_OUT"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path("experiments") / "global_vae_translation_check" / cfg.NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
