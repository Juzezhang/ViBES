"""3-way trajectory visualization of the Global VAE (V1 / V2 / V3) on the SAME test samples.

The Global VAE recovers global root translation from the lower-body pose (predicts root local
velocity, then integrates it with the pelvis orientation). This script overlays the ground-truth
translation against the recovered translation of three checkpoints — V1 (released), V2 (retrained,
velocity-only) and V3 (velocity + foot-contacts) — on identical BEAT2 / AMASS *test* sequences, so
the per-axis drift (especially the vertical Y axis) is directly comparable across versions.

For each picked sample it writes:
  * ``<id>_xyz.png``       — X / Y / Z translation vs time, GT + V1 + V2 + V3 overlaid (err in title)
  * ``<id>_topdown.png``   — the X–Z ground path (plan view), GT + V1 + V2 + V3

All three checkpoints share the same architecture (``vae_test_dim=61``), so one model is built and the
``vae_global`` state dict is swapped per version. Data is loaded once from the combined BEAT2+AMASS
config and filtered per-sample by ``dataset_name`` in the loop (restricting the loader corrupts the
representation — see eval_global_vae_translation.py).

Env overrides:
  VIBES_CKPT_V1 / V2 / V3   the three checkpoints (required; no real paths baked in)
  VIBES_BEAT2_ROOT / VIBES_AMASS_ROOT   real dataset roots (override assets.yaml placeholders)
  VIBES_VIZ_N               samples per dataset (default 4)
  VIBES_VIZ_MIN_FRAMES      skip sequences shorter than this (default 16)
  VIBES_VIZ_OUT            output root (default /path/to/viz_demos/global_vae_trans_3way)

  python -m scripts.viz_global_vae_3way --cfg configs/config_global_rerun_beat2amass.yaml
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from multimodal_tokenizers.config import parse_args
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.data.utils import conversation_collate
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.load_checkpoint import _extract_state_dict

# version label -> (env var, line color)
VERSIONS = [
    ("V1", "VIBES_CKPT_V1", "tab:red"),
    ("V2", "VIBES_CKPT_V2", "tab:green"),
    ("V3", "VIBES_CKPT_V3", "tab:orange"),
]
GT_COLOR = "black"


def _load_vae_global_sd(ckpt_path):
    sd = _extract_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False), ckpt_path)
    g = {k.replace("vae_global.", ""): v for k, v in sd.items() if k.startswith("vae_global.")}
    return g or sd


def _recover_trans(model, vae_sd, lower, global_dim, tar_go6d, init_pos, device):
    """Load a version's vae_global weights and integrate its predicted velocity to translation."""
    model.vae_global.load_state_dict(vae_sd, strict=True)
    to_global = lower[:, :, :global_dim].clone()
    if to_global.shape[2] > 54:
        to_global[:, :, 54:] = 0.0
    with torch.no_grad():
        rec_vel = model.vae_global(to_global.to(device))["rec_pose"][:, :, 54:57]
    rec_trans, _ = model._integrate_local_velocity(rec_vel, tar_go6d, init_pos=init_pos)
    return rec_trans[0].cpu().numpy()


def _collect_indices(dataset, target, n_want, min_frames):
    """n_want test indices for `target`, spread evenly across the matching test samples (so we
    don't pick e.g. 4 clips from the same speaker), each with length >= min_frames."""
    matches = []
    for idx in range(len(dataset)):
        batch = conversation_collate([dataset[idx]])
        dsname = str(batch.get("dataset_name", ["?"])[0]).lower()
        if target not in dsname:
            continue
        n = batch.get("motion_len", [batch["lower"].shape[1]])[0]
        if int(n) < min_frames:
            continue
        sid = str(batch.get("id_name", [f"sample{idx}"])[0])
        matches.append((idx, sid))
    if not matches:
        return []
    if len(matches) <= n_want:
        return matches
    sel = np.linspace(0, len(matches) - 1, n_want).round().astype(int)
    return [matches[i] for i in sel]


def main():
    cfg = parse_args(phase="test")
    if cfg.Selected_part != "global":
        raise ValueError("viz_global_vae_3way expects Selected_part == 'global'")
    if os.environ.get("VIBES_BEAT2_ROOT"):
        cfg.DATASET.BEAT2.ROOT = os.environ["VIBES_BEAT2_ROOT"]
    if os.environ.get("VIBES_AMASS_ROOT"):
        cfg.DATASET.AMASS.ROOT = os.environ["VIBES_AMASS_ROOT"]

    ckpts = {}
    for label, env, _ in VERSIONS:
        p = os.environ.get(env)
        if not p or not os.path.exists(p):
            raise ValueError(f"{label} checkpoint missing: set {env} to an existing path (got {p!r})")
        ckpts[label] = _load_vae_global_sd(p)

    n_want = int(os.environ.get("VIBES_VIZ_N", "4"))
    min_frames = int(os.environ.get("VIBES_VIZ_MIN_FRAMES", "16"))
    out_root = Path(os.environ.get("VIBES_VIZ_OUT",
                                   "/path/to/viz_demos/global_vae_trans_3way"))
    fps = float(cfg.DATASET.pose_fps)

    # CPU by default — the VAE is tiny and this avoids contending with training/inference on GPUs.
    device = torch.device(os.environ.get("VIBES_VIZ_DEVICE",
                                         "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg).to(device).eval()

    cfg.TRAIN.NUM_WORKERS = cfg.EVAL.NUM_WORKERS = cfg.TEST.NUM_WORKERS = 0
    datamodule = build_data(cfg)
    cfg.TEST.SPLIT = "test"
    datamodule._test_dataset = None
    dataset = datamodule.test_dataset
    global_dim = cfg.model.params.modality_tokenizer.vae_global.params.vae_test_dim
    print(f"Loaded test set: {len(dataset)} samples | global_dim={global_dim} | device={device}")

    for target in ["beat2", "amass"]:
        picks = _collect_indices(dataset, target, n_want, min_frames)
        if not picks:
            print(f"  [{target}] no matching test samples — skipping")
            continue
        out_dir = out_root / target.upper()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{target}] {len(picks)} samples -> {out_dir}")

        for k, (idx, sid) in enumerate(picks):
            batch = conversation_collate([dataset[idx]])
            n = int(batch.get("motion_len", [batch["lower"].shape[1]])[0])
            lower = batch["lower"][:, :n].to(device)
            tar_go6d = lower[:, :, :6]
            tar_trans, _ = model._integrate_local_velocity(lower[:, :, 54:57], tar_go6d)
            init_pos = tar_trans[:, :1, :].squeeze(1)
            gt = tar_trans[0].cpu().numpy()

            recs = {label: _recover_trans(model, ckpts[label], lower, global_dim,
                                          tar_go6d, init_pos, device)
                    for label, _, _ in VERSIONS}
            t = np.arange(n) / fps
            safe_sid = sid.replace("/", "_")

            # --- per-axis X/Y/Z over time ---
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for a, name in enumerate(["X (m)", "Y / up (m)", "Z (m)"]):
                axes[a].plot(t, gt[:, a], color=GT_COLOR, lw=2.2, label="GT", zorder=5)
                errs = []
                for label, _, color in VERSIONS:
                    rc = recs[label]
                    e = np.abs(gt[:, a] - rc[:, a]).mean() * 1000
                    errs.append(f"{label}={e:.0f}")
                    axes[a].plot(t, rc[:, a], color=color, ls="--", lw=1.4,
                                 label=f"{label} ({e:.0f}mm)")
                axes[a].set_ylabel(name)
                axes[a].set_title(f"{name.split()[0]}   mean|err|: " + "  ".join(errs),
                                  fontsize=9, loc="left")
                axes[a].grid(alpha=0.3)
            axes[0].legend(loc="upper left", fontsize=8, ncol=4)
            axes[-1].set_xlabel("time (s)")
            fig.suptitle(f"{target.upper()} · {sid} · root translation: GT vs V1/V2/V3", fontsize=12)
            fig.tight_layout()
            fig.savefig(out_dir / f"{k:02d}_{safe_sid}_xyz.png", dpi=110)
            plt.close(fig)

            # --- top-down X-Z ground path (plan view) ---
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(gt[:, 0], gt[:, 2], color=GT_COLOR, lw=2.4, label="GT", zorder=5)
            ax.scatter([gt[0, 0]], [gt[0, 2]], color=GT_COLOR, marker="o", s=40, zorder=6)
            for label, _, color in VERSIONS:
                rc = recs[label]
                fd = np.linalg.norm(gt[-1] - rc[-1]) * 1000
                ax.plot(rc[:, 0], rc[:, 2], color=color, ls="--", lw=1.5,
                        label=f"{label} (drift {fd:.0f}mm)")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
            ax.set_aspect("equal", "datalim")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_title(f"{target.upper()} · {sid} · ground path (top-down)", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"{k:02d}_{safe_sid}_topdown.png", dpi=110)
            plt.close(fig)
            print(f"    [{k}] {sid}  n={n}")

    print(f"\nDone. Plots under {out_root}")


if __name__ == "__main__":
    main()
