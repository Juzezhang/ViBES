"""Per-axis trajectory plots for the Global VAE (VAE_Global_from_Lower54).

For a few test sequences, plots the ground-truth vs recovered root translation along each
axis (X / Y / Z) over time, so the per-axis drift (especially vertical Y) is visible. Also
saves a per-axis mean-|error| bar chart for the evaluated subset.

Env overrides (same as eval_global_vae_translation.py): VIBES_GLOBAL_CKPT, VIBES_BEAT2_ROOT,
VIBES_AMASS_ROOT, VIBES_EVAL_MODALITIES. Extra: VIBES_PLOT_N (samples, default 6),
VIBES_PLOT_OUT (output dir).

  python -m scripts.plot_global_vae_trans --cfg configs/config_mixed_stage1_vae_global_wo_mesh_lr1e-4.yaml
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


def _load_vae_global(model, ckpt_path):
    sd = _extract_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False), ckpt_path)
    g = {k.replace("vae_global.", ""): v for k, v in sd.items() if k.startswith("vae_global.")}
    model.vae_global.load_state_dict(g or sd, strict=True)


def main():
    cfg = parse_args(phase="test")
    if os.environ.get("VIBES_BEAT2_ROOT"):
        cfg.DATASET.BEAT2.ROOT = os.environ["VIBES_BEAT2_ROOT"]
    if os.environ.get("VIBES_AMASS_ROOT"):
        cfg.DATASET.AMASS.ROOT = os.environ["VIBES_AMASS_ROOT"]
    if os.environ.get("VIBES_EVAL_MODALITIES"):
        cfg.DATASET.MODALITIES = {m.strip(): ["lower"]
                                  for m in os.environ["VIBES_EVAL_MODALITIES"].split(",") if m.strip()}
    ckpt = os.environ.get("VIBES_GLOBAL_CKPT") or OmegaConf.select(cfg, "TEST.CHECKPOINTS")
    n_plot = int(os.environ.get("VIBES_PLOT_N", "6"))
    tag = os.environ.get("VIBES_EVAL_MODALITIES", "data").split(",")[0]
    out_dir = Path(os.environ.get("VIBES_PLOT_OUT", f"/path/to/viz_demos/global_vae_trans/{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(cfg.DATASET.pose_fps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    _load_vae_global(model, ckpt)
    cfg.TEST.NUM_WORKERS = 0
    datamodule = build_data(cfg)
    cfg.TEST.SPLIT = "test"
    datamodule._test_dataset = None
    dataset = datamodule.test_dataset
    global_dim = cfg.model.params.modality_tokenizer.vae_global.params.vae_test_dim

    axis_abs = []
    plotted = 0
    idx = 0
    while plotted < n_plot and idx < len(dataset):
        batch = conversation_collate([dataset[idx]]); idx += 1
        lower = batch["lower"].to(device)
        n = int(batch.get("motion_len", [lower.shape[1]])[0])
        if n < 8:
            continue
        lower = lower[:, :n]
        tar_go6d = lower[:, :, :6]
        tar_trans, _ = model._integrate_local_velocity(lower[:, :, 54:57], tar_go6d)
        to_global = lower[:, :, :global_dim].clone()
        if to_global.shape[2] > 54:
            to_global[:, :, 54:] = 0.0
        with torch.no_grad():
            rec_vel = model.vae_global(to_global)["rec_pose"][:, :, 54:57]
        rec_trans, _ = model._integrate_local_velocity(rec_vel, tar_go6d, init_pos=tar_trans[:, 0, :])

        gt = tar_trans[0].cpu().numpy(); rc = rec_trans[0].cpu().numpy()
        axis_abs.append(np.abs(gt - rc).mean(axis=0))
        t = np.arange(n) / fps
        sid = batch.get("id_name", [f"sample{idx}"])[0]

        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
        for a, name in enumerate(["X (forward/side)", "Y (up)", "Z"]):
            axes[a].plot(t, gt[:, a], color="tab:blue", label="GT")
            axes[a].plot(t, rc[:, a], color="tab:orange", ls="--", label="recovered")
            err = np.abs(gt[:, a] - rc[:, a]).mean() * 1000
            axes[a].set_ylabel(f"{name} (m)")
            axes[a].set_title(f"{name}   mean|err| = {err:.0f} mm", fontsize=9, loc="left")
            axes[a].grid(alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"{tag} · {sid} · Global VAE: GT vs recovered translation", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{plotted:02d}_{sid}_trans_xyz.png", dpi=110)
        plt.close(fig)
        plotted += 1

    # per-axis summary bar chart
    axis_abs = np.array(axis_abs) * 1000  # mm
    means = axis_abs.mean(axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["X", "Y (up)", "Z"], means, color=["tab:gray", "tab:red", "tab:gray"])
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("mean |error| (mm)")
    ax.set_title(f"{tag}: per-axis translation error (n={plotted} shown)")
    fig.tight_layout()
    fig.savefig(out_dir / "_per_axis_error.png", dpi=110)
    plt.close(fig)
    print(f"Saved {plotted} trajectory plots + per-axis bar to {out_dir}")


if __name__ == "__main__":
    main()
