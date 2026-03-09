import argparse
import json
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from multimodal_tokenizers.config import get_module_config


class RunningStatsVec:
    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)

    def update_batch(self, x):
        if x.size == 0:
            return
        x = x.astype(np.float64)
        batch_count = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_m2 = ((x - batch_mean) ** 2).sum(axis=0)
        batch_min = x.min(axis=0)
        batch_max = x.max(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min = batch_min
            self.max = batch_max
            return

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + (delta ** 2) * (self.count * batch_count / total)
        self.count = total
        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)

    def finalize(self):
        if self.count < 2:
            std = np.zeros_like(self.mean)
        else:
            std = np.sqrt(self.m2 / (self.count - 1))
        return {
            "count": int(self.count),
            "mean": self.mean.astype(np.float32),
            "std": std.astype(np.float32),
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
        }


def load_cfg(cfg_path, cfg_assets_path):
    cfg_assets = OmegaConf.load(cfg_assets_path)
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load(cfg_path))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    return cfg


def collect_npz_files(root, pose_rep, subdirs=None):
    base = Path(root) / pose_rep
    if not subdirs:
        return list(base.rglob("*.npz"))
    files = []
    for sub in subdirs:
        files.extend(list((base / sub).rglob("*.npz")))
    return files


def compute_stats_for_files(files, dim, max_files=0, seed=0):
    if max_files and len(files) > max_files:
        rng = np.random.default_rng(seed)
        files = list(rng.choice(files, size=max_files, replace=False))

    stats = RunningStatsVec(dim)
    bad = 0
    for fpath in tqdm(files, desc="motion_vector stats"):
        try:
            data = np.load(fpath, allow_pickle=True)
            if "motion_vector" not in data:
                continue
            mv = data["motion_vector"]
            if mv.ndim != 2 or mv.shape[1] < dim:
                continue
            mv = mv[:, :dim]
            stats.update_batch(mv)
        except Exception:
            bad += 1
            continue
    return stats.finalize(), len(files), bad


def summarize(name, stats, low_std_thresh, topk):
    std = stats["std"]
    rng = stats["max"] - stats["min"]
    low_std = np.where(std < low_std_thresh)[0]
    low_std_ratio = float(len(low_std)) / len(std)
    topk = min(topk, len(std))
    lowest = np.argsort(std)[:topk]
    highest = np.argsort(std)[-topk:][::-1]
    summary = {
        "dataset": name,
        "frames": stats["count"],
        "mean_std": float(std.mean()),
        "min_std": float(std.min()),
        "max_std": float(std.max()),
        "mean_range": float(rng.mean()),
        "low_std_ratio": low_std_ratio,
        "low_std_dims": low_std.tolist(),
        "lowest_std_dims": lowest.tolist(),
        "highest_std_dims": highest.tolist(),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="GENMO motion_vector distribution stats")
    parser.add_argument("--cfg", type=str, required=True, help="config yaml")
    parser.add_argument("--cfg_assets", type=str, default="./configs/assets.yaml", help="assets yaml")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "beat2_genmo", "amass_genmo"])
    parser.add_argument("--max_files", type=int, default=0, help="max files per dataset (0 = all)")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for sampling")
    parser.add_argument("--out_dir", type=str, default="", help="output directory to save stats")
    parser.add_argument("--low_std", type=float, default=1e-4, help="threshold for low std ratio")
    parser.add_argument("--topk", type=int, default=10, help="top-k dims for std report")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg, args.cfg_assets)
    dim = 145

    outputs = {}

    if args.dataset in ["all", "beat2_genmo"]:
        beat2_root = OmegaConf.select(cfg, "DATASET.BEAT2_Genmo.ROOT")
        pose_rep = OmegaConf.select(cfg, "DATASET.BEAT2_Genmo.pose_rep")
        subdirs = OmegaConf.select(cfg, "DATASET.BEAT2_Genmo.pose_rep_subdirs")
        files = collect_npz_files(beat2_root, pose_rep, subdirs=subdirs)
        stats, file_count, bad = compute_stats_for_files(files, dim, args.max_files, args.seed)
        summary = summarize("beat2_genmo", stats, args.low_std, args.topk)
        summary["files"] = file_count
        summary["bad_files"] = bad
        outputs["beat2_genmo"] = {"summary": summary, "stats": stats}

    if args.dataset in ["all", "amass_genmo"]:
        amass_root = OmegaConf.select(cfg, "DATASET.AMASS_Genmo.ROOT")
        pose_rep = OmegaConf.select(cfg, "DATASET.AMASS_Genmo.pose_rep")
        files = collect_npz_files(amass_root, pose_rep, subdirs=None)
        stats, file_count, bad = compute_stats_for_files(files, dim, args.max_files, args.seed)
        summary = summarize("amass_genmo", stats, args.low_std, args.topk)
        summary["files"] = file_count
        summary["bad_files"] = bad
        outputs["amass_genmo"] = {"summary": summary, "stats": stats}

    for name, payload in outputs.items():
        summary = payload["summary"]
        print(f"\n[{name}]")
        print(f"  frames: {summary['frames']} files: {summary['files']} bad_files: {summary['bad_files']}")
        print(f"  mean_std: {summary['mean_std']:.6f} min_std: {summary['min_std']:.6f} max_std: {summary['max_std']:.6f}")
        print(f"  mean_range: {summary['mean_range']:.6f}")
        print(f"  low_std_ratio (<{args.low_std}): {summary['low_std_ratio']*100:.2f}%")
        print(f"  lowest_std_dims: {summary['lowest_std_dims']}")
        print(f"  highest_std_dims: {summary['highest_std_dims']}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, payload in outputs.items():
            stats = payload["stats"]
            np.save(out_dir / f"{name}_mean.npy", stats["mean"])
            np.save(out_dir / f"{name}_std.npy", stats["std"])
            np.save(out_dir / f"{name}_min.npy", stats["min"])
            np.save(out_dir / f"{name}_max.npy", stats["max"])
        with open(out_dir / "summary.json", "w") as f:
            json.dump({k: v["summary"] for k, v in outputs.items()}, f, indent=2)
        print(f"\nSaved stats to {out_dir}")


if __name__ == "__main__":
    main()
