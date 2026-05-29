"""Aggregate multiple Global-VAE eval JSONs into a comparison table + bar chart.

Each input JSON is produced by scripts/eval_global_vae_translation.py and self-describes its
`label` and `dataset`. A JSON with a `refined` block (V3 foot-contact refine) contributes an
extra "<label> (refined)" row. Use this to compare V1 (released) / V2 (retrained) / V3
(velocity + foot contacts + refine).

    python scripts/compare_global_vae.py run1.json run2.json ... [--out compare.png] [--md compare.md]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rows_from_json(path):
    d = json.loads(Path(path).read_text())
    label = d.get("label", Path(path).stem)
    ds = d.get("dataset", "?")
    base = {
        "label": label, "dataset": ds,
        "trans_err": d["trans_err_mm_mean"], "final_drift": d["final_drift_mm_mean"],
        "x": d["trans_err_axis_mm"]["x"], "y": d["trans_err_axis_mm"]["y"],
        "z": d["trans_err_axis_mm"]["z"], "vel_mae": d.get("local_vel_mae_mean", float("nan")),
    }
    rows = [base]
    if "refined" in d:
        r = d["refined"]
        rows.append({
            "label": label + " (refined)", "dataset": ds,
            "trans_err": r["trans_err_mm_mean"], "final_drift": r["final_drift_mm_mean"],
            "x": r["trans_err_axis_mm"]["x"], "y": r["trans_err_axis_mm"]["y"],
            "z": r["trans_err_axis_mm"]["z"], "vel_mae": base["vel_mae"],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="eval JSON files")
    ap.add_argument("--out", default="/path/to/viz_demos/global_vae_train/compare.png")
    ap.add_argument("--md", default="/path/to/viz_demos/global_vae_train/compare.md")
    args = ap.parse_args()

    rows = []
    for p in args.jsons:
        rows.extend(_rows_from_json(p))

    # ---- markdown table ----
    hdr = "| version | dataset | trans_err (mm) | final_drift (mm) | x | y | z | vel_mae |"
    sep = "|" + "---|" * 8
    lines = [hdr, sep]
    for r in rows:
        lines.append(f"| {r['label']} | {r['dataset']} | {r['trans_err']:.1f} | {r['final_drift']:.1f} "
                     f"| {r['x']:.1f} | {r['y']:.1f} | {r['z']:.1f} | {r['vel_mae']:.5f} |")
    md = "\n".join(lines)
    print(md)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("# Global VAE 3-way comparison\n\n" + md + "\n")

    # ---- grouped bar chart: trans_err + final_drift per version, faceted by dataset ----
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        dr = [r for r in rows if r["dataset"] == ds]
        labels = [r["label"] for r in dr]
        x = np.arange(len(dr)); w = 0.38
        ax.bar(x - w / 2, [r["trans_err"] for r in dr], w, label="trans err (mean)")
        ax.bar(x + w / 2, [r["final_drift"] for r in dr], w, label="final drift")
        for i, r in enumerate(dr):
            ax.text(i - w / 2, r["trans_err"], f"{r['trans_err']:.0f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + w / 2, r["final_drift"], f"{r['final_drift']:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("error (mm)"); ax.set_title(ds); ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle("Global VAE translation error — version comparison")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"\nSaved table -> {args.md}\nSaved chart -> {args.out}")


if __name__ == "__main__":
    main()
