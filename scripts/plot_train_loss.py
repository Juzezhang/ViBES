"""Plot a training loss curve from a ViBES tokenizer training log.

Parses ``Epoch <N>: loss_total <val>`` lines (what train_tokenizer.py logs) — works on a live
sbatch ``.out`` while training is still running.

    python scripts/plot_train_loss.py <train_log> [--out curve.png] [--title T]
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"Epoch (\d+): loss_total ([0-9.eE+-]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    ep, ls = [], []
    for line in Path(args.log).open(errors="ignore"):
        m = PAT.search(line)
        if m:
            ep.append(int(m.group(1))); ls.append(float(m.group(2)))
    if not ep:
        print("No 'Epoch N: loss_total ...' lines found yet.")
        return
    print(f"parsed {len(ep)} epochs; last: epoch {ep[-1]} loss {ls[-1]:.4e}")

    out = args.out or str(Path(args.log).with_suffix(".loss.png"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, ls, marker="o", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss_total"); ax.set_yscale("log")
    ax.set_title(args.title or f"training loss — {Path(args.log).name} (epoch {ep[-1]})")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
