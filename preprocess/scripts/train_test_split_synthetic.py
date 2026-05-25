#!/usr/bin/env python3
"""
Assign subsequences (e.g., 202008675_0018) to train/val/test according to
predefined *base-ID* splits, where each split is first merged from its
processed & unprocessed lists.

Merges (under --splits-dir):
  - train_processed.txt + train_unprocessed.txt  -> TRAIN base IDs
  - val_processed.txt   + val_unprocessed.txt    -> VAL   base IDs
  - test_processed.txt  + test_unprocessed.txt   -> TEST  base IDs

Then scans --audio-dir for *.wav like 202008675_0018.wav, extracts base ID
(the part before the first underscore), and assigns the subsequence to the
corresponding split.

Outputs (to --output-dir, default: parent of --audio-dir):
  - train.txt
  - val.txt
  - test.txt
Each contains one subsequence stem per line (no extension).
"""

from pathlib import Path
import argparse
import sys
from typing import Set, Tuple, Dict, List

def read_id_list(path: Path) -> Set[str]:
    """Read base IDs (one per line). Ignores blank lines and lines starting with '#'. """
    out: Set[str] = set()
    if not path.exists():
        return out
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line.split()[0])
    return out

def merge_two_lists(a: Path, b: Path) -> Set[str]:
    """Union of two ID files."""
    return read_id_list(a) | read_id_list(b)

def check_overlaps(train_ids: Set[str], val_ids: Set[str], test_ids: Set[str]) -> Dict[str, Set[str]]:
    """Return any overlaps among the three sets."""
    overlaps = {
        "train&val": train_ids & val_ids,
        "train&test": train_ids & test_ids,
        "val&test": val_ids & test_ids,
    }
    return {k: v for k, v in overlaps.items() if v}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path,
        default=Path("/path/to/YouTube_Talking_Synthetic/audios"),
        help="Directory containing subsequence .wav files (e.g., 202008675_0018.wav).")
    ap.add_argument("--splits-dir", type=Path,
        default=Path("/path/to/conversational_agent/datasets/YouTube_Talking"),
        help="Directory containing train/val/test *_processed.txt and *_unprocessed.txt.")
    ap.add_argument("--output-dir", type=Path, default=None,
        help="Where to write train/val/test.txt. Default: parent of --audio-dir.")
    ap.add_argument("--ext", type=str, default=".wav",
        help="Audio extension to scan (default: .wav).")
    ap.add_argument("--unknown-policy", choices=["skip", "error"], default="skip",
        help="If a subsequence base ID is not present in any split: skip or raise (default: skip).")
    ap.add_argument("--overlap-policy", choices=["error", "train>val>test"], default="error",
        help="If a base ID appears in multiple splits: error (default) or resolve by priority train>val>test.")
    args = ap.parse_args()

    audio_dir: Path = args.audio_dir
    if not audio_dir.is_dir():
        print(f"[ERR] Audio directory does not exist: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or audio_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) Merge processed + unprocessed for each split ---
    sd = args.splits_dir
    train_ids = merge_two_lists(sd / "train_processed.txt", sd / "train_unprocessed.txt")
    val_ids   = merge_two_lists(sd / "val_processed.txt",   sd / "val_unprocessed.txt")
    test_ids  = merge_two_lists(sd / "test_processed.txt",  sd / "test_unprocessed.txt")

    if not any([train_ids, val_ids, test_ids]):
        print(f"[WARN] No base IDs loaded from {sd}. Check your split files.", file=sys.stderr)

    # --- 2) Handle overlaps if any ---
    overlaps = check_overlaps(train_ids, val_ids, test_ids)
    if overlaps:
        if args.overlap-policy == "error":
            for k, ids in overlaps.items():
                print(f"[ERR] Overlap between {k}: {sorted(list(ids))[:10]}{' ...' if len(ids)>10 else ''}",
                      file=sys.stderr)
            sys.exit(2)
        else:
            # Resolve by priority: train > val > test
            # Anything in TRAIN wins; remove from others.
            val_ids  -= train_ids
            test_ids -= train_ids
            # Anything in VAL then wins over TEST.
            test_ids -= val_ids

    # --- 3) Scan subsequences and assign by base ID ---
    wavs = sorted(p for p in audio_dir.glob(f"*{args.ext}") if p.is_file())
    stems = [p.stem for p in wavs]

    buckets = {"train": [], "val": [], "test": []}
    unknown: List[str] = []

    for stem in stems:
        base = stem.split("_", 1)[0]  # '202008675_0018' -> '202008675'
        if base in train_ids:
            buckets["train"].append(stem)
        elif base in val_ids:
            buckets["val"].append(stem)
        elif base in test_ids:
            buckets["test"].append(stem)
        else:
            unknown.append(stem)
            if args.unknown_policy == "error":
                raise RuntimeError(f"Base ID not found in any split: {base} (from {stem})")

    # --- 4) Sort & write outputs (stems only) ---
    for k in buckets:
        buckets[k].sort()

    (out_dir / "train.txt").write_text("\n".join(buckets["train"]) + ("\n" if buckets["train"] else ""))
    (out_dir / "val.txt").write_text("\n".join(buckets["val"]) + ("\n" if buckets["val"] else ""))
    (out_dir / "test.txt").write_text("\n".join(buckets["test"]) + ("\n" if buckets["test"] else ""))

    # --- 5) Report ---
    total = len(stems)
    assigned = sum(len(v) for v in buckets.values())
    print(f"[OK] Scanned {total} subsequences in {audio_dir}")
    print(f"[OK] Assigned: train {len(buckets['train'])}, val {len(buckets['val'])}, test {len(buckets['test'])}")
    if unknown:
        print(f"[WARN] Unassigned subsequences: {len(unknown)} (policy={args.unknown_policy}). "
              f"Example: {unknown[:5]}")
    print(f"[OK] Wrote:\n  {out_dir/'train.txt'}\n  {out_dir/'val.txt'}\n  {out_dir/'test.txt'}")

if __name__ == "__main__":
    main()