#!/usr/bin/env python3
"""
Build the Converse3D Q2M evaluation set (motion_segments/) from your own BEAT2 and AMASS_talking.

The released split files ({train,val,test}.jsonl[.gz]) list every question with the motion segment it
belongs to. This script cuts those segments out of the preprocessed source datasets:

    beat2  ->  <BEAT2_ROOT>/smplxflame_25/<source_id>.npz, frames [start_frame_25fps, end_frame_25fps)
    amass  ->  <AMASS_TALKING_ROOT>/amass_data_align_25_audios_rotation/<id>.npz (whole clip)

Produce the two source folders with docs/1-data/beat2.md (Steps 1-3) and docs/1-data/amass.md
(Steps 1-5). The output is the layout scripts/eval_vibes_body_q2m.py reads:

    <output_root>/{train,val,test}.jsonl
    <output_root>/motion_segments/{train,val,test}/<segment_id>.npz

Usage:
    python preprocess/build_converse3d_eval.py \
        --jsonl_dir          <downloaded q2m_eval/> \
        --beat2_root         <BEAT2_ROOT> \
        --amass_talking_root <AMASS_TALKING_ROOT> \
        --output_root        <CONVERSE3D_EVAL_ROOT> \
        [--splits test]
"""
import argparse
import gzip
import json
import os
import shutil

import numpy as np
from tqdm import tqdm

BEAT2_SLICED_KEYS = ("poses", "expressions", "trans")
AMASS_DIR = "amass_data_align_25_audios_rotation"


def beat2_segment(record, beat2_root):
    src = np.load(os.path.join(beat2_root, "smplxflame_25", f"{record['source_id']}.npz"), allow_pickle=True)
    start, end = record["start_frame_25fps"], record["end_frame_25fps"]
    # A few segments end past the last motion frame (the split came from the audio timeline);
    # slicing stops at the end of the motion, as in the released set.
    return {k: (src[k][start:end] if k in BEAT2_SLICED_KEYS else src[k]) for k in src.files}


def amass_source(record, amass_talking_root):
    clip_id = record["segment_id"][len("amass_"):]       # e.g. 004822 or M004822 (mirrored)
    return os.path.join(amass_talking_root, AMASS_DIR, f"{clip_id}.npz")


def read_split_file(jsonl_dir, split):
    """Read <split>.jsonl or <split>.jsonl.gz; returns the raw text."""
    path = os.path.join(jsonl_dir, f"{split}.jsonl")
    if os.path.exists(path):
        return open(path).read()
    with gzip.open(path + ".gz", "rt") as f:
        return f.read()


def build_split(split, args):
    text = read_split_file(args.jsonl_dir, split)
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    out_dir = os.path.join(args.output_root, "motion_segments", split)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args.output_root, f"{split}.jsonl"), "w") as f:
        f.write(text)

    missing = []
    for record in tqdm(records, desc=split):
        out = os.path.join(out_dir, f"{record['segment_id']}.npz")
        if os.path.exists(out) and not args.overwrite:
            continue
        try:
            if record["source"] == "beat2":
                np.savez(out, **beat2_segment(record, args.beat2_root))
            elif record["source"] == "amass":
                shutil.copyfile(amass_source(record, args.amass_talking_root), out)
            else:
                raise ValueError(f"unknown source {record['source']}")
        except (FileNotFoundError, ValueError) as e:
            missing.append((record["segment_id"], str(e)))
    print(f"{split}: {len(records) - len(missing)}/{len(records)} segments written to {out_dir}")
    for seg_id, err in missing[:10]:
        print(f"  missing {seg_id}: {err}")
    return len(missing)


def main():
    parser = argparse.ArgumentParser(description="Build Converse3D motion_segments/ from BEAT2 + AMASS_talking.")
    parser.add_argument("--jsonl_dir", required=True,
                        help="Folder with the released {train,val,test}.jsonl (or .jsonl.gz)")
    parser.add_argument("--beat2_root", required=True, help="BEAT2 root containing smplxflame_25/")
    parser.add_argument("--amass_talking_root", required=True,
                        help=f"AMASS_talking root containing {AMASS_DIR}/")
    parser.add_argument("--output_root", required=True, help="Where to write the evaluation set")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        help="Splits to build (test is enough for evaluation; train/val are for "
                             "training the Q2M evaluator)")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite segments that already exist")
    args = parser.parse_args()
    n_missing = sum(build_split(split, args) for split in args.splits)
    if n_missing:
        raise SystemExit(f"{n_missing} segments could not be built; check the source folders above")


if __name__ == "__main__":
    main()
