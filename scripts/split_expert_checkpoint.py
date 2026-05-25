"""Split a full ViBES Stage-2 checkpoint into an Expert-1-only (motion) checkpoint.

A full checkpoint bundles Expert-0 (text/audio, == the GLM-4-Voice base) + Expert-1 (motion).
Expert-0 is frozen and redundant with the GLM-4-Voice base, so this keeps ONLY Expert-1
(~half the size). Load the result with ``training/expert_io.py:load_expert1_checkpoint``,
which rebuilds Expert-0 from the GLM-4-Voice base and merges in Expert-1.

Usage:
    python scripts/split_expert_checkpoint.py --input <full_ckpt_dir> --output <expert1_dir>
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from expert_io import filter_expert1_state_dict, load_state_dict_from_dir, write_expert_marker  # noqa: E402

# Files that belong to the (discarded) full model / training state, not the per-expert model.
_SKIP = {"model.safetensors.index.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
         "trainer_state.json", "training_args.bin"}


def main():
    ap = argparse.ArgumentParser(description="Keep only the motion Expert-1 weights of a ViBES checkpoint.")
    ap.add_argument("--input", required=True, help="Full checkpoint dir (both experts).")
    ap.add_argument("--output", required=True, help="Output dir for the Expert-1-only checkpoint.")
    ap.add_argument("--base", default="glm-4-voice-9b",
                    help="Name of the GLM-4-Voice base that provides Expert-0 (recorded in the marker).")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading full state_dict from {args.input} ...")
    full = load_state_dict_from_dir(args.input)
    e1 = filter_expert1_state_dict(full)
    sz_full = sum(v.numel() * v.element_size() for v in full.values()) / 1e9
    sz_e1 = sum(v.numel() * v.element_size() for v in e1.values()) / 1e9
    print(f"  full: {len(full)} tensors ({sz_full:.1f} GB)  ->  Expert-1: {len(e1)} tensors ({sz_e1:.1f} GB)")
    if not e1:
        raise RuntimeError("No Expert-1 params matched — is this a ViBES MotExpertNum2 checkpoint?")

    from safetensors.torch import save_file
    e1 = {k: v.contiguous().cpu() for k, v in e1.items()}
    out_st = os.path.join(args.output, "model.safetensors")
    save_file(e1, out_st, metadata={"format": "pt"})
    print(f"  wrote {out_st}")

    # Copy config / modeling code / tokenizer / generation config (skip full-model weights + trainer state).
    for fn in os.listdir(args.input):
        if fn in _SKIP or fn.endswith(".safetensors") or fn.startswith("pytorch_model"):
            continue
        src = os.path.join(args.input, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output, fn))

    write_expert_marker(args.output, base=args.base)
    print(f"✓ Expert-1-only checkpoint written to {args.output}")
    print("  Load it with training/expert_io.py:load_expert1_checkpoint(model, <dir>, <glm_base_path>).")


if __name__ == "__main__":
    main()
