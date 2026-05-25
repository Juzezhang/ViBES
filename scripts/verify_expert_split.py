"""Verify that an Expert-1-only split is a lossless representation of a full ViBES checkpoint.

The merged model used at load time is ``{reconstructed Expert-0 (GLM base, renamed)} +
{Expert-1 (from the split)}``. The split's Expert-1 is a literal byte-subset of the full
checkpoint, so the only thing that can differ is Expert-0. This script confirms that the
GLM-base-reconstructed Expert-0 is *identical* to the full checkpoint's (frozen) Expert-0 —
which guarantees the merged model is bit-for-bit the same as the original full checkpoint
(no GPU / forward pass needed).

    python scripts/verify_expert_split.py --full ViBES-Face --glm_base_path <glm-4-voice-9b dir>
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from expert_io import (  # noqa: E402
    glm_base_to_expert0_state_dict,
    is_expert1_param,
    load_state_dict_from_dir,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="Full (both-expert) checkpoint dir, e.g. ViBES-Face.")
    ap.add_argument("--glm_base_path", required=True, help="GLM-4-Voice base dir / HF id.")
    args = ap.parse_args()

    print(f"Loading full checkpoint Expert-0 from {args.full} ...")
    full = load_state_dict_from_dir(args.full)
    e0_ref = {k: v for k, v in full.items() if not is_expert1_param(k)}
    print(f"  Expert-0 in checkpoint: {len(e0_ref)} tensors")

    print(f"Reconstructing Expert-0 from GLM base {args.glm_base_path} ...")
    e0_recon = glm_base_to_expert0_state_dict(args.glm_base_path, torch_dtype=torch.bfloat16)
    print(f"  Expert-0 from GLM base : {len(e0_recon)} tensors")

    ref_keys, recon_keys = set(e0_ref), set(e0_recon)
    missing = ref_keys - recon_keys          # in checkpoint, not reconstructed
    extra = recon_keys - ref_keys            # reconstructed but not in checkpoint (e.g. rotary buffers)
    common = ref_keys & recon_keys

    n_mismatch, max_diff = 0, 0.0
    worst = None
    for k in sorted(common):
        a, b = e0_ref[k].float(), e0_recon[k].float()
        if a.shape != b.shape:
            n_mismatch += 1
            continue
        d = (a - b).abs().max().item()
        if d > max_diff:
            max_diff, worst = d, k
        if d != 0.0:
            n_mismatch += 1

    print("\n===== RESULT =====")
    print(f"  common Expert-0 tensors : {len(common)}")
    print(f"  in-ckpt-not-reconstructed: {len(missing)}  {sorted(missing)[:5]}")
    print(f"  reconstructed-only       : {len(extra)}   {sorted(extra)[:5]} (rotary buffers OK)")
    print(f"  value-mismatched tensors : {n_mismatch}")
    print(f"  max abs diff (Expert-0)  : {max_diff:.3e}  (worst: {worst})")

    ok = (len(missing) == 0 and n_mismatch == 0 and max_diff == 0.0)
    print("\n  " + ("✅ LOSSLESS: reconstructed Expert-0 == checkpoint Expert-0 (bit-identical merge)."
                     if ok else
                     "❌ MISMATCH: Expert-0 differs — the per-expert split is NOT lossless. Investigate."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
