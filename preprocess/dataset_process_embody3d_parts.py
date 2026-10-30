#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Embody3D raw SMPL-X captures into part-wise motion arrays (upper / lower / hand).

Embody3D (Meta) ships each participant's SMPL-X motion as SEPARATE per-component .npy files
under ``<capture>/<participant>/smplx_mesh_*/`` (30 fps):

    smplx_mesh_global_orient   (n, 3)
    smplx_mesh_body_pose       (n, 63)     # 21 body joints
    smplx_mesh_jaw_pose        (n, 3)      # face; absent/zero for body-only subsets
    smplx_mesh_leye_pose       (n, 3)
    smplx_mesh_reye_pose       (n, 3)
    smplx_mesh_left_hand_pose  (n, 45)     # 15 joints
    smplx_mesh_right_hand_pose (n, 45)
    smplx_mesh_transl          (n, 3)
    smplx_mesh_betas           (n, 300)
    (smplx_mesh_expression     (n, 100))   # not used here
    missing/<clip>.npy         (n, 1)      # 1 = good tracking, 0 = drop frame

This script assembles those into the SMPL-X ``poses`` (n, 165) + ``trans`` (n, 3) layout
    poses = [global_orient(3) | body_pose(63) | jaw(3) | leye(3) | reye(3) | lhand(45) | rhand(45)]
and runs the same part split as the ViBES body pipeline (ported from the conversational-agent
``process_smplx_npz``), producing per clip:

    upper : (n, 78)   = 13 upper joints × 6D
    lower : (n, 61)   = 9 leg joints × 6D (54) + root trans (3) + foot contacts (4, zeros)
    hand  : (n, 180)  = 30 hand joints × 6D

The (n,61) lower matches the Global VAE / lower-VQ input convention (contacts left as zeros here,
as in the reference recipe; the Global VAE only consumes the rotations + integrates velocity).

Output: one ``.npz`` per (capture, participant) under ``<output_dir>`` with keys
``upper, lower, hand, trans, betas, mocap_frame_rate, id_name`` — the layout the ViBES motion
tokenizer (``preprocess/scripts/get_compositional_motion_code.py``) consumes.

Usage:
    python -m preprocess.dataset_process_embody3d_parts \
        --data_root  <EMBODY3D_ROOT>/datasets/dyadic \
        --output_dir <EMBODY3D_ROOT>/embody3d_parts_25 \
        [--scene_filter AIAGENT]      # only captures whose name contains this (aiagent subset)
        [--drop_missing]              # drop frames flagged bad by missing/
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
from scipy import interpolate
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))  # repo root

from preprocess.utils.rotation_conversions import axis_angle_to_6d_np  # noqa: E402
from multimodal_tokenizers.data.mixed_dataset.data_tools import (  # noqa: E402
    JOINT_MASK_FULL,
    JOINT_MASK_UPPER,
    JOINT_MASK_LOWER,
)

# SMPL-X pose layout (axis-angle), 55 joints × 3 = 165
_POSE_SLICES = [
    ("smplx_mesh_global_orient", 3),
    ("smplx_mesh_body_pose", 63),
    ("smplx_mesh_jaw_pose", 3),
    ("smplx_mesh_leye_pose", 3),
    ("smplx_mesh_reye_pose", 3),
    ("smplx_mesh_left_hand_pose", 45),
    ("smplx_mesh_right_hand_pose", 45),
]  # sums to 165


def assemble_poses(participant_dir, clip_stem):
    """Assemble Embody3D per-component .npy files into SMPL-X poses (n,165) + trans (n,3).

    Face components (jaw / leye / reye) are zero-filled when absent (body-only subsets).
    Returns (poses, trans, betas, missing) or None if the required components are missing.
    """
    def _load(comp):
        p = os.path.join(participant_dir, comp, clip_stem + ".npy")
        return np.load(p).astype(np.float64) if os.path.exists(p) else None

    cols, n = [], None
    for comp, dim in _POSE_SLICES:
        a = _load(comp)
        if a is None:
            if comp in ("smplx_mesh_jaw_pose", "smplx_mesh_leye_pose", "smplx_mesh_reye_pose"):
                a = None  # fill later once n is known
            else:
                return None  # a required body/hand/orient component is missing
        elif n is None:
            n = a.shape[0]
        cols.append((comp, dim, a))

    if n is None:
        return None
    poses_parts = []
    for comp, dim, a in cols:
        poses_parts.append(np.zeros((n, dim), dtype=np.float64) if a is None else a)
    poses = np.concatenate(poses_parts, axis=1)
    assert poses.shape[1] == 165, f"assembled poses must be 165-D, got {poses.shape[1]}"

    trans = _load("smplx_mesh_transl")
    if trans is None:
        return None
    betas = _load("smplx_mesh_betas")
    betas = betas[0, :10] if betas is not None else np.zeros(10, dtype=np.float64)

    miss_p = os.path.join(participant_dir, "missing", clip_stem + ".npy")
    missing = np.load(miss_p).reshape(-1).astype(bool) if os.path.exists(miss_p) else None
    return poses, trans, betas, missing


def resample_fps(arr, src_fps, tgt_fps):
    """Cubic-interp resample along time (axis 0), matching ViBES's beat2/amass 30->25 converter."""
    if src_fps == tgt_fps:
        return arr
    n = arr.shape[0]
    duration = (n - 1) / float(src_fps)
    n_tgt = int(np.floor(duration * tgt_fps)) + 1
    t_src = np.linspace(0, duration, n)
    t_tgt = np.linspace(0, duration, n_tgt)
    return interpolate.interp1d(t_src, arr, axis=0, kind="cubic",
                                bounds_error=False, fill_value="extrapolate")(t_tgt)


def process_smplx_npz(poses, trans):
    """SMPL-X poses(n,165)+trans(n,3) -> {upper(n,78), lower(n,61), hand(n,180)} 6D parts.

    Ported from the conversational-agent ``process_smplx_npz``. Uses the shared ViBES joint masks.
    """
    assert poses.ndim == 2 and poses.shape[1] == 165, f"poses must be (n,165), got {poses.shape}"
    n = poses.shape[0]
    assert trans.shape == (n, 3), f"trans must be (n,3), got {trans.shape}"

    full_mask_bool = JOINT_MASK_FULL.astype(bool)
    pose_processed = (poses * JOINT_MASK_FULL)[:, full_mask_bool]

    upper_filtered = JOINT_MASK_UPPER[full_mask_bool].astype(bool)
    tar_pose_upper = pose_processed[:, upper_filtered].reshape(n, 13, 3)
    tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

    lower_filtered = JOINT_MASK_LOWER[full_mask_bool].astype(bool)
    tar_pose_leg = pose_processed[:, lower_filtered].reshape(n, 9, 3)
    tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
    contacts = np.zeros((n, 4), dtype=trans.dtype)
    tar_pose_lower = np.concatenate([tar_pose_leg_6d, trans, contacts], axis=1)

    tar_pose_hands = pose_processed[:, 25 * 3:55 * 3].reshape(n, 30, 3)
    tar_pose_hands_6d = axis_angle_to_6d_np(tar_pose_hands).reshape(n, 30 * 6)

    assert tar_pose_upper_6d.shape == (n, 78)
    assert tar_pose_lower.shape == (n, 61)
    assert tar_pose_hands_6d.shape == (n, 180)
    return {"upper": tar_pose_upper_6d, "lower": tar_pose_lower, "hands": tar_pose_hands_6d}


def find_participants(data_root, scene_filter=None):
    """Yield (capture_dir, participant_id, clip_stem) for every participant in every capture."""
    for capture_dir in sorted(glob.glob(os.path.join(data_root, "c--*"))):
        cap = os.path.basename(capture_dir)
        if scene_filter and scene_filter.lower() not in cap.lower():
            continue
        # clip stem == capture dir name (Embody3D names the .npy after the full capture id)
        for participant in sorted(os.listdir(capture_dir)):
            pdir = os.path.join(capture_dir, participant)
            if not os.path.isdir(pdir) or not os.path.isdir(os.path.join(pdir, "smplx_mesh_transl")):
                continue
            yield capture_dir, participant, cap


def main():
    ap = argparse.ArgumentParser(description="Embody3D raw SMPL-X -> upper/lower/hand parts")
    ap.add_argument("--data_root", required=True,
                    help="Embody3D category dir containing c--* captures (e.g. .../datasets/dyadic)")
    ap.add_argument("--output_dir", required=True, help="Output dir for per-clip parts .npz")
    ap.add_argument("--scene_filter", default=None,
                    help="Only captures whose name contains this substring (e.g. AIAGENT for the aiagent subset)")
    ap.add_argument("--drop_missing", action="store_true",
                    help="Drop frames flagged bad (0) by the missing/ mask (applied before resampling)")
    ap.add_argument("--min_frames", type=int, default=16)
    ap.add_argument("--src_fps", type=int, default=30, help="Embody3D native fps")
    ap.add_argument("--target_fps", type=int, default=25,
                    help="Output fps (25 to match beat2_parts_25 / amass_parts_25)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = list(find_participants(args.data_root, args.scene_filter))
    print(f"Found {len(items)} (capture, participant) entries under {args.data_root}"
          + (f" [scene_filter={args.scene_filter}]" if args.scene_filter else ""))

    n_ok, n_skip = 0, 0
    for capture_dir, participant, clip_stem in tqdm(items, desc="embody3d->parts"):
        out_path = out / f"{clip_stem}__{participant}.npz"
        if out_path.exists():
            n_ok += 1
            continue
        try:
            res = assemble_poses(os.path.join(capture_dir, participant), clip_stem)
            if res is None:
                n_skip += 1
                continue
            poses, trans, betas, missing = res
            if args.drop_missing and missing is not None and missing.shape[0] == poses.shape[0]:
                poses, trans = poses[missing], trans[missing]
            if poses.shape[0] < args.min_frames or not np.isfinite(poses).all():
                n_skip += 1
                continue
            # 30 -> 25 fps to match the other ViBES datasets (beat2_parts_25 / amass_parts_25)
            poses = resample_fps(poses, args.src_fps, args.target_fps)
            trans = resample_fps(trans, args.src_fps, args.target_fps)
            parts = process_smplx_npz(poses, trans)
            np.savez_compressed(
                out_path,
                upper=parts["upper"].astype(np.float32),
                lower=parts["lower"].astype(np.float32),
                hand=parts["hands"].astype(np.float32),
                trans=trans.astype(np.float32),
                betas=betas.astype(np.float32),
                mocap_frame_rate=np.int64(args.mocap_frame_rate),
                id_name=f"{clip_stem}__{participant}",
            )
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_skip += 1
            if n_skip <= 10:
                print(f"  skip {clip_stem}/{participant}: {e}")

    print(f"\nDone: {n_ok} written, {n_skip} skipped -> {out}")


if __name__ == "__main__":
    main()
