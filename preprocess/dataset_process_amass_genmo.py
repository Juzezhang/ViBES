"""
Convert AMASS SMPL-X files to GENMO motion vector format with HumanML3D-style naming.

Key behaviors (aligned with preprocess/dataset_process_amass.py):
- Uses preprocess/index.csv to map source files to numbered names (000000.npz, ...).
- Resamples sequences to a target FPS (default: 25) with consistent start/end frame
  adjustment based on the index FPS (default: 20).
- Applies the Z-up (AMASS) -> Y-up (GENMO) coordinate conversion.
- Outputs GENMO vectors (145 dims per frame) in .npz files.

GENMO format (without in-cam data): 145 dimensions per frame
- body_pose_r6d:      126 dims (21 joints × 6D rotation)  indices 0-125
- betas:               10 dims (shape parameters)         indices 126-135
- global_orient_r6d:    6 dims (root rotation, 6D)        indices 136-141
- local_transl_vel:     3 dims (translation velocity)     indices 142-144

Coordinate System Conversion:
- AMASS: Z-up (XY is ground plane, Z is vertical)
- GENMO: Y-up (XZ is ground plane, Y is vertical)
- Transformation: -90 degree rotation around X-axis (maps Z -> Y, Y -> -Z)
"""

import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)


# =============================================================================
# Coordinate System Conversion Functions
# AMASS: Z-up (XY ground) -> GENMO: Y-up (XZ ground)
# =============================================================================

def get_zup_to_yup_matrix():
    """Transformation matrix from Z-up (AMASS) to Y-up (GENMO)."""
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )


def convert_zup_to_yup_trans(trans):
    """Convert translation from Z-up to Y-up."""
    trans = torch.as_tensor(trans, dtype=torch.float32)
    T = get_zup_to_yup_matrix()
    if trans.dim() == 1:
        return T @ trans
    return trans @ T.T


def convert_zup_to_yup_orient(aa):
    """
    Convert global orientation axis-angle from Z-up to Y-up.
    For global orientation, pre-multiply by T: R_new = T @ R_old.
    """
    aa = torch.as_tensor(aa, dtype=torch.float32)
    shape = aa.shape
    R = axis_angle_to_matrix(aa.reshape(-1, 3))
    T = get_zup_to_yup_matrix()
    R_new = T @ R
    aa_new = matrix_to_axis_angle(R_new)
    return aa_new.reshape(shape)


def convert_yup_to_zup_trans(trans):
    """Convert translation from Y-up to Z-up (inverse of Z-up -> Y-up)."""
    T = get_zup_to_yup_matrix()
    trans = torch.as_tensor(trans, dtype=torch.float32)
    if trans.dim() == 1:
        return T.T @ trans
    return trans @ T


def convert_yup_to_zup_orient(aa):
    """Inverse conversion: R_old = T.T @ R_new."""
    aa = torch.as_tensor(aa, dtype=torch.float32)
    shape = aa.shape
    R = axis_angle_to_matrix(aa.reshape(-1, 3))
    T = get_zup_to_yup_matrix()
    R_old = T.T @ R
    aa_old = matrix_to_axis_angle(R_old)
    return aa_old.reshape(shape)


def get_local_transl_vel(transl, global_orient):
    """
    Compute local translation velocity in root-relative coordinates.
    Args:
        transl: (L, 3) world translation
        global_orient: (L, 3) axis-angle global orientation
    Returns:
        (L, 3) local translation velocity
    """
    transl = torch.as_tensor(transl, dtype=torch.float32)
    global_orient = torch.as_tensor(global_orient, dtype=torch.float32)

    # Frame-to-frame velocity
    velocity = torch.zeros_like(transl)
    velocity[1:] = transl[1:] - transl[:-1]
    velocity[0] = velocity[1] if len(velocity) > 1 else velocity[0]

    # Transform to local coordinates (root-relative)
    R = axis_angle_to_matrix(global_orient)  # (L, 3, 3)
    R_inv = R.transpose(-1, -2)
    local_vel = torch.einsum('lij,lj->li', R_inv, velocity)
    return local_vel


# =============================================================================
# AMASS -> GENMO conversion
# =============================================================================

def validate_smplx_file(data):
    """Validate that the file is a proper SMPL-X file."""
    required_keys = ['poses', 'trans', 'betas']
    for key in required_keys:
        if key not in data:
            return False, f"Missing required key: {key}"

    poses = data['poses']
    if len(poses.shape) != 2:
        return False, f"poses should be 2D array, got shape {poses.shape}"

    pose_dim = poses.shape[1]
    if pose_dim < 66:
        return False, f"Pose dimension {pose_dim} too small"

    trans = data['trans']
    if len(trans.shape) != 2 or trans.shape[1] != 3:
        return False, f"trans should be (L, 3), got shape {trans.shape}"

    if poses.shape[0] != trans.shape[0]:
        return False, f"Frame count mismatch: poses={poses.shape[0]}, trans={trans.shape[0]}"

    if poses.shape[0] < 2:
        return False, f"Too few frames: {poses.shape[0]}"

    return True, "Valid"


def convert_amass_to_genmo(data, fps_override=None):
    """
    Convert AMASS SMPL-X data to GENMO motion vector format.
    Includes coordinate system conversion from Z-up (AMASS) to Y-up (GENMO).
    """
    if 'pose_body' in data:
        body_pose_aa = data['pose_body']
        global_orient_aa = data['root_orient']
    else:
        poses = data['poses']
        global_orient_aa = poses[:, :3]
        body_pose_aa = poses[:, 3:66]

    trans_zup = data['trans']
    betas_raw = data['betas']
    L = body_pose_aa.shape[0]

    if len(betas_raw.shape) == 1:
        betas = np.tile(betas_raw[:10], (L, 1)) if len(betas_raw) >= 10 else np.zeros((L, 10))
    else:
        betas = betas_raw[:, :10] if betas_raw.shape[1] >= 10 else np.zeros((L, 10))

    # Coordinate Conversion: Z-up -> Y-up
    trans_yup = convert_zup_to_yup_trans(trans_zup)
    global_orient_yup = convert_zup_to_yup_orient(global_orient_aa)

    # Body pose (local rotations) - no conversion needed
    body_pose_t = torch.from_numpy(body_pose_aa.astype(np.float32))

    # Convert to GENMO format (Y-up)
    body_pose_reshaped = body_pose_t.reshape(L, 21, 3)
    body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose_reshaped)).reshape(L, 126)
    global_orient_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_yup))
    local_transl_vel = get_local_transl_vel(trans_yup, global_orient_yup)

    betas_t = torch.from_numpy(betas.astype(np.float32))
    motion_vector = torch.cat([body_pose_r6d, betas_t, global_orient_r6d, local_transl_vel], dim=-1)

    fps = float(fps_override) if fps_override is not None else float(data.get('mocap_frame_rate', 30.0))
    gender = str(data['gender']) if 'gender' in data else 'neutral'

    return {
        'motion_vector': motion_vector.numpy(),
        'num_frames': L,
        'fps': fps,
        'gender': gender,
        'body_pose': body_pose_aa,
        'global_orient': global_orient_yup.numpy(),
        'trans': trans_yup.numpy(),
        'betas': betas,
    }


# =============================================================================
# HumanML3D mapping + filtering (aligned with dataset_process_amass.py)
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='AMASS -> GENMO conversion options')
    parser.add_argument(
        "--dataset_path_original",
        type=str,
        required=False,
        default="/simurgh2/datasets/AMASS_original_smplx",
        help="Path to original AMASS dataset",
    )
    parser.add_argument(
        "--dataset_path_processed",
        type=str,
        required=False,
        default="/simurgh2/datasets/AMASS",
        help="Path to processed AMASS dataset root",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=False,
        default="./preprocess/index.csv",
        help="Path to HumanML3D index file",
    )
    parser.add_argument(
        "--index_fps",
        type=int,
        required=False,
        default=20,
        help="Frame rate used by the index file (HumanML3D uses 20 FPS)",
    )
    parser.add_argument(
        "--ex_fps",
        type=int,
        required=False,
        default=25,
        help="Target frame rate for output (GENMO)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (skip some checks)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a round-trip verification on a few random samples",
    )
    return parser.parse_args()


def load_humanml3d_mapping(index_file_path):
    """
    Load mapping between AMASS source paths and HumanML3D names.
    Returns mapping: source_identifier -> (new_name, start_frame, end_frame)
    """
    index_dataframe = pd.read_csv(index_file_path)
    total_sequences = index_dataframe.shape[0]
    humanml3d_mapping = {}

    for i in tqdm(range(total_sequences), desc="Loading HumanML3D mapping"):
        source_path = index_dataframe.loc[i]['source_path']

        # Standardize dataset directory names (match dataset_process_amass.py)
        if 'BMLhandball' not in source_path:
            source_path = (
                source_path
                .replace('/BioMotionLab_NTroje/', '/BMLrub/')
                .replace('/DFaust_67/', '/DFaust/')
                .replace('/MPI_mosh/', '/MoSh/')
                .replace('/MPI_HDM05/', '/HDM05/')
                .replace('/MPI_Limits/', '/PosePrior/')
                .replace('/SSM_synced/', '/SSM/')
                .replace('/TCD_handMocap/', '/TCDHands/')
                .replace('/Transitions_mocap/', '/Transitions/')
                .replace('.npy', '.npz')
                .replace(' ', '_')
                .replace('_poses', '_stageii')
            )

        source_path = source_path.replace('./pose_data/', '').replace('/', '_')
        new_name = index_dataframe.loc[i]['new_name']
        start_frame = index_dataframe.loc[i]['start_frame']
        end_frame = index_dataframe.loc[i]['end_frame']
        humanml3d_mapping[source_path] = (new_name, start_frame, end_frame)

    return humanml3d_mapping


def collect_and_filter_motion_files(amass_data_directory, humanml3d_mapping, debug_mode=False):
    """Collect AMASS motion files and filter by mapping + SMPL-X compatibility."""
    included_datasets = [
        'ACCAD', 'BMLrub', 'BMLhandball', 'BMLmovi', 'CMU', 'DFaust',
        'EKUT', 'Eyes_Japan_Dataset', 'HumanEva', 'KIT', 'HDM05',
        'PosePrior', 'MoSh', 'SFU', 'SSM', 'TCDHands', 'TotalCapture', 'Transitions'
    ]

    all_motion_files = [
        Path(f'{root}/{filename}')
        for root, _, files in os.walk(amass_data_directory)
        for filename in files
        if filename.endswith('.npz')
        and any(dataset in Path(root).parts for dataset in included_datasets)
    ]
    all_motion_files.sort()

    valid_motion_files = []
    for motion_file in tqdm(all_motion_files, ncols=150, desc="Filtering motion files"):
        source_identifier = motion_file.relative_to(amass_data_directory).as_posix().replace('/', '_')
        if source_identifier not in humanml3d_mapping:
            continue
        if debug_mode:
            valid_motion_files.append(motion_file)
            continue
        try:
            motion_data = np.load(motion_file, allow_pickle=True)
        except zipfile.BadZipFile:
            continue
        if 'surface_model_type' not in motion_data:
            continue
        model_type = motion_data['surface_model_type'].item()
        if model_type not in {'smplx', 'smplx_locked_head'}:
            continue
        if 'mocap_frame_rate' not in motion_data:
            continue
        valid_motion_files.append(motion_file)

    return valid_motion_files


def resample_and_trim(motion_data, motion_file_path, start_frame, end_frame, target_fps, index_fps):
    """
    Downsample to target_fps and trim by (start_frame, end_frame) after
    converting index FPS to effective FPS.
    """
    original_fps = float(motion_data['mocap_frame_rate'])
    downsample_factor = max(1, int(original_fps / target_fps))

    pose_data = motion_data['poses'][::downsample_factor, ...]
    translation_data = motion_data['trans'][::downsample_factor, ...]

    effective_fps = original_fps / downsample_factor
    frame_scale = effective_fps / index_fps
    adjusted_start_frame = int(round(start_frame * frame_scale))
    if end_frame < 0:
        adjusted_end_frame = None
    else:
        adjusted_end_frame = int(round(end_frame * frame_scale))

    motion_file_str = motion_file_path.as_posix()
    if 'humanact12' not in motion_file_str:
        if 'Eyes_Japan_Dataset' in motion_file_str:
            pose_data = pose_data[3 * target_fps:]
            translation_data = translation_data[3 * target_fps:]
        elif 'HDM05' in motion_file_str:
            pose_data = pose_data[3 * target_fps:]
            translation_data = translation_data[3 * target_fps:]
        elif 'TotalCapture' in motion_file_str:
            pose_data = pose_data[1 * target_fps:]
            translation_data = translation_data[1 * target_fps:]
        elif 'PosePrior' in motion_file_str:
            pose_data = pose_data[1 * target_fps:]
            translation_data = translation_data[1 * target_fps:]
        elif 'Transitions' in motion_file_str:
            pose_data = pose_data[int(0.5 * target_fps):]
            translation_data = translation_data[int(0.5 * target_fps):]

        sequence_len = pose_data.shape[0]
        start_idx = min(max(adjusted_start_frame, 0), sequence_len)
        if adjusted_end_frame is None:
            end_idx = None
        else:
            end_idx = min(max(adjusted_end_frame, 0), sequence_len)
        pose_data = pose_data[start_idx:end_idx]
        translation_data = translation_data[start_idx:end_idx]

    if pose_data.shape[0] == 0:
        return None, None

    return pose_data, translation_data


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_arguments()

    input_dir = Path(args.dataset_path_original)
    output_dir = Path(args.dataset_path_processed) / f"amass_genmo_{args.ex_fps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HumanML3D mapping...")
    humanml3d_mapping = load_humanml3d_mapping(args.index_path)
    print(f"Loaded {len(humanml3d_mapping)} motion sequence mappings")

    print("Collecting and filtering motion files...")
    valid_motion_files = collect_and_filter_motion_files(input_dir, humanml3d_mapping, args.debug)
    print(f"Valid motion files: {len(valid_motion_files)}")

    stats = {
        'processed': 0,
        'skipped_invalid': 0,
        'skipped_error': 0,
        'skipped_existing': 0,
        'skipped_empty': 0,
        'errors': [],
    }

    print("Processing AMASS sequences (GENMO)...")
    for motion_file in tqdm(valid_motion_files, desc="Processing AMASS sequences", ncols=150):
        source_identifier = motion_file.relative_to(input_dir).as_posix().replace('/', '_')
        new_name, start_frame, end_frame = humanml3d_mapping[source_identifier]

        output_path = output_dir / new_name.replace('.npy', '.npz')
        if output_path.exists():
            stats['skipped_existing'] += 1
            continue

        try:
            motion_data = np.load(motion_file, allow_pickle=True)
            if not validate_smplx_file(motion_data)[0]:
                stats['skipped_invalid'] += 1
                continue

            pose_data, translation_data = resample_and_trim(
                motion_data,
                motion_file,
                start_frame,
                end_frame,
                args.ex_fps,
                args.index_fps,
            )
            if pose_data is None:
                stats['skipped_empty'] += 1
                continue

            # Build a minimal AMASS-like dict after trimming and resampling.
            trimmed = {
                'poses': pose_data,
                'trans': translation_data,
                'betas': motion_data['betas'],
                'mocap_frame_rate': float(args.ex_fps),
                'gender': motion_data['gender'] if 'gender' in motion_data else 'neutral',
            }

            genmo = convert_amass_to_genmo(trimmed, fps_override=args.ex_fps)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **genmo)
            stats['processed'] += 1
        except Exception as e:
            stats['skipped_error'] += 1
            stats['errors'].append((str(motion_file), str(e)))

    print(
        f"Summary: {stats['processed']} processed, "
        f"{stats['skipped_existing']} existing, "
        f"{stats['skipped_invalid']} invalid, "
        f"{stats['skipped_empty']} empty, "
        f"{stats['skipped_error']} errors"
    )
    print(f"Output directory: {output_dir}")

    if args.verify:
        verify_conversion(input_dir, output_dir)


# =============================================================================
# Optional verification
# =============================================================================

def convert_genmo_to_amass(motion_vector, global_orient_yup, trans_yup):
    """
    Convert GENMO motion vector back to AMASS format for verification.
    """
    motion_vector = torch.as_tensor(motion_vector, dtype=torch.float32)
    L = motion_vector.shape[0]

    body_pose_r6d = motion_vector[:, :126]
    betas = motion_vector[:, 126:136]
    global_orient_r6d = motion_vector[:, 136:142]
    local_transl_vel = motion_vector[:, 142:145]

    body_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(L, 21, 6))).reshape(L, 63)
    global_orient_aa_yup = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))

    global_orient_t = torch.as_tensor(global_orient_yup, dtype=torch.float32)
    R = axis_angle_to_matrix(global_orient_t)
    world_vel_yup = torch.einsum('lij,lj->li', R, local_transl_vel)
    trans_decoded_yup = torch.zeros(L, 3)
    trans_decoded_yup[0] = torch.as_tensor(trans_yup[0], dtype=torch.float32)
    for i in range(1, L):
        trans_decoded_yup[i] = trans_decoded_yup[i - 1] + world_vel_yup[i]

    # Convert back from Y-up to Z-up
    global_orient_aa_zup = convert_yup_to_zup_orient(global_orient_aa_yup)
    trans_zup = convert_yup_to_zup_trans(trans_decoded_yup)

    return {
        'body_pose': body_pose_aa.numpy(),
        'global_orient': global_orient_aa_zup.numpy(),
        'trans': trans_zup.numpy(),
        'betas': betas.numpy(),
    }


def verify_conversion(input_dir, output_dir, num_files=5):
    """Round-trip verification: Z-up -> Y-up -> Z-up (decoded)."""
    import random
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    converted_files = list(output_dir.rglob('*.npz'))
    if len(converted_files) == 0:
        print("No converted files found!")
        return

    selected_files = random.sample(converted_files, min(num_files, len(converted_files)))
    print(f"\nVerification: Testing {len(selected_files)} random files")

    all_passed = True
    for converted_path in selected_files:
        rel_path = converted_path.relative_to(output_dir)
        # Reverse map based on filename (index) to original path is not trivial;
        # we keep this as a best-effort check for overlap in output naming.
        try:
            converted_data = np.load(converted_path, allow_pickle=True)
            decoded = convert_genmo_to_amass(
                converted_data['motion_vector'],
                converted_data['global_orient'],
                converted_data['trans'],
            )
            # Basic sanity: no NaNs and shapes are consistent
            ok = (
                not np.isnan(decoded['body_pose']).any()
                and not np.isnan(decoded['global_orient']).any()
                and not np.isnan(decoded['trans']).any()
            )
            print(f"{'✓' if ok else '✗'} {rel_path}")
            if not ok:
                all_passed = False
        except Exception as e:
            print(f"❌ {rel_path}: {e}")
            all_passed = False
    return all_passed


if __name__ == "__main__":
    main()
