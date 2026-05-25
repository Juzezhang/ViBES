"""
Convert AMASS SMPL-X files into part-wise motion arrays (upper/lower/face/hand).

Outputs per sequence (.npz):
  - upper: 78D (13 joints × 6D rotation)
  - lower: 71D (54D lower pose + 3D local vel + 4D contact + 10D betas)
  - face: 106D (jaw 6D + 100 expr) -> AMASS uses zeros
  - face_with_head: 112D (head 6D + jaw 6D + 100 expr) -> AMASS zeros
  - hand: 180D (30 joints × 6D) -> AMASS zeros
  - global_orient for reference

Coordinate System Conversion:
  - AMASS: Z-up (XY is ground plane, Z is vertical)
  - Output: Y-up (XZ is ground plane, Y is vertical)

Usage:
    python dataset_process_amass_parts.py [--output_dir OUTPUT_DIR]
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import smplx
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_axis_angle,
)

# =============================================================================
# Joint Masks
# =============================================================================

# Upper body joint indices in full SMPLX (13 joints)
# spine1=3, spine2=6, spine3=9, neck=12, left_collar=13, right_collar=14,
# head=15, left_shoulder=16, right_shoulder=17, left_elbow=18, right_elbow=19,
# left_wrist=20, right_wrist=21
UPPER_JOINT_INDICES = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

# Lower body joint indices (9 joints)
# pelvis=0, left_hip=1, right_hip=2, left_knee=4, right_knee=5,
# left_ankle=7, right_ankle=8, left_foot=10, right_foot=11
LOWER_JOINT_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11]


def create_joint_mask(joint_indices, total_dims=66):
    """Create a boolean mask for specified joints in AMASS format."""
    mask = np.zeros(total_dims, dtype=bool)
    for joint_idx in joint_indices:
        mask[joint_idx * 3:(joint_idx + 1) * 3] = True
    return mask


# AMASS uses 66 dims for global_orient(3) + body_pose(63) = 22 joints
JOINT_MASK_UPPER_AMASS = create_joint_mask(UPPER_JOINT_INDICES, 66)
JOINT_MASK_LOWER_AMASS = create_joint_mask(LOWER_JOINT_INDICES, 66)


# =============================================================================
# Coordinate System Conversion Functions
# =============================================================================

def get_zup_to_yup_matrix():
    """Transformation matrix from Z-up (AMASS) to Y-up."""
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
    """Convert global orientation axis-angle from Z-up to Y-up."""
    aa = torch.as_tensor(aa, dtype=torch.float32)
    shape = aa.shape
    R = axis_angle_to_matrix(aa.reshape(-1, 3))
    T = get_zup_to_yup_matrix()
    R_new = T @ R
    aa_new = matrix_to_axis_angle(R_new)
    return aa_new.reshape(shape)


# =============================================================================
# Helper Functions
# =============================================================================

def get_local_transl_vel(transl, global_orient):
    """Compute local translation velocity in root-relative coordinates."""
    transl = torch.as_tensor(transl, dtype=torch.float32)
    global_orient = torch.as_tensor(global_orient, dtype=torch.float32)

    velocity = torch.zeros_like(transl)
    velocity[1:] = transl[1:] - transl[:-1]
    velocity[0] = velocity[1] if len(velocity) > 1 else velocity[0]

    R = axis_angle_to_matrix(global_orient)
    R_inv = R.transpose(-1, -2)
    local_vel = torch.einsum('lij,lj->li', R_inv, velocity)
    return local_vel


def axis_angle_to_6d_np(aa):
    """Convert axis-angle to 6D rotation representation."""
    aa = torch.from_numpy(aa.astype(np.float32))
    R = axis_angle_to_matrix(aa)
    r6d = matrix_to_rotation_6d(R)
    return r6d.numpy()


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


# =============================================================================
# Foot Contact Computation
# =============================================================================

def compute_foot_contacts(smplx_model, poses, trans, betas, max_batch=128):
    """Compute foot contacts from motion data using SMPLX."""
    n = poses.shape[0]

    betas_raw = betas
    beta_dim = betas_raw.shape[-1] if len(betas_raw.shape) > 0 else 0
    if beta_dim == 16:
        padded_betas = np.zeros(300)
        padded_betas[:16] = betas_raw if len(betas_raw.shape) == 1 else betas_raw[0]
    elif beta_dim >= 300:
        padded_betas = betas_raw[:300] if len(betas_raw.shape) == 1 else betas_raw[0, :300]
    else:
        padded_betas = np.zeros(300)
        if len(betas_raw.shape) == 1:
            padded_betas[:beta_dim] = betas_raw
        else:
            padded_betas[:min(beta_dim, 300)] = betas_raw[0, :min(beta_dim, 300)]

    betas_tiled = np.tile(padded_betas.reshape(1, 300), (n, 1))
    betas_tensor = torch.from_numpy(betas_tiled.astype(np.float32)).cuda()
    poses_tensor = torch.from_numpy(poses.astype(np.float32)).cuda()
    trans_tensor = torch.from_numpy(trans.astype(np.float32)).cuda()

    s, r = n // max_batch, n % max_batch
    all_joints = []

    for i in range(s):
        start, end = i * max_batch, (i + 1) * max_batch
        batch_size = end - start
        with torch.no_grad():
            # Must pass all optional pose tensors with correct batch size
            # Otherwise SMPLX uses default batch=1 which causes concatenation errors
            output = smplx_model(
                betas=betas_tensor[start:end],
                transl=trans_tensor[start:end],
                expression=torch.zeros((batch_size, 100), dtype=torch.float32, device='cuda'),
                global_orient=poses_tensor[start:end, :3],
                body_pose=poses_tensor[start:end, 3:66],
                jaw_pose=torch.zeros((batch_size, 3), dtype=torch.float32, device='cuda'),
                leye_pose=torch.zeros((batch_size, 3), dtype=torch.float32, device='cuda'),
                reye_pose=torch.zeros((batch_size, 3), dtype=torch.float32, device='cuda'),
                left_hand_pose=torch.zeros((batch_size, 45), dtype=torch.float32, device='cuda'),
                right_hand_pose=torch.zeros((batch_size, 45), dtype=torch.float32, device='cuda'),
                return_joints=True,
            )
            joints = output['joints'][:, (7, 8, 10, 11), :].cpu()
        all_joints.append(joints)

    if r != 0:
        start = s * max_batch
        batch_size = r
        with torch.no_grad():
            output = smplx_model(
                betas=betas_tensor[start:start + r],
                transl=trans_tensor[start:start + r],
                expression=torch.zeros((batch_size, 100), dtype=torch.float32, device='cuda'),
                global_orient=poses_tensor[start:start + r, :3],
                body_pose=poses_tensor[start:start + r, 3:66],
                jaw_pose=torch.zeros((batch_size, 3), dtype=torch.float32, device='cuda'),
                leye_pose=torch.zeros((batch_size, 3), dtype=torch.float32, device='cuda'),
                reye_pose=torch.zeros((batch_size, 3), dtype=torch.float32, device='cuda'),
                left_hand_pose=torch.zeros((batch_size, 45), dtype=torch.float32, device='cuda'),
                right_hand_pose=torch.zeros((batch_size, 45), dtype=torch.float32, device='cuda'),
                return_joints=True,
            )
            joints = output['joints'][:, (7, 8, 10, 11), :].cpu()
        all_joints.append(joints)

    joints = torch.cat(all_joints, dim=0)
    joints = joints.permute(1, 0, 2)
    feetv = torch.zeros(joints.shape[0], joints.shape[1])
    feetv[:, 1:] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
    feetv[:, 0] = feetv[:, 1]

    contacts = (feetv < 0.01).numpy().astype(float)
    contacts = contacts.transpose(1, 0)

    return contacts


# =============================================================================
# Main Conversion Function
# =============================================================================

def convert_amass_to_upper_lower(data, smplx_model, beta_dim=10, fps_override=None):
    """
    Convert AMASS SMPL-X data to Upper+Lower body motion vector format.
    Includes coordinate system conversion from Z-up to Y-up.
    """
    if 'pose_body' in data:
        body_pose_aa = data['pose_body']
        global_orient_aa_zup = data['root_orient']
    else:
        poses = data['poses']
        global_orient_aa_zup = poses[:, :3]
        body_pose_aa = poses[:, 3:66]

    trans_zup = data['trans']
    betas_raw = data['betas']
    L = global_orient_aa_zup.shape[0]

    # Coordinate conversion: Z-up -> Y-up
    trans_yup = convert_zup_to_yup_trans(trans_zup).numpy()
    global_orient_yup = convert_zup_to_yup_orient(global_orient_aa_zup).numpy()

    # Reconstruct full pose in Y-up
    poses_yup = np.concatenate([global_orient_yup, body_pose_aa], axis=-1)

    # Extract upper body joints (13 joints)
    upper_pose_aa = poses_yup[:, JOINT_MASK_UPPER_AMASS].reshape(L, 13, 3)
    upper_pose_6d = axis_angle_to_6d_np(upper_pose_aa).reshape(L, 78)

    # Extract lower body joints (9 joints)
    lower_pose_aa = poses_yup[:, JOINT_MASK_LOWER_AMASS].reshape(L, 9, 3)
    lower_pose_6d = axis_angle_to_6d_np(lower_pose_aa).reshape(L, 54)

    # Compute local translation velocity
    local_transl_vel = get_local_transl_vel(
        torch.from_numpy(trans_yup),
        torch.from_numpy(global_orient_yup)
    ).numpy()

    # Compute foot contacts
    contacts = compute_foot_contacts(smplx_model, poses_yup, trans_yup, betas_raw)

    # Extract betas
    if len(betas_raw.shape) == 1:
        betas = np.tile(betas_raw[:beta_dim], (L, 1))
    else:
        betas = np.tile(betas_raw[0, :beta_dim], (L, 1))

    if betas.shape[1] < beta_dim:
        padded_betas = np.zeros((L, beta_dim))
        padded_betas[:, :betas.shape[1]] = betas
        betas = padded_betas

    # Construct lower (71D) and upper (78D) representations
    lower = np.concatenate([
        lower_pose_6d,      # 54D
        local_transl_vel,   # 3D
        contacts,           # 4D
        betas,              # 10D
    ], axis=-1)  # Total: 71D
    upper = upper_pose_6d.astype(np.float32)

    # AMASS has no face/hand annotations in this pipeline.
    # Provide empty placeholders so downstream loaders can keep consistent keys.
    face_exps = np.zeros((L, 100), dtype=np.float32)
    jaw_6d = np.zeros((L, 6), dtype=np.float32)
    head_6d = np.zeros((L, 6), dtype=np.float32)
    face = np.concatenate([jaw_6d, face_exps], axis=-1)            # 106D
    face_with_head = np.concatenate([head_6d, jaw_6d, face_exps], axis=-1)  # 112D
    hand = np.zeros((L, 180), dtype=np.float32)

    # Keep a combined motion vector for backward compatibility
    motion_vector = np.concatenate([upper, lower], axis=-1)  # 149D

    fps = float(fps_override) if fps_override is not None else float(data.get('mocap_frame_rate', 30.0))
    gender = str(data['gender']) if 'gender' in data else 'neutral'

    return {
        'motion_vector': motion_vector.astype(np.float32),
        'num_frames': L,
        'fps': fps,
        'gender': gender,
        'upper': upper.astype(np.float32),
        'lower': lower.astype(np.float32),
        'hand': hand,
        'face': face.astype(np.float32),
        'face_with_head': face_with_head.astype(np.float32),
        'upper_pose': upper_pose_aa.reshape(L, 39).astype(np.float32),
        'lower_pose': lower_pose_aa.reshape(L, 27).astype(np.float32),
        'global_orient': global_orient_yup.astype(np.float32),
    }


# =============================================================================
# HumanML3D Mapping
# =============================================================================

def load_humanml3d_mapping(index_file_path):
    """Load mapping between AMASS source paths and HumanML3D names."""
    index_dataframe = pd.read_csv(index_file_path)
    total_sequences = index_dataframe.shape[0]
    humanml3d_mapping = {}

    for i in tqdm(range(total_sequences), desc="Loading HumanML3D mapping"):
        source_path = index_dataframe.loc[i]['source_path']

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
    """Downsample to target_fps and trim by (start_frame, end_frame)."""
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='AMASS -> Upper+Lower format conversion')
    parser.add_argument(
        "--dataset_path_original",
        type=str,
        default="/path/to/AMASS_original_smplx",
        help="Path to original AMASS dataset",
    )
    parser.add_argument(
        "--dataset_path_processed",
        type=str,
        default="/path/to/AMASS",
        help="Path to processed AMASS dataset root",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="./preprocess/index.csv",
        help="Path to HumanML3D index file",
    )
    parser.add_argument(
        "--index_fps",
        type=int,
        default=20,
        help="Frame rate used by the index file",
    )
    parser.add_argument(
        "--ex_fps",
        type=int,
        default=25,
        help="Target frame rate for output",
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="./model_files/smplx_models",
        help="Path to SMPLX model files (parent directory, smplx.create will append /smplx)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    input_dir = Path(args.dataset_path_original)
    output_dir = Path(args.dataset_path_processed) / f"amass_parts_{args.ex_fps}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing SMPLX model...")
    smplx_model = smplx.create(
        args.smplx_path,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext='npz',
        use_pca=False,
    ).cuda().eval()

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

    print("Processing AMASS sequences (Upper+Lower)...")
    for motion_file in tqdm(valid_motion_files, desc="Processing AMASS sequences", ncols=150):
        source_identifier = motion_file.relative_to(input_dir).as_posix().replace('/', '_')
        new_name, start_frame, end_frame = humanml3d_mapping[source_identifier]

        output_path = output_dir / new_name.replace('.npy', '.npz')
        if output_path.exists():
            stats['skipped_existing'] += 1
            continue

        try:
            motion_data = np.load(motion_file, allow_pickle=True)
            valid, reason = validate_smplx_file(motion_data)
            if not valid:
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

            trimmed = {
                'poses': pose_data,
                'trans': translation_data,
                'betas': motion_data['betas'],
                'mocap_frame_rate': float(args.ex_fps),
                'gender': motion_data['gender'] if 'gender' in motion_data else 'neutral',
            }

            result = convert_amass_to_upper_lower(trimmed, smplx_model, fps_override=args.ex_fps)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **result)
            stats['processed'] += 1

        except Exception as e:
            stats['skipped_error'] += 1
            stats['errors'].append((str(motion_file), str(e)))

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped (existing): {stats['skipped_existing']}")
    print(f"  Skipped (invalid): {stats['skipped_invalid']}")
    print(f"  Skipped (empty): {stats['skipped_empty']}")
    print(f"  Skipped (error): {stats['skipped_error']}")
    print(f"\nOutput directory: {output_dir}")

    if stats['errors']:
        print("\nFirst 5 errors:")
        for path, err in stats['errors'][:5]:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
