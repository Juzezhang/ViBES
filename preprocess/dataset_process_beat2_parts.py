"""
Convert BEAT2 SMPL-X files into part-wise motion arrays (upper/lower/face/hand).

Outputs per sequence (.npz):
  - upper: 78D (13 joints × 6D rotation)
  - lower: 71D (54D lower pose + 3D local vel + 4D contact + 10D betas)
  - face: 106D (jaw 6D + 100 expr)
  - face_with_head: 112D (head 6D + jaw 6D + 100 expr)
  - hand: 180D (30 joints × 6D)
  - global_orient for reference

Upper body joints (13 joints):
  spine1, spine2, spine3, neck, left_collar, right_collar, head,
  left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist

Lower body joints (9 joints):
  pelvis, left_hip, right_hip, left_knee, right_knee,
  left_ankle, right_ankle, left_foot, right_foot

Usage:
  python dataset_process_beat2_parts.py [--output_dir OUTPUT_DIR]
"""
import argparse
import os
import sys
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
)

# =============================================================================
# Joint Masks
# =============================================================================

# Upper body joint indices (13 joints)
# spine1=3, spine2=6, spine3=9, neck=12, left_collar=13, right_collar=14,
# head=15, left_shoulder=16, right_shoulder=17, left_elbow=18, right_elbow=19,
# left_wrist=20, right_wrist=21
UPPER_JOINT_INDICES = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

# Lower body joint indices (9 joints)
# pelvis=0, left_hip=1, right_hip=2, left_knee=4, right_knee=5,
# left_ankle=7, right_ankle=8, left_foot=10, right_foot=11
LOWER_JOINT_INDICES = [0, 1, 2, 4, 5, 7, 8, 10, 11]


def create_joint_mask(joint_indices, total_joints=55):
    """Create a boolean mask for specified joints (axis-angle format)."""
    mask = np.zeros(total_joints * 3, dtype=bool)
    for joint_idx in joint_indices:
        mask[joint_idx * 3:(joint_idx + 1) * 3] = True
    return mask


JOINT_MASK_UPPER = create_joint_mask(UPPER_JOINT_INDICES)
JOINT_MASK_LOWER = create_joint_mask(LOWER_JOINT_INDICES)


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
    return local_vel.numpy()


def axis_angle_to_6d_np(aa):
    """Convert axis-angle to 6D rotation representation."""
    aa = torch.from_numpy(aa.astype(np.float32))
    R = axis_angle_to_matrix(aa)
    r6d = matrix_to_rotation_6d(R)
    return r6d.numpy()


def validate_beat2_file(data):
    """Validate that the file is a proper BEAT2 SMPL-X file."""
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

def compute_foot_contacts(smplx_model, m_data, max_batch=128):
    """Compute foot contacts from motion data using SMPLX."""
    # Extract data as numpy arrays (handle potential npz loading issues)
    betas = np.array(m_data["betas"])
    poses = np.array(m_data["poses"]).astype(np.float32)
    trans = np.array(m_data["trans"]).astype(np.float32)

    n = poses.shape[0]

    # Handle betas - ensure proper shape
    if betas.ndim == 1:
        betas_300 = np.zeros(300, dtype=np.float32)
        betas_300[:min(len(betas), 300)] = betas[:min(len(betas), 300)]
    else:
        betas_300 = np.zeros(300, dtype=np.float32)
        betas_300[:min(betas.shape[1], 300)] = betas[0, :min(betas.shape[1], 300)]

    # Tile betas for all frames
    betas_tiled = np.tile(betas_300.reshape(1, 300), (n, 1)).astype(np.float32)

    # Convert to tensors
    betas_tensor = torch.from_numpy(betas_tiled).cuda()
    poses_tensor = torch.from_numpy(poses).cuda()
    trans_tensor = torch.from_numpy(trans).cuda()

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

def convert_beat2_to_upper_lower(data, smplx_model, beta_dim=10):
    """
    Convert BEAT2 SMPL-X data to Upper+Lower body motion vector format.

    Returns dict with:
        - motion_vector: (L, 149) Upper+Lower format
        - metadata
    """
    poses = data['poses']
    trans = data['trans']
    betas_raw = data['betas']
    L = poses.shape[0]

    # Extract upper body pose (13 joints)
    upper_pose_aa = poses[:, JOINT_MASK_UPPER].reshape(L, 13, 3)
    upper_pose_6d = axis_angle_to_6d_np(upper_pose_aa).reshape(L, 78)

    # Extract lower body pose (9 joints)
    lower_pose_aa = poses[:, JOINT_MASK_LOWER].reshape(L, 9, 3)
    lower_pose_6d = axis_angle_to_6d_np(lower_pose_aa).reshape(L, 54)

    # Extract global orient for local velocity
    global_orient = poses[:, :3]
    local_transl_vel = get_local_transl_vel(trans, global_orient)

    # Compute foot contacts
    contacts = compute_foot_contacts(smplx_model, data)

    # Extract betas
    if len(betas_raw.shape) == 1:
        betas = np.tile(betas_raw[:beta_dim], (L, 1))
    else:
        betas = np.tile(betas_raw[0, :beta_dim], (L, 1))

    # Build lower (71D) and upper (78D) representations
    lower = np.concatenate([
        lower_pose_6d,      # 54D
        local_transl_vel,   # 3D
        contacts,           # 4D
        betas,              # 10D
    ], axis=-1)  # Total: 71D
    upper = upper_pose_6d.astype(np.float32)

    # Face: jaw 6D + expressions (pad/trim to 100)
    expressions = data.get("expressions")
    if expressions is None:
        expressions = np.zeros((L, 100), dtype=np.float32)
    else:
        expressions = np.array(expressions, dtype=np.float32)
        if expressions.shape[1] < 100:
            padded = np.zeros((L, 100), dtype=np.float32)
            padded[:, :expressions.shape[1]] = expressions
            expressions = padded
        elif expressions.shape[1] > 100:
            expressions = expressions[:, :100]

    jaw_pose = poses[:, 66:69]
    jaw_pose_6d = axis_angle_to_6d_np(jaw_pose).reshape(L, 6)
    head_pose = poses[:, 45:48]
    head_pose_6d = axis_angle_to_6d_np(head_pose).reshape(L, 6)
    face = np.concatenate([jaw_pose_6d, expressions], axis=-1)  # 106D
    face_with_head = np.concatenate([head_pose_6d, jaw_pose_6d, expressions], axis=-1)  # 112D

    # Hand: 30 joints (15 left + 15 right) in axis-angle -> 6D
    hand_pose = poses[:, 25 * 3:55 * 3].reshape(L, 30, 3)
    hand_pose_6d = axis_angle_to_6d_np(hand_pose).reshape(L, 180)

    # Combined vector for backward compatibility
    motion_vector = np.concatenate([upper, lower], axis=-1)  # 149D

    fps = float(data.get('mocap_frame_rate', 25.0)) if isinstance(data, dict) else 25.0
    gender = str(data['gender']) if 'gender' in data else 'neutral'

    return {
        'motion_vector': motion_vector.astype(np.float32),
        'num_frames': L,
        'fps': fps,
        'gender': gender,
        'upper': upper.astype(np.float32),
        'lower': lower.astype(np.float32),
        'hand': hand_pose_6d.astype(np.float32),
        'face': face.astype(np.float32),
        'face_with_head': face_with_head.astype(np.float32),
        # Optional: for verification/visualization
        'upper_pose': upper_pose_aa.reshape(L, 39).astype(np.float32),
        'lower_pose': lower_pose_aa.reshape(L, 27).astype(np.float32),
        'global_orient': global_orient.astype(np.float32),
    }


# =============================================================================
# Processing Functions
# =============================================================================

def process_directory(input_dir, output_dir, smplx_model, max_files=None):
    """Process all .npz files in a directory."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    npz_files = sorted(list(input_dir.rglob('*.npz')))

    stats = {
        'processed': 0,
        'skipped_invalid': 0,
        'skipped_error': 0,
        'skipped_existing': 0,
        'errors': []
    }

    processed_count = 0
    for npz_path in tqdm(npz_files, desc=f"Processing {input_dir.name}"):
        if max_files is not None and processed_count >= max_files:
            break

        rel_path = npz_path.relative_to(input_dir)
        out_path = output_dir / rel_path

        if out_path.exists():
            stats['skipped_existing'] += 1
            processed_count += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = dict(np.load(npz_path, allow_pickle=True))
            valid, reason = validate_beat2_file(data)
            if not valid:
                stats['skipped_invalid'] += 1
                stats['errors'].append((str(rel_path), reason))
                continue

            result = convert_beat2_to_upper_lower(data, smplx_model)
            np.savez_compressed(out_path, **result)
            stats['processed'] += 1
            processed_count += 1

        except Exception as e:
            stats['skipped_error'] += 1
            stats['errors'].append((str(rel_path), str(e)))

    return stats


def parse_arguments():
    parser = argparse.ArgumentParser(description='BEAT2 -> Upper+Lower format conversion')
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/path/to/BEAT2/beat_english_v2.0.0",
        help="Path to BEAT2 dataset root",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/BEAT2/beat_english_v2.0.0/beat2_parts_25",
        help="Output directory for Upper+Lower format data",
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="./model_files/smplx_models",
        help="Path to SMPLX model files (parent directory, smplx.create will append /smplx)",
    )
    parser.add_argument(
        "--subdirs",
        type=str,
        nargs="+",
        default=["smplxflame_25", "smplxflame_25_mirror"],
        help="Subdirectories to process",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum files to process per subdir (for testing)",
    )
    parser.add_argument(
        "--beta_dim",
        type=int,
        default=10,
        help="Number of beta dimensions to include",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

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

    total_stats = {
        'processed': 0,
        'skipped_invalid': 0,
        'skipped_error': 0,
        'skipped_existing': 0,
    }

    for subdir in args.subdirs:
        input_dir = Path(args.input_dir) / subdir
        output_dir = Path(args.output_dir) / subdir

        if not input_dir.exists():
            print(f"Skipping {subdir}: directory not found")
            continue

        print(f"\nProcessing {subdir}...")
        stats = process_directory(
            input_dir,
            output_dir,
            smplx_model,
            max_files=args.max_files
        )

        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

        if stats['errors']:
            print(f"  Errors in {subdir}:")
            for path, err in stats['errors'][:5]:
                print(f"    {path}: {err}")
            if len(stats['errors']) > 5:
                print(f"    ... and {len(stats['errors']) - 5} more errors")

    print(f"\n{'='*60}")
    print(f"Final Summary:")
    print(f"  Processed: {total_stats['processed']}")
    print(f"  Skipped (existing): {total_stats['skipped_existing']}")
    print(f"  Skipped (invalid): {total_stats['skipped_invalid']}")
    print(f"  Skipped (error): {total_stats['skipped_error']}")
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()
