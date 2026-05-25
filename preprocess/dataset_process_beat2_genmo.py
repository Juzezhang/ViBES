"""
Convert BEAT2 SMPL-X files to GENMO motion vector format.

GENMO format (without in-cam data): 145 dimensions per frame
- body_pose_r6d:      126 dims (21 joints × 6D rotation)  indices 0-125
- betas:               10 dims (shape parameters)         indices 126-135
- global_orient_r6d:    6 dims (root rotation, 6D)        indices 136-141
- local_transl_vel:     3 dims (translation velocity)     indices 142-144

Coordinate System:
- BEAT2: Already Y-up (XZ is ground plane, Y is vertical)
- GENMO: Y-up (XZ is ground plane, Y is vertical)
- No coordinate conversion needed, but we keep the structure for consistency.

Usage:
    python format_BEAT2.py
"""
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add current directory to path for relative imports if needed
sys.path.append(str(Path(__file__).parent))

# Import rotation conversion functions from GENMO codebase
from utils.rotation_conversions import (
    axis_angle_to_matrix, 
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
)

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
    
    # Compute velocity (frame-to-frame difference)
    velocity = torch.zeros_like(transl)
    velocity[1:] = transl[1:] - transl[:-1]
    velocity[0] = velocity[1] if len(velocity) > 1 else velocity[0]
    
    # Transform to local coordinates (root-relative)
    R = axis_angle_to_matrix(global_orient)  # (L, 3, 3)
    R_inv = R.transpose(-1, -2)  # Inverse of rotation matrix
    
    # Local velocity = R^T @ world_velocity
    local_vel = torch.einsum('lij,lj->li', R_inv, velocity)
    
    return local_vel


def validate_beat2_file(data):
    """
    Validate that the file is a proper BEAT2 SMPL-X file.
    Returns: (is_valid, reason, is_already_genmo)
    """
    # Check if file is already in GENMO format
    if 'motion_vector' in data and 'num_frames' in data:
        motion_vector = data['motion_vector']
        if len(motion_vector.shape) == 2 and motion_vector.shape[1] in [145, 149]:
            return True, "Already GENMO format", True

    # Check if it's raw SMPL-X format
    required_keys = ['poses', 'trans', 'betas']
    for key in required_keys:
        if key not in data:
            return False, f"Missing required key: {key}", False

    poses = data['poses']
    if len(poses.shape) != 2:
        return False, f"poses should be 2D array, got shape {poses.shape}", False

    pose_dim = poses.shape[1]
    if pose_dim < 66:
        return False, f"Pose dimension {pose_dim} too small", False

    trans = data['trans']
    if len(trans.shape) != 2 or trans.shape[1] != 3:
        return False, f"trans should be (L, 3), got shape {trans.shape}", False

    if poses.shape[0] != trans.shape[0]:
        return False, f"Frame count mismatch: poses={poses.shape[0]}, trans={trans.shape[0]}", False

    if poses.shape[0] < 2:
        return False, f"Too few frames: {poses.shape[0]}", False

    return True, "Valid", False


def convert_beat2_to_genmo(data):
    """
    Convert BEAT2 SMPL-X data to GENMO motion vector format.
    BEAT2 is already Y-up, so no rotation is applied.
    """
    poses = data['poses']
    global_orient_aa = poses[:, :3]
    body_pose_aa = poses[:, 3:66]
    
    trans = data['trans']
    betas_raw = data['betas']
    L = body_pose_aa.shape[0]
    
    # Extract first 10 betas and tile (GENMO standard)
    if len(betas_raw.shape) == 1:
        betas = np.tile(betas_raw[:10], (L, 1)) if len(betas_raw) >= 10 else np.zeros((L, 10))
    else:
        betas = betas_raw[:, :10] if betas_raw.shape[1] >= 10 else np.zeros((L, 10))
    
    # BEAT2 is already Y-up
    trans_yup = torch.from_numpy(trans.astype(np.float32))
    global_orient_yup = torch.from_numpy(global_orient_aa.astype(np.float32))
    
    # Body pose (local rotations)
    body_pose_t = torch.from_numpy(body_pose_aa.astype(np.float32))
    
    # Convert to GENMO format (Y-up)
    body_pose_reshaped = body_pose_t.reshape(L, 21, 3)
    body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose_reshaped)).reshape(L, 126)
    global_orient_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient_yup))
    local_transl_vel = get_local_transl_vel(trans_yup, global_orient_yup)
    
    betas_t = torch.from_numpy(betas.astype(np.float32))
    motion_vector = torch.cat([body_pose_r6d, betas_t, global_orient_r6d, local_transl_vel], dim=-1)
    
    fps = float(data['mocap_frame_rate']) if 'mocap_frame_rate' in data else 30.0
    gender = str(data['gender']) if 'gender' in data else 'neutral'
    
    return {
        'motion_vector': motion_vector.numpy(),
        'num_frames': L,
        'fps': fps,
        'gender': gender,
        'body_pose': body_pose_aa,
        'global_orient': global_orient_aa,
        'trans': trans,
        'betas': betas, # Only save the 10-dim version
    }


def convert_genmo_to_beat2(motion_vector, global_orient_yup, trans_yup):
    """
    Convert GENMO motion vector back to BEAT2-like format for verification.
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
        trans_decoded_yup[i] = trans_decoded_yup[i-1] + world_vel_yup[i]
    
    return {
        'body_pose': body_pose_aa.numpy(),
        'global_orient': global_orient_aa_yup.numpy(),
        'trans': trans_decoded_yup.numpy(),
        'betas': betas.numpy(),
    }


def verify_conversion(input_dir, output_dir, num_files=5):
    """
    Round-trip verification.
    """
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
        original_path = input_dir / rel_path
        if not original_path.exists(): continue
        
        try:
            converted_data = np.load(converted_path, allow_pickle=True)
            original_data = np.load(original_path, allow_pickle=True)
            
            orig_poses = original_data['poses']
            orig_global_orient = orig_poses[:, :3]
            orig_body_pose = orig_poses[:, 3:66]
            orig_trans = original_data['trans']
            orig_betas = original_data['betas']
            
            L = orig_body_pose.shape[0]
            orig_betas_10 = np.tile(orig_betas[:10] if len(orig_betas.shape)==1 else orig_betas[0, :10], (L, 1))
            
            decoded = convert_genmo_to_beat2(converted_data['motion_vector'], 
                                            converted_data['global_orient'], 
                                            converted_data['trans'])
            
            orig_body_R = axis_angle_to_matrix(torch.from_numpy(orig_body_pose.astype(np.float32)).reshape(-1, 21, 3))
            decoded_body_R = axis_angle_to_matrix(torch.from_numpy(decoded['body_pose'].astype(np.float32)).reshape(-1, 21, 3))
            body_diff = (orig_body_R - decoded_body_R).abs().max().item()
            
            orig_orient_R = axis_angle_to_matrix(torch.from_numpy(orig_global_orient.astype(np.float32)))
            decoded_orient_R = axis_angle_to_matrix(torch.from_numpy(decoded['global_orient'].astype(np.float32)))
            orient_diff = (orig_orient_R - decoded_orient_R).abs().max().item()
            
            betas_diff = np.abs(decoded['betas'] - orig_betas_10).max()
            trans_diff = np.abs(decoded['trans'] - orig_trans).max()
            
            ok = (body_diff < 1e-3 and orient_diff < 1e-3 and betas_diff < 1e-6 and trans_diff < 1e-3)
            print(f"{'✓' if ok else '✗'} {rel_path} (body:{body_diff:.1e}, orient:{orient_diff:.1e}, trans:{trans_diff:.1e})")
            if not ok: 
                all_passed = False
                print(f"  FAILED: body_diff={body_diff}, orient_diff={orient_diff}, trans_diff={trans_diff}")
        except Exception as e:
            print(f"❌ {rel_path}: {e}")
            all_passed = False
    return all_passed


def process_directory(input_dir, output_dir, max_files=None):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    npz_files = list(input_dir.rglob('*.npz'))
    stats = {'processed': 0, 'copied': 0, 'skipped_invalid': 0, 'skipped_error': 0, 'skipped_existing': 0, 'errors': []}
    
    processed_count = 0
    for npz_path in tqdm(npz_files, desc="Processing"):
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
            data = np.load(npz_path, allow_pickle=True)
            is_valid, reason, is_already_genmo = validate_beat2_file(data)

            if not is_valid:
                stats['skipped_invalid'] += 1
                continue

            if is_already_genmo:
                # File is already in GENMO format, just copy it
                import shutil
                shutil.copy2(npz_path, out_path)
                stats['copied'] += 1
            else:
                # Convert raw SMPL-X to GENMO format
                res = convert_beat2_to_genmo(data)
                np.savez_compressed(out_path, **res)
                stats['processed'] += 1

            processed_count += 1
        except Exception as e:
            stats['skipped_error'] += 1
            stats['errors'].append((str(rel_path), str(e)))
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert BEAT2 SMPL-X motion (25 fps + mirror) to GENMO 145D format.",
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="[INPUT] BEAT2 root containing the smplxflame_25/ and smplxflame_25_mirror/ subfolders (e.g. <BEAT2_ROOT>).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="[OUTPUT] Where to write GENMO 145D .npz files, mirroring the input subfolder structure (e.g. <BEAT2_ROOT>/beat2_genmo_25).")
    parser.add_argument("--subdirs", type=str, nargs="+",
                        default=["smplxflame_25", "smplxflame_25_mirror"],
                        help="Subdirectories under --input_dir to process (default: both original and mirror).")
    parser.add_argument("--max_files_per_subdir", type=int, default=None,
                        help="Cap files per subdirectory for quick testing (default: process everything).")
    args = parser.parse_args()

    base_input = Path(args.input_dir)
    base_output = Path(args.output_dir)
    subdirs = args.subdirs
    max_files = args.max_files_per_subdir

    total_stats = {'processed': 0, 'copied': 0, 'skipped_invalid': 0, 'skipped_error': 0, 'skipped_existing': 0}

    for subdir in subdirs:
        input_dir = base_input / subdir
        output_dir = base_output / subdir
        if not input_dir.exists():
            print(f"  (skipping {subdir} — not found at {input_dir})")
            continue

        print(f"\nProcessing {subdir}...")
        stats = process_directory(input_dir, output_dir, max_files=max_files)
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    print(f"\nFinal Summary: {total_stats['processed']} converted, {total_stats['copied']} copied (already GENMO), {total_stats['skipped_existing']} skipped, {total_stats['skipped_invalid']} invalid, {total_stats['skipped_error']} errors")

    if total_stats['processed'] > 0:
        verify_conversion(base_input, base_output)

if __name__ == "__main__":
    main()
