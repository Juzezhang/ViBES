"""
AMASS Dataset Processing Script

This script processes AMASS motion capture data by:
1. Loading and filtering AMASS motion files
2. Converting coordinate systems and applying transformations
3. Creating both original and mirrored versions of motion sequences
4. Saving processed data in a standardized format

The script maintains compatibility with HumanML3D dataset structure.
"""

import sys, os
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.rotation_tools import matrot2aa, aa2matrot
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pathlib
from collections import defaultdict
import pandas as pd
from collections import defaultdict
import os
import trimesh
from scipy.spatial.transform import Rotation as R
import smplx
import argparse


def parse_arguments():
    """Parse command line arguments for dataset processing configuration."""
    parser = argparse.ArgumentParser(description='Dataset processing options')
    parser.add_argument("--smplx_path", type=str, required=False, 
                       default="./model_files/smplx_models", 
                       help="Path to SMPL-X model files")
    parser.add_argument("--dataset_path_original", type=str, required=False, 
                       default="/scr/juze/datasets/BEAT2/beat_english_v2.0.0/smplxflame_25", 
                       help="Path to original BEAT2 dataset")
    parser.add_argument("--dataset_path_processed", type=str, required=False, 
                       default="/scr/juze/datasets/BEAT2/beat_english_v2.0.0/smplxflame_25_mirror", 
                       help="Path to processed BEAT2 dataset")
    parser.add_argument("--index_path", type=str, required=False, 
                       default="./preprocess/index.csv", 
                       help="Path to index file")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    
    return parser.parse_args()


def initialize_smplx_model(smplx_model_path, num_shape_params=16):
    """
    Initialize SMPL-X body model for motion processing.
    
    Args:
        smplx_model_path (str): Path to SMPL-X model files
        num_shape_params (int): Number of shape parameters to use
        
    Returns:
        smplx.SMPLX: Initialized SMPL-X model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    neutral_body_model = smplx.create(
        smplx_model_path,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=num_shape_params,
        num_expression_coeffs=100,
        ext='npz',
        use_pca=False,
    ).eval().to(device)
    
    return neutral_body_model



def collect_and_filter_motion_files(beat2_data_directory, debug_mode=False):
    """
    Collect and filter motion files based on various criteria.
    
    Args:
        beat2_data_directory (pathlib.Path): Directory containing BEAT2 data
        debug_mode (bool): Whether to run in debug mode
        
    Returns:
        list: List of valid motion files to process
    """


    # Find all .npz files in the specified datasets
    all_motion_files = [
        pathlib.Path(f'{root}/{filename}')
        for root, dirs, files in os.walk(beat2_data_directory)
        for filename in files
        if filename.endswith('.npz')
    ]
    all_motion_files.sort()
    
    # Initialize tracking lists for file categorization
    valid_motion_files = []
    files_not_readable = []
    files_not_in_humanml3d = []
    files_incompatible_model = []
    files_missing_framerate = []
    files_invalid_framerate = []
    
    # Track statistics
    model_type_counts = defaultdict(int)
    framerate_counts = defaultdict(int)
    
    print(f"Processing {len(all_motion_files)} motion files...")
    
    # Process each file and categorize based on various criteria
    for motion_file in tqdm(all_motion_files, ncols=150, desc="Filtering motion files"):
        
        
        
        # Skip detailed checks in debug mode
        if debug_mode:
            valid_motion_files.append(motion_file)
            continue
        
        # Try to load and validate the motion file
        try:
            motion_data = np.load(motion_file, allow_pickle=True)
        except zipfile.BadZipFile:
            files_not_readable.append(motion_file)
            continue
        
        # Check model type compatibility
        model_type = motion_data['model'].item()
        model_type_counts[model_type] += 1
        
        if model_type not in {'smplx', 'smplx2020', 'smplx_locked_head'}:
            files_incompatible_model.append(motion_file)
            continue
        
        # Check for frame rate information
        if 'mocap_frame_rate' not in motion_data:
            files_missing_framerate.append(motion_file)
            continue
        
        # Track frame rate statistics
        frame_rate = int(motion_data['mocap_frame_rate'].item())
        framerate_counts[frame_rate] += 1
        
        valid_motion_files.append(motion_file)
    
    # Print comprehensive statistics
    print(f'\nFile Processing Statistics:')
    print(f'Total files found: {len(all_motion_files)}')
    print(f'Valid files for processing: {len(valid_motion_files)}')
    print(f'Files not readable: {len(files_not_readable)}')
    print(f'Files with incompatible model type: {len(files_incompatible_model)}')
    print(f'Files missing frame rate info: {len(files_missing_framerate)}')
    print(f'Files with invalid frame rate: {len(files_invalid_framerate)}')
    print(f'Model types found: {sorted(model_type_counts.items())}')
    print(f'Frame rates found: {sorted(framerate_counts.items())}')
    
    # Verify that all files are accounted for
    total_categorized = (len(valid_motion_files) + 
                        len(files_not_readable) + len(files_incompatible_model) + 
                        len(files_missing_framerate) + len(files_invalid_framerate))
    assert len(all_motion_files) == total_categorized, "File count mismatch in categorization"
    
    return valid_motion_files


def save_mesh_as_obj(output_path, vertices, faces):
    """
    Save mesh vertices and faces as OBJ file.
    
    Args:
        output_path (str): Path to save the OBJ file
        vertices (np.ndarray): Mesh vertices
        faces (np.ndarray): Mesh faces
    """
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    triangle_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
    triangle_mesh.export(output_path)


def create_rotation_matrix(rx, ry, rz, use_degrees=True):
    """
    Create rotation matrix from Euler angles.
    
    Args:
        rx, ry, rz (float): Rotation angles around x, y, z axes
        use_degrees (bool): Whether angles are in degrees
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    if use_degrees:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    
    sin_x, sin_y, sin_z = np.sin(rx), np.sin(ry), np.sin(rz)
    cos_x, cos_y, cos_z = np.cos(rx), np.cos(ry), np.cos(rz)

    # Create rotation matrices for each axis
    rotation_x = np.array([[1.0, 0.0, 0.0],
                          [0.0, cos_x, -sin_x],
                          [0.0, sin_x, cos_x]])
    
    rotation_y = np.array([[cos_y, 0.0, sin_y],
                          [0.0, 1.0, 0.0],
                          [-sin_y, 0.0, cos_y]])
    
    rotation_z = np.array([[cos_z, -sin_z, 0.0],
                          [sin_z, cos_z, 0.0],
                          [0.0, 0.0, 1.0]])

    # Combine rotations in ZYX order
    combined_rotation = np.matmul(np.matmul(rotation_z, rotation_y), rotation_x)
    return combined_rotation


def process_and_align_motion_sequence(motion_file_path, output_mirrored_path, 
                                    right_joint_indices, left_joint_indices):
    """
    Process and align a single motion sequence from BEAT2 data.
    
    Args:
        motion_file_path (pathlib.Path): Input motion file path
        output_mirrored_path (pathlib.Path): Output path for mirrored motion
        right_joint_indices (list): Indices of right-side joints
        left_joint_indices (list): Indices of left-side joints
        
    Returns:
        int: Original frame rate of the motion file, or 0 if processing failed
    """
    
    # Load motion data
    try:
        motion_data = np.load(motion_file_path, allow_pickle=True)
    except zipfile.BadZipFile:
        return 0
    
    # Extract frame rate information
    original_fps = 0
    try:
        original_fps = motion_data['mocap_frame_rate']
    except:
        return original_fps

    # Verify model compatibility
    assert motion_data['model'].item() == 'smplx' or motion_data['model'].item() == 'smplx2020', "Only SMPL-X models are supported"

    # Calculate downsampling rate and apply it
    pose_data = motion_data['poses']
    translation_data = motion_data['trans']
    
    # Apply dataset-specific temporal adjustments
    # Extract the specified frame range
    pose_data = pose_data[:]
    translation_data = translation_data[:]

    # Create mirrored version of the motion
    pose_data_reshaped = pose_data.reshape(-1, 55, 3)
    mirrored_pose_data = pose_data_reshaped.copy()
    mirrored_translation_data = translation_data.copy()
    
    # Swap left and right joints for mirroring
    mirrored_pose_data[:, left_joint_indices] = pose_data_reshaped[:, right_joint_indices]
    mirrored_pose_data[:, right_joint_indices] = pose_data_reshaped[:, left_joint_indices]
    
    # Mirror Y and Z rotations
    mirrored_pose_data[:, :, 1:3] *= -1
    
    # Mirror X translation
    mirrored_translation_data[..., 0] *= -1
    mirrored_translation_data[:, 0] -= mirrored_translation_data[0, 0]
    mirrored_translation_data[:, 2] -= mirrored_translation_data[0, 2]
    
    # Reshape back to original format
    mirrored_pose_data = mirrored_pose_data.reshape(-1, 55 * 3)

    # Save mirrored motion data
    mirrored_motion_data = dict(motion_data)
    mirrored_motion_data['poses'] = mirrored_pose_data
    mirrored_motion_data['trans'] = mirrored_translation_data
    np.savez(output_mirrored_path, **mirrored_motion_data)

    return original_fps


def setup_joint_mapping(smplx_model_path):
    """
    Set up joint mapping for left-right mirroring operations.
    
    Args:
        smplx_model_path (str): Path to SMPL-X model files
        
    Returns:
        tuple: (left_joint_indices, right_joint_indices, joints_to_exclude)
    """
    model_params_file = pathlib.Path(os.path.join(smplx_model_path, 'smplx/SMPLX_NEUTRAL_2020.npz'))
    model_params = np.load(model_params_file, allow_pickle=True)
    joint_name_to_index = model_params['joint2num'].item()
    index_to_joint_name = {v: k for k, v in joint_name_to_index.items()}

    # Identify left and right joints for mirroring
    left_joint_indices = []
    right_joint_indices = []
    
    for joint_name in joint_name_to_index:
        if joint_name.startswith('L_'):
            left_joint_name = joint_name
            right_joint_name = joint_name.replace('L_', 'R_')
            left_joint_indices.append(joint_name_to_index[left_joint_name])
            right_joint_indices.append(joint_name_to_index[right_joint_name])

    # Define joints to be excluded from processing
    joints_to_exclude = [
        joint_name_to_index['Jaw'],
        joint_name_to_index['L_Eye'],
        joint_name_to_index['R_Eye']
    ]

    # Print joint mapping information
    print(f'\nJoint Mapping Information:')
    print(f'Number of joint pairs to swap: {len(left_joint_indices)}')
    print(f'Left joint indices: {left_joint_indices}')
    print(f'Right joint indices: {right_joint_indices}')
    print(f'Joints to exclude: {joints_to_exclude}')
    print('\nJoint pair mappings:')
    for left_idx, right_idx in sorted(zip(left_joint_indices, right_joint_indices)):
        left_name = index_to_joint_name[left_idx]
        right_name = index_to_joint_name[right_idx]
        print(f'{left_name:10} ({left_idx:2}) <--> ({right_idx:2}) {right_name}')

    return left_joint_indices, right_joint_indices, joints_to_exclude


def main():
    """Main function to orchestrate the AMASS dataset processing pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    debug_mode = args.debug
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_shape_params = 300  # Number of shape parameters for SMPL-X model

    print("Starting AMASS dataset processing...")
    print(f"Using device: {device}")
    print(f"Debug mode: {debug_mode}")

    # Initialize SMPL-X model
    print("\nInitializing SMPL-X model...")
    neutral_body_model = initialize_smplx_model(args.smplx_path, num_shape_params)


    # Collect and filter motion files
    print("\nCollecting and filtering motion files...")
    beat2_data_directory = pathlib.Path(args.dataset_path_original)
    valid_motion_files = collect_and_filter_motion_files(
        beat2_data_directory, debug_mode
    )

    # Set up joint mapping for mirroring
    print("\nSetting up joint mapping...")
    left_joint_indices, right_joint_indices, joints_to_exclude = setup_joint_mapping(args.smplx_path)

    # Define output directories
    processed_data_directory = pathlib.Path(args.dataset_path_processed)

    # Process each motion file
    print(f"\nProcessing {len(valid_motion_files)} motion files...")
    
    for motion_file in tqdm(valid_motion_files, desc='Processing AMASS sequences', ncols=150):
        # Get mapping information for this file
        
        # Define output file paths
        output_mirrored_path = processed_data_directory / ('M_' + motion_file.relative_to(beat2_data_directory).as_posix().replace('/', '_').replace('.npy', '.npz'))

        # Create output directories if they don't exist
        output_mirrored_path.parent.mkdir(parents=True, exist_ok=True)

        # Process and align motion data
        process_and_align_motion_sequence(
            motion_file, output_mirrored_path,
            right_joint_indices, left_joint_indices
        )

    print(f"\nProcessing completed!")
    print(f"Processed data saved to: {args.dataset_path_processed}")


if __name__ == "__main__":
    main()


