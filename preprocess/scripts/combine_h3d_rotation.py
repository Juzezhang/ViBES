#!/usr/bin/env python3
"""
Script to read motion data from BEAT2 dataset
"""
import os
import sys
import numpy as np
import argparse

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.hybrik_loc2rot import HybrIKJointsToRotmat

import torch
import smplx
from models.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np, rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d

from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
)
from utils.motion_process import recover_from_ric
from models.tokenizers.lom_vq import VQVAEConvZeroDSUS_PaperVersion, VQVAEConvZeroDSUS1_PaperVersion, VAEConvZero
from models.tokenizers.mgpt_vq import VQVae
import multimodal_tokenizers.render.matplot.plot_3d_global as plot_3d
import moviepy.editor as mp


# Note: inverse_selection_tensor function removed as it requires torch
# You can add it back when torch dependencies are available

def load_h3d_data(pkl_dir):
    """Load H3D data (pose and shape/beta) from pkl files"""
    import glob
    import joblib
    
    # Get all pkl files in the directory
    pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    pkl_files.sort()  # Sort to ensure consistent order
    
    print(f"Found {len(pkl_files)} pkl files in directory")
    
    if len(pkl_files) == 0:
        print("No pkl files found in the directory!")
        return None, None
    
    # Load all pkl files and combine them
    all_data = []
    successful_files = 0
    
    for i, pkl_file in enumerate(pkl_files):
        data = joblib.load(pkl_file)
        all_data.append(data)
        successful_files += 1
        
        if i < 5:  # Show details for first 5 files
            print(f"File {i+1}: {os.path.basename(pkl_file)} - dict with keys: {list(data.keys())}")
            # Show details of the first dict
            if i == 0:
                for key, value in data.items():
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
    
    print(f"\nSuccessfully loaded {successful_files} out of {len(pkl_files)} files")
    
    # Combine all data
    print(f"Combining {len(all_data)} dictionaries...")
    combined_dict = {}
    for i, data_dict in enumerate(all_data):
        for key, value in data_dict.items():
            if key not in combined_dict:
                combined_dict[key] = []
            combined_dict[key].append(value)
    
    # Convert lists to arrays
    for key, value_list in combined_dict.items():
        combined_dict[key] = np.concatenate(value_list, axis=0)
        print(f"  {key}: combined shape {combined_dict[key].shape}")
    
    # Separate pose and shape (beta) data
    pose_data = combined_dict['pose']  # shape: (1368, 72)
    shape_data = combined_dict['beta']  # shape: (1368, 10)
    
    print(f"\nPose data:")
    print(f"  Shape: {pose_data.shape}")
    print(f"  Total frames: {pose_data.shape[0]}")
    print(f"  Features per frame: {pose_data.shape[1]}")
    print(f"  Data type: {pose_data.dtype}")
    print(f"  Min/Max values: {np.min(pose_data):.4f} / {np.max(pose_data):.4f}")
    
    print(f"\nShape data (beta):")
    print(f"  Shape: {shape_data.shape}")
    print(f"  Total frames: {shape_data.shape[0]}")
    print(f"  Features per frame: {shape_data.shape[1]}")
    print(f"  Data type: {shape_data.dtype}")
    print(f"  Min/Max values: {np.min(shape_data):.4f} / {np.max(shape_data):.4f}")
    
    return pose_data, shape_data


def load_rotation_data(rotation_path):
    """Load rotation data from npz file"""
    rotation_data = np.load(rotation_path)
    
    print(f"\nLoaded rotation data from: {rotation_path}")
    print(f"Rotation data keys: {list(rotation_data.keys())}")
    
    # Show rotation data details
    for key in rotation_data.keys():
        data = rotation_data[key]
        print(f"  {key}: shape {data.shape}, dtype {data.dtype}")
    
    # Process rotation data separately
    rotation_array = rotation_data['poses']  # shape: (1369, 165)
    
    print(f"\nRotation data:")
    print(f"  Shape: {rotation_array.shape}")
    print(f"  Total frames: {rotation_array.shape[0]}")
    print(f"  Features per frame: {rotation_array.shape[1]}")
    print(f"  Data type: {rotation_array.dtype}")
    print(f"  Min/Max values: {np.min(rotation_array):.4f} / {np.max(rotation_array):.4f}")
    
    return rotation_array


def main():
    parser = argparse.ArgumentParser(description='Test GLM-4-Voice checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       default="/path/to/experiments/glm4voice_conversational_mot_layernum_5_modalitynum_6_beat2_body_only_v1/checkpoint-514000",
                       help='Path to the trained checkpoint')
    parser.add_argument('--output_dir', type=str, default="/path/to/BEAT2/beat_english_v2.0.0/new_joints_25fps_reconstructed_hybrik", 
                       help='Output directory')
    parser.add_argument('--device', type=str, default="cuda",
                       help='Device: cuda or cpu')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:0"

    lower_idx = [0,1,3,4,6,7,9,10]
    lower_idx_with_root = [0,1,2,4,5,7,8,10,11]
    h3d_feat_index_lower = [0,1,2,3]
    ## 4: 4 + 63
    for i in range(len(lower_idx)):
        h3d_feat_index_lower.extend(range( 3 * lower_idx[i] + 4, 3 * lower_idx[i] + 3 + 4))
    ## 67 : 67 + 126 
    for i in range(len(lower_idx)):
        h3d_feat_index_lower.extend(range( 6 * lower_idx[i] + 67, 6 * lower_idx[i] + 6 + 67))
    ## 4 + 63 + 126 : 
    for i in range(len(lower_idx_with_root)):
        h3d_feat_index_lower.extend(range( 3 * lower_idx_with_root[i] + 193, 3 * lower_idx_with_root[i] + 3 +  193))
    h3d_feat_index_lower.extend(range(259, 263))

    joints_data = np.load("/path/to/BEAT2/beat_english_v2.0.0/new_joints_25fps/2_scott_0_26_26.npy")
    joints_data = torch.tensor(joints_data, device=device).unsqueeze(0)


    # h3d_tokens = np.load("/path/to/BEAT2/beat_english_v2.0.0/TOKENS_AGENT_25_H3D_LOWER/2_scott_0_26_26.npy")
    h3d_tokens = np.load("/path/to/BEAT2/beat_english_v2.0.0/TOKENS_AGENT_25_H3D_LOWER/2_scott_0_26_26.npy")
    h3d_tokens = h3d_tokens[:, :100]


    
    vae_lower = VQVae(
                nfeats=107,
                quantizer= "ema_reset",
                code_num=512,
                code_dim=512,
                output_emb_width=512,
                down_t=2,
                stride_t=2,
                width=512,
                depth=3,
                dilation_growth_rate=3,
                norm=None,
                activation="relu")

    # checkpoint_lower = torch.load('/path/to/MotionGPT/checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar', map_location="cpu", weights_only=False)
    checkpoint_lower = torch.load('/path/to/MotionGPT/experiments/mgpt/VQVAE_HumanML3D_Beat2/checkpoints/epoch=9069.ckpt', map_location="cpu", weights_only=False)


    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in checkpoint_lower['state_dict'].items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v

    vae_lower.load_state_dict(vae_dict, strict=True)
    vae_lower.eval()
    vae_lower.to(device)
    lower_tokens = torch.tensor(h3d_tokens, device=device).unsqueeze(0)
    rec_lower_h3d = vae_lower.decode(lower_tokens.int())

    dis_data_root = '/path/to/MotionGPT/deps/t2m/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
    mean = np.load(os.path.join(dis_data_root, "mean.npy"))
    std = np.load(os.path.join(dis_data_root, "std.npy"))
    mean = mean[h3d_feat_index_lower]
    std = std[h3d_feat_index_lower]
    mean = torch.tensor(mean).to(device).float()
    std = torch.tensor(std).to(device).float()


    rec_lower_h3d = (rec_lower_h3d * std) + mean
    # h3d_joints = recover_from_ric(rec_lower_h3d, 22)[0]

    feat_data = np.load("/path/to/BEAT2/beat_english_v2.0.0/new_joint_vecs_25fps/2_scott_0_26_26.npy")
    feat_data = feat_data[:1000, h3d_feat_index_lower]
    feat_data = torch.tensor(feat_data, device=device).unsqueeze(0)

    h3d_joints = recover_from_ric(rec_lower_h3d, 9)[0]
    # h3d_joints = recover_from_ric(feat_data, 9)[0]
    # joints_data = h3d_joints.detach().cpu().numpy()

    output_gif_path = './demo/2_scott_0_26_26_rec_v3.gif'
    if len(h3d_joints.shape) == 3:
        h3d_joints = h3d_joints[None]
    if isinstance(h3d_joints, torch.Tensor):
        h3d_joints = h3d_joints.detach().cpu().numpy()
    # pose_vis = plot_3d.draw_to_batch(h3d_joints, [''], [output_gif_path])
    pose_vis = plot_3d.draw_to_batch_lower(h3d_joints, [''], [output_gif_path])
    out_video = mp.VideoFileClip(output_gif_path)
    out_video.write_videofile(output_gif_path.replace('.gif', '.mp4'))


    pose_data = np.load("/path/to/BEAT2/beat_english_v2.0.0/smplxflame_25/2_scott_0_26_26.npz", allow_pickle=True)['poses']
    n_pose_data = pose_data.shape[0]
    n_joints_data = joints_data.shape[0]
    tar_pose_upper = pose_data[:, JOINT_MASK_UPPER.astype(bool)].reshape(n_pose_data, 13, 3)
    tar_pose_lower = pose_data[:, JOINT_MASK_LOWER.astype(bool)].reshape(n_pose_data, 9, 3)
    n = min(n_pose_data, n_joints_data)
    tar_pose_upper = tar_pose_upper[:n, :, :]
    tar_pose_lower = tar_pose_lower[:n, :, :]
    joints_data = joints_data[:n, :, :]


    pose_generator = HybrIKJointsToRotmat()
    rec_rot_mats = pose_generator(joints_data)
    rec_rot_mats = torch.tensor(rec_rot_mats, dtype=torch.float32).to(device)

    n = rec_rot_mats.shape[0]
    rec_rot_mats = matrix_to_rotation_6d(rec_rot_mats)
    rec_pose = rotation_6d_to_axis_angle(rec_rot_mats).reshape(n, 22, 3)

    lower_idx = [0,1,2,4,5,7,8,10,11]
    upper_idx = [3,6,7,12,13,14,15,16,17,18,19,20,21]

    tar_pose_upper = torch.tensor(tar_pose_upper, dtype=torch.float32).to(device)
    tar_pose_lower = torch.tensor(tar_pose_lower, dtype=torch.float32).to(device)
    rec_pose_full = torch.zeros(n, 22, 3, dtype=torch.float32).to(device)
    rec_pose_full[:, lower_idx, :] = rec_pose[:, lower_idx, :]
    # rec_pose_full[:, lower_idx, :] = tar_pose_lower
    rec_pose_full[:, upper_idx, :] = tar_pose_upper
    rec_pose_full = rec_pose_full.reshape(n, 22 * 3)

    # rec_lower = rec_rot_mats[:, lower_idx, :].reshape(bs, n, 9 * 6)
    rec_trans = torch.tensor(joints_data[:,0, :3], dtype=torch.float32).to(device)

    smplx_path = "./model_files/smplx_models/"
    smplxmodel = smplx.create(smplx_path,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext='npz',
        use_pca=False,
        ).eval().to(device)
    rec_beta = torch.zeros(300)
    rec_beta = torch.tile(rec_beta, (n, 1))
    rec_beta = rec_beta.to(device)
    outputp = smplxmodel(betas=rec_beta,
                            transl= rec_trans,
                            expression=torch.zeros(rec_beta.shape[0], 100).to(device),
                            jaw_pose=torch.zeros(rec_beta.shape[0], 3).to(device),
                            global_orient=rec_pose_full[:, :3],
                            body_pose=rec_pose_full[:, 3 :21 * 3 + 3],
                            left_hand_pose=torch.zeros(rec_beta.shape[0], 15 * 3).to(device),
                            right_hand_pose=torch.zeros(rec_beta.shape[0], 15 * 3).to(device),
                            leye_pose=torch.zeros(rec_beta.shape[0], 3).to(device),
                            reye_pose=torch.zeros(rec_beta.shape[0], 3).to(device),
                            )

    vertex_saved = outputp.vertices.cpu().numpy()
    np.save(os.path.join(args.output_dir, "2_scott_0_26_26_rec.npy"), vertex_saved)
    pass





    # # Load H3D data from pkl files (pose and shape separated)
    # pkl_dir = "/path/to/BEAT2/beat_english_v2.0.0/new_joints_25fps_reconstructed/10_kieks_0_10_10/"
    # pose_data, shape_data = load_h3d_data(pkl_dir)
    
    # if pose_data is None or shape_data is None:
    #     return
    
    # # Load rotation data from npz file
    # rotation_path = "/path/to/BEAT2/beat_english_v2.0.0/smplxflame_25/10_kieks_0_10_10.npz"
    # rotation_array = load_rotation_data(rotation_path)
    
    # # Frame alignment check - use pose_data as reference
    # if rotation_array.shape[0] != pose_data.shape[0]:
    #     print(f"\nFrame count mismatch:")
    #     print(f"  Pose data: {pose_data.shape[0]} frames")
    #     print(f"  Shape data: {shape_data.shape[0]} frames")
    #     print(f"  Rotation data: {rotation_array.shape[0]} frames")
        
    #     if rotation_array.shape[0] > pose_data.shape[0]:
    #         rotation_array = rotation_array[:pose_data.shape[0]]
    #         print(f"  Trimmed rotation data to: {rotation_array.shape}")
    #     else:
    #         last_frame = rotation_array[-1:]
    #         repeat_count = pose_data.shape[0] - rotation_array.shape[0]
    #         padding = np.repeat(last_frame, repeat_count, axis=0)
    #         rotation_array = np.concatenate([rotation_array, padding], axis=0)
    #         print(f"  Padded rotation data to: {rotation_array.shape}")
    
    # # Keep pose, shape, and rotation data separate
    # hd_3d_rotation = {
    #     'pose': pose_data,
    #     'shape': shape_data,
    #     'rotation': rotation_array
    # }
    
    # print(f"\nFinal data structure:")
    # print(f"  Pose data shape: {hd_3d_rotation['pose'].shape}")
    # print(f"  Shape data shape: {hd_3d_rotation['shape'].shape}")
    # print(f"  Rotation data shape: {hd_3d_rotation['rotation'].shape}")
    
    # # Show available alternatives in the same directory
    # print("\nAvailable alternatives in the same directory:")
    
    # import glob
    # # List .ply files
    # ply_files = glob.glob(os.path.join(pkl_dir, "*.ply"))
    # print(f"Found {len(ply_files)} .ply files")
    
    # # List .pkl files
    # pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    # print(f"Found {len(pkl_files)} .pkl files")
    
    # # Suggest alternative path
    # alternative_path = "/path/to/BEAT2/beat_english_v2.0.0/joints_25fps/10_kieks_0_10_10.npy"
    # if os.path.exists(alternative_path):
    #     print(f"\nAlternative: Use the .npy file at:")
    #     print(f"{alternative_path}")
    #     print("This file contains the actual motion data in NumPy format.")
    

if __name__ == "__main__":
    main() 
