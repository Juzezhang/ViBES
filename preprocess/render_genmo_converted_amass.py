"""
Render videos from GENMO-formatted AMASS files.
Randomly selects files and renders them using the GENMO rendering pipeline.

Usage: python render_genmo_converted_amass.py
"""

import os
import sys
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from einops import einsum

ROOT_DIR = Path(__file__).resolve().parents[1]
GVHMR_ASSET_DIR = ROOT_DIR / "model_files" / "gvhmr"

from utils.genmo.pylogger import Log
from utils.genmo.video_io_utils import save_video
from utils.genmo.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from utils.genmo.camera import create_camera_sensor
from utils.genmo.smplx_utils import make_smplx

from utils.genmo.rotation_conversions import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
)


def load_genmo_formatted_file(npz_path):
    """
    Load a GENMO-formatted AMASS file and extract parameters FROM the motion_vector.
    
    145-dim motion_vector structure:
    - 0:126   -> body_pose_r6d (21 joints * 6)
    - 126:136 -> betas (10)
    - 136:142 -> global_orient_r6d (6)
    - 142:145 -> local_transl_vel (3)
    """
    data = np.load(npz_path, allow_pickle=True)
    motion_vector = torch.from_numpy(data['motion_vector'].astype(np.float32))
    L = motion_vector.shape[0]
    
    # 1. Extract and decode body pose (indices 0-125)
    body_pose_r6d = motion_vector[:, :126].reshape(L, 21, 6)
    body_pose_R = rotation_6d_to_matrix(body_pose_r6d)
    body_pose_aa = matrix_to_axis_angle(body_pose_R).reshape(L, 63)
    
    # 2. Extract betas (indices 126-135)
    betas = motion_vector[:, 126:136]
    
    # 3. Extract and decode global orientation (indices 136-141)
    global_orient_r6d = motion_vector[:, 136:142]
    global_orient_R = rotation_6d_to_matrix(global_orient_r6d)
    global_orient_aa = matrix_to_axis_angle(global_orient_R)
    
    # 4. Extract and integrate local translation velocity (indices 142-144)
    local_transl_vel = motion_vector[:, 142:145]
    
    # World velocity = R_global @ local_velocity
    world_vel = torch.einsum('lij,lj->li', global_orient_R, local_transl_vel)
    
    # Integrate velocity to get translation (starting from origin)
    # The renderer's normalize_coordinates will handle centering anyway.
    trans = torch.cumsum(world_vel, dim=0)
    
    fps = float(data['fps']) if 'fps' in data else 30.0
    
    return {
        'body_pose': body_pose_aa.numpy(),
        'global_orient': global_orient_aa.numpy(),
        'transl': trans.numpy(),
        'betas': betas.numpy(),
        'num_frames': L,
        'fps': fps,
    }


def render_amass_motion(smpl_params, out_path, width=1280, height=720, max_duration=60.0):
    """
    Render a video from SMPL parameters (AMASS format).
    
    Important: SMPL-X body model expects Y-up coordinates internally.
    The normalize_coordinates function handles transformation for rendering.
    
    Args:
        smpl_params: dict with body_pose, global_orient, transl, betas (all in Y-up)
        out_path: path to save the output video
        width, height: video dimensions
        max_duration: maximum duration in seconds (default: 60s)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize body models
    Log.info("Initializing body models...")
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(GVHMR_ASSET_DIR / "smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(GVHMR_ASSET_DIR / "smpl_neutral_J_regressor.pt").cuda()
    
    # Prepare SMPL-X parameters
    # SMPL-X expects body_pose as (L, 21, 3) or (L, 63)
    fps = smpl_params.get('fps', 30.0)
    L_original = smpl_params['num_frames']
    
    # Limit to max_duration seconds
    max_frames = int(max_duration * fps)
    L = min(L_original, max_frames)
    
    if L < L_original:
        Log.info(f"Truncating from {L_original} to {L} frames ({max_duration}s at {fps}fps)")
    
    body_pose = torch.from_numpy(smpl_params['body_pose'][:L].astype(np.float32))  # (L, 63)
    global_orient = torch.from_numpy(smpl_params['global_orient'][:L].astype(np.float32))  # (L, 3)
    transl = torch.from_numpy(smpl_params['transl'][:L].astype(np.float32))  # (L, 3)
    betas = torch.from_numpy(smpl_params['betas'][:L].astype(np.float32))  # (L, 10)
    
    # Use first frame's betas for the whole sequence (common practice)
    betas_single = betas[0] if len(betas.shape) > 1 else betas  # (10,)
    
    # Create parameter dict for SMPL-X
    smplx_params = {
        'body_pose': body_pose.cuda(),  # (L, 63)
        'global_orient': global_orient.cuda(),  # (L, 3)
        'transl': transl.cuda(),  # (L, 3)
        'betas': betas_single.unsqueeze(0).expand(L, -1).cuda(),  # (L, 10)
    }
    
    # Generate vertices from SMPL parameters
    Log.info("Generating vertices...")
    smpl_out = smplx(**smplx_params)
    verts_ay = torch.stack([torch.matmul(smplx2smpl, v) for v in smpl_out.vertices])
    
    # Transform to world coordinates (Origin XZ, Floor level, Face +Z)
    def normalize_coordinates(v):
        v = v.clone()
        # Move to origin
        offset = einsum(J_regressor, v[0], "j v, v i -> j i")[0]
        offset[1] = v[:, :, 1].min()
        v -= offset
        # Face-Z alignment
        T_rot = compute_T_ayfz2ay(einsum(J_regressor, v[[0]], "j v, l v i -> l j i"), inverse=True)
        return apply_T_on_points(v, T_rot)
    
    Log.info("Normalizing coordinates...")
    verts_glob = normalize_coordinates(verts_ay)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")
    
    # Setup renderer (24mm lens)
    Log.info(f"Setting up renderer ({width}x{height})...")
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)
    
    # Setup ground
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(max(scale, 3) * 1.5, cx, cz)
    
    # Get optimal static cameras
    cam_R, cam_T, lights = get_global_cameras_static(verts_glob.cpu())
    color = torch.tensor([0.69, 0.39, 0.96]).cuda()  # GENMO Purple
    
    # Render frames
    frames = []
    num_frames = len(verts_glob)
    Log.info(f"Rendering {num_frames} frames...")
    for i in tqdm(range(num_frames), desc="Rendering"):
        cams = renderer.create_camera(cam_R[i], cam_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cams, lights)
        frames.append(img)
    
    # Save video
    Log.info(f"Saving video to: {out_path} (fps={fps})")
    save_video(np.array(frames), str(out_path), fps=int(fps), crf=23)
    Log.info("Done!")
    
    return out_path


def main():
    # Configuration
    input_dir = Path("/path/to/AMASS/amass_genmo_25")
    output_dir = Path("/path/to/AMASS/rendered_outputs")
    num_files = 3
    max_duration = 20.0  # Maximum duration in seconds
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("GENMO-Formatted AMASS Renderer")
    print("=" * 80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files to render: {num_files}")
    print(f"Max duration: {max_duration}s")
    print()
    
    # Find all .npz files
    npz_files = list(input_dir.rglob('*.npz'))
    
    if len(npz_files) == 0:
        print("ERROR: No .npz files found in input directory!")
        sys.exit(1)
    
    print(f"Found {len(npz_files)} .npz files")
    
    # Randomly select files
    num_to_render = min(num_files, len(npz_files))
    selected_files = random.sample(npz_files, num_to_render)
    
    print(f"\nSelected {num_to_render} random files:")
    for f in selected_files:
        print(f"  - {f.name}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Render each selected file
    results = []
    for i, npz_path in enumerate(selected_files):
        print(f"\n{'=' * 80}")
        print(f"[{i+1}/{num_to_render}] Rendering: {npz_path.name}")
        print("=" * 80)
        
        try:
            # Load SMPL parameters
            smpl_params = load_genmo_formatted_file(npz_path)
            
            fps = smpl_params['fps']
            total_frames = smpl_params['num_frames']
            duration = total_frames / fps
            frames_to_render = min(total_frames, int(max_duration * fps))
            
            print(f"  Total frames: {total_frames} ({duration:.1f}s at {fps}fps)")
            print(f"  Rendering: {frames_to_render} frames ({min(duration, max_duration):.1f}s)")
            
            # Create output path (preserve relative structure)
            rel_path = npz_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix('.mp4')
            
            # Render (limited to max_duration)
            render_amass_motion(smpl_params, out_path, max_duration=max_duration)
            
            results.append((npz_path.name, "SUCCESS", str(out_path)))
            
        except Exception as e:
            Log.error(f"Failed to render {npz_path.name}: {e}")
            results.append((npz_path.name, "FAILED", str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r[1] == "SUCCESS")
    print(f"Rendered: {success_count}/{num_to_render}")
    print()
    
    for name, status, info in results:
        if status == "SUCCESS":
            print(f"  ✓ {name}")
            print(f"      -> {info}")
        else:
            print(f"  ✗ {name}")
            print(f"      Error: {info}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    main()
