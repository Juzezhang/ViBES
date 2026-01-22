import torch
from smplx import FLAME
import numpy as np
import trimesh
import torch
from os.path import join, exists
import os
import argparse
from conver_agent.utils.renderer_utils import RenderMesh
import matplotlib
import matplotlib.pyplot as plt
import torchaudio
from conver_agent.utils.utils_videos import write_video
import cv2
matplotlib.use('Agg')
from tqdm import tqdm
from conver_agent.utils.rotation_conversions import rotation_6d_to_axis_angle, axis_angle_to_6d, axis_angle_to_6d_np

def save_obj(verts, faces, path):
    """Save mesh vertices and faces to an OBJ file."""
    mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=faces)
    mesh.export(path)

import soundfile as sf

# ---------------------- Configurable Dataset and Model Paths ----------------------
FLAME_COEFFS_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/FLAME_coeffs"
FLAME_COEFFS_25_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/FLAME_coeffs_25"
VIDEO_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/video_20241226"
AUDIO_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/audios_original"
MODEL_DIR = "./model_files/FLAME2020/"
OUTPUT_DIR = "./output_videos"

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description='Process facial meshes for visualization')
parser.add_argument('--video_name', type=str, default="203011908.mp4", 
                    help='Video name within the sequence')
parser.add_argument('--start_frame', type=int, default=0, 
                    help='Starting frame index')
parser.add_argument('--end_frame', type=int, default=500, 
                    help='Ending frame index')
parser.add_argument('--batch_size', type=int, default=100, 
                    help='Batch size for processing')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA even if available')
parser.add_argument('--gpu_id', type=int, default=2,
                    help='GPU ID to use if CUDA is available (default: 0)')
args = parser.parse_args()

# ---------------------- Device Setup ----------------------
use_cuda = torch.cuda.is_available() and not args.no_cuda
if use_cuda:
    torch.cuda.set_device(args.gpu_id)
    print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    device = torch.device(f"cuda:{args.gpu_id}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ---------------------- Path and Variable Setup ----------------------
video_name = args.video_name
start_frame = args.start_frame
end_frame = args.end_frame
batch_size = args.batch_size

# Load FLAME coefficients (expression, shape, pose) for the video (original FPS)
coeffs_path = join(FLAME_COEFFS_DIR, video_name.replace(".mp4", ".npz"))
data = np.load(coeffs_path)
exp = data['exp']
shape = data['shape']
pose = data['pose']

# Load FLAME coefficients for the video (25 FPS)
coeffs_25_path = join(FLAME_COEFFS_25_DIR, video_name.replace(".mp4", ".npz"))
data_25 = np.load(coeffs_25_path)
exp_25 = data_25['exp']
shape_25 = data_25['shape']
pose_25 = data_25['pose']

# Convert numpy arrays to torch tensors and move to device
exp = torch.FloatTensor(exp).to(device)
shape = torch.FloatTensor(shape).to(device)
pose = torch.FloatTensor(pose).to(device)

exp_25 = torch.FloatTensor(exp_25).to(device)
shape_25 = torch.FloatTensor(shape_25).to(device)
pose_25 = torch.FloatTensor(pose_25).to(device)

# Open video file for reading frames
video_path = join(VIDEO_DIR, video_name)
cap = cv2.VideoCapture(video_path)

# Load audio file (try soundfile first, fallback to torchaudio)
audio_path = join(AUDIO_DIR, video_name.replace('.mp4', '.wav'))
try:
    audio_data, sr = sf.read(audio_path)
    audio = torch.FloatTensor(audio_data)
    if len(audio.shape) > 1:
        audio = audio.mean(dim=1)  # Convert to mono if stereo
except Exception as e:
    print(f"Failed to load with soundfile, trying torchaudio: {e}")
    try:
        audio, sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)  # Convert to mono if stereo
    except Exception as e:
        print(f"Failed to load audio file: {e}")
        # Fallback: create silent audio of appropriate length
        duration_frames = end_frame - start_frame
        audio = torch.zeros(int(duration_frames / 25.0 * 16000))
        sr = 16000

# Resample audio to 16kHz if needed
if sr != 16000:
    audio = torchaudio.transforms.Resample(sr, 16000)(audio)

# ---------------------- FLAME Model Setup ----------------------
# Initialize FLAME model with batch size and move to device
flame_model = FLAME(MODEL_DIR, num_expression_coeffs=50, ext='pkl', batch_size=batch_size).to(device)

# Convert FLAME faces to tensor
faces = torch.LongTensor(flame_model.faces.astype(np.int64))
# Initialize mesh renderer
mesh_renderer = RenderMesh(image_size=256, faces=faces, scale=1.0)

pred_images = []  # Store rendered frames for video output

# Determine the minimum number of frames to compare
n_frames = min(exp.shape[0], exp_25.shape[0], end_frame - start_frame)

# ---------------------- Main Processing Loop ----------------------
for batch_start in tqdm(range(0, n_frames, batch_size)):
    actual_batch_size = min(batch_size, n_frames - batch_start)
    batch_end = batch_start + actual_batch_size

    # Prepare FLAME parameters for the current batch (original FPS)
    flame_param = {
        'global_orient': pose[batch_start:batch_end, :3],
        'expression': exp[batch_start:batch_end, :],
        'jaw_pose': pose[batch_start:batch_end, 3:],
        'shape': shape[batch_start:batch_end, :],
    }
    # Prepare FLAME parameters for the current batch (25 FPS)
    flame_param_25 = {
        'global_orient': pose_25[batch_start:batch_end, :3],
        'expression': exp_25[batch_start:batch_end, :],
        'jaw_pose': pose_25[batch_start:batch_end, 3:],
        'shape': shape_25[batch_start:batch_end, :],
    }

    # Run FLAME model to get mesh vertices (inference mode)
    with torch.no_grad():
        flame_out = flame_model(
            global_orient=flame_param['global_orient'],
            expression=flame_param['expression'],
            jaw_pose=flame_param['jaw_pose'],
            shape=flame_param['shape'],
        )
        flame_out_25 = flame_model(
            global_orient=flame_param_25['global_orient'],
            expression=flame_param_25['expression'],
            jaw_pose=flame_param_25['jaw_pose'],
            shape=flame_param_25['shape'],
        )
    verts = flame_out['vertices'].detach()
    verts_25 = flame_out_25['vertices'].detach()

    # Render all meshes in the batch
    render_output = mesh_renderer(verts)
    image_renders = render_output[0] / 255.0  # Normalize to [0,1]
    render_output_25 = mesh_renderer(verts_25)
    image_renders_25 = render_output_25[0] / 255.0

    # Process each frame in the batch
    for b in range(actual_batch_size):
        # Get rendered mesh image for this frame (original FPS)
        image_render = image_renders[b]
        image_render_np = image_render.cpu().numpy()
        image_render_np = np.transpose(image_render_np, (1, 2, 0))  # [C,H,W] -> [H,W,C]

        # Get rendered mesh image for this frame (25 FPS)
        image_render_25 = image_renders_25[b]
        image_render_25_np = image_render_25.cpu().numpy()
        image_render_25_np = np.transpose(image_render_25_np, (1, 2, 0))  # [C,H,W] -> [H,W,C]

        # Resize both to the same height if needed
        height = max(image_render_np.shape[0], image_render_25_np.shape[0])
        width1 = image_render_np.shape[1]
        width2 = image_render_25_np.shape[1]
        if image_render_np.shape[0] != height:
            image_render_np = cv2.resize(image_render_np, (width1, height))
        if image_render_25_np.shape[0] != height:
            image_render_25_np = cv2.resize(image_render_25_np, (width2, height))

        # Concatenate the two images horizontally
        combined_image = np.concatenate([image_render_np, image_render_25_np], axis=1)
        combined_image = np.asarray(combined_image).astype(np.float32)
        combined_image = torch.FloatTensor(combined_image)
        pred_images.append(combined_image)

# ---------------------- Save Output Video ----------------------
if len(pred_images) > 0:
    # Stack tensors and permute to [N, 3, H, W] for video writing
    pred_images_tensor = torch.stack(pred_images)  # [N, H, W, 3]
    pred_images_tensor = pred_images_tensor.permute(0, 3, 1, 2)  # [N, 3, H, W]

    # Prepare output directory and file path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dump_path = join(OUTPUT_DIR, f"{video_name.split('/')[0]}_{start_frame}_{start_frame+n_frames}_compare.mp4")

    print(f"Saving video to: {dump_path}")
    # Extract audio segment corresponding to the selected frames (optional: use original audio, or silence)
    # Here, just use silence for simplicity
    audio_clip = torch.zeros(int(n_frames / 25.0 * 16000))
    write_video(pred_images_tensor*255.0, dump_path, 25, audio_clip, 16000, "aac")
    print("Video saved successfully!")
else:
    print("Error: No frames were processed successfully")

