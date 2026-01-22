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
FLAME_COEFFS_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/FLAME_coeffs_25"
# FLAME_COEFFS_DIR = "/simurgh/group/yuheng/youtube_spectre_mica/Talk_video_summary_English_20241226/FLAME_coeffs_1"
# FLAME_COEFFS_DIR = "/simurgh/group/yuheng/youtube_spectre_mica/Talk_video_summary_English_20241226/FLAME_coeffs_1kfrm_0814"
VIDEO_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/video_20241226"
AUDIO_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/audios_original"
MODEL_DIR = "./model_files/FLAME2020/"
OUTPUT_DIR = "/simurgh/group/juze/datasets/YouTube_Talking/Reconstructed_videos_first1k"
# OUTPUT_DIR = "./output_videos_testset"

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description='Process facial meshes for visualization')
# parser.add_argument('--video_name', type=str, default="202010655.mp4", 
#                     help='Video name within the sequence')
parser.add_argument('--sequence_dir', type=str, default=FLAME_COEFFS_DIR, 
                    help='Sequence directory')
parser.add_argument('--start_frame', type=int, default=0, 
                    help='Starting time index')
parser.add_argument('--end_frame', type=int, default=500, 
                    help='Ending time index')
parser.add_argument('--batch_size', type=int, default=100, 
                    help='Batch size for processing')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA even if available')
parser.add_argument('--gpu_id', type=int, default=1,
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
sequence_dir = args.sequence_dir

for sequence_name in os.listdir(sequence_dir):
    # if sequence_name != "202008767.npz":
    #     continue
    if sequence_name in [
            '202008830.npz']:
        continue

    print(f"Processing sequence: {sequence_name}")

    sequence_name = sequence_name.replace(".npz", "")
    start_frame = args.start_frame
    end_frame = args.end_frame
    batch_size = args.batch_size

    # Prepare output directory and file path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dump_path = join(OUTPUT_DIR, f"{sequence_name}_{start_frame}_{end_frame}.mp4")
    if os.path.exists(dump_path):
        print(f"Video already exists: {dump_path}")
        continue

    # Load FLAME coefficients (expression, shape, pose) for the video
    coeffs_path = join(sequence_dir, sequence_name + ".npz")
    data = np.load(coeffs_path)
    exp = data['exp']
    shape = data['shape']
    pose = data['pose']

    # Convert numpy arrays to torch tensors and move to device
    exp = torch.FloatTensor(exp).to(device)
    shape = torch.FloatTensor(shape).to(device)
    pose = torch.FloatTensor(pose).to(device)

    # Open video file for reading frames
    video_path = join(VIDEO_DIR, sequence_name + ".mp4")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    end_time = end_frame / video_fps
    start_time = start_frame / video_fps

    # Load audio file (try soundfile first, fallback to torchaudio)
    audio_path = join(AUDIO_DIR, sequence_name + ".wav")
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

    # ---------------------- Main Processing Loop ----------------------
    for batch_start in tqdm(range(start_frame, end_frame, batch_size)):
        # Determine actual batch size (may be smaller at the end)
        actual_batch_size = min(batch_size, end_frame - batch_start)
        batch_end = batch_start + actual_batch_size

        # Prepare FLAME parameters for the current batch
        flame_param = {
            'global_orient': pose[batch_start:batch_end, :3],
            'expression': exp[batch_start:batch_end, :],
            'jaw_pose': pose[batch_start:batch_end, 3:],
            'shape': shape[batch_start:batch_end, :],
        }

        # Run FLAME model to get mesh vertices (inference mode)
        with torch.no_grad():
            flame_out = flame_model(
                global_orient=flame_param['global_orient'],
                expression=flame_param['expression'],
                jaw_pose=flame_param['jaw_pose'],
                shape=flame_param['shape'],
            )
        verts = flame_out['vertices'].detach()

        # Render all meshes in the batch
        render_output = mesh_renderer(verts)
        image_renders = render_output[0] / 255.0  # Normalize to [0,1]

        # Process each frame in the batch
        for b in range(actual_batch_size):
            frame_idx = batch_start + b

            # Read the original frame from the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, image_original = cap.read()
            if not ret:
                # If frame not found, use a black image
                image_original = np.zeros((256, 256, 3), dtype=np.uint8)

            # Get rendered mesh image for this frame
            image_render = image_renders[b]
            image_render_np = image_render.cpu().numpy()
            image_render_np = np.transpose(image_render_np, (1, 2, 0))  # [C,H,W] -> [H,W,C]

            # Resize rendered image to match original frame height
            original_height = image_original.shape[0]
            render_height, render_width = image_render_np.shape[0], image_render_np.shape[1]
            if render_height <= 0:
                render_height = 1
            new_width = int((original_height / render_height) * render_width)
            if new_width <= 0:
                new_width = original_height
            image_render_np = cv2.resize(image_render_np, (new_width, original_height))

            # Convert original image to RGB and normalize
            image_original_rgb = image_original[:, :, ::-1] / 255.0

            # Concatenate original and rendered images side by side
            combined_width = image_original_rgb.shape[1] + new_width
            combined_image = np.zeros((original_height, combined_width, 3))
            combined_image[:, :image_original_rgb.shape[1], :] = image_original_rgb
            combined_image[:, image_original_rgb.shape[1]:, :] = image_render_np

            # Convert to float tensor and append to results
            combined_image = np.asarray(combined_image).astype(np.float32)
            combined_image = torch.FloatTensor(combined_image)
            pred_images.append(combined_image)

    # ---------------------- Save Output Video ----------------------
    if len(pred_images) > 0:
        # Stack tensors and permute to [N, 3, H, W] for video writing
        pred_images_tensor = torch.stack(pred_images)  # [N, H, W, 3]
        pred_images_tensor = pred_images_tensor.permute(0, 3, 1, 2)  # [N, 3, H, W]
        print(f"Saving video to: {dump_path}")
        # Extract audio segment corresponding to the selected frames
        # audio_clip = audio[int(start_frame/25.0*16000):int(end_frame/25.0*16000)]
        audio_clip = audio[int(start_frame/video_fps*16000):int(end_frame/video_fps*16000)]
        write_video(pred_images_tensor*255.0, dump_path, int(video_fps), audio_clip, 16000, "aac")
        print("Video saved successfully!")
    else:
        print("Error: No frames were processed successfully")

