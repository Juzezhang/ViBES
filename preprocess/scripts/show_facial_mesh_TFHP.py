import torch
from smplx import FLAME
import numpy as np
import trimesh
import torch
from os.path import join, exists
import os
import argparse
from multimodal_tokenizers.utils.renderer_utils import RenderMesh
import matplotlib
import matplotlib.pyplot as plt
import torchaudio
from multimodal_tokenizers.utils.utils_videos import write_video
import cv2
matplotlib.use('Agg')
from tqdm import tqdm
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle, axis_angle_to_6d, axis_angle_to_6d_np
def save_obj(verts, faces, path):
    mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=faces)
    mesh.export(path)

import lmdb
import pickle  # 如果存的是序列化对象
import os
import io
import soundfile as sf


# Define configurable variables for sequence and video
parser = argparse.ArgumentParser(description='Process facial meshes for visualization')
parser.add_argument('--seq_name', type=str, default="TH_00139/000/000", 
                    help='Sequence folder name')
parser.add_argument('--video_name', type=str, default="TH_00139/000.mp4", 
                    help='Video name within the sequence')
parser.add_argument('--start_frame', type=int, default=0, 
                    help='Starting frame index')
parser.add_argument('--end_frame', type=int, default=100, 
                    help='Ending frame index')
parser.add_argument('--batch_size', type=int, default=50, 
                    help='Batch size for processing')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA even if available')
parser.add_argument('--gpu_id', type=int, default=2,
                    help='GPU ID to use if CUDA is available (default: 0)')
args = parser.parse_args()


# LMDB 路径是文件夹，不是 .mdb 文件本身
lmdb_path = "/path/to/conversational_agent/datasets/TFHP/HDTF_TFHP-lmdb"

# 打开 LMDB
env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

# Check if CUDA is available and not disabled by user
use_cuda = torch.cuda.is_available() and not args.no_cuda

if use_cuda:
    # Set the GPU to use
    torch.cuda.set_device(args.gpu_id)
    print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    device = torch.device(f"cuda:{args.gpu_id}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Use the variables in the paths
seq_name = args.seq_name
video_name = args.video_name
start_frame = args.start_frame
end_frame = args.end_frame
batch_size = args.batch_size

target_key = f"{seq_name}"

with env.begin() as txn:
    value = txn.get(target_key.encode())
    if value is None:
        print("Key not found!")
    else:
        data = pickle.loads(value)
        audio_bytes = data['audio']
        # Decode FLAC
        audio_io = io.BytesIO(audio_bytes)
        audio, sample_rate = sf.read(audio_io)
        
        coef = data['coef']
        exp = coef['exp']
        shape = coef['shape']
        pose = coef['pose']

        exp = torch.FloatTensor(exp).to(device)
        shape = torch.FloatTensor(shape).to(device)
        pose = torch.FloatTensor(pose).to(device)

        print("Sample rate:", sample_rate)
        print("Audio shape:", audio.shape)


video_path = f"/path/to/TFHP/data/{video_name}"

cap = cv2.VideoCapture(video_path)

# # Ensure batch size is not larger than total frames
# total_frames = end_frame - start_frame
# if batch_size > total_frames:
#     batch_size = total_frames
#     print(f"Adjusted batch size to match total frames: {batch_size}")

# # Construct the paths using the variables
# # flame_path = f"/path/to/CANDOR_spectre_mica/FLAME_coeffs/{seq_name}/{video_name}.npz"
# flame_path = f"/path/to/CANDOR_spectre_mica/FLAME_coeffs/{seq_name}/{video_name}.npz"
# audio_path = f"/path/to/CANDOR_processed/{seq_name}/{video_name}.mp3"
# # image_folder = f"/path/to/CANDOR_spectre_mica/SPECTRE_coeffs/{seq_name}/{video_name}"
# video_path = f"/path/to/CANDOR_processed/{seq_name}/{video_name}.mp4"

# cap = cv2.VideoCapture(video_path)
# # Alternate path example (commented out)
# # flame_path = f"/path/to/CANDOR_spectre_mica/SPECTRE_coeffs/230a227f-e1e3-46ef-8817-37912ba9f87a/5c75471e5eb59f000131c7a4.npz"

# print(f"Processing sequence: {seq_name}")
# print(f"Video: {video_name}")
# print(f"Frames: {start_frame} to {end_frame} (total: {total_frames})")
# print(f"Batch size: {batch_size}")

# audio, sr = torchaudio.load(audio_path)
# audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

# flame_data = np.load(flame_path)

# # Convert NumPy arrays to PyTorch tensors and move to GPU
# test = flame_data['exp']
# print("Expression data shape:", test.shape, "dtype:", test.dtype)

# # Convert data with explicit dtype handling
# exp_np = np.asarray(flame_data['exp']).astype(np.float32)
# shape_np = np.asarray(flame_data['shape']).astype(np.float32)
# pose_np = np.asarray(flame_data['pose']).astype(np.float32)

# exp = torch.FloatTensor(exp_np).to(device)
# shape = torch.FloatTensor(shape_np).to(device)
# pose = torch.FloatTensor(pose_np).to(device)

# head_temp =  torch.tensor(axis_angle_to_6d_np(flame_data['pose']))
# head_temp = axis_angle_to_6d(pose[:, :3])
# pose[:, :3] = rotation_6d_to_axis_angle(head_temp)


model_path = "./model_files/FLAME2020/"

# Initialize FLAME model with the adjusted batch size and move to GPU
flame_model = FLAME(model_path, num_expression_coeffs=50, ext='pkl', batch_size=batch_size).to(device)

# Convert faces to tensor explicitly
faces = torch.LongTensor(flame_model.faces.astype(np.int64))
mesh_renderer = RenderMesh(image_size=256, faces=faces, scale=1.0)
pred_images = []

# Process faces in batches
for batch_start in tqdm(range(start_frame, end_frame, batch_size)):
    # Calculate actual batch size (may be smaller at the end)
    actual_batch_size = min(batch_size, end_frame - batch_start)
    batch_end = batch_start + actual_batch_size
    
    # Prepare FLAME parameters for the batch (already on GPU)
    flame_param = {
        'global_orient': pose[batch_start:batch_end, :3],
        'expression': exp[batch_start:batch_end, :],
        'jaw_pose': pose[batch_start:batch_end, 3:],
        'shape': shape[batch_start:batch_end, :],
    }
    
    # Run FLAME model
    with torch.no_grad():  # Add no_grad for inference to save memory
        flame_out = flame_model(
            global_orient=flame_param['global_orient'],
            expression=flame_param['expression'],
            jaw_pose=flame_param['jaw_pose'],
            shape=flame_param['shape'],
        )
    verts = flame_out['vertices'].detach()
    
    # Render all meshes in the batch at once
    # mesh_renderer returns a tuple where the first element is the rendered image data
    render_output = mesh_renderer(verts)
    image_renders = render_output[0] / 255.0  # Access the first element of the tuple
    
    # Process each frame in the batch
    for b in range(actual_batch_size):
        frame_idx = batch_start + b
        
        # Get the original image
        # image_path = join(image_folder, f'{frame_idx:06d}.jpg')
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, image_original = cap.read()
        if not ret:
            image_original = np.zeros((256, 256, 3), dtype=np.uint8)

        # Get current frame's rendered image
        image_render = image_renders[b]
        
        # Convert tensor to numpy and rearrange dimensions
        image_render_np = image_render.cpu().numpy()
        image_render_np = np.transpose(image_render_np, (1, 2, 0))  # Convert from [C,H,W] to [H,W,C]
        
        # Get dimensions for resizing
        original_height = image_original.shape[0]
        render_height, render_width = image_render_np.shape[0], image_render_np.shape[1]
        
        # Safety checks for dimensions
        if render_height <= 0:
            render_height = 1
        
        # Calculate new width to maintain aspect ratio
        new_width = int((original_height / render_height) * render_width)
        if new_width <= 0:
            new_width = original_height
        
        # Resize the rendered image
        image_render_resized = cv2.resize(image_render_np, (new_width, original_height))
        
        # Convert original image to RGB
        # image_original_rgb = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB) / 255.0
        image_original_rgb = image_original[:, :, ::-1] / 255.0
        # Create a combined image (side by side)
        combined_width = image_original_rgb.shape[1] + new_width
        combined_image = np.zeros((original_height, combined_width, 3))
        combined_image[:, :image_original_rgb.shape[1], :] = image_original_rgb
        combined_image[:, image_original_rgb.shape[1]:, :] = image_render_resized
        
        # Convert to tensor and append to results
        combined_image = np.asarray(combined_image).astype(np.float32)
        combined_image = torch.FloatTensor(combined_image)
        # combined_image = torch.from_numpy(combined_image).float()
        pred_images.append(combined_image)

# Save video
if len(pred_images) > 0:
    # Stack the tensors and convert from [N, H, W, 3] to [N, 3, H, W]
    pred_images_tensor = torch.stack(pred_images)  # Shape: [N, H, W, 3]
    pred_images_tensor = pred_images_tensor.permute(0, 3, 1, 2)  # Reshape to [N, 3, H, W]
    
    # Use sequence and video name in the output filename
    output_dir = "./output_videos"
    os.makedirs(output_dir, exist_ok=True)
    dump_path = join(output_dir, f"{video_name.split('/')[0]}_{start_frame}_{end_frame}.mp4")
    
    print(f"Saving video to: {dump_path}")
    audio_clip = audio[int(start_frame/25.0*16000):int(end_frame/25.0*16000)]
    write_video(pred_images_tensor*255.0, dump_path, 25, audio_clip, 16000, "aac")
    print("Video saved successfully!")
else:
    print("Error: No frames were processed successfully")

