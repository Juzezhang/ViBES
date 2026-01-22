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
    mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=faces)
    mesh.export(path)

# Define configurable variables for sequence and video
parser = argparse.ArgumentParser(description='Process facial meshes for visualization')
parser.add_argument('--seq_name', type=str, default="e12fed95-8f11-4255-b8c7-f5c0346b8e59", 
                    help='Sequence folder name')
parser.add_argument('--video_name', type=str, default="5f113ec4c304894ad573296c", 
                    help='Video name within the sequence')
parser.add_argument('--start_frame', type=int, default=45000, 
                    help='Starting frame index')
parser.add_argument('--end_frame', type=int, default=45500, 
                    help='Ending frame index')
parser.add_argument('--batch_size', type=int, default=25, 
                    help='Batch size for processing')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA even if available')
parser.add_argument('--gpu_id', type=int, default=2,
                    help='GPU ID to use if CUDA is available (default: 0)')
args = parser.parse_args()

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

# Ensure batch size is not larger than total frames
total_frames = end_frame - start_frame
if batch_size > total_frames:
    batch_size = total_frames
    print(f"Adjusted batch size to match total frames: {batch_size}")

# Construct the paths using the variables
# flame_path = f"/simurgh/group/yuheng/CANDOR_spectre_mica/FLAME_coeffs/{seq_name}/{video_name}.npz"
flame_path = f"/simurgh/group/yuheng/CANDOR_spectre_mica/FLAME_coeffs/{seq_name}/{video_name}.npz"
audio_path = f"/simurgh/group/yuheng/CANDOR_processed/{seq_name}/{video_name}.mp3"
image_folder = f"/simurgh/group/yuheng/CANDOR_spectre_mica/SPECTRE_coeffs/{seq_name}/{video_name}"

# Alternate path example (commented out)
# flame_path = f"/simurgh/group/yuheng/CANDOR_spectre_mica/SPECTRE_coeffs/230a227f-e1e3-46ef-8817-37912ba9f87a/5c75471e5eb59f000131c7a4.npz"

print(f"Processing sequence: {seq_name}")
print(f"Video: {video_name}")
print(f"Frames: {start_frame} to {end_frame} (total: {total_frames})")
print(f"Batch size: {batch_size}")

audio, sr = torchaudio.load(audio_path)
audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

flame_data = np.load(flame_path)

# Convert NumPy arrays to PyTorch tensors and move to GPU
exp = torch.tensor(flame_data['exp']).float().to(device)
shape = torch.tensor(flame_data['shape']).float().to(device)
pose = torch.tensor(flame_data['pose']).float().to(device)

# head_temp =  torch.tensor(axis_angle_to_6d_np(flame_data['pose']))
# head_temp = axis_angle_to_6d(pose[:, :3])
# pose[:, :3] = rotation_6d_to_axis_angle(head_temp)


model_path = "./model_files/FLAME2020/"

# Initialize FLAME model with the adjusted batch size and move to GPU
flame_model = FLAME(model_path, num_expression_coeffs=50, ext='pkl', batch_size=batch_size).to(device)

mesh_renderer = RenderMesh(image_size=256, faces=flame_model.faces, scale=1.0)
pred_images = []

# Process faces in batches
for batch_start in tqdm(range(5000, 5125, batch_size)):
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
    
    # Get current frame's rendered image
    for b in range(actual_batch_size):
        image_render = image_renders[b]

        # Convert tensor to numpy and rearrange dimensions
        image_render_np = image_render.cpu().numpy()
        image_render_np = np.transpose(image_render_np, (1, 2, 0))  # Convert from [C,H,W] to [H,W,C]

        # Convert to tensor and append to results
        combined_image = torch.from_numpy(image_render_np).float()
        pred_images.append(combined_image)


# Save video
if len(pred_images) > 0:
    # Stack the tensors and convert from [N, H, W, 3] to [N, 3, H, W]
    pred_images_tensor = torch.stack(pred_images)  # Shape: [N, H, W, 3]
    pred_images_tensor = pred_images_tensor.permute(0, 3, 1, 2)  # Reshape to [N, 3, H, W]
    
    # Use sequence and video name in the output filename
    output_dir = "./output_videos"
    os.makedirs(output_dir, exist_ok=True)
    dump_path = join(output_dir, f"random.mp4")
    
    print(f"Saving video to: {dump_path}")
    audio_clip = 0 * audio[int(5000/25.0*16000):int(5125/25.0*16000)]
    write_video(pred_images_tensor*255.0, dump_path, 25, audio_clip, 16000, "aac")
    print("Video saved successfully!")
else:
    print("Error: No frames were processed successfully")

