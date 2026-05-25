import torch
import sys
# sys.path.append("..")
from smplx import FLAME
import numpy as np
import trimesh
import torch
from os.path import join, exists
import os
import argparse
from multimodal_tokenizers.utils.renderer_utils import RenderBodyMesh
import matplotlib
import matplotlib.pyplot as plt
import torchaudio
from multimodal_tokenizers.utils.utils_videos import write_video
import cv2
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from tqdm import tqdm
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle, axis_angle_to_6d, axis_angle_to_6d_np, matrix_to_axis_angle
import smplx
import pickle
from joblib import load
from multimodal_tokenizers.utils.utils_4dhumans import Renderer, cam_crop_to_full

def save_obj(verts, faces, path):
    """Save mesh vertices and faces to an OBJ file."""
    mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=faces)
    mesh.export(path)
    
import soundfile as sf

# ---------------------- Configurable Dataset and Model Paths ----------------------
# FLAME_COEFFS_DIR = "/path/to/YouTube_Talking/FLAME_coeffs"
# FLAME_COEFFS_DIR = "/path/to/youtube_spectre_mica/Talk_video_summary_English_20241226/FLAME_coeffs_1"
SMPL_COEFFS_DIR = "/path/to/YouTube_Talking/4d_humans_results"

VIDEO_DIR = "/path/to/YouTube_Talking/video_20241226"
AUDIO_DIR = "/path/to/YouTube_Talking/audios_original"
MODEL_DIR = "./model_files/FLAME2020/"
SMPL_PATH = "../language_of_motion/model_files/smplx_models"
# OUTPUT_DIR = "/path/to/YouTube_Talking/Reconstructed_videos_body"
OUTPUT_DIR = "/path/to/YouTube_Talking/Reconstructed_videos_body_first1k"

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description='Process facial meshes for visualization')
parser.add_argument('--video_name', type=str, default=None, 
                    help='Video name within the sequence. If not specified, process all sequences in SMPL_COEFFS_DIR')
parser.add_argument('--start_frame', type=int, default=0, 
                    help='Starting frame index')
parser.add_argument('--end_frame', type=int, default=500, 
                    help='Ending frame index')
parser.add_argument('--batch_size', type=int, default=1, 
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

# ---------------------- Helper Functions ----------------------
def find_all_sequences(smpl_coeffs_dir):
    """Find all sequences in the SMPL_COEFFS_DIR that have valid pkl files."""
    sequences = []
    if not os.path.exists(smpl_coeffs_dir):
        print(f"Warning: SMPL_COEFFS_DIR does not exist: {smpl_coeffs_dir}")
        return sequences
    
    for item in os.listdir(smpl_coeffs_dir):
        item_path = join(smpl_coeffs_dir, item)
        if os.path.isdir(item_path):
            # Check if results directory exists
            results_dir = join(item_path, 'results')
            if os.path.exists(results_dir):
                # Look for demo_*.pkl files
                pkl_file = join(results_dir, f"demo_{item}.pkl")
                if os.path.exists(pkl_file):
                    sequences.append(item)
    
    return sorted(sequences)


def process_sequence(sequence_name, start_frame, end_frame, batch_size, device, smplx_model, renderer, faces):
    """Process a single sequence and generate output video."""
    print(f"\n{'='*60}")
    print(f"Processing sequence: {sequence_name}")
    print(f"{'='*60}")
    
    # Determine video name (sequence_name might be without .mp4 extension)
    if sequence_name.endswith('.mp4'):
        video_name = sequence_name
        sequence_name_clean = sequence_name.replace(".mp4", "")
    else:
        video_name = f"{sequence_name}.mp4"
        sequence_name_clean = sequence_name
    
    # Load SMPL coefficients
    smplx_coeffs_path = join(SMPL_COEFFS_DIR, sequence_name_clean, 'results', f"demo_{sequence_name_clean}.pkl")
    
    if not os.path.exists(smplx_coeffs_path):
        print(f"Warning: SMPL coefficients file not found: {smplx_coeffs_path}")
        return False
    
    try:
        data = load(smplx_coeffs_path)
    except Exception as e:
        print(f"Error loading SMPL coefficients: {e}")
        return False
    
    # Open video file for reading frames
    video_path = join(VIDEO_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return False
    
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

    # Process data
    pred_images = []  # Store rendered frames for video output
    global_orient_list = []
    body_pose_list = []
    betas_list = []
    camera_t_list = []
    
    for data_idx, data_name in enumerate(tqdm(data, desc=f"Loading data for {sequence_name}")):
        if data_idx >= end_frame or data_idx < start_frame:
            continue

        data_frame = data[data_name]
        data_smpl = data_frame['smpl']
        if len(data_smpl) != 0:
            global_orient = data_smpl[0]['global_orient']
            body_pose = data_smpl[0]['body_pose']
            betas = data_smpl[0]['betas']
            global_orient = matrix_to_axis_angle(torch.FloatTensor(global_orient).reshape(1, 3, 3)).to(device)
            body_pose = matrix_to_axis_angle(torch.FloatTensor(body_pose).reshape(23, 3, 3)).reshape(1, 23 * 3).to(device)
            tar_beta = torch.FloatTensor(betas).reshape(1, 10).to(device)
            camera_t = torch.FloatTensor(data_frame['camera'][0]).reshape(1, 3).to(device)
        else:
            global_orient = torch.zeros((1, 3), device=device)
            body_pose = torch.zeros((1, 23 * 3), device=device)
            tar_beta = torch.zeros((1, 10), device=device)
            camera_t = torch.zeros((1, 3), device=device)
        global_orient_list.append(global_orient)
        body_pose_list.append(body_pose)
        betas_list.append(tar_beta.reshape(1, 10))
        camera_t_list.append(camera_t)

    if len(global_orient_list) == 0:
        print(f"Warning: No valid frames found for sequence {sequence_name}")
        cap.release()
        return False

    global_orient_list = torch.cat(global_orient_list, dim=0)
    body_pose_list = torch.cat(body_pose_list, dim=0)
    betas_list = torch.cat(betas_list, dim=0)
    camera_t_list = torch.cat(camera_t_list, dim=0)
    
    # Adjust frame range based on actual data length
    actual_end_frame = min(end_frame, len(global_orient_list))
    if start_frame >= actual_end_frame:
        print(f"Warning: start_frame ({start_frame}) >= actual_end_frame ({actual_end_frame})")
        cap.release()
        return False

    # ---------------------- Main Processing Loop ----------------------
    img_size = 256
    focal_length = 5000.
    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    
    for batch_start in tqdm(range(start_frame, actual_end_frame, batch_size), desc=f"Rendering {sequence_name}"):
        # Determine actual batch size (may be smaller at the end)
        actual_batch_size = min(batch_size, actual_end_frame - batch_start)
        batch_end = batch_start + actual_batch_size

        global_orient = global_orient_list[batch_start:batch_end]
        body_pose = body_pose_list[batch_start:batch_end]
        betas = betas_list[batch_start:batch_end]

        ## The forward function of smplx model is good.
        output_tar = smplx_model(
            betas=torch.zeros((actual_batch_size, 300), device=device), 
            transl=torch.zeros((actual_batch_size, 3), device=device), 
            expression=torch.zeros((actual_batch_size, 100), device=device),
            jaw_pose=torch.zeros((actual_batch_size, 3), device=device), 
            global_orient=global_orient[:, :3], 
            body_pose=body_pose[:, :63], 
            left_hand_pose=torch.zeros((actual_batch_size, 15*3), device=device), 
            right_hand_pose=torch.zeros((actual_batch_size, 15*3), device=device), 
            return_verts=True,
            return_joints=True,
            leye_pose=torch.zeros((actual_batch_size, 3), device=device), 
            reye_pose=torch.zeros((actual_batch_size, 3), device=device),
        )

        verts = output_tar.vertices.detach()

        # Read the original frame from the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, batch_start)
        ret, image_original = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {batch_start}")
            continue
            
        # Resize rendered image to match original frame height
        original_height, original_width = image_original.shape[0], image_original.shape[1]
        scale = 256 / original_height
        image_original = cv2.resize(image_original, (int(original_width * scale) + 1, int(original_height * scale)))
        image_original = image_original[:, :, ::-1] / 255.0
        
        original_img_size = [image_original.shape[1], image_original.shape[0]]
        scaled_focal_length = focal_length / img_size * max(original_img_size)
        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        regression_img = renderer.render_rgba_multiple([verts[0].detach().cpu().numpy()],
                            [camera_t_list[batch_start].detach().cpu().numpy()],
                            render_res=torch.tensor(original_img_size),
                            **misc_args
                            )
        # Overlay image
        input_img = image_original.astype(np.float32)
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)  # Add alpha channel
        regression_img_overlay = input_img[:,:,:3] * (1-regression_img[:,:,3:]) + regression_img[:,:,:3] * regression_img[:,:,3:]

        pred_images.append(torch.FloatTensor(regression_img_overlay))

    cap.release()

    # ---------------------- Save Output Video ----------------------
    if len(pred_images) > 0:
        # Stack tensors and permute to [N, 3, H, W] for video writing
        pred_images_tensor = torch.stack(pred_images)  # [N, H, W, 3]
        pred_images_tensor = pred_images_tensor.permute(0, 3, 1, 2)  # [N, 3, H, W]

        # Prepare output directory and file path
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dump_path = join(OUTPUT_DIR, f"{sequence_name_clean}_{start_frame}_{actual_end_frame}.mp4")

        print(f"Saving video to: {dump_path}")
        # Extract audio segment corresponding to the selected frames
        audio_clip = audio[int(start_frame/25.0*16000):int(actual_end_frame/25.0*16000)]
        write_video(pred_images_tensor*255.0, dump_path, 25, audio_clip, 16000, "aac")
        print(f"✓ Successfully saved: {dump_path}")
        return True
    else:
        print(f"Error: No frames were processed successfully for {sequence_name}")
        return False


# ---------------------- Main Execution ----------------------
# ---------------------- Path and Variable Setup ----------------------
start_frame = args.start_frame
end_frame = args.end_frame
batch_size = args.batch_size

# ---------------------- SMPLX Model Setup ----------------------
# Initialize FLAME model with batch size and move to device
smplx_model = smplx.create(SMPL_PATH,
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
            ).eval().to(device)

# Convert FLAME faces to tensor
faces = torch.LongTensor(smplx_model.faces.astype(np.int64))

img_size = 256
focal_length = 5000.
# Setup the renderer
renderer = Renderer(focal_length=focal_length, img_res=img_size, faces=faces.cpu().numpy())

# ---------------------- Process Sequences ----------------------
if args.video_name is not None:
    # Process only the specified video
    if args.video_name.endswith('.mp4'):
        sequence_name = args.video_name.replace('.mp4', '')
    else:
        sequence_name = args.video_name
    sequences_to_process = [sequence_name]
    print(f"Processing single sequence: {sequence_name}")
else:
    # Find and process all sequences
    sequences_to_process = find_all_sequences(SMPL_COEFFS_DIR)
    print(f"Found {len(sequences_to_process)} sequences to process")
    if len(sequences_to_process) == 0:
        print("No sequences found. Exiting.")
        exit(1)

# Process each sequence
success_count = 0
failed_sequences = []
for seq_name in tqdm(sequences_to_process, desc="Processing sequences"):
    try:
        success = process_sequence(seq_name, start_frame, end_frame, batch_size, device, smplx_model, renderer, faces)
        if success:
            success_count += 1
        else:
            failed_sequences.append(seq_name)
    except Exception as e:
        print(f"\n✗ Error processing {seq_name}: {e}")
        import traceback
        traceback.print_exc()
        failed_sequences.append(seq_name)

# Print summary
print(f"\n{'='*60}")
print(f"Processing complete!")
print(f"Successfully processed: {success_count}/{len(sequences_to_process)}")
if failed_sequences:
    print(f"Failed sequences: {len(failed_sequences)}")
    for seq in failed_sequences:
        print(f"  - {seq}")
print(f"{'='*60}")

