#!/usr/bin/env python3

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
from multimodal_tokenizers.utils.utils_videos import write_video
import cv2
matplotlib.use('Agg')
from tqdm import tqdm
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle, axis_angle_to_6d, axis_angle_to_6d_np
from transformers import AutoTokenizer, WhisperFeatureExtractor, AutoModel
from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion, VQVAEConvZeroDSUS1_PaperVersion
import uuid
from transformers.generation.streamers import BaseStreamer
import sys
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")
from audio_process import AudioStreamProcessor
from flow_inference import AudioDecoder
import logging
import re

def save_obj(verts, faces, path):
    """Save mesh vertices and faces to an OBJ file."""
    mesh = trimesh.Trimesh(vertices=verts[0].numpy(), faces=faces)
    mesh.export(path)

import soundfile as sf



def load_speaking_track_segments(transcript_dir, video_id):
    """
    Load all segments from speaking track transcript file.
    
    Args:
        transcript_dir: Directory containing transcript files
        video_id: Video ID
        
    Returns:
        List of segments with text and timestamps
    """
    speaking_track_path = os.path.join(transcript_dir, video_id, f"{video_id}_speaking_track.txt")
    
    if not os.path.exists(speaking_track_path):
        logging.warning(f"No speaking track transcript found for {video_id}")
        return []
    
    segments = []
    
    try:
        with open(speaking_track_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse segments with word-level timestamps
        in_segments_section = False
        current_segment = None
        
        for line in content.split('\n'):
            # Start of segments section
            if line.startswith("Segments with word-level timestamps:"):
                in_segments_section = True
                continue
            
            if not in_segments_section:
                continue
            
            # New segment
            if line.startswith("Segment ") and ":" in line:
                # Save previous segment if exists
                if current_segment and current_segment.get('words'):
                    segments.append(current_segment)
                
                current_segment = {
                    'text': '',
                    'words': [],
                    'start_time': None,
                    'end_time': None
                }
                continue
            
            # Timestamp line
            if current_segment is not None and line.startswith("Timestamp:"):
                match = re.match(r'Timestamp:\s*([\d.]+)s\s*-\s*([\d.]+)s', line)
                if match:
                    current_segment['start_time'] = float(match.group(1))
                    current_segment['end_time'] = float(match.group(2))
                continue
            
            # Text line
            if current_segment is not None and line.startswith("Text:"):
                text = line[5:].strip()
                # Remove "A: " prefix if present
                if text.startswith("A: "):
                    text = text[3:]
                current_segment['text'] = text
                continue
            
            # Words section
            if current_segment is not None and line.startswith("Words:"):
                continue
            
            # Word with timestamp
            if current_segment is not None and ':' in line and 's' in line and '-' in line:
                # Parse word timestamp line
                match = re.match(r'^\s*(.+?):\s*([\d.]+)s\s*-\s*([\d.]+)s', line)
                if match:
                    word = match.group(1).strip()
                    start = float(match.group(2))
                    end = float(match.group(3))
                    current_segment['words'].append({
                        'word': word,
                        'start': start,
                        'end': end
                    })
        
        # Don't forget the last segment
        if current_segment and current_segment.get('words'):
            segments.append(current_segment)
    
    except Exception as e:
        logging.error(f"Error loading speaking track transcript for {video_id}: {e}")
        return []
    
    return segments


def merge_close_segments(segments, gap_threshold=2.0):
    """
    Merge segments that have a time gap less than the threshold.
    
    Args:
        segments: List of segments with text and timestamps
        gap_threshold: Maximum gap in seconds between segments to merge (default: 2.0)
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x['start_time'])
    
    merged_segments = []
    current_merged = None
    
    for segment in sorted_segments:
        if current_merged is None:
            # First segment
            current_merged = segment.copy()
        else:
            # Check gap between current merged segment and this segment
            gap = segment['start_time'] - current_merged['end_time']
            
            if gap < gap_threshold:
                # Merge segments
                current_merged['end_time'] = segment['end_time']
                current_merged['text'] += ' ' + segment['text']
                current_merged['words'].extend(segment['words'])
            else:
                # Gap is too large, save current merged segment and start new one
                merged_segments.append(current_merged)
                current_merged = segment.copy()
    
    # Don't forget the last merged segment
    if current_merged:
        merged_segments.append(current_merged)
    
    return merged_segments





# ---------------------- Configurable Dataset and Model Paths ----------------------
FLAME_COEFFS_DIR = "/path/to/YouTube_Talking/FLAME_coeffs_25"
FACE_CODE_DIR = "/path/to/YouTube_Talking/TOKENS_DS4_512_512_DS1_wo_meshloss"
VIDEO_DIR = "/path/to/YouTube_Talking/video_20241226"
TRANSCRIPT_DIR = "/path/to/YouTube_Talking/transcript"
AUDIO_CODE_DIR = "/path/to/YouTube_Talking/audios_token_glm"
MODEL_DIR = "./model_files/FLAME2020/"
OUTPUT_DIR = "./output_videos"

# ---------------------- Argument Parsing ----------------------
parser = argparse.ArgumentParser(description='Process facial meshes for visualization')
parser.add_argument('--video_name', type=str, default="203011115.mp4", 
                    help='Video name within the sequence')
parser.add_argument('--start_time', type=int, default=20, 
                    help='Starting time index')
parser.add_argument('--end_time', type=int, default=60, 
                    help='Ending time index')
parser.add_argument('--batch_size', type=int, default=100, 
                    help='Batch size for processing')
parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA even if available')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='GPU ID to use if CUDA is available (default: 0)')
parser.add_argument('--show_speaker', default=True,
                    help='Show speaker in the video')
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
start_time = args.start_time
end_time = args.end_time
batch_size = args.batch_size
show_speaker = args.show_speaker

# Open video file for reading frames
video_path = join(VIDEO_DIR, video_name)
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
video_start_frame = int(start_time * video_fps)
video_end_frame = int(end_time * video_fps)
transcript_segments = load_speaking_track_segments(TRANSCRIPT_DIR, video_name.split('.')[0])

# Merge segments with gaps less than 2 seconds
transcript_segments = merge_close_segments(transcript_segments, gap_threshold=2.0)
print(f"Total segments after merging: {len(transcript_segments)}")

for segment in transcript_segments:

    # segment_start_frame = int(segment['start_time'] * video_fps)
    # segment_end_frame = int(segment['end_time'] * video_fps)
    segment_start_frame = int(segment['start_time'] * 25)
    segment_end_frame = int(segment['end_time'] * 25)
    segment_start_time = segment['start_time']
    segment_end_time = segment['end_time']
    
    # if segment_start_time >= video_start_frame and segment_end_time <= video_end_frame:
    #     break

    # ---------------------- Load Audio Tokens ----------------------
    audio_start_frame = int(segment_start_time * 12.5)
    audio_end_frame = int(segment_end_time * 12.5)

    # Load FLAME coefficients (expression, shape, pose) for the video
    audio_code_path = join(AUDIO_CODE_DIR, video_name.replace(".mp4", ".npy"))
    audio_tokens = np.load(audio_code_path)[audio_start_frame:audio_end_frame]

    this_uuid = str(uuid.uuid4())
    audio_processor = AudioStreamProcessor()
    # Initialize variables before processing files
    prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
    # Flow & Hift
    flow_config = os.path.join("./glm-4-voice-decoder", "config.yaml")
    flow_checkpoint = os.path.join("./glm-4-voice-decoder", 'flow.pt')
    hift_checkpoint = os.path.join("./glm-4-voice-decoder", 'hift.pt')
    audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                    hift_ckpt_path=hift_checkpoint,
                                    device=device)

    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
    tts_speech, tts_mel = audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                prompt_token=flow_prompt_speech_token.to(device),
                                                prompt_feat=prompt_speech_feat.to(device),
                                                finalize=True)
    final_speech = tts_speech[0].cpu()

    # ---------------------- Load Face Tokens ----------------------
    face_token_start_frame = int(segment_start_time * 25)
    face_token_end_frame = int(segment_end_time * 25)

    # Load FLAME coefficients (expression, shape, pose) for the video
    face_code_path = join(FACE_CODE_DIR, video_name.replace(".mp4", ".npy"))
    face_tokens = np.load(face_code_path)[:, face_token_start_frame:face_token_end_frame]

    checkpoint = torch.load('/path/to/conversational_agent/experiments/conversational_agent/VQVAE_Mixed_Face_Only_512_DS1_Dim_512_wo_meshloss/checkpoints/epoch=79.ckpt', map_location="cpu", weights_only=False)
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    # Create new state dict with modified keys
    state_dict_face = {}
    for key, value in state_dict.items():
        if 'vae_face' in key:
            new_key = key.replace('vae_face.', '')
            state_dict_face[new_key] = value

    vae_face = VQVAEConvZeroDSUS1_PaperVersion(
        vae_layer=3,
        code_num=512,
        codebook_size=512,
        vae_quantizer_lambda=1,
        vae_test_dim=112
    )

    # Save only the modified state_dict
    vae_face.load_state_dict(state_dict_face, strict=True)
    vae_face.eval()
    vae_face.to(device)
    face_tokens = torch.tensor(face_tokens, device=device)
    rec_face = vae_face.decode(face_tokens.int())
    rec_head_pose = 0 * rotation_6d_to_axis_angle(rec_face[0, :, :6])
    rec_jaw_pose = rotation_6d_to_axis_angle(rec_face[0, :, 6:12])
    rec_exp = rec_face[0, :, 12:]
    n = rec_head_pose.shape[0]
    model_path = "./model_files/FLAME2020/"
    batch_size_visualize = 100
    # Initialize FLAME model with the adjusted batch size and move to GPU
    flame_model = FLAME(model_path, num_expression_coeffs=100, ext='pkl', batch_size=batch_size_visualize).to(device)
    faces = torch.tensor(flame_model.faces.astype(np.int32), dtype=torch.int64)
    mesh_renderer = RenderMesh(image_size=256, faces=faces, scale=1.0)

    pred_images = []
    for i in range(0, n, batch_size_visualize):
        if batch_size_visualize > n - i:
            batch_size_visualize = n - i
            flame_model = FLAME(model_path, num_expression_coeffs=100, ext='pkl', batch_size=batch_size_visualize).to(device)
        actual_visualize_batch_size = batch_size_visualize
        # Run FLAME model for both reconstructed and ground truth
        with torch.no_grad():
            flame_out = flame_model(
                global_orient=rec_head_pose[i:i+actual_visualize_batch_size, :],
                expression=rec_exp[i:i+actual_visualize_batch_size, :],
                jaw_pose=rec_jaw_pose[i:i+actual_visualize_batch_size, :],
                shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
            )
        verts = flame_out['vertices'].detach()
        for v in tqdm(verts):
            rgb = mesh_renderer(v[None])[0]
            pred_images.append(rgb.cpu()[0] / 255.0)
    pred_images_tensor = torch.stack(pred_images)


    # ---------------------- Load FLAME Coefficients (Ground Truth) ----------------------
    face_coeff_start_frame = segment_start_frame
    face_coeff_end_frame = segment_end_frame
    # Load FLAME coefficients (expression, shape, pose) for the video
    coeffs_path = join(FLAME_COEFFS_DIR, video_name.replace(".mp4", ".npz"))
    data = np.load(coeffs_path)
    exp = data['exp'][face_coeff_start_frame:face_coeff_end_frame]
    shape = data['shape'][face_coeff_start_frame:face_coeff_end_frame]
    pose = data['pose'][face_coeff_start_frame:face_coeff_end_frame]
    # Convert numpy arrays to torch tensors and move to device
    exp = torch.FloatTensor(exp).to(device)
    shape = torch.FloatTensor(shape).to(device)
    pose = torch.FloatTensor(pose).to(device)
    head_pose = 0 * pose[:, :3]
    jaw_pose = pose[:, 3:]
    n = exp.shape[0]
    batch_size_visualize = 100

    flame_model_gt = FLAME(model_path, num_expression_coeffs=50, ext='pkl', batch_size=batch_size_visualize).to(device)
    pred_images_gt = []
    for i in range(0, n, batch_size_visualize):
        if batch_size_visualize > n - i:
            batch_size_visualize = n - i
            flame_model_gt = FLAME(model_path, num_expression_coeffs=50, ext='pkl', batch_size=batch_size_visualize).to(device)
        actual_visualize_batch_size = batch_size_visualize
        # Run FLAME model for both reconstructed and ground truth
        with torch.no_grad():
            flame_out = flame_model_gt(
                global_orient=head_pose[i:i+actual_visualize_batch_size, :],
                expression=exp[i:i+actual_visualize_batch_size, :],
                jaw_pose=jaw_pose[i:i+actual_visualize_batch_size, :],
                shape=torch.zeros(actual_visualize_batch_size, 100).to(device),
            )
        verts = flame_out['vertices'].detach()
        for v in tqdm(verts):
            rgb = mesh_renderer(v[None])[0]
            pred_images_gt.append(rgb.cpu()[0] / 255.0)
    pred_images_tensor_gt = torch.stack(pred_images_gt)

    frame_num_gt, _,  gt_height, gt_width = pred_images_tensor_gt.shape
    frame_num, _,  height, width = pred_images_tensor.shape

    if frame_num_gt != frame_num:
        frame_num_gt = min(frame_num_gt, frame_num)

    combined_width = gt_width + width
    # Concatenate previously generated images horizontally
    combined_image = np.zeros((frame_num_gt, 3, gt_height, combined_width))
    combined_image[:, :, :, :gt_width] = pred_images_tensor_gt[:frame_num_gt]
    combined_image[:, :, :, gt_width:] = pred_images_tensor[:frame_num_gt]

    # Convert to float tensor and append to results
    combined_image = np.asarray(combined_image).astype(np.float32)
    combined_image = torch.FloatTensor(combined_image)

    # Use sequence and video name in the output filename
    # Prepare output directory and file path
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dump_path = join(OUTPUT_DIR, f"{video_name.split('/')[0]}_{segment_start_time}_{segment_end_time}_fromcode.mp4")

    print(f"Saving video to: {dump_path}")
    audio_clip = final_speech
    audio_clip = audio_clip[:int(combined_image.shape[0]/25.0*22050)]
    write_video(combined_image*255.0, dump_path, 25, audio_clip, 22050, "aac")
    print("Video saved successfully!")


