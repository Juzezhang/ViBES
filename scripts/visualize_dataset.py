#!/usr/bin/env python3
"""
Visualize samples from the preprocessed HuggingFace dataset.

Decodes text, audio, and motion tokens from the tokenized dataset and
produces rendered SMPLX video with synchronized audio.

Token ID scheme (after subtracting TOKEN_OFFSET=168736 from motion tokens):
  - 0-511:   face VQ codes (512 entries, not present in body_only datasets)
  - 512-767:  upper body VQ codes (256 entries, subtract 512 for codebook index)
  - 768-1023: lower body VQ codes (256 entries, subtract 768 for codebook index)
  - 1024-1279: hand VQ codes (256 entries, subtract 1024 for codebook index)
  - 1280: segment separator token

Motion tokens are interleaved as: [sep], U, L, H, U, L, H, ..., [sep], U, L, H, ...

Usage:
    python inference/visualize_dataset.py \
        --dataset_path /path/to/seamless_interaction_processed/generated_dataset_hf_0201/combined_pair \
        --sample_idx 0 \
        --output_dir viz_output/
"""
import sys
import os
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

speech_related_path = os.path.join(ROOT_DIR, "speech_related")
cosyvoice_path = os.path.join(ROOT_DIR, "speech_related", "cosyvoice")
matcha_path = os.path.join(ROOT_DIR, "speech_related", "Matcha-TTS")
for p in [speech_related_path, cosyvoice_path, matcha_path]:
    if os.path.exists(p):
        if p in sys.path:
            sys.path.remove(p)
        # Insert after ROOT_DIR to avoid cosyvoice/utils shadowing SAMPA utils
        sys.path.insert(1, p)

import uuid
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer

from multimodal_tokenizers.utils.rotation_conversions import (
    rotation_6d_to_axis_angle,
    axis_angle_to_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)
from multimodal_tokenizers.utils.utils_videos import write_video
from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
)
from utils.tensor_utils import inverse_selection_tensor
from utils.model_loader import load_vae_models, load_smplx_model

# ============================================================================
# Constants
# ============================================================================
TOKEN_OFFSET = 168736

# Motion token ranges (after subtracting TOKEN_OFFSET)
FACE_CODE_START = 0
FACE_CODE_END = 512
UPPER_CODE_START = 512
UPPER_CODE_END = 768
LOWER_CODE_START = 768
LOWER_CODE_END = 1024
HAND_CODE_START = 1024
HAND_CODE_END = 1280
SEPARATOR_TOKEN = 1280

MOTION_FPS = 25
AUDIO_OUTPUT_SAMPLE_RATE = 22050

# Feature dimensions
LOWER_BODY_FEATURE_DIM = 54
GLOBAL_TRANSLATION_START = 54
GLOBAL_TRANSLATION_END = 57
GLOBAL_FEATURE_PADDING = 7

# SMPLX constants
SMPLX_NUM_BETAS = 300
SMPLX_NUM_EXPRESSIONS = 100
SMPLX_NUM_JOINTS = 55
SMPLX_DIMS_PER_JOINT = 3

# Pose indices
POSE_JAW = (66, 69)
POSE_GLOBAL_ORIENT = (0, 3)
POSE_BODY_START = 3
POSE_BODY_END = 21 * SMPLX_DIMS_PER_JOINT + POSE_BODY_START
POSE_LEFT_HAND = (25 * SMPLX_DIMS_PER_JOINT, 40 * SMPLX_DIMS_PER_JOINT)
POSE_RIGHT_HAND = (40 * SMPLX_DIMS_PER_JOINT, 55 * SMPLX_DIMS_PER_JOINT)
POSE_LEFT_EYE = (69, 72)
POSE_RIGHT_EYE = (72, 75)

RENDER_WIDTH = 1280
RENDER_HEIGHT = 720

# Default model paths (can be overridden via CLI)
DEFAULT_TOKENIZER_PATH = os.path.join(ROOT_DIR, "vibes")
DEFAULT_VAE_BODY_CKPT = "./model_files/pretrained_cpt/body/lom_vq.ckpt"
DEFAULT_VAE_FACE_CKPT = "./model_files/pretrained_cpt/face/face.ckpt"
DEFAULT_VAE_GLOBAL_CKPT = '/path/to/experiments/multimodal_tokenizer/VAE_Global_from_Lower54/checkpoints/last.ckpt' #"./experiments/multimodal_tokenizer/VAE_Global_from_Lower54/checkpoints/last.ckpt"
DEFAULT_SMPLX_DIR = "./model_files/smplx_models"
DEFAULT_AUDIO_DECODER_DIR = "./speech_related/glm-4-voice-decoder"


# ============================================================================
# Token parsing
# ============================================================================

def parse_sample(sample, tokenizer):
    """Parse a dataset sample into text, audio, and motion token arrays."""
    input_ids = np.array(sample['input_ids'])
    mask_0 = np.array(sample['modality_masks_0'])
    mask_1 = np.array(sample['modality_masks_1'])
    mask_2 = np.array(sample['modality_masks_2'])

    text_ids = input_ids[mask_0].tolist()
    audio_ids = input_ids[mask_1].tolist()
    motion_vocab_ids = input_ids[mask_2].tolist()

    # Decode motion vocab IDs back to strings like "<|upper_XX|>", "<|lower_XX|>", etc.
    # and parse out the codebook index
    import re
    face_pattern = re.compile(r"<\|face_(\d+)\|>")
    upper_pattern = re.compile(r"<\|upper_(\d+)\|>")
    lower_pattern = re.compile(r"<\|lower_(\d+)\|>")
    hand_pattern = re.compile(r"<\|hand_(\d+)\|>")

    face_tokens = []
    upper_tokens = []
    lower_tokens = []
    hand_tokens = []
    for tid in motion_vocab_ids:
        decoded_str = tokenizer.decode([tid])
        m = face_pattern.search(decoded_str)
        if m:
            face_tokens.append(int(m.group(1)))
            continue
        m = upper_pattern.search(decoded_str)
        if m:
            upper_tokens.append(int(m.group(1)))
            continue
        m = lower_pattern.search(decoded_str)
        if m:
            lower_tokens.append(int(m.group(1)))
            continue
        m = hand_pattern.search(decoded_str)
        if m:
            hand_tokens.append(int(m.group(1)))
            continue
        # skip begin_of_motion, separators, etc.

    return {
        'text_ids': text_ids,
        'audio_ids': audio_ids,
        'face_tokens': face_tokens,
        'upper_tokens': upper_tokens,
        'lower_tokens': lower_tokens,
        'hand_tokens': hand_tokens,
        'metadata': {
            'conv_id': sample.get('conv_id', ''),
            'sequence_name': sample.get('sequence_name', ''),
            'num_turns': sample.get('num_turns', 0),
        },
    }


# ============================================================================
# Decode text
# ============================================================================

def decode_text(text_ids, tokenizer):
    """Decode text token IDs to string."""
    return tokenizer.decode(text_ids, skip_special_tokens=False)


def dump_token_structure(sample, tokenizer, output_path):
    """Write a detailed token-by-token structure visualization to a file.

    For each token position, shows the modality (TEXT/AUDIO/MOTION), raw token ID,
    decoded text or codebook index, and which turn/segment it belongs to.
    """
    input_ids = np.array(sample['input_ids'])
    mask_0 = np.array(sample['modality_masks_0'])
    mask_1 = np.array(sample['modality_masks_1'])
    mask_2 = np.array(sample['modality_masks_2'])
    labels = np.array(sample['labels'])
    pos_enc = np.array(sample['position_encoding_indices']) if 'position_encoding_indices' in sample else None

    lines = []
    lines.append(f"Token Structure Visualization")
    lines.append(f"conv_id: {sample.get('conv_id', '')}")
    lines.append(f"sequence_name: {sample.get('sequence_name', '')}")
    lines.append(f"num_turns: {sample.get('num_turns', 0)}")
    lines.append(f"total tokens: {len(input_ids)}")
    lines.append(f"  text (mask_0):   {mask_0.sum()}")
    lines.append(f"  audio (mask_1):  {mask_1.sum()}")
    lines.append(f"  motion (mask_2): {mask_2.sum()}")
    lines.append(f"  labels != -100:  {(labels != -100).sum()} (training targets)")
    lines.append("")
    if pos_enc is not None:
        lines.append(f"{'Pos':>5} {'PosEnc':>10} {'Modality':>8} {'RawID':>8} {'Info':<60} {'Label':>8}")
        lines.append("-" * 107)
    else:
        lines.append(f"{'Pos':>5} {'Modality':>8} {'RawID':>8} {'Info':<60} {'Label':>8}")
        lines.append("-" * 95)

    # Track segments for motion
    motion_seg_idx = 0
    motion_step_in_seg = 0

    for i in range(len(input_ids)):
        tid = int(input_ids[i])
        lab = int(labels[i])
        lab_str = str(lab) if lab != -100 else "-100"

        if mask_2[i]:
            # Motion token
            mot_id = tid - TOKEN_OFFSET
            if mot_id == SEPARATOR_TOKEN:
                info = f"[MOTION SEP] (turn boundary)"
                motion_seg_idx += 1
                motion_step_in_seg = 0
            elif FACE_CODE_START <= mot_id < FACE_CODE_END:
                info = f"[MOTION FACE]  code={mot_id - FACE_CODE_START:3d}  step={motion_step_in_seg}"
                motion_step_in_seg += 1
            elif UPPER_CODE_START <= mot_id < UPPER_CODE_END:
                info = f"[MOTION UPPER] code={mot_id - UPPER_CODE_START:3d}  step={motion_step_in_seg}"
            elif LOWER_CODE_START <= mot_id < LOWER_CODE_END:
                info = f"[MOTION LOWER] code={mot_id - LOWER_CODE_START:3d}  step={motion_step_in_seg}"
            elif HAND_CODE_START <= mot_id < HAND_CODE_END:
                info = f"[MOTION HAND]  code={mot_id - HAND_CODE_START:3d}  step={motion_step_in_seg}"
                motion_step_in_seg += 1  # increment after hand (U,L,H = one timestep)
            else:
                info = f"[MOTION ???]   offset_id={mot_id}"
            modality = "MOTION"
        elif mask_1[i]:
            # Audio token
            if tid >= 152353:
                codebook_idx = tid - 152353
                info = f"[AUDIO] codebook_idx={codebook_idx}"
            elif tid == 151343:
                info = f"[AUDIO] <begin_of_audio>"
            elif tid == 151344:
                info = f"[AUDIO] <end_of_audio>"
            else:
                decoded = tokenizer.decode([tid])
                info = f"[AUDIO SPECIAL] decoded={repr(decoded)}"
            modality = "AUDIO"
        elif mask_0[i]:
            # Text token
            decoded = tokenizer.decode([tid])
            info = f"[TEXT] {repr(decoded)}"
            modality = "TEXT"
        else:
            # No mask set (shouldn't happen normally)
            decoded = tokenizer.decode([tid])
            info = f"[NONE] {repr(decoded)}"
            modality = "NONE"

        if pos_enc is not None:
            pe = float(pos_enc[i])
            lines.append(f"{i:5d} {pe:10.3f} {modality:>8} {tid:8d} {info:<60} {lab_str:>8}")
        else:
            lines.append(f"{i:5d} {modality:>8} {tid:8d} {info:<60} {lab_str:>8}")

    # Append summary statistics at the end
    lines.append("")
    lines.append("=" * 95)
    lines.append("SUMMARY")
    lines.append("=" * 95)

    # Per-turn breakdown
    motion_ids_raw = input_ids[mask_2] - TOKEN_OFFSET
    sep_positions = np.where(motion_ids_raw == SEPARATOR_TOKEN)[0]
    lines.append(f"\nPer-turn motion token counts (28 turns × 13 timesteps × 3 body parts = 1092 motion tokens):")
    for i in range(len(sep_positions)):
        start = sep_positions[i] + 1
        end = sep_positions[i + 1] if i + 1 < len(sep_positions) else len(motion_ids_raw)
        seg = motion_ids_raw[start:end]
        non_sep = seg[seg < SEPARATOR_TOKEN]
        n_steps = len(non_sep) // 3
        lines.append(f"  Turn {i:2d}: {len(non_sep):3d} tokens ({n_steps} timesteps, "
                      f"{n_steps * 4} decoded frames, {n_steps * 4 / MOTION_FPS:.1f}s)")

    # Audio breakdown
    audio_ids = input_ids[mask_1]
    n_codebook = sum(1 for t in audio_ids if 152353 <= t <= 168735)
    n_special = len(audio_ids) - n_codebook
    lines.append(f"\nAudio: {n_codebook} codebook tokens + {n_special} special tokens")

    # VQ-VAE temporal info
    n_motion_steps = int(np.sum((motion_ids_raw >= UPPER_CODE_START) & (motion_ids_raw < UPPER_CODE_END)))
    lines.append(f"\nMotion: {n_motion_steps} VQ tokens per body part")
    lines.append(f"  VQ-VAE decoder 4x upsample: {n_motion_steps} → {n_motion_steps * 4} frames")
    lines.append(f"  Duration at {MOTION_FPS} FPS: {n_motion_steps * 4 / MOTION_FPS:.1f}s")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Token structure saved to {output_path} ({len(lines)} lines)")


# ============================================================================
# Decode audio
# ============================================================================

def decode_audio(audio_ids, device, audio_decoder_dir, tokenizer):
    """Decode audio token IDs to waveform using GLM-4-Voice AudioDecoder.

    Audio tokens in the dataset are vocab IDs produced by tokenizing strings like
    "<|audio_7233|>".  We decode them back to strings, parse the codebook index,
    and feed that to the AudioDecoder.
    """
    import re
    from speech_related.flow_inference import AudioDecoder

    config_path = os.path.join(audio_decoder_dir, "config.yaml")
    flow_path = os.path.join(audio_decoder_dir, "flow.pt")
    hift_path = os.path.join(audio_decoder_dir, "hift.pt")

    for p in [config_path, flow_path, hift_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Audio decoder file not found: {p}")

    audio_decoder = AudioDecoder(
        config_path=config_path,
        flow_ckpt_path=flow_path,
        hift_ckpt_path=hift_path,
        device=device,
    )

    # Decode each vocab ID back to string and parse codebook index from "<|audio_XXXX|>"
    audio_pattern = re.compile(r"<\|audio_(\d+)\|>")
    codebook_ids = []
    for tid in audio_ids:
        decoded_str = tokenizer.decode([tid])
        m = audio_pattern.search(decoded_str)
        if m:
            codebook_ids.append(int(m.group(1)))

    if not codebook_ids:
        print("  Warning: no audio codebook tokens found")
        return None

    print(f"  Audio codebook tokens: {len(codebook_ids)}, index range: {min(codebook_ids)}-{max(codebook_ids)}")
    tts_token = torch.tensor(codebook_ids, device=device).unsqueeze(0)
    prompt_feat = torch.zeros(1, 0, 80).to(device)
    prompt_token = torch.zeros(1, 0, dtype=torch.int64).to(device)

    tts_speech, _ = audio_decoder.token2wav(
        tts_token,
        uuid=str(uuid.uuid4()),
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        finalize=True,
    )
    return tts_speech[0].cpu()


# ============================================================================
# Decode motion
# ============================================================================

def decode_motion(upper_tokens, lower_tokens, hand_tokens,
                  vae_upper, vae_lower, vae_hand, vae_global, device):
    """Decode body part VQ tokens to continuous motion features.

    If tokens are empty (length 0), pads with token 0 (neutral T-pose).
    """
    # Get sequence length from any available body part
    T_upper = len(upper_tokens)
    T_lower = len(lower_tokens)
    T_hand = len(hand_tokens)

    # Use any available token length as reference
    n = max(T_upper, T_lower, T_hand)
    if n == 0:
        n = 1  # Minimum 1 frame

    # Pad missing tokens with 0 (neutral token)
    if T_upper == 0:
        upper_tokens = [0] * n
    if T_lower == 0:
        lower_tokens = [0] * n
    if T_hand == 0:
        hand_tokens = [0] * n

    with torch.no_grad():
        upper_t = torch.tensor(upper_tokens, device=device).unsqueeze(0)
        lower_t = torch.tensor(lower_tokens, device=device).unsqueeze(0)
        hand_t = torch.tensor(hand_tokens, device=device).unsqueeze(0)

        rec_upper = vae_upper.decode(upper_t.int())   # (1, T, 78)
        rec_lower = vae_lower.decode(lower_t.int())   # (1, T, 54)
        rec_hand = vae_hand.decode(hand_t.int())     # (1, T, 180)

    # Align lengths
    n = min(rec_upper.shape[1], rec_lower.shape[1], rec_hand.shape[1])
    rec_upper = rec_upper[:, :n, :]
    rec_lower = rec_lower[:, :n, :LOWER_BODY_FEATURE_DIM]
    rec_hand = rec_hand[:, :n, :]

    return rec_upper, rec_lower, rec_hand, n


def reconstruct_body_pose(rec_upper, rec_lower, batch_size, num_frames, device):
    """Reconstruct full SMPLX pose from upper and lower body features."""
    # Upper body: 13 joints * 6D -> axis-angle -> full 165D
    pose_upper = rec_upper.reshape(batch_size, num_frames, 13, 6)
    pose_upper = rotation_6d_to_axis_angle(pose_upper).reshape(
        batch_size * num_frames, 13 * 3)
    pose_upper_full = inverse_selection_tensor(
        pose_upper.to(device), JOINT_MASK_UPPER, batch_size * num_frames)

    # Lower body: 9 joints * 6D -> axis-angle -> full 165D
    pose_legs = rec_lower[:, :, :LOWER_BODY_FEATURE_DIM]
    pose_lower = pose_legs.reshape(batch_size, num_frames, 9, 6)
    pose_lower_mat = rotation_6d_to_matrix(pose_lower)
    lower2global = matrix_to_rotation_6d(pose_lower_mat.clone()).reshape(
        batch_size, num_frames, 9 * 6)
    pose_lower = matrix_to_axis_angle(pose_lower_mat).reshape(
        batch_size * num_frames, 9 * 3)
    pose_lower_full = inverse_selection_tensor(
        pose_lower, JOINT_MASK_LOWER, batch_size * num_frames)

    # Hands: zeros (hand VAE decodes features but not yet used for pose)
    pose_hands = torch.zeros(batch_size * num_frames, 30 * 3, device=device)
    pose_hands_full = inverse_selection_tensor(
        pose_hands, JOINT_MASK_HANDS, batch_size * num_frames)

    # Combine
    rec_pose = pose_upper_full + pose_lower_full + pose_hands_full
    # Zero jaw (no face model in body_only)
    rec_pose[:, POSE_JAW[0]:POSE_JAW[1]] = 0.0

    return rec_pose, lower2global


def compute_global_translation(rec_lower, lower2global, rec_pose, vae_global):
    """Compute world-space root translation from lower body features."""
    to_global = rec_lower.clone()
    if to_global.shape[2] == LOWER_BODY_FEATURE_DIM:
        to_global = F.pad(to_global, (0, GLOBAL_FEATURE_PADDING))
    to_global[:, :, GLOBAL_TRANSLATION_START:GLOBAL_TRANSLATION_END] = 0.0
    to_global[:, :, :LOWER_BODY_FEATURE_DIM] = lower2global

    with torch.no_grad():
        rec_global = vae_global(to_global)

    rec_trans_v_s = rec_global["rec_pose"][:, :, GLOBAL_TRANSLATION_START:GLOBAL_TRANSLATION_END]
    global_orient_aa = rec_pose[:, POSE_GLOBAL_ORIENT[0]:POSE_GLOBAL_ORIENT[1]]

    # Integrate local velocity to world positions
    global_6d = axis_angle_to_6d(global_orient_aa)
    R = rotation_6d_to_matrix(global_6d)
    world_vel = torch.einsum("tij,tj->ti", R, rec_trans_v_s[0])
    pos = torch.zeros_like(world_vel)
    pos[0] = 0.0
    if world_vel.shape[0] > 1:
        pos[1:] = torch.cumsum(world_vel[1:], dim=0)

    return pos.unsqueeze(0)


# ============================================================================
# Render
# ============================================================================

def render_video(rec_pose, rec_trans, smplx_model, n, device,
                 output_path, audio_waveform=None, fixed_camera=True):
    """Render SMPLX mesh to video with optional audio."""
    from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
    from utils.genmo.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
    from utils.genmo.camera import create_camera_sensor

    rec_beta = torch.zeros(n, SMPLX_NUM_BETAS, device=device)
    rec_exps = torch.zeros(n, SMPLX_NUM_EXPRESSIONS, device=device)

    with torch.no_grad():
        vertices_rec = smplx_model(
            betas=rec_beta,
            transl=rec_trans.reshape(n, 3),
            expression=rec_exps,
            jaw_pose=rec_pose[:, POSE_JAW[0]:POSE_JAW[1]],
            global_orient=rec_pose[:, POSE_GLOBAL_ORIENT[0]:POSE_GLOBAL_ORIENT[1]],
            body_pose=rec_pose[:, POSE_BODY_START:POSE_BODY_END],
            left_hand_pose=rec_pose[:, POSE_LEFT_HAND[0]:POSE_LEFT_HAND[1]],
            right_hand_pose=rec_pose[:, POSE_RIGHT_HAND[0]:POSE_RIGHT_HAND[1]],
            leye_pose=rec_pose[:, POSE_LEFT_EYE[0]:POSE_LEFT_EYE[1]],
            reye_pose=rec_pose[:, POSE_RIGHT_EYE[0]:POSE_RIGHT_EYE[1]],
        )

    render_faces = torch.from_numpy(smplx_model.faces.astype(np.int64))
    verts = vertices_rec.vertices.detach()
    joints_raw = vertices_rec.joints.detach()

    # Normalize
    offset = joints_raw[0, 0].clone()
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset
    joints_norm = joints_raw - offset
    T_rot = compute_T_ayfz2ay(joints_norm[[0]], inverse=True)
    verts = apply_T_on_points(verts, T_rot)
    root_points = apply_T_on_points(joints_norm, T_rot)[:, 0, :]

    # Renderer setup
    _, _, K = create_camera_sensor(RENDER_WIDTH, RENDER_HEIGHT, 24)
    renderer = Renderer(
        width=RENDER_WIDTH, height=RENDER_HEIGHT,
        device=device, faces=render_faces, K=K, bin_size=0,
    )
    scale, cx, cz = get_ground_params_from_points(root_points, verts)
    renderer.set_ground(max(scale, 3) * 1.5, cx, cz)

    if fixed_camera:
        from inference.inference_face_body import get_fixed_camera
        cam_R, cam_T, lights = get_fixed_camera(n, beta=2.5, device=device)
    else:
        cam_R, cam_T, lights = get_global_cameras_static(
            verts.cpu(), beta=2.5, vec_rot=45, cam_height_degree=30)

    pred_images = []
    color = torch.tensor([0.69, 0.39, 0.96], device=device)
    for i in tqdm(range(verts.shape[0]), desc="Rendering frames"):
        cams = renderer.create_camera(cam_R[i], cam_T[i])
        img = renderer.render_with_ground(verts[[i]], color[None], cams, lights)
        pred_images.append(img)

    pred_tensor = torch.from_numpy(np.stack(pred_images)).permute(0, 3, 1, 2)

    if audio_waveform is not None:
        audio_clip = audio_waveform[:int(pred_tensor.shape[0] / MOTION_FPS * AUDIO_OUTPUT_SAMPLE_RATE)]
        write_video(pred_tensor, output_path, MOTION_FPS,
                    audio_clip, AUDIO_OUTPUT_SAMPLE_RATE, "aac")
    else:
        write_video(pred_tensor, output_path, MOTION_FPS)

    print(f"  Video saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize samples from preprocessed HuggingFace dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to HuggingFace dataset directory")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Index of sample to visualize")
    parser.add_argument("--output_dir", type=str, default="./viz_output",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--skip_audio", action="store_true",
                        help="Skip audio decoding")
    parser.add_argument("--skip_render", action="store_true",
                        help="Only print text and token stats, no rendering")
    parser.add_argument("--dump_structure", action="store_true",
                        help="Write detailed per-token structure visualization to file")
    parser.add_argument("--fixed_camera", action="store_true", default=True,
                        help="Use fixed camera position")

    # Body mode
    parser.add_argument("--mode", type=str, default="auto",
                        choices=["auto", "compositional", "upper_lower_genmo", "fullbody_genmo"],
                        help="Body motion decode mode. 'auto' detects from token counts.")

    # Model paths
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--vae_body_ckpt", type=str, default=DEFAULT_VAE_BODY_CKPT)
    parser.add_argument("--vae_face_ckpt", type=str, default=DEFAULT_VAE_FACE_CKPT)
    parser.add_argument("--vae_global_ckpt", type=str, default=DEFAULT_VAE_GLOBAL_CKPT)
    parser.add_argument("--smplx_dir", type=str, default=DEFAULT_SMPLX_DIR)
    parser.add_argument("--audio_decoder_dir", type=str, default=DEFAULT_AUDIO_DECODER_DIR)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        args.device = "cpu"
    device = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================
    # Step 1: Load dataset and parse sample
    # ========================================================================
    print(f"=== Loading dataset from {args.dataset_path} ===")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset has {len(dataset)} samples")

    if args.sample_idx >= len(dataset):
        print(f"Error: sample_idx {args.sample_idx} out of range (max {len(dataset)-1})")
        return

    sample = dataset[args.sample_idx]

    # Load tokenizer (needed for parsing tokens and decoding text)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    parsed = parse_sample(sample, tokenizer)
    meta = parsed['metadata']
    prefix = f"sample_{args.sample_idx}"

    print(f"\n=== Sample {args.sample_idx} ===")
    print(f"  conv_id: {meta['conv_id']}")
    print(f"  sequence_name: {meta['sequence_name']}")
    print(f"  num_turns: {meta['num_turns']}")
    print(f"  text tokens: {len(parsed['text_ids'])}")
    print(f"  audio tokens: {len(parsed['audio_ids'])}")
    print(f"  motion tokens: face={len(parsed['face_tokens'])}, upper={len(parsed['upper_tokens'])}, "
          f"lower={len(parsed['lower_tokens'])}, hand={len(parsed['hand_tokens'])}")

    # ========================================================================
    # Step 2: Decode text
    # ========================================================================
    print(f"\n=== Decoding Text ===")
    text_content = decode_text(parsed['text_ids'], tokenizer)
    text_path = os.path.join(args.output_dir, f"{prefix}_text.txt")
    with open(text_path, 'w') as f:
        f.write(text_content)
    print(f"  Text saved to {text_path}")
    print(f"  Content preview: {text_content[:200]}...")

    # ========================================================================
    # Step 2b: Dump full token structure (optional)
    # ========================================================================
    if args.dump_structure:
        print(f"\n=== Dumping Token Structure ===")
        structure_path = os.path.join(args.output_dir, f"{prefix}_structure.txt")
        dump_token_structure(sample, tokenizer, structure_path)

    if args.skip_render:
        print("\n=== Skipping render (--skip_render) ===")
        return

    # ========================================================================
    # Step 3: Detect mode and decode motion tokens
    # ========================================================================
    mode = args.mode
    if mode == "auto":
        n_upper = len(parsed['upper_tokens'])
        n_lower = len(parsed['lower_tokens'])
        n_hand = len(parsed['hand_tokens'])
        if n_upper > 0 and n_lower == 0 and n_hand == 0:
            mode = "fullbody_genmo"
        elif n_upper > 0 and n_lower > 0 and n_hand == 0:
            mode = "upper_lower_genmo"
        else:
            mode = "compositional"
        print(f"  Auto-detected mode: {mode}")

    print(f"\n=== Decoding Motion (mode={mode}) ===")

    if mode == "compositional":
        _, vae_upper, vae_lower, vae_global, vae_hand = load_vae_models(
            device, args.vae_body_ckpt, args.vae_face_ckpt, args.vae_global_ckpt
        )
        rec_upper, rec_lower, rec_hand, n = decode_motion(
            parsed['upper_tokens'], parsed['lower_tokens'], parsed['hand_tokens'],
            vae_upper, vae_lower, vae_hand, vae_global, device
        )
        print(f"  Decoded {n} timesteps ({n/MOTION_FPS:.1f}s at {MOTION_FPS} FPS)")

        print(f"\n=== Reconstructing Body Pose ===")
        bs = 1
        rec_pose, lower2global = reconstruct_body_pose(
            rec_upper, rec_lower, bs, n, device)
        rec_trans = compute_global_translation(
            rec_lower, lower2global, rec_pose, vae_global)
        rec_pose = rec_pose.to(device)
        rec_trans = rec_trans.to(device)

    elif mode == "upper_lower_genmo":
        from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion as LOMVQ
        from multimodal_tokenizers.archs.motiongpt_vq import MotionGPTVQVaeAdapter
        from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle

        print("Loading VAE models...")
        vae_upper = LOMVQ(vae_layer=3, code_num=256, vae_test_dim=78, codebook_size=256, vae_quantizer_lambda=1)
        sd = torch.load(os.path.join(ROOT_DIR, 'model_files/pretrained_cpt/body/lom_vq.ckpt'), map_location='cpu', weights_only=False)['state_dict']
        vae_upper.load_state_dict({k.replace('vae_upper.', ''): v for k, v in sd.items() if k.startswith('vae_upper.')}, strict=True)
        vae_upper.eval().to(device)

        vae_lower = MotionGPTVQVaeAdapter(nfeats=61, code_dim=128, code_num=256, mu=0.95, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=512, output_emb_width=128, norm='GN')
        sd2 = torch.load(os.path.join(ROOT_DIR, 'model_files/pretrained_cpt/VQVAE_0318_NormalUpper_GenmoLower/vqvar_genmo_lower_global_last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
        vae_lower.load_state_dict({k.replace('vae_lower.', ''): v for k, v in sd2.items() if 'vae_lower' in k}, strict=True)
        vae_lower.eval().to(device)
        print("VAE models loaded successfully!")

        upper_t = torch.tensor(parsed['upper_tokens'], device=device).unsqueeze(0)
        lower_t = torch.tensor(parsed['lower_tokens'], device=device).unsqueeze(0)
        with torch.no_grad():
            rec_u = vae_upper.decode(upper_t.int())  # (1, T, 78)
            rec_l = vae_lower.decode(lower_t.int())  # (1, T, 61)
        n = min(rec_u.shape[1], rec_l.shape[1])
        rec_u = rec_u[:, :n]; rec_l = rec_l[:, :n]
        print(f"  Decoded {n} timesteps ({n/MOTION_FPS:.1f}s at {MOTION_FPS} FPS)")

        print(f"\n=== Reconstructing Body Pose ===")
        bs = 1

        # Upper: 13 joints * 6D → matrix → axis-angle → scatter via JOINT_MASK_UPPER
        pose_upper = rec_u[0].reshape(n, 13, 6)
        pose_upper_mat = rotation_6d_to_matrix(pose_upper)
        pose_upper_aa = matrix_to_axis_angle(pose_upper_mat).reshape(n, 13 * 3)
        pose_upper_full = inverse_selection_tensor(pose_upper_aa, JOINT_MASK_UPPER, n)

        # Lower: first 54D = 9 joints * 6D → matrix → axis-angle → scatter via JOINT_MASK_LOWER
        pose_lower = rec_l[0, :, :54].reshape(n, 9, 6)
        pose_lower_mat = rotation_6d_to_matrix(pose_lower)
        lower2global = matrix_to_rotation_6d(pose_lower_mat.clone()).reshape(1, n, 9 * 6)
        pose_lower_aa = matrix_to_axis_angle(pose_lower_mat).reshape(n, 9 * 3)
        pose_lower_full = inverse_selection_tensor(pose_lower_aa, JOINT_MASK_LOWER, n)

        # Combine upper + lower (zeros for hands)
        rec_pose = pose_upper_full + pose_lower_full
        rec_pose[:, POSE_JAW[0]:POSE_JAW[1]] = 0.0

        # Translation: use local_vel from lower genmo output
        local_vel = rec_l[0, :n, 54:57]
        go_6d = lower2global[0, :, :6]  # pelvis/global_orient 6D from lower
        R = rotation_6d_to_matrix(go_6d)
        world_vel = torch.einsum('tij,tj->ti', R, local_vel)
        rec_trans = torch.cumsum(world_vel, dim=0)

    elif mode == "fullbody_genmo":
        from multimodal_tokenizers.archs.motiongpt_vq import MotionGPTVQVaeAdapter
        from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle

        print("Loading VAE model...")
        vae_full = MotionGPTVQVaeAdapter(nfeats=135, code_dim=256, code_num=256, mu=0.99, down_t=2, stride_t=2, depth=3, dilation_growth_rate=3, width=512, output_emb_width=256)
        sd = torch.load(os.path.join(ROOT_DIR, 'model_files/pretrained_cpt/VQVAE_0320_GenmoFull/last.ckpt'), map_location='cpu', weights_only=False)['state_dict']
        vae_full.load_state_dict({k.replace('vae_body.', ''): v for k, v in sd.items() if 'vae_body' in k}, strict=True)
        vae_full.eval().to(device)
        print("VAE model loaded successfully!")

        upper_t = torch.tensor(parsed['upper_tokens'], device=device).unsqueeze(0)
        with torch.no_grad():
            rec135 = vae_full.decode(upper_t.int())[0]  # (T, 135)
        n = rec135.shape[0]
        rec145 = torch.cat([rec135[:, :126], torch.zeros(n, 10, device=device), rec135[:, 126:]], dim=-1)
        print(f"  Decoded {n} timesteps ({n/MOTION_FPS:.1f}s at {MOTION_FPS} FPS)")

        body_r6d = rec145[:, :126].reshape(n, 21, 6)
        go_r6d = rec145[:, 136:142]
        local_vel = rec145[:, 142:145]

        body_aa = matrix_to_axis_angle(rotation_6d_to_matrix(body_r6d.reshape(-1, 6))).reshape(n, 21, 3)
        go_aa = matrix_to_axis_angle(rotation_6d_to_matrix(go_r6d))

        # Build (n, 165) pose
        rec_pose = torch.zeros(n, 165, device=device)
        rec_pose[:, 0:3] = go_aa
        rec_pose[:, 3:66] = body_aa.reshape(n, -1)

        R = rotation_6d_to_matrix(go_r6d)
        world_vel = torch.einsum('tij,tj->ti', R, local_vel)
        rec_trans = torch.cumsum(world_vel, dim=0)

        print(f"\n=== Reconstructing Body Pose ===")

    # Save motion as NPZ
    npz_path = os.path.join(args.output_dir, f"{prefix}_motion.npz")
    np.savez(
        npz_path,
        betas=np.zeros(SMPLX_NUM_BETAS),
        poses=rec_pose.detach().cpu().numpy().reshape(n, SMPLX_NUM_JOINTS * 3),
        expressions=np.zeros((n, SMPLX_NUM_EXPRESSIONS)),
        trans=rec_trans.detach().cpu().numpy().reshape(n, 3),
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=MOTION_FPS,
    )
    print(f"  Motion NPZ saved to {npz_path}")

    # ========================================================================
    # Step 5: Decode audio (after motion to avoid CUDA context corruption)
    # ========================================================================
    audio_waveform = None
    if not args.skip_audio:
        print(f"\n=== Decoding Audio ===")
        try:
            audio_waveform = decode_audio(parsed['audio_ids'], device, args.audio_decoder_dir, tokenizer)
            if audio_waveform is not None:
                audio_path = os.path.join(args.output_dir, f"{prefix}_audio.wav")
                sf.write(audio_path, audio_waveform.numpy(), AUDIO_OUTPUT_SAMPLE_RATE)
                print(f"  Audio saved to {audio_path} "
                      f"({len(audio_waveform)/AUDIO_OUTPUT_SAMPLE_RATE:.1f}s)")
        except Exception as e:
            print(f"  Warning: audio decoding failed: {e}")
            print(f"  Continuing without audio...")
    else:
        print(f"\n=== Skipping audio (--skip_audio) ===")

    # ========================================================================
    # Step 6: Render video
    # ========================================================================
    print(f"\n=== Rendering Video ===")
    smplx_model = load_smplx_model(args.smplx_dir, device)

    video_path = os.path.join(args.output_dir, f"{prefix}_motion.mp4")
    render_video(
        rec_pose, rec_trans, smplx_model, n, device,
        video_path, audio_waveform=audio_waveform,
        fixed_camera=args.fixed_camera,
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n=== Done ===")
    print(f"  Text:   {text_path}")
    if audio_waveform is not None:
        print(f"  Audio:  {os.path.join(args.output_dir, f'{prefix}_audio.wav')}")
    print(f"  Motion: {npz_path}")
    print(f"  Video:  {video_path}")


if __name__ == "__main__":
    main()
