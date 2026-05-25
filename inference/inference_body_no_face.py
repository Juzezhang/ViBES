#!/usr/bin/env python3
"""
Body-only inference script (no face, no hand decoding).

Generates body motion (upper + lower + hand tokens) from text prompts using
a single body model checkpoint. Produces an MP4 video rendered from SMPLX
with neutral face (jaw=0, expression=0).

Pipeline:
  1. Load body model -> generate audio + body tokens (upper/lower/hand) from text
  2. Decode audio tokens -> waveform
  3. Decode body tokens -> upper/lower body poses + global translation
  4. Render body video (SMPLX) with synchronized audio

Usage:
    python inference/inference_body_no_face.py \\
        --checkpoint /path/to/body_checkpoint \\
        --user_text "Hi, how are you?" \\
        --cam_beta 3.5 --fixed_camera --front_view
"""
import sys
import os
import argparse
import uuid

# Setup sys.path before other imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))

if not os.path.exists(os.path.join(ROOT_DIR, "utils")):
    raise RuntimeError(
        f"Cannot find 'utils' directory in project root. "
        f"Expected at: {os.path.join(ROOT_DIR, 'utils')}. "
        f"Please run this script from the project root or inference directory."
    )

if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

_conversational_agent_dir_env = os.getenv('CONVERSATIONAL_AGENT_DIR')
if _conversational_agent_dir_env and os.path.exists(_conversational_agent_dir_env):
    CONVERSATIONAL_AGENT_DIR = _conversational_agent_dir_env
else:
    CONVERSATIONAL_AGENT_DIR = ROOT_DIR

if os.path.exists(CONVERSATIONAL_AGENT_DIR):
    if CONVERSATIONAL_AGENT_DIR in sys.path:
        sys.path.remove(CONVERSATIONAL_AGENT_DIR)
    sys.path.insert(1, CONVERSATIONAL_AGENT_DIR)

# Add speech_related subdirectories to sys.path
speech_related_path = os.path.join(ROOT_DIR, "speech_related")
cosyvoice_path = os.path.join(ROOT_DIR, "speech_related", "cosyvoice")
matcha_path = os.path.join(ROOT_DIR, "speech_related", "Matcha-TTS")
sys.path.insert(0, speech_related_path)
if os.path.exists(cosyvoice_path):
    if cosyvoice_path in sys.path:
        sys.path.remove(cosyvoice_path)
    sys.path.insert(0, cosyvoice_path)
if os.path.exists(matcha_path):
    if matcha_path in sys.path:
        sys.path.remove(matcha_path)
    sys.path.insert(0, matcha_path)

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.modeling_utils import load_sharded_checkpoint

from speech_related.flow_inference import AudioDecoder
from multimodal_tokenizers.utils.rotation_conversions import (
    rotation_6d_to_axis_angle,
    axis_angle_to_6d,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)
from multimodal_tokenizers.utils.utils_videos import write_video

# Re-ensure ROOT_DIR is at the front of sys.path after other imports
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from utils.genmo.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from utils.genmo.camera import create_camera_sensor
from utils.token_utils import extract_modality_tokens_from_response
from utils.tensor_utils import apply_body_token_offset, inverse_selection_tensor
from utils.model_loader import load_smplx_model
from utils.inference_utils import prepare_modality_masks, create_prompt

from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion, VAEConvZero
from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
)

# ============================================================================
# Constants
# ============================================================================

TOKEN_OFFSET = 168736
MOTION_FPS = 25
AUDIO_OUTPUT_SAMPLE_RATE = 22050

LOWER_BODY_FEATURE_DIM = 54
GLOBAL_TRANSLATION_START = 54
GLOBAL_TRANSLATION_END = 57
GLOBAL_FEATURE_PADDING = 7

AUDIO_MEL_DIM = 80

SMPLX_NUM_BETAS = 300
SMPLX_NUM_EXPRESSIONS = 100
SMPLX_NUM_JOINTS = 55
SMPLX_DIMS_PER_JOINT = 3

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

NUM_MODALITIES = 3
MODALITY_MOTION_IDX = 2

VAE_CHECKPOINT_MAIN = './model_files/pretrained_cpt/body/lom_vq.ckpt'
VAE_CHECKPOINT_GLOBAL = os.environ.get(
    'VIBES_VAE_GLOBAL_CKPT',
    os.path.join("/path/to", 'experiments', 'multimodal_tokenizer',
                 'VAE_Global_from_Lower54', 'checkpoints', 'last.ckpt'),
)

SMPLX_MODEL_DIR = os.environ.get(
    'VIBES_SMPLX_MODEL_DIR',
    os.path.join(ROOT_DIR, 'model_files', 'smplx_models'),
)

AUDIO_DECODER_CONFIG = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "config.yaml")
AUDIO_DECODER_FLOW = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", 'flow.pt')
AUDIO_DECODER_HIFT = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", 'hift.pt')


# ============================================================================
# Helper Functions (same as inference_face_body.py)
# ============================================================================

def get_fixed_camera(num_frames, beta=2.5, cam_height_degree=30,
                     target_center_height=1.0, vec_rot=45, device="cuda",
                     front_view=False):
    """Compute a fixed camera position independent of mesh data."""
    from pytorch3d.renderer import PointLights
    from pytorch3d.renderer.cameras import look_at_rotation

    if front_view:
        vec_rot = 0
        cam_height_degree = 0

    vec_rad = vec_rot / 180 * np.pi
    vec = torch.tensor([np.sin(vec_rad), 0, np.cos(vec_rad)]).float()
    vec = vec / torch.norm(vec)

    target_center = torch.zeros(3)
    position = target_center + vec * beta
    position[1] = beta * np.tan(np.pi * cam_height_degree / 180) + target_center_height

    positions = position.unsqueeze(0).repeat(num_frames, 1)
    target_centers = target_center.unsqueeze(0).repeat(num_frames, 1)
    target_centers[:, 1] = target_center_height
    rotation = look_at_rotation(positions, target_centers).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)

    lights = PointLights(device=device, location=[position.tolist()])
    return rotation, translation, lights


def integrate_local_velocity(local_vel, global_orient_aa, init_pos=None):
    """Integrate root local velocity into world positions using global orientation."""
    global_6d = axis_angle_to_6d(global_orient_aa)
    R = rotation_6d_to_matrix(global_6d)
    world_vel = torch.einsum("tij,tj->ti", R, local_vel)
    pos = torch.zeros_like(world_vel)
    if init_pos is None:
        pos[0] = 0.0
    else:
        pos[0] = init_pos
    if world_vel.shape[0] > 1:
        pos[1:] = pos[0:1] + torch.cumsum(world_vel[1:], dim=0)
    return pos


def load_model(tokenizer_path, checkpoint_path, device):
    """Load a model from config and checkpoint."""
    config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModel.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="flash_attention_2",
    ).to(device)
    load_sharded_checkpoint(model, checkpoint_path)
    model.eval()
    return model


def run_generation(model, tokenizer, device, user_text, body_part,
                   max_new_tokens=1024, temperature=0.2, top_p=0.8):
    """Run model generation and return extracted modality tokens."""
    prompt = create_prompt(user_text)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    batch_size, seq_len = inputs.input_ids.shape
    modality_masks, position_encoding_indices = prepare_modality_masks(
        batch_size, seq_len, num_modalities=NUM_MODALITIES, device=device
    )

    with torch.no_grad():
        output_ids, output_modality_masks = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            modality_masks=modality_masks,
            use_cache=True,
            position_encoding_indices=position_encoding_indices,
            body_part=body_part,
        )

    output_ids = apply_body_token_offset(
        output_ids, output_modality_masks, TOKEN_OFFSET, modality_idx=MODALITY_MOTION_IDX
    )

    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(full_response)

    return extract_modality_tokens_from_response(full_response)


def reconstruct_body_pose(rec_upper, rec_lower, batch_size, num_frames, device):
    """
    Reconstruct full body pose from upper + lower body parts.
    Jaw and hands are set to zero (no face model).
    """
    # Upper body: 13 joints * 6D -> axis-angle -> full 165D
    pose_upper = rec_upper.reshape(batch_size, num_frames, 13, 6)
    pose_upper = rotation_6d_to_axis_angle(pose_upper).reshape(batch_size * num_frames, 13 * 3)
    pose_upper_full = inverse_selection_tensor(pose_upper.to(device), JOINT_MASK_UPPER, batch_size * num_frames)

    # Lower body: 9 joints * 6D -> rotation matrix -> axis-angle -> full 165D
    pose_legs = rec_lower[:, :, :LOWER_BODY_FEATURE_DIM]
    pose_lower = pose_legs.reshape(batch_size, num_frames, 9, 6)
    pose_lower_mat = rotation_6d_to_matrix(pose_lower)
    lower2global = matrix_to_rotation_6d(pose_lower_mat.clone()).reshape(batch_size, num_frames, 9 * 6)
    pose_lower = matrix_to_axis_angle(pose_lower_mat).reshape(batch_size * num_frames, 9 * 3)
    pose_lower_full = inverse_selection_tensor(pose_lower, JOINT_MASK_LOWER, batch_size * num_frames)

    # Hands: zeros
    pose_hands = torch.zeros(batch_size * num_frames, 30 * 3, device=device)
    pose_hands_full = inverse_selection_tensor(pose_hands, JOINT_MASK_HANDS, batch_size * num_frames)

    # Combine (jaw stays zero)
    rec_pose = pose_upper_full + pose_lower_full + pose_hands_full

    return rec_pose, lower2global


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_body_from_text(
    model,
    tokenizer,
    device,
    user_text="If you had a superpower for one day, what would you choose?",
    output_dir="./demo",
    output_prefix="response",
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.1,
    cam_beta=2.5,
    fixed_camera=False,
    front_view=False,
):
    """Generate body motion from text (no face)."""
    print("\n=== Loading VAE Models ===")
    smplx_2020 = load_smplx_model(SMPLX_MODEL_DIR, device)

    # Load body VAEs only (no face VAE)
    main_ckpt = torch.load(VAE_CHECKPOINT_MAIN, map_location="cpu", weights_only=False)['state_dict']

    vae_upper = VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=78)
    vae_upper.load_state_dict({k.replace('vae_upper.', ''): v for k, v in main_ckpt.items() if k.startswith('vae_upper.')}, strict=True)
    vae_upper.eval().to(device)

    vae_lower = VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=54)
    vae_lower.load_state_dict({k.replace('vae_lower.', ''): v for k, v in main_ckpt.items() if k.startswith('vae_lower.')}, strict=True)
    vae_lower.eval().to(device)

    vae_hand = VQVAEConvZeroDSUS_PaperVersion(vae_layer=3, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=180)
    vae_hand.load_state_dict({k.replace('vae_hand.', ''): v for k, v in main_ckpt.items() if k.startswith('vae_hand.')}, strict=True)
    vae_hand.eval().to(device)

    vae_global = VAEConvZero(vae_layer=4, code_num=256, codebook_size=256, vae_quantizer_lambda=1, vae_test_dim=61)
    global_ckpt = torch.load(VAE_CHECKPOINT_GLOBAL, map_location="cpu", weights_only=False)
    global_sd = global_ckpt['state_dict'] if 'state_dict' in global_ckpt else global_ckpt
    vae_global.load_state_dict({k.replace('vae_global.', ''): v for k, v in global_sd.items() if 'vae_global' in k}, strict=True)
    vae_global.eval().to(device)

    # ========================================================================
    # Step 1: Generate body tokens + audio tokens
    # ========================================================================
    print("\n=== Generating Body Tokens ===")
    modality_tokens = run_generation(
        model, tokenizer, device, user_text, body_part="body",
        max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
    )
    audio_tokens = modality_tokens['audio']
    upper_tokens = modality_tokens['upper']
    lower_tokens = modality_tokens['lower']
    hand_tokens = modality_tokens['hand']
    print(f"  audio: {len(audio_tokens)}, upper: {len(upper_tokens)}, "
          f"lower: {len(lower_tokens)}, hand: {len(hand_tokens)}")

    # ========================================================================
    # Step 2: Decode audio
    # ========================================================================
    print("\n=== Decoding Audio ===")
    this_uuid = str(uuid.uuid4())
    prompt_speech_feat = torch.zeros(1, 0, AUDIO_MEL_DIM).to(device)
    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)

    audio_decoder = AudioDecoder(
        config_path=AUDIO_DECODER_CONFIG,
        flow_ckpt_path=AUDIO_DECODER_FLOW,
        hift_ckpt_path=AUDIO_DECODER_HIFT,
        device=device,
    )

    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
    tts_speech, _ = audio_decoder.token2wav(
        tts_token,
        uuid=this_uuid,
        prompt_token=flow_prompt_speech_token.to(device),
        prompt_feat=prompt_speech_feat.to(device),
        finalize=True,
    )
    final_speech = tts_speech[0].cpu()

    # ========================================================================
    # Step 3: Decode body tokens
    # ========================================================================
    print("\n=== Decoding Body Tokens ===")
    upper_token_tensor = torch.tensor(upper_tokens, device=device).unsqueeze(0)
    lower_token_tensor = torch.tensor(lower_tokens, device=device).unsqueeze(0)
    hand_token_tensor = torch.tensor(hand_tokens, device=device).unsqueeze(0)

    rec_upper = vae_upper.decode(upper_token_tensor.int())
    rec_lower = vae_lower.decode(lower_token_tensor.int())
    rec_hand = vae_hand.decode(hand_token_tensor.int())

    n = min(rec_upper.shape[1], rec_lower.shape[1], rec_hand.shape[1])
    rec_upper = rec_upper[:, :n, :]
    rec_lower = rec_lower[:, :n, :LOWER_BODY_FEATURE_DIM]

    # ========================================================================
    # Step 4: Reconstruct body pose (no face)
    # ========================================================================
    print("\n=== Reconstructing Body Pose ===")
    bs = 1
    rec_pose, rec_lower2global = reconstruct_body_pose(
        rec_upper, rec_lower, bs, n, device
    )

    # ========================================================================
    # Step 5: Global motion (translation)
    # ========================================================================
    to_global = rec_lower
    if to_global.shape[2] == LOWER_BODY_FEATURE_DIM:
        to_global = F.pad(to_global, (0, GLOBAL_FEATURE_PADDING))
    to_global[:, :, GLOBAL_TRANSLATION_START:GLOBAL_TRANSLATION_END] = 0.0
    to_global[:, :, :LOWER_BODY_FEATURE_DIM] = rec_lower2global
    rec_global = vae_global(to_global)

    rec_trans_v_s = rec_global["rec_pose"][:, :, GLOBAL_TRANSLATION_START:GLOBAL_TRANSLATION_END]
    global_orient_aa = rec_pose[:, POSE_GLOBAL_ORIENT[0]:POSE_GLOBAL_ORIENT[1]]
    rec_trans = integrate_local_velocity(rec_trans_v_s[0], global_orient_aa).unsqueeze(0)

    rec_beta = torch.zeros(SMPLX_NUM_BETAS, device=device)
    rec_pose = rec_pose.to(device)
    rec_trans = rec_trans.to(device)
    rec_exps = torch.zeros(1, n, SMPLX_NUM_EXPRESSIONS, device=device)
    rec_beta = torch.tile(rec_beta, (n, 1))

    # ========================================================================
    # Step 6: SMPLX mesh generation
    # ========================================================================
    print("\n=== Generating SMPLX Mesh ===")
    with torch.no_grad():
        vertices_rec = smplx_2020(
            betas=rec_beta.reshape(n, SMPLX_NUM_BETAS),
            transl=rec_trans.reshape(n, 3),
            expression=rec_exps.reshape(n, SMPLX_NUM_EXPRESSIONS),
            jaw_pose=rec_pose[:, POSE_JAW[0]:POSE_JAW[1]],
            global_orient=rec_pose[:, POSE_GLOBAL_ORIENT[0]:POSE_GLOBAL_ORIENT[1]],
            body_pose=rec_pose[:, POSE_BODY_START:POSE_BODY_END],
            left_hand_pose=rec_pose[:, POSE_LEFT_HAND[0]:POSE_LEFT_HAND[1]],
            right_hand_pose=rec_pose[:, POSE_RIGHT_HAND[0]:POSE_RIGHT_HAND[1]],
            leye_pose=rec_pose[:, POSE_LEFT_EYE[0]:POSE_LEFT_EYE[1]],
            reye_pose=rec_pose[:, POSE_RIGHT_EYE[0]:POSE_RIGHT_EYE[1]],
        )

    # Save motion parameters
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"{output_prefix}.npz")
    np.savez(
        npz_path,
        betas=torch.zeros(SMPLX_NUM_BETAS).numpy(),
        poses=rec_pose.detach().cpu().numpy().reshape(n, SMPLX_NUM_JOINTS * 3),
        expressions=rec_exps.detach().cpu().numpy().reshape(n, SMPLX_NUM_EXPRESSIONS),
        trans=rec_trans.detach().cpu().numpy().reshape(n, 3),
        model='smplx2020',
        gender='neutral',
        mocap_frame_rate=MOTION_FPS,
    )
    print(f"  Motion parameters saved to {npz_path}")

    # ========================================================================
    # Step 7: Render video
    # ========================================================================
    print("\n=== Rendering Video (SMPLX) ===")
    render_faces = torch.from_numpy(smplx_2020.faces.astype(np.int64))
    verts_smplx_raw = vertices_rec.vertices.detach()
    joints_raw = vertices_rec.joints.detach()

    verts = verts_smplx_raw.clone()
    offset = joints_raw[0, 0].clone()
    offset[1] = verts[:, :, 1].min()
    verts = verts - offset
    joints_norm = joints_raw - offset
    T_rot = compute_T_ayfz2ay(joints_norm[[0]], inverse=True)
    verts = apply_T_on_points(verts, T_rot)
    root_points = apply_T_on_points(joints_norm, T_rot)[:, 0, :]

    _, _, K = create_camera_sensor(RENDER_WIDTH, RENDER_HEIGHT, 24)
    renderer = Renderer(
        width=RENDER_WIDTH, height=RENDER_HEIGHT,
        device=device, faces=render_faces, K=K, bin_size=0,
    )
    scale, cx, cz = get_ground_params_from_points(root_points, verts)
    renderer.set_ground(max(scale, 3) * 1.5, cx, cz)

    vec_rot = 0 if front_view else 45
    cam_height_degree = 0 if front_view else 30

    if fixed_camera:
        cam_R, cam_T, lights = get_fixed_camera(
            n, beta=cam_beta, device=device, front_view=front_view)
    else:
        cam_R, cam_T, lights = get_global_cameras_static(
            verts.cpu(), beta=cam_beta, vec_rot=vec_rot,
            cam_height_degree=cam_height_degree)

    pred_images = []
    color = torch.tensor([0.69, 0.39, 0.96], device=device)
    for i in tqdm(range(verts.shape[0]), desc="Rendering frames"):
        cams = renderer.create_camera(cam_R[i], cam_T[i])
        img = renderer.render_with_ground(verts[[i]], color[None], cams, lights)
        pred_images.append(img)

    pred_tensor = torch.from_numpy(np.stack(pred_images)).permute(0, 3, 1, 2)
    video_path = os.path.join(output_dir, f"{output_prefix}.mp4")
    audio_clip = final_speech[:int(pred_tensor.shape[0] / MOTION_FPS * AUDIO_OUTPUT_SAMPLE_RATE)]
    write_video(pred_tensor, video_path, MOTION_FPS,
                audio_clip, AUDIO_OUTPUT_SAMPLE_RATE, "aac")
    print(f"  Video saved to {video_path}")

    print(f"\n=== Done ===")
    print(f"  Video:      {video_path}")
    print(f"  Motion NPZ: {npz_path}")

    return True


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Body-Only Motion Generation (no face)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to body model checkpoint directory')
    parser.add_argument('--output_dir', type=str, default="./test_output",
                        help='Output directory for generated videos')
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"],
                        help='Computing device')
    parser.add_argument('--user_text', type=str,
                        default="If you had a superpower for one day, what would you choose?",
                        help='User input text prompt')
    parser.add_argument('--output_prefix', type=str, default="response",
                        help='Prefix for output filenames (produces <prefix>.mp4, <prefix>.npz)')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='Maximum number of tokens to generate per model')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature for body model')
    parser.add_argument('--top_p', type=float, default=0.1,
                        help='Top-p sampling for body model')
    parser.add_argument('--cam_beta', type=float, default=2.5,
                        help='Camera distance multiplier (lower = closer)')
    parser.add_argument('--fixed_camera', action='store_true',
                        help='Use a fixed camera position (same viewpoint for all frames)')
    parser.add_argument('--front_view', action='store_true',
                        help='Camera faces the front of the SMPL body (eye-level, no elevation)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")
        device = "cpu"
    else:
        device = "cuda:0" if args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("\n=== Step 1: Load Tokenizer ===")
    tokenizer_path = os.path.join(ROOT_DIR, "vibes")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load body model
    print(f"\n=== Step 2: Load Body Model from {args.checkpoint} ===")
    model = load_model(tokenizer_path, args.checkpoint, device)

    # Generate
    generate_body_from_text(
        model, tokenizer, device,
        user_text=args.user_text,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        cam_beta=args.cam_beta,
        fixed_camera=args.fixed_camera,
        front_view=args.front_view,
    )

    print("\n=== Inference Completed ===")


if __name__ == "__main__":
    main()
