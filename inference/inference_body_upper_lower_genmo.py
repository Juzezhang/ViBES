#!/usr/bin/env python3
"""
Inference script for Audio-to-Motion Body generation (upper + lower_genmo variant).

This script generates full-body motion sequences from text prompts using a multimodal
generative model. It uses the standard LOM upper-body VAE together with a MotionGPT-based
GENMO lower-body VAE (no hand VAE, no global VAE).

Features:
- Text-to-motion generation with audio synthesis
- Upper body: standard VQ-VAE (13 joints x 6D = 78D)
- Lower body: MotionGPT GENMO VQ-VAE (9 joints x 6D + local_vel + contact = 61D)
- Local velocity integration for world translation
- SMPLX mesh generation and video rendering

Usage:
    # Recommended: Run from project root directory
    python -m inference.inference_body_upper_lower_genmo \\
        --checkpoint <path_to_checkpoint> \\
        --user_text "Your text prompt here" \\
        --output_dir ./output

    # Alternative: Run directly from inference/ directory
    cd inference
    python inference_body_upper_lower_genmo.py \\
        --checkpoint <path_to_checkpoint> \\
        --user_text "Your text prompt here" \\
        --output_dir ./output
"""
import sys
import os
import argparse
import uuid

# Setup sys.path before other imports
# Ensure we get the project root directory regardless of where the script is run from
_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))

# Validate that ROOT_DIR is correct (should contain utils directory)
if not os.path.exists(os.path.join(ROOT_DIR, "utils")):
    raise RuntimeError(
        f"Cannot find 'utils' directory in project root. "
        f"Expected at: {os.path.join(ROOT_DIR, 'utils')}. "
        f"Please run this script from the project root or inference directory."
    )

# Force ROOT_DIR to be at the beginning of sys.path (remove and re-insert to ensure priority)
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# Add conversational_agent directory to sys.path for conver_agent imports
_conversational_agent_dir_env = os.getenv('CONVERSATIONAL_AGENT_DIR')
if _conversational_agent_dir_env and os.path.exists(_conversational_agent_dir_env):
    CONVERSATIONAL_AGENT_DIR = _conversational_agent_dir_env
else:
    _relative_path = ROOT_DIR
    CONVERSATIONAL_AGENT_DIR = _relative_path

if os.path.exists(CONVERSATIONAL_AGENT_DIR):
    if CONVERSATIONAL_AGENT_DIR in sys.path:
        sys.path.remove(CONVERSATIONAL_AGENT_DIR)
    sys.path.insert(1, CONVERSATIONAL_AGENT_DIR)  # Insert at position 1, after ROOT_DIR

# Add speech_related subdirectories to sys.path
cosyvoice_path = os.path.join(ROOT_DIR, "speech_related", "cosyvoice")
matcha_path = os.path.join(ROOT_DIR, "speech_related", "Matcha-TTS")
if os.path.exists(cosyvoice_path):
    if cosyvoice_path in sys.path:
        sys.path.remove(cosyvoice_path)
    sys.path.insert(0, cosyvoice_path)
if os.path.exists(matcha_path):
    if matcha_path in sys.path:
        sys.path.remove(matcha_path)
    sys.path.insert(0, matcha_path)

# Import external dependencies first
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from einops import einsum as einsum_fn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
from transformers.modeling_utils import load_sharded_checkpoint

from speech_related.flow_inference import AudioDecoder
from multimodal_tokenizers.utils.rotation_conversions import (
    axis_angle_to_6d,
    rotation_6d_to_matrix,
    rotation_6d_to_axis_angle,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)
from multimodal_tokenizers.utils.utils_videos import write_video

# Re-ensure ROOT_DIR is at the front of sys.path after other imports
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# Per-expert checkpoint loading (Expert-1-only checkpoints + GLM-base reconstruction of Expert-0)
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))
from expert_io import is_expert1_checkpoint, load_expert1_checkpoint

# Now import project-local utils modules (after dependencies, but with path re-asserted)
from utils.genmo.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from utils.genmo.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from utils.genmo.camera import create_camera_sensor
from utils.genmo.smplx_utils import make_smplx
from utils.token_utils import extract_modality_tokens_from_response
from utils.tensor_utils import apply_body_token_offset, inverse_selection_tensor
from utils.model_loader import load_smplx_model, extract_state_dict_keys, _extract_state_dict, _load_module_state
from utils.inference_utils import prepare_modality_masks, create_prompt

from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS_PaperVersion
from multimodal_tokenizers.archs.motiongpt_vq import MotionGPTVQVaeAdapter
from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER,
    JOINT_MASK_LOWER,
    JOINT_MASK_HANDS,
)

# ============================================================================
# Constants
# ============================================================================

# Token ID constants
BODY_TOKEN_OFFSET = 168736  # Offset for body motion tokens in vocabulary

# Motion and audio constants
MOTION_FPS = 25  # Frames per second for motion sequences
AUDIO_OUTPUT_SAMPLE_RATE = 22050  # Audio sample rate in Hz

# SMPLX model constants
SMPLX_NUM_BETAS = 300  # Number of shape parameters in SMPLX model
SMPLX_NUM_EXPRESSIONS = 100  # Number of facial expression parameters
SMPLX_NUM_JOINTS = 55  # Number of joints in the skeleton
SMPLX_NUM_DIMS_PER_JOINT = 3  # Dimension per joint (axis-angle rotation)

# Motion feature dimensions
FACE_FEATURE_DIM = 112  # Total dimension of face features
FACE_EXPRESSION_START_IDX = 12  # Starting index for facial expressions
FACE_JAW_POSE_START_IDX = 6  # Starting index for jaw pose
FACE_JAW_POSE_END_IDX = 12  # Ending index for jaw pose

# GENMO lower body feature layout (61D total):
#   [0:54]  = 9 joints x 6D rotation
#   [54:57] = local_vel (3D)
#   [57:61] = contact (4D)
GENMO_LOWER_JOINT_DIM = 54  # 9 joints x 6D
GENMO_LOWER_VEL_START = 54
GENMO_LOWER_VEL_END = 57
GENMO_LOWER_CONTACT_START = 57
GENMO_LOWER_CONTACT_END = 61

# Audio processing constants
AUDIO_MEL_DIM = 80  # Mel-spectrogram dimension for audio prompts

# Pose reconstruction indices (SMPLX joint structure)
POSE_JAW_INDICES = (66, 69)  # Jaw joint indices
POSE_GLOBAL_ORIENT_INDICES = (0, 3)  # Global orientation indices
POSE_BODY_START_IDX = 3  # Body pose starts after global orientation
POSE_BODY_END_IDX = 21 * SMPLX_NUM_DIMS_PER_JOINT + POSE_BODY_START_IDX  # Body pose end
POSE_LEFT_HAND_START_IDX = 25 * SMPLX_NUM_DIMS_PER_JOINT  # Left hand start
POSE_LEFT_HAND_END_IDX = 40 * SMPLX_NUM_DIMS_PER_JOINT  # Left hand end
POSE_RIGHT_HAND_START_IDX = 40 * SMPLX_NUM_DIMS_PER_JOINT  # Right hand start
POSE_RIGHT_HAND_END_IDX = 55 * SMPLX_NUM_DIMS_PER_JOINT  # Right hand end
POSE_LEFT_EYE_INDICES = (69, 72)  # Left eye pose indices
POSE_RIGHT_EYE_INDICES = (72, 75)  # Right eye pose indices

# Video rendering constants
RENDER_WIDTH = 1280  # Output video width
RENDER_HEIGHT = 720  # Output video height
RENDER_SCALE = 6.0  # Scale factor for mesh rendering
VIDEO_COLOR_SCALE = 255.0  # Color scale for video output (0-255 range)
RENDER_CAMERA_Y_OFFSET = -1.2  # Camera Y translation

# VAE model paths
VAE_CHECKPOINT_UPPER = './model_files/pretrained_cpt/body/lom_vq.ckpt'
VAE_CHECKPOINT_LOWER_GENMO = './model_files/pretrained_cpt/VQVAE_0318_NormalUpper_GenmoLower/vqvar_genmo_lower_global_last.ckpt'

# SMPLX model path
SMPLX_MODEL_DIR = os.environ.get(
    'VIBES_SMPLX_MODEL_DIR',
    os.path.join(ROOT_DIR, 'model_files', 'smplx_models'),
)
# GVHMR asset path (smplx2smpl, J_regressor, smpl_faces)
GVHMR_ASSET_DIR = os.path.join(ROOT_DIR, 'model_files', 'gvhmr')

# Audio decoder paths
AUDIO_DECODER_CONFIG = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "config.yaml")
AUDIO_DECODER_FLOW = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", 'flow.pt')
AUDIO_DECODER_HIFT = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", 'hift.pt')

# Model configuration
NUM_MODALITIES = 3  # Number of modalities: text, audio, motion
MODALITY_BODY_IDX = 2  # Index of body motion modality

# ============================================================================
# VAE Loading
# ============================================================================

def load_upper_lower_genmo_vae_models(device):
    """
    Load VAE models for the upper_lower_genmo configuration.

    Upper: VQVAEConvZeroDSUS_PaperVersion loaded from lom_vq.ckpt (vae_upper.* keys)
    Lower: MotionGPTVQVaeAdapter loaded from vqvar_genmo_lower_global_last.ckpt (vae_lower.* keys)

    Returns:
        Tuple of (vae_upper, vae_lower)
    """
    print("Loading VAE models (upper + lower_genmo)...")

    for path in [VAE_CHECKPOINT_UPPER, VAE_CHECKPOINT_LOWER_GENMO]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Upper body VAE (same architecture as original)
    vae_upper = VQVAEConvZeroDSUS_PaperVersion(
        vae_layer=3,
        code_num=256,
        vae_test_dim=78,
        codebook_size=256,
        vae_quantizer_lambda=1,
    )

    # Lower body VAE (MotionGPT GENMO architecture)
    vae_lower = MotionGPTVQVaeAdapter(
        nfeats=61,
        code_dim=128,
        code_num=256,
        mu=0.95,
        down_t=2,
        stride_t=2,
        depth=3,
        dilation_growth_rate=3,
        width=512,
        output_emb_width=128,
        norm='GN',
    )

    # Load upper weights from lom_vq.ckpt
    checkpoint_upper = torch.load(VAE_CHECKPOINT_UPPER, map_location="cpu", weights_only=False)
    state_dict_upper = _extract_state_dict(checkpoint_upper, VAE_CHECKPOINT_UPPER)
    _load_module_state(vae_upper, state_dict_upper, "vae_upper")

    # Load lower weights from genmo lower checkpoint
    checkpoint_lower = torch.load(VAE_CHECKPOINT_LOWER_GENMO, map_location="cpu", weights_only=False)
    state_dict_lower = _extract_state_dict(checkpoint_lower, VAE_CHECKPOINT_LOWER_GENMO)
    _load_module_state(vae_lower, state_dict_lower, "vae_lower")

    for vae in [vae_upper, vae_lower]:
        vae.eval()
        vae.to(device)

    print("VAE models loaded successfully!")
    return vae_upper, vae_lower


# ============================================================================
# Main Generation Function
# ============================================================================

def integrate_local_velocity(local_vel, global_orient_aa, init_pos=None):
    """
    Integrate root local velocity into world positions using global orientation.
    local_vel: (T, 3), global_orient_aa: (T, 3)
    """
    global_6d = axis_angle_to_6d(global_orient_aa)
    R = rotation_6d_to_matrix(global_6d)  # (T, 3, 3)
    world_vel = torch.einsum("tij,tj->ti", R, local_vel)
    pos = torch.zeros_like(world_vel)
    if init_pos is None:
        pos[0] = 0.0
    else:
        pos[0] = init_pos
    if world_vel.shape[0] > 1:
        pos[1:] = pos[0:1] + torch.cumsum(world_vel[1:], dim=0)
    return pos


def reconstruct_upper_lower_genmo_pose(
    rec_upper, rec_lower_genmo, batch_size, num_frames, device
):
    """
    Reconstruct full body pose from upper (LOM) and lower (GENMO) VAE outputs.

    Upper: (B, T, 78) = 13 joints x 6D  -- joints [3,6,9,12,13,14,15,16,17,18,19,20,21]
    Lower: (B, T, 61) = [0:54] 9 joints x 6D, [54:57] local_vel, [57:61] contact
           -- joints [0,1,2,4,5,7,8,10,11]

    Returns:
        rec_pose: (B*T, 165) axis-angle full body pose
        local_vel: (B, T, 3)
        contact: (B, T, 4)
    """
    # --- Upper body: 6D -> axis-angle -> scatter into 55-joint skeleton ---
    rec_pose_upper = rec_upper.reshape(batch_size, num_frames, 13, 6)
    rec_pose_upper = rotation_6d_to_axis_angle(rec_pose_upper).reshape(
        batch_size * num_frames, 13 * 3
    )
    rec_pose_upper_recover = inverse_selection_tensor(
        rec_pose_upper.to(device), JOINT_MASK_UPPER, batch_size * num_frames
    )

    # --- Lower body: extract joint rotations, local_vel, contact ---
    lower_joints_6d = rec_lower_genmo[:, :, :GENMO_LOWER_JOINT_DIM]  # (B, T, 54)
    local_vel = rec_lower_genmo[:, :, GENMO_LOWER_VEL_START:GENMO_LOWER_VEL_END]  # (B, T, 3)
    contact = rec_lower_genmo[:, :, GENMO_LOWER_CONTACT_START:GENMO_LOWER_CONTACT_END]  # (B, T, 4)

    rec_pose_lower = lower_joints_6d.reshape(batch_size, num_frames, 9, 6)
    rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
    rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(
        batch_size * num_frames, 9 * 3
    )
    rec_pose_lower_recover = inverse_selection_tensor(
        rec_pose_lower, JOINT_MASK_LOWER, batch_size * num_frames
    )

    # --- Combine upper + lower (no hands) ---
    rec_pose_hands = torch.zeros(batch_size * num_frames, 30 * 3, device=device)
    rec_pose_hands_recover = inverse_selection_tensor(
        rec_pose_hands, JOINT_MASK_HANDS, batch_size * num_frames
    )

    # Jaw pose zeroed out (no face generation)
    rec_pose_jaw = torch.zeros(batch_size * num_frames, 3, device=device)

    rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
    rec_pose[:, POSE_JAW_INDICES[0]:POSE_JAW_INDICES[1]] = rec_pose_jaw

    return rec_pose, local_vel, contact


def generate_motion_from_text(
    model,
    tokenizer,
    device,
    user_text="If you had a superpower for one day, what would you choose?",
    output_dir="./demo",
    output_filename="response.mp4",
    max_new_tokens=1024,
    temperature=0.0,
    top_p=0.1
):
    """Generate motion and audio from text prompt using the trained model.

    This function performs the complete pipeline:
    1. Loads VAE models for motion decoding (upper + lower_genmo only)
    2. Tokenizes input text and generates multimodal tokens
    3. Decodes audio tokens to waveform
    4. Decodes motion tokens to 3D poses
    5. Integrates local velocity for world translation
    6. Reconstructs full body pose using SMPLX model
    7. Renders video with synchronized audio

    Args:
        model: The trained multimodal generative model
        tokenizer: Tokenizer for the model vocabulary
        device: Computing device (cuda/cpu)
        user_text: User input text prompt
        output_dir: Directory to save output video
        output_filename: Output video filename
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Top-p (nucleus) sampling parameter

    Returns:
        bool: True if generation succeeded
    """
    print(f"\n=== Step 4: Generate Motion from Text ===")

    # Load SMPLX body model for pose reconstruction
    smplx_2020 = load_smplx_model(SMPLX_MODEL_DIR, device)

    # Load VAE models (upper + lower_genmo only)
    vae_upper, vae_lower = load_upper_lower_genmo_vae_models(device)

    # Prepare input prompt with proper formatting
    prompt = create_prompt(user_text)

    # Tokenize input text
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    print(f"Starting generation...")
    original_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    print(f"Original text: {original_text}")

    # Prepare modality masks for multimodal generation
    batch_size, seq_len = inputs.input_ids.shape[0], inputs.input_ids.shape[1]
    modality_masks_original, position_encoding_indices = prepare_modality_masks(
        batch_size, seq_len, num_modalities=NUM_MODALITIES, device=inputs.input_ids.device
    )

    # Generate multimodal tokens (text, audio, motion)
    with torch.no_grad():
        output_ids, output_modality_masks = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            modality_masks=modality_masks_original,
            use_cache=True,
            position_encoding_indices=position_encoding_indices,
            body_part="upper_lower_genmo"
        )

    # Apply body token offset to correct token IDs
    output_ids = apply_body_token_offset(
        output_ids,
        output_modality_masks,
        BODY_TOKEN_OFFSET,
        modality_idx=MODALITY_BODY_IDX
    )

    # Decode generated tokens back to text for inspection
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(full_response)

    # Extract tokens for each modality from the generated response
    # Only extract audio, upper, and lower (no hand)
    modality_tokens = extract_modality_tokens_from_response(
        full_response,
        modality_names=['audio', 'upper', 'lower']
    )
    audio_tokens = modality_tokens['audio']
    upper_tokens = modality_tokens['upper']
    lower_tokens = modality_tokens['lower']

    # ========================================================================
    # Audio Decoding
    # ========================================================================

    # Prepare audio decoder with empty prompts for zero-shot generation
    this_uuid = str(uuid.uuid4())
    prompt_speech_feat = torch.zeros(1, 0, AUDIO_MEL_DIM).to(device)
    flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)

    audio_decoder = AudioDecoder(
        config_path=AUDIO_DECODER_CONFIG,
        flow_ckpt_path=AUDIO_DECODER_FLOW,
        hift_ckpt_path=AUDIO_DECODER_HIFT,
        device=device
    )

    # Decode audio tokens to waveform
    tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
    tts_speech, tts_mel = audio_decoder.token2wav(
        tts_token,
        uuid=this_uuid,
        prompt_token=flow_prompt_speech_token.to(device),
        prompt_feat=prompt_speech_feat.to(device),
        finalize=True
    )
    final_speech = tts_speech[0].cpu()

    # ========================================================================
    # Motion Token Decoding
    # ========================================================================

    # Convert token lists to tensors and decode using VAE models
    upper_token_tensor = torch.tensor(upper_tokens, device=device).unsqueeze(0)
    lower_token_tensor = torch.tensor(lower_tokens, device=device).unsqueeze(0)

    # Decode motion tokens to continuous features using VAE decoders
    rec_upper = vae_upper.decode(upper_token_tensor.int())   # (1, T, 78)
    rec_lower = vae_lower.decode(lower_token_tensor.int())   # (1, T, 61)

    # Align sequence lengths across upper and lower
    n = min(rec_upper.shape[1], rec_lower.shape[1])
    rec_upper = rec_upper[:, :n, :]
    rec_lower = rec_lower[:, :n, :]

    # ========================================================================
    # Full Body Pose Reconstruction
    # ========================================================================

    pose_batch_size = 1
    rec_pose, local_vel, contact = reconstruct_upper_lower_genmo_pose(
        rec_upper, rec_lower, pose_batch_size, n, device
    )

    # ========================================================================
    # Calculate Translation from Local Velocity
    # ========================================================================

    # Extract local velocity from GENMO lower output and integrate to world translation
    rec_trans_v_s = local_vel  # (1, T, 3)

    # Convert local velocity to world translation using global orientation
    global_orient_aa = rec_pose[:, POSE_GLOBAL_ORIENT_INDICES[0]:POSE_GLOBAL_ORIENT_INDICES[1]]
    rec_trans = integrate_local_velocity(
        rec_trans_v_s[0], global_orient_aa
    ).unsqueeze(0)

    # Initialize shape parameters (betas) for SMPLX model
    rec_beta = torch.zeros(SMPLX_NUM_BETAS, device=device)

    # Initialize face features (zeros, no face generation)
    rec_exps = torch.zeros(pose_batch_size, n, SMPLX_NUM_EXPRESSIONS, device=device, dtype=torch.float32)

    # Convert axis-angle rotations to 6D rotation representation
    rec_pose = rec_pose.to(device)
    rec_trans = rec_trans.to(device)
    rec_pose_6d = axis_angle_to_6d(
        rec_pose.reshape(n, SMPLX_NUM_JOINTS, SMPLX_NUM_DIMS_PER_JOINT)
    ).reshape(n, SMPLX_NUM_JOINTS * 6)
    rec_exps = rec_exps.to(device)
    rec_beta = torch.tile(rec_beta, (n, 1))
    rec_pose_6d = rec_pose_6d.to(device)

    # ========================================================================
    # SMPLX Mesh Generation
    # ========================================================================

    # Generate 3D mesh vertices using SMPLX model
    with torch.no_grad():
        vertices_rec = smplx_2020(
            betas=rec_beta.reshape(n, SMPLX_NUM_BETAS),
            transl=rec_trans.reshape(n, 3),
            expression=rec_exps.reshape(n, SMPLX_NUM_EXPRESSIONS),
            jaw_pose=rec_pose[:, POSE_JAW_INDICES[0]:POSE_JAW_INDICES[1]],
            global_orient=rec_pose[:, POSE_GLOBAL_ORIENT_INDICES[0]:POSE_GLOBAL_ORIENT_INDICES[1]],
            body_pose=rec_pose[:, POSE_BODY_START_IDX:POSE_BODY_END_IDX],
            left_hand_pose=rec_pose[:, POSE_LEFT_HAND_START_IDX:POSE_LEFT_HAND_END_IDX],
            right_hand_pose=rec_pose[:, POSE_RIGHT_HAND_START_IDX:POSE_RIGHT_HAND_END_IDX],
            leye_pose=rec_pose[:, POSE_LEFT_EYE_INDICES[0]:POSE_LEFT_EYE_INDICES[1]],
            reye_pose=rec_pose[:, POSE_RIGHT_EYE_INDICES[0]:POSE_RIGHT_EYE_INDICES[1]],
        )

    # ========================================================================
    # Video Rendering (GENMO-style: SMPLX -> SMPL conversion + proper camera)
    # ========================================================================

    # Load SMPL conversion assets
    smplx2smpl = torch.load(
        os.path.join(GVHMR_ASSET_DIR, "smplx2smpl_sparse.pt"), weights_only=True
    ).to(device)
    J_regressor = torch.load(
        os.path.join(GVHMR_ASSET_DIR, "smpl_neutral_J_regressor.pt"), weights_only=True
    ).to(device)
    faces_smpl = make_smplx("smpl").faces

    # Convert SMPLX vertices to SMPL vertices
    verts_smplx = vertices_rec.vertices.detach()
    verts_smpl = torch.stack([smplx2smpl @ v for v in verts_smplx])

    # Normalize: origin XZ, floor level, face +Z
    def _normalize_vertices(v, J_reg):
        v = v.clone()
        offset = einsum_fn(J_reg, v[0], "j v, v i -> j i")[0]
        offset[1] = v[:, :, 1].min()
        v -= offset
        joints_for_rot = einsum_fn(J_reg, v[[0]], "j v, l v i -> l j i")
        T_rot = compute_T_ayfz2ay(joints_for_rot, inverse=True)
        return apply_T_on_points(v, T_rot)

    verts = _normalize_vertices(verts_smpl, J_regressor)
    joints = einsum_fn(J_regressor, verts, "j v, l v i -> l j i")

    # Setup camera with 24mm lens
    _, _, K = create_camera_sensor(RENDER_WIDTH, RENDER_HEIGHT, 24)
    renderer = Renderer(
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        device=device,
        faces=faces_smpl,
        K=K,
        bin_size=0,
    )
    scale, cx, cz = get_ground_params_from_points(joints[:, 0], verts)
    renderer.set_ground(max(scale, 3) * 1.5, cx, cz)
    cam_R, cam_T, lights = get_global_cameras_static(verts.cpu())

    pred_images = []
    color = torch.tensor([0.69, 0.39, 0.96], device=device)
    print("Rendering video frames...")
    for i in tqdm(range(verts.shape[0])):
        cams = renderer.create_camera(cam_R[i], cam_T[i])
        img = renderer.render_with_ground(verts[[i]], color[None], cams, lights)
        pred_images.append(img)

    # Stack frames and prepare audio
    pred_images_tensor = torch.from_numpy(np.stack(pred_images)).permute(0, 3, 1, 2)
    os.makedirs(output_dir, exist_ok=True)
    dump_path = os.path.join(output_dir, output_filename)

    # Trim audio to match video duration
    print(f"Saving video to: {dump_path}")
    audio_clip = final_speech
    audio_clip = audio_clip[:int(pred_images_tensor.shape[0] / MOTION_FPS * AUDIO_OUTPUT_SAMPLE_RATE)]

    # Write video with synchronized audio
    write_video(
        pred_images_tensor,
        dump_path,
        MOTION_FPS,
        audio_clip,
        AUDIO_OUTPUT_SAMPLE_RATE,
        "aac"
    )
    print("Video saved successfully!")

    return True


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(
        description='Audio-to-Motion Body Generation Inference (Upper + Lower GENMO)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="/path/to/experiments/glm4voice_conversational_mot_layernum_40_modalitynum_3_rotation_body_a2m_v6/checkpoint-109000",
        help='Path to the trained model checkpoint directory'
    )
    parser.add_argument(
        '--glm_base_path',
        type=str,
        default="THUDM/glm-4-voice-9b",
        help='GLM-4-Voice base used to reconstruct the frozen text/audio expert (Expert-0) '
             'when --checkpoint is an Expert-1-only (motion) checkpoint. Ignored for full checkpoints.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="./test_output",
        help='Output directory for generated videos'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='Computing device: cuda or cpu'
    )
    parser.add_argument(
        '--user_text',
        type=str,
        default="If you had a superpower for one day, what would you choose?",
        help='User input text prompt for motion generation'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        default="response.mp4",
        help='Output video filename'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1024,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature (0.0 = deterministic, higher = more diverse)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.1,
        help='Top-p (nucleus) sampling parameter (0.0-1.0)'
    )
    args = parser.parse_args()

    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU instead.")
        device = "cpu"
    else:
        device = "cuda:0" if args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    print("\n=== Step 1: Load Tokenizer ===")
    tokenizer_path = os.path.join(ROOT_DIR, "vibes")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load model configuration and create model instance
    print("\n=== Step 2: Load Base Model ===")
    config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)

    base_model = AutoModel.from_config(
        config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="flash_attention_2",
    ).to(device)

    # Load model weights from checkpoint.
    # Expert-1-only checkpoints (marked with expert_checkpoint.json) store just the trained motion
    # expert; reconstruct the frozen text/audio expert (Expert-0) from the GLM-4-Voice base and
    # merge. Full checkpoints (no marker) load normally for backward compatibility.
    print(f"\n=== Step 3: Load Model Weights from {args.checkpoint} ===")
    if is_expert1_checkpoint(args.checkpoint):
        print(f"  Detected Expert-1-only checkpoint; reconstructing Expert-0 from {args.glm_base_path}")
        _, _, unexpected = load_expert1_checkpoint(base_model, args.checkpoint, args.glm_base_path)
        unexpected = [k for k in unexpected if "rotary_pos_emb" not in k]
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    else:
        load_sharded_checkpoint(base_model, args.checkpoint)

    model = base_model
    model.eval()

    # Generate motion from text
    generate_motion_from_text(
        model,
        tokenizer,
        device,
        user_text=args.user_text,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print("\n=== Inference Completed Successfully ===")


if __name__ == "__main__":
    main()
