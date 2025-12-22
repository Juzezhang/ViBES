#!/usr/bin/env python3
"""
Inference script for Audio-to-Motion Face generation.

This script generates facial motion sequences from text prompts using a multimodal
generative model. It combines text-to-speech, face token generation, and 3D face
reconstruction to produce animated video outputs with synchronized audio.

Features:
- Text-to-face motion generation with audio synthesis
- Multi-modal token extraction and decoding
- VAE-based face motion reconstruction
- FLAME mesh generation and video rendering

Usage:
    # Recommended: Run from project root directory
    python -m inference.inference_a2m_face \\
        --checkpoint <path_to_checkpoint> \\
        --user_text "Your text prompt here" \\
        --output_dir ./output
    
    # Alternative: Run directly from inference/ directory
    cd inference
    python inference_a2m_face.py \\
        --checkpoint <path_to_checkpoint> \\
        --user_text "Your text prompt here" \\
        --output_dir ./output
"""
import sys
import os
import argparse
import uuid
import time

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
# This enables imports like: from conver_agent.archs.lom_vq import ...
# This is needed when using conda activate conver_agent environment
# Path resolution priority:
#   1) CONVERSATIONAL_AGENT_DIR environment variable (if set)
#   2) Relative path from project root (../conversational_agent)
#   3) Default absolute path as fallback
_conversational_agent_dir_env = os.getenv('CONVERSATIONAL_AGENT_DIR')
if _conversational_agent_dir_env and os.path.exists(_conversational_agent_dir_env):
    CONVERSATIONAL_AGENT_DIR = _conversational_agent_dir_env
else:
    # _relative_path = os.path.join(os.path.dirname(ROOT_DIR), 'conversational_agent')
    _relative_path = ROOT_DIR
    CONVERSATIONAL_AGENT_DIR = _relative_path

if os.path.exists(CONVERSATIONAL_AGENT_DIR):
    if CONVERSATIONAL_AGENT_DIR in sys.path:
        sys.path.remove(CONVERSATIONAL_AGENT_DIR)
    sys.path.insert(1, CONVERSATIONAL_AGENT_DIR)  # Insert at position 1, after ROOT_DIR

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

# Import external dependencies first
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
from transformers.modeling_utils import load_sharded_checkpoint
from speech_related.flow_inference import AudioDecoder
from multimodal_tokenizers.utils.rotation_conversions import rotation_6d_to_axis_angle
from multimodal_tokenizers.utils.renderer_utils import RenderMesh
from multimodal_tokenizers.utils.utils_videos import write_video

# Re-ensure ROOT_DIR is at the front of sys.path after other imports
# (some imports may modify sys.path, so we need to re-assert priority)
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# # Now import project-local modules (after dependencies, but with path re-asserted)
# from qwen2_5_omni import ChatGLMForConditionalGenerationMotExpertNum2

# Import conver_agent modules (after path setup)
from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS1_PaperVersion
from smplx import FLAME

# ============================================================================
# Constants
# ============================================================================

# Token ID constants
FACE_TOKEN_OFFSET = 168736  # Offset for face motion tokens in vocabulary
AUDIO_TOKEN_MIN = 152353  # Minimum audio token ID
AUDIO_TOKEN_MAX = 168735  # Maximum audio token ID
AUDIO_TOKENS_PER_GROUP = 26  # Number of audio tokens per group
FACE_TOKENS_PER_GROUP = 53  # Number of face tokens per group

# Motion and audio constants
MOTION_FPS = 25  # Frames per second for motion sequences
AUDIO_OUTPUT_SAMPLE_RATE = 22050  # Audio sample rate in Hz

# Face feature dimensions
FACE_FEATURE_DIM = 112  # Total dimension of face features
FACE_HEAD_POSE_START_IDX = 0  # Starting index for head pose (6D rotation)
FACE_HEAD_POSE_END_IDX = 6  # Ending index for head pose
FACE_JAW_POSE_START_IDX = 6  # Starting index for jaw pose (6D rotation)
FACE_JAW_POSE_END_IDX = 12  # Ending index for jaw pose
FACE_EXPRESSION_START_IDX = 12  # Starting index for facial expressions

# Audio processing constants
AUDIO_MEL_DIM = 80  # Mel-spectrogram dimension for audio prompts

# Video rendering constants
RENDER_IMAGE_SIZE = 512  # Output image size for video rendering
RENDER_SCALE = 1.0  # Scale factor for mesh rendering
VIDEO_COLOR_SCALE = 255.0  # Color scale for video output (0-255 range)

# FLAME model constants
FLAME_NUM_EXPRESSIONS = 100  # Number of facial expression parameters
FLAME_BATCH_SIZE_VISUALIZE = 100  # Batch size for FLAME visualization

# VAE model paths
VAE_CHECKPOINT_FACE = os.path.join(CONVERSATIONAL_AGENT_DIR, 'model_files/pretrained_cpt/face/face.ckpt')

# FLAME model path
FLAME_MODEL_DIR = os.path.join(ROOT_DIR, "model_files/FLAME2020")

# Audio decoder paths
AUDIO_DECODER_CONFIG = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "config.yaml")
AUDIO_DECODER_FLOW = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", 'flow.pt')
AUDIO_DECODER_HIFT = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", 'hift.pt')

# Model configuration
NUM_MODALITIES = 3  # Number of modalities: text, audio, face motion
MODALITY_FACE_IDX = 2  # Index of face motion modality

# ============================================================================
# Utility Functions
# ============================================================================

def string_to_token(string_list):
    """
    Extract token IDs from token string representations.
    
    Converts strings like "<|audio_1|>" or "<|face_1|>" to integer token IDs.
    
    Args:
        string_list: List of token strings containing modality tokens
        
    Returns:
        List of integer token IDs
    """
    return [int(val.split('_')[1].split('|')[0]) for val in string_list if isinstance(val, str)]


def extract_modality_tokens_from_response(response_text):
    """
    Extract tokens for each modality from the generated response text.
    
    Args:
        response_text: Full response text containing modality tokens
        
    Returns:
        Dictionary with keys: 'audio', 'face', 'upper', 'lower', 'hand'
        Each value is a list of token IDs
    """
    response_split = response_text.split("<|")
    return {
        'audio': string_to_token([s for s in response_split if "audio_" in s]),
        'face': string_to_token([s for s in response_split if "face_" in s]),
        'upper': string_to_token([s for s in response_split if "upper_" in s]),
        'lower': string_to_token([s for s in response_split if "lower_" in s]),
        'hand': string_to_token([s for s in response_split if "hand_" in s])
    }


def load_vae_face_model(device):
    """
    Load and initialize VAE model for face motion.
    
    Args:
        device: Device to load model on (e.g., 'cuda' or 'cpu')
        
    Returns:
        Initialized VAE face model
    """
    print("Loading VAE face model...")
    
    # Initialize VAE model
    vae_face = VQVAEConvZeroDSUS1_PaperVersion(
        vae_layer=3,
        code_num=512,
        codebook_size=512,
        vae_quantizer_lambda=1,
        vae_test_dim=FACE_FEATURE_DIM
    )
    
    # Load checkpoint
    state_dict_face_old = torch.load(VAE_CHECKPOINT_FACE, map_location="cpu", weights_only=False)
    # state_dict_face_old = checkpoint_face['state_dict']
    # state_dict_face_old = checkpoint_face
    
    # Create new state dict with modified keys
    state_dict_face = {}
    for key, value in state_dict_face_old.items():
        if 'vae_face' in key:
            new_key = key.replace('vae_face.', '')
            state_dict_face[new_key] = value
    
    # Load state dictionary
    vae_face.load_state_dict(state_dict_face, strict=True)
    
    # Set to evaluation mode and move to device
    vae_face.eval()
    vae_face.to(device)
    
    print("VAE face model loaded successfully!")
    return vae_face


def create_prompt(user_text):
    """
    Create formatted prompt for face generation.
    
    Args:
        user_text: User input text prompt
        
    Returns:
        Formatted prompt string
    """
    text = "<|user|>\n" + user_text
    prompt = "<|system|>\nUser will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
    prompt += text
    prompt += "<|assistant|>streaming_transcription\n"
    return prompt


def prepare_modality_masks(batch_size, seq_len, num_modalities, device):
    """
    Prepare modality masks for multimodal generation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_modalities: Number of modalities
        device: Device for tensors
        
    Returns:
        Tuple of (modality_masks, position_encoding_indices)
    """
    # Initialize modality masks: text=1, audio=0, face=0
    modality_ids_0 = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    modality_ids_1 = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    modality_ids_2 = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Stack to create modality_masks: (num_modalities, batch_size, seq_len)
    modality_masks = torch.stack([modality_ids_0, modality_ids_1, modality_ids_2], dim=0)
    position_encoding_indices = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0)
    
    return modality_masks, position_encoding_indices


def apply_face_token_offset(output_ids, output_modality_masks, offset, modality_idx):
    """
    Apply token offset to face tokens in the output.
    
    Args:
        output_ids: Generated token IDs
        output_modality_masks: Modality masks indicating which tokens belong to which modality
        offset: Token offset to apply (FACE_TOKEN_OFFSET)
        modality_idx: Index of face modality in modality_masks
        
    Returns:
        Modified output_ids with face token offset applied
    """
    for i in range(output_ids.shape[0]):
        for j in range(output_ids.shape[1]):
            if output_modality_masks[modality_idx][i, j]:
                output_ids[i, j] += offset
    return output_ids

# ============================================================================
# Main Generation Function
# ============================================================================

def generate_face_from_text(
    model,
    tokenizer,
    device,
    user_text="If you had a superpower for one day, what would you choose?",
    output_dir="./demo",
    output_filename="response.mp4",
    max_new_tokens=1024,
    temperature=0.2,
    top_p=0.8
):
    """
    Generate face motion and audio from text prompt using the trained model.
    
    This function performs the complete pipeline:
    1. Loads VAE model for face motion decoding
    2. Tokenizes input text and generates multimodal tokens
    3. Decodes audio tokens to waveform
    4. Decodes face tokens to 3D face features
    5. Reconstructs face mesh using FLAME model
    6. Renders video with synchronized audio
    
    Args:
        model: The trained multimodal generative model
        tokenizer: Tokenizer for the model vocabulary
        device: Computing device (cuda/cpu)
        user_text: User input text prompt
        output_dir: Directory to save output video
        output_filename: Output video filename
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.2 = moderate diversity)
        top_p: Top-p (nucleus) sampling parameter
        
    Returns:
        bool: True if generation succeeded
    """
    print(f"\n=== Step 4: Generate Face Motion from Text ===")

    # Load VAE model for face motion
    vae_face = load_vae_face_model(device)

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

    # Generate multimodal tokens (text, audio, face motion)
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
            body_part="face"
        )

    # Apply face token offset to correct token IDs
    output_ids = apply_face_token_offset(
        output_ids,
        output_modality_masks,
        FACE_TOKEN_OFFSET,
        modality_idx=MODALITY_FACE_IDX
    )

    # Decode generated tokens back to text for inspection
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(full_response)
    
    # Extract tokens for each modality from the generated response
    modality_tokens = extract_modality_tokens_from_response(full_response)
    audio_tokens = modality_tokens['audio']
    face_tokens = modality_tokens['face']

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
    # Face Token Decoding
    # ========================================================================
    
    # Convert face token list to tensor and decode using VAE model
    face_token_tensor = torch.tensor(face_tokens, device=device).unsqueeze(0)
    rec_face = vae_face.decode(face_token_tensor.int())
    
    # Extract face features: head pose, jaw pose, and expressions
    rec_head_pose = rotation_6d_to_axis_angle(rec_face[0, :, FACE_HEAD_POSE_START_IDX:FACE_HEAD_POSE_END_IDX])
    rec_jaw_pose = rotation_6d_to_axis_angle(rec_face[0, :, FACE_JAW_POSE_START_IDX:FACE_JAW_POSE_END_IDX])
    rec_exp = rec_face[0, :, FACE_EXPRESSION_START_IDX:]
    n = rec_face.shape[1]

    # ========================================================================
    # FLAME Mesh Generation
    # ========================================================================
    
    # Initialize FLAME model for face mesh generation
    flame_model = FLAME(
        FLAME_MODEL_DIR,
        num_expression_coeffs=FLAME_NUM_EXPRESSIONS,
        ext='pkl',
        batch_size=FLAME_BATCH_SIZE_VISUALIZE
    ).to(device)
    
    faces = torch.tensor(flame_model.faces.astype(np.int32), dtype=torch.int64)
    mesh_renderer = RenderMesh(image_size=RENDER_IMAGE_SIZE, faces=faces, scale=RENDER_SCALE)

    pred_images = []

    # Process face sequence in batches
    for i in range(0, n, FLAME_BATCH_SIZE_VISUALIZE):
        # Adjust batch size for the last batch
        if FLAME_BATCH_SIZE_VISUALIZE > n - i:
            current_batch_size = n - i
            flame_model = FLAME(
                FLAME_MODEL_DIR,
                num_expression_coeffs=FLAME_NUM_EXPRESSIONS,
                ext='pkl',
                batch_size=current_batch_size
            ).to(device)
        else:
            current_batch_size = FLAME_BATCH_SIZE_VISUALIZE

        # Run FLAME model to generate face mesh
        with torch.no_grad():
            flame_out = flame_model(
                global_orient=rec_head_pose[i:i+current_batch_size, :],
                expression=rec_exp[i:i+current_batch_size, :],
                jaw_pose=rec_jaw_pose[i:i+current_batch_size, :],
                shape=torch.zeros(current_batch_size, 100).to(device),
            )
        
        # Render each frame
        verts = flame_out['vertices'].detach()
        for v in tqdm(verts):
            rgb = mesh_renderer(v[None])[0]
            pred_images.append(rgb.cpu()[0] / VIDEO_COLOR_SCALE)

    # ========================================================================
    # Video Rendering
    # ========================================================================
    
    # Stack frames and prepare audio
    pred_images_tensor = torch.stack(pred_images)
    os.makedirs(output_dir, exist_ok=True)
    dump_path = os.path.join(output_dir, output_filename)
    
    # Trim audio to match video duration
    print(f"Saving video to: {dump_path}")
    audio_clip = final_speech
    audio_clip = audio_clip[:int(pred_images_tensor.shape[0] / MOTION_FPS * AUDIO_OUTPUT_SAMPLE_RATE)]
    
    # Write video with synchronized audio
    write_video(
        pred_images_tensor * VIDEO_COLOR_SCALE,
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
        description='Audio-to-Motion Face Generation Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="./ViBES-Face",
        help='Path to the trained model checkpoint directory'
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
        help='User input text prompt for face motion generation'
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
        default=0.2,
        help='Sampling temperature (0.0 = deterministic, higher = more diverse)'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
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

    # Load model weights from checkpoint
    print(f"\n=== Step 3: Load Model Weights from {args.checkpoint} ===")
    load_sharded_checkpoint(base_model, args.checkpoint)

    model = base_model
    model.eval()
    
    # Generate face motion from text
    generate_face_from_text(
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
