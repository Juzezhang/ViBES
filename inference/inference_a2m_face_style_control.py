#!/usr/bin/env python3
"""
Audio-to-Face inference with style control.

Takes an audio file + a style reference (from ARTalk pre-defined styles or custom .pt file)
and generates styled 3D facial motion video. No TTS needed - audio is provided directly.

Available ARTalk styles:
    natural_0~7, happy_0~2, curious_0, angry_0, doubtful_0~1

Usage:
    python -m inference.inference_a2m_face_style_control \
        --audio /path/to/audio.wav \
        --style natural_0 \
        --checkpoint ./ViBES-Face \
        --output_dir ./output

    # Custom style .pt file
    python -m inference.inference_a2m_face_style_control \
        --audio /path/to/audio.wav \
        --style /path/to/custom_style.pt \
        --checkpoint ./ViBES-Face
"""
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# Add speech_related to sys.path for speech_tokenizer imports
speech_related_path = os.path.join(ROOT_DIR, "speech_related")
if speech_related_path not in sys.path:
    sys.path.insert(1, speech_related_path)

import argparse
import torch
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    WhisperFeatureExtractor,
)
from transformers.modeling_utils import load_sharded_checkpoint
from smplx import FLAME

from multimodal_tokenizers.archs.lom_vq import VQVAEConvZeroDSUS1_PaperVersion
from multimodal_tokenizers.utils.rotation_conversions import (
    rotation_6d_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from multimodal_tokenizers.utils.renderer_utils import RenderMesh
from multimodal_tokenizers.utils.utils_videos import write_video
from speech_tokenizer.modeling_whisper import WhisperVQEncoder

# ============================================================================
# Constants
# ============================================================================
FACE_TOKEN_OFFSET = 168736
FACE_FEATURE_DIM = 112
FACE_HEAD_POSE_END_IDX = 6
FACE_JAW_POSE_END_IDX = 12
FACE_EXPRESSION_START_IDX = 12
FLAME_NUM_EXPRESSIONS = 100
FLAME_BATCH_SIZE = 100
MOTION_FPS = 25
AUDIO_OUTPUT_SAMPLE_RATE = 22050
RENDER_IMAGE_SIZE = 512
RENDER_SCALE = 1.0
VIDEO_COLOR_SCALE = 255.0
AUDIO_GROUP_SIZE = 25
FACE_GROUP_SIZE = 51  # 1 begin_of_motion + 50 face

VAE_CHECKPOINT_FACE = os.path.join(ROOT_DIR, 'model_files/pretrained_cpt/face/face.ckpt')
FLAME_MODEL_DIR = os.environ.get(
    "VIBES_FLAME_MODEL_DIR",
    os.path.join(ROOT_DIR, "model_files", "FLAME2020"),
)
DEFAULT_STYLE_DIR = "/path/to/ARTalk/assets/style_motion"


# ============================================================================
# Audio Tokenization (GLM-4-Voice)
# ============================================================================

_resample_buffer = {}


def tokenize_audio(audio_path, whisper_model, feature_extractor, device):
    """
    Tokenize an audio file using GLM-4-Voice whisper tokenizer.

    Args:
        audio_path: path to .wav file
    Returns:
        list of int: audio token IDs
    """
    audio, sample_rate = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Process in 30-second chunks
    audios = []
    time_step = 0
    while time_step * 16000 < audio.shape[0]:
        audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
        audios.append(audio_segment)
        time_step += 30

    pooling_kernel_size = whisper_model.config.pooling_kernel_size or 1
    stride = (whisper_model.conv1.stride[0] * whisper_model.conv2.stride[0]
              * pooling_kernel_size * feature_extractor.hop_length)

    all_tokens = []
    for start in range(0, len(audios), 128):
        features = feature_extractor(
            audios[start:start + 128], sampling_rate=16000,
            return_attention_mask=True, return_tensors="pt",
            device=device, padding="longest", pad_to_multiple_of=stride
        ).to(device=device)
        outputs = whisper_model(**features)
        speech_tokens = outputs.quantized_token_ids
        attention_mask = features.attention_mask[:, ::whisper_model.conv1.stride[0] * whisper_model.conv2.stride[0]]
        attention_mask = attention_mask[:, ::whisper_model.config.pooling_kernel_size]
        for i in range(len(speech_tokens)):
            token = speech_tokens[i][attention_mask[i].bool()].tolist()
            all_tokens.extend(token)

    return all_tokens


# ============================================================================
# Style Loading and Conversion
# ============================================================================

def load_artalk_style(style_name_or_path, style_dir=DEFAULT_STYLE_DIR):
    """
    Load an ARTalk style tensor.

    Args:
        style_name_or_path: style name (e.g. "natural_0") or path to .pt file
        style_dir: directory containing ARTalk style .pt files

    Returns:
        torch.Tensor: (50, 106) style motion tensor
    """
    if os.path.isfile(style_name_or_path):
        path = style_name_or_path
    else:
        path = os.path.join(style_dir, f"{style_name_or_path}.pt")
    if not os.path.exists(path):
        available = [f.replace('.pt', '') for f in os.listdir(style_dir) if f.endswith('.pt')]
        raise FileNotFoundError(
            f"Style '{style_name_or_path}' not found at {path}.\n"
            f"Available styles: {sorted(available)}"
        )
    style = torch.load(path, map_location='cpu', weights_only=True)
    assert style.shape == (50, 106), f"Expected (50, 106), got {style.shape}"
    return style


def artalk_to_vibes_format(artalk_motion):
    """
    Convert ARTalk 106D motion to ViBES 112D format.

    ARTalk: [0:100] expression, [100:103] global orient (axis-angle), [103:106] jaw pose (axis-angle)
    ViBES:  [0:6] head pose (6D rot), [6:12] jaw pose (6D rot), [12:112] expression

    Args:
        artalk_motion: (N, 106) tensor
    Returns:
        (N, 112) tensor in ViBES format
    """
    expression = artalk_motion[:, :100]                    # (N, 100)
    global_orient_aa = artalk_motion[:, 100:103]           # (N, 3) axis-angle
    jaw_pose_aa = artalk_motion[:, 103:106]                # (N, 3) axis-angle

    # Convert axis-angle to 6D rotation
    head_mat = axis_angle_to_matrix(global_orient_aa)      # (N, 3, 3)
    head_6d = matrix_to_rotation_6d(head_mat)              # (N, 6)
    jaw_mat = axis_angle_to_matrix(jaw_pose_aa)            # (N, 3, 3)
    jaw_6d = matrix_to_rotation_6d(jaw_mat)                # (N, 6)

    # ViBES order: head_6d + jaw_6d + expression
    vibes_motion = torch.cat([head_6d, jaw_6d, expression], dim=-1)  # (N, 112)
    return vibes_motion


def encode_style_to_tokens(style_motion_vibes, vae_face, device):
    """
    Encode ViBES-format style motion through VQ encoder to get token IDs.

    Args:
        style_motion_vibes: (50, 112) tensor
        vae_face: VQ-VAE face model
        device: torch device

    Returns:
        list of int: 50 face codebook indices
    """
    with torch.no_grad():
        motion = style_motion_vibes.unsqueeze(0).to(device)  # (1, 50, 112)
        tokens = vae_face.map2index(motion)  # (1, T)
        return tokens.squeeze(0).tolist()


# ============================================================================
# Model Loading
# ============================================================================

def load_vae_face_model(device):
    """Load VAE face encoder/decoder."""
    vae_face = VQVAEConvZeroDSUS1_PaperVersion(
        vae_layer=3, code_num=512, codebook_size=512,
        vae_quantizer_lambda=1, vae_test_dim=FACE_FEATURE_DIM
    )
    state_dict_old = torch.load(VAE_CHECKPOINT_FACE, map_location="cpu", weights_only=False)
    state_dict = {k.replace('vae_face.', ''): v for k, v in state_dict_old.items() if 'vae_face' in k}
    vae_face.load_state_dict(state_dict, strict=True)
    vae_face.eval().to(device)
    return vae_face


# ============================================================================
# Sequence Building
# ============================================================================

def build_sequence(audio_tokens, tokenizer, device):
    """
    Build the full input sequence for the model.

    Structure:
        <|assistant|>streaming_transcription\n
        style_control: <|face_X|> x 50   (style tokens, modality 2)
        [audio_chunk_1 x 25]              (modality 1)
        <|begin_of_motion|> + face x 50   (modality 2, to be generated)
        [audio_chunk_2 x 25]
        ...
        <eos>

    Returns:
        input_ids, modality_masks, position_encoding_indices (all tensors on device)
    """
    # Build sequence with prefix + audio ONLY (no style, no face).
    # Style and face tokens will be inserted later by add_face_tokens_to_output_style_control.
    # This matches the normal generate() flow: first expert processes text+audio,
    # then face tokens are added before second expert runs.

    prefix = "<|assistant|>streaming_transcription\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]

    # Audio token IDs (tokenizer IDs - these go to expert 0)
    audio_tokenizer_ids = []
    for tok in audio_tokens:
        token_str = f"<|audio_{tok}|>"
        ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
        audio_tokenizer_ids.extend(ids)

    # Build sequence: prefix (text) + audio tokens
    input_ids = []
    mod0, mod1, mod2 = [], [], []

    def append_tokens(ids, m0, m1, m2_val):
        input_ids.extend(ids)
        mod0.extend([m0] * len(ids))
        mod1.extend([m1] * len(ids))
        mod2.extend([m2_val] * len(ids))

    # 1. Prefix (text)
    append_tokens(prefix_ids, True, False, False)

    # 2. Audio tokens (all of them, contiguous)
    append_tokens(audio_tokenizer_ids, False, True, False)

    # Convert to tensors
    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    modality_masks = torch.stack([
        torch.tensor([mod0], dtype=torch.bool, device=device),
        torch.tensor([mod1], dtype=torch.bool, device=device),
        torch.tensor([mod2], dtype=torch.bool, device=device),
    ], dim=0)  # (3, 1, seq_len)

    # Simple sequential position encoding for prefix+audio (no face tokens yet)
    position_encoding_indices = torch.arange(
        input_ids_t.shape[1], dtype=torch.float32, device=device
    ).unsqueeze(0)

    return input_ids_t, modality_masks, position_encoding_indices


# ============================================================================
# Main Inference
# ============================================================================

def generate_face_from_audio(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_path = os.path.join(ROOT_DIR, "vibes")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Load model: create from config, then load checkpoint weights
    print("Loading base model from config...")
    config = AutoConfig.from_pretrained(tokenizer_path, trust_remote_code=True)
    config.num_layers = args.layer_num
    model = AutoModel.from_config(
        config, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    load_sharded_checkpoint(model, args.checkpoint)
    model.eval()

    # Load VAE
    print("Loading VAE face model...")
    vae_face = load_vae_face_model(device)

    # Load audio tokenizer
    print("Loading audio tokenizer...")
    whisper_model = WhisperVQEncoder.from_pretrained('THUDM/glm-4-voice-tokenizer').eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained('THUDM/glm-4-voice-tokenizer')

    # Load and encode style
    print(f"Loading style: {args.style}")
    artalk_style = load_artalk_style(args.style, args.style_dir)
    vibes_style = artalk_to_vibes_format(artalk_style)
    style_token_ids = encode_style_to_tokens(vibes_style, vae_face, device)
    print(f"Style encoded to {len(style_token_ids)} tokens")

    # Tokenize audio
    print(f"Tokenizing audio: {args.audio}")
    audio_tokens = tokenize_audio(args.audio, whisper_model, feature_extractor, device)
    print(f"Audio: {len(audio_tokens)} tokens ({len(audio_tokens) / 12.5:.1f}s)")

    # Build prefix + audio sequence
    print("Building input sequence...")
    input_ids, modality_masks, position_encoding_indices = build_sequence(
        audio_tokens, tokenizer, device
    )
    print(f"Prefix + audio sequence: {input_ids.shape[1]} tokens")

    # Insert style + face placeholders using model's built-in method
    print("Inserting style + face tokens...")
    input_ids, modality_masks = model.add_face_tokens_to_output_style_control(
        input_ids, modality_masks, style_token_ids=style_token_ids
    )

    # Recompute position encoding for the full sequence
    from vibes.modeling_chatglm import calculate_position_encoding_indices
    pos_indices = calculate_position_encoding_indices(
        modality_masks, modality_fps={1: 12.5, 2: 25.0}
    )
    position_encoding_indices = torch.tensor(pos_indices, dtype=torch.float32, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, input_ids.shape[1], dtype=torch.int64, device=device)

    # Offset adjust (same as training collator)
    input_ids[input_ids >= FACE_TOKEN_OFFSET] -= FACE_TOKEN_OFFSET

    # Zero out face token positions (model will regenerate via second expert)
    # Keep begin_of_motion (1280) intact
    BOM_RELATIVE_ID = 1280
    mask_condition_2 = (
        modality_masks[2, 0].bool() & (input_ids[0] != BOM_RELATIVE_ID)
    )
    input_ids[:, mask_condition_2] = 0

    print(f"Full sequence: {input_ids.shape[1]} tokens")

    # Stage 1: generate_first_expert_cache → KV cache
    print("Stage 1: Building KV cache...")
    with torch.no_grad():
        past_key_values = model.generate_first_expert_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            modality_masks=modality_masks,
            position_encoding_indices=position_encoding_indices,
        )

    # Stage 2: generate_second_expert → face tokens
    print("Stage 2: Generating face tokens...")
    with torch.no_grad():
        output_ids, output_masks = model.generate_second_expert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            modality_masks=modality_masks,
            position_encoding_indices=position_encoding_indices,
            past_key_values=past_key_values,
            body_part="face",
        )

    # Extract generated face tokens
    # Add offset back for tokenizer decoding
    for i in range(output_ids.shape[0]):
        for j in range(output_ids.shape[1]):
            if output_masks[0][i, j] == False and output_masks[1][i, j] == False:
                output_ids[i, j] += FACE_TOKEN_OFFSET

    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    response_split = full_response.split("<|")
    face_tokens = [int(s.split('_')[1].split('|')[0]) for s in response_split if "face_" in s]

    print(f"Generated {len(face_tokens)} face tokens ({len(face_tokens) * 2 / MOTION_FPS:.1f}s)")

    if len(face_tokens) == 0:
        print("No face tokens generated!")
        return

    # Decode face tokens → FLAME params
    print("Decoding face tokens...")
    face_tensor = torch.tensor(face_tokens, device=device).unsqueeze(0)
    rec_face = vae_face.decode(face_tensor.int())  # (1, n_frames, 112)

    rec_head_pose = rotation_6d_to_axis_angle(rec_face[0, :, :FACE_HEAD_POSE_END_IDX])
    rec_jaw_pose = rotation_6d_to_axis_angle(rec_face[0, :, FACE_HEAD_POSE_END_IDX:FACE_JAW_POSE_END_IDX])
    rec_exp = rec_face[0, :, FACE_EXPRESSION_START_IDX:]
    n_frames = rec_face.shape[1]

    # FLAME mesh generation
    print(f"Generating FLAME mesh for {n_frames} frames...")
    flame_model = FLAME(
        FLAME_MODEL_DIR, num_expression_coeffs=FLAME_NUM_EXPRESSIONS,
        ext='pkl', batch_size=FLAME_BATCH_SIZE
    ).to(device)
    faces = torch.tensor(flame_model.faces.astype(np.int32), dtype=torch.int64)
    mesh_renderer = RenderMesh(image_size=RENDER_IMAGE_SIZE, faces=faces, scale=RENDER_SCALE)

    pred_images = []
    for i in range(0, n_frames, FLAME_BATCH_SIZE):
        bs = min(FLAME_BATCH_SIZE, n_frames - i)
        if bs != FLAME_BATCH_SIZE:
            flame_model = FLAME(
                FLAME_MODEL_DIR, num_expression_coeffs=FLAME_NUM_EXPRESSIONS,
                ext='pkl', batch_size=bs
            ).to(device)

        with torch.no_grad():
            flame_out = flame_model(
                global_orient=rec_head_pose[i:i+bs],  # Use predicted head pose for video rendering
                expression=rec_exp[i:i+bs],
                jaw_pose=rec_jaw_pose[i:i+bs],
                shape=torch.zeros(bs, 100, device=device),
            )
            # Note: for metrics computation, global_orient=zeros is used (see eval script).
            # For video rendering, we keep the predicted head pose for visual quality.
        verts = flame_out['vertices'].detach()
        for v in tqdm(verts, leave=False, desc="Rendering"):
            rgb = mesh_renderer(v[None])[0]
            pred_images.append(rgb.cpu()[0] / VIDEO_COLOR_SCALE)

    # Load input audio for video
    audio_raw, sr = sf.read(args.audio, dtype='float32')
    if audio_raw.ndim > 1:
        audio_raw = audio_raw[:, 0]
    if sr != AUDIO_OUTPUT_SAMPLE_RATE:
        audio_raw = librosa.resample(audio_raw, orig_sr=sr, target_sr=AUDIO_OUTPUT_SAMPLE_RATE)
    audio_clip = torch.tensor(audio_raw, dtype=torch.float32)
    audio_clip = audio_clip[:int(len(pred_images) / MOTION_FPS * AUDIO_OUTPUT_SAMPLE_RATE)]

    # Save video
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    pred_images_tensor = torch.stack(pred_images)

    print(f"Saving video to {output_path}")
    write_video(
        pred_images_tensor * VIDEO_COLOR_SCALE,
        output_path,
        MOTION_FPS,
        audio_clip,
        AUDIO_OUTPUT_SAMPLE_RATE,
        "aac"
    )
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Audio-to-Face with Style Control")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio .wav file")
    parser.add_argument("--style", type=str, default="natural_0",
                        help="Style name (e.g. natural_0, happy_1) or path to .pt file")
    parser.add_argument("--style_dir", type=str, default=DEFAULT_STYLE_DIR,
                        help="Directory containing ARTalk style .pt files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--output_filename", type=str, default="styled_face.mp4", help="Output filename")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--layer_num", type=int, default=40, help="Number of model layers")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling")
    args = parser.parse_args()
    generate_face_from_audio(args)


if __name__ == "__main__":
    main()
