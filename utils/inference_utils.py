"""
Inference utilities for model generation and output processing.
Handles modality mask preparation, prompt construction, and other inference helpers.
"""
import torch
from typing import List, Tuple, Optional


def prepare_modality_masks(
    batch_size: int,
    seq_len: int,
    num_modalities: int = 3,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare modality masks and position encoding indices for model generation.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_modalities: Number of modalities (default: 3 for text, audio, motion)
        device: Device to create tensors on
        
    Returns:
        Tuple of (modality_masks, position_encoding_indices)
        - modality_masks: Shape (num_modalities, batch_size, seq_len)
        - position_encoding_indices: Shape (batch_size, seq_len)
        
    Example:
        >>> masks, positions = prepare_modality_masks(1, 128, device=torch.device('cuda'))
        >>> masks.shape
        torch.Size([3, 1, 128])
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize modality masks: [text, audio, motion]
    modality_masks = []
    for i in range(num_modalities):
        if i == 0:
            # First modality (text) is active by default
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        else:
            # Other modalities are inactive initially
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        modality_masks.append(mask)
    
    modality_masks_tensor = torch.stack(modality_masks, dim=0)
    position_encoding_indices = torch.arange(
        seq_len,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)
    
    return modality_masks_tensor, position_encoding_indices


# System message of the instruction-following ("agent") body training data — verbatim from
# preprocess/preprocess_hf_amass_dataset_body.py (text_system_prompt), typos included, so the
# ViBES-Body-Agent checkpoint sees exactly the prompt it was trained on.
AGENT_SYSTEM_MESSAGE = (
    "User will provide you with a text instruction. Do it step by step. First, think about the "
    "instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. "
    "Please follow these steps carefully: 1. Think about the instruction first. 2. Respond in an "
    "interleaved manner: output 13 text tokens followed by 26 audio tokens. 3. In your reply, imagine "
    "that you have a body and are already moving, pretending to perform 'the motion required by the "
    "question. 4. Make sure your answer aligns with both the question and the motion being asked. "
    "5. Remember: the motion is imaginary (pretend), not real. 6. If you describe the motion, use the "
    "first-person perspective (e.g., 'my hand,' 'my body,' 'my movement'). Please reply as if you are "
    "experiencing and expressing the motion yourself."
)


def create_prompt(
    user_text: str,
    system_message: Optional[str] = None,
    agent: bool = False
) -> str:
    """
    Create a formatted prompt for the conversational model.
    
    Args:
        user_text: User's input text
        system_message: Optional custom system message
        agent: Use the instruction-following layout (AGENT_SYSTEM_MESSAGE + the
            ``streaming_transcription_with_performed_motion`` assistant prefix) that the
            ViBES-Body-Agent checkpoint was trained with
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> prompt = create_prompt("Hello, how are you?")
        >>> print(prompt)
        <|system|>
        User will provide you with a text instruction...
    """
    if system_message is None and agent:
        system_message = AGENT_SYSTEM_MESSAGE
    if system_message is None:
        system_message = (
            "User will provide you with a text instruction. "
            "Do it step by step. First, think about the instruction and respond "
            "in a interleaved manner, with 13 text token followed by 26 audio tokens."
        )
    
    prompt = f"<|system|>\n{system_message}"
    prompt += f"<|user|>\n{user_text}"
    if agent:
        prompt += "<|assistant|> streaming_transcription_with_performed_motion\n"
    else:
        prompt += "<|assistant|> streaming_transcription\n"

    return prompt


# ============================================================================
# Audio-to-motion (a2m): condition motion on a given speech clip
# ============================================================================

# Same assistant prefix the body/face training sequences start with (see preprocess/preprocess_hf_*).
A2M_ASSISTANT_PREFIX = "<|assistant|>streaming_transcription\n"
AUDIO_TOKEN_ID_RANGE = (152353, 168735)

# body_part -> (motion-token insertion method, motion-token fps); mirrors ChatGLM generate().
A2M_BODY_PARTS = {
    "body": ("add_body_tokens_to_output", 18.75),
    "upper_lower_genmo": ("add_upper_lower_genmo_tokens_to_output", 12.5),
    "fullbody_genmo": ("add_fullbody_genmo_tokens_to_output", 6.25),
}


def tokenize_audio_file(
    audio_path: str,
    device: torch.device,
    audio_tokenizer_path: str = "THUDM/glm-4-voice-tokenizer"
) -> List[int]:
    """
    Tokenize a speech file into GLM-4-Voice audio tokens (12.5 tokens/s).

    Same procedure as inference/inference_a2m_face_style_control.py: 16 kHz mono, 30 s chunks.
    The Whisper-VQ encoder is freed afterwards to keep GPU memory for the main model.
    """
    import librosa
    from transformers import WhisperFeatureExtractor
    from multimodal_tokenizers.archs.speech_tokenizer.modeling_whisper import WhisperVQEncoder

    whisper_model = WhisperVQEncoder.from_pretrained(audio_tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(audio_tokenizer_path)

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audios = [audio[s:s + 30 * 16000] for s in range(0, len(audio), 30 * 16000)]

    pooling_kernel_size = whisper_model.config.pooling_kernel_size or 1
    stride = (whisper_model.conv1.stride[0] * whisper_model.conv2.stride[0]
              * pooling_kernel_size * feature_extractor.hop_length)

    all_tokens = []
    with torch.no_grad():
        for start in range(0, len(audios), 128):
            features = feature_extractor(
                audios[start:start + 128], sampling_rate=16000,
                return_attention_mask=True, return_tensors="pt",
                device=device, padding="longest", pad_to_multiple_of=stride
            ).to(device=device)
            speech_tokens = whisper_model(**features).quantized_token_ids
            attention_mask = features.attention_mask[:, ::whisper_model.conv1.stride[0] * whisper_model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::pooling_kernel_size]
            for i in range(len(speech_tokens)):
                all_tokens.extend(speech_tokens[i][attention_mask[i].bool()].tolist())

    del whisper_model
    torch.cuda.empty_cache()
    return all_tokens


def load_audio_waveform(audio_path: str, sample_rate: int) -> torch.Tensor:
    """Load a speech file as a mono float tensor [T] at `sample_rate` (used as the video soundtrack)."""
    import librosa
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return torch.from_numpy(audio)


def generate_motion_from_audio_tokens(
    model,
    tokenizer,
    audio_tokens: List[int],
    body_part: str,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    top_p: float = 0.1,
    generation_mode: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Audio-to-motion generation: teacher-force the given audio tokens through Expert 0, then let
    Expert 1 generate the motion tokens interleaved after every 26-token audio group.

    Sequence = A2M_ASSISTANT_PREFIX + audio tokens (the training layout, minus the time-aligned
    transcript), with motion placeholders inserted by the model's own add_*_tokens_to_output.
    Returns (output_ids, output_modality_masks) in the same format as model.generate(), so the
    caller's token extraction / decoding is unchanged.

    generation_mode follows model.generate(): "mask" (default; Expert-0 KV cache fast path, as in
    the face a2m script) or "mask_full" (no cache, explicit full-prefix interleaved attention mask).
    """
    import os
    import sys

    if body_part not in A2M_BODY_PARTS:
        raise ValueError(f"Unsupported body_part for a2m: {body_part}")
    insert_method, motion_fps = A2M_BODY_PARTS[body_part]
    generation_mode = generation_mode or os.environ.get("VIBES_ATTENTION_MODE", "mask")

    prefix_ids = tokenizer(A2M_ASSISTANT_PREFIX, add_special_tokens=False)["input_ids"]
    audio_ids = tokenizer.convert_tokens_to_ids([f"<|audio_{t}|>" for t in audio_tokens])
    if not all(AUDIO_TOKEN_ID_RANGE[0] <= i <= AUDIO_TOKEN_ID_RANGE[1] for i in audio_ids):
        raise ValueError("Audio tokens did not map into the tokenizer's audio-token range")

    n_prefix = len(prefix_ids)
    input_ids = torch.tensor([prefix_ids + audio_ids], dtype=torch.long, device=device)
    modality_masks = torch.zeros(3, 1, input_ids.shape[1], dtype=torch.bool, device=device)
    modality_masks[0, :, :n_prefix] = True   # text
    modality_masks[1, :, n_prefix:] = True   # audio

    input_ids, modality_masks = getattr(model, insert_method)(input_ids, modality_masks)

    # Use the position helper from the model's own (remote-code) module.
    calculate_position_encoding_indices = sys.modules[type(model).__module__].calculate_position_encoding_indices
    position_encoding_indices = torch.tensor(
        calculate_position_encoding_indices(modality_masks, modality_fps={1: 12.5, 2: motion_fps}),
        device=device
    ).unsqueeze(0)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.int64, device=device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        modality_masks=modality_masks,
        position_encoding_indices=position_encoding_indices,
        body_part=body_part,
    )
    print(f"a2m: {len(audio_tokens)} audio tokens ({len(audio_tokens) / 12.5:.1f}s), "
          f"sequence {input_ids.shape[1]} tokens, generation_mode={generation_mode}")
    with torch.no_grad():
        if generation_mode in ("mask_full", "masked_full", "debug_full_mask"):
            return model.generate_second_expert_masked(**gen_kwargs)
        past_key_values = model.generate_first_expert_cache(
            input_ids=input_ids,
            attention_mask=attention_mask,
            modality_masks=modality_masks,
            position_encoding_indices=position_encoding_indices,
        )
        return model.generate_second_expert(past_key_values=past_key_values, **gen_kwargs)

