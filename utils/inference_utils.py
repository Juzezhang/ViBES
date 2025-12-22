"""
Inference utilities for model generation and output processing.
Handles modality mask preparation, prompt construction, and other inference helpers.
"""
import torch
from typing import Tuple, Optional


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


def create_prompt(
    user_text: str,
    system_message: Optional[str] = None
) -> str:
    """
    Create a formatted prompt for the conversational model.
    
    Args:
        user_text: User's input text
        system_message: Optional custom system message
        
    Returns:
        Formatted prompt string
        
    Example:
        >>> prompt = create_prompt("Hello, how are you?")
        >>> print(prompt)
        <|system|>
        User will provide you with a text instruction...
    """
    if system_message is None:
        system_message = (
            "User will provide you with a text instruction. "
            "Do it step by step. First, think about the instruction and respond "
            "in a interleaved manner, with 13 text token followed by 26 audio tokens."
        )
    
    prompt = f"<|system|>\n{system_message}"
    prompt += f"<|user|>\n{user_text}"
    prompt += "<|assistant|> streaming_transcription\n"
    
    return prompt

