import numpy as np
import torch
from typing import List, Dict, Tuple, Union, Optional, Any


def combine_audio_face_tokens(
    audio_tokens: Union[List, np.ndarray, torch.Tensor], 
    face_tokens: Union[List, np.ndarray, torch.Tensor],
    face_token_offset: int = 1024,  # Default offset, assuming audio tokens range from 0-1023
    combination_mode: str = 'concat',  # 'concat' or 'interleave'
    max_length: Optional[int] = None,
    padding_value: int = 0
) -> Union[List, np.ndarray, torch.Tensor]:
    """
    Combine audio tokens and facial expression tokens into a single token sequence for LLM use.
    
    Args:
        audio_tokens: Audio token sequence
        face_tokens: Facial expression token sequence
        face_token_offset: Offset added to facial tokens to avoid overlap with audio tokens
        combination_mode: Combination mode - 'concat' or 'interleave'
        max_length: If specified, the output sequence will be truncated or padded to this length
        padding_value: Value used for padding (if needed)
    
    Returns:
        Combined token sequence
    """
    # Ensure inputs are of the same type
    is_tensor = isinstance(audio_tokens, torch.Tensor)
    is_numpy = isinstance(audio_tokens, np.ndarray)
    
    # Convert to lists for processing
    audio_list = audio_tokens.tolist() if (is_tensor or is_numpy) else list(audio_tokens)
    face_list = face_tokens.tolist() if (isinstance(face_tokens, (torch.Tensor, np.ndarray))) else list(face_tokens)
    
    # Apply offset to facial tokens
    face_list_offset = [token + face_token_offset for token in face_list]
    
    # Combine tokens according to specified mode
    if combination_mode == 'interleave':
        # Interleave audio and facial tokens
        combined = []
        min_len = min(len(audio_list), len(face_list_offset))
        
        for i in range(min_len):
            combined.append(audio_list[i])
            combined.append(face_list_offset[i])
            
        # Add remaining tokens
        if len(audio_list) > min_len:
            combined.extend(audio_list[min_len:])
        if len(face_list_offset) > min_len:
            combined.extend(face_list_offset[min_len:])
    else:  # 'concat' mode
        # Simply concatenate the two sequences
        combined = audio_list + face_list_offset
    
    # Apply maximum length constraint (if specified)
    if max_length is not None:
        if len(combined) > max_length:
            # Truncate
            combined = combined[:max_length]
        elif len(combined) < max_length:
            # Pad
            combined = combined + [padding_value] * (max_length - len(combined))
    
    # Return result in the same type as input
    if is_tensor:
        return torch.tensor(combined, dtype=audio_tokens.dtype, device=audio_tokens.device)
    elif is_numpy:
        return np.array(combined, dtype=audio_tokens.dtype)
    else:
        return combined


def separate_audio_face_tokens(
    combined_tokens: Union[List, np.ndarray, torch.Tensor],
    face_token_offset: int = 1024,  # Same offset used in the combine function
    known_audio_length: Optional[int] = None
) -> Tuple[Union[List, np.ndarray, torch.Tensor], Union[List, np.ndarray, torch.Tensor]]:
    """
    Separate a combined token sequence back into audio tokens and facial expression tokens.
    
    Args:
        combined_tokens: Combined token sequence
        face_token_offset: Offset added to facial tokens
        known_audio_length: If known, can specify the length of audio tokens (for concat mode)
    
    Returns:
        Tuple of (audio_tokens, face_tokens)
    """
    # Ensure input is of the same type
    is_tensor = isinstance(combined_tokens, torch.Tensor)
    is_numpy = isinstance(combined_tokens, np.ndarray)
    
    # Convert to list for processing
    combined_list = combined_tokens.tolist() if (is_tensor or is_numpy) else list(combined_tokens)
    
    # If audio length is known, use it to separate
    if known_audio_length is not None:
        audio_list = combined_list[:known_audio_length]
        face_list_offset = combined_list[known_audio_length:]
    else:
        # Separate based on token values
        audio_list = []
        face_list_offset = []
        
        for token in combined_list:
            if token < face_token_offset:
                audio_list.append(token)
            else:
                face_list_offset.append(token)
    
    # Remove offset from facial tokens
    face_list = [token - face_token_offset for token in face_list_offset]
    
    # Return results in the same type as input
    if is_tensor:
        audio_tokens = torch.tensor(audio_list, dtype=combined_tokens.dtype, device=combined_tokens.device)
        face_tokens = torch.tensor(face_list, dtype=combined_tokens.dtype, device=combined_tokens.device)
    elif is_numpy:
        audio_tokens = np.array(audio_list, dtype=combined_tokens.dtype)
        face_tokens = np.array(face_list, dtype=combined_tokens.dtype)
    else:
        audio_tokens = audio_list
        face_tokens = face_list
        
    return audio_tokens, face_tokens


def prepare_multimodal_tokens_for_lm(
    audio_tokens: Union[List, np.ndarray, torch.Tensor],
    face_tokens: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    max_sequence_length: int = 2048,
    audio_sample_rate: float = 16000.0,
    audio_token_fps: float = 25.0,
    face_token_fps: float = 12.5,
    face_token_offset: int = 1024,
    mode: str = 'concat',
    add_special_tokens: bool = True,
    bos_token_id: int = 1,  # Beginning of sequence token ID
    eos_token_id: int = 2,  # End of sequence token ID
    pad_token_id: int = 0,  # Padding token ID
    inference_mode: bool = False,  # Whether generating for inference
    model_type: str = "",  # Type of model (e.g., "glm", "gpt", "llama")
    special_tokens: Optional[Dict[str, int]] = None  # Dictionary of special token IDs
) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Prepare multimodal token sequences for language model training and inference.
    
    Args:
        audio_tokens: Audio token sequence
        face_tokens: Facial expression token sequence (can be None for inference)
        max_sequence_length: Maximum sequence length
        audio_sample_rate: Audio sampling rate
        audio_token_fps: Frame rate of audio tokens
        face_token_fps: Frame rate of facial tokens
        face_token_offset: Offset added to facial tokens to avoid overlap with audio tokens
        mode: Combination mode - 'concat' or 'interleave'
        add_special_tokens: Whether to add special tokens (BOS/EOS)
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        inference_mode: Whether this is for inference (generation) or training
        model_type: Type of model to format data for (e.g., "glm", "gpt", "llama")
        special_tokens: Dictionary of special token IDs for specific models
    
    Returns:
        Dictionary containing processed token sequences and attention masks
    """
    # Use default special tokens if not provided
    if special_tokens is None:
        special_tokens = {
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id
        }
    else:
        # Ensure required tokens are present
        for key in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if key not in special_tokens:
                special_tokens[key] = locals()[key]  # Use the default value

    # Handle case where face_tokens is None (for inference)
    if face_tokens is None and inference_mode:
        # For inference, we only need to preprocess audio tokens
        # Adjust max length to allow space for generated face tokens
        audio_tokens_max_length = int(max_sequence_length * 0.5)  # Reserve half for face tokens
        
        # Convert to list if needed
        is_tensor = isinstance(audio_tokens, torch.Tensor)
        is_numpy = isinstance(audio_tokens, np.ndarray)
        audio_list = audio_tokens.tolist() if (is_tensor or is_numpy) else list(audio_tokens)
        
        # Truncate audio tokens if necessary
        if len(audio_list) > audio_tokens_max_length:
            audio_list = audio_list[:audio_tokens_max_length]
        
        # Add special tokens
        if add_special_tokens:
            input_ids = [special_tokens["bos_token_id"]] + audio_list + [special_tokens["eos_token_id"]]
        else:
            input_ids = audio_list
            
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Return in appropriate format
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Add model-specific fields
        if model_type.lower() == "glm":
            # For GLM, add position_ids and other required fields
            position_ids = list(range(len(input_ids)))
            result["position_ids"] = position_ids
            
            # Add token_type_ids - 0 for context, 1 for target
            result["token_type_ids"] = [0] * len(input_ids)  # All tokens are context for inference
            
        # Return early for inference mode
        return result
    
    # Adjust time alignment of facial tokens
    # If audio and facial token frame rates differ, we need to adjust facial tokens for alignment
    if face_tokens is not None and audio_token_fps != face_token_fps and len(face_tokens) > 0:
        scale_factor = audio_token_fps / face_token_fps
        
        if scale_factor > 1:
            # Facial tokens need upsampling
            if isinstance(face_tokens, torch.Tensor):
                # Use nearest neighbor interpolation for upsampling
                face_tokens = torch.nn.functional.interpolate(
                    face_tokens.float().unsqueeze(0).unsqueeze(0),
                    size=int(len(face_tokens) * scale_factor),
                    mode='nearest'
                ).squeeze(0).squeeze(0).long()
            else:
                # Simple repetition for numpy arrays or lists
                is_numpy = isinstance(face_tokens, np.ndarray)
                face_list = face_tokens.tolist() if is_numpy else list(face_tokens)
                upsampled = []
                
                for token in face_list:
                    # Repeat each token approximately scale_factor times
                    repeats = int(scale_factor)
                    upsampled.extend([token] * repeats)
                
                # Handle decimal part - randomly decide whether to add extra token
                if scale_factor % 1 > 0:
                    for i, token in enumerate(face_list):
                        if np.random.random() < (scale_factor % 1):
                            insert_pos = min(i * int(scale_factor) + int(scale_factor), len(upsampled))
                            upsampled.insert(insert_pos, token)
                
                # Convert back to original type
                if is_numpy:
                    face_tokens = np.array(upsampled, dtype=face_tokens.dtype)
                else:
                    face_tokens = upsampled
        
        elif scale_factor < 1:
            # Facial tokens need downsampling
            if isinstance(face_tokens, torch.Tensor):
                # Use average pooling for downsampling
                face_tokens = torch.nn.functional.avg_pool1d(
                    face_tokens.float().unsqueeze(0),
                    kernel_size=int(1/scale_factor),
                    stride=int(1/scale_factor)
                ).squeeze(0).long()
            else:
                # Sample from numpy array or list
                is_numpy = isinstance(face_tokens, np.ndarray)
                face_list = face_tokens.tolist() if is_numpy else list(face_tokens)
                downsampled = []
                
                stride = int(1/scale_factor)
                for i in range(0, len(face_list), stride):
                    downsampled.append(face_list[i])
                
                # Convert back to original type
                if is_numpy:
                    face_tokens = np.array(downsampled, dtype=face_tokens.dtype)
                else:
                    face_tokens = downsampled
    
    # Calculate length available for actual tokens
    available_length = max_sequence_length
    if add_special_tokens:
        available_length -= 2  # Subtract space for BOS and EOS tokens
    
    # Combine audio and facial tokens
    if face_tokens is not None:
        combined = combine_audio_face_tokens(
            audio_tokens, face_tokens,
            face_token_offset=face_token_offset,
            combination_mode=mode,
            max_length=available_length
        )
    else:
        # If no face tokens, just use audio tokens
        is_tensor = isinstance(audio_tokens, torch.Tensor)
        is_numpy = isinstance(audio_tokens, np.ndarray)
        combined = audio_tokens.tolist() if (is_tensor or is_numpy) else list(audio_tokens)
        
        # Apply max length constraint
        if len(combined) > available_length:
            combined = combined[:available_length]
    
    # Add special tokens
    if add_special_tokens:
        if isinstance(combined, torch.Tensor):
            device = combined.device
            dtype = combined.dtype
            input_ids = torch.cat([
                torch.tensor([special_tokens["bos_token_id"]], dtype=dtype, device=device),
                combined,
                torch.tensor([special_tokens["eos_token_id"]], dtype=dtype, device=device)
            ])
        elif isinstance(combined, np.ndarray):
            input_ids = np.concatenate([
                np.array([special_tokens["bos_token_id"]], dtype=combined.dtype),
                combined,
                np.array([special_tokens["eos_token_id"]], dtype=combined.dtype)
            ])
        else:
            input_ids = [special_tokens["bos_token_id"]] + combined + [special_tokens["eos_token_id"]]
    else:
        input_ids = combined
    
    # Create attention mask (1 for tokens, 0 for padding)
    if isinstance(input_ids, torch.Tensor):
        attention_mask = torch.ones_like(input_ids)
    elif isinstance(input_ids, np.ndarray):
        attention_mask = np.ones_like(input_ids)
    else:
        attention_mask = [1] * len(input_ids)
    
    # Create result dictionary
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    # Add model-specific fields
    if model_type.lower() == "glm":
        # For GLM, add position_ids and other required fields
        if isinstance(input_ids, torch.Tensor):
            position_ids = torch.arange(len(input_ids), device=input_ids.device)
            token_type_ids = torch.zeros_like(input_ids)  # 0 for all tokens in typical usage
        elif isinstance(input_ids, np.ndarray):
            position_ids = np.arange(len(input_ids))
            token_type_ids = np.zeros_like(input_ids)
        else:
            position_ids = list(range(len(input_ids)))
            token_type_ids = [0] * len(input_ids)
            
        result["position_ids"] = position_ids
        result["token_type_ids"] = token_type_ids
    
    # Return the prepared data
    return result 