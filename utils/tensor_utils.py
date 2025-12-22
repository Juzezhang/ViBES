"""
Tensor manipulation utilities for inference.
Provides helper functions for tensor operations commonly used in motion reconstruction.
"""
import torch
import numpy as np
from typing import Union


def inverse_selection_tensor(
    filtered_t: torch.Tensor,
    selection_array: np.ndarray,
    n: int
) -> torch.Tensor:
    """
    Inverse selection tensor operation.
    Reconstructs a full tensor from a filtered tensor using a selection mask.
    
    Args:
        filtered_t: Filtered tensor with shape (n, filtered_dim)
        selection_array: Binary numpy array indicating which dimensions to keep
        n: Number of samples/batch size
        
    Returns:
        Reconstructed tensor with shape (n, 165)
        
    Example:
        >>> filtered = torch.randn(10, 50)  # 50 selected dimensions
        >>> selection = np.zeros(165, dtype=int)
        >>> selection[:50] = 1  # First 50 dimensions selected
        >>> result = inverse_selection_tensor(filtered, selection, 10)
        >>> result.shape
        torch.Size([10, 165])
    """
    selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
    original_shape_t = torch.zeros((n, 165), device=filtered_t.device, dtype=filtered_t.dtype)
    selected_indices = torch.where(selection_array == 1)[0]
    
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    
    return original_shape_t


def apply_body_token_offset(
    output_ids: torch.Tensor,
    modality_masks: torch.Tensor,
    body_token_offset: int,
    modality_idx: int = 2
) -> torch.Tensor:
    """
    Apply body token offset to output token IDs based on modality masks.
    
    Args:
        output_ids: Token IDs tensor with shape (batch_size, seq_len)
        modality_masks: Modality masks tensor with shape (num_modalities, batch_size, seq_len)
        body_token_offset: Offset value to add to body tokens
        modality_idx: Index of the body modality in modality_masks (default: 2)
        
    Returns:
        Modified output_ids tensor with offset applied
    """
    output_ids = output_ids.clone()
    for i in range(output_ids.shape[0]):
        for j in range(output_ids.shape[1]):
            if modality_masks[modality_idx][i, j]:
                output_ids[i, j] += body_token_offset
    return output_ids

