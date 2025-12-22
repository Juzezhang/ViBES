"""
Motion reconstruction utilities for inference.
Handles reconstruction of body poses from VAE outputs and token decoding.
"""
import torch
from typing import Tuple

from multimodal_tokenizers.utils.rotation_conversions import (
    rotation_6d_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)
from multimodal_tokenizers.data.mixed_dataset.data_tools import (
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
)
from .tensor_utils import inverse_selection_tensor


def reconstruct_upper_body_pose(
    rec_upper: torch.Tensor,
    batch_size: int,
    num_frames: int,
    device: torch.device
) -> torch.Tensor:
    """
    Reconstruct upper body pose from 6D rotations.
    
    Args:
        rec_upper: Upper body VAE output with shape (batch_size, num_frames, 13*6)
        batch_size: Batch size
        num_frames: Number of frames
        device: Device to perform computations on
        
    Returns:
        Reconstructed upper body pose with shape (batch_size * num_frames, 165)
    """
    rec_pose_upper = rec_upper.reshape(batch_size, num_frames, 13, 6)
    rec_pose_upper = rotation_6d_to_axis_angle(rec_pose_upper).reshape(batch_size * num_frames, 13 * 3)
    rec_pose_upper_recover = inverse_selection_tensor(
        rec_pose_upper.to(device),
        JOINT_MASK_UPPER,
        batch_size * num_frames
    )
    return rec_pose_upper_recover


def reconstruct_lower_body_pose(
    rec_pose_legs: torch.Tensor,
    batch_size: int,
    num_frames: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct lower body pose from 6D rotations.
    
    Args:
        rec_pose_legs: Lower body VAE output with shape (batch_size, num_frames, 9*6)
        batch_size: Batch size
        num_frames: Number of frames
        
    Returns:
        Tuple of (reconstructed_lower_pose, lower_to_global_rotation)
        - reconstructed_lower_pose: Shape (batch_size * num_frames, 165)
        - lower_to_global_rotation: Shape (batch_size, num_frames, 9*6)
    """
    rec_pose_lower = rec_pose_legs.reshape(batch_size, num_frames, 9, 6)
    rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
    rec_lower2global = matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(batch_size, num_frames, 9 * 6)
    rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(batch_size * num_frames, 9 * 3)
    rec_pose_lower_recover = inverse_selection_tensor(
        rec_pose_lower,
        JOINT_MASK_LOWER,
        batch_size * num_frames
    )
    return rec_pose_lower_recover, rec_lower2global


def reconstruct_full_body_pose(
    rec_upper: torch.Tensor,
    rec_lower: torch.Tensor,
    rec_pose_jaw: torch.Tensor,
    batch_size: int,
    num_frames: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct full body pose from individual body parts.
    
    Args:
        rec_upper: Upper body VAE output
        rec_lower: Lower body VAE output (will use first 54 dimensions)
        rec_pose_jaw: Jaw pose tensor with shape (batch_size, num_frames, 6)
        batch_size: Batch size
        num_frames: Number of frames
        device: Device to perform computations on
        
    Returns:
        Tuple of (full_body_pose, lower_to_global_rotation)
        - full_body_pose: Complete pose with shape (batch_size * num_frames, 165)
        - lower_to_global_rotation: Shape (batch_size, num_frames, 9*6)
    """
    rec_pose_upper_recover = reconstruct_upper_body_pose(rec_upper, batch_size, num_frames, device)
    rec_pose_legs = rec_lower[:, :, :54]
    rec_pose_lower_recover, rec_lower2global = reconstruct_lower_body_pose(rec_pose_legs, batch_size, num_frames)
    
    rec_pose_hands = torch.zeros(batch_size * num_frames, 30 * 3, device=device)
    rec_pose_hands_recover = inverse_selection_tensor(rec_pose_hands, JOINT_MASK_HANDS, batch_size * num_frames)
    
    rec_pose_jaw = rec_pose_jaw.reshape(batch_size * num_frames, 6)
    rec_pose_jaw = 0 * rotation_6d_to_axis_angle(rec_pose_jaw).reshape(batch_size * num_frames, 1 * 3)
    
    rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
    rec_pose[:, 66:69] = rec_pose_jaw
    
    return rec_pose, rec_lower2global

