"""
Model loading utilities for inference.
Handles loading of VAE models, SMPLX models, and checkpoint management.
"""
import torch
import os
from typing import Tuple, Optional
import smplx

from multimodal_tokenizers.archs.lom_vq import (
    VQVAEConvZeroDSUS_PaperVersion,
    VQVAEConvZeroDSUS1_PaperVersion,
    VAEConvZero,
)


def extract_state_dict_keys(state_dict: dict, prefix: str) -> dict:
    """
    Extract and rename state_dict keys by removing a prefix.
    
    Args:
        state_dict: State dictionary with prefixed keys
        prefix: Prefix to remove from keys (e.g., 'vae_face')
        
    Returns:
        New state dictionary with prefix removed from keys
        
    Example:
        >>> state_dict = {'vae_face.layer1.weight': ..., 'vae_face.layer2.weight': ...}
        >>> extract_state_dict_keys(state_dict, 'vae_face')
        {'layer1.weight': ..., 'layer2.weight': ...}
    """
    extracted = {}
    prefix_dot = f"{prefix}."
    
    for key, value in state_dict.items():
        if key.startswith(prefix_dot):
            new_key = key.replace(prefix_dot, "")
            extracted[new_key] = value
    
    return extracted


def load_vae_models(
    device: torch.device,
    checkpoint_main: str,
    checkpoint_face: str,
    checkpoint_global: str
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """
    Load and initialize VAE models for face, upper body, lower body, hand, and global motion.
    
    Args:
        device: Device to load models on (e.g., 'cuda' or 'cpu')
        checkpoint_main: Path to main VAE checkpoint file
        checkpoint_face: Path to face VAE checkpoint file
        checkpoint_global: Path to global VAE checkpoint file
        
    Returns:
        Tuple of (vae_face, vae_upper, vae_lower, vae_global, vae_hand) initialized VAE models
        
    Raises:
        FileNotFoundError: If any checkpoint file is not found
        RuntimeError: If model loading fails
    """
    print("Loading VAE models...")
    
    # Validate checkpoint paths
    for path in [checkpoint_main, checkpoint_face, checkpoint_global]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    # Initialize VAE models
    vae_face = VQVAEConvZeroDSUS1_PaperVersion(
        vae_layer=3,
        code_num=512,
        codebook_size=512,
        vae_quantizer_lambda=1,
        vae_test_dim=112
    )
    vae_upper = VQVAEConvZeroDSUS_PaperVersion(
        vae_layer=3,
        code_num=256,
        codebook_size=256,
        vae_quantizer_lambda=1,
        vae_test_dim=78
    )
    vae_lower = VQVAEConvZeroDSUS_PaperVersion(
        vae_layer=3,
        code_num=256,
        codebook_size=256,
        vae_quantizer_lambda=1,
        vae_test_dim=54
    )
    vae_hand = VQVAEConvZeroDSUS_PaperVersion(
        vae_layer=3,
        code_num=256,
        codebook_size=256,
        vae_quantizer_lambda=1,
        vae_test_dim=180
    )
    vae_global = VAEConvZero(
        vae_layer=4,
        code_num=256,
        codebook_size=256,
        vae_quantizer_lambda=1,
        vae_test_dim=61
    )
    
    # Load checkpoints
    checkpoint = torch.load(checkpoint_main, map_location="cpu", weights_only=False)
    checkpoint_face_data = torch.load(checkpoint_face, map_location="cpu", weights_only=False)
    checkpoint_global_data = torch.load(checkpoint_global, map_location="cpu", weights_only=False)
    
    # Extract state dictionaries
    state_dict_old = checkpoint['state_dict']
    state_dict_face_old = checkpoint_face_data['state_dict']
    state_dict_global_old = checkpoint_global_data['state_dict']
    
    # Extract and rename state dict keys
    state_dict_face = extract_state_dict_keys(state_dict_face_old, 'vae_face')
    state_dict_upper = extract_state_dict_keys(state_dict_old, 'vae_upper')
    state_dict_lower = extract_state_dict_keys(state_dict_old, 'vae_lower')
    state_dict_hand = extract_state_dict_keys(state_dict_old, 'vae_hand')
    state_dict_global = extract_state_dict_keys(state_dict_global_old, 'vae_global')
    
    # Load state dictionaries
    vae_face.load_state_dict(state_dict_face, strict=True)
    vae_upper.load_state_dict(state_dict_upper, strict=True)
    vae_lower.load_state_dict(state_dict_lower, strict=True)
    vae_global.load_state_dict(state_dict_global, strict=True)
    vae_hand.load_state_dict(state_dict_hand, strict=True)
    
    # Set to evaluation mode and move to device
    for vae in [vae_face, vae_upper, vae_lower, vae_global, vae_hand]:
        vae.eval()
        vae.to(device)
    
    print("VAE models loaded successfully!")
    return vae_face, vae_upper, vae_lower, vae_global, vae_hand


def load_smplx_model(
    model_dir: str,
    device: torch.device,
    gender: str = 'NEUTRAL_2020',
    num_betas: int = 300,
    num_expression_coeffs: int = 100,
    use_face_contour: bool = False,
    use_pca: bool = False
) -> torch.nn.Module:
    """
    Load and initialize SMPLX body model.
    
    Args:
        model_dir: Directory containing SMPLX model files
        device: Device to load model on
        gender: Gender variant ('NEUTRAL_2020', 'MALE_2020', 'FEMALE_2020')
        num_betas: Number of shape coefficients
        num_expression_coeffs: Number of expression coefficients
        use_face_contour: Whether to use face contour
        use_pca: Whether to use PCA for hand pose
        
    Returns:
        Initialized SMPLX model in evaluation mode
        
    Raises:
        FileNotFoundError: If model directory doesn't exist
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"SMPLX model directory not found: {model_dir}")
    
    smplx_model = smplx.create(
        model_dir,
        model_type='smplx',
        gender=gender,
        use_face_contour=use_face_contour,
        num_betas=num_betas,
        num_expression_coeffs=num_expression_coeffs,
        ext='npz',
        use_pca=use_pca,
    ).to(device).eval()
    
    return smplx_model

