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


def _is_state_dict(maybe_state_dict: dict) -> bool:
    if not isinstance(maybe_state_dict, dict):
        return False
    for value in maybe_state_dict.values():
        if isinstance(value, torch.Tensor):
            return True
    return False


def _extract_state_dict(checkpoint: dict, checkpoint_path: str = "") -> dict:
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if _is_state_dict(checkpoint):
            return checkpoint
    raise KeyError(f"Missing state_dict in checkpoint: {checkpoint_path}")


def _load_module_state(module: torch.nn.Module, state_dict: dict, prefix: str) -> None:
    """
    Load a module state dict with prefix stripping + safe fallback.
    """
    extracted = extract_state_dict_keys(state_dict, prefix)
    if extracted:
        # strict=False: module may carry newer zero-init layers (e.g. dataset_embedding from
        # 3-dataset conditioning) absent in older released checkpoints — loads them as a no-op.
        module.load_state_dict(extracted, strict=False)
        return

    module_keys = set(module.state_dict().keys())
    if all(k in module_keys for k in state_dict.keys()):
        # strict=False: the module may carry newer zero-init layers (e.g. dataset_embedding
        # from 3-dataset conditioning) absent in older released checkpoints — load as no-op.
        module.load_state_dict(state_dict, strict=False)
        return

    filtered = {k: v for k, v in state_dict.items() if k in module_keys}
    if not filtered:
        raise RuntimeError(
            f"No matching keys found for module when loading state dict. "
            f"Checkpoint keys (first 5): {list(state_dict.keys())[:5]}, "
            f"Module keys (first 5): {list(module_keys)[:5]}"
        )
    module.load_state_dict(filtered, strict=False)


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
    # Load checkpoints
    checkpoint = torch.load(checkpoint_main, map_location="cpu", weights_only=False)
    checkpoint_face_data = torch.load(checkpoint_face, map_location="cpu", weights_only=False)
    checkpoint_global_data = torch.load(checkpoint_global, map_location="cpu", weights_only=False)

    # Extract state dictionaries
    state_dict_old = _extract_state_dict(checkpoint, checkpoint_main)
    state_dict_face_old = _extract_state_dict(checkpoint_face_data, checkpoint_face)
    state_dict_global_old = _extract_state_dict(checkpoint_global_data, checkpoint_global)

    # Auto-detect the shape-aware ("uncondMeasure") Global VAE: it carries a `beta_proj` layer that
    # projects body measurements into a latent offset. If present, build the net with feed_betas so the
    # weights load and the shape input is used; otherwise the vanilla velocity-only net (default ckpt).
    global_feed_betas = any(k.endswith("beta_proj.weight") for k in state_dict_global_old.keys())
    global_beta_dim = (
        int(state_dict_global_old[next(k for k in state_dict_global_old if k.endswith("beta_proj.weight"))].shape[1])
        if global_feed_betas else 10
    )
    vae_global = VAEConvZero(
        vae_layer=4,
        code_num=256,
        codebook_size=256,
        vae_quantizer_lambda=1,
        vae_test_dim=61,
        feed_betas=global_feed_betas,
        beta_dim=global_beta_dim,
    )
    if global_feed_betas:
        print(f"  Global VAE: shape-aware (beta_dim={global_beta_dim}); pass measurements at inference.")

    
    # Extract and rename state dict keys
    # Load state dictionaries (supports checkpoints with/without state_dict wrapper)
    _load_module_state(vae_face, state_dict_face_old, "vae_face")
    _load_module_state(vae_upper, state_dict_old, "vae_upper")
    _load_module_state(vae_lower, state_dict_old, "vae_lower")
    _load_module_state(vae_hand, state_dict_old, "vae_hand")
    _load_module_state(vae_global, state_dict_global_old, "vae_global")
    
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
