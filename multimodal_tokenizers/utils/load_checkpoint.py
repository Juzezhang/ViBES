import torch
# from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Any


def _is_state_dict(maybe_state_dict):
    if not isinstance(maybe_state_dict, dict):
        return False
    for value in maybe_state_dict.values():
        if isinstance(value, torch.Tensor):
            return True
    return False


def _extract_state_dict(checkpoint, checkpoint_path=""):
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if _is_state_dict(checkpoint):
            return checkpoint
    raise KeyError(f"Missing state_dict in checkpoint: {checkpoint_path}")

def load_pretrained(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(checkpoint, ckpt_path)

    model.load_state_dict(state_dict, strict=False)
    return model


def load_pretrained_debug(cfg, model, logger=None, phase="train"):
    from collections import OrderedDict
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    # Add OmegaConf classes to safe globals before loading
    # add_safe_globals([ListConfig, ContainerMetadata])
    states = torch.load(ckpt_path, weights_only=True, map_location=torch.device('cpu'))
    model_state = states.get('model_state')
    if model_state is None:
        model_state = _extract_state_dict(states, ckpt_path)

    new_weights = OrderedDict()
    flag=False
    for k, v in model_state.items():
        #print(k)
        if "module" not in k:
            break
        else:
            new_weights[k.replace('module','vae')]=v
            flag=True
    if flag:
        try:
            model.load_state_dict(new_weights, strict=False)
        except:
            #print(states['model_state'])
            model.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    return model


def load_pretrained_without_vqvae(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")

    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS

    # Add all required classes to safe globals before loading
    # add_safe_globals([ListConfig, ContainerMetadata, Any])
    #if logger is not None:
    #    logger.info(f"Checkpoint keys: {list(model.keys())}")
    #else:
    #    print(f"Checkpoint keys: {list(model.keys())}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    return model

def load_pretrained_without_vae(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")

    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS

    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(checkpoint, ckpt_path)
    model.load_state_dict(state_dict, strict=False)

    return model

def load_pretrained_vae_face(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train":
        checkpoint_path= cfg.TRAIN.PRETRAINED_VQ_FACE
    elif phase == "token":
        checkpoint_path = cfg.TEST.CHECKPOINTS_FACE
    elif phase == "test":
        checkpoint_path = cfg.TEST.CHECKPOINTS
    
    # Load full checkpoint and extract only the state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    
    # Create new state dict with modified keys
    state_dict_face = {}
    for key, value in state_dict.items():
        if 'vae_face' in key:
            new_key = key.replace('vae_face.', '')
            state_dict_face[new_key] = value
    
    # Save only the modified state_dict
    model.vae_face.load_state_dict(state_dict_face, strict=True)
    logger.info(f"Loaded pretrained VAE model from {checkpoint_path}")

    return model


def load_pretrained_vae_upper(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train":
        checkpoint_path= cfg.TRAIN.PRETRAINED_VQ_UPPER
    elif phase == "token":
        checkpoint_path = cfg.TEST.CHECKPOINTS_UPPER
    elif phase == "test":
        checkpoint_path = cfg.TEST.CHECKPOINTS


    # Load full checkpoint and extract only the state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    
    # Create new state dict with modified keys
    state_dict_upper = {}
    for key, value in state_dict.items():
        if 'vae_upper' in key:
            new_key = key.replace('vae_upper.', '')
            state_dict_upper[new_key] = value
    
    # Save only the modified state_dict
    model.vae_upper.load_state_dict(state_dict_upper, strict=True)
    logger.info(f"Loaded pretrained VAE model from {checkpoint_path}")

    return model


def load_pretrained_vae_lower(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train" or phase == "token":
        checkpoint_path= cfg.TRAIN.PRETRAINED_VQ_LOWER
    elif phase == "test":
        checkpoint_path = cfg.TEST.CHECKPOINTS
    
    # Load full checkpoint and extract only the state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    
    # Create new state dict with modified keys
    state_dict_lower = {}
    for key, value in state_dict.items():
        if 'vae_lower' in key:
            new_key = key.replace('vae_lower.', '')
            state_dict_lower[new_key] = value
    
    # Save only the modified state_dict
    model.vae_lower.load_state_dict(state_dict_lower, strict=True)
    logger.info(f"Loaded pretrained VAE model from {checkpoint_path}")

    return model


def load_pretrained_vae_hand(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train" or phase == "token":
        checkpoint_path= cfg.TRAIN.PRETRAINED_HAND
    elif phase == "test":
        checkpoint_path = cfg.TEST.CHECKPOINTS
    
    # Load full checkpoint and extract only the state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    
    # Create new state dict with modified keys
    state_dict_hand = {}
    for key, value in state_dict.items():
        if 'vae_hand' in key:
            new_key = key.replace('vae_hand.', '')
            state_dict_hand[new_key] = value
    
    # Save only the modified state_dict
    model.vae_hand.load_state_dict(state_dict_hand, strict=True)
    logger.info(f"Loaded pretrained VAE model from {checkpoint_path}")

    return model


def load_pretrained_vae_compositional(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    # checkpoint_path= cfg.TRAIN.PRETRAINED_VQ
    if cfg.TEST.CHECKPOINTS_FACE != '':   
        checkpoint_path_face = cfg.TEST.CHECKPOINTS_FACE
        checkpoint_face = torch.load(checkpoint_path_face, map_location="cpu", weights_only=False)
        state_dict_face = _extract_state_dict(checkpoint_face, checkpoint_path_face)
        # Create new state dict with modified keys
        state_dict_face_new = {}
        for key, value in state_dict_face.items():
            if 'vae_face' in key:
                new_key = key.replace('vae_face.', '')
                state_dict_face_new[new_key] = value
        model.vae_face.load_state_dict(state_dict_face_new, strict=True)
        logger.info(f"Loaded face VAE from {checkpoint_path_face}")

    if cfg.TEST.CHECKPOINTS_UPPER != '':   
        checkpoint_path_upper = cfg.TEST.CHECKPOINTS_UPPER
        checkpoint_upper = torch.load(checkpoint_path_upper, map_location="cpu", weights_only=False)
        state_dict_upper = _extract_state_dict(checkpoint_upper, checkpoint_path_upper)
        # Create new state dict with modified keys
        state_dict_upper_new = {}
        for key, value in state_dict_upper.items():
            if 'vae_upper' in key:
                new_key = key.replace('vae_upper.', '')
                state_dict_upper_new[new_key] = value
        model.vae_upper.load_state_dict(state_dict_upper_new, strict=True)
        logger.info(f"Loaded upper body vqvae from {checkpoint_path_upper}")
    if cfg.TEST.CHECKPOINTS_LOWER != '':   
        checkpoint_path_lower = cfg.TEST.CHECKPOINTS_LOWER
        checkpoint_lower = torch.load(checkpoint_path_lower, map_location="cpu", weights_only=False)
        state_dict_lower = _extract_state_dict(checkpoint_lower, checkpoint_path_lower)
        # Create new state dict with modified keys
        state_dict_lower_new = {}
        for key, value in state_dict_lower.items():
            if 'vae_lower' in key:
                new_key = key.replace('vae_lower.', '')
                state_dict_lower_new[new_key] = value
        model.vae_lower.load_state_dict(state_dict_lower_new, strict=True)
        logger.info(f"Loaded lower body vqvae from {checkpoint_path_lower}")
    if cfg.TEST.CHECKPOINTS_HAND != '':   
        checkpoint_path_hand = cfg.TEST.CHECKPOINTS_HAND
        checkpoint_hand = torch.load(checkpoint_path_hand, map_location="cpu", weights_only=False)
        state_dict_hand = _extract_state_dict(checkpoint_hand, checkpoint_path_hand)
        # Create new state dict with modified keys
        state_dict_hand_new = {}
        for key, value in state_dict_hand.items():
            if 'vae_hands' in key:
                new_key = key.replace('vae_hands.', '')
                state_dict_hand_new[new_key] = value
            elif 'vae_hand' in key:
                new_key = key.replace('vae_hand.', '')
                state_dict_hand_new[new_key] = value
        model.vae_hand.load_state_dict(state_dict_hand_new, strict=True)
        logger.info(f"Loaded hand vqvae from {checkpoint_path_hand}")
    if cfg.TEST.CHECKPOINTS_GLOBAL != '':   
        checkpoint_path_global = cfg.TEST.CHECKPOINTS_GLOBAL
        checkpoint_global = torch.load(checkpoint_path_global, map_location="cpu", weights_only=False)
        state_dict_global = _extract_state_dict(checkpoint_global, checkpoint_path_global)
        # Create new state dict with modified keys
        state_dict_global_new = {}
        for key, value in state_dict_global.items():
            if 'vae_global' in key:
                new_key = key.replace('vae_global.', '')
                state_dict_global_new[new_key] = value
        model.vae_global.load_state_dict(state_dict_global_new, strict=True)
        logger.info(f"Loaded global vqvae from {checkpoint_path_global}")

    return model


def load_pretrained_vae(cfg, model, logger, phase="train"):
    # Load pretrained VAE model
    if phase == "train" or phase == "token":
        checkpoint_path= cfg.TRAIN.PRETRAINED_VQ
    elif phase == "test":
        checkpoint_path = cfg.TEST.CHECKPOINTS
    
    # Load full checkpoint and extract only the state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    
    # Create new state dict with modified keys
    state_dict_face = {}
    for key, value in state_dict.items():
        if 'vae_face' in key:
            new_key = key.replace('vae_face.', '')
            state_dict_face[new_key] = value
    
    state_dict_upper = {}
    for key, value in state_dict.items():
        if 'vae_upper' in key:
            new_key = key.replace('vae_upper.', '')
            state_dict_upper[new_key] = value
    
    state_dict_lower = {}
    for key, value in state_dict.items():
        if 'vae_lower' in key:
            new_key = key.replace('vae_lower.', '')
            state_dict_lower[new_key] = value
    
    state_dict_hand = {}
    for key, value in state_dict.items():
        if 'vae_hand' in key:
            new_key = key.replace('vae_hand.', '')
            state_dict_hand[new_key] = value
    
    state_dict_global = {}
    for key, value in state_dict.items():
        if 'vae_global' in key:
            new_key = key.replace('vae_global.', '')
            state_dict_global[new_key] = value
    
    # Save only the modified state_dict
    if hasattr(model, 'vae_face') and state_dict_face:
        model.vae_face.load_state_dict(state_dict_face, strict=True)
        logger.info(f"Loaded face VAE from {checkpoint_path}")
    
    if hasattr(model, 'vae_upper') and state_dict_upper:
        model.vae_upper.load_state_dict(state_dict_upper, strict=True)
        logger.info(f"Loaded upper body VAE from {checkpoint_path}")
    
    if hasattr(model, 'vae_lower') and state_dict_lower:
        model.vae_lower.load_state_dict(state_dict_lower, strict=True)
        logger.info(f"Loaded lower body VAE from {checkpoint_path}")
    
    if hasattr(model, 'vae_hand') and state_dict_hand:
        model.vae_hand.load_state_dict(state_dict_hand, strict=True)
        logger.info(f"Loaded hand VAE from {checkpoint_path}")
    
    if hasattr(model, 'vae_global') and state_dict_global:
        model.vae_global.load_state_dict(state_dict_global, strict=True)
        logger.info(f"Loaded global VAE from {checkpoint_path}")
    
    logger.info(f"Completed loading pretrained VAE model from {checkpoint_path}")

    return model


def load_pretrained_tokenizer(model, save_path):
    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    checkpoint = torch.load(save_path, map_location="cpu", weights_only=True)
    state_dict = _extract_state_dict(checkpoint, save_path)

    # Extract encoder/decoder
    from collections import OrderedDict
    vae_dict = OrderedDict()
    for k, v in state_dict.items():
        if "motion_vae" in k:
            name = k.replace("motion_vae.", "")
            vae_dict[name] = v
        elif "vae" in k:
            name = k.replace("vae.", "")
            vae_dict[name] = v
    if hasattr(model, 'vae'):
        model.load_state_dict(vae_dict, strict=True)
    else:
        model.load_state_dict(vae_dict, strict=True)

    return model
