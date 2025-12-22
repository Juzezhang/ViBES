import torch
# from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Any

def load_pretrained(cfg, model, logger=None, phase="train"):
    if logger is not None:
        logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
        
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    # Add OmegaConf classes to safe globals before loading
    add_safe_globals([ListConfig, ContainerMetadata])
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]

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

    new_weights = OrderedDict()
    flag=False
    for k, v in states['model_state'].items():
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
            model.load_state_dict(states['model_state'])
    else:
        model.load_state_dict(states['model_state'])

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
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
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
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
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
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    
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
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    
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
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    
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
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    
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
        state_dict_face = checkpoint_face['state_dict']
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
        state_dict_upper = checkpoint_upper['state_dict']
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
        state_dict_lower = checkpoint_lower['state_dict']
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
        state_dict_hand = checkpoint_hand['state_dict']
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
        state_dict_global = checkpoint_global['state_dict']
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
    state_dict = checkpoint['state_dict']  # Get only the state_dict
    
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
    state_dict = torch.load(save_path,
                            map_location="cpu", weights_only=True)['state_dict']

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
