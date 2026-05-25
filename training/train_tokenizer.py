# import os
# import glob
import torch
import os
import sys

# Setup sys.path before other imports
# Ensure we get the project root directory regardless of where the script is run from
_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))

# Validate that ROOT_DIR is correct (should contain utils directory)
if not os.path.exists(os.path.join(ROOT_DIR, "utils")):
    raise RuntimeError(
        f"Cannot find 'utils' directory in project root. "
        f"Expected at: {os.path.join(ROOT_DIR, 'utils')}. "
        f"Please run this script from the project root or inference directory."
    )

# Force ROOT_DIR to be at the beginning of sys.path (remove and re-insert to ensure priority)
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

# Add conversational_agent directory to sys.path for conver_agent imports
# This enables imports like: from multimodal_tokenizers.archs.lom_vq import ...
# This is needed when using conda activate conver_agent environment
# Path resolution priority:
#   1) CONVERSATIONAL_AGENT_DIR environment variable (if set)
#   2) Relative path from project root (../conversational_agent)
#   3) Default absolute path as fallback
_conversational_agent_dir_env = os.getenv('CONVERSATIONAL_AGENT_DIR')
if _conversational_agent_dir_env and os.path.exists(_conversational_agent_dir_env):
    CONVERSATIONAL_AGENT_DIR = _conversational_agent_dir_env
else:
    # _relative_path = os.path.join(os.path.dirname(ROOT_DIR), 'conversational_agent')
    _relative_path = ROOT_DIR
    CONVERSATIONAL_AGENT_DIR = _relative_path

if os.path.exists(CONVERSATIONAL_AGENT_DIR):
    if CONVERSATIONAL_AGENT_DIR in sys.path:
        sys.path.remove(CONVERSATIONAL_AGENT_DIR)
    sys.path.insert(1, CONVERSATIONAL_AGENT_DIR)  # Insert at position 1, after ROOT_DIR

import pytorch_lightning as pl
from omegaconf import OmegaConf
from multimodal_tokenizers.callback import build_callbacks
from multimodal_tokenizers.config import parse_args, instantiate_from_config
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.logger import create_logger
from multimodal_tokenizers.utils.load_checkpoint import load_pretrained, load_pretrained_vae, load_pretrained_without_vqvae, load_pretrained_vae_face, load_pretrained_vae_upper, load_pretrained_vae_lower, load_pretrained_vae_hand


def main():
    # # Check debug mode
    # debug_tokens = os.environ.get('DEBUG_TOKENS', 'False').lower() in ('true', '1', 't')
    # if debug_tokens:
    #     print("==== TOKEN DEBUG MODE ENABLED ====")
        
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tenosrboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    # model = build_model(cfg, datamodule)
    model = build_model(cfg)
    # if cfg.TRAIN.FORCE_BF16 and cfg.TRAIN.PRECISION == 'bf16':
    #     model.to(torch.bfloat16)  # convert model weight to BF16

    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision=cfg.TRAIN.PRECISION,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
        # num_sanity_val_steps=0
    )
    logger.info("Trainer initialized")

    # Strict load pretrianed model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained_without_vqvae(cfg, model, logger)

    # Strict load vae model
    # if OmegaConf.select(cfg.TRAIN, 'PRETRAINED_VQ') is not None:
    if OmegaConf.select(cfg.TRAIN, 'PRETRAINED_VQ_FACE') is not None:
        load_pretrained_vae_face(cfg, model, logger)
    if OmegaConf.select(cfg.TRAIN, 'PRETRAINED_VQ_UPPER') is not None:
        load_pretrained_vae_upper(cfg, model, logger)
    if OmegaConf.select(cfg.TRAIN, 'PRETRAINED_VQ_LOWER') is not None:
        load_pretrained_vae_lower(cfg, model, logger)
    if OmegaConf.select(cfg.TRAIN, 'PRETRAINED_VQ_HAND') is not None:
        load_pretrained_vae_hand(cfg, model, logger)

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")



if __name__ == "__main__":
    main()
