"""
Training script for the Text-Motion CLIP evaluator.

Follows the same pattern as training/train_tokenizer.py.

Usage:
    python training/train_text_motion_clip.py \
        --cfg configs/evaluator/text_motion_clip.yaml --nodebug
"""

import os
import sys

# Setup sys.path before other imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))

if not os.path.exists(os.path.join(ROOT_DIR, "multimodal_tokenizers")):
    raise RuntimeError(
        f"Cannot find 'multimodal_tokenizers' directory in project root. "
        f"Expected at: {os.path.join(ROOT_DIR, 'multimodal_tokenizers')}. "
        f"Please run this script from the project root."
    )

if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import pytorch_lightning as pl
from omegaconf import OmegaConf
from multimodal_tokenizers.callback import build_callbacks
from multimodal_tokenizers.config import parse_args, instantiate_from_config
from multimodal_tokenizers.data.build_data import build_data
from multimodal_tokenizers.models.build_model import build_model
from multimodal_tokenizers.utils.logger import create_logger


def main():
    # Configs
    cfg = parse_args(phase="train")

    # Logger
    logger = create_logger(cfg, phase="train")
    logger.info(OmegaConf.to_yaml(cfg))

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tensorboard' or (
            loggerName == 'wandb'
            and OmegaConf.select(cfg, "LOGGER.WANDB.params.project")
        ):
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format(
        cfg.DATASET.target.split('.')[-1]))
    try:
        datamodule.setup("fit")
        logger.info("Train dataset: %d samples", len(datamodule.train_dataset))
        logger.info("Val dataset: %d samples", len(datamodule.val_dataset))
    except Exception as e:
        logger.warning("Datamodule setup failed before training: %s", e)

    # Model
    model = build_model(cfg)
    logger.info("model {} loaded".format(cfg.model.target))

    # Lightning Trainer
    num_sanity_val_steps = getattr(cfg.LOGGER, "NUM_SANITY_VAL_STEPS", 2)
    gradient_clip_val = getattr(cfg.TRAIN, 'GRADIENT_CLIP_VAL', None)
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        precision=cfg.TRAIN.PRECISION,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        num_sanity_val_steps=num_sanity_val_steps,
        log_every_n_steps=cfg.LOGGER.get("LOG_EVERY_N_STEPS", 50),
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true"
        if len(cfg.DEVICE) > 1 else 'auto',
        gradient_clip_val=gradient_clip_val,
        benchmark=False,
        deterministic=False,
    )
    logger.info("Trainer initialized")

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
