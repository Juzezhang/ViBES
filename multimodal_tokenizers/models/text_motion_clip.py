"""
Text-Motion CLIP Evaluator.

CLIP-style contrastive model that learns a shared embedding space for text descriptions
and motion sequences. Adapted from InterGen's InterCLIP for single-person AMASS data
with full SMPLX parameters (343-dim per frame: 330 6D rotations + 3 translation + 10 betas).

Reference: text-to-motion (train_tex_mot_match.py) for high-level evaluator training idea;
InterGen (evaluator_models.py) for CLIP architecture with learnable temperature.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

import numpy as np
from pytorch_lightning import LightningModule
from multimodal_tokenizers.config import get_obj_from_str


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """InterGen-style cosine schedule with linear warmup."""

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1) * 1.0 / self.warmup
        return lr_factor


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class MotionEncoder(nn.Module):
    """Transformer encoder for motion sequences with CLS-style query token pooling."""

    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.MODEL.INPUT_DIM
        self.latent_dim = cfg.MODEL.LATENT_DIM
        self.ff_size = cfg.MODEL.FF_SIZE
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.num_heads = cfg.MODEL.NUM_HEADS
        self.dropout = cfg.MODEL.DROPOUT
        self.activation = cfg.MODEL.ACTIVATION
        self.embed_dim = cfg.MODEL.EMBED_DIM

        self.embed_motion = nn.Linear(self.input_dim, self.latent_dim)
        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, self.dropout, max_len=2000
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, self.embed_dim)

    def forward(self, motion, mask):
        B, T, D = motion.shape
        x_emb = self.embed_motion(motion)

        query = self.query_token[
            torch.zeros(B, dtype=torch.long, device=motion.device)
        ][:, None]
        emb = torch.cat([query, x_emb], dim=1)

        token_mask = torch.ones((B, 1), dtype=torch.bool, device=motion.device)
        valid_mask = torch.cat([token_mask, mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        return self.out(h[:, 0])


class TextEncoder(nn.Module):
    """InterGen-style text encoder: frozen CLIP token embeddings -> trainable Transformer."""

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.MODEL.EMBED_DIM

        clip_model, _ = clip.load(
            cfg.MODEL.get("CLIP_MODEL", "ViT-L/14@336px"), device="cpu", jit=False
        )
        # Frozen CLIP token + positional embeddings
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        for p in self.token_embedding.parameters():
            p.requires_grad = False

        # Trainable 8-layer Transformer on top of CLIP token embeddings
        text_trans_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=cfg.MODEL.NUM_HEADS,
            dim_feedforward=cfg.MODEL.FF_SIZE,
            dropout=cfg.MODEL.DROPOUT,
            activation=cfg.MODEL.ACTIVATION,
            batch_first=True,
        )
        self.text_transformer = nn.TransformerEncoder(
            text_trans_layer, num_layers=cfg.MODEL.NUM_LAYERS
        )
        self.text_ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, self.embed_dim)

    def forward(self, raw_text, device):
        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)
            pe_tokens = x + self.positional_embedding.type(self.dtype)

        out = self.text_transformer(pe_tokens.float())
        out = self.text_ln(out)
        # Extract at [EOS] token position (same as InterGen)
        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return self.out(out)


class LightweightTextEncoder(nn.Module):
    """Full frozen CLIP text encoder + single trainable projection layer."""

    def __init__(self, cfg):
        super().__init__()
        clip_model, _ = clip.load(
            cfg.MODEL.get("CLIP_MODEL", "ViT-L/14@336px"), device="cpu", jit=False
        )
        # Freeze everything and save dtype before deleting visual
        for p in clip_model.parameters():
            p.requires_grad = False
        self._clip_dtype = clip_model.dtype
        self.clip_model = clip_model
        # Delete visual encoder to save memory (breaks clip_model.dtype property)
        del self.clip_model.visual

        # Single trainable projection: CLIP hidden dim -> embed_dim
        clip_dim = clip_model.ln_final.weight.shape[0]  # 768 for ViT-L/14
        self.projection = nn.Linear(clip_dim, cfg.MODEL.EMBED_DIM)

    def forward(self, raw_text, device):
        with torch.no_grad():
            tokens = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.clip_model.token_embedding(tokens).type(self._clip_dtype)
            x = x + self.clip_model.positional_embedding.type(self._clip_dtype)
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x)
            # Extract at EOS token position (before CLIP's text_projection)
            text_features = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)].float()
        return self.projection(text_features)


class TextMotionCLIP(LightningModule):
    """
    Text-Motion CLIP evaluator.

    Learns a shared embedding space for text and motion using symmetric
    contrastive loss with learnable temperature scaling.

    Follows SAMPA's BaseModel epoch-based logging pattern:
    losses are accumulated during steps and logged at epoch end.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"], logger=False)

        # Encoders
        self.motion_encoder = MotionEncoder(cfg)

        text_encoder_type = cfg.MODEL.get('TEXT_ENCODER_TYPE', 'transformer')
        if text_encoder_type == 'lightweight':
            self.text_encoder = LightweightTextEncoder(cfg)
        else:
            self.text_encoder = TextEncoder(cfg)

        # Load pre-trained motion encoder weights if specified
        pretrained_encoder = getattr(cfg.TRAIN, 'PRETRAINED_ENCODER', '')
        if pretrained_encoder:
            self._load_pretrained_motion_encoder(pretrained_encoder)

        # Learnable temperature scale (squared in logit computation)
        self.latent_scale = nn.Parameter(torch.tensor([1.0]))

        # Loss function
        self.loss_ce = nn.CrossEntropyLoss()

        # Epoch-level loss accumulators (registered as buffers for device sync)
        for split in ["train", "val"]:
            self.register_buffer(f"_loss_total_{split}", torch.tensor(0.0))
            self.register_buffer(f"_loss_m2t_{split}", torch.tensor(0.0))
            self.register_buffer(f"_loss_t2m_{split}", torch.tensor(0.0))
            self.register_buffer(f"_count_{split}", torch.tensor(0.0))

    def _reset_accumulators(self, split):
        getattr(self, f"_loss_total_{split}").zero_()
        getattr(self, f"_loss_m2t_{split}").zero_()
        getattr(self, f"_loss_t2m_{split}").zero_()
        getattr(self, f"_count_{split}").zero_()

    def generate_src_mask(self, T, lengths):
        """Generate padding mask: True for valid positions."""
        return torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    def encode_motion(self, motion, lengths):
        """Encode motion and return L2-normalized, scaled embeddings."""
        B, T, _ = motion.shape
        mask = self.generate_src_mask(T, lengths)
        motion_emb = self.motion_encoder(motion, mask)
        return motion_emb / motion_emb.norm(dim=-1, keepdim=True) * self.latent_scale

    def encode_text(self, raw_text, device):
        """Encode text and return L2-normalized, scaled embeddings."""
        text_emb = self.text_encoder(raw_text, device)
        return text_emb / text_emb.norm(dim=-1, keepdim=True) * self.latent_scale

    def forward(self, text, motion, mask):
        motion_emb = self.motion_encoder(motion, mask)
        text_emb = self.text_encoder(text, motion.device)
        return motion_emb, text_emb

    def compute_loss(self, motion_emb, text_emb):
        """Symmetric CLIP contrastive loss."""
        motion_norm = F.normalize(motion_emb, dim=-1)
        text_norm = F.normalize(text_emb, dim=-1)

        logit_scale = self.latent_scale ** 2
        logits_per_motion = logit_scale * motion_norm @ text_norm.t()
        logits_per_text = logits_per_motion.t()

        B = motion_emb.shape[0]
        labels = torch.arange(B, dtype=torch.long, device=motion_emb.device)
        loss_m2t = self.loss_ce(logits_per_motion, labels)
        loss_t2m = self.loss_ce(logits_per_text, labels)
        loss = (loss_m2t + loss_t2m) / 2.0

        return loss, loss_m2t, loss_t2m

    def allsplit_step(self, split, batch, batch_idx):
        text, motion, lengths = batch["text"], batch["motion"], batch["length"]

        B, T, _ = motion.shape
        mask = torch.arange(T, device=motion.device).unsqueeze(0) < lengths.unsqueeze(1)

        motion_emb, text_emb = self(text, motion, mask)
        loss, loss_m2t, loss_t2m = self.compute_loss(motion_emb, text_emb)

        # Accumulate losses for epoch-level logging
        getattr(self, f"_loss_total_{split}").add_(loss.detach())
        getattr(self, f"_loss_m2t_{split}").add_(loss_m2t.detach())
        getattr(self, f"_loss_t2m_{split}").add_(loss_t2m.detach())
        getattr(self, f"_count_{split}").add_(1)

        return loss

    def training_step(self, batch, batch_idx):
        return self.allsplit_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.allsplit_step("val", batch, batch_idx)

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        dico = self._loss_log_dict("train")
        dico["epoch"] = float(self.trainer.current_epoch)
        dico["latent_scale"] = self.latent_scale.item()
        self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._reset_accumulators("val")
            return

        dico = {}
        dico["epoch"] = float(self.trainer.current_epoch)

        # Log train losses if available
        if self._count_train.item() > 0:
            dico.update(self._loss_log_dict("train"))

        # Log val losses
        if self._count_val.item() > 0:
            dico.update(self._loss_log_dict("val"))

        # Alias for build_callbacks checkpoint monitor (expects val_mpjpe, lower=better)
        if "loss/total/val" in dico:
            dico["val_mpjpe"] = dico["loss/total/val"]

        self.log_dict(dico, sync_dist=True, rank_zero_only=True)

    def _loss_log_dict(self, split):
        count = getattr(self, f"_count_{split}")
        if count.item() == 0:
            return {}
        dico = {
            f"loss/total/{split}": (
                getattr(self, f"_loss_total_{split}") / count
            ).item(),
            f"loss/m2t/{split}": (
                getattr(self, f"_loss_m2t_{split}") / count
            ).item(),
            f"loss/t2m/{split}": (
                getattr(self, f"_loss_t2m_{split}") / count
            ).item(),
        }
        self._reset_accumulators(split)
        return dico

    def _load_pretrained_motion_encoder(self, ckpt_path):
        """Load motion encoder weights from a MotionAutoEncoder checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # Extract keys with 'motion_encoder.' prefix
        prefix = "motion_encoder."
        encoder_state = {
            k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        missing, unexpected = self.motion_encoder.load_state_dict(encoder_state, strict=True)
        print(f"[TextMotionCLIP] Loaded pre-trained motion encoder from {ckpt_path}")
        print(f"  Loaded {len(encoder_state)} keys, missing={missing}, unexpected={unexpected}")

    def configure_optimizers(self):
        # Follow BaseModel pattern: generic, config-driven
        optim_target = self.cfg.TRAIN.OPTIM.target
        if len(optim_target.split('.')) == 1:
            optim_target = 'torch.optim.' + optim_target
        optimizer = get_obj_from_str(optim_target)(
            params=[p for p in self.parameters() if p.requires_grad],
            **self.cfg.TRAIN.OPTIM.params,
        )

        scheduler_target = self.cfg.TRAIN.LR_SCHEDULER.target
        sched_params = dict(self.cfg.TRAIN.LR_SCHEDULER.params)
        if scheduler_target == 'CosineWarmup':
            # InterGen-style: linear warmup + cosine decay
            lr_scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup=sched_params['warmup_epochs'],
                max_iters=sched_params['max_epochs'],
            )
        else:
            if len(scheduler_target.split('.')) == 1:
                scheduler_target = 'torch.optim.lr_scheduler.' + scheduler_target
            lr_scheduler = get_obj_from_str(scheduler_target)(
                optimizer=optimizer, **sched_params
            )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
