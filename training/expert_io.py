"""Per-expert checkpoint I/O for the ViBES Stage-2 MoME model.

The Stage-2 model ``ChatGLMForConditionalGenerationMotExpertNum2`` has two experts:

* **Expert 0** — text/audio. Frozen during training; its weights are *exactly* the
  GLM-4-Voice base (warm-started + ``requires_grad=False``).
* **Expert 1** — motion (the trained "face" / "body" expert).

To avoid re-saving the ~10 GB frozen Expert-0 in every checkpoint (which doubled
checkpoint size), training saves **only Expert-1**. At load time Expert-0 is rebuilt
from the GLM-4-Voice base and merged with the saved Expert-1.

A checkpoint that contains only Expert-1 is marked with ``expert_checkpoint.json`` so
loaders can detect it and merge automatically (falling back to a normal full-checkpoint
load when the marker is absent).
"""

import glob
import json
import os

import torch

# Param-name substrings identifying the trainable motion expert (Expert 1).
# Mirrors the freeze allow-list in train_vibes.py / train_vibes_face_style_control.py.
EXPERT1_MARKERS = (
    "transformer.embedding.word_embeddings.1",
    "local_experts_input_layernorm.1",
    "local_experts_post_attention_layernorm.1",
    "local_experts_mlp.1.dense_h_to_4h",
    "local_experts_mlp.1.dense_4h_to_h",
    "self_attention.query_key_value.1",
    "self_attention.dense.1",
    "transformer.encoder.local_experts_final_layernorm.1",
    "transformer.output_layer.1",
)

EXPERT_MARKER_FILE = "expert_checkpoint.json"


def is_expert1_param(name: str) -> bool:
    """True if a parameter name belongs to the motion expert (Expert 1)."""
    return any(m in name for m in EXPERT1_MARKERS)


def filter_expert1_state_dict(state_dict: dict) -> dict:
    """Keep only the trainable motion-expert (Expert 1) tensors."""
    return {k: v for k, v in state_dict.items() if is_expert1_param(k)}


def glm_base_to_expert0_state_dict(glm_base_path, torch_dtype=torch.bfloat16) -> dict:
    """Load the GLM-4-Voice base and rename its params into the Expert-0 (``.0``) namespace.

    This is the canonical source of the frozen Expert-0 weights (kept identical to it
    throughout training), so it reconstructs Expert-0 losslessly for per-expert loads.
    """
    from transformers import AutoModel

    base = AutoModel.from_pretrained(
        glm_base_path, trust_remote_code=True, torch_dtype=torch_dtype
    )
    out = {}
    for name, param in base.state_dict().items():
        name = name.replace("transformer.embedding.word_embeddings", "transformer.embedding.word_embeddings.0")
        name = name.replace("input_layernorm", "local_experts_input_layernorm.0")
        name = name.replace("post_attention_layernorm", "local_experts_post_attention_layernorm.0")
        name = name.replace("mlp.dense_h_to_4h", "local_experts_mlp.0.dense_h_to_4h")
        name = name.replace("mlp.dense_4h_to_h", "local_experts_mlp.0.dense_4h_to_h")
        name = name.replace("self_attention.query_key_value", "self_attention.query_key_value.0")
        name = name.replace("self_attention.dense", "self_attention.dense.0")
        name = name.replace("transformer.encoder.final_layernorm", "transformer.encoder.local_experts_final_layernorm.0")
        name = name.replace("transformer.output_layer", "transformer.output_layer.0")
        out[name] = param
    del base
    return out


def write_expert_marker(output_dir, base="glm-4-voice-9b"):
    """Drop the ``expert_checkpoint.json`` marker into a per-expert checkpoint dir."""
    with open(os.path.join(output_dir, EXPERT_MARKER_FILE), "w") as f:
        json.dump(
            {
                "expert": "motion",
                "format": "expert1_only",
                "base": base,
                "note": (
                    "Contains ONLY the trained motion expert (Expert 1). Expert 0 "
                    "(text/audio) == the GLM-4-Voice base; reconstruct + merge via "
                    "training/expert_io.py:load_expert1_checkpoint."
                ),
            },
            f,
            indent=2,
        )


def is_expert1_checkpoint(path) -> bool:
    """True if `path` is an Expert-1-only checkpoint (has the marker file)."""
    return os.path.isfile(os.path.join(path, EXPERT_MARKER_FILE))


def load_state_dict_from_dir(ckpt_dir) -> dict:
    """Load a state_dict from a checkpoint dir (safetensors shards or .bin)."""
    sd = {}
    sts = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if sts:
        from safetensors.torch import load_file

        for f in sts:
            sd.update(load_file(f))
    else:
        bins = sorted(glob.glob(os.path.join(ckpt_dir, "pytorch_model*.bin")))
        if not bins:
            raise FileNotFoundError(f"No .safetensors or pytorch_model*.bin in {ckpt_dir}")
        for f in bins:
            sd.update(torch.load(f, map_location="cpu"))
    return sd


def load_expert1_checkpoint(model, expert1_dir, glm_base_path):
    """Load Expert-0 (from the GLM base) + Expert-1 (from `expert1_dir`) into `model`.

    Returns (model, missing_keys, unexpected_keys). Rotary buffers are recomputed by the
    freshly-constructed model, so they are expected to appear in `missing_keys` harmlessly.
    """
    target_dtype = next(model.parameters()).dtype
    e0 = glm_base_to_expert0_state_dict(glm_base_path, torch_dtype=target_dtype)
    e1 = filter_expert1_state_dict(load_state_dict_from_dir(expert1_dir))
    merged = {**e0, **e1}
    missing, unexpected = model.load_state_dict(merged, strict=False)
    return model, missing, unexpected


def make_expert_save_trainer(base="glm-4-voice-9b"):
    """Return a HuggingFace ``Trainer`` subclass that saves ONLY the Expert-1 (motion)
    weights at each checkpoint, instead of the full ~20 GB model.

    HF Trainer passes the (DeepSpeed-gathered) full state_dict into ``_save``; we filter it
    to Expert-1 before delegating to the parent, then drop the marker file. DeepSpeed resume
    shards (``global_step*/``) are written separately and are untouched, so resume still works.
    """
    from transformers import Trainer

    class ExpertSaveTrainer(Trainer):
        def _save(self, output_dir=None, state_dict=None):
            if state_dict is None:
                state_dict = self.model.state_dict()
            state_dict = filter_expert1_state_dict(state_dict)
            super()._save(output_dir=output_dir, state_dict=state_dict)
            out = output_dir if output_dir is not None else self.args.output_dir
            if getattr(self.args, "should_save", True):
                write_expert_marker(out, base=base)

    return ExpertSaveTrainer
