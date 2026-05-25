# Evaluation

## Question-to-Motion (Q2M) CLIP Evaluator

A CLIP-style contrastive model that learns a shared embedding space between text questions and SMPLX body motion sequences, used to compute motion generation evaluation metrics.

### Architecture

- **MotionEncoder**: Transformer encoder (8 layers) with CLS-style query token pooling -> (B, 512) embedding
- **TextEncoder**: Frozen CLIP ViT-L/14 + single trainable Linear projection
- **Loss**: Symmetric cross-entropy with learnable temperature scaling

### Motion Representation (135-dim)

```
[22 body joints x 6D rotation (132)] [root_velocity (3)]
```

22 body joints (indices 0-21): pelvis, left/right hip, spine1, left/right knee, spine2, left/right ankle, spine3, left/right foot, neck, left/right collar, head, left/right shoulder, left/right elbow, left/right wrist.

### Root Velocity Computation

Root velocity is computed in root-relative coordinates:

```python
from multimodal_tokenizers.utils.rotation_conversions import axis_angle_to_matrix

velocity = torch.zeros_like(transl)
velocity[1:] = transl[1:] - transl[:-1]
velocity[0] = velocity[1]

R_inv = axis_angle_to_matrix(global_orient).transpose(-1, -2)
local_vel = torch.einsum('lij,lj->li', R_inv, velocity)
```

### Converting from Axis-Angle

```python
from multimodal_tokenizers.utils.rotation_conversions import axis_angle_to_6d_np
from multimodal_tokenizers.data.converse3d_question_motion import get_local_transl_vel

rot_aa = data["poses"].reshape(T, 55, 3)
rot_6d = axis_angle_to_6d_np(rot_aa)[:, :22, :].reshape(T, 132)
local_vel = get_local_transl_vel(data["trans"], data["poses"][:, :3])
motion = np.concatenate([rot_6d, local_vel], axis=-1)  # (T, 135)
```

### Config and Checkpoint

| Parameter | Value |
|-----------|-------|
| Config | `configs/evaluator/question_motion_clip_v3_body.yaml` |
| Checkpoint | `model_files/pretrained_cpt/evaluator/question_motion_clip_v3_best.ckpt` |
| Input dim | 135 |
| Num joints | 22 (body only) |
| Embed dim | 512 |
| Num layers | 8 |
| Batch size | 64 |

---

## Balanced R-Precision Protocol

### Problem

Standard R-Precision fails with mixed question types. The Converse3D test set contains:

| Type | Source | Example | Correct Match |
|------|--------|---------|---------------|
| **Motion-descriptive** | AMASS | "Show a person walking in place" | Must match its specific paired motion |
| **Conversational** | BEAT2 | "What does happiness mean to you?" | Any conversational gesture is valid |

Conversational gestures are interchangeable, artificially deflating standard R-Precision.

### Balanced Batch Construction

Each batch of R=32:
```
[16 motion-descriptive] + [16 conversational]
```

Per replication:
1. Shuffle both pools independently
2. Create `min(N_motion // 16, N_conv // 16)` balanced batches
3. Remaining motion-descriptive samples form pure motion batches (groups of 32)
4. 20 replications with fresh shuffles, 95% confidence intervals

With Converse3D test set (3128 motion + 1793 conv):
- 112 balanced batches (1792 motion + 1792 conv)
- 41 pure motion batches (1312 motion)

### Type-Aware Matching

**Motion-descriptive (strict 1-to-1):**
```
correct@k = 1  if index i in top-k nearest
```

**Conversational (relaxed):**
```
correct@k = 1  if ANY conversational index in top-k nearest
```

---

## Metric Definitions

### Balanced R-Precision
```
R-Prec@k (Balanced) = (correct_motion + correct_conv) / (N_motion + N_conv)
```

Sample-count-weighted average (~63% motion, ~37% conv since N_motion > N_conv).

### Motion-only R-Precision (strict)
```
R-Prec@k (Motion) = correct_motion / N_motion
```
Random baseline: 1/32 = 3.1%. Good evaluator: >50% top-1.

### Conv-only R-Precision (relaxed)
```
R-Prec@k (Conv) = correct_conv / N_conv
```
Random baseline: 16/32 = 50%. Good evaluator: >95% top-1.

### MM Dist (Matching Score)
```
MM Dist = (1/N) * sum_i dist(text_emb_i, motion_emb_i)
```
Average Euclidean distance between paired embeddings. Lower = better.

### FID
```
FID = ||mu_gen - mu_gt||^2 + Tr(Sigma_gen + Sigma_gt - 2*sqrt(Sigma_gen * Sigma_gt))
```
Distributional distance between generated and GT motion embeddings. Embeddings scaled by `emb_scale=6.0`. Lower = better.

### Diversity
```
Diversity = (1/P) * sum_p ||emb_a - emb_b|| / pair_divisor
```
P=300 random pairs, `emb_scale=6.0`, `pair_divisor=2.0`. Higher = more diverse.

### Replication Protocol
- R-Precision and MM Dist: 20 replications, 95% CI = `1.96 * std / sqrt(20)`
- FID and Diversity: computed once (distribution-level)

---

## GT Baseline

Evaluated on Converse3D test set (4921 samples: 3128 motion-descriptive + 1793 conversational).

| Metric | Balanced | Motion-only | Conv-only |
|--------|----------|-------------|-----------|
| R-Prec Top-1 | 0.7122 +/- 0.0017 | 0.5464 +/- 0.0027 | 0.9993 +/- 0.0003 |
| R-Prec Top-2 | 0.8298 +/- 0.0014 | 0.7316 +/- 0.0021 | 1.0000 +/- 0.0001 |
| R-Prec Top-3 | 0.8834 +/- 0.0013 | 0.8162 +/- 0.0020 | 1.0000 +/- 0.0000 |
| MM Dist | 2.3415 +/- 0.0003 | | |
| FID | 0.0000 | | |
| Diversity | 11.0476 | | |

---

## Evaluating ViBES Body Models

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vibes_body_q2m.py \
    --vibes_checkpoint /path/to/checkpoint \
    --save_dir eval_results/body_v6
```

### Multi-GPU (recommended)

```bash
python scripts/eval_vibes_body_q2m.py \
    --vibes_checkpoint /path/to/checkpoint \
    --save_dir eval_results/body_v6 \
    --gpus 0,1,2,3
```

4 GPUs: ~1.75 hours instead of ~7 hours.

### Evaluate from Cached Results

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vibes_body_q2m.py \
    --eval_only --save_dir eval_results/body_v6
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vibes_checkpoint` | - | ViBES body checkpoint path |
| `--save_dir` | `eval_results/body_v6` | Directory for generated .npz files |
| `--eval_only` | False | Skip generation, evaluate from cache |
| `--gpus` | `0` | Comma-separated GPU IDs |
| `--data_root` | `<CONVERSE3D_EVAL_ROOT>` | Dataset root |
| `--seed` | 42 | Random seed |

Features: resume support, multi-GPU parallel generation, cached evaluation.

---

## Python API

```python
import torch
from omegaconf import OmegaConf
from multimodal_tokenizers.models.text_motion_clip import TextMotionCLIP
from multimodal_tokenizers.metrics.clip_eval import extract_embeddings, evaluate_clip

cfg = OmegaConf.load("configs/evaluator/question_motion_clip_v3_body.yaml")
model = TextMotionCLIP(cfg)
ckpt = torch.load("model_files/pretrained_cpt/evaluator/question_motion_clip_v3_best.ckpt",
                   map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.to("cuda").eval()

# DataLoaders must yield {"text": str, "motion": Tensor(T, 135), "length": int}
results = evaluate_clip(model, test_loader, train_loader, device="cuda",
                        replication_times=20, R_size=32, diversity_times=300,
                        emb_scale=6.0, diversity_pair_divisor=2.0)
```

## Training from Scratch

```bash
CUDA_VISIBLE_DEVICES=0 python training/train_text_motion_clip.py \
    --cfg configs/evaluator/question_motion_clip_v3_body.yaml --nodebug
```

- Peak performance ~epoch 44, then overfitting
- Best checkpoint saved as `best-val-*.ckpt`
- ~10 hours for 150 epochs on 1 GPU

---

## File Reference

| File | Purpose |
|------|---------|
| `multimodal_tokenizers/models/text_motion_clip.py` | TextMotionCLIP model |
| `multimodal_tokenizers/metrics/clip_eval.py` | R-Precision, FID, Diversity, extract_embeddings |
| `multimodal_tokenizers/metrics/utils.py` | FID, Diversity, distance matrix utilities |
| `multimodal_tokenizers/data/converse3d_question_motion.py` | Dataset + DataModule |
| `scripts/eval_question_motion_stratified.py` | Balanced evaluation script |
| `scripts/eval_vibes_body_q2m.py` | ViBES body model evaluation |
| `configs/evaluator/question_motion_clip_v3_body.yaml` | Evaluator config |
| `training/train_text_motion_clip.py` | Training entry point |
