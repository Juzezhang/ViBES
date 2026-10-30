# Evaluation

## Reproducing the Paper

| Paper table | Benchmark | Script | Status |
|---|---|---|---|
| Table 2 | Conversational body motion on Converse3D (Q2M) | `scripts/eval_vibes_body_q2m.py` | Reproduced with the released `ViBES-Body-Agent` ([below](#evaluating-vibes-body-models-paper-table-2)); evaluation set from [`1-data/converse3d_eval.md`](1-data/converse3d_eval.md) |
| Table 4 | Talking head on TFHP (LVE / MOD / FDD) | `evaluation/eval_face_style_control.py` | Reproduced exactly with the paper's style-control face expert ([below](#talking-head-on-tfhp-paper-table-4)) |
| Table 5 | Co-speech gestures on BEAT2 speaker 2 (FGD / BC / Diversity) | `evaluation/eval_beat2_cospeech.py` | Script provided; the speaker-2 model is not released yet ([below](#co-speech-gestures-on-beat2-paper-table-5)) |
| Table 3 | Speech quality | — | Not covered |

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

## Evaluating ViBES Body Models (paper Table 2)

The paper's "Ours" row is the agent body expert, released as `ViBES-Body-Agent`. The script asks
each test question, decodes the generated body tokens to SMPL-X (root translation from the Global
VAE) and scores them with the Q2M evaluator. Generation dominates the cost: about 37 s per question
on one L40S, so the 4921 test questions take roughly 50 GPU-hours. Split them across GPUs.

### Generate + evaluate on one machine

```bash
python scripts/eval_vibes_body_q2m.py \
    --vibes_checkpoint ./ViBES-Body-Agent \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --data_root <CONVERSE3D_EVAL_ROOT> \
    --save_dir eval_results/body_agent \
    --gpus 0,1,2,3,4,5,6,7
```

### Sharded generation (e.g. one SLURM array task per shard)

```bash
# task k of N; finished samples are skipped, so a timed-out task can simply be resubmitted
python scripts/eval_vibes_body_q2m.py \
    --vibes_checkpoint ./ViBES-Body-Agent --glm_base_path ./model_files/glm-4-voice-9b \
    --data_root <CONVERSE3D_EVAL_ROOT> --save_dir eval_results/body_agent \
    --num_shards N --shard_id k

# once every shard is done
python scripts/eval_vibes_body_q2m.py --eval_only \
    --data_root <CONVERSE3D_EVAL_ROOT> --save_dir eval_results/body_agent
```

Metrics are printed and written to `<save_dir>/q2m_results.json`.

| | R1-Bal | R1-Motion | R1-Conv | R3-Bal | MM Dist | FID | Div |
|---|---|---|---|---|---|---|---|
| Paper, Table 2 | 0.467 | 0.308 | 0.745 | 0.671 | 3.178 | 93.9 | 10.73 |
| Released `ViBES-Body-Agent`, regenerated | 0.462 | 0.308 | 0.730 | 0.669 | 3.180 | 94.7 | 10.35 |

- **Sampling:** generation samples (temperature 0.1, top-p 0.1), so reruns move by about the
  confidence interval (±0.004 on R1-Balanced, ±0.01 on R1-Conv).
- **Skipped questions:** about 1.7% of questions (82 of 4921) produce no decodable motion and
  are skipped; the script reports how many samples were scored.
- **Evaluation set:** `<CONVERSE3D_EVAL_ROOT>` holds `test.jsonl` and `motion_segments/test/`
  (4921 questions: 3128 motion-descriptive from AMASS / HumanML3D, 1793 conversational from
  BEAT2). Build it with [`1-data/converse3d_eval.md`](1-data/converse3d_eval.md): download our
  split files, then cut the motion from your own BEAT2 and AMASS.

### Global VAE

The root velocity the evaluator sees comes from the Global VAE (`--global_checkpoint`, default
`model_files/pretrained_cpt/VAE_Global_from_Lower54/last.ckpt`). The paper used an earlier Global
VAE. Decoding the same tokens with either version gives the same scores to three decimals. Each
generated `.npz` also stores the body tokens, so the same generations can be decoded with another
Global VAE without regenerating:

```bash
python scripts/eval_vibes_body_q2m.py --redecode_from eval_results/body_agent \
    --global_checkpoint <other_global_vae.ckpt> \
    --data_root <CONVERSE3D_EVAL_ROOT> --save_dir eval_results/body_agent_other_global
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vibes_checkpoint` | `./ViBES-Body-Agent` | Body checkpoint: Expert-1-only (released) or full |
| `--glm_base_path` | `THUDM/glm-4-voice-9b` | GLM-4-Voice base that rebuilds Expert-0 for Expert-1-only checkpoints |
| `--global_checkpoint` | released Global VAE | Lower-body pose -> root velocity |
| `--agent` | False | Use the agent system prompt (`create_prompt(agent=True)`); the paper numbers use the default prompt |
| `--save_dir` | `eval_results/body_v6` | Directory for generated `.npz` files |
| `--eval_only` | False | Skip generation, evaluate from cache |
| `--redecode_from` | - | Re-decode the tokens saved in this directory into `--save_dir`, then evaluate |
| `--num_shards` / `--shard_id` | 1 / 0 | Generate one shard and exit |
| `--gpus` | `0` | Comma-separated GPU IDs (one generation process per GPU) |
| `--results_json` | `<save_dir>/q2m_results.json` | Where to write the metrics |
| `--data_root` | `<CONVERSE3D_EVAL_ROOT>` | Evaluation set built with [`1-data/converse3d_eval.md`](1-data/converse3d_eval.md) |
| `--seed` | 42 | Random seed |

The Q2M text encoder is the frozen CLIP ViT-L/14@336px; the `clip` package downloads it (~0.9 GB)
on first use.

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

## Talking Head on TFHP (paper Table 4)

LVE / MOD / FDD on the 120 TFHP test sessions, computed on FLAME vertices with ARTalk's metric code
(`evaluation/face_metrics.py`). The face expert gets the test audio and regenerates every face token
with greedy decoding, so a run is deterministic.

**1. Build the test set** (style-control variant, one sequence per test session; inputs as in
[`1-data/tfhp.md`](1-data/tfhp.md)):

```bash
python preprocess/preprocess_hf_tfhp_face_style.py \
    --data_root   <TFHP_ROOT> \
    --output_path <TFHP_ROOT>/processed_tfhp_tokenized_face_style_test \
    --split test --max_seq_length 10000000
```

The large `--max_seq_length` is what the paper uses: each test session stays one sequence (120
records). The default 2048 packs the sessions into 313 chunks and gives different numbers.

**2. Evaluate:**

```bash
python -m evaluation.eval_face_style_control \
    --checkpoint ./ViBES-Face \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --test_dataset <TFHP_ROOT>/processed_tfhp_tokenized_face_style_test/tokenized_dataset \
    --output_dir eval_results/face_tfhp
```

About 55 s per session on one L40S (~1.8 h for the test set). Results go to `eval_results.json`.

| Checkpoint | LVE (mm) ↓ | MOD (mm) ↓ | FDD ↓ |
|---|---|---|---|
| Paper, Table 4 | 10.87 | 2.29 | 6.05 |
| Style-control face expert used in the paper (rerun) | 10.87 | 2.29 | 6.05 |
| Released `ViBES-Face` | 10.11 | 2.13 | 8.30 |

- **The rerun matches the paper exactly:** the JSON values are identical to the original run.
- **The released `ViBES-Face` is a different checkpoint:** it was trained without the style
  prefix, and the table shows it evaluated on the same test set.
- **MOD:** the script prints this column as `MHD`, the root-mean-square vertex distance over the
  whole face.

---

## Co-speech Gestures on BEAT2 (paper Table 5)

The BEAT2 benchmark in the EMAGE setting: speaker 2 ("scott"), 15 test conversations, 30 fps.
The model regenerates every face / body / hand token of a test conversation from its text and
audio. The tokens are decoded with EMAGE's speaker-2 VQ-VAEs and scored with EMAGE's metrics
(`multimodal_tokenizers/metrics/co_speech.py`):

- **FGD:** Fréchet distance between AESKConv latents of the generated and GT 55-joint poses.
- **BC:** beat consistency between audio onsets and upper-body motion.
- **Diversity:** L1 diversity of the generated joints.

```bash
python -m evaluation.eval_beat2_cospeech \
    --checkpoint /path/to/beat2_speaker2_checkpoint \
    --vq_checkpoint /path/to/vq_emage_speaker_2.ckpt \
    --test_dataset /path/to/beat2_speaker2_test/tokenized_dataset \
    --beat2_root <BEAT2_ROOT> \
    --output_dir eval_results/beat2_cospeech
```

- **Inputs:** `<BEAT2_ROOT>` must hold `smplxflame_30/`, `wave16k/` and EMAGE's `weights/`
  (`AESKConv_240_100.bin`, `mean_vel_smplxflame_30.npy`).
- **Not released yet:** the speaker-2 ViBES model, the VQ-VAEs and the tokenized test set.
- **Score saved tokens:** `--from_dir <earlier output_dir>` scores the tokens a previous run
  saved, without generating again.
- **Reading the table:** `paper_scale` in `eval_results.json` gives the numbers in the paper's
  form, where FGD is the raw value and BC is the raw value × 10.
- **Run-to-run variation:** generation samples (temperature 0.2, top-p 0.8), so reruns differ by
  about ±0.05 FGD and ±0.2 BC / Diversity.

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
| `evaluation/eval_face_style_control.py` | TFHP talking-head evaluation (Table 4) |
| `evaluation/face_metrics.py` | LVE / MHD (paper: MOD) / FDD on FLAME vertices |
| `evaluation/eval_beat2_cospeech.py` | BEAT2 speaker-2 co-speech evaluation (Table 5) |
| `multimodal_tokenizers/metrics/co_speech.py` | EMAGE FGD / BC / Diversity |
