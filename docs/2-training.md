# Training

ViBES is trained in **two stages**: (1) train per-part VQ-VAE tokenizers, then (2) train the autoregressive SLB (speech-language-behavior) model on top of the discrete token vocabulary.

> 🚧 **Status**: The face expert training pipeline is fully released. The body expert (Stage 2 over GENMO body tokens) is being prepared for release. See the [TODO list](../README.md#-todo-list).

---

## Stage 1 — Tokenizer Training

VQ-VAE tokenizers convert continuous SMPL-X motion into discrete codes. Each part (face / upper / lower / hand / full body) is trained with its own config. All entry points live under `training/train_tokenizer.py`; pass `--nodebug` to actually log and checkpoint (the script defaults to `DEBUG=True` for safety during development).

**Face tokenizer:**

```bash
python -m training.train_tokenizer --cfg configs/config_mixed_stage1_face.yaml --nodebug
```

**Body tokenizer (GenmoFull, 145D motion vector):**

```bash
python -m training.train_tokenizer --cfg configs/config_mixed_stage1_body_genmo.yaml --nodebug
```

**Compositional Lower + Global (71D):**

Includes translation + betas in the lower part.

```bash
python -m training.train_tokenizer \
    --cfg configs/config_mixed_stage1_vq_compositional_lower_global.yaml \
    --nodebug
```

**Global VAE from Lower_54:**

Supervises only the global loss using local velocity.

```bash
python -m training.train_tokenizer \
    --cfg configs/config_mixed_stage1_vae_global_wo_mesh_lr1e-4.yaml \
    --nodebug
```

Render GT pose + reconstructed translation for the global VAE:

```bash
python -m scripts.render_global_vae_translation \
    --cfg configs/config_mixed_stage1_vae_global_wo_mesh_lr1e-4.yaml
```

---

## Stage 2 — SLB Model Training (LLM Expert)

The Stage 2 model is a transformer with a mixture-of-modality-experts (MoME) architecture that consumes interleaved audio / text / motion tokens. Training uses **DeepSpeed** for multi-GPU data parallelism.

### Face Expert with Style Control

The released face-expert entry point is `training/train_vibes_face_style_control.py`. Example 4-GPU launch:

```bash
deepspeed --include localhost:0,1,2,3 --master_port=29507 --master_addr=127.0.0.1 \
    training/train_vibes_face_style_control.py \
    --tokenized_dataset /path/to/processed_tfhp_tokenized_face_style_train/tokenized_dataset \
    --output_dir /path/to/experiments/vibes_face_style_v6 \
    --pretrained_model_path /path/to/vibes_face_v5/checkpoint-52000 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 20000000 \
    --layer_num 40 \
    --save_steps 1000
```

Key arguments:

| Flag | Example | Description |
|---|---|---|
| `--tokenized_dataset` | `.../tokenized_dataset` | HF dataset built from TFHP / Converse3D |
| `--output_dir` | `.../vibes_face_style_v6` | Checkpoint + log directory |
| `--pretrained_model_path` | `.../checkpoint-52000` | Warm start from a prior face checkpoint (drop to train from scratch) |
| `--batch_size` | `8` | Per-GPU batch size |
| `--learning_rate` | `1e-4` | AdamW LR (paper used 1e-4 for style-control fine-tune) |
| `--epochs` | `20000000` | Effectively "until you stop it" — checkpoint by `--save_steps` |
| `--layer_num` | `40` | Number of transformer layers in the MoME backbone |
| `--save_steps` | `1000` | Checkpoint cadence |
| `--save_total_limit` | `5` | Keep last-N rolling checkpoints |
| `--resume_from_checkpoint` | `.../checkpoint-NNNN` | Resume a crashed run |
| `--glm_base_path` | `THUDM/glm-4-voice-9b` | GLM-4-Voice base for the frozen text/audio expert (Expert-0). Also the source reconstructed at load time for per-expert checkpoints |

The DeepSpeed launcher injects `--local_rank` automatically; don't set it by hand.

### Required inputs

1. **Pretrained Stage 1 tokenizers** — face / body VQ-VAE checkpoints from above (or downloaded per [`0-overview.md`](0-overview.md))
2. **Tokenized HuggingFace datasets** — preprocessed conversational data (audio + text + motion tokens) — see the per-dataset guides in [`1-data/`](1-data/)
3. **GLM-4-Voice components** — speech tokenizer/decoder from `./scripts/download_glm4voice_modules.sh`
4. **(Optional) Warm-start checkpoint** — for style-control fine-tuning, pass `--pretrained_model_path` pointing at an earlier face-expert checkpoint

---

## Distributed training

- **Stage 1** uses PyTorch Lightning + DDP / DeepSpeed; multi-GPU is controlled by `Trainer(devices=...)` in the config.
- **Stage 2** is launched directly through the `deepspeed` CLI (see the command above). `--include localhost:0,1,2,3` selects which GPUs participate; `--master_port` should be unique per concurrent run.

## Monitoring

**Stage 1 (Lightning)** — TensorBoard logs under `experiments/<exp_name>/lightning_logs/`:

```bash
tensorboard --logdir experiments/<exp_name>
```

Key metrics to watch for VQ-VAE training:

- `train/recon_loss` — reconstruction quality
- `train/commit_loss` — VQ commitment cost (should decrease then stabilize)
- `val/mpjpe`, `val/pampjpe`, `val/accel` — see [`4-evaluation.md`](4-evaluation.md)
- Codebook perplexity (logged per validation epoch)

**Stage 2 (HF Trainer)** — TensorBoard logs under `<output_dir>/runs/`:

```bash
tensorboard --logdir <output_dir>/runs
```

Key metrics:

- `train/loss` — autoregressive next-token cross-entropy
- `train/learning_rate` — LR schedule
- Per-modality loss components if the script logs them separately

---

## Configuration & Checkpoint Output (reference)

Stage 1 and Stage 2 use different configuration styles:

| Stage | Framework | Config style |
|---|---|---|
| Stage 1 (VQ-VAE) | PyTorch Lightning | YAML — merges `configs/default.yaml` + experiment YAML + `configs/assets.yaml` |
| Stage 2 (LLM expert) | HuggingFace `Trainer` + DeepSpeed | CLI flags only (no YAML); model architecture is selected by `--layer_num` and the pretrained backbone |

**Stage 2 checkpoint output** (HuggingFace `Trainer`):

- Saves every `--save_steps` to `<output_dir>/checkpoint-<step>/`
- Keeps the last `--save_total_limit` rolling checkpoints
- Resume with `--resume_from_checkpoint <output_dir>/checkpoint-<step>`

> **Per-expert checkpoints.** The MoME model has two experts: Expert-0 (text/audio) is *frozen* and
> identical to the GLM-4-Voice base, while Expert-1 (motion) is the only part trained. To avoid
> re-saving the ~19 GB frozen expert in every checkpoint, the gathered model file stores **only the
> motion expert (~0.86 GB)**, marked by `expert_checkpoint.json`. Expert-0 is reconstructed from
> `--glm_base_path` and merged at load time (bit-identical to a full checkpoint). DeepSpeed resume
> shards (`global_step*/`) are untouched, so `--resume_from_checkpoint` works unchanged. The helper
> logic lives in [`training/expert_io.py`](../training/expert_io.py); to shrink an existing full
> checkpoint after the fact, use `scripts/split_expert_checkpoint.py`.

**Stage 1 checkpoint output** (Lightning `ModelCheckpoint`, configured in `multimodal_tokenizers/callback.py`):

- Last-N rolling checkpoints for resume
- Best-val checkpoint by `val_mpjpe` for evaluation

---

## Codebook Diagnostics (reference)

Compute codebook usage statistics (perplexity, dead codes, top-k coverage, commitment loss) to assess VQ-VAE health.

**From model + dataset** (loads the checkpoint, runs the dataloader, computes full loss stats):

```bash
python -m scripts.vq_codebook_stats \
    --cfg configs/config_mixed_stage1_body_genmo.yaml \
    --mode model \
    --split test \
    --topk 0.1,0.05 \
    --max_batches 200 \
    --out_dir experiments/codebook_stats
```

**From saved tokens** (skips the model, reads pre-extracted `.npy` token files only — much faster):

```bash
python -m scripts.vq_codebook_stats \
    --cfg configs/config_mixed_stage1_vq_compositional_lower_global.yaml \
    --mode tokens \
    --topk 0.1,0.05 \
    --out_dir experiments/codebook_stats
```

Optional flags: `--parts face,upper,lower,hand,body`, `--ckpt /path/to/ckpt` (single-part override).

---

## Reconstruction Validation (reference)

Render side-by-side GT vs. reconstructed motion to validate a trained tokenizer:

```bash
python -m scripts.render_genmo_reconstruction \
    --cfg configs/config_mixed_stage1_body_genmo.yaml
```

> ℹ️ **Rendering the *released* full-body GENMO tokenizer.** `config_mixed_stage1_body_genmo.yaml` builds the 145D `VQVAEConvZeroDSUS_PaperVersion` arch (for training that variant from scratch). The **released** checkpoint `model_files/pretrained_cpt/VQVAE_0320_GenmoFull/last.ckpt` is instead the 135D `MotionGPTVQVaeAdapter` (`GENMO_FULL_INCLUDE_BETAS: false`); loading it into the 145D config raises a `state_dict` mismatch. To render/tokenize with the released checkpoint, use its **shipped config** `model_files/pretrained_cpt/VQVAE_0320_GenmoFull/genmo_full.yaml` (matching arch + dims; add a `TEST.RENDER` block for rendering). Also ensure PyTorch3D was built with GPU support (see README) and `ffmpeg-python` is installed.
>
> **135D vs 145D = the 10 SMPL-X `betas` (body shape).** 145D layout is `body_r6d(126) + betas(10) + global_orient_r6d(6) + local_transl_vel(3)`; 135D drops the `betas` block. The **released 135D tokenizer does not model shape** — on decode, betas are filled with zeros (`insert_zero_betas`), so reconstructions use the **neutral SMPL-X body** regardless of the subject's real shape. (The GENMO `.npz` *data* from preprocessing still stores real per-sequence betas at dims 126:136; the 135D tokenizer simply ignores them.)

For the compositional lower_global variant:

```bash
python -m scripts.render_genmo_reconstruction \
    --cfg configs/config_mixed_stage1_vq_compositional_lower_global.yaml
```

Outputs MP4s to `experiments/multimodal_tokenizer/VQVAE_Mixed_Genmo/genmo_recon_videos/`. The renderer uses the same GENMO/SMPL-X conversion as `preprocess/render_genmo_converted_amass.py`.

Optional config knobs: `TEST.NUM_SAMPLES`, `TEST.MAX_SECONDS`, `TEST.RENDER_CHUNK`.

**GENMO translation check** — quick sanity check for the velocity → translation integration:

```bash
python -m scripts.render_genmo_translation_check \
    --cfg configs/config_mixed_stage1_body_genmo.yaml
```

Output: `experiments/genmo_translation_check/<EXP_NAME>/<timestamp>/<split>/<dataset>/`.

---

## Token Formats (reference)

The three released body tokenizer variants (see [`0-overview.md`](0-overview.md) for download paths):

| Variant | Input dim | Codebook | Downsample | Checkpoint |
|---|---|---|---|---|
| **GenmoFull** | 135D (GENMO 145D minus 10D betas) | 256 × 256 | ~4× | `VQVAE_0320_GenmoFull/last.ckpt` |
| **Hybrid Lower_Genmo** | 61D (9 joints × 6D + 3D vel + 4D foot contact) | 256 × 128 | ~4× | `VQVAE_0318_NormalUpper_GenmoLower/vqvar_genmo_lower_global_last.ckpt` |
| **LOM Upper** | 78D (13 upper joints × 6D rotation) | 256 | — | `body/lom_vq.ckpt` |

**Output paths** (one `.npy` per source motion sequence):

```
<DATASET_ROOT>/TOKENS_AGENT_25/
├── upper/{seq_name}.npy           ← LOM Upper
├── lower_genmo/{seq_name}.npy     ← Hybrid Lower_Genmo
└── fullbody_genmo/{seq_name}.npy  ← GenmoFull
```

**GENMO 145D vector layout** (per frame):

```
[0:126]   body_pose_r6d      21 body joints × 6D rotation
[126:136] betas              10 shape parameters
[136:142] global_orient_r6d  6D root rotation
[142:145] local_transl_vel   3D local translation velocity
```

### Reconstruction quality (AMASS test, 50 samples)

| Model | MPJPE (mm) | PA-MPJPE (mm) |
|---|---|---|
| Hybrid (NormalUpper + Lower_Genmo) | 43.30 | 61.86 |
| LOM (upper + lower) | 50.02 | 71.98 |
| GenmoFull (135D) | 59.59 | 74.27 |
