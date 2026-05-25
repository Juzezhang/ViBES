# BEAT2 Preprocessing

## Step 1 — Download BEAT2

BEAT2 is a co-speech gesture dataset hosted at [H-Liu1997/BEAT2 on Hugging Face](https://huggingface.co/datasets/H-Liu1997/BEAT2). We only use the English portion.

```bash
huggingface-cli download H-Liu1997/BEAT2 \
    --repo-type dataset \
    --include "beat_english_v2.0.0/*" \
    --local-dir <BEAT2_PARENT_ROOT>
```

The `--include "beat_english_v2.0.0/*"` pattern recursively matches everything under the English subset.

After download, you'll get the following structure (everything below comes straight from the HF release):

```
<BEAT2_PARENT_ROOT>/
└── beat_english_v2.0.0/         ← we'll call this <BEAT2_ROOT> from here on
    ├── smplxflame_30/           (raw SMPL-X data at 30 fps — Steps 2-4 use this)
    ├── wave16k/                 (audio at 16 kHz)
    ├── textgrid/                (word/phoneme-level alignment)
    ├── sem/                     (semantic annotations)
    ├── weights/                 (pretrained evaluator weights, e.g. mean_vel_smplxflame_30.npy)
    ├── train_test_split.csv     (train/val/test split — used by downstream loaders)
    └── readme.md                (BEAT2 author's own readme)
```

## Step 2 — Generate the mirrored version

Mirror the SMPL-X motion (horizontal augmentation):

```bash
python preprocess/mirror_motion_beat2.py \
    --smplx_path ./model_files/smplx_models \
    --dataset_path_original <BEAT2_ROOT>/smplxflame_30 \
    --dataset_path_processed <BEAT2_ROOT>/smplxflame_30_mirror
```

## Step 3 — Convert frame rate from 30 fps to 25 fps

BEAT2 ships at 30 fps; ViBES uses 25 fps. Run the converter on **both** the original and the mirrored folders:

```bash
# Original
python preprocess/beat2_motion_fps_converter.py \
    --motion_folder <BEAT2_ROOT>/smplxflame_30 \
    --output_dir   <BEAT2_ROOT>/smplxflame_25

# Mirror
python preprocess/beat2_motion_fps_converter.py \
    --motion_folder <BEAT2_ROOT>/smplxflame_30_mirror \
    --output_dir   <BEAT2_ROOT>/smplxflame_25_mirror
```

## Step 4 — Convert to GENMO 145D format

```bash
python preprocess/dataset_process_beat2_genmo.py \
    --input_dir  <BEAT2_ROOT> \
    --output_dir <BEAT2_ROOT>/beat2_genmo_25
```

Reads `<BEAT2_ROOT>/smplxflame_25/` and `<BEAT2_ROOT>/smplxflame_25_mirror/`, writes GENMO 145D `.npz` files into `<BEAT2_ROOT>/beat2_genmo_25/smplxflame_25/` and `<BEAT2_ROOT>/beat2_genmo_25/smplxflame_25_mirror/` respectively.

**Optional flags:**

- `--max_files_per_subdir N` — process only the first N files per subfolder (useful for quick smoke tests)
- `--subdirs smplxflame_25` — only process the original (skip the mirror), or vice versa

After this step, your `<BEAT2_ROOT>/` should look like:

```
<BEAT2_ROOT>/
├── smplxflame_30/             (source, 30 fps — from HF)
├── smplxflame_30_mirror/      (mirrored source, 30 fps — Step 2 output)
├── smplxflame_25/             (downsampled, 25 fps — Step 3 output)
├── smplxflame_25_mirror/      (downsampled mirror, 25 fps — Step 3 output)
├── beat2_genmo_25/            (Step 4 output)
│   ├── smplxflame_25/         (GENMO 145D, original)
│   └── smplxflame_25_mirror/  (GENMO 145D, mirror)
├── wave16k/                   (from HF)
├── textgrid/                  (from HF)
├── sem/                       (from HF)
├── weights/                   (from HF)
├── train_test_split.csv       (from HF — used by downstream loaders)
└── readme.md                  (from HF)
```

## Step 5 (optional) — Generate foot contacts

The Hybrid Lower tokenizer (61D) and several downstream loaders consume a `foot_contacts_25/` folder (per-frame foot-ground contact labels, shape `(T, 4)`). It's not in the HF release — generate it with the parts-format script:

```bash
python preprocess/dataset_process_beat2_parts.py \
    --input_dir  <BEAT2_ROOT> \
    --output_dir <BEAT2_ROOT>/beat2_parts_25 \
    --smplx_path ./model_files/smplx_models
```

This produces both the BEAT2 parts format and the per-clip foot contacts. Mirror files are prefixed `M_` (e.g. `M_10_kieks_0_103_103.npy`). You only need to run this step if you plan to train the Hybrid Lower VQ-VAE or use any loader that reads `foot_contacts_25/`.

## Step 6 — Tokenize audio with GLM-4-Voice

```bash
PYTHONPATH=./speech_related python preprocess/scripts/get_audio_code_glm.py \
    --wav_folder <BEAT2_ROOT>/wave16k \
    --output_dir <BEAT2_ROOT>/audios_token_glm
```

Produces `<BEAT2_ROOT>/audios_token_glm/<clip>.npy` — one `(N,) int64` array of GLM-4-Voice discrete audio tokens per `.wav`. Make sure you've already run `./scripts/download_glm4voice_modules.sh` from the README setup.

## Step 7 — Tokenize motion with the body VQ-VAE

This step needs a trained body VQ-VAE tokenizer. You can either:

- **Use our pretrained tokenizer**, downloaded as part of [`docs/0-overview.md`](../0-overview.md) (see the `model_files/pretrained_cpt/` table), or
- **Train your own** by following the Stage 1 instructions in [`docs/2-training.md`](../2-training.md).

Edit `configs/config_mixed_stage1_vq_compositional.yaml` (or your variant) so that:
- `cfg.DATASET.BEAT2.ROOT` points at `<BEAT2_ROOT>`
- The checkpoint path under each VAE block points at the tokenizer you're using

Then extract motion tokens:

```bash
python -m preprocess.scripts.get_compositional_motion_code \
    --cfg configs/<your-vq-compositional-config>.yaml
```

`get_compositional_motion_code.py` is the single, config-driven tokenizer for every dataset — which dataset(s) to process comes from `cfg.DATASET.datasets`, and which body parts to tokenize from `cfg.DATASET.MODALITIES[<dataset>]`.

Produces `<BEAT2_ROOT>/TOKENS_AGENT_25/{face,hand,upper,lower,fullbody_genmo,lower_genmo}/<clip>.npy` — one `(T,) int64` array of motion token IDs per source motion file, per body part.

## Step 8 — Build the HuggingFace dataset

Three preprocessing variants are available (run any subset — they write to different `--output_path` and share the same `<BEAT2_ROOT>` inputs):

### 8a — Hybrid Upper + Lower (GENMO)

Reads `TOKENS_AGENT_25/upper/` + `TOKENS_AGENT_25/lower_genmo/` and merges them into a single sequence per clip:

```bash
python preprocess/preprocess_hf_beat2_dataset_body_upper_lower_genmo.py \
    --data_root   <BEAT2_ROOT> \
    --output_path <BEAT2_ROOT>/preprocess_hf_beat2_dataset_body_upper_lower_genmo_train \
    --split train

python preprocess/preprocess_hf_beat2_dataset_body_upper_lower_genmo.py \
    --data_root   <BEAT2_ROOT> \
    --output_path <BEAT2_ROOT>/preprocess_hf_beat2_dataset_body_upper_lower_genmo_test \
    --split test
```

### 8b — Single-stream Full body (GENMO 135D)

Reads `TOKENS_AGENT_25/fullbody_genmo/`:

```bash
python preprocess/preprocess_hf_beat2_dataset_body_fullbody_genmo.py \
    --data_root   <BEAT2_ROOT> \
    --output_path <BEAT2_ROOT>/preprocess_hf_beat2_dataset_body_fullbody_genmo_train \
    --split train

python preprocess/preprocess_hf_beat2_dataset_body_fullbody_genmo.py \
    --data_root   <BEAT2_ROOT> \
    --output_path <BEAT2_ROOT>/preprocess_hf_beat2_dataset_body_fullbody_genmo_test \
    --split test
```

### 8c — Body (interleaved upper / lower / hand, supports motion variants)

The recommended path — reads `TOKENS_AGENT_25/{upper,lower,hand}/`, interleaves them per audio group, and supports a `--motion_variant` flag to pick which body parts to encode:

```bash
python preprocess/preprocess_hf_beat2_dataset_body.py \
    --data_root        <BEAT2_ROOT> \
    --output_path      <BEAT2_ROOT>/preprocess_hf_beat2_dataset_body_train \
    --split            train \
    --motion_variant   body_only
```

| `--motion_variant` | Per-group composition | Motion FPS |
|---|---|---|
| `body_only` (default) | 13 upper + 13 lower + 13 hand | 18.75 |
| `upper_hand` | 13 upper + 13 hand (no lower) | 12.5 |
| `lower_only` | 13 lower (no upper / hand) | 6.25 |

Padded positions are correctly masked as `-100` (no loss); `begin_of_motion` is treated as a structural separator (also `-100`, not supervised).

Training instructions for the body expert live in [`../2-training.md`](../2-training.md).

After Step 8, your `<BEAT2_ROOT>` should look like:

```
<BEAT2_ROOT>/
├── smplxflame_30/ smplxflame_30_mirror/         (from HF + Step 2)
├── smplxflame_25/ smplxflame_25_mirror/         (Step 3)
├── beat2_genmo_25/                              (Step 4)
├── beat2_parts_25/ foot_contacts_25/            (Step 5, optional)
├── wave16k/ textgrid/ sem/ weights/ train_test_split.csv  (from HF)
├── audios_token_glm/                            (Step 6)
├── TOKENS_AGENT_25/{face,hand,upper,lower,fullbody_genmo,lower_genmo}/   (Step 7)
├── preprocess_hf_beat2_dataset_body_upper_lower_genmo_{train,test}/      (Step 8a)
└── preprocess_hf_beat2_dataset_body_fullbody_genmo_{train,test}/         (Step 8b)
```

---

## Reference

### Coordinate System

BEAT2 is already **Y-up**, matching GENMO — no conversion needed.

### Input

`.npz` archives:

| Key | Shape | Description |
|---|---|---|
| `poses` | `(L, ≥66)` | SMPL-X pose (first 3 = global orient, next 63 = body pose) |
| `trans` | `(L, 3)` | World-space translation |
| `betas` | `(10,)` or `(L, 10)` | Shape parameters |
| `mocap_frame_rate` | scalar | Frame rate (default 30) |

Two source variants exist: `smplxflame_25/` (original) and `smplxflame_25_mirror/` (horizontal mirror augmentation). Both are processed identically.

### GENMO 145D Output Format

| Component | Dims | Indices | Description |
|---|---|---|---|
| `body_pose_r6d` | 126 | 0–125 | 21 body joints × 6D rotation |
| `betas` | 10 | 126–135 | SMPL-X shape (tiled per frame) |
| `global_orient_r6d` | 6 | 136–141 | Root orientation × 6D rotation |
| `local_transl_vel` | 3 | 142–144 | Translation velocity in root-local coords |

6D rotation (Zhou et al., 2019) is used instead of axis-angle for continuity. `local_transl_vel = R^T @ world_velocity`, making it invariant to global heading.

> ⚠️ **145D data vs 135D tokenizer.** This `.npz` stores the full **145D** vector *including* the 10 `betas` (dims 126–135). The released full-body GENMO tokenizer (`VQVAE_0320_GenmoFull`, used in Step 8b → `fullbody_genmo`) is **135D and drops `betas`** — it does not model body shape, so its reconstructions use the **neutral SMPL-X body**. The betas remain in the data for completeness (and for the 145D variant), but are ignored by the released tokenizer.

### Output `.npz` Fields

| Key | Shape | Description |
|---|---|---|
| `motion_vector` | `(L, 145)` | GENMO motion vector |
| `num_frames` | scalar | |
| `fps` | scalar | |
| `gender` | string | |
| `body_pose` | `(L, 63)` | Original axis-angle |
| `global_orient` | `(L, 3)` | Original axis-angle (Y-up) |
| `trans` | `(L, 3)` | Translation (Y-up) |
| `betas` | `(L, 10)` | |

Files already in GENMO format (detected by `motion_vector` with 145 or 149 dims) are copied directly.

### Foot Contacts

Pre-computed in `foot_contacts_25/` as `.npy` (shape `(T, 4)`). Mirror files prefixed `M_`. Used by the Hybrid Lower tokenizer (61D), not by GENMO preprocessing.

### Round-Trip Verification

Built-in verification decodes the GENMO vector back to axis-angle + translation and checks rotation/translation drift against the original.
