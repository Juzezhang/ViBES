# AMASS Preprocessing

## Step 1 — Download AMASS

Make sure you have registered at [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/) and agreed to the SMPL-X license terms before running the download script.

```bash
# Download the SMPL-X version of AMASS (requires registration)
./preprocess/amass_download.sh

# Download the SMPL+H version — needed for text-to-motion (HumanML3D format)
./preprocess/amass_download_smplh.sh
```

## Step 2 — Dataset Structure

`amass_download.sh` and `amass_download_smplh.sh` each prompt for an "AMASS path" — pass the same parent path to both. The scripts then download per-dataset `.tar.bz2` files and extract them in place. The resulting layout is:

```
<AMASS_PATH>/
├── AMASS_original_smplx/        (output of amass_download.sh — Steps 3 / Alternative use this)
│   ├── ACCAD/
│   ├── BMLhandball/
│   ├── BMLmovi/
│   ├── BMLrub/
│   ├── CMU/
│   ├── ... (24 sub-datasets in total)
│   └── WEIZMANN/
└── AMASS_original_smplh/        (output of amass_download_smplh.sh — used by the HumanML3D pipeline in Step 4)
    └── ... (same per-dataset folders)
```

## Step 3 — Preprocess the AMASS dataset

```bash
python preprocess/dataset_process_amass.py \
    --smplx_path "/path/to/your/smplx_models" \
    --dataset_path_original "/path/to/your/data" \
    --dataset_path_processed "/path/to/your/data" \
    --index_path "/path/to/your/index.csv" \
    --ex_fps 25
```

For example:

```bash
python preprocess/dataset_process_amass.py \
    --smplx_path ./model_files/smplx_models \
    --dataset_path_original /path/to/AMASS_original_smplx \
    --dataset_path_processed /path/to/AMASS \
    --index_path ./preprocess/index.csv \
    --ex_fps 25
```

## Step 4 — Get HumanML3D annotations

Follow the [HumanML3D repository](https://github.com/EricGuo5513/HumanML3D) to run their preprocessing pipeline. Once finished, copy the following artifacts into your `AMASS_talking/` root:

- `train.txt`, `val.txt`, `test.txt` — split files
- `texts/` — per-clip text annotations

After this step, your `AMASS_talking/` root should contain `train.txt`, `val.txt`, `test.txt`, and `texts/`.

## Step 5 — Extract the text label index and convert to audio format

The `texts_label_index.zip` file is shipped at `preprocess/texts_label_index.zip`. Extract it into your `AMASS_talking/` root:

```bash
unzip preprocess/texts_label_index.zip -d /path/to/AMASS_talking/
# Produces: /path/to/AMASS_talking/texts_label_index/{000002.txt, 000003.txt, ...}
```

Then process the dataset to match the audio format:

```bash
python preprocess/convert_text_to_transcript_amass.py \
    --root_folder /path/to/AMASS_talking \
    --text_folder /path/to/AMASS_talking/texts \
    --motion_folder /path/to/AMASS_talking/amass_data_align_25 \
    --motion_folder_audio_rotation /path/to/AMASS_talking/amass_data_align_25_audios_rotation \
    --text_folder_audio /path/to/AMASS_talking/texts_for_transcripts \
    --text_label_index_dir /path/to/AMASS_talking/texts_label_index
```

## Step 6 — Download speaker and answer audio

Download the speaker audio and answer audio for the AMASS dataset from our [Google Drive folder](https://drive.google.com/drive/folders/1iqjzmgSy7FYQ2OH5ZJMEw2uRkrhq8E0Z?usp=sharing).

You can also fetch the folder non-interactively with [`gdown`](https://github.com/wkentaro/gdown):

```bash
pip install gdown
cd /path/to/AMASS_talking
gdown --folder https://drive.google.com/drive/folders/1iqjzmgSy7FYQ2OH5ZJMEw2uRkrhq8E0Z
```

The folder contains `audios_q/`, `audios_answer/`, `transcripts_question/`, and `transcripts_answer/`. Unzip any archive inside, and place each subdirectory directly under `AMASS_talking/` so it matches the layout shown below.

## Step 7 — Tokenize question audio with GLM-4-Voice

```bash
PYTHONPATH=./speech_related python preprocess/scripts/get_audio_code_glm.py \
    --wav_folder <AMASS_TALKING_ROOT>/audios_q \
    --output_dir <AMASS_TALKING_ROOT>/audios_q_token_glm
```

Produces `<AMASS_TALKING_ROOT>/audios_q_token_glm/<idx>.npy` — one `(N,) int64` array of GLM-4-Voice discrete audio tokens per `.wav` file. Make sure you've already run `./scripts/download_glm4voice_modules.sh` from the README setup.

## Step 8 — Tokenize motion with the body VQ-VAE

This step needs a trained body VQ-VAE tokenizer. You can either:

- **Use our pretrained tokenizer**, downloaded as part of [`docs/0-overview.md`](../0-overview.md) (see the `model_files/pretrained_cpt/` table), or
- **Train your own** by following the Stage 1 instructions in [`docs/2-training.md`](../2-training.md).

Edit `configs/config_mixed_stage1_vq_compositional.yaml` (or your variant) so that:
- `cfg.DATASET.AMASS_talking.ROOT` points at `<AMASS_TALKING_ROOT>`
- The checkpoint path under each VAE block points at the tokenizer you're using

Then extract motion tokens:

```bash
python -m preprocess.scripts.get_compositional_motion_code \
    --cfg configs/<your-vq-compositional-config>.yaml
```

`get_compositional_motion_code.py` is the single, config-driven tokenizer for every dataset — which dataset(s) to process comes from `cfg.DATASET.datasets`, and which body parts from `cfg.DATASET.MODALITIES[<dataset>]`.

Produces `<AMASS_TALKING_ROOT>/TOKENS_AGENT_25/{face,hand,upper,lower,fullbody_genmo,lower_genmo}/<idx>.npy` — one `(T,) int64` array of motion token IDs per source motion file, per body part.

## Step 9 — Build the HuggingFace dataset

Three preprocessing variants are available (run any subset — they write to different `--output_path` and share the same `<AMASS_TALKING_ROOT>` inputs):

### 9a — Hybrid Upper + Lower (GENMO)

Reads `TOKENS_AGENT_25/upper/` + `TOKENS_AGENT_25/lower_genmo/` and merges them into a single sequence per session:

```bash
python preprocess/preprocess_hf_amass_dataset_body_upper_lower_genmo.py \
    --data_root   <AMASS_TALKING_ROOT> \
    --output_path <AMASS_TALKING_ROOT>/preprocess_hf_amass_dataset_body_upper_lower_genmo_train \
    --split train

python preprocess/preprocess_hf_amass_dataset_body_upper_lower_genmo.py \
    --data_root   <AMASS_TALKING_ROOT> \
    --output_path <AMASS_TALKING_ROOT>/preprocess_hf_amass_dataset_body_upper_lower_genmo_test \
    --split test
```

### 9b — Single-stream Full body (GENMO 135D)

Reads `TOKENS_AGENT_25/fullbody_genmo/` (one stream covers the whole body):

```bash
python preprocess/preprocess_hf_amass_dataset_body_fullbody_genmo.py \
    --data_root   <AMASS_TALKING_ROOT> \
    --output_path <AMASS_TALKING_ROOT>/preprocess_hf_amass_dataset_body_fullbody_genmo_train \
    --split train

python preprocess/preprocess_hf_amass_dataset_body_fullbody_genmo.py \
    --data_root   <AMASS_TALKING_ROOT> \
    --output_path <AMASS_TALKING_ROOT>/preprocess_hf_amass_dataset_body_fullbody_genmo_test \
    --split test
```

### 9c — Body (interleaved upper / lower / hand, supports motion variants)

The recommended path — reads `TOKENS_AGENT_25/{upper,lower,hand}/`, interleaves them per audio group, and supports a `--motion_variant` flag to pick which body parts to encode:

```bash
python preprocess/preprocess_hf_amass_dataset_body.py \
    --data_root        <AMASS_TALKING_ROOT> \
    --output_path      <AMASS_TALKING_ROOT>/preprocess_hf_amass_dataset_body_train \
    --split            train \
    --motion_variant   body_only
```

| `--motion_variant` | Per-group composition | Motion FPS |
|---|---|---|
| `body_only` (default) | 13 upper + 13 lower + 13 hand | 18.75 |
| `upper_hand` | 13 upper + 13 hand (no lower) | 12.5 |
| `lower_only` | 13 lower (no upper / hand) | 6.25 |

Padded positions are correctly masked as `-100` (no loss); `begin_of_motion` is treated as a structural separator (also `-100`, not supervised).

> 💡 **AMASS-specific behaviour**: AMASS upstream mocap has no finger data, so `TOKENS_AGENT_25/hand/` is empty for most AMASS sequences. The script handles this automatically — it inserts zero-filled hand tokens at the correct positions to preserve the 13:13:13 grouping, but masks their labels to `-100` so the model is **not supervised on hand for AMASS data**. You don't need to do anything; just run with the default `body_only` variant and AMASS hand will be skipped from the loss while BEAT2 hand (which has real data) will be supervised.

Training instructions for the body expert live in [`../2-training.md`](../2-training.md).

After Step 9, your `<AMASS_TALKING_ROOT>` should look like this:

```
<AMASS_TALKING_ROOT>/
├── amass_data_align_25/                                  (Step 3)
├── amass_data_align_25_audios_rotation/                  (Step 5)
├── texts/ texts_label_index/ texts_for_transcripts/      (Steps 4–5)
├── train.txt / val.txt / test.txt                        (Step 4)
├── audios_q/ audios_answer/                              (Step 6)
├── transcripts_question/ transcripts_answer/             (Step 6)
├── audios_q_token_glm/                                   (Step 7)
├── TOKENS_AGENT_25/{face,hand,upper,lower,fullbody_genmo,lower_genmo}/  (Step 8)
├── preprocess_hf_amass_dataset_body_upper_lower_genmo_{train,test}/    (Step 9a)
└── preprocess_hf_amass_dataset_body_fullbody_genmo_{train,test}/        (Step 9b)
```

---

## Alternative: Convert AMASS to GENMO 145D

If you only need the GENMO 145D motion vector (e.g., for the body tokenizer training described in [`../2-training.md`](../2-training.md)), use the dedicated GENMO conversion script instead of the full step-by-step pipeline above:

```bash
python preprocess/dataset_process_amass_genmo.py \
    --dataset_path_original <AMASS_ORIGINAL_SMPLX_ROOT> \
    --dataset_path_processed <AMASS_ROOT> \
    --index_path ./preprocess/index.csv \
    --ex_fps 25 \
    --index_fps 20
```

**Output:** `<AMASS_ROOT>/amass_genmo_25/*.npz` (HumanML3D-style numbered filenames)

**Optional flags:**

- `--debug` — skip SMPL-X type and FPS validation
- `--verify` — round-trip verification on random samples

---

## Reference

### Coordinate System

AMASS is **Z-up**; GENMO is **Y-up**. A -90° rotation around X is applied:

- Translation: `trans_yup = trans_zup @ T^T`
- Global orientation: `R_yup = T @ R_zup`
- Body pose: unchanged (local joint rotations are coordinate-invariant)

### Input

`.npz` archives with SMPL-X parameters. Two formats supported:

- Standard: `poses`, `trans`, `betas`
- Split: `pose_body`, `root_orient`, `trans`, `betas`

Plus metadata: `mocap_frame_rate`, `gender`, `surface_model_type`.

### HumanML3D Index Mapping

The script uses `preprocess/index.csv` (from HumanML3D) to map source files to standardized numbered names, specify frame boundaries, and filter sequences. Dataset names are standardized (e.g., `BioMotionLab_NTroje` → `BMLrub`).

### Filtering

Included sub-datasets: ACCAD, BMLrub, BMLhandball, BMLmovi, CMU, DFaust, EKUT, Eyes_Japan_Dataset, HumanEva, KIT, HDM05, PosePrior, MoSh, SFU, SSM, TCDHands, TotalCapture, Transitions.

Files filtered by: index mapping, `surface_model_type ∈ {smplx, smplx_locked_head}`, frame-rate key presence.

### Resampling

Integer decimation from source FPS to `--ex_fps` (default 25). Start/end frames (at 20 FPS in the index) are scaled accordingly. Dataset-specific lead-in trims:

| Dataset                   | Trim  |
| ------------------------- | ----- |
| Eyes_Japan_Dataset, HDM05 | 3 s   |
| TotalCapture, PosePrior   | 1 s   |
| Transitions               | 0.5 s |

### Output Format

Same 145D vector as BEAT2 — see [`beat2.md#genmo-145d-output-format`](beat2.md#genmo-145d-output-format).
