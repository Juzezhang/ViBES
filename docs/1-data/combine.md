# Combine Datasets into Training Sets

The per-dataset guides in this folder each take **one** source dataset all the way to its own
tokenized HuggingFace dataset. A Stage-2 expert is then trained on a **single** dataset that mixes
the relevant sources. This page covers that last step: packing each source into a HF dataset and
**concatenating** them with `preprocess/combine_dataset.py`.

```
per-dataset preprocessing            per-dataset HF packing                 merge
(see each 1-data guide)              (preprocess_hf_*.py / packer)           (combine_dataset.py)
────────────────────────             ──────────────────────                 ────────────────────
BEAT2   → tokens + audio tokens  ─┐
AMASS   → tokens                  │
TFHP    → face tokens + audio     ├─► processed_<dataset>_<modality>_<split>/ ─► <combined>/ ─► train_vibes_*
Embody3D→ tokens + audio          │                                                  │
YouTube → tokens + audio          ┘                                          --tokenized_dataset
```

> ℹ️ **Combine within a modality, never across.** Face datasets share one schema (audio + face
> tokens) and body datasets share another (audio + upper/lower/hand tokens), so the **face expert**
> and the **body expert** each get their own combined set. `concatenate_datasets` requires matching
> features.

This repo defines **one face training target** and **three body training targets** (see
[`2-training.md`](../2-training.md) for the matching launch commands):

| Expert | Target | Sources |
|---|---|---|
| Face | `face` | BEAT2 + TFHP + YouTube_Talking + WebTalk-Synthetic |
| Body | **`cospeech`** | BEAT2 + Embody3D |
| Body | **`text2motion`** | HumanML3D (alone) |
| Body | **`full`** | BEAT2 + AMASS + Embody3D (all body sources **except** YouTube) |

---

## Step 1 — Pack each source into a HuggingFace dataset

Each per-dataset packing script (`preprocess/preprocess_hf_*.py`, plus
`preprocess_embody3d_dataset_body.py` for Embody3D) reads one preprocessed dataset root and writes a
HF dataset under `<output_path>/tokenized_dataset/`. Common flags: `--data_root`, `--output_path`,
`--split` (`train`/`val`/`test`), `--model_name` (tokenizer/processor), `--max_seq_length`.

**Face expert sources:**

| Dataset | Script |
|---|---|
| BEAT2 | `preprocess_hf_beat2_dataset_face.py` |
| TFHP | `preprocess_hf_tfhp_face.py` (style-control: `preprocess_hf_tfhp_face_style.py`) |
| YouTube_Talking | `preprocess_hf_youtube_dataset_face.py` |
| WebTalk-Synthetic | `preprocess_hf_youtube_synthetic_dataset_face.py` |

**Body expert sources:**

| Dataset | Script |
|---|---|
| BEAT2 | `preprocess_hf_beat2_dataset_body.py` (GENMO: `..._body_fullbody_genmo.py` / `..._body_upper_lower_genmo.py`) |
| AMASS | `preprocess_hf_amass_dataset_body.py` (+ `..._fullbody_genmo.py` / `..._upper_lower_genmo.py`) |
| Embody3D | `preprocess_embody3d_dataset_body.py` (assistant-only; `--data_root` = the tokenized `aiagent` dir — see [`embody3d.md`](embody3d.md)) |
| YouTube_Talking | `preprocess_hf_youtube_dataset_body.py` |

**Text→motion (HumanML3D):** `preprocess_hf_h3d_text2motion.py`

Example (BEAT2 body, train split):

```bash
python preprocess/preprocess_hf_beat2_dataset_body.py \
    --data_root <BEAT2_ROOT> \
    --output_path <OUT>/processed_beat2_body_train \
    --split train
```

Run the matching script for every source in the target you are building, **for each split** you need.

---

## Step 2 — Build a combined set per target

`preprocess/combine_dataset.py` loads each source's `tokenized_dataset/` with `load_from_disk`,
concatenates them with `datasets.concatenate_datasets`, and writes the merged dataset to
`--output_path`. **Pass each source as its `…/tokenized_dataset` subdir**; the combined output is
written directly to `--output_path` (point training there).

| Flag | Meaning |
|---|---|
| `--datasets PATH [PATH ...]` | source `…/tokenized_dataset` dirs to merge (space-separated) |
| `--dataset_list FILE` | …or a text file with one path per line |
| `--output_path DIR` | where to write the combined dataset (the arrow dataset lands here directly) |
| `--validate` | skip missing paths (warn) instead of failing |

Build a separate combined set per split (`*_train`, `*_test`, …).

### Face expert

```bash
python preprocess/combine_dataset.py \
    --datasets <OUT>/processed_beat2_face_train/tokenized_dataset \
               <OUT>/processed_tfhp_face_train/tokenized_dataset \
               <OUT>/processed_youtube_face_train/tokenized_dataset \
               <OUT>/processed_webtalk_synthetic_face_train/tokenized_dataset \
    --output_path <OUT>/processed_face_train_combined --validate
```

### Body — `cospeech` (BEAT2 + Embody3D)

The conversational / co-speech body target: speech-driven gesture from the two conversational
sources.

```bash
python preprocess/combine_dataset.py \
    --datasets <OUT>/processed_beat2_body_train/tokenized_dataset \
               <OUT>/processed_embody3d_body_train/tokenized_dataset \
    --output_path <OUT>/processed_body_cospeech_train --validate
```

### Body — `text2motion` (HumanML3D, alone)

A **single source** — no concatenation needed. Just pack HumanML3D (see [`humanml3d.md`](humanml3d.md))
and train directly on its `tokenized_dataset`:

```bash
python preprocess/preprocess_hf_h3d_text2motion.py \
    --data_root <HUMANML3D_ROOT> \
    --output_path <OUT>/processed_h3d_text2motion_train \
    --split train
# train target -> <OUT>/processed_h3d_text2motion_train/tokenized_dataset
```

### Body — `full` (all body except YouTube)

BEAT2 + AMASS + Embody3D — the full body-motion mix (excludes YouTube):

```bash
python preprocess/combine_dataset.py \
    --datasets <OUT>/processed_beat2_body_train/tokenized_dataset \
               <OUT>/processed_amass_body_train/tokenized_dataset \
               <OUT>/processed_embody3d_body_train/tokenized_dataset \
    --output_path <OUT>/processed_body_full_train --validate
```

---

## Step 3 — Train on the combined set

Point the matching Stage-2 launcher at the combined dataset — see [`2-training.md`](../2-training.md),
which defines the launch command for each of the three body targets (`cospeech` / `text2motion` /
`full`) and the face target. For example, the co-speech body model:

```bash
deepspeed ... training/train_vibes.py \
    --tokenized_dataset <OUT>/processed_body_cospeech_train \
    ...
```

> ⚠️ `concatenate_datasets` fails if the sources have mismatched features. If a source was packed
> with a different `--model_name`, `--max_seq_length`, or token layout, re-pack it with the same
> settings before combining.
