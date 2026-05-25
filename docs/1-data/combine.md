# Combine Datasets into a Unified Training Set

The per-dataset guides in this folder each take **one** source dataset all the way to
its own tokenized HuggingFace dataset. The Stage-2 expert (face or body) is trained on a
**single** dataset that mixes all the sources together. This page covers that last step:
packing each source into a HF dataset and **concatenating** them with
`preprocess/combine_dataset.py`.

```
per-dataset preprocessing            per-dataset HF packing                merge
(see each 1-data guide)              (preprocess_hf_*.py)                  (combine_dataset.py)
────────────────────────             ──────────────────────               ────────────────────
BEAT2  → tokens + audio tokens  ─┐
AMASS  → tokens                  ├─►  processed_<dataset>_<modality>_<split>/  ─►  unified_<modality>_<split>/  ─► train_vibes_*
TFHP   → face tokens + audio     │                                                       │
YouTube→ tokens + audio          ┘                                              --tokenized_dataset
```

> ℹ️ Combine **within a modality**, not across. Face datasets share one schema (audio +
> face tokens) and body datasets share another (audio + upper/lower/hand or GENMO tokens),
> so you build **one combined face set** for the face expert and **one combined body set**
> for the body expert. `concatenate_datasets` requires matching features.

---

## Step 1 — Pack each source into a HuggingFace dataset

Each `preprocess/preprocess_hf_*.py` reads one preprocessed dataset root and writes a HF
dataset for one split. Common flags: `--data_root`, `--output_path`, `--split`
(`train`/`val`/`test`), `--model_name` (tokenizer/processor), `--max_seq_length`.

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
| YouTube_Talking | `preprocess_hf_youtube_dataset_body.py` |

**Text→motion (HumanML3D):** `preprocess_hf_h3d_text2motion.py`

Example (face, one source, train split):

```bash
python preprocess/preprocess_hf_beat2_dataset_face.py \
    --data_root <BEAT2_ROOT> \
    --output_path <OUT>/processed_beat2_face_train \
    --split train
```

Run the matching script for every source you want in the mix (and for each split you need).

---

## Step 2 — Concatenate into one training set

`preprocess/combine_dataset.py` loads each HF dataset with `load_from_disk`, concatenates
them with `datasets.concatenate_datasets`, and writes the result with `save_to_disk`.

```bash
# Option A: list the dataset paths inline
python preprocess/combine_dataset.py \
    --datasets <OUT>/processed_beat2_face_train \
               <OUT>/processed_tfhp_face_train \
               <OUT>/processed_youtube_face_train \
               <OUT>/processed_webtalk_synthetic_face_train \
    --output_path <OUT>/processed_face_train_combined \
    --validate

# Option B: one path per line in a text file
python preprocess/combine_dataset.py \
    --dataset_list <OUT>/face_train_sources.txt \
    --output_path <OUT>/processed_face_train_combined
```

| Flag | Meaning |
|---|---|
| `--datasets PATH [PATH ...]` | HF dataset dirs to merge (space-separated) |
| `--dataset_list FILE` | …or a text file with one HF dataset path per line |
| `--output_path DIR` | where to write the combined dataset |
| `--validate` | skip missing paths (warn) instead of failing |

`--datasets` and `--dataset_list` are mutually exclusive. Build a separate combined set per
split (`*_train_combined`, `*_test_combined`, …).

---

## Step 3 — Train on the combined set

Point the Stage-2 launcher at the combined dataset (see [`2-training.md`](../2-training.md)):

```bash
deepspeed ... training/train_vibes_face_style_control.py \
    --tokenized_dataset <OUT>/processed_face_train_combined/tokenized_dataset \
    ...
```

> ⚠️ `concatenate_datasets` fails if the sources have mismatched features. If a source was
> packed with a different `--model_name`, `--max_seq_length`, or token layout, re-pack it
> with the same settings before combining.
