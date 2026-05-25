# TFHP Face Preprocessing

Convert the TFHP dataset (videos + LMDB) into the tokenized HuggingFace dataset used for ViBES face training with style control.

## Step 1 — Obtain the TFHP raw dataset

TFHP is not distributed under an open license. Contact the [DiffPoseTalk authors](https://github.com/DiffPoseTalk/DiffPoseTalk/blob/main/datasets/HDTF_TFHP/README.md) to request access. After they grant access, you should have:

```
TFHP/                       ← we'll call this <TFHP_RAW_ROOT>
├── HDTF_TFHP-lmdb/         (LMDB database with per-clip audio + FLAME coefficients + splits)
│   ├── data.mdb
│   ├── lock.mdb
│   ├── keys.txt            (one "<speaker>/<session>" key per line)
│   ├── stats_train.npz     (normalization stats over the train split)
│   ├── train.txt / val.txt / test.txt
│   └── LICENSE
├── data/                   (raw video files, one per clip)
│   ├── TH_00000/000.mp4
│   ├── TH_00001/000.mp4
│   └── ...
├── TFHP_raw.zip            (original archive of the video set)
└── LICENSE
```

> 💡 Two naming systems coexist: raw videos use sequential `TH_XXXXX` IDs (~341 clips), while the LMDB and downstream split files use HDTF-style `<speaker>/<session>` keys (~1100 entries; e.g. `RD_AmandaStuck/000`). Steps 2–6 operate entirely on the `<speaker>/<session>` namespace driven by the LMDB.

## Step 2 — Extract audio / transcripts / FLAME coefficients from the LMDB

```bash
pip install lmdb faster-whisper          # not in requirements.txt by default
python preprocess/scripts/get_transcript_TFHP.py \
    --lmdb_path  <TFHP_RAW_ROOT>/HDTF_TFHP-lmdb \
    --output_dir <TFHP_ROOT> \
    --whisper_device cuda                 # use cpu if no GPU; cpu ≈ 6 s/clip, cuda is much faster
```

Produces:

```
<TFHP_ROOT>/
├── audios/{speaker}/{session}/{idx}.wav
├── transcripts/{speaker}/{session}/{idx}.txt
└── coef/{speaker}/{session}/{idx}.npz
```

**Optional flag:** `--max_samples N` caps to the first N LMDB keys (useful for smoke tests; default processes everything).

## Step 3 — Tokenize audio with GLM-4-Voice

```bash
pip install soundfile                     # used by the script for WAV loading
PYTHONPATH=./speech_related python preprocess/scripts/get_audio_code_glm.py \
    --wav_folder <TFHP_ROOT>/audios \
    --output_dir <TFHP_ROOT>/audios_token_glm
```

The `PYTHONPATH=./speech_related` makes the GLM-4-Voice tokenizer importable. Make sure you've already run `./scripts/download_glm4voice_modules.sh` from the README setup.

Produces:

```
<TFHP_ROOT>/audios_token_glm/{speaker}/{session}/{idx}.npy   # one (50,) int64 array per .wav
```

## Step 4 — Tokenize face motion with the Face VQ-VAE

This step needs a trained face VQ-VAE tokenizer checkpoint. You can either:

- Use the pretrained face tokenizer downloaded as part of [`docs/0-overview.md`](../0-overview.md) (e.g., `model_files/pretrained_cpt/face/face.ckpt`), or
- Train your own following [Stage 1 in `docs/2-training.md`](../2-training.md).

Edit `configs/config_get_face_code_TFHP.yaml` (or your variant) so that `cfg.DATASET.TFHP.ROOT` points at `<TFHP_ROOT>`, then run:

```bash
python -m preprocess.scripts.get_compositional_motion_code \
    --cfg configs/<your-tfhp-face-code-config>.yaml
```

`get_compositional_motion_code.py` is the single, config-driven tokenizer for every dataset; for TFHP set `cfg.DATASET.MODALITIES.TFHP: [face]` so only the face VQ-VAE runs.

Produces:

```
<TFHP_ROOT>/TOKENS_AGENT_25/{speaker}/{session}/{idx}.npy
```

## Step 5 — Copy the train / val / test split files

```bash
cp <TFHP_RAW_ROOT>/HDTF_TFHP-lmdb/{train,val,test}.txt <TFHP_ROOT>/
```

Each line is one `<speaker>/<session>` entry (e.g., `RD_AmandaStuck/000`).

## Step 6 — Build the HuggingFace dataset

There are **two preprocessing variants**. Pick whichever matches your downstream use case (or run both — they write to different `--output_path` and share the same `<TFHP_ROOT>` inputs):

### 6a — Style control variant (recommended for fair comparison with ARTalk-style baselines)

The face expert receives a sequence of 50 reference face tokens as a "style" prefix. This is what we use for our reported metrics against talking-head baselines.

```bash
# Train set (style = placeholder; collator overrides at runtime)
python preprocess/preprocess_hf_tfhp_face_style.py \
    --data_root   <TFHP_ROOT> \
    --output_path <TFHP_ROOT>/processed_tfhp_tokenized_face_style_train \
    --split train

# Test set (style = fixed middle of GT; deterministic)
python preprocess/preprocess_hf_tfhp_face_style.py \
    --data_root   <TFHP_ROOT> \
    --output_path <TFHP_ROOT>/processed_tfhp_tokenized_face_style_test \
    --split test
```

Per-turn group: **25 audio + 1 begin_of_motion + 50 face tokens**. The HF dataset has 12 features — adds `style_pool` on top of the normal-variant fields.

### 6b — Normal variant (no style control — used for ViBES's own face model)

The face expert generates motion directly from the audio context, with no reference-face prefix. We do **not** report metrics from this variant in the paper, but it's the simpler version and useful if you don't need ARTalk-style style conditioning.

```bash
# Train set
python preprocess/preprocess_hf_tfhp_face.py \
    --data_root   <TFHP_ROOT> \
    --output_path <TFHP_ROOT>/processed_tfhp_tokenized_face_normal_train \
    --split train

# Test set — uses a different script (each clip becomes one sequence, no merging)
python preprocess/preprocess_hf_tfhp_face_test.py \
    --data_root   <TFHP_ROOT> \
    --output_path <TFHP_ROOT>/processed_tfhp_tokenized_face_normal_test \
    --split test
```

Per-turn group: **26 audio + 1 begin_of_motion + 52 face tokens** (slightly different from the style-control variant). The HF dataset has 11 features — no `style_pool`.

Training instructions for the face expert live in [`../2-training.md`](../2-training.md).

After Step 6, your `<TFHP_ROOT>` should look like:

```
<TFHP_ROOT>/
├── audios/                                              (Step 2)
├── transcripts/                                         (Step 2)
├── coef/                                                (Step 2)
├── audios_token_glm/                                    (Step 3)
├── TOKENS_AGENT_25/                                     (Step 4)
├── train.txt / val.txt / test.txt                       (Step 5)
├── processed_tfhp_tokenized_face_style_train/           (Step 6a)
├── processed_tfhp_tokenized_face_style_test/            (Step 6a)
├── processed_tfhp_tokenized_face_normal_train/          (Step 6b)
└── processed_tfhp_tokenized_face_normal_test/           (Step 6b)
```

---

## Reference

### Preprocessing variants (Step 6)

Three released scripts:

| Script | Style control? | Output features | When to use |
|---|---|---|---|
| `preprocess/preprocess_hf_tfhp_face_style.py` | ✅ yes | 12 (incl. `style_pool`) | **Step 6a** — fair comparison vs. ARTalk-style baselines (used in our paper's Table 4) |
| `preprocess/preprocess_hf_tfhp_face.py` | ❌ no | 11 | **Step 6b** train/val — ViBES's own face model |
| `preprocess/preprocess_hf_tfhp_face_test.py` | ❌ no | 11 | **Step 6b** test split — one file = one sequence (no merging across files) |

All three read the same `<TFHP_ROOT>` layout (`transcripts/`, `audios_token_glm/`, `TOKENS_AGENT_25/`, `train.txt`/`val.txt`/`test.txt`) — only the output structure differs.

> An ablation variant `preprocess/mics/preprocess_hf_tfhp_face_style_no_interp.py` exists for the sequential-position-encoding baseline; it is not used in the paper.

### Sequence structure

```
<|assistant|>streaming_transcription\n     (text, mod 0, label=-100)
style_control:<|face_X|> x 50              (style prefix, label=-100)
[Turn 1]
  text_tokens                               (mod 0, label=-100)
  <|audio_X|> x 25                          (mod 1, label=-100)
  <|begin_of_motion|>                       (mod 2, label=-100)
  <|face_X|> x 50                           (mod 2, label=token_id, supervised)
[Turn 2] ...
<eos>                                       (mod 0, label=-100)
```

Group sizes (aligned with ARTalk):

| Component | Count | FPS | Duration |
|---|---|---|---|
| Audio tokens per turn | 25 | 12.5 | 2.0 s |
| Face tokens per turn | 50 | 25.0 | 2.0 s |
| Style control tokens | 50 | — | reference clip |

### Style selection strategy

| Split | Preprocessing | Runtime |
|---|---|---|
| **train** | `000.npy[0, :50]` placeholder | Collator randomly samples from `style_pool` each batch |
| **test/val** | Fixed middle of session's GT face tokens | Used as-is (deterministic) |

Each record carries a `style_pool` field with **all face token IDs from the session** for dynamic replacement during training.

### Output fields (Step 6 HuggingFace dataset)

| Field | Type | Description |
|---|---|---|
| `input_ids` | `List[int]` | Full token sequence |
| `labels` | `List[int]` | `-100` for unsupervised; token id for supervised face tokens |
| `attention_mask` | `List[int]` | All 1s (no padding) |
| `modality_masks_0` | `List[bool]` | Text tokens |
| `modality_masks_1` | `List[bool]` | Audio tokens |
| `modality_masks_2` | `List[bool]` | Motion tokens (face + begin_of_motion + style) |
| `position_encoding_indices` | `List[float]` | Position indices |
| `style_pool` | `List[int]` | Session face token IDs for dynamic style sampling |

### Step 6 arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_root` | ✅ | — | TFHP processed root (`<TFHP_ROOT>`) |
| `--output_path` | ✅ | — | Output HF dataset directory |
| `--split` | ✅ | — | `train` / `test` / `val` |
| `--model_name` | | `THUDM/glm-4-voice-9b` | Tokenizer model |
| `--max_seq_length` | | 2048 | Max sequence length |
| `--debug` | | False | Small subset only |

### Key files

| File | Purpose |
|---|---|
| `preprocess/scripts/get_transcript_TFHP.py` | LMDB → audios / transcripts / coef (Step 2) |
| `preprocess/scripts/get_audio_code_glm.py` | WAV → GLM-4-Voice audio tokens (Step 3) |
| `preprocess/scripts/get_compositional_motion_code.py` | Unified VQ-VAE token extraction (Step 4); face-only via `MODALITIES.TFHP: [face]` |
| `preprocess/preprocess_hf_tfhp_face_style.py` | HF dataset builder (Step 6a, style control, interpolated) |
| `preprocess/mics/preprocess_hf_tfhp_face_style_no_interp.py` | HF dataset builder (ablation, style control, sequential position encoding) |
| `preprocess/preprocess_hf_tfhp_face.py` | HF dataset builder (Step 6b, normal, train) |
| `preprocess/preprocess_hf_tfhp_face_test.py` | HF dataset builder (Step 6b, normal, test) |
| `training/train_vibes_face_style_control.py` | Training with dynamic style sampling (separate doc) |
| `evaluation/eval_face_style_control.py` | Evaluation (LVE / MHD / FDD) |
| `evaluation/face_metrics.py` | Metric implementations |

### Design reference

Inspired by [ARTalk](https://arxiv.org/abs/2502.20323):

- ARTalk: 100 frames (4 s @ 25 fps) for style/motion, random sampling during training
- ViBES: 50 tokens per group (2 s @ 25 fps), random sampling from session's `style_pool` via collator
