# WebTalk-Synthetic Face Preprocessing

WebTalk-Synthetic is a large set of **synthetic co-speech facial motion** (FLAME
coefficients) paired with in-the-wild conversational speech. The face motion is
*generated* — it is produced by an audio-driven face model from filtered
in-the-wild talking audio, not motion-captured. It provides ~12.7 k assistant-side
clips used to train the ViBES face expert with extra in-the-wild coverage.

> ⚠️ **Research use only.** The driving audio is segmented from public
> talking-head videos. The dataset is released **for non-commercial research
> only** (recommended license: CC-BY-NC-4.0); do not use the audio for
> commercial purposes.

---

## Step 1 — Obtain the dataset

Download from Hugging Face ([`JuzeZhang/WebTalk-Synthetic`](https://huggingface.co/datasets/JuzeZhang/WebTalk-Synthetic)):

```bash
huggingface-cli download JuzeZhang/WebTalk-Synthetic \
    --repo-type dataset --local-dir <WEBTALK_ROOT>
```

The repo ships each modality as a single `.tar` (fewer, larger files upload and
sync much faster than ~50k loose files). After download, extract them in place:

```bash
cd <WEBTALK_ROOT>
for f in audios audios_token_glm FLAME_coeffs_25 transcripts; do tar -xf "$f.tar"; done
# optionally: rm *.tar
```

After extraction:

```
<WEBTALK_ROOT>/
├── audios/                 (16 kHz mono WAV, one per clip)
├── audios_token_glm/       (GLM-4-Voice audio tokens — lets you skip Step 2)
├── FLAME_coeffs_25/        (synthetic face motion — one .npz per clip, 25 fps)
├── transcripts/            (one .txt per clip)
└── train.txt / val.txt / test.txt   (split files, one clip stem per line)
```

Each clip stem looks like `<session>_<segment>` (e.g. `202008647_0001`), and all
modalities are keyed by it. The dataset already includes `audios_token_glm/`, so
**Step 2 is optional** — skip it unless you want to re-tokenize the audio yourself.

## Step 2 — Tokenize audio with GLM-4-Voice

```bash
PYTHONPATH=./speech_related python preprocess/scripts/get_audio_code_glm.py \
    --wav_folder <WEBTALK_ROOT>/audios \
    --output_dir <WEBTALK_ROOT>/audios_token_glm
```

Produces `<WEBTALK_ROOT>/audios_token_glm/<clip>.npy` — one `(N,) int64` array of
GLM-4-Voice discrete audio tokens per `.wav`. Make sure you've already run
`./scripts/download_glm4voice_modules.sh` from the README setup.

## Step 3 — Tokenize face motion with the Face VQ-VAE

This step needs a trained face VQ-VAE tokenizer. Use the pretrained face tokenizer
downloaded as part of [`docs/0-overview.md`](../0-overview.md)
(`model_files/pretrained_cpt/face/face.ckpt`), or train your own following
[Stage 1 in `docs/2-training.md`](../2-training.md).

Edit your VQ config so that `cfg.DATASET.WebTalk_Synthetic.ROOT` points at
`<WEBTALK_ROOT>` and `cfg.DATASET.MODALITIES.WebTalk_Synthetic: [face]` (face-only),
then run the unified tokenizer:

```bash
python -m preprocess.scripts.get_compositional_motion_code \
    --cfg configs/<your-face-code-config>.yaml
```

Produces `<WEBTALK_ROOT>/TOKENS_AGENT_25/face/<clip>.npy` — one `(T,) int64` array
of face token IDs per clip.

## Step 4 — Build the HuggingFace dataset

```bash
# Train split
python preprocess/preprocess_hf_youtube_synthetic_dataset_face.py \
    --data_root   <WEBTALK_ROOT> \
    --output_path <WEBTALK_ROOT>/processed_webtalk_synthetic_face_train \
    --split train

# Test split
python preprocess/preprocess_hf_youtube_synthetic_dataset_face.py \
    --data_root   <WEBTALK_ROOT> \
    --output_path <WEBTALK_ROOT>/processed_webtalk_synthetic_face_test \
    --split test
```

Reads `audios_token_glm/`, `TOKENS_AGENT_25/face/`, and `transcripts/`; interleaves
them per clip into fixed-size groups. Per group: **26 audio + 1 begin_of_motion +
52 face tokens**; only the face tokens are supervised (`labels = -100` elsewhere).

After Step 4:

```
<WEBTALK_ROOT>/
├── audios/ FLAME_coeffs_25/ transcripts/         (Steps 1)
├── audios_token_glm/                             (Step 2)
├── TOKENS_AGENT_25/face/                         (Step 3)
└── processed_webtalk_synthetic_face_{train,test}/ (Step 4)
```

Training instructions for the face expert live in [`../2-training.md`](../2-training.md).

---

## Reference

### Coordinate / motion format

Face motion is stored as FLAME 2020 coefficients (`.npz`, one per clip), 25 fps:

| Key | Shape | Description |
|---|---|---|
| `exp` | `(T, 100)` float32 | FLAME expression coefficients |
| `shape` | `(T, 100)` float64 | FLAME shape coefficients |
| `pose` | `(T, 6)` float32 | head pose (3) + jaw pose (3), axis-angle |
| `mocap_frame_rate` | scalar | 25 |

The face VQ-VAE consumes a 6D-rotation + expression vector (head 6D, jaw 6D,
expression) derived from these coefficients.

### Audio

16 kHz mono PCM WAV, ~8 s per clip. Segmented from public talking-head videos;
research use only.

### Splits

`train.txt`, `val.txt`, `test.txt` — one clip stem (`<session>_<segment>`) per line.

### Key files

| Purpose | File |
|---|---|
| Audio → GLM-4-Voice tokens (Step 2) | `preprocess/scripts/get_audio_code_glm.py` |
| Face VQ-VAE token extraction (Step 3) | `preprocess/scripts/get_compositional_motion_code.py` (face-only via `MODALITIES.WebTalk_Synthetic: [face]`) |
| HuggingFace dataset builder (Step 4) | `preprocess/preprocess_hf_youtube_synthetic_dataset_face.py` |
