# Inference

Run trained ViBES models to generate conversational facial expressions, body motion, and text-to-motion, and to tokenize/decode audio.

## Face Inference (Main Demo)

Generates facial animation + synchronized audio response from a text prompt:

```bash
python inference/inference_face.py \
    --checkpoint ./ViBES-Face \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --user_text "If you had a superpower for one day, what would you choose?"
```

### Pipeline

1. Synthesize speech from `--user_text` via TTS
2. Tokenize the audio with the GLM-4-Voice speech tokenizer
3. Generate motion tokens through the face expert (autoregressive)
4. Decode tokens → FLAME parameters (expression + jaw)
5. Render FLAME mesh to video, mux with audio

### Common Flags

| Flag | Default | Description |
|---|---|---|
| `--user_text` | a sample question | Text prompt to convert into a conversational response |
| `--output_dir` | `./test_output` | Output directory for the generated video |
| `--output_filename` | `response.mp4` | Name of the output video |
| `--checkpoint` | `./ViBES-Face` | Path to the face-expert checkpoint directory |
| `--glm_base_path` | `THUDM/glm-4-voice-9b` | GLM-4-Voice base used to reconstruct the frozen text/audio expert (see note) |
| `--vibes_config` | `./vibes` (9B) | MoME config + tokenizer; `vibes_0.5b` for the 0.5B model |
| `--device` | `cuda` | `cuda` or `cpu` |

> **Loading the released checkpoint.** `ViBES-Face` (and the body checkpoint) store **only the trained
> motion expert (~0.86 GB)**, marked by `expert_checkpoint.json`. The inference scripts detect this
> marker and reconstruct the frozen text/audio expert from `--glm_base_path`, merging the two into the
> full model automatically (bit-identical to a full checkpoint). Point `--glm_base_path` at the
> GLM-4-Voice base you downloaded during setup to avoid a duplicate ~18 GB download. Older full
> checkpoints (no marker) still load normally.

Output: `<output_dir>/<output_filename>` (FLAME render with the speech).

### 0.5B face model

`ViBES-Face-0.5B` is the face expert on [ViBES-Audio](https://huggingface.co/JuzeZhang/ViBES-Audio),
our distilled 0.5B GLM-4-Voice, so the whole model is ~0.85B parameters instead of ~10B. On TFHP it
matches the 9B `ViBES-Face` (LVE 10.15 vs 10.11, FDD 7.51 vs 8.30;
[docs/4-evaluation.md](4-evaluation.md#talking-head-on-tfhp-paper-table-4)). The spoken answer also
comes from the 0.5B backbone.

```bash
huggingface-cli download JuzeZhang/ViBES-Audio --local-dir ./model_files/ViBES-Audio
gdown --folder https://drive.google.com/drive/folders/17yH0Olqi3C5y5GpXuyg5rOZKQN027qeL -O ./ViBES-Face-0.5B

python inference/inference_face.py \
    --checkpoint ./ViBES-Face-0.5B \
    --vibes_config vibes_0.5b \
    --glm_base_path ./model_files/ViBES-Audio \
    --user_text "If you had a superpower for one day, what would you choose?"
```

---

## Body Inference

Body generation produces a full-body SMPL-X animation with synchronized speech. Two body experts are
released, on Google Drive for now (a Hugging Face mirror will follow):

| Checkpoint | Conditioning | What it does |
|---|---|---|
| `ViBES-Body-Cospeech` | Cospeech | Gestures follow the speech the model generates from your text, or a speech clip you pass with `--audio`. No instruction about which motion to perform. |
| `ViBES-Body-Agent` | Instruction + cospeech | With `--agent`, the model also performs the motion the user asks for (e.g. "wave hello") while it answers, staying speech-synchronized. |

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1GsoWZMcZYq-_IOYuKaWWu3bBZClV3E17 -O ./ViBES-Body-Cospeech
gdown --folder https://drive.google.com/drive/folders/1EiN7n2anqbHaKXRj28bOZzIC955qrhhH -O ./ViBES-Body-Agent
```

Both use the **Upper+Lower+Hand** representation: separate upper / lower / hand VQ-VAEs
(`model_files/pretrained_cpt/body/lom_vq.ckpt`), with root translation recovered from the lower-body
pose by the Global VAE (`model_files/pretrained_cpt/VAE_Global_from_Lower54/last.ckpt`). Translation
can drift on long sequences. Both checkpoints use the **Expert-1-only format** and load exactly like the
[face checkpoint](#face-inference-main-demo): the frozen text/audio expert is reconstructed from the
GLM-4-Voice base given by `--glm_base_path`.

```bash
# cospeech
python inference/inference_body.py \
    --checkpoint ./ViBES-Body-Cospeech \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --user_text "If you had a superpower for one day, what would you choose?" \
    --output_dir ./results/body

# agent: phrase --user_text as the request
python inference/inference_body.py \
    --checkpoint ./ViBES-Body-Agent \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --agent --user_text "Can you show me how you wave hello to a friend?" \
    --output_dir ./results/body
```

Output: `<output_dir>/<...>.mp4` — an SMPL-X body render muxed with synthesized speech, plus
intermediate token / audio files.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `./ViBES-Body-Cospeech` | Stage-2 body checkpoint directory |
| `--glm_base_path` | `THUDM/glm-4-voice-9b` | GLM-4-Voice base for reconstructing the frozen text/audio expert |
| `--user_text` | (prompt) | Text prompt; with `--agent`, the motion request |
| `--agent` | off | Use the instruction (agent) prompt that `ViBES-Body-Agent` was trained with; see [Conditioning prompts](#conditioning-prompts). Ignored with `--audio` |
| `--output_dir` | `./test_output` | Output directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--audio` | — | Audio-to-motion: drive the body with this speech file instead of `--user_text` |
| `--audio_tokenizer` | `THUDM/glm-4-voice-tokenizer` | GLM-4-Voice speech tokenizer (HF id or local dir), used with `--audio` |
| `--global_checkpoint` | `VAE_Global_from_Lower54/last.ckpt` | Global VAE override, e.g. the shape-aware `VAE_Global_uncondMeasure/last.ckpt` together with `--body_betas` |

### GENMO representation (checkpoints not released yet)

The **GENMO body+hand** representation is one full-body VQ-VAE (135D: 21-joint body 6D + global
orient 6D + root local velocity 3D, tokenizer `model_files/pretrained_cpt/VQVAE_0320_GenmoFull/`). It
keeps global translation stable without a Global VAE. Its inference script,
`inference/inference_body_fullbody_genmo.py`, takes the same flags as `inference_body.py`. So does the
GENMO upper/lower ablation `inference/inference_body_upper_lower_genmo.py` (upper-body VQ-VAE + GENMO
lower-body VQ-VAE). Stage-2 checkpoints for both are not released yet.

### Audio-to-motion (your own speech)

All three body scripts can also generate body motion for a **given speech clip** instead of speech
they synthesize from text. Pass `--audio`; `--user_text` is then ignored and no TTS runs:

```bash
python inference/inference_body.py \
    --checkpoint ./ViBES-Body-Cospeech \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --audio speech.wav \
    --output_dir ./results/body
```

The clip (any format librosa reads) is tokenized with the GLM-4-Voice speech tokenizer at 12.5
tokens/s, and the model generates motion after every 26-token audio group, the same layout as the
cospeech training sequences. The rendered video uses the input clip as its soundtrack. With the
Upper+Lower+Hand representation a trailing audio group shorter than 26 tokens gets no motion, so the
video can end up to ~2 s before the clip does.

### Conditioning prompts

The conditioning mode is set by the system message passed to `create_prompt`
(`utils/inference_utils.py`):

- **Cospeech only** (default) — the model is asked to respond with interleaved speech tokens; gestures
  follow from the speech:

  > User will provide you with a text instruction. Do it step by step. First, think about the
  > instruction and respond in an interleaved manner, with 13 text tokens followed by 26 audio tokens.

- **Instruction + cospeech** (`--agent`) — the model is additionally told to *embody and perform* the
  requested motion while speaking (first-person, "imagine you have a body and are already moving …").
  `--agent` (`create_prompt(user_text, agent=True)`) uses this longer system message, verbatim from the
  instruction training data (`AGENT_SYSTEM_MESSAGE`), and the matching assistant prefix
  `streaming_transcription_with_performed_motion`.

---

## Face + Body Inference (co-speech)

Speech, facial expressions and co-speech body motion from one text prompt. This is the README Quick Start:

```bash
python inference/inference_face_body.py \
    --checkpoint_face ./ViBES-Face \
    --checkpoint_body ./ViBES-Body-Cospeech \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --user_text "If you had a superpower for one day, what would you choose?" \
    --output_dir ./results/face_body --front_view
```

1. The face expert answers `--user_text` with speech tokens and face tokens.
2. The body expert generates gestures **for that speech** (the same audio-to-motion path as
   `inference_body.py --audio`), so body, face and audio are synchronized.
3. Face (jaw + 100 expressions) and body (upper / lower; translation from the Global VAE) are merged
   into one SMPL-X 2020 sequence, hands included (decoded by the hand VQ-VAE).

The two 9B models are loaded one after the other, so only one is on the GPU at a time.

| Output | Content |
|---|---|
| `<prefix>.mp4` | SMPL-X render with the speech (`--front_view` for a frontal camera) |
| `<prefix>.npz` | SMPL-X 2020 parameters at 25 fps: `poses` (N, 165), `expressions` (N, 100), `trans` (N, 3), `betas` (300,) |
| `<prefix>.wav` | The generated speech, 22.05 kHz |

`--temperature_face` / `--top_p_face` and `--temperature_body` / `--top_p_body` set the sampling for
each expert; `--output_prefix` names the files.

**Blender rendering.** The `.npz` + `.wav` pair matches the single-person input of
[BlenderRenderKit](https://github.com/Juzezhang/BlenderRenderKit): put them in `<seq>/motion.npz` and
`<seq>/audio.wav`, then render with its `render_single.py`.

---

## Text-to-Motion (HumanML3D)

Plain text → motion on the HumanML3D benchmark (263-D features, MotionGPT VQ-VAE tokens). The
released model, **ViBES-T2M** ([`JuzeZhang/ViBES-T2M`](https://huggingface.co/JuzeZhang/ViBES-T2M)),
puts a ViBES motion expert (~0.27B trainable) on a fully frozen Qwen3-1.7B. It lives in
[`qwen_mome/`](../qwen_mome/README.md) and needs `transformers` ≥ 4.51 (Qwen3), so it runs in its own
environment rather than the ViBES one.

```bash
python -m qwen_mome.generate_t2m \
    --ckpt ./ViBES-T2M/vibes_t2m_qwen_mome_1p7b.pt \
    --vq_ckpt /path/to/MotionGPT/checkpoints/MotionGPT-base/motiongpt_s3_h3d.tar \
    --meta_dir /path/to/MotionGPT/deps/t2m/t2m/t2m/Comp_v6_KLD01/meta \
    --text "a person walks forward, then squats down" \
    --out_dir ./results/t2m
```

Output per caption: `NN_motion.npy` (263-D), `NN_joints.npy` (`[T, 22, 3]`, 20 fps) and a skeleton
`NN.gif`.

| Flag | Default | Description |
|---|---|---|
| `--ckpt` | (required) | `vibes_t2m_qwen_mome_1p7b.pt` from the HF repo |
| `--vq_ckpt` | (required) | MotionGPT's `motiongpt_s3_h3d.tar` (HF `OpenMotionLab/MotionGPT-base`) |
| `--meta_dir` | (required) | `Comp_v6_KLD01/meta` (mean/std) from MotionGPT's `prepare/download_t2m_evaluators.sh` |
| `--text` | (required) | One or more captions |
| `--cfg_scale` | `3.0` | Classifier-free guidance scale |
| `--temperature` / `--top_p` | `0.7` / `0.9` | Sampling; `--greedy` for argmax decoding |

HumanML3D test results (R@1 0.486 / R@3 0.769 / FID 0.265 / MM-Dist 3.196) and training notes are in
[`qwen_mome/README.md`](../qwen_mome/README.md).

---

## Render Mesh NPY Files

Render pre-computed mesh `.npy` files (shape `(T, V, 3)`) to MP4 videos. Auto-detects SMPL (6890 verts) vs. SMPL-X (10475 verts).

```bash
# Basic usage — renders all *_mesh.npy in the directory
python scripts/render_mesh_npy.py --input_dir paper_result/question2motion/motiongpt

# Custom output, resolution, fps
python scripts/render_mesh_npy.py \
    --input_dir paper_result/question2motion/motiongpt \
    --output_dir results/rendered_videos \
    --fps 30 --width 1280 --height 720

# Custom mesh color (RGB, 0-1 range)
python scripts/render_mesh_npy.py \
    --input_dir paper_result/question2motion/motiongpt \
    --color 0.4 0.7 0.9
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--input_dir` | (required) | Directory containing `*_mesh.npy` files |
| `--output_dir` | `<input_dir>/videos` | Where to write `.mp4` files |
| `--fps` | `30` | Output frame rate |
| `--width` | `1280` | Render width |
| `--height` | `720` | Render height |
| `--color` | `0.69 0.39 0.96` | Mesh RGB color |
| `--crf` | `23` | H.264 quality (lower = better) |
| `--pattern` | `*_mesh.npy` | Glob pattern for input files |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--cam_beta` | `2.5` | Camera distance multiplier (lower = closer) |
| `--fixed_camera` | off | Use a single fixed camera for all sequences |
| `--front_view` | off | Frontal eye-level camera |

---

## Audio Tokenization Round-Trip

Encode any audio file into GLM-4-Voice discrete tokens, then decode back to a waveform. Useful for sanity-checking the audio tokenizer:

```bash
# Basic round-trip (outputs to results/audio_roundtrip/)
python scripts/audio_tokenize_roundtrip.py --input path/to/audio.wav

# Save intermediate tokens + custom output dir
python scripts/audio_tokenize_roundtrip.py \
    --input path/to/audio.mp3 \
    --output_dir results/audio_roundtrip \
    --save_tokens
```

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Input audio file (wav, mp3, flac, ...) |
| `--output_dir` | `results/audio_roundtrip` | Output directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--save_tokens` | off | Also save the integer token array as `.npy` |

### Outputs

- `<stem>_reconstructed.wav` — decoded audio from tokens (22050 Hz)
- `<stem>_original_22050hz.wav` — original resampled to 22050 Hz for comparison
- `<stem>_tokens.npy` — integer token array (if `--save_tokens`)

---

## Troubleshooting

**CUDA OOM during face inference** — Lower the max-context length or run on a GPU with more memory; the FLAME renderer holds the full mesh sequence on device.

**Generated video has no audio** — Make sure `ffmpeg` is on your `PATH`; the muxing step shells out to it.

**Mesh appears far away or off-screen in `render_mesh_npy.py`** — Tune `--cam_beta` (try 1.5–4.0) or pass `--front_view`.
