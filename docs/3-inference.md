# Inference

Run trained ViBES models to generate conversational facial expressions, body motion, and tokenize/decode audio.

## Face Inference (Main Demo)

Generates facial animation + synchronized audio response from a text prompt:

```bash
python inference/inference_face.py \
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
| `--user_text` | (required) | Text prompt to convert into a conversational response |
| `--output_dir` | `results/face/` | Output directory for the generated video |
| `--checkpoint` | `./ViBES-Face` | Path to the face-expert checkpoint directory |
| `--glm_base_path` | `THUDM/glm-4-voice-9b` | GLM-4-Voice base used to reconstruct the frozen text/audio expert (see note) |
| `--device` | `cuda:0` | GPU device |

> **Loading the released checkpoint.** `ViBES-Face` (and the body checkpoint) store **only the trained
> motion expert (~0.86 GB)**, marked by `expert_checkpoint.json`. The inference scripts detect this
> marker and reconstruct the frozen text/audio expert from `--glm_base_path`, merging the two into the
> full model automatically (bit-identical to a full checkpoint). Point `--glm_base_path` at the
> GLM-4-Voice base you downloaded during setup to avoid a duplicate ~18 GB download. Older full
> checkpoints (no marker) still load normally.

Output: `<output_dir>/<timestamp>/result.mp4` plus intermediate token / audio files.

---

## Body Inference

> 🚧 **Status: draft for the upcoming body release.** The face expert is fully released; the body
> expert is being finalized (see the [TODO list](../README.md#-todo-list)). This section documents the
> body inference variants so they can be turned on as soon as the checkpoints are published.

Body generation produces a full-body SMPL-X animation with synchronized speech from a text prompt.
There are **four variants** — every combination of two *conditioning* modes and two *motion
representations*. Pick the row you want in the table below and run the shared command template.

- **Conditioning** — *what drives the motion:*
  - **Cospeech** — gestures emerge from the speech the model generates from your text; there is no
    explicit instruction about which motion to perform (audio-to-motion).
  - **Instruction + cospeech** — the model also follows a text instruction (e.g. "wave hello") while
    staying speech-synchronized, via the instruction system prompt (see [Conditioning prompts](#conditioning-prompts)).
- **Representation** — *how motion tokens decode to SMPL-X:*
  - **GENMO body+hand** — one full-body VQ-VAE (135D: 21-joint body 6D + global orient 6D + root local
    velocity 3D); keeps **global translation stable** (the reason we adopted it). Tokenizer:
    `model_files/pretrained_cpt/VQVAE_0320_GenmoFull/`.
  - **Upper+Lower+Hand** — the original LOM split (separate upper/lower/hand VQ-VAEs + a global
    branch); global translation drifts more than GENMO.

### The four variants

| Conditioning | Representation | Inference script | Checkpoint (`--checkpoint`) |
|---|---|---|---|
| Cospeech | GENMO body+hand | `inference/inference_body_fullbody_genmo.py` | `/path/to/ViBES-Body-Genmo-Cospeech` |
| Cospeech | Upper+Lower+Hand | `inference/inference_body.py` | `/path/to/ViBES-Body-Cospeech` |
| Instruction + cospeech | GENMO body+hand | `inference/inference_body_fullbody_genmo.py` | `/path/to/ViBES-Body-Genmo-Instruct` |
| Instruction + cospeech | Upper+Lower+Hand | `inference/inference_body.py` | `/path/to/ViBES-Body-Instruct` |

<!-- TODO(body release): replace the /path/to/ViBES-Body-* placeholders with the published HF repos.
     Internal training runs that back these rows (private — do not commit real paths):
       cospeech/upper+lower+hand = rotation_body_a2m_v6   instruction/upper+lower+hand = rotation_body_v6
       GENMO = fullbody_genmo_v1 (+ upper_lower_genmo_v1 for the upper/lower GENMO split)
     Confirm which GENMO run is cospeech vs instruction before filling the GENMO rows. -->

All four share the same Stage-2 MoME backbone and the **Expert-1-only checkpoint format** (loaded
exactly like the [face checkpoint](#face-inference-main-demo): the frozen text/audio expert is
reconstructed from the GLM-4-Voice base via `--glm_base_path`). To run a variant, drop its **script**
and **checkpoint** from the table into this template:

```bash
python <inference-script-from-the-table> \
    --checkpoint <checkpoint-from-the-table> \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --user_text "If you had a superpower for one day, what would you choose?" \
    --output_dir ./results/body
```

For an **instruction + cospeech** checkpoint, phrase `--user_text` as the instruction itself
(e.g. `"Wave hello, then tell me about your day."`).

Output: `<output_dir>/<...>.mp4` — an SMPL-X body render muxed with synthesized speech, plus
intermediate token / audio files.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (from table) | Stage-2 body checkpoint directory |
| `--glm_base_path` | `THUDM/glm-4-voice-9b` | GLM-4-Voice base for reconstructing the frozen text/audio expert |
| `--user_text` | (prompt) | Text prompt, or the motion instruction for instruction variants |
| `--output_dir` | `./results/body` | Output directory |
| `--device` | `cuda:0` | GPU device |

> **Ablation variant:** the GENMO upper/lower split (`inference/inference_body_upper_lower_genmo.py` —
> standard upper-body VQ-VAE + GENMO lower-body VQ-VAE) is available but is not part of the four-variant
> release matrix.

### Conditioning prompts

The conditioning mode is set by the system message passed to `create_prompt`
(`utils/inference_utils.py`):

- **Cospeech only** (default) — the model is asked to respond with interleaved speech tokens; gestures
  follow from the speech:

  > User will provide you with a text instruction. Do it step by step. First, think about the
  > instruction and respond in an interleaved manner, with 13 text tokens followed by 26 audio tokens.

- **Instruction + cospeech** — the model is additionally told to *embody and perform* the requested
  motion while speaking (first-person, "imagine you have a body and are already moving …"). Pass this
  longer system message via `create_prompt(user_text, system_message=...)`.

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
| `--device` | `cuda:0` | GPU device |
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
| `--device` | `cuda:0` | GPU device |
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
