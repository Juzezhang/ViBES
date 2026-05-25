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
| `--checkpoint` | from config | Override the face-expert checkpoint path |
| `--device` | `cuda:0` | GPU device |

Output: `<output_dir>/<timestamp>/result.mp4` plus intermediate token / audio files.

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

## Body Inference

> 🚧 Body inference scripts are pending release. Earlier prototypes used the LOM motion representation, but global translation was unstable under autoregressive generation. The released body pipeline will use the GENMO 145D representation — see the [TODO list](../README.md#-todo-list) for status.

---

## Troubleshooting

**CUDA OOM during face inference** — Lower the max-context length or run on a GPU with more memory; the FLAME renderer holds the full mesh sequence on device.

**Generated video has no audio** — Make sure `ffmpeg` is on your `PATH`; the muxing step shells out to it.

**Mesh appears far away or off-screen in `render_mesh_npy.py`** — Tune `--cam_beta` (try 1.5–4.0) or pass `--front_view`.
