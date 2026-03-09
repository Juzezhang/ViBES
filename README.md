# ViBES

[![arXiv](https://img.shields.io/badge/arXiv-2412.10523-b31b1b.svg)](https://arxiv.org/pdf/2512.14234)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ai.stanford.edu/~juze/ViBES/)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/JuzeZhang/ViBES-Face)

This repository contains the official implementation of "ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body".

## 🔍 Overview

ViBES is a speech-language-behavior (SLB) model with a mixture-of-modality-experts (MoME) architecture that ingests audio, motion, or text and shares cross-modal information via speech-language-behavior Attention (SLB-Attn).

![Teaser](./assets/teaser.png)


## ✅ TODO List

- [x] Initial code release
- [x] Inference code for conversational behavior of facial expressions
- [] Inference code for conversational behavior of body (Note: we use motion representation from previous methods, but experiments show global translation is unstable, so we decided to use the representation from GENMO. We will release the body part when ready.)
- [] Training code for face expert
- [] Training code for body expert
- [] Dataset release (facial part)
- [] Dataset release (body part)
- [] Dataset preprocessing


## 🛠️ Environment Setup

Requires Conda, CUDA 12.8+, and GCC 9.0+. Follow installation order carefully.

```bash

git clone --recurse-submodules https://github.com/Juzezhang/ViBES.git
cd ViBES

# Create environment
conda create --name ViBES -y python=3.10
conda activate ViBES

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4

# Install dependencies
pip install -r requirements.txt
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install FlashAttention2 (may take 10-30 min to compile)
export MAX_JOBS=4 NVCC_THREADS=1
pip install -U flash-attn --no-build-isolation

# Install Chumpy and apply patch
pip install "chumpy==0.70" --no-build-isolation
chmod +x ./scripts/patch_chumpy_numpy2.sh
./scripts/patch_chumpy_numpy2.sh

# Download GLM-4-Voice components
./scripts/download_glm4voice_modules.sh

# Build resources
./scripts/build_resources.sh

# Download our pretrained model
huggingface-cli download JuzeZhang/ViBES-Face --local-dir ./ViBES-Face
```

**Notes:**
- FlashAttention compilation may take 10-30 minutes
- PyTorch3D requires separate installation


## 🚀 Quick Start

<summary><b>Conversation with Text input</b></summary>

```bash
python inference/inference_a2m_face.py --user_text "If you had a superpower for one day, what would you choose?"
```

Demo output

https://github.com/user-attachments/assets/cd0191fa-394d-4476-aec7-c8aed7fe1690

*Example output showing conversational facial animation with synchronized audio*



## Dataset Preprocessing for tokenization

Please refer to:
- `preprocess/dataset_process_amass_genmo.py` to generate GENMO motion vectors (145D) for AMASS.
- `preprocess/dataset_process_beat2_genmo.py` to generate GENMO motion vectors (145D) for BEAT2.

Expected outputs (25 fps):
- `/simurgh2/datasets/AMASS/amass_genmo_25`
- `/simurgh2/datasets/BEAT2/beat_english_v2.0.0/beat2_genmo_25/{smplxflame_25,smplxflame_25_mirror}`

Update dataset roots in `configs/assets.yaml`:
- `DATASET.AMASS_Genmo.ROOT`
- `DATASET.BEAT2_Genmo.ROOT`


## Tokenization Training

Face tokenizer:
```bash
python -m training.train_tokenizer --cfg configs/config_mixed_stage1_face.yaml --nodebug
```

Body tokenizer (GENMO 145D motion_vector):
```bash
python -m training.train_tokenizer --cfg configs/config_mixed_stage1_body_genmo.yaml --nodebug
```

Compositional tokenizer with lower+global (includes translation + betas in lower, 71D):
```bash
python -m training.train_tokenizer --cfg configs/config_mixed_stage1_vq_compositional_lower_global.yaml --nodebug
```

Global VAE from lower_54 (supervise only global loss using local velocity):
```bash
python -m training.train_tokenizer --cfg configs/config_mixed_stage1_vae_global_wo_mesh_lr1e-4.yaml --nodebug
```

Render GT pose + recon translation for the global VAE:
```bash
python -m scripts.render_global_vae_translation --cfg configs/config_mixed_stage1_vae_global_wo_mesh_lr1e-4.yaml
```

## Codebook Health (VQ Diagnostics)

Compute codebook usage stats (perplexity, dead codes, top-k coverage, commitment loss).

From model + dataset (uses checkpoint and dataloader):
```bash
python -m scripts.vq_codebook_stats \
  --cfg configs/config_mixed_stage1_body_genmo.yaml \
  --mode model \
  --split test \
  --topk 0.1,0.05 \
  --max_batches 200 \
  --out_dir experiments/codebook_stats
```

From saved tokens (fast, no loss stats):
```bash
python -m scripts.vq_codebook_stats \
  --cfg configs/config_mixed_stage1_vq_compositional_lower_global.yaml \
  --mode tokens \
  --topk 0.1,0.05 \
  --out_dir experiments/codebook_stats
```

Optional flags: `--parts face,upper,lower,hand,body`, `--ckpt /path/to/ckpt` (single-part override).

## GENMO Reconstruction (Test Set)

Set checkpoints in the config, then render GT vs recon on the chosen split:

```bash
python -m scripts.render_genmo_reconstruction --cfg configs/config_mixed_stage1_body_genmo.yaml
```

For compositional + lower_global:
```bash
python -m scripts.render_genmo_reconstruction --cfg configs/config_mixed_stage1_vq_compositional_lower_global.yaml
```

Outputs MP4s to `experiments/multimodal_tokenizer/VQVAE_Mixed_Genmo/genmo_recon_videos`. The rendering follows the same GENMO/SMPL-X conversion used in `preprocess/render_genmo_converted_amass.py`. Optional knobs: `TEST.NUM_SAMPLES`, `TEST.MAX_SECONDS`, and `TEST.RENDER_CHUNK`.

## Render Mesh NPY Files

Render pre-computed mesh `.npy` files (shape `(T, V, 3)`) to MP4 videos. Auto-detects SMPL (6890 verts) and SMPLX (10475 verts).

```bash
# Basic usage — renders all *_mesh.npy in the directory
python scripts/render_mesh_npy.py --input_dir paper_result/question2motion/motiongpt

# Custom output directory, resolution, and fps
python scripts/render_mesh_npy.py \
    --input_dir paper_result/question2motion/motiongpt \
    --output_dir results/rendered_videos \
    --fps 30 --width 1280 --height 720

# Custom mesh color (RGB, 0-1 range)
python scripts/render_mesh_npy.py \
    --input_dir paper_result/question2motion/motiongpt \
    --color 0.4 0.7 0.9
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | (required) | Directory containing `*_mesh.npy` files |
| `--output_dir` | `<input_dir>/videos` | Output directory for `.mp4` files |
| `--fps` | `30` | Video frame rate |
| `--width` | `1280` | Render width |
| `--height` | `720` | Render height |
| `--color` | `0.69 0.39 0.96` | Mesh RGB color |
| `--crf` | `23` | H.264 compression quality (lower = better) |
| `--pattern` | `*_mesh.npy` | Glob pattern for input files |
| `--device` | `cuda:0` | GPU device |
| `--cam_beta` | `2.5` | Camera distance multiplier (lower = closer) |
| `--fixed_camera` | off | Use a fixed camera position for all sequences |
| `--front_view` | off | Camera faces the front of the SMPL body (eye-level) |

## Audio Tokenization Round-Trip

Encode any audio file into GLM-4-Voice discrete tokens, then decode back to a waveform. Useful for evaluating audio tokenizer reconstruction quality.

```bash
# Basic round-trip (outputs to results/audio_roundtrip/)
python scripts/audio_tokenize_roundtrip.py --input path/to/audio.wav

# Save intermediate tokens as .npy + custom output directory
python scripts/audio_tokenize_roundtrip.py \
    --input path/to/audio.mp3 \
    --output_dir results/audio_roundtrip \
    --save_tokens
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Input audio file (wav, mp3, flac, etc.) |
| `--output_dir` | `results/audio_roundtrip` | Output directory |
| `--device` | `cuda:0` | GPU device |
| `--save_tokens` | off | Also save the intermediate token array as `.npy` |

Outputs:
- `<stem>_reconstructed.wav` — decoded audio from tokens (22050 Hz)
- `<stem>_original_22050hz.wav` — original resampled to 22050 Hz for comparison
- `<stem>_tokens.npy` — integer token array (if `--save_tokens`)

## GENMO Translation Check (GT vs Recon)

Render side-by-side GT vs reconstructed motion to validate the velocity→translation integration:

```bash
python -m scripts.render_genmo_translation_check --cfg configs/config_mixed_stage1_body_genmo.yaml
```

Outputs videos to `experiments/genmo_translation_check/<EXP_NAME>/<timestamp>/<split>/<dataset>/`.


## Dataset Processing for SLB model training


Please refer to xxx


## Acknowledgements

We thank the following projects for sharing their great work.
- **GLM-4-Voice**: https://github.com/zai-org/GLM-4-Voice
- **Language of Motion**: https://github.com/Juzezhang/language_of_motion
- **ARTalk**: https://github.com/xg-chu/ARTalk/tree/main
- **FLAME**: https://flame.is.tue.mpg.de
- **EMICA**: https://github.com/radekd91/inferno


## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@inproceedings{
      zhang2026vibes,
        title={ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body},
      author={Juze Zhang and Changan Chen and Xin Chen and Heng Yu and Tiange Xiang and Ali Sartaz Khan and Shrinidhi Kowshika Lakshmikanth and Ehsan Adeli},
      booktitle={CVPR},
      year={2026},
}
```
