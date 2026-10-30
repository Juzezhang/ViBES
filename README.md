# ViBES

[![arXiv](https://img.shields.io/badge/arXiv-2512.14234-b31b1b.svg)](https://arxiv.org/pdf/2512.14234)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://ai.stanford.edu/~juze/ViBES/)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/JuzeZhang/ViBES-Face)

This repository contains the official implementation of "ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body".

## 🔍 Overview

ViBES is a speech-language-behavior (SLB) model with a mixture-of-modality-experts (MoME) architecture that ingests audio, motion, or text and shares cross-modal information via speech-language-behavior Attention (SLB-Attn).

![Teaser](./assets/teaser.png)

### Model variants

The speech models come in two sizes that differ only in the frozen speech/text backbone (Expert-0); the motion experts are the same. The same motion-expert design on a frozen text-only LLM gives a text-to-motion model:

| Variant | Frozen backbone (Expert-0) | Notes |
|---|---|---|
| **ViBES (9B)** | [GLM-4-Voice-9B](https://huggingface.co/THUDM/glm-4-voice-9b) | speech + face/body, best quality |
| **ViBES (0.5B)** | [ViBES-Audio](https://huggingface.co/JuzeZhang/ViBES-Audio) — our distilled 0.5B GLM-4-Voice | ~15× smaller backbone, low-latency / on-device |
| **ViBES-T2M (Qwen-MoME)** | [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (text only) | text-to-motion on HumanML3D (R@1 0.486, FID 0.265); [`qwen_mome/`](qwen_mome/README.md), weights at [`JuzeZhang/ViBES-T2M`](https://huggingface.co/JuzeZhang/ViBES-T2M) |


## ✅ TODO List

- [x] Initial code release
- [x] Inference code for conversational behavior of facial expressions
- [x] Inference code for conversational behavior of body — `inference/inference_body.py` (upper+hands+lower via compositional VQs; global translation reconstructed by the Global VAE from lower-body pose and may drift on long sequences). A GENMO-representation upgrade for stable translation is in progress; combined face+body inference: `inference/inference_face_body.py`.
- [x] Body checkpoints — co-speech (`ViBES-Body-Cospeech`) and agent (`ViBES-Body-Agent`, `--agent`), on Google Drive (links in [Environment Setup](#-environment-setup))
- [x] Text-to-motion (HumanML3D) — ViBES motion expert on a frozen Qwen3-1.7B, [`qwen_mome/`](qwen_mome/README.md), weights at [`JuzeZhang/ViBES-T2M`](https://huggingface.co/JuzeZhang/ViBES-T2M)
- [x] Training code for face expert
- [x] Training code for body expert
- [x] Evaluation code for the paper's benchmarks — [docs/4-evaluation.md](docs/4-evaluation.md): Q2M on Converse3D ([evaluation set](docs/1-data/converse3d_eval.md)), TFHP talking head (with `ViBES-Face-TFHP-Eval`), BEAT2 co-speech gestures
- [x] Dataset release (facial part) — [`JuzeZhang/Converse3D`](https://huggingface.co/datasets/JuzeZhang/Converse3D) (tokenized face training set), [`JuzeZhang/YouTube_Talking`](https://huggingface.co/datasets/JuzeZhang/YouTube_Talking), [`JuzeZhang/WebTalk-Synthetic`](https://huggingface.co/datasets/JuzeZhang/WebTalk-Synthetic)
- [x] Dataset release (body part) — the motion can't be redistributed, so we release our AMASS questions/answers/speech (Google Drive) and the Embody3D clip list, plus the recipe to rebuild the released checkpoints' training sets from your own BEAT2 / AMASS / Embody3D: [docs/1-data/body_training_sets.md](docs/1-data/body_training_sets.md)
- [x] Dataset preprocessing — per-dataset guides and scripts in [docs/1-data/](docs/1-data/)


## 🛠️ Environment Setup

Requires Conda, CUDA 12.8+, and GCC 9.0+. Follow installation order carefully.

<details>
<summary><b>Show full installation steps</b></summary>

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
# PyTorch3D — build WITH GPU support (run on a machine that has a GPU + nvcc).
# A CPU-only build silently omits CUDA kernels and rendering later fails with
# "Not compiled with GPU support" at rasterize_meshes.
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install FlashAttention2 (may take 10-30 min to compile)
export MAX_JOBS=4 NVCC_THREADS=1
pip install -U flash-attn --no-build-isolation

# Install Chumpy and apply patch
pip install "chumpy==0.70" --no-build-isolation
chmod +x ./scripts/patch_chumpy_numpy2.sh
./scripts/patch_chumpy_numpy2.sh

# Download GLM-4-Voice components (cosyvoice, speech tokenizer, decoder)
./scripts/download_glm4voice_modules.sh

# Download the GLM-4-Voice base model (provides the frozen text/audio expert; ~18 GB)
huggingface-cli download THUDM/glm-4-voice-9b --local-dir ./model_files/glm-4-voice-9b

# Build resources
./scripts/build_resources.sh

# Download our pretrained face model
# (motion expert only, ~0.86 GB; the frozen text/audio expert is reconstructed from the GLM base at load)
huggingface-cli download JuzeZhang/ViBES-Face --local-dir ./ViBES-Face

# Download the body experts (co-speech + agent, motion expert only, ~0.86 GB each)
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1GsoWZMcZYq-_IOYuKaWWu3bBZClV3E17 -O ./ViBES-Body-Cospeech
gdown --folder https://drive.google.com/drive/folders/1EiN7n2anqbHaKXRj28bOZzIC955qrhhH -O ./ViBES-Body-Agent
```

The body experts are on Google Drive for now ([co-speech](https://drive.google.com/drive/folders/1GsoWZMcZYq-_IOYuKaWWu3bBZClV3E17),
[agent](https://drive.google.com/drive/folders/1EiN7n2anqbHaKXRj28bOZzIC955qrhhH)); a Hugging Face mirror will follow. The Stage-1 tokenizers are in the
Google Drive folder listed in [docs/0-overview.md](docs/0-overview.md), the text-to-motion weights in
[`JuzeZhang/ViBES-T2M`](https://huggingface.co/JuzeZhang/ViBES-T2M) and the 0.5B backbone in
[`JuzeZhang/ViBES-Audio`](https://huggingface.co/JuzeZhang/ViBES-Audio).

**Notes:**
- FlashAttention compilation may take 10-30 minutes
- PyTorch3D requires separate installation

For SMPL-X / FLAME body models, sign up on https://smpl-x.is.tue.mpg.de and https://flame.is.tue.mpg.de, then place the files under `model_files/smplx_models/` and `model_files/FLAME2020/` respectively.

</details>


## 🚀 Quick Start

<details open>
<summary><b>Conversation with text input — speech, face and co-speech body</b></summary>

```bash
python inference/inference_face_body.py \
    --checkpoint_face ./ViBES-Face \
    --checkpoint_body ./ViBES-Body-Cospeech \
    --glm_base_path ./model_files/glm-4-voice-9b \
    --user_text "If you had a superpower for one day, what would you choose?" \
    --output_dir ./results/face_body --front_view
```

The face expert answers with speech and facial expressions. The co-speech body expert then generates
gestures for that speech, so body, face and audio stay in sync. Both checkpoints ship the trained
**motion expert only**: the frozen text/audio expert is rebuilt from the GLM-4-Voice base given by
`--glm_base_path` (default `THUDM/glm-4-voice-9b`, auto-downloaded to the HF cache). The two models are
loaded one after the other, so only one 9B model is on the GPU at a time.

Outputs in `--output_dir`: `<prefix>.mp4` (SMPL-X render with the speech), `<prefix>.npz` (SMPL-X
`poses` / `expressions` / `trans`, 25 fps) and `<prefix>.wav`. Face-only, body-only, audio-driven and
agent (`--agent`) runs are covered in [docs/3-inference.md](docs/3-inference.md).

Demo output

https://github.com/user-attachments/assets/2ecc0816-e0cb-462c-9dcf-e3a6001ad429

*Example output of the command above: speech, face and co-speech body (hands included), rendered in Blender*

</details>


## 📚 Documentation

Full guides live under [`docs/`](docs/index.md):

| Topic | Link |
|---|---|
| Project overview | [docs/0-overview.md](docs/0-overview.md) |
| Dataset preprocessing | [docs/1-data/](docs/1-data/) |
| Training (tokenizers + LLM expert) | [docs/2-training.md](docs/2-training.md) |
| Inference & utilities | [docs/3-inference.md](docs/3-inference.md) |
| Evaluation (how to reproduce the paper's tables) | [docs/4-evaluation.md](docs/4-evaluation.md) |


## Acknowledgements

<details>
<summary><b>Projects we build on</b></summary>

We thank the following projects for sharing their great work.
- **GLM-4-Voice**: https://github.com/zai-org/GLM-4-Voice
- **Language of Motion**: https://github.com/Juzezhang/language_of_motion
- **ARTalk**: https://github.com/xg-chu/ARTalk/tree/main
- **FLAME**: https://flame.is.tue.mpg.de
- **EMICA**: https://github.com/radekd91/inferno

</details>


## 📖 Citation

If you find our work useful in your research, please consider citing:

<details open>
<summary><b>BibTeX</b></summary>

```bibtex
@inproceedings{
      zhang2026vibes,
        title={ViBES: A Conversational Agent with Behaviorally-Intelligent 3D Virtual Body},
      author={Juze Zhang and Changan Chen and Xin Chen and Heng Yu and Tiange Xiang and Ali Sartaz Khan and Shrinidhi Kowshika Lakshmikanth and Ehsan Adeli},
      booktitle={CVPR},
      year={2026},
}
```

</details>
