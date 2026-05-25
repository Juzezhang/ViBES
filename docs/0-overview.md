# Overview

ViBES is a **speech-language-behavior (SLB)** model that generates synchronized 3D facial and body motion in response to conversational input. It takes any combination of audio, text, and motion as input and produces co-speech facial expressions and body gestures as output.

## Architecture in One Paragraph

A single transformer backbone with three modality-specific experts (text, audio, motion) shares cross-modal information through **SLB-Attention**. Continuous SMPL-X motion is quantized into discrete tokens by per-part VQ-VAE tokenizers (face / upper / lower / hand / full body), so motion can be predicted by the same autoregressive head that handles text and audio. For body motion we use the **GENMO 145D** root-relative representation for stable global translation under autoregressive generation.

## Pretrained Checkpoints

All Stage 1 tokenizer checkpoints + the Q2M evaluator are hosted on Google Drive:

**📥 [ViBES_pretrained_checkpoints (Google Drive folder)](https://drive.google.com/drive/folders/1pASh6gc6skACDzLKfO8VllAUsZffXCFv)**

Recommended download (mirrors the layout of `model_files/pretrained_cpt/` — drop the downloaded folder in place):

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1pASh6gc6skACDzLKfO8VllAUsZffXCFv \
      --output ./model_files/pretrained_cpt
```

The face Stage 2 expert is hosted separately on Hugging Face (see [README → Environment Setup](../README.md#-environment-setup)):

```bash
huggingface-cli download JuzeZhang/ViBES-Face --local-dir ./ViBES-Face
```

The table below lists every file in the Google Drive folder, in the same layout as `model_files/pretrained_cpt/`:

### Face tokenizer (single version)


| Path under `model_files/pretrained_cpt/` | Size  | Description                                       |
| ---------------------------------------- | ----- | ------------------------------------------------- |
| `face/face.ckpt`                         | 69 MB | Face VQ-VAE — 112D FLAME params → discrete tokens |


### Body tokenizer (two versions — pick one for your pipeline)


| Version                       | Path under `model_files/pretrained_cpt/`                                                                | Size         | Description                                                                                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Legacy (LOM)**              | `body/lom_vq.ckpt`                                                                                      | 69 MB        | Single file containing all 5 parts: face / upper / lower / hand / global. Used by the original Language-of-Motion model.                                                |
| **GENMO Fullbody**            | `VQVAE_0320_GenmoFull/last.ckpt`                                                                        | 287 MB       | Single **135D** stream covering the whole body (body pose + global orient + translation velocity). **Does NOT model SMPL-X shape — betas are dropped; reconstructions use the neutral body.** Load with its shipped `genmo_full.yaml` (`MotionGPTVQVaeAdapter`). Simpler to use; the default for new training. |
| **GENMO Hybrid** *(optional)* | `VQVAE_0318_NormalUpper_GenmoLower/vqvae_normal_upper_last.ckpt` + `vqvar_genmo_lower_global_last.ckpt` | 165 + 315 MB | Two files: Normal Upper (78D, 13 joints × 6D) plus Genmo Lower (61D) with foot contact. Best reconstruction (see `[2-training.md](2-training.md)` for numbers). |


### Q2M evaluator (optional, for the metrics in `[4-evaluation.md](4-evaluation.md)`)


| Path under `model_files/pretrained_cpt/`      | Size   | Description                                                                                   |
| --------------------------------------------- | ------ | --------------------------------------------------------------------------------------------- |
| `evaluator/question_motion_clip_v3_best.ckpt` | 1.3 GB | Q2M CLIP v3 — text↔motion contrastive matcher used for Balanced R-Precision / FID / Diversity |


### Body / external models (must be obtained separately under their own licenses)


| Resource                   | Download site                                                | Target path                                             |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
| SMPL-X v1.1 (NEUTRAL 2020) | [https://smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de) | `model_files/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz` |
| FLAME 2020                 | [https://flame.is.tue.mpg.de](https://flame.is.tue.mpg.de)   | `model_files/FLAME2020/`                                |
| SMPL (neutral, optional)   | [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)     | `model_files/smpl_models/SMPL_NEUTRAL.pkl`              |


SMPL is only needed for the joint-to-SMPL fitting scripts under `preprocess/scripts/fit_batch_sp.py`.

## Where to Go Next

- Install env & download checkpoints → [README](../README.md#-environment-setup)
- Try the demo → [README Quick Start](../README.md#-quick-start)
- Dataset preprocessing → `[1-data/](1-data/)`
- Motion tokenization → `[2-training.md](2-training.md)`
- Training → `[2-training.md](2-training.md)`
- Inference & utilities → `[3-inference.md](3-inference.md)`
- Evaluation → `[4-evaluation.md](4-evaluation.md)`

