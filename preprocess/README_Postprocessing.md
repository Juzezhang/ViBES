# ViBES Data Postprocessing

This directory contains all the scripts and tools for postprocessing multimodal datasets used in ViBES training. The postprocessing pipeline handles various datasets including AMASS, BEAT2, TFHP, YouTube videos, and synthetic data.

## 6. Multimodal Tokenization Processing

This section covers the tokenization process for different modalities (motion and audio) across various datasets.

### Motion Tokenization

Before running motion tokenization, specify the checkpoint paths in the configuration file `configs/config_mixed_stage1_vq_compositional.yaml`:

```yaml
TEST:
  CHECKPOINTS_FACE: './model_files/pretrained_cpt/face/face.ckpt'
  CHECKPOINTS_HAND: /path/to/conversational_agent/model_files/pretrained_cpt/lom_vq_ds_new/lom_vq.ckpt
  CHECKPOINTS_UPPER: /path/to/conversational_agent/model_files/pretrained_cpt/lom_vq_ds_new/lom_vq.ckpt
  CHECKPOINTS_LOWER: /path/to/conversational_agent/model_files/pretrained_cpt/lom_vq_ds_new/lom_vq.ckpt
  CHECKPOINTS_GLOBAL: ''
```

Then extract motion tokens from motion data:

```bash
python -m scripts.get_compositional_motion_code --cfg configs/config_mixed_stage1_vq_compositional.yaml
```

### Audio Tokenization

Extract audio tokens from audio files using the GLM-4-Voice tokenizer:

```bash
python preprocess/get_audio_code.py \
    --wav_folder /path/to/your/audio/folder \
    --output_dir /path/to/output/token/folder
```

**Note:** The script automatically processes all `.mp3` files in subdirectories of the specified `--wav_folder` and saves the quantized audio tokens as `.npy` files in the output directory.

## 7. Sentence Construction for MLLM Model Format

*(To be documented)*

## 8. Dataset Combination from Different Sources

*(To be documented)*
