# ViBES Documentation

## Getting Started

- [Overview](0-overview.md) — Architecture in one paragraph + pretrained checkpoint downloads

## Data

- [AMASS](1-data/amass.md) — Convert AMASS SMPL-X data to GENMO 145D (Z-up → Y-up), audio tokens, motion tokens, HF datasets
- [BEAT2](1-data/beat2.md) — Convert BEAT2 SMPL-X data to GENMO 145D, audio tokens, motion tokens, HF datasets
- [TFHP Face](1-data/tfhp.md) — TFHP face dataset preprocessing with style control
- [WebTalk-Synthetic](1-data/webtalk_synthetic.md) — Synthetic in-the-wild co-speech face motion (FLAME); audio tokens, face tokens, HF dataset
- [YouTube_Talking](1-data/youtube_talking.md) — In-the-wild talking-head dataset; recipe-only release (URLs + TalkNet results + reproduce-from-video scripts)
- [HumanML3D](1-data/humanml3d.md) — Pointer to the upstream HumanML3D pipeline + expected output layout
- [Embody3D](1-data/embody3d.md) — Meta's Embody3D conversational dataset as a body-expert source (download + HF packing)
- [Combine datasets](1-data/combine.md) — Pack each source to a HF dataset (`preprocess_hf_*`) then concatenate into one unified per-modality training set (`combine_dataset.py`)

## Training

- [Training](2-training.md) — Stage 1 (per-part VQ-VAE tokenizers) + Stage 2 (LLM expert with DeepSpeed); includes token formats, codebook diagnostics, reconstruction validation

## Inference

- [Inference](3-inference.md) — Face / body inference scripts, render mesh NPY, audio round-trip

## Evaluation

- [Evaluation](4-evaluation.md) — Q2M CLIP evaluator (v1/v2/v3), Balanced R-Precision protocol, body metrics

## Developer Reference

- [README](../README.md) — Project overview, environment setup, quick start, and demos
- [CLAUDE.md](../CLAUDE.md) — Claude Code development context
- [AGENTS.md](../AGENTS.md) — Repository guidelines
