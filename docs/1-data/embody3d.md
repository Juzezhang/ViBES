# Embody3D Preprocessing

[**Embody 3D**](https://www.meta.com/emerging-tech/codec-avatars/embody-3d/) is Meta's
large-scale multimodal motion-and-behavior dataset (439 participants, ~500 hours, ~54 M
frames of tracked 3D motion with per-participant audio and text annotations, including
multi-person **conversational** scenarios). ViBES uses its assistant-side conversational
clips as a **body-expert** training source (audio + upper/lower/hand motion tokens; no
face).

> ⚠️ **Obtain it from Meta under their license.** Embody3D is **not** redistributed here —
> download it yourself and accept Meta's terms.

## Step 0 — Download Embody3D

- Dataset page (request access + license): <https://www.meta.com/emerging-tech/codec-avatars/embody-3d/>
- Official tools / loaders: <https://github.com/facebookresearch/embody-3d>
- Paper: [Embody 3D (arXiv:2510.16258)](https://arxiv.org/abs/2510.16258)

Embody3D ships per-subset manifests (`datasets/<subset>/dataset.json`) and a downloader
(`src/download.py`, with `--category {charades,daylife,dyadic,hands,locomotion,multiperson,scenarios}`
and `--feat {smplx,audio,text,videos}`). ViBES uses the **`aiagent`** subset (`datasets/aiagent/`) —
the dyadic AI-agent conversational takes, whose names contain `AIAGENT_scene_*` and
`PROXEMICS_AIAGENT_scene_*`. Point the packer's `--data_root` at this `aiagent` directory.

## Step 1 — Motion → SMPL-X parts → tokenize

**1a. Raw SMPL-X components → upper/lower/hand parts.** Embody3D stores each participant's motion as
separate per-component `.npy` files (`smplx_mesh_{global_orient,body_pose,jaw_pose,leye_pose,reye_pose,
left_hand_pose,right_hand_pose,transl,betas}/`). Assemble them into the SMPL-X `poses(165)+trans(3)`
layout, resample 30→25 fps (to match BEAT2/AMASS), and split into the ViBES body parts:

```bash
python -m preprocess.dataset_process_embody3d_parts \
    --data_root    <EMBODY3D_ROOT>/datasets/dyadic \
    --output_dir   <EMBODY3D_ROOT>/embody3d_parts_25 \
    --scene_filter AIAGENT          # restrict to the aiagent conversational captures
```

This writes one `.npz` per (capture, participant) with `upper (n,78)`, `lower (n,61)`, `hand (n,180)`,
`trans`, `betas`.

**1b. Tokenize motion + audio** with the shared tokenizers (same tooling as the other datasets — see
[`beat2.md`](beat2.md) / [`amass.md`](amass.md) and
`preprocess/scripts/get_compositional_motion_code.py` + `get_audio_code_glm.py`). The body
packer below expects this **tokenized** layout under `<EMBODY3D_ROOT>`:

```
<EMBODY3D_ROOT>/
└── c--*/**/
    ├── tokens/
    │   ├── audio/{clip}.npy      # GLM-4-Voice audio tokens (12.5 fps)
    │   ├── upper/{clip}.npy      # upper-body motion tokens  (6.25 fps)
    │   ├── lower/{clip}.npy      # lower-body motion tokens
    │   └── hand/{clip}.npy       # hand motion tokens
    └── audio_separated/
        ├── {clip}.json           # word-level transcript (per speaker)
        └── {clip}.wav            # separated speaker audio
```

## Step 2 — Build the HuggingFace dataset (body expert)

```bash
python preprocess/preprocess_embody3d_dataset_body.py \
    --data_root   <EMBODY3D_ROOT>/aiagent \
    --output_path <EMBODY3D_ROOT>/processed_embody3d_body_train \
    --split train
```

Produces assistant-only training sequences with 3 modalities (text / audio / motion).
Per-group token layout: `text + 26 audio + 1 begin_of_motion + 39 motion`
(13 upper + 13 lower + 13 hand interleaved 1:1:1). Key flags: `--audio_fps 12.5`,
`--{upper,lower,hand}_fps 6.25`, `--max_seq_length`, `--model_name` (GLM-4-Voice tokenizer),
`--split`.

## Step 3 — Combine with the other body sources

Embody3D's HF dataset is concatenated with the other body datasets (BEAT2 / AMASS / …) into
one unified body-expert training set — see [`combine.md`](combine.md).

---

### License

Embody3D is © Meta and governed by Meta's dataset license. This repository ships only the
**preprocessing script**, not any Embody3D data.

Sources: [Meta Embody 3D](https://www.meta.com/emerging-tech/codec-avatars/embody-3d/) ·
[facebookresearch/embody-3d](https://github.com/facebookresearch/embody-3d) ·
[arXiv:2510.16258](https://arxiv.org/abs/2510.16258)
