# HumanML3D Preprocessing

## Why HumanML3D is in this repo

HumanML3D is **not used to train any model reported in the ViBES paper**. We include it here only to demonstrate that **the ViBES framework degrades gracefully to the classic text-to-motion task**: if you swap the generated text (audio response transcript) with the *input* text prompt that describes the motion you want, the same MoME architecture and the same body VQ-VAE tokens can be used to do plain text → motion generation as in standard text-to-motion benchmarks. No code change to the model is needed — only the dataset format.

This guide covers the ViBES-specific piece: converting a preprocessed HumanML3D directory into the HuggingFace text-to-motion dataset that ViBES training scripts consume. The upstream HumanML3D pipeline (videos → motion features → text annotations) is **not** documented here — follow [the HumanML3D repository](https://github.com/EricGuo5513/HumanML3D) end-to-end first.

---

## Step 1 — Get the preprocessed HumanML3D dataset

Run the upstream [HumanML3D pipeline](https://github.com/EricGuo5513/HumanML3D).

> ⚠️ **Body model gotcha** — unlike the rest of ViBES (which uses SMPL-X), the HumanML3D upstream pipeline requires **SMPL+H** and **DMPL** body models, *not* SMPL-X. Specifically:
> - **SMPL+H** — download "Extended SMPL+H model used in AMASS project" from [https://mano.is.tue.mpg.de/download.php](https://mano.is.tue.mpg.de/download.php) (file: `smplh.tar.xz`)
> - **DMPL** — download "DMPLs compatible with SMPL" from [https://smpl.is.tue.mpg.de/download.php](https://smpl.is.tue.mpg.de/download.php) (file: `dmpls.tar.xz`)
>
> Both tarballs unpack flat (no top-level dir), so extract each into its own subdirectory:
> ```bash
> mkdir -p HumanML3D/body_models/smplh HumanML3D/body_models/dmpls
> tar -xf smplh.tar.xz -C HumanML3D/body_models/smplh
> tar -xf dmpls.tar.xz -C HumanML3D/body_models/dmpls
> ```
> Final layout: `body_models/smplh/{male,female,neutral}/...` and `body_models/dmpls/{male,female,neutral}/...`. Registration with MPI is required to download.

<details>
<summary>🩹 <b>Running the upstream notebooks on a modern (numpy ≥1.24, multi-GPU) environment</b></summary>

The HumanML3D notebooks predate current numpy/CUDA setups. We hit and fixed the following — apply the same if you run them headless (`jupyter nbconvert --to script *.ipynb`):

1. **AMASS SMPL+H folder names already match** — the SMPL+H download extracts to HumanML3D's expected names (`BioMotionLab_NTroje`, `DFaust_67`, `MPI_HDM05`, `MPI_Limits`, `MPI_mosh`, `SSM_synced`, `Transitions_mocap`, `Eyes_Japan_Dataset`, …), so symlink/copy them straight into `./amass_data/` (no renaming).
2. If `./amass_data/<dataset>` are **symlinks**, pass `os.walk('./amass_data', followlinks=True)` (default doesn't descend symlinks).
3. After nbconvert, delete the `get_ipython()` magic lines; only process `.npz` files in the AMASS walk (skip stray `LICENSE.txt`/`info.txt`).
4. `comp_device` is hardcoded to `cuda:2` in `raw_pose_processing.ipynb` — change to an available device (`cuda:0`).
5. `unzip pose_data/humanact12.zip` before the index/segment loop, and `mkdir -p ./joints` (the segment loop writes there without creating it).
6. numpy ≥1.24 removed `np.float`/`np.int`/`np.bool` — `common/quaternion.py` uses `np.finfo(np.float)`; replace with `np.float64`.
7. The notebooks' `reference1 = np.load('./HumanML3D/…')` lines double-check against the *official* release files (which you won't have on a fresh run) — comment them out.

</details>

After it finishes, your `<HUMANML3D_ROOT>` should look like:

```
<HUMANML3D_ROOT>/
├── new_joint_vecs/        (263D motion features, one .npy per clip)
├── new_joints/            (3D joint positions, one .npy per clip)
├── texts/                 (motion descriptions, one .txt per clip — multiple lines per file = multiple captions)
├── Mean.npy               (per-dim mean over new_joint_vecs)
├── Std.npy                (per-dim std over new_joint_vecs)
├── all.txt                (all clip stems)
├── train.txt              (training split)
├── val.txt                (validation split)
├── test.txt               (test split)
└── train_val.txt          (training + validation combined)
```

## Step 2 — Tokenize motion with the MotionGPT VQ-VAE

Tokenize the HumanML3D motion features into discrete codes. The expected output is a flat folder of one `.npy` per clip stem:

```
<HUMANML3D_ROOT>/TOKENS/
├── 000000.npy          (motion token sequence for clip 000000)
├── 000001.npy
├── ...
├── M000000.npy         (mirror of 000000)
└── ...
```

### Why the MotionGPT VQ-VAE (and not a ViBES body tokenizer)?

HumanML3D motion is stored in the **263-dimensional kinematic feature** format of Guo et al. (root
angular/linear velocity, local joint positions, 6D joint rotations, joint velocities, and foot-contact
labels) — the de-facto standard representation for text-to-motion benchmarks. This is **not** the
representation ViBES uses natively: the ViBES body experts operate on SMPL-X part rotations
(upper/lower/hand) and the released full-body tokenizer on the GENMO 135D vector. ViBES's own body
VQ-VAEs therefore cannot tokenize the 263D features directly.

So for HumanML3D we use the **MotionGPT VQ-VAE** ([OpenMotionLab/MotionGPT](https://github.com/OpenMotionLab/MotionGPT)),
the reference VQ-VAE for the 263D HumanML3D representation. Its architecture is ported into ViBES at
[`multimodal_tokenizers/archs/motiongpt_vq.py`](../../multimodal_tokenizers/archs/motiongpt_vq.py)
(`MotionGPTVQVae` / `MotionGPTVQVaeAdapter` — a 1D-convolutional encoder/decoder with an EMA-reset
codebook, downsampling motion to the token FPS). Using the same tokenizer the text-to-motion
literature uses means:

- the resulting motion tokens are **directly comparable** to standard text-to-motion baselines, and
- it demonstrates the graceful-degradation point above — the ViBES MoME model consumes these standard
  tokens unchanged to do plain text → motion, with no model change.

> ℹ️ The released full-body **GENMO** tokenizer uses this *same* MotionGPT VQ-VAE architecture (see
> [`../2-training.md`](../2-training.md)), but trained on the GENMO 135D representation rather than
> HumanML3D's 263D — i.e. the architecture is shared, the input representation and codebook differ.

## Step 3 — Build the HuggingFace text-to-motion dataset

```bash
# Train split
python preprocess/preprocess_hf_h3d_text2motion.py \
    --data_root   <HUMANML3D_ROOT> \
    --output_path <HUMANML3D_ROOT>/processed_h3d_text2motion_train \
    --split train

# Test split
python preprocess/preprocess_hf_h3d_text2motion.py \
    --data_root   <HUMANML3D_ROOT> \
    --output_path <HUMANML3D_ROOT>/processed_h3d_text2motion_test \
    --split test
```

One sample is created **per (clip, text description)** pair — since HumanML3D usually has 2-4 captions per clip, the resulting dataset has more rows than the number of clips.

After this step, `<HUMANML3D_ROOT>/` should look like:

```
<HUMANML3D_ROOT>/
├── (upstream artifacts from Step 1)
├── TOKENS/                                    (Step 2)
├── processed_h3d_text2motion_train/           (Step 3)
└── processed_h3d_text2motion_test/            (Step 3)
```

---

## Reference

### Sequence layout

The script produces a "pure text-to-motion" sequence (no system prompt, no audio):

```
<|user|> <motion description text> <|assistant|> <begin_of_motion> <motion tokens> <eos>
```

Only the motion tokens after `<begin_of_motion>` are supervised; the user-side text is masked with `-100` labels.

### Output fields (HuggingFace dataset)

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Row index |
| `conv_id` | `str` | `{clip_stem}_{text_idx}_text` |
| `sequence_name` | `str` | `{conv_id}_seq{N}` |
| `num_turns` | `int` | Always 1 for the pure t2m setting |
| `input_ids` | `List[int]` | Full token sequence (text + motion) |
| `attention_mask` | `List[int]` | All 1s |
| `labels` | `List[int]` | `-100` for unsupervised; token id for supervised motion tokens |
| `modality_masks_0` | `List[bool]` | Text-modality positions |
| `modality_masks_1` | `List[bool]` | Audio-modality positions (always False for HumanML3D) |
| `modality_masks_2` | `List[bool]` | Motion-modality positions |
| `position_encoding_indices` | `List[float]` | Sequential `0, 1, 2, ...` positions |

### Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_root` | ✅ | — | `<HUMANML3D_ROOT>` |
| `--output_path` | ✅ | — | Output HF dataset directory |
| `--split` | ✅ | — | `train` / `test` / `val` |
| `--texts_dir` | | `texts` | Subdir under `--data_root` with text descriptions |
| `--lower_dir` | | `TOKENS` | Subdir under `--data_root` with motion tokens |
| `--lower_fps` | | `7.5` | Motion token FPS (after VQ-VAE downsampling) |
| `--model_name` | | `THUDM/glm-4-voice-9b` | Tokenizer model |
| `--max_seq_length` | | `2048` | Max sequence length |
| `--debug` | | False | Process only a small subset |
| `--limit_videos` | | — | Optional cap on number of clips |

Earlier ablation variants live under `preprocess/mics/` and are kept for internal reference; they are not part of the open-source release.
