# YouTube_Talking Preprocessing

YouTube_Talking is a large in-the-wild conversational dataset: ~2,983 English
single-speaker talking-head videos (interviews + speeches) with derived
multimodal annotations — speech audio, FLAME face motion, SMPL-X body, MANO/HaMeR
hands, active-speaker segments, and motion tokens.

> ⚠️ **Research use only.** The source videos are copyrighted YouTube content and
> are **not redistributed** — download them yourself (Step 1). The release **does
> ship the expensive model-output annotations** (TalkNet, 4D-Humans body, MHR body
> as SMPL-X, FLAME face, fitted SMPL-X, HaMeR hands) so you skip hundreds of GPU-hours; the
> cheap steps (audio extraction, tokenization, HF packing) are reproduced with the
> shipped scripts. No raw video, audio, or rendered frames are redistributed.

## What the release provides

From the Hugging Face dataset [`JuzeZhang/YouTube_Talking`](https://huggingface.co/datasets/JuzeZhang/YouTube_Talking):

```
JuzeZhang/YouTube_Talking/
├── video_urls.csv        (2,983 rows: id,url,language,type,body_parts,num_people,duration_min)
├── download_youtube_talking.py
├── talknet_pywork.tar    (TalkNet active-speaker results — pywork/ per video, ~12 GB)
├── 4d_humans_results.tar (4D-Humans body recovery)
├── smplxflame_25.tar     (fitted SMPL-X body+face @25fps)
├── hamer_results.tar     (HaMeR hands)
├── {train,val,test}_processed.txt    (346 / 9 / 11 fully-processed clip ids)
├── {train,val,test}_unprocessed.txt
├── README.md
└── LICENSE
```

Two more annotation tars are on Google Drive for now (a Hugging Face copy will follow):

```
Google Drive: ViBES_release/YouTube_Talking/
├── FLAME_coeffs_25.tar      (FLAME face coefficients @25 fps, 2,480 clips, 146 GB)
└── smplx_mhr_official.tar   (MHR body converted to SMPL-X, 714 clips, 31 GB)
```

```bash
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1HohvE8Jic1nMh-plLeIf9X3MvWbIarNJ -O <YT_ROOT>
```

- **`FLAME_coeffs_25/<id>.npz`:** `exp` (T×50), `shape` (T×100), `pose` (T×6) at 25 fps.
  This is Step 4's output.
- **`smplx_mhr_official/<id>_mhr_official.npz`:** MHR body converted to SMPL-X with MHR's
  official conversion. Fields: `transl`, `global_orient`, `body_pose`, `left_hand_pose`,
  `right_hand_pose`, `betas`, `expression` and `fitting_errors` (mean ≈ 1 cm), plus
  `frame_ids`, `fps` and `video_total_frames`. Each `_meta.json` holds the same summary.
- **SMPL-X frames:** `frame_ids` index the source video at its own frame rate `fps`. Most
  videos are 24 fps, not 25. 38 videos were processed every 5th frame (`frame_step` in
  `_meta.json`), and 20 cover less than 90% of their video, so compare `frame_ids`
  with `video_total_frames`.
- **Not shipped:** the raw per-frame MHR outputs (`mhr_results/`, ~800 GB).

Extract each `*.tar` in place (`for f in *.tar; do tar -xf "$f"; done`). The
shipped annotations let you skip Steps 2, 4, 5 and 6 below. The skippable steps are
documented for completeness / if you want to regenerate from scratch. All
preprocessing scripts live in this repository (`preprocess/`).

`talknet_pywork.tar` extracts to `talknet_output/<id>/pywork/{scores,tracks,faces,scene}.pckl`
— the active-speaker-detection output that Step 3 consumes, shipped so you can
**skip re-running TalkNet** (the most expensive step).

## Step 0 — Download the release from Hugging Face

```bash
# requires: pip install -U "huggingface_hub[cli]"
huggingface-cli download JuzeZhang/YouTube_Talking --repo-type dataset --local-dir <YT_ROOT>
cd <YT_ROOT> && tar -xf talknet_pywork.tar      # -> talknet_output/<id>/pywork/
```

## Step 1 — Download the source videos

```bash
python preprocess/download_youtube_talking.py \
    --url_csv <YT_ROOT>/video_urls.csv \
    --output_dir <YT_ROOT>/videos \
    --workers 4
```

Saves `<YT_ROOT>/videos/<id>.mp4` (resumable; needs `yt-dlp` + `ffmpeg`). Videos
that have been removed/privated since collection are skipped and listed in
`videos/_missing_ids.txt`.

## Step 2 — Active-speaker detection (TalkNet)

The shipped `talknet_output/<id>/pywork/` lets you **skip this step**. To
regenerate it instead, run [TalkNet-ASD](https://github.com/TaoRuijie-TalkNet-ASD)
on each video; only the `pywork/` outputs (`scores.pckl`, `tracks.pckl`) are
needed downstream.

## Step 3 — Audio, speaking segments, transcripts

```bash
python preprocess/scripts/get_question_transcript_youtube.py \
    --video_dir       <YT_ROOT>/videos \
    --talknet_output  <YT_ROOT>/talknet_output \
    --output_dir      <YT_ROOT>
```

Reads each video + its `pywork/{scores,tracks}.pckl`, then:
- extracts the full 16 kHz mono track → `audios_original/<id>.wav`,
- crops the active-speaker (speaking) audio using the TalkNet segments → `audios/<id>.wav`,
- writes per-video speaking/non-speaking segments → `speaking_segments/<id>_speaking_segments.json`,
- transcribes with Whisper → `transcripts/`.

## Step 4 — Face: FLAME coefficients

Run a per-frame FLAME face tracker on each video to produce
`FLAME_coeffs/<id>.npz` (original video fps; `exp`, `shape`, `pose`). Then
resample to the common 25 fps used by ViBES:

```
FLAME_coeffs/        (tracker output, original fps)
   ↓ resample to 25 fps
FLAME_coeffs_25/     (used by Steps 7-9)
```

## Step 5 — Body: SMPL-X via 4D-Humans + MHR

Run [4D-Humans](https://github.com/shubham-goel/4D-Humans) and MHR on each video
for body recovery (`4d_humans_results/`, `mhr_results/`), then fit the body to
SMPL-X at 25 fps → `smplxflame_25/`. The MHR results converted to SMPL-X are shipped
as `smplx_mhr_official/` (see above).

## Step 6 — Hands: HaMeR

Run [HaMeR](https://github.com/geopavlakos/hamer) for hand recovery →
`hamer_results/<id>/<frame>.pkl`.

## Step 7 — Tokenize audio (GLM-4-Voice)

```bash
PYTHONPATH=./speech_related python preprocess/scripts/get_audio_code_glm.py \
    --wav_folder <YT_ROOT>/audios \
    --output_dir <YT_ROOT>/audios_token_glm
```

## Step 8 — Tokenize motion (VQ-VAE)

Point your VQ config at `<YT_ROOT>` and the body parts present
(`cfg.DATASET.MODALITIES.YouTube_Talking`), then:

```bash
python -m preprocess.scripts.get_compositional_motion_code \
    --cfg configs/<your-vq-compositional-config>.yaml
```

Produces `<YT_ROOT>/TOKENS_AGENT_25/{face,upper,lower,hand}/<id>.npy`.

## Step 9 — Build the HuggingFace training dataset

```bash
# Face expert
python preprocess/preprocess_hf_youtube_dataset_face.py \
    --data_root <YT_ROOT> --output_path <YT_ROOT>/processed_youtube_face_train --split train
# Body expert
python preprocess/preprocess_hf_youtube_dataset_body.py \
    --data_root <YT_ROOT> --output_path <YT_ROOT>/processed_youtube_body_train --split train
```

Splits come from `{train,val,test}_processed.txt` in the release (the
`*_processed.txt` lists are the clips with full multimodal processing).

---

## Reference

### `video_urls.csv`

| Column | Description |
|---|---|
| `id` | clip id (e.g. `202008647`); all modalities are keyed by it |
| `url` | source YouTube URL |
| `language` | always `English` |
| `type` | `interview` (2,140) or `speech` (843) |
| `body_parts` | framing (e.g. `upper body`) |
| `num_people` | always `Single` |
| `duration_min` | approx. video length in minutes |

### Pipeline summary

| Step | Output | Tool |
|---|---|---|
| 1 | `videos/<id>.mp4` | `download_youtube_talking.py` (repo) |
| 2 | `talknet_output/<id>/pywork/` | TalkNet-ASD (external; **shipped**, skip) |
| 3 | `audios/`, `audios_original/`, `speaking_segments/`, `transcripts/` | `get_question_transcript_youtube.py` (repo) |
| 4 | `FLAME_coeffs/` → `FLAME_coeffs_25/` | FLAME tracker (external) + fps resample |
| 5 | `4d_humans_results/`, `mhr_results/`, `smplxflame_25/` | 4D-Humans + MHR (external) |
| 6 | `hamer_results/` | HaMeR (external) |
| 7 | `audios_token_glm/` | `get_audio_code_glm.py` (repo) |
| 8 | `TOKENS_AGENT_25/` | `get_compositional_motion_code.py` (repo) |
| 9 | `processed_youtube_*` | `preprocess_hf_youtube_dataset_{face,body}.py` (repo) |

### License

Research use only. The source videos remain under their original YouTube terms;
this release redistributes none of them.
