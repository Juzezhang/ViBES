# Body Training Sets (Released Checkpoints)

How to rebuild the training sets behind the two released body checkpoints from your own copies of
BEAT2, AMASS and Embody3D, and how close a rebuild gets to the sets we trained on.

| Checkpoint | Target ([`combine.md`](combine.md)) | Sources (rows) | Total |
|---|---|---|---|
| `ViBES-Body-Cospeech` | `cospeech` | BEAT2 (2,004) + Embody3D (3,381) | 5,385 |
| `ViBES-Body-Agent` | `full` | BEAT2 (2,004) + AMASS (33,116) + Embody3D (3,381) | 38,501 |

Both sets use the same BEAT2 and Embody3D rows. BEAT2 and AMASS are packed with
`--max_seq_length 4096`, Embody3D with the default 2048. Other values give different rows:
BEAT2 at 2048 splits into 3,482 shorter rows, and Embody3D at 4096 merges into 1,874 longer ones.

## What we release

The motion in all three datasets, and the motion tokens derived from it, may not be
redistributed. You get the datasets from their owners, and we release everything else:

| Asset | Where |
|---|---|
| AMASS_talking questions and answers (text, TTS speech, question audio tokens) | Google Drive, see [`amass.md`](amass.md) Step 6 |
| Embody3D clip list and train/test split | [`preprocess/embody3d_aiagent_split.txt`](../../preprocess/embody3d_aiagent_split.txt) |
| BEAT2 | nothing extra: speech, transcripts and motion all come from BEAT2 |
| Preprocessing and packing scripts | this repo, [`docs/1-data/`](.) |

## Step 1 — Preprocess each source

| Source | Guide | Produces |
|---|---|---|
| BEAT2 | [`beat2.md`](beat2.md) Steps 1, 3, 6, 7 | `<BEAT2_ROOT>/{audios_token_glm,TOKENS_AGENT_25}/` |
| AMASS | [`amass.md`](amass.md) Steps 1–8 | `<AMASS_TALKING_ROOT>/{transcripts_*,audios_q_token_glm,TOKENS_AGENT_25}/` |
| Embody3D | [`embody3d.md`](embody3d.md) Step 1 | `<EMBODY3D_ROOT>/aiagent/c--*/<participant>/tokens/` |

## Step 2 — Pack each source

```bash
python preprocess/preprocess_hf_beat2_dataset_body.py \
    --data_root <BEAT2_ROOT> --split train --max_seq_length 4096 \
    --output_path <OUT>/processed_beat2_body_train

python preprocess/preprocess_hf_amass_dataset_body.py \
    --data_root <AMASS_TALKING_ROOT> --split train --max_seq_length 4096 \
    --output_path <OUT>/processed_amass_body_train

python preprocess/preprocess_embody3d_dataset_body.py \
    --data_root <EMBODY3D_ROOT>/aiagent --split train --max_seq_length 2048 \
    --split_file preprocess/embody3d_aiagent_split.txt \
    --output_path <OUT>/processed_embody3d_body_train
```

- **`--split_file`** pins the 988 training clips (1,098 clips in all). Without it, the packer
  takes a sorted 9:1 split of whatever clips are on disk, so a different download gives a
  different split.
- **The Embody3D log** should report `Using 988/988 train clips`.
- **Row counts:** BEAT2 2,004, AMASS 33,116, Embody3D 3,381.

## Step 3 — Combine

Keep this source order; it is the order of the original sets.

```bash
# ViBES-Body-Cospeech
python preprocess/combine_dataset.py \
    --datasets <OUT>/processed_beat2_body_train/tokenized_dataset \
               <OUT>/processed_embody3d_body_train/tokenized_dataset \
    --output_path <OUT>/processed_body_cospeech_train

# ViBES-Body-Agent
python preprocess/combine_dataset.py \
    --datasets <OUT>/processed_beat2_body_train/tokenized_dataset \
               <OUT>/processed_amass_body_train/tokenized_dataset \
               <OUT>/processed_embody3d_body_train/tokenized_dataset \
    --output_path <OUT>/processed_body_full_train
```

Then train with [`2-training.md`](../2-training.md) (Body Expert).

## How close a rebuild gets

We packed each source with the commands above and compared the result, row by row, with the set
the checkpoints were trained on:

| Source | Rows | Identical | Different |
|---|---|---|---|
| Embody3D | 3,381 / 3,381 | every row, all fields | — |
| BEAT2 | 2,004 / 2,004 | text, masks, position indices; 99.9% of audio tokens, 99.4% of motion tokens | `begin_of_motion` is no longer supervised |
| AMASS | 33,116 / 33,116 | same clips; questions, answers and speech tokens in 99.99% of rows | motion tokens, see below |

- **BEAT2 tokens.** The training set was packed from earlier token files. We re-extracted the
  audio tokens after training, and the motion tokens on disk differ from the packed ones at 0.6%
  of positions.
- **AMASS motion tokens.** The AMASS rows of `ViBES-Body-Agent` came from an earlier
  tokenization run that no longer exists:
  - **Hand tokens:** that run had hand tokens for AMASS, and they were supervised. The current
    pipeline has none for AMASS, so it fills the hand slots with zeros and masks them.
  - **Upper and lower tokens:** 73% are identical.
  - **Motion groups per clip:** the old run kept fewer 13-token groups at the end of each clip.
    Its packer dropped the last partial group, often along with one more. The current packer
    pads the last group instead, so 98% of the rebuilt rows have one or two more groups.
- **Embody3D tokens.** Our Embody3D rows were packed from tokens extracted before
  `preprocess/dataset_process_embody3d_parts.py` existed. That script reproduces the SMPL-X
  assembly exactly, but its 30→25 fps resampling differs slightly near the end of each
  sequence, so tokens you extract will differ a little.

A rebuilt Cospeech set is nearly identical to ours. A rebuilt Agent set has the same clips,
text and speech, but its AMASS motion tokens differ as described above. We have not retrained a
checkpoint on a rebuilt set.

## License

BEAT2, AMASS and Embody3D stay under their own licenses. Get each from its official source and
follow its terms. The released files contain our questions, answers, speech and clip lists only.
