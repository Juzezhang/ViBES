# ViBES Data Preprocessing

This directory contains all the scripts and tools for preprocessing multimodal datasets used in ViBES training. The preprocessing pipeline handles various datasets including AMASS, BEAT2, TFHP, YouTube videos, and synthetic data.

## 📋 Supported Datasets

### Human Motion Datasets
- **AMASS**: Large-scale human motion capture dataset with SMPL-X annotations
- **BEAT2**: Multimodal dataset with speech, text, and 3D human motion
- **TFHP (Talk For Hours Person)**: Long-form talking head videos
- **HumanML3D**: Don't be confused about this part, this is basically same as AMASS dataset but we want to convert it to HumanML3D representation.


### Video Datasets
- **YouTube Videos**: Web-crawled talking head videos
- **Synthetic Videos**: Generated videos with controlled conditions
- **CANDOR**: Conversational dataset

## 🛠️ Preprocessing Pipeline

### 1. Dataset Download

#### AMASS Dataset

Make sure you have registered at https://smpl-x.is.tue.mpg.de/ and agreed to the SMPLX license terms before running the download script.

```bash
# Download SMPL-X version of AMASS (requires registration)
./preprocess/amass_download.sh

# Download SMPL+H version, for text-to-motion we need smplh that convert to humanml3d format.
./preprocess/amass_download_smplh.sh
```

#### BEAT2: A co-speech gesture dataset. Available from the [BEAT website](https://huggingface.co/datasets/H-Liu1997/BEAT2). We only used the English portion.
####  TFHP: Available from the [website](https://github.com/DiffPoseTalk/DiffPoseTalk/blob/main/datasets/HDTF_TFHP/README.md).
####  Embody3d: a multimodal dataset of 500 individual hours of 3D motion data. Available from the [website](https://github.com/facebookresearch/embody-3d).
#### YouTube: please download our youtube dataset here (coming soon)

## Dataset Structure

Organize your downloaded datasets according to the following directory structure:
```
datasets/
├── AMASS/
    ├── ACCAD/
    ├── BMLhandball/
    ...
├── BEAT2/
    ├── beat_chinese_v2.0.0/
    ├── beat_english_v2.0.0/
    ├── beat_japanese_v2.0.0/
    ├── beat_spanish_v2.0.0/
└── TFHP/
    ├── HDTF_TFHP-lmdb/
    ├── TFHP_raw.zip/
    ├── data/  ## Extracted from TFHP_raw.zip
└── Embody3d/
    ├── c--20250321--1019--WBM368--BVO565--pilot--MotionPrior2--AIAGENT_scene_009--054393-059790/
    ...
└── YouTube_Talking/
    ├── audios/
    ├── FLAME_coeffs_25/
    ├── smplxflame_25/
    ├── talknet_output/
```

### 2. Dataset Preprocess for each dataset

#### 2.1 AMASS Dataset

```bash
python preprocess/dataset_process_amass.py
    --smplx_path "/path/to/your/smplx_models"
    --dataset_path_original "/path/to/your/data"
    --dataset_path_processed "/path/to/your/data"
    --index_path "/path/to/your/index.csv"
    --ex_fps 25
```

For example

```bash
python preprocess/dataset_process_amass.py \
    --smplx_path ./model_files/smplx_models \
    --dataset_path_original /simurgh2/datasets/AMASS_original_smplx \
    --dataset_path_processed /simurgh2/datasets/AMASS \
    --index_path ./preprocess/index.csv \
    --ex_fps 25
```


please download the train/val/test split file(train.txt, val.txt and test.txt) from [HumanML3D](https://github.com/EricGuo5513/HumanML3D/tree/main/HumanML3D) dataset annotation. And the put them to the AMASS root path. And also download the text annotation(texts.zip).


also, please note that the texts_label_index.zip file is already in preprocess/texts_label_index.zip, please extract it. then, process the dataset to match the audio format:

```bash
python preprocess/convert_text_to_transcript_amass.py \
    --root_folder /path/to/AMASS \
    --text_folder /path/to/AMASS/texts \
    --motion_folder /path/to/AMASS/amass_data_align_25 \
    --motion_folder_audio_rotation /path/to/AMASS/amass_data_align_25_audios_rotation \
    --text_folder_audio /path/to/AMASS/texts_for_transcripts \
    --text_label_index_dir /path/to/AMASS/texts_label_index
```

Then download the speaker audio and answer audio on AMASS dataset from our [provided link](https://drive.google.com/drive/folders/1iqjzmgSy7FYQ2OH5ZJMEw2uRkrhq8E0Z?usp=sharing):


After all preprocessing, your AMASS folder structure will look like this:

```
datasets/
├── AMASS_talking/
    ├── amass_data_align_25_audios_rotation/
    ├── texts/
    ├── texts_label_index/
    ├── texts_for_transcripts/
    ├── train.txt/
    ├── val.txt/
    ├── test.txt/
    ├── audios_answer/
    ├── audios_q_token_glm/
    ├── audios_q/
    ├── transcripts_answer/
    ├── transcripts_question/
```

Now the current structure of AMASS is ready for training your motion tokenization. Let's go back to the main page of README and try to train your  motion tokenization. Once you've done, you could use the checkpoint to get the dataset's codebook.

Next, to the motion token by using our tokenization or your own trained tokenization by using the following command:

```bash
python -m scripts.get_compositional_motion_code --cfg configs/config_mixed_stage1_vq_compositional.yaml
```


#### 2.2 BEAT2 Dataset
Motion Data frame rate convertion

For BEAT2 motion data, we will mirror it first

```bash
python preprocess/mirror_motion_beat2.py \
    --smplx_path ./model_files/smplx_models \
    --dataset_path_original /path/to/beat2/smplxflame_30 \
    --dataset_path_processed /path/to/beat2/smplxflame_30_mirror
```

As the original BEAT2 smplx dataset are provided with 30fps, so we need to convert it to 25 fps.
```bash
python preprocess/beat2_motion_fps_converter.py \
    --motion_folder /path/to/beat2/data \
    --output_dir /path/to/output/25fps
```

For example

```bash
python preprocess/beat2_motion_fps_converter.py \
    --motion_folder /simurgh2/datasets/BEAT2/beat_english_v2.0.0/smplxflame_30 \
    --output_dir /simurgh2/datasets/BEAT2/beat_english_v2.0.0/smplxflame_25
```

please also do on the mirror version

```bash
python preprocess/beat2_motion_fps_converter.py \
    --motion_folder /simurgh2/datasets/BEAT2/beat_english_v2.0.0/smplxflame_30_mirror \
    --output_dir /simurgh2/datasets/BEAT2/beat_english_v2.0.0/smplxflame_25_mirror
```


#### 2.3 TFHP Dataset Preprocess

Don't need to much modification for the dataset structure

#### 2.4 Youtube Dataset Preprocess

Here's the original face portation, please download from here:





## 🆘 Troubleshooting

### Common Issues
- **SMPL-X Registration**: Must register at https://smpl-x.is.tue.mpg.de/
- **Memory Errors**: Reduce batch size or use `--num_workers 1`
- **Path Issues**: Use absolute paths for dataset locations

