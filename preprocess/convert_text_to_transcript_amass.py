import torchaudio
import torch
import os
import numpy as np
from os.path import join
import argparse
from tqdm import tqdm
from scipy import interpolate
import codecs as cs
from rich.progress import track
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(
    description="Convert HumanML3D text annotations into AMASS-talking transcripts aligned with audio.",
)
parser.add_argument('--root_folder', type=str, required=True,
                    help="[INPUT] AMASS_talking root directory (must contain train.txt / val.txt / test.txt).")
parser.add_argument('--text_folder', type=str, required=True,
                    help="[INPUT] HumanML3D `texts/` folder with one .txt per clip.")
parser.add_argument('--motion_folder', type=str, required=True,
                    help="[INPUT] Processed AMASS motion folder produced by dataset_process_amass.py (e.g. amass_data_align_25).")
parser.add_argument('--motion_folder_audio_rotation', type=str, required=True,
                    help="[OUTPUT] Where to write audio-aligned motion sequences with rotation conversion (e.g. amass_data_align_25_audios_rotation).")
parser.add_argument('--text_folder_audio', type=str, required=True,
                    help="[OUTPUT] Where to write per-clip transcript files ready for TTS (e.g. texts_for_transcripts).")
parser.add_argument('--text_label_index_dir', type=str, required=True,
                    help="[INPUT] Folder of label-index files (the extracted contents of preprocess/texts_label_index.zip).")
args = parser.parse_args()

text_folder = args.text_folder  
motion_folder = args.motion_folder
motion_folder_audio_rotation = args.motion_folder_audio_rotation
text_folder_audio = args.text_folder_audio
text_label_index_dir = args.text_label_index_dir

os.makedirs(motion_folder_audio_rotation, exist_ok=True)
os.makedirs(text_folder_audio, exist_ok=True)

root_folder = args.root_folder

text_dir_amass = text_folder
motion_dir_amass = motion_folder
# audio_dir_amass = pjoin(data_root_amass, 'audios_token')

split_file = join(root_folder, 'train.txt')
# Data id list
id_list_train = []
with cs.open(split_file, "r") as f:
    for line in f.readlines():
        id_list_train.append(line.strip())


# split_file_val = join(root_folder, 'val.txt')
# # Load test data IDs
# id_list_val = []
# with cs.open(split_file_val, "r") as f:
#     for line in f.readlines():
#         id_list_val.append(line.strip())


# split_file_test = join(root_folder, 'test.txt')
# # Load test data IDs
# id_list_test = []
# with cs.open(split_file_test, "r") as f:
#     for line in f.readlines():
#         id_list_test.append(line.strip())

# id_list_amass = id_list_train + id_list_val + id_list_test
id_list_amass = id_list_train

# Iterate through the files to load data
enumerator = enumerate(track(id_list_amass, f"Loading AMASS"))
metadata_amass = []
data_dict_amass = {}

# Fast loading
for i, name in enumerator:
    try:
        # Load motion tokens
        motion_data = np.load(join(motion_dir_amass, f'{name}.npz'))
        poses = motion_data['poses']  # (n_frames, 165)
        expressions = motion_data['expressions']  # (n_frames, 100)
        trans = motion_data['trans']  # (n_frames, 3)
        betas = motion_data['betas']  # (16,)
        mocap_frame_rate = motion_data['mocap_frame_rate']
        surface_model_type = motion_data['surface_model_type']
        gender = motion_data['gender']
        # Read text
        with cs.open(join(text_dir_amass, name + '.txt')) as f:
            text_data = []
            flag = False
            lines = f.readlines()

        index_path = join(text_label_index_dir, name + '.txt')
        with cs.open(index_path, "r") as f:
            index_lines = [line.strip() for line in f.readlines() if line.strip()]
        if not index_lines:
            raise ValueError(f"Empty text label index file: {index_path}")
        index_token = index_lines[0].split()[0].split(",")[0]
        if ":" in index_token:
            index_token = index_token.split(":")[-1]
        try:
            text_label_id = int(index_token)
        except ValueError as exc:
            raise ValueError(f"Invalid text label index '{index_lines[0]}' in {index_path}") from exc
        if text_label_id < 0 or text_label_id >= len(lines):
            raise IndexError(
                f"text_label_id {text_label_id} out of range for {name} (lines={len(lines)})"
            )
        line = lines[text_label_id]
        text_dict = {}
        line_split = line.strip().split('#')
        caption = line_split[0]
        t_tokens = line_split[1].split(' ')
        f_tag = float(line_split[2])
        to_tag = float(line_split[3])
        f_tag = 0.0 if np.isnan(f_tag) else f_tag
        to_tag = 0.0 if np.isnan(to_tag) else to_tag
        text_dict['caption'] = caption
        text_dict['tokens'] = t_tokens
        if f_tag == 0.0 and to_tag == 0.0:
            motion_data_saved = motion_data
            poses_saved = poses
            expressions_saved = expressions
            trans_saved = trans
        else:
            start_index = int(f_tag * mocap_frame_rate)
            end_index = int(to_tag * mocap_frame_rate)
            poses_saved = poses[start_index:end_index]
            expressions_saved = expressions[start_index:end_index]
            trans_saved = trans[start_index:end_index]
            pass

        length_poses = poses_saved.shape[0]

        motion_save_name = name + '.npz'
        motion_data_saved = {}
        motion_data_saved['poses'] = poses_saved
        motion_data_saved['expressions'] = expressions_saved
        motion_data_saved['trans'] = trans_saved
        motion_data_saved['mocap_frame_rate'] = np.array(25)
        motion_data_saved['betas'] = betas
        motion_data_saved['gender'] = gender
        motion_data_saved['surface_model_type'] = surface_model_type
        motion_data_saved['text_label_id'] = text_label_id
        np.savez(join(motion_folder_audio_rotation, motion_save_name), **motion_data_saved)

        text_save_name = name + '.txt'
        with cs.open(join(text_folder_audio, text_save_name), "w") as f:
            f.write(caption)

    except Exception as e:
        print(f"Error processing file {name}: {str(e)}")
        pass
