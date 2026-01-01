import torchaudio
import torch
import os
import numpy as np
from os.path import join
import argparse
from tqdm import tqdm
from conver_agent.archs.hubert.hubert_tokenizer import HubertTokenizer

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(
            wav, orig_freq=sr, new_freq=16000
        )
    return wav
    
def load_audio_input_tokenize(audio_path, audio_tokenizer, channel_id=None):

    max_wav_chunk=100 * 16_000
    min_wav_chunk=400
    audio = load_audio(audio_path)
    audio = audio.squeeze()
    if len(audio.shape) == 2:
        assert (
            audio.shape[0] == 2
        ), f"expected a stereo wav of shape (2,x), found {audio.shape}"
        if channel_id is None:
            logger.info(
                "Found stereo audio input, averaging audio from 2 channels. If you want to extract"
                "only one channel, set channel_id to 0 or 1"
            )
            audio = audio.mean(0)
        else:
            audio = audio[channel_id]
    assert len(audio.shape) == 1, audio.shape

    hubert_units = []
    for start in range(0, len(audio), max_wav_chunk):
        audio_chunk = audio[start : start + max_wav_chunk]
        if len(audio_chunk) < min_wav_chunk:
            continue
        hubert_units.extend([str(i.item()) for i in audio_tokenizer(audio_chunk)])

    audio_token = torch.tensor(list(map(int, hubert_units)), device=audio_tokenizer.device)

    return audio_token

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser('exp_motion command line tools')
parser.add_argument('--wav_folder', type=str, default="/simurgh/group/yuheng/CANDOR_processed/", help="Path to the folder containing .wav files")
parser.add_argument('--output_dir', type=str, default="/simurgh/u/juze/datasets/CANDOR/audios_token_hubert",
                    help="Directory to save the quantized outputs")
args = parser.parse_args()

wav_folder = args.wav_folder
output_dir = args.output_dir

audio_tokenizer_path = 'model_files/hubert_25hz'
tokenizer = HubertTokenizer(
            hubert_ckpt=join(audio_tokenizer_path, "mhubert_base_25hz.pt"),
            hubert_layer=11,
            quantizer_ckpt=join(audio_tokenizer_path, "L11_quantizer_500.pt"),
            is_linear_quantizer=True,
        ).to(device)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each .wav file in the provided folder
for subfolder in tqdm(os.listdir(wav_folder)):
    for wav_file in os.listdir(join(wav_folder, subfolder)):
        if wav_file.endswith(".mp3"):

            try:
                wav_path = join(wav_folder, subfolder, wav_file)
                output_sub_dir = join(output_dir, subfolder)
                os.makedirs(output_sub_dir, exist_ok=True)
                output_path = join(output_sub_dir, wav_file.replace(".mp3", ".npy"))

                if os.path.exists(output_path):
                    print(f"Skipping {wav_file} because it already exists")
                    continue

                ## encode_units
                # print('\nEncode audio into units (not deduplicated) \n', '-' * 20)
                # units = tokenizer.encode_units(wav_path)
                units = load_audio_input_tokenize(wav_path, tokenizer)
                # print(units)
                quantized_indices = units
                # Convert the string into a list of integers
                # quantized_array = np.array(list(map(int, quantized_indices.split())))
                quantized_array = np.array(quantized_indices.cpu())
                # Save the quantized indices
                np.save(output_path, quantized_array)
                # print(f"Processed and saved tokens for {wav_file}")

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue

print("Processing complete!")