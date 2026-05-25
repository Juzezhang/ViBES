import torchaudio
import torch
import os
import numpy as np
import soundfile as sf
from os.path import join
import argparse
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from pathlib import Path


def _load_wav(path):
    """Load wav using soundfile (avoids torchaudio.load's torchcodec dep on torchaudio>=2.9)."""
    audio_np, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); convert to (channels, samples) to match torchaudio.
    audio = torch.from_numpy(audio_np.T.copy())
    return audio, sample_rate

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(
    description=(
        "Tokenize a folder of .wav files into GLM-4-Voice discrete audio tokens (one .npy per .wav). "
        "Works on any dataset's audio: TFHP <TFHP_ROOT>/audios, BEAT2 <BEAT2_ROOT>/wave16k, "
        "AMASS_talking <AMASS_TALKING_ROOT>/audios_q, etc. Walks the input folder recursively."
    ),
)
parser.add_argument('--wav_folder', type=str, required=True,
                    help="[INPUT] Folder of .wav files (walked recursively).")
parser.add_argument('--output_dir', type=str, required=True,
                    help="[OUTPUT] Where to write the .npy token arrays (mirrors the input folder structure).")
args = parser.parse_args()

wav_folder = args.wav_folder
output_dir = args.output_dir

_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

# Speech tokenizer
whisper_model = WhisperVQEncoder.from_pretrained('THUDM/glm-4-voice-tokenizer').eval().to(device)
feature_extractor = WhisperFeatureExtractor.from_pretrained('THUDM/glm-4-voice-tokenizer')

def extract_speech_token(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, utts):
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = _load_wav(utt)
            audio = audio.cuda()
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to('cuda')
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = 128
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device='cuda',
                                         padding="longest", pad_to_multiple_of=stride)
            features = features.to(device="cuda")
            outputs = model(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
        return all_speech_tokens


# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each .wav file in the provided folder
for root, dirs, files in tqdm(os.walk(wav_folder)):

    for wav_file in files:
        if not wav_file.endswith(".wav"):
            continue

        try:
            wav_path = join(root, wav_file)
            # Compute output path that mirrors input folder structure under output_dir.
            # os.path.relpath handles both flat (root == wav_folder) and nested layouts.
            rel = os.path.relpath(root, wav_folder)
            output_sub_dir = output_dir if rel == "." else join(output_dir, rel)
            output_path = join(output_sub_dir, wav_file.replace(".wav", ".npy"))
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if os.path.exists(output_path):
                print(f"Skipping {wav_file} because it already exists")
                continue

            audio_tokens = extract_speech_token(
                whisper_model, feature_extractor, [wav_path]
            )[0]
            if len(audio_tokens) == 0:
                raise Exception("No audio tokens extracted")
            quantized_array = np.array(audio_tokens)
            # Save the quantized indices
            np.save(output_path, quantized_array)

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

print("Processing complete!")