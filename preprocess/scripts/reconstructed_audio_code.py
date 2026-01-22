import torchaudio
import torch
import os
import numpy as np
from os.path import join
import argparse
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, AutoTokenizer
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from flow_inference import AudioDecoder
import uuid
import sys
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")
# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser('exp_motion command line tools')
parser.add_argument('--token_folder', type=str, default="/simurgh/u/juze/datasets/CANDOR/audios_token_glm/", help="Path to the folder containing .wav files")
parser.add_argument('--output_dir', type=str, default="/simurgh/u/juze/datasets/CANDOR/audios_token_glm_reconstructed",
                    help="Directory to save the quantized outputs")
args = parser.parse_args()

token_folder = args.token_folder
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
                audio, sample_rate = torchaudio.load(utt)
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

# Flow & Hift
flow_config = os.path.join("./glm-4-voice-decoder", "config.yaml")
flow_checkpoint = os.path.join("./glm-4-voice-decoder", 'flow.pt')
hift_checkpoint = os.path.join("./glm-4-voice-decoder", 'hift.pt')
audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                hift_ckpt_path=hift_checkpoint,
                device="cuda")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)



start_sec = 1000
end_sec = 1010


# Process each .wav file in the provided folder
for subfolder in tqdm(os.listdir(token_folder)):
    for token_file in os.listdir(join(token_folder, subfolder)):
        if token_file.endswith(".npy"):

            try:
                token_path = join(token_folder, subfolder, token_file)
                output_sub_dir = join(output_dir, subfolder)
                os.makedirs(output_sub_dir, exist_ok=True)
                original_wav_path = join('/simurgh/group/yuheng/CANDOR_processed/', subfolder, token_file.replace(".npy", ".mp3"))
                output_path = join(output_sub_dir, token_file.replace(".npy", ".wav"))

                if os.path.exists(output_path):
                    print(f"Skipping {token_file} because it already exists")
                    continue

                audio_tokens = np.load(token_path)
                if len(audio_tokens) == 0:
                    raise Exception("No audio tokens extracted")

                # Initialize variables before processing files
                this_uuid = str(uuid.uuid4())
                tts_speechs = []
                tts_mels = []
                prev_mel = None
                prompt_speech_feat = torch.zeros(1, 0, 80).to(device)
                flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device)
                # for chunk_idx in range(0, len(audio_tokens), 25):
                for chunk_idx in range(int(start_sec * 12.5), int(end_sec * 12.5) , 25):
                    chunk = audio_tokens[chunk_idx:chunk_idx+25]
                    is_finalize = (chunk_idx + 25 >= len(audio_tokens))
                    # Create tensor from chunk instead of full audio_tokens
                    tts_token = torch.tensor(chunk, device=device).unsqueeze(0)
                    if prev_mel is not None:
                        prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                    tts_speech, tts_mel = audio_decoder.token2wav(tts_token, uuid=this_uuid,
                                                                prompt_token=flow_prompt_speech_token.to(device),
                                                                prompt_feat=prompt_speech_feat.to(device),
                                                                finalize=is_finalize)
                    prev_mel = tts_mel
                    tts_speechs.append(tts_speech.squeeze())
                    tts_mels.append(tts_mel)
                    flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)

                # Save the quantized indices
                # np.save(output_path, quantized_array)
                # Merge and save audio after processing all chunks
                # if tts_speechs:
                final_speech = torch.cat(tts_speechs, dim=-1).cpu()
                torchaudio.save(output_path, final_speech.unsqueeze(0), 22050)
                wav_input, sample_rate = torchaudio.load(original_wav_path)
                torchaudio.save(output_path.replace(".wav", "_input.wav"), wav_input[:, int(sample_rate * start_sec) : int(sample_rate * end_sec)], sample_rate)

            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue

print("Processing complete!")