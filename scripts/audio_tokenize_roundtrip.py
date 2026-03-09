"""
Audio tokenization round-trip: encode any audio file to GLM-4-Voice tokens,
then decode the tokens back to a waveform.

Usage:
    python scripts/audio_tokenize_roundtrip.py --input audio.wav --output_dir results/audio_roundtrip
    python scripts/audio_tokenize_roundtrip.py --input audio.mp3 --output_dir results/audio_roundtrip --save_tokens
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# ── path setup ────────────────────────────────────────────────────────────
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

speech_related_dir = os.path.join(ROOT_DIR, "speech_related")
if speech_related_dir not in sys.path:
    sys.path.insert(0, speech_related_dir)

cosyvoice_dir = os.path.join(speech_related_dir, "cosyvoice")
if os.path.exists(cosyvoice_dir) and cosyvoice_dir not in sys.path:
    sys.path.insert(0, cosyvoice_dir)

matcha_dir = os.path.join(speech_related_dir, "Matcha-TTS")
if os.path.exists(matcha_dir) and matcha_dir not in sys.path:
    sys.path.insert(0, matcha_dir)

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from transformers import WhisperFeatureExtractor
from speech_related.flow_inference import AudioDecoder

# ── constants ─────────────────────────────────────────────────────────────
AUDIO_DECODER_CONFIG = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "config.yaml")
AUDIO_DECODER_FLOW = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "flow.pt")
AUDIO_DECODER_HIFT = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "hift.pt")
OUTPUT_SAMPLE_RATE = 22050

_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


# ── audio I/O helpers ────────────────────────────────────────────────────
def load_audio(path):
    """Load audio file via soundfile. Returns (waveform_tensor [C, T], sample_rate)."""
    data, sr = sf.read(path, dtype="float32")  # (T,) or (T, C)
    if data.ndim == 1:
        data = data[np.newaxis, :]  # (1, T)
    else:
        data = data.T  # (C, T)
    return torch.from_numpy(data), sr


def save_audio(path, waveform, sample_rate):
    """Save waveform tensor [C, T] to wav via soundfile."""
    data = waveform.cpu().numpy()
    if data.ndim == 2:
        data = data.T  # (T, C) for soundfile
    sf.write(str(path), data, sample_rate)


# ── encoder ───────────────────────────────────────────────────────────────
def extract_speech_tokens(model, feature_extractor, audio_path, device):
    """Encode a single audio file into a list of integer token IDs."""
    audio, sample_rate = load_audio(audio_path)
    audio = audio.to(device)

    # resample to 16 kHz if needed
    if sample_rate != 16000:
        if sample_rate not in _resample_buffer:
            _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            ).to(device)
        audio = _resample_buffer[sample_rate](audio)

    # mono, numpy
    audio_np = audio[0].cpu().numpy()

    # chunk into 30-second segments (same as get_audio_code.py)
    segments = []
    time_step = 0
    while time_step * 16000 < audio_np.shape[0]:
        seg = audio_np[time_step * 16000 : (time_step + 30) * 16000]
        segments.append(seg)
        time_step += 30

    # compute stride for padding
    pooling_kernel_size = model.config.pooling_kernel_size or 1
    stride = (
        model.conv1.stride[0]
        * model.conv2.stride[0]
        * pooling_kernel_size
        * feature_extractor.hop_length
    )

    all_tokens = []
    with torch.no_grad():
        for start in range(0, len(segments), 128):
            batch = segments[start : start + 128]
            features = feature_extractor(
                batch,
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt",
                device=str(device),
                padding="longest",
                pad_to_multiple_of=stride,
            ).to(device=device)

            outputs = model(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[
                :, :: model.conv1.stride[0] * model.conv2.stride[0]
            ]
            attention_mask = attention_mask[:, :: model.config.pooling_kernel_size]

            for i in range(len(speech_tokens)):
                tok = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_tokens.extend(tok)

    return all_tokens


# ── decoder ───────────────────────────────────────────────────────────────
def decode_tokens_to_audio(tokens, audio_decoder, device):
    """Decode a list of integer token IDs back to a waveform tensor."""
    token_tensor = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)
    tts_speech = audio_decoder.offline_inference(token_tensor)
    return tts_speech[0]  # (samples,)


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Audio tokenization round-trip (encode -> tokens -> decode)"
    )
    parser.add_argument("--input", type=str, required=True, help="Input audio file path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/audio_roundtrip",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--save_tokens",
        action="store_true",
        help="Also save the intermediate token array as .npy",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    stem = input_path.stem

    # ── 1. Load encoder ──────────────────────────────────────────────────
    print("Loading speech tokenizer (WhisperVQEncoder) ...")
    whisper_model = (
        WhisperVQEncoder.from_pretrained("THUDM/glm-4-voice-tokenizer")
        .eval()
        .to(device)
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        "THUDM/glm-4-voice-tokenizer"
    )

    # ── 2. Encode ────────────────────────────────────────────────────────
    print(f"Encoding: {input_path}")
    tokens = extract_speech_tokens(whisper_model, feature_extractor, str(input_path), device)
    print(f"  -> {len(tokens)} tokens extracted")

    if args.save_tokens:
        token_path = output_dir / f"{stem}_tokens.npy"
        np.save(token_path, np.array(tokens))
        print(f"  -> tokens saved to {token_path}")

    # free encoder memory before loading decoder
    del whisper_model, feature_extractor
    torch.cuda.empty_cache()

    # ── 3. Load decoder ──────────────────────────────────────────────────
    print("Loading audio decoder (flow + hift) ...")
    audio_decoder = AudioDecoder(
        config_path=AUDIO_DECODER_CONFIG,
        flow_ckpt_path=AUDIO_DECODER_FLOW,
        hift_ckpt_path=AUDIO_DECODER_HIFT,
        device=device,
    )

    # ── 4. Decode ────────────────────────────────────────────────────────
    print("Decoding tokens back to audio ...")
    waveform = decode_tokens_to_audio(tokens, audio_decoder, device)
    print(f"  -> waveform shape: {waveform.shape}  sample_rate: {OUTPUT_SAMPLE_RATE}")

    # ── 5. Save ──────────────────────────────────────────────────────────
    out_path = output_dir / f"{stem}_reconstructed.wav"
    save_audio(out_path, waveform.unsqueeze(0), OUTPUT_SAMPLE_RATE)
    print(f"  -> saved to {out_path}")

    # also copy original at same sample rate for easy comparison
    orig_audio, orig_sr = load_audio(str(input_path))
    if orig_sr != OUTPUT_SAMPLE_RATE:
        orig_audio = torchaudio.transforms.Resample(orig_sr, OUTPUT_SAMPLE_RATE)(orig_audio)
    orig_out = output_dir / f"{stem}_original_{OUTPUT_SAMPLE_RATE}hz.wav"
    save_audio(orig_out, orig_audio, OUTPUT_SAMPLE_RATE)
    print(f"  -> original (resampled) saved to {orig_out}")

    print("Done.")


if __name__ == "__main__":
    main()
