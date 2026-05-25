#!/usr/bin/env python3
"""
Generate answer audio for Converse3D_eval AMASS test set using GLM-4-Voice.

Reads question text from test.jsonl, generates text+audio response using the
base GLM-4-Voice model, decodes audio tokens to wav via AudioDecoder.

Usage:
    # Single GPU
    python scripts/generate_converse3d_answer_audio.py \
        --data_root /path/to/Converse3D_eval \
        --split test --source amass --gpus 0

    # Multi-GPU (4 GPUs)
    python scripts/generate_converse3d_answer_audio.py \
        --data_root /path/to/Converse3D_eval \
        --split test --source amass --gpus 0,1,2,3
"""
import os
import sys
import json
import hashlib
import argparse
import re
import uuid
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

_script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(_script_dir, ".."))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

speech_related_path = os.path.join(ROOT_DIR, "speech_related")
cosyvoice_path = os.path.join(ROOT_DIR, "speech_related", "cosyvoice")
matcha_path = os.path.join(ROOT_DIR, "speech_related", "Matcha-TTS")
for p in [speech_related_path, cosyvoice_path, matcha_path]:
    if os.path.exists(p):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(1, p)

# ============================================================================
# Constants
# ============================================================================
AUDIO_SAMPLE_RATE = 22050
AUDIO_TOKEN_MIN = 152353
AUDIO_TOKEN_MAX = 168735

SYSTEM_PROMPT = (
    "<|system|>\nUser will provide you with a text instruction. Do it step by step. "
    "First, think about the instruction and respond in a interleaved manner, with 13 "
    "text token followed by 26 audio tokens. Please follow these steps carefully: "
    "1. Think about the instruction first. "
    "2. Respond in an interleaved manner: output 13 text tokens followed by 26 audio tokens. "
    "3. In your reply, imagine that you have a body and are already moving, pretending to "
    "perform 'the motion required by the question. "
    "4. Make sure your answer aligns with both the question and the motion being asked. "
    "5. Remember: the motion is imaginary (pretend), not real. "
    "6. If you describe the motion, use the first-person perspective (e.g., 'my hand,' "
    "'my body,' 'my movement'). Please reply as if you are experiencing and expressing "
    "the motion yourself."
)

AUDIO_DECODER_CONFIG = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "config.yaml")
AUDIO_DECODER_FLOW = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "flow.pt")
AUDIO_DECODER_HIFT = os.path.join(ROOT_DIR, "speech_related", "glm-4-voice-decoder", "hift.pt")


# ============================================================================
# Audio token extraction
# ============================================================================

def extract_audio_tokens(response_text):
    """Extract audio token indices from model response string."""
    audio_pattern = re.compile(r'<\|audio_(\d+)\|>')
    return [int(m.group(1)) for m in audio_pattern.finditer(response_text)]


# ============================================================================
# Generation worker
# ============================================================================

def generate_worker(samples, gpu_id, data_root, split, batch_size):
    """Generate answer audio for a shard of samples on a specific GPU."""
    from transformers import AutoModel, AutoTokenizer
    from speech_related.flow_inference import AudioDecoder

    device = f"cuda:{gpu_id}"
    print(f"  [GPU {gpu_id}] Loading GLM-4-Voice model...")

    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        "THUDM/glm-4-voice-9b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    audio_decoder = AudioDecoder(
        config_path=AUDIO_DECODER_CONFIG,
        flow_ckpt_path=AUDIO_DECODER_FLOW,
        hift_ckpt_path=AUDIO_DECODER_HIFT,
        device=device,
    )

    output_dir = os.path.join(data_root, "segments", split)
    os.makedirs(output_dir, exist_ok=True)

    # Filter already generated
    remaining = []
    for s in samples:
        wav_path = os.path.join(output_dir, f"{s['segment_id']}.wav")
        if not os.path.exists(wav_path):
            remaining.append(s)
    print(f"  [GPU {gpu_id}] {len(remaining)}/{len(samples)} to generate")

    failed = 0
    for idx, s in enumerate(tqdm(remaining, desc=f"GPU {gpu_id}")):
        try:
            prompt = SYSTEM_PROMPT + f"<|user|>\n{s['question']}<|assistant|>streaming_transcription\n"
            inputs = tokenizer([prompt], return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.8,
                    use_cache=True,
                )

            response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
            audio_tokens = extract_audio_tokens(response)

            if not audio_tokens:
                print(f"  [GPU {gpu_id}] [{idx}] {s['segment_id']}: no audio tokens, skipping")
                failed += 1
                continue

            # Decode audio tokens to wav
            tts_token = torch.tensor(audio_tokens, device=device).unsqueeze(0)
            prompt_feat = torch.zeros(1, 0, 80).to(device)
            prompt_token = torch.zeros(1, 0, dtype=torch.int64).to(device)

            with torch.no_grad():
                tts_speech, _ = audio_decoder.token2wav(
                    tts_token,
                    uuid=str(uuid.uuid4()),
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    finalize=True,
                )

            wav_path = os.path.join(output_dir, f"{s['segment_id']}.wav")
            sf.write(wav_path, tts_speech[0].cpu().numpy(), AUDIO_SAMPLE_RATE)

            # Also save answer transcript
            assistant_start = response.find("streaming_transcription\n")
            if assistant_start != -1:
                answer_text = response[assistant_start + len("streaming_transcription\n"):]
                answer_text_clean = re.sub(r'<\|audio_\d+\|>', '', answer_text).strip()
                txt_path = os.path.join(output_dir, f"{s['segment_id']}_answer.txt")
                with open(txt_path, 'w') as f:
                    f.write(answer_text_clean)

        except Exception as e:
            print(f"  [GPU {gpu_id}] [{idx}] {s['segment_id']}: ERROR {e}")
            failed += 1

    print(f"  [GPU {gpu_id}] Done. Failed: {failed}/{len(remaining)}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate answer audio for Converse3D_eval")
    parser.add_argument("--data_root", type=str, default="/path/to/Converse3D_eval")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"])
    parser.add_argument("--source", type=str, default="amass", choices=["amass", "beat2", "all"],
                        help="Which source samples to generate for")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g. '0,1,2,3')")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU for generation")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    num_gpus = len(gpu_ids)

    # Load samples from JSONL
    jsonl_path = os.path.join(args.data_root, f"{args.split}.jsonl")
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if not rec.get("question"):
                continue
            if args.source != "all" and rec.get("source") != args.source:
                continue
            samples.append({
                "segment_id": rec["segment_id"],
                "question": rec["question"],
                "source": rec.get("source", "unknown"),
            })

    # Shuffle deterministically
    samples.sort(key=lambda s: hashlib.md5(s["segment_id"].encode()).hexdigest())
    print(f"Total samples ({args.source}, {args.split}): {len(samples)}")

    # Check existing
    output_dir = os.path.join(args.data_root, "segments", args.split)
    existing = sum(1 for s in samples if os.path.exists(os.path.join(output_dir, f"{s['segment_id']}.wav")))
    print(f"Already generated: {existing}/{len(samples)}")

    if existing == len(samples):
        print("All done!")
        return

    if num_gpus > 1:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        # Shard samples
        shards = [[] for _ in range(num_gpus)]
        for i, s in enumerate(samples):
            shards[i % num_gpus].append(s)

        processes = []
        for rank, gid in enumerate(gpu_ids):
            print(f"  Launching GPU {gid} with {len(shards[rank])} samples")
            p = mp.Process(
                target=generate_worker,
                args=(shards[rank], gid, args.data_root, args.split, args.batch_size),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        generate_worker(samples, gpu_ids[0], args.data_root, args.split, args.batch_size)

    # Final count
    final = sum(1 for s in samples if os.path.exists(os.path.join(output_dir, f"{s['segment_id']}.wav")))
    print(f"\nDone. Generated: {final}/{len(samples)}")


if __name__ == "__main__":
    main()
