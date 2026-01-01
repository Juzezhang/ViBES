#!/usr/bin/env python3
import os
import torch
import numpy as np
import torchaudio
import argparse
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
from peft import PeftModel
from speech_tokenizer.modeling_whisper import WhisperVQEncoder
import time

def extract_speech_tokens(audio_path, whisper_model, feature_extractor, device="cuda:0"):
    """Extract speech tokens from audio file"""
    _resample_buffer = {}
    
    print(f"Processing audio file: {audio_path}")
    with torch.no_grad():
        # Load audio
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.to(device)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            print(f"Converting sample rate from {sample_rate}Hz to 16000Hz...")
            if sample_rate not in _resample_buffer:
                _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=16000
                ).to(device)
            audio = _resample_buffer[sample_rate](audio)
        
        # Take the first channel
        audio = audio[0]
        audio = audio.cpu().numpy()
        
        # Process long audio in segments
        audios, indices = [], []
        time_step = 0
        while time_step * 16000 < audio.shape[0]:
            audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
            audios.append(audio_segment)
            indices.append(0)  # All segments belong to the same audio
            time_step += 30
        
        # Model parameter settings
        pooling_kernel_size = whisper_model.config.pooling_kernel_size or 1
        stride = whisper_model.conv1.stride[0] * whisper_model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_speech_tokens = [[]]  # Only one audio file
        
        # Batch processing
        batch_size = 128
        for start in range(0, len(audios), batch_size):
            end = min(start + batch_size, len(audios))
            print(f"Processing audio segments {start+1}-{end} / {len(audios)}")
            
            features = feature_extractor(audios[start:end], sampling_rate=16000,
                                       return_attention_mask=True, return_tensors="pt", device=device,
                                       padding="longest", pad_to_multiple_of=stride)
            features = features.to(device=device)
            outputs = whisper_model(**features)
            speech_tokens = outputs.quantized_token_ids
            
            # Process attention mask
            attention_mask = features.attention_mask[:, ::whisper_model.conv1.stride[0] * whisper_model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            
            # Collect tokens
            for i in range(speech_tokens.shape[0]):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
    
    return all_speech_tokens[0]

def generate_response(audio_tokens, face_tokens, model, tokenizer, args):
    """Generate response using model"""
    # Convert to model's expected input format
    print(f"Formatting input...")
    audio_str = ''.join([f"<|audio_{token}|>" for token in audio_tokens])
    face_str = ''
    if face_tokens is not None:
        face_str = ''.join([f"<|face_{token}|>" for token in face_tokens])
    
    # Create complete formatted text and system prompt
    formatted_text = f"<|system|>\nThe user will provide an audio conversation along with face motion data. Do it step by step.First, think about the audio conversation and respond in a interleaved manner, with 13 face token followed by 26 audio tokens.\n<|user|>\n {audio_str}{face_str}\n<|assistant|>\n"
    
    # Record input length
    print(f"Number of input tokens: audio={len(audio_tokens)}, face={len(face_tokens) if face_tokens else 0}")
    
    # Tokenize input
    input_ids = tokenizer(formatted_text, return_tensors="pt").input_ids.to(model.device)
    
    # Generate response
    print(f"Generating response...")
    generation_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=min(args.max_length, 2048),
            top_p=args.top_p,
            temperature=args.temperature,
            do_sample=True,
        )
    generation_time = time.time() - generation_start
    print(f"Generation completed, time taken: {generation_time:.2f} seconds")
    
    # Decode output
    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    return response

def extract_tokens_from_response(response_text):
    """Extract audio and face tokens from model response"""
    import re
    
    # Extract audio tokens
    audio_tokens = []
    audio_matches = re.findall(r"<\|audio_(\d+)\|>", response_text)
    for match in audio_matches:
        audio_tokens.append(int(match))
    
    # Extract face tokens
    face_tokens = []
    face_matches = re.findall(r"<\|face_(\d+)\|>", response_text)
    for match in face_matches:
        face_tokens.append(int(match))
    
    return {
        "audio_tokens": audio_tokens,
        "face_tokens": face_tokens,
        "full_response": response_text
    }

def main():
    parser = argparse.ArgumentParser(description='End-to-end audio processing and generation')
    parser.add_argument('--audio', type=str, required=True, 
                       help='Input audio file path')
    parser.add_argument('--model_path', type=str, default="./glm4voice-conversational-agent/final_model", 
                       help='Model path')
    parser.add_argument('--output_dir', type=str, default="./output", 
                       help='Output directory')
    parser.add_argument('--face_tokens', type=str, default=None,
                       help='Face tokens file path (optional)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum generation length')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus sampling) value')
    parser.add_argument('--no_cuda', action='store_true',
                       help='Do not use CUDA')
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda:0"
    print(f"Using device: {device}")
    
    # Set output file name base
    audio_name = os.path.basename(args.audio).split('.')[0]
    output_base = os.path.join(args.output_dir, audio_name)
    
    start_time = time.time()
    print("=== Step 1: Load audio tokenizer ===")
    # Load audio tokenizer
    whisper_model = WhisperVQEncoder.from_pretrained('THUDM/glm-4-voice-tokenizer').eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained('THUDM/glm-4-voice-tokenizer')
    
    print("\n=== Step 2: Extract audio tokens ===")
    # Extract tokens from audio
    audio_tokens = extract_speech_tokens(args.audio, whisper_model, feature_extractor, device)
    print(f"Successfully extracted {len(audio_tokens)} audio tokens")
    np.save(f"{output_base}_input_audio_tokens.npy", np.array(audio_tokens))
    
    # Load face tokens if provided
    face_tokens = None
    if args.face_tokens and os.path.exists(args.face_tokens):
        print(f"Loading face tokens: {args.face_tokens}")
        face_tokens = np.load(args.face_tokens).tolist()
        print(f"Loaded {len(face_tokens)} face tokens")
    
    print("\n=== Step 3: Load GLM model ===")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    tokens = [f"<|face_{i}|>" for i in range(256)]
    tokenizer.add_tokens(tokens, special_tokens=False)
    print(f"Tokenizer vocabulary size after expansion: {len(tokenizer)}")
    
    # Load model
    base_model = AutoModel.from_pretrained(
        "THUDM/glm-4-voice-9b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    
    # Resize model vocabulary
    from transformers import resize_token_embeddings
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.vocab_size = len(tokenizer)
    print(f"Resized model vocabulary size to: {base_model.config.vocab_size}")
    
    # Try to load LoRA weights
    try:
        model = PeftModel.from_pretrained(base_model, args.model_path).to(device)
        print("Successfully loaded LoRA fine-tuned model")
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        print("Falling back to base model")
        model = base_model
    
    # Set to evaluation mode
    model.eval()
    
    print("\n=== Step 4: Generate response ===")
    # Generate response
    response = generate_response(audio_tokens, face_tokens, model, tokenizer, args)
    
    # Extract tokens from response
    extracted = extract_tokens_from_response(response)
    
    print("\n=== Step 5: Save results ===")
    # Save complete response
    with open(f"{output_base}_response.txt", "w") as f:
        f.write(response)
    
    # Save audio tokens
    if extracted["audio_tokens"]:
        np.save(f"{output_base}_output_audio_tokens.npy", np.array(extracted["audio_tokens"]))
        print(f"Saved {len(extracted['audio_tokens'])} generated audio tokens")
    
    # Save face tokens
    if extracted["face_tokens"]:
        np.save(f"{output_base}_output_face_tokens.npy", np.array(extracted["face_tokens"]))
        print(f"Saved {len(extracted['face_tokens'])} generated face tokens")
    
    total_time = time.time() - start_time
    print(f"\nProcessing completed! Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")
    
    # Print token counts
    print("\n=== Generation Results Summary ===")
    print(f"Input: {len(audio_tokens)} audio tokens" + (f", {len(face_tokens)} face tokens" if face_tokens else ""))
    print(f"Output: {len(extracted['audio_tokens'])} audio tokens, {len(extracted['face_tokens'])} face tokens")

if __name__ == "__main__":
    main() 