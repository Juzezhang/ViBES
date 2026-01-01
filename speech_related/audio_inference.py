#!/usr/bin/env python3
import os
import torch
import numpy as np
import torchaudio
import argparse
import time
import re
from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor
from peft import PeftModel
from speech_tokenizer.modeling_whisper import WhisperVQEncoder

def resize_glm_vocab(model, new_size):
    """Adjust the vocabulary size of the GLM model"""
    # Adjust the input embedding layer
    model.resize_token_embeddings(new_size)
    
    # Adjust the output layer
    old_layer = model.transformer.output_layer
    hidden_dim = old_layer.weight.shape[1]
    new_layer = torch.nn.Linear(hidden_dim, new_size, bias=False)
    
    # Copy weights (preserve existing knowledge)
    with torch.no_grad():
        new_layer.weight[:old_layer.weight.shape[0], :] = old_layer.weight.data
    
    # Replace the layer
    model.transformer.output_layer = new_layer
    model.config.vocab_size = new_size
    
    return model

def extract_speech_tokens(audio_path, whisper_model, feature_extractor, device="cuda:0"):
    """
    Extract speech tokens from an audio file
    Referencing the method in get_audio_code.py
    """
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

def chunk_audio_tokens(audio_tokens, max_tokens=800):
    """
    Split long audio tokens into multiple chunks, ensuring each chunk does not exceed the maximum length
    Reference the processing method in the training code
    """
    if len(audio_tokens) <= max_tokens:
        return [audio_tokens]
    
    chunks = []
    for i in range(0, len(audio_tokens), max_tokens):
        chunk = audio_tokens[i:i + max_tokens]
        if len(chunk) > 0:
            chunks.append(chunk)
    
    print(f"Audio tokens total {len(audio_tokens)}, divided into {len(chunks)} chunks, each with maximum {max_tokens} tokens")
    return chunks

def format_conversation_input(audio_tokens, face_tokens=None):
    """
    Format conversation input, converting audio and optional face tokens into model input format
    """
    # Convert to string format expected by the model
    audio_str = ''.join([f"<|audio_{token}|>" for token in audio_tokens])
    
    # Include face tokens if provided
    face_str = ''
    if face_tokens is not None:
        face_str = ''.join([f"<|face_{token}|>" for token in face_tokens])
    
    # Create complete formatted text including system prompt
    formatted_text = f"<|system|>\nThe user will provide an audio conversation along with face motion data. Do it step by step.First, think about the audio conversation and respond in a interleaved manner, with 13 face token followed by 26 audio tokens.\n<|user|>\n {audio_str}{face_str}\n<|assistant|>\n"
    
    return formatted_text

def count_tokens(tokenizer, text):
    """
    Count the number of tokens in text
    """
    return len(tokenizer(text).input_ids)

def extract_tokens_from_response(response_text):
    """
    Extract audio and face tokens from model response
    """
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
    
    # Extract assistant's plain text response (if any)
    assistant_text = ""
    if "<|assistant|>" in response_text:
        assistant_part = response_text.split("<|assistant|>", 1)[1].strip()
        # Remove all token markers, keep only plain text
        assistant_text = re.sub(r"<\|[^|]+\|>", "", assistant_part).strip()
    
    return {
        "audio_tokens": audio_tokens,
        "face_tokens": face_tokens,
        "assistant_text": assistant_text,
        "full_response": response_text
    }

def main():
    parser = argparse.ArgumentParser(description='GLM-4-Voice Inference')
    parser.add_argument('--audio', type=str, required=True, 
                       help='Input audio file path')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the trained checkpoint')
    parser.add_argument('--output_dir', type=str, default="./output", 
                       help='Output directory')
    parser.add_argument('--face_tokens', type=str, default=None,
                       help='Face tokens file path (optional)')
    parser.add_argument('--max_audio_tokens', type=int, default=800,
                       help='Maximum audio tokens per chunk')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum generation length')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p (nucleus sampling) value')
    parser.add_argument('--device', type=str, default="cuda",
                       help='Device: cuda or cpu')
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.audio):
        print(f"Error: Audio file {args.audio} does not exist")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:0"
    print(f"Using device: {device}")
    
    # Set output filename base
    audio_name = os.path.basename(args.audio).split('.')[0]
    output_base = os.path.join(args.output_dir, audio_name)
    
    start_time = time.time()
    
    # Step 1: Load audio tokenizer
    print("=== Step 1: Load Audio Tokenizer ===")
    whisper_model = WhisperVQEncoder.from_pretrained('THUDM/glm-4-voice-tokenizer').eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained('THUDM/glm-4-voice-tokenizer')
    print(f"Audio tokenizer loaded")
    
    # Step 2: Extract audio tokens
    print("\n=== Step 2: Extract Audio Tokens ===")
    audio_tokens = extract_speech_tokens(args.audio, whisper_model, feature_extractor, device)
    print(f"Successfully extracted {len(audio_tokens)} audio tokens")
    np.save(f"{output_base}_input_audio_tokens.npy", np.array(audio_tokens))
    
    # Free audio tokenizer memory
    del whisper_model
    del feature_extractor
    torch.cuda.empty_cache()
    
    # Load face tokens (if provided)
    face_tokens = None
    if args.face_tokens and os.path.exists(args.face_tokens):
        print(f"Loading face tokens: {args.face_tokens}")
        face_tokens = np.load(args.face_tokens).tolist()
        print(f"Loaded {len(face_tokens)} face tokens")
    
    # Step 3: Load GLM model and Checkpoint
    print("\n=== Step 3: Load GLM Model and Checkpoint ===")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-voice-9b", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens
    tokens = [f"<|face_{i}|>" for i in range(256)]
    tokenizer.add_tokens(tokens, special_tokens=False)
    print(f"Extended tokenizer vocabulary size: {len(tokenizer)}")
    
    # Load base model
    base_model = AutoModel.from_pretrained(
        "THUDM/glm-4-voice-9b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    
    # Adjust model vocabulary size
    base_model = resize_glm_vocab(base_model, len(tokenizer))
    print(f"Adjusted model vocabulary size to: {base_model.config.vocab_size}")
    
    # Load LoRA checkpoint
    try:
        model = PeftModel.from_pretrained(base_model, args.checkpoint).to(device)
        print("Successfully loaded LoRA fine-tuned model")
        # Ensure model is in evaluation mode
        model.eval()
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        print("Falling back to base model")
        model = base_model
        model.eval()
    
    # Step 4: Process audio tokens and generate responses
    print("\n=== Step 4: Process Audio and Generate Responses ===")
    
    # Split audio tokens into multiple chunks
    audio_chunks = chunk_audio_tokens(audio_tokens, args.max_audio_tokens)
    
    # Process each chunk and generate responses
    all_responses = []
    all_audio_tokens = []
    all_face_tokens = []
    
    for i, chunk in enumerate(audio_chunks):
        print(f"\nProcessing chunk {i+1}/{len(audio_chunks)} (length: {len(chunk)})")
        
        # Format input
        formatted_input = format_conversation_input(chunk, face_tokens)
        input_token_count = count_tokens(tokenizer, formatted_input)
        print(f"Input token count: {input_token_count}")
        
        # Check input length
        if input_token_count > args.max_length:
            print(f"Warning: Input length {input_token_count} exceeds maximum length {args.max_length}, input will be truncated")
        
        # Generate response
        print(f"Generating response...")
        generation_start = time.time()
        with torch.no_grad():
            input_ids = tokenizer(formatted_input, return_tensors="pt").input_ids.to(device)
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=min(args.max_length, 2048),
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=True,
            )
        
        # Decode output
        response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        generation_time = time.time() - generation_start
        print(f"Generation completed, time: {generation_time:.2f} seconds")
        
        # Extract tokens
        extracted = extract_tokens_from_response(response)
        all_responses.append(response)
        all_audio_tokens.extend(extracted["audio_tokens"])
        all_face_tokens.extend(extracted["face_tokens"])
        
        # Save current chunk results
        chunk_output_base = f"{output_base}_chunk{i+1}"
        with open(f"{chunk_output_base}_response.txt", "w") as f:
            f.write(response)
        np.save(f"{chunk_output_base}_audio_tokens.npy", np.array(extracted["audio_tokens"]))
        if extracted["face_tokens"]:
            np.save(f"{chunk_output_base}_face_tokens.npy", np.array(extracted["face_tokens"]))
        
        print(f"This chunk generated {len(extracted['audio_tokens'])} audio tokens, {len(extracted['face_tokens'])} face tokens")
    
    # Step 5: Save merged results
    print("\n=== Step 5: Save Final Results ===")
    
    # Save all responses
    with open(f"{output_base}_all_responses.txt", "w") as f:
        for i, resp in enumerate(all_responses):
            f.write(f"\n--- Chunk {i+1} Response ---\n")
            f.write(resp)
            f.write("\n")
    
    # Save merged audio tokens
    np.save(f"{output_base}_output_audio_tokens.npy", np.array(all_audio_tokens))
    print(f"Saved a total of {len(all_audio_tokens)} audio tokens")
    
    # Save merged face tokens
    if all_face_tokens:
        np.save(f"{output_base}_output_face_tokens.npy", np.array(all_face_tokens))
        print(f"Saved a total of {len(all_face_tokens)} face tokens")
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\nProcessing complete! Total time: {total_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary
    print("\n=== Final Results ===")
    print(f"Input: {len(audio_tokens)} audio tokens" + (f", {len(face_tokens)} face tokens" if face_tokens else ""))
    print(f"Output: {len(all_audio_tokens)} audio tokens, {len(all_face_tokens)} face tokens")
    print(f"Processed a total of {len(audio_chunks)} chunks")

if __name__ == "__main__":
    main() 