#!/usr/bin/env python
"""
Inference script to use trained Hugging Face models for multimodal generation.
This script loads a trained model and generates facial expressions from audio tokens.
"""

import os
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from multimodal_tokenizers.utils.token_utils import (
    prepare_multimodal_tokens_for_lm,
    separate_audio_face_tokens
)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with trained Transformers model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with saved model")
    parser.add_argument("--audio_tokens", type=str, required=True, help="Path to audio tokens file (.npy)")
    parser.add_argument("--output_dir", type=str, default="./generated", help="Output directory")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--face_token_offset", type=int, default=1024, help="Offset for face tokens")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability")
    parser.add_argument("--config", type=str, help="Path to config file with fps settings")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load audio tokens
    audio_tokens = np.load(args.audio_tokens)
    
    # Load config if provided
    audio_token_fps = 50
    face_token_fps = 25
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            audio_token_fps = config.get("audio_token_fps", 50)
            pose_fps = config.get("pose_fps", 25)
            unit_length = config.get("unit_length", 1)
            face_token_fps = pose_fps / unit_length
    
    # Prepare input data
    combined_data = prepare_multimodal_tokens_for_lm(
        audio_tokens=audio_tokens,
        face_tokens=None,  # No face tokens for inference (we'll generate them)
        max_sequence_length=args.max_length,
        audio_token_fps=audio_token_fps,
        face_token_fps=face_token_fps,
        face_token_offset=args.face_token_offset,
        mode='concat',
        add_special_tokens=True,
        inference_mode=True  # Important for setting up generation properly
    )
    
    # Convert to tensor and move to device
    input_ids = torch.tensor(combined_data['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(combined_data['attention_mask']).unsqueeze(0).to(device)
    
    # Expected output length (approximately)
    expected_length = min(args.max_length, len(audio_tokens) * (face_token_fps / audio_token_fps) + 50)
    max_new_tokens = int(expected_length)
    
    print(f"Input length: {input_ids.shape[1]}, Expected output length: {expected_length}")
    print(f"Generating {args.num_samples} samples...")
    
    # Generate samples
    all_outputs = []
    for i in range(args.num_samples):
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Extract generated tokens (excluding input)
        generated_sequence = output[0, input_ids.shape[1]:].cpu().numpy()
        
        # Save the generated sequence
        output_path = os.path.join(args.output_dir, f"generated_sample_{i}.npy")
        np.save(output_path, generated_sequence)
        
        # Separate audio and face tokens if needed
        full_sequence = output[0].cpu().numpy()
        audio_tokens_out, face_tokens_out = separate_audio_face_tokens(
            combined_tokens=full_sequence, 
            face_token_offset=args.face_token_offset
        )
        
        # Save separated tokens
        face_tokens_path = os.path.join(args.output_dir, f"face_tokens_sample_{i}.npy")
        np.save(face_tokens_path, face_tokens_out)
        
        all_outputs.append(generated_sequence)
        print(f"Generated sample {i+1}/{args.num_samples}, length: {len(generated_sequence)}")
    
    print(f"Generated samples saved to {args.output_dir}")
    
if __name__ == "__main__":
    main() 