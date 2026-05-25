#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess H3D dataset for text-to-motion (t2m) generation V3 format WITHOUT SYSTEM PROMPT,
organizing unified motion tokens and saving in HuggingFace datasets format.

V3 Features:
- Motion-only modality: Only motion tokens (NO text or audio in response)
- V3 format: 3 modalities (text, audio, motion) instead of 6
- Position encoding: Simple sequential indices (no interpolation)
- Text-to-motion format: <|user|> text description <|assistant|> motion tokens only
- Motion tokens FPS: Lower body 7.5
- Supervision: Lower body motion tokens supervised as unified motion modality

This script targets the H3D dataset at
  /path/to/datasets/HumanML3D
and assumes:
- Text descriptions under data_root/texts/*.txt (multiple descriptions per motion)
- Lower body tokens under data_root/TOKENS_AGENT_25_H3D (supervised)
- Split information in data_root/{split}.txt

Key features:
- NO SYSTEM PROMPT (woprompt = without prompt)
- Motion-only generation: Only motion tokens in response (no text, no audio)
- V3 format: 3 modalities (text, audio, motion) - motion in modality 2 only
- Supports train/test/val splits based on split files in the dataset root
- Creates one sample per text description with corresponding motion tokens
- Pure t2m setting: text input -> motion output only
- Supervision: Lower body motion tokens supervised only

Usage:
    python preprocess_hf_h3d_text2motion.py \
        --data_root /path/to/datasets/HumanML3D \
        --output_path ./processed_h3d_v3 \
        --split train
"""

import os
import json
import numpy as np
from pathlib import Path
import re
import logging
from tqdm import tqdm
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from typing import Optional, Tuple, Union, List, Dict, Any


def calculate_position_encoding_indices_simple(seq_length):
    """
    Calculate simple sequential position encoding indices for t2m dataset.
    Unlike other datasets, this one doesn't need complex interpolation.
    Each token gets its sequential position as the encoding index.
    
    Args:
        seq_length: Length of the sequence
    
    Returns:
        List of float values representing sequential position indices.
        For t2m dataset, position_index[i] = float(i)
    """
    # Simple sequential indexing - no interpolation needed
    return [float(i) for i in range(seq_length)]


def compute_rotary_embeddings_from_precomputed_indices(
    rotary_embedding_module,
    position_indices,
    base: int = 10000
):
    """
    Compute actual rotary embeddings from precomputed position indices.
    This function can be used in the model to efficiently compute rotary embeddings
    using the position indices stored in the dataset.
    
    Args:
        rotary_embedding_module: The RotaryEmbedding module instance
        position_indices: List or tensor of precomputed position indices
        base: Base value for rotary embeddings computation
        
    Returns:
        Rotary position embeddings with shape [seq_length, dim, 2]
        
    Usage example in model:
        # Load precomputed indices from dataset
        position_indices = batch["position_encoding_indices"]
        
        # Compute rotary embeddings efficiently
        rotary_pos_emb = compute_rotary_embeddings_from_precomputed_indices(
            self.rotary_pos_emb, position_indices
        )
    """
    import torch
    
    if isinstance(position_indices, list):
        position_indices = torch.tensor(position_indices, dtype=torch.float32)
    
    seq_length = len(position_indices)
    
    # Prepare for angle-based computation
    base_with_ratio = base * rotary_embedding_module.rope_ratio
    inv_freq = 1.0 / (base_with_ratio ** (torch.arange(
        0, rotary_embedding_module.dim, 2, 
        dtype=rotary_embedding_module.inv_freq.dtype, 
        device=rotary_embedding_module.inv_freq.device
    ) / rotary_embedding_module.dim))
    
    # Initialize the final rotary embeddings
    final_rope = torch.zeros(
        (seq_length, rotary_embedding_module.dim // 2, 2),
        dtype=rotary_embedding_module.inv_freq.dtype,
        device=rotary_embedding_module.inv_freq.device
    )
    
    # Move position indices to the same device
    position_indices = position_indices.to(device=rotary_embedding_module.inv_freq.device)
    
    # Compute rotary embeddings for each position
    for i, pos_idx in enumerate(position_indices):
        theta = pos_idx * inv_freq
        final_rope[i, :, 0] = torch.cos(theta)
        final_rope[i, :, 1] = torch.sin(theta)
    
    return final_rope


def safe_token_to_string(token, modality):
    """
    Safely convert token to string format.
    
    Args:
        token: Token value (could be int, float, list, or numpy array)
        modality: Type of token ("face", "upper", "lower", or "hand")
        
    Returns:
        String representation of the token
    """
    if isinstance(token, (list, tuple)):
        # For nested lists/tuples, flatten and take first element
        while isinstance(token, (list, tuple)) and len(token) > 0:
            token = token[0]
    
    # Convert to int if numeric
    if isinstance(token, (int, float)):
        token_int = int(token)
    else:
        try:
            token_int = int(token)
        except:
            # If conversion fails, use 0 as fallback
            print(f"Warning: Could not convert {modality} token to int: {token}, using 0")
            token_int = 0
    
    if modality == "face":
        return f"<|face_{token_int}|>"
    elif modality == "upper":
        return f"<|upper_{token_int}|>"
    elif modality == "lower":
        return f"<|motion_{token_int}|>"  # Use motion_ prefix for unified motion modality
    elif modality == "hand":
        return f"<|hand_{token_int}|>"
    elif modality == "motion":
        return f"<|motion_{token_int}|>"  # Direct motion modality support
    else:
        raise ValueError(f"Unknown modality: {modality}")



def process_full_video(sequence_id, texts_dir, lower_dir, lower_fps=7.5):
    """
    Process text descriptions and motion tokens into format suitable for t2m dataset creation.
    Only motion tokens are used (no text or audio in response).
    
    Args:
        sequence_id: Sequence ID
        texts_dir: Directory containing text descriptions
        lower_dir: Directory containing lower body motion tokens
        lower_fps: Lower body motion token frame rate
        
    Returns:
        List of processed segments with text descriptions and motion tokens only
    """
    processed_segments = []

    # Load text descriptions and motion tokens for t2m format (motion-only output)
    text_file = os.path.join(texts_dir, f"{sequence_id}.txt")
    lower_token_file = os.path.join(lower_dir, f"{sequence_id}.npy")
    
    # Check if required files exist
    if not os.path.exists(text_file):
        logging.warning(f"Text file not found: {text_file}")
        return []
    
    # Must have motion token file (only motion tokens are used)
    if not os.path.exists(lower_token_file):
        logging.warning(f"No motion token file found for {sequence_id}")
        return []
    
    # Load text descriptions
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_lines = f.readlines()
        logging.info(f"Loaded {len(text_lines)} text descriptions for {sequence_id}")
    except Exception as e:
        logging.error(f"Error loading text descriptions for {sequence_id}: {e}")
        return []
    
    
    # Load motion tokens only (this is the only modality in the output)
    try:
        lower_data = np.load(lower_token_file, allow_pickle=True)
        if lower_data.ndim > 1:
            lower_data = lower_data[0]
        logging.info(f"Loaded {len(lower_data)} motion tokens for {sequence_id}")
            
    except Exception as e:
        logging.error(f"Error loading motion tokens for {sequence_id}: {e}")
        return []
    
    # Process each text description to create text-motion pairs
    for line_idx, line in enumerate(text_lines):
        line = line.strip()
        if not line:
            continue
            
        # Parse text format: description#tokens#start_time#end_time
        parts = line.split('#')
        if len(parts) != 4:
            logging.warning(f"Invalid line format in {text_file}, line {line_idx}: {line}")
            continue
            
        description = parts[0].strip()
        try:
            f_tag = float(parts[2])
            to_tag = float(parts[3])
            start_time = 0.0 if np.isnan(f_tag) else f_tag
            end_time = 0.0 if np.isnan(to_tag) else to_tag
        except (ValueError, IndexError):
            start_time = 0.0
            end_time = 0.0
        
        # Extract lower body motion tokens based on time
        if start_time == 0.0 and end_time == 0.0:
            # Use entire sequence
            motion_lower_tokens = lower_data.tolist()
        else:
            # Extract time segment using lower_fps (7.5)
            lower_start_idx = int(start_time * lower_fps)
            lower_end_idx = int(end_time * lower_fps)
            
            lower_start = max(0, min(lower_start_idx, len(lower_data)))
            lower_end = max(0, min(lower_end_idx, len(lower_data)))
            motion_lower_tokens = lower_data[lower_start:lower_end].tolist() if lower_start < lower_end else []
        
        # Check if we have valid motion tokens
        if len(motion_lower_tokens) == 0:
            logging.warning(f"No motion tokens for {sequence_id}, line {line_idx}")
            continue
        
        # Create segment in compatible format for the convert function (motion tokens only)
        processed_segments.append({
            "segment_id": f"{sequence_id}_{line_idx}",
            "video_id": sequence_id,
            "start_time": start_time,
            "end_time": end_time,
            "transcripts_question": description,  # Text description as input
            "transcripts_answer": "<|assistant|>\n",  # Assistant prefix for t2m
            "lower_tokens": motion_lower_tokens,  # Only motion tokens in output
            "audio_question_tokens": []  # No audio in t2m (motion-only)
        })
        
        logging.info(f"Created t2m pair {sequence_id}_{line_idx}: '{description[:50]}...' -> {len(motion_lower_tokens)} motion tokens")

    return processed_segments
def process_chunk_to_record(chunk, conv_id, tokenized_records):
    """
    Process a chunk of turns into a tokenized record with unified motion modality V3 support.
    V3 format: 3 modalities instead of 6 (modality 0: text, modality 1: audio, modality 2: unified motion)
    
    Args:
        chunk: Dictionary containing input_ids, labels, turns, modality_masks_0, modality_masks_1, modality_masks_2
        conv_id: Conversation ID
        tokenized_records: List to append the processed record to
    """
    # Sequences are split before this point based on tokenizer lengths.
    # Do not truncate here; rely on upstream splitting to respect max_seq_length.
    
    # Ensure all arrays have the same length
    seq_len = len(chunk["input_ids"])
    if len(chunk["labels"]) != seq_len:
        logging.warning(f"Labels length mismatch, truncating to {seq_len}")
        chunk["labels"] = chunk["labels"][:seq_len]
    if len(chunk["modality_masks_0"]) != seq_len:
        logging.warning(f"Modality_masks_0 length mismatch, truncating to {seq_len}")
        chunk["modality_masks_0"] = chunk["modality_masks_0"][:seq_len]
    if len(chunk["modality_masks_1"]) != seq_len:
        logging.warning(f"Modality_masks_1 length mismatch, truncating to {seq_len}")
        chunk["modality_masks_1"] = chunk["modality_masks_1"][:seq_len]
    if len(chunk["modality_masks_2"]) != seq_len:
        logging.warning(f"Modality_masks_2 length mismatch, truncating to {seq_len}")
        chunk["modality_masks_2"] = chunk["modality_masks_2"][:seq_len]
    
    # Create attention mask (all 1s since we don't have padding)
    attention_mask = [1] * seq_len
    
    # Calculate position encoding indices for this sequence
    # For t2m dataset, we use simple sequential indexing - no complex interpolation needed
    # Each token position corresponds directly to its sequence index
    position_encoding_indices = calculate_position_encoding_indices_simple(seq_len)
    
    # Validate position encoding indices
    assert len(position_encoding_indices) == seq_len, f"Position indices length {len(position_encoding_indices)} != sequence length {seq_len}"
    
    # Log statistics for validation
    pos_min, pos_max = min(position_encoding_indices), max(position_encoding_indices)
    # logging.debug(f"Calculated sequential position encoding indices for sequence of length {seq_len}, range: [{pos_min:.3f}, {pos_max:.3f}]")
    
    # Generate sequence name for this record
    sequence_name = f"{conv_id}_seq{str(len(tokenized_records)).zfill(2)}"
    
    tokenized_record = {
        "id": len(tokenized_records),
        "conv_id": conv_id,
        "sequence_name": sequence_name,
        "num_turns": len(chunk["turns"]),
        "input_ids": chunk["input_ids"],
        "attention_mask": attention_mask,
        "labels": chunk["labels"],
        "modality_masks_0": chunk["modality_masks_0"],  # True for text tokens (modality 0)
        "modality_masks_1": chunk["modality_masks_1"],  # True for audio tokens (modality 1)
        "modality_masks_2": chunk["modality_masks_2"],  # True for unified motion tokens (modality 2)
        "position_encoding_indices": position_encoding_indices,  # Precomputed position indices (sequential)
    }
    tokenized_records.append(tokenized_record)

def convert_to_huggingface_dataset(
    output_path,
    interleaved_turns,
    tokenizer,
    max_seq_length=1024,
    limit_sequences: int | None = None,
    split="train",
):
    """
    Convert turns into a HuggingFace dataset with tokenized format and motion-only supervision.
    Format: text-to-motion (t2m) - only motion tokens in response (no text, no audio).
    V3 format uses unified motion modality (modality 2).
    
    Args:
        output_path: Where to save the processed dataset
        interleaved_turns: List of processed chunks
        tokenizer_name: Name of the tokenizer to use
        max_seq_length: Maximum sequence length for tokenization
        limit_sequences: Limit number of sequences to process (for debugging)
        split: Dataset split (train/test/val)
    Returns:
        Dataset: The created HuggingFace dataset
    """
    logging.info(f"Converting to Hugging Face dataset (MOT version) with tokenizer: {tokenizer.__class__.__name__}")
    os.makedirs(output_path, exist_ok=True)
    
    # Use the provided tokenizer (already configured in main())
    eos_token = tokenizer.eos_token
    
    # No system prompt for this version (woprompt = without prompt)

    # Note: assistant_prefix is already included in the answer_text from transcripts_answer files
    # assistant_prefix = "<|assistant|>streaming_transcription\n"  # Not needed - already in answer_text
    # assistant_prefix_tokens = tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
    eos_token_ids = tokenizer(eos_token, add_special_tokens=False)["input_ids"]

    # Group turns by conversation ID
    conversations = {}
    for turn in interleaved_turns:
        conv_id = turn.get("conversation_id")
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(turn)
    
    tokenized_records = []
    
    for conv_idx, (conv_id, turns) in enumerate(conversations.items()):
        # Process turns as text-to-motion pairs
        if not turns:
            continue
        
        # Initialize accumulator for packing multiple turns
        current_input_ids = []
        current_labels = []
        current_turns = []
        current_modality_masks_0 = []  # Text tokens (modality 0)
        current_modality_masks_1 = []  # Audio tokens (modality 1)
        current_modality_masks_2 = []  # Unified motion tokens (modality 2)
        
        # Process the conversation - take question from first turn and full answer text
        if len(turns) > 0:
            first_turn = turns[0]  # Get question from first turn
            input_type = first_turn.get("input_type", "text")
            question_text = first_turn.get("question_text", "")
            question_audio_tokens = first_turn.get("question_audio_tokens", [])
            answer_text = first_turn.get("answer_text", "")  # Full answer text with all audio tokens
            
            # No system prompt for this version (woprompt = without prompt)
            
            # Get the full motion token arrays directly (no need to reconstruct)
            full_lower_tokens = first_turn.get("lower_tokens", [])  # Full lower token array
            
            # Add user question based on input type
            if input_type == "text" and question_text:
                # Text input: use question text
                user_prompt = f"<|user|>\n{question_text}"
                user_tokens = tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
                current_input_ids.extend(user_tokens)
                current_labels.extend([-100] * len(user_tokens))  # User doesn't contribute to loss
                current_modality_masks_0.extend([True] * len(user_tokens))   # User is text (modality 0)
                current_modality_masks_1.extend([False] * len(user_tokens))  # User is not audio (modality 1)
                current_modality_masks_2.extend([False] * len(user_tokens))  # User is not motion (modality 2)
            
            # Note: assistant prefix is already included in answer_text, so we don't add it here
            
            # Process the full answer text with embedded audio tokens
            answer_tokens = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            
            # For t2m: Only motion tokens in response (no text, no audio)
            
            # For t2m: Process motion tokens only
            # Get motion tokens (this is the only modality in the output)
            full_lower_tokens = first_turn.get("lower_tokens", [])
            
            if len(full_lower_tokens) == 0:
                # No motion tokens available - skip this turn
                continue
            
            # Motion-only processing (no interleaving needed)
            # Create motion sequence with assistant prefix and motion tokens only
            final_tokens = answer_tokens.copy()  # Start with assistant prefix
            final_labels = [-100] * len(answer_tokens)  # Assistant prefix doesn't contribute to loss
            final_modality_0 = [True] * len(answer_tokens)   # Assistant prefix is text (modality 0)
            final_modality_1 = [False] * len(answer_tokens)  # Assistant prefix is not audio (modality 1)
            final_modality_2 = [False] * len(answer_tokens)  # Assistant prefix is not motion (modality 2)
            
            # Add begin_of_motion token (V3: unified motion modality)
            begin_motion_token = tokenizer("<|begin_of_motion|>", add_special_tokens=False)["input_ids"][0]
            final_tokens.append(begin_motion_token)
            final_labels.append(begin_motion_token)  # Begin token is supervised
            final_modality_0.append(False)
            final_modality_1.append(False)
            final_modality_2.append(True)   # Begin of motion token is unified motion (modality 2)
            
            # Add all motion tokens as unified motion modality
            for token in full_lower_tokens:
                token_str = safe_token_to_string(token, "motion")  # Use "motion" instead of "lower"
                token_ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
                for token_id in token_ids:
                    final_tokens.append(token_id)
                    final_labels.append(token_id)  # Motion tokens are supervised
                    final_modality_0.append(False)
                    final_modality_1.append(False)
                    final_modality_2.append(True)   # Motion tokens are modality 2
            
            # Add end_of_motion token
            end_of_motion_token = tokenizer("<|end_of_motion|>", add_special_tokens=False)["input_ids"][0]
            final_tokens.append(end_of_motion_token)
            final_labels.append(end_of_motion_token)  # End token is supervised
            final_modality_0.append(False)
            final_modality_1.append(False)
            final_modality_2.append(True)   # End of motion token is unified motion (modality 2)
            
            # Set the final turn data
            turn_input_ids = final_tokens
            turn_labels = final_labels
            turn_modality_0 = final_modality_0
            turn_modality_1 = final_modality_1
            turn_modality_2 = final_modality_2
            # For woprompt version: No length restrictions, just add the turn
            
            # Add this turn to the current sequence
            current_input_ids.extend(turn_input_ids)
            current_labels.extend(turn_labels)
            current_modality_masks_0.extend(turn_modality_0)
            current_modality_masks_1.extend(turn_modality_1)
            current_modality_masks_2.extend(turn_modality_2)
            current_turns.append(first_turn)  # Use first_turn instead of undefined turn
        
        # Finalize the last sequence if it has content (no eos_token needed, end_of_motion is sufficient)
        if current_input_ids:
            
            tokenized_record = {
                "input_ids": current_input_ids,
                "labels": current_labels,
                "turns": current_turns,
                "modality_masks_0": current_modality_masks_0,
                "modality_masks_1": current_modality_masks_1,
                "modality_masks_2": current_modality_masks_2,
            }
            process_chunk_to_record(tokenized_record, conv_id, tokenized_records)
        
        if limit_sequences is not None and len(tokenized_records) >= limit_sequences:
            break
    
    # Create dataset from records
    if not tokenized_records:
        logging.error("No valid records created")
        return None
    
    tokenized_dataset = Dataset.from_list(tokenized_records)
    tokenized_dataset_path = os.path.join(output_path, "tokenized_dataset")
    tokenized_dataset.save_to_disk(tokenized_dataset_path)
    
    # Save metadata
    metadata = {
        "split": split,
        f"{split}_size": len(tokenized_records),
        "motion_tokens_per_chunk": "all_available_tokens",
        "format_version": "3.0",
        "format_type": "user_assistant_motion_only_with_position_encoding_v3",
        "text_format": "text description input -> motion tokens output only (no text, no audio in response)",
        "position_encoding": "precomputed_sequential_indices_no_interpolation",
        "position_encoding_type": "sequential",  # No interpolation, direct sequential indexing
        "supervision": "motion tokens only (as unified motion modality 2)",
        "tokenized": True,
        "max_seq_length": max_seq_length,
        "system_prompt": "none",
        "assistant_prefix": "included_in_answer_text",
        "modality_masks": "masks_0 for text (modality 0), masks_1 for audio (modality 1), masks_2 for unified motion (modality 2)",
        "modality_supervision": "motion_only_v3",
        "source": "H3D_t2m_v3"
    }
    
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Log position encoding statistics
    if tokenized_records:
        sample_record = tokenized_records[0]
        if "position_encoding_indices" in sample_record:
            pos_indices = sample_record["position_encoding_indices"]
            logging.info(f"Position encoding indices computed successfully (sequential indexing)")
            logging.info(f"Sample position indices range: {min(pos_indices):.3f} to {max(pos_indices):.3f}")
            logging.info(f"Sample sequence length: {len(pos_indices)}")
        else:
            logging.warning("Position encoding indices not found in records")
    
    logging.info(f"Dataset saved to {output_path}")
    logging.info(f"You can load it with: from datasets import load_from_disk; dataset = load_from_disk('{tokenized_dataset_path}')")
    
    return tokenized_dataset

def main():
    """Main function to preprocess AMASS dataset for text-to-motion generation."""
    parser = argparse.ArgumentParser(description="Build the HuggingFace text-to-motion dataset from preprocessed HumanML3D + motion tokens.")

    # Required arguments
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to the HumanML3D dataset root (e.g., <HUMANML3D_ROOT>)")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Where to save processed dataset")
    parser.add_argument("--model_name", type=str, default="THUDM/glm-4-voice-9b", 
                       help="Tokenizer model name")
    
    # Data directories
    parser.add_argument("--texts_dir", type=str, default="texts", 
                       help="Directory containing text descriptions (relative to data_root)")                       
    parser.add_argument("--lower_dir", type=str, default="TOKENS", 
                       help="Directory containing lower body token files (relative to data_root)")
    # Processing parameters
    parser.add_argument("--lower_fps", type=float, default=7.5, 
                       help="Lower body motion tokens per second")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    parser.add_argument("--limit_videos", type=int, default=1e8,
                       help="Limit number of videos to process (for debugging)")
    parser.add_argument("--split", type=str, choices=["train", "test", "val"], required=True, 
                       help="Which split to process (train, test, or val)")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting preprocessing of AMASS dataset (text-to-motion version)")
    logging.info(f"Args: {args}")

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Load train/test split
    split_file = os.path.join(args.data_root, f"{args.split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        selected_sequence_ids = set(line.strip() for line in f if line.strip())
    
    logging.info(f"Loaded {len(selected_sequence_ids)} sequences for {args.split} split")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token
    
    # Add special tokens for face, upper body, and lower body modalities
    motion_tokens = [f"<|motion_{i}|>" for i in range(512)]
    tokenizer.add_tokens(motion_tokens, special_tokens=False)
    tokenizer.add_tokens([f"<|begin_of_motion|>"], special_tokens=True)
    tokenizer.add_tokens([f"<|end_of_motion|>"], special_tokens=True)
    print(f"Extended tokenizer vocab size: {len(tokenizer)}")

    # Get list of sequences to process (AMASS t2m layout: texts directory)
    texts_dir = os.path.join(args.data_root, args.texts_dir)
    lower_dir = os.path.join(args.data_root, args.lower_dir)
    sequence_ids = []
    if not os.path.isdir(texts_dir):
        logging.error("Required directory does not exist: texts_dir")
        return

    for text_path in Path(texts_dir).glob("*.txt"):
        sequence_id = text_path.stem
        
        # Only process sequences that are in the selected split
        if sequence_id not in selected_sequence_ids:
            continue
            
        # Check if lower body motion token file exists
        lower_file = os.path.join(lower_dir, f"{sequence_id}.npy")
        if os.path.exists(lower_file):
            sequence_ids.append(sequence_id)
        else:
            logging.debug(f"Skipping {sequence_id}, missing lower body token file")
        if len(sequence_ids) >= args.limit_videos:
            logging.info(f"Limiting to {len(sequence_ids)} videos")
            break
    
    logging.info(f"Found {len(sequence_ids)} videos with required files")
    
    # Process each video
    all_turns = []
    
    for sequence_id in tqdm(sequence_ids, desc="Processing videos"):

        # Process text-motion pairs
        processed_segments = process_full_video(
            sequence_id, texts_dir, lower_dir, args.lower_fps
        )

        if not processed_segments:
            logging.warning(f"No valid segments processed for {sequence_id}")
            continue
        
        # Process segments directly without chunking - create two samples per sequence
        for seg in processed_segments:
            # Create text input version
            all_turns.append({
                "conversation_id": f"{seg['segment_id']}_text",
                "turn_id": f"{seg['segment_id']}_text",
                "input_type": "text",
                "question_text": seg["transcripts_question"],
                "question_audio_tokens": [],  # No audio tokens for text input
                "answer_text": seg["transcripts_answer"],  # Full answer text with all embedded audio tokens
                "lower_tokens": seg["lower_tokens"],  # Only lower body tokens
                "speaker_type": "assistant",
            })
            
        logging.info(f"Processed {len(processed_segments)} text-to-motion samples for {sequence_id}")

        if args.debug:
            logging.info(f"Debug mode enabled, processing only 1 sequence")
            break
    
    logging.info(f"Total interleaved groups created: {len(all_turns)}")
    
    # Convert to HuggingFace dataset
    if all_turns:
        dataset = convert_to_huggingface_dataset(
            output_path=args.output_path,
            interleaved_turns=all_turns,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            limit_sequences=(100 if args.debug else None),
            split=args.split,
        )
        
        if dataset:
            logging.info(f"Successfully created dataset with {len(dataset)} samples")
        else:
            logging.error("Failed to create dataset")
    else:
        logging.error("No chunks to process")

    logging.info("Preprocessing finished")

if __name__ == "__main__":
    main() 