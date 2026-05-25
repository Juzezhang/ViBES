#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess AMASS dataset into question-answer sequences with unified motion modality,
organizing audio and motion tokens (upper, lower, hand) into fixed-size chunks and saving in
HuggingFace datasets format. Body only version without face tokens.

This script targets the AMASS dataset at
  /path/to/AMASS
and assumes:
- Audio tokens are embedded within transcripts_answer text files
- Answer transcripts under data_root/transcripts_answer (with audio tokens)
- Question transcripts under data_root/transcripts_question (text only)
- Question audio tokens under data_root/audios_q_token_glm
- Upper body tokens under data_root/TOKENS_AGENT_25_Rotation/upper
- Lower body tokens under data_root/TOKENS_AGENT_25_Rotation/lower
- Hand tokens under data_root/TOKENS_AGENT_25_Rotation/hand_generated
- Split information in data_root/{split}.txt

Key features:
- Supports train/test/val splits based on split files in the dataset root
- Creates TWO samples per sequence: one with text input, one with audio input
- Processes question-answer pairs with unified motion supervision
- Unifies upper, lower, and hand tokens into a single motion modality (modality 2)
- Audio FPS: 12.5, Motion FPS: 18.75 (6.25*3) for body only
- Hand tokens: supervised if available, zero-padded if not
- Group size: question (text OR audio) + answer text (with embedded audio) + 1 begin_of_motion + 39 interleaved motion tokens per group
- Token breakdown: 1 begin_of_motion + 39 motion tokens (13 upper + 13 lower + 13 hand in 1:1:1 alternating pattern)
- 3 modalities: text(0), audio(1), motion(2)
- Different system prompts for text vs audio input

Usage:
    python preprocess_amass_dataset_w_transcript_tokenized_varying_mot_encode_position_body_only_v3.py \
        --data_root /path/to/AMASS \
        --output_path ./processed_amass_body_only_train \
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
import sys
from typing import Optional, Tuple, Union, List, Dict, Any

# Global precomputed token ID sets for efficient attention mask generation
HAND_TOKEN_IDS = None
UPPER_TOKEN_IDS = None
LOWER_TOKEN_IDS = None

def initialize_token_sets(tokenizer):
    """
    Initialize global token ID sets once at the beginning of processing.
    This avoids recomputing token IDs for every chunk.
    
    Args:
        tokenizer: Tokenizer instance to convert tokens to IDs
    """
    global HAND_TOKEN_IDS, UPPER_TOKEN_IDS, LOWER_TOKEN_IDS
    
    # Initialize hand token IDs
    HAND_TOKEN_IDS = set()
    for i in range(256):
        try:
            token_str = f"<|hand_{i}|>"
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id != tokenizer.unk_token_id:
                HAND_TOKEN_IDS.add(token_id)
        except:
            continue
    
    # Initialize upper token IDs
    UPPER_TOKEN_IDS = set()
    for i in range(256):
        try:
            token_str = f"<|upper_{i}|>"
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id != tokenizer.unk_token_id:
                UPPER_TOKEN_IDS.add(token_id)
        except:
            continue
    
    # Initialize lower token IDs
    LOWER_TOKEN_IDS = set()
    for i in range(256):
        try:
            token_str = f"<|lower_{i}|>"
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id != tokenizer.unk_token_id:
                LOWER_TOKEN_IDS.add(token_id)
        except:
            continue
    
    logging.info(f"Initialized token sets: {len(HAND_TOKEN_IDS)} hand, {len(UPPER_TOKEN_IDS)} upper, {len(LOWER_TOKEN_IDS)} lower tokens")

def _find_consecutive_groups(positions: torch.Tensor) -> List[torch.Tensor]:
    """
    Find groups of consecutive positions in a sorted tensor.
    
    Args:
        positions: Sorted tensor of positions
    
    Returns:
        List of tensors, each containing a group of consecutive positions
    """
    if len(positions) == 0:
        return []
    
    groups = []
    current_group = [positions[0]]
    
    for i in range(1, len(positions)):
        if positions[i] == positions[i-1] + 1:
            current_group.append(positions[i])
        else:
            groups.append(torch.tensor(current_group, device=positions.device))
            current_group = [positions[i]]
    
    groups.append(torch.tensor(current_group, device=positions.device))
    return groups


class DummyRotaryEmbedding:
    """
    Dummy RotaryEmbedding class for position encoding calculation during preprocessing.
    This mimics the interface needed by the position encoding function.
    """
    def __init__(self, dim=4096, rope_ratio=1.0):
        self.dim = dim
        self.rope_ratio = rope_ratio
        # Create dummy inv_freq with appropriate dtype and device
        self.inv_freq = torch.ones(dim // 2, dtype=torch.float32)
    
    def forward_impl(self, seq_len, n_elem, dtype, device, base=10000):
        """
        Dummy implementation that returns position indices instead of actual embeddings.
        This is used during preprocessing to compute position mappings.
        """
        # Return simple position indices as a placeholder
        # The actual rotary embeddings will be computed at runtime using these indices
        positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(-1).unsqueeze(-1)
        return positions.expand(-1, n_elem // 2, 2)


def resolve_motion_variant_config(motion_variant: str) -> Dict[str, float]:
    """Return segment sizes and motion fps for the selected motion subset."""
    if motion_variant == "upper_hand":
        return {
            "upper_segment_size": 13,
            "lower_segment_size": 0,
            "hand_segment_size": 13,
            "motion_fps": 12.5,
        }
    if motion_variant == "lower_only":
        return {
            "upper_segment_size": 0,
            "lower_segment_size": 13,
            "hand_segment_size": 0,
            "motion_fps": 6.25,
        }
    return {
        "upper_segment_size": 13,
        "lower_segment_size": 13,
        "hand_segment_size": 13,
        "motion_fps": 18.75,
    }


def calculate_position_encoding_indices(modality_masks, modality_fps=None):
    """
    Calculate position encoding indices for each token based on modality masks.
    This function extracts the position mapping logic from the rotary embedding computation
    and returns the interpolated position indices that can be stored in the dataset.
    
    Args:
        modality_masks: List of modality masks [mask_0, mask_1, mask_2]
                       where each mask is a list of booleans indicating token presence
        modality_fps: Dictionary mapping modality index to fps value
                     Default: {1: 12.5, 2: 18.75}
    
    Returns:
        List of float values representing the interpolated position index for each token.
        These indices can be used later to compute the actual rotary embeddings efficiently.
    """
    if modality_fps is None:
        modality_fps = {1: 12.5, 2: 18.75}
    
    # Convert modality masks to tensor format expected by the function
    seq_length = len(modality_masks[0])
    n_modalities = len(modality_masks)
    
    # Create tensor format: [n_modalities, seq_length]
    modality_masks_tensor = torch.zeros((n_modalities, seq_length), dtype=torch.bool)
    for i, mask in enumerate(modality_masks):
        modality_masks_tensor[i] = torch.tensor(mask, dtype=torch.bool)
    
    # Create dummy rotary embedding module for position calculation
    dummy_rotary = DummyRotaryEmbedding()
    
    # Compute position mappings using the same logic as the model
    position_indices = extract_position_indices_from_rotary_computation(
        dummy_rotary, modality_masks_tensor, modality_fps
    )
    
    return position_indices.tolist()


def extract_position_indices_from_rotary_computation(rotary_embedding_module, modality_masks, modality_fps):
    """
    Extract position indices from the rotary embedding computation without computing actual embeddings.
    This function follows the same logic as create_rotary_embeddings_from_modality_masks_multiple_modalities
    but only tracks the position indices.
    """
    # Handle batch dimension
    if modality_masks.dim() == 3:
        modality_masks = modality_masks[:, 0, :]
    
    n_modalities, seq_length = modality_masks.shape
    
    # Initialize position indices array
    position_indices = torch.arange(seq_length, dtype=torch.float32)
    
    # Get positions for all modalities
    modality_positions = {}
    for i in range(n_modalities):
        mask = modality_masks[i].bool()
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        if len(positions) > 0:
            modality_positions[i] = positions
    
    # Combine modality 0 and 1 to form the primary sequence
    primary_positions = []
    if 0 in modality_positions:
        primary_positions.append(modality_positions[0])
    if 1 in modality_positions:
        primary_positions.append(modality_positions[1])
    
    if not primary_positions:
        return position_indices
    
    primary_positions = torch.cat(primary_positions)
    primary_positions, _ = torch.sort(primary_positions)
    
    if 1 not in modality_positions:
        return position_indices
        
    mod1_positions = modality_positions[1]
    if len(mod1_positions) == 0:
        return position_indices
    
    # Set primary position indices (these remain as their sequential positions)
    for i, pos in enumerate(primary_positions):
        position_indices[pos] = float(i)
    
    # Create pos→rope mapping for fast lookups
    pos_to_rope_idx = {}
    for i, pos in enumerate(primary_positions):
        pos_to_rope_idx[pos.item()] = i
    
    # Find consecutive groups of modality 1 positions (cycles)
    mod1_groups = _find_consecutive_groups(mod1_positions)
    
    for group_idx, mod1_group in enumerate(mod1_groups):
        # Get timing information for this mod1 cycle
        first_mod1_pos = mod1_group[0].item()
        first_mod1_idx = pos_to_rope_idx[first_mod1_pos]
        
        base_fps = modality_fps.get(1, 12.5)
        cycle_duration = len(mod1_group) / base_fps  # Duration in seconds

        # Create a global token list for this cycle with timestamps
        all_cycle_tokens = []
        
        # Process modality 2 (unified motion) and collect all tokens with timestamps
        for modality_idx in range(2, min(3, n_modalities)):
            if modality_idx not in modality_positions:
                continue
                
            positions = modality_positions[modality_idx]
            fps = modality_fps.get(modality_idx, 12.5)
            
            # Find positions that belong to this cycle
            cycle_positions = []
            for pos in positions:
                if pos.item() >= first_mod1_pos:
                    if group_idx + 1 < len(mod1_groups):
                        next_mod1_pos = mod1_groups[group_idx + 1][0].item()
                        if pos.item() < next_mod1_pos:
                            cycle_positions.append(pos)
                    else:
                        cycle_positions.append(pos)
            
            if len(cycle_positions) > 0:
                # Add start token (first position) with special timing
                start_pos = cycle_positions[0]
                start_offset = -0.5  # Unified motion modality
                all_cycle_tokens.append({
                    'position': start_pos.item(),
                    'modality': modality_idx,
                    'timestamp': start_offset,
                    'is_start_token': True,
                    'priority': modality_idx
                })
                
                # Add regular tokens with calculated timestamps
                if len(cycle_positions) > 1:
                    regular_positions = cycle_positions[1:]
                    time_per_token = 1.0 / fps

                    for i, pos in enumerate(regular_positions):
                        token_timestamp = i * time_per_token
                        K = len(mod1_group) - 1
                        normalized_timestamp = (token_timestamp / cycle_duration) * K
                        
                        all_cycle_tokens.append({
                            'position': pos.item(),
                            'modality': modality_idx,
                            'timestamp': normalized_timestamp,
                            'is_start_token': False,
                            'priority': modality_idx
                        })
        
        # Sort all tokens by timestamp, then by modality priority
        all_cycle_tokens.sort(key=lambda x: (x['timestamp'], x['priority']))
        
        # Separate start tokens and regular tokens
        start_tokens = [token for token in all_cycle_tokens if token['is_start_token']]
        regular_tokens = [token for token in all_cycle_tokens if not token['is_start_token']]
        
        # Handle start tokens first
        for token_info in start_tokens:
            pos = token_info['position']
            timestamp = token_info['timestamp']  # This is the negative offset
            
            # Start tokens: positioned before first mod1 token with fixed offset
            interpolated_position = first_mod1_idx + timestamp
            position_indices[pos] = interpolated_position
        
        # Handle regular tokens with uniform interpolation between mod1 positions
        if len(regular_tokens) > 0 and len(mod1_group) > 0:
            # Get mod1 rope indices
            mod1_rope_indices = []
            for mod1_pos in mod1_group:
                rope_idx = pos_to_rope_idx[mod1_pos.item()]
                mod1_rope_indices.append(rope_idx)
            
            if len(mod1_rope_indices) == 1:
                # Only one mod1 token, place all regular tokens after it
                base_rope_idx = mod1_rope_indices[0]
                for i, token_info in enumerate(regular_tokens):
                    pos = token_info['position']
                    interpolated_position = base_rope_idx + (i + 1) * (1.0 / (len(regular_tokens) + 1))
                    position_indices[pos] = interpolated_position
            else:
                K = len(mod1_rope_indices) - 1

                # Group regular tokens by interval
                intervals = []
                for i in range(K):
                    intervals.append([])
                tail_tokens = []

                for token_info in regular_tokens:
                    timestamp = token_info['timestamp']
                    if timestamp < 0:
                        interval_idx = 0
                        intervals[interval_idx].append(token_info)
                    elif timestamp >= K:
                        tail_tokens.append(token_info)
                    else:
                        interval_idx = int(timestamp)
                        intervals[interval_idx].append(token_info)

                for interval_idx, interval_tokens in enumerate(intervals):
                    if len(interval_tokens) > 0:
                        start_rope_idx = mod1_rope_indices[interval_idx]
                        end_rope_idx = mod1_rope_indices[interval_idx + 1]
                        for i, token_info in enumerate(interval_tokens):
                            pos = token_info['position']
                            alpha = (i + 1) / (len(interval_tokens) + 1)
                            interpolated_position = start_rope_idx + alpha * (end_rope_idx - start_rope_idx)
                            position_indices[pos] = interpolated_position

                if len(tail_tokens) > 0:
                    last_start_rope_idx = mod1_rope_indices[-2]
                    last_end_rope_idx = mod1_rope_indices[-1]
                    interval_span = last_end_rope_idx - last_start_rope_idx
                    for i, token_info in enumerate(tail_tokens):
                        pos = token_info['position']
                        extrapolation_offset = (i + 1) * (interval_span / (len(tail_tokens) + 1))
                        interpolated_position = last_end_rope_idx + extrapolation_offset
                        position_indices[pos] = interpolated_position
    
    return position_indices


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
        modality: Type of token ("audio" or "upper")
        
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
    
    if modality == "audio":
        return f"<|audio_{token_int}|>"
    elif modality == "upper":
        return f"<|upper_{token_int}|>"
    elif modality == "lower":
        return f"<|lower_{token_int}|>"
    elif modality == "hand":
        return f"<|hand_{token_int}|>"
    else:
        raise ValueError(f"Unknown modality: {modality}")



def process_full_video(sequence_id, transcripts_question_dir, transcripts_answer_dir, audio_question_dir, upper_dir, lower_dir, hand_dir,
                       audio_fps=12.5, upper_fps=6.25, lower_fps=6.25, hand_fps=6.25):
    """
    Process transcript segments into format suitable for dataset creation.
    
    Args:
        sequence_id: Video ID
        transcripts_question_dir: Directory containing transcripts_question text
        transcripts_answer_dir: Directory containing transcripts_answer text (with embedded audio tokens)
        audio_question_dir: Directory containing audio tokens for questions
        upper_dir: Directory containing upper body tokens
        lower_dir: Directory containing lower body tokens
        hand_dir: Directory containing hand tokens
        upper_fps: Upper body token frame rate
        lower_fps: Lower body token frame rate
        hand_fps: Hand token frame rate
        
    Returns:
        List of processed segments with text, audio, upper body, and hand tokens
    """
    processed_segments = []

    # Load transcripts, audio question tokens, and motion tokens
    transcripts_question_file = os.path.join(transcripts_question_dir, f"{sequence_id}.txt")
    transcripts_answer_file = os.path.join(transcripts_answer_dir, f"{sequence_id}.txt")
    audio_question_file = os.path.join(audio_question_dir, f"{sequence_id}.npy")
    upper_token_file = os.path.join(upper_dir, f"{sequence_id}.npy")
    lower_token_file = os.path.join(lower_dir, f"{sequence_id}.npy")
    hand_token_file = os.path.join(hand_dir, f"{sequence_id}.npy")
    
    if not os.path.exists(transcripts_question_file):
        logging.warning(f"No transcripts_question found for {sequence_id}, skipping")
        return []
    
    if not os.path.exists(transcripts_answer_file):
        logging.warning(f"No transcripts_answer found for {sequence_id}, skipping")
        return []
    

    if not os.path.exists(upper_token_file):
        logging.warning(f"No upper body tokens found for {sequence_id}, skipping")
        return []
    
    if not os.path.exists(lower_token_file):
        logging.warning(f"No lower body tokens found for {sequence_id}, skipping")
        return []
    
    # Hand file is OPTIONAL — AMASS upstream usually has no hand tokens. If missing,
    # we fall through with has_hand_file=False so downstream supervision masks hand labels to -100.
    has_hand_file = os.path.exists(hand_token_file)
    if not has_hand_file:
        logging.info(f"No hand tokens for {sequence_id}; will use zero placeholders (labels masked to -100)")

    if not os.path.exists(audio_question_file):
        logging.warning(f"No audio question tokens found for {sequence_id}, skipping")
        return []
    
    # Load transcripts and audio question tokens
    try:
        with open(transcripts_question_file, 'r', encoding='utf-8') as f:
            transcripts_question = f.read().strip()
        with open(transcripts_answer_file, 'r', encoding='utf-8') as f:
            transcripts_answer = f.read().strip()
        
        # Load audio question tokens
        audio_question_data = np.load(audio_question_file, allow_pickle=True)
        logging.info(f"Loaded transcripts and audio question tokens for {sequence_id}")
        logging.info(f"Question text: {transcripts_question[:100]}...")
        logging.info(f"Question audio tokens: {len(audio_question_data)} tokens")
        logging.info(f"Answer (first 200 chars): {transcripts_answer[:200]}...")
    except Exception as e:
        logging.error(f"Error loading transcripts or audio question tokens for {sequence_id}: {e}")
        return []
    
    # No face tokens in body only version
    
    # Load upper, lower, and hand body tokens
    try:
        upper_data = np.load(upper_token_file, allow_pickle=True)
        lower_data = np.load(lower_token_file, allow_pickle=True)
        if has_hand_file:
            hand_data = np.load(hand_token_file, allow_pickle=True)
            if hand_data.ndim > 1:
                hand_data = hand_data[0]  # Take first element if nested
        else:
            # Match lower's length as a sensible placeholder; downstream masks labels to -100.
            hand_data = np.zeros_like(lower_data[0] if lower_data.ndim > 1 else lower_data)
        if upper_data.ndim > 1:
            upper_data = upper_data[0]  # Take first element if nested
        if lower_data.ndim > 1:
            lower_data = lower_data[0]  # Take first element if nested
        
        # Calculate target duration based on upper tokens and upper FPS (use upper as reference)
        target_duration = len(upper_data) / upper_fps
        logging.info(f"Target duration based on upper tokens: {target_duration:.2f}s")
        
        # Calculate required number of upper tokens based on duration and upper FPS
        target_upper_length = int(target_duration * upper_fps)
        if len(upper_data) < target_upper_length:
            # Repeat the sequence to match target duration
            repeat_times = (target_upper_length + len(upper_data) - 1) // len(upper_data)
            upper_data_extended = np.tile(upper_data, repeat_times)[:target_upper_length]
            logging.info(f"Extended upper tokens from {len(upper_data)} to {target_upper_length} by repeating (duration: {target_duration:.2f}s)")
        elif len(upper_data) > target_upper_length:
            # Truncate to match target duration
            upper_data_extended = upper_data[:target_upper_length]
            logging.info(f"Truncated upper tokens from {len(upper_data)} to {target_upper_length} (duration: {target_duration:.2f}s)")
        else:
            upper_data_extended = upper_data
            logging.info(f"Upper tokens already match target duration: {len(upper_data)} tokens for {target_duration:.2f}s")
        
        # Calculate required number of lower tokens based on duration and lower FPS
        target_lower_length = int(target_duration * lower_fps)
        if len(lower_data) < target_lower_length:
            # Repeat the sequence to match target duration
            repeat_times = (target_lower_length + len(lower_data) - 1) // len(lower_data)
            lower_data_extended = np.tile(lower_data, repeat_times)[:target_lower_length]
            logging.info(f"Extended lower tokens from {len(lower_data)} to {target_lower_length} by repeating (duration: {target_duration:.2f}s)")
        elif len(lower_data) > target_lower_length:
            # Truncate to match target duration
            lower_data_extended = lower_data[:target_lower_length]
            logging.info(f"Truncated lower tokens from {len(lower_data)} to {target_lower_length} (duration: {target_duration:.2f}s)")
        else:
            lower_data_extended = lower_data
            logging.info(f"Lower tokens already match target duration: {len(lower_data)} tokens for {target_duration:.2f}s")
        
        # Calculate required number of hand tokens based on duration and hand FPS
        target_hand_length = int(target_duration * hand_fps)
        if len(hand_data) < target_hand_length:
            # Repeat the sequence to match target duration
            repeat_times = (target_hand_length + len(hand_data) - 1) // len(hand_data)
            hand_data_extended = np.tile(hand_data, repeat_times)[:target_hand_length]
            logging.info(f"Extended hand tokens from {len(hand_data)} to {target_hand_length} by repeating (duration: {target_duration:.2f}s)")
        elif len(hand_data) > target_hand_length:
            # Truncate to match target duration
            hand_data_extended = hand_data[:target_hand_length]
            logging.info(f"Truncated hand tokens from {len(hand_data)} to {target_hand_length} (duration: {target_duration:.2f}s)")
        else:
            hand_data_extended = hand_data
            logging.info(f"Hand tokens already match target duration: {len(hand_data)} tokens for {target_duration:.2f}s")
        
        # Create list of (token, timestamp) tuples based on their respective FPS
        upper_tokens_ts = [(upper_data_extended[i], i / upper_fps) for i in range(len(upper_data_extended))]
        lower_tokens_ts = [(lower_data_extended[i], i / lower_fps) for i in range(len(lower_data_extended))]
        hand_tokens_ts = [(hand_data_extended[i], i / hand_fps) for i in range(len(hand_data_extended))]
        
        logging.info(f"Final upper body tokens: {len(upper_data_extended)}, duration: {len(upper_data_extended) / upper_fps:.2f}s")
        logging.info(f"Final lower body tokens: {len(lower_data_extended)}, duration: {len(lower_data_extended) / lower_fps:.2f}s")
        logging.info(f"Final hand tokens: {len(hand_data_extended)}, duration: {len(hand_data_extended) / hand_fps:.2f}s")
        
    except Exception as e:
        logging.error(f"Error loading upper/lower/hand body tokens for {sequence_id}: {e}")
        return []
    
    # Build a single segment with all data
    start_time = 0.0
    end_time = target_duration
    # word_timestamps = []
    segment_upper = upper_tokens_ts
    segment_lower = lower_tokens_ts
    segment_hand = hand_tokens_ts
    if transcripts_question and transcripts_answer:
        processed_segments.append({
            "segment_id": f"{sequence_id}_full",
            "video_id": sequence_id,
            "start_time": start_time,
            "end_time": end_time,
            "upper_tokens": [t for t, _ in segment_upper],
            "lower_tokens": [t for t, _ in segment_lower],
            "hand_tokens": [t for t, _ in segment_hand],
            # "word_timestamps": word_timestamps
            "transcripts_answer": transcripts_answer,
            "transcripts_question": transcripts_question,
            "audio_question_tokens": audio_question_data.tolist(),
            # Add file existence flags for supervision logic.
            # has_hand_file is set early (line 540ish) and reused here so the value is consistent
            # with the actual load path.
            "has_upper_file": os.path.exists(upper_token_file),
            "has_lower_file": os.path.exists(lower_token_file),
            "has_hand_file": has_hand_file,
        })

    return processed_segments



def process_chunk_to_record(chunk, conv_id, tokenized_records, tokenizer, motion_fps=18.75):
    """
    Process a chunk of turns into a tokenized record with unified motion modality support.
    
    Args:
        chunk: Dictionary containing input_ids, labels, turns, modality_masks_0, modality_masks_1, modality_masks_2
        conv_id: Conversation ID
        tokenized_records: List to append the processed record to
        tokenizer: Tokenizer instance for decoding tokens
    """
    # Use global precomputed token ID sets for efficient attention mask generation
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
    
    attention_mask = [1] * seq_len
    # # Create attention mask efficiently using global precomputed hand token IDs
    # attention_mask = []
    # for i in range(seq_len):
    #     token_id = chunk["input_ids"][i]
    #     # Use global precomputed set for O(1) lookup instead of expensive decode
    #     if token_id in HAND_TOKEN_IDS:
    #         attention_mask.append(0)  # Hand tokens get 0 attention mask
    #     else:
    #         attention_mask.append(1)  # All other tokens get 1 attention mask
    
    # Calculate position encoding indices for this sequence
    modality_masks = [
        chunk["modality_masks_0"],  # Text tokens (modality 0)
        chunk["modality_masks_1"],  # Audio tokens (modality 1) 
        chunk["modality_masks_2"],  # Unified motion tokens (modality 2)
    ]
    
    # Calculate position encoding indices using the same logic as the model
    # This MUST succeed during preprocessing
    position_encoding_indices = calculate_position_encoding_indices(
        modality_masks,
        modality_fps={1: 12.5, 2: motion_fps},
    )
    
    # Validate position encoding indices
    assert len(position_encoding_indices) == seq_len, f"Position indices length {len(position_encoding_indices)} != sequence length {seq_len}"
    
    # Log statistics for validation
    pos_min, pos_max = min(position_encoding_indices), max(position_encoding_indices)
    # logging.debug(f"Calculated position encoding indices for sequence of length {seq_len}, range: [{pos_min:.3f}, {pos_max:.3f}]")
    
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
        "modality_masks_0": chunk["modality_masks_0"],  # True for text tokens
        "modality_masks_1": chunk["modality_masks_1"],  # True for audio tokens
        "modality_masks_2": chunk["modality_masks_2"],  # True for unified motion tokens
        "position_encoding_indices": position_encoding_indices,  # Precomputed position indices
    }
    tokenized_records.append(tokenized_record)

def convert_to_huggingface_dataset(
    output_path,
    interleaved_turns,
    tokenizer,
    max_seq_length=2048,
    upper_segment_size=13,
    lower_segment_size=13,
    hand_segment_size=13,
    motion_fps: float = 18.75,
    limit_sequences: int | None = None,
    split="train",
):
    """
    Convert interleaved turns into a HuggingFace dataset with unified motion modality support.
    Following AMASS format: assistant-only, no system prompt, unified motion supervision.
    Packs multiple turns into sequences up to max_seq_length.
    Body only version: upper, lower, and hand tokens in 1:1:1 interleaved pattern.
    
    Args:
        output_path: Where to save the processed dataset
        interleaved_turns: List of processed chunks
        tokenizer: The tokenizer to use
        max_seq_length: Maximum sequence length for tokenization
        upper_segment_size: Number of upper body tokens per group (default: 13)
        lower_segment_size: Number of lower body tokens per group (default: 13)
        hand_segment_size: Number of hand tokens per group (default: 13)
        limit_sequences: Limit number of sequences to process
        split: Dataset split name
    Returns:
        Dataset: The created HuggingFace dataset
    """
    logging.info(f"Converting to Hugging Face dataset (MOT version) with tokenizer: {tokenizer.__class__.__name__}")
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize global token ID sets once for efficient attention mask generation
    initialize_token_sets(tokenizer)
    
    # Use the provided tokenizer (already configured in main())
    eos_token = tokenizer.eos_token
    
    # Prepare system prompts for both input types
    text_system_prompt = "<|system|>\nUser will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. Please follow these steps carefully: 1. Think about the instruction first. 2. Respond in an interleaved manner: output 13 text tokens followed by 26 audio tokens. 3. In your reply, imagine that you have a body and are already moving, pretending to perform 'the motion required by the question. 4. Make sure your answer aligns with both the question and the motion being asked. 5. Remember: the motion is imaginary (pretend), not real. 6. If you describe the motion, use the first-person perspective (e.g., 'my hand,' 'my body,' 'my movement'). Please reply as if you are experiencing and expressing the motion yourself."
    
    audio_system_prompt = "<|system|>\nUser will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens. Please follow these steps carefully: 1. Think about the instruction first. 2. Respond in an interleaved manner: output 13 text tokens followed by 26 audio tokens. 3. In your reply, imagine that you have a body and are already moving, pretending to perform 'the motion required by the question. 4. Make sure your answer aligns with both the question and the motion being asked. 5. Remember: the motion is imaginary (pretend), not real. 6. If you describe the motion, use the first-person perspective (e.g., 'my hand,' 'my body,' 'my movement'). Please reply as if you are experiencing and expressing the motion yourself."

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
        # Process turns as question-answer pairs
        if not turns:
            continue
        
        # Initialize accumulator for packing multiple turns
        current_input_ids = []
        current_labels = []
        current_turns = []
        current_modality_masks_0 = []
        current_modality_masks_1 = []
        current_modality_masks_2 = []
        
        # Process the conversation - take question from first turn and full answer text
        if len(turns) > 0:
            first_turn = turns[0]  # Get question from first turn
            input_type = first_turn.get("input_type", "text")
            question_text = first_turn.get("question_text", "")
            question_audio_tokens = first_turn.get("question_audio_tokens", [])
            answer_text = first_turn.get("answer_text", "")  # Full answer text with all audio tokens


            if "<|assistant|> streaming_transcription\n" in answer_text:
                answer_text = answer_text.replace("<|assistant|> streaming_transcription\n", "<|assistant|> streaming_transcription_with_performed_motion\n")
            elif "<|assistant|>streaming_transcription_by_motion\n" in answer_text:
                answer_text = answer_text.replace("<|assistant|>streaming_transcription\n", "<|assistant|> streaming_transcription_with_performed_motion\n")
            elif "<|assistant|>\n" in answer_text:
                answer_text = answer_text.replace("<|assistant|>\n", "<|assistant|> streaming_transcription_with_performed_motion\n")
            else:
                answer_text = answer_text.replace("<|assistant|>", "<|assistant|> streaming_transcription_with_performed_motion")
            
            # Choose system prompt based on input type
            if input_type == "audio":
                system_prompt = audio_system_prompt
            else:
                system_prompt = text_system_prompt
            
            system_prompt_tokens = tokenizer(system_prompt, add_special_tokens=False)["input_ids"]
            
            # Add system prompt at the beginning
            current_input_ids.extend(system_prompt_tokens)
            current_labels.extend([-100] * len(system_prompt_tokens))  # System doesn't contribute to loss
            current_modality_masks_0.extend([True] * len(system_prompt_tokens))   # System is text
            current_modality_masks_1.extend([False] * len(system_prompt_tokens))  # System is not audio
            current_modality_masks_2.extend([False] * len(system_prompt_tokens))  # System is not motion
            
            # Get the full motion token arrays directly (no need to reconstruct)
            full_upper_tokens = first_turn.get("upper_tokens", [])  # Full upper token array
            full_lower_tokens = first_turn.get("lower_tokens", [])  # Full lower token array
            full_hand_tokens = first_turn.get("hand_tokens", [])  # Full hand token array
            
            # Add user question based on input type
            if input_type == "text" and question_text:
                # Text input: use question text
                user_prompt = f"<|user|>\n{question_text}"
                user_tokens = tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
                current_input_ids.extend(user_tokens)
                current_labels.extend([-100] * len(user_tokens))  # User doesn't contribute to loss
                current_modality_masks_0.extend([True] * len(user_tokens))   # User is text
                current_modality_masks_1.extend([False] * len(user_tokens))  # User is not audio
                current_modality_masks_2.extend([False] * len(user_tokens))  # User is not motion
            elif input_type == "audio" and question_audio_tokens:
                # Audio input: use question audio tokens
                user_prompt = "<|user|>\n"
                user_text_tokens = tokenizer(user_prompt, add_special_tokens=False)["input_ids"]
                current_input_ids.extend(user_text_tokens)
                current_labels.extend([-100] * len(user_text_tokens))  # User text doesn't contribute to loss
                current_modality_masks_0.extend([True] * len(user_text_tokens))   # User text is text
                current_modality_masks_1.extend([False] * len(user_text_tokens))  # User text is not audio
                current_modality_masks_2.extend([False] * len(user_text_tokens))  # User text is not motion
                
                # Add audio question tokens
                for audio_token in question_audio_tokens:
                    token_str = safe_token_to_string(audio_token, "audio")
                    audio_token_ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
                    current_input_ids.extend(audio_token_ids)
                    current_labels.extend([-100] * len(audio_token_ids))  # Audio doesn't contribute to loss
                    current_modality_masks_0.extend([False] * len(audio_token_ids))  # Audio is not text
                    current_modality_masks_1.extend([True] * len(audio_token_ids))   # Audio is modality 1
                    current_modality_masks_2.extend([False] * len(audio_token_ids))  # Audio is not motion
            
            # Note: assistant prefix is already included in answer_text, so we don't add it here
            
            # Process the full answer text with embedded audio tokens
            answer_tokens = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            
            # Remove spaces (token ID 220) between audio tokens
            cleaned_answer_tokens = []
            for i, token_id in enumerate(answer_tokens):
                if token_id == 220:  # Space token
                    # Check if this space is between audio tokens
                    prev_is_audio = i > 0 and 152353 <= answer_tokens[i-1] <= 168735
                    next_is_audio = i < len(answer_tokens) - 1 and 152353 <= answer_tokens[i+1] <= 168735
                    
                    if prev_is_audio and next_is_audio:
                        continue  # Skip spaces between audio tokens
                    else:
                        cleaned_answer_tokens.append(token_id)  # Keep regular spaces in text
                else:
                    cleaned_answer_tokens.append(token_id)
            
            # Find audio positions and group them by 26
            audio_positions = []
            for i, token_id in enumerate(cleaned_answer_tokens):
                if 152353 <= token_id <= 168735:
                    audio_positions.append(i)
            
            # Group audio positions by 26
            all_audio_groups = []
            for i in range(0, len(audio_positions), 26):
                group = audio_positions[i:i+26]
                if len(group) == 26:  # Only add complete groups
                    all_audio_groups.append(group)
            
            # Calculate required audio groups based on enabled motion parts.
            required_groups_candidates = []
            if upper_segment_size > 0:
                required_groups_candidates.append(
                    (len(full_upper_tokens) + upper_segment_size - 1) // upper_segment_size
                )
            if lower_segment_size > 0:
                required_groups_candidates.append(
                    (len(full_lower_tokens) + lower_segment_size - 1) // lower_segment_size
                )
            if hand_segment_size > 0:
                required_groups_candidates.append(
                    (len(full_hand_tokens) + hand_segment_size - 1) // hand_segment_size
                )
            required_audio_groups = max(required_groups_candidates) if required_groups_candidates else 0
            
            # Adjust audio groups to match motion tokens length (truncate if too many)
            if len(all_audio_groups) > required_audio_groups:
                # Truncate audio groups if too many
                audio_groups = all_audio_groups[:required_audio_groups]
                logging.info(
                    f"Truncated audio groups from {len(all_audio_groups)} to {required_audio_groups} to match selected motion token lengths"
                )
            else:
                # Use all available audio groups if not enough
                audio_groups = all_audio_groups
                if len(audio_groups) < required_audio_groups:
                    logging.warning(
                        f"Only {len(audio_groups)} audio groups available, but {required_audio_groups} are needed for selected motion tokens"
                    )
            
            # Use motion tokens as-is (no adjustment needed)
            # Motion tokens are already aligned in the first stage (process_full_video)
            adjusted_upper_tokens = full_upper_tokens if upper_segment_size > 0 else []
            adjusted_lower_tokens = full_lower_tokens if lower_segment_size > 0 else []
            adjusted_hand_tokens = full_hand_tokens if hand_segment_size > 0 else []
            
            # Determine supervision based on file existence rather than token values
            # If the file exists, we should supervise even if all tokens are 0
            upper_has_real_data = first_turn.get("has_upper_file", False) and upper_segment_size > 0
            lower_has_real_data = first_turn.get("has_lower_file", False) and lower_segment_size > 0
            hand_has_real_data = first_turn.get("has_hand_file", False) and hand_segment_size > 0
            
            # Now insert unified motion tokens after each audio group
            final_tokens = []
            final_labels = []
            final_modality_0 = []
            final_modality_1 = []
            final_modality_2 = []
            
            current_pos = 0
            
            for group_idx, group in enumerate(audio_groups):
                # Copy content before audio group
                if group[0] > current_pos:
                    segment = cleaned_answer_tokens[current_pos:group[0]]
                    final_tokens.extend(segment)
                    for token_id in segment:
                        final_labels.append(-100)  # Text doesn't contribute to loss
                        final_modality_0.append(True)   # Text is modality 0
                        final_modality_1.append(False)  # Text is not audio
                        final_modality_2.append(False)  # Text is not motion
                
                # Copy audio tokens
                audio_segment = [cleaned_answer_tokens[i] for i in group]
                final_tokens.extend(audio_segment)
                for token_id in audio_segment:
                    final_labels.append(-100)  # Audio doesn't contribute to loss
                    final_modality_0.append(False)  # Audio is not text
                    final_modality_1.append(True)   # Audio is modality 1
                    final_modality_2.append(False)  # Audio is not motion
                
                # Get motion tokens for this specific audio group
                # No face tokens in body only version
                
                # Get tokens for this specific group from adjusted arrays
                upper_start = group_idx * upper_segment_size if upper_segment_size > 0 else 0
                upper_end = (group_idx + 1) * upper_segment_size if upper_segment_size > 0 else 0
                upper_tokens_chunk = adjusted_upper_tokens[upper_start:upper_end] if upper_start < len(adjusted_upper_tokens) else []
                orig_upper_count = len(upper_tokens_chunk)
                if len(upper_tokens_chunk) < upper_segment_size:
                    upper_tokens_chunk.extend([0] * (upper_segment_size - len(upper_tokens_chunk)))
                
                lower_start = group_idx * lower_segment_size if lower_segment_size > 0 else 0
                lower_end = (group_idx + 1) * lower_segment_size if lower_segment_size > 0 else 0
                lower_tokens_chunk = adjusted_lower_tokens[lower_start:lower_end] if lower_start < len(adjusted_lower_tokens) else []
                orig_lower_count = len(lower_tokens_chunk)
                if len(lower_tokens_chunk) < lower_segment_size:
                    lower_tokens_chunk.extend([0] * (lower_segment_size - len(lower_tokens_chunk)))
                
                hand_start = group_idx * hand_segment_size if hand_segment_size > 0 else 0
                hand_end = (group_idx + 1) * hand_segment_size if hand_segment_size > 0 else 0
                hand_tokens_chunk = adjusted_hand_tokens[hand_start:hand_end] if hand_start < len(adjusted_hand_tokens) else []
                orig_hand_count = len(hand_tokens_chunk)
                if len(hand_tokens_chunk) < hand_segment_size:
                    hand_tokens_chunk.extend([0] * (hand_segment_size - len(hand_tokens_chunk)))
                
                # Create unified motion token sequence with 1:1:1 interleaved pattern (body only)
                motion_token_ids = []
                motion_labels = []
                motion_modality_0 = []
                motion_modality_1 = []
                motion_modality_2 = []
                
                # Add unified begin_of_motion token
                begin_motion_token = tokenizer("<|begin_of_motion|>", add_special_tokens=False)["input_ids"][0]
                motion_token_ids.append(begin_motion_token)
                motion_labels.append(-100)
                motion_modality_0.append(False)  # Motion is not text
                motion_modality_1.append(False)  # Motion is not audio
                motion_modality_2.append(True)   # Motion is modality 2
                
                # Process tokens in 1:1:1 interleaved pattern (body only)
                # Pattern: upper_0, lower_0, hand_0, upper_1, lower_1, hand_1, ...
                upper_idx = 0
                lower_idx = 0
                hand_idx = 0
                
                while upper_idx < len(upper_tokens_chunk) or lower_idx < len(lower_tokens_chunk) or hand_idx < len(hand_tokens_chunk):
                    # Add 1 upper token
                    if upper_idx < len(upper_tokens_chunk):
                        upper_token = upper_tokens_chunk[upper_idx]
                        if isinstance(upper_token, np.ndarray):
                            upper_token_val = upper_token.tolist() if hasattr(upper_token, 'tolist') else upper_token
                        else:
                            upper_token_val = upper_token
                        upper_token_str = safe_token_to_string(upper_token_val, "upper")
                        upper_token_ids = tokenizer(upper_token_str, add_special_tokens=False)["input_ids"]
                        motion_token_ids.extend(upper_token_ids)
                        is_padding = (upper_idx >= orig_upper_count)
                        motion_labels.extend([-100] * len(upper_token_ids) if is_padding else upper_token_ids)
                        motion_modality_0.extend([False] * len(upper_token_ids))  # Upper is not text
                        motion_modality_1.extend([False] * len(upper_token_ids))  # Upper is not audio
                        motion_modality_2.extend([True] * len(upper_token_ids))   # Upper is modality 2 (motion)
                        upper_idx += 1
                    
                    # Add 1 lower token
                    if lower_idx < len(lower_tokens_chunk):
                        lower_token = lower_tokens_chunk[lower_idx]
                        if isinstance(lower_token, np.ndarray):
                            lower_token_val = lower_token.tolist() if hasattr(lower_token, 'tolist') else lower_token
                        else:
                            lower_token_val = lower_token
                        lower_token_str = safe_token_to_string(lower_token_val, "lower")
                        lower_token_ids = tokenizer(lower_token_str, add_special_tokens=False)["input_ids"]
                        motion_token_ids.extend(lower_token_ids)
                        is_padding = (lower_idx >= orig_lower_count)
                        motion_labels.extend([-100] * len(lower_token_ids) if is_padding else lower_token_ids)
                        motion_modality_0.extend([False] * len(lower_token_ids))  # Lower is not text
                        motion_modality_1.extend([False] * len(lower_token_ids))  # Lower is not audio
                        motion_modality_2.extend([True] * len(lower_token_ids))   # Lower is modality 2 (motion)
                        lower_idx += 1
                    
                    # Add 1 hand token
                    if hand_idx < len(hand_tokens_chunk):
                        hand_token = hand_tokens_chunk[hand_idx]
                        hand_token_str = safe_token_to_string(hand_token, "hand")
                        hand_token_ids = tokenizer(hand_token_str, add_special_tokens=False)["input_ids"]
                        motion_token_ids.extend(hand_token_ids)
                        is_padding = (hand_idx >= orig_hand_count)
                        if is_padding or not hand_has_real_data:
                            motion_labels.extend([-100] * len(hand_token_ids))
                        else:
                            motion_labels.extend(hand_token_ids)
                        motion_modality_0.extend([False] * len(hand_token_ids))  # Hand is not text
                        motion_modality_1.extend([False] * len(hand_token_ids))  # Hand is not audio
                        motion_modality_2.extend([True] * len(hand_token_ids))   # Hand is modality 2 (motion)
                        hand_idx += 1
                
                # Add all motion tokens to final sequence
                final_tokens.extend(motion_token_ids)
                final_labels.extend(motion_labels)
                final_modality_0.extend(motion_modality_0)
                final_modality_1.extend(motion_modality_1)
                final_modality_2.extend(motion_modality_2)
                
                current_pos = group[-1] + 1
            
            # Copy remaining content after the last audio group
            if current_pos < len(cleaned_answer_tokens):
                remaining_tokens = cleaned_answer_tokens[current_pos:]
                final_tokens.extend(remaining_tokens)
                for token_id in remaining_tokens:
                    final_labels.append(-100)  # Text doesn't contribute to loss
                    final_modality_0.append(True)   # Text is modality 0
                    final_modality_1.append(False)  # Text is not audio
                    final_modality_2.append(False)  # Text is not motion
            
            # Set the final turn data
            turn_input_ids = final_tokens
            turn_labels = final_labels
            turn_modality_0 = final_modality_0
            turn_modality_1 = final_modality_1
            turn_modality_2 = final_modality_2
            # For AMASS dataset: Simple truncation approach
            # Note: Each AMASS motion sequence is typically only 4-5 seconds long, 
            # so sequences rarely exceed max_seq_length (2048 tokens). 
            # Simple truncation is sufficient instead of complex splitting.
            
            total_length = len(current_input_ids) + len(turn_input_ids) + len(eos_token_ids)
            if total_length > max_seq_length:
                logging.warning(f"Sequence from {conv_id} is too large ({total_length} tokens), truncating to {max_seq_length}")
                
                # Calculate available space for the turn
                available_space = max_seq_length - len(current_input_ids) - len(eos_token_ids)
                
                if available_space > 0:
                    # Truncate the turn to fit
                    turn_input_ids = turn_input_ids[:available_space]
                    turn_labels = turn_labels[:available_space]
                    turn_modality_0 = turn_modality_0[:available_space]
                    turn_modality_1 = turn_modality_1[:available_space]
                    turn_modality_2 = turn_modality_2[:available_space]
                else:
                    # No space left, skip this turn
                    logging.warning(f"No space left for turn from {conv_id}, skipping")
                    continue
            
            # Add this turn to the current sequence
            current_input_ids.extend(turn_input_ids)
            current_labels.extend(turn_labels)
            current_modality_masks_0.extend(turn_modality_0)
            current_modality_masks_1.extend(turn_modality_1)
            current_modality_masks_2.extend(turn_modality_2)
            current_turns.append(first_turn)  # Use first_turn instead of undefined turn
        
        # Finalize the last sequence if it has content
        if current_input_ids and len(current_input_ids) > len(system_prompt_tokens):
            current_input_ids.extend(eos_token_ids)
            current_labels.extend([-100] * len(eos_token_ids))
            current_modality_masks_0.extend([True] * len(eos_token_ids))   # EOS is text
            current_modality_masks_1.extend([False] * len(eos_token_ids))  # EOS is not audio
            current_modality_masks_2.extend([False] * len(eos_token_ids))  # EOS is not motion
            tokenized_record = {
                "input_ids": current_input_ids,
                "labels": current_labels,
                "turns": current_turns,
                "modality_masks_0": current_modality_masks_0,
                "modality_masks_1": current_modality_masks_1,
                "modality_masks_2": current_modality_masks_2,
            }
            process_chunk_to_record(
                tokenized_record,
                conv_id,
                tokenized_records,
                tokenizer,
                motion_fps=motion_fps,
            )
        
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
        "audio_tokens_per_chunk": "embedded_in_answer_text",
        "upper_tokens_per_chunk": upper_segment_size,
        "lower_tokens_per_chunk": lower_segment_size,
        "hand_tokens_per_chunk": hand_segment_size,
        "format_version": "3.0",
        "format_type": "user_assistant_with_system_and_unified_motion_modality_body_only",
        "text_format": "question (text OR audio) + answer text (with embedded audio) + begin_of_motion + interleaved selected motion tokens with unified motion supervision and precomputed position encoding indices",
        "position_encoding": "precomputed_indices_based_on_modality_fps",
        "modality_fps": {"1": 12.5, "2": motion_fps},
        "supervision": "unified motion tokens (selected parts supervised if available)",
        "tokenized": True,
        "max_seq_length": max_seq_length,
        "system_prompt": "text_and_audio_system_prompts_available",
        "assistant_prefix": "included_in_answer_text",
        "modality_masks": "masks_0 for text, masks_1 for audio, masks_2 for unified motion",
        "modality_supervision": "unified_motion_selected_parts",
        "source": "AMASS_question_answer_v3_body_only"
    }
    
    with open(os.path.join(output_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Log position encoding statistics
    if tokenized_records:
        sample_record = tokenized_records[0]
        if "position_encoding_indices" in sample_record:
            pos_indices = sample_record["position_encoding_indices"]
            logging.info(f"Position encoding indices computed successfully")
            logging.info(f"Sample position indices range: {min(pos_indices):.3f} to {max(pos_indices):.3f}")
            logging.info(f"Sample sequence length: {len(pos_indices)}")
        else:
            logging.warning("Position encoding indices not found in records")
    
    logging.info(f"Dataset saved to {output_path}")
    logging.info(f"You can load it with: from datasets import load_from_disk; dataset = load_from_disk('{tokenized_dataset_path}')")
    
    return tokenized_dataset

def main():
    """Main function to preprocess AMASS dataset with question-answer format and MOT support."""
    parser = argparse.ArgumentParser(description="Preprocess AMASS dataset with question-answer format and MOT support")

    # Required arguments
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to AMASS dataset root (e.g., /path/to/AMASS_talking)")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Where to save processed dataset")
    parser.add_argument("--model_name", type=str, default="THUDM/glm-4-voice-9b", 
                       help="Tokenizer model name")
    
    # Data directories
    # parser.add_argument("--audio_dir", type=str, default="audios_token_glm", 
    #                    help="Directory containing audio token files (relative to data_root)")
    parser.add_argument("--transcripts_answer_dir", type=str, default="transcripts_answer", 
                       help="Directory containing answer text files with embedded audio tokens (relative to data_root)")
    parser.add_argument("--transcripts_question_dir", type=str, default="transcripts_question", 
                       help="Directory containing question text files (relative to data_root)")
    parser.add_argument("--audio_question_dir", type=str, default="audios_q_token_glm", 
                       help="Directory containing audio token files for questions (relative to data_root)")                       
    parser.add_argument("--upper_dir", type=str, default="TOKENS_AGENT_25_Rotation/upper", 
                       help="Directory containing upper body token files (relative to data_root)")
    parser.add_argument("--lower_dir", type=str, default="TOKENS_AGENT_25_Rotation/lower", 
                       help="Directory containing lower body token files (relative to data_root)")
    parser.add_argument("--hand_dir", type=str, default="TOKENS_AGENT_25_Rotation/hand_generated", 
                       help="Directory containing hand token files (relative to data_root)")
    # Processing parameters
    parser.add_argument("--audio_fps", type=float, default=12.5, 
                       help="Audio tokens per second")
    parser.add_argument("--upper_fps", type=float, default=6.25, 
                       help="Upper body tokens per second")
    parser.add_argument("--lower_fps", type=float, default=6.25, 
                       help="Lower body tokens per second")
    parser.add_argument("--hand_fps", type=float, default=6.25, 
                       help="Lower body tokens per second")
    parser.add_argument("--max_seq_length", type=int, default=2048, 
                       help="Maximum sequence length")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    parser.add_argument("--limit_videos", type=int, default=1e8,
                       help="Limit number of videos to process (for debugging)")
    parser.add_argument("--split", type=str, choices=["train", "test", "val"], required=True, 
                       help="Which split to process (train, test, or val)")
    parser.add_argument(
        "--motion_variant",
        type=str,
        choices=["body_only", "upper_hand", "lower_only"],
        default="body_only",
        help="Motion subset to encode: full body_only (upper+lower+hand), upper_hand, or lower_only",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting preprocessing of AMASS dataset (question-answer version)")
    logging.info(f"Args: {args}")
    motion_cfg = resolve_motion_variant_config(args.motion_variant)
    logging.info(
        "Using motion variant %s with segment sizes upper=%d lower=%d hand=%d and motion_fps=%.2f",
        args.motion_variant,
        motion_cfg["upper_segment_size"],
        motion_cfg["lower_segment_size"],
        motion_cfg["hand_segment_size"],
        motion_cfg["motion_fps"],
    )

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
    
    # Add special tokens for face, upper body, lower body, and hand modalities
    face_tokens = [f"<|face_{i}|>" for i in range(512)]
    upper_tokens = [f"<|upper_{i}|>" for i in range(256)]
    lower_tokens = [f"<|lower_{i}|>" for i in range(256)]
    hand_tokens = [f"<|hand_{i}|>" for i in range(256)]
    tokenizer.add_tokens(face_tokens, special_tokens=False)
    tokenizer.add_tokens(upper_tokens, special_tokens=False)
    tokenizer.add_tokens(lower_tokens, special_tokens=False)
    tokenizer.add_tokens(hand_tokens, special_tokens=False)
    tokenizer.add_tokens([f"<|begin_of_motion|>"], special_tokens=True)
    tokenizer.add_tokens([f"<|end_of_motion|>"], special_tokens=True)
    print(f"Extended tokenizer vocab size: {len(tokenizer)}")

    # Get list of videos to process (AMASS layout: transcripts_answer directory)
    transcripts_answer_dir = os.path.join(args.data_root, args.transcripts_answer_dir)
    transcripts_question_dir = os.path.join(args.data_root, args.transcripts_question_dir)
    audio_question_dir = os.path.join(args.data_root, args.audio_question_dir)
    upper_dir = os.path.join(args.data_root, args.upper_dir)
    lower_dir = os.path.join(args.data_root, args.lower_dir)
    hand_dir = os.path.join(args.data_root, args.hand_dir)
    sequence_ids = []
    if not os.path.isdir(transcripts_answer_dir) or not os.path.isdir(transcripts_question_dir) or not os.path.isdir(audio_question_dir) or not os.path.isdir(upper_dir) or not os.path.isdir(lower_dir) or not os.path.isdir(hand_dir):
        logging.error("One or more required directories do not exist: transcripts_answer_dir, transcripts_question_dir, audio_question_dir, upper_dir, lower_dir, hand_dir")
        return

    for transcripts_answer_path in Path(transcripts_answer_dir).glob("*.txt"):
        sequence_id = transcripts_answer_path.stem
        
        # Only process sequences that are in the selected split
        if sequence_id not in selected_sequence_ids:
            continue
            
        upper_file = os.path.join(upper_dir, f"{sequence_id}.npy")
        lower_file = os.path.join(lower_dir, f"{sequence_id}.npy")
        hand_file = os.path.join(hand_dir, f"{sequence_id}.npy")
        transcripts_question_file = os.path.join(transcripts_question_dir, f"{sequence_id}.txt")
        audio_question_file = os.path.join(audio_question_dir, f"{sequence_id}.npy")
        # Only require token files for body parts the chosen motion_variant actually uses.
        ok_upper = (motion_cfg["upper_segment_size"] == 0) or os.path.exists(upper_file)
        ok_lower = (motion_cfg["lower_segment_size"] == 0) or os.path.exists(lower_file)
        # Hand is optional even when hand_segment_size > 0 (AMASS upstream usually lacks hand
        # tokens); downstream supervision via has_hand_file=False will mask hand labels to -100.
        ok_hand = True
        if ok_upper and ok_lower and ok_hand and os.path.exists(transcripts_question_file) and os.path.exists(audio_question_file):
            sequence_ids.append(sequence_id)
        else:
            logging.debug(f"Skipping {sequence_id}, missing required files for motion_variant={args.motion_variant}")
        if len(sequence_ids) >= args.limit_videos:
            logging.info(f"Limiting to {len(sequence_ids)} videos")
            break
    
    logging.info(f"Found {len(sequence_ids)} videos with required files")
    
    # Process each video
    all_turns = []
    
    for sequence_id in tqdm(sequence_ids, desc="Processing videos"):

        # Process full video into single A-only segment
        processed_segments = process_full_video(
            sequence_id, transcripts_question_dir, transcripts_answer_dir, audio_question_dir, upper_dir, lower_dir, hand_dir, args.audio_fps, args.upper_fps, args.lower_fps, args.hand_fps
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
                "upper_tokens": seg["upper_tokens"],  # Full upper token array
                "lower_tokens": seg["lower_tokens"],  # Full lower token array
                "hand_tokens": seg["hand_tokens"],  # Hand tokens from actual data
                "speaker_type": "assistant",
                # Pass file existence flags for supervision logic
                "has_upper_file": seg.get("has_upper_file", False),
                "has_lower_file": seg.get("has_lower_file", False),
                "has_hand_file": seg.get("has_hand_file", False),
            })
            
            # Create audio input version
            all_turns.append({
                "conversation_id": f"{seg['segment_id']}_audio",
                "turn_id": f"{seg['segment_id']}_audio",
                "input_type": "audio",
                "question_text": "",  # No text for audio input
                "question_audio_tokens": seg["audio_question_tokens"],  # Audio tokens for question
                "answer_text": seg["transcripts_answer"],  # Full answer text with all embedded audio tokens
                "upper_tokens": seg["upper_tokens"],  # Full upper token array
                "lower_tokens": seg["lower_tokens"],  # Full lower token array
                "hand_tokens": seg["hand_tokens"],  # Hand tokens from actual data
                "speaker_type": "assistant",
                # Pass file existence flags for supervision logic
                "has_upper_file": seg.get("has_upper_file", False),
                "has_lower_file": seg.get("has_lower_file", False),
                "has_hand_file": seg.get("has_hand_file", False),
            })
        logging.info(f"Processed 1 full transcript into 2 samples (text + audio input) for {sequence_id}")

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
            upper_segment_size=motion_cfg["upper_segment_size"],
            lower_segment_size=motion_cfg["lower_segment_size"],
            hand_segment_size=motion_cfg["hand_segment_size"],
            motion_fps=motion_cfg["motion_fps"],
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
