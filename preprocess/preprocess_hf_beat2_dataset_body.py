#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess BEAT2 dataset into A-only sequences (assistant-only) for Rotation version,
organizing audio and motion tokens into fixed-size chunks with unified motion modality.
Body only version without face tokens.

This script targets the BEAT2 dataset at
  /path/to/BEAT2/beat_english_v2.0.0
and assumes:
- Audio tokens under data_root/audios_token_glm
- Upper, lower, hand tokens under data_root/TOKENS_AGENT_25/{upper,lower,hand}
- Transcripts under data_root/textgrid (TextGrid format)
- Split information in data_root/train_test_split.csv

Key features for Rotation version (Body Only):
- Supports train/test/val splits with 'additional' data included
- Uses all speakers [1-30] as specified
- Uses TextGrid format for transcript parsing with word-level timestamps
- Processes upper, lower, hand motion tokens with unified modality (no face)
- Audio FPS: 12.5, Motion FPS: 18.75 (6.25*3) for body only
- Group size: text + 26 audio + 1 begin_of_motion + 39 interleaved motion tokens per group
- Token breakdown: 1 begin_of_motion + 39 motion tokens (13 upper + 13 lower + 13 hand in 1:1:1 alternating pattern)
- 3 modalities: text(0), audio(1), motion(2)

Usage:
    python preprocess_hf_beat2_dataset_body.py \
        --data_root /path/to/BEAT2/beat_english_v2.0.0 \
        --output_path ./processed_beat2_body_only_train \
        --split train
"""

import os
import json
import numpy as np
from pathlib import Path
import random
import re
import csv
import logging
from tqdm import tqdm
import argparse
from datasets import Dataset, DatasetDict, Features, Value, Array2D, Array3D
from transformers import AutoTokenizer
import pandas as pd
import pickle
import glob
import torch
import traceback
import datetime
import sys
import textgrid as tg
from typing import Optional, Tuple, Union, List, Dict, Any




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
                     Default: {1: 12.5, 2: 18.75} (Body only version, 3 modalities only, motion=18.75fps=6.25*3)
    
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
        
        # Process modality 2 (motion) and collect all tokens with timestamps
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
                # For unified motion modality (modality 2), use -0.5 offset for begin_of_motion token
                start_offset = -0.5
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

def clean_transcript_text(text):
    """
    Clean transcript text, removing special characters and formatting issues.
    
    Args:
        text: Original transcript text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra spaces and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove timestamp markers
    text = re.sub(r'\[\d+:\d+:\d+\]', '', text)
    text = re.sub(r'\[\d+:\d+\]', '', text)
    
    # Replace common non-standard punctuation
    text = text.replace('..', '…').replace('...', '…')
    
    return text

def parse_full_transcript(textgrid_dir: str, video_id: str):
    """
    Load and parse the full transcript from TextGrid for a given video_id.
    Returns (full_text, word_timestamps) where word_timestamps is a list of (word, start, end).
    Expected file: textgrid_dir/{video_id}.TextGrid
    """
    textgrid_file = os.path.join(textgrid_dir, f"{video_id}.TextGrid")
    
    if not os.path.exists(textgrid_file):
        logging.warning(f"TextGrid not found for {video_id} in {textgrid_dir}")
        return "", []

    try:
        tgrid = tg.TextGrid.fromFile(textgrid_file)
        
        # Extract words from the first tier (typically 'words' tier)
        word_timestamps = []
        full_words = []
        
        if len(tgrid) > 0:
            word_tier = tgrid[0]  # First tier contains word-level annotations
            for interval in word_tier:
                word_text = interval.mark.strip()
                start_time = interval.minTime
                end_time = interval.maxTime
                
                # Skip empty intervals
                if word_text and word_text != "":
                    # Clean the word text
                    cleaned_word = clean_transcript_text(word_text)
                    if cleaned_word:
                        word_timestamps.append((cleaned_word, start_time, end_time))
                        full_words.append(cleaned_word)
        
        # Join all words to create full text
        full_text = " ".join(full_words) if full_words else ""
        
        return full_text, word_timestamps
        
    except Exception as e:
        logging.error(f"Error reading TextGrid for {video_id}: {e}")
        return "", []

def load_speaking_track_segments(transcript_dir, video_id):
    # Deprecated in synthetic pipeline (no segment parsing). Kept for compatibility if needed.
    return []



def merge_close_segments(segments, gap_threshold=2.0):
    # Deprecated in synthetic pipeline (no segment parsing). Kept for compatibility if needed.
    return segments




def get_tokens_for_timerange(tokens_with_timestamps, start_time, end_time):
    """
    Extract tokens that fall within a specific time range.
    
    Args:
        tokens_with_timestamps: List of (token, timestamp) tuples
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        List of (token, timestamp) tuples within the time range
    """
    return [(token, ts) for token, ts in tokens_with_timestamps if start_time <= ts < end_time]

def process_full_video(video_id, transcript_text, audio_dir, upper_dir, lower_dir, hand_dir,
                       audio_fps=12.5, upper_fps=6.25, lower_fps=6.25, hand_fps=6.25):
    """
    Process transcript segments into format suitable for dataset creation.
    
    Args:
        video_id: Video ID
        transcript_text: Full transcript text
        audio_dir: Directory containing audio tokens
        upper_dir: Directory containing upper body tokens
        lower_dir: Directory containing lower body tokens
        hand_dir: Directory containing hand tokens
        audio_fps: Audio token frame rate
        upper_fps: Upper body token frame rate
        lower_fps: Lower body token frame rate
        hand_fps: Hand token frame rate
        
    Returns:
        List of processed segments with text, audio, upper, lower, and hand body tokens
    """
    processed_segments = []

    # Load audio and motion tokens
    # All tokens are saved as complete video files (e.g., 1_wayne_0_1_1.npy)
    audio_token_file = os.path.join(audio_dir, f"{video_id}.npy")
    upper_token_file = os.path.join(upper_dir, f"{video_id}.npy")
    lower_token_file = os.path.join(lower_dir, f"{video_id}.npy")
    hand_token_file = os.path.join(hand_dir, f"{video_id}.npy")
    if not os.path.exists(audio_token_file):
        logging.warning(f"No audio tokens found for {video_id}, skipping")
        return []

    if not os.path.exists(upper_token_file):
        logging.warning(f"No upper body tokens found for {video_id}, skipping")
        return []
    
    if not os.path.exists(lower_token_file):
        logging.warning(f"No lower body tokens found for {video_id}, skipping")
        return []
    
    if not os.path.exists(hand_token_file):
        logging.warning(f"No hand tokens found for {video_id}, skipping")
        return []
    
    # Load tokens
    try:
        audio_data = np.load(audio_token_file, allow_pickle=True)
        # Create list of (token, timestamp) tuples based on audio FPS
        audio_tokens_ts = [(audio_data[i], i / audio_fps) for i in range(len(audio_data))]
        logging.info(f"Loaded {len(audio_data)} audio tokens for {video_id}, duration: {len(audio_data) / audio_fps:.2f}s")
    except Exception as e:
        logging.error(f"Error loading audio tokens for {video_id}: {e}")
        return []
    
    # No face tokens in body only version
    
    # Load upper, lower, and hand body tokens
    try:
        upper_data = np.load(upper_token_file, allow_pickle=True)
        lower_data = np.load(lower_token_file, allow_pickle=True)
        hand_data = np.load(hand_token_file, allow_pickle=True)
        if upper_data.ndim > 1:
            upper_data = upper_data[0]  # Take first element if nested
        if lower_data.ndim > 1:
            lower_data = lower_data[0]  # Take first element if nested
        if hand_data.ndim > 1:
            hand_data = hand_data[0]  # Take first element if nested
        # Create list of (token, timestamp) tuples based on respective FPS
        upper_tokens_ts = [(upper_data[i], i / upper_fps) for i in range(len(upper_data))]
        lower_tokens_ts = [(lower_data[i], i / lower_fps) for i in range(len(lower_data))]
        hand_tokens_ts = [(hand_data[i], i / hand_fps) for i in range(len(hand_data))]
        logging.info(f"Loaded {len(upper_data)} upper body tokens for {video_id}, duration: {len(upper_data) / upper_fps:.2f}s")
        logging.info(f"Loaded {len(lower_data)} lower body tokens for {video_id}, duration: {len(lower_data) / lower_fps:.2f}s")
        logging.info(f"Loaded {len(hand_data)} hand tokens for {video_id}, duration: {len(hand_data) / hand_fps:.2f}s")
    except Exception as e:
        logging.error(f"Error loading upper/lower/hand body tokens for {video_id}: {e}")
        return []
    
    # Build a single A-only segment covering the full duration
    start_time = 0.0
    end_time = len(audio_data) / audio_fps if audio_fps > 0 else 0.0
    text = transcript_text
    word_timestamps = []

    segment_audio = audio_tokens_ts
    segment_upper = upper_tokens_ts
    segment_lower = lower_tokens_ts
    segment_hand = hand_tokens_ts
    if segment_audio and text:
        processed_segments.append({
            "segment_id": f"{video_id}_full",
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "text": text,
            "audio_tokens": [t for t, _ in segment_audio],
            "upper_tokens": [t for t, _ in segment_upper],
            "lower_tokens": [t for t, _ in segment_lower],
            "hand_tokens": [t for t, _ in segment_hand],
            "word_timestamps": word_timestamps
        })

    return processed_segments

def interleave_groups_from_full_segment(segment, audio_group_size: int = 26, upper_group_size: int = 13, lower_group_size: int = 13, hand_group_size: int = 13, audio_fps: float = 12.5):
    """
    From one full A-only segment, build interleaved groups following BEAT2 format:
    per group: text (words in time window), 26 audio tokens, 1 begin_of_motion token, 39 interleaved motion tokens (13 upper + 13 lower + 13 hand in 1:1:1 alternating pattern).
    """
    audio_tokens = segment["audio_tokens"]
    upper_tokens = segment["upper_tokens"]
    lower_tokens = segment["lower_tokens"]
    hand_tokens = segment.get("hand_tokens", [])
    text = segment["text"]
    word_timestamps = segment.get("word_timestamps", [])
    start_time = float(segment.get("start_time", 0.0))

    groups = []
    total_groups = max(1, (len(audio_tokens) + audio_group_size - 1) // audio_group_size)
    for i in range(total_groups):
        a_start = i * audio_group_size
        a_end = min((i + 1) * audio_group_size, len(audio_tokens))
        chunk_audio = audio_tokens[a_start:a_end]
        if len(chunk_audio) < audio_group_size:
            chunk_audio = chunk_audio + [0] * (audio_group_size - len(chunk_audio))

        # No face tokens in body only version

        u_start = i * upper_group_size
        u_end = min((i + 1) * upper_group_size, len(upper_tokens))
        chunk_upper = upper_tokens[u_start:u_end] if u_start < len(upper_tokens) else []
        orig_upper_count = len(chunk_upper)
        if len(chunk_upper) < upper_group_size:
            chunk_upper = chunk_upper + [0] * (upper_group_size - len(chunk_upper))

        l_start = i * lower_group_size
        l_end = min((i + 1) * lower_group_size, len(lower_tokens))
        chunk_lower = lower_tokens[l_start:l_end] if l_start < len(lower_tokens) else []
        orig_lower_count = len(chunk_lower)
        if len(chunk_lower) < lower_group_size:
            chunk_lower = chunk_lower + [0] * (lower_group_size - len(chunk_lower))

        h_start = i * hand_group_size
        h_end = min((i + 1) * hand_group_size, len(hand_tokens))
        chunk_hand = hand_tokens[h_start:h_end] if h_start < len(hand_tokens) else []
        orig_hand_count = len(chunk_hand)
        if len(chunk_hand) < hand_group_size:
            chunk_hand = chunk_hand + [0] * (hand_group_size - len(chunk_hand))

        # Time window for this group from audio indices
        group_start_time = start_time + (a_start / max(audio_fps, 1e-6))
        group_end_time = start_time + (a_end / max(audio_fps, 1e-6))
        if word_timestamps:
            words = [w for (w, s, e) in word_timestamps if e > group_start_time and s < group_end_time]
            group_text = " ".join(words)
        else:
            group_text = text if i == 0 else ""

        groups.append({
            "text": group_text,
            "audio_tokens": chunk_audio,
            "upper_tokens": chunk_upper,
            "lower_tokens": chunk_lower,
            "hand_tokens": chunk_hand,
            "orig_upper_count": orig_upper_count,
            "orig_lower_count": orig_lower_count,
            "orig_hand_count": orig_hand_count,
        })
    return groups

def process_chunk_to_record(chunk, conv_id, tokenized_records, motion_fps=18.75):
    """
    Process a chunk of turns into a tokenized record with MOT support and position encoding indices.
    
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
    modality_masks = [
        chunk["modality_masks_0"],  # Text tokens (modality 0)
        chunk["modality_masks_1"],  # Audio tokens (modality 1) 
        chunk["modality_masks_2"],  # Motion tokens (modality 2) - unified face, upper, lower, hand
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
        "modality_masks_2": chunk["modality_masks_2"],  # True for motion tokens (unified face, upper, lower, hand)
        "position_encoding_indices": position_encoding_indices,  # Precomputed position indices
    }
    tokenized_records.append(tokenized_record)

def convert_to_huggingface_dataset(
    output_path,
    interleaved_turns,
    tokenizer_name,
    max_seq_length=1024,
    audio_segment_size=26,
    upper_segment_size=13,
    lower_segment_size=13,
    hand_segment_size=13,
    motion_fps: float = 18.75,
    limit_sequences: int | None = None,
    audio_fps: float = 12.5,
    split="train",
):
    """
    Convert interleaved turns into a HuggingFace dataset with tokenized format and MOT support.
    Following BEAT2 format: assistant-only, no system prompt, motion supervision with unified modality.
    Packs multiple turns into sequences up to max_seq_length.
    Body only version: upper, lower, and hand tokens in 1:1:1 interleaved pattern.
    
    Args:
        output_path: Where to save the processed dataset
        interleaved_turns: List of processed chunks
        tokenizer_name: Name of the tokenizer to use
        max_seq_length: Maximum sequence length for tokenization
        audio_segment_size: Number of audio tokens per group (default: 26)
        upper_segment_size: Number of upper body tokens per group (default: 13, used in interleaved pattern)
        lower_segment_size: Number of lower body tokens per group (default: 13, used in interleaved pattern)
        hand_segment_size: Number of hand tokens per group (default: 13, used in interleaved pattern)
    Returns:
        Dataset: The created HuggingFace dataset
    """
    logging.info(f"Converting to Hugging Face dataset (MOT version) with tokenizer: {tokenizer_name}")
    os.makedirs(output_path, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token
    
    # Add special tokens for face, upper body, lower body, and hand modalities
    face_tokens = [f"<|face_{i}|>" for i in range(512)] ###  512 codebook size for face
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
    
    # Prepare assistant prefix (following TFHP format)
    assistant_prefix = "<|assistant|>streaming_transcription\n"
    assistant_prefix_tokens = tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
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
        # All turns are assistant turns in this version
        asst_turns = turns
        if not asst_turns:
            continue
        
        # Initialize accumulator for packing multiple turns
        current_input_ids = []
        current_labels = []
        current_turns = []
        current_modality_masks_0 = []
        current_modality_masks_1 = []
        current_modality_masks_2 = []
        
        for turn_idx, asst_turn in enumerate(asst_turns):
            # Prepare content tokens for this turn
            asst_text = asst_turn.get("text", "")
            text_tokens = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
            text_labels = [-100] * len(text_tokens)  # Text doesn't contribute to loss
            text_modality_0 = [True] * len(text_tokens)   # Text is modality 0
            text_modality_1 = [False] * len(text_tokens)  # Text is not audio
            text_modality_2 = [False] * len(text_tokens)  # Text is not motion
            
            # Process audio tokens (already grouped 26 per turn)
            audio_tokens = asst_turn.get("audio_tokens", [])
            audio_token_ids = []
            for token in audio_tokens:
                if isinstance(token, np.ndarray):
                    token_val = token.tolist() if hasattr(token, 'tolist') else token
                else:
                    token_val = token
                token_str = safe_token_to_string(token_val, "audio")
                ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
                audio_token_ids.extend(ids)
            audio_labels = [-100] * len(audio_token_ids)
            audio_modality_0 = [False] * len(audio_token_ids)  # Audio is not text
            audio_modality_1 = [True] * len(audio_token_ids)   # Audio is modality 1
            audio_modality_2 = [False] * len(audio_token_ids)  # Audio is not motion

            # Process motion tokens in 1:1:1 interleaved pattern: upper, lower, hand, upper, lower, hand...
            upper_tokens = asst_turn.get("upper_tokens", []) if upper_segment_size > 0 else []
            lower_tokens = asst_turn.get("lower_tokens", []) if lower_segment_size > 0 else []
            hand_tokens = asst_turn.get("hand_tokens", []) if hand_segment_size > 0 else []
            orig_upper_count = asst_turn.get("orig_upper_count", len(upper_tokens))
            orig_lower_count = asst_turn.get("orig_lower_count", len(lower_tokens))
            orig_hand_count = asst_turn.get("orig_hand_count", len(hand_tokens))
            
            # Initialize motion token arrays
            motion_token_ids = []
            motion_labels = []
            motion_modality_0 = []
            motion_modality_1 = []
            motion_modality_2 = []
            
            # Add unified begin_of_motion token
            if upper_tokens or lower_tokens or hand_tokens:
                begin_motion_tokens = tokenizer("<|begin_of_motion|>", add_special_tokens=False)["input_ids"]
                motion_token_ids.extend(begin_motion_tokens)
                motion_labels.extend([-100] * len(begin_motion_tokens))  # Begin token is NOT supervised
                motion_modality_0.extend([False] * len(begin_motion_tokens))  # Begin is not text
                motion_modality_1.extend([False] * len(begin_motion_tokens))  # Begin is not audio
                motion_modality_2.extend([True] * len(begin_motion_tokens))   # Begin is modality 2 (motion)
            
            # Process tokens in 1:1:1 interleaved pattern
            # Pattern: upper_0, lower_0, hand_0, upper_1, lower_1, hand_1, ...
            upper_idx = 0
            lower_idx = 0
            hand_idx = 0
            
            while upper_idx < len(upper_tokens) or lower_idx < len(lower_tokens) or hand_idx < len(hand_tokens):
                
                # Add 1 upper token
                if upper_idx < len(upper_tokens):
                    upper_token = upper_tokens[upper_idx]
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
                if lower_idx < len(lower_tokens):
                    lower_token = lower_tokens[lower_idx]
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
                if hand_idx < len(hand_tokens):
                    hand_token = hand_tokens[hand_idx]
                    if isinstance(hand_token, np.ndarray):
                        hand_token_val = hand_token.tolist() if hasattr(hand_token, 'tolist') else hand_token
                    else:
                        hand_token_val = hand_token
                    hand_token_str = safe_token_to_string(hand_token_val, "hand")
                    hand_token_ids = tokenizer(hand_token_str, add_special_tokens=False)["input_ids"]
                    motion_token_ids.extend(hand_token_ids)
                    is_padding = (hand_idx >= orig_hand_count)
                    motion_labels.extend([-100] * len(hand_token_ids) if is_padding else hand_token_ids)
                    motion_modality_0.extend([False] * len(hand_token_ids))  # Hand is not text
                    motion_modality_1.extend([False] * len(hand_token_ids))  # Hand is not audio
                    motion_modality_2.extend([True] * len(hand_token_ids))   # Hand is modality 2 (motion)
                    hand_idx += 1

            # Compose the full turn with interleaved motion tokens
            turn_input_ids = text_tokens + audio_token_ids + motion_token_ids
            turn_labels = text_labels + audio_labels + motion_labels
            turn_modality_0 = text_modality_0 + audio_modality_0 + motion_modality_0
            turn_modality_1 = text_modality_1 + audio_modality_1 + motion_modality_1
            turn_modality_2 = text_modality_2 + audio_modality_2 + motion_modality_2
            
            # Check if this single turn is too large to fit in any sequence (even alone)
            min_required_space = len(assistant_prefix_tokens) + len(eos_token_ids)
            if len(turn_input_ids) + min_required_space > max_seq_length:
                logging.warning(f"Turn {turn_idx} from {conv_id} is too large ({len(turn_input_ids)} tokens), splitting across multiple sequences")
                
                # First, finalize current sequence if it has content
                if current_input_ids and len(current_input_ids) > len(assistant_prefix_tokens):
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
                    process_chunk_to_record(tokenized_record, conv_id, tokenized_records, motion_fps=motion_fps)
                
                # Split the oversized turn across multiple sequences
                turn_offset = 0
                chunk_idx = 0
                while turn_offset < len(turn_input_ids):
                    # Start new sequence with assistant prefix
                    current_input_ids = assistant_prefix_tokens.copy()
                    current_labels = [-100] * len(assistant_prefix_tokens)
                    current_modality_masks_0 = [True] * len(assistant_prefix_tokens)
                    current_modality_masks_1 = [False] * len(assistant_prefix_tokens)
                    current_modality_masks_2 = [False] * len(assistant_prefix_tokens)
                    
                    # Calculate how much of the turn we can fit
                    available_space = max_seq_length - len(current_input_ids) - len(eos_token_ids)
                    chunk_size = min(available_space, len(turn_input_ids) - turn_offset)
                    
                    if chunk_size <= 0:
                        logging.error(f"Cannot fit any tokens from oversized turn, skipping remainder")
                        break
                    
                    # Add chunk of the turn
                    current_input_ids.extend(turn_input_ids[turn_offset:turn_offset + chunk_size])
                    current_labels.extend(turn_labels[turn_offset:turn_offset + chunk_size])
                    current_modality_masks_0.extend(turn_modality_0[turn_offset:turn_offset + chunk_size])
                    current_modality_masks_1.extend(turn_modality_1[turn_offset:turn_offset + chunk_size])
                    current_modality_masks_2.extend(turn_modality_2[turn_offset:turn_offset + chunk_size])
                    
                    # Add EOS tokens
                    current_input_ids.extend(eos_token_ids)
                    current_labels.extend([-100] * len(eos_token_ids))
                    current_modality_masks_0.extend([True] * len(eos_token_ids))
                    current_modality_masks_1.extend([False] * len(eos_token_ids))
                    current_modality_masks_2.extend([False] * len(eos_token_ids))
                    
                    # Create turn metadata for this chunk
                    chunk_turn = asst_turn.copy()
                    chunk_turn["turn_id"] = f"{asst_turn.get('turn_id', 'turn')}_chunk_{chunk_idx}"
                    
                    tokenized_record = {
                        "input_ids": current_input_ids,
                        "labels": current_labels,
                        "turns": [chunk_turn],
                        "modality_masks_0": current_modality_masks_0,
                        "modality_masks_1": current_modality_masks_1,
                        "modality_masks_2": current_modality_masks_2,
                    }
                    process_chunk_to_record(tokenized_record, conv_id, tokenized_records, motion_fps=motion_fps)
                    
                    turn_offset += chunk_size
                    chunk_idx += 1
                    
                    if limit_sequences is not None and len(tokenized_records) >= limit_sequences:
                        break
                
                # Reset for next turn
                current_input_ids = []
                current_labels = []
                current_modality_masks_0 = []
                current_modality_masks_1 = []
                current_modality_masks_2 = []
                current_turns = []
                continue  # Skip to next turn
            else:
                # Normal case: single turn can fit in a sequence
                # Now check if adding this turn to current sequence would exceed max_seq_length
                
                # If starting a new sequence, add the assistant prefix
                if not current_input_ids:
                    current_input_ids.extend(assistant_prefix_tokens)
                    current_labels.extend([-100] * len(assistant_prefix_tokens))
                    current_modality_masks_0.extend([True] * len(assistant_prefix_tokens))   # Prefix is text
                    current_modality_masks_1.extend([False] * len(assistant_prefix_tokens))  # Prefix is not audio
                    current_modality_masks_2.extend([False] * len(assistant_prefix_tokens))  # Prefix is not face
                
                # Check if adding this turn to current sequence would exceed max_seq_length (including EOS)
                # This is different from the first check: first check was for single turn alone, this is for current sequence + turn
                if len(current_input_ids) + len(turn_input_ids) + len(eos_token_ids) > max_seq_length:
                    # Finalize current sequence
                    current_input_ids.extend(eos_token_ids)
                    current_labels.extend([-100] * len(eos_token_ids))
                    current_modality_masks_0.extend([True] * len(eos_token_ids))   # EOS is text
                    current_modality_masks_1.extend([False] * len(eos_token_ids))  # EOS is not audio
                    current_modality_masks_2.extend([False] * len(eos_token_ids))
  # EOS is not upper body
                    
                    tokenized_record = {
                        "input_ids": current_input_ids,
                        "labels": current_labels,
                        "turns": current_turns,
                        "modality_masks_0": current_modality_masks_0,
                        "modality_masks_1": current_modality_masks_1,
                        "modality_masks_2": current_modality_masks_2,
                    }
                    process_chunk_to_record(tokenized_record, conv_id, tokenized_records, motion_fps=motion_fps)
                    
                    # Start a new sequence
                    current_input_ids = assistant_prefix_tokens.copy()
                    current_labels = [-100] * len(assistant_prefix_tokens)
                    current_modality_masks_0 = [True] * len(assistant_prefix_tokens)
                    current_modality_masks_1 = [False] * len(assistant_prefix_tokens)
                    current_modality_masks_2 = [False] * len(assistant_prefix_tokens)
                    current_turns = []
                    
                    if limit_sequences is not None and len(tokenized_records) >= limit_sequences:
                        break
                
                # Add this turn to the current sequence
                current_input_ids.extend(turn_input_ids)
                current_labels.extend(turn_labels)
                current_modality_masks_0.extend(turn_modality_0)
                current_modality_masks_1.extend(turn_modality_1)
                current_modality_masks_2.extend(turn_modality_2)
                current_turns.append(asst_turn)
        
        # Finalize the last sequence if it has content
        if current_input_ids and len(current_input_ids) > len(assistant_prefix_tokens):
            current_input_ids.extend(eos_token_ids)
            current_labels.extend([-100] * len(eos_token_ids))
            current_modality_masks_0.extend([True] * len(eos_token_ids))   # EOS is text
            current_modality_masks_1.extend([False] * len(eos_token_ids))  # EOS is not audio
            current_modality_masks_2.extend([False] * len(eos_token_ids))
            
            tokenized_record = {
                "input_ids": current_input_ids,
                "labels": current_labels,
                "turns": current_turns,
                "modality_masks_0": current_modality_masks_0,
                "modality_masks_1": current_modality_masks_1,
                "modality_masks_2": current_modality_masks_2,
            }
            process_chunk_to_record(tokenized_record, conv_id, tokenized_records, motion_fps=motion_fps)
        
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
        "audio_tokens_per_chunk": audio_segment_size,
        "upper_tokens_per_chunk": upper_segment_size,
        "lower_tokens_per_chunk": lower_segment_size,
        "hand_tokens_per_chunk": hand_segment_size,
        "format_version": "3.0",
        "format_type": "assistant_only_mot_with_position_encoding_body_only",
        "text_format": "text+audio+interleaved_motion_tokens (selected motion parts) with motion supervision and precomputed position encoding indices",
        "position_encoding": "precomputed_indices_based_on_modality_fps",
        "modality_fps": {"1": 12.5, "2": motion_fps},
        "supervision": "selected body motion tokens",
        "tokenized": True,
        "max_seq_length": max_seq_length,
        "assistant_prefix": assistant_prefix.strip(),
        "modality_masks": "masks_0 for text, masks_1 for audio, masks_2 for motion (selected motion parts)",
        "modality_supervision": "selected_motion_parts",
        "source": "BEAT2_A_only_body_only"
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


def main():
    """Main function to preprocess BEAT2 dataset (A-only version) with MOT support."""
    parser = argparse.ArgumentParser(description="Preprocess BEAT2 dataset (A-only) with MOT support")

    # Required arguments
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to BEAT2 dataset root (e.g., /path/to/BEAT2/beat_english_v2.0.0)")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Where to save processed dataset")
    parser.add_argument("--model_name", type=str, default="THUDM/glm-4-voice-9b", 
                       help="Tokenizer model name")
    
    # Data directories
    parser.add_argument("--audio_dir", type=str, default="audios_token_glm", 
                       help="Directory containing audio token files (relative to data_root)")
    parser.add_argument("--upper_dir", type=str, default="TOKENS_AGENT_25/upper", 
                       help="Directory containing upper body token files (relative to data_root)")
    parser.add_argument("--lower_dir", type=str, default="TOKENS_AGENT_25/lower", 
                       help="Directory containing lower body token files (relative to data_root)")
    parser.add_argument("--hand_dir", type=str, default="TOKENS_AGENT_25/hand", 
                       help="Directory containing hand token files (relative to data_root)")
    # Processing parameters
    parser.add_argument("--audio_fps", type=float, default=12.5, 
                       help="Audio tokens per second")
    parser.add_argument("--upper_fps", type=float, default=6.25, 
                       help="Upper body tokens per second")
    parser.add_argument("--lower_fps", type=float, default=6.25, 
                       help="Lower body tokens per second")
    parser.add_argument("--hand_fps", type=float, default=6.25, 
                       help="Hand tokens per second")
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
    logging.info("Starting preprocessing of BEAT2 dataset (A-only body only version)")
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
    split_file = os.path.join(args.data_root, "train_test_split.csv")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    split_rule = pd.read_csv(split_file)
    # Filter for the specified split, using all speakers [1-30]
    training_speakers = list(range(1, 31))  # [1,2,3,...,30]
    selected_videos = split_rule.loc[
        (split_rule['type'] == args.split) &
        (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))
    ]
    
    # Also include 'additional' data for all splits
    additional_videos = split_rule.loc[
        (split_rule['type'] == 'additional') & 
        (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))
    ]
    selected_videos = pd.concat([selected_videos, additional_videos])
    selected_video_ids = set(selected_videos['id'].tolist())
    
    logging.info(f"Loaded {len(selected_video_ids)} videos for {args.split} split from speakers {training_speakers}")

    # Get list of videos to process (BEAT2 layout: textgrid directory)
    textgrid_dir = os.path.join(args.data_root, "textgrid")
    audio_dir = os.path.join(args.data_root, args.audio_dir)
    upper_dir = os.path.join(args.data_root, args.upper_dir)
    lower_dir = os.path.join(args.data_root, args.lower_dir)
    hand_dir = os.path.join(args.data_root, args.hand_dir)
    video_ids = []
    if not os.path.isdir(audio_dir) or not os.path.isdir(upper_dir) or not os.path.isdir(lower_dir) or not os.path.isdir(hand_dir) or not os.path.isdir(textgrid_dir):
        logging.error("One or more required directories do not exist: audio_dir, upper_dir, lower_dir, hand_dir, textgrid_dir")
        return

    for audio_path in Path(audio_dir).glob("*.npy"):
        video_id = audio_path.stem
        
        # Only process videos that are in the selected split
        if video_id not in selected_video_ids:
            continue
            
        upper_file = os.path.join(upper_dir, f"{video_id}.npy")
        lower_file = os.path.join(lower_dir, f"{video_id}.npy")
        hand_file = os.path.join(hand_dir, f"{video_id}.npy")
        textgrid_file = os.path.join(textgrid_dir, f"{video_id}.TextGrid")
        # Only require token files for body parts the chosen motion_variant actually uses.
        ok_upper = (motion_cfg["upper_segment_size"] == 0) or os.path.exists(upper_file)
        ok_lower = (motion_cfg["lower_segment_size"] == 0) or os.path.exists(lower_file)
        ok_hand  = (motion_cfg["hand_segment_size"]  == 0) or os.path.exists(hand_file)
        if ok_upper and ok_lower and ok_hand and os.path.exists(textgrid_file):
            video_ids.append(video_id)
        else:
            logging.debug(f"Skipping {video_id}, missing required files for motion_variant={args.motion_variant}")
        if len(video_ids) >= args.limit_videos:
            logging.info(f"Limiting to {len(video_ids)} videos")
            break
    
    logging.info(f"Found {len(video_ids)} videos with required files")
    
    # Process each video
    all_turns = []
    
    for video_id in tqdm(video_ids, desc="Processing videos"):
        # Load full transcript text and word timestamps (A-only)
        transcript_text, word_ts = parse_full_transcript(textgrid_dir, video_id)
        if not transcript_text:
            logging.warning(f"No transcript found for {video_id}, skipping")
            continue

        # Process full video into single A-only segment
        processed_segments = process_full_video(
            video_id, transcript_text, audio_dir, upper_dir, lower_dir, hand_dir, args.audio_fps, args.upper_fps, args.lower_fps, args.hand_fps
        )
        # Attach word timestamps into the only segment if present
        if processed_segments and word_ts:
            processed_segments[0]["word_timestamps"] = word_ts
        
        if not processed_segments:
            logging.warning(f"No valid segments processed for {video_id}")
            continue
        
        # Interleave into BEAT2-style groups (text + 26 audio + 1 begin_of_motion + 39 interleaved motion tokens per group)
        for seg in processed_segments:
            groups = interleave_groups_from_full_segment(
                seg,
                audio_group_size=26,
                upper_group_size=motion_cfg["upper_segment_size"],
                lower_group_size=motion_cfg["lower_segment_size"],
                hand_group_size=motion_cfg["hand_segment_size"],
                audio_fps=args.audio_fps,
            )
            for idx, g in enumerate(groups):
                all_turns.append({
                    "conversation_id": seg["video_id"],
                    "turn_id": f"{seg['segment_id']}_group_{idx}",
                    "text": g["text"],
                    "audio_tokens": g["audio_tokens"],
                    "upper_tokens": g["upper_tokens"],
                    "lower_tokens": g["lower_tokens"],
                    "hand_tokens": g["hand_tokens"],
                    "orig_upper_count": g["orig_upper_count"],
                    "orig_lower_count": g["orig_lower_count"],
                    "orig_hand_count": g["orig_hand_count"],
                    "speaker_type": "assistant",
                })
        logging.info(f"Processed 1 full transcript into {len(groups)} groups for {video_id}")

        if args.debug:
            logging.info(f"Debug mode enabled, processing only 1 video")
            break
    
    logging.info(f"Total interleaved groups created: {len(all_turns)}")
    
    # Convert to HuggingFace dataset
    if all_turns:
        dataset = convert_to_huggingface_dataset(
            output_path=args.output_path,
            interleaved_turns=all_turns,
            tokenizer_name=args.model_name,
            max_seq_length=args.max_seq_length,
            audio_segment_size=26,
            upper_segment_size=motion_cfg["upper_segment_size"],
            lower_segment_size=motion_cfg["lower_segment_size"],
            hand_segment_size=motion_cfg["hand_segment_size"],
            motion_fps=motion_cfg["motion_fps"],
            limit_sequences=(100 if args.debug else None),
            audio_fps=args.audio_fps,
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
