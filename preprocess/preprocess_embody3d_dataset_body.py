#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess Embody3D dataset into A-only sequences (assistant-only) for Rotation version,
organizing audio and motion tokens into fixed-size chunks with unified motion modality.
Body only version without face tokens.

This script targets the Embody3D dataset at
  /path/to/embody_3d/datasets/aiagent
and assumes:
- Audio tokens under data_root/c--*/**/tokens/audio/{filename}.npy
- Upper, lower, hand tokens under data_root/c--*/**/tokens/{upper,lower,hand}/{filename}.npy
- Transcripts under data_root/c--*/**/audio_separated/*.json (JSON format with word-level timestamps)
- WAV files under data_root/c--*/**/audio_separated/*.wav

Key features for Rotation version (Body Only):
- Processes all files found in the dataset structure
- Uses JSON format for transcript parsing with word-level timestamps
- Processes upper, lower, hand motion tokens with unified modality (no face)
- Audio FPS: 12.5, Motion FPS: 18.75 (6.25*3) for body only
- Group size: text + 26 audio + 1 begin_of_motion + 39 interleaved motion tokens per group
- Token breakdown: 1 begin_of_motion + 39 motion tokens (13 upper + 13 lower + 13 hand in 1:1:1 alternating pattern)
- 3 modalities: text(0), audio(1), motion(2)

Usage:
    python preprocess_embody3d_dataset_w_transcript_tokenized_varying_mot_encode_position_body_only.py \
        --data_root /path/to/embody_3d/datasets/aiagent \
        --output_path /path/to/embody_3d \
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
import pickle
import glob
import torch
import traceback
import datetime
import sys
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
        
        # Calculate time duration for this mod1 cycle
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
                        K = len(mod1_group) - 1  # Number of intervals between mod1 anchors
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
                # Multiple mod1 tokens, determine interval assignment
                K = len(mod1_rope_indices) - 1  # Number of intervals between mod1 anchors
                
                # Group regular tokens by interval
                intervals = []
                for i in range(K):
                    intervals.append([])
                tail_tokens = []
                
                # Assign each regular token to its corresponding interval
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
                
                # Process tokens within intervals (interpolation)
                for interval_idx, interval_tokens in enumerate(intervals):
                    if len(interval_tokens) > 0:
                        start_rope_idx = mod1_rope_indices[interval_idx]
                        end_rope_idx = mod1_rope_indices[interval_idx + 1]
                        
                        # Uniformly distribute tokens in this interval
                        for i, token_info in enumerate(interval_tokens):
                            pos = token_info['position']
                            alpha = (i + 1) / (len(interval_tokens) + 1)
                            interpolated_position = start_rope_idx + alpha * (end_rope_idx - start_rope_idx)
                            position_indices[pos] = interpolated_position
                
                # Process tail tokens (extrapolation beyond last mod1 anchor)
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

def find_wav_files_embody3d(root: Path) -> list[Path]:
    """
    Find all WAV files in Embody3D dataset structure.
    Matches the traversal pattern: iterate c--* directories and find any subfolders 
    named "audio_separated" containing wav files.
    
    Args:
        root: Dataset root directory
        
    Returns:
        List of Path objects to WAV files
    """
    files: list[Path] = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith("c--"):
            for p in entry.rglob("audio_separated/*.wav"):
                if p.is_file():
                    files.append(p)
    return sorted(files)


def parse_full_transcript(json_file: Path):
    """
    Load and parse the full transcript from JSON for a given file.
    Returns (full_text, word_timestamps) where word_timestamps is a list of (word, start, end).
    Expected file: JSON file with segments and word-level timestamps.
    
    Args:
        json_file: Path to JSON transcript file
        
    Returns:
        Tuple of (full_text, word_timestamps)
    """
    if not json_file.exists():
        logging.warning(f"JSON transcript not found: {json_file}")
        return "", []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract full text if available
        full_text = data.get("full_text", "")
        
        # Extract word timestamps from segments
        word_timestamps = []
        full_words = []
        
        segments = data.get("segments", [])
        for segment in segments:
            words = segment.get("words", [])
            for word_info in words:
                word_text = word_info.get("word", "").strip()
                start_time = word_info.get("start", 0.0)
                end_time = word_info.get("end", 0.0)
                
                # Skip empty words
                if word_text and word_text != "":
                    # Clean the word text
                    cleaned_word = clean_transcript_text(word_text)
                    if cleaned_word:
                        word_timestamps.append((cleaned_word, start_time, end_time))
                        full_words.append(cleaned_word)
        
        # If full_text is not provided, construct from words
        if not full_text and full_words:
            full_text = " ".join(full_words)
        
        return full_text, word_timestamps
        
    except Exception as e:
        logging.error(f"Error reading JSON transcript for {json_file}: {e}")
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

def process_full_video(video_id, speaker_id, transcript_text, audio_dir, upper_dir, lower_dir, hand_dir,
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
    # Audio tokens use {video_id}.npy, motion tokens use {video_id}_amass.npy
    audio_token_file = os.path.join(audio_dir, f"{video_id}.npy")
    upper_token_file = os.path.join(upper_dir, f"{video_id}_amass.npy")
    lower_token_file = os.path.join(lower_dir, f"{video_id}_amass.npy")
    hand_token_file = os.path.join(hand_dir, f"{video_id}_amass.npy")
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
            "segment_id": f"{video_id}--SPEAKER_{speaker_id}_full",
            "video_id": f"{video_id}--SPEAKER_{speaker_id}",
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
        if len(chunk_upper) < upper_group_size:
            chunk_upper = chunk_upper + [0] * (upper_group_size - len(chunk_upper))

        l_start = i * lower_group_size
        l_end = min((i + 1) * lower_group_size, len(lower_tokens))
        chunk_lower = lower_tokens[l_start:l_end] if l_start < len(lower_tokens) else []
        if len(chunk_lower) < lower_group_size:
            chunk_lower = chunk_lower + [0] * (lower_group_size - len(chunk_lower))

        h_start = i * hand_group_size
        h_end = min((i + 1) * hand_group_size, len(hand_tokens))
        chunk_hand = hand_tokens[h_start:h_end] if h_start < len(hand_tokens) else []
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
        })
    return groups

def process_chunk_to_record(chunk, conv_id, tokenized_records):
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
    position_encoding_indices = calculate_position_encoding_indices(modality_masks)
    
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
            upper_tokens = asst_turn.get("upper_tokens", [])
            lower_tokens = asst_turn.get("lower_tokens", [])
            hand_tokens = asst_turn.get("hand_tokens", [])
            
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
                motion_labels.extend(begin_motion_tokens)  # Begin token is supervised
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
                    motion_labels.extend(upper_token_ids)  # Upper tokens are supervised
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
                    motion_labels.extend(lower_token_ids)  # Lower tokens are supervised
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
                    motion_labels.extend(hand_token_ids)  # Hand tokens are supervised
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
                    process_chunk_to_record(tokenized_record, conv_id, tokenized_records)
                
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
                    process_chunk_to_record(tokenized_record, conv_id, tokenized_records)
                    
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
                    process_chunk_to_record(tokenized_record, conv_id, tokenized_records)
                    
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
        "audio_tokens_per_chunk": audio_segment_size,
        "upper_tokens_per_chunk": upper_segment_size,
        "lower_tokens_per_chunk": lower_segment_size,
        "hand_tokens_per_chunk": hand_segment_size,
        "format_version": "3.0",
        "format_type": "assistant_only_mot_with_position_encoding_body_only",
        "text_format": "text+audio+interleaved_motion_tokens (upper,lower,hand in 1:1:1 alternating pattern) with motion supervision and precomputed position encoding indices",
        "position_encoding": "precomputed_indices_based_on_modality_fps",
        "modality_fps": {"1": 12.5, "2": 18.75},
        "supervision": "upper, lower, hand body tokens (body only)",
        "tokenized": True,
        "max_seq_length": max_seq_length,
        "assistant_prefix": assistant_prefix.strip(),
        "modality_masks": "masks_0 for text, masks_1 for audio, masks_2 for motion (unified upper, lower, hand)",
        "modality_supervision": "upper_lower_hand_body_only",
        "source": "EMBODY3D_A_only_body_only"
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
    """Main function to preprocess Embody3D dataset (A-only version) with MOT support."""
    parser = argparse.ArgumentParser(description="Preprocess Embody3D dataset (A-only) with MOT support")

    # Required arguments
    parser.add_argument("--data_root", type=str, default="/path/to/embody_3d/subset",
                       help="Path to Embody3D dataset root (e.g., /path/to/embody_3d/datasets/aiagent)")
    parser.add_argument("--output_path", type=str, 
                       help="Where to save processed dataset", default="/path/to/embody_3d_subset")
    parser.add_argument("--model_name", type=str, default="THUDM/glm-4-voice-9b", 
                       help="Tokenizer model name")
    
    # Token directory names (relative to the same parent directory as audio_separated)
    parser.add_argument("--tokens_dirname", type=str, default="tokens", 
                       help="Name of tokens directory (relative to parent of audio_separated)")
    parser.add_argument("--audio_subdir", type=str, default="audio", 
                       help="Subdirectory for audio tokens (relative to tokens_dirname)")
    parser.add_argument("--upper_subdir", type=str, default="upper", 
                       help="Subdirectory for upper body tokens (relative to tokens_dirname)")
    parser.add_argument("--lower_subdir", type=str, default="lower", 
                       help="Subdirectory for lower body tokens (relative to tokens_dirname)")
    parser.add_argument("--hand_subdir", type=str, default="hand", 
                       help="Subdirectory for hand tokens (relative to tokens_dirname)")
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
    parser.add_argument("--split", type=str, choices=["train", "test", "val"], default="train", 
                       help="Dataset split to process (train:test = 9:1 ratio). Files are sorted and split deterministically.")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting preprocessing of Embody3D dataset (A-only body only version)")
    logging.info(f"Args: {args}")

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Get list of WAV files to process (Embody3D layout: c--*/**/audio_separated/*.wav)
    data_root_path = Path(args.data_root)
    wav_files = find_wav_files_embody3d(data_root_path)
    
    if not wav_files:
        logging.error("No WAV files found in dataset structure")
        return
    
    # Sort files for consistent splitting
    wav_files = sorted(wav_files)
    
    # Split into train:test with 9:1 ratio
    total_files = len(wav_files)
    train_size = int(total_files * 0.9)
    test_size = total_files - train_size
    
    train_files = wav_files[:train_size]
    test_files = wav_files[train_size:]
    
    # Filter files based on split argument
    if args.split == "train":
        wav_files = train_files
        logging.info(f"Found {total_files} total WAV files, using {len(wav_files)} for train split (9:1 ratio)")
    elif args.split == "test":
        wav_files = test_files
        logging.info(f"Found {total_files} total WAV files, using {len(wav_files)} for test split (9:1 ratio)")
    else:
        # For "val" or other splits, use test files (or you can add a separate val split)
        wav_files = test_files
        logging.info(f"Found {total_files} total WAV files, using {len(wav_files)} for {args.split} split")
    
    if not wav_files:
        logging.error(f"No WAV files found for {args.split} split")
        return
    
    # Process each WAV file
    all_turns = []
    processed_count = 0
    
    for wav_path in tqdm(wav_files, desc="Processing videos"):
        # Extract filename (without extension) to match token files
        video_id = wav_path.stem
        
        # Find corresponding JSON transcript file
        json_file = wav_path.with_suffix(".json")
        
        # Load full transcript text and word timestamps (A-only)
        transcript_text, word_ts = parse_full_transcript(json_file)
        if not transcript_text:
            logging.warning(f"No transcript found for {video_id}, skipping")
            continue
        
        # Find token files relative to the WAV file's directory structure
        # WAV is at: data_root/c--*/**/audio_separated/{filename}.wav
        # Tokens are at: data_root/c--*/**/tokens/{audio,upper,lower,hand}/{filename}.npy
        # So we need to go up from audio_separated to the parent, then into tokens/
        wav_parent = wav_path.parent  # audio_separated directory
        wav_grandparent = wav_parent.parent  # parent of audio_separated
        tokens_base_dir = wav_grandparent / args.tokens_dirname
        audio_token_file = tokens_base_dir / args.audio_subdir / f"{video_id}.npy"
        upper_token_file = tokens_base_dir / args.upper_subdir / f"{video_id}_amass.npy"
        lower_token_file = tokens_base_dir / args.lower_subdir / f"{video_id}_amass.npy"
        hand_token_file = tokens_base_dir / args.hand_subdir / f"{video_id}_amass.npy"

        speaker_id = wav_grandparent.stem
        # Check if token files exist
        missing_files = [f for f in [audio_token_file, upper_token_file, lower_token_file, hand_token_file] if not f.exists()]
        if missing_files:
            logging.warning(f"Skipping {video_id}, missing token files:")
            for f in missing_files:
                logging.warning(f"  - {f}")
            continue
        
        # Convert Path objects to strings for process_full_video
        audio_dir = str(audio_token_file.parent)
        upper_dir = str(upper_token_file.parent)
        lower_dir = str(lower_token_file.parent)
        hand_dir = str(hand_token_file.parent)
        
        # Process full video into single A-only segment
        processed_segments = process_full_video(
            video_id, speaker_id, transcript_text, audio_dir, upper_dir, lower_dir, hand_dir, 
            args.audio_fps, args.upper_fps, args.lower_fps, args.hand_fps
        )
        
        # Attach word timestamps into the only segment if present
        if processed_segments and word_ts:
            processed_segments[0]["word_timestamps"] = word_ts
        
        if not processed_segments:
            logging.warning(f"No valid segments processed for {video_id}")
            continue
        
        # Interleave into groups (text + 26 audio + 1 begin_of_motion + 39 interleaved motion tokens per group)
        for seg in processed_segments:
            groups = interleave_groups_from_full_segment(
                seg, audio_group_size=26, upper_group_size=13, lower_group_size=13, 
                hand_group_size=13, audio_fps=args.audio_fps
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
                    "speaker_type": "assistant",
                })
        processed_count += 1
        logging.info(f"Processed 1 full transcript into {len(groups)} groups for {video_id}_{speaker_id}")

        if args.debug and processed_count >= 1:
            logging.info(f"Debug mode enabled, processing only 1 video")
            break
        
        if processed_count >= args.limit_videos:
            logging.info(f"Limiting to {processed_count} videos")
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
            upper_segment_size=13, ### 0.5 x 26 = 13, Body only version
            lower_segment_size=13, ### 0.5 x 26 = 13, Body only version
            hand_segment_size=13, ### 0.5 x 26 = 13, Body only version
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