#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess YouTube_Talking conversation data by extracting only speaking segments (A parts)
based on TalkNet output, organizing audio and motion tokens into fixed-size chunks,
and saving in HuggingFace datasets format with unified motion modality and V3 features.
Body only version without face tokens.

V3 Features:
- Unified motion modality: All motion tokens (upper, lower, hand) consolidated into modality 2
- Interleaved motion tokens: 1:1:1 alternating pattern (1 upper + 1 lower + 1 hand)
- Motion FPS: 18.75 (6.25*3 for upper/lower/hand)
- Token breakdown: 1 begin_of_motion + 39 motion tokens (13 upper + 13 lower + 13 hand)
- Supervision: Upper and hand tokens supervised if available, lower tokens not supervised (labels=-100 for lower, labels=token_ids for upper/hand if available, attention_mask=1 for text/audio/upper/lower/hand tokens)
- Position encoding: Precomputed indices based on modality FPS

This script processes YouTube_Talking dataset using pre-processed speaking track transcripts 
to extract only segments where the person is actively speaking, with multimodal token support.

Motion Token Supervision:
    - Upper tokens: Supervised (if available)  
    - Lower tokens: NOT supervised (labels=-100)
    - Hand tokens: Supervised (if available, labels=token_ids)
    
Sequence Filtering:
    - Requires speaking track, audio tokens, and at least upper tokens
    - Sequences without upper tokens are discarded

Split Support:
    - Train: Merges train_processed.txt and train_unprocessed.txt
    - Test/Val: Uses single split files (test.txt, val.txt)

Speaking Track Processing:
    The script reads from speaking track transcript files that contain:
    - Segment timestamps (start and end times)
    - Transcript text for each segment
    - Word-level timestamps
    
    These files are generated from TalkNet speaker detection and have already:
    - Separated speaking (A) from non-speaking (Q) segments
    - Provided accurate word-level timestamps
    - Aligned audio with transcript

Usage:
    python preprocess_youtube_dataset_w_transcript_tokenized_varying_mot_A_only_encode_position_body_only_v3.py \
        --data_root /path/to/YouTube_Talking \
        --output_path ./processed_youtube_train \
        --split train
        
    python preprocess_youtube_dataset_w_transcript_tokenized_varying_mot_A_only_encode_position_body_only_v3.py \
        --data_root /path/to/YouTube_Talking \
        --output_path ./processed_youtube_test \
        --split test
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
from typing import Optional, Tuple, Union, List, Dict, Any

# Global precomputed token ID sets for efficient attention mask generation
UPPER_TOKEN_IDS = None
LOWER_TOKEN_IDS = None
HAND_TOKEN_IDS = None

def initialize_token_sets(tokenizer):
    """
    Initialize global token ID sets once at the beginning of processing.
    This avoids recomputing token IDs for every chunk.
    
    Args:
        tokenizer: Tokenizer instance to convert tokens to IDs
    """
    global UPPER_TOKEN_IDS, LOWER_TOKEN_IDS, HAND_TOKEN_IDS
    
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
    
    logging.info(f"Initialized token sets: {len(UPPER_TOKEN_IDS)} upper, {len(LOWER_TOKEN_IDS)} lower, {len(HAND_TOKEN_IDS)} hand tokens")




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
        
        # Process modalities 2 and collect all tokens with timestamps
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
                start_offset = -0.5  # Fixed offset for unified motion modality
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
        modality: Type of token ("audio", "upper", "lower", or "hand")
        
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

def load_speaking_segments(speaking_segments_dir, video_id):
    """
    Load speaking segments from pre-processed JSON files.
    
    Args:
        speaking_segments_dir: Directory containing speaking segment JSON files
        video_id: Video ID to load segments for
        
    Returns:
        List of (start_time, end_time) tuples for speaking segments
    """
    # Load from speaking segments JSON file
    segments_file = os.path.join(speaking_segments_dir, f"{video_id}_speaking_segments.json")
    
    if not os.path.exists(segments_file):
        logging.warning(f"Speaking segments file not found: {segments_file}")
        return []
    
    try:
        with open(segments_file, 'r') as f:
            data = json.load(f)
        
        speaking_segments = [(seg[0], seg[1]) for seg in data.get("speaking_segments", [])]
        logging.info(f"Loaded {len(speaking_segments)} speaking segments for {video_id}")
        
        # Log additional metadata if available
        if "duration" in data:
            logging.info(f"Video duration: {data['duration']:.2f} seconds")
        
        return speaking_segments
        
    except Exception as e:
        logging.error(f"Error loading speaking segments for {video_id}: {e}")
        return []

def load_speaking_track_segments(transcript_dir, video_id):
    """
    Load all segments from speaking track transcript file.
    
    Args:
        transcript_dir: Directory containing transcript files
        video_id: Video ID
        
    Returns:
        List of segments with text and timestamps
    """
    speaking_track_path = os.path.join(transcript_dir, video_id, f"{video_id}_speaking_track.txt")
    
    if not os.path.exists(speaking_track_path):
        logging.warning(f"No speaking track transcript found for {video_id}")
        return []
    
    segments = []
    
    try:
        with open(speaking_track_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse segments with word-level timestamps
        in_segments_section = False
        current_segment = None
        
        for line in content.split('\n'):
            # Start of segments section
            if line.startswith("Segments with word-level timestamps:"):
                in_segments_section = True
                continue
            
            if not in_segments_section:
                continue
            
            # New segment
            if line.startswith("Segment ") and ":" in line:
                # Save previous segment if exists
                if current_segment and current_segment.get('words'):
                    segments.append(current_segment)
                
                current_segment = {
                    'text': '',
                    'words': [],
                    'start_time': None,
                    'end_time': None
                }
                continue
            
            # Timestamp line
            if current_segment is not None and line.startswith("Timestamp:"):
                match = re.match(r'Timestamp:\s*([\d.]+)s\s*-\s*([\d.]+)s', line)
                if match:
                    current_segment['start_time'] = float(match.group(1))
                    current_segment['end_time'] = float(match.group(2))
                continue
            
            # Text line
            if current_segment is not None and line.startswith("Text:"):
                text = line[5:].strip()
                # Remove "A: " prefix if present
                if text.startswith("A: "):
                    text = text[3:]
                current_segment['text'] = text
                continue
            
            # Words section
            if current_segment is not None and line.startswith("Words:"):
                continue
            
            # Word with timestamp
            if current_segment is not None and ':' in line and 's' in line and '-' in line:
                # Parse word timestamp line
                match = re.match(r'^\s*(.+?):\s*([\d.]+)s\s*-\s*([\d.]+)s', line)
                if match:
                    word = match.group(1).strip()
                    start = float(match.group(2))
                    end = float(match.group(3))
                    current_segment['words'].append({
                        'word': word,
                        'start': start,
                        'end': end
                    })
        
        # Don't forget the last segment
        if current_segment and current_segment.get('words'):
            segments.append(current_segment)
    
    except Exception as e:
        logging.error(f"Error loading speaking track transcript for {video_id}: {e}")
        return []
    
    return segments



def merge_close_segments(segments, gap_threshold=2.0):
    """
    Merge segments that have a time gap less than the threshold.
    
    Args:
        segments: List of segments with text and timestamps
        gap_threshold: Maximum gap in seconds between segments to merge (default: 2.0)
        
    Returns:
        List of merged segments
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x['start_time'])
    
    merged_segments = []
    current_merged = None
    
    for segment in sorted_segments:
        if current_merged is None:
            # First segment
            current_merged = segment.copy()
        else:
            # Check gap between current merged segment and this segment
            gap = segment['start_time'] - current_merged['end_time']
            
            if gap < gap_threshold:
                # Merge segments
                current_merged['end_time'] = segment['end_time']
                current_merged['text'] += ' ' + segment['text']
                current_merged['words'].extend(segment['words'])
            else:
                # Gap is too large, save current merged segment and start new one
                merged_segments.append(current_merged)
                current_merged = segment.copy()
    
    # Don't forget the last merged segment
    if current_merged:
        merged_segments.append(current_merged)
    
    return merged_segments




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

def process_transcript_segments(video_id, transcript_segments, audio_dir, 
                              upper_dir, lower_dir, hand_dir,
                              audio_fps=12.5, upper_fps=6.25, lower_fps=6.25, hand_fps=6.25):
    """
    Process transcript segments into format suitable for dataset creation.
    
    Args:
        video_id: Video ID
        transcript_segments: List of segments from speaking track transcript
        audio_dir: Directory containing audio tokens
        upper_dir: Directory containing upper body tokens
        lower_dir: Directory containing lower body tokens
        hand_dir: Directory containing hand tokens
        audio_fps: Audio token frame rate
        upper_fps: Upper body token frame rate
        lower_fps: Lower body token frame rate
        hand_fps: Hand token frame rate
        
    Returns:
        List of processed segments with text, audio, upper, lower, and hand tokens
    """
    processed_segments = []
    
    # Load audio and motion tokens
    audio_token_file = os.path.join(audio_dir, f"{video_id}.npy")
    upper_token_file = os.path.join(upper_dir, f"{video_id}.npy")
    lower_token_file = os.path.join(lower_dir, f"{video_id}.npy")
    hand_token_file = os.path.join(hand_dir, f"{video_id}.npy")
    
    if not os.path.exists(audio_token_file):
        logging.warning(f"No audio tokens found for {video_id}, skipping")
        return []
    
    # Check for upper tokens (required)
    has_upper = os.path.exists(upper_token_file)
    if not has_upper:
        logging.warning(f"No upper tokens found for {video_id}, skipping")
        return []
    
    # Load audio tokens (always required)
    try:
        audio_data = np.load(audio_token_file, allow_pickle=True)
        audio_tokens_ts = [(audio_data[i], i / audio_fps) for i in range(len(audio_data))]
        logging.info(f"Loaded {len(audio_data)} audio tokens for {video_id}, duration: {len(audio_data) / audio_fps:.2f}s")
    except Exception as e:
        logging.error(f"Error loading audio tokens for {video_id}: {e}")
        return []
    
    # Load motion tokens (upper, lower, hand)
    upper_tokens_ts = []
    lower_tokens_ts = []
    hand_tokens_ts = []
    
    # Load upper tokens if available
    if has_upper:
        try:
            upper_data = np.load(upper_token_file, allow_pickle=True)
            if upper_data.ndim > 1:
                upper_data = upper_data[0]  # Take first element if nested
            upper_tokens_ts = [(upper_data[i], i / upper_fps) for i in range(len(upper_data))]
            logging.info(f"Loaded {len(upper_data)} upper tokens for {video_id}, duration: {len(upper_data) / upper_fps:.2f}s")
        except Exception as e:
            logging.error(f"Error loading upper tokens for {video_id}: {e}")
            upper_tokens_ts = []
    
    # Generate dummy lower tokens (always 0, not supervised)
    audio_duration = len(audio_data) / audio_fps
    lower_token_count = int(audio_duration * lower_fps)
    lower_tokens_ts = [(0, i / lower_fps) for i in range(lower_token_count)]
    logging.info(f"Generated {lower_token_count} zero lower tokens for {video_id} (not supervised)")
    
    # Load hand tokens if available
    if os.path.exists(hand_token_file):
        try:
            hand_data = np.load(hand_token_file, allow_pickle=True)
            if hand_data.ndim > 1:
                hand_data = hand_data[0]  # Take first element if nested
            hand_tokens_ts = [(hand_data[i], i / hand_fps) for i in range(len(hand_data))]
            logging.info(f"Loaded {len(hand_data)} hand tokens for {video_id}, duration: {len(hand_data) / hand_fps:.2f}s")
        except Exception as e:
            logging.error(f"Error loading hand tokens for {video_id}: {e}")
            # Fallback to zero hand tokens
            audio_duration = len(audio_data) / audio_fps
            hand_token_count = int(audio_duration * hand_fps)
            hand_tokens_ts = [(0, i / hand_fps) for i in range(hand_token_count)]
            logging.info(f"Generated {hand_token_count} zero hand tokens for {video_id} (fallback)")
    else:
        # Create dummy hand tokens based on audio duration
        audio_duration = len(audio_data) / audio_fps
        hand_token_count = int(audio_duration * hand_fps)
        hand_tokens_ts = [(0, i / hand_fps) for i in range(hand_token_count)]
        logging.info(f"Generated {hand_token_count} zero hand tokens for {video_id} (no file)")
    
    # Process each transcript segment
    for seg_idx, segment in enumerate(transcript_segments):
        start_time = segment['start_time']
        end_time = segment['end_time']
        text = segment['text']
        word_data = segment['words']
        
        # Get tokens for this time range
        segment_audio = get_tokens_for_timerange(audio_tokens_ts, start_time, end_time)
        segment_upper = get_tokens_for_timerange(upper_tokens_ts, start_time, end_time)
        segment_lower = get_tokens_for_timerange(lower_tokens_ts, start_time, end_time)
        segment_hand = get_tokens_for_timerange(hand_tokens_ts, start_time, end_time)
        
        # Convert word data to word timestamps format
        word_timestamps = [(w['word'], w['start'], w['end']) for w in word_data]
        
        # Only process segments with content and upper tokens
        if segment_audio and text and segment_upper:
            processed_segments.append({
                "segment_id": f"{video_id}_seg_{seg_idx}",
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "audio_tokens": [t for t, _ in segment_audio],
                "upper_tokens": [t for t, _ in segment_upper],
                "lower_tokens": [t for t, _ in segment_lower],
                "hand_tokens": [t for t, _ in segment_hand],
                "word_timestamps": word_timestamps,
                # Add file existence flags for supervision logic
                "has_upper_file": has_upper,
                "has_lower_file": os.path.exists(lower_token_file),
                "has_hand_file": os.path.exists(hand_token_file)
            })
            
            # Log segment info for debugging
            logging.debug(f"Segment {seg_idx}: {start_time:.2f}s-{end_time:.2f}s, "
                         f"audio tokens: {len(segment_audio)}, "
                         f"upper tokens: {len(segment_upper)}, lower tokens: {len(segment_lower)}, "
                         f"hand tokens: {len(segment_hand)}, words: {len(word_data)}")
    
    return processed_segments

def split_into_chunks(processed_segments, audio_chunk_size=26, 
                     upper_chunk_size=13, lower_chunk_size=13, hand_chunk_size=13):
    """
    Split processed segments into fixed-size chunks with all motion modalities.
    
    Args:
        processed_segments: List of processed segments with text and tokens
        audio_chunk_size: Number of audio tokens per chunk
        upper_chunk_size: Number of upper tokens per chunk
        lower_chunk_size: Number of lower tokens per chunk
        hand_chunk_size: Number of hand tokens per chunk
        
    Returns:
        List of chunks ready for dataset creation
    """
    all_chunks = []
    
    for segment in processed_segments:
        audio_tokens = segment["audio_tokens"]
        upper_tokens = segment["upper_tokens"]
        lower_tokens = segment["lower_tokens"]
        hand_tokens = segment["hand_tokens"]
        text = segment["text"]
        word_timestamps = segment["word_timestamps"]
        
        # Calculate number of chunks needed
        num_chunks = max(1, (len(audio_tokens) + audio_chunk_size - 1) // audio_chunk_size)
        
        for chunk_idx in range(num_chunks):
            # Extract audio chunk
            audio_start = chunk_idx * audio_chunk_size
            audio_end = min((chunk_idx + 1) * audio_chunk_size, len(audio_tokens))
            chunk_audio = audio_tokens[audio_start:audio_end]
            
            # Pad audio if needed
            if len(chunk_audio) < audio_chunk_size:
                chunk_audio = chunk_audio + [0] * (audio_chunk_size - len(chunk_audio))
            
            # Extract corresponding upper chunk
            upper_start = chunk_idx * upper_chunk_size
            upper_end = min((chunk_idx + 1) * upper_chunk_size, len(upper_tokens))
            chunk_upper = upper_tokens[upper_start:upper_end] if upper_start < len(upper_tokens) else []
            
            # Pad upper if needed
            if len(chunk_upper) < upper_chunk_size:
                chunk_upper = chunk_upper + [0] * (upper_chunk_size - len(chunk_upper))
            
            # Extract corresponding lower chunk
            lower_start = chunk_idx * lower_chunk_size
            lower_end = min((chunk_idx + 1) * lower_chunk_size, len(lower_tokens))
            chunk_lower = lower_tokens[lower_start:lower_end] if lower_start < len(lower_tokens) else []
            
            # Pad lower if needed
            if len(chunk_lower) < lower_chunk_size:
                chunk_lower = chunk_lower + [0] * (lower_chunk_size - len(chunk_lower))
            
            # Extract corresponding hand chunk
            hand_start = chunk_idx * hand_chunk_size
            hand_end = min((chunk_idx + 1) * hand_chunk_size, len(hand_tokens))
            chunk_hand = hand_tokens[hand_start:hand_end] if hand_start < len(hand_tokens) else []
            
            # Pad hand if needed
            if len(chunk_hand) < hand_chunk_size:
                chunk_hand = chunk_hand + [0] * (hand_chunk_size - len(chunk_hand))
            
            # Estimate text for this chunk based on time proportion
            if num_chunks > 1 and word_timestamps:
                # Calculate time range for this chunk
                chunk_start_time = segment["start_time"] + (chunk_idx * audio_chunk_size / 12.5)
                chunk_end_time = segment["start_time"] + ((chunk_idx + 1) * audio_chunk_size / 12.5)
                
                # Extract words for this time range
                chunk_words = []
                for word, word_start, word_end in word_timestamps:
                    if word_end > chunk_start_time and word_start < chunk_end_time:
                        chunk_words.append(word)
                chunk_text = " ".join(chunk_words)
            else:
                # Use full text for single chunk or when no timestamps
                chunk_text = text if chunk_idx == 0 else ""
            
            all_chunks.append({
                "conversation_id": segment["video_id"],
                "turn_id": f"{segment['segment_id']}_chunk_{chunk_idx}",
                "text": chunk_text,
                "audio_tokens": chunk_audio,
                "upper_tokens": chunk_upper,
                "lower_tokens": chunk_lower,
                "hand_tokens": chunk_hand,
                "speaker_type": "assistant",
                # Pass file existence flags for supervision logic
                "has_upper_file": segment.get("has_upper_file", False),
                "has_lower_file": segment.get("has_lower_file", False),
                "has_hand_file": segment.get("has_hand_file", False)
            })
    
    return all_chunks

def process_chunk_to_record(chunk, conv_id, tokenized_records, tokenizer, max_seq_length=2048):
    """
    Process a chunk of turns into a tokenized record with unified motion modality support.
    
    Args:
        chunk: Dictionary containing input_ids, labels, turns, modality_masks_0, modality_masks_1, modality_masks_2
        conv_id: Conversation ID
        tokenized_records: List to append the processed record to
        tokenizer: Tokenizer for decoding tokens
        max_seq_length: Maximum sequence length for truncation
    """
    # Use global precomputed token ID sets for efficient attention mask generation
    # Final check for sequence length - this will be updated dynamically based on max_seq_length
    max_length = max_seq_length
    if len(chunk["input_ids"]) > max_length:
        logging.warning(f"Final check found sequence length {len(chunk['input_ids'])} > {max_length}, truncating")
        chunk["input_ids"] = chunk["input_ids"][:max_length]
        chunk["labels"] = chunk["labels"][:max_length]
        chunk["modality_masks_0"] = chunk["modality_masks_0"][:max_length]
        chunk["modality_masks_1"] = chunk["modality_masks_1"][:max_length]
        chunk["modality_masks_2"] = chunk["modality_masks_2"][:max_length]
    
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
    
    # Create attention mask efficiently using global precomputed token IDs
    attention_mask = [1] * seq_len  # Initialize all tokens to participate in attention
    
    # # Set attention_mask to 0 for lower and hand tokens using global O(1) lookup
    # # In YouTube V3 Body Only: text, audio, upper, lower, and hand tokens participate in attention
    # # All tokens participate in attention in body only version
    # for i, token_id in enumerate(chunk["input_ids"]):
    #     if token_id in LOWER_TOKEN_IDS or token_id in HAND_TOKEN_IDS:
    #         attention_mask[i] = 0
    
    # Calculate position encoding indices for this sequence
    modality_masks = [
        chunk["modality_masks_0"],  # Text tokens (modality 0)
        chunk["modality_masks_1"],  # Audio tokens (modality 1) 
        chunk["modality_masks_2"],  # Unified motion tokens (modality 2)
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
        "modality_masks_2": chunk["modality_masks_2"],  # True for unified motion tokens
        "position_encoding_indices": position_encoding_indices,  # Precomputed position indices
    }
    tokenized_records.append(tokenized_record)

def convert_to_huggingface_dataset(
    output_path,
    interleaved_turns,
    tokenizer_name,
    max_seq_length=2048,
    audio_segment_size=26,
    upper_segment_size=13,
    lower_segment_size=13,
    hand_segment_size=13,
    split="train"
):
    """
    Convert interleaved turns into a HuggingFace dataset with unified motion modality and V3 format.
    Supports upper+lower+hand supervision, body only version.
    
    Args:
        output_path: Where to save the processed dataset
        interleaved_turns: List of processed chunks
        tokenizer_name: Name of the tokenizer to use
        max_seq_length: Maximum sequence length for tokenization
        audio_segment_size: Number of audio tokens per group (default: 26)
        upper_segment_size: Number of upper tokens per group (default: 13)
        lower_segment_size: Number of lower tokens per group (default: 13)
        hand_segment_size: Number of hand tokens per group (default: 13)
        split: Which split this is (train/test/val)
        
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
    
    
    # Add special tokens for all motion modalities
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


    # Initialize global token ID sets once for efficient attention mask generation
    initialize_token_sets(tokenizer)


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
        
        current_input_ids = []
        current_labels = []
        current_turns = []
        current_modality_masks_0 = []  # For text tokens
        current_modality_masks_1 = []  # For audio tokens
        current_modality_masks_2 = []  # For unified motion tokens
        
        for turn_idx, asst_turn in enumerate(asst_turns):
            # Prepare content tokens for this turn
            asst_text = asst_turn.get("text", "")
            text_tokens = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
            text_labels = [-100] * len(text_tokens)  # Text doesn't contribute to loss
            text_modality_0 = [True] * len(text_tokens)   # Text is modality 0
            text_modality_1 = [False] * len(text_tokens)  # Text is not modality 1
            text_modality_2 = [False] * len(text_tokens)  # Text is not modality 2
            
            # Process audio tokens
            audio_tokens = asst_turn.get("audio_tokens", [])[:audio_segment_size]
            audio_token_ids = []
            for token in audio_tokens:
                if isinstance(token, np.ndarray):
                    token_val = token.tolist() if hasattr(token, 'tolist') else token
                else:
                    token_val = token
                token_str = safe_token_to_string(token_val, "audio")
                ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
                audio_token_ids.extend(ids)
            audio_labels = [-100] * len(audio_token_ids)  # Audio doesn't contribute to loss
            audio_modality_0 = [False] * len(audio_token_ids)  # Audio is not modality 0
            audio_modality_1 = [True] * len(audio_token_ids)   # Audio is modality 1
            audio_modality_2 = [False] * len(audio_token_ids)  # Audio is not modality 2
            
            # Process motion tokens with unified begin token and 1:1:1 interleaved pattern
            upper_tokens = asst_turn.get("upper_tokens", [])[:upper_segment_size]
            lower_tokens = asst_turn.get("lower_tokens", [])[:lower_segment_size]
            hand_tokens = asst_turn.get("hand_tokens", [])[:hand_segment_size]
            
            motion_token_ids = []
            motion_labels = []
            motion_modality_0 = []
            motion_modality_1 = []
            motion_modality_2 = []
            
            # Add unified begin_of_motion token if there are any motion tokens
            if upper_tokens or lower_tokens or hand_tokens:
                begin_motion_tokens = tokenizer("<|begin_of_motion|>", add_special_tokens=False)["input_ids"]
                motion_token_ids.extend(begin_motion_tokens)
                motion_labels.extend([-100] * len(begin_motion_tokens))  # Begin token not supervised
                motion_modality_0.extend([False] * len(begin_motion_tokens))
                motion_modality_1.extend([False] * len(begin_motion_tokens))
                motion_modality_2.extend([True] * len(begin_motion_tokens))  # Begin token belongs to motion modality
            
            upper_token_ids = []
            for token in upper_tokens:
                if isinstance(token, np.ndarray):
                    token_val = token.tolist() if hasattr(token, 'tolist') else token
                else:
                    token_val = token
                token_str = safe_token_to_string(token_val, "upper")
                ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
                upper_token_ids.extend(ids)
            
            lower_token_ids = []
            for token in lower_tokens:
                if isinstance(token, np.ndarray):
                    token_val = token.tolist() if hasattr(token, 'tolist') else token
                else:
                    token_val = token
                token_str = safe_token_to_string(token_val, "lower")
                ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
                lower_token_ids.extend(ids)
            
            hand_token_ids = []
            for token in hand_tokens:
                if isinstance(token, np.ndarray):
                    token_val = token.tolist() if hasattr(token, 'tolist') else token
                else:
                    token_val = token
                token_str = safe_token_to_string(token_val, "hand")
                ids = tokenizer(token_str, add_special_tokens=False)["input_ids"]
                hand_token_ids.extend(ids)
            
            # Implement 1:1:1 interleaved pattern (13 rounds of 3 tokens each = 39 motion tokens)
            # Each round: 1 upper + 1 lower + 1 hand
            for round_idx in range(13):
                # Add 1 upper token
                if round_idx < len(upper_token_ids):
                    motion_token_ids.append(upper_token_ids[round_idx])
                    motion_labels.append(upper_token_ids[round_idx])  # Upper tokens supervised
                    motion_modality_0.append(False)
                    motion_modality_1.append(False)
                    motion_modality_2.append(True)
                
                # Add 1 lower token
                if round_idx < len(lower_token_ids):
                    motion_token_ids.append(lower_token_ids[round_idx])
                    motion_labels.append(-100)  # Lower tokens not supervised
                    motion_modality_0.append(False)
                    motion_modality_1.append(False)
                    motion_modality_2.append(True)
                
                # Add 1 hand token
                if round_idx < len(hand_token_ids):
                    motion_token_ids.append(hand_token_ids[round_idx])
                    # Determine supervision based on file existence
                    hand_has_real_data = asst_turn.get("has_hand_file", False)
                    motion_labels.append(hand_token_ids[round_idx] if hand_has_real_data else -100)  # Hand tokens supervised if available
                    motion_modality_0.append(False)
                    motion_modality_1.append(False)
                    motion_modality_2.append(True)
            
            # Compose the full turn
            turn_input_ids = text_tokens + audio_token_ids + motion_token_ids
            turn_labels = text_labels + audio_labels + motion_labels
            turn_modality_0 = text_modality_0 + audio_modality_0 + motion_modality_0
            turn_modality_1 = text_modality_1 + audio_modality_1 + motion_modality_1
            turn_modality_2 = text_modality_2 + audio_modality_2 + motion_modality_2
            
            # If starting a new sequence, add the assistant prefix
            if not current_input_ids:
                current_input_ids.extend(assistant_prefix_tokens)
                current_labels.extend([-100] * len(assistant_prefix_tokens))
                current_modality_masks_0.extend([True] * len(assistant_prefix_tokens))   # Prefix is text
                current_modality_masks_1.extend([False] * len(assistant_prefix_tokens))  # Prefix is not audio
                current_modality_masks_2.extend([False] * len(assistant_prefix_tokens))  # Prefix is not motion
            
            # Check if adding this turn would exceed max_seq_length (including EOS)
            if len(current_input_ids) + len(turn_input_ids) + len(eos_token_ids) > max_seq_length:
                # Finalize current sequence
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
                process_chunk_to_record(tokenized_record, conv_id, tokenized_records, tokenizer, max_seq_length)
                
                # Start a new sequence
                current_input_ids = assistant_prefix_tokens.copy()
                current_labels = [-100] * len(assistant_prefix_tokens)
                current_modality_masks_0 = [True] * len(assistant_prefix_tokens)
                current_modality_masks_1 = [False] * len(assistant_prefix_tokens)
                current_modality_masks_2 = [False] * len(assistant_prefix_tokens)
                current_turns = []
            
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
            current_modality_masks_2.extend([False] * len(eos_token_ids))  # EOS is not motion
            
            tokenized_record = {
                "input_ids": current_input_ids,
                "labels": current_labels,
                "turns": current_turns,
                "modality_masks_0": current_modality_masks_0,
                "modality_masks_1": current_modality_masks_1,
                "modality_masks_2": current_modality_masks_2,
            }
            process_chunk_to_record(tokenized_record, conv_id, tokenized_records, tokenizer, max_seq_length)
    
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
        "text_format": "text+audio+1 begin_of_motion+39 interleaved motion tokens (13 upper + 13 lower + 13 hand in 1:1:1 alternating pattern) with motion supervision and precomputed position encoding indices",
        "position_encoding": "precomputed_indices_based_on_modality_fps",
        "modality_fps": {"1": 12.5, "2": 18.75},
        "supervision": "unified motion tokens (upper and hand supervised if available, lower not supervised, text/audio/upper/lower/hand tokens participate in attention)",
        "tokenized": True,
        "max_seq_length": max_seq_length,
        "assistant_prefix": assistant_prefix.strip(),
        "modality_masks": "masks_0 for text, masks_1 for audio, masks_2 for unified motion",
        "modality_supervision": "upper_lower_and_hand_supervised_body_only",
        "source": "YouTube_Talking_A_only_body_only_v3"
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
    """Main function to preprocess YouTube_Talking dataset (A-only version) with MOT support."""
    parser = argparse.ArgumentParser(description="Preprocess YouTube_Talking dataset (A-only) with MOT support")

    # Required arguments
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to YouTube_Talking dataset root")
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
    parser.add_argument("--hand_dir", type=str, default="TOKENS_AGENT_25/hand_generated", 
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
    parser.add_argument("--limit_videos", type=int, default=None,
                       help="Limit number of videos to process (for debugging)")
    parser.add_argument("--split", type=str, choices=["train", "test", "val"], required=True,
                       help="Which split to process (train, test, or val)")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting preprocessing of YouTube_Talking dataset (A-only body only version)")
    logging.info(f"Args: {args}")

    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)

    # Load split files
    selected_video_ids = set()
    if args.split == "train":
        # Merge train_processed.txt and train_unprocessed.txt for train split
        for split_file in ["train_processed.txt", "train_unprocessed.txt"]:
            split_path = os.path.join(args.data_root, split_file)
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    video_ids_from_file = set(line.strip() for line in f if line.strip())
                    selected_video_ids.update(video_ids_from_file)
                    logging.info(f"Loaded {len(video_ids_from_file)} video IDs from {split_file}")
            else:
                logging.warning(f"Split file not found: {split_path}")
    else:
        # For test and val splits, use single file
        split_file = os.path.join(args.data_root, f"{args.split}.txt")
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                selected_video_ids = set(line.strip() for line in f if line.strip())
            logging.info(f"Loaded {len(selected_video_ids)} video IDs for {args.split} split")
        else:
            logging.error(f"Split file not found: {split_file}")
            return
    
    logging.info(f"Total selected video IDs for {args.split} split: {len(selected_video_ids)}")

    # Get list of videos to process
    transcript_dir = os.path.join(args.data_root, "transcript")
    audio_dir = os.path.join(args.data_root, args.audio_dir)
    upper_dir = os.path.join(args.data_root, args.upper_dir)
    lower_dir = os.path.join(args.data_root, args.lower_dir)
    hand_dir = os.path.join(args.data_root, args.hand_dir)
    
    video_ids = []
    for video_dir in os.listdir(transcript_dir):
        video_path = os.path.join(transcript_dir, video_dir)
        if os.path.isdir(video_path) and video_dir in selected_video_ids:
            # Check if required files exist
            speaking_track_file = os.path.join(video_path, f"{video_dir}_speaking_track.txt")
            audio_file = os.path.join(audio_dir, f"{video_dir}.npy")
            upper_file = os.path.join(upper_dir, f"{video_dir}.npy")
            lower_file = os.path.join(lower_dir, f"{video_dir}.npy")
            hand_file = os.path.join(hand_dir, f"{video_dir}.npy")
            
            # Require speaking track, audio tokens, and upper tokens
            has_upper = os.path.exists(upper_file)
            
            if os.path.exists(speaking_track_file) and os.path.exists(audio_file) and has_upper:
                video_ids.append(video_dir)
            else:
                # Log which files are missing
                missing_files = []
                if not os.path.exists(speaking_track_file):
                    missing_files.append("speaking track")
                if not os.path.exists(audio_file):
                    missing_files.append("audio tokens")
                if not has_upper:
                    missing_files.append("upper tokens")
                logging.debug(f"Skipping {video_dir}, missing: {', '.join(missing_files)}")
    
    logging.info(f"Found {len(video_ids)} videos with required files")
    
    # Limit videos if requested
    if args.limit_videos:
        video_ids = video_ids[:args.limit_videos]
        logging.info(f"Limiting to {len(video_ids)} videos")
    
    # Process each video
    all_chunks = []
    
    for video_id in tqdm(video_ids, desc="Processing videos"):
        # Load transcript segments from speaking track file
        transcript_segments = load_speaking_track_segments(transcript_dir, video_id)
        # Merge segments with gaps less than 2 seconds
        transcript_segments = merge_close_segments(transcript_segments, gap_threshold=2.0)
        print(f"Total segments after merging: {len(transcript_segments)}")

        if not transcript_segments:
            logging.warning(f"No transcript segments found for {video_id}, skipping")
            continue
        
        # Process transcript segments
        processed_segments = process_transcript_segments(
            video_id, transcript_segments, 
            audio_dir, upper_dir, lower_dir, hand_dir,
            args.audio_fps, args.upper_fps, args.lower_fps, args.hand_fps
        )
        
        if not processed_segments:
            logging.warning(f"No valid segments processed for {video_id}")
            continue
        
        # Split into fixed-size chunks
        chunks = split_into_chunks(processed_segments)
        all_chunks.extend(chunks)
        
        logging.info(f"Processed {len(transcript_segments)} transcript segments into {len(chunks)} chunks for {video_id}")
    
    logging.info(f"Total chunks created: {len(all_chunks)}")
    
    # Convert to HuggingFace dataset
    if all_chunks:
        dataset = convert_to_huggingface_dataset(
            output_path=args.output_path,
            interleaved_turns=all_chunks,
            tokenizer_name=args.model_name,
            max_seq_length=args.max_seq_length,
            audio_segment_size=26,
            upper_segment_size=13,
            lower_segment_size=13,
            hand_segment_size=13,
            split=args.split
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