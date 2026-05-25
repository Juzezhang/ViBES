#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess TFHP dataset into sequences with unified motion modality V3 format.
Face-Only V3 version: organizing audio and face motion tokens into fixed-size chunks.

Key features for Face-Only V3 version:
- Only face motion tokens (no upper, lower, hand)
- Unified motion modality: Face tokens mapped to modality 2
- 3 modalities: text(0), audio(1), motion(2) instead of 6
- Motion FPS: 25.0 (face only)
- Group size: text + 25 audio + 1 begin_of_motion + 50 face motion tokens
- Supervision: Face tokens supervised as unified motion modality

Sequence Processing Logic:
- Merge all files from each session into combined tokens
- Group tokens into fixed-size chunks (25 audio + 50 face)
- All chunks from same session share conversation_id = speaker_session
- Multiple sessions may merge into sequences up to max_seq_length
- sequence_name format: speaker_session_seq{id}

Supports train/test/val splits based on split files in the dataset root.

Usage:
    python preprocess_hf_tfhp_face_style.py \
        --data_root /path/to/TFHP \
        --output_path ./processed_tfhp_train_face_only_v3 \
        --split train
    
    python preprocess_hf_tfhp_face_style.py \
        --data_root /path/to/TFHP \
        --output_path ./processed_tfhp_test_face_only_v3 \
        --split test
"""

import os
import json
import numpy as np
import re
import logging
from tqdm import tqdm
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
import torch
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
    Face-Only V3: Computes interpolated position indices for 3 modalities (text, audio, motion).
    
    This function extracts the position mapping logic from the rotary embedding computation
    and returns the interpolated position indices that can be stored in the dataset.
    
    Args:
        modality_masks: List of modality masks [mask_0, mask_1, mask_2]
                       where each mask is a list of booleans indicating token presence
                       mask_0: text, mask_1: audio, mask_2: motion (face only, unified modality)
        modality_fps: Dictionary mapping modality index to fps value
                     Default: {1: 12.5, 2: 25.0} (Face-only V3: audio=12.5, motion=25.0)
    
    Returns:
        List of float values representing the interpolated position index for each token.
        These indices can be used later to compute the actual rotary embeddings efficiently.
    """
    if modality_fps is None:
        modality_fps = {1: 12.5, 2: 25.0}  # Face-only: motion FPS = 25.0 (face only)

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

    # Handle style_control face tokens that appear before the first audio cycle
    if 2 in modality_positions and len(mod1_groups) > 0:
        first_audio_pos = mod1_groups[0][0].item()
        pre_audio_face = [p for p in modality_positions[2].tolist() if p < first_audio_pos]

        if pre_audio_face:
            # Find the primary rope index immediately before and after this block
            before_idx = None
            after_idx = None
            for p in sorted(pos_to_rope_idx.keys()):
                if p < pre_audio_face[0]:
                    before_idx = pos_to_rope_idx[p]
                if p > pre_audio_face[-1] and after_idx is None:
                    after_idx = pos_to_rope_idx[p]

            if before_idx is not None and after_idx is not None:
                n = len(pre_audio_face)
                for i, pos in enumerate(pre_audio_face):
                    alpha = (i + 1) / (n + 1)
                    position_indices[pos] = before_idx + alpha * (after_idx - before_idx)

    for group_idx, mod1_group in enumerate(mod1_groups):
        # Get timing information for this mod1 cycle
        first_mod1_pos = mod1_group[0].item()
        first_mod1_idx = pos_to_rope_idx[first_mod1_pos]
        
        # Calculate time duration for this mod1 cycle
        base_fps = modality_fps.get(1, 12.5)
        cycle_duration = len(mod1_group) / base_fps  # Duration in seconds
        
        # Create a global token list for this cycle with timestamps
        all_cycle_tokens = []
        
        # Process modality 2 (face) and collect all tokens with timestamps
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
                # Face only version: only modality 2 (face) has special timing
                if modality_idx == 2:
                    start_offset = -0.5  # Face tokens start before audio cycle
                else:
                    start_offset = 0.0  # Other modalities (shouldn't happen in face only)
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



def safe_token_to_string(val, prefix, max_val=1e10):
    """Convert token values to special token strings with safety handling."""
    try:
        safe_val = max(0, min(int(val), max_val))
        return f"<|{prefix}_{safe_val}|>"
    except Exception:
        return f"<|{prefix}_0|>"

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
    
    # Remove common non-dialogue content markers (like timestamps)
    text = re.sub(r'\[\d+:\d+\]', '', text)
    
    # Replace common non-standard punctuation
    text = text.replace('..', '…').replace('...', '…')
    
    return text

def merge_session_fragments(transcript_dir, audio_dir, face_dir):
    """
    Merge multiple segment files from a session into combined sequences.
    
    Args:
        transcript_dir: Directory containing transcript .txt files
        audio_dir: Directory containing audio token .npy files  
        face_dir: Directory containing face token .npy files
        
    Returns:
        Tuple of (texts, audios, faces, word_timestamps) for all segments
    """
    txt_files = sorted([f for f in os.listdir(transcript_dir) if f.endswith('.txt')])
    merged_text = []
    merged_audio = []
    merged_face = []
    merged_word_timestamps = []
    for txt_file in txt_files:
        idx = txt_file.replace('.txt', '')
        txt_path = os.path.join(transcript_dir, txt_file)
        audio_path = os.path.join(audio_dir, f"{idx}.npy")
        face_path = os.path.join(face_dir, f"{idx}.npy")
        if not (os.path.exists(audio_path) and os.path.exists(face_path)):
            continue
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        segment_start = 0.0
        segment_end = 0.0
        word_timestamps = []
        for i, line in enumerate(lines):
            if line.strip().startswith("Timestamp:"):
                ts = line.strip().replace("Timestamp:", "").replace("s", "").split("-")
                if len(ts) == 2:
                    try:
                        segment_start = float(ts[0])
                        segment_end = float(ts[1])
                    except Exception:
                        segment_start = 0.0
                        segment_end = 0.0
            if line.strip().startswith("Text:"):
                text = clean_transcript_text(line.strip().replace("Text:", ""))
                merged_text.append(text)
            if line.strip().startswith("Words:"):
                j = i + 1
                while j < len(lines) and lines[j].strip() and ':' in lines[j]:
                    word_line = lines[j].strip()
                    try:
                        word, times = word_line.split(':')
                        word = word.strip()
                        start, end = times.strip().split('-')
                        start = float(start.replace('s', '').strip())
                        end = float(end.replace('s', '').strip())
                        word_timestamps.append((word, start, end))
                    except Exception:
                        pass
                    j += 1
        merged_word_timestamps.append(word_timestamps)
        audio_tokens = np.load(audio_path, allow_pickle=True)
        face_tokens = np.load(face_path, allow_pickle=True)
        face_tokens = face_tokens.flatten()  # Ensure 1D
        merged_face.append(face_tokens.tolist())
        merged_audio.append(audio_tokens.tolist())
    return merged_text, merged_audio, merged_face, merged_word_timestamps

def interleave_split_sequence(texts, audios, faces, word_timestamps):
    """
    Concatenate all audio and face tokens from all segments, then group by fixed size.
    Face-Only V3: Groups tokens into fixed-size chunks following the format.
    
    Grouping logic:
    - Concatenate all audio tokens and face tokens from segments
    - Group by fixed sizes: 25 audio tokens + 50 face tokens per turn (2s @ 12.5/25fps)
    - Extract text corresponding to each group based on timestamps

    Args:
        texts: List of text segments
        audios: List of audio token segments
        faces: List of face token segments
        word_timestamps: List of word-level timestamps for each segment

    Returns:
        List of interleaved groups with fixed sizes:
        - text: Extracted based on timestamp overlap
        - audio_tokens: Exactly 25 tokens per group (padded if needed)
        - face_tokens: Exactly 50 tokens per group (padded if needed)
    """
    AUDIO_GROUP_SIZE = 25
    FACE_GROUP_SIZE = 50

    results = []

    # 1. Concatenate all audio and face tokens
    all_audio_tokens = []
    all_face_tokens = []
    segment_boundaries = []  # Record the boundary positions of each segment

    accumulated_audio_len = 0
    for seg_idx in range(len(audios)):
        audio = audios[seg_idx]
        face = faces[seg_idx]

        # Record segment boundary
        segment_boundaries.append({
            'start_audio_idx': accumulated_audio_len,
            'end_audio_idx': accumulated_audio_len + len(audio),
            'segment_idx': seg_idx
        })

        # Concatenate tokens
        all_audio_tokens.extend(audio)
        all_face_tokens.extend(face)
        accumulated_audio_len += len(audio)

    # 2. Group by fixed size
    audio_groups = []
    face_groups = []
    total_groups = len(all_audio_tokens) // AUDIO_GROUP_SIZE

    # Process complete groups
    for i in range(total_groups):
        audio_start = i * AUDIO_GROUP_SIZE
        audio_end = (i + 1) * AUDIO_GROUP_SIZE

        # Ensure not to exceed the range
        if audio_end <= len(all_audio_tokens):
            audio_groups.append(all_audio_tokens[audio_start:audio_end])

        # Process face tokens (2:1 ratio with audio due to 25fps vs 12.5fps)
        face_start = i * FACE_GROUP_SIZE
        face_end = (i + 1) * FACE_GROUP_SIZE

        # Ensure not to exceed the range
        if face_end <= len(all_face_tokens):
            face_groups.append(all_face_tokens[face_start:face_end])
        else:
            # If the remaining face tokens are less than FACE_GROUP_SIZE, pad with zeros
            face_chunk = all_face_tokens[face_start:] if face_start < len(all_face_tokens) else []
            padded_face = face_chunk + [0] * (FACE_GROUP_SIZE - len(face_chunk))
            face_groups.append(padded_face)

    # Handle the last group if audio tokens are not a multiple of AUDIO_GROUP_SIZE
    if len(all_audio_tokens) % AUDIO_GROUP_SIZE > 0:
        last_start = total_groups * AUDIO_GROUP_SIZE
        last_audio = all_audio_tokens[last_start:]
        padded_audio = last_audio + [0] * (AUDIO_GROUP_SIZE - len(last_audio))
        audio_groups.append(padded_audio)

        # Handle the corresponding face tokens
        last_face_start = total_groups * FACE_GROUP_SIZE
        last_face = all_face_tokens[last_face_start:] if last_face_start < len(all_face_tokens) else []
        padded_face = last_face + [0] * (FACE_GROUP_SIZE - len(last_face))
        face_groups.append(padded_face)

    # 3. For each group, find the corresponding text
    for i, (audio_group, face_group) in enumerate(zip(audio_groups, face_groups)):
        audio_start_idx = i * AUDIO_GROUP_SIZE
        audio_end_idx = audio_start_idx + AUDIO_GROUP_SIZE
        
        # Calculate the actual time range for the group
        group_start_time = audio_start_idx / 12.5
        group_end_time = audio_end_idx / 12.5
        
        # Find which segment this group belongs to
        segment_texts = []
        segment_word_timestamps = []
        
        for boundary in segment_boundaries:
            seg_start = boundary['start_audio_idx']
            seg_end = boundary['end_audio_idx']
            seg_idx = boundary['segment_idx']
            
            # Check if this group overlaps with the current segment
            if (audio_start_idx < seg_end and audio_end_idx > seg_start):
                # Calculate the start and end time relative to the segment
                relative_start = max(0, audio_start_idx - seg_start) / 12.5
                relative_end = min(seg_end - seg_start, audio_end_idx - seg_start) / 12.5
                
                # Find words in this time range
                segment_words = []
                segment_timestamps = []
                for word, start, end in word_timestamps[seg_idx]:
                    # Check if the word is in the time range
                    if end > relative_start and start < relative_end:
                        segment_words.append(word)
                        segment_timestamps.append((word, start, end))
                
                if segment_words:
                    segment_texts.append(' '.join(segment_words))
                    segment_word_timestamps.extend(segment_timestamps)
        
        # Merge all found texts
        group_text = ' '.join(segment_texts)
        
        # Create result object (face only version)
        results.append({
            "text": group_text,
            "audio_tokens": audio_group,
            "face_tokens": face_group,
            "word_timestamps": segment_word_timestamps
        })
    
    return results

def process_chunk_to_record(chunk, conv_id, tokenized_records, max_seq_length=2048, style_pool=None):
    """
    Process a chunk of turns into a tokenized record with MOT support and position encoding indices.
    Face-Only V3: Computes position indices for 3 modalities (text, audio, motion).
    
    Args:
        chunk: Dictionary containing:
            - input_ids: Tokenized sequence IDs
            - labels: Supervision labels (-100 for unsupervised tokens)
            - turns: List of turn metadata
            - modality_masks_0/1/2: Boolean masks for text/audio/motion tokens
        conv_id: Conversation ID (format: speaker_session)
        tokenized_records: List to append the processed record to
        max_seq_length: Maximum sequence length for truncation
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
        chunk["modality_masks_2"],  # Motion tokens (modality 2) - unified face only
    ]
    
    # Calculate position encoding indices using the same logic as the model
    # This MUST succeed during preprocessing
    position_encoding_indices = calculate_position_encoding_indices(modality_masks)
    
    # Validate position encoding indices
    assert len(position_encoding_indices) == seq_len, f"Position indices length {len(position_encoding_indices)} != sequence length {seq_len}"
    
    # Log statistics for validation
    pos_min, pos_max = min(position_encoding_indices), max(position_encoding_indices)
    # logging.debug(f"Calculated position encoding indices for sequence of length {seq_len}, range: [{pos_min:.3f}, {pos_max:.3f}]")
    
    # Generate sequence name for this record that matches the file structure
    # Extract speaker and session from conv_id (format: speaker_session)
    conv_parts = conv_id.split('_')
    if len(conv_parts) >= 2:
        speaker = '_'.join(conv_parts[:-1])  # Handle multi-part speaker names
        session = conv_parts[-1]
        # Create a generic sequence name that doesn't imply specific file correspondence
        # Format: speaker_session_seq{id} (e.g., WDA_MarkWarner_000_seq01)
        seq_id = str(len(tokenized_records)).zfill(2)
        sequence_name = f"{speaker}_{session}_seq{seq_id}"
    else:
        # Fallback for unexpected format
        sequence_name = f"{conv_id}_sequence_{len(tokenized_records)}"
    
    tokenized_record = {
        "id": len(tokenized_records),
        "conv_id": conv_id,
        "sequence_name": sequence_name,  # Human-readable sequence identifier
        "num_turns": len(chunk["turns"]),
        "input_ids": chunk["input_ids"],
        "attention_mask": attention_mask,
        "labels": chunk["labels"],
        "modality_masks_0": chunk["modality_masks_0"],  # True for text tokens
        "modality_masks_1": chunk["modality_masks_1"],  # True for audio tokens
        "modality_masks_2": chunk["modality_masks_2"],  # True for motion tokens (unified face only)
        "position_encoding_indices": position_encoding_indices,  # Precomputed position indices
        "style_pool": style_pool if style_pool is not None else [],  # All session face token IDs for dynamic style sampling
    }
    tokenized_records.append(tokenized_record)

def convert_to_huggingface_dataset(
    data_root,
    output_path,
    interleaved_turns,
    tokenizer_name,
    max_seq_length=2048,
    audio_segment_size=25,
    face_segment_size=50,
    split="train",
):
    """
    Convert interleaved turns into a HuggingFace dataset with tokenized format and MOT support (Face-Only V3).
    Face-Only V3: Following TFHP format with session-level sequence packing.

    Key features:
    - Face-Only V3: assistant-only, unified motion modality, face-only supervision
    - 3 modalities: text(0), audio(1), motion(2) with FPS {1: 12.5, 2: 25.0}
    - Group size: text + 25 audio + 1 begin_of_motion + 50 face per turn
    - session-level packing: All chunks from same session share conversation_id = speaker_session
    - Cross-session merging: Multiple sessions may merge into sequences up to max_seq_length
    - sequence_name format: speaker_session_seq{id}
    
    Args:
        data_root: Path to the TFHP dataset root
        output_path: Where to save the processed dataset
        interleaved_turns: List of processed interleaved samples (already grouped by fixed-size chunks)
        tokenizer_name: Name of the tokenizer to use
        max_seq_length: Maximum sequence length for tokenization
        audio_segment_size: Number of audio tokens per group (default: 26)
        face_segment_size: Number of face tokens per group (default: 52)
        
    Returns:
        Dataset: The created HuggingFace dataset
    """
    logging.info(f"Converting to Hugging Face dataset (MOT version) with tokenizer: {tokenizer_name}")
    os.makedirs(output_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_token = tokenizer.eos_token
    
    # Add special tokens for face modality only (face only version)
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

    # Pre-compute lookup tables for codebook value → tokenizer ID (avoids per-token tokenizer calls)
    face_id_lookup = {}
    for i in range(512):
        token_str = f"<|face_{i}|>"
        face_id_lookup[i] = tokenizer(token_str, add_special_tokens=False)["input_ids"]
    audio_id_lookup = {}  # lazy-filled cache for audio tokens

    def get_token_ids(val, prefix):
        """Fast token ID lookup with lazy caching."""
        val = max(0, int(val))
        if prefix == "face":
            return face_id_lookup.get(min(val, 511), face_id_lookup[0])
        elif prefix == "audio":
            if val not in audio_id_lookup:
                audio_id_lookup[val] = tokenizer(f"<|audio_{val}|>", add_special_tokens=False)["input_ids"]
            return audio_id_lookup[val]
        else:
            return tokenizer(f"<|{prefix}_{val}|>", add_special_tokens=False)["input_ids"]

    # Prepare assistant prefix and cached tokens
    assistant_prefix = "<|assistant|>streaming_transcription\n"
    assistant_prefix_tokens = tokenizer(assistant_prefix, add_special_tokens=False)["input_ids"]
    eos_token_ids = tokenizer(eos_token, add_special_tokens=False)["input_ids"]
    begin_motion_token_ids = tokenizer("<|begin_of_motion|>", add_special_tokens=False)["input_ids"]

    # # Group turns by conversation ID
    # conversations = {}

    # for conv_id, turns in interleaved_turns.items():
    # # for turn in interleaved_turns:
    #     conv_id = turn.get("conversation_id")
    #     if conv_id not in conversations:
    #         conversations[conv_id] = []
    #     conversations[conv_id].append(turn)
    
    tokenized_records = []
    raw_sequences = []  # For saving raw string sequences
    raw_sequence_metadata = []  # For saving sequence metadata (names, IDs, etc.)
    
    for conv_idx, (conv_id, turns) in enumerate(interleaved_turns.items()):
        asst_turns = [turn for turn in turns if turn.get("speaker_type") == "assistant"]
        if not asst_turns:
            continue

        # Build style_pool: collect ALL face tokens from this session and convert to tokenizer IDs
        style_pool_ids = []
        for t in asst_turns:
            for token in t.get("face_tokens", []):
                token_val = token.tolist() if isinstance(token, np.ndarray) else token
                style_pool_ids.extend(get_token_ids(token_val, "face"))

        # Prepare content tokens and raw string for this turn
        style_control_prefix_text = "style_control:"
        style_control_prefix_tokens = tokenizer(style_control_prefix_text, add_special_tokens=False)["input_ids"]
        style_control_prefix_labels = [-100] * len(style_control_prefix_tokens)  # Text doesn't contribute to loss
        style_control_prefix_modality_0 = [True] * len(style_control_prefix_tokens)   # Text is modality 0
        style_control_prefix_modality_1 = [False] * len(style_control_prefix_tokens)  # Text is not audio
        style_control_prefix_modality_2 = [False] * len(style_control_prefix_tokens)  # Text is not motion

        style_control_content_tokens = turns[0].get("style_control_tokens", [])[:50]
        style_control_token_ids = []
        style_control_token_strs = []
        for token in style_control_content_tokens:
            token_val = token.tolist() if isinstance(token, np.ndarray) else token
            ids = get_token_ids(token_val, "face")
            style_control_token_ids.extend(ids)
            style_control_token_strs.append(f"<|face_{max(0, int(token_val))}|>")
        style_control_content_labels = [-100] * len(style_control_token_ids)  # Text doesn't contribute to loss
        style_control_content_modality_0 = [False] * len(style_control_token_ids)  # Text is not modality 0
        style_control_content_modality_1 = [False] * len(style_control_token_ids)  # Text is not modality 1
        style_control_content_modality_2 = [True] * len(style_control_token_ids)  # Text is modality 2
        

        
        current_input_ids = []
        current_labels = []
        current_turns = []
        current_raw_sequence = []  # For building the raw string
        current_modality_masks_0 = []  # Text tokens
        current_modality_masks_1 = []  # Audio tokens
        current_modality_masks_2 = []  # Motion tokens (unified face)
        
        for turn_idx, asst_turn in enumerate(asst_turns):
            # Prepare content tokens and raw string for this turn
            asst_text = asst_turn.get("text", "")
            text_tokens = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
            text_labels = [-100] * len(text_tokens)  # Text doesn't contribute to loss
            text_modality_0 = [True] * len(text_tokens)   # Text is modality 0
            text_modality_1 = [False] * len(text_tokens)  # Text is not audio
            text_modality_2 = [False] * len(text_tokens)  # Text is not motion
            
            audio_tokens = asst_turn.get("audio_tokens", [])[:audio_segment_size]
            audio_token_ids = []
            audio_token_strs = []
            for token in audio_tokens:
                token_val = token.tolist() if isinstance(token, np.ndarray) else token
                ids = get_token_ids(token_val, "audio")
                audio_token_ids.extend(ids)
                audio_token_strs.append(f"<|audio_{max(0, int(token_val))}|>")
            audio_labels = [-100] * len(audio_token_ids)  # Audio doesn't contribute to loss
            audio_modality_0 = [False] * len(audio_token_ids)  # Audio is not text
            audio_modality_1 = [True] * len(audio_token_ids)   # Audio is modality 1
            audio_modality_2 = [False] * len(audio_token_ids)  # Audio is not motion
            
            face_tokens = asst_turn.get("face_tokens", [])[:face_segment_size]
            face_token_ids = []
            face_token_strs = []
            
            # Initialize motion token arrays
            motion_token_ids = []
            motion_labels = []
            motion_modality_0 = []
            motion_modality_1 = []
            motion_modality_2 = []
            
            # Add unified begin_of_motion token if there are face tokens (V3)
            if face_tokens:
                motion_token_ids.extend(begin_motion_token_ids)
                motion_labels.extend([-100] * len(begin_motion_token_ids))  # Begin token not supervised
                motion_modality_0.extend([False] * len(begin_motion_token_ids))
                motion_modality_1.extend([False] * len(begin_motion_token_ids))
                motion_modality_2.extend([True] * len(begin_motion_token_ids))
            
            # Process individual face tokens
            for token in face_tokens:
                token_val = token.tolist() if isinstance(token, np.ndarray) else token
                ids = get_token_ids(token_val, "face")
                motion_token_ids.extend(ids)
                # Face tokens are supervised as unified motion modality (V3)
                motion_labels.extend(ids)  # Face tokens supervised
                motion_modality_0.extend([False] * len(ids))
                motion_modality_1.extend([False] * len(ids))
                motion_modality_2.extend([True] * len(ids))
            
            # Also keep face_token_ids for raw sequence building (backward compatibility)
            face_token_ids = motion_token_ids.copy()
            
            # Compose the full turn (V3: text + audio + motion)
            turn_input_ids = text_tokens + audio_token_ids + face_token_ids
            turn_labels = text_labels + audio_labels + motion_labels
            turn_modality_0 = text_modality_0 + audio_modality_0 + motion_modality_0
            turn_modality_1 = text_modality_1 + audio_modality_1 + motion_modality_1
            turn_modality_2 = text_modality_2 + audio_modality_2 + motion_modality_2
            turn_raw_str = asst_text + ''.join(audio_token_strs) + ''.join(face_token_strs)
            
            # If starting a new sequence, add the assistant prefix + style control
            if not current_input_ids:
                current_input_ids.extend(assistant_prefix_tokens)
                current_labels.extend([-100] * len(assistant_prefix_tokens))
                current_modality_masks_0.extend([True] * len(assistant_prefix_tokens))
                current_modality_masks_1.extend([False] * len(assistant_prefix_tokens))
                current_modality_masks_2.extend([False] * len(assistant_prefix_tokens))
                current_raw_sequence.append(assistant_prefix)

                current_input_ids.extend(style_control_prefix_tokens)
                current_labels.extend(style_control_prefix_labels)
                current_modality_masks_0.extend(style_control_prefix_modality_0)
                current_modality_masks_1.extend(style_control_prefix_modality_1)
                current_modality_masks_2.extend(style_control_prefix_modality_2)
                current_raw_sequence.append(style_control_prefix_text)

                current_input_ids.extend(style_control_token_ids)
                current_labels.extend(style_control_content_labels)
                current_modality_masks_0.extend(style_control_content_modality_0)
                current_modality_masks_1.extend(style_control_content_modality_1)
                current_modality_masks_2.extend(style_control_content_modality_2)
                current_raw_sequence.append(''.join(style_control_token_strs))

            # Check if adding this turn would exceed max_seq_length
            if len(current_input_ids) + len(turn_input_ids) + len(eos_token_ids) > max_seq_length:
                # Finalize current sequence
                current_input_ids.extend(eos_token_ids)
                current_labels.extend([-100] * len(eos_token_ids))
                current_modality_masks_0.extend([True] * len(eos_token_ids))
                current_modality_masks_1.extend([False] * len(eos_token_ids))
                current_modality_masks_2.extend([False] * len(eos_token_ids))

                tokenized_record = {
                    "input_ids": current_input_ids,
                    "labels": current_labels,
                    "turns": current_turns,
                    "modality_masks_0": current_modality_masks_0,
                    "modality_masks_1": current_modality_masks_1,
                    "modality_masks_2": current_modality_masks_2,
                }
                process_chunk_to_record(tokenized_record, conv_id, tokenized_records, max_seq_length, style_pool=style_pool_ids)
                raw_sequences.append(''.join(current_raw_sequence).strip())

                # Start a new sequence with prefix + style control
                current_input_ids = list(assistant_prefix_tokens)
                current_labels = [-100] * len(assistant_prefix_tokens)
                current_modality_masks_0 = [True] * len(assistant_prefix_tokens)
                current_modality_masks_1 = [False] * len(assistant_prefix_tokens)
                current_modality_masks_2 = [False] * len(assistant_prefix_tokens)
                current_turns = []
                current_raw_sequence = [assistant_prefix]

                current_input_ids.extend(style_control_prefix_tokens)
                current_labels.extend(style_control_prefix_labels)
                current_modality_masks_0.extend(style_control_prefix_modality_0)
                current_modality_masks_1.extend(style_control_prefix_modality_1)
                current_modality_masks_2.extend(style_control_prefix_modality_2)
                current_raw_sequence.append(style_control_prefix_text)

                current_input_ids.extend(style_control_token_ids)
                current_labels.extend(style_control_content_labels)
                current_modality_masks_0.extend(style_control_content_modality_0)
                current_modality_masks_1.extend(style_control_content_modality_1)
                current_modality_masks_2.extend(style_control_content_modality_2)
                current_raw_sequence.append(''.join(style_control_token_strs))

            # Add this turn to the current sequence
            current_input_ids.extend(turn_input_ids)
            current_labels.extend(turn_labels)
            current_modality_masks_0.extend(turn_modality_0)
            current_modality_masks_1.extend(turn_modality_1)
            current_modality_masks_2.extend(turn_modality_2)
            current_turns.append(asst_turn)
            current_raw_sequence.append(turn_raw_str)
        
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
            process_chunk_to_record(tokenized_record, conv_id, tokenized_records, max_seq_length, style_pool=style_pool_ids)
            raw_sequences.append(''.join(current_raw_sequence).strip())
            # Save sequence metadata for final sequence
            sequence_metadata = {
                "sequence_id": len(raw_sequences) - 1,
                "conversation_id": conv_id,
                "turn_range": f"turn_{current_turns[0]['turn_id'].split('_turn_')[-1]}_to_turn_{current_turns[-1]['turn_id'].split('_turn_')[-1]}" if current_turns else "unknown",
                "num_turns_in_sequence": len(current_turns),
                "sequence_length": len(current_input_ids),
            }
            raw_sequence_metadata.append(sequence_metadata)
    
    # Create dataset from records
    if not tokenized_records:
        return None
    
    tokenized_dataset = Dataset.from_list(tokenized_records)
    tokenized_dataset_path = os.path.join(output_path, "tokenized_dataset")
    tokenized_dataset.save_to_disk(tokenized_dataset_path)
    
    # Save raw string sequences for inspection
    raw_sequences_path = os.path.join(output_path, "raw_sequences.txt")
    with open(raw_sequences_path, "w", encoding="utf-8") as f:
        for seq in raw_sequences:
            f.write(seq + "\n")
    
    # Save sequence metadata with names and IDs
    raw_sequences_metadata_path = os.path.join(output_path, "raw_sequences_metadata.json")
    with open(raw_sequences_metadata_path, "w", encoding="utf-8") as f:
        json.dump(raw_sequence_metadata, f, indent=2, ensure_ascii=False)
    
    # Save a combined view with sequence names and content
    raw_sequences_named_path = os.path.join(output_path, "raw_sequences_with_names.txt")
    with open(raw_sequences_named_path, "w", encoding="utf-8") as f:
        for i, (seq, metadata) in enumerate(zip(raw_sequences, raw_sequence_metadata)):
            f.write(f"=== Sequence {i} ===\n")
            f.write(f"Conversation ID: {metadata['conversation_id']}\n")
            f.write(f"Turn Range: {metadata['turn_range']}\n")
            f.write(f"Turns in Sequence: {metadata['num_turns_in_sequence']}\n")
            f.write(f"Sequence Length: {metadata['sequence_length']}\n")
            f.write(f"Content:\n{seq}\n\n")
    
    # Save metadata
    metadata = {
        "split": split,
        f"{split}_size": len(tokenized_records),
        "audio_tokens_per_chunk": audio_segment_size,
        "face_tokens_per_chunk": face_segment_size,
        "format_version": "3.0",
        "format_type": "assistant_only_unified_motion_face_only_with_position_encoding_v3",
        "text_format": "text+audio+face_motion_tokens only with motion supervision and precomputed position encoding indices",
        "position_encoding": "precomputed_indices_based_on_modality_fps",
        "modality_fps": {"1": 12.5, "2": 25.0},
        "supervision": "face body tokens only (as unified motion modality)",
        "tokenized": True,
        "max_seq_length": max_seq_length,
        "assistant_prefix": assistant_prefix.strip(),
        "modality_masks": "masks_0 for text (modality 0), masks_1 for audio (modality 1), masks_2 for unified motion (modality 2)",
        "modality_supervision": "unified_motion_face_only_v3",
        "source": "TFHP_face_only_v3",
        "sequence_tracking": {
            "raw_sequences_file": "raw_sequences.txt",
            "metadata_file": "raw_sequences_metadata.json", 
            "named_sequences_file": "raw_sequences_with_names.txt",
            "total_sequences": len(raw_sequences)
        }
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
    
    return tokenized_dataset

def main():
    """
    Main function to preprocess TFHP dataset with Face-Only V3 format.
    Processes each session by merging all files and grouping into fixed-size chunks.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to TFHP dataset root")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save processed dataset")
    parser.add_argument("--model_name", type=str, default="THUDM/glm-4-voice-9b", help="Tokenizer model name")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--split", type=str, choices=["train", "test", "val"], required=True, help="Which split to process (train, test, or val)")
    parser.add_argument("--debug", action="store_true", help="If set, only process a small number of samples for debugging.")
    args = parser.parse_args()
    
    # Load train/test split
    split_file = os.path.join(args.data_root, f"{args.split}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    with open(split_file, 'r') as f:
        split_sessions = set(line.strip() for line in f if line.strip())
    
    print(f"Loaded {len(split_sessions)} sessions for {args.split} split")
    
    # Define paths to data directories
    transcripts_root = os.path.join(args.data_root, "transcripts")
    audio_root = os.path.join(args.data_root, "audios_token_glm")
    face_root = os.path.join(args.data_root, "TOKENS_AGENT_25")
    
    # Process each speaker and session
    interleaved_turns = {}
    speakers = os.listdir(transcripts_root)
    if args.debug:
        speakers = speakers[:1]
    for speaker in tqdm(speakers, desc="Processing speakers"):
        speaker_dir = os.path.join(transcripts_root, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        sessions = os.listdir(speaker_dir)
        if args.debug:
            sessions = sessions[:10]
        for session in sessions:
            session_dir = os.path.join(speaker_dir, session)
            if not os.path.isdir(session_dir):
                continue
            session_key = f"{speaker}/{session}"
            if session_key not in split_sessions:
                continue
            audio_dir = os.path.join(audio_root, speaker, session)
            face_dir = os.path.join(face_root, speaker, session)
            if not (os.path.exists(audio_dir) and os.path.exists(face_dir)):
                continue

            texts, audios, faces, word_timestamps = merge_session_fragments(session_dir, audio_dir, face_dir)
            interleave_samples = interleave_split_sequence(texts, audios, faces, word_timestamps)

            # Collect all face tokens from this session for style selection
            all_session_face = []
            for sample in interleave_samples:
                all_session_face.extend(sample["face_tokens"])

            # Style selection depends on split:
            #   train: first 50 tokens from 000.npy (will be overridden by collator anyway)
            #   test/val: fixed middle position (ARTalk-style deterministic eval)
            if args.split == "train":
                style_control_path = os.path.join(face_dir, '000.npy')
                try:
                    style_control_tokens = np.load(style_control_path)[0, :50].tolist()
                except Exception:
                    continue
            else:
                # ARTalk-style: select from middle of all session face tokens
                n = len(all_session_face)
                style_len = 50
                if n <= style_len:
                    style_control_tokens = all_session_face
                else:
                    start = min(n // 2, n - style_len)
                    style_control_tokens = all_session_face[start:start + style_len]

            interleaved_turns[f"{speaker}_{session}"] = []
            for idx, sample in enumerate(interleave_samples):
                interleaved_turns[f"{speaker}_{session}"].append({
                    "conversation_id": f"{speaker}_{session}",
                    "turn_id": f"{speaker}_{session}_turn_{idx}",
                    "text": sample["text"],
                    "audio_tokens": sample["audio_tokens"],
                    "face_tokens": sample["face_tokens"],
                    "speaker_type": "assistant",
                    "style_control_tokens": style_control_tokens
                })
    
    # Convert to HuggingFace dataset format with MOT support (face only version)
    convert_to_huggingface_dataset(
        data_root=args.data_root,
        output_path=args.output_path,
        interleaved_turns=interleaved_turns,
        tokenizer_name=args.model_name,
        max_seq_length=args.max_seq_length,
        audio_segment_size=25,
        face_segment_size=50,
        split=args.split,
    )

if __name__ == "__main__":
    main() 