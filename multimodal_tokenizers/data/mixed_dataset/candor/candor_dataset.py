import os
import json
import numpy as np
import torch
from torch.utils import data
from pathlib import Path
from os.path import join as pjoin
import logging
from rich.progress import track
import random

logger = logging.getLogger(__name__)

class CandorDataset(data.Dataset):
    """
    Dataset loader for the CANDOR dataset with audio and face tokens.
    
    This dataset handles loading of pre-tokenized audio and face data from the CANDOR dataset,
    processing them into chunks suitable for training a conversational agent.
    """
    
    def __init__(
        self,
        data_root,
        audio_token_dir="audios_token_glm",
        face_token_dir="TOKENS_DS4",
        structure_file="candor_structure.json",
        audio_fps=12.5,
        face_fps=6.25,
        audio_tokens_per_chunk=26,
        face_tokens_per_chunk=13,
        split="train",
        debug=False,
        max_data=None,
        **kwargs
    ):
        """
        Initialize the CANDOR dataset.
        
        Args:
            data_root: Root directory for the CANDOR dataset
            audio_token_dir: Directory containing audio tokens relative to data_root
            face_token_dir: Directory containing face tokens relative to data_root
            structure_file: JSON file containing conversation structure
            audio_fps: Frames per second for audio tokens
            face_fps: Frames per second for face tokens
            audio_tokens_per_chunk: Number of audio tokens per chunk
            face_tokens_per_chunk: Number of face tokens per chunk
            split: Data split (train/val)
            debug: If True, load a small subset of data for debugging
            max_data: Maximum number of samples to load (useful for debugging)
        """
        self.data_root = Path(data_root)
        self.audio_token_dir = self.data_root / audio_token_dir
        self.face_token_dir = self.data_root / face_token_dir
        self.structure_path = self.data_root / structure_file
        
        self.audio_fps = audio_fps
        self.face_fps = face_fps
        self.audio_tokens_per_chunk = audio_tokens_per_chunk
        self.face_tokens_per_chunk = face_tokens_per_chunk
        
        self.split = split
        self.debug = debug
        
        # Set maximum data size for debugging
        if debug:
            self.max_data = 10 if max_data is None else max_data
        else:
            self.max_data = max_data
        
        # Load conversation structure
        self.structure = self._load_structure()
        
        # Split data based on split parameter
        self.conv_ids = list(self.structure.keys())
        if not debug:
            # Deterministic split based on conversation ID hash
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(self.conv_ids)
            
            if split == "train":
                self.conv_ids = self.conv_ids[:int(len(self.conv_ids) * 0.8)]
            elif split == "val":
                self.conv_ids = self.conv_ids[int(len(self.conv_ids) * 0.8):int(len(self.conv_ids) * 0.9)]
            elif split == "test":
                # Use the last 10% of shuffled IDs for testing
                self.conv_ids = self.conv_ids[int(len(self.conv_ids) * 0.9):]
            # If split is not one of the above, use all conversations
        
        # Load and process data
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} CANDOR conversation chunks for {split} split")
    
    def _load_structure(self):
        """Load the conversation structure from JSON file."""
        try:
            with open(self.structure_path, 'r') as f:
                structure = json.load(f)
            print(f"Loaded conversation structure with {len(structure)} conversations")
            return structure
        except Exception as e:
            print(f"Failed to load conversation structure: {e}")
            return {}
    
    def _load_data(self):
        """Load audio and face tokens and organize them into chunks."""
        data = []
        
        # Limit the number of conversations for debugging
        conv_ids = self.conv_ids[:self.max_data] if self.max_data else self.conv_ids
        
        for conv_id in track(conv_ids, description=f"Loading CANDOR {self.split}"):


            # Get speaker IDs for this conversation
            speaker_files = self.structure[conv_id]
            
            # Load audio and face tokens for both speakers
            speaker_data = []
            for speaker_file in speaker_files:
                speaker_id = speaker_file.split('.')[0]  # Remove file extension
                
                # Path to audio tokens
                audio_path = self.audio_token_dir / conv_id / f"{speaker_id}.npy"
                if not os.path.exists(audio_path):
                    print(f"Audio tokens not found: {audio_path}")
                    continue
                
                # Path to face tokens
                face_path = self.face_token_dir / conv_id / f"{speaker_id}.npy"
                if not os.path.exists(face_path):
                    print(f"Face tokens not found: {face_path}")
                    continue
                
                # Load tokens
                audio_tokens = np.load(audio_path)
                face_tokens = np.load(face_path)
                
                speaker_data.append({
                    "speaker_id": speaker_id,
                    "audio_tokens": audio_tokens,
                    "face_tokens": face_tokens
                })
            
            # Skip if we don't have both speakers
            if len(speaker_data) != 2:
                print(f"Skipping conversation {conv_id}: only {len(speaker_data)} speakers found")
                continue
            
            # Process the conversation data into chunks
            chunks = self._process_conversation(speaker_data, conv_id)
            data.extend(chunks)
                
        
        return data
    
    def _process_conversation(self, speaker_data, conv_id):
        """
        Process a conversation into chunks with interleaved turns.
        
        Args:
            speaker_data: List of dictionaries with speaker audio and face tokens
            conv_id: Conversation ID
            
        Returns:
            List of conversation chunks
        """
        chunks = []
        
        # Estimate speaking turns based on audio activity
        # Simplified approach: detect silence gaps as turn boundaries
        speaker0_turns = self._estimate_speaking_turns(speaker_data[0]["audio_tokens"])
        speaker1_turns = self._estimate_speaking_turns(speaker_data[1]["audio_tokens"])
        
        # Interleave turns from both speakers
        all_turns = []
        for s0_turn in speaker0_turns:
            all_turns.append({"speaker_idx": 0, "start": s0_turn["start"], "end": s0_turn["end"]})
        for s1_turn in speaker1_turns:
            all_turns.append({"speaker_idx": 1, "start": s1_turn["start"], "end": s1_turn["end"]})
        
        # Sort turns by start time
        all_turns.sort(key=lambda x: x["start"])
        
        # Process each turn into chunks
        for i, turn in enumerate(all_turns):
            speaker_idx = turn["speaker_idx"]
            start_idx = turn["start"]
            end_idx = turn["end"]
            
            # Skip very short turns
            if end_idx - start_idx < self.audio_tokens_per_chunk:
                continue
            
            # Extract audio and face tokens for this turn
            audio_tokens = speaker_data[speaker_idx]["audio_tokens"][start_idx:end_idx]
            
            # Calculate corresponding face token indices
            # Face tokens are at half the frame rate of audio tokens
            face_start_idx = start_idx // 2
            face_end_idx = end_idx // 2
            face_tokens = speaker_data[speaker_idx]["face_tokens"][face_start_idx:face_end_idx]
            
            # Split into chunks
            for chunk_start in range(0, len(audio_tokens) - self.audio_tokens_per_chunk + 1, 
                                    self.audio_tokens_per_chunk // 2):  # 50% overlap between chunks
                
                # Extract audio chunk
                audio_chunk_end = chunk_start + self.audio_tokens_per_chunk
                if audio_chunk_end > len(audio_tokens):
                    continue
                
                audio_chunk = audio_tokens[chunk_start:audio_chunk_end]
                
                # Extract corresponding face chunk
                face_chunk_start = chunk_start // 2
                face_chunk_end = face_chunk_start + self.face_tokens_per_chunk
                if face_chunk_end > len(face_tokens):
                    continue
                
                face_chunk = face_tokens[face_chunk_start:face_chunk_end]
                
                # Create chunk data
                chunk = {
                    "conv_id": conv_id,
                    "chunk_id": len(chunks),
                    "speaker_id": speaker_data[speaker_idx]["speaker_id"],
                    "speaker_idx": speaker_idx,
                    "audio_tokens": audio_chunk,
                    "face_tokens": face_chunk,
                    "turn_idx": i
                }
                
                chunks.append(chunk)
        
        return chunks
    
    def _estimate_speaking_turns(self, audio_tokens, silence_threshold=0.2, min_gap=10):
        """
        Estimate speaking turns by detecting silence gaps in audio tokens.
        
        This is a simplified approach that treats sequences of low-activity 
        audio tokens as turn boundaries.
        
        Args:
            audio_tokens: Array of audio tokens
            silence_threshold: Activity threshold below which tokens are considered silence
            min_gap: Minimum gap length (in tokens) to consider a turn boundary
            
        Returns:
            List of speaking turns with start and end indices
        """
        # Calculate activity measure (using L2 norm as a simple proxy)
        if audio_tokens.ndim > 1:
            # If audio_tokens is a 2D array (e.g., features per token)
            activity = np.linalg.norm(audio_tokens, axis=1)
        else:
            # If audio_tokens is a 1D array
            activity = np.abs(audio_tokens)
        
        # Normalize activity
        if activity.max() > 0:
            activity = activity / activity.max()
        
        # Label silent segments
        is_silent = activity < silence_threshold
        
        # Find continuous speaking segments
        turns = []
        in_turn = False
        turn_start = 0
        
        for i, silent in enumerate(is_silent):
            if not in_turn and not silent:
                # Start of a new turn
                turn_start = i
                in_turn = True
            elif in_turn and silent:
                # Potential end of turn - check if gap is long enough
                gap_start = i
                gap_end = i
                while gap_end < len(is_silent) and is_silent[gap_end]:
                    gap_end += 1
                
                gap_length = gap_end - gap_start
                
                if gap_length >= min_gap:
                    # End of turn
                    turns.append({"start": turn_start, "end": i})
                    in_turn = False
        
        # Handle case where the last turn goes until the end
        if in_turn:
            turns.append({"start": turn_start, "end": len(is_silent)})
        
        return turns
    
    def __getitem__(self, index):
        """Retrieve a specific chunk from the dataset."""
        return self.data[index]
    
    def __len__(self):
        """Return the total number of chunks in the dataset."""
        return len(self.data) 