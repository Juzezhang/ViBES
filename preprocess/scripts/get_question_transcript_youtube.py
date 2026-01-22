"""
Video processing script to extract speaking and non-speaking segments and generate transcripts.

This script uses TalkNet for active speaker detection in videos and Whisper for transcription.
It can process individual videos or entire directories, extracting speaking/non-speaking segments,
transcribing them, and creating organized transcripts in Q&A format.

Key features:
- Uses TalkNet for precise speaker detection
- Transcribes audio with Whisper for high-quality results
- Generates separate transcripts for speaking and non-speaking segments
- Creates combined, chronological Q&A format transcripts
- Supports both individual segment transcription and full-track transcription
- Organizes transcripts by timeline based on speaker detection results
- Properly formats transcripts with clear speaker indicators (A: for answers, Q: for questions)

The combined transcript generation uses TalkNet speaker detection results to create a precise
timeline of speaking and non-speaking segments, ensuring accurate assignment of text to the
appropriate segments while maintaining chronological order.
"""

import os
import numpy as np
import torch
from tqdm import tqdm
import librosa
from faster_whisper import WhisperModel
import cv2
import subprocess
import json
# import moviepy.editor as mp
from pathlib import Path
import warnings
import soundfile as sf
from scipy.io import wavfile
import sys
import pickle
import glob
import argparse
import re

# Ignore warnings
warnings.filterwarnings("ignore")

def load_speaking_segments(video_path, talknet_output_dir=None):
    """
    Load TalkNet speaking segments from saved output
    Args:
        video_path: Path to the video file
        talknet_output_dir: Directory containing TalkNet outputs
    Returns:
        [(start1, end1), (start2, end2), ...] speaking intervals (seconds) or None if results don't exist
    """
    if talknet_output_dir is None:
        print("Error: talknet_output_dir is required")
        return None
    
    # Ensure input video exists
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return None
    
    # Get video name for organizing outputs
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Check for TalkNet output
    talknet_video_dir = os.path.join(talknet_output_dir, video_name)
    pywork_dir = os.path.join(talknet_video_dir, "pywork")
    scores_pkl = os.path.join(pywork_dir, "scores.pckl")
    tracks_pkl = os.path.join(pywork_dir, "tracks.pckl")
    
    # Check if TalkNet results exist
    if not os.path.exists(scores_pkl) or not os.path.exists(tracks_pkl):
        print(f"TalkNet results not found for {video_name}. Skipping this video.")
        return None
    
    try:
        # Load scores and tracks
        with open(scores_pkl, "rb") as f:
            scores = pickle.load(f)
        with open(tracks_pkl, "rb") as f:
            tracks = pickle.load(f)
        
        # Extract speaking segments
        speaking_segments = []
        for track_idx, (track, score) in enumerate(zip(tracks, scores)):
            try:
                # Extract speaking probability for each timestamp
                prob = score
                if isinstance(prob, np.ndarray) and len(prob.shape) > 1 and prob.shape[1] > 1:
                    prob = prob[:, 1]  # If 2D array with second dimension > 1, take second column as speaking probability
                
                # Apply threshold to detect speaking parts (using default 0.5 threshold)
                # threshold = 0.5
                # Modify by Juze
                threshold = 0.0
                speaking = prob > threshold
                
                # Identify continuous speaking segments
                start = None
                for idx, is_speaking in enumerate(speaking):
                    t = track['track']['frame'][idx] / 25.0  # Convert to seconds
                    if is_speaking and start is None:
                        start = t
                    elif not is_speaking and start is not None:
                        speaking_segments.append((start, t))
                        start = None
                        
                # Handle case where speaking continues until the end
                if start is not None:
                    t = track['track']['frame'][-1] / 25.0
                    speaking_segments.append((start, t))
            except Exception as e:
                print(f"Error processing track {track_idx}: {e}")
        
        # Merge adjacent speaking segments
        speaking_segments = merge_segments(speaking_segments)
        
        print(f"Loaded {len(speaking_segments)} speaking segments from TalkNet results for {video_name}")
        return speaking_segments
    
    except Exception as e:
        print(f"Error loading TalkNet results for {video_name}: {e}")
        return None

def merge_segments(segments, tol=0.2):
    """Merge overlapping or adjacent intervals"""
    if not segments:
        return []
    segments = sorted(segments)
    merged = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + tol:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged

def extract_audio_segment(video_path, output_path, start_time, end_time):
    """
    Extract a segment of audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio
        start_time: Start time of the segment in seconds
        end_time: End time of the segment in seconds
    """
    try:
        # Use FFmpeg to extract the audio segment
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"Error extracting audio segment: {e}")

def extract_full_audio_tracks(video_path, speaking_segments, video_duration, speaking_audio_path, non_speaking_audio_path):
    """
    Extract full speaking and non-speaking audio tracks from a video.
    
    Args:
        video_path: Path to the video file
        speaking_segments: List of speaking segments, format [(start1, end1), (start2, end2), ...]
        video_duration: Total duration of the video (seconds)
        speaking_audio_path: Path to save the speaking audio
        non_speaking_audio_path: Path to save the non-speaking audio
    
    Returns:
        bool: Whether extraction was successful
    """
    try:
        # Create a temporary directory for intermediate files
        temp_dir = os.path.join(os.path.dirname(speaking_audio_path), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 1. First extract the full audio from the video
        full_audio_path = os.path.join(temp_dir, "full_audio.wav")
        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            full_audio_path
        ]
        subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 2. Create complex filter strings for muting/keeping speaking parts
        speaking_filter = "volume=0"  # Default: mute all
        non_speaking_filter = "volume=1"  # Default: keep all
        
        # Process speaking segments
        if speaking_segments:
            speaking_parts = []
            non_speaking_parts = []
            
            # Divide the video timeline into speaking and non-speaking parts
            time_points = [0]
            for start, end in speaking_segments:
                if start > time_points[-1]:
                    time_points.append(start)
                time_points.append(end)
            if time_points[-1] < video_duration:
                time_points.append(video_duration)
            
            # Create filter expressions
            for i in range(len(time_points) - 1):
                start = time_points[i]
                end = time_points[i+1]
                duration = end - start
                
                # Check if this interval is a speaking part
                is_speaking_part = any(s <= start < e or s < end <= e or (start <= s and end >= e) 
                                      for s, e in speaking_segments)
                
                if is_speaking_part:
                    speaking_parts.append(f"volume=1:enable='between(t,{start},{end})'")
                    non_speaking_parts.append(f"volume=0:enable='between(t,{start},{end})'")
                else:
                    speaking_parts.append(f"volume=0:enable='between(t,{start},{end})'")
                    non_speaking_parts.append(f"volume=1:enable='between(t,{start},{end})'")
            
            speaking_filter = ','.join(speaking_parts)
            non_speaking_filter = ','.join(non_speaking_parts)
        
        # 3. Create speaking audio
        speaking_cmd = [
            "ffmpeg", "-y", "-i", full_audio_path,
            "-af", speaking_filter,
            speaking_audio_path
        ]
        subprocess.run(speaking_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 4. Create non-speaking audio
        non_speaking_cmd = [
            "ffmpeg", "-y", "-i", full_audio_path,
            "-af", non_speaking_filter,
            non_speaking_audio_path
        ]
        subprocess.run(non_speaking_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 5. Clean up temporary files
        if os.path.exists(full_audio_path):
            os.remove(full_audio_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
        
        return True
    
    except Exception as e:
        print(f"Error extracting audio tracks: {e}")
        return False

def process_audio(audio_path):
    """
    Process audio file to prepare for Whisper transcription.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Processed audio data
    """
    sample_rate, audio_data = wavfile.read(audio_path)
    
    # Ensure audio is float32 and in [-1, 1] range
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    if np.abs(audio_data).max() > 0:  # Prevent division by zero
        audio_data = audio_data / np.abs(audio_data).max()
    
    # If stereo, convert to mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = audio_data.mean(axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_data = librosa.resample(
            y=audio_data, 
            orig_sr=sample_rate, 
            target_sr=16000
        )
    
    return audio_data

def get_transcription(model, audio_data):
    """
    Get transcription with word-level timestamps using Whisper.
    
    Args:
        model: Whisper model
        audio_data: Audio data to transcribe
        
    Returns:
        List of segments with transcriptions and timestamps
    """
    segments, info = model.transcribe(
        audio_data,
        task="transcribe",
        language="en",
        word_timestamps=True
    )
    
    return list(segments), info

def save_transcript(output_file, segments, info, segment_type):
    """
    Save transcription to a file.
    
    Args:
        output_file: Path to save the transcript
        segments: List of transcription segments
        info: Transcription info
        segment_type: Type of segment (speaker or non_speaker)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Determine speaker indicator
        speaker_indicator = "A: " if segment_type == "answer" or segment_type == "speaking_track" else "Q: "
        f.write(f"Type: {segment_type} ({speaker_indicator})\n\n")
        
        # Write full text with speaker indicators
        full_text = ' '.join(segment.text for segment in segments)
        
        # Make sure we don't duplicate speaker prefixes
        if full_text.startswith(speaker_indicator):
            f.write(f"Full text: {full_text}\n\n")
        else:
            f.write(f"Full text: {speaker_indicator}{full_text}\n\n")
        
        # Write language and detection info
        f.write(f"Detected language: {info.language} (probability: {info.language_probability:.3f})\n\n")
        
        # Write detailed segment info
        f.write("Segments with word-level timestamps:\n")
        for i, segment in enumerate(segments, 1):
            f.write(f"\nSegment {i}:\n")
            f.write(f"Timestamp: {segment.start:.3f}s - {segment.end:.3f}s\n")
            
            # Make sure we don't duplicate speaker prefixes for segment text
            segment_text = segment.text
            if not segment_text.startswith(speaker_indicator):
                segment_text = f"{speaker_indicator}{segment_text}"
            f.write(f"Text: {segment_text}\n")
            
            if hasattr(segment, 'words') and segment.words:
                f.write("Words:\n")
                for word in segment.words:
                    f.write(f"{word.word}: {word.start:.3f}s - {word.end:.3f}s")
                    if hasattr(word, 'confidence'):
                        f.write(f" (confidence: {word.confidence:.3f})")
                    f.write("\n")
            
            f.write("-" * 50 + "\n")

def save_combined_transcript(video_name, transcript_dir, speaking_segments, non_speaking_segments, use_full_tracks=False):
    """
    Combine speaker and non-speaker transcripts into a single chronological Q&A file.
    
    Args:
        video_name: Name of the video
        transcript_dir: Directory containing transcript files
        speaking_segments: List of speaking segments with timestamps based on TalkNet detection
        non_speaking_segments: List of non-speaking segments with timestamps
        use_full_tracks: Whether to use full track transcripts instead of segments
    
    Returns:
        Path to the combined transcript file
    """
    combined_transcript_path = os.path.join(transcript_dir, f"{video_name}_combined_qa.txt")
    
    # Skip if combined transcript already exists
    if os.path.exists(combined_transcript_path):
        print(f"[SKIP] Combined Q&A transcript already exists at {combined_transcript_path}")
        return combined_transcript_path
        
    # Sort speaking_segments by time to ensure they are processed in order
    speaking_segments = sorted(speaking_segments)
    
    # Print TalkNet speaking segments for debugging
    print(f"TalkNet speaking segments for {video_name}:")
    for i, (start, end) in enumerate(speaking_segments):
        print(f"  Speaking {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    # Load word-level timestamp data
    speaking_words = []
    non_speaking_words = []
    
    if use_full_tracks:
        # Use full track transcripts
        speaking_track_path = os.path.join(transcript_dir, f"{video_name}_speaking_track.txt")
        non_speaking_track_path = os.path.join(transcript_dir, f"{video_name}_non_speaking_track.txt")
        
        # Load words from speaking track transcript
        if os.path.exists(speaking_track_path):
            try:
                with open(speaking_track_path, 'r', encoding='utf-8') as f:
                    speaking_content = f.read()
                
                # Parse word-level timestamps
                segments_mode = False
                current_segment = None
                
                for line in speaking_content.split('\n'):
                    # Detect entering word-level timestamp section
                    if line.startswith("Segments with word-level timestamps:"):
                        segments_mode = True
                        continue
                    
                    # Detect new segment
                    if segments_mode and line.startswith("Segment "):
                        current_segment = {}
                        continue
                        
                    # Parse segment timestamps
                    if segments_mode and current_segment is not None and line.startswith("Timestamp:"):
                        timestamp_parts = line.replace("Timestamp:", "").strip().split(" - ")
                        if len(timestamp_parts) == 2:
                            segment_start = float(timestamp_parts[0].replace("s", ""))
                            segment_end = float(timestamp_parts[1].replace("s", ""))
                            current_segment["start"] = segment_start
                            current_segment["end"] = segment_end
                        continue
                    
                    # Parse segment text (remove prefix)
                    if segments_mode and current_segment is not None and line.startswith("Text:"):
                        text = line.replace("Text:", "").strip()
                        if text.startswith("A: "):
                            text = text[3:].strip()
                        current_segment["text"] = text
                        continue
                    
                    # Detect start of word-level timestamp section
                    if segments_mode and current_segment is not None and line.startswith("Words:"):
                        current_segment["words"] = []
                        continue
                    
                    # Parse word-level timestamps
                    if segments_mode and current_segment is not None and "words" in current_segment and ":" in line and "-" in line and not line.startswith("Segment") and len(line.strip()) > 0:
                        try:
                            match = re.match(r'^(.*?):\s*([\d.]+)s\s*-\s*([\d.]+)s(?:\s*\(confidence:\s*([\d.]+)\))?', line.strip())
                            if match:
                                word = match.group(1).strip()
                                start_time = float(match.group(2))
                                end_time = float(match.group(3))
                                confidence = float(match.group(4)) if match.group(4) else None
                                word_info = {
                                    "word": word,
                                    "start": start_time,
                                    "end": end_time,
                                    "confidence": confidence
                                }
                                speaking_words.append(word_info)
                            else:
                                print(f"Error parsing word timestamp: {line} - format not matched")
                        except Exception as e:
                            print(f"Error parsing word timestamp: {line} - {e}")
                        continue
            except Exception as e:
                print(f"Error reading speaking track transcript: {e}")
        
        # Load words from non-speaking track transcript
        if os.path.exists(non_speaking_track_path):
            try:
                with open(non_speaking_track_path, 'r', encoding='utf-8') as f:
                    non_speaking_content = f.read()
                
                # Parse word-level timestamps
                segments_mode = False
                current_segment = None
                
                for line in non_speaking_content.split('\n'):
                    # Detect entering word-level timestamp section
                    if line.startswith("Segments with word-level timestamps:"):
                        segments_mode = True
                        continue
                    
                    # Detect new segment
                    if segments_mode and line.startswith("Segment "):
                        current_segment = {}
                        continue
                        
                    # Parse segment timestamps
                    if segments_mode and current_segment is not None and line.startswith("Timestamp:"):
                        timestamp_parts = line.replace("Timestamp:", "").strip().split(" - ")
                        if len(timestamp_parts) == 2:
                            segment_start = float(timestamp_parts[0].replace("s", ""))
                            segment_end = float(timestamp_parts[1].replace("s", ""))
                            current_segment["start"] = segment_start
                            current_segment["end"] = segment_end
                        continue
                    
                    # Parse segment text (remove prefix)
                    if segments_mode and current_segment is not None and line.startswith("Text:"):
                        text = line.replace("Text:", "").strip()
                        if text.startswith("Q: "):
                            text = text[3:].strip()
                        current_segment["text"] = text
                        continue
                    
                    # Detect start of word-level timestamp section
                    if segments_mode and current_segment is not None and line.startswith("Words:"):
                        current_segment["words"] = []
                        continue
                    
                    # Parse word-level timestamps
                    if segments_mode and current_segment is not None and "words" in current_segment and ":" in line and "-" in line and not line.startswith("Segment") and len(line.strip()) > 0:
                        try:
                            match = re.match(r'^(.*?):\s*([\d.]+)s\s*-\s*([\d.]+)s(?:\s*\(confidence:\s*([\d.]+)\))?', line.strip())
                            if match:
                                word = match.group(1).strip()
                                start_time = float(match.group(2))
                                end_time = float(match.group(3))
                                confidence = float(match.group(4)) if match.group(4) else None
                                word_info = {
                                    "word": word,
                                    "start": start_time,
                                    "end": end_time,
                                    "confidence": confidence
                                }
                                non_speaking_words.append(word_info)
                            else:
                                print(f"Error parsing word timestamp: {line} - format not matched")
                        except Exception as e:
                            print(f"Error parsing word timestamp: {line} - {e}")
                        continue
            except Exception as e:
                print(f"Error reading non-speaking track transcript: {e}")
    else:
        # Process from individual segment files
        # Load all answer segments
        for i, (start, end) in enumerate(speaking_segments):
            segment_id = f"{video_name}_answer_{i:03d}"
            transcript_path = os.path.join(transcript_dir, f"{segment_id}.txt")
            
            if os.path.exists(transcript_path):
                try:
                    # Extract words from file
                    with open(transcript_path, 'r', encoding='utf-8') as tf:
                        transcript_lines = tf.readlines()
                    
                    # Extract all words with timestamps
                    segments_mode = False
                    
                    for line in transcript_lines:
                        # Detect entering word-level timestamp section
                        if line.startswith("Segments with word-level timestamps:"):
                            segments_mode = True
                            continue
                        
                        # Detect start of word-level timestamp section
                        if segments_mode and line.startswith("Words:"):
                            continue
                        
                        # Parse word-level timestamps
                        if segments_mode and ":" in line and "-" in line and not line.startswith("Segment") and not line.startswith("Timestamp:") and not line.startswith("Text:") and len(line.strip()) > 0:
                            try:
                                match = re.match(r'^(.*?):\s*([\d.]+)s\s*-\s*([\d.]+)s(?:\s*\(confidence:\s*([\d.]+)\))?', line.strip())
                                if match:
                                    word = match.group(1).strip()
                                    start_time = float(match.group(2))
                                    end_time = float(match.group(3))
                                    confidence = float(match.group(4)) if match.group(4) else None
                                    word_info = {
                                        "word": word,
                                        "start": start_time + start,  # Adjust timestamp to video time
                                        "end": end_time + start,      # Adjust timestamp to video time
                                        "confidence": confidence
                                    }
                                    speaking_words.append(word_info)
                                else:
                                    print(f"Error parsing word timestamp: {line} - format not matched")
                            except Exception as e:
                                print(f"Error parsing word timestamp: {line} - {e}")
                        
                except Exception as e:
                    print(f"Error reading transcript {transcript_path}: {e}")
        
        # Load all question segments
        for i, (start, end) in enumerate(non_speaking_segments):
            segment_id = f"{video_name}_question_{i:03d}"
            transcript_path = os.path.join(transcript_dir, f"{segment_id}.txt")
            
            if os.path.exists(transcript_path):
                try:
                    # Extract words from file
                    with open(transcript_path, 'r', encoding='utf-8') as tf:
                        transcript_lines = tf.readlines()
                    
                    # Extract all words with timestamps
                    segments_mode = False
                    
                    for line in transcript_lines:
                        # Detect entering word-level timestamp section
                        if line.startswith("Segments with word-level timestamps:"):
                            segments_mode = True
                            continue
                        
                        # Detect start of word-level timestamp section
                        if segments_mode and line.startswith("Words:"):
                            continue
                        
                        # Parse word-level timestamps
                        if segments_mode and ":" in line and "-" in line and not line.startswith("Segment") and not line.startswith("Timestamp:") and not line.startswith("Text:") and len(line.strip()) > 0:
                            try:
                                match = re.match(r'^(.*?):\s*([\d.]+)s\s*-\s*([\d.]+)s(?:\s*\(confidence:\s*([\d.]+)\))?', line.strip())
                                if match:
                                    word = match.group(1).strip()
                                    start_time = float(match.group(2))
                                    end_time = float(match.group(3))
                                    confidence = float(match.group(4)) if match.group(4) else None
                                    word_info = {
                                        "word": word,
                                        "start": start_time + start,  # Adjust timestamp to video time
                                        "end": end_time + start,      # Adjust timestamp to video time
                                        "confidence": confidence
                                    }
                                    non_speaking_words.append(word_info)
                                else:
                                    print(f"Error parsing word timestamp: {line} - format not matched")
                            except Exception as e:
                                print(f"Error parsing word timestamp: {line} - {e}")
                        
                except Exception as e:
                    print(f"Error reading transcript {transcript_path}: {e}")
    
    # Create final transcript segments
    transcript_segments = []
    
    # First, strictly create A segments according to TalkNet speaking segments
    for start, end in speaking_segments:
        # Create an A segment for each TalkNet speaking segment
        segment = {
            'start': start,
            'end': end,
            'type': 'A',
            'words': []
        }
        transcript_segments.append(segment)
    
    # Find all gaps between speaking_segments and create Q segments
    # First handle the gap from the start of the video to the first speaking segment
    if speaking_segments and speaking_segments[0][0] > 0:
        transcript_segments.append({
            'start': 0,
            'end': speaking_segments[0][0],
            'type': 'Q',
            'words': []
        })
    
    # Then handle all gaps between speaking segments
    for i in range(len(speaking_segments) - 1):
        current_end = speaking_segments[i][1]
        next_start = speaking_segments[i+1][0]
        
        if next_start > current_end:  # If there is a gap
            transcript_segments.append({
                'start': current_end,
                'end': next_start,
                'type': 'Q',
                'words': []
            })
    
    # Finally, handle the gap from the last speaking segment to the end of the video
    if speaking_segments:
        last_end = speaking_segments[-1][1]
        # Assume the maximum time is based on the max end time of all words
        max_time = 0
        for word in speaking_words + non_speaking_words:
            max_time = max(max_time, word['end'])
        
        if max_time > last_end:
            transcript_segments.append({
                'start': last_end,
                'end': max_time,
                'type': 'Q',
                'words': []
            })
    
    # Sort all segments by start time
    transcript_segments.sort(key=lambda x: x['start'])
    
    # Now assign words to the corresponding segments
    # Process speaking words
    for word in speaking_words:
        word_start = word['start']
        
        # Find the A segment this word belongs to
        for segment in transcript_segments:
            if segment['type'] == 'A' and word_start >= segment['start'] and word_start < segment['end']:
                segment['words'].append(word)
                break
    
    # Process non-speaking words
    for word in non_speaking_words:
        word_start = word['start']
        
        # Find the Q segment this word belongs to
        for segment in transcript_segments:
            if segment['type'] == 'Q' and word_start >= segment['start'] and word_start < segment['end']:
                segment['words'].append(word)
                break
    
    # Filter out Q segments with no words and merge adjacent A segments
    filtered_segments = []
    i = 0
    while i < len(transcript_segments):
        current_segment = transcript_segments[i]
        
        # If current segment is A, keep it
        if current_segment['type'] == 'A':
            # Initialize a new merged segment
            merged_segment = {
                'start': current_segment['start'],
                'end': current_segment['end'],
                'type': 'A',
                'words': current_segment['words'].copy()
            }
            
            # Find the next segment
            j = i + 1
            # Skip Q segments with no words and merge adjacent A segments
            while j < len(transcript_segments):
                next_segment = transcript_segments[j]
                # If next segment is Q and has no words, skip
                if next_segment['type'] == 'Q' and not next_segment.get('words', []):
                    j += 1
                    # If after skipping Q, there is another A segment, merge it
                    if j < len(transcript_segments) and transcript_segments[j]['type'] == 'A':
                        next_a_segment = transcript_segments[j]
                        merged_segment['end'] = next_a_segment['end']
                        merged_segment['words'].extend(next_a_segment['words'])
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Add the merged A segment
            filtered_segments.append(merged_segment)
            i = j
        # If current segment is Q and has words, keep it
        elif current_segment['type'] == 'Q' and current_segment.get('words', []):
            filtered_segments.append(current_segment)
            i += 1
        # Otherwise, skip (Q segment with no words)
        else:
            i += 1
    
    # Sort words within each segment by time
    for segment in filtered_segments:
        segment['words'].sort(key=lambda x: x['start'])
    
    # Write the combined transcript file
    with open(combined_transcript_path, 'w', encoding='utf-8') as f:
        f.write(f"# Combined Q&A Transcript for {video_name}\n\n")
        
        for segment in filtered_segments:
            # Format time as MM:SS
            start_time = f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}"
            end_time = f"{int(segment['end']//60):02d}:{int(segment['end']%60):02d}"
            
            # Write segment type prefix with timestamps
            f.write(f"[{start_time}-{end_time}] {segment['type']}:\n")
            
            # Write word-level timestamps
            for word_info in segment['words']:
                word = word_info['word']
                word_start = word_info['start']
                word_end = word_info['end']
                confidence = word_info.get('confidence')
                
                # Format word time as seconds
                f.write(f"{word} [{word_start:.3f}s-{word_end:.3f}s] ")
            
            f.write("\n\n")
    
    print(f"✓ Combined Q&A transcript saved to {combined_transcript_path}")
    return combined_transcript_path

def transcribe_full_track(audio_file, whisper_model, output_file, segment_type="complete_track"):
    """
    Transcribe the full audio track
    
    Args:
        audio_file: Path to the audio file
        whisper_model: Whisper model
        output_file: Path to save the transcribed text
        segment_type: Description of the audio type
    
    Returns:
        bool: Whether transcription was successful
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} does not exist.")
        return False
        
    if os.path.exists(output_file):
        print(f"[SKIP] Transcript already exists at {output_file}")
        return True
        
    try:
        print(f"Transcribing full audio track: {audio_file}")
        # Process and transcribe audio
        audio_data = process_audio(audio_file)
        segments, info = get_transcription(whisper_model, audio_data)
        
        # Only save if there's actual content
        if segments and any(segment.text.strip() for segment in segments):
            save_transcript(output_file, segments, info, segment_type)
            print(f"✓ Transcript saved to {output_file}")
            return True
        else:
            print(f"No speech detected in {audio_file}")
            return False
    except Exception as e:
        print(f"Error transcribing audio file {audio_file}: {e}")
        return False

def process_video(video_path, whisper_model, output_dir_audio, output_dir_transcript, 
                 ignore_talknet=False, only_talknet=False, talknet_output_dir=None, extract_tracks=False,
                 transcribe_full_tracks=False, max_gap_duration=2.0, generate_combined=False,
                 skip_segments=False):
    """
    Process a video to extract and transcribe speaking and non-speaking segments.
    
    Args:
        video_path: Path to the video file
        whisper_model: Loaded Whisper model
        output_dir_audio: Directory to save audio files
        output_dir_transcript: Directory to save transcripts
        ignore_talknet: Skip TalkNet detection and transcribe the entire video
        only_talknet: Only verify TalkNet detection, do not transcribe
        talknet_output_dir: Directory with TalkNet outputs
        extract_tracks: Whether to extract full speaking/non-speaking audio tracks
        transcribe_full_tracks: Whether to transcribe the full audio tracks (instead of segments)
        max_gap_duration: Maximum allowed gap duration (seconds) for merging speaking segments
        generate_combined: Force generation of combined Q&A transcript even if segment transcription is skipped
        skip_segments: Skip processing of individual segments and only handle full tracks
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"Processing video: {video_name}")
    
    # Create output directories for this video
    video_audio_dir = os.path.join(output_dir_audio, video_name)
    video_transcript_dir = os.path.join(output_dir_transcript, video_name)
    os.makedirs(video_audio_dir, exist_ok=True)
    os.makedirs(video_transcript_dir, exist_ok=True)
    
    # Paths for extracted tracks
    speaking_track_path = os.path.join(video_audio_dir, f"{video_name}_speaking_track.wav")
    non_speaking_track_path = os.path.join(video_audio_dir, f"{video_name}_non_speaking_track.wav")
    
    # Paths for full track transcripts
    speaking_transcript_path = os.path.join(video_transcript_dir, f"{video_name}_speaking_track.txt")
    non_speaking_transcript_path = os.path.join(video_transcript_dir, f"{video_name}_non_speaking_track.txt")
    
    # Path for combined Q&A transcript
    combined_transcript_path = os.path.join(video_transcript_dir, f"{video_name}_combined_qa.txt")
    
    # Check if tracks already exist
    tracks_exist = os.path.exists(speaking_track_path) and os.path.exists(non_speaking_track_path)
    
    # Check if full track transcripts already exist
    transcripts_exist = os.path.exists(speaking_transcript_path) and os.path.exists(non_speaking_transcript_path)
    
    # Check if combined transcript already exists
    combined_exists = os.path.exists(combined_transcript_path)
    
    # If tracks exist but extraction is requested
    if extract_tracks and tracks_exist:
        print(f"[SKIP] Full audio tracks already exist for {video_name}. Skipping track extraction.")
        extract_tracks = False
    
    # If only extracting tracks and tracks exist
    if extract_tracks and only_talknet and tracks_exist:
        print(f"[SKIP] Full audio tracks already exist for {video_name}. Skipping.")
        return
    
    # If not processing full tracks and transcripts already exist, skip
    if not transcribe_full_tracks and not extract_tracks and not only_talknet:
        existing_transcripts = list(Path(video_transcript_dir).glob("*.txt"))
        if existing_transcripts and combined_exists and not generate_combined:
            print(f"[SKIP] Found {len(existing_transcripts)} existing transcript files for {video_name}. Skipping.")
            return
    
    # Get video duration
    try:
        duration_result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        ).stdout
        video_duration = float(json.loads(duration_result)["format"]["duration"])
        print(f"Video duration: {video_duration:.2f} seconds")
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        print("Trying alternative method...")
        try:
            # Alternative method to get duration
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file")
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0
            cap.release()
            print(f"Video duration (from OpenCV): {video_duration:.2f} seconds")
        except Exception as inner_e:
            print(f"Error using alternative method: {str(inner_e)}")
            print("Using default duration of 60 seconds")
            video_duration = 60.0
    
    if ignore_talknet:
        # Transcribe the entire video without TalkNet segmentation
        if only_talknet:
            print("Warning: --ignore_talknet and --only_talknet options cannot be used together")
            return
            
        print("Skipping TalkNet detection, transcribing entire video...")
        
        # Extract full audio
        full_audio_path = os.path.join(video_audio_dir, f"{video_name}_full.wav")
        extract_audio_segment(video_path, full_audio_path, 0, video_duration)
        
        # Process and transcribe
        try:
            audio_data = process_audio(full_audio_path)
            segments, info = get_transcription(whisper_model, audio_data)
            
            if segments and any(segment.text.strip() for segment in segments):
                full_transcript_path = os.path.join(video_transcript_dir, f"{video_name}_full.txt")
                save_transcript(full_transcript_path, segments, info, "full_transcription")
                print(f"Full transcription saved to {full_transcript_path}")
            else:
                print("No speech detected in the video")
        except Exception as e:
            print(f"Error transcribing full video: {e}")
        return
    
    # Load speaking segments from TalkNet output
    speaking_segments = load_speaking_segments(video_path, talknet_output_dir)
    
    if not speaking_segments:
        print(f"No speaking segments found in TalkNet results for {video_name}")
        return
    
    # Merge short-gap speaking segments
    original_segment_count = len(speaking_segments)
    speaking_segments = merge_short_gaps(speaking_segments, max_gap_duration)
    
    if len(speaking_segments) != original_segment_count:
        print(f"Original {original_segment_count} segments, merged to {len(speaking_segments)} segments")
    
    # Create non-speaking segments for processing
    non_speaking_segments = []
    last_end = 0
    for start, end in speaking_segments:
        if start > last_end + 0.5:  # At least 0.5 seconds of silence
            non_speaking_segments.append((last_end, start))
        last_end = end
    
    # Add final non-speaking segment if needed
    if last_end < video_duration - 0.5:
        non_speaking_segments.append((last_end, video_duration))
    
    # Extract full speaking/non-speaking audio tracks if needed
    if extract_tracks or transcribe_full_tracks:
        if not tracks_exist:
            print(f"Extracting full speaking/non-speaking audio tracks for {video_name}...")
            success = extract_full_audio_tracks(
                video_path, 
                speaking_segments, 
                video_duration, 
                speaking_track_path, 
                non_speaking_track_path
            )
            if success:
                print(f"✓ Successfully extracted audio tracks:")
                print(f"  - Speaking track: {speaking_track_path}")
                print(f"  - Non-speaking track: {non_speaking_track_path}")
                tracks_exist = True
            else:
                print(f"✗ Failed to extract audio tracks for {video_name}")
                # If only need to transcribe full tracks but extraction failed, exit
                if transcribe_full_tracks and not extract_tracks:
                    return
        else:
            print(f"Using existing audio tracks for {video_name}")
    
    # If only extracting tracks, stop here
    if only_talknet:
        print(f"Only verified TalkNet detection for {video_name}, found {len(speaking_segments)} speaking segments")
        return
    
    # If need to transcribe full tracks
    if transcribe_full_tracks and tracks_exist:
        if not transcripts_exist:
            print(f"Transcribing full audio tracks for {video_name}...")
            
            # Transcribe speaking track
            speaking_success = transcribe_full_track(
                speaking_track_path,
                whisper_model,
                speaking_transcript_path,
                segment_type="speaking_track"
            )
            
            # Transcribe non-speaking track
            non_speaking_success = transcribe_full_track(
                non_speaking_track_path,
                whisper_model,
                non_speaking_transcript_path,
                segment_type="non_speaking_track"
            )
            
            if speaking_success or non_speaking_success:
                print(f"✓ Successfully transcribed audio tracks for {video_name}")
                
                # Generate combined transcript from the full tracks
                if not combined_exists or generate_combined:
                    try:
                        # Use actual speaking segments from TalkNet detection for timeline
                        if speaking_success and os.path.exists(speaking_transcript_path):
                            save_combined_transcript(video_name, video_transcript_dir, speaking_segments, non_speaking_segments, use_full_tracks=True)
                    except Exception as e:
                        print(f"Error generating combined Q&A transcript from tracks: {e}")
                
                # If only want to transcribe full tracks, skip segment processing
                if not extract_tracks or skip_segments:
                    return
            else:
                print(f"No speech content found in tracks for {video_name}")
        else:
            print(f"[SKIP] Full track transcripts already exist for {video_name}")
            
            # Generate combined transcript if it doesn't exist
            if (not combined_exists or generate_combined) and (os.path.exists(speaking_transcript_path) or os.path.exists(non_speaking_transcript_path)):
                try:
                    # Use actual speaking segments from TalkNet detection for timeline
                    save_combined_transcript(video_name, video_transcript_dir, speaking_segments, non_speaking_segments, use_full_tracks=True)
                except Exception as e:
                    print(f"Error generating combined Q&A transcript from existing tracks: {e}")
            
            # If only want to transcribe full tracks, skip segment processing
            if not extract_tracks or skip_segments:
                return
    
    # If not processing segments, stop here
    if transcribe_full_tracks and not extract_tracks or skip_segments:
        return
    
    # Process speaking segments (answers)
    for i, (start, end) in enumerate(speaking_segments):
        # Skip very short segments
        if end - start < 1.0:
            continue
            
        segment_id = f"{video_name}_answer_{i:03d}"
        audio_path = os.path.join(video_audio_dir, f"{segment_id}.wav")
        transcript_path = os.path.join(video_transcript_dir, f"{segment_id}.txt")
        
        # Skip if transcript already exists
        if os.path.exists(transcript_path):
            print(f"[SKIP] Transcript already exists for {segment_id}")
            continue
        
        # Extract audio segment
        extract_audio_segment(video_path, audio_path, start, end)
        
        # Process and transcribe
        try:
            audio_data = process_audio(audio_path)
            segments, info = get_transcription(whisper_model, audio_data)
            
            # Only save if there's actual content
            if segments and any(segment.text.strip() for segment in segments):
                save_transcript(transcript_path, segments, info, "answer")
            else:
                # Clean up empty segments
                os.remove(audio_path)
                if os.path.exists(transcript_path):
                    os.remove(transcript_path)
        except Exception as e:
            print(f"Error processing speaking segment {segment_id}: {e}")
    
    # Process non-speaking segments (potential questions)
    for i, (start, end) in enumerate(non_speaking_segments):
        # Skip very short segments
        if end - start < 1.0:
            continue
            
        segment_id = f"{video_name}_question_{i:03d}"
        audio_path = os.path.join(video_audio_dir, f"{segment_id}.wav")
        transcript_path = os.path.join(video_transcript_dir, f"{segment_id}.txt")
        
        # Skip if transcript already exists
        if os.path.exists(transcript_path):
            print(f"[SKIP] Transcript already exists for {segment_id}")
            continue
        
        # Extract audio segment
        extract_audio_segment(video_path, audio_path, start, end)
        
        # Process and transcribe
        try:
            audio_data = process_audio(audio_path)
            segments, info = get_transcription(whisper_model, audio_data)
            
            # Only save non-speaking segments that have speech (questions)
            if segments and any(segment.text.strip() for segment in segments):
                save_transcript(transcript_path, segments, info, "question")
            else:
                # Clean up segments without speech
                os.remove(audio_path)
                if os.path.exists(transcript_path):
                    os.remove(transcript_path)
        except Exception as e:
            print(f"Error processing non-speaking segment {segment_id}: {e}")
    
    # Always generate combined Q&A transcript at the end
    if not combined_exists or generate_combined:
        try:
            # Use actual speaking and non-speaking segments for timeline-based organization
            save_combined_transcript(video_name, video_transcript_dir, speaking_segments, non_speaking_segments)
        except Exception as e:
            print(f"Error generating combined Q&A transcript: {e}")

def merge_short_gaps(speaking_segments, max_gap_duration=2.0):
    """
    Merge speaking segments if the gap between them is too short.
    
    Args:
        speaking_segments: List of speaking segments, format [(start1, end1), (start2, end2), ...]
        max_gap_duration: Maximum allowed gap duration (seconds). Gaps less than or equal to this value will be merged.
        
    Returns:
        List of merged speaking segments
    """
    if not speaking_segments or len(speaking_segments) < 2:
        return speaking_segments
    
    # Sort by start time
    sorted_segments = sorted(speaking_segments)
    
    # Initialize result list
    merged = [sorted_segments[0]]
    
    # Iterate and merge short gaps
    for current_start, current_end in sorted_segments[1:]:
        prev_start, prev_end = merged[-1]
        
        # Calculate gap between current and previous segment
        gap = current_start - prev_end
        
        # Merge if gap is less than or equal to threshold
        if gap <= max_gap_duration:
            merged[-1] = (prev_start, max(prev_end, current_end))
        else:
            merged.append((current_start, current_end))
    
    # Print info if merging happened
    if len(merged) < len(speaking_segments):
        print(f"Merged {len(speaking_segments) - len(merged)} short-gap segments (threshold: {max_gap_duration:.1f}s)")
        if len(merged) <= 5:
            print("Merged segments:")
            for i, (start, end) in enumerate(merged):
                print(f"  {i+1}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    
    return merged

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process video dataset to extract speaking segments and transcripts")
    parser.add_argument("--video_dir", type=str, default="/simurgh/group/juze/datasets/YouTube_Talking/video_20241226", help="Dataset directory containing videos to process")
    parser.add_argument("--output_dir_audio", type=str, default="/simurgh/group/juze/datasets/YouTube_Talking/audios", help="Output directory for extracted audio segments")
    parser.add_argument("--output_dir_transcript", type=str, default="/simurgh/group/juze/datasets/YouTube_Talking/transcript", help="Output directory for transcripts")
    parser.add_argument("--talknet_output_dir", type=str, default="/simurgh/group/juze/datasets/YouTube_Talking/talknet_output", help="Directory containing TalkNet results")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Whisper model (cuda/cpu)")
    parser.add_argument("--ignore_talknet", action="store_true", help="Skip TalkNet processing, directly transcribe entire video")
    parser.add_argument("--only_talknet", action="store_true", help="Only verify TalkNet results, skip transcription")
    parser.add_argument("--extract_tracks", action="store_true", help="Extract full speaking/non-speaking audio tracks")
    parser.add_argument("--transcribe_full_tracks", action="store_true", help="Transcribe the full audio tracks (instead of segments)")
    parser.add_argument("--max_gap_duration", type=float, default=1, help="Maximum allowed gap duration (seconds) for merging speaking segments (default: 2.0)")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of videos to process in parallel (default: 1)")
    parser.add_argument("--whisper_model", type=str, default="large", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--generate_combined", action="store_true", help="Force generation of combined Q&A transcript even if segments transcription is skipped")
    parser.add_argument("--skip_segments", action="store_true", help="Skip processing of individual segments, only extract and transcribe full tracks")
    
    args = parser.parse_args()
    
    # Set output directories
    output_dir_audio = args.output_dir_audio
    output_dir_transcript = args.output_dir_transcript
    talknet_output_dir = args.talknet_output_dir
    
    # Create output directories
    os.makedirs(output_dir_audio, exist_ok=True)
    os.makedirs(output_dir_transcript, exist_ok=True)
    
    # Verify TalkNet output directory exists
    if not os.path.exists(talknet_output_dir) and not args.ignore_talknet:
        print(f"Error: TalkNet output directory {talknet_output_dir} does not exist.")
        print("Please run process_talknet.py first or use --ignore_talknet option.")
        return
    
    # Ensure parameter combination is valid
    if args.transcribe_full_tracks and not args.extract_tracks and not os.path.exists(output_dir_audio):
        print(f"Warning: --transcribe_full_tracks requires audio tracks, but --extract_tracks was not specified.")
        print(f"Automatically enabling --extract_tracks")
        args.extract_tracks = True
    
    # Load Whisper model if not only verifying TalkNet or only extracting tracks
    whisper_model = None
    if not args.only_talknet and not (args.extract_tracks and args.only_talknet and not args.transcribe_full_tracks):
        print(f"Loading Whisper model ({args.whisper_model}) on {args.device}...")
        try:
            # Use device from command line parameter
            whisper_model = WhisperModel(args.whisper_model, device=args.device, compute_type="float32")
            print(f"Whisper model loaded successfully on {args.device}")
        except Exception as e:
            print(f"Error: Cannot load Whisper model on {args.device}: {e}")
            if args.device == "cuda":
                print("Trying to fall back to CPU...")
                try:
                    whisper_model = WhisperModel(args.whisper_model, device="cpu", compute_type="float32")
                    print("Whisper model loaded successfully on CPU")
                except Exception as e2:
                    print(f"Error: Cannot load Whisper model on CPU either: {e2}")
                    print("Try using a smaller model with --whisper_model=tiny or --whisper_model=base")
                    if not args.only_talknet and not args.extract_tracks:
                        print("Consider using --only_talknet option to only verify speech detection without transcription")
                        print("Or use --extract_tracks --only_talknet to only extract audio tracks without transcription")
                        return
                    else:
                        print("Continuing without Whisper model since only TalkNet verification or track extraction was requested")
            else:
                print("Try using a smaller model with --whisper_model=tiny or --whisper_model=base")
                if not args.only_talknet and not args.extract_tracks:
                    print("Consider using --only_talknet option to only verify speech detection without transcription")
                    print("Or use --extract_tracks --only_talknet to only extract audio tracks without transcription")
                    return
                else:
                    print("Continuing without Whisper model since only TalkNet verification or track extraction was requested")

    # Process all videos in the dataset directory
    video_dir = args.video_dir
    print(f"Processing dataset at: {video_dir}")
    print(f"Using TalkNet results from: {talknet_output_dir}")
    print(f"Merging gap threshold: {args.max_gap_duration:.1f} seconds (speaking segments with gaps less than or equal to this value will be merged)")
    
    # Get all video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(Path(video_dir).glob(f"**/*{ext}")))  # Using ** to search subdirectories recursively
    
    print(f"Found {len(video_files)} video files in dataset")
    
    if len(video_files) == 0:
        print(f"No video files found in {video_dir}. Please check the directory path.")
        return
    
    # Process each video
    for video_path in tqdm(video_files, desc="Processing dataset videos"):
        try:
            process_video(str(video_path), whisper_model, output_dir_audio, output_dir_transcript,
                         ignore_talknet=args.ignore_talknet, 
                         only_talknet=args.only_talknet,
                         talknet_output_dir=talknet_output_dir,
                         extract_tracks=args.extract_tracks,
                         transcribe_full_tracks=args.transcribe_full_tracks,
                         max_gap_duration=args.max_gap_duration,
                         generate_combined=args.generate_combined,
                         skip_segments=args.skip_segments)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
    
    print("\nDataset processing complete.")
    print(f"Audio segments and tracks saved to {output_dir_audio}")
    if not args.only_talknet:
        print(f"Transcripts saved to {output_dir_transcript}")
    else:
        print("No transcription was performed (--only_talknet option was used)")

if __name__ == "__main__":
    main()
