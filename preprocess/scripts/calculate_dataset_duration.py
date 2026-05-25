#!/usr/bin/env python3
"""
Calculate total duration of audio files in the aiagent dataset.
Audio files are located at: data_root/c--*/**/audio_separated/*.wav
"""

import os
import glob
import soundfile as sf
from tqdm import tqdm


def calculate_dataset_duration(data_root):
    """Calculate total duration of all audio files in the dataset."""
    # Find all wav files matching the pattern
    pattern = os.path.join(data_root, "c--*", "*", "audio_separated", "*.wav")
    wav_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(wav_files)} audio files")
    
    total_duration = 0.0
    failed_files = []
    
    # Process each file with progress bar
    for wav_file in tqdm(wav_files, desc="Calculating durations"):
        try:
            # Get file info without loading the entire audio
            info = sf.info(wav_file)
            duration = info.duration
            total_duration += duration
        except Exception as e:
            failed_files.append((wav_file, str(e)))
            if len(failed_files) <= 10:  # Only print first 10 errors
                print(f"\nWarning: Failed to process {wav_file}: {e}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Total audio files: {len(wav_files)}")
    print(f"Successfully processed: {len(wav_files) - len(failed_files)}")
    print(f"Failed files: {len(failed_files)}")
    print(f"\nTotal duration: {total_duration:.2f} seconds")
    print(f"Total duration: {total_duration/60:.2f} minutes")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"{'='*60}")
    
    if failed_files:
        print("\nFailed files:")
        for file_path, error in failed_files[:10]:  # Show first 10 failures
            print(f"  {file_path}: {error}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    return total_duration


if __name__ == "__main__":
    data_root = "/path/to/embody_3d/datasets/aiagent"
    calculate_dataset_duration(data_root)

