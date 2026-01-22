#!/usr/bin/env python3
"""
Calculate total duration of npz files in the CANDOR FLAME_coeffs dataset.
Files are at 25 fps (typical for FLAME data).
"""

import os
import glob
import numpy as np
from tqdm import tqdm


def calculate_npz_dataset_duration(data_root, fps=25):
    """Calculate total duration of all npz files in the dataset."""
    # Find all npz files recursively (files are in subdirectories)
    pattern = os.path.join(data_root, "*", "*.npz")
    npz_files = glob.glob(pattern)
    
    print(f"Found {len(npz_files)} npz files")
    
    total_frames = 0
    total_duration = 0.0
    failed_files = []
    
    # Process each file with progress bar
    for npz_file in tqdm(npz_files, desc="Calculating durations"):
        try:
            data = np.load(npz_file)
            
            # Get frame count from the first dimension of any array
            # Check common keys: exp, shape, pose
            frame_count = None
            for key in ['exp', 'shape', 'pose']:
                if key in data:
                    if hasattr(data[key], 'shape') and len(data[key].shape) > 0:
                        frame_count = data[key].shape[0]
                        break
            
            if frame_count is None:
                raise ValueError("Could not determine frame count from file")
            
            duration = frame_count / fps
            total_frames += frame_count
            total_duration += duration
            
            data.close()
        except Exception as e:
            failed_files.append((npz_file, str(e)))
            if len(failed_files) <= 10:  # Only print first 10 errors
                print(f"\nWarning: Failed to process {npz_file}: {e}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Total npz files: {len(npz_files)}")
    print(f"Successfully processed: {len(npz_files) - len(failed_files)}")
    print(f"Failed files: {len(failed_files)}")
    print(f"\nFrame rate: {fps} fps")
    print(f"Total frames: {total_frames:,}")
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
    data_root = "/simurgh/u/juze/datasets/CANDOR/FLAME_coeffs"
    # Resolve symlink to actual path
    actual_path = os.path.realpath(data_root)
    print(f"Data root (resolved): {actual_path}")
    calculate_npz_dataset_duration(actual_path, fps=25)


