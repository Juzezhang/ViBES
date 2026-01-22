import os
import re
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import tempfile


def parse_video_filename(filename):
    """
    Parse video filename to extract components
    Expected format: TH_00003_000_000.mp4
    Returns: (base_name, segment_index) where base_name is TH_00003_000 and segment_index is 000
    """
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    if len(parts) >= 4:
        # Join first 3 parts as base name, last part as segment index
        base_name = '_'.join(parts[:-1])
        segment_index = parts[-1]
        return base_name, segment_index
    else:
        # Fallback: try to extract pattern with regex
        match = re.match(r'(.+)_(\d+)$', name_without_ext)
        if match:
            return match.group(1), match.group(2)
        else:
            print(f"Warning: Could not parse filename {filename}")
            return None, None


def get_video_groups(input_dir):
    """
    Group video files by their base name
    """
    groups = defaultdict(list)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            base_name, segment_index = parse_video_filename(filename)
            if base_name and segment_index:
                full_path = os.path.join(input_dir, filename)
                groups[base_name].append((segment_index, full_path))
    
    # Sort each group by segment index
    for base_name in groups:
        groups[base_name].sort(key=lambda x: int(x[0]))
    
    return groups


def merge_videos_with_ffmpeg(video_list, output_path):
    """
    Merge videos using ffmpeg
    """
    # Create a temporary file list for ffmpeg
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for video_path in video_list:
            f.write(f"file '{video_path}'\n")
        temp_file = f.name
    
    try:
        # Use ffmpeg to concatenate videos
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_file,
            '-c', 'copy',
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        else:
            print(f"Successfully merged to: {output_path}")
            return True
            
    except Exception as e:
        print(f"Error running ffmpeg: {e}")
        return False
    finally:
        # Clean up temporary file
        os.unlink(temp_file)


def main():
    parser = argparse.ArgumentParser(description='Merge video segments into complete videos')
    parser.add_argument('--input_dir', '-i', required=True, help='Input directory containing video segments')
    parser.add_argument('--output_dir', '-o', required=True, help='Output directory for merged videos')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be done without actually merging')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video groups
    print(f"Scanning for video segments in: {input_dir}")
    groups = get_video_groups(str(input_dir))
    
    if not groups:
        print("No video segments found!")
        return
    
    print(f"Found {len(groups)} video groups:")
    
    # Process each group
    for base_name, segments in groups.items():
        print(f"\nGroup: {base_name}")
        print(f"  Segments: {len(segments)}")
        
        # Show segment details
        for segment_index, video_path in segments:
            print(f"    {segment_index}: {os.path.basename(video_path)}")
        
        # Create output filename
        output_filename = f"{base_name}.mp4"
        output_path = output_dir / output_filename
        
        if args.dry_run:
            print(f"  Would merge to: {output_path}")
        else:
            # Extract video paths
            video_paths = [video_path for _, video_path in segments]
            
            # Merge videos
            print(f"  Merging to: {output_path}")
            success = merge_videos_with_ffmpeg(video_paths, str(output_path))
            
            if success:
                print(f"  ✓ Successfully merged {len(segments)} segments")
            else:
                print(f"  ✗ Failed to merge segments")
    
    print(f"\nMerging completed. Output files saved to: {output_dir}")


if __name__ == "__main__":
    main() 