#!/usr/bin/env python3
"""
Fast video comparison using ffmpeg command line
Much faster than MoviePy for video processing
"""
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
import codecs as cs
import json

def find_common_videos(lom_dir, ours_layer5_dir, ours_layer40_dir, motiongpt_dir, momask_dir):
    """
    Find video IDs that exist in at least 2 of the 5 directories
    Note: For ours videos, both layer5 and layer40 must exist
    """
    # Find all mp4 files in ours layer5 directory
    ours_layer5_videos = set()
    for mp4_file in Path(ours_layer5_dir).glob("*.mp4"):
        video_id = mp4_file.stem
        ours_layer5_videos.add(video_id)
    
    # Find all mp4 files in ours layer40 directory
    ours_layer40_videos = set()
    for mp4_file in Path(ours_layer40_dir).glob("*.mp4"):
        video_id = mp4_file.stem
        ours_layer40_videos.add(video_id)
    
    # Find all mp4 files in MotionGPT directory
    motiongpt_videos = set()
    for mp4_file in Path(motiongpt_dir).glob("*.mp4"):
        video_id = mp4_file.stem
        motiongpt_videos.add(video_id)
    
    # Find all mp4 files in LOM directory (directly in the folder)
    lom_videos = set()
    for mp4_file in Path(lom_dir).glob("*.mp4"):
        video_id = mp4_file.stem
        lom_videos.add(video_id)
    
    # Find all mp4 files in MoMask directory (ignore files ending with 'ik')
    momask_videos = set()
    for mp4_file in Path(momask_dir).glob("*.mp4"):
        video_id = mp4_file.stem
        # Ignore files ending with 'ik'
        if not video_id.endswith('ik'):
            momask_videos.add(video_id)
    
    # Ours videos: both layer5 AND layer40 must exist
    ours_videos = ours_layer5_videos & ours_layer40_videos
    
    # Find IDs that exist in at least 2 directories (counting ours as one)
    all_ids = ours_videos | motiongpt_videos | lom_videos | momask_videos
    valid_ids = set()
    
    for vid in all_ids:
        count = sum([
            vid in ours_videos,  # Both layer5 and layer40
            vid in motiongpt_videos,
            vid in lom_videos,
            vid in momask_videos
        ])
        if count >= 2:
            valid_ids.add(vid)
    
    print(f"Found {len(ours_layer5_videos)} videos in Ours Layer5")
    print(f"Found {len(ours_layer40_videos)} videos in Ours Layer40")
    print(f"Found {len(ours_videos)} videos in Ours (both layers)")
    print(f"Found {len(motiongpt_videos)} videos in MotionGPT")
    print(f"Found {len(lom_videos)} videos in LOM")
    print(f"Found {len(momask_videos)} videos in MoMask")
    print(f"Found {len(valid_ids)} videos with at least 2 sources")
    
    return sorted(list(valid_ids))

def get_text_description(texts_dir, video_id):
    """
    Read the first line of text description from HumanML3D texts directory
    """
    text_file = Path(texts_dir) / f"{video_id}.txt"
    
    if not text_file.exists():
        return "No description available"
    
    try:
        with cs.open(text_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Extract caption before the first '#' if exists
            if '#' in first_line:
                caption = first_line.split('#')[0].strip()
            else:
                caption = first_line
            return caption
    except Exception as e:
        print(f"Error reading text for {video_id}: {e}")
        return "Error reading description"

def get_video_info(video_path):
    """
    Get video duration and dimensions using ffprobe
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration',
        '-of', 'json',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info['streams'][0]
        width = int(stream['width'])
        height = int(stream['height'])
        duration = float(stream.get('duration', 0))
        return width, height, duration
    except Exception as e:
        print(f"Error getting video info for {video_path}: {e}")
        return None, None, None

def concatenate_videos_ffmpeg(lom_path, ours_layer5_path, ours_layer40_path, motiongpt_path, 
                              momask_path, text_description, output_path, max_duration=20.0):
    """
    Concatenate five videos horizontally using ffmpeg (much faster!)
    Supports missing videos by filling with blank space
    """
    try:
        # Check which videos exist
        lom_exists = lom_path.exists()
        ours_layer5_exists = ours_layer5_path.exists()
        ours_layer40_exists = ours_layer40_path.exists()
        mgpt_exists = motiongpt_path.exists()
        momask_exists = momask_path.exists()
        
        # Need at least 2 videos (and if ours exists, both layers must exist)
        exists_count = sum([lom_exists, (ours_layer5_exists and ours_layer40_exists), mgpt_exists, momask_exists])
        if exists_count < 2:
            print(f"Not enough videos (need at least 2)")
            return False
        
        # Get video info for existing videos
        lom_w, lom_h, lom_dur = get_video_info(lom_path) if lom_exists else (None, None, 0)
        ours_layer5_w, ours_layer5_h, ours_layer5_dur = get_video_info(ours_layer5_path) if ours_layer5_exists else (None, None, 0)
        ours_layer40_w, ours_layer40_h, ours_layer40_dur = get_video_info(ours_layer40_path) if ours_layer40_exists else (None, None, 0)
        mgpt_w, mgpt_h, mgpt_dur = get_video_info(motiongpt_path) if mgpt_exists else (None, None, 0)
        momask_w, momask_h, momask_dur = get_video_info(momask_path) if momask_exists else (None, None, 0)
        
        # Calculate target dimensions from existing videos
        valid_widths = [w for w in [lom_w, ours_layer5_w, ours_layer40_w, mgpt_w, momask_w] if w is not None]
        valid_heights = [h for h in [lom_h, ours_layer5_h, ours_layer40_h, mgpt_h, momask_h] if h is not None]
        
        if not valid_widths or not valid_heights:
            print(f"Failed to get video info from existing videos")
            return False
        
        target_h = max(valid_heights)
        target_w = max(valid_widths)
        
        # Calculate target duration (min of max duration or longest video)
        target_dur = min(max(lom_dur, ours_layer5_dur, ours_layer40_dur, mgpt_dur, momask_dur), max_duration)
        
        # Label height and text height
        label_h = 50
        text_h = 80
        
        # Final dimensions (5 videos side by side)
        final_w = target_w * 5
        final_h = target_h + label_h + text_h
        
        # Escape text for ffmpeg
        text_escaped = text_description.replace("'", "'\\''").replace(":", "\\:")
        if len(text_escaped) > 150:
            text_escaped = text_escaped[:150] + "..."
        
        # Build complex ffmpeg filter
        # Step 1: Scale all videos to same size and trim to target duration
        # If video doesn't exist, create a blank video
        filter_parts = []
        input_idx = 0
        
        # Process LOM video
        if lom_exists:
            filter_parts.append(f"[{input_idx}:v]scale={target_w}:{target_h},trim=duration={target_dur},setpts=PTS-STARTPTS[v0];")
            input_idx += 1
        else:
            # Create blank video
            filter_parts.append(f"color=c=black:s={target_w}x{target_h}:d={target_dur}:r=30[v0];")
        
        # Process Ours Layer5 video
        if ours_layer5_exists:
            filter_parts.append(f"[{input_idx}:v]scale={target_w}:{target_h},trim=duration={target_dur},setpts=PTS-STARTPTS[v1];")
            input_idx += 1
        else:
            filter_parts.append(f"color=c=black:s={target_w}x{target_h}:d={target_dur}:r=30[v1];")
        
        # Process Ours Layer40 video
        if ours_layer40_exists:
            filter_parts.append(f"[{input_idx}:v]scale={target_w}:{target_h},trim=duration={target_dur},setpts=PTS-STARTPTS[v2];")
            input_idx += 1
        else:
            filter_parts.append(f"color=c=black:s={target_w}x{target_h}:d={target_dur}:r=30[v2];")
        
        # Process MotionGPT video
        if mgpt_exists:
            filter_parts.append(f"[{input_idx}:v]scale={target_w}:{target_h},trim=duration={target_dur},setpts=PTS-STARTPTS[v3];")
            input_idx += 1
        else:
            filter_parts.append(f"color=c=black:s={target_w}x{target_h}:d={target_dur}:r=30[v3];")
        
        # Process MoMask video
        if momask_exists:
            filter_parts.append(f"[{input_idx}:v]scale={target_w}:{target_h},trim=duration={target_dur},setpts=PTS-STARTPTS[v4];")
            input_idx += 1
        else:
            filter_parts.append(f"color=c=black:s={target_w}x{target_h}:d={target_dur}:r=30[v4];")
        
        # Step 2: Add labels on top of each video
        lom_label = 'LOM' if lom_exists else 'LOM (missing)'
        ours_layer5_label = 'Ours-L5' if ours_layer5_exists else 'Ours-L5 (missing)'
        ours_layer40_label = 'Ours-L40' if ours_layer40_exists else 'Ours-L40 (missing)'
        mgpt_label = 'MotionGPT' if mgpt_exists else 'MotionGPT (missing)'
        momask_label = 'MoMask' if momask_exists else 'MoMask (missing)'
        
        filter_parts.extend([
            f"[v0]drawtext=text='{lom_label}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=30:fontcolor=white:box=1:boxcolor=black@1.0:boxborderw=5:x=(w-text_w)/2:y=10[v0l];",
            f"[v1]drawtext=text='{ours_layer5_label}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=30:fontcolor=white:box=1:boxcolor=black@1.0:boxborderw=5:x=(w-text_w)/2:y=10[v1l];",
            f"[v2]drawtext=text='{ours_layer40_label}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=30:fontcolor=white:box=1:boxcolor=black@1.0:boxborderw=5:x=(w-text_w)/2:y=10[v2l];",
            f"[v3]drawtext=text='{mgpt_label}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=30:fontcolor=white:box=1:boxcolor=black@1.0:boxborderw=5:x=(w-text_w)/2:y=10[v3l];",
            f"[v4]drawtext=text='{momask_label}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=30:fontcolor=white:box=1:boxcolor=black@1.0:boxborderw=5:x=(w-text_w)/2:y=10[v4l];",
            
            # Step 3: Stack horizontally (5 videos)
            "[v0l][v1l][v2l][v3l][v4l]hstack=inputs=5[stacked];",
            
            # Step 4: Add text description at bottom
            f"[stacked]pad=w={final_w}:h={final_h}:x=0:y=0:color=black[padded];",
            f"[padded]drawtext=text='{text_escaped}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=50:fontcolor=white:x=(w-text_w)/2:y=h-{text_h//2}-th/2[out]"
        ])
        
        filter_complex = "".join(filter_parts)
        
        # Build ffmpeg command - only add existing video inputs
        cmd = ['ffmpeg', '-y']  # Overwrite output
        
        # Add input videos (only the ones that exist)
        if lom_exists:
            cmd.extend(['-i', str(lom_path)])
        if ours_layer5_exists:
            cmd.extend(['-i', str(ours_layer5_path)])
        if ours_layer40_exists:
            cmd.extend(['-i', str(ours_layer40_path)])
        if mgpt_exists:
            cmd.extend(['-i', str(motiongpt_path)])
        if momask_exists:
            cmd.extend(['-i', str(momask_path)])
        
        # Add filter and output options
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-r', '30',  # 30 fps
            '-t', str(target_dur),
            '-pix_fmt', 'yuv420p',
            '-an',  # No audio
            str(output_path)
        ])
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error creating comparison video: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fast video comparison using ffmpeg')
    parser.add_argument('--max_duration', type=float, default=20.0,
                       help='Maximum duration in seconds (default: 20.0)')
    args = parser.parse_args()
    
    # Paths
    lom_dir = "/path/to/result/lom"
    ours_layer5_dir = "/path/to/conversational_agent/demo/t2m_qualityresult_layer5"
    ours_layer40_dir = "/path/to/conversational_agent/demo/t2m_qualityresult_layer40"
    motiongpt_dir = "/path/to/MotionGPT/output"
    momask_dir = "/path/to/momask-codes/generation/momask/animations"
    texts_dir = "/path/to/HumanML3D/texts"
    output_dir = "/path/to/conversational_agent/demo/comparison_videos_fast"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Max duration per video: {args.max_duration} seconds")
    print(f"Using ffmpeg for fast processing\n")
    
    # Find common videos
    common_ids = find_common_videos(lom_dir, ours_layer5_dir, ours_layer40_dir, motiongpt_dir, momask_dir)
    
    if len(common_ids) == 0:
        print("No common videos found!")
        return
    
    print(f"\nProcessing {len(common_ids)} videos...")
    
    # Process each common video
    success_count = 0
    for video_id in tqdm(common_ids, desc="Creating comparison videos"):
        # Construct paths
        lom_path = Path(lom_dir) / f"{video_id}.mp4"
        ours_layer5_path = Path(ours_layer5_dir) / f"{video_id}.mp4"
        ours_layer40_path = Path(ours_layer40_dir) / f"{video_id}.mp4"
        motiongpt_path = Path(motiongpt_dir) / f"{video_id}.mp4"
        momask_path = Path(momask_dir) / f"{video_id}.mp4"
        output_path = Path(output_dir) / f"{video_id}_comparison.mp4"
        
        # Skip if output already exists
        if output_path.exists():
            # print(f"Skipping {video_id} (already exists)")
            success_count += 1
            continue
        
        # Check that at least 2 sources exist (ours requires both layers)
        ours_exists = ours_layer5_path.exists() and ours_layer40_path.exists()
        exists_count = sum([lom_path.exists(), ours_exists, motiongpt_path.exists(), momask_path.exists()])
        if exists_count < 2:
            print(f"\nSkipping {video_id}: only {exists_count} source(s) found")
            continue
        
        # Get text description
        text_description = get_text_description(texts_dir, video_id)
        
        # Create comparison video using ffmpeg
        success = concatenate_videos_ffmpeg(
            lom_path,
            ours_layer5_path,
            ours_layer40_path,
            motiongpt_path,
            momask_path,
            text_description,
            output_path,
            max_duration=args.max_duration
        )
        
        if success:
            success_count += 1
        else:
            print(f"\n✗ Failed: {video_id}")
    
    print(f"\n{'='*60}")
    print(f"Completed! Successfully created {success_count}/{len(common_ids)} comparison videos")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()

