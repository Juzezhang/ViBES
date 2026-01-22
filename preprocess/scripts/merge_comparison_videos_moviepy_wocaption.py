#!/usr/bin/env python3
"""
Alternative version using MoviePy to merge videos with frame numbers.
This version is slower but more compatible and easier to debug.
"""
import os
from pathlib import Path
from tqdm import tqdm

try:
    from moviepy.editor import VideoFileClip, CompositeVideoClip, clips_array, ImageClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not available. Install with: pip install moviepy")

import numpy as np


def find_common_videos(directories):
    """Find video IDs that exist in all provided directories."""
    if not directories:
        return []
    
    first_dir = Path(directories[0])
    all_video_ids = set()
    for mp4_file in first_dir.glob("*.mp4"):
        video_id = mp4_file.stem
        all_video_ids.add(video_id)
    
    common_video_ids = []
    for video_id in all_video_ids:
        exists_in_all = True
        for directory in directories[1:]:
            video_path = Path(directory) / f"{video_id}.mp4"
            if not video_path.exists():
                exists_in_all = False
                break
        
        if exists_in_all:
            common_video_ids.append(video_id)
    
    return sorted(common_video_ids)


def merge_videos_with_frame_numbers_moviepy(video_paths, audio_path, output_path):
    """
    Merge videos horizontally without any text overlay using MoviePy.
    
    Args:
        video_paths: List of video file paths to merge
        audio_path: Path to video file to extract audio from
        output_path: Output video path
    """
    if not MOVIEPY_AVAILABLE:
        print("MoviePy not available!")
        return False
    
    try:
        # Check if all videos exist
        existing_videos = []
        for video_path in video_paths:
            if Path(video_path).exists():
                existing_videos.append(video_path)
        
        if len(existing_videos) == 0:
            print("No videos to merge!")
            return False
        
        # Load video clips
        video_clips = []
        for video_path in existing_videos:
            clip = VideoFileClip(str(video_path))
            video_clips.append(clip)
        
        # Get minimum duration
        min_duration = min(clip.duration for clip in video_clips)
        
        # Resize clips to same size (use first clip's dimensions)
        target_size = video_clips[0].size
        resized_clips = [clip.resize(target_size) for clip in video_clips]
        
        # Prepare clips for merging (no text overlay)
        fps = video_clips[0].fps
        clips_for_merge = []
        
        for clip in resized_clips:
            # Simply subclip to minimum duration without adding any text
            clip_sub = clip.subclip(0, min_duration)
            clips_for_merge.append(clip_sub)
        
        # Stack horizontally
        final_video = clips_array([[clip for clip in clips_for_merge]])
        
        # Add audio from first video
        if Path(audio_path).exists():
            audio_clip = VideoFileClip(str(audio_path)).audio
            if audio_clip is not None:
                audio_clip = audio_clip.subclip(0, min_duration)
                final_video = final_video.set_audio(audio_clip)
        
        # Write output
        final_video.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            preset='medium'
        )
        
        # Close all clips
        for clip in video_clips:
            clip.close()
        for clip in clips_for_merge:
            clip.close()
        final_video.close()
        
        return True
        
    except Exception as e:
        print(f"Error merging videos with MoviePy: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Define directories
    # directories = [
    #     "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/ours_wo_head_pose_layernum_40",
    #     "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/artalk_wo_head_pose",
    #     "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/diffposetalk_wo_head_pose",
    #     "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/scantalk_wo_head_pose",
    #     "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/multitalk_wo_head_pose",
    # ]
    
    # output_dir = "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/compare_layernum_40_picked"
    directories = [
        "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/ours_with_head_pose_layernum_40",
        "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/artalk",
        "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/diffposetalk",
    ]
    output_dir = "/simurgh/u/juze/code/conversational_agent/paper_result/a2f/compare_layernum_40_with_head_pose_picked"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if not MOVIEPY_AVAILABLE:
        print("Error: MoviePy is not installed. Please install it first:")
        print("  pip install moviepy")
        return
    
    print("Finding common videos across all directories...")
    
    # Find common videos
    common_video_ids = find_common_videos(directories)
    
    if len(common_video_ids) == 0:
        print("No common videos found!")
        return
    
    print(f"Found {len(common_video_ids)} common videos")
    print(f"Output directory: {output_dir}\n")
    
    # Process each common video
    success_count = 0
    for video_id in tqdm(common_video_ids, desc="Merging videos"):

        if video_id not in ["TH_00104_000_000","TH_00314_000_009","TH_00314_001_001", "TH_00314_007_007", "TH_00314_006_004"]:
            continue
        # Construct paths
        video_paths = [Path(directory) / f"{video_id}.mp4" for directory in directories]
        audio_path = video_paths[0]  # Use audio from first directory
        output_path = Path(output_dir) / f"{video_id}.mp4"
        
        # Skip if output already exists
        if output_path.exists():
            success_count += 1
            continue
        
        # Merge videos
        success = merge_videos_with_frame_numbers_moviepy(
            video_paths,
            audio_path,
            output_path
        )
        
        if success:
            success_count += 1
        else:
            print(f"\n✗ Failed: {video_id}")
    
    print(f"\n{'='*60}")
    print(f"Completed! Successfully created {success_count}/{len(common_video_ids)} merged videos")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

