#!/usr/bin/env python3
"""
Compare videos from three different methods: LOM, Ours, and MotionGPT
Concatenate them horizontally with labels and text descriptions
"""
import os
import glob
from pathlib import Path
import moviepy.editor as mp
from tqdm import tqdm
import codecs as cs
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def find_common_videos(lom_dir, ours_dir, motiongpt_dir):
    """
    Find video IDs that exist in all three directories
    """
    # Find all mp4 files in ours directory
    ours_videos = set()
    for mp4_file in Path(ours_dir).glob("*.mp4"):
        video_id = mp4_file.stem
        ours_videos.add(video_id)
    
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
    
    # Find common IDs
    common_ids = ours_videos & motiongpt_videos & lom_videos
    
    print(f"Found {len(ours_videos)} videos in Ours")
    print(f"Found {len(motiongpt_videos)} videos in MotionGPT")
    print(f"Found {len(lom_videos)} videos in LOM")
    print(f"Found {len(common_ids)} common videos")
    
    return sorted(list(common_ids))

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

def create_text_image(text, width, height, fontsize=40, text_color='white', bg_color='black'):
    """
    Create a text image using PIL
    """
    # Create image
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fontsize)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", fontsize)
        except:
            font = ImageFont.load_default()
    
    # Get text bbox to center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center text
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, font=font, fill=text_color)
    
    return np.array(img)

def concatenate_videos(lom_path, ours_path, motiongpt_path, text_description, output_path, max_duration=20.0):
    """
    Concatenate three videos horizontally with labels and text description at bottom
    Args:
        max_duration: Maximum duration in seconds (default 20s), longer videos will be truncated
    """
    try:
        # Load videos
        lom_video = mp.VideoFileClip(str(lom_path))
        ours_video = mp.VideoFileClip(str(ours_path))
        motiongpt_video = mp.VideoFileClip(str(motiongpt_path))
        
        # Get dimensions and duration - use MAXIMUM duration as baseline, but cap at max_duration
        duration = min(max(lom_video.duration, ours_video.duration, motiongpt_video.duration), max_duration)
        video_height = max(lom_video.h, ours_video.h, motiongpt_video.h)
        video_width = max(lom_video.w, ours_video.w, motiongpt_video.w)
        
        # Resize all videos to the same size
        lom_video = lom_video.resize(height=video_height, width=video_width)
        ours_video = ours_video.resize(height=video_height, width=video_width)
        motiongpt_video = motiongpt_video.resize(height=video_height, width=video_width)
        
        # Truncate or loop videos to match the target duration
        if lom_video.duration > duration:
            lom_video = lom_video.subclip(0, duration)
        elif lom_video.duration < duration:
            lom_video = lom_video.loop(duration=duration)
            
        if ours_video.duration > duration:
            ours_video = ours_video.subclip(0, duration)
        elif ours_video.duration < duration:
            ours_video = ours_video.loop(duration=duration)
            
        if motiongpt_video.duration > duration:
            motiongpt_video = motiongpt_video.subclip(0, duration)
        elif motiongpt_video.duration < duration:
            motiongpt_video = motiongpt_video.loop(duration=duration)
        
        # Create label images
        label_height = 50
        lom_label_img = create_text_image("LOM", video_width, label_height, fontsize=30)
        ours_label_img = create_text_image("Ours", video_width, label_height, fontsize=30)
        motiongpt_label_img = create_text_image("MotionGPT", video_width, label_height, fontsize=30)
        
        # Convert to clips
        lom_label = mp.ImageClip(lom_label_img).set_duration(duration)
        ours_label = mp.ImageClip(ours_label_img).set_duration(duration)
        motiongpt_label = mp.ImageClip(motiongpt_label_img).set_duration(duration)
        
        # Stack labels on top of videos
        lom_with_label = mp.clips_array([[lom_label], [lom_video]])
        ours_with_label = mp.clips_array([[ours_label], [ours_video]])
        motiongpt_with_label = mp.clips_array([[motiongpt_label], [motiongpt_video]])
        
        # Concatenate horizontally
        combined_video = mp.clips_array([[lom_with_label, ours_with_label, motiongpt_with_label]])
        
        # Create text description clip at the bottom
        total_width = video_width * 3
        text_height = 80
        
        # Wrap long text
        if len(text_description) > 150:
            text_description = text_description[:150] + "..."
        
        text_img = create_text_image(
            text_description,
            total_width,
            text_height,
            fontsize=20,
            text_color='white',
            bg_color='black'
        )
        text_clip = mp.ImageClip(text_img).set_duration(duration)
        
        # Stack description at bottom
        final_video = mp.clips_array([[combined_video], [text_clip]])
        
        # Write output
        final_video.write_videofile(
            str(output_path),
            codec='libx264',
            audio=False,
            fps=30
        )
        
        # Close all clips
        lom_video.close()
        ours_video.close()
        motiongpt_video.close()
        combined_video.close()
        final_video.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating comparison video: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare videos from LOM, Ours, and MotionGPT')
    parser.add_argument('--max_duration', type=float, default=20.0,
                       help='Maximum duration in seconds (default: 20.0)')
    args = parser.parse_args()
    
    # Paths
    lom_dir = "/simurgh/group/juze/result/lom"
    ours_dir = "/simurgh/u/juze/code/conversational_agent/demo/t2m_qualityresult_layer5"
    motiongpt_dir = "/simurgh/u/juze/code/MotionGPT/output"
    texts_dir = "/simurgh/u/juze/datasets/HumanML3D/texts"
    output_dir = "/simurgh/u/juze/code/conversational_agent/demo/comparison_videos"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Max duration per video: {args.max_duration} seconds")
    
    # Find common videos
    common_ids = find_common_videos(lom_dir, ours_dir, motiongpt_dir)
    
    if len(common_ids) == 0:
        print("No common videos found!")
        return
    
    print(f"\nProcessing {len(common_ids)} videos...")
    
    # Process each common video
    success_count = 0
    for video_id in tqdm(common_ids, desc="Creating comparison videos"):
        # Construct paths
        lom_path = Path(lom_dir) / f"{video_id}.mp4"
        ours_path = Path(ours_dir) / f"{video_id}.mp4"
        motiongpt_path = Path(motiongpt_dir) / f"{video_id}.mp4"
        output_path = Path(output_dir) / f"{video_id}_comparison.mp4"
        
        # Skip if output already exists
        if output_path.exists():
            print(f"\nSkipping {video_id} (already exists)")
            success_count += 1
            continue
        
        # Verify all paths exist
        if not lom_path.exists():
            print(f"\nWarning: LOM video not found: {lom_path}")
            continue
        if not ours_path.exists():
            print(f"\nWarning: Ours video not found: {ours_path}")
            continue
        if not motiongpt_path.exists():
            print(f"\nWarning: MotionGPT video not found: {motiongpt_path}")
            continue
        
        # Get text description
        text_description = get_text_description(texts_dir, video_id)
        
        # Create comparison video
        success = concatenate_videos(
            lom_path,
            ours_path,
            motiongpt_path,
            text_description,
            output_path,
            max_duration=args.max_duration
        )
        
        if success:
            success_count += 1
            print(f"\n✓ Created: {output_path.name}")
        else:
            print(f"\n✗ Failed: {video_id}")
    
    print(f"\n{'='*60}")
    print(f"Completed! Successfully created {success_count}/{len(common_ids)} comparison videos")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()

