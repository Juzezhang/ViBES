#!/usr/bin/env python3
"""
Merge body and face videos by alternating between two directories and stacking up
to 16 videos horizontally per output clip.
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

try:
    from moviepy.editor import VideoFileClip, clips_array

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not available. Install with: pip install moviepy")


def list_videos(directory: Path) -> List[Path]:
    """Return a sorted list of .mp4 files inside the directory."""
    if not directory.exists():
        print(f"Warning: directory does not exist: {directory}")
        return []
    return sorted(p for p in directory.glob("*.mp4") if p.is_file())


def interleave_sequences(body_videos: List[Path], face_videos: List[Path]) -> List[Path]:
    """Interleave videos from body and face lists."""
    combined: List[Path] = []
    idx_body, idx_face = 0, 0
    pick_body = True
    while idx_body < len(body_videos) or idx_face < len(face_videos):
        if pick_body and idx_body < len(body_videos):
            combined.append(body_videos[idx_body])
            idx_body += 1
        elif not pick_body and idx_face < len(face_videos):
            combined.append(face_videos[idx_face])
            idx_face += 1
        elif idx_body < len(body_videos):
            combined.append(body_videos[idx_body])
            idx_body += 1
        elif idx_face < len(face_videos):
            combined.append(face_videos[idx_face])
            idx_face += 1
        pick_body = not pick_body
    return combined


def chunk_sequence(items: List[Path], chunk_size: int) -> Iterable[List[Path]]:
    """Yield chunks of items with at most chunk_size elements."""
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def merge_video_group(video_paths: List[Path], output_path: Path) -> bool:
    """Merge a list of videos horizontally without text overlay."""
    if not MOVIEPY_AVAILABLE:
        print("MoviePy not available!")
        return False

    existing_videos = [path for path in video_paths if path.exists()]
    if not existing_videos:
        print("No videos to merge for this group!")
        return False

    try:
        video_clips = [VideoFileClip(str(path)) for path in existing_videos]
        min_duration = min(clip.duration for clip in video_clips)
        # Ensure all clips share the same height to keep horizontal stack aligned
        target_height = min(int(clip.h) for clip in video_clips)

        clips_for_merge = []
        for clip in video_clips:
            resized = clip.resize(height=target_height).subclip(0, min_duration)
            clips_for_merge.append(resized)

        final_video = clips_array([clips_for_merge])

        audio_source = existing_videos[0]
        audio_clip = VideoFileClip(str(audio_source)).audio
        if audio_clip is not None:
            audio_clip = audio_clip.subclip(0, min_duration)
            final_video = final_video.set_audio(audio_clip)

        fps = video_clips[0].fps
        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset="medium",
        )

        for clip in video_clips:
            clip.close()
        for clip in clips_for_merge:
            clip.close()
        final_video.close()

        return True

    except Exception as exc:
        print(f"Error merging videos: {exc}")
        import traceback

        traceback.print_exc()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge body and face videos horizontally.")
    parser.add_argument(
        "--body_dir",
        type=Path,
        default=Path("/path/to/conversational_agent/paper_result/video/dataset/body"),
        help="Directory containing body videos.",
    )
    parser.add_argument(
        "--face_dir",
        type=Path,
        default=Path("/path/to/conversational_agent/paper_result/video/dataset/face"),
        help="Directory containing face videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/path/to/conversational_agent/paper_result/video/dataset/merged_body_face"),
        help="Directory to store merged videos.",
    )
    parser.add_argument(
        "--max_clips_per_video",
        type=int,
        default=20,
        help="Maximum number of clips per merged video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not MOVIEPY_AVAILABLE:
        print("MoviePy is not installed. Please run: pip install moviepy")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    body_videos = list_videos(args.body_dir)
    face_videos = list_videos(args.face_dir)

    if not body_videos and not face_videos:
        print("No videos found in either directory. Exiting.")
        return

    combined_sequence = interleave_sequences(body_videos, face_videos)
    print(f"Total videos to process: {len(combined_sequence)}")

    success_count = 0
    total_groups = 0

    for group_idx, group in enumerate(
        tqdm(
            list(chunk_sequence(combined_sequence, args.max_clips_per_video)),
            desc="Merging groups",
        )
    ):
        total_groups += 1
        output_path = args.output_dir / f"merged_{group_idx:04d}.mp4"

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            success_count += 1
            continue

        success = merge_video_group(group, output_path)
        if success:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Completed merging groups: {success_count}/{total_groups}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()


