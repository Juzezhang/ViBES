"""
Extract audio segments based on TalkNet speaking segments.

Defaults:
- Input audios:     /path/to/YouTube_Talking/audios
- TalkNet outputs:  /path/to/YouTube_Talking/talknet_output
- Output segments:  /path/to/YouTube_Talking_Synthetic/audios

For each audio file, we find its speaking segments from TalkNet results (by matching the
file's stem as the video_id), slice the audio accordingly, and save each segment to the
output directory using the original filename plus a suffix and an index.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
# Optional dependency: pydub (preferred for slicing). Fallback to ffmpeg CLI if absent.
try:
    from pydub import AudioSegment  # type: ignore
    HAVE_PYDUB = True
except Exception:  # pragma: no cover - env dependent
    AudioSegment = None  # type: ignore
    HAVE_PYDUB = False
import subprocess


# =====================
# Configuration defaults
# =====================
DEFAULT_AUDIO_DIR = "/path/to/YouTube_Talking/audios_original"
DEFAULT_TALKNET_DIR = "/path/to/YouTube_Talking/talknet_output"
DEFAULT_OUTPUT_DIR = "/path/to/YouTube_Talking_Synthetic/audios"
DEFAULT_SUFFIX = "_"
MIN_SEGMENT_DURATION = 2.0  # seconds; discard merged segments shorter than this


def merge_segments(segments: Sequence[Tuple[float, float]], tol: float = 2.0) -> List[Tuple[float, float]]:
    """
    Merge adjacent or overlapping segments when the gap between them is <= tol seconds.
    Expects segments as (start, end) in seconds.
    """
    if not segments:
        return []

    segments_sorted = sorted((float(s), float(e)) for s, e in segments if e > s)
    merged: List[Tuple[float, float]] = []

    cur_s, cur_e = segments_sorted[0]
    for s, e in segments_sorted[1:]:
        if s - cur_e <= tol:  # overlap or small gap
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def load_speaking_segments_by_video_id(video_id: str, talknet_output_dir: str) -> List[Tuple[float, float]] | None:
    """
    Load TalkNet speaking segments by video_id directly from saved output.

    Returns a list of (start, end) in seconds, or None if results don't exist.
    """
    video_id = os.path.splitext(os.path.basename(video_id))[0]
    talknet_video_dir = os.path.join(talknet_output_dir, video_id)
    pywork_dir = os.path.join(talknet_video_dir, "pywork")
    scores_pkl = os.path.join(pywork_dir, "scores.pckl")
    tracks_pkl = os.path.join(pywork_dir, "tracks.pckl")

    if not os.path.exists(scores_pkl) or not os.path.exists(tracks_pkl):
        logging.warning(f"TalkNet results not found for {video_id}. Skipping.")
        return None

    try:
        with open(scores_pkl, "rb") as f:
            scores = pickle.load(f)
        with open(tracks_pkl, "rb") as f:
            tracks = pickle.load(f)

        speaking_segments: List[Tuple[float, float]] = []
        for track_idx, (track, score) in enumerate(zip(tracks, scores)):
            try:
                prob = score
                if isinstance(prob, np.ndarray) and prob.ndim > 1 and prob.shape[1] > 1:
                    prob = prob[:, 1]

                threshold = 0.0  # as per user's modified usage
                speaking = prob > threshold

                start: float | None = None
                for idx, is_speaking in enumerate(speaking):
                    t = track["track"]["frame"][idx] / 25.0  # seconds
                    if is_speaking and start is None:
                        start = t
                    elif (not is_speaking) and (start is not None):
                        speaking_segments.append((start, t))
                        start = None
                if start is not None:  # continues until the end
                    t = track["track"]["frame"][-1] / 25.0
                    speaking_segments.append((start, t))
            except Exception as e:
                logging.error(f"Error processing track {track_idx} for {video_id}: {e}")

        # Merge segments with a 2.0s tolerance between neighboring segments
        speaking_segments = merge_segments(speaking_segments, tol=2.0)
        # Discard merged segments shorter than 1.0s
        speaking_segments = [
            (s, e) for (s, e) in speaking_segments if (e - s) >= MIN_SEGMENT_DURATION
        ]
        logging.info(f"Loaded {len(speaking_segments)} speaking segments for {video_id}")
        return speaking_segments
    except Exception as e:
        logging.error(f"Error loading TalkNet results for {video_id}: {e}")
        return None


def slice_audio_with_pydub(
    audio_path: Path,
    segments: Sequence[Tuple[float, float]],
    output_dir: Path,
    suffix: str = DEFAULT_SUFFIX,
    overwrite: bool = False,
) -> int:
    """
    Slice an audio file into segments and export each as WAV.

    Returns the number of segments successfully saved.
    """
    if not HAVE_PYDUB:
        return 0

    audio = AudioSegment.from_file(str(audio_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    stem = audio_path.stem
    for idx, (start_s, end_s) in enumerate(segments, start=1):
        if end_s - start_s < MIN_SEGMENT_DURATION:
            continue
        start_ms = max(0, int(start_s * 1000))
        end_ms = max(0, int(end_s * 1000))
        if end_ms <= start_ms:
            continue

        out_path = output_dir / f"{stem}{suffix}{idx:04d}.wav"
        if out_path.exists() and not overwrite:
            logging.info(f"Skip existing: {out_path}")
            continue

        clip = audio[start_ms:end_ms]
        clip.export(str(out_path), format="wav")
        saved += 1
    return saved


def slice_audio_with_ffmpeg(
    audio_path: Path,
    segments: Sequence[Tuple[float, float]],
    output_dir: Path,
    suffix: str = DEFAULT_SUFFIX,
    overwrite: bool = False,
) -> int:
    """
    Slice an audio file using ffmpeg CLI and export each segment as WAV.
    Returns the number of segments successfully saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    stem = audio_path.stem

    for idx, (start_s, end_s) in enumerate(segments, start=1):
        if end_s - start_s < MIN_SEGMENT_DURATION:
            continue
        out_path = output_dir / f"{stem}{suffix}{idx:04d}.wav"
        if out_path.exists() and not overwrite:
            logging.info(f"Skip existing: {out_path}")
            continue

        # Use accurate trim: place -ss/-to after -i; re-encode to mono 16k wav
        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(audio_path),
            "-ss", f"{start_s:.3f}",
            "-to", f"{end_s:.3f}",
            "-ac", "1", "-ar", "16000",
            "-vn", "-acodec", "pcm_s16le",
            str(out_path),
        ]
        try:
            # Capture only errors; show them if returncode != 0
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode != 0:
                logging.error(f"ffmpeg failed for {audio_path.name} seg {idx}: {proc.stderr.decode(errors='ignore')[:200]}")
                continue
            saved += 1
        except FileNotFoundError:
            logging.error("ffmpeg is not installed or not in PATH")
            break
    return saved


def collect_audio_files(audio_dir: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
    files: List[Path] = []
    for p in audio_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Extract audio segments using TalkNet speaking segments.")
    parser.add_argument("--audio_dir", default=DEFAULT_AUDIO_DIR, help="Directory containing input audios")
    parser.add_argument("--talknet_dir", default=DEFAULT_TALKNET_DIR, help="Directory containing TalkNet outputs")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save sliced audio segments")
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX, help="Suffix to append to base filename")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N audio files (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    audio_dir = Path(args.audio_dir)
    talknet_dir = args.talknet_dir
    output_dir = Path(args.output_dir)

    if not audio_dir.is_dir():
        logging.error(f"Audio directory does not exist: {audio_dir}")
        return
    if not os.path.isdir(talknet_dir):
        logging.error(f"TalkNet output directory does not exist: {talknet_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = collect_audio_files(audio_dir)
    if args.limit is not None:
        audio_files = audio_files[: args.limit]

    total_saved = 0
    for idx, audio_path in enumerate(audio_files, start=1):
        video_id = audio_path.stem
        logging.info(f"[{idx}/{len(audio_files)}] Processing {audio_path.name} (id={video_id})")

        segments = load_speaking_segments_by_video_id(video_id, talknet_dir)
        if not segments:
            logging.info(f"No segments for {video_id}; skipping")
            continue

        if HAVE_PYDUB:
            saved = slice_audio_with_pydub(
                audio_path, segments, output_dir, suffix=args.suffix, overwrite=args.overwrite
            )
        else:
            saved = slice_audio_with_ffmpeg(
                audio_path, segments, output_dir, suffix=args.suffix, overwrite=args.overwrite
            )
        logging.info(f"Saved {saved} segments for {video_id}")
        total_saved += saved

    logging.info(f"Done. Total segments saved: {total_saved}")


if __name__ == "__main__":
    main()
