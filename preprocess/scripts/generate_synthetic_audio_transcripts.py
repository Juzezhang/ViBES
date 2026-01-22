"""
Simplified audio transcription script.

Purpose:
- Scan all audio files under an input directory
- Transcribe each using faster-whisper
- Save plain-text transcripts to an output directory, mirroring the folder structure

Defaults:
- Input audios:     /simurgh/group/juze/datasets/YouTube_Talking_Synthetic/audios
- Output transcripts:/simurgh/group/juze/datasets/YouTube_Talking_Synthetic/transcripts
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from faster_whisper import WhisperModel


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
DEFAULT_INPUT_DIR = "/simurgh/group/juze/datasets/YouTube_Talking_Synthetic/audios"
DEFAULT_OUTPUT_DIR = "/simurgh/group/juze/datasets/YouTube_Talking_Synthetic/transcripts"


def find_audio_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(path)
    return sorted(files, reverse=True)


def transcribe_audio_file(model: WhisperModel, audio_path: Path):
    segments, info = model.transcribe(
        str(audio_path), task="transcribe", language="en", word_timestamps=True
    )
    return list(segments), info


def save_transcript(output_file: Path, segments, info) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    full_text = " ".join(s.text.strip() for s in segments if getattr(s, "text", "").strip())
    with output_file.open("w", encoding="utf-8") as f:
        f.write(
            f"Detected language: {getattr(info, 'language', 'unknown')} (prob: {getattr(info, 'language_probability', 0.0):.3f})\n\n"
        )
        f.write("Full text:\n")
        f.write(full_text + "\n\n")
        f.write("Segments with word-level timestamps:\n")
        for i, s in enumerate(segments, start=1):
            f.write(f"\nSegment {i}:\n")
            f.write(f"Timestamp: {getattr(s, 'start', 0.0):.3f}s - {getattr(s, 'end', 0.0):.3f}s\n")
            f.write(f"Text: {getattr(s, 'text', '').strip()}\n")
            if getattr(s, "words", None):
                f.write("Words:\n")
                for w in s.words:
                    w_start = getattr(w, "start", 0.0)
                    w_end = getattr(w, "end", 0.0)
                    f.write(f"{w.word}: {w_start:.3f}s - {w_end:.3f}s")
                    if hasattr(w, "confidence") and w.confidence is not None:
                        f.write(f" (confidence: {w.confidence:.3f})")
                    f.write("\n")
            

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe all audios in a directory using faster-whisper"
    )
    parser.add_argument(
        "--input_dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save transcripts",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (e.g., tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device to run the model on"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcripts if present",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info(f"Scanning audios in: {input_dir}")

    audio_files = find_audio_files(input_dir)
    logging.info(f"Found {len(audio_files)} audio files")

    logging.info(f"Loading Whisper model: {args.model} on {args.device}")
    try:
        model = WhisperModel(args.model, device=args.device, compute_type="float32")
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {e}")
        raise SystemExit(1)

    processed = 0
    for audio_path in audio_files:
        rel = audio_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".txt")
        if out_path.exists() and not args.overwrite:
            logging.info(f"[SKIP] {out_path} exists")
            continue
        logging.info(f"Transcribing: {audio_path}")
        try:
            segments, info = transcribe_audio_file(model, audio_path)
            save_transcript(out_path, segments, info)
            processed += 1
        except Exception as e:
            logging.error(f"Failed to transcribe {audio_path}: {e}")

    logging.info(f"Done. Wrote {processed} transcript files to {output_dir}")


if __name__ == "__main__":
    main()


