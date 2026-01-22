import os
import cv2
from pathlib import Path
import sys

VIDEO_DIR = '/simurgh/group/juze/datasets/YouTube_Talking/video_20241226'
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# Suppress OpenCV/FFmpeg stderr output
try:
    # For OpenCV >= 4.2
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    # For OpenCV >= 4.5
    try:
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except AttributeError:
        pass  # If not available, fallback to context manager below

from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr (for FFmpeg warnings)."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def get_video_fps(video_path):
    with suppress_stderr():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

def main():
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(Path(VIDEO_DIR).rglob(f'*{ext}'))
    if not video_files:
        print(f'No video files found in {VIDEO_DIR}')
        return
    for video_path in video_files:
        try:
            fps = get_video_fps(video_path)
            if fps is not None and fps > 0:
                print(f'{video_path}: {fps:.2f} FPS')
            else:
                print(f'{video_path}: Unable to read FPS')
        except Exception as e:
            print(f'{video_path}: Error - {e}')

if __name__ == '__main__':
    main() 