#!/usr/bin/env python3
"""Download the source videos for the YouTube_Talking dataset.

The YouTube_Talking annotations (FLAME / SMPL-X / HaMeR / speaking segments /
tokens) are released openly, but the **source videos are not redistributed**.
This script downloads them from YouTube using the shipped ``video_urls.csv``
(one row per clip: ``id,url,...``). Videos that have since been removed or made
private are skipped and logged.

Requires `yt-dlp` (``pip install yt-dlp``) and ``ffmpeg`` on PATH.

Usage:
    python download_youtube_talking.py \
        --url_csv video_urls.csv \
        --output_dir <YOUTUBE_TALKING_ROOT>/videos \
        [--workers 4]

Each video is saved as ``<output_dir>/<id>.mp4``. Re-running skips ids that
already exist, so the download is resumable.
"""

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_one(row, output_dir):
    vid = row["id"]
    url = row["url"]
    out_path = os.path.join(output_dir, f"{vid}.mp4")
    if os.path.exists(out_path):
        return vid, "skip-exists"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        "--no-playlist",
        "--quiet", "--no-warnings",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        return vid, "ok" if os.path.exists(out_path) else "no-file"
    except subprocess.CalledProcessError:
        return vid, "failed-unavailable"
    except subprocess.TimeoutExpired:
        return vid, "failed-timeout"
    except Exception as e:  # noqa: BLE001
        return vid, f"failed-{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url_csv", required=True, help="video_urls.csv (id,url,...)")
    ap.add_argument("--output_dir", required=True, help="where to save <id>.mp4")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = list(csv.DictReader(open(args.url_csv)))
    print(f"{len(rows)} videos to fetch -> {args.output_dir}")

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, r, args.output_dir) for r in rows]
        for i, fut in enumerate(as_completed(futs), 1):
            vid, status = fut.result()
            results[status] = results.get(status, 0) + 1
            if i % 50 == 0 or status.startswith("failed"):
                print(f"[{i}/{len(rows)}] {vid}: {status}")

    print("\n=== summary ===")
    for status, n in sorted(results.items()):
        print(f"  {status}: {n}")

    # Write the list of ids that are missing after the run (for a retry pass).
    missing = [
        r["id"] for r in rows
        if not os.path.exists(os.path.join(args.output_dir, f"{r['id']}.mp4"))
    ]
    if missing:
        miss_path = os.path.join(args.output_dir, "_missing_ids.txt")
        with open(miss_path, "w") as f:
            f.write("\n".join(missing) + "\n")
        print(f"\n{len(missing)} videos unavailable; ids written to {miss_path}")


if __name__ == "__main__":
    main()
