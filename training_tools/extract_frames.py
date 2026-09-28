#!/usr/bin/env python3
"""Extract candidate training frames from phone videos.

Walking videos contain hundreds of nearly identical frames. This tool samples frames at a
target rate, skips near-duplicates (camera standing still, very slow walking) and writes a
manifest CSV. Output files are named ``<video>_f<frame>.jpg`` so that
``split_dataset.py`` can keep all frames of one video in the same split (frames of the
same video in train *and* val would make validation scores look better than reality).

Examples::

    # ~2 frames per second from one video
    python training_tools/extract_frames.py videos/line_03.mp4 --out dataset/candidates --fps 2

    # every 15th frame of every video in a folder, stricter duplicate filtering
    python training_tools/extract_frames.py videos/ --out dataset/candidates --every 15 --min-diff 10

    # footage that plays back sideways
    python training_tools/extract_frames.py sideways.mp4 --out dataset/candidates --rotate 90

Everything stays on your machine. Keep the output folder out of git (the repository is
public; ``dataset/candidates`` is already git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".3gp", ".webm"}
THUMB_SIZE = (64, 36)  # width, height of the grayscale thumbnail used for similarity
MANIFEST_FIELDS = ["file", "video", "frame_index", "time_s", "width", "height", "diff", "sharpness"]


@dataclass
class ExtractOptions:
    fps: float = 2.0
    every: int = 0
    min_diff: float = 6.0
    max_per_video: int = 0
    max_side: int = 1280
    rotate: int = 0
    min_sharpness: float = 0.0
    jpeg_quality: int = 92
    dry_run: bool = False


@dataclass
class VideoStats:
    video: str
    read: int = 0
    considered: int = 0
    kept: int = 0
    duplicates: int = 0
    blurry: int = 0


def find_videos(inputs: list[Path]) -> list[Path]:
    videos: list[Path] = []
    for p in inputs:
        if p.is_dir():
            videos += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
        elif p.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(p)
        else:
            print(f"skipping {p}: not a video file", file=sys.stderr)
    return videos


def thumbnail(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def frame_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference (0..255) between two thumbnails, after removing the
    global brightness offset so auto-exposure changes alone do not count as 'new'."""
    return float(np.mean(np.abs((a - a.mean()) - (b - b.mean()))))


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian on a 320 px wide grayscale copy (higher = sharper)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w > 320:
        gray = cv2.resize(gray, (320, max(1, round(h * 320 / w))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def resize_max_side(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def safe_stem(path: Path) -> str:
    """File-system and split-friendly video name (no '_f<digits>' ambiguity)."""
    stem = "".join(c if c.isalnum() or c in "-." else "-" for c in path.stem)
    return stem.strip("-") or "video"


def extract_video(video: Path, out_dir: Path, opts: ExtractOptions, writer: csv.DictWriter | None) -> VideoStats:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(native_fps) or native_fps <= 0:
        native_fps = 30.0
    step = opts.every if opts.every > 0 else max(1, round(native_fps / max(opts.fps, 1e-3)))
    name = safe_stem(video)
    stats = VideoStats(video=video.name)
    last_thumb: np.ndarray | None = None
    index = -1
    while True:
        if not cap.grab():
            break
        index += 1
        stats.read += 1
        if index % step != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue
        stats.considered += 1
        frame = rotate_frame(frame, opts.rotate)
        thumb = thumbnail(frame)
        diff = frame_difference(thumb, last_thumb) if last_thumb is not None else float("inf")
        if diff < opts.min_diff:
            stats.duplicates += 1
            continue
        sharp = sharpness(frame)
        if sharp < opts.min_sharpness:
            stats.blurry += 1
            continue
        last_thumb = thumb
        frame = resize_max_side(frame, opts.max_side)
        file_name = f"{name}_f{index:06d}.jpg"
        if not opts.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out_dir / file_name), frame, [cv2.IMWRITE_JPEG_QUALITY, opts.jpeg_quality]):
                raise RuntimeError(f"failed to write {out_dir / file_name}")
            if writer is not None:
                writer.writerow({
                    "file": file_name,
                    "video": video.name,
                    "frame_index": index,
                    "time_s": round(index / native_fps, 3),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "diff": "" if diff == float("inf") else round(diff, 2),
                    "sharpness": round(sharp, 1),
                })
        stats.kept += 1
        if opts.max_per_video and stats.kept >= opts.max_per_video:
            break
    cap.release()
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="video files and/or folders")
    ap.add_argument("--out", type=Path, required=True, help="output folder for frames")
    ap.add_argument("--fps", type=float, default=2.0, help="approximate frames per second to consider (default 2)")
    ap.add_argument("--every", type=int, default=0, help="consider every N-th frame instead of --fps")
    ap.add_argument("--min-diff", type=float, default=6.0,
                    help="skip frames whose thumbnail differs less than this (0-255) from the last kept "
                         "frame; 0 keeps everything (default 6)")
    ap.add_argument("--max-per-video", type=int, default=0, help="cap on frames per video (0 = no cap)")
    ap.add_argument("--max-side", type=int, default=1280, help="downscale so the longest side <= this (0 = keep)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate frames clockwise (for footage that plays back sideways)")
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="drop frames blurrier than this (variance of Laplacian). Default 0 keeps blurry "
                         "frames on purpose: they are realistic training examples")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--subdir-per-video", action="store_true", help="write each video's frames into its own subfolder")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be extracted")
    args = ap.parse_args(argv)

    videos = find_videos(args.inputs)
    if not videos:
        print("no videos found", file=sys.stderr)
        return 2
    opts = ExtractOptions(
        fps=args.fps, every=args.every, min_diff=args.min_diff, max_per_video=args.max_per_video,
        max_side=args.max_side, rotate=args.rotate, min_sharpness=args.min_sharpness,
        jpeg_quality=args.jpeg_quality, dry_run=args.dry_run,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.csv"
    new_file = not manifest.exists()
    total = 0
    with open(manifest, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new_file and not args.dry_run:
            writer.writeheader()
        for video in videos:
            out_dir = args.out / safe_stem(video) if args.subdir_per_video else args.out
            s = extract_video(video, out_dir, opts, None if args.dry_run else writer)
            total += s.kept
            print(f"{s.video}: read {s.read} frames, considered {s.considered}, kept {s.kept} "
                  f"(skipped {s.duplicates} near-duplicates, {s.blurry} blurry)")
    print(f"{'would keep' if args.dry_run else 'wrote'} {total} frames to {args.out}")
    if args.dry_run and new_file and manifest.exists() and manifest.stat().st_size == 0:
        manifest.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
