#!/usr/bin/env python3
"""Render a SYNTHETIC "walk along a pallet line" video with a known pallet count.

For smoke-testing the video tooling (extract_frames.py, predict_video.py, the replay CLI)
without workplace footage. The drawing matches make_synthetic_dataset.py, so a model
trained on that synthetic dataset detects these pallets. Real footage is still required
for a real model.

Example::

    python training_tools/make_synthetic_video.py --out /tmp/synthetic_walk.mp4 --pallets 12 --pause 2 --reverse 1.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synthetic_dataset import draw_half_pallet_stack, draw_pallet_base  # noqa: E402


def render_panorama(rng: np.random.Generator, pallets: int, pitch: int, width: int, height: int, margin: int):
    pano_w = margin * 2 + pallets * pitch
    img = np.zeros((height, pano_w, 3), np.uint8)
    horizon = int(height * 0.72)
    img[:horizon] = 150
    img[horizon:] = 115
    img = np.clip(img.astype(np.int16) + rng.normal(0, 5, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    pw = int(pitch * 0.94)
    ph = int(pw * 0.24)
    centers = []
    for i in range(pallets):
        x0 = margin + i * pitch
        y_bottom = horizon + int(height * 0.08)
        draw_pallet_base(img, rng, x0, y_bottom - ph, pw, ph)
        draw_half_pallet_stack(img, rng, x0, y_bottom - ph - 1, pw, max(4, int(ph * 0.75)), int(rng.integers(9, 14)))
        centers.append(x0 + pw / 2)
    return img, centers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pallets", type=int, default=12)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=0.25, help="walking speed in frame widths per second")
    ap.add_argument("--pause", type=float, default=0.0, help="stand still this many seconds halfway")
    ap.add_argument("--reverse", type=float, default=0.0, help="walk back this many seconds after the pause")
    ap.add_argument("--blur", action="store_true", help="add horizontal motion blur")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    pitch = int(args.width * 0.36)
    margin = args.width  # start and end with empty floor so every pallet crosses the centre
    pano, centers = render_panorama(rng, args.pallets, pitch, args.width, args.height, margin)
    v = args.speed * args.width / args.fps  # px per frame
    positions: list[float] = []  # left edge of the camera window in the panorama
    x = 0.0
    end = pano.shape[1] - args.width
    half = end / 2
    while x < half:
        positions.append(x)
        x += v
    positions += [x] * int(args.pause * args.fps)
    for _ in range(int(args.reverse * args.fps)):
        x = max(0.0, x - v)
        positions.append(x)
    while x < end:
        positions.append(x)
        x += v

    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        print("cannot open video writer", file=sys.stderr)
        return 2
    kernel = None
    if args.blur:
        k = 9
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
    for p in positions:
        frame = pano[:, int(p):int(p) + args.width].copy()
        if kernel is not None:
            frame = cv2.filter2D(frame, -1, kernel)
        writer.write(frame)
    writer.release()
    start_c, end_c = positions[0] + args.width / 2, positions[-1] + args.width / 2
    expected = sum(1 for c in centers if start_c < c <= end_c)
    print(f"wrote {args.out}: {len(positions)} frames, expected count {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
