#!/usr/bin/env python3
"""Generate a small SYNTHETIC dataset for smoke-testing the training/export pipeline.

The images are procedural drawings (grey floor/wall, black "half-pallet" grids standing on
brown "EUR-pallet" bases, distractor boxes). A model trained on them is USELESS on real
footage. The only purpose is to check, in minutes and without any workplace imagery, that
``train.py`` → ``export_model.py`` → TFLite decoding works on your machine, and to produce
test fixtures for the decoder.

Example::

    python training_tools/make_synthetic_dataset.py --out /tmp/synthetic_dataset --train 160 --val 40 --test 20
    python training/train.py --dataset /tmp/synthetic_dataset --epochs 5 --imgsz 320 --name smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def wood_texture(rng: np.random.Generator, w: int, h: int, dark: float) -> np.ndarray:
    base = np.array([60, 120, 170], dtype=np.float32) * dark  # BGR light wood
    tex = np.tile(base, (h, w, 1))
    grain = rng.normal(0, 12, (h, 1, 1)).astype(np.float32)
    tex += grain
    tex += rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    return np.clip(tex, 0, 255).astype(np.uint8)


def draw_pallet_base(img: np.ndarray, rng: np.random.Generator, x: int, y: int, w: int, h: int) -> None:
    """Deck boards on top, three support blocks with dark fork openings between them."""
    dark = rng.uniform(0.55, 1.1)
    img[y:y + h, x:x + w] = wood_texture(rng, w, h, dark)
    top = max(2, h // 5)
    img[y + top:y + h - top, x:x + w] = (img[y + top:y + h - top, x:x + w] * 0.9).astype(np.uint8)
    block_w = max(3, w // 7)
    for bx in (x, x + w // 2 - block_w // 2, x + w - block_w):
        cv2.rectangle(img, (bx, y + top), (bx + block_w, y + h - top), (70, 130, 185), -1)
    for ox in (x + block_w, x + w // 2 + block_w // 2):
        gap_end = ox + (w // 2 - block_w * 3 // 2)
        cv2.rectangle(img, (ox, y + top + 1), (gap_end, y + h - top - 1), (25, 25, 30), -1)


def draw_half_pallet_stack(img: np.ndarray, rng: np.random.Generator, x: int, y_bottom: int, w: int, layer_h: int, layers: int) -> None:
    for k in range(layers):
        yb = y_bottom - k * layer_h
        yt = yb - layer_h + 1
        if yt < 0:
            break
        shade = int(rng.uniform(15, 45))
        cv2.rectangle(img, (x, yt), (x + w, yb), (shade, shade + int(rng.uniform(0, 12)), shade), -1)
        cells = int(rng.integers(5, 9))
        for c in range(cells):
            cx = x + int((c + 0.5) * w / cells)
            cv2.rectangle(img, (cx - w // (cells * 3), yt + layer_h // 4), (cx + w // (cells * 3), yb - layer_h // 4),
                          (5, 5, 5), -1)


def make_image(rng: np.random.Generator, width: int, height: int, negative: bool) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    img = np.zeros((height, width, 3), np.uint8)
    horizon = int(height * rng.uniform(0.55, 0.8))
    img[:horizon] = np.clip(rng.normal(150, 25), 60, 230)
    img[horizon:] = np.clip(rng.normal(120, 20), 50, 200)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    boxes: list[tuple[float, float, float, float]] = []
    if not negative:
        pw = int(width * rng.uniform(0.18, 0.4))
        ph = max(8, int(pw * rng.uniform(0.2, 0.3)))
        gap = int(pw * rng.uniform(0.02, 0.15))
        x = int(rng.uniform(-0.5, 0.2) * pw)
        y_base_bottom = min(height - 2, horizon + int(rng.uniform(0.0, 0.15) * height))
        while x < width:
            x0, x1 = max(0, x), min(width, x + pw)
            if x1 - x0 > pw * 0.35:
                draw_pallet_base(img, rng, x0, y_base_bottom - ph, x1 - x0, ph)
                layer_h = max(4, int(ph * rng.uniform(0.6, 0.9)))
                draw_half_pallet_stack(img, rng, x0, y_base_bottom - ph - 1, x1 - x0, layer_h, int(rng.integers(6, 16)))
                boxes.append(((x0 + x1) / 2 / width, (y_base_bottom - ph / 2) / height, (x1 - x0) / width, ph / height))
            x += pw + gap
    for _ in range(int(rng.integers(0, 3))):  # distractors: plain brown/grey boxes, no pallet structure
        dw, dh = int(width * rng.uniform(0.05, 0.2)), int(height * rng.uniform(0.03, 0.1))
        dx, dy = int(rng.uniform(0, width - dw)), int(rng.uniform(0, horizon))
        color = tuple(int(v) for v in rng.uniform(40, 200, 3))
        cv2.rectangle(img, (dx, dy), (dx + dw, dy + dh), color, -1)
    gain = rng.uniform(0.6, 1.4)
    img = np.clip(img.astype(np.float32) * gain + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = int(rng.integers(3, 9)) | 1
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k  # horizontal motion blur
        img = cv2.filter2D(img, -1, kernel)
    return img, boxes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=int, default=160)
    ap.add_argument("--val", type=int, default=40)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--negatives", type=float, default=0.1, help="fraction of images without pallets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.train), ("val", args.val), ("test", args.test)):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, boxes = make_image(rng, args.width, args.height, rng.random() < args.negatives)
            stem = f"synthetic-{split}_f{i:06d}"
            cv2.imwrite(str(args.out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            (args.out / "labels" / split / f"{stem}.txt").write_text(
                "".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes))
    print(f"wrote synthetic dataset to {args.out} (NOT real data; for pipeline smoke tests only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
