#!/usr/bin/env python3
"""Check what off-the-shelf detectors see in your pallet photos (step 0 of the project).

Runs, on each image:
  * COCO-pretrained YOLO (80 everyday classes -- there is no "pallet" class), and
  * an open-vocabulary YOLOE model prompted with pallet-related text,
and writes annotated copies plus a text report to --out.

Everything runs locally. Images and results are written only to --out; keep that folder
outside the git repository (the repository is public).

Example:
    python training/try_pretrained.py photos/*.jpg --out ~/pallet_eval --rotate 90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROMPTS = ["wooden pallet", "pallet", "euro pallet", "stack of plastic pallets"]


def load_image(path: Path, rotate: int):
    import cv2
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # honour EXIF orientation
    if rotate:
        im = im.rotate(-rotate, expand=True)  # PIL rotates counter-clockwise
    import numpy as np

    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output folder (outside the repo)")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate images clockwise before detection (sideways footage)")
    ap.add_argument("--coco", default="yolo11n.pt", help="COCO-pretrained weights")
    ap.add_argument("--yoloe", default="yoloe-11s-seg.pt", help="open-vocabulary YOLOE weights ('' to skip)")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    models = [("coco", YOLO(args.coco), None)]
    if args.yoloe:
        try:
            from ultralytics import YOLOE

            ye = YOLOE(args.yoloe)
            ye.set_classes(PROMPTS, ye.get_text_pe(PROMPTS))
            models.append(("yoloe", ye, PROMPTS))
        except Exception as e:  # weights/text encoder unavailable offline, etc.
            report.append(f"YOLOE unavailable: {type(e).__name__}: {e}")

    for path in args.images:
        img = load_image(path, args.rotate)
        report.append(f"\n== {path.name} ({img.shape[1]}x{img.shape[0]})")
        for tag, model, _ in models:
            res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            names = res.names
            counts: dict[str, int] = {}
            for c in res.boxes.cls.tolist():
                counts[names[int(c)]] = counts.get(names[int(c)], 0) + 1
            top = sorted(zip(res.boxes.conf.tolist(), [names[int(c)] for c in res.boxes.cls.tolist()]), reverse=True)[:8]
            report.append(f"  [{tag}] {len(res.boxes)} boxes: {counts}")
            report.append("         top: " + ", ".join(f"{n} {s:.2f}" for s, n in top))
            cv2.imwrite(str(args.out / f"{path.stem}_{tag}.jpg"), res.plot(line_width=3))
    text = "\n".join(report)
    (args.out / "report.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
