#!/usr/bin/env python3
"""Write draft YOLO labels for new frames with an existing model, to speed up annotation.

After the first model exists, labeling new frames becomes "correct the drafts" instead of
"draw every box". Open the folder in your annotation tool (it reads the ``.txt`` files next
to the images), fix wrong boxes, add missed ones, delete false ones.

Never train on unchecked drafts: the model would learn its own mistakes.

Examples::

    python training_tools/prelabel.py --model runs/detect/v1/weights/best.pt --images dataset/candidates/line_09
    python training_tools/prelabel.py --model model/eur_pallet.tflite --images dataset/candidates --conf 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help=".pt weights or exported .tflite")
    ap.add_argument("--images", type=Path, required=True, help="folder with images (searched recursively)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--overwrite", action="store_true", help="replace existing label files")
    args = ap.parse_args(argv)

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print("no images found", file=sys.stderr)
        return 2

    from PIL import Image, ImageOps

    if args.model.suffix == ".tflite":
        from palletlib.tflite_runner import TfliteDetector

        det = TfliteDetector(args.model)
        predict = lambda rgb: det.detect(rgb, conf=args.conf)  # noqa: E731
    else:
        import cv2
        from ultralytics import YOLO

        model = YOLO(str(args.model))

        def predict(rgb):
            r = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            return [tuple(b) + (s, int(c)) for b, s, c in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

    written = skipped = boxes = 0
    for img in images:
        label = img.with_suffix(".txt")
        if label.exists() and not args.overwrite:
            skipped += 1
            continue
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(img)).convert("RGB"))
        dets = [d for d in predict(rgb) if d[4] >= args.conf]
        label.write_text("".join(
            f"0 {(l + r) / 2:.6f} {(t + b) / 2:.6f} {r - l:.6f} {b - t:.6f}\n" for l, t, r, b, _, _ in dets))
        written += 1
        boxes += len(dets)
    print(f"wrote {written} draft label files ({boxes} boxes), skipped {skipped} already labeled. "
          "Review every draft before training!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
