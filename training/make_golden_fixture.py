#!/usr/bin/env python3
"""Developer tool: record a golden decoder test case from a real exported model.

Runs a .tflite model on one image, stores the raw output tensor plus the detections
decoded by (a) the Python mirror of the app decoder and (b) Ultralytics itself (if the
.pt weights are given). ``core/src/test/.../YoloGoldenTest.kt`` then checks that the
Kotlin decoder used by the app reproduces them.

Only use synthetic or public images: fixtures are committed to the (public) repository.

    python training/make_golden_fixture.py --model model/x.tflite --weights best.pt \\
        --image synthetic.jpg --out core/src/test/resources/golden --name yolo11n_raw
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palletlib.tflite_runner import TfliteDetector  # noqa: E402
from palletlib.yolo_decode import decode, resolve_layout  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--weights", type=Path, help="matching .pt for the Ultralytics reference")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args(argv)

    from PIL import Image

    img = np.asarray(Image.open(args.image).convert("RGB"))
    det = TfliteDetector(args.model)
    raw, lb = det.raw(img)
    fmt, n, c = resolve_layout(raw.shape, lb.dst_w, lb.dst_h, det.sidecar.get("output_format", "auto"))
    coords = det.sidecar.get("coordinates", "auto")
    py = decode(raw, lb, conf=args.conf, fmt=fmt, coords=coords, target_classes=det.target_classes)
    reference = []
    if args.weights:
        import cv2
        from ultralytics import YOLO

        res = YOLO(str(args.weights)).predict(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), imgsz=[lb.dst_h, lb.dst_w],
                                               conf=args.conf, verbose=False)[0]
        reference = [list(map(float, b)) + [float(s)] for b, s in zip(res.boxes.xyxyn.tolist(), res.boxes.conf.tolist())]
    args.out.mkdir(parents=True, exist_ok=True)
    raw.astype("<f4").tofile(args.out / f"{args.name}.output.bin")
    meta = {
        "shape": [int(v) for v in raw.shape],
        "src_width": int(img.shape[1]),
        "src_height": int(img.shape[0]),
        "input_width": int(lb.dst_w),
        "input_height": int(lb.dst_h),
        "output_format": fmt,
        "coordinates": coords,
        "conf": args.conf,
        "python_decoded": [[round(v, 6) for v in d[:5]] + [int(d[5])] for d in py],
        "ultralytics": [[round(v, 6) for v in d] for d in reference],
        "model": args.model.name,
        "note": "raw output of a model trained on SYNTHETIC images; regenerate with training/make_golden_fixture.py",
    }
    (args.out / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
    print(f"{args.name}: shape {meta['shape']} {fmt}/{coords}, {len(py)} python detections, {len(reference)} ultralytics")
    return 0


if __name__ == "__main__":
    sys.exit(main())
