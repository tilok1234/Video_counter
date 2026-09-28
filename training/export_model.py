#!/usr/bin/env python3
"""Export a trained detector to TFLite for the Android app and verify the export.

Steps:
  1. Ultralytics TFLite export (fp16 by default; int8 needs --data for calibration).
  2. Inspect the model's tensors and work out the output layout / coordinate space.
  3. Optionally compare TFLite detections (decoded exactly like the app does) with the
     PyTorch model on a few images (--check-images).
  4. Write ``<out>/<name>.tflite`` plus the sidecar ``<out>/<name>.json`` the app reads.

Examples::

    python training/export_model.py --weights runs/detect/v1/weights/best.pt --imgsz 416 --out model/
    python training/export_model.py --weights best.pt --imgsz 320 --precision int8 --data dataset/dataset.yaml \\
        --check-images dataset/images/test --out model/ --name eur_pallet_int8

Copy both files to the phone and import them (Settings → Import model), or put them in
``app/src/main/assets/models/`` as ``eur_pallet.tflite`` / ``eur_pallet.json`` and rebuild.
See docs/ANDROID_MODEL_INTEGRATION.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palletlib.tflite_runner import TfliteDetector  # noqa: E402
from palletlib.yolo_decode import END_TO_END, iou, resolve_layout  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(items: list[Path], limit: int) -> list[Path]:
    out: list[Path] = []
    for p in items:
        if p.is_dir():
            out += sorted(q for q in p.glob("*") if q.suffix.lower() in IMAGE_EXTENSIONS)
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return out[:limit]


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def torch_predictions(model, img_rgb: np.ndarray, imgsz, conf: float) -> list[tuple]:
    import cv2

    res = model.predict(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), imgsz=imgsz, conf=conf, verbose=False)[0]
    boxes = res.boxes
    return [tuple(map(float, b)) + (float(s), int(c)) for b, s, c in
            zip(boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]


def match_rate(a: list[tuple], b: list[tuple], thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching; returns (matched, only_in_a, only_in_b)."""
    used = set()
    matched = 0
    for da in sorted(a, key=lambda d: -d[4]):
        best, best_j = 0.0, -1
        for j, db in enumerate(b):
            if j in used:
                continue
            v = iou(da, db)
            if v > best:
                best, best_j = v, j
        if best >= thr:
            used.add(best_j)
            matched += 1
    return matched, len(a) - matched, len(b) - matched


def detect_coordinates(det: TfliteDetector, images: list[np.ndarray]) -> str:
    """Normalized outputs never exceed ~1 for real detections; pixel outputs do."""
    samples = images or [np.full((480, 640, 3), 128, np.uint8)]
    for img in samples:
        y, lb = det.raw(img)
        fmt, n, c = resolve_layout(y.shape, lb.dst_w, lb.dst_h)
        flat = y.reshape(-1)
        if fmt == END_TO_END:
            rows = flat[: n * c].reshape(n, c)
            rows = rows[rows[:, 4] > 0.25]
            coords = rows[:, :4]
        else:
            mat = flat[: n * c].reshape(c, n).T if fmt == "yolo_raw" else flat[: n * c].reshape(n, c)
            coords = mat[mat[:, 4:].max(axis=1) > 0.25][:, :4]
        if len(coords):
            return "normalized" if float(coords.max()) <= 2.0 else "pixels"
    return "normalized"  # Ultralytics TFLite exports normalize boxes to the input size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help="trained .pt (e.g. runs/detect/v1/weights/best.pt)")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[416],
                    help="model input size: one value (square) or H W. 320-416 for budget phones, 640 for best accuracy")
    ap.add_argument("--precision", choices=["fp16", "fp32", "int8"], default="fp16")
    ap.add_argument("--data", type=Path, help="dataset.yaml (required for int8 calibration)")
    ap.add_argument("--out", type=Path, default=Path("model"))
    ap.add_argument("--name", default="eur_pallet")
    ap.add_argument("--check-images", type=Path, nargs="*", default=[], help="images/folders for the export check")
    ap.add_argument("--check-limit", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.35, help="suggested operating threshold stored in the sidecar")
    args = ap.parse_args(argv)

    if args.precision == "int8" and not args.data:
        ap.error("--precision int8 needs --data <dataset.yaml> for calibration images")
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else list(args.imgsz)

    from ultralytics import YOLO, __version__ as ul_version

    model = YOLO(str(args.weights))
    kwargs = dict(format="tflite", imgsz=imgsz, half=args.precision == "fp16", int8=args.precision == "int8", nms=False)
    if args.data:
        kwargs["data"] = str(args.data)
    exported = Path(model.export(**kwargs))
    print(f"exported: {exported}")

    det = TfliteDetector(exported)
    in_shape = det.inp["shape"].tolist()
    out_shape = det.out["shape"].tolist()
    fmt, anchors, channels = resolve_layout(out_shape, det.in_w, det.in_h)
    print(f"input {in_shape} {det.inp['dtype'].__name__}, output {out_shape} {det.out['dtype'].__name__} -> {fmt}")

    check_paths = list_images(args.check_images, args.check_limit)
    check_imgs = [load_rgb(p) for p in check_paths]
    coordinates = detect_coordinates(det, check_imgs)
    names = [model.names[k] for k in sorted(model.names)]
    targets = ["eur_pallet_base"] if "eur_pallet_base" in names else names

    sidecar = {
        "name": args.name,
        "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "output_format": fmt,
        "coordinates": coordinates,
        "input_width": det.in_w,
        "input_height": det.in_h,
        "input_normalization": "zero_one",
        "letterbox_pad_value": 114,
        "classes": names,
        "target_classes": targets,
        "confidence_threshold": args.conf,
        "iou_threshold": 0.5,
        "framework": f"ultralytics {ul_version}",
        "source_weights": args.weights.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quantization": args.precision,
    }

    if check_imgs:
        det.sidecar = sidecar
        det.target_classes = {names.index(t) for t in targets}
        tot = [0, 0, 0]
        for img in check_imgs:
            ref = [d for d in torch_predictions(model, img, imgsz, args.conf) if names[d[5]] in targets]
            got = det.detect(img, conf=args.conf)
            m, only_pt, only_tfl = match_rate(ref, got)
            tot = [tot[0] + m, tot[1] + only_pt, tot[2] + only_tfl]
        total_ref = tot[0] + tot[1]
        rate = tot[0] / total_ref if total_ref else 1.0
        print(f"export check on {len(check_imgs)} images: {tot[0]} boxes matched, {tot[1]} only in PyTorch, "
              f"{tot[2]} only in TFLite (match rate {rate:.0%})")
        sidecar["metrics"] = {"export_match_rate": round(rate, 4)}
        if rate < 0.8 and total_ref >= 5:
            print("WARNING: TFLite detections differ a lot from PyTorch; check precision (try fp32) and imgsz")

    args.out.mkdir(parents=True, exist_ok=True)
    target_model = args.out / f"{args.name}.tflite"
    shutil.copy2(exported, target_model)
    (args.out / f"{args.name}.json").write_text(json.dumps(sidecar, indent=2))
    size_mb = target_model.stat().st_size / 1e6
    print(f"wrote {target_model} ({size_mb:.1f} MB) and {args.out / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
