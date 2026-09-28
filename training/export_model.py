#!/usr/bin/env python3
"""Export a trained detector to TFLite for the Android app and verify the export.

Steps:
  1. Ultralytics LiteRT export (``format="litert"``, which replaced ``tflite`` in
     Ultralytics 8.4.83). Runs on Linux x86-64 and macOS only (on Windows use WSL2).
     Precision: ``fp32`` (default; the phone's GPU delegate runs it in FP16 anyway),
     ``w8a32`` (int8 weights, ~4x smaller, no calibration), ``int8`` / ``w8a16`` (static
     quantization, needs --data for calibration images).
  2. Inspect the model's tensors and work out the output layout / coordinate space.
  3. Optionally compare TFLite detections (decoded exactly like the app does) with the
     PyTorch model on a few images (--check-images).
  4. Write ``<out>/<name>.tflite`` plus the sidecar ``<out>/<name>.json`` the app reads.

Examples::

    python training/export_model.py --weights runs/detect/v1/weights/best.pt --imgsz 416 --out model/
    python training/export_model.py --weights best.pt --imgsz 320 --precision int8 --data dataset/dataset.yaml \\
        --check-images dataset/images/test --out model/ --name eur_pallet_int8

The exported model's input is NCHW float32 and its output is either the raw YOLO head
(``[1, 4+classes, anchors]``, normalized boxes) or an end-to-end head (``[1, 300, 6]``,
pixel boxes, YOLO26); the sidecar records which, and the app handles both.

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
    ap.add_argument("--precision", choices=["fp32", "w8a32", "int8", "w8a16"], default="fp32",
                    help="fp32 (default), w8a32 (dynamic int8 weights), int8 / w8a16 (static, need --data)")
    ap.add_argument("--data", type=Path, help="dataset.yaml (required for int8 calibration)")
    ap.add_argument("--out", type=Path, default=Path("model"))
    ap.add_argument("--name", default="eur_pallet")
    ap.add_argument("--check-images", type=Path, nargs="*", default=[], help="images/folders for the export check")
    ap.add_argument("--check-limit", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.35, help="suggested operating threshold stored in the sidecar")
    ap.add_argument("--end2end", action="store_true",
                    help="export the NMS-free end-to-end head ([1,300,6], YOLO26/YOLOv10 only); default is the raw "
                         "head with NMS in the app, which works for every YOLO version")
    args = ap.parse_args(argv)

    if args.precision in ("int8", "w8a16") and not args.data:
        ap.error(f"--precision {args.precision} needs --data <dataset.yaml> for calibration images")
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else list(args.imgsz)

    from ultralytics import YOLO, __version__ as ul_version

    model = YOLO(str(args.weights))
    quantize = {"fp32": None, "w8a32": "w8a32", "int8": 8, "w8a16": "w8a16"}[args.precision]
    kwargs = dict(format="litert", imgsz=imgsz)
    if args.end2end:
        kwargs["nms"] = False  # Ultralytics: nms=False selects the NMS-free one-to-one head
    if quantize is not None:
        kwargs["quantize"] = quantize
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
    if names == ["item"]:  # models trained with single_cls=True lose the class name
        names = ["eur_pallet_base"]
    targets = ["eur_pallet_base"] if "eur_pallet_base" in names else names

    sidecar = {
        "name": args.name,
        "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "output_format": fmt,
        "coordinates": coordinates,
        "input_width": int(det.in_w),
        "input_height": int(det.in_h),
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
        # Compare boxes, not score calibration: each side above --conf must be found by the
        # other side above conf/2. A decoding/layout error gives ~0 matches.
        lo = args.conf / 2
        found_ref = total_ref = found_tfl = total_tfl = 0
        for img in check_imgs:
            ref = [d for d in torch_predictions(model, img, imgsz, lo) if names[d[5]] in targets]
            got = det.detect(img, conf=lo)
            ref_hi = [d for d in ref if d[4] >= args.conf]
            got_hi = [d for d in got if d[4] >= args.conf]
            found_ref += match_rate(ref_hi, got)[0]
            total_ref += len(ref_hi)
            found_tfl += match_rate(got_hi, ref)[0]
            total_tfl += len(got_hi)
        recall = found_ref / total_ref if total_ref else 1.0
        precision = found_tfl / total_tfl if total_tfl else 1.0
        print(f"export check on {len(check_imgs)} images: {found_ref}/{total_ref} PyTorch boxes found by TFLite, "
              f"{found_tfl}/{total_tfl} TFLite boxes found by PyTorch")
        sidecar["metrics"] = {"export_recall_vs_pytorch": round(recall, 4), "export_precision_vs_pytorch": round(precision, 4)}
        if min(recall, precision) < 0.8 and max(total_ref, total_tfl) >= 5:
            print("WARNING: TFLite detections differ from PyTorch. With --end2end a short training can leave the "
                  "one-to-one head poorly calibrated; otherwise check --imgsz and precision (try fp32).")

    args.out.mkdir(parents=True, exist_ok=True)
    target_model = args.out / f"{args.name}.tflite"
    shutil.copy2(exported, target_model)
    (args.out / f"{args.name}.json").write_text(json.dumps(sidecar, indent=2))
    size_mb = target_model.stat().st_size / 1e6
    print(f"wrote {target_model} ({size_mb:.1f} MB) and {args.out / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
