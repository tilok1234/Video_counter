#!/usr/bin/env python3
"""Evaluate a detector on the held-out split, with an eye on *counting* errors.

1. Standard detection metrics (mAP50, mAP50-95, precision, recall) via Ultralytics.
2. A per-image analysis at the operating threshold: predicted vs labelled box count,
   false positives and misses, so you can see *which* images fail and add similar
   examples to the training set. With ``--render`` those images are drawn:
   green = correct, red = false positive, magenta = missed label.

Works with ``.pt`` weights and exported ``.tflite`` models (the latter are run through the
same preprocessing/decoding as the Android app).

Examples::

    python training/evaluate.py --weights runs/detect/v1/weights/best.pt --dataset dataset
    python training/evaluate.py --weights model/eur_pallet.tflite --dataset dataset --render /tmp/eval_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palletlib.yolo_decode import iou  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                x, y, w, h = map(float, p[1:])
                boxes.append((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    return boxes


def match(preds: list[tuple], gts: list[tuple], thr: float) -> tuple[list[int], list[int], list[int]]:
    """Greedy matching by score. Returns (tp_pred_idx, fp_pred_idx, fn_gt_idx)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][4])
    used: set[int] = set()
    tp, fp = [], []
    for i in order:
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(preds[i], g)
            if v > best:
                best, best_j = v, j
        if best >= thr:
            used.add(best_j)
            tp.append(i)
        else:
            fp.append(i)
    return tp, fp, [j for j in range(len(gts)) if j not in used]


def make_predictor(weights: Path, imgsz: int):
    if weights.suffix == ".tflite":
        from palletlib.tflite_runner import TfliteDetector

        det = TfliteDetector(weights)
        return lambda rgb, conf: det.detect(rgb, conf=conf)
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(rgb, conf):
        import cv2

        res = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=imgsz, conf=conf, verbose=False)[0]
        return [tuple(b) + (s, int(c)) for b, s, c in zip(res.boxes.xyxyn.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]

    return predict


def render(rgb: np.ndarray, preds, gts, tp, fp, fn, out: Path) -> None:
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    lw = max(2, w // 400)

    def rect(b, color):
        cv2.rectangle(img, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), color, lw)

    for i in tp:
        rect(preds[i], (0, 200, 0))
    for i in fp:
        rect(preds[i], (0, 0, 255))
        cv2.putText(img, f"FP {preds[i][4]:.2f}", (int(preds[i][0] * w), int(preds[i][1] * h) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    for j in fn:
        rect(gts[j], (255, 0, 255))
        cv2.putText(img, "MISSED", (int(gts[j][0] * w), int(gts[j][3] * h) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help=".pt or .tflite")
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=640, help="inference size for .pt weights")
    ap.add_argument("--conf", type=float, default=0.5, help="operating threshold for the per-image analysis")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU for a prediction to count as correct")
    ap.add_argument("--render", type=Path, help="write images with errors here")
    ap.add_argument("--no-map", action="store_true", help="skip the Ultralytics mAP computation")
    ap.add_argument("--worst", type=int, default=15)
    args = ap.parse_args(argv)

    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"no images in {img_dir}", file=sys.stderr)
        return 2

    if not args.no_map:
        from ultralytics import YOLO

        yaml_path = args.dataset / "dataset.yaml"
        if not yaml_path.exists():
            print(f"{yaml_path} missing - run training/train.py once, or pass --no-map", file=sys.stderr)
        else:
            m = YOLO(str(args.weights)).val(data=str(yaml_path), split=args.split, imgsz=args.imgsz, verbose=False, plots=False)
            print(f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  precision {m.box.mp:.3f}  recall {m.box.mr:.3f}")

    from PIL import Image, ImageOps

    predict = make_predictor(args.weights, args.imgsz)
    rows = []
    totals = np.zeros(3, dtype=int)
    for path in images:
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        gts = read_labels(lbl_dir / (path.stem + ".txt"))
        preds = [p for p in predict(rgb, args.conf) if p[4] >= args.conf]
        tp, fp, fn = match(preds, gts, args.iou)
        totals += (len(tp), len(fp), len(fn))
        rows.append((path.name, len(gts), len(preds), len(fp), len(fn)))
        if args.render and (fp or fn):
            render(rgb, preds, gts, tp, fp, fn, args.render / path.name)

    exact = sum(1 for r in rows if r[1] == r[2] and r[3] == 0)
    abs_err = [abs(r[2] - r[1]) for r in rows]
    tp, fp, fn = totals
    print(f"\n{args.split}: {len(rows)} images at conf {args.conf}: {tp} correct, {fp} false positives, {fn} missed "
          f"(precision {tp / max(1, tp + fp):.3f}, recall {tp / max(1, tp + fn):.3f})")
    print(f"per-image box count exactly right: {exact}/{len(rows)}; mean |count error| {np.mean(abs_err):.2f}")
    worst = sorted(rows, key=lambda r: -(r[3] + r[4]))[: args.worst]
    worst = [r for r in worst if r[3] + r[4] > 0]
    if worst:
        print("\nworst images (labels / predicted / FP / missed):")
        for name, g, p, f, m in worst:
            print(f"  {name:45s} {g:3d} {p:3d} {f:3d} {m:3d}")
    if args.render:
        print(f"\nerror renders written to {args.render}")
    print("\nNote: per-image counts are not line counts. Test counting on videos with predict_video.py + "
          "the replay CLI (docs/TESTING.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
