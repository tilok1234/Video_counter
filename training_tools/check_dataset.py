#!/usr/bin/env python3
"""Validate a YOLO-format dataset before training and print useful statistics.

Checks every label file for format errors (wrong number of values, class ids, coordinates
outside 0..1, zero-size boxes), finds suspicious annotations (tiny boxes, duplicated
boxes), and warns about leakage between splits (frames of the same video in more than one
split, or near-identical images across splits).

Examples::

    python training_tools/check_dataset.py --dataset dataset
    python training_tools/check_dataset.py --dataset dataset --render /tmp/label_check   # draw boxes

Exit status 1 if any label file is malformed.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
GROUP_RE = re.compile(r"^(?P<group>.+)_f\d+$")


def parse_label_file(path: Path, num_classes: int) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes, errors = [], []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"{path.name}:{n}: expected 5 values, got {len(parts)} (segmentation labels are not supported)")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"{path.name}:{n}: not numeric")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"{path.name}:{n}: class {cls} outside 0..{num_classes - 1}")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"{path.name}:{n}: coordinates must be normalized (0..1): {x} {y} {w} {h}")
        elif x - w / 2 < -0.01 or x + w / 2 > 1.01 or y - h / 2 < -0.01 or y + h / 2 > 1.01:
            errors.append(f"{path.name}:{n}: box extends outside the image")
        boxes.append((cls, x, y, w, h))
    return boxes, errors


def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def dhash(path: Path) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def render(img_path: Path, boxes, out_dir: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    for k, (_, x, y, bw, bh) in enumerate(boxes, start=1):
        p1 = (int((x - bw / 2) * w), int((y - bh / 2) * h))
        p2 = (int((x + bw / 2) * w), int((y + bh / 2) * h))
        cv2.rectangle(img, p1, p2, (0, 200, 255), max(2, w // 400))
        cv2.putText(img, str(k), (p1[0] + 4, p1[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / img_path.name), img)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--num-classes", type=int, default=1)
    ap.add_argument("--tiny", type=float, default=0.0004, help="flag boxes smaller than this area fraction")
    ap.add_argument("--render", type=Path, help="write images with drawn boxes to this folder")
    ap.add_argument("--no-hash", action="store_true", help="skip the near-duplicate check across splits")
    args = ap.parse_args(argv)

    errors: list[str] = []
    warnings: list[str] = []
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    hashes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    all_w, all_h, per_image = [], [], []
    for split in SPLITS:
        img_dir = args.dataset / "images" / split
        lbl_dir = args.dataset / "labels" / split
        images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if img_dir.exists() else []
        n_boxes = negatives = missing = 0
        for img in images:
            label = lbl_dir / (img.stem + ".txt")
            m = GROUP_RE.match(img.stem)
            groups_by_split[split].add(m.group("group") if m else img.stem)
            if not label.exists():
                missing += 1
                warnings.append(f"{split}/{img.name}: no label file (treated as negative by the trainer)")
                boxes = []
            else:
                boxes, errs = parse_label_file(label, args.num_classes)
                errors += [f"{split}/{e}" for e in errs]
            n_boxes += len(boxes)
            negatives += not boxes
            per_image.append(len(boxes))
            for i, b in enumerate(boxes):
                all_w.append(b[3])
                all_h.append(b[4])
                if b[3] * b[4] < args.tiny:
                    warnings.append(f"{split}/{img.name}: tiny box #{i + 1} ({b[3]:.3f} x {b[4]:.3f})")
                for j in range(i + 1, len(boxes)):
                    if iou_xywh(b, boxes[j]) > 0.9:
                        warnings.append(f"{split}/{img.name}: boxes #{i + 1} and #{j + 1} are duplicates")
            if not args.no_hash:
                h = dhash(img)
                if h is not None:
                    hashes[split].append((h, img.name))
            if args.render:
                render(img, boxes, args.render / split)
        orphan_labels = [p for p in lbl_dir.glob("*.txt")] if lbl_dir.exists() else []
        orphans = [p.name for p in orphan_labels if not any((img_dir / (p.stem + e)).exists() for e in IMAGE_EXTENSIONS)]
        if orphans:
            warnings.append(f"{split}: {len(orphans)} label files without an image, e.g. {orphans[0]}")
        print(f"{split:5s}: {len(images):5d} images, {n_boxes:6d} boxes, {negatives:4d} negatives, "
              f"{missing} without label file")

    if all_w:
        q = lambda v: np.percentile(v, [5, 50, 95])  # noqa: E731
        w5, w50, w95 = q(all_w)
        h5, h50, h95 = q(all_h)
        print(f"box width  (fraction of image) p5/p50/p95: {w5:.3f} / {w50:.3f} / {w95:.3f}")
        print(f"box height (fraction of image) p5/p50/p95: {h5:.3f} / {h50:.3f} / {h95:.3f}")
        print(f"boxes per image: mean {np.mean(per_image):.2f}, max {max(per_image)}")

    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                shared = groups_by_split[a] & groups_by_split[b]
                if shared:
                    warnings.append(f"leakage: {len(shared)} video group(s) in both {a} and {b}, e.g. {sorted(shared)[0]}")
    if not args.no_hash:
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if not hashes[a] or not hashes[b]:
                continue
            ha = np.array([h for h, _ in hashes[a]], dtype=np.uint64)
            near = 0
            example = None
            for h, name in hashes[b]:
                x = np.bitwise_xor(ha, np.uint64(h))
                dist = np.array([bin(int(v)).count("1") for v in x])
                if dist.min() <= 3:
                    near += 1
                    example = example or name
            if near:
                warnings.append(f"{near} images in {b} look nearly identical to {a} images, e.g. {example}")

    for w in warnings[:50]:
        print("warning:", w)
    if len(warnings) > 50:
        print(f"... and {len(warnings) - 50} more warnings")
    for e in errors[:100]:
        print("ERROR:", e)
    if not any((args.dataset / "images" / s).exists() for s in SPLITS):
        print("ERROR: no dataset/images/{train,val,test} folders found")
        return 1
    print("OK" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
