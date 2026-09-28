#!/usr/bin/env python3
"""Split labeled images into dataset/images/{train,val,test} and dataset/labels/{...}.

Frames from the same video are nearly identical, so they are split *by video* (group):
all frames of one video end up in the same split. Otherwise validation/test scores would
measure memorisation of neighbouring frames instead of generalisation to new lines.

Input folder layout (what annotation tools export): images plus YOLO ``.txt`` files with
the same name, either side by side or in a ``labels/`` subfolder::

    labeled/line_03_f000120.jpg   labeled/line_03_f000120.txt
    labeled/...                   (or labeled/labels/line_03_f000120.txt)

An image whose label file is *empty* is a negative example (no target pallet) and is kept.
An image with *no* label file is treated as not yet labeled and skipped, unless
``--missing-as-negative`` is given.

Examples::

    python training_tools/split_dataset.py --src labeled/ --dataset dataset --val 0.2 --test 0.1
    python training_tools/split_dataset.py --src labeled/ --dataset dataset --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
DEFAULT_GROUP_REGEX = r"^(?P<group>.+)_f\d+$"


def group_of(stem: str, pattern: re.Pattern[str]) -> str:
    m = pattern.match(stem)
    return m.group("group") if m else stem


def find_pairs(src: Path, missing_as_negative: bool) -> tuple[list[tuple[Path, Path | None]], list[Path]]:
    pairs: list[tuple[Path, Path | None]] = []
    unlabeled: list[Path] = []
    for img in sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        candidates = [img.with_suffix(".txt"), img.parent / "labels" / (img.stem + ".txt"),
                      src / "labels" / (img.stem + ".txt")]
        label = next((c for c in candidates if c.exists()), None)
        if label is None and not missing_as_negative:
            unlabeled.append(img)
            continue
        pairs.append((img, label))
    return pairs, unlabeled


def assign_groups(groups: dict[str, list], val: float, test: float, seed: int) -> dict[str, str]:
    """Greedy assignment of whole groups to splits, approximating the requested ratios."""
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    total = sum(len(v) for v in groups.values())
    targets = {"test": test * total, "val": val * total}
    counts = {s: 0 for s in SPLITS}
    assignment: dict[str, str] = {}
    for name in names:
        size = len(groups[name])
        if counts["test"] < targets["test"] and counts["test"] + size / 2 <= targets["test"] + 1e-9:
            split = "test"
        elif counts["val"] < targets["val"] and counts["val"] + size / 2 <= targets["val"] + 1e-9:
            split = "val"
        else:
            split = "train"
        assignment[name] = split
        counts[split] += size
    # Guarantee a non-empty val split when possible (training needs one).
    if val > 0 and counts["val"] == 0 and len(names) >= 2:
        smallest = min((n for n in names if assignment[n] == "train"), key=lambda n: len(groups[n]), default=None)
        if smallest is not None:
            assignment[smallest] = "val"
    return assignment


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder with labeled images + YOLO txt files")
    ap.add_argument("--dataset", type=Path, default=Path("dataset"), help="dataset root (default: dataset)")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX,
                    help="regex with a 'group' named group applied to the file stem (default: video name "
                         "before _f<frame>); images that do not match form their own group")
    ap.add_argument("--by-image", action="store_true", help="split individual images (NOT recommended for video frames)")
    ap.add_argument("--missing-as-negative", action="store_true",
                    help="treat images without a label file as negatives (empty label)")
    ap.add_argument("--move", action="store_true", help="move files instead of copying")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not 0 <= args.val < 1 or not 0 <= args.test < 1 or args.val + args.test >= 1:
        ap.error("--val and --test must be in [0, 1) and sum to < 1")
    pairs, unlabeled = find_pairs(args.src, args.missing_as_negative)
    if unlabeled:
        print(f"note: {len(unlabeled)} images have no label file and are skipped "
              f"(use --missing-as-negative if they are true negatives), e.g. {unlabeled[0].name}")
    if not pairs:
        print("no labeled images found", file=sys.stderr)
        return 2

    pattern = re.compile(args.group_regex)
    groups: dict[str, list[tuple[Path, Path | None]]] = defaultdict(list)
    for img, label in pairs:
        key = img.stem if args.by_image else group_of(img.stem, pattern)
        groups[key].append((img, label))
    if not args.by_image and len(groups) < 3:
        print(f"warning: only {len(groups)} group(s) (videos). A group-wise split needs at least 3; "
              "record more videos or use --by-image (validation scores will then be optimistic).")
    assignment = assign_groups(groups, args.val, args.test, args.seed)

    summary = {s: {"images": 0, "negatives": 0, "boxes": 0, "groups": []} for s in SPLITS}
    for name, items in groups.items():
        split = assignment[name]
        summary[split]["groups"].append(name)
        for img, label in items:
            text = label.read_text() if label else ""
            n_boxes = sum(1 for line in text.splitlines() if line.strip())
            summary[split]["images"] += 1
            summary[split]["boxes"] += n_boxes
            summary[split]["negatives"] += n_boxes == 0
            if args.dry_run:
                continue
            img_dir = args.dataset / "images" / split
            lbl_dir = args.dataset / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            op = shutil.move if args.move else shutil.copy2
            op(str(img), str(img_dir / img.name))
            target_label = lbl_dir / (img.stem + ".txt")
            if label is not None:
                op(str(label), str(target_label))
            else:
                target_label.write_text("")

    for s in SPLITS:
        d = summary[s]
        print(f"{s:5s}: {d['images']:5d} images ({d['negatives']} negatives), {d['boxes']:6d} boxes, "
              f"{len(d['groups'])} groups")
    if not args.dry_run:
        report = args.dataset / "split_report.json"
        report.write_text(json.dumps({"seed": args.seed, "val": args.val, "test": args.test,
                                      "assignment": assignment, "summary": summary}, indent=2))
        print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
