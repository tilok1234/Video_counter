#!/usr/bin/env python3
"""Run a detector over a recorded video and write a detection log for the replay CLI.

The log uses the same JSONL format the app records, so the video can then be counted with
exactly the app's tracking + counting code on a PC::

    python training/predict_video.py line_07.mp4 --model model/eur_pallet.tflite --expected 18 --out logs/line_07.jsonl
    ./gradlew -PcoreOnly :core:installDist
    core/build/install/pallet-core/bin/pallet-core replay logs/line_07.jsonl
    core/build/install/pallet-core/bin/pallet-core eval logs/          # many videos at once

Use the exported ``.tflite`` to reproduce the phone as closely as possible (same
letterbox, normalisation and decoding). ``.pt`` weights work too (via Ultralytics).
Frames are sampled at ``--fps`` (the phone typically runs the detector at 5-15 fps).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palletlib import detlog  # noqa: E402


def rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    import cv2

    return {90: lambda f: cv2.rotate(f, cv2.ROTATE_90_CLOCKWISE),
            180: lambda f: cv2.rotate(f, cv2.ROTATE_180),
            270: lambda f: cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE)}.get(degrees, lambda f: f)(frame)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", type=Path, required=True, help=".tflite (recommended) or .pt")
    ap.add_argument("--out", type=Path, help="output .jsonl (default: next to the video)")
    ap.add_argument("--fps", type=float, default=10.0, help="detector frames per second to simulate")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="decoder threshold; keep low like the app (the tracker uses low-score boxes)")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size for .pt models")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="rotate frames clockwise")
    ap.add_argument("--expected", type=int, help="true pallet count (stored in the log for `eval`)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after this much video (0 = all)")
    ap.add_argument("--render", type=Path, help="optional .mp4 with raw detections drawn")
    args = ap.parse_args(argv)

    import cv2

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 2
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1.0, native / args.fps)
    out = args.out or args.video.with_suffix(".jsonl")

    if args.model.suffix == ".tflite":
        from palletlib.tflite_runner import TfliteDetector

        det = TfliteDetector(args.model)
        info = {"name": det.sidecar.get("name", args.model.stem), "kind": "tflite-yolo",
                "inputWidth": det.in_w, "inputHeight": det.in_h, "accelerator": "CPU (python)"}
        detect = lambda rgb: det.detect(rgb, conf=args.conf)  # noqa: E731
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.model))
        info = {"name": args.model.stem, "kind": "ultralytics", "inputWidth": args.imgsz,
                "inputHeight": args.imgsz, "accelerator": "python"}

        def detect(rgb):
            r = model.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            return [tuple(b) + (s, int(c)) for b, s, c in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

    writer = None
    frames = 0
    total_ms = 0.0
    next_index = 0.0
    index = -1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        fh.write(detlog.header("python-predict", info, args.expected, args.video.name) + "\n")
        while True:
            if not cap.grab():
                break
            index += 1
            if index + 1e-6 < next_index:
                continue
            next_index += step
            ok, frame = cap.retrieve()
            if not ok:
                continue
            t = index / native
            if args.max_seconds and t > args.max_seconds:
                break
            frame = rotate(frame, args.rotate)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            dets = detect(rgb)
            ms = (time.perf_counter() - t0) * 1000
            total_ms += ms
            h, w = frame.shape[:2]
            fh.write(detlog.frame(int(round(t * 1e9)), frames, dets, w, h, ms) + "\n")
            frames += 1
            if args.render:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.render), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                for l, tp, r, b, s, _ in dets:
                    if s >= 0.3:
                        cv2.rectangle(frame, (int(l * w), int(tp * h)), (int(r * w), int(b * h)), (0, 200, 255), 2)
                        cv2.putText(frame, f"{s:.2f}", (int(l * w), int(tp * h) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()
    print(f"wrote {out}: {frames} frames at ~{args.fps:g} fps, mean inference {total_ms / max(1, frames):.1f} ms (PC)")
    print(f"next: core/build/install/pallet-core/bin/pallet-core replay {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
