"""Python mirror of the app's letterbox + YOLO output decoding (see YoloDecoder.kt)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RAW_CHANNELS_FIRST = "yolo_raw"
RAW_CHANNELS_LAST = "yolo_raw_channels_last"
END_TO_END = "yolo_end2end"


def _round_half_up(x: float) -> int:
    # Kotlin's roundToInt(); Python's round() is banker's rounding.
    return int(math.floor(x + 0.5))


@dataclass
class Letterbox:
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float = field(init=False)
    scaled_w: int = field(init=False)
    scaled_h: int = field(init=False)
    pad_left: int = field(init=False)
    pad_top: int = field(init=False)

    def __post_init__(self) -> None:
        self.scale = min(self.dst_w / self.src_w, self.dst_h / self.src_h)
        self.scaled_w = min(max(_round_half_up(self.src_w * self.scale), 1), self.dst_w)
        self.scaled_h = min(max(_round_half_up(self.src_h * self.scale), 1), self.dst_h)
        self.pad_left = (self.dst_w - self.scaled_w) // 2
        self.pad_top = (self.dst_h - self.scaled_h) // 2

    def input_pixels_to_source(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        return (
            (x1 - self.pad_left) / self.scaled_w,
            (y1 - self.pad_top) / self.scaled_h,
            (x2 - self.pad_left) / self.scaled_w,
            (y2 - self.pad_top) / self.scaled_h,
        )


def letterbox_image(img_rgb: np.ndarray, dst_w: int, dst_h: int, pad_value: int = 114) -> tuple[np.ndarray, Letterbox]:
    """Aspect-preserving resize into a (dst_h, dst_w, 3) uint8 canvas, centred."""
    import cv2

    h, w = img_rgb.shape[:2]
    lb = Letterbox(w, h, dst_w, dst_h)
    resized = cv2.resize(img_rgb, (lb.scaled_w, lb.scaled_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((dst_h, dst_w, 3), pad_value, dtype=np.uint8)
    canvas[lb.pad_top:lb.pad_top + lb.scaled_h, lb.pad_left:lb.pad_left + lb.scaled_w] = resized
    return canvas, lb


def expected_anchor_count(in_w: int, in_h: int, strides=(8, 16, 32)) -> int:
    return sum(((in_h + s - 1) // s) * ((in_w + s - 1) // s) for s in strides)


def resolve_layout(shape, in_w: int, in_h: int, requested: str = "auto") -> tuple[str, int, int]:
    """Returns (format, num_anchors, num_channels) exactly like YoloLayout.resolve."""
    dims = list(shape)
    if len(dims) == 3:
        if dims[0] != 1:
            raise ValueError(f"only batch size 1 is supported, got {shape}")
        dims = dims[1:]
    if len(dims) != 2:
        raise ValueError(f"unsupported YOLO output shape {shape}")
    a, b = dims
    if requested == RAW_CHANNELS_FIRST:
        return RAW_CHANNELS_FIRST, b, a
    if requested == RAW_CHANNELS_LAST:
        return RAW_CHANNELS_LAST, a, b
    if requested == END_TO_END:
        return END_TO_END, a, b
    expected = expected_anchor_count(in_w, in_h)
    if b == expected and a >= 5:
        return RAW_CHANNELS_FIRST, b, a
    if a == expected and b >= 5:
        return RAW_CHANNELS_LAST, a, b
    if b == 6 and a < expected:
        return END_TO_END, a, b
    if 5 <= a < b:
        return RAW_CHANNELS_FIRST, b, a
    if 5 <= b < a:
        return RAW_CHANNELS_LAST, a, b
    raise ValueError(f"cannot infer YOLO layout from {shape} for input {in_w}x{in_h}")


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list, iou_threshold: float, max_det: int = 100, class_agnostic: bool = True) -> list:
    dets = sorted(dets, key=lambda d: -d[4])[:1000]
    kept: list = []
    suppressed = [False] * len(dets)
    for i, a in enumerate(dets):
        if suppressed[i]:
            continue
        kept.append(a)
        if len(kept) >= max_det:
            break
        for j in range(i + 1, len(dets)):
            if suppressed[j] or (not class_agnostic and dets[j][5] != a[5]):
                continue
            if iou(a, dets[j]) > iou_threshold:
                suppressed[j] = True
    return kept


def decode(output: np.ndarray, lb: Letterbox, conf: float = 0.1, iou_threshold: float = 0.5,
           fmt: str = "auto", coords: str = "auto", target_classes: set[int] | None = None,
           max_det: int = 100, nms_end_to_end: bool = False) -> list[tuple[float, float, float, float, float, int]]:
    """Decode a YOLO output tensor into (left, top, right, bottom, score, class) tuples in
    normalized source-image coordinates, clipped to [0, 1]."""
    shape = output.shape
    fmt, n, c = resolve_layout(shape, lb.dst_w, lb.dst_h, fmt)
    flat = output.reshape(-1).astype(np.float32)
    cands = []
    if fmt == END_TO_END:
        rows = flat[: n * c].reshape(n, c)
        for r in rows:
            score, cls = float(r[4]), int(round(float(r[5])))
            if score < conf or (target_classes is not None and cls not in target_classes):
                continue
            cands.append((float(r[0]), float(r[1]), float(r[2]), float(r[3]), score, cls))
    else:
        mat = flat[: n * c].reshape(c, n).T if fmt == RAW_CHANNELS_FIRST else flat[: n * c].reshape(n, c)
        scores = mat[:, 4:]
        if target_classes is not None:
            mask = np.full(scores.shape[1], -1.0, dtype=np.float32)
            idx = [k for k in target_classes if k < scores.shape[1]]
            mask[idx] = 0.0
            scores = np.where(mask >= 0, scores, -1.0)
        best_cls = scores.argmax(axis=1)
        best = scores[np.arange(len(scores)), best_cls]
        for i in np.nonzero(best >= conf)[0]:
            cx, cy, w, h = (float(v) for v in mat[i, :4])
            cands.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, float(best[i]), int(best_cls[i])))
    if not cands:
        return []
    max_coord = max(max(v[0], v[1], v[2], v[3]) for v in cands)
    normalized = coords == "normalized" or (coords == "auto" and max_coord <= 2.0)
    sx = lb.dst_w if normalized else 1.0
    sy = lb.dst_h if normalized else 1.0
    dets = []
    for x1, y1, x2, y2, s, k in cands:
        l, t, r, b = lb.input_pixels_to_source(x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        l, t, r, b = (min(max(v, 0.0), 1.0) for v in (l, t, r, b))
        if r - l < 1e-3 or b - t < 1e-3:
            continue
        dets.append((l, t, r, b, s, k))
    if fmt == END_TO_END and not nms_end_to_end:
        return sorted(dets, key=lambda d: -d[4])[:max_det]
    return nms(dets, iou_threshold, max_det)
