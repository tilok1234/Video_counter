"""Run an exported .tflite detector the same way the Android app does."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .yolo_decode import Letterbox, decode, letterbox_image


def load_interpreter(model_path: Path, num_threads: int = 4):
    try:
        from ai_edge_litert.interpreter import Interpreter  # LiteRT (Linux/macOS)
    except ImportError:  # pragma: no cover - fallback for platforms without LiteRT wheels
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interp.allocate_tensors()
    return interp


def read_sidecar(model_path: Path) -> dict:
    side = Path(model_path).with_suffix(".json")
    return json.loads(side.read_text()) if side.exists() else {}


class TfliteDetector:
    def __init__(self, model_path: Path, num_threads: int = 4):
        self.model_path = Path(model_path)
        self.sidecar = read_sidecar(self.model_path)
        self.interp = load_interpreter(self.model_path, num_threads)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        shape = list(self.inp["shape"])
        self.nchw = shape[1] == 3 and shape[3] != 3
        self.in_h, self.in_w = (shape[2], shape[3]) if self.nchw else (shape[1], shape[2])
        self.pad = int(self.sidecar.get("letterbox_pad_value", 114))
        classes = self.sidecar.get("classes", ["eur_pallet_base"])
        targets = self.sidecar.get("target_classes", ["eur_pallet_base"])
        ids = {classes.index(t) for t in targets if t in classes}
        self.target_classes = ids or ({0} if len(classes) <= 1 else None)

    def raw(self, img_rgb: np.ndarray) -> tuple[np.ndarray, Letterbox]:
        canvas, lb = letterbox_image(img_rgb, self.in_w, self.in_h, self.pad)
        x = canvas.astype(np.float32)
        if self.sidecar.get("input_normalization", "zero_one") != "none":
            x /= 255.0
        if self.nchw:
            x = x.transpose(2, 0, 1)
        x = x[None]
        dtype = self.inp["dtype"]
        if dtype in (np.int8, np.uint8):
            scale, zero = self.inp["quantization"]
            x = np.clip(np.round(x / scale + zero), np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        if y.dtype in (np.int8, np.uint8):
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y.astype(np.float32), lb

    def detect(self, img_rgb: np.ndarray, conf: float = 0.1, iou: float = 0.5):
        y, lb = self.raw(img_rgb)
        return decode(y, lb, conf=conf, iou_threshold=iou,
                      fmt=self.sidecar.get("output_format", "auto"),
                      coords=self.sidecar.get("coordinates", "auto"),
                      target_classes=self.target_classes)
