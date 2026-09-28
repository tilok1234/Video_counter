# Android model integration

## Installing a model

**On the phone, no rebuild (recommended while iterating):**

1. Copy `eur_pallet.tflite` and `eur_pallet.json` to the phone (USB, cloud drive you are
   allowed to use, …).
2. App → Settings → **Import .tflite**, then **Import .json**.
3. The model is validated by loading it; the Settings screen shows name, version, size,
   SHA-256 prefix, output format and classes. Detector mode switches to *Trained model*.

An imported model takes priority over a bundled one. *Remove imported model* reverts.

**Bundled in the APK:** put the files at

```
app/src/main/assets/models/eur_pallet.tflite
app/src/main/assets/models/eur_pallet.json
```

and rebuild. These paths are git-ignored — the public repository never contains your model.
Models are stored uncompressed in the APK (`noCompress += "tflite"`) and memory-mapped.

## Model contract

| | Supported |
|---|---|
| Runtime | LiteRT (TensorFlow Lite) `Interpreter`, `com.google.ai.edge.litert:litert:1.4.2` |
| Input | 1 image, `[1, 3, H, W]` (NCHW, current Ultralytics export) or `[1, H, W, 3]` (NHWC) |
| Input type | float32 (RGB, 0–1 unless the sidecar says `input_normalization: none`), int8/uint8 with quantization params |
| Preprocessing | upright frame, aspect-preserving letterbox, centred, padding value 114 (like Ultralytics) |
| Output | raw YOLO head `[1, 4+C, N]` or `[1, N, 4+C]` (cx, cy, w, h + class scores), or end-to-end `[1, N, 6]` (x1, y1, x2, y2, score, class) |
| Output type | float32 or int8/uint8 (dequantized) |
| Coordinates | normalized to the input size or input pixels (sidecar or auto-detected) |
| Classes | any; the classes listed in `target_classes` are counted (default: class 0) |

The decoder (`core/.../YoloDecoder.kt`) is tested against raw outputs recorded from real
exported models (`core/src/test/resources/golden`) and the tiny on-device test models.

## Sidecar JSON

Written by `training/export_model.py`; every field is optional (missing → auto).

```json
{
  "name": "eur_pallet",
  "version": "20260928-131623",
  "output_format": "yolo_raw",
  "coordinates": "normalized",
  "input_width": 416,
  "input_height": 416,
  "input_normalization": "zero_one",
  "letterbox_pad_value": 114,
  "classes": ["eur_pallet_base"],
  "target_classes": ["eur_pallet_base"],
  "confidence_threshold": 0.35,
  "quantization": "fp32",
  "framework": "ultralytics 8.4.164",
  "metrics": {"export_recall_vs_pytorch": 1.0, "export_precision_vs_pytorch": 1.0}
}
```

| Field | Values |
|---|---|
| `output_format` | `yolo_raw` (channels first), `yolo_raw_channels_last`, `yolo_end2end`, `auto` |
| `coordinates` | `normalized`, `pixels`, `auto` |
| `input_normalization` | `zero_one` (pixel / 255) or `none` (0–255) |
| `input_width/height` | informational; the interpreter's tensor shape is authoritative |
| `target_classes` | class names that are counted |
| `confidence_threshold` | suggested operating point from the export (photo mode default); the tracker uses its own thresholds |

## Choosing performance settings (Settings → Performance)

| Setting | Default | Notes |
|---|---|---|
| Accelerator | Auto | GPU delegate if the device supports it, otherwise CPU (XNNPACK). "GPU" forces an attempt, falls back to CPU on failure. |
| CPU threads | 4 | 2 on very hot/old phones. |
| Max detector rate | 10 fps | Tracking works from ~5 fps at walking speed; more is smoother, hotter. |
| Analysis resolution | 1280×960 | 640×480 saves conversion time; captures get smaller. |

Read the HUD while scanning: `infer … ms` (model time), `det … fps` (detector frames),
`cam … fps` (camera), `thermal`. If `det fps` is below ~5 at walking speed, export a
smaller input size (e.g. 416 → 320), try `w8a32`, or switch accelerator. When the phone
reports a SEVERE thermal state the app halves the detector rate automatically.

Budget phones (e.g. Samsung Galaxy A04s, Exynos 850) should start with a 320–416 input.
Actual latency depends on the phone, model and accelerator — measure it with the HUD;
the numbers here are guidance, not measurements.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| "No trained model installed" | Import a model, or switch detector to SIMULATION for plumbing tests. |
| Detector error on start | File is not a LiteRT model, or unsupported ops. Re-export with `export_model.py`; check the message in the banner. |
| No boxes at all | Wrong sidecar (import the matching `.json`), threshold too high, or model trained on different footage. Try *Show raw detections*. |
| Boxes shifted/scaled | Sidecar `coordinates` wrong (set `auto`), or `input_normalization` wrong. |
| Many boxes everywhere | Model not trained enough / wrong classes in `target_classes`. |
| Boxes fine, count wrong | Tracking/counting parameters — see [FAILURE_CASES.md](FAILURE_CASES.md) and replay the scan log on a PC ([TESTING.md](TESTING.md)). |

## Using a different model family or runtime

Implement `PalletDetector<FrameInput>` (see `TfliteYoloDetector` and `SimulatedDetector`):
return `DetectionFrame` with boxes in normalized upright-frame coordinates. Wire it into
`ModelManager.createDetector()`. Nothing else in the app changes.
