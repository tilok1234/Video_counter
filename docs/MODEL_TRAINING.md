# Model training

Train → evaluate → export → test on video → find failures → add examples → retrain.
Everything runs locally; no image leaves your machine.

## Step 0: do pretrained models help?

Checked on 2026-09-28 with three sample photos of pallet stacks (run
`training/try_pretrained.py` on your own photos to repeat it):

| Model | Setting | Result |
|---|---|---|
| YOLO11n, COCO (80 everyday classes, no "pallet") | conf ≥ 0.15 | 0 detections on all photos |
| YOLO11n, COCO | conf ≥ 0.03 | only spurious "car" (0.04–0.08) and "train" (0.05–0.11) boxes |
| YOLOE-11S open-vocabulary, prompts "wooden pallet", "pallet", "euro pallet", "stack of plastic pallets" | conf ≥ 0.15 | 0 / 0 / 1 boxes; the one box is a black stack |
| YOLOE-11L open-vocabulary, same prompts | conf ≥ 0.03 | black stacks segmented as "stack of plastic pallets" (0.03–0.42); only 3 "wooden pallet" boxes (0.06–0.10), none on an actual base |

**Conclusion:** nothing off the shelf localizes the wooden bases, not even for
pre-labeling. A custom detector trained on your footage is required. COCO-pretrained
weights are still the right *starting point* (transfer learning): they already know edges,
textures and objects, so a few hundred labeled frames go a long way.

## Environment

* Python 3.10–3.12, **Linux x86-64 or macOS**. The LiteRT export used by Ultralytics
  (`litert_torch`) does not run on Windows — use **WSL2 (Ubuntu)** there.
* A CUDA GPU makes training ~20× faster but is optional (see timings below).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
```

The requirements are **pinned to a set verified together** (Ultralytics 8.4.164, torch
2.13.0, torchvision 0.28.0, litert-torch 0.9.4, ai-edge-litert 2.2.0). Install them before
the first export: otherwise Ultralytics auto-installs `litert-torch` during the export,
which downgraded torch under the running process and crashed it in our tests.

## Smoke test (5 minutes, synthetic data)

Before labeling anything, check the whole toolchain on your machine:

```bash
python training_tools/make_synthetic_dataset.py --out /tmp/synth
python training/train.py --dataset /tmp/synth --epochs 5 --imgsz 320 --name smoke
python training/export_model.py --weights runs/detect/smoke/weights/best.pt --imgsz 320 \
    --check-images /tmp/synth/images/test --out /tmp/smoke_model --name smoke
```

The export must print `export check … N/N PyTorch boxes found by TFLite`. The synthetic
model is useless on real footage; it only proves the pipeline works.

## Which model

| Model | Notes |
|---|---|
| **`yolo11n.pt` (default)** | Small (2.6 M params), fast, export verified end to end (raw head + app NMS). |
| `yolo26n.pt` | Newest Ultralytics edge model; also exports to the raw head by default (verified). `--end2end` export gives an NMS-free head (verified), but needs a well-trained model for calibrated scores. |
| `yolo11s.pt` | ~3.5× slower, a bit more accurate. Consider for mid/high-end phones. |

Licence: Ultralytics YOLO is AGPL-3.0 (commercial licences available from Ultralytics).
For an internal tool this is usually unproblematic, but check with your employer before
distributing the app. The app does not depend on Ultralytics at runtime: any detector that
exports a compatible LiteRT model (or a new `PalletDetector`) can be swapped in.

## Train

```bash
python training/train.py --dataset dataset --model yolo11n.pt --imgsz 640 --epochs 150 --name v1
```

* `--imgsz 640` for training even if you export smaller; small, far pallets benefit.
* Stops early after 40 epochs without improvement (`--patience`).
* Augmentation (in `train.py`): strong brightness/saturation changes (sun, shadows, wet
  wood), small rotations/shear/perspective, horizontal flips, mosaic; **no vertical flips**
  (pallets are always under the stacks).
* Output: `runs/detect/v1/weights/best.pt`, training curves and sample predictions in
  `runs/detect/v1/` (git-ignored).
* Timings (rough): 500 images × 150 epochs at 640 ≈ 2–3 h on a laptop CPU, 10–15 min on a
  mid-range NVIDIA GPU.
* Next round: start from your previous model (`--model runs/detect/v1/weights/best.pt`)
  after adding new labeled data.

Data must first be split and checked — see [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md).

## Evaluate

```bash
python training/evaluate.py --weights runs/detect/v1/weights/best.pt --dataset dataset --render /tmp/eval_v1
```

* mAP50 / mAP50-95 / precision / recall on the **test** split (videos never seen in training).
* Per-image analysis at the operating threshold: false positives and misses per image and
  the worst images; `--render` draws them (green correct, red false positive, magenta missed).
* What matters for counting: **recall of front-row bases** (a missed pallet over several
  frames = a missed count) and **persistent false positives** (a false box that is tracked
  across the line = an extra count). Single-frame errors are largely absorbed by tracking.
* The real benchmark is counting on test videos ([TESTING.md](TESTING.md)).

## Export for Android

```bash
python training/export_model.py --weights runs/detect/v1/weights/best.pt --imgsz 416 \
    --check-images dataset/images/test --out model/ --name eur_pallet
```

* Writes `model/eur_pallet.tflite` and the sidecar `model/eur_pallet.json` (layout,
  coordinate space, input size, classes, suggested threshold, export check metrics).
* **Input size:** 320–416 for budget phones (e.g. Galaxy A04s-class), 640 for newer phones.
  Smaller is faster but misses small/distant bases. Try 416 first and read the HUD's
  *infer ms* on the phone.
* **Precision:** `fp32` (default; the GPU delegate runs it in FP16 anyway), `w8a32`
  (int8 weights, ~4× smaller file, faster on CPU), `int8`/`w8a16` (static quantization,
  needs `--data dataset/dataset.yaml`). Always compare the export check and a test video
  before switching precision.
* The exported model takes **NCHW float32** input and outputs the raw YOLO head
  `[1, 5, anchors]` with normalized boxes (or `[1, 300, 6]` pixel boxes with `--end2end`).
  The app reads both; see [ANDROID_MODEL_INTEGRATION.md](ANDROID_MODEL_INTEGRATION.md).
* `export check` compares TFLite (decoded exactly like the app) with PyTorch. If far fewer
  boxes match, do not deploy that file.

## Iterate

1. Scan real lines; use MISSED / FALSE + / BAD TRACK capture buttons on failures.
2. Record new videos of conditions that fail (see [FAILURE_CASES.md](FAILURE_CASES.md)).
3. Extract frames, pre-label with the current model (`training_tools/prelabel.py`), correct.
4. Add to the dataset (`split_dataset.py` again with the new folder), check, retrain from
   the last `best.pt`, evaluate, export, run the test videos, deploy if counts improved.

Keep a small table of model versions with their test-video results; the sidecar `version`
and `metrics` fields and the app's history make it easy to see which model produced what.

## Developer tools

* `training/make_golden_fixture.py` — records raw output of an exported model for the
  Kotlin decoder tests (synthetic images only; fixtures are committed).
* `training/make_android_test_models.py` — builds the tiny models used by the Android
  instrumented tests.
