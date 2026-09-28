# EUR-Pallet Counter

Android app that counts **wooden EUR-pallets underneath stacks of black plastic
half-pallets** from a live video sweep: walk along a line with the phone pointed at the
stack bases, the app detects each wooden base with an on-device model, tracks it across
frames and counts it once when it crosses a virtual count line. Then:

```
EURO pallets 17  ×  stack size 30  ×  one side ×2  =  1020 half-pallets
```

Everything runs offline on the phone. The app has no internet permission; camera frames
are discarded after processing unless you explicitly save a sample.

## Status — what works and what needs your data

| Part | Status |
|---|---|
| Tracking + counting core (Kotlin) | **Implemented and tested**: unit tests, 5-seed simulated sweeps (noise, blackouts, stops, reversals, re-scans, fast walking, camera jerks, far-row slivers), golden tests on real exported model outputs, replay of real model output on a synthetic walk video |
| Android app | **Implemented, builds in CI** (debug APK artifact). Camera sweep, live overlay + debug HUD, review with ±1 correction, photo mode, video replay, sample capture, model import, settings. Instrumented tests on an emulator in CI. Not yet tried on a physical phone by me |
| Training / export tooling (Python) | **Implemented and verified** end to end on synthetic data with the current Ultralytics/LiteRT versions (train → LiteRT export → export check → decode → video → count) |
| Pallet detector model | **Does not exist yet.** No pretrained model detects these bases (tested, see [MODEL_TRAINING.md](docs/MODEL_TRAINING.md#step-0-do-pretrained-models-help)). It must be trained on your own labeled footage — the tools and guides for that are here |
| SIMULATION detector | Plumbing test only: synthetic boxes, camera image ignored, red banner. **Not computer vision** |

## Getting the app

* **From CI:** GitHub → Actions → latest `ci` run → artifact **pallet-counter-debug-apk**
  (a zip with the APK). Install on the phone (allow "install unknown apps").
* **Build yourself:** Android Studio (recent) → open the repository → run `app`. Or
  `./gradlew :app:assembleDebug` → `app/build/outputs/apk/debug/app-debug.apk`.
  Requirements: JDK 17+, Android SDK (compileSdk 37), minSdk 26 (Android 8.0).

Without a model the app starts in "No trained model installed". Settings → Detector →
**SIMULATION** lets you try the whole flow (it fakes a pallet line; the count is not real).

## Using the app

1. **Stack size** 20 / 30 and **line mode** *one side ×2* / *whole line ×1*.
2. **START VIDEO SWEEP.** Landscape recommended (wider view = pallets stay visible longer).
3. Start **before the first pallet**. Keep the wooden bases inside the dashed band
   ("KEEP PALLET BASES IN THIS AREA"), walk steadily. Stopping or stepping back is fine —
   walking back un-counts, walking forward counts again.
4. Watch for **MOVING TOO FAST**. Press **FINISH** after the last pallet crossed the yellow line.
5. **Review:** check the thumbnails of the counted pallets, correct with −1 / +1, change
   stack size or line mode if needed, **Accept** (saved to the local history) or **Rescan**.

Other modes: **Replay video** (run a recorded video through the same pipeline, optionally
with the true count for comparison), **Photo mode** (single image, tap boxes to exclude).
For multi-tier stacks choose Settings → Guide band → *Full height* so every visible
wooden layer is counted.

## Building the detector (your part)

```
film lines ─► extract frames ─► label bases ─► split ─► train ─► evaluate ─► export
    ▲                                                                          │
    └──── capture failures in the app ◄── test on videos / phone ◄── import ◄──┘
```

| Step | Guide / command |
|---|---|
| Film | [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) |
| Extract frames | `python training_tools/extract_frames.py videos/ --out dataset/candidates --fps 2` |
| Label | [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md) (one class: `eur_pallet_base`) |
| Split + check | `split_dataset.py --src labeled/ --dataset dataset`, `check_dataset.py --dataset dataset` |
| Train | `python training/train.py --dataset dataset --model yolo11n.pt --name v1` — [docs/MODEL_TRAINING.md](docs/MODEL_TRAINING.md) |
| Evaluate | `python training/evaluate.py --weights runs/detect/v1/weights/best.pt --dataset dataset` |
| Export | `python training/export_model.py --weights … --imgsz 416 --out model/` |
| Install on phone | Settings → Import .tflite / .json — [docs/ANDROID_MODEL_INTEGRATION.md](docs/ANDROID_MODEL_INTEGRATION.md) |
| Test counting offline | `predict_video.py` + `pallet-core eval` — [docs/TESTING.md](docs/TESTING.md) |

First model: 300–500 labeled frames from 15+ videos (5-minute smoke test of the toolchain
on synthetic data first: see MODEL_TRAINING.md).

## Repository layout

```
app/              Android app (Kotlin, Compose, CameraX, LiteRT)
core/             Pure Kotlin: detection types, YOLO decoding, tracker, counter, replay CLI
training/         Python: train, export, evaluate, predict_video (+ palletlib helpers)
training_tools/   Python: extract_frames, split_dataset, check_dataset, prelabel, synthetic data
dataset/          Dataset folder structure (contents git-ignored)
model/            Exported models go here (git-ignored)
docs/             Architecture, data collection, annotation, training, integration, testing, failure cases
```

Design and data flow: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Failure cases and how they are handled/diagnosed: [docs/FAILURE_CASES.md](docs/FAILURE_CASES.md).

## Development

```bash
./gradlew -PcoreOnly :core:test                 # core tests, no Android SDK needed
./gradlew :app:assembleDebug                    # APK
./gradlew :app:connectedDebugAndroidTest        # on a connected phone/emulator
python -m pytest                                # Python tools (pip install -r training_tools/requirements.txt pytest)
./gradlew -PcoreOnly :core:installDist && core/build/install/pallet-core/bin/pallet-core --help
```

Versions: Gradle 9.8, AGP 9.4.1, Kotlin 2.4.20, Compose BOM 2026.09.00, CameraX 1.6.2,
LiteRT 1.4.2, Ultralytics 8.4.164.

## Privacy

This repository is **public**. `.gitignore` keeps workplace footage, frames, labels,
captures and model weights out of git — keep it that way. The app stores captures and
detection logs only in its private storage and exports them only when you choose to.
