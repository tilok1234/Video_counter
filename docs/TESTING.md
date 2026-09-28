# Testing

## Test layers

| Layer | What | Run |
|---|---|---|
| Core unit tests | geometry, letterbox, YOLO decoding, NMS, Hungarian, tracker, counter rules, count math, log format | `./gradlew -PcoreOnly :core:test` |
| Simulated sweeps | 16-pallet line walks with noisy detections: misses, blackouts, stop-and-go, reversals, re-scans, fast walking at 6 fps, camera jerks, both directions, far-row slivers — each over 5 random seeds | part of `:core:test` (`SimulatedSweepTest`) |
| Golden decoder tests | raw output tensors recorded from **real exported LiteRT models** (raw and end-to-end heads) must decode exactly like the Python reference and match Ultralytics' predictions | part of `:core:test` (`YoloGoldenTest`) |
| Real-model replay | detection log produced by `predict_video.py` with an exported model on a synthetic walk video (12 pallets, pause, walk back) must count 12 | part of `:core:test` (`PythonLogReplayTest`) |
| Python tools | frame extraction/dedup/rotation, grouped split, dataset checker, decoder mirror, log writer | `python -m pytest` |
| Android unit tests | settings → ROI/pipeline mapping, settings JSON compatibility | `./gradlew :app:testDebugUnitTest` |
| Android on-device | LiteRT with tiny test models (NCHW/NHWC, RGB order, normalisation, rotation, letterbox, GPU fallback); upright ↔ sensor pixel mapping; scan engine (count, safe double close, RESET splits the detection log and each part replays to its recorded count); full UI flow with emulated camera + simulation detector | `./gradlew :app:connectedDebugAndroidTest` (emulator/phone) |

CI (`.github/workflows/ci.yml`) runs all of them on every push: core tests, APK build
(downloadable artifact `pallet-counter-debug-apk`) with a check that the APK requests no
permission except the camera, Python tests and the instrumented tests on an API 34
emulator (the full UI flow three times, to catch intermittent failures; logcat crashes are
printed into the job log).

**What the tests cannot tell you:** how well a detector trained on your footage finds real
pallets. That is measured with real test videos, below.

## Offline counting evaluation ("expected 18 — app 18")

Record test videos that are **not used for training**, count each line by hand, and keep
the number with the video. Then evaluate any model/parameter combination without walking
the yard:

```bash
# 1. detections for every test video, using the exported model like the phone does
for v in testvideos/*.mp4; do
  n=$(basename "$v" .mp4 | sed -E 's/.*count([0-9]+).*/\1/')   # e.g. line07_count18.mp4
  python training/predict_video.py "$v" --model model/eur_pallet.tflite --fps 10 --expected "$n" \
      --out logs/$(basename "$v" .mp4).jsonl
done

# 2. build the replay tool once
./gradlew -PcoreOnly :core:installDist
PC=core/build/install/pallet-core/bin/pallet-core

# 3. count them with the app's tracking/counting code
$PC eval logs/
```

```
log                                      expected  counted   diff
line03_count16.jsonl                           16       16      0
line07_count18.jsonl                           18       17     -1
...
12 logs with ground truth: 11 exact, 1 wrong, mean abs error 0.08 pallets
```

`eval` exits with status 1 if any count is wrong, so it can gate a model release.

Inspect a single failure:

```bash
$PC replay logs/line07_count18.jsonl --events          # every count event with time + track id
$PC replay logs/line07_count18.jsonl --trace | less    # one line per frame: tracks and sides
```

Try parameters without re-running the model (detections are reused):

```bash
$PC eval logs/ --line 0.45 --hysteresis 0.05 --min-hits 2 --lost-ms 2000 --size-filter 0.75
$PC replay logs/x.jsonl --config my_pipeline.json      # full PipelineConfig JSON
```

When a parameter set is better on all logs, set the same values in the app's Settings.

## Evaluating on the phone

* **Replay video** (setup screen): pick a recorded video, choose the detector rate and
  rotation, optionally enter the true count. The video runs through exactly the live
  pipeline; the review screen shows "Expected 18, app counted 18 ✓".
* **Scan logs**: every live scan writes a JSONL detection log (Settings → Debug → *Record
  detection logs*). Settings → *Export logs* → unzip on a PC → `pallet-core replay` gives
  the identical count (`DetectionLogTest.replayOfRecordedLogReproducesLiveCount`) and lets
  you debug a bad real-world scan offline. Logs contain boxes and scores only, no images.

## Plumbing tests without a model

Settings → Detector → **SIMULATION** makes the app generate synthetic boxes for an endless
pallet line (camera image ignored, red banner). Use it to check the camera, overlay,
counting, review and history on a phone before a model exists. It proves nothing about
detection.

`pallet-core simulate --scenario reverse --out sim.jsonl` writes synthetic logs
(scenarios: steady, noisy, reverse, jerks, blackout) for trying the replay tools.

## Adding a regression test for a real failure

1. Export the scan log (or run `predict_video.py` on the video).
2. Put the `.jsonl` (boxes only) into your private test-log folder with `expected_count` in
   its header, and add it to your `pallet-core eval` set.
3. If it exposes a tracking/counting bug, reproduce it in `LineCounterTest` or as a
   `SimulatedSweepTest` scenario, fix, and keep the test. Do not commit logs from real
   scans to the public repository unless you are sure they reveal nothing sensitive
   (they contain only box coordinates, but still).
