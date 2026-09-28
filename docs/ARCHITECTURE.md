# Architecture

## Goal in one sentence

Count the wooden EUR-pallet bases under stacks of black half-pallets by filming a line from
the side while walking, detecting the bases with an on-device model, tracking them across
frames and counting each physical pallet once when it crosses a virtual line.

## Module layout

```
core/            Pure Kotlin/JVM. No Android dependencies. Unit-tested on any machine.
  geometry/        Box (normalized upright-frame coordinates), IoU, DIoU
  detection/       Detection, DetectionFrame, PalletDetector<F> interface,
                   Letterbox, YoloDecoder (+ layout detection), NMS, ModelSidecar
  tracking/        ByteTracker, Track, per-coordinate Kalman filters, Hungarian solver
  counting/        LineCounter (net line crossings, hysteresis, stitching)
  pipeline/        ScanPipeline = ROI filter -> tracker -> counter, ScanSnapshot
  session/         StackSize, LineMode, CountResult (pallets x stack x multiplier)
  replay/          JSONL detection logs, Replay runner
  sim/             SyntheticLineScene (test/plumbing simulator, NOT computer vision)
  cli/             pallet-core replay | eval | simulate

app/             Android (Kotlin, Jetpack Compose, CameraX, LiteRT)
  camera/          CameraBinder (Preview + ImageAnalysis), FrameAnalyzer (frame gate)
  detection/       FrameInput, Preprocessor, TfliteYoloDetector, SimulatedDetector, ModelManager
  scan/            ScanEngine (inference worker, stats, thumbnails, captures, logs)
  video/           VideoFrameSource (recorded video -> frames)
  debug/           CaptureStore (sample capture), LogStore (detection logs)
  data/            AppSettings, SettingsRepository, HistoryRepository
  ui/              Setup, Scan, Review, Photo, VideoReplay, Settings screens, overlay

training/        Python: train, export (LiteRT), evaluate, predict_video, golden fixtures
training_tools/  Python: extract_frames, split_dataset, check_dataset, prelabel, synthetic data
```

Camera ≠ detector ≠ tracker ≠ counter ≠ UI: each only talks to the next through small data
types (`FrameInput` → `DetectionFrame` → `TrackerOutput` → `CounterUpdate` → `ScanSnapshot`).

## Data flow (live sweep)

```
 CameraX Preview ───────────────────────────────► PreviewView (smooth, full rate)
 CameraX ImageAnalysis (RGBA, 4:3, KEEP_ONLY_LATEST)
      │ every frame: camera-FPS meter
      │ FrameGate.tryBegin(): detector idle AND >= 1/maxFps since last frame?
      │     no  -> close frame immediately (no copy)
      │     yes -> copy into one reused bitmap -> FrameInput(bitmap, rotation, timestamp)
      ▼
 ScanEngine worker thread ("pallet-inference")
      │ PalletDetector.detect(frame)        TfliteYoloDetector | SimulatedDetector
      │     Preprocessor: rotate + letterbox in one Canvas draw -> input tensor
      │     LiteRT Interpreter (GPU delegate or XNNPACK CPU)
      │     YoloDecoder (core): layout, coordinates, threshold, NMS -> DetectionFrame
      │ ScanPipeline.process(DetectionFrame)            (core, identical everywhere)
      │     ROI filter -> ByteTracker -> LineCounter -> ScanSnapshot
      │ DetectionLogWriter (JSONL, optional)  thumbnails of counted pallets  sample capture
      ▼
 StateFlow<ScanUiState> ──► Compose overlay (ROI band, count line, tracks, HUD, count)
```

Video replay (`VideoFrameSource`) and the desktop CLI feed the same `ScanPipeline`; the
CLI starts from recorded `DetectionFrame`s instead of pixels. A scan log therefore
reproduces the app's count exactly on a PC (tested in `DetectionLogTest`).

## Coordinates

Everything after the detector uses normalized coordinates of the **upright** analysed frame:
x → right, y → down, (0,0) top-left, (1,1) bottom-right. Camera buffers arrive in sensor
orientation; `FrameInput.rotationDegrees` says how to rotate them. Rotation, letterboxing
and model input size are undone inside the detector, so tracking, counting, logs and
overlays never depend on camera resolution, phone orientation or model size.

Preview and analysis use the same 4:3 field of view and the preview is shown with
`FIT_CENTER`, so the overlay maps normalized boxes linearly onto the fitted preview rect.
The scan screen requests a fixed orientation (landscape by default, Settings → Scan
orientation) and binds the camera only after the display has turned, because the camera's
target rotation is taken from the display at bind time. Frames are therefore upright even
when system auto-rotate is off, and no rotation can swap the axes mid-scan.

## Detector abstraction

```kotlin
interface PalletDetector<in F> : AutoCloseable {
    val info: DetectorInfo                       // name, input size, accelerator, isSimulation
    fun detect(frame: F, timestampNanos: Long): DetectionFrame
}
data class Detection(val box: Box, val confidence: Float, val classId: Int)
data class DetectionFrame(val timestampNanos: Long, val detections: List<Detection>, ...)
```

`TfliteYoloDetector` handles any YOLO-family LiteRT model: float32/int8/uint8 inputs and
outputs, NCHW or NHWC input, raw heads (`[1, 4+classes, anchors]`, either orientation) and
NMS-free end-to-end heads (`[1, N, 6]`), normalized or pixel coordinates. The layout is
read from the sidecar JSON or inferred from tensor shapes. Another architecture or runtime
(ONNX Runtime, MediaPipe, a different decoder) means implementing `PalletDetector` — the
tracker, counter, UI and tools stay unchanged.

`SimulatedDetector` ignores the image and replays a synthetic line walk. It exists only to
test camera → tracking → counting → UI before a model exists; it is labelled SIMULATION
everywhere (red banner, history, logs, review screen).

## Tracking (core/tracking)

ByteTrack (Zhang et al. 2022) adapted for variable detector timing:

1. **Predict** every track to the frame timestamp: independent constant-velocity Kalman
   filters for centre x/y (real `dt`, not "one step per frame"), random-walk filters for
   width/height. Noise is scaled by the track's own size.
2. **Associate high-score detections** (≥ 0.5) with confirmed and lost tracks via Hungarian
   assignment on IoU (DIoU for lost tracks, which tolerates drift). Width/height ratios are
   gated so an unrelated small box cannot hijack a track.
3. **Camera-jerk recovery**: if recently seen tracks all fail at once, try the offsets
   implied by (track, detection) pairs; accept an offset that explains ≥ 2 matches (or one
   match in the walking direction).
4. **Associate low-score detections** (0.1–0.5) with remaining confirmed and just-lost tracks —
   motion blur and partial occlusion mostly *lower* scores rather than remove detections.
5. **Tentative tracks** need `minHits` (3) matches before they may count; unmatched
   high-score detections ≥ 0.55 start tracks.
6. **Lost tracks** are kept 1.5 s and predicted with the **camera-motion estimate** (median
   velocity of confirmed tracks — all pallets in a line move together in the image), so a
   pallet hidden while the camera stops is predicted to stop too.

## Counting (core/counting)

A vertical **count line** (default x = 0.5) with a **dead band** (±0.04 frame widths).
Each track remembers the last side (LEFT/RIGHT) where it was seen *outside* the band.
Observing it on the other side is a crossing: +1 for left→right, −1 for right→left.

* **Count = |net crossings|.** Walking back over pallets un-counts them, walking forward
  again re-counts them, so re-scanning a section never double counts — even when track ids
  change — and the walking direction never needs to be detected or configured.
* **One track never contributes more than ±1**: its crossings necessarily alternate.
* **Jitter on the line changes nothing** (the band), a stopped camera changes nothing.
* **Crossings of unconfirmed tracks are held** and applied once the track is confirmed
  (fast pallets), or dropped if it dies first (most false positives).
* **Short-term memory (stitching)**: when a new track appears where a *removed* track is
  predicted (camera motion × elapsed time), it inherits that track's side and contribution.
  If the old track is merely *lost*, the link is provisional: it is merged only when the new
  track is confirmed (or the old one times out) and dropped if the old track is recovered.
  (An earlier version merged immediately; simulation showed a false positive near the line
  could then delete the real pallet's lost track — see git history.)
* **Optional relative size filter** ignores tracks much smaller than the upper quartile of
  confirmed track heights (far-row pallets visible through gaps).

Counted pallets get a thumbnail; the review screen lists them in order so duplicates and
misses can be spotted and corrected with ±1.

## Region of interest

A detection is kept only if its centre lies inside the ROI and at least 50 % of its area
is inside; everything else is dropped before tracking. Presets: centre band (0.25–0.85,
default), low band, full height (for lines stacked in several tiers, so every visible
wooden layer is counted), custom.
The ROI and the count line are configurable in Settings and drawn on the preview
("KEEP PALLET BASES IN THIS AREA").

## Threads and performance

| Thread | Work |
|---|---|
| main | Compose UI, camera binding |
| camera-analysis | frame gate, RGBA copy into a reused bitmap (only when the detector is free) |
| pallet-inference | preprocessing, LiteRT, decoding, tracking, counting, logging |

* Detector rate is capped (default 10 fps) and halved automatically when the phone reports
  a SEVERE thermal state; preview stays at full camera rate.
* Frames that arrive while the detector is busy are closed without copying.
* The GPU delegate is created, used and closed on the inference thread (required by LiteRT).
* Inference input size is chosen at export time (320/416 for budget phones, 640 for best
  accuracy); tracking bridges the gaps between detector frames.

## Persistence and privacy

* No `INTERNET` permission — the app cannot upload anything.
* Frames are processed in memory and discarded. Nothing is written unless the user taps a
  capture button (image + JSON + draft YOLO label in app-private storage).
* Detection logs contain boxes and scores only (no pixels).
* Captures and logs leave the phone only via "Export … ZIP" through the system file picker.
* Accepted counts are stored locally in `history.json`.
* App data is excluded from cloud backup and from device-to-device transfer
  (`data_extraction_rules.xml`), and the recent-apps snapshot is disabled (Android 13+),
  because screens show camera frames and pallet thumbnails.
* The manifest strips `INTERNET` even if a library tries to add it; CI checks the APK's
  permissions.

## Extending

| Change | Where |
|---|---|
| New model file | import in Settings or `app/src/main/assets/models/` — no code |
| Different YOLO output format | `YoloDecoder` / sidecar `output_format` |
| Different runtime (ONNX, MediaPipe) | new `PalletDetector<FrameInput>` + `ModelManager` |
| Different counting rule | `LineCounter` (tests in `LineCounterTest`, `SimulatedSweepTest`) |
| Adjustable ROI on the preview | `AppSettings.roi()` already drives the pipeline and overlay |
