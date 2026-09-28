# Failure cases: design, diagnosis, tests

For each case from the requirements: how the system is designed to cope, how to see it
happening (HUD, overlay, review screen, logs) and what covers it in tests. "Data" means the
fix is more/better training examples, not code.

Diagnostics referenced below:

* **Overlay** — track boxes: yellow dashed = tentative, blue = tracked, green = counted,
  grey dashed = lost (predicted), orange = size-filtered; labels `ID · score · TRACKED n
  FRAMES · status`, `←#id` = stitched to an earlier track.
* **HUD** — detector/camera fps, inference ms, raw/ROI detections, tracks, camera motion,
  `L→R / R→L` crossings, stitches, `late` tracks (first seen past the line, never counted).
* **Review** — thumbnails of every counted pallet in order, crossing/stitch statistics.
* **Replay** — `pallet-core replay log.jsonl --events/--trace` on an exported scan log.

| # | Case | Design | Diagnose | Tests |
|---|---|---|---|---|
| 1 | Adjacent pallets touching | One box per pallet (annotation rule); NMS IoU 0.5 keeps touching boxes; Hungarian assignment keeps ids apart | two boxes and two ids on the overlay | `NmsTest.keepsAdjacentTouchingPallets`, `ByteTrackerTest.twoAdjacentObjectsKeepSeparateIds`; Data: many touching examples |
| 2 | Two-wide lines, different depths | Label front row only; ROI band; optional relative size filter; tracker size gating | orange "SMALL (far row?)" boxes; review thumbnails of far pallets | `SimulatedSweepTest.farRowSliversNeedTheSizeFilter`; Data: negatives with far row visible |
| 3 | One pallet blocking another | Low-score detections keep tracks alive; lost tracks predicted with camera motion; stitching | grey "LOST" boxes that turn blue again, `←#id` | `ByteTrackerTest.shortOcclusionRecoversSameId`, `LineCounterTest.trackLostAcrossTheLineIsStitchedAndCountedOnce` |
| 4 | Shrink wrap over wood | Detector must learn it | missed boxes on wrapped stacks | Data: wrapped stacks in every lighting |
| 5 | Steep viewing angle | Detector + boxes of any aspect; tracker gates on size ratios, not shape | boxes jumping or missing at an angle | Data: oblique views |
| 6 | Pallet only briefly visible | Crossings of tentative tracks are held until confirmation (3 hits) | "PENDING" label; `delayed` in replay stats | `LineCounterTest.fastPalletCrossingBeforeConfirmationIsCountedOnConfirmation` |
| 7 | Camera moving quickly | Kalman prediction with real dt, camera-motion prior for new tracks, jerk recovery; "MOVING TOO FAST" warning | HUD `step %` of pallet width, orange warning banner | `SimulatedSweepTest.fastWalkLowFrameRate`, `cameraJerks` |
| 8 | Motion blur | Low-score detections extend tracks (ByteTrack second stage); blurry frames kept in training data | scores drop but ids stay | `ByteTrackerTest.lowScoreDetectionsKeepConfirmedTrackAlive`, `steadyWalkNoisyDetector`; Data: blurred frames |
| 9 | Camera stopping | Count changes only on crossings outside the dead band; lost tracks follow camera motion (stop) | HUD "STATIONARY" | `LineCounterTest.palletParkedOnTheLineWithJitterDoesNotCount`, `ByteTrackerTest.lostTrackFollowsCameraMotionWhenCameraStops`, `stopAndGo` |
| 10 | Camera reversing | Net crossings: back = −1, forward again = +1 | L→R and R→L both non-zero in HUD/replay | `LineCounterTest.alreadyCountedObjectCrossingAgainIsNotDuplicated`, `shortReversal`, `PythonLogReplayTest` |
| 11 | Detector losing/reacquiring | 1.5 s lost buffer + low-score recovery + stitching window 1.5 s after removal | grey boxes, `←#id` | `shortOcclusionRecoversSameId`, `detectorBlackouts` |
| 12 | Tracker changing ids | Harmless unless the switch happens across the line; provisional links + stitching carry the side over | `stitches` count, `←#id` labels | `idSwitchAfterCountingDoesNotDoubleCount`, `trackLostAcrossTheLineIsStitchedAndCountedOnce` |
| 13 | Same section scanned twice | Net crossings (within one scan); review thumbnails in order | thumbnails repeat? L→R vs R→L | `longReversalRescansSection`, `endsHalfwayBack` |
| 14 | Empty pallet nearby | Not labeled → negative; ROI excludes most | red boxes on empty pallets (FALSE + capture) | Data: negatives |
| 15 | Background pallet stacks | ROI band; not labeled; size gating | raw detections outside the band (enable *Show raw detections*) | `LineCounterTest.detectionsOutsideRoiAreIgnored`; Data: negatives |
| 16 | Pallet-shaped wooden objects | Negatives in training | FALSE + captures | Data |
| 17 | Strong sunshine | Brightness augmentation; data | scores per lighting in `evaluate.py` renders | Data: hard shadows, backlight |
| 18 | Deep shadows | Same | same | Data |
| 19 | Wet/dark wood | Same | same | Data |
| 20 | Dirty/damaged pallets | Same | same | Data |
| 21 | Very crowded frames | Up to 100 detections/frame; Hungarian (global optimum) instead of greedy matching | tracks per frame in HUD | `twoAdjacentObjectsKeepSeparateIds`; Data: crowded lines |
| 22 | Only a small part of a base visible | Annotation: label if ≥ 1/3 visible; width ratio gating tolerates frame-edge truncation | boxes growing at frame edges | Data |

## Known limitations (honest list)

* **Start and end of a scan:** a pallet already past the count line when the scan starts
  is not counted; one that has not crossed when you press FINISH is not counted. Start
  before the first pallet and finish after the last one has crossed the yellow line.
* **A pallet first detected past the line** (missed for the whole first half of the frame)
  cannot be counted without risking double counts; it shows up as a `late` track. Frequent
  `late` tracks mean the detector misses pallets → more training data.
* **Multi-tier stacks:** each visible wooden layer inside the ROI is counted. With the
  default centre band only the tiers inside the band count; use *Full height* to count all
  visible tiers, and label every tier consistently (annotation guide).
* **Hold the phone the way the scan screen is shown.** If the screen is in landscape but
  the phone is held upright, the frames are sideways: the model sees rotated bases and the
  pallets move parallel to the count line, so nothing is counted.
* **Real detection quality is unknown until trained on your footage.** Everything above the
  detector is tested; the detector itself is only as good as the data.
