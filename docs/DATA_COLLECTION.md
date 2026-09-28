# Data collection

The app can only be as good as the detector, and the detector can only learn what it has
seen. No public or pretrained model detects these pallet bases (see
[MODEL_TRAINING.md](MODEL_TRAINING.md#step-0-do-pretrained-models-help)), so the first model
has to be trained on **your own footage**. Videos are the easiest source: one 30-second
walk gives dozens of useful, varied frames.

> **Privacy.** This repository is public. Never commit workplace videos, frames, labels or
> trained weights — `.gitignore` already excludes `dataset/images`, `dataset/labels`,
> `videos/`, `*.mp4`, `*.tflite`, `*.pt` and friends. Keep footage on your own PC/phone.
> Check with your employer whether filming in the yard is allowed.

## How to film

Film exactly the way you will scan, so training data looks like what the app will see:

* **Landscape**, phone roughly level, camera about hip to chest height.
* Point at the **bases** of the stacks; bases in the middle-to-lower part of the frame.
* Walk **along** the line at normal, steady speed (about 1 step per second).
* Start before the first pallet of the line and stop after the last one.
* Two-wide lines: walk down the aisle and film the **near** row. The far row behind it
  (visible through gaps) is not a target — the *one side ×2* mode accounts for it.
* Use the normal camera app, 1080p or 720p, 30 fps. Higher resolution is not needed.
* 20–60 s per video is plenty. Many short videos beat a few long ones (variety).

## What to film — the variety list

Aim for coverage of every condition below. Each bullet is a few videos, not hundreds.

| Category | Examples |
|---|---|
| Geometry | straight-on views, oblique views (30–60°), close (1 m) and far (4 m+), low and high camera |
| Lines | single stacks, two-wide lines, very crowded lines, gaps between stacks, 20-stacks and 30-stacks, multi-tier stacks |
| Pallets | clean, dirty, dark/wet, damaged, repaired, new light wood, old grey wood |
| Covering | wrapped (shrink wrap over the base), unwrapped, straps, labels |
| Light | sunshine with hard shadows, overcast, rain, evening/low light, indoor warehouse lights, backlight |
| Motion | slow walk, fast walk (motion blur), stopping, stepping back |
| Occlusion | pallet partly hidden by a person, forklift, post; base partly out of frame |
| Background | trailers, buildings, machinery, other pallet piles behind the line |

## Negative examples — equally important

The model must learn what **not** to count. Record videos/photos that contain:

* wooden pallets that are **not** under a stack of the line: empty pallets lying around,
  pallet piles in the background, pallets on a forklift;
* wooden objects that look pallet-like: wall cladding, fences, crates, wooden trailer beds;
* the second (far) row of a two-wide line visible through gaps;
* stretches of yard with **no** target pallets at all.

Frames without target pallets are kept with an *empty* label file (see the annotation
guide). Aim for roughly 10–20 % such frames.

## How much data

| Stage | Labeled frames | Expectation |
|---|---|---|
| First model | 300–500 frames from 15+ videos | usable in conditions similar to the training videos |
| Solid model | 1 000–2 000 frames from 40+ videos covering the variety list | robust day-to-day |
| Later | + hard examples from app captures, a few hundred per round | fixes specific failure modes |

Variety matters more than volume: 300 frames from 30 different videos beat 3 000 frames
from 3 videos.

## From video to candidate frames

```bash
python training_tools/extract_frames.py videos/ --out dataset/candidates --fps 2
```

* `--fps 2` keeps ~2 frames per second of video; `--min-diff` (default 6) skips frames that
  are nearly identical to the previous kept one (standing still).
* Output names are `<video>_f<frame>.jpg` so all frames of a video stay in the same
  train/val/test split later.
* Sideways footage: `--rotate 90` (or 180/270).
* Blurry frames are **kept** on purpose — the app sees motion blur too. Use
  `--min-sharpness` only if you have far too many.
* A `manifest.csv` lists source video, time and sharpness of every frame.

Then label the frames ([ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)).

## Capturing hard examples with the app

While scanning (Settings → Debug → *Sample capture buttons*), tap:

| Button | When |
|---|---|
| **MISSED** | a pallet base is visible but has no box / was not counted |
| **FALSE +** | a box is on something that is not a target pallet |
| **BAD TRACK** | ids jump between pallets, a pallet is counted twice, etc. |
| **GOOD** | a typical frame that works (keeps the dataset balanced) |

Each tap saves the current frame, a JSON with all detections/tracks/settings and a draft
YOLO label file into app storage. Settings → *Export captures* writes a ZIP through the
system file picker. Unzip on your PC, correct the draft labels, add to the dataset, retrain.
Nothing is uploaded automatically.

## Recording test videos (for evaluating counting)

Keep a separate set of videos **never used for training**, each with a hand-counted number
of pallets (write it in the file name, e.g. `line07_count18.mp4`). They are the benchmark
for every new model or parameter change — see [TESTING.md](TESTING.md).
