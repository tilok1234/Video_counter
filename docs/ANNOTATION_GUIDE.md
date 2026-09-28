# Annotation guide — class `eur_pallet_base`

One class, one rule, applied the same way on every image. Inconsistent boxes hurt the
model more than missing data.

## The rule

> Draw one box around the **visible wooden EUR-pallet base under each stack of the line
> you are filming**, in the **front row** (the row whose stacks face the camera), at any
> tier/height.

## What goes inside the box

* **Width:** from the left end to the right end of the pallet: deck boards and support
  blocks, as far as they are visible.
* **Height:** from the top edge of the top deck boards to the bottom of the blocks (where
  the pallet meets the floor or the stack below).
* **Not** the black half-pallets above it, **not** the floor, **not** shadows.
* Tight: the box edges touch the wood. A few pixels of slack is fine; half a board is not.
* Two pallets side by side = **two boxes**, even if they touch. Split at the visible gap or
  at the point where the blocks/boards of one pallet end.

```
  ┌──────────── black half-pallet stack ────────────┐
  │                                                  │
  ├──────────────────────────────────────────────────┤
  ┃▓▓▓▓▓▓▓▓▓▓▓▓▓▓ deck boards ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┃  ◄ box top
  ┃██  fork opening  ██  fork opening  ██             ┃
  ┃██ block ░░░░░░░░ ██ block ░░░░░░░░ ██ block       ┃  ◄ box bottom
  ═══════════════════════ floor ════════════════════════
```

## Partially visible pallets

| Situation | Label? |
|---|---|
| Cut off by the image edge, **≥ ~1/3 of its width visible** | Yes — box the visible part only |
| Cut off by the image edge, less than ~1/3 visible | No |
| Part hidden by a person/post/forklift, both ends still visible | Yes — box the whole pallet |
| Part hidden, only one end visible, ≥ 1/3 visible | Yes — box the visible part |
| Mostly hidden, you could not tell it is a pallet without context | No |
| Shrink wrap over the base, pallet still recognisable | Yes — box the pallet under the wrap |
| Motion-blurred but recognisable | Yes |

## What NOT to label

* Pallets in the **second / far row** of the line (visible through gaps between stacks).
* Pallets of **other lines** further away, pallet piles in the background.
* **Empty** pallets lying around, pallets on forklifts, loose boards, wooden walls/fences.
* Anything you are not sure is a front-row pallet base of the line.

These are *background*: leave them unlabeled so the model learns to ignore them.

## Uncertain cases

If you cannot decide whether something must be labeled, **remove the whole image from the
dataset** rather than guessing. An unlabeled real pallet teaches the model to ignore real
pallets; a labeled non-pallet teaches it to invent pallets. Both are worse than one image
fewer. Keep a short list of such cases — they often reveal a rule that needs clarifying.

## Images without target pallets

Keep them. Their label file must exist and be **empty** — that is how YOLO learns
negatives. Most tools write an empty `.txt` when you save an image without boxes; check it.

## Label format (YOLO)

One `.txt` per image, same name, one line per box, all values normalized to 0–1:

```
<class> <x_center> <y_center> <width> <height>
0 0.512 0.771 0.301 0.098
```

The class id is always `0` (`eur_pallet_base`).

## Tools

Any tool that exports YOLO detection format works. Recommended, all run **locally** (no
upload of workplace imagery):

* **X-AnyLabeling** (desktop app): open the frames folder, class list `eur_pallet_base`,
  draw rectangles, export/save as YOLO. It can also load our exported model for automatic
  pre-labeling in later rounds.
* **CVAT** (self-hosted with Docker): create a task with the label `eur_pallet_base`, export
  "YOLO 1.1".
* **Label Studio** (`pip install label-studio`, runs in your browser on localhost): object
  detection template, export YOLO.

Cloud labeling services (e.g. cvat.ai, Roboflow) upload your images — only use them if
that is acceptable for your workplace.

## Workflow

1. Extract frames: `python training_tools/extract_frames.py videos/ --out dataset/candidates --fps 2`
2. Label them in your tool; export images + YOLO `.txt` files into one folder, e.g. `labeled/`.
3. Split by video: `python training_tools/split_dataset.py --src labeled/ --dataset dataset --val 0.2 --test 0.1`
4. Check: `python training_tools/check_dataset.py --dataset dataset --render /tmp/check`
   and look through the rendered images — mistakes are easy to spot there.
5. Train ([MODEL_TRAINING.md](MODEL_TRAINING.md)).

From the second round on, let the model draft the labels and only correct them:

```bash
python training_tools/prelabel.py --model runs/detect/v1/weights/best.pt --images dataset/candidates/new_batch
```

Always review drafts before training; never train on unchecked model output.

## Train / validation / test split

* Split **by video**, never by frame: neighbouring frames are nearly identical, so frames
  of one video in both train and validation make scores look better than reality.
  `split_dataset.py` does this automatically from the `<video>_f<frame>` file names.
* 70 % train / 20 % validation / 10 % test is a good default.
* Keep a few whole videos (with hand counts) completely outside the dataset for counting
  tests ([TESTING.md](TESTING.md)).
