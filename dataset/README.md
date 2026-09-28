# dataset/

YOLO-format dataset for the single class `eur_pallet_base` (id 0).

```
dataset/
  images/{train,val,test}/   <video>_f<frame>.jpg
  labels/{train,val,test}/   <video>_f<frame>.txt   (one "0 cx cy w h" line per box, normalized; empty = negative)
  dataset.yaml               written by training/train.py
  split_report.json          written by training_tools/split_dataset.py
```

Contents are **git-ignored** (the repository is public). Fill it with
`training_tools/split_dataset.py`, check it with `training_tools/check_dataset.py`.
See docs/DATA_COLLECTION.md and docs/ANNOTATION_GUIDE.md.
