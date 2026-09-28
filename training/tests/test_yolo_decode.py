"""Tests for the Python mirror of the app's letterbox + YOLO decoder."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from palletlib.yolo_decode import END_TO_END, RAW_CHANNELS_FIRST, RAW_CHANNELS_LAST, Letterbox, decode, resolve_layout  # noqa: E402

GOLDEN = ROOT / "core" / "src" / "test" / "resources" / "golden"


@pytest.mark.parametrize(
    "src,dst,scaled,pad",
    [
        ((1280, 960), (640, 640), (640, 480), (0, 80)),
        ((960, 1280), (640, 640), (480, 640), (80, 0)),
        ((1920, 1080), (640, 640), (640, 360), (0, 140)),
        ((640, 480), (320, 320), (320, 240), (0, 40)),
    ],
)
def test_letterbox_matches_kotlin(src, dst, scaled, pad):
    lb = Letterbox(*src, *dst)
    assert (lb.scaled_w, lb.scaled_h) == scaled
    assert (lb.pad_left, lb.pad_top) == pad


def test_layout_detection_matches_kotlin_rules():
    assert resolve_layout((1, 5, 8400), 640, 640)[0] == RAW_CHANNELS_FIRST
    assert resolve_layout((1, 8400, 5), 640, 640)[0] == RAW_CHANNELS_LAST
    assert resolve_layout((1, 300, 6), 640, 640)[0] == END_TO_END
    assert resolve_layout((1, 6, 8400), 640, 640)[0] == RAW_CHANNELS_FIRST
    with pytest.raises(ValueError):
        resolve_layout((2, 5, 8400), 640, 640)


def test_decode_normalized_raw_box():
    n = 8400
    out = np.zeros((1, 5, n), np.float32)
    out[0, :, 10] = [0.5, 0.5, 0.5, 0.15, 0.9]
    lb = Letterbox(1280, 960, 640, 640)
    dets = decode(out, lb, conf=0.25)
    assert len(dets) == 1
    l, t, r, b, s, c = dets[0]
    assert (round(l, 4), round(t, 4), round(r, 4), round(b, 4)) == (0.25, 0.4, 0.75, 0.6)
    assert s == pytest.approx(0.9)


@pytest.mark.parametrize("meta_file", sorted(GOLDEN.glob("*.json")), ids=lambda p: p.stem)
def test_golden_fixtures_are_consistent(meta_file):
    meta = json.loads(meta_file.read_text())
    raw = np.fromfile(meta_file.with_name(meta_file.stem + ".output.bin"), dtype="<f4").reshape(meta["shape"])
    lb = Letterbox(meta["src_width"], meta["src_height"], meta["input_width"], meta["input_height"])
    dets = decode(raw, lb, conf=meta["conf"], fmt=meta["output_format"], coords=meta["coordinates"], target_classes={0})
    expected = meta["python_decoded"]
    assert len(dets) == len(expected)
    for d, e in zip(sorted(dets, key=lambda x: -x[4]), sorted(expected, key=lambda x: -x[4])):
        assert d[:5] == pytest.approx(e[:5], abs=1e-4)
