"""The Python log writer must produce the JSONL format the Kotlin replay CLI parses."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))
from palletlib import detlog  # noqa: E402


def test_frame_line_is_compact_valid_json():
    line = detlog.frame(1_000_000, 3, [(0.1, 0.2, 0.3456789, 0.4, 0.91234, 0)], 1280, 960, 12.345)
    obj = json.loads(line)
    assert obj["type"] == "frame" and obj["t"] == 1_000_000 and obj["i"] == 3
    assert obj["w"] == 1280 and obj["h"] == 960 and obj["ms"] == 12.35
    assert obj["d"] == [[0.1, 0.2, 0.3457, 0.4, 0.912, 0]]


def test_header_contains_expected_count():
    obj = json.loads(detlog.header("python-predict", {"name": "m", "kind": "tflite-yolo"}, 18, "v.mp4"))
    assert obj["type"] == "header" and obj["expected_count"] == 18 and obj["detector"]["name"] == "m"


def test_negative_zero_is_normalized():
    assert '"d":[[0,0,0.5,0.5,0.5,0]]' in detlog.frame(0, 0, [(-0.00001, 0.0, 0.5, 0.5, 0.5, 0)])
