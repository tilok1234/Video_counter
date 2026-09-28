"""End-to-end tests of the data tools on synthetic data (no workplace imagery)."""
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOLS / script), *args], capture_output=True, text=True)


@pytest.fixture(scope="module")
def walk_video(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("video") / "line-01.mp4"
    r = run("make_synthetic_video.py", "--out", str(out), "--pallets", "4", "--pause", "2", "--speed", "0.5")
    assert r.returncode == 0, r.stderr
    assert "expected count 4" in r.stdout
    return out


def test_extract_frames_names_and_dedup(walk_video, tmp_path):
    r = run("extract_frames.py", str(walk_video), "--out", str(tmp_path / "all"), "--fps", "4", "--min-diff", "0")
    assert r.returncode == 0, r.stderr
    everything = sorted((tmp_path / "all").glob("*.jpg"))
    assert everything and everything[0].name.startswith("line-01_f")
    r = run("extract_frames.py", str(walk_video), "--out", str(tmp_path / "dedup"), "--fps", "4")
    deduped = sorted((tmp_path / "dedup").glob("*.jpg"))
    # The 2 s pause produces identical frames that must be skipped.
    assert len(deduped) < len(everything)
    assert "near-duplicates" in r.stdout
    assert (tmp_path / "dedup" / "manifest.csv").read_text().startswith("file,video,frame_index")


def test_rotate_option_swaps_dimensions(walk_video, tmp_path):
    import cv2

    r = run("extract_frames.py", str(walk_video), "--out", str(tmp_path), "--fps", "1", "--rotate", "90", "--max-per-video", "1")
    assert r.returncode == 0, r.stderr
    img = cv2.imread(str(next(tmp_path.glob("*.jpg"))))
    assert img.shape[0] > img.shape[1]  # portrait after rotating a landscape video


def test_split_is_grouped_by_video_and_check_passes(tmp_path):
    src = tmp_path / "labeled"
    src.mkdir()
    # 6 "videos" x 5 frames, labels next to images
    import cv2
    import numpy as np

    for v in range(6):
        for f in range(5):
            stem = f"line-{v:02d}_f{f * 15:06d}"
            cv2.imwrite(str(src / f"{stem}.jpg"), np.full((48, 64, 3), 10 * v + f, np.uint8))
            (src / f"{stem}.txt").write_text("" if f == 0 else "0 0.5 0.8 0.4 0.1\n")
    ds = tmp_path / "dataset"
    r = run("split_dataset.py", "--src", str(src), "--dataset", str(ds), "--val", "0.2", "--test", "0.2")
    assert r.returncode == 0, r.stderr
    groups = {}
    for split in ("train", "val", "test"):
        for img in (ds / "images" / split).glob("*.jpg"):
            groups.setdefault(img.stem.split("_f")[0], set()).add(split)
    assert all(len(s) == 1 for s in groups.values()), groups
    assert len(list((ds / "images" / "val").glob("*.jpg"))) > 0
    r = run("check_dataset.py", "--dataset", str(ds), "--no-hash")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "leakage" not in r.stdout


def test_check_dataset_rejects_malformed_labels(tmp_path):
    import cv2
    import numpy as np

    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        cv2.imwrite(str(tmp_path / "images" / split / "a_f000000.jpg"), np.zeros((20, 20, 3), np.uint8))
    (tmp_path / "labels" / "train" / "a_f000000.txt").write_text("0 50 50 10 10\n")  # pixels, not normalized
    (tmp_path / "labels" / "val" / "a_f000000.txt").write_text("0 0.5 0.5 0.2\n")  # 4 values
    r = run("check_dataset.py", "--dataset", str(tmp_path), "--no-hash")
    assert r.returncode == 1
    assert "normalized" in r.stdout and "expected 5 values" in r.stdout


def test_synthetic_dataset_generator(tmp_path):
    r = run("make_synthetic_dataset.py", "--out", str(tmp_path), "--train", "3", "--val", "2", "--test", "1")
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "images" / "train").glob("*.jpg"))) == 3
    assert run("check_dataset.py", "--dataset", str(tmp_path), "--no-hash").returncode == 0
