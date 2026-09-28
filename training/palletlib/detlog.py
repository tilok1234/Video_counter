"""Writer for the app's JSONL detection-log format (see DetectionLog.kt / docs/TESTING.md)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import IO, Iterable


def _f(v: float, decimals: int = 4) -> str:
    s = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def header(source: str, detector: dict | None = None, expected_count: int | None = None,
           video: str | None = None, notes: str | None = None) -> str:
    obj = {"type": "header", "version": 1, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": source}
    if detector:
        obj["detector"] = detector
    if expected_count is not None:
        obj["expected_count"] = expected_count
    if video:
        obj["video"] = video
    if notes:
        obj["notes"] = notes
    return json.dumps(obj, separators=(",", ":"))


def frame(timestamp_ns: int, index: int, dets: Iterable[tuple], width: int = 0, height: int = 0,
          inference_ms: float | None = None) -> str:
    parts = [f'{{"type":"frame","t":{int(timestamp_ns)},"i":{int(index)}']
    if width:
        parts.append(f',"w":{int(width)}')
    if height:
        parts.append(f',"h":{int(height)}')
    if inference_ms is not None:
        parts.append(f',"ms":{_f(inference_ms, 2)}')
    boxes = ",".join(
        f"[{_f(l)},{_f(t)},{_f(r)},{_f(b)},{_f(s, 3)},{int(c)}]" for l, t, r, b, s, c in dets
    )
    parts.append(f',"d":[{boxes}]}}')
    return "".join(parts)


def write_lines(fh: IO[str], lines: Iterable[str]) -> None:
    for line in lines:
        fh.write(line)
        fh.write("\n")
