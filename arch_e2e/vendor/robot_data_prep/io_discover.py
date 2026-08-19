"""Discover image / pointcloud files and timestamps from filenames."""

from __future__ import annotations

import bisect
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

_TS_RE = re.compile(r"(\d{13,20})\.(jpg|jpeg|png|npz)$", re.IGNORECASE)


def parse_timestamp_ns_from_name(path: Path) -> int:
    m = _TS_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from filename: {path.name}")
    digits = m.group(1)
    value = int(digits)
    n = len(digits)

    # Robot recordings usually store media filenames in microseconds (16 digits).
    # Normalize all supported timestamp granularities to nanoseconds.
    if n <= 10:  # seconds
        return value * int(1e9)
    if n <= 13:  # milliseconds
        return value * int(1e6)
    if n <= 16:  # microseconds
        return value * int(1e3)
    if n <= 19:  # nanoseconds
        return value
    # If sub-ns precision ever appears, keep nanosecond part only.
    return int(digits[:19])


def list_sorted_media(
    root: Path,
    subdir_globs: Tuple[str, ...],
    exts: Tuple[str, ...],
) -> List[Tuple[int, Path]]:
    """Return (timestamp_ns, path) sorted by time."""
    items: List[Tuple[int, Path]] = []
    seen: Set[Path] = set()
    for pat in subdir_globs:
        for p in root.glob(pat):
            if p.suffix.lower() not in exts:
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            try:
                ts = parse_timestamp_ns_from_name(p)
            except ValueError:
                continue
            items.append((ts, p))
    items.sort(key=lambda x: x[0])
    return items


def nearest_by_time(
    query_ns: int,
    items: List[Tuple[int, Path]],
    max_delta_ns: int,
) -> Tuple[Optional[Path], Optional[int]]:
    """Return (path, signed_delta_ns) or (None, None) if empty / too far."""
    if not items:
        return None, None
    times = [t for t, _ in items]
    i = bisect.bisect_left(times, query_ns)
    candidates = []
    if i < len(times):
        candidates.append(i)
    if i > 0:
        candidates.append(i - 1)
    best_i = min(candidates, key=lambda k: abs(times[k] - query_ns))
    dt = int(times[best_i] - query_ns)
    if abs(dt) > max_delta_ns:
        return None, None
    return items[best_i][1], dt
