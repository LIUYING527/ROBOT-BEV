"""
Task-mode conditioning for robot DiffusionDrive training (replaces fixed [1,0,0,0] command).

Uses the same 8-d status layout as NAVSIM Transfuser:
  [driving_command(4), ego_velocity(2), ego_acceleration(2)]

Modes (one-hot command):
  cruise  -> [1,0,0,0]  straight / corridor driving
  turn    -> [0,1,0,0]  lane change / cornering
  inspect -> [0,0,1,0]  slow approach / face rack / in-place check
  exit    -> [0,0,0,1]  leave inspection pose, resume cruise
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

MODES = ("cruise", "turn", "inspect", "exit")
TASK_MODES_FILENAME = "task_modes.jsonl"
TASK_MODES_AUDIT_FILENAME = "task_mode_annotations.jsonl"


class TaskMode(str, Enum):
    CRUISE = "cruise"
    TURN = "turn"
    INSPECT = "inspect"
    EXIT = "exit"


def mode_to_command(mode: str) -> np.ndarray:
    m = TaskMode(mode.lower())
    if m == TaskMode.CRUISE:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if m == TaskMode.TURN:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    if m == TaskMode.INSPECT:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    if m == TaskMode.EXIT:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    raise ValueError(f"Unknown mode: {mode}")


def build_status_vector(
    mode: str,
    vx: float,
    vy: float,
    ax: float,
    ay: float,
) -> np.ndarray:
    cmd = mode_to_command(mode)
    vel = np.array([vx, vy], dtype=np.float32)
    acc = np.array([ax, ay], dtype=np.float32)
    return np.concatenate([cmd, vel, acc], axis=0)


def validate_mode(mode: str) -> str:
    m = str(mode).lower()
    if m not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
    return m


def load_mode_overrides(recording_dir: Path) -> List[Tuple[int, int, str]]:
    """
    Manual labels: recording_dir/task_modes.jsonl
    Each line: {"t_start_ns": ..., "t_end_ns": ..., "mode": "inspect"|...}
    """
    path = recording_dir / TASK_MODES_FILENAME
    if not path.is_file():
        return []
    spans: List[Tuple[int, int, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            spans.append(
                (
                    int(o["t_start_ns"]),
                    int(o["t_end_ns"]),
                    validate_mode(str(o["mode"])),
                )
            )
    return merge_adjacent_spans(spans)


def save_mode_spans(recording_dir: Path, spans: List[Tuple[int, int, str]]) -> Path:
    """Write merged spans to recording_dir/task_modes.jsonl."""
    path = recording_dir / TASK_MODES_FILENAME
    spans = merge_adjacent_spans(spans)
    with open(path, "w", encoding="utf-8") as f:
        for t0, t1, mode in spans:
            rec = {"t_start_ns": int(t0), "t_end_ns": int(t1), "mode": mode}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def merge_adjacent_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int, str]] = [spans[0]]
    for t0, t1, mode in spans[1:]:
        p0, p1, pm = merged[-1]
        if mode == pm and t0 <= p1:
            merged[-1] = (p0, max(p1, t1), mode)
        else:
            merged.append((t0, t1, mode))
    return merged


def apply_range_to_spans(
    spans: List[Tuple[int, int, str]],
    t_start_ns: int,
    t_end_ns: int,
    mode: str,
) -> List[Tuple[int, int, str]]:
    """Label [t_start_ns, t_end_ns] inclusive; overlapping spans are split/replaced."""
    mode = validate_mode(mode)
    if t_start_ns > t_end_ns:
        t_start_ns, t_end_ns = t_end_ns, t_start_ns
    kept: List[Tuple[int, int, str]] = []
    for a, b, m in spans:
        if b < t_start_ns or a > t_end_ns:
            kept.append((a, b, m))
            continue
        if a < t_start_ns:
            kept.append((a, t_start_ns - 1, m))
        if b > t_end_ns:
            kept.append((t_end_ns + 1, b, m))
    kept.append((t_start_ns, t_end_ns, mode))
    return merge_adjacent_spans(kept)


def mode_for_timestamp(anchor_ns: int, spans: List[Tuple[int, int, str]]) -> Optional[str]:
    for t0, t1, mode in spans:
        if t0 <= anchor_ns <= t1:
            return mode
    return None


def is_timestamp_labeled(anchor_ns: int, spans: List[Tuple[int, int, str]]) -> bool:
    return mode_for_timestamp(anchor_ns, spans) is not None


def mode_at_timestamp(
    anchor_ns: int,
    vx: float,
    vy: float,
    wz: float,
    spans: List[Tuple[int, int, str]],
    *,
    use_heuristic: bool,
    cruise_vx: float,
    inspect_wz: float,
    turn_vy: float,
) -> Optional[str]:
    labeled = mode_for_timestamp(anchor_ns, spans)
    if labeled is not None:
        return labeled
    if not use_heuristic:
        return None
    return infer_mode_heuristic(vx, vy, wz, cruise_vx, inspect_wz, turn_vy)


def infer_mode_heuristic(
    vx: float,
    vy: float,
    wz: float,
    cruise_vx: float = 0.06,
    inspect_wz: float = 0.06,
    turn_vy: float = 0.04,
) -> str:
    """Bootstrap labels only — refine with task_modes.jsonl for exit/inspect boundaries."""
    if abs(wz) >= inspect_wz and abs(vx) < cruise_vx:
        return TaskMode.INSPECT.value
    if abs(vy) >= turn_vy and abs(vx) < cruise_vx * 1.5:
        return TaskMode.TURN.value
    if vx >= cruise_vx and abs(wz) < inspect_wz * 0.8:
        return TaskMode.CRUISE.value
    return TaskMode.CRUISE.value


def infer_mode_temporal(
    anchor_ns: int,
    times_ns: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    wz: np.ndarray,
    spans: List[Tuple[int, int, str]],
    *,
    use_heuristic: bool,
    lookback_s: float = 2.0,
) -> Optional[str]:
    """Heuristic with short history: after inspect-like motion, forward -> exit."""
    labeled = mode_for_timestamp(anchor_ns, spans)
    if labeled is not None:
        return labeled
    if not use_heuristic:
        return None

    q = float(anchor_ns)
    t = times_ns.astype(np.float64) * 1e-9
    tq = q * 1e-9
    mask = (t >= tq - lookback_s) & (t <= tq)
    if not np.any(mask):
        ivx = float(np.interp(tq, t, vx))
        ivy = float(np.interp(tq, t, vy))
        iwz = float(np.interp(tq, t, wz))
        return infer_mode_heuristic(ivx, ivy, iwz)

    wz_win = wz[mask]
    vx_win = vx[mask]
    vy_win = vy[mask]
    ivx = float(np.interp(tq, t, vx))
    ivy = float(np.interp(tq, t, vy))
    iwz = float(np.interp(tq, t, wz))

    inspect_recent = float(np.mean(np.abs(wz_win) > 0.06)) > 0.35
    if inspect_recent and ivx > 0.08 and abs(iwz) < 0.05 and abs(ivy) < 0.05:
        return TaskMode.EXIT.value

    return infer_mode_heuristic(ivx, ivy, iwz)
