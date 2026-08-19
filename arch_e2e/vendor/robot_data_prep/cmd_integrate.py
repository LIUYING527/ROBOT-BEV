"""Integrate geometry_msgs/Twist-style velocities to a planar pose path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_svtrobot_cmd_jsonl(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns times_ns (N,), vx, vy, wz (each N,) in SI units."""
    times: List[int] = []
    vx: List[float] = []
    vy: List[float] = []
    wz: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            times.append(int(o["_timestamp_ns"]))
            lin = o["linear"]
            ang = o["angular"]
            vx.append(float(lin["x"]))
            vy.append(float(lin["y"]))
            wz.append(float(ang["z"]))
    return (
        np.asarray(times, dtype=np.int64),
        np.asarray(vx, dtype=np.float64),
        np.asarray(vy, dtype=np.float64),
        np.asarray(wz, dtype=np.float64),
    )


def integrate_world_path(
    times_ns: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    wz: np.ndarray,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    psi0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Piecewise-constant body-frame twist integrated in world frame.
    Returns (times_ns same length as input, poses Nx3 as x,y,psi_rad).
    """
    n = len(times_ns)
    if n == 0:
        return times_ns, np.zeros((0, 3), dtype=np.float64)
    x, y, psi = float(x0), float(y0), float(psi0)
    poses = np.zeros((n, 3), dtype=np.float64)
    poses[0] = (x, y, psi)
    for i in range(n - 1):
        dt = (times_ns[i + 1] - times_ns[i]) * 1e-9
        if dt <= 0:
            dt = 1e-6
        c, s = np.cos(psi), np.sin(psi)
        x += (vx[i] * c - vy[i] * s) * dt
        y += (vx[i] * s + vy[i] * c) * dt
        psi = psi + wz[i] * dt
        psi = np.arctan2(np.sin(psi), np.cos(psi))
        poses[i + 1] = (x, y, psi)
    return times_ns, poses


def interpolate_pose_at_time(
    times_ns: np.ndarray,
    poses: np.ndarray,
    query_times_ns: np.ndarray,
) -> np.ndarray:
    """Interpolate x,y and unwrap psi. Returns (M,3)."""
    if len(times_ns) < 2:
        raise ValueError("Need at least 2 cmd samples to interpolate.")
    t = times_ns.astype(np.float64) * 1e-9
    q = query_times_ns.astype(np.float64) * 1e-9
    x = np.interp(q, t, poses[:, 0])
    y = np.interp(q, t, poses[:, 1])
    psi_u = np.unwrap(poses[:, 2])
    psi = np.interp(q, t, psi_u)
    psi = np.arctan2(np.sin(psi), np.cos(psi))
    return np.stack([x, y, psi], axis=-1)


def future_trajectory_in_ego_frame(
    pose_anchor: np.ndarray,
    future_poses: np.ndarray,
) -> np.ndarray:
    """
    pose_anchor: (3,) x,y,psi world
    future_poses: (K,3) world
    Returns (K,3) x,y,heading in ego frame at anchor (relative SE2).
    """
    xa, ya, psia = pose_anchor
    c, s = np.cos(psia), np.sin(psia)
    out = np.zeros_like(future_poses)
    dx = future_poses[:, 0] - xa
    dy = future_poses[:, 1] - ya
    out[:, 0] = c * dx + s * dy
    out[:, 1] = -s * dx + c * dy
    out[:, 2] = np.arctan2(
        np.sin(future_poses[:, 2] - psia),
        np.cos(future_poses[:, 2] - psia),
    )
    return out.astype(np.float32)


def build_anchor_grid_ns(t0_ns: int, t1_ns: int, hz: float) -> np.ndarray:
    """Inclusive-ish grid from t0 to t1 at hz (skip if beyond)."""
    step = int(round(1e9 / hz))
    out: List[int] = []
    t = t0_ns
    while t <= t1_ns:
        out.append(int(t))
        t += step
    return np.asarray(out, dtype=np.int64)

