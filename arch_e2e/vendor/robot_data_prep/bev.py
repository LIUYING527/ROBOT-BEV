"""Point cloud → BEV histogram (Transfuser / DiffusionDrive compatible)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class BEVParams:
    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5
    max_height_lidar: float = 100.0
    lidar_split_height: float = 0.2
    use_ground_plane: bool = False


def load_xyz_from_npz(path: str) -> np.ndarray:
    """Load Nx3 points in meters (or auto-scaled from mm)."""
    z = np.load(path, allow_pickle=True)
    arr = None
    for k in ("points", "xyz", "points_xyz", "xyzrgba", "data", "arr_0"):
        if k in z.files:
            arr = np.asarray(z[k])
            break
    if arr is None:
        raise ValueError(f"No known array key in {path}: {z.files}")
    if arr.ndim == 1:
        raise ValueError(f"Unexpected shape in {path}")
    if arr.shape[-1] < 3:
        raise ValueError(f"Need at least xyz columns, got shape {arr.shape}")
    pts = arr[..., :3].astype(np.float64)
    if pts.ndim < 2:
        raise ValueError(f"Unexpected shape in {path}: {pts.shape}")
    if pts.ndim > 2:
        pts = pts.reshape(-1, pts.shape[-1])
    # Heuristic: ZED / devices often store mm
    if np.nanmax(np.abs(pts)) > 50.0:
        pts = pts / 1000.0
    return pts


def points_to_bev_tensor(
    lidar_pc: np.ndarray,
    p: BEVParams,
) -> np.ndarray:
    """
    Returns float32 tensor [C, H, W] with H=W from bin counts (same logic as Transfuser).
    """
    if lidar_pc.size == 0:
        h = int((p.lidar_max_x - p.lidar_min_x) * p.pixels_per_meter)
        w = int((p.lidar_max_y - p.lidar_min_y) * p.pixels_per_meter)
        c = 2 if p.use_ground_plane else 1
        return np.zeros((c, h, w), dtype=np.float32)

    lidar_pc = lidar_pc[np.isfinite(lidar_pc).all(axis=-1)]
    lidar_pc = lidar_pc[lidar_pc[..., 2] < p.max_height_lidar]

    def splat_points(point_cloud_xy: np.ndarray) -> np.ndarray:
        xbins = np.linspace(
            p.lidar_min_x,
            p.lidar_max_x,
            int((p.lidar_max_x - p.lidar_min_x) * int(p.pixels_per_meter)) + 1,
        )
        ybins = np.linspace(
            p.lidar_min_y,
            p.lidar_max_y,
            int((p.lidar_max_y - p.lidar_min_y) * int(p.pixels_per_meter)) + 1,
        )
        hist = np.histogramdd(point_cloud_xy[:, :2], bins=(xbins, ybins))[0]
        hist[hist > p.hist_max_per_pixel] = p.hist_max_per_pixel
        overhead_splat = hist / float(p.hist_max_per_pixel)
        return overhead_splat.astype(np.float32)

    below = lidar_pc[lidar_pc[..., 2] <= p.lidar_split_height]
    above = lidar_pc[lidar_pc[..., 2] > p.lidar_split_height]
    above_features = splat_points(above)
    if p.use_ground_plane:
        below_features = splat_points(below)
        features = np.stack([below_features, above_features], axis=-1)
    else:
        features = np.stack([above_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    return features


def bev_params_from_dict(d: Dict[str, Any]) -> BEVParams:
    return BEVParams(
        lidar_min_x=float(d["lidar_min_x"]),
        lidar_max_x=float(d["lidar_max_x"]),
        lidar_min_y=float(d["lidar_min_y"]),
        lidar_max_y=float(d["lidar_max_y"]),
        pixels_per_meter=float(d["pixels_per_meter"]),
        hist_max_per_pixel=int(d["hist_max_per_pixel"]),
        max_height_lidar=float(d["max_height_lidar"]),
        lidar_split_height=float(d["lidar_split_height"]),
        use_ground_plane=bool(d["use_ground_plane"]),
    )
