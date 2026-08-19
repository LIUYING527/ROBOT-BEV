"""
Build DiffusionDrive-style training samples from a SVTROBO recording folder.

Expected layout (typical):
  <recording>/svtrobot_cmd.jsonl
  <recording>/images/zed/*.jpg   (filename timestamp_us)
  <recording>/pointcloud/zed/*.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from robot_data_prep.bev import BEVParams, bev_params_from_dict, load_xyz_from_npz, points_to_bev_tensor
from robot_data_prep.cmd_integrate import (
    future_trajectory_in_ego_frame,
    integrate_world_path,
    interpolate_pose_at_time,
    load_svtrobot_cmd_jsonl,
)
from robot_data_prep.io_discover import list_sorted_media, nearest_by_time, parse_timestamp_ns_from_name
from robot_data_prep.task_modes import build_status_vector, infer_mode_temporal, load_mode_overrides


def _rel_if_possible(path: Path, base: Path) -> str:
    """Relative path when under base; otherwise absolute string (Python 3.8+ safe)."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Install PyYAML: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resize_rgb_u8(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _read_image_bgr(path: Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im


def nearest_cmd_kinematics(
    anchor_ns: int,
    times_ns: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    wz: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Return vx,vy,wz at anchor (interp), and simple acc from neighbor."""
    t = times_ns.astype(np.float64) * 1e-9
    q = float(anchor_ns) * 1e-9
    ivx = float(np.interp(q, t, vx))
    ivy = float(np.interp(q, t, vy))
    iwz = float(np.interp(q, t, wz))
    # crude accel: backward diff
    dt = 0.1
    q0 = max(t[0], q - dt)
    vx0 = float(np.interp(q0, t, vx))
    vy0 = float(np.interp(q0, t, vy))
    ax = (ivx - vx0) / (q - q0 + 1e-6)
    ay = (ivy - vy0) / (q - q0 + 1e-6)
    return ivx, ivy, iwz, ax, ay


def run(
    recording_dir: Path,
    output_dir: Path,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    recording_dir = recording_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    hz = float(defaults["target_hz"])
    time_horizon = float(defaults["time_horizon_s"])
    interval = float(defaults["interval_length_s"])
    k_future = int(round(time_horizon / interval))
    offset_ns = int(defaults.get("time_offset_ns", 0))
    max_dt_img = int(float(defaults["max_delta_image_ms"]) * 1e6)
    max_dt_pc = int(float(defaults["max_delta_lidar_ms"]) * 1e6)
    require_lidar = bool(defaults["require_lidar"])
    mode_heuristic = bool(defaults.get("mode_heuristic", True))
    require_mode_labels = bool(defaults.get("require_mode_labels", False))
    if require_mode_labels:
        mode_heuristic = False
    cw = int(defaults["camera_width"])
    ch = int(defaults["camera_height"])
    bev_p = bev_params_from_dict(defaults)
    mode_spans = load_mode_overrides(recording_dir)

    cmd_path = recording_dir / "svtrobot_cmd.jsonl"
    if not cmd_path.exists():
        raise FileNotFoundError(cmd_path)

    times_ns, vx, vy, wz = load_svtrobot_cmd_jsonl(cmd_path)
    if times_ns.size == 0:
        raise RuntimeError(f"No command records in {cmd_path} (svtrobot_cmd.jsonl is empty).")
    if times_ns.size < 2:
        raise RuntimeError(
            f"Not enough command records in {cmd_path} (need >=2, got {times_ns.size})."
        )
    times_ns, poses_w = integrate_world_path(times_ns, vx, vy, wz)

    images = list_sorted_media(
        recording_dir,
        ("images/zed/*.jpg", "images/zed/*.jpeg", "images/zed/*.png"),
        (".jpg", ".jpeg", ".png"),
    )
    if not images:
        images = list_sorted_media(
            recording_dir,
            ("images/**/*.jpg", "images/**/*.jpeg", "images/**/*.png"),
            (".jpg", ".jpeg", ".png"),
        )
    pcs = list_sorted_media(
        recording_dir,
        ("pointcloud/zed/*.npz",),
        (".npz",),
    )
    if not pcs:
        pcs = list_sorted_media(
            recording_dir,
            ("pointcloud/**/*.npz",),
            (".npz",),
        )
    if not images:
        raise FileNotFoundError(f"No images under {recording_dir}/images")
    if require_lidar and not pcs:
        raise FileNotFoundError(f"No pointcloud npz under {recording_dir}/pointcloud")

    future_times = (np.arange(1, k_future + 1, dtype=np.float64) * interval * 1e9).astype(np.int64)
    max_future_ns = int(future_times[-1]) if len(future_times) else 0

    # Anchor policy: use each image timestamp as one sample anchor.
    # Optional time_offset_ns can shift the anchor for sensor-delay compensation.
    image_ts = np.asarray([t for t, _ in images], dtype=np.int64)
    anchors = image_ts + np.int64(offset_ns)
    valid_cmd_mask = (anchors >= times_ns[0]) & ((anchors + max_future_ns) <= times_ns[-1])
    num_image_anchors = int(anchors.size)
    num_dropped_cmd_horizon = int((~valid_cmd_mask).sum())
    anchors = anchors[valid_cmd_mask]
    if anchors.size == 0:
        raise RuntimeError("No valid anchors: image timestamps do not have enough cmd horizon for trajectory labels.")

    manifest_rows: List[Dict[str, Any]] = []
    skipped = 0
    skipped_unlabeled = 0
    sample_idx = 0

    for a_ns in anchors.tolist():
        a_ns = int(a_ns)
        img_path, dt_img = nearest_by_time(a_ns, images, max_dt_img)
        if img_path is None:
            skipped += 1
            continue
        pc_path, dt_pc = (None, None)
        if pcs:
            pc_path, dt_pc = nearest_by_time(a_ns, pcs, max_dt_pc)
        if require_lidar and pc_path is None:
            skipped += 1
            continue

        pose_a = interpolate_pose_at_time(times_ns, poses_w, np.array([a_ns], dtype=np.int64))[0]
        ft_ns = a_ns + future_times
        poses_f = interpolate_pose_at_time(times_ns, poses_w, ft_ns.astype(np.int64))
        traj_ego = future_trajectory_in_ego_frame(pose_a, poses_f)

        ivx, ivy, iwz, ax, ay = nearest_cmd_kinematics(a_ns, times_ns, vx, vy, wz)
        task_mode = infer_mode_temporal(
            a_ns,
            times_ns,
            vx,
            vy,
            wz,
            mode_spans,
            use_heuristic=mode_heuristic,
        )
        if task_mode is None:
            skipped_unlabeled += 1
            continue
        status = build_status_vector(task_mode, ivx, ivy, ax, ay)

        im_bgr = _read_image_bgr(Path(img_path))
        im_rgb = im_bgr[:, :, ::-1].copy()
        im_small = _resize_rgb_u8(im_rgb, cw, ch)

        if pc_path is not None:
            xyz = load_xyz_from_npz(str(pc_path))
            lidar_t = points_to_bev_tensor(xyz, bev_p)
        else:
            c = 2 if bev_p.use_ground_plane else 1
            h = int((bev_p.lidar_max_x - bev_p.lidar_min_x) * bev_p.pixels_per_meter)
            w = int((bev_p.lidar_max_y - bev_p.lidar_min_y) * bev_p.pixels_per_meter)
            lidar_t = np.zeros((c, h, w), dtype=np.float32)

        out_npz = samples_dir / f"{sample_idx:06d}.npz"
        np.savez_compressed(
            out_npz,
            image_u8=im_small.astype(np.uint8),
            lidar_feature=lidar_t.astype(np.float32),
            trajectory=traj_ego.astype(np.float32),
            status=status.astype(np.float32),
            task_mode=np.array(task_mode),
            anchor_timestamp_ns=np.int64(a_ns),
            dt_image_ns=np.int64(dt_img if dt_img is not None else 0),
            dt_lidar_ns=np.int64(dt_pc if dt_pc is not None else 0),
            image_source=_rel_if_possible(img_path, recording_dir),
            lidar_source=_rel_if_possible(Path(pc_path), recording_dir) if pc_path else "",
        )

        manifest_rows.append(
            {
                "idx": sample_idx,
                "npz": str(out_npz.relative_to(output_dir)),
                "anchor_timestamp_ns": a_ns,
                "dt_image_ms": (dt_img or 0) / 1e6,
                "dt_lidar_ms": (dt_pc or 0) / 1e6,
                "task_mode": task_mode,
            }
        )
        sample_idx += 1

    manifest_path = output_dir / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    summary = {
        "recording_dir": str(recording_dir),
        "output_dir": str(output_dir),
        "anchor_source": "image_timestamps",
        "num_image_anchors": num_image_anchors,
        "dropped_by_cmd_horizon": num_dropped_cmd_horizon,
        "num_anchors_considered": int(len(anchors)),
        "num_samples": sample_idx,
        "skipped": skipped,
        "skipped_unlabeled": skipped_unlabeled,
        "k_future_poses": k_future,
        "hz": hz,
        "mode_heuristic": mode_heuristic,
        "require_mode_labels": require_mode_labels,
        "num_mode_spans": len(mode_spans),
        "defaults": defaults,
    }
    with open(output_dir / "preprocess_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Visualization: first / last / middle sample
    try:
        from robot_data_prep import visualize as viz_mod

        if sample_idx > 0:
            for j in {0, sample_idx // 2, sample_idx - 1}:
                viz_mod.render_sample_npz(samples_dir / f"{j:06d}.npz", viz_dir / f"sample_{j:06d}.png", ch, cw)
    except Exception as e:
        warnings.warn(f"Skipping QA visualization (install matplotlib): {e}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess SVTROBO recording for DiffusionDrive-style training.")
    ap.add_argument("--recording-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--defaults", type=Path, default=None, help="YAML overriding built-in defaults.")
    ap.add_argument("--target-hz", type=float, default=None)
    ap.add_argument("--require-lidar", action="store_true", default=None)
    ap.add_argument("--no-require-lidar", action="store_false", dest="require_lidar")
    ap.add_argument(
        "--mode-heuristic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Bootstrap task_mode from vx/vy/wz when task_modes.jsonl is missing (default from defaults.yaml).",
    )
    ap.add_argument(
        "--require-mode-labels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip image anchors without task_modes.jsonl coverage (unlabeled = dirty data).",
    )
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    base = _load_yaml(here / "defaults.yaml")
    if args.defaults and args.defaults.exists():
        ov = _load_yaml(args.defaults)
        base.update(ov)
    if args.target_hz is not None:
        base["target_hz"] = args.target_hz
    if args.require_lidar is not None:
        base["require_lidar"] = args.require_lidar
    if args.mode_heuristic is not None:
        base["mode_heuristic"] = args.mode_heuristic
    if args.require_mode_labels is not None:
        base["require_mode_labels"] = args.require_mode_labels

    summary = run(args.recording_dir, args.output_dir, base)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
