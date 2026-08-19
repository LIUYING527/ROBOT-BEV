"""
Diagnose why robot_data_prep produces fewer NPZ samples than labeled images.

Usage:
  python -m robot_data_prep.tools.diagnose_session \\
    --recording-dir /mnt/.../2026-04-24/100829
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from robot_data_prep.cmd_integrate import load_svtrobot_cmd_jsonl
from robot_data_prep.io_discover import list_sorted_media, parse_timestamp_ns_from_name
from robot_data_prep.task_modes import load_mode_overrides, mode_for_timestamp


def _ns_to_s(t_ns: int) -> float:
    return float(t_ns) * 1e-9


def _human_range(t0: int, t1: int) -> str:
    return f"{t0} .. {t1}  ({_ns_to_s(t1 - t0):.2f}s span)"


def diagnose(recording_dir: Path, *, future_horizon_s: float = 4.0) -> Dict[str, Any]:
    recording_dir = recording_dir.resolve()
    max_future_ns = int(round(future_horizon_s * 1e9))

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

    cmd_path = recording_dir / "svtrobot_cmd.jsonl"
    times_ns, _, _, _ = load_svtrobot_cmd_jsonl(cmd_path)
    spans = load_mode_overrides(recording_dir)

    img_ts = [t for t, _ in images]
    img_t0, img_t1 = img_ts[0], img_ts[-1]
    cmd_t0, cmd_t1 = int(times_ns[0]), int(times_ns[-1])

    # Detect common unit mismatch (cmd stored as us, images as us*1000 ns).
    ratio = (img_t0 / cmd_t0) if cmd_t0 else 0.0
    unit_warning = None
    if 900 <= ratio <= 1100:
        unit_warning = (
            f"Image timestamps are ~{ratio:.0f}x cmd _timestamp_ns. "
            "Images use ns from filename (us*1000); cmd may be in microseconds "
            "mislabeled as _timestamp_ns. Preprocess compares raw integers → most anchors fail."
        )

    pass_cmd = 0
    pass_cmd_and_label = 0
    by_mode: Dict[str, int] = {}
    fail_cmd = 0
    fail_label = 0

    for t in img_ts:
        ok_cmd = (t >= cmd_t0) and ((t + max_future_ns) <= cmd_t1)
        if not ok_cmd:
            fail_cmd += 1
            continue
        pass_cmd += 1
        mode = mode_for_timestamp(t, spans)
        if mode is None:
            fail_label += 1
            continue
        pass_cmd_and_label += 1
        by_mode[mode] = by_mode.get(mode, 0) + 1

    labeled_images = sum(1 for t in img_ts if mode_for_timestamp(t, spans) is not None)

    return {
        "recording_dir": str(recording_dir),
        "num_images": len(images),
        "image_time_range_ns": _human_range(img_t0, img_t1),
        "num_cmd_records": int(times_ns.size),
        "cmd_time_range_ns": _human_range(cmd_t0, cmd_t1),
        "cmd_minus_image_end_s": round(_ns_to_s(cmd_t1 - img_t1), 3),
        "future_horizon_s": future_horizon_s,
        "num_mode_spans": len(spans),
        "labeled_images_by_spans": labeled_images,
        "expected_npz_if_only_label_filter": labeled_images,
        "anchors_pass_cmd_horizon": pass_cmd,
        "anchors_pass_cmd_and_label": pass_cmd_and_label,
        "expected_npz_samples": pass_cmd_and_label,
        "dropped_cmd_horizon": fail_cmd,
        "dropped_unlabeled_gap": fail_label,
        "expected_npz_by_mode": by_mode,
        "unit_mismatch_warning": unit_warning,
        "hint": (
            "expected_npz_samples is what preprocess should write with "
            "--require-mode-labels. If this is << labeled_images, cmd ends too early "
            "or timestamp units differ."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose preprocess sample count vs labels.")
    ap.add_argument("--recording-dir", type=Path, required=True)
    ap.add_argument("--future-horizon-s", type=float, default=4.0)
    args = ap.parse_args()
    report = diagnose(args.recording_dir, future_horizon_s=args.future_horizon_s)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
