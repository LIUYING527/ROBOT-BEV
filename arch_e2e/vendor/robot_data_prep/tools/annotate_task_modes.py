"""
Label task modes (cruise / turn / inspect / exit) on a SVTROBO recording session.

Writes spans to <recording_dir>/task_modes.jsonl (same format preprocess reads).
Unlabeled images are skipped during training when preprocessing with --require-mode-labels.

Example:
  python -m robot_data_prep.tools.annotate_task_modes \\
    --recording-dir /mnt/.../2026-04-24/100829 \\
    --start-image /mnt/.../images/zed/1776996513948853.jpg \\
    --end-image   /mnt/.../images/zed/1776996525123456.jpg \\
    --mode cruise

  python -m robot_data_prep.tools.annotate_task_modes --recording-dir ... --report
  python -m robot_data_prep.tools.annotate_task_modes --recording-dir ... --list-unlabeled
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from robot_data_prep.io_discover import list_sorted_media, parse_timestamp_ns_from_name
from robot_data_prep.task_modes import (
    MODES,
    TASK_MODES_AUDIT_FILENAME,
    TASK_MODES_FILENAME,
    apply_range_to_spans,
    load_mode_overrides,
    merge_adjacent_spans,
    mode_for_timestamp,
    save_mode_spans,
    validate_mode,
)


def find_recording_dir(path: Path) -> Path:
    """Walk up from a file or folder until a session root is found."""
    p = path.resolve()
    if p.is_file():
        p = p.parent
    for _ in range(8):
        has_images = (p / "images" / "zed").is_dir() or any(p.glob("images/**/*.jpg"))
        if has_images:
            return p
        p = p.parent
    raise ValueError(
        f"Cannot find recording session root from {path}. "
        "Expected a directory containing images/zed/*.jpg"
    )


def list_session_images(recording_dir: Path) -> List[Tuple[int, Path]]:
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
    if not images:
        raise FileNotFoundError(f"No timestamped images under {recording_dir}/images")
    return images


def image_timestamp_ns(image_path: Path) -> int:
    return parse_timestamp_ns_from_name(image_path.resolve())


def images_in_range(
    images: List[Tuple[int, Path]],
    t_start_ns: int,
    t_end_ns: int,
) -> List[Tuple[int, Path]]:
    if t_start_ns > t_end_ns:
        t_start_ns, t_end_ns = t_end_ns, t_start_ns
    return [(t, p) for t, p in images if t_start_ns <= t <= t_end_ns]


def append_audit_log(
    recording_dir: Path,
    *,
    mode: str,
    t_start_ns: int,
    t_end_ns: int,
    start_image: str,
    end_image: str,
    num_images: int,
) -> None:
    audit_path = recording_dir / TASK_MODES_AUDIT_FILENAME
    rec = {
        "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "t_start_ns": t_start_ns,
        "t_end_ns": t_end_ns,
        "start_image": start_image,
        "end_image": end_image,
        "num_images": num_images,
    }
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def coverage_report(
    recording_dir: Path,
    images: List[Tuple[int, Path]],
    spans: List[Tuple[int, int, str]],
) -> Dict[str, object]:
    labeled = 0
    by_mode: Dict[str, int] = {m: 0 for m in MODES}
    unlabeled: List[Tuple[int, Path]] = []
    for t, p in images:
        m = mode_for_timestamp(t, spans)
        if m is None:
            unlabeled.append((t, p))
        else:
            labeled += 1
            by_mode[m] = by_mode.get(m, 0) + 1
    total = len(images)
    pct = (100.0 * labeled / total) if total else 0.0
    return {
        "recording_dir": str(recording_dir),
        "task_modes_file": str(recording_dir / TASK_MODES_FILENAME),
        "total_images": total,
        "labeled_images": labeled,
        "unlabeled_images": len(unlabeled),
        "coverage_pct": round(pct, 2),
        "by_mode": by_mode,
        "num_spans": len(spans),
        "spans": [
            {"t_start_ns": a, "t_end_ns": b, "mode": m, "num_images": len(images_in_range(images, a, b))}
            for a, b, m in spans
        ],
        "unlabeled_paths": [str(p) for _, p in unlabeled[:20]],
    }


def annotate_range(
    recording_dir: Path,
    start_image: Path,
    end_image: Path,
    mode: str,
    *,
    dry_run: bool = False,
) -> Dict[str, object]:
    recording_dir = recording_dir.resolve()
    start_image = start_image.resolve()
    end_image = end_image.resolve()
    mode = validate_mode(mode)

    if not start_image.is_file():
        raise FileNotFoundError(start_image)
    if not end_image.is_file():
        raise FileNotFoundError(end_image)

    images = list_session_images(recording_dir)
    t0 = image_timestamp_ns(start_image)
    t1 = image_timestamp_ns(end_image)
    selected = images_in_range(images, t0, t1)
    if not selected:
        raise RuntimeError(
            f"No session images between {start_image.name} and {end_image.name} "
            f"({t0} .. {t1} ns). Check --recording-dir."
        )

    t_start_ns = selected[0][0]
    t_end_ns = selected[-1][0]
    spans = load_mode_overrides(recording_dir)
    new_spans = apply_range_to_spans(spans, t_start_ns, t_end_ns, mode)

    result = {
        "recording_dir": str(recording_dir),
        "mode": mode,
        "t_start_ns": t_start_ns,
        "t_end_ns": t_end_ns,
        "start_image": str(start_image),
        "end_image": str(end_image),
        "num_images_labeled": len(selected),
        "first_image": str(selected[0][1]),
        "last_image": str(selected[-1][1]),
        "dry_run": dry_run,
        "spans_before": len(spans),
        "spans_after": len(new_spans),
    }

    if dry_run:
        result["would_write"] = str(recording_dir / TASK_MODES_FILENAME)
        return result

    out_path = save_mode_spans(recording_dir, new_spans)
    append_audit_log(
        recording_dir,
        mode=mode,
        t_start_ns=t_start_ns,
        t_end_ns=t_end_ns,
        start_image=str(start_image),
        end_image=str(end_image),
        num_images=len(selected),
    )
    result["written"] = str(out_path)
    cov = coverage_report(recording_dir, images, new_spans)
    result["coverage_pct"] = cov["coverage_pct"]
    result["unlabeled_images"] = cov["unlabeled_images"]
    return result


def annotate_from_csv(recording_dir: Path, csv_path: Path, *, dry_run: bool = False) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"start_image", "end_image", "mode"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must have columns: {sorted(required)}")
        for row in reader:
            results.append(
                annotate_range(
                    recording_dir,
                    Path(row["start_image"].strip()),
                    Path(row["end_image"].strip()),
                    row["mode"].strip(),
                    dry_run=dry_run,
                )
            )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Annotate cruise/turn/inspect/exit spans on a recording session (task_modes.jsonl)."
    )
    ap.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help="Session folder (contains images/zed/). Inferred from image paths if omitted.",
    )
    ap.add_argument("--start-image", type=Path, default=None, help="First image in range (inclusive).")
    ap.add_argument("--end-image", type=Path, default=None, help="Last image in range (inclusive).")
    ap.add_argument("--mode", choices=list(MODES), default=None, help="Task mode for the range.")
    ap.add_argument(
        "--batch-csv",
        type=Path,
        default=None,
        help="CSV with columns start_image,end_image,mode (batch annotate).",
    )
    ap.add_argument("--report", action="store_true", help="Print coverage report and exit.")
    ap.add_argument("--list-unlabeled", action="store_true", help="Print unlabeled image paths.")
    ap.add_argument("--dry-run", action="store_true", help="Show effect without writing task_modes.jsonl.")
    args = ap.parse_args()

    recording_dir = args.recording_dir
    if recording_dir is None:
        hint = args.start_image or args.end_image or args.batch_csv
        if hint is None:
            ap.error("Provide --recording-dir or an image path to infer it.")
        recording_dir = find_recording_dir(Path(hint))
    else:
        recording_dir = recording_dir.resolve()

    images = list_session_images(recording_dir)
    spans = load_mode_overrides(recording_dir)

    if args.report or args.list_unlabeled:
        report = coverage_report(recording_dir, images, spans)
        if args.list_unlabeled:
            for t, p in images:
                if mode_for_timestamp(t, spans) is None:
                    print(p)
            print(f"# unlabeled: {report['unlabeled_images']} / {report['total_images']}", flush=True)
            return
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.batch_csv:
        if not args.batch_csv.is_file():
            ap.error(f"Batch CSV not found: {args.batch_csv}")
        results = annotate_from_csv(recording_dir, args.batch_csv, dry_run=args.dry_run)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    if not (args.start_image and args.end_image and args.mode):
        ap.error("Annotate requires --start-image, --end-image, and --mode (or use --batch-csv / --report).")

    result = annotate_range(
        recording_dir,
        args.start_image,
        args.end_image,
        args.mode,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
