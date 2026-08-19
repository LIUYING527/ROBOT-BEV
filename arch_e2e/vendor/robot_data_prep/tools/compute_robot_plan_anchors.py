#!/usr/bin/env python3
"""K-means trajectory anchors from robot NPZ manifests (replace NAVSIM kmeans for fine-tune)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="manifest.csv from robot_data_prep")
    ap.add_argument("--output", type=Path, required=True, help="Output .npy shape (K, T, 2)")
    ap.add_argument("--num-modes", type=int, default=20)
    ap.add_argument("--max-samples", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = args.manifest.resolve().parent
    xy_list = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            z = np.load(root / row["npz"], allow_pickle=False)
            traj = z["trajectory"][..., :2].astype(np.float32)
            xy_list.append(traj)
            if len(xy_list) >= args.max_samples:
                break
    if not xy_list:
        raise SystemExit("No trajectories loaded.")

    data = np.stack(xy_list, axis=0)  # N, T, 2
    n, t, _ = data.shape
    flat = data.reshape(n, t * 2)

    rng = np.random.default_rng(args.seed)
    k = min(args.num_modes, n)
    idx = rng.choice(n, size=k, replace=False)
    centers = flat[idx].copy()

    for _ in range(30):
        dist = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        assign = dist.argmin(axis=1)
        for j in range(k):
            mask = assign == j
            if np.any(mask):
                centers[j] = flat[mask].mean(axis=0)

    anchors = centers.reshape(k, t, 2).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, anchors)
    print(f"Saved {anchors.shape} anchors to {args.output}")


if __name__ == "__main__":
    main()
