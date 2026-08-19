"""QA visualization for preprocessed robot samples (.npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def render_sample_npz(
    npz_path: Path,
    out_png: Path,
    image_height: int,
    image_width: int,
) -> None:
    """
    Save a PNG: camera RGB | LiDAR BEV (channel sum) | trajectory (x–y in ego).

    image_height / image_width are the resized camera dimensions stored in the npz (for titles only).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=False)
    rgb = data["image_u8"]
    lidar = np.asarray(data["lidar_feature"], dtype=np.float32)
    traj = np.asarray(data["trajectory"], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"camera {rgb.shape[1]}×{rgb.shape[0]}")
    axes[0].axis("off")

    bev = lidar.sum(axis=0) if lidar.ndim == 3 else lidar
    axes[1].imshow(bev, cmap="magma", origin="lower")
    axes[1].set_title(f"BEV C={lidar.shape[0]} {bev.shape[1]}×{bev.shape[0]}")
    axes[1].axis("off")

    xs, ys = traj[:, 0], traj[:, 1]
    axes[2].plot(xs, ys, "o-", color="C0")
    axes[2].scatter([0.0], [0.0], c="g", s=40, zorder=5)
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("x (ego)")
    axes[2].set_ylabel("y (ego)")
    axes[2].set_title("future trajectory (ego)")

    fig.suptitle(f"{npz_path.name}  (target {image_width}×{image_height})", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
