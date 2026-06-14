"""数据格式自动探测。

拿到数据后第一时间跑这个，把输出贴出来，用于确认 P0 阻塞项：
  - 深度图 dtype / 单位 / 通道
  - RGB-Depth 对应关系与分辨率
  - 总帧数

用法：
    cd robot_bev_sim
    python scripts/01_check_data.py
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

COLOR_DIR = "data/color"
DEPTH_DIR = "data/depth"


def main():
    color_files = sorted(f for f in os.listdir(COLOR_DIR) if not f.startswith("."))
    depth_files = sorted(f for f in os.listdir(DEPTH_DIR) if not f.startswith("."))

    print(f"[INFO] Color frames: {len(color_files)}")
    print(f"[INFO] Depth frames: {len(depth_files)}")
    if not color_files or not depth_files:
        print("[WARN] 数据目录为空。请先把数据拷到 data/color 和 data/depth。")
        return

    print(f"[INFO] First color: {color_files[0]}")
    print(f"[INFO] First depth: {depth_files[0]}")
    if len(color_files) != len(depth_files):
        print(f"[WARN] 帧数不一致：color={len(color_files)} vs depth={len(depth_files)}")

    color = cv2.imread(os.path.join(COLOR_DIR, color_files[0]), cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(os.path.join(DEPTH_DIR, depth_files[0]), cv2.IMREAD_UNCHANGED)

    print("\n=== Color ===")
    print(f"  shape: {color.shape}, dtype: {color.dtype}")
    print(f"  range: [{color.min()}, {color.max()}]")

    print("\n=== Depth ===")
    print(f"  shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"  range: [{depth.min()}, {depth.max()}]")
    print(f"  非零像素数: {(depth > 0).sum()}")
    if (depth > 0).any():
        nz = depth[depth > 0]
        print(f"  非零深度均值: {nz.mean():.2f}  中位数: {np.median(nz):.2f}")
        print("  → 若均值在几百~上万量级，多半是 uint16 毫米图 (DEPTH_SCALE=1000)")
        print("  → 若均值在 0~10 量级，多半已是米为单位的浮点/归一化图")

    # 分辨率一致性检查
    if color.shape[:2] != depth.shape[:2]:
        print(f"\n[WARN] RGB 与 Depth 分辨率不一致："
              f"color={color.shape[:2]} depth={depth.shape[:2]}，可能未 align。")

    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if color.ndim == 3:
        axes[0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(color, cmap="gray")
    axes[0].set_title("Color")
    im = axes[1].imshow(depth, cmap="viridis")
    axes[1].set_title(f"Depth ({depth.dtype})")
    fig.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig("outputs/data_check.png", dpi=120)
    print("\n[OK] 已保存 outputs/data_check.png")


if __name__ == "__main__":
    main()
