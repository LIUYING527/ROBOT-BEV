#!/usr/bin/env python
"""从 prepare 生成的 nerfstudio transforms.json 造一条 off-path 新视角轨迹,
喂 ArtiFixer 的 --trajectory_path。off-path = 沿相机 right 轴横向平移(+可选朝墙偏航),
正对准单目前视单程采集没覆盖、hqbest 会拖糊的视角。

输出为 transforms 式 JSON:顶层内参(w,h,fl_x,fl_y,cx,cy) + frames[{transform_matrix}],
**不含 file_path**(target-only,ArtiFixer 要求)。

OpenGL c2w 约定: 列 = [right, up, -forward, position]。
"""
import json, sys, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="prepare 生成的 nerfstudio transforms.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=700)
    ap.add_argument("--count", type=int, default=80)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--lateral", type=float, default=0.35, help="沿 right 轴横向平移(COLMAP 单位)")
    ap.add_argument("--yaw_deg", type=float, default=0.0, help="绕 up 轴朝平移侧偏航角(度,0=朝向不变)")
    args = ap.parse_args()

    d = json.load(open(args.src))
    intr = {k: d[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    frames = d["frames"]
    idxs = list(range(args.start, min(args.start + args.count * args.stride, len(frames)), args.stride))

    yaw = np.deg2rad(args.yaw_deg)
    out_frames = []
    for i in idxs:
        c2w = np.array(frames[i]["transform_matrix"], dtype=np.float64)
        R = c2w[:3, :3].copy()
        pos = c2w[:3, 3].copy()
        right = R[:, 0] / (np.linalg.norm(R[:, 0]) + 1e-9)
        up = R[:, 1] / (np.linalg.norm(R[:, 1]) + 1e-9)
        # 横向平移
        pos = pos + args.lateral * right
        # 朝平移侧偏航(绕 up 轴旋转 R)
        if abs(yaw) > 1e-6:
            ux, uy, uz = up
            K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
            Rrot = np.eye(3) + np.sin(yaw) * K + (1 - np.cos(yaw)) * (K @ K)  # Rodrigues
            R = Rrot @ R
        new = np.eye(4)
        new[:3, :3] = R
        new[:3, 3] = pos
        out_frames.append({"transform_matrix": new.tolist()})

    out = {**intr, "camera_model": "OPENCV", "frames": out_frames}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}: {len(out_frames)} frames, "
          f"src[{args.start}:{args.start+args.count*args.stride}:{args.stride}], "
          f"lateral={args.lateral} yaw={args.yaw_deg}deg")


if __name__ == "__main__":
    main()
