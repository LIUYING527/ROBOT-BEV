"""把点云(如 corridor_merged.ply)转成 DISCOVERSE 可渲染的 3DGS ply(每点=小各向同性高斯)。
颜色: ZED真彩坏了(float16打包)→ 暂用高度伪彩(turbo), 让结构可见。
用法: ~/discoverse_venv/bin/python scripts/pc2gaussian.py <in.ply> <out.ply> [--scale 0.025]
"""
import sys, argparse
import numpy as np

C0 = 0.28209479177387814


def main():
    import open3d as o3d
    import matplotlib.cm as cm
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=0.025)
    ap.add_argument("--opacity", type=float, default=0.9)
    args = ap.parse_args()
    pc = o3d.io.read_point_cloud(args.inp)
    P = np.asarray(pc.points, np.float32)
    col = np.asarray(pc.colors, np.float32)
    real = not (col.size == 0 or col.max() < 0.02)
    if not real:                               # 无真彩→高度伪彩兜底
        h = P[:, 2]
        hn = np.clip((h - (-0.2)) / (2.8 - (-0.2)), 0, 1)
        col = cm.get_cmap("turbo")(hn)[:, :3].astype(np.float32)
    n = len(P)
    f_dc = (col - 0.5) / C0
    opa = np.full((n, 1), np.log(args.opacity / (1 - args.opacity)), np.float32)
    scl = np.full((n, 3), np.log(args.scale), np.float32)
    rot = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([P, np.zeros((n, 3), np.float32), f_dc, opa, scl, rot], 1).astype(np.float32)
    with open(args.out, "wb") as f:
        hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
        hdr += "".join("property float %s\n" % p for p in fields) + "end_header\n"
        f.write(hdr.encode()); f.write(data.tobytes())
    print(f"[pc2gs] {n} 点 -> {args.out} (scale={args.scale}, {'真彩' if real else '高度伪彩'})")


if __name__ == "__main__":
    main()
