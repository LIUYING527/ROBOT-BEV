#!/usr/bin/env python3
"""把 110744s2 会话导成 ConceptGraphs(Replica格式) posed RGB-D 数据集。

- RGB:   frames_zed 510帧 → {scene}/results/frame%05d.jpg
- 深度:  data/110744/pointcloud/zed/{ts}.npz 的 xyzrgba z通道(光轴深度mm) → depth%05d.png (uint16, scale=1000)
- 位姿:  cameras.npz(COLMAP帧world2cam) → c2w → 变换到【对齐米制世界】traj.txt(每行16float)
         world = (k/s_norm)·Rz·(x_colmap - center) + t_world  (与 capture_fpv/locate_target 同源,已验证)
- yaml:  dataset_name=replica, camera_params 全分辨率内参

用法: python3 scripts/prep_conceptgraphs_lab.py [--scene lab110744]
"""
import os, sys, glob, re
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = os.path.join(ROOT, "outputs/vggto_colmapjoint_110744s2")
SIMW = os.path.join(ROOT, "outputs/sim_world_colmapjoint_110744s2.npz")
PCDIR = os.path.join(ROOT, "data/110744/pointcloud/zed")
OUT = os.path.join(ROOT, "outputs/cg_lab110744")
TS = re.compile(r"(\d{16})")


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "lab110744"
    sdir = os.path.join(OUT, scene)
    res = os.path.join(sdir, "results")
    os.makedirs(res, exist_ok=True)

    W = np.load(SIMW, allow_pickle=True)
    k, Rz, tw = float(W["k"]), W["Rz"].astype(np.float64), W["t_world"].astype(np.float64)
    center, s = W["center"].astype(np.float64), float(W["s_norm"])
    A = (k / s) * Rz
    b = tw - A @ center                       # x_world = A @ x_colmap + b

    C = np.load(os.path.join(SESS, "cameras.npz"))
    ext, intr = C["extrinsic"].astype(np.float64), C["intrinsic"].astype(np.float64)
    N = len(ext)
    frames = sorted(glob.glob(os.path.join(SESS, "frames_zed", "*")))
    assert len(frames) == N, f"{len(frames)} != {N}"
    pc_idx = {TS.search(os.path.basename(p)).group(1): p for p in glob.glob(os.path.join(PCDIR, "*.npz"))}

    T_align = np.eye(4); T_align[:3, :3] = A; T_align[:3, 3] = b
    traj_lines = []
    n_depth = 0
    for i, f in enumerate(frames):
        # RGB
        img = cv2.imread(f)
        cv2.imwrite(os.path.join(res, f"frame{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # 深度 (z通道 mm → uint16)
        ts = TS.search(os.path.basename(f)).group(1)
        arr = np.load(pc_idx[ts])["xyzrgba"]
        z = arr[:, :, 2].astype(np.float32)
        valid = np.isfinite(z) & (z > 1e-3) & (z < 19990)
        z = np.where(valid, z, 0.0)
        cv2.imwrite(os.path.join(res, f"depth{i:05d}.png"), z.astype(np.uint16))
        n_depth += int(valid.sum() > 0)
        # c2w (COLMAP world2cam → c2w → 对齐世界)
        R, t = ext[i, :3, :3], ext[i, :, 3]
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ t
        c2w_world = T_align @ c2w
        traj_lines.append(" ".join(f"{v:.8f}" for v in c2w_world.flatten()))
    with open(os.path.join(sdir, "traj.txt"), "w") as fp:
        fp.write("\n".join(traj_lines) + "\n")

    K = np.median(intr, axis=0)
    yaml = f"""dataset_name: 'replica'
camera_params:
  image_height: 720
  image_width: 1280
  fx: {K[0,0]:.4f}
  fy: {K[1,1]:.4f}
  cx: {K[0,2]:.4f}
  cy: {K[1,2]:.4f}
  png_depth_scale: 1000.0
  crop_edge: 0
"""
    with open(os.path.join(OUT, f"{scene}.yaml"), "w") as fp:
        fp.write(yaml)
    print(f"[prep] {scene}: {N}帧 (有深度{n_depth}) → {sdir}")
    print(f"[prep] traj.txt 位姿=对齐米制世界; yaml → {OUT}/{scene}.yaml")


if __name__ == "__main__":
    main()
