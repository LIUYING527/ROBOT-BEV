"""把 fuse_zed.py 产出的 zed_fused.ply(VGGT 原始帧)变换到仿真器世界帧(z-up 米制),
与 gs_vggto_<s>_world.ply 对齐。供 fit_world_geometry.py 从 ZED 深度提干净几何用。

变换链(复用 align_gs_world 已存的参数, 在 sim_world_<s>.npz 里):
  x_norm  = (x_vggt - center) / s_norm        # VGGT原始 -> train归一化 (.norm.npy 同值)
  x_world = k * (Rz @ x_norm) + t_world        # 归一化 -> 世界(z-up 米制 地面z=0)
用法: ~/discoverse_venv/bin/python scripts/zed_world.py <session>
产出: outputs/vggto_<session>/zed_fused_world.ply
"""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = sys.argv[1] if len(sys.argv) > 1 else "114830c"


def main():
    import open3d as o3d
    vdir = os.path.join(ROOT, "outputs", f"vggto_{S}")
    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{S}.npz"))
    center, s_norm = W["center"].astype(np.float64), float(W["s_norm"])
    Rz, k, t_world = W["Rz"].astype(np.float64), float(W["k"]), W["t_world"].astype(np.float64)

    pc = o3d.io.read_point_cloud(os.path.join(vdir, "zed_fused.ply"))
    P = np.asarray(pc.points, dtype=np.float64)
    Pn = (P - center) / s_norm
    Pw = (k * (Pn @ Rz.T)) + t_world
    pc.points = o3d.utility.Vector3dVector(Pw)
    out = os.path.join(vdir, "zed_fused_world.ply")
    o3d.io.write_point_cloud(out, pc)
    print(f"[zed_world] {len(Pw)} 点 -> {out}  z[{Pw[:,2].min():.2f},{Pw[:,2].max():.2f}]m "
          f"span {np.ptp(Pw[:,0]):.1f}x{np.ptp(Pw[:,1]):.1f}m")


if __name__ == "__main__":
    main()
