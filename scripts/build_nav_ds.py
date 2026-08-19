"""为 191715 场景建最小 nav_ds/meta.npz, 供 RealFrameLookup(位姿→最近真实帧, 模拟真机相机)。
   注意: 这只模拟【传感器】(按位姿取真图), 不给机器人任何预知坐标/地图 —— 决策仍只看图, 守真机约束。
poses = waypoints_xy(世界x,y) + yaw ; fpv = frames_zed 真图绝对路径(330张, 与waypoints 1:1)。
用法: ~/discoverse_venv/bin/python scripts/build_nav_ds_191715.py
"""
import os, glob, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
SESSION = "colmapjoint_" + (sys.argv[1] if len(sys.argv)>1 else "demo191715")
OUT = os.path.join(ROOT, "outputs", "nav_ds_" + (sys.argv[1] if len(sys.argv)>1 else "demo191715"))


def main():
    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{SESSION}.npz"))
    wp = W["waypoints_xy"].astype(np.float32)          # (N,2)
    yaw = W["yaw"].astype(np.float32)                  # (N,)
    frames = sorted(glob.glob(os.path.join(ROOT, "outputs", f"vggto_{SESSION}", "frames_zed", "*.jpg")))
    N = min(len(wp), len(yaw), len(frames))
    assert N > 0, "没找到帧/waypoints"
    if not (len(wp) == len(yaw) == len(frames)):
        print(f"[warn] 数目不齐 wp={len(wp)} yaw={len(yaw)} frames={len(frames)} -> 取前 {N}")
    poses = np.stack([wp[:N, 0], wp[:N, 1], yaw[:N]], 1).astype(np.float32)   # (N,3)
    fpv = np.array([os.path.abspath(frames[i]) for i in range(N)])
    merged_ply = os.path.join(ROOT, "outputs", f"gs_vggto_{SESSION}_world.ply")

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "meta.npz"), poses=poses, fpv=fpv, merged_ply=merged_ply)
    print(f"[ok] {OUT}/meta.npz  N={N}")
    print(f"     pose范围 x[{poses[:,0].min():.2f},{poses[:,0].max():.2f}] "
          f"y[{poses[:,1].min():.2f},{poses[:,1].max():.2f}] yaw[{poses[:,2].min():.2f},{poses[:,2].max():.2f}]")
    print(f"     首帧 {fpv[0]}")


if __name__ == "__main__":
    main()
