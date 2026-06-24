"""Part B 交付:在真实 DISCOVERSE 仿真器(MuJoCo + 机器人 + VGGT→3DGS 背景)里,
让机器人沿真实采集轨迹穿行,headless EGL 离屏渲成视频。
证明"VGGT 输出已是仿真器可用资产,且在真仿真器里跑起来"。

用法:
  MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python scripts/sim_walk_discoverse.py [session] [--nframes 240]
产出: outputs/sim_walk_<session>.mp4
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "discoverse"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from sim_walk_common import make_env, ROOT


def resample(wp, yaw, n):
    """把 64 个 waypoint 重采样成 n 个平滑帧位姿。"""
    m = len(wp)
    t = np.linspace(0, m - 1, n)
    i0 = np.floor(t).astype(int).clip(0, m - 1)
    i1 = np.minimum(i0 + 1, m - 1)
    a = (t - i0)[:, None]
    p = wp[i0] * (1 - a) + wp[i1] * a
    # yaw 插值(解卷绕)
    yu = np.unwrap(yaw)
    yi = yu[i0] * (1 - a[:, 0]) + yu[i1] * a[:, 0]
    return p, yi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default="114830")
    ap.add_argument("--nframes", type=int, default=180)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--full", action="store_true",
                    help="渲完整往返(默认只渲前向去程,返程因3DGS前视欠观测会糊)")
    ap.add_argument("--compare", action="store_true",
                    help="双层对比: 每帧同位姿渲 左=几何物理 右=3DGS皮肤, 拼接成 sim_compare_<s>.mp4")
    args = ap.parse_args()

    env, W = make_env(args.session, width=args.width, height=args.height)
    wp, yaw = W["waypoints_xy"], W["yaw"]
    # yaw 直接用 align 存的**相机实际朝向**(npz), 不再从waypoint切向重算
    # (近直线/抖动轨迹下切向yaw会乱转→相机拍进墙里糊;相机朝向=3DGS被观测处=清晰)
    if not args.full:
        # 只取去程:从起点到离起点最远的 waypoint(覆盖最好的前向扫掠)
        far = int(np.argmax(np.linalg.norm(wp - wp[0], axis=1)))
        wp, yaw = wp[:far + 1], yaw[:far + 1]
        print(f"[sim] 去程 waypoints {len(wp)} (到最远点 idx={far})", flush=True)
    poses, yaws = resample(wp, yaw, args.nframes)

    tag = "compare" if args.compare else "walk"
    out = os.path.join(ROOT, "outputs", f"sim_{tag}_{args.session}.mp4")
    # H.264 + yuv420p + faststart:浏览器/QuickTime/Windows 通用可播(cv2 的 mp4v 浏览器不认)
    import imageio.v2 as imageio
    writer = imageio.get_writer(out, fps=30, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "23"])
    mode = "双层对比(左几何/右皮肤)" if args.compare else "3DGS皮肤"
    print(f"[sim] rendering {args.nframes} frames @ {args.width}x{args.height} [{mode}] ...", flush=True)

    def _label(img, text):
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    for i in range(args.nframes):
        env.set_pose(poses[i, 0], poses[i, 1], yaws[i])
        if args.compare:
            env.show_gaussian_img = False
            env.render(); geom = _label(env.frame().copy(), "Geometry (physics)")
            env.show_gaussian_img = True
            if hasattr(env, "gs_renderer"):
                env.gs_renderer.need_rerender = True
            env.render(); skin = _label(env.frame().copy(), "3DGS skin")
            img = np.hstack([geom, skin])
        else:
            env.render()
            img = env.frame()                  # RGB uint8
        writer.append_data(img)
        if i % 30 == 0:
            print(f"  {i}/{args.nframes}  pose=({poses[i,0]:.1f},{poses[i,1]:.1f},{np.degrees(yaws[i]):.0f}deg)"
                  f"  nonblack {(img.sum(2)>10).mean()*100:.0f}%", flush=True)
    writer.close()
    print(f"[ok] -> {out}", flush=True)


if __name__ == "__main__":
    main()
