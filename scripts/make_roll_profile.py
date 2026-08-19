#!/usr/bin/env python3
"""
生成【水平找平 profile】(相机绕光轴的 roll 修正, 随沿走廊 x): 供 test_env 在【相机侧】把
普通 discoverse 渲染的画面转正(治"整体画面斜着")。

⭐关键: 找平量直接取【真实采集相机自身的 roll】(=让 capcam 全程水平的那个基准), 不是拟合地面
法线(远端融化的地面点会把法线拉向竖直→严重低估, 实测真roll远端+6.7° vs 地面法线仅~3°)。
单程室外重建的 SfM roll 漂移让远端场景整体滚了~6.7°, 采集相机同步滚(所以capcam用真外参看着正),
但普通 level 相机看这倾斜场景=画面斜。修=让普通相机也按这个 roll 绕光轴转→画面转正, 且是平滑曲线不抖。

用法: python3 scripts/make_roll_profile.py colmapjoint_111450
产物: outputs/roll_profile_<session>.npz  (x_samp, roll_deg)  # roll_deg=相机绕光轴要补的角度(度)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_profile(session):
    """每张【平滑】采集相机(=capcam渲染用的那套)的 roll + 其对齐世界xy中心。
    test_env 按机器人(x,y)最近邻查此 roll→给普通相机绕光轴补→画面与capcam同款水平。
    按(x,y)最近邻(而非x-median)=正确处理同一x处去程/返程两批不同手持roll。"""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from capture_fpv import CaptureFPV
    fpv = CaptureFPV(session)
    Rz = fpv.Rz
    xy, rolls = [], []
    for i in range(len(fpv._Vs_smooth)):
        Rc2col = fpv._Vs_smooth[i][:3, :3].T          # cam->colmap-world, 列=cam轴
        right_w = Rz @ Rc2col[:, 0]
        roll = np.degrees(np.arcsin(np.clip(right_w[2] / (np.linalg.norm(right_w) + 1e-9), -1, 1)))
        xy.append(fpv._cc_world[i, :2]); rolls.append(float(roll))
    return np.asarray(xy, np.float64), np.asarray(rolls, np.float64)


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "colmapjoint_111450"
    xy, rolls = build_profile(session)
    out = os.path.join(ROOT, "outputs", f"roll_profile_{session}.npz")
    np.savez(out, cam_xy=xy, roll_deg=rolls)
    xs = xy[:, 0]
    print(f"[profile] session={session} 采集帧 {len(rolls)} 张 (按xy最近邻查roll)", flush=True)
    print(f"[profile] 找平roll(度)逐3m均值: " + " ".join(
        f"x{xt}={rolls[(xs>=xt)&(xs<xt+3)].mean():+.1f}" for xt in range(0, 28, 3)
        if ((xs >= xt) & (xs < xt + 3)).any()), flush=True)
    print(f"[profile] 幅度 max|roll|={np.abs(rolls).max():.2f}° mean={np.abs(rolls).mean():.2f}°", flush=True)
    print(f"[ok] -> {out}", flush=True)


if __name__ == "__main__":
    main()
