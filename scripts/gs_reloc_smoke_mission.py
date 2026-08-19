"""RELOC闭环(伪传感器路径)单元冒烟: 取sim_world采集位姿当真位姿, 渲染ED→伪zed观测,
注入漂移→reloc_correct同数学优化→测残差。验证_cam_CR/viewmat一致性, 不烧整趟mission。
用法: CUDA_VISIBLE_DEVICES=5 ~/discoverse_venv/bin/python scripts/gs_reloc_smoke_mission.py
"""
import os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gs_relocate import GSReloc

PITCH = np.radians(-22.0)          # lab_mission capfpv.pitch
CAM_H = 1.2                        # capfpv.cam_height(世界系)


def cam_CR(x, y, yaw):
    C_w = np.array([x, y, CAM_H], np.float64)
    c, s = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(PITCH), np.sin(PITCH)
    fwd = np.array([c * cp, s * cp, sp]); right = np.array([s, -c, 0.0])
    down = np.cross(fwd, right); down /= np.linalg.norm(down)
    return C_w, np.stack([right, down, fwd], axis=1)


def main():
    g = GSReloc("colmapjoint_110744s2", "data/lab/110744")
    W = np.load(os.path.join(ROOT, "outputs", "sim_world_colmapjoint_110744s2.npz"))
    wp, yaws = W["waypoints_xy"], W["yaw"]
    rng = np.random.RandomState(0)
    print(f"{'帧':>4} {'漂移':>12} {'残差':>14} {'yaw残':>7} loss")
    for j in (50, 150, 250, 350, 450):
        x, y, yaw = float(wp[j, 0]), float(wp[j, 1]), float(yaws[j])
        obs = {}
        with torch.no_grad():
            C_t, R_t = cam_CR(x, y, yaw)
            for (Wo, Ho) in ((320, 180), (640, 360)):
                V_t = g.viewmat_from_world(torch.tensor(C_t, device="cuda", dtype=torch.float32),
                                           torch.tensor(R_t, device="cuda", dtype=torch.float32))
                ed, al = g.render_ed(V_t, Ho, Wo)
                d_real = (ed / g.obs_scale).cpu().numpy()
                sig = 0.01 + 0.01 * d_real
                d_n = d_real + rng.normal(0, sig).astype(np.float32)
                m = (al.cpu().numpy() > 0.6) & (d_real > 0.3) & (d_real < 12.0) & (rng.rand(Ho, Wo) > 0.05)
                obs[(Wo, Ho)] = (torch.tensor(np.where(m, d_n * g.obs_scale, 0.0).astype(np.float32), device="cuda"),
                                 torch.tensor(m, device="cuda"))
        # 注入漂移
        rr = 0.25 * (0.6 + 0.4 * rng.rand()); aa = rng.uniform(0, 2 * np.pi)
        dx, dy = rr * np.cos(aa), rr * np.sin(aa)
        dyaw = np.radians(rng.uniform(4, 14) * rng.choice([-1, 1]))
        C_b, R_b = cam_CR(x + dx, y + dy, yaw + dyaw)
        C1, R1, trace = g.optimize(0, C_b, R_b, obs=obs)
        res = float(np.hypot(C1[0] - x, C1[1] - y))
        yaw1 = float(np.arctan2(R1[1, 2], R1[0, 2]))
        yres = abs(np.degrees((yaw1 - yaw + np.pi) % (2 * np.pi) - np.pi))
        print(f"{j:>4} {rr:>5.2f}m/{abs(np.degrees(dyaw)):>4.1f}° {res*100:>8.1f}cm {yres:>6.2f}° "
              f"{trace[0]:.3f}→{trace[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
