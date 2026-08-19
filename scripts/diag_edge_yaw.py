"""0813晚: EDGE伺服yaw争议裁决——渲染站位yaw扫描, 同_edge_line测斜率+出对比图。
用法: PYTHONPATH=third_party/discoverse ~/discoverse_venv/bin/python scripts/diag_edge_yaw.py
"""
import os, sys, json
import numpy as np
import cv2
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gs_relocate import GSReloc

OUT = os.path.join(ROOT, "outputs", "_diag_edge_yaw")
os.makedirs(OUT, exist_ok=True)

def edge_line(fpv_bgr, v_target=None):
    """与lab_mission._edge_line逐行一致"""
    g = cv2.cvtColor(fpv_bgr, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
    if lines is None:
        return None, []
    Hf, Wf = g.shape
    cand = []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = map(float, l)
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L < 0.08 * Wf:
            continue
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        a = ((a + 90) % 180) - 90
        if abs(a) > 6:
            continue
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        if abs(xm - Wf / 2) > 0.40 * Wf or not (0.10 * Hf < ym < 0.85 * Hf):
            continue
        cand.append((a, ym, L, (x1, y1, x2, y2)))
    if not cand:
        return None, []
    if v_target is not None:
        band = [c for c in cand if abs(c[1] - v_target) < 0.15 * Hf]
        if band:
            cand = band
    Lsum = sum(c[2] for c in cand)
    a_w = sum(c[0] * c[2] for c in cand) / Lsum
    return float(a_w), cand

def main():
    g = GSReloc("colmapjoint_110744s2", "data/lab/110744")
    Wz = np.load(os.path.join(ROOT, "outputs", "sim_world_colmapjoint_110744s2.npz"))
    cam_h = float(Wz["cam_height"])
    tj = json.load(open(os.path.join(ROOT, "outputs", "lab_mission_targets_zed_front_perp_lvl.json")))
    tr = np.load(os.path.join(ROOT, "outputs", "lab_mission_trace.npz"), allow_pickle=True)
    xyyaw = tr["xyyaw_vxvywz"].astype(float)
    Ho, Wo = 360, 640
    pitch = np.radians(-12.0)
    stations = {  # trace站位帧(dock块前)
        "iron_cabinet": 228, "microscope": 545, "pipette_table": 856,
    }
    for nm, fi in stations.items():
        sx, sy = xyyaw[fi, 0], xyyaw[fi, 1]
        fd = tj["front_deg"][nm]
        yaws = [fd - 10, fd - 5, fd, fd + 2.5, fd + 5, fd + 10]
        tiles = []
        print(f"== {nm} 站位({sx:.2f},{sy:.2f}) 地图front={fd}° ==")
        for yd in yaws:
            yaw = np.radians(yd)
            c, s = np.cos(yaw), np.sin(yaw)
            cp, sp = np.cos(pitch), np.sin(pitch)
            # 与capture_fpv._world_to_colmap_cam逐行一致(roll=0): fwd/right/down → R_c2w列
            fwd = np.array([c * cp, s * cp, sp]); right = np.array([s, -c, 0.0])
            down = np.cross(fwd, right); down /= np.linalg.norm(down)
            R = np.stack([right, down, fwd], axis=1)
            C_w = np.array([sx, sy, cam_h])
            V = g.viewmat_from_world(torch.tensor(C_w, device="cuda", dtype=torch.float32),
                                     torch.tensor(R, device="cuda", dtype=torch.float32))
            from gsplat import rasterization
            K = torch.tensor(g.K0, device="cuda", dtype=torch.float32)
            K[0] *= Wo / g.W0; K[1] *= Ho / g.H0
            out, _, _ = rasterization(g.means, g.quats, g.scales, g.opac, g.colors,
                                      V[None], K[None], Wo, Ho, sh_degree=g.deg,
                                      near_plane=0.01, far_plane=1e4, render_mode="RGB")
            img = (out[0, :, :, :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            a_w, cand = edge_line(img)
            vis = img.copy()
            for (a, ym, L, (x1, y1, x2, y2)) in cand:
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
            cv2.putText(vis, f"yaw={yd:+.1f} edge_slope={a_w if a_w is None else round(a_w,2)}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            tiles.append(vis)
            print(f"  yaw={yd:+7.1f}°  edge_slope={a_w if a_w is None else round(a_w,2)}  (cand={len(cand)})")
        sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.imwrite(os.path.join(OUT, f"{nm}_yawsweep.png"), sheet)
    print("->", OUT)

if __name__ == "__main__":
    main()
