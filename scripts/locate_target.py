"""WHERE定位器: 机器人在某处【扫视一圈】→每个朝向渲FPV+深度→VLM报目标方位(left/center/right/none)
→在报到的方向把对应图像列反投影出世界3D点→多帧多方向投票聚合→目标世界坐标(x,y)。
这是'机器人自己找命名目标'的核心: 语义(VLM认物+方位)+几何(点云反投影)+聚合, 不预先手标坐标。

用法(库): loc = TargetLocator(session); xy, dbg = loc.locate("the microscope", scan_xy=(0.4,-0.5))
自测:  ~/discoverse_venv/bin/python scripts/locate_target.py colmapjoint_110744s2 "the microscope"
"""
import os, sys, re, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arch_e2e", "pipeline"))
from capture_fpv import CaptureFPV
import sim_fullchain as sf   # vlm_api, 默认自研 10.0.0.18

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_bbox(ans):
    """解析robo2vlm归一化bbox '[x1,y1,x2,y2]'(0~1) → (cx,cy,w,h) 归一化, 或None。
    ⭐铁律: 只信bbox不信口头left/right(VLM口头方位是垃圾)。"""
    nums = re.findall(r"[-+]?\d*\.?\d+", ans or "")
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in nums[:4]]
    # 若像素坐标(>1.5)则无法确定分辨率, 跳过; robo2vlm这里输出归一化
    if max(x1, y1, x2, y2) > 1.5:
        return None
    x1, x2 = sorted([x1, x2]); y1, y2 = sorted([y1, y2])
    w, h = x2 - x1, y2 - y1
    if w < 0.01 or h < 0.01 or w > 0.98 or h > 0.98:   # 太小=噪声, 太大=整图(没真定位)
        return None
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2), w, h)


class TargetLocator:
    def __init__(self, session):
        self.fpv = CaptureFPV(session)
        self.session = session

    def locate(self, target_phrase, scan_xy, n_dirs=12, max_new=48, verbose=True):
        """在 scan_xy 处原地扫 n_dirs 个朝向, 定位 target_phrase 的世界(x,y)。
        返回 (xy 或 None, debug列表)。"""
        x, y = float(scan_xy[0]), float(scan_xy[1])
        q = (f"Detect {target_phrase}. Output its bounding box as [x1,y1,x2,y2] normalized 0-1. "
             "If not visible, answer none.")
        hits = []   # 反投影得到的世界点
        dbg = []
        for k in range(n_dirs):
            yaw = -np.pi + 2 * np.pi * k / n_dirs
            V = self.fpv._world_to_colmap_cam(x, y, yaw)
            rgb, depth = self.fpv._render_rgbd_V(V)     # rgb=(H,W) 输出分辨率; depth=(Hr,Wr) 渲染分辨率
            Hr, Wr = depth.shape
            ans, _ = sf.vlm_api(rgb, q, max_new_tokens=max_new, timeout=40)
            bb = _parse_bbox(ans)
            rec = {"k": k, "yaw": float(np.degrees(yaw)), "ans": (ans or "").strip()[:22], "world": None}
            if bb is not None:
                cxn, cyn, wn, hn = bb
                uu, vv = int(cxn * Wr), int(cyn * Hr)                # bbox中心像素(渲染分辨率)
                # bbox内一小片深度中位(避噪), 反投影
                du, dv = max(2, int(0.15 * wn * Wr)), max(2, int(0.15 * hn * Hr))
                patch = depth[max(0, vv - dv):vv + dv, max(0, uu - du):uu + du]
                valid = patch[(patch > 0.05) & (patch < 40)]
                if len(valid) > 15:
                    d = float(np.median(valid))
                    pw = self.fpv.backproject_world(V, uu, vv, d)
                    if pw[2] < 2.2:                      # 排除天花板/远墙误命中
                        hits.append(pw[:2]); rec["world"] = [round(float(pw[0]), 2), round(float(pw[1]), 2)]
            dbg.append(rec)
            if verbose:
                print(f"  dir{k:2d} yaw{np.degrees(yaw):4.0f}  VLM='{rec['ans']:22s}'  world={rec['world']}", flush=True)
        if len(hits) < 2:
            return None, dbg
        H = np.array(hits)
        # ⭐密度聚类取最大一致簇(小房间多方向可见+偶发误框→点散, 取最密簇=真目标)
        best_inl, best_c = None, None
        for i in range(len(H)):
            d = np.linalg.norm(H - H[i], axis=1)
            inl = d < 0.6                                   # 0.6m半径邻域
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl, best_c = inl, H[i]
        xy = H[best_inl].mean(0)
        if verbose:
            print(f"  [聚合] {len(H)}命中 → 最大簇{best_inl.sum()}点 → {np.round(xy,2)}", flush=True)
        return xy, dbg


    def locate_from_captures(self, target_phrase, frame_dir, n_sample=40, verbose=True):
        """⭐从真实采集帧定位(比合成扫描稳: 真图锐→VLM bbox准; 深度用该帧真实外参渲染=同位姿对齐)。
        对采样的真实帧: VLM在真图出bbox → 用_Vs[j]渲深度 → bbox中心反投影世界 → 全局密度聚类。"""
        import glob
        frames = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        n = min(len(frames), len(self.fpv._Vs))
        idx = np.linspace(0, n - 1, min(n_sample, n)).astype(int)
        q = (f"Detect {target_phrase}. Output its bounding box as [x1,y1,x2,y2] normalized 0-1. "
             "If not visible, answer none.")
        hits = []
        for j in idx:
            real = cv2.imread(frames[j])
            if real is None: continue
            ans, _ = sf.vlm_api(real, q, max_new_tokens=48, timeout=40)
            bb = _parse_bbox(ans)
            if bb is None: continue
            cxn, cyn, wn, hn = bb
            V = self.fpv._Vs[j]
            _, depth = self.fpv._render_rgbd_V(V)
            Hr, Wr = depth.shape
            uu, vv = int(cxn * Wr), int(cyn * Hr)
            du, dv = max(2, int(0.15 * wn * Wr)), max(2, int(0.15 * hn * Hr))
            patch = depth[max(0, vv - dv):vv + dv, max(0, uu - du):uu + du]
            valid = patch[(patch > 0.05) & (patch < 40)]
            if len(valid) < 15: continue
            pw = self.fpv.backproject_world(V, uu, vv, float(np.median(valid)))
            if pw[2] < 2.2:
                hits.append(pw[:2])
                if verbose: print(f"  f#{j:3d} bbox={np.round(bb,2)} -> ({pw[0]:.2f},{pw[1]:.2f})", flush=True)
        if len(hits) < 3:
            return None, hits
        H = np.array(hits)
        best_inl = None
        for i in range(len(H)):
            inl = np.linalg.norm(H - H[i], axis=1) < 0.5
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl = inl
        xy = H[best_inl].mean(0)
        if verbose: print(f"  [聚合] {len(H)}命中 最大簇{best_inl.sum()} → {np.round(xy,2)}", flush=True)
        return xy, hits

    def locate_multi(self, target_phrase, scan_pts, n_dirs=12, verbose=True):
        """⭐多视角三角化: 从多个扫视点各扫一圈, 汇总所有反投影点→全局密度聚类取最大簇。
        真目标从多点被稳定命中→累积密簇; 偶发误框→散点被稀释。返回 (xy, all_hits)。"""
        all_hits = []
        for sp in scan_pts:
            if verbose: print(f"  --- scan@({sp[0]:.1f},{sp[1]:.1f}) ---", flush=True)
            _, dbg = self.locate(target_phrase, sp, n_dirs=n_dirs, verbose=verbose)
            for d in dbg:
                if d["world"]: all_hits.append(d["world"])
        if len(all_hits) < 3:
            return None, all_hits
        H = np.array(all_hits)
        best_inl, best_c = None, None
        for i in range(len(H)):
            inl = np.linalg.norm(H - H[i], axis=1) < 0.6
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl, best_c = inl, H[i]
        xy = H[best_inl].mean(0)
        if verbose:
            print(f"  [多视角聚合] {len(H)}命中 → 最大簇{best_inl.sum()}点 → {np.round(xy,2)}", flush=True)
        return xy, all_hits


if __name__ == "__main__":
    sess = sys.argv[1] if len(sys.argv) > 1 else "colmapjoint_110744s2"
    target = sys.argv[2] if len(sys.argv) > 2 else "the microscope"
    sx = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    sy = float(sys.argv[4]) if len(sys.argv) > 4 else -0.5
    loc = TargetLocator(sess)
    print(f"[locate] target='{target}' scan@({sx},{sy})")
    xy, dbg = loc.locate(target, (sx, sy))
    nhit = sum(1 for d in dbg if d["world"])
    print(f"[locate] {nhit}/{len(dbg)}方向命中 → 目标世界坐标 = {None if xy is None else np.round(xy,2)}")
