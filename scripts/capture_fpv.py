"""CaptureFPV: 用【真实采集相机内参 + gsplat】在机器人任意世界位姿渲染 FPV(=capture-cam质量, 锐且正)。
把机器人世界位姿(x,y,yaw)映射回 COLMAP 帧, 渲染 pre-align 归一化高斯(与 render_best_walkthrough 同管线)。
映射(来自 align_gs_world): world = k·Rz·norm + t_world; colmap = s_norm·norm + center。
用法: 见 __main__ 自测 / run_inspection 集成。
"""
import os, numpy as np, torch, torch.nn.functional as F, pycolmap, cv2
from gsplat import rasterization
DEV = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_gauss(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"; f.readline()
        n = int(f.readline().split()[-1]); props = []
        ln = f.readline()
        while ln.strip() != b"end_header":
            if ln.startswith(b"property"): props.append(ln.split()[-1].decode())
            ln = f.readline()
        d = np.frombuffer(f.read(n * len(props) * 4), np.float32).reshape(n, len(props))
    g = {p: d[:, i] for i, p in enumerate(props)}
    means = torch.tensor(np.stack([g["x"], g["y"], g["z"]], 1), device=DEV)
    quats = F.normalize(torch.tensor(np.stack([g["rot_0"], g["rot_1"], g["rot_2"], g["rot_3"]], 1), device=DEV), dim=-1)
    scales = torch.tensor(np.exp(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1)), device=DEV)
    opac = torch.tensor(1 / (1 + np.exp(-g["opacity"])), device=DEV)
    fdc = np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1)[:, None, :]
    rest = [k for k in props if k.startswith("f_rest_")]
    if rest:
        fr = np.stack([g[k] for k in sorted(rest, key=lambda x: int(x.split("_")[-1]))], 1)
        fr = fr.reshape(len(means), 3, -1).transpose(0, 2, 1)
        sh = np.concatenate([fdc, fr], axis=1); deg = int(round(sh.shape[1] ** 0.5)) - 1
    else:
        sh = fdc; deg = 0
    colors = torch.tensor(sh, device=DEV)
    cs = np.load(path + ".norm.npy"); center, s = cs[:3].astype(np.float32), float(cs[3])
    means = means * s + torch.tensor(center, device=DEV); scales = scales * s
    return means, quats, scales, opac, colors, deg


class CaptureFPV:
    def __init__(self, session, W=640, H=400, ss=1.5):
        short = session[len("colmapjoint_"):] if session.startswith("colmapjoint_") else session
        ply = os.path.join(ROOT, "outputs", f"gs_vggto_{session}.ply")
        self.means, self.quats, self.scales, self.opac, self.colors, self.deg = _load_gauss(ply)
        # 刚性 world<->colmap 参数
        Wz = np.load(os.path.join(ROOT, "outputs", f"sim_world_{session}.npz"))
        self.Rz = Wz["Rz"].astype(np.float64); self.k = float(Wz["k"])
        self.t_world = Wz["t_world"].astype(np.float64); self.center = Wz["center"].astype(np.float64)
        self.s_norm = float(Wz["s_norm"]); self.cam_height = float(Wz["cam_height"])
        # 真实相机内参(COLMAP), 按输出分辨率缩放
        for sub in ("sparse/2", "sparse/1", "sparse/0"):
            d = os.path.join(ROOT, "outputs", f"_colmap_joint_{short}", sub)
            if os.path.isdir(d):
                rec = pycolmap.Reconstruction(d)
                if rec.cameras:
                    cam = max(rec.cameras.values(), key=lambda c: c.width * c.height); break
        K0 = np.array(cam.calibration_matrix(), np.float64); W0, H0 = int(cam.width), int(cam.height)
        self.W, self.H = W, H
        self.Wr, self.Hr = int(W * ss), int(H * ss)
        # K 从采集分辨率缩放到渲染分辨率(x/y独立, 保持视场)
        K = K0.copy(); K[0] *= self.Wr / W0; K[1] *= self.Hr / H0
        self.K = torch.tensor(K, device=DEV, dtype=torch.float32)[None]
        # ⭐采集相机的真实俯仰角(手持采集通常俯视~13°): 机器人相机必须匹配, 否则viewing angle偏离训练视角=糊+透视错(显歪)
        self.pitch = self._capture_pitch(rec)
        print(f"[capfpv] 匹配采集相机俯仰 pitch={np.degrees(self.pitch):+.1f}°", flush=True)
        # ⭐每张采集图的【精确外参】V(world->cam,colmap帧) + 相机中心在对齐世界(供render_nearest)
        ims = sorted(rec.images.values(), key=lambda im: im.name)
        Vs = []; cc_world = []
        for im in ims:
            R = np.array(im.cam_from_world().rotation.matrix(), np.float64)
            t = np.array(im.cam_from_world().translation, np.float64)
            V = np.eye(4); V[:3, :3] = R; V[:3, 3] = t; Vs.append(V)
            cc = -R.T @ t                                            # colmap帧相机中心
            norm = (cc - self.center) / self.s_norm
            cc_world.append(self.k * (self.Rz @ norm) + self.t_world)   # → 对齐世界
        self._Vs = np.stack(Vs); self._cc_world = np.stack(cc_world)
        from scipy.spatial import cKDTree
        self._captree = cKDTree(self._cc_world[:, :2])
        self._Vs_smooth = self._smooth_extrinsics(self._Vs, win=7)   # ⭐去手持抖动(roll±14/pitch范围大→帧间跳=画面抖)
        # ⭐render_smooth 用: 平滑外参的旋转四元数(xyzw) + colmap帧相机中心, 供沿采集序列【连续插值】
        from scipy.spatial.transform import Rotation as _Rt
        Rsm = self._Vs_smooth[:, :3, :3]; tsm = self._Vs_smooth[:, :3, 3]
        self._q_sm = _Rt.from_matrix(Rsm).as_quat()                  # (N,4) xyzw
        for i in range(1, len(self._q_sm)):
            if np.dot(self._q_sm[i], self._q_sm[i - 1]) < 0: self._q_sm[i] = -self._q_sm[i]
        self._Ccol_sm = np.einsum("nij,nj->ni", np.transpose(Rsm, (0, 2, 1)), -tsm)  # -R^T t = 相机中心(colmap帧)
        # ⭐每帧真实相机 roll(right向量在对齐世界的z倾角)=render(x,y,yaw)重建时要补的roll→画面与真实采集同款水平
        rw = np.einsum("ij,njk->nik", self.Rz, np.transpose(Rsm, (0, 2, 1)))[:, :, 0]  # 各帧cam right在对齐世界
        self._roll_deg = np.degrees(np.arcsin(np.clip(rw[:, 2] / (np.linalg.norm(rw, axis=1) + 1e-9), -1, 1)))

    @staticmethod
    def _smooth_extrinsics(Vs, win=7):
        """沿采集序列滑动平滑相机旋转(四元数)+平移, 消手持抖动→FPV流畅。"""
        from scipy.spatial.transform import Rotation
        R = Vs[:, :3, :3]; t = Vs[:, :3, 3]
        q = Rotation.from_matrix(R).as_quat()                        # (N,4) xyzw
        for i in range(1, len(q)):                                   # 消双重覆盖符号翻转
            if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
        k = win
        pad_q = np.pad(q, ((k, k), (0, 0)), mode="edge")
        pad_t = np.pad(t, ((k, k), (0, 0)), mode="edge")
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        qs = np.stack([np.convolve(pad_q[:, c], ker, "valid") for c in range(4)], 1)
        ts = np.stack([np.convolve(pad_t[:, c], ker, "valid") for c in range(3)], 1)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        Rs = Rotation.from_quat(qs).as_matrix()
        out = np.tile(np.eye(4), (len(Vs), 1, 1))
        out[:, :3, :3] = Rs; out[:, :3, 3] = ts
        return out

    def _capture_pitch(self, rec):
        ps = []
        for im in rec.images.values():
            Rc2w = np.array(im.cam_from_world().rotation.matrix()).T
            fwd = self.Rz @ Rc2w[:, 2]; fwd /= np.linalg.norm(fwd)   # 相机前向在对齐世界
            ps.append(np.arcsin(np.clip(fwd[2], -1, 1)))
        return float(np.median(ps))

    def _roll_at(self, x, y, k=9):
        """机器人(x,y)最近k采集帧的真实相机roll中位(度)=该处场景倾斜→render重建时补上即水平。"""
        _, idx = self._captree.query([x, y], k=k)
        return float(np.median(self._roll_deg[np.atleast_1d(idx)]))

    def _world_to_colmap_cam(self, x, y, yaw, roll_deg=None):
        """机器人世界位姿 → COLMAP帧的 view matrix(world->cam, OpenCV)。
        roll_deg=None→自动取该处真实采集相机roll(画面与采集同款水平); 传0=水平相机(旧行为)。"""
        C_w = np.array([x, y, self.cam_height], np.float64)
        c, s = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)             # 匹配采集俯仰
        # OpenCV cam-to-world(世界): x右 y下 z前; 前向按pitch俯视(sp<0向下)
        fwd = np.array([c * cp, s * cp, sp]); right = np.array([s, -c, 0.0])
        down = np.cross(fwd, right); down /= np.linalg.norm(down)
        # ⭐补场景roll: 绕前向轴把right/down旋roll_deg→画面与真实采集同款水平(治单程远端整体斜~6.7°)
        rr = np.radians(self._roll_at(x, y) if roll_deg is None else roll_deg)
        cr, sr = np.cos(rr), np.sin(rr)
        right, down = right * cr + down * sr, -right * sr + down * cr
        R_c2w = np.stack([right, down, fwd], axis=1)                 # 列=cam轴在世界
        # → colmap帧: 方向 norm=Rz.T@world_dir, colmap dir=norm dir; 位置 colmap=s_norm·Rz.T·(C-t)/k+center
        R_c2col = self.Rz.T @ R_c2w
        C_col = self.s_norm * (self.Rz.T @ (C_w - self.t_world) / self.k) + self.center
        R_view = R_c2col.T; t_view = -R_view @ C_col
        V = np.eye(4); V[:3, :3] = R_view; V[:3, 3] = t_view
        return V

    def _render_V(self, V):
        vm = torch.tensor(V, device=DEV, dtype=torch.float32)[None]
        with torch.no_grad():
            img, _, _ = rasterization(self.means, self.quats, self.scales, self.opac, self.colors,
                                      vm, self.K, self.Wr, self.Hr, sh_degree=self.deg,
                                      near_plane=0.01, far_plane=1e4)
        rgb = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)   # 超采样降回=抗锯齿

    def _render_rgbd_V(self, V):
        """渲染 RGB + 期望深度(colmap归一化单位, 沿光轴z)。用于点云反投影定位。"""
        vm = torch.tensor(V, device=DEV, dtype=torch.float32)[None]
        with torch.no_grad():
            out, _, _ = rasterization(self.means, self.quats, self.scales, self.opac, self.colors,
                                      vm, self.K, self.Wr, self.Hr, sh_degree=self.deg,
                                      near_plane=0.01, far_plane=1e4, render_mode="RGB+ED")
        arr = out[0].cpu().numpy()
        rgb = (np.clip(arr[:, :, :3], 0, 1) * 255).astype(np.uint8)
        depth = arr[:, :, 3].astype(np.float32)                       # colmap单位
        return cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA), depth

    def backproject_world(self, V, u, v, depth_col):
        """渲染分辨率像素(u,v)+colmap深度 → 对齐世界3D坐标(米)。
        (u,v)在渲染分辨率(Wr,Hr)下; depth_col=该像素沿相机光轴深度(colmap单位)。"""
        K = self.K[0].cpu().numpy().astype(np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p_cam = np.array([(u - cx) / fx * depth_col, (v - cy) / fy * depth_col, depth_col], np.float64)
        R, t = V[:3, :3], V[:3, 3]
        C_col = -R.T @ t                                              # colmap相机中心
        p_col = R.T @ p_cam + C_col                                   # colmap世界点
        p_world = self.t_world + (self.k / self.s_norm) * (self.Rz @ (p_col - self.center))
        return p_world                                                # (3,) 对齐世界(米)

    def render(self, x, y, yaw):
        """从(x,y,yaw)重建位姿渲染(机器人任意朝向, 如转身/横扫; 但丢采集pitch/roll→偏)。"""
        return self._render_V(self._world_to_colmap_cam(x, y, yaw))

    def render_nearest(self, x, y):
        """⭐用【最近采集帧的平滑精确外参】渲染=walk_111450_capcam同款锐且正+去手持抖动流畅。巡航时用。"""
        _, j = self._captree.query([x, y])
        return self._render_V(self._Vs_smooth[j])

    def render_smooth(self, x, y):
        """⭐又正又不抖: 沿采集序列【连续插值】平滑外参渲染(替代render_nearest的离散抓帧)。
        capcam"正"来自真实采集外参; "抖"来自render_nearest按位置抓最近离散帧(机器人连续移动→帧下标跳/停)。
        这里把(x,y)投影到采集轨迹相邻帧的线段→分数位置tt→SLERP旋转+lerp相机中心→连续相机=不抖, 外参仍真实=正。"""
        from scipy.spatial.transform import Rotation as _Rt
        _, j = self._captree.query([x, y])
        cc = self._cc_world[:, :2]; p = np.array([x, y], np.float64)
        # 在采集序列相邻两段(j-1,j)/(j,j+1)里挑投影落在段内且最近的→得端点a,b与分数tt∈[0,1]
        best = (j, min(j + 1, len(cc) - 1), 0.0, np.inf)
        for a in (j - 1, j):
            b = a + 1
            if a < 0 or b >= len(cc): continue
            d = cc[b] - cc[a]; L2 = float(d @ d)
            tt = 0.0 if L2 < 1e-9 else float(np.clip((p - cc[a]) @ d / L2, 0.0, 1.0))
            proj = cc[a] + tt * d; dist = float(np.hypot(*(p - proj)))
            if dist < best[3]: best = (a, b, tt, dist)
        a, b, tt, _ = best
        # SLERP 旋转(小角度) + lerp 相机中心(colmap帧)
        q0, q1 = self._q_sm[a], self._q_sm[b]
        dot = float(np.clip(q0 @ q1, -1, 1))
        if dot < 0: q1 = -q1; dot = -dot
        if dot > 0.9995:
            q = q0 + tt * (q1 - q0)
        else:
            th = np.arccos(dot); s = np.sin(th)
            q = (np.sin((1 - tt) * th) * q0 + np.sin(tt * th) * q1) / s
        q /= np.linalg.norm(q)
        Rv = _Rt.from_quat(q).as_matrix()
        Cc = (1 - tt) * self._Ccol_sm[a] + tt * self._Ccol_sm[b]
        V = np.eye(4); V[:3, :3] = Rv; V[:3, 3] = -Rv @ Cc
        return self._render_V(V)


if __name__ == "__main__":
    import sys
    sess = sys.argv[1] if len(sys.argv) > 1 else "colmapjoint_111450"
    fpv = CaptureFPV(sess)
    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{sess}.npz"))
    wp, yaw = W["waypoints_xy"], W["yaw"]
    for i in (100, 300):
        rgb = fpv.render(float(wp[i, 0]), float(wp[i, 1]), float(yaw[i]))
        f = rgb[:, :, ::-1].copy()
        for yy in range(0, f.shape[0], 40): cv2.line(f, (0, yy), (f.shape[1], yy), (0, 255, 0), 1)
        cv2.imwrite(f"/tmp/capfpv_{i}.jpg", f)
        print(f"saved /tmp/capfpv_{i}.jpg  pose=({wp[i,0]:.1f},{wp[i,1]:.1f},{yaw[i]:.2f})")
