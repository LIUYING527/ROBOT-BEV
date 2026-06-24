"""闭环测试场环境: 训练好的模型(VLM+DiffusionDrive)插进来跑测试。
obs = {fpv: 3DGS前视RGB(VLM用), bev: 局部占据窗(DiffusionDrive用), pose: (x,y,yaw)}
action = (v, omega);  step里运动学积分 + 占据碰撞判定 + 到达判定。
视觉皮肤=单段锐3DGS(114830c);导航底座=合并占据图(corridor_merged,两段全覆盖,视角无关)。

用法(自带脚本化策略跑闭环demo,出视频证明可用):
  CUDA_VISIBLE_DEVICES=N MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=N PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python scripts/test_env.py [--skin_session 114830c] [--steps 200]
"""
import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def build_occupancy(merged_ply, res, robot_r, traj=None):
    import trimesh, cv2                                  # 用trimesh读(无GL,避开与MuJoCo EGL冲突)
    P = np.asarray(trimesh.load(merged_ply, process=False).vertices)
    mid = P[(P[:, 2] > 0.3) & (P[:, 2] < 2.0)]
    pts2 = mid[:, :2] if traj is None else np.vstack([mid[:, :2], traj])   # 边界含轨迹
    lo = pts2.min(0) - 1.5
    hi = pts2.max(0) + 1.5
    W = int(np.ceil((hi[0] - lo[0]) / res)); H = int(np.ceil((hi[1] - lo[1]) / res))
    occ = np.zeros((H, W), np.uint8)
    ix = np.clip(((mid[:, 0] - lo[0]) / res).astype(int), 0, W - 1)
    iy = np.clip(((mid[:, 1] - lo[1]) / res).astype(int), 0, H - 1)
    cnt = np.zeros((H, W), np.int32); np.add.at(cnt, (iy, ix), 1)
    occ[cnt >= 3] = 1                                   # >=3点的格=占据(滤孤立噪点)
    occ = cv2.dilate(occ, np.ones((int(robot_r / res) * 2 + 1,) * 2, np.uint8))  # 膨胀机器人半径
    return occ, lo, res


class CorridorTestEnv:
    def __init__(self, skin_session="114830c", merged="outputs/corridor_merged.ply",
                 res=0.05, robot_r=0.3, W=640, H=400):
        from sim_walk_common import make_env
        self.env, self.Wd = make_env(skin_session, width=W, height=H, skin_only=True)
        wp = self.Wd["waypoints_xy"]
        self.occ, self.lo, self.res = build_occupancy(os.path.join(ROOT, merged), res, robot_r, traj=wp)
        self._clear_path(wp, robot_r + 0.45)            # 机器人走过的轨迹沿线清free(去人/杂物对路径污染)
        kf = min(6, len(wp) - 1)                         # 朝前方第6个路点定开局朝向(避首点梯度噪声)
        self.start = (float(wp[0, 0]), float(wp[0, 1]),
                      float(np.arctan2(wp[kf, 1] - wp[0, 1], wp[kf, 0] - wp[0, 0])))
        self.goal = (float(wp[-1, 0]), float(wp[-1, 1]))
        self.win_m = 8.0                                # BEV局部窗 8m
        self.reset()

    def _clear_path(self, wp, r):
        import cv2
        rad = int(r / self.res)
        for a, b in zip(wp[:-1], wp[1:]):
            ga = (int((a[0] - self.lo[0]) / self.res), int((a[1] - self.lo[1]) / self.res))
            gb = (int((b[0] - self.lo[0]) / self.res), int((b[1] - self.lo[1]) / self.res))
            cv2.line(self.occ, ga, gb, 0, thickness=2 * rad)   # 沿轨迹清成free

    def _occupied(self, x, y):
        ix = int((x - self.lo[0]) / self.res); iy = int((y - self.lo[1]) / self.res)
        if iy < 0 or iy >= self.occ.shape[0] or ix < 0 or ix >= self.occ.shape[1]:
            return True
        return self.occ[iy, ix] > 0

    def reset(self):
        self.x, self.y, self.yaw = self.start
        self.t = 0; self.collided = False; self.reached = False
        return self._obs()

    def step(self, v, w, dt=0.25):
        nx = self.x + v * np.cos(self.yaw) * dt
        ny = self.y + v * np.sin(self.yaw) * dt
        if self._occupied(nx, ny):
            self.collided = True                        # 撞了:不前进
        else:
            self.x, self.y = nx, ny
        self.yaw += w * dt; self.t += 1
        if np.hypot(self.x - self.goal[0], self.y - self.goal[1]) < 1.2:
            self.reached = True
        term = self.collided or self.reached
        return self._obs(), term, {"collided": self.collided, "reached": self.reached, "t": self.t}

    def _obs(self):
        self.env.set_pose(self.x, self.y, self.yaw); self.env.render()
        fpv = self.env.frame().copy()                   # (H,W,3) 3DGS前视
        bev = self._bev_window()                        # 局部占据窗(机器人朝上)
        return {"fpv": fpv, "bev": bev, "pose": (self.x, self.y, self.yaw)}

    def _bev_window(self, n=64):
        # 机器人为中心、朝向朝上的局部占据窗
        half = self.win_m / 2; out = np.zeros((n, n), np.uint8)
        ca, sa = np.cos(-self.yaw + np.pi / 2), np.sin(-self.yaw + np.pi / 2)
        for j in range(n):
            for i in range(n):
                lx = (i - n / 2) / n * self.win_m; ly = (j - n / 2) / n * self.win_m
                wx = self.x + (lx * ca - ly * sa); wy = self.y + (lx * sa + ly * ca)
                out[n - 1 - j, i] = 1 if self._occupied(wx, wy) else 0
        return out


def scripted_policy(bev):
    """脚本化反应式策略: 看前方局部占据窗,前方堵就转向更空的一侧,否则直行。"""
    n = bev.shape[0]; cx = n // 2
    front = bev[: n // 2, cx - 6:cx + 6]                # 前方区域
    if front.mean() < 0.08:
        return 0.7, 0.0                                  # 前方空 -> 直行
    left = bev[: n // 2, cx - 18:cx - 6].mean()
    right = bev[: n // 2, cx + 6:cx + 18].mean()
    return 0.3, (1.2 if left < right else -1.2)          # 转向更空侧


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--skin_session", default="114830c")
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    env = CorridorTestEnv(skin_session=args.skin_session)
    print(f"[testenv] 占据图 {env.occ.shape} 自由{100*(env.occ==0).mean():.0f}% start={env.start[:2]} goal={env.goal}")
    import imageio.v2 as imageio
    out = os.path.join(ROOT, "outputs", "test_env_demo.mp4")
    writer = imageio.get_writer(out, fps=12, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    obs = env.reset()
    for k in range(args.steps):
        v, w = scripted_policy(obs["bev"])
        obs, term, info = env.step(v, w)
        # 可视化: 左FPV | 右BEV局部窗
        fpv = np.ascontiguousarray(obs["fpv"][:, :, ::-1])
        bev = cv2.resize(((1 - obs["bev"]) * 255).astype(np.uint8), (400, 400), interpolation=cv2.INTER_NEAREST)
        bev = cv2.cvtColor(bev.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.circle(bev, (200, 200), 6, (0, 0, 255), -1)                  # 机器人
        cv2.arrowedLine(bev, (200, 200), (200, 160), (0, 0, 255), 2)     # 朝向(上)
        cv2.putText(bev, "BEV occ (DiffusionDrive in)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(fpv, "FPV 3DGS (VLM in)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        h = fpv.shape[0]; bev = cv2.resize(bev, (h, h))
        writer.append_data(np.hstack([fpv, bev])[:, :, ::-1])
        if k % 30 == 0:
            print(f"  step{k} pose=({env.x:.1f},{env.y:.1f}) v={v:.1f} w={w:.1f}", flush=True)
        if term:
            print(f"[testenv] 终止 step{k}: {info}"); break
    writer.close()
    print(f"[testenv] ✅ 闭环demo -> {out}")


if __name__ == "__main__":
    main()
