"""Stage B(非反应式评测) — 加载(原始架构)DiffusionDrive快脑,在Stage A的sim观测上
逐帧预测4s轨迹,在静态占据里展开算简化PDMS(无碰撞/进度/最近净空),出对比视频+指标。

在【模型 env】跑(artifixer + _nuplan_shim,不含renderer):
  CUDA_VISIBLE_DEVICES=N /home/DongBaorong/micromamba/envs/artifixer/bin/python \
    scripts/eval_diffusiondrive_nonreactive.py --session colmapjoint_all

架构原则: BEV用官方 sim_bev.official_lidar_feature(=points_to_bev_tensor); 模型用V2直连+shim。
学长ckpt仅作烟雾测试快脑(非最终)。sim的3DGS FPV与真实ZED有domain gap,轨迹未必好——
本脚本目标=验证整条架构数据流通+模型可被sim驱动。
"""
import os, sys, argparse, json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC = "/data/DongBaorong/BEVLOOK/robot_orig_code"
for p in [os.path.join(ROOT, "_nuplan_shim"), f"{RC}/DiffusionDrive-main", RC, ROOT]:
    sys.path.insert(0, p)
import torch
import sim_bev   # 官方 BEV 生成件


# ---------------- 极简二进制 ply xyz 读取(无第三方依赖) ----------------
def read_ply_xyz(path, max_pts=1_500_000):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.readline()
        lines = head.decode("ascii", "replace").splitlines()
        n = next(int(l.split()[-1]) for l in lines if l.startswith("element vertex"))
        props = [l.split()[-1] for l in lines if l.startswith("property")]
        assert props[:3] == ["x", "y", "z"], props[:3]
        stride = len(props)  # 全 float32
        buf = np.frombuffer(f.read(n * stride * 4), dtype="<f4").reshape(n, stride)
    xyz = buf[:, :3].astype(np.float64)
    if len(xyz) > max_pts:
        xyz = xyz[np.random.RandomState(0).choice(len(xyz), max_pts, replace=False)]
    return xyz


# ---------------- 静态占据(展开碰撞用) ----------------
def build_occ(xyz, res=0.08, robot_r=0.3):
    from scipy import ndimage
    z = xyz[:, 2]; floor = np.percentile(z, 2)
    m = (z - floor > 0.3) & (z - floor < 2.0)
    p = xyz[m][:, :2]
    lo = p.min(0) - 1.5; hi = p.max(0) + 1.5
    nx = int((hi[0] - lo[0]) / res) + 1; ny = int((hi[1] - lo[1]) / res) + 1
    grid = np.zeros((ny, nx), np.int32)
    ix = ((p[:, 0] - lo[0]) / res).astype(int); iy = ((p[:, 1] - lo[1]) / res).astype(int)
    np.add.at(grid, (iy, ix), 1)
    occ = grid >= 3
    occ = ndimage.binary_dilation(occ, iterations=max(1, int(robot_r / res)))
    return occ, lo, res, floor


def occupied(occ, lo, res, x, y):
    ix = int((x - lo[0]) / res); iy = int((y - lo[1]) / res)
    if ix < 0 or iy < 0 or iy >= occ.shape[0] or ix >= occ.shape[1]:
        return True
    return bool(occ[iy, ix])


# ---------------- 模型加载(V2 直连 + shim) ----------------
def load_model(device):
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
    cfg = TransfuserConfig()
    cfg.use_auxiliary_supervision = False
    cfg.plan_anchor_path = f"{RC}/kmeans_navsim_traj_20.npy"
    cfg.bkb_path = f"{RC}/DiffusionDrive-main/weights/resnet34.a1_in1k/pytorch_model.bin"
    model = V2TransfuserModel(cfg)
    ck = torch.load(f"{RC}/best-val-loss.ckpt", map_location="cpu", weights_only=False)
    new = {k.replace("agent._transfuser_model.", ""): v
           for k, v in ck["state_dict"].items() if k.startswith("agent._transfuser_model.")}
    model.load_state_dict(new, strict=False)
    return model.to(device).eval()


# ---------------- 特征构造 ----------------
def camera_feature(fpv_bgr, device):
    rgb = cv2.cvtColor(fpv_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1024, 256), interpolation=cv2.INTER_LINEAR)  # 同上车
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return t.unsqueeze(0).to(device)


def lidar_feature_fov(xyz, pose, floor, device, fov_deg=110.0, R=32.0):
    """从合并几何按位姿裁前向FOV扇形 -> 官方直方图 [1,1,256,256]。"""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    dx = xyz[:, 0] - x; dy = xyz[:, 1] - y
    ex = c * dx + s * dy; ey = -s * dx + c * dy          # ego: x前 y左
    ez = xyz[:, 2] - floor                                # 高度(地面上方)
    ang = np.degrees(np.arctan2(ey, ex))
    m = (ex > 0.2) & (np.abs(ang) < fov_deg / 2) & (np.hypot(ex, ey) < R)  # 前向FOV锥内
    ego = np.stack([ex[m], ey[m], ez[m]], 1)
    bev = sim_bev.official_lidar_feature(ego, lidar_range=R)              # [1,256,256]
    return torch.from_numpy(bev).unsqueeze(0).to(device)                  # [1,1,256,256]


def status_feature(device, vx=0.4):
    s = np.array([1, 0, 0, 0, vx, 0.0, 0.0, 0.0], np.float32)             # cruise + 名义前速
    return torch.from_numpy(s).unsqueeze(0).to(device)


# ---------------- 轨迹展开(简化PDMS) ----------------
def unroll(traj, pose, occ, lo, res):
    """traj (8,3) ego累积位姿; 转世界逐点查占据。返回(碰撞,进度m,最近净空m,世界点)。"""
    x, y, th = pose; c, s = np.cos(th), np.sin(th)
    pts = []; collided = False; clear = 1e9
    for (fx, fy, _) in traj:
        wx = x + c * fx - s * fy; wy = y + s * fx + c * fy
        pts.append((wx, wy))
        if occupied(occ, lo, res, wx, wy):
            collided = True
    prog = float(np.hypot(traj[-1, 0], traj[-1, 1]))                      # 末点离原点(前进量)
    return collided, prog, np.array(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seqdir = os.path.join(ROOT, "outputs", f"obs_seq_{args.session}")
    P = np.load(os.path.join(seqdir, "poses.npz"), allow_pickle=True)
    poses = P["poses"]; merged_ply = str(P["merged_ply"])
    print(f"[stageB] 观测帧 {len(poses)}  ply {merged_ply}")

    print("[stageB] 读 ply + 建占据...")
    xyz = read_ply_xyz(merged_ply)
    occ, lo, res, floor = build_occ(xyz)
    print(f"  ply {len(xyz)}点 占据 {int(occ.sum())}格 floor={floor:.2f}")

    print("[stageB] 加载模型...")
    model = load_model(device)

    import imageio
    writer = imageio.get_writer(os.path.join(ROOT, "outputs", f"eval_nonreactive_{args.session}.mp4"), fps=6)
    metrics = []
    for i, pose in enumerate(poses):
        fpv = cv2.imread(os.path.join(seqdir, f"{i:03d}.jpg"))
        feats = {
            "camera_feature": camera_feature(fpv, device),
            "lidar_feature": lidar_feature_fov(xyz, pose, floor, device),
            "status_feature": status_feature(device),
        }
        with torch.no_grad():
            out = model(feats)
        traj = out["trajectory"][0].cpu().numpy()                        # (8,3)
        mode = int(out["mode_probs"][0].argmax()) if "mode_probs" in out else 0
        collided, prog, wpts = unroll(traj, pose, occ, lo, res)
        metrics.append({"i": i, "collided": collided, "progress": prog, "mode": mode})

        # 可视化: FPV(左) + BEV占据(叠预测轨迹,右)
        bevimg = (feats["lidar_feature"][0, 0].cpu().numpy() * 255).astype(np.uint8)
        bevimg = cv2.cvtColor(cv2.resize(bevimg, (360, 360)), cv2.COLOR_GRAY2BGR)
        # 预测轨迹画到 bev(ego系, ±32m -> 360px; x前=上, y左=左)
        for (fx, fy, _) in traj:
            px = int(180 - fy / 32.0 * 180); py = int(180 - fx / 32.0 * 180)
            if 0 <= px < 360 and 0 <= py < 360:
                cv2.circle(bevimg, (px, py), 3, (0, 200, 255), -1)
        cv2.circle(bevimg, (180, 180), 4, (0, 0, 255), -1)
        cv2.putText(bevimg, f"BEV(official) traj mode={mode} coll={collided}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        fpv2 = cv2.resize(fpv, (640, 360))
        cv2.putText(fpv2, "FPV 3DGS -> camera_feature", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        writer.append_data(np.hstack([fpv2, bevimg])[:, :, ::-1])
        if i % 10 == 0:
            print(f"  frame {i}: mode={mode} progress={prog:.2f}m coll={collided}", flush=True)
    writer.close()

    ncoll = sum(m["collided"] for m in metrics)
    summary = {"frames": len(metrics), "collision_free_rate": 1 - ncoll / len(metrics),
               "mean_progress_m": float(np.mean([m["progress"] for m in metrics])),
               "per_frame": metrics}
    json.dump(summary, open(os.path.join(ROOT, "outputs", f"eval_nonreactive_{args.session}.json"), "w"), indent=1)
    print(f"[stageB] 完成. 无碰撞率={summary['collision_free_rate']:.2f} "
          f"平均进度={summary['mean_progress_m']:.2f}m -> outputs/eval_nonreactive_{args.session}.mp4")


if __name__ == "__main__":
    main()
