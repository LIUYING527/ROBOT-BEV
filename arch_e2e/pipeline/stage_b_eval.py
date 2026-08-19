"""Stage B (非反应式评测) — 加载 DiffusionDrive 快脑, 在 Stage A 的 sim 观测上逐帧预测
4s 轨迹, 在静态占据里展开算简化 PDMS(碰撞/进度), 出对比视频 + 指标。（arch_e2e 解耦版）

在【模型 env】跑(artifixer, 只用本仓 vendor/weights, 不碰 renderer, 不碰 robot_orig_code):
  CUDA_VISIBLE_DEVICES=N /home/DongBaorong/micromamba/envs/artifixer/bin/python \
    arch_e2e/pipeline/stage_b_eval.py --session colmapjoint_all

架构原则: BEV 用官方 sim_bev.official_lidar_feature(=points_to_bev_tensor); 模型 V2 直连 + nuplan-shim。
学长 ckpt 仅作烟雾测试快脑(非最终)。sim 的 3DGS FPV 与真实 ZED 有 domain gap, 轨迹未必好——
本脚本目标 = 验证整条架构数据流通 + 模型可被 sim 驱动。

⚠️注意架构归属: 这里的 4 运动模式(cruise/turn/inspect/exit) one-hot 是【学长 ckpt 自带的接口】,
不是我们的架构。我们的 VLM 慢脑做【语义场景/任务理解】, 见 pipeline/stage_vlm_slowbrain.py。
本文件仅用学长 ckpt 做快脑烟雾测试, 故沿用他的 4 模式槽; 换成我们的语义任务条件需按新通路重训。
"""
import os, sys, argparse, json
import numpy as np
import cv2

from _paths import inject_vendor, WEIGHTS, RUNS
inject_vendor()
import torch
import sim_bev   # 官方 BEV 生成件(vendored)


# ---------------- 学长 ckpt 的 4 运动模式槽(非我们架构; 仅为跑通他的 ckpt) ----------------
MODES = ["cruise", "turn", "inspect", "exit"]   # 学长上车的 4 运动基元 one-hot 顺序

def onehot_of(mode):
    oh = np.zeros(4, np.float32); oh[MODES.index(mode) if mode in MODES else 0] = 1.0
    return oh


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


# ---------------- 模型加载(V2 直连 + shim, 全部本地 weights) ----------------
def load_model(device):
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    from navsim.agents.diffusiondrive.transfuser_model_v2 import V2TransfuserModel
    cfg = TransfuserConfig()
    cfg.use_auxiliary_supervision = False
    cfg.plan_anchor_path = os.path.join(WEIGHTS, "kmeans_navsim_traj_20.npy")
    cfg.bkb_path = os.path.join(WEIGHTS, "resnet34.a1_in1k", "pytorch_model.bin")
    model = V2TransfuserModel(cfg)
    ck = torch.load(os.path.join(WEIGHTS, "best-val-loss.ckpt"), map_location="cpu", weights_only=False)
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


def fov_select_ego(xyz, pose, floor, fov_deg=110.0, R=32.0):
    """位姿pose的前向FOV锥选点。返回(ego系点[M,3] z已减floor, 选中点的世界系xyz[M,3])。
    ego系: x前 y左。世界系点供历史累积做跨位姿刚体变换(只施一次当前位姿变换, 防链式符号错)。"""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    dx = xyz[:, 0] - x; dy = xyz[:, 1] - y
    ex = c * dx + s * dy; ey = -s * dx + c * dy          # ego: x前 y左
    ez = xyz[:, 2] - floor                                # 高度(地面上方)
    ang = np.degrees(np.arctan2(ey, ex))
    m = (ex > 0.2) & (np.abs(ang) < fov_deg / 2) & (np.hypot(ex, ey) < R)  # 前向FOV锥内
    return np.stack([ex[m], ey[m], ez[m]], 1), xyz[m]


def lidar_feature_fov(xyz, pose, floor, device, fov_deg=110.0, R=32.0):
    """从合并几何按位姿裁前向 FOV 扇形 -> 官方直方图 [1,1,256,256]。"""
    ego, _ = fov_select_ego(xyz, pose, floor, fov_deg, R)
    bev = sim_bev.official_lidar_feature(ego, lidar_range=R)              # [1,256,256]
    return torch.from_numpy(bev).unsqueeze(0).to(device)                  # [1,1,256,256]


def hist_merge_ego(cur_ego, hist_world_list, pose, floor, gamma, seed=0):
    """⭐观察策略参数化(0802): 当前帧ego点 + 历史位姿FOV点(世界系)衰减融合 -> 合并ego点。
    衰减=遗忘: lag h(1=最近)的历史点用种子化RandomState确定性保留 gamma^h 比例。
    合并后在当前ego系重施 hypot>0.2 近距掩码(历史点可能落到机器人脚下→ego原点假亮斑),
    但【不】施FOV锥(身后点正是历史的意义)。seed由样本索引派生=可复现。"""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    parts = [cur_ego]
    for lag, wsel in enumerate(hist_world_list, start=1):
        if wsel is None or len(wsel) == 0:
            continue
        keep = float(gamma) ** lag
        rng = np.random.RandomState((int(seed) * 131 + lag) & 0x7FFFFFFF)
        wsel = wsel[rng.rand(len(wsel)) < keep]
        if len(wsel) == 0:
            continue
        dx = wsel[:, 0] - x; dy = wsel[:, 1] - y
        ex = c * dx + s * dy; ey = -s * dx + c * dy
        parts.append(np.stack([ex, ey, wsel[:, 2] - floor], 1))
    if len(parts) == 1:
        return cur_ego
    ego = np.concatenate(parts)
    return ego[np.hypot(ego[:, 0], ego[:, 1]) > 0.2]


def lidar_feature_fov_hist(xyz, pose, floor, device, hist_poses=None, W=0, gamma=1.0,
                           fov_deg=110.0, R=32.0, seed=0):
    """参数化时序历史BEV: 当前帧FOV(权重1) + 最近W个历史位姿FOV点按gamma^lag衰减融合。
    hist_poses: (H,3)世界系历史位姿, 由近到远(lag=1..H)。W=0 → 与lidar_feature_fov逐位一致。"""
    ego, _ = fov_select_ego(xyz, pose, floor, fov_deg, R)
    if W > 0 and hist_poses is not None and len(hist_poses) > 0:
        hw = [fov_select_ego(xyz, hist_poses[k], floor, fov_deg, R)[1]
              for k in range(min(W, len(hist_poses)))]
        ego = hist_merge_ego(ego, hw, pose, floor, gamma, seed)
    bev = sim_bev.official_lidar_feature(ego, lidar_range=R)
    return torch.from_numpy(bev).unsqueeze(0).to(device)


def status_feature(mode_onehot, device, vx=0.4):
    s = np.concatenate([mode_onehot, [vx, 0.0, 0.0, 0.0]]).astype(np.float32)  # [mode4 | vx vy ... ]
    return torch.from_numpy(s).unsqueeze(0).to(device)


# ---------------- 轨迹展开(简化 PDMS) ----------------
def unroll(traj, pose, occ, lo, res):
    """traj (8,3) ego 累积位姿; 转世界逐点查占据。返回(碰撞, 进度m, 世界点)。"""
    x, y, th = pose; c, s = np.cos(th), np.sin(th)
    pts = []; collided = False
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
    ap.add_argument("--seqdir", default=None, help="Stage A 观测目录(默认 runs/obs_seq_<session>)")
    ap.add_argument("--mode", default="cruise", choices=MODES, help="不接VLM时强制的4模式(烟雾测试)")
    ap.add_argument("--use_vlm", action="store_true",
                    help="⭐接 VLM 慢脑: 每帧 Qwen3.5-4B 出语义任务 -> 桥接4模式 -> 驱动快脑(完整链路)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seqdir = args.seqdir or os.path.join(RUNS, f"obs_seq_{args.session}")
    P = np.load(os.path.join(seqdir, "poses.npz"), allow_pickle=True)
    poses = P["poses"]; merged_ply = str(P["merged_ply"])
    print(f"[stageB] 观测帧 {len(poses)}  seqdir {seqdir}  ply {merged_ply}")

    print("[stageB] 读 ply + 建占据...")
    xyz = read_ply_xyz(merged_ply)
    occ, lo, res, floor = build_occ(xyz)
    print(f"  ply {len(xyz)}点 占据 {int(occ.sum())}格 floor={floor:.2f}")

    print("[stageB] 加载快脑 DiffusionDrive(本地 weights)...")
    model = load_model(device)

    vlm = None
    if args.use_vlm:                                     # ⭐完整链路: 单进程再加载慢脑
        import task_vocab, stage_vlm_slowbrain as sb
        print("[stageB] 加载慢脑 VLM ...")
        vproc, vmodel = sb.load_vlm(sb.find_model_dir(), device)
        vlm = (sb, vproc, vmodel, task_vocab)

    os.makedirs(RUNS, exist_ok=True)
    import imageio
    tag = "fullchain" if args.use_vlm else "smoke"
    mp4 = os.path.join(RUNS, f"eval_{tag}_{args.session}.mp4")
    writer = imageio.get_writer(mp4, fps=6)
    metrics = []
    for i, pose in enumerate(poses):
        fpv = cv2.imread(os.path.join(seqdir, f"{i:03d}.jpg"))
        if vlm is not None:                              # 慢脑 -> 语义任务 -> 桥接 4 模式
            sb, vproc, vmodel, task_vocab = vlm
            raw, parsed = sb.infer(vproc, vmodel, os.path.join(seqdir, f"{i:03d}.jpg"), device)
            vtask = (parsed or {}).get("task", "") or raw
            sem_task, mode_name = task_vocab.vlm_text_to_ckpt_mode(vtask)
        else:                                            # 不接 VLM: 强制模式(烟雾测试)
            sem_task, mode_name = None, args.mode
        feats = {
            "camera_feature": camera_feature(fpv, device),
            "lidar_feature": lidar_feature_fov(xyz, pose, floor, device),
            "status_feature": status_feature(onehot_of(mode_name), device),
        }
        with torch.no_grad():
            out = model(feats)
        traj = out["trajectory"][0].cpu().numpy()                        # (8,3)
        mpred = int(out["mode_probs"][0].argmax()) if "mode_probs" in out else 0
        collided, prog, wpts = unroll(traj, pose, occ, lo, res)
        metrics.append({"i": i, "vlm_sem_task": sem_task, "bridged_mode": mode_name,
                        "collided": collided, "progress": prog, "model_mode": mpred})

        # 可视化: FPV(左) + BEV占据(叠预测轨迹, 右)
        bevimg = (feats["lidar_feature"][0, 0].cpu().numpy() * 255).astype(np.uint8)
        bevimg = cv2.cvtColor(cv2.resize(bevimg, (360, 360)), cv2.COLOR_GRAY2BGR)
        for (fx, fy, _) in traj:                                          # ego系 ±32m -> 360px
            px = int(180 - fy / 32.0 * 180); py = int(180 - fx / 32.0 * 180)
            if 0 <= px < 360 and 0 <= py < 360:
                cv2.circle(bevimg, (px, py), 3, (0, 200, 255), -1)
        cv2.circle(bevimg, (180, 180), 4, (0, 0, 255), -1)
        cv2.putText(bevimg, f"BEV mode={mode_name} coll={collided}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        fpv2 = cv2.resize(fpv, (640, 360))
        label = f"VLM: {sem_task} -> {mode_name}" if sem_task else "FPV 3DGS -> camera_feature"
        cv2.putText(fpv2, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        writer.append_data(np.hstack([fpv2, bevimg])[:, :, ::-1])
        if i % 5 == 0:
            print(f"  frame {i}: sem={sem_task} mode={mode_name} prog={prog:.2f}m coll={collided}", flush=True)
    writer.close()

    ncoll = sum(m["collided"] for m in metrics)
    summary = {"frames": len(metrics), "use_vlm": args.use_vlm,
               "collision_free_rate": 1 - ncoll / len(metrics),
               "mean_progress_m": float(np.mean([m["progress"] for m in metrics])),
               "per_frame": metrics}
    js = os.path.join(RUNS, f"eval_{tag}_{args.session}.json")
    json.dump(summary, open(js, "w"), ensure_ascii=False, indent=1)
    print(f"[stageB] 完成({tag}). 无碰撞率={summary['collision_free_rate']:.2f} "
          f"平均进度={summary['mean_progress_m']:.2f}m\n  -> {mp4}\n  -> {js}")


if __name__ == "__main__":
    main()
