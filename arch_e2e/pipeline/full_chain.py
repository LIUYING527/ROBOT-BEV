"""⭐真·全链路(快慢脑都用上, 一整个系统, 从输入到输出): 每帧真实FPV →
   VLM慢脑(Qwen3.5-4B, 读图决策巡检动作) → cond_id →
   DiffusionDrive快脑(条件微调版, 出轨迹) → 投影叠回真实FPV。

⭐一个进程、两个脑【同时加载】、同一循环里跑(这才是双脑架构本体, 不拆开)。
相机投影用【解析重建】(相机在机器人上方1.2m朝前, 同 sim_walk_common), 不依赖 discoverse。

用法(artifixer):
  CUDA_VISIBLE_DEVICES=N /home/DongBaorong/micromamba/envs/artifixer/bin/python \
    arch_e2e/pipeline/full_chain.py --session colmapjoint_all
产物: runs/full_chain_<session>.png(代表帧montage) + runs/full_chain_<session>.mp4(全程) + .json(每帧决策)
"""
import os, sys, json, argparse
import numpy as np, cv2

from _paths import inject_vendor, RUNS, REPO
inject_vendor()
sys.path.insert(0, os.path.join(REPO, "scripts"))
import torch
import stage_b_eval as ddb                                # 快脑 + 特征
import stage_vlm_slowbrain as vlm                         # 慢脑
from gen_conditional_data import project_ground, ego_to_world, forward_leg_poses
from train_conditional import robot_anchors

# --- 我们的 4 语义条件(与训练一致) ---
CONDITIONS = ["inspect_left", "cruise", "inspect_right", "pause"]
COND_COLOR = {"cruise": (0, 200, 255), "inspect_left": (0, 255, 0),
              "inspect_right": (255, 128, 0), "pause": (150, 150, 150)}

# --- 让 VLM 慢脑【直接决策巡检动作】(覆盖它默认的 scene/task prompt) ---
vlm.PROMPT = (
    "你是巡检机器人的慢脑。看这张前视相机画面, 决定机器人此刻该执行的巡检动作。\n"
    '只输出一个JSON: {"action": "四选一", "reason": "一句话依据"}\n'
    "action 从这四个里选一个(照抄): 直行巡航 / 靠近左侧设备 / 靠近右侧设备 / 暂停\n"
    "规则: 画面明显有阀门/配电柜/仪表/管道等设备偏左→靠近左侧设备; 偏右→靠近右侧设备; "
    "只是普通走廊无明显目标→直行巡航; 画面全黑/严重模糊/故障→暂停。只输出JSON。"
)


def action_to_cond(text):
    """VLM 动作文本 -> cond_id(index in CONDITIONS)。"""
    t = text or ""
    if any(k in t for k in ["暂停", "停", "全黑", "故障", "模糊", "无信号"]):
        return 3  # pause
    if "左" in t:
        return 0  # inspect_left
    if "右" in t:
        return 2  # inspect_right
    return 1      # cruise


def cam_pose_from_robot(x, y, yaw, cam_h=1.2):
    """解析重建 FPV 相机世界位姿(同 sim_walk_common: 相机在body上方cam_h, 朝前)。
    相机轴(body系): right=-y, up=+z, back(+z_cam)=-x。"""
    R_cb = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)   # 列=cam right/up/back 在body系
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    cam_mat = Rz @ R_cb
    cam_pos = np.array([x, y, cam_h])
    return cam_pos, cam_mat


def draw_traj_on_fpv(img, pose, traj, color, fovy=75.0, label=None):
    H, W = img.shape[:2]
    cpos, cmat = cam_pose_from_robot(*pose)
    wxy = ego_to_world(traj, pose)
    uv = project_ground(cpos, cmat, fovy, W, H, wxy)
    uv0 = project_ground(cpos, cmat, fovy, W, H, [ego_to_world(np.zeros((1, 3)), pose)[0]])[0]
    chain = [uv0] + uv
    for a, b in zip(chain[:-1], chain[1:]):
        if a and b:
            cv2.line(img, a, b, color, 4, cv2.LINE_AA)
    if uv[-1]:
        cv2.circle(img, uv[-1], 6, color, -1)
    if label and uv[-1]:
        cv2.putText(img, label, (uv[-1][0] + 6, uv[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--montage_idx", default="4,8,17,18,20,25")
    ap.add_argument("--vlm_device", default="cuda")
    ap.add_argument("--dd_device", default="cpu",
                    help="快脑设备; 默认cpu(与9G VLM共卡易OOM, 快脑小放CPU即可, 仍同进程一体)")
    args = ap.parse_args()
    vdev, ddev = args.vlm_device, args.dd_device
    seqdir = os.path.join(REPO, "outputs", f"obs_seq_{args.session}")
    P = np.load(os.path.join(seqdir, "poses.npz"), allow_pickle=True)
    poses_all = P["poses"].astype(np.float64); ply = str(P["merged_ply"])
    fwd = forward_leg_poses(poses_all); N = len(fwd)
    frames = [os.path.join(seqdir, f"{i:03d}.jpg") for i in range(N)]

    # ---------- 两个脑【同时加载】(一整个系统) ----------
    print("[全链路] 加载 VLM 慢脑 + DiffusionDrive 快脑(同进程共存) ...")
    vproc, vmodel = vlm.load_vlm(vlm.find_model_dir(), device)
    xyz = ddb.read_ply_xyz(ply); _, _, _, floor = ddb.build_occ(xyz)
    model = ddb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_anchors(), device=device)
    model._trajectory_head.load_state_dict(torch.load(os.path.join(RUNS, "cond_finetuned.pt"))["trajectory_head"])
    model.eval()
    st = torch.zeros((1, 8), device=device); st[0, 4] = 0.4

    montage_idx = [int(x) for x in args.montage_idx.split(",")]
    import imageio
    writer = imageio.get_writer(os.path.join(RUNS, f"full_chain_{args.session}.mp4"), fps=4)
    decisions, montage = [], []
    # ---------- 同一循环: 每帧 FPV -> 慢脑 -> cond -> 快脑 -> 轨迹 -> 叠图 ----------
    for i in range(N):
        pose = fwd[i]
        # 慢脑: 读图出动作
        raw, parsed = vlm.infer(vproc, vmodel, frames[i], device)
        action = (parsed or {}).get("action", "") or raw
        cid = action_to_cond(action)
        # 快脑: 按 cond 出轨迹
        cam = ddb.camera_feature(cv2.imread(frames[i]), device)
        bev = ddb.lidar_feature_fov(xyz, pose, floor, device)
        feats = {"camera_feature": cam, "lidar_feature": bev, "status_feature": st,
                 "cond_id": torch.tensor([cid], device=device)}
        with torch.no_grad():
            traj = model(feats)["trajectory"][0].cpu().numpy()
        decisions.append({"i": i, "action": action, "cond": CONDITIONS[cid], "cond_id": cid,
                          "reason": (parsed or {}).get("reason", "")})
        # 叠图: 真实FPV + 快脑轨迹 + 顶部"输入->慢脑->快脑"横幅
        img = cv2.imread(frames[i])
        draw_traj_on_fpv(img, pose, traj, COND_COLOR[CONDITIONS[cid]])
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, f"VLM: {action}  ->  cond={CONDITIONS[cid]}  -> DiffusionDrive traj",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COND_COLOR[CONDITIONS[cid]], 2)
        writer.append_data(img[:, :, ::-1])
        if i in montage_idx:
            montage.append(img)
        if i % 5 == 0:
            print(f"  frame{i}: 慢脑'{action}' -> {CONDITIONS[cid]} -> 快脑轨迹", flush=True)
    writer.close()
    if montage:
        cv2.imwrite(os.path.join(RUNS, f"full_chain_{args.session}.png"), np.vstack(montage))
    json.dump(decisions, open(os.path.join(RUNS, f"full_chain_{args.session}.json"), "w"),
              ensure_ascii=False, indent=1)
    # 决策统计
    from collections import Counter
    cnt = Counter(d["cond"] for d in decisions)
    print(f"[done] 全链路 {N} 帧。慢脑决策分布: {dict(cnt)}")
    print(f"  montage -> runs/full_chain_{args.session}.png")
    print(f"  video   -> runs/full_chain_{args.session}.mp4")


if __name__ == "__main__":
    main()
