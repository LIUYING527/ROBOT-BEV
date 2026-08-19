"""办法C: 命令门控场景 + 样本重平衡, 让 DD 学会纯原地转。
⭐诊断依据: strafe等平移命令本就可靠跟随(走目标点通路), 唯独纯旋转(x=y=0只转yaw)被场景强加的
   前向平移压死。模型改动(transfuser_model_v2.CustomTransformerDecoderLayer)已加【命令门 scene_gate】:
   g=sigmoid(scene_gate(status_encoding)+4)∈(0,1), traj = g*场景 + (1-g)*原特征。零初始化=初始g≈0.98全开
   =warm-start无损; 训练让"转向命令"学会把 g 压低 -> 关掉场景前向推力 -> 干净原地转。
重平衡: 原地转样本提到与平移~1:1(原9:1前进先验太强, 门学不动)。
warm-start: 从 goalnav_ft.pt(已验证 cruise+strafe 好)载, 只补学"转向+门"。
用法(空卡):
  CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=third_party/discoverse \
    ~/discoverse_venv/bin/python goalnav/pipeline/train_goalnav_gate.py --epochs 240
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2, torch
import torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
import stage_b_eval as sb
from train_goalnav import robot_kmeans_anchors, smooth_traj, with_yaw, GOAL_AHEAD, SX, SY
from train_goalnav_turn import status_gh, turn_gt

GOALNAV = os.path.join(ROOT, "goalnav", "runs")
T = 8


def gate_report(model, device):
    """探针: 前向命令 vs 转向命令下的门值(应: 前向≈全开, 转向被压低)。"""
    layer = model._trajectory_head.diff_decoder.layers[0]
    with torch.no_grad():
        st_fwd = status_gh(np.array([GOAL_AHEAD, 0], np.float32), np.array([1.0, 0.0], np.float32), device=device)
        st_turn = status_gh(np.zeros(2, np.float32), np.array([0.0, 1.0], np.float32), device=device)
        se_f = model._status_encoding(st_fwd)[:, None]; se_t = model._status_encoding(st_turn)[:, None]
        gf = torch.sigmoid(layer.scene_gate(se_f) + layer.scene_gate_bias).item()
        gt_ = torch.sigmoid(layer.scene_gate(se_t) + layer.scene_gate_bias).item()
    return gf, gt_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_turn", type=int, default=2000, help="合成原地转样本数(左右各半, ~1:1平衡)")
    ap.add_argument("--warm", default="goalnav_ft.pt", help="热启动ckpt(cruise+strafe已好)")
    ap.add_argument("--gate_reg", type=float, default=0.3,
                    help="门正则权重: 平移命令→门开(g→1), 纯转向命令→门关(g→0)。>0 才能稳定锁定设计意图")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ⭐固定种子: 唯一非确定源=训练循环的 np.random.permutation(batch打乱)。之前两次同初始却分岔
    #   (一次门理想分离/一次两门齐塌)证明门被 trajectory_loss 欠约束→靠种子运气, 见 gate_reg。
    np.random.seed(0); torch.manual_seed(0)

    dsdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}")
    M = np.load(os.path.join(dsdir, "meta.npz"), allow_pickle=True)
    poses = M["poses"].astype(np.float64); gdir = M["goal_dirs"].astype(np.float32)
    base = M["base_ego"].astype(np.float32); fpvs = [str(x) for x in M["fpv"]]
    xyz = sb.read_ply_xyz(str(M["merged_ply"])); _, _, _, floor = sb.build_occ(xyz)
    N = len(poses)
    print(f"[data] {N} 平移样本 featurize(+镜像) + 合成{args.n_turn}原地转(重平衡~1:1) ...", flush=True)

    cams, bevs, gts, heads = [], [], [], []
    for i in range(N):
        img = cv2.imread(os.path.join(dsdir, fpvs[i]))
        bev_i = sb.lidar_feature_fov(xyz, poses[i], floor, device)
        for flip in (False, True):
            im2 = cv2.flip(img, 1) if flip else img
            cams.append(sb.camera_feature(im2, device))
            bevs.append(torch.flip(bev_i, dims=[-1]) if flip else bev_i)
            b = base[i].copy(); gd = gdir[i].copy()
            if flip: b[:, 1] = -b[:, 1]; gd[1] = -gd[1]
            gts.append(with_yaw(smooth_traj(b)))
            heads.append((gd * GOAL_AHEAD, np.array([1.0, 0.0], np.float32)))
        if i % 300 == 0: print(f"  featurize {i}/{N}", flush=True)
    n_trans = len(cams)

    # ---- 原地转样本(重平衡到 ~n_trans): 复用真实观测(含大量走廊图=对抗配对), GT=yaw爬±90°, 目标点=0 ----
    rng = np.random.RandomState(0)
    for t in range(args.n_turn):
        src = int(rng.randint(0, n_trans))
        cams.append(cams[src]); bevs.append(bevs[src])
        left = (t % 2 == 0)
        dyaw = (np.pi / 2) * (1 if left else -1) * rng.uniform(0.6, 1.0)
        gts.append(turn_gt(dyaw))
        heads.append((np.zeros(2, np.float32), np.array([np.cos(dyaw), np.sin(dyaw)], np.float32)))
    N2 = len(cams)
    st_all = status_gh(np.stack([h[0] for h in heads]), np.stack([h[1] for h in heads]), device=device)
    gts = torch.tensor(np.stack(gts), dtype=torch.float32, device=device)
    print(f"[data] 共{N2} (平移{n_trans} + 原地转{args.n_turn}) = {n_trans/args.n_turn:.1f}:1", flush=True)
    # 门监督目标: 平移样本[0:n_trans]→1(场景保留), 纯转向样本[n_trans:]→0(关场景前向推力)
    gate_tgt = torch.zeros(N2, device=device); gate_tgt[:n_trans] = 1.0

    model = sb.load_model(device)
    model._trajectory_head.plan_anchor.data = torch.tensor(robot_kmeans_anchors(), device=device)
    # warm-start: 载 goalnav_ft(cruise+strafe已好); scene_gate是新增(strict=False保留零初始化=全开)
    wp = os.path.join(GOALNAV, args.warm)
    if os.path.exists(wp):
        ck = torch.load(wp)
        model._trajectory_head.load_state_dict(ck["trajectory_head"], strict=False)
        model._status_encoding.load_state_dict(ck["status_encoding"])
        print(f"[warm] 载 {args.warm} (cruise+strafe) + 新增门=全开", flush=True)

    for p in model.parameters(): p.requires_grad = False
    for p in model._trajectory_head.parameters(): p.requires_grad = True   # 含 scene_gate
    for p in model._status_encoding.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    B = args.batch
    gf, gt_ = gate_report(model, device)
    print(f"[gate] 初始: 前向命令门={gf:.3f} 转向命令门={gt_:.3f} (期望训练后转向门显著下降)", flush=True)
    print(f"[train] {N2}样本 batch={B} {args.epochs}ep (命令门控+重平衡) ...", flush=True)
    model.train()
    for ep in range(args.epochs):
        order = np.random.permutation(N2); tot = 0.0; nb = 0
        for s in range(0, N2, B):
            bi = order[s:s + B]
            cam = torch.cat([cams[i] for i in bi]); bev = torch.cat([bevs[i] for i in bi])
            st = st_all[bi]; gt = gts[bi]
            out = model({"camera_feature": cam, "lidar_feature": bev, "status_feature": st},
                        {"trajectory": gt})
            loss = out["trajectory_loss"]
            if args.gate_reg > 0:
                # 门正则: 对本 batch 每样本算两层门 g, BCE 拉向目标(平移1/转向0)。
                # 门被 trajectory_loss 欠约束(有goal驱动,场景冗余)→需显式监督锁定"只转向时关场景"。
                se = model._status_encoding(st)[:, None]                       # (b,1,D)
                tgt = gate_tgt[bi][:, None, None]                              # (b,1,1)
                greg = 0.0
                for lyr in model._trajectory_head.diff_decoder.layers:
                    g = torch.sigmoid(lyr.scene_gate(se) + lyr.scene_gate_bias)
                    greg = greg + F.binary_cross_entropy(g, tgt.expand_as(g))
                loss = loss + args.gate_reg * greg
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            gf, gt_ = gate_report(model, device)
            print(f"  ep{ep:03d} loss={tot / nb:.4f}  门[前向{gf:.3f} 转向{gt_:.3f}]", flush=True)
        if ep >= 40 and ep % 40 == 0:
            torch.save({"trajectory_head": model._trajectory_head.state_dict(),
                        "status_encoding": model._status_encoding.state_dict()},
                       os.path.join(GOALNAV, "goalnav_gate.pt"))
    torch.save({"trajectory_head": model._trajectory_head.state_dict(),
                "status_encoding": model._status_encoding.state_dict()},
               os.path.join(GOALNAV, "goalnav_gate.pt"))
    gf, gt_ = gate_report(model, device)
    print(f"[done] -> goalnav_gate.pt  终门[前向{gf:.3f} 转向{gt_:.3f}]", flush=True)


if __name__ == "__main__":
    main()
