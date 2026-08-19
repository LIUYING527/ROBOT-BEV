"""快脑训练集v5 (0803): 骨架化搜索行为数据 — 论文§4.1.3(Qwen-RobotNav)"最短路GT教不出搜索行为"。

相对v4的升级(v4基数据集逐样本保留, mode=0):
  ①⭐搜索样本(mode=1): occ自由空间骨架化(skimage官方skeletonize, 非手搓)→随机DFS遍历
    (交叉点优先未访问边, 死胡同沿原路回退)→沿walk采样训练窗。
    GT=未来walk路径(含回退折返=A*永远学不出的行为), goal/heading=(0,0)(语义=无目标)。
  ②⭐mode标志进meta["mode"](0=goal-seek/1=search)→训练填status[7](v4及以前恒0=旧行为兼容)。
    搜索=覆盖式观察: W~U{4..8}偏高+γ~U[0.7,0.95](论文: 搜索要覆盖式历史)。
  ③hist_poses=walk上真实过去位姿(0.4m lag, 天然时序, 比ref轨迹查表更真)。
用法: CUDA_VISIBLE_DEVICES=? MUJOCO_GL=egl PYTHONPATH=third_party/discoverse \
      ~/discoverse_venv/bin/python scripts/gen_nav_dataset_lab_v5.py --session colmapjoint_110744s2
产出: outputs/nav_ds_colmapjoint_110744s2_v5/ (不覆盖v4)
"""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "arch_e2e", "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "goalnav", "pipeline"))
from gen_nav_dataset import T
from gen_nav_dataset_lab_v3 import path_to_ego, HORIZON
from gen_nav_dataset_lab_v4 import HIST_MAX, HIST_STRIDE

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_walk(free, start_rc, rng, max_len_m=25.0, res=0.05):
    """骨架随机DFS遍历(死胡同回退): 返回格序列(含回退重复段)。
    交叉点优先未访问出边(系统性覆盖), 死胡同沿父链回退(论文核心行为=A*教不出)。"""
    from skimage.morphology import skeletonize
    skel = skeletonize(free > 0)
    Hh, Ww = skel.shape
    def nbs(c):
        return [(c[0] + dr, c[1] + dc) for dr, dc in N8
                if 0 <= c[0] + dr < Hh and 0 <= c[1] + dc < Ww and skel[c[0] + dr, c[1] + dc]]
    visited_edge = set()                                 # 无向边(a,b) frozenset
    walk = [start_rc]; cur = start_rc; parent = {start_rc: None}
    max_cells = int(max_len_m / res)
    while len(walk) < max_cells:
        unvis = [n for n in nbs(cur) if frozenset((cur, n)) not in visited_edge]
        if unvis:                                        # 优先没走过的边(覆盖式)
            nxt = unvis[rng.randint(len(unvis))]
            visited_edge.add(frozenset((cur, nxt))); parent[nxt] = cur
            cur = nxt; walk.append(cur)
        else:                                            # 死胡同/全访问过: 回退一格(沿父链)
            if parent[cur] is None:
                break
            cur = parent[cur]; walk.append(cur)
    return walk, skel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--n_search", type=int, default=260, help="搜索样本目标数")
    ap.add_argument("--n_episode", type=int, default=3, help="搜索walk条数(不同起点)")
    ap.add_argument("--stride_m", type=float, default=0.4, help="沿walk采样间距")
    args = ap.parse_args()
    import sim_fullchain as sf
    from test_env import CorridorTestEnv
    env = CorridorTestEnv(skin_session=args.session)
    res, lo = env.res, env.lo
    free = (env.occ == 0)

    # ---- v4基数据集原样载入(mode=0) ----
    v4dir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}_v4")
    M = np.load(os.path.join(v4dir, "meta.npz"), allow_pickle=True)
    outdir = os.path.join(ROOT, "outputs", f"nav_ds_{args.session}_v5"); os.makedirs(outdir, exist_ok=True)
    import shutil
    for f in M["fpv"]:                                   # FPV软复制(不重复渲染)
        src, dst = os.path.join(v4dir, str(f)), os.path.join(outdir, str(f))
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    poses = list(M["poses"]); goals = list(M["goal_dirs"]); labels = list(M["base_ego"])
    fpv_paths = [str(x) for x in M["fpv"]]; kinds = list(M["kinds"]); vxs = list(M["vx"])
    hposes = list(M["hist_poses"]); hws = list(M["hist_w"]); hgammas = list(M["hist_gamma"])
    modes = [0] * len(poses)
    n = len(poses)
    print(f"[v5] v4基数据集 {n} 样本(mode=0)已并入", flush=True)

    # ---- 骨架搜索样本(mode=1) ----
    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    ref = W["waypoints_xy"].astype(np.float64)
    rng1 = np.random.RandomState(2)                      # ⭐v5新增随机性独立流(v4用0/1, 别碰)
    merged_ply = str(M["merged_ply"])
    n_search = 0; ep = 0; tries = 0
    # 起点候选: ref轨迹点→最近骨架格
    from skimage.morphology import skeletonize
    skel = skeletonize(free)
    skel_rc = np.argwhere(skel)
    while n_search < args.n_search and ep < args.n_episode * 3:
        rp = ref[rng1.randint(len(ref))]
        d2 = ((skel_rc[:, 1] - (rp[0] - lo[0]) / res) ** 2 + (skel_rc[:, 0] - (rp[1] - lo[1]) / res) ** 2)
        start_rc = tuple(skel_rc[int(d2.argmin())])
        walk, _ = skeleton_walk(free, start_rc, rng1, res=res)
        ep += 1
        if len(walk) < 40:
            continue
        # 格序列→世界折线(格中心), 弧长重采样(骨架格子锯齿沿walk自然平滑, 采样间距0.4m>>格0.05m)
        wxy = np.array([[lo[0] + (c[1] + 0.5) * res, lo[1] + (c[0] + 0.5) * res] for c in walk])
        seg = np.linalg.norm(np.diff(wxy, axis=0), axis=1); arc = np.concatenate([[0], np.cumsum(seg)])
        ss = np.arange(0, arc[-1] - HORIZON - 0.3, args.stride_m)
        for si in ss:
            i0 = int(np.searchsorted(arc, si)); i0 = min(i0, len(wxy) - 2)
            i1 = min(i0 + int(0.6 / res), len(wxy) - 1)  # 切线看前0.6m
            x, y = wxy[i0]
            yaw = float(np.arctan2(wxy[i1, 1] - wxy[i0, 1], wxy[i1, 0] - wxy[i0, 0]))
            yaw += float(rng1.uniform(-0.1, 0.1))
            fut = wxy[i0:]                               # 未来walk(GT来源)
            lab, plen = path_to_ego(fut, x, y, yaw, False)
            if lab is None or plen < HORIZON * 0.6:
                continue
            # ⚠不强制x单调(搜索含死胡同折返=先退后进, 这正是要教的行为)
            # 历史=walk真实过去(朝向=当时切线)
            hist = np.zeros((HIST_MAX, 3), np.float32); n_avail = 0
            for lag in range(1, HIST_MAX + 1):
                s_h = si - HIST_STRIDE * lag
                if s_h <= 0:
                    break
                jh = min(int(np.searchsorted(arc, s_h)), len(wxy) - 2)
                jh1 = min(jh + int(0.6 / res), len(wxy) - 1)
                hist[lag - 1] = [wxy[jh, 0], wxy[jh, 1],
                                 float(np.arctan2(wxy[jh1, 1] - wxy[jh, 1], wxy[jh1, 0] - wxy[jh, 0]))]
                n_avail = lag
            w = int(rng1.randint(4, HIST_MAX + 1)); w = min(w, n_avail)
            if w == 0:
                continue                                 # walk开头没历史, 跳过(搜索样本必须带覆盖历史)
            gam = float(rng1.uniform(0.7, 0.95))
            vx = float(rng1.uniform(0.3, 0.6))
            env.env.set_pose(x, y, yaw); env.env.render()
            fpv_saved = f"{n:04d}.jpg"
            cv2.imwrite(os.path.join(outdir, fpv_saved), env.env.frame()[:, :, ::-1].copy())
            poses.append([x, y, yaw]); goals.append(np.zeros(2, np.float32))   # ⭐goal=(0,0)=无目标语义
            labels.append(lab); fpv_paths.append(fpv_saved); kinds.append("search")
            vxs.append(vx); hposes.append(hist); hws.append(w); hgammas.append(gam)
            modes.append(1); n += 1; n_search += 1
            if n_search >= args.n_search:
                break
        print(f"  episode{ep}: walk {len(walk)}格/{arc[-1]:.1f}m, 累计搜索样本{n_search}", flush=True)
    np.savez(os.path.join(outdir, "meta.npz"),
             poses=np.array(poses, np.float32), goal_dirs=np.array(goals, np.float32),
             base_ego=np.array(labels, np.float32), fpv=np.array(fpv_paths),
             kinds=np.array(kinds), vx=np.array(vxs, np.float32),
             hist_poses=np.array(hposes, np.float32), hist_w=np.array(hws, np.int32),
             hist_gamma=np.array(hgammas, np.float32),
             mode=np.array(modes, np.int32), merged_ply=merged_ply)   # ⭐mode: 0=goal-seek 1=search
    print(f"[done] {n} 样本(搜索{n_search}) -> {outdir}/meta.npz", flush=True)


if __name__ == "__main__":
    main()
