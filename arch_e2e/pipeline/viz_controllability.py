"""可控性铁证图(publication 级) —— 纯离线, 读 model_trajs_<session>.npz + obs_seq FPV, 不占卡。

一句话: 同一观测(FPV) + 只换语义 cond -> 快脑输出的轨迹按语义分化。
    左  = 该观测的真实前视 FPV(接地气, 证明是真走廊)
    中  = 位置行为(cruise/inspect_L/R/strafe_L/R): ego 系轨迹在走廊里明显分叉
    右  = 转视角/停(turn_L/R 朝向扇形 + pause/stop 近停): 位置几乎不动, 靠朝向/停走区分

用法:
    python3 arch_e2e/pipeline/viz_controllability.py --session colmapjoint_all --pose 20
产物: arch_e2e/runs/controllability_hero_<session>.png
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "arch_e2e", "runs")

# 语义分组 + 配色(matplotlib RGB 0-1) + 中文/英文标签
GROUPS = {
    "move":   ["cruise", "inspect_left", "inspect_right", "strafe_left", "strafe_right"],
    "rotate": ["turn_left", "turn_right", "pause", "stop"],
}
COLOR = {
    "cruise": "#f5a623", "inspect_left": "#2ecc40", "inspect_right": "#0074d9",
    "strafe_left": "#39cccc", "strafe_right": "#b10dc9",
    "turn_left": "#ff851b", "turn_right": "#85144b", "pause": "#aaaaaa", "stop": "#ff4136",
}
LABEL = {
    "cruise": "cruise (go straight)", "inspect_left": "inspect_left (veer L)",
    "inspect_right": "inspect_right (veer R)", "strafe_left": "strafe_left",
    "strafe_right": "strafe_right", "turn_left": "turn_left (rotate)",
    "turn_right": "turn_right (rotate)", "pause": "pause (creep)", "stop": "stop",
}


def smooth(xy, k=3):
    """轻量居中滑动平均, 去锯齿(可视化用; 原始路点也画成散点保持诚实)。"""
    if len(xy) < 3:
        return xy
    out = xy.copy()
    for i in range(len(xy)):
        a, b = max(0, i - k), min(len(xy), i + k + 1)
        out[i] = xy[a:b].mean(0)
    return out


def draw_corridor(ax, half=1.6, x0=-0.5, x1=3.6):
    ax.axhspan(-half - 0.5, -half, xmin=0, xmax=1, color="0.85", zorder=0)  # placeholder(用竖直墙)
    ax.axvspan(0, 0, color="none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="colmapjoint_all")
    ap.add_argument("--pose", type=int, default=20, help="代表观测位姿下标")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--tag", default="hero", help="输出文件名后缀")
    args = ap.parse_args()

    npz = args.npz or os.path.join(RUNS, f"model_trajs_{args.session}.npz")
    d = np.load(npz, allow_pickle=True)
    trajs = d["trajs"]                       # (N,9,8,3) ego (x前,y左,yaw)
    conds = [str(c) for c in d["conds"]]
    ci = {c: i for i, c in enumerate(conds)}
    p = min(args.pose, trajs.shape[0] - 1)

    fpv_path = os.path.join(ROOT, "outputs", f"obs_seq_{args.session}", f"{p:03d}.jpg")
    fpv = plt.imread(fpv_path) if os.path.exists(fpv_path) else None

    fig = plt.figure(figsize=(16.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.0], wspace=0.22)

    # ---------- (1) 观测 FPV ----------
    ax0 = fig.add_subplot(gs[0, 0])
    if fpv is not None:
        ax0.imshow(fpv)
    ax0.set_title(f"Observation — front camera (pose #{p})", fontsize=12, fontweight="bold")
    ax0.axis("off")
    ax0.text(0.5, -0.06, "same observation fed to fast brain for ALL commands below",
             transform=ax0.transAxes, ha="center", va="top", fontsize=9.5, color="0.3")

    # ---------- (2) 位置行为(走廊俯视, ego) ----------
    ax1 = fig.add_subplot(gs[0, 1])
    half = 1.6
    ax1.axvspan(-2.4, -half, color="0.88", zorder=0)   # 左墙(ego: 左=+y=图上方? 用 x前朝上, y左朝左)
    ax1.axvspan(half, 2.4, color="0.88", zorder=0)      # 右墙
    for cond in GROUPS["move"]:
        tr = trajs[p, ci[cond]]
        xy = tr[:, :2]                                  # x前, y左
        # 画到 ego 俯视: 横轴=y左(正=左), 纵轴=x前
        Y = np.concatenate([[0], xy[:, 1]]); X = np.concatenate([[0], xy[:, 0]])
        Ys = smooth(np.stack([Y, X], 1))[:, 0]; Xs = smooth(np.stack([Y, X], 1))[:, 1]
        ax1.plot(Ys, Xs, "-", color=COLOR[cond], lw=3.2, label=LABEL[cond], zorder=3, solid_capstyle="round")
        ax1.plot(Y, X, ".", color=COLOR[cond], ms=4, alpha=0.5, zorder=2)   # 原始路点(诚实)
        ax1.annotate("", xy=(Ys[-1], Xs[-1]), xytext=(Ys[-2], Xs[-2]),
                     arrowprops=dict(arrowstyle="-|>", color=COLOR[cond], lw=3.2), zorder=4)
    ax1.plot(0, 0, "k^", ms=13, zorder=5)               # 机器人
    ax1.set_xlim(2.4, -2.4)                              # 左=+y 放左边(镜像): 反转x轴
    ax1.set_ylim(-0.4, 3.4)
    ax1.set_aspect("equal")
    ax1.set_xlabel("lateral  (+left)  [m]", fontsize=10)
    ax1.set_ylabel("forward  [m]", fontsize=10)
    ax1.set_title("Position behaviors — same obs, different command", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper center", fontsize=8.7, ncol=2, framealpha=0.9)
    ax1.grid(alpha=0.25)

    # ---------- (3) 转视角 / 停(朝向扇形 + 近停) ----------
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axvspan(-2.4, -half, color="0.88", zorder=0)
    ax2.axvspan(half, 2.4, color="0.88", zorder=0)
    for cond in GROUPS["rotate"]:
        tr = trajs[p, ci[cond]]
        xy = tr[:, :2]; yaw_end = float(tr[-1, 2])
        Y = np.concatenate([[0], xy[:, 1]]); X = np.concatenate([[0], xy[:, 0]])
        if cond in ("turn_left", "turn_right"):
            # 朝向扇形: 从机器人画一根随 yaw 转的箭头(位置几乎不动)
            L = 1.1
            ax2.annotate("", xy=(-L * np.sin(yaw_end), L * np.cos(yaw_end)), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=COLOR[cond], lw=3.2), zorder=4)
            ax2.text(-L * np.sin(yaw_end) * 1.12, L * np.cos(yaw_end) * 1.12, LABEL[cond],
                     color=COLOR[cond], fontsize=8.7, ha="center", fontweight="bold")
        else:
            ax2.plot(Y, X, "-", color=COLOR[cond], lw=3.2, label=LABEL[cond], zorder=3, solid_capstyle="round")
            ax2.plot(Y[-1], X[-1], "o", color=COLOR[cond], ms=7, zorder=4)
    ax2.plot(0, 0, "k^", ms=13, zorder=5)
    ax2.plot([0, 0], [0, 1.0], "k--", lw=1, alpha=0.4, zorder=1)   # 原朝向参考
    ax2.set_xlim(1.6, -1.6); ax2.set_ylim(-0.4, 1.5); ax2.set_aspect("equal")
    ax2.set_xlabel("lateral  (+left)  [m]", fontsize=10)
    ax2.set_title("Rotate-in-place / stop\n(position ~fixed, heading changes)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower center", fontsize=8.7, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.suptitle("Semantic controllability: one observation → slow-brain command → fast-brain trajectory follows the command",
                 fontsize=13.5, fontweight="bold", y=1.02)
    out = os.path.join(RUNS, f"controllability_{args.tag}_{args.session}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
