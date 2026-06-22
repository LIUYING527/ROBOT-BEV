"""我们的架构图(参考 DiffusionDrive 图(a)总体流水线风格)。
- 示意图(锚定高斯分布 / 噪声轨迹)直接画出来
- 只给"实际场景图"留虚线占位框(输入真实场景照片 + 输出多模轨迹场景图,用户自行插入)
- 扩散解码器作单个框,不画内部;不标"创新点"
- 重点=双脑:VLM 慢脑→任务 id 作为条件
输出: outputs/arch_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(7)

BLUE = "#2563eb"; BLUE_F = "#e8f1ff"
ORG = "#ea580c"; ORG_F = "#fff3e8"
PURP = "#7c3aed"; PURP_F = "#f0ebff"
GRN = "#16a34a"; GRN_F = "#e9f9ef"
GRY = "#475569"; GRY_F = "#eef2f7"
PH = "#94a3b8"
COLS = ["#ec4899", "#f59e0b", "#06b6d4", "#8b5cf6", "#10b981"]

fig, ax = plt.subplots(figsize=(15, 6.4), dpi=170)
ax.set_xlim(0, 15); ax.set_ylim(0, 6.4); ax.axis("off")


def box(x, y, w, h, text, ec, fc, fs=12, bold=False, lw=2.0, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#0f172a", zorder=z + 1, fontweight="bold" if bold else "normal")


def scene_ph(x, y, w, h, text):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc="#f1f5f9", ec=GRY, lw=2.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11.5,
            color="#0f172a", zorder=4, fontweight="bold")


def arr(x1, y1, x2, y2, c=GRY, lw=2.2, rad=0.0, ms=16, z=5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def tag(x, y, text, c, ha="left"):
    ax.text(x, y, text, fontsize=10.5, color=c, fontweight="bold", ha=ha, va="center", zorder=6)


def draw_gaussian(cx, cy):
    for k, col in enumerate(COLS):
        ang = -0.6 + k * 0.3
        n = 80
        tt = np.random.rand(n)
        px = cx + tt * 0.95 * np.cos(ang) + np.random.randn(n) * 0.05
        py = cy + tt * 0.95 * np.sin(ang) + np.random.randn(n) * 0.05
        ax.scatter(px, py, s=5, color=col, alpha=0.55, zorder=4, edgecolors="none")


def draw_trajs(x0, y0, L, amps, lw=2.6):
    t = np.linspace(0, 1, 36)
    for i, a in enumerate(amps):
        ax.plot(x0 + t * L, y0 + a * (t ** 1.6), color=COLS[i % len(COLS)], lw=lw, alpha=0.95, zorder=4)


# ---------- 标题 ----------
ax.text(7.5, 6.12, "我们的架构：VLM 慢脑识别任务  →  DiffusionDrive 快脑按任务条件生成轨迹",
        fontsize=14, fontweight="bold", ha="center", color="#0f172a")

# ---------- 上路:轨迹先验(示意图,直接画) ----------
ax.text(1.5, 5.78, "锚定高斯分布", fontsize=11, fontweight="bold", ha="center", color="#0f172a")
draw_gaussian(1.15, 4.75)
ax.text(4.35, 5.78, "N 条噪声轨迹", fontsize=11, fontweight="bold", ha="center", color="#0f172a")
draw_trajs(3.45, 4.75, 1.55, [-0.34, -0.05, 0.32])
arr(2.55, 4.78, 3.35, 4.78)
ax.text(2.95, 5.0, "采样", fontsize=9.5, ha="center", color=GRY)

# ---------- 输入(实际场景图占位) ----------
scene_ph(0.4, 2.3, 2.3, 1.05, "真实场景图像 (ZED)")

# ---------- 慢脑:VLM → 任务 id ----------
tag(0.42, 1.95, "慢脑 (~0.2 Hz，异步)", BLUE)
box(0.4, 0.55, 2.3, 1.0, "VLM 慢脑\n(Qwen3-3B)", BLUE, BLUE_F, fs=11.5, bold=True)
box(3.35, 0.65, 2.15, 0.85, "离散任务 id", BLUE, "#cfe0ff", fs=12, bold=True)
arr(2.7, 1.05, 3.35, 1.07, c=BLUE)
arr(1.55, 2.3, 1.55, 1.55, c=GRY)

# ---------- 快脑:感知 + 扩散解码器 ----------
box(3.35, 2.4, 2.15, 1.0, "感知 / BEV·PV\n(ResNet)", ORG, ORG_F, fs=11)
arr(2.7, 2.82, 3.35, 2.9, c=GRY)
tag(6.1, 5.0, "快脑 (实时)", PURP)
box(6.0, 2.15, 2.15, 2.55, "扩散解码器\nDiffusionDrive\n2 步去噪", PURP, PURP_F, fs=11.5, bold=True)

# 三路进解码器:噪声轨迹 / 感知 / 任务 id(箭头颜色区分来源,不另加文字避免压线)
arr(5.6, 4.6, 6.0, 4.15, c=GRY)
arr(5.5, 2.9, 6.0, 3.25, c=ORG)
arr(5.5, 1.07, 6.0, 2.55, c=BLUE, lw=2.4, rad=0.12)

# ---------- 输出(实际场景图占位)→ 控制器 → 机器人 ----------
scene_ph(8.55, 2.65, 2.5, 1.7, "多模轨迹 / Top-1")
arr(8.15, 3.45, 8.55, 3.45, c=PURP)
box(11.55, 3.35, 2.45, 1.1, "控制器\n路点 →(v, ω) + 避障", GRN, GRN_F, fs=11)
box(11.55, 1.75, 2.45, 1.1, "机器人执行\nJetson Orin / 仿真器", GRN, GRN_F, fs=11, bold=True)
arr(11.0, 3.5, 11.55, 3.9, c=GRN)
arr(12.77, 3.35, 12.77, 2.85, c=GRN)

# ---------- 图例 ----------
xx = 0.5
for c, t in [(BLUE, "慢脑 VLM"), (ORG, "快脑 感知"), (PURP, "DiffusionDrive 解码器"), (GRN, "执行")]:
    ax.add_patch(Rectangle((xx, 0.1), 0.28, 0.28, fc=c, ec="none"))
    ax.text(xx + 0.38, 0.24, t, fontsize=10, va="center", color="#334155")
    xx += len(t) * 0.26 + 1.5

fig.savefig("outputs/arch_diagram.png", bbox_inches="tight", facecolor="white")
print("[ok] outputs/arch_diagram.png")
