"""当前跑通链路的架构图(sim_fullchain.py 单进程实时闭环)。
在 draw_arch.py 基础上改成"现在真在跑"的版本。框内只放短标签(防重叠),
细节(9 类条件等)见汇报文字, 不塞进图里。
输出: outputs/arch_diagram_current.png
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

W, H = 16.0, 7.2
fig, ax = plt.subplots(figsize=(W, H), dpi=170)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, text, ec, fc, fs=11.5, bold=False, lw=2.0, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#0f172a", zorder=z + 1, fontweight="bold" if bold else "normal")


def arr(x1, y1, x2, y2, c=GRY, lw=2.2, rad=0.0, ms=16, z=5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def tag(x, y, text, c, ha="left", fs=10):
    ax.text(x, y, text, fontsize=fs, color=c, fontweight="bold", ha=ha, va="center", zorder=6)


def draw_gaussian(cx, cy):
    for k, col in enumerate(COLS):
        ang = -0.6 + k * 0.3; n = 80; tt = np.random.rand(n)
        px = cx + tt * 0.9 * np.cos(ang) + np.random.randn(n) * 0.05
        py = cy + tt * 0.9 * np.sin(ang) + np.random.randn(n) * 0.05
        ax.scatter(px, py, s=5, color=col, alpha=0.55, zorder=4, edgecolors="none")


def draw_trajs(x0, y0, L, amps, lw=2.6):
    t = np.linspace(0, 1, 36)
    for i, a in enumerate(amps):
        ax.plot(x0 + t * L, y0 + a * (t ** 1.6), color=COLS[i % len(COLS)], lw=lw, alpha=0.95, zorder=4)


# ---------- 标题 ----------
ax.text(W / 2, 6.92, "当前跑通链路：自研 VLM 慢脑(定位) → 语义条件注入 → DiffusionDrive 快脑生成可控轨迹",
        fontsize=13.5, fontweight="bold", ha="center", color="#0f172a")
ax.text(W / 2, 6.55, "单进程实时闭环 (sim_fullchain.py)", fontsize=10.5, ha="center", color=GRY)

# ---------- 上路:轨迹先验(示意图) ----------
ax.text(1.5, 6.15, "锚定高斯分布 (20 锚点)", fontsize=10.5, fontweight="bold", ha="center", color="#0f172a")
draw_gaussian(1.2, 5.15)
ax.text(4.5, 6.15, "N 条噪声轨迹", fontsize=10.5, fontweight="bold", ha="center", color="#0f172a")
draw_trajs(3.7, 5.15, 1.5, [-0.32, -0.04, 0.30])
arr(2.7, 5.18, 3.55, 5.18); ax.text(3.1, 5.42, "采样", fontsize=9.5, ha="center", color=GRY)

# ---------- 输入 FPV ----------
box(0.35, 3.35, 2.7, 1.05, "FPV 前视图像\n(ZED 重建 3DGS 皮肤)", GRY, "#f1f5f9", fs=10.5, bold=True)

# ---------- 慢脑:自研 VLM → 语义条件 ----------
tag(0.4, 2.95, "慢脑 (异步 · HTTP API)", BLUE, fs=9.5)
box(0.35, 1.55, 2.7, 1.15, "自研 VLM 慢脑\n定位: 目标 左/中/右", BLUE, BLUE_F, fs=10.5, bold=True)
box(3.7, 1.62, 2.6, 1.0, "语义条件 cond_id\n(9 类)", BLUE, "#cfe0ff", fs=11, bold=True)
arr(3.05, 2.1, 3.7, 2.1, c=BLUE)
arr(1.7, 3.35, 1.7, 2.7, c=GRY)

# ---------- 快脑:感知 + 扩散解码器 ----------
box(3.7, 3.35, 2.6, 1.05, "感知编码\n相机特征 + BEV 点云特征", ORG, ORG_F, fs=9.8, bold=True)
arr(3.05, 3.85, 3.7, 3.85, c=GRY)
tag(7.0, 5.15, "快脑 (实时)", PURP, fs=11)
box(6.9, 2.55, 2.4, 2.3, "扩散解码器\nDiffusionDrive\n2 步去噪", PURP, PURP_F, fs=11.5, bold=True)

# 三路进解码器:噪声轨迹 / 感知 / 语义条件(FiLM)
arr(5.25, 4.9, 6.9, 4.35, c=GRY)
arr(6.3, 3.85, 6.9, 3.75, c=ORG)
arr(6.3, 2.12, 6.9, 2.85, c=BLUE, lw=2.4, rad=0.14)
tag(4.4, 1.2, "cond_id → cond_embed → FiLM 逐层调制注入解码器", BLUE, ha="left", fs=9)

# ---------- 输出 → 控制器 → 机器人 ----------
box(9.75, 3.05, 2.5, 1.5, "多模轨迹 / Top-1", GRY, "#f1f5f9", fs=11.5, bold=True)
arr(9.3, 3.8, 9.75, 3.8, c=PURP)
box(12.75, 3.65, 2.9, 1.1, "控制器\n路点 →(vx, vy, wz) 全向 + 避障", GRN, GRN_F, fs=9.6, bold=True)
box(12.75, 1.95, 2.9, 1.1, "机器人执行\n仿真器(现) / Jetson Orin 真机(将)", GRN, GRN_F, fs=9.6, bold=True)
arr(12.25, 3.9, 12.75, 4.2, c=GRN)
arr(14.2, 3.65, 14.2, 3.05, c=GRN)

# ---------- 闭环回边(背景, 走底部, 不压框) ----------
arr(14.2, 1.95, 1.7, 1.55, c=PH, lw=1.5, rad=-0.22, ms=13, z=2)
ax.text(8.0, 0.62, "闭环: 执行后环境渲下一帧 FPV, 回慢脑/快脑 (receding horizon)",
        fontsize=9, ha="center", color=PH, style="italic", zorder=6)

# ---------- 图例(底部整行) ----------
xx = 0.5
for c, t in [(BLUE, "慢脑 VLM(语义条件)"), (ORG, "快脑 感知"), (PURP, "DiffusionDrive 解码器"), (GRN, "执行")]:
    ax.add_patch(Rectangle((xx, 0.12), 0.26, 0.26, fc=c, ec="none"))
    ax.text(xx + 0.36, 0.25, t, fontsize=9.2, va="center", color="#334155")
    xx += len(t) * 0.32 + 1.0

fig.savefig("outputs/arch_diagram_current.png", bbox_inches="tight", facecolor="white")
print("[ok] outputs/arch_diagram_current.png")
