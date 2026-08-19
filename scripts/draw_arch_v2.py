"""当前架构图 v2(紧凑短箭头·统一字号·沿用原版横向流程)。
FPV前视框、多模轨迹框 = 用户放真实原图/巡检轨迹图的位置(同大小)。
输出: outputs/arch_diagram_v2.png
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
GRY = "#475569"
PH = "#94a3b8"
COLS = ["#ec4899", "#f59e0b", "#06b6d4", "#8b5cf6", "#10b981"]
FS = 9.5   # ⭐所有框内统一字号

W, H = 17.0, 7.6
fig, ax = plt.subplots(figsize=(W, H), dpi=170)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, text, ec, fc, lw=2.0, z=4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS,
            color="#0f172a", zorder=z + 1, fontweight="bold")


def arr(x1, y1, x2, y2, c=GRY, lw=2.2, rad=0.0, ms=15, z=5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def draw_gaussian(cx, cy):
    for k, col in enumerate(COLS):
        ang = -0.6 + k * 0.3; n = 80; tt = np.random.rand(n)
        px = cx + tt * 0.85 * np.cos(ang) + np.random.randn(n) * 0.05
        py = cy + tt * 0.85 * np.sin(ang) + np.random.randn(n) * 0.05
        ax.scatter(px, py, s=5, color=col, alpha=0.55, zorder=4, edgecolors="none")


def draw_trajs(x0, y0, L, amps, lw=2.6):
    t = np.linspace(0, 1, 36)
    for i, a in enumerate(amps):
        ax.plot(x0 + t * L, y0 + a * (t ** 1.6), color=COLS[i % len(COLS)], lw=lw, alpha=0.95, zorder=4)


# ---------- 标题 ----------
ax.text(W / 2, 7.28, "当前架构：自研 VLM 慢脑(项目经理) → 命令门控 → DiffusionDrive 快脑生成可控轨迹",
        fontsize=13.5, fontweight="bold", ha="center", color="#0f172a")
ax.text(W / 2, 6.92, "巡检闭环 (run_inspection.py) · 慢脑决策(找目标/检视/到头) · 快脑 走/原地转/横扫 · 往返",
        fontsize=10.5, ha="center", color=GRY)

CY = 3.72
# ---------- 主流程(紧凑, 短箭头) ----------
box(0.9, 2.95, 2.7, 1.55, "FPV 前视图像\n(真实场景原图)", GRY, "#f1f5f9")
box(3.95, 3.22, 2.3, 1.0, "感知编码\n相机+BEV点云", ORG, ORG_F)
box(6.6, 2.72, 2.7, 2.0, "扩散解码器\nDiffusionDrive\n命令门控(办法C)\n走/原地转/横扫", PURP, PURP_F)
box(9.65, 2.95, 2.7, 1.55, "多模轨迹 / Top-1\n(巡检轨迹图)", GRY, "#f1f5f9")
box(12.7, 3.9, 3.4, 0.82, "控制器 路点→(vx,vy,wz)\n全向+避障", GRN, GRN_F)
box(12.7, 2.72, 3.4, 0.82, "机器人执行\n仿真(现)/Orin真机(将)", GRN, GRN_F)

arr(3.6, CY, 3.95, CY)
arr(6.25, CY, 6.6, CY)
arr(9.3, CY, 9.65, CY, c=PURP)
arr(12.35, CY, 12.7, 4.3, c=PURP)
arr(14.4, 3.9, 14.4, 3.54, c=GRN)

# ---------- 上路:轨迹先验(上移) ----------
ax.text(2.15, 6.35, "锚定高斯分布 (20 锚点)", fontsize=10.3, fontweight="bold", ha="center", color="#0f172a")
draw_gaussian(1.75, 5.35)
ax.text(5.35, 6.35, "N 条噪声轨迹", fontsize=10.3, fontweight="bold", ha="center", color="#0f172a")
draw_trajs(4.5, 5.55, 1.5, [-0.28, -0.02, 0.28])
arr(3.45, 5.6, 4.4, 5.6); ax.text(3.9, 5.83, "采样", fontsize=9.3, ha="center", color=GRY)
arr(6.15, 5.45, 7.7, 4.78, c=GRY, rad=-0.28)   # N噪声轨迹→快脑(平滑曲线)
ax.text(7.95, 4.95, "快脑 (实时) · 命令门控", ha="center", va="center",
        fontsize=10.5, color=PURP, fontweight="bold", zorder=6)

# ---------- 慢脑 ----------
ax.text(0.92, 2.72, "慢脑 (异步 · HTTP API)", color=BLUE, fontsize=9.5, fontweight="bold", ha="left", zorder=6)
box(0.9, 1.35, 2.7, 1.15, "自研VLM慢脑(项目经理)\n找目标·决策 检视/到头", BLUE, BLUE_F)
box(3.95, 1.42, 2.3, 1.0, "语义命令 status\n(目标点+朝向)", BLUE, "#cfe0ff")
arr(3.6, 1.9, 3.95, 1.9, c=BLUE)                 # 慢脑→命令
arr(6.25, 1.95, 6.9, 2.72, c=BLUE, lw=2.3, rad=0.14)  # 命令→快脑
arr(3.3, 2.95, 3.3, 2.5, c=GRY)                  # FPV→慢脑(右侧, 避开左标签)

# ---------- 可控性机制(语义命令右边/解码器下方) ----------
ax.text(9.4, 1.75, "命令 status → scene_gate 门控场景特征\n→ 可控轨迹 (同观测换命令 → 轨迹分布变)",
        ha="center", va="center", fontsize=9, color=PURP, fontweight="bold", zorder=6)

# ---------- 闭环回边(底层) ----------
arr(14.4, 2.72, 3.3, 2.95, c=PH, lw=1.5, rad=-0.26, ms=13, z=2)
ax.text(8.7, 0.6, "闭环: 执行后环境渲下一帧 FPV, 回慢脑/快脑 (receding horizon) · 到头原地转 180° 返程",
        fontsize=9, ha="center", color=PH, style="italic", zorder=6)

# ---------- 图例(固定位置) ----------
LEG = [(0.6, BLUE, "慢脑 VLM(语义命令/编排)"), (5.4, ORG, "快脑 感知"),
       (8.3, PURP, "DiffusionDrive(命令门控)"), (12.6, GRN, "控制 / 执行")]
for xx, c, t in LEG:
    ax.add_patch(Rectangle((xx, 0.12), 0.26, 0.26, fc=c, ec="none"))
    ax.text(xx + 0.38, 0.25, t, fontsize=9.2, va="center", color="#334155")

fig.savefig("outputs/arch_diagram_v2.png", bbox_inches="tight", facecolor="white")
print("[ok] outputs/arch_diagram_v2.png")
