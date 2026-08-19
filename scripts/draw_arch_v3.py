"""当前架构图 v3(0805版, 沿用v2横向流程风格)。
相比v2(0715走廊巡检版)的更新:
- 加离线建图带(3DGS深度监督+z场扶平 -> 场景图RAG 185节点)
- 慢脑 = 四段推理链confirm + key-frame复核; 接口 = NavCmd参数包{L,tau,Phi}
- 快脑 = scene_gate + 时序历史(W,gamma)走status[5:6] + 搜索行为(骨架DFS数据)
- 控制器 = turn-then-go + safe_vel; 待接 = GS重定位(深度偏差断点)
输出: outputs/arch_diagram_v3.png
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
FS = 9.5   # 所有框内统一字号

W, H = 17.0, 8.6
fig, ax = plt.subplots(figsize=(W, H), dpi=170)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def box(x, y, w, h, text, ec, fc, lw=2.0, z=4, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=z,
                                linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS,
            color="#0f172a", zorder=z + 1, fontweight="bold")


def arr(x1, y1, x2, y2, c=GRY, lw=2.2, rad=0.0, ms=15, z=5, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                 color=c, lw=lw, zorder=z, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))


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
ax.text(W / 2, 8.28, "当前架构(0805): 场景图RAG慢脑 → NavCmd参数包 → DiffusionDrive快脑(时序历史+搜索行为)",
        fontsize=13.5, fontweight="bold", ha="center", color="#0f172a")
ax.text(W / 2, 7.94, "任务闭环 lab_mission.py · 慢脑 检索定位/四段confirm · 快脑 巡航/对准/搜索 · turn-then-go 控制",
        fontsize=10.5, ha="center", color=GRY)

# ---------- 离线建图带(顶部) ----------
ax.text(0.9, 7.72, "离线建图 (一次性)", color=GRY, fontsize=9.5, fontweight="bold", ha="left", zorder=6)
box(0.9, 6.78, 2.9, 0.88, "ZED posed RGB-D 采集\nCOLMAP位姿+度量深度", GRY, "#f1f5f9")
box(4.15, 6.78, 2.9, 0.88, "3DGS 重建\n深度监督 · 世界扶平(z场)", ORG, ORG_F)
box(7.4, 6.78, 3.3, 0.88, "场景图 RAG (185节点)\n坐标 · 正面front · crops · tag", BLUE, BLUE_F)
arr(3.8, 7.22, 4.15, 7.22)
arr(7.05, 7.22, 7.4, 7.22)

# ---------- 待接: GS重定位(中带右侧空白) ----------
box(12.7, 5.5, 3.4, 0.95, "待接: GS重定位\n(render-compare, 深度偏差断点)", PH, "#f8fafc", dashed=True)
arr(9.6, 6.78, 13.3, 6.46, c=PH, dashed=True, lw=1.6, rad=-0.15, z=2)  # 离线地图→GS重定位

CY = 3.72
# ---------- 主流程(紧凑, 短箭头) ----------
box(0.9, 2.95, 2.7, 1.55, "FPV 渲染 + BEV 占据\n时序历史(W窗 γ衰减)\n(真实原图位)", GRY, "#f1f5f9")
box(3.95, 3.22, 2.3, 1.0, "感知编码\n相机+BEV点云", ORG, ORG_F)
box(6.6, 2.72, 2.7, 2.0, "快脑: 扩散解码器\nDiffusionDrive\nscene_gate 门控\n巡航/对准/搜索(骨架DFS)", PURP, PURP_F)
box(9.65, 2.95, 2.7, 1.55, "多模轨迹 / Top-1\n(巡检轨迹图位)", GRY, "#f1f5f9")
box(12.7, 3.9, 3.4, 0.82, "控制器 turn-then-go\n+safe_vel → (vx,vy,wz)", GRN, GRN_F)
box(12.7, 2.72, 3.4, 0.82, "机器人执行\n仿真(现)/Orin真机(将)", GRN, GRN_F)

arr(3.6, CY, 3.95, CY)
arr(6.25, CY, 6.6, CY)
arr(9.3, CY, 9.65, CY, c=PURP)
arr(12.35, CY, 12.7, 4.3, c=PURP)
arr(14.4, 3.9, 14.4, 3.54, c=GRN)

# ---------- 上路: 扩散先验 ----------
ax.text(2.15, 6.35, "锚定高斯分布 (20 锚点)", fontsize=10.3, fontweight="bold", ha="center", color="#0f172a")
draw_gaussian(1.75, 5.35)
ax.text(5.35, 6.35, "N 条噪声轨迹", fontsize=10.3, fontweight="bold", ha="center", color="#0f172a")
draw_trajs(4.5, 5.55, 1.5, [-0.28, -0.02, 0.28])
arr(3.45, 5.6, 4.4, 5.6); ax.text(3.9, 5.83, "采样", fontsize=9.3, ha="center", color=GRY)
arr(6.15, 5.45, 7.7, 4.78, c=GRY, rad=-0.28)   # N噪声轨迹→快脑

# ---------- 慢脑 ----------
ax.text(0.92, 2.72, "慢脑 (异步 · HTTP API)", color=BLUE, fontsize=9.5, fontweight="bold", ha="left", zorder=6)
box(0.9, 1.35, 2.7, 1.15, "慢脑 VLM (Qwen3.5-4B)\n四段推理链confirm\nkey-frame复核≥2帧yes", BLUE, BLUE_F)
box(3.95, 1.35, 2.5, 1.15, "RAG检索 文本→节点τ\nNavCmd 参数包\n{L, τ, Φ=(W,γ,vx)}", BLUE, "#cfe0ff")
arr(3.6, 1.9, 3.95, 1.9, c=BLUE)                 # 慢脑→参数包
arr(6.45, 1.95, 6.9, 2.72, c=BLUE, lw=2.3, rad=0.14)  # 参数包→快脑
arr(3.3, 2.95, 3.3, 2.5, c=GRY)                  # FPV→慢脑
arr(8.4, 6.78, 6.2, 2.56, c=BLUE, dashed=True, lw=1.6, rad=0.16, z=2)  # 场景图→RAG检索(建图喂查询, 走底层)

# ---------- 可控性机制 ----------
ax.text(9.2, 2.38, "NavCmd Φ → status[5:6]+scene_gate\n同观测换参数 → 轨迹分布变 (可控性)",
        ha="center", va="center", fontsize=9, color=PURP, fontweight="bold", zorder=6)

# ---------- 闭环回边(底层) ----------
arr(14.4, 2.72, 3.3, 2.95, c=PH, lw=1.5, rad=-0.26, ms=13, z=2)
ax.text(8.7, 0.6, "闭环: 执行后环境渲下一帧 FPV+BEV 并 hist_update(巡航/搜索同待遇) · 到站四段confirm → 派发下一腿",
        fontsize=9, ha="center", color=PH, style="italic", zorder=6)

# ---------- 图例(固定位置) ----------
LEG = [(0.6, BLUE, "慢脑 VLM / RAG"), (4.4, ORG, "感知 / 重建"),
       (7.6, PURP, "DiffusionDrive 快脑"), (11.2, GRN, "控制 / 执行"),
       (14.0, PH, "待接(断点)")]
for xx, c, t in LEG:
    ax.add_patch(Rectangle((xx, 0.12), 0.26, 0.26, fc=c, ec="none"))
    ax.text(xx + 0.38, 0.25, t, fontsize=9.2, va="center", color="#334155")

fig.savefig("outputs/arch_diagram_v3.png", bbox_inches="tight", facecolor="white")
print("[ok] outputs/arch_diagram_v3.png")
