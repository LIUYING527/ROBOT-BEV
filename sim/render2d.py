"""游戏感 2D 俯视渲染（matplotlib, headless）。

风格参考：蓝色棋盘格地面 + 暗红粗墙线 + 箭头机器人 + 彩色分类目标。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrow

CAT_COLOR = {"blue": "#2a3ed8", "green": "#3fae46", "red": "#e23b2e"}
CAT_MARKER = {"blue": "o", "green": "o", "red": "s"}
FLOOR_A = "#9fc4e8"   # 棋盘格浅色
FLOOR_B = "#8fb8e0"   # 棋盘格深色
WALL_C = "#7a1f2b"    # 暗红墙


def _draw_floor(ax, bounds, tile=1.0):
    xmin, ymin, xmax, ymax = bounds
    nx = int(np.ceil((xmax - xmin) / tile))
    ny = int(np.ceil((ymax - ymin) / tile))
    checker = np.indices((ny, nx)).sum(0) % 2
    ax.imshow(checker, cmap=matplotlib.colors.ListedColormap([FLOOR_A, FLOOR_B]),
              extent=[xmin, xmax, ymin, ymax], origin="lower",
              interpolation="nearest", zorder=0, alpha=0.9)


def _draw_walls(ax, walls, lw=6):
    segs = [[(x1, y1), (x2, y2)] for (x1, y1, x2, y2) in walls]
    lc = LineCollection(segs, colors=WALL_C, linewidths=lw,
                        capstyle="round", zorder=3)
    ax.add_collection(lc)


def _draw_targets(ax, targets, size=180):
    for cat in CAT_COLOR:
        pts = [(x, y) for (x, y, c) in targets if c == cat]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=size, c=CAT_COLOR[cat], marker=CAT_MARKER[cat],
                       edgecolors="white", linewidths=1.2, zorder=4,
                       label=f"{cat} target")


def draw_robot(ax, x, y, theta, L=0.9):
    dx, dy = L * np.cos(theta), L * np.sin(theta)
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.28, head_width=0.7,
                            head_length=0.55, length_includes_head=True,
                            color="#111111", zorder=6))
    ax.scatter([x], [y], s=60, c="#1b6cff", edgecolors="white",
               linewidths=1.5, zorder=7)


def render_world(world, robot_pose=None, title="2D Inspection Sim",
                 trail=None, save=None, show_legend=True):
    xmin, ymin, xmax, ymax = world.bounds
    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_floor(ax, world.bounds)
    _draw_walls(ax, world.walls)
    _draw_targets(ax, world.targets)
    if trail is not None and len(trail) > 1:
        t = np.asarray(trail)
        ax.plot(t[:, 0], t[:, 1], "-", color="#1b6cff", lw=2, alpha=0.8, zorder=5)
    if robot_pose is not None:
        draw_robot(ax, *robot_pose)
    ax.set_xlim(xmin - 0.3, xmax + 0.3)
    ax.set_ylim(ymin - 0.3, ymax + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140)
        print(f"[OK] 已保存 {save}")
    return fig, ax
