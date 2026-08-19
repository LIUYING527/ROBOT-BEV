"""诊断105731室外场景真实形状: 对齐轨迹俯视(带朝向箭头) + 高斯点云俯视 + 采样FPV帧。
用法: ~/discoverse_venv/bin/python scripts/_viz_105731_scene.py
产出: outputs/_scene_105731.png
"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        break

W = np.load(os.path.join(ROOT, "outputs", "sim_world_colmapjoint_105731.npz"))
wp = W["waypoints_xy"]; yaw = W["yaw"]; N = len(wp)


def read_ply_xyz_rgb(path, stride=40):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.read(1)
        props = [l.split()[-1] for l in head.decode("ascii", "ignore").splitlines()
                 if l.startswith("property float")]
        nverts = int([l for l in head.decode("ascii", "ignore").splitlines()
                      if l.startswith("element vertex")][0].split()[-1])
        data = np.frombuffer(f.read(nverts * len(props) * 4), np.float32).reshape(nverts, len(props))
    xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
    xyz = data[::stride, [xi, yi, zi]]
    # DC color -> rgb
    try:
        ri, gi, bi = props.index("f_dc_0"), props.index("f_dc_1"), props.index("f_dc_2")
        rgb = 0.5 + 0.2820948 * data[::stride, [ri, gi, bi]]
        rgb = np.clip(rgb, 0, 1)
    except ValueError:
        rgb = None
    return xyz, rgb


fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(3, 4, height_ratios=[2.2, 1, 1])

# --- 俯视: 高斯点云 + 轨迹 + 朝向 ---
ax = fig.add_subplot(gs[0, :])
xyz, rgb = read_ply_xyz_rgb(os.path.join(ROOT, "outputs", "gs_vggto_colmapjoint_105731_world.ply"))
# 只画地面上方合理高度区间, 俯视 x-y
zc = xyz[:, 2]
zlo, zhi = np.percentile(zc, 2), np.percentile(zc, 98)
m = (zc > zlo) & (zc < zhi)
if rgb is not None:
    ax.scatter(xyz[m, 0], xyz[m, 1], s=0.5, c=rgb[m], alpha=0.35, linewidths=0)
else:
    ax.scatter(xyz[m, 0], xyz[m, 1], s=0.5, c="0.6", alpha=0.35, linewidths=0)
ax.plot(wp[:, 0], wp[:, 1], "-", c="red", lw=1.5, label="对齐轨迹(937帧)")
step = max(1, N // 40)
ax.quiver(wp[::step, 0], wp[::step, 1], np.cos(yaw[::step]), np.sin(yaw[::step]),
          color="lime", scale=25, width=0.003, label="朝向yaw")
ax.scatter(wp[0, 0], wp[0, 1], c="blue", s=120, marker="o", label="起点", zorder=5)
ax.scatter(wp[-1, 0], wp[-1, 1], c="black", s=120, marker="s", label="终点", zorder=5)
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper right")
ax.set_title(f"105731 室外场景俯视: 高斯点云(色)+对齐轨迹(红)+朝向(绿) | "
             f"x跨{np.ptp(wp[:,0]):.1f}m y跨{np.ptp(wp[:,1]):.1f}m 路径{np.sum(np.linalg.norm(np.diff(wp,axis=0),axis=1)):.1f}m "
             f"yaw累计{np.ptp(np.unwrap(yaw)):.1f}rad")

# --- 采样 FPV 帧 ---
frames = sorted(glob.glob(os.path.join(ROOT, "outputs", "vggto_colmapjoint_105731", "frames_zed", "*.jpg")))
picks = np.linspace(0, len(frames) - 1, 8).astype(int)
for j, fi in enumerate(picks):
    r, c = 1 + j // 4, j % 4
    axi = fig.add_subplot(gs[r, c])
    im = cv2.cvtColor(cv2.imread(frames[fi]), cv2.COLOR_BGR2RGB)
    axi.imshow(im); axi.axis("off")
    axi.set_title(f"帧{fi}  x={wp[fi,0]:.1f} y={wp[fi,1]:.1f} yaw={np.degrees(yaw[fi]):.0f}°", fontsize=9)

plt.tight_layout()
out = os.path.join(ROOT, "outputs", "_scene_105731.png")
plt.savefig(out, dpi=95, bbox_inches="tight")
print("[ok] ->", out)
