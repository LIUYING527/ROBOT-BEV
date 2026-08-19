"""俯视分析111450: 点云投影 + 路径 + 每格离地高度(找立柱=高凸出物) + 距路径侧别。
用法: ~/discoverse_venv/bin/python scripts/_analyze_poles_111450.py [session]
产出: outputs/_poles_111450.png
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name(); break

SESS = sys.argv[1] if len(sys.argv) > 1 else "111450"
W = np.load(os.path.join(ROOT, "outputs", f"sim_world_colmapjoint_{SESS}.npz"))
wp = W["waypoints_xy"]; yaw = W["yaw"]; gz = float(W["ground_z"])


def read_ply_xyz(path, stride=8):
    with open(path, "rb") as f:
        head = b""
        while b"end_header" not in head:
            head += f.read(1)
        lines = head.decode("ascii", "ignore").splitlines()
        props = [l.split()[-1] for l in lines if l.startswith("property float")]
        nv = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
        data = np.frombuffer(f.read(nv * len(props) * 4), np.float32).reshape(nv, len(props))
    xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
    return data[::stride, [xi, yi, zi]]


xyz = read_ply_xyz(os.path.join(ROOT, "outputs", f"gs_vggto_colmapjoint_{SESS}_world.ply"))
# 裁到路径周围包围盒(去远处离群高斯) + 有限值
pad = 8.0
bb = ((xyz[:, 0] > wp[:, 0].min() - pad) & (xyz[:, 0] < wp[:, 0].max() + pad) &
      (xyz[:, 1] > wp[:, 1].min() - pad) & (xyz[:, 1] < wp[:, 1].max() + pad) &
      np.isfinite(xyz).all(1) & (xyz[:, 2] - gz > -1.0) & (xyz[:, 2] - gz < 12.0))
xyz = xyz[bb]
print(f"[info] 裁后点数 {len(xyz)}")
h = xyz[:, 2] - gz   # 离地高度
# 网格化: 每 0.5m 格, 求该格最高点(立柱/桥柱=高)
res = 0.4
xmin, ymin = xyz[:, 0].min(), xyz[:, 1].min()
gx = ((xyz[:, 0] - xmin) / res).astype(int); gy = ((xyz[:, 1] - ymin) / res).astype(int)
nx, ny = gx.max() + 1, gy.max() + 1
maxh = np.full((nx, ny), -1e9)
for i in range(len(xyz)):
    if h[i] > maxh[gx[i], gy[i]]:
        maxh[gx[i], gy[i]] = h[i]
maxh[maxh < -1e8] = np.nan

fig, axs = plt.subplots(2, 1, figsize=(20, 11))
# 上: 离地高度俯视(立柱=亮)
ax = axs[0]
im = ax.imshow(maxh.T, origin="lower", extent=[xmin, xmin + nx * res, ymin, ymin + ny * res],
               aspect="equal", cmap="turbo", vmin=0, vmax=np.nanpercentile(maxh, 98))
ax.plot(wp[:, 0], wp[:, 1], "-", c="white", lw=2, label="路径")
st = max(1, len(wp) // 40)
ax.quiver(wp[::st, 0], wp[::st, 1], np.cos(yaw[::st]), np.sin(yaw[::st]),
          color="black", scale=25, width=0.003, label="朝向")
ax.scatter(wp[0, 0], wp[0, 1], c="lime", s=120, zorder=5, label="起点")
ax.legend(loc="upper left"); plt.colorbar(im, ax=ax, label="离地高度m")
ax.set_title(f"{SESS} 每格最高点(立柱/桥柱=暖色高) + 路径/朝向")
# 下: 侧别分析 —— 沿路径参数s, 高结构在路径左还是右
ax = axs[1]
# 对每个高格(h>1.5m)算它在最近路径点的左/右
tall = np.column_stack(np.where(maxh > 1.5))
if len(tall):
    tx = xmin + (tall[:, 0] + 0.5) * res; ty = ymin + (tall[:, 1] + 0.5) * res
    from scipy.spatial import cKDTree
    tree = cKDTree(wp)
    _, idx = tree.query(np.column_stack([tx, ty]))
    # 左右: (点-路径点) 在 该处朝向法向的投影
    dx = tx - wp[idx, 0]; dy = ty - wp[idx, 1]
    nrmx, nrmy = -np.sin(yaw[idx]), np.cos(yaw[idx])   # 左法向
    sideval = dx * nrmx + dy * nrmy
    sc = ax.scatter(tx, ty, c=sideval, cmap="coolwarm", s=8, vmin=-6, vmax=6)
    ax.plot(wp[:, 0], wp[:, 1], "-", c="black", lw=2)
    plt.colorbar(sc, ax=ax, label="相对路径左(+)/右(-)")
    ax.set_title(f"高结构(>1.5m)相对路径侧别: 蓝=左 红=右  (共{len(tall)}格)  "
                 f"左均值{sideval[sideval>0].mean() if (sideval>0).any() else 0:.1f} 右均值{sideval[sideval<0].mean() if (sideval<0).any() else 0:.1f}"
                 f" | 左占{(sideval>0).mean()*100:.0f}% 右占{(sideval<0).mean()*100:.0f}%")
ax.set_aspect("equal")
plt.tight_layout()
out = os.path.join(ROOT, "outputs", f"_poles_{SESS}.png")
plt.savefig(out, dpi=90, bbox_inches="tight"); print("[ok]", out)
