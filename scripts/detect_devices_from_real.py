"""从【真实采集数据】提取设备巡检站位 -> device_clusters.npy。
⭐动机: 重建几何(A3D/gs_vggto占据)里中段设备贴墙糊成一片, 几何检不出(采集天花板, 已确证)。
   但喂给重建的【真实图】里设备是真真切切拍到的(泵站/阀门/压力表), 且真实采集本身含
   "巡检机动": 操作员开到设备旁->转身横扫(strafe)->继续。nav_ds_real 的 kinds 标了每帧
   cruise/strafe, strafe段=操作员正在检视某设备 => strafe站位中心=该设备的巡检站位,
   站位时的 yaw 指向设备 => 定左右侧。坐标系与仿真env一致(都源自同一次真实采集, x∈[-0.3,34.5])。

用法: ~/discoverse_venv/bin/python scripts/detect_devices_from_real.py
产出: outputs/device_clusters.npy  (K,5)=[station_x, station_y, side(+1左/-1右), x_lo前沿, x_hi后沿]
      前沿/后沿=该设备真实横扫的沿廊起止 -> 巡检机动: 到x_lo转正对, crab扫到x_hi转回。
      outputs/_device_from_real.png (叠占据+真图站位核对图)
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MERGE_X = 2.5      # 站位中心 x 相距 <MERGE_X 米 视为同一设备(横扫会被切成多小段)
MIN_FRAMES = 8     # 合并后总帧数 >=MIN_FRAMES 才算真设备(滤噪声/转弯瞬间)
EDGE_SKIP = 2.0    # 走廊两端 <EDGE_SKIP 或 >max-EDGE_SKIP 的段=起终点掉头, 丢
LEN_CAP = 3.5      # 单设备横扫长度上限(真实末端有来回长扫, 截断到合理值防重叠)


def main():
    meta = np.load(os.path.join(ROOT, "outputs", "nav_ds_real", "meta.npz"), allow_pickle=True)
    poses = meta["poses"]; kinds = meta["kinds"]; fpv = meta["fpv"]
    xmin, xmax = poses[:, 0].min(), poses[:, 0].max()

    # 1) 横扫(strafe)连续段
    scan = kinds != "cruise"
    idx = np.where(scan)[0]
    segs = []; g = [idx[0]]
    for i in idx[1:]:
        if i - g[-1] <= 3:
            g.append(i)
        else:
            segs.append(g); g = [i]
    segs.append(g)

    # 2) 每段的站位中心 + 朝向
    raw = []
    for s in segs:
        p = poses[s]
        cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
        # side: 站位时 yaw 指向设备, sin(yaw)>0 => 设备在 +y(左/上)
        syaw = float(np.mean(np.sin(p[:, 2])))
        raw.append({"x": cx, "y": cy, "n": len(s), "syaw": syaw,
                    "xlo": float(p[:, 0].min()), "xhi": float(p[:, 0].max()),
                    "frame": int(s[len(s) // 2])})

    # 3) 按 x 合并近邻段 -> 设备
    raw.sort(key=lambda r: r["x"])
    dev = []
    for r in raw:
        if dev and abs(r["x"] - dev[-1]["x"]) < MERGE_X:
            d = dev[-1]                                       # 帧数加权并入
            tot = d["n"] + r["n"]
            d["x"] = (d["x"] * d["n"] + r["x"] * r["n"]) / tot
            d["y"] = (d["y"] * d["n"] + r["y"] * r["n"]) / tot
            d["syaw"] = (d["syaw"] * d["n"] + r["syaw"] * r["n"]) / tot
            d["xlo"] = min(d["xlo"], r["xlo"]); d["xhi"] = max(d["xhi"], r["xhi"])
            d["n"] = tot; d["frames"].append(r["frame"])
        else:
            dev.append({**r, "frames": [r["frame"]]})

    # 4) 过滤: 帧数够 + 不在走廊端点掉头区
    out = []
    for d in dev:
        if d["n"] < MIN_FRAMES:
            continue
        if d["x"] < xmin + EDGE_SKIP or d["x"] > xmax - EDGE_SKIP:
            continue
        side = 1.0 if d["syaw"] > 0 else -1.0
        # 横扫长度截断到 LEN_CAP 并以站位为中心(末端真实来回长扫会重叠, 归一化)
        L = min(d["xhi"] - d["xlo"], LEN_CAP)
        xlo = d["x"] - L / 2; xhi = d["x"] + L / 2
        out.append((round(d["x"], 2), round(d["y"], 2), side,
                    round(xlo, 2), round(xhi, 2), d["n"], d["frames"]))

    # 相邻设备扫描区间去重叠: 边界取两站位中点
    out.sort(key=lambda o: o[0])
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if a[4] > b[3]:                                       # a后沿 > b前沿 => 重叠
            mid = round((a[0] + b[0]) / 2, 2)
            out[i] = (*a[:4], min(a[4], mid), *a[5:])
            out[i + 1] = (*b[:3], max(b[3], mid), *b[4:])

    print(f"[real-dev] 横扫段{len(segs)} -> 合并过滤后设备{len(out)}个:")
    arr = []
    for x, y, side, xlo, xhi, n, frames in out:
        print(f"  站位x={x:5.1f} y={y:5.1f} side={'左' if side>0 else '右'}({int(side):+d}) "
              f"前沿x_lo={xlo:.1f} 后沿x_hi={xhi:.1f} 长{xhi-xlo:.1f}m 帧数={n} 样帧={frames[0]}")
        arr.append([x, y, side, xlo, xhi])
    arr = np.array(arr, np.float32)
    np.save(os.path.join(ROOT, "outputs", "device_clusters.npy"), arr)
    print(f"[ok] -> outputs/device_clusters.npy  {arr.shape}  格式[x,y,side,x_lo,x_hi]")

    # 5) 核对图: 占据 + 站位 + 真图缩略
    try:
        _viz(out, poses, fpv)
    except Exception as e:
        print("[viz skip]", e)


def _viz(out, poses, fpv):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import cv2
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(poses[:, 0], poses[:, 1], '.', ms=1, c='0.7', label='real cruise path')
    for x, y, side, xlo, xhi, n, frames in out:
        c = 'blue' if side > 0 else 'red'
        ax.plot([xlo, xhi], [y, y], c=c, lw=3, alpha=.5)      # 前沿->后沿扫描区间
        ax.scatter(x, y, c=c, s=60 + n, marker=('v' if side > 0 else '^'))
        ax.annotate(f"x{x:.0f}\n{'L' if side>0 else 'R'} {xhi-xlo:.1f}m", (x, y),
                    fontsize=8, color=c)
    ax.legend(); ax.set_aspect('equal'); ax.grid(alpha=.3)
    ax.set_title(f"设备巡检站位(真实横扫段) {len(out)}个: L=蓝v R=红^")
    plt.savefig(os.path.join(ROOT, "outputs", "_device_from_real.png"), dpi=95, bbox_inches='tight')
    print("[viz] -> outputs/_device_from_real.png")


if __name__ == "__main__":
    main()
