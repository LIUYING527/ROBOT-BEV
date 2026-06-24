"""从可导航几何(world_geometry_<s>.json)栅格化出 BEV 占据图 = DiffusionDrive 的输入观测。
占据=墙+障碍盒(膨胀机器人半径), 自由=走廊内部。视角无关, 任意机器人位置都能取局部窗口。

用法: ~/discoverse_venv/bin/python scripts/occupancy_bev.py <session> [--res 0.05] [--robot_r 0.25]
产出: outputs/occ_bev_<s>.npy (HxW uint8: 0自由/1占据) + outputs/occ_bev_<s>.png(可视化)
"""
import os, sys, json, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rasterize(geom, res, robot_r, pad=1.0):
    walls = geom["walls"]; boxes = geom.get("boxes", [])
    # 收集所有几何 xy 范围
    pts = []
    for w in walls:
        for a, b in w["segs"]:
            pts += [a, b]
    for bx in boxes:
        c, h = bx["center"], bx["half"]
        pts += [[c[0] - h[0], c[1] - h[1]], [c[0] + h[0], c[1] + h[1]]]
    pts = np.array(pts)
    lo = pts.min(0) - pad; hi = pts.max(0) + pad
    W = int(np.ceil((hi[0] - lo[0]) / res)); H = int(np.ceil((hi[1] - lo[1]) / res))
    occ = np.zeros((H, W), np.uint8)

    def w2g(x, y):
        return int((x - lo[0]) / res), int((y - lo[1]) / res)

    import cv2
    r_px = max(1, int(robot_r / res))
    for w in walls:
        th = max(1, int((w["thick"] / 2 + robot_r) / res))   # 墙厚+机器人半径膨胀
        for a, b in w["segs"]:
            ga = w2g(*a); gb = w2g(*b)
            cv2.line(occ, ga, gb, 1, thickness=2 * th)
    for bx in boxes:
        c, h = bx["center"], bx["half"]
        p0 = w2g(c[0] - h[0] - robot_r, c[1] - h[1] - robot_r)
        p1 = w2g(c[0] + h[0] + robot_r, c[1] + h[1] + robot_r)
        cv2.rectangle(occ, p0, p1, 1, -1)
    return occ, lo, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--robot_r", type=float, default=0.25)
    args = ap.parse_args()
    geom = json.load(open(os.path.join(ROOT, "outputs", f"world_geometry_{args.session}.json")))
    occ, lo, res = rasterize(geom, args.res, args.robot_r)
    np.save(os.path.join(ROOT, "outputs", f"occ_bev_{args.session}.npy"), occ)

    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    W = np.load(os.path.join(ROOT, "outputs", f"sim_world_{args.session}.npz"))
    wp = W["waypoints_xy"]
    fig, ax = plt.subplots(figsize=(6, 11))
    ax.imshow(occ, origin="lower", cmap="gray_r",
              extent=[lo[0], lo[0] + occ.shape[1] * res, lo[1], lo[1] + occ.shape[0] * res])
    ax.plot(wp[:, 0], wp[:, 1], "-", c="tab:green", lw=2, label="robot path")
    ax.scatter(wp[0, 0], wp[0, 1], c="lime", s=60, label="start")
    ax.set_aspect("equal"); ax.legend()
    ax.set_title(f"BEV occupancy {args.session} (black=occupied, white=free)\n机器人半径膨胀 res={res}m")
    fig.savefig(os.path.join(ROOT, "outputs", f"occ_bev_{args.session}.png"), dpi=90, bbox_inches="tight")
    free = 100 * (occ == 0).mean()
    print(f"[occ] {occ.shape} grid @res{res}m, 自由空间{free:.0f}%, 路点{len(wp)} -> occ_bev_{args.session}.npy/.png")


if __name__ == "__main__":
    main()
