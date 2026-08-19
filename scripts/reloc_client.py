"""RELOC 机器人端客户端: 取zed点云 → POST到GPU服务器 → 得纠正位姿(真实米制)。
三种用法:
  ① 真机(在robot ws里, import camera_driver): python reloc_client.py --x 0 --y 0 --yaw 0
  ② 离线npz测试: python reloc_client.py --npz <path.npz> --x .. --y .. --yaw ..
  ③ 服务器验证(在GPU服务器上跑, 用采集npz+COLMAP GT自动扰动打分):
     CUDA_VISIBLE_DEVICES= python reloc_client.py --selftest --frames 8
依赖仅 numpy+requests(机器人环境没有requests则 pip install requests 或换urllib)。
"""
import os, sys, io, json, argparse
import numpy as np

try:
    import requests
except ImportError:
    requests = None


def post_npz(server, npz_bytes, x, y, yaw_deg, pitch_deg=0.0, cam_h=1.2, timeout=180):
    assert requests, "需要requests"
    r = requests.post(f"{server}/reloc",
                      params=dict(x=x, y=y, yaw=yaw_deg, pitch_deg=pitch_deg, cam_h=cam_h),
                      data=npz_bytes, timeout=timeout)
    return r.json()


def npz_bytes_from_pc(pc):
    """(H,W,4) float32/float16 mm → npz字节流(与采集同格式float16)。"""
    buf = io.BytesIO()
    np.savez(buf, xyzrgba=pc.astype(np.float16))
    return buf.getvalue()


def live_grab():
    """真机: 从camera_driver取最新点云。返回(H,W,4)float32 mm。"""
    from camera_driver.zed_camera import ZedCamera   # 在robot ws内
    import time
    cam = ZedCamera()
    cam.start()
    for _ in range(50):
        pc = cam.capture_pointcloud()
        if pc is not None:
            return pc
        time.sleep(0.1)
    raise RuntimeError("取不到zed点云")


def selftest(server, raw_dir, n_frames, seed=0):
    """服务器端验证: 采集npz+COLMAP GT(世界系→真实米制)加扰动POST, 测纠正误差。"""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from gs_relocate import GSReloc
    g = GSReloc.__new__(GSReloc)                      # 只借坐标映射, 不加载高斯(服务器已载)
    import pycolmap
    short = "110744s2"
    rec = None
    for sub in ("sparse/2", "sparse/1", "sparse/0"):
        d = os.path.join(ROOT, "outputs", f"_colmap_joint_{short}", sub)
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, "images.bin")) or
                                 os.path.exists(os.path.join(d, "images.txt"))):
            rec = pycolmap.Reconstruction(d); break
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    Wz = np.load(os.path.join(ROOT, "outputs", "sim_world_colmapjoint_110744s2.npz"))
    Rz, k, s_norm = Wz["Rz"], float(Wz["k"]), float(Wz["s_norm"])
    t_world, center = Wz["t_world"], Wz["center"]
    s = float(os.environ.get("OBS_SCALE", "0.70"))    # 真实→世界
    rng = np.random.RandomState(seed)
    pcdir = os.path.join(ROOT, raw_dir, "pointcloud", "zed")
    errs = []
    cand = np.linspace(5, len(ims) - 6, n_frames * 2).astype(int)
    used = 0
    for j in cand:
        p = os.path.join(pcdir, ims[j].name.replace(".jpg", ".npz"))
        if not os.path.exists(p):
            continue
        V = np.eye(4); V[:3, :3] = ims[j].cam_from_world().rotation.matrix()
        V[:3, 3] = ims[j].cam_from_world().translation
        Rc, tc = V[:3, :3], V[:3, 3]
        C_col = -Rc.T @ tc
        C_w = t_world + (k / s_norm) * (Rz @ (C_col - center))       # 世界系GT
        R_c2w = Rz @ Rc.T
        x_gt, y_gt = C_w[0] / s, C_w[1] / s                          # 真实米制GT
        yaw_gt = np.degrees(np.arctan2(R_c2w[1, 2], R_c2w[0, 2]))
        pitch_gt = np.degrees(np.arcsin(np.clip(R_c2w[2, 2], -1, 1)))
        a = rng.uniform(0, 2 * np.pi); rr = 0.3 * np.sqrt(rng.uniform(0, 1))
        dx, dy = rr * np.cos(a), rr * np.sin(a); dyaw = rng.uniform(-15, 15)
        with open(p, "rb") as f:
            nb = f.read()
        out = post_npz(server, nb, x_gt + dx, y_gt + dy, yaw_gt + dyaw,
                       pitch_deg=pitch_gt, cam_h=C_w[2] / s)
        if not out.get("ok"):
            print(f"  f#{j}: 服务器拒绝 {out.get('err')}", flush=True); continue
        e = np.hypot(out["x"] - x_gt, out["y"] - y_gt)
        ye = abs((out["yaw_deg"] - yaw_gt + 180) % 360 - 180)
        errs.append((e, ye, out["accepted"], out["elapsed_s"]))
        used += 1
        print(f"  f#{j}: init({rr:.2f}m,{abs(dyaw):.0f}°) → 误差{e*100:.1f}cm/{ye:.2f}° "
              f"{'OK' if out['accepted'] else 'REJECT'} {out['elapsed_s']}s", flush=True)
        if used >= n_frames:
            break
    if errs:
        E = np.array([e[0] for e in errs]); Y = np.array([e[1] for e in errs])
        acc = np.mean([e[2] for e in errs]); el = np.mean([e[3] for e in errs])
        print(f"\n[selftest] n={len(errs)} 水平中位{np.median(E)*100:.1f}cm p90={np.percentile(E,90)*100:.1f} "
              f"yaw中位{np.median(Y):.2f}° 接受率{acc:.0%} 平均{el:.0f}s/次")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://10.0.0.20:8321")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--x", type=float, default=0.0); ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch_deg", type=float, default=0.0)
    ap.add_argument("--cam_h", type=float, default=1.2, help="相机离地高(真实米)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw", default="data/lab/110744")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    if args.selftest:
        selftest(args.server, args.raw, args.frames)
        return
    if args.npz:
        with open(args.npz, "rb") as f:
            nb = f.read()
    else:
        nb = npz_bytes_from_pc(live_grab())
    out = post_npz(args.server, nb, args.x, args.y, args.yaw, args.pitch_deg, args.cam_h)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if out.get("ok") and out.get("accepted"):
        print(f"\n纠正量: Δx={out['dx']:+.3f}m Δy={out['dy']:+.3f}m Δyaw={out['dyaw_deg']:+.1f}°")
        print(f"纠正后位姿(真实米制): x={out['x']:.3f} y={out['y']:.3f} yaw={out['yaw_deg']:.1f}°")


if __name__ == "__main__":
    main()
