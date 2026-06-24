"""组装两段采集(114830c + 113628d)到统一坐标系的相机集,供训练一个统一3DGS。
114830c 世界帧为参考;113628d 用配准T搬过来。输出 outputs/vggto_unified/{cameras.npz, frames_zed/, recon.ply}。
含 sanity: 把合并点云投到组装的相机,和该帧真实RGB并排,确认对得上。
用法: ~/discoverse_venv/bin/python scripts/build_unified.py
"""
import os, glob, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def metric_cam2world(extr_i, sw):
    """VGGT-orig world2cam -> 该session米制世界帧的 cam->world (4x4)。"""
    R = extr_i[:3, :3]; t = extr_i[:3, 3]; c = -R.T @ t
    c_m = sw["k"] * (sw["Rz"] @ ((c - sw["center"]) / sw["s_norm"])) + sw["t_world"]
    M = np.eye(4); M[:3, :3] = sw["Rz"] @ R.T; M[:3, 3] = c_m
    return M


def session_frames(s, n):
    fz = sorted(glob.glob(os.path.join(ROOT, "outputs", f"vggto_{s}", "frames_zed", "*")))
    if len(fz) == n:
        return fz
    idx = np.linspace(0, len(fz) - 1, n).astype(int)
    return [fz[i] for i in idx]


def main():
    import cv2
    udir = os.path.join(ROOT, "outputs", "vggto_unified")
    fzdir = os.path.join(udir, "frames_zed")
    os.makedirs(fzdir, exist_ok=True)
    for f in glob.glob(fzdir + "/*"):
        os.remove(f)

    T = np.load(os.path.join(ROOT, "outputs", "_reg_T_113628d_to_114830c.npy"))   # 113628d->114830c
    extr_all, intr_all, frame_src = [], [], []
    k = 0
    for s, Tref in [("114830c", np.eye(4)), ("113628d", T)]:
        cams = np.load(os.path.join(ROOT, "outputs", f"vggto_{s}", "cameras.npz"))
        extr, intr = cams["extrinsic"], cams["intrinsic"]
        sw = dict(np.load(os.path.join(ROOT, "outputs", f"sim_world_{s}.npz")))
        frames = session_frames(s, len(extr))
        for i in range(len(extr)):
            M = Tref @ metric_cam2world(extr[i], sw)          # cam->world(统一帧)
            w2c = np.linalg.inv(M)
            extr_all.append(w2c[:3, :4].astype(np.float32))
            intr_all.append(intr[i].astype(np.float32))
            dst = os.path.join(fzdir, f"{k:04d}_{os.path.basename(frames[i])}")
            os.symlink(os.path.abspath(frames[i]), dst)
            frame_src.append(frames[i]); k += 1
    extr_all = np.stack(extr_all); intr_all = np.stack(intr_all)
    np.savez(os.path.join(udir, "cameras.npz"), extrinsic=extr_all, intrinsic=intr_all)
    img = os.path.join(udir, "images")
    if os.path.islink(img) or os.path.exists(img):
        os.remove(img) if os.path.islink(img) else shutil.rmtree(img)
    os.symlink(os.path.abspath(fzdir), img)
    # init 点云 = 合并云
    shutil.copy(os.path.join(ROOT, "outputs", "corridor_merged.ply"), os.path.join(udir, "recon.ply"))
    print(f"[unified] {k} 帧 (114830c + 113628d) -> {udir}")

    # sanity: 投影合并点云到组装相机, 和真实RGB并排
    import open3d as o3d
    P = np.asarray(o3d.io.read_point_cloud(os.path.join(udir, "recon.ply")).points)
    Pc = np.asarray(o3d.io.read_point_cloud(os.path.join(udir, "recon.ply")).colors)
    rows = []
    for ci in [0, 30, len(extr_all) // 2, len(extr_all) // 2 + 30]:
        K = intr_all[ci]; W = int(K[0, 2] * 2); H = int(K[1, 2] * 2)
        Rt = extr_all[ci]; Pc_cam = (Rt[:3, :3] @ P.T).T + Rt[:3, 3]
        z = Pc_cam[:, 2]; u = Pc_cam[:, 0] / z * K[0, 0] + K[0, 2]; v = Pc_cam[:, 1] / z * K[1, 1] + K[1, 2]
        m = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        proj = np.zeros((H, W, 3), np.uint8)
        o = np.argsort(-z[m]); proj[v[m][o].astype(int), u[m][o].astype(int)] = (Pc[m][o][:, ::-1] * 255).astype(np.uint8)
        gt = cv2.resize(cv2.imread(frame_src[ci]), (W, H))
        rows.append(np.hstack([cv2.resize(proj, (W, H)), gt]))
    cv2.imwrite(os.path.join(ROOT, "outputs", "_unified_sanity.png"), np.vstack([cv2.resize(r, (760, 214)) for r in rows]))
    print("[unified] sanity -> _unified_sanity.png (左投影/右真实RGB, 对得上=相机正确)")


if __name__ == "__main__":
    main()
