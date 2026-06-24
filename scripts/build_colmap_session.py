"""把 COLMAP 联合BA结果(_colmap_joint/sparse 最大模型)导成 vggto 风格训练输入,
供 train_3dgs_vggt 训一个锐利的统一3DGS(紧位姿→不糊;两段→多角度清)。
产出 outputs/vggto_colmapjoint/{cameras.npz(world2cam+K), recon.ply(BA稀疏点), frames_zed/}。
用法: ~/discoverse_venv/bin/python scripts/build_colmap_session.py
"""
import os, glob, shutil, sys
import numpy as np
import pycolmap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""        # 空=_colmap_joint;"dense"=_colmap_joint_dense
    work = os.path.join(ROOT, "outputs", "_colmap_joint" + (("_" + tag) if tag else ""))
    sess = "colmapjoint" + (("_" + tag) if tag else "")
    recs = {int(os.path.basename(d)): d for d in glob.glob(os.path.join(work, "sparse", "*")) if os.path.isdir(d)}
    rec = None; best = -1
    for d in recs.values():
        r = pycolmap.Reconstruction(d)
        if r.num_reg_images() > best:
            best = r.num_reg_images(); rec = r
    print(f"[colmap2gs] {work} 最大模型 注册{rec.num_reg_images()}张 3D点{rec.num_points3D()}")

    out = os.path.join(ROOT, "outputs", f"vggto_{sess}")
    fz = os.path.join(out, "frames_zed")
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(fz)
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    extr, intr = [], []
    for k, im in enumerate(imgs):
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = cfw.rotation.matrix(); t = np.asarray(cfw.translation)
        E = np.zeros((3, 4), np.float32); E[:3, :3] = R; E[:3, 3] = t
        cam = rec.cameras[im.camera_id]
        try:
            K = cam.calibration_matrix().astype(np.float32)
        except Exception:
            p = cam.params  # 假定 SIMPLE_PINHOLE/PINHOLE: f(,fy),cx,cy
            fx = p[0]; fy = p[1] if len(p) > 3 else p[0]
            cx = p[-2]; cy = p[-1]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        extr.append(E); intr.append(K)
        src = os.path.join(work, "images", im.name)
        os.symlink(os.path.realpath(src), os.path.join(fz, f"{k:04d}_{im.name}"))
    np.savez(os.path.join(out, "cameras.npz"), extrinsic=np.stack(extr), intrinsic=np.stack(intr))
    img = os.path.join(out, "images")
    os.symlink(os.path.abspath(fz), img)

    # recon.ply = BA 稀疏点(xyz+rgb), 与cameras同帧, 作3DGS初始化
    pts = rec.points3D
    xyz = np.array([p.xyz for p in pts.values()], np.float32)
    rgb = np.array([p.color for p in pts.values()], np.uint8)
    n = len(xyz)
    hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % n)
    with open(os.path.join(out, "recon.ply"), "wb") as f:
        f.write(hdr.encode())
        for i in range(n):
            f.write(xyz[i].tobytes() + bytes(rgb[i].tolist()))
    print(f"[colmap2gs] -> {out}  ({len(imgs)}帧, {n}初始点, K分辨率~{int(intr[0][0,2]*2)}x{int(intr[0][1,2]*2)})")


if __name__ == "__main__":
    main()
