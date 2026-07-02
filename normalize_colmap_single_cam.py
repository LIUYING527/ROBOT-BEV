#!/usr/bin/env python
"""把多相机(每图一 camera_id、内参逐图微调)的 COLMAP 模型压成单个共享内参,
以满足 ArtiFixer `prepare_colmap_artifixer_inputs` 的 "one shared intrinsic" 断言。

我们的 _colmap_joint_all 是 114830+113628 两段联合 BA,COLMAP 给每张图各注册了一个
SIMPLE_RADIAL 相机(物理上同一台 ZED 左目,fl_x≈534.86 附近浮动)。这里取所有被使用相机
参数的中位数作为单一共享相机(id=1),所有 image 重映射到该相机。

用法:
  python normalize_colmap_single_cam.py <src_colmap_dir> <dst_colmap_dir>
  # src 需含 images/ 和 sparse/0/{cameras,images,points3D}.bin
"""
import sys
from pathlib import Path

# 复用 artifixer 仓库的 COLMAP IO
AF = Path(__file__).resolve().parent / "third_party/artifixer"
sys.path.insert(0, str(AF))
import numpy as np
import data_processing.prepare_colmap_artifixer_inputs as P


def main() -> None:
    src = Path(sys.argv[1]).expanduser().resolve()
    dst = Path(sys.argv[2]).expanduser().resolve()
    img_dir, sparse_dir = P.resolve_colmap_paths(src)
    scene = P.read_colmap_scene(sparse_dir)

    # 只统计真正被图像使用的相机
    used_ids = sorted({im.camera_id for im in scene.images})
    cam_by_id = {c.id: c for c in scene.cameras}
    used = [cam_by_id[i] for i in used_ids]
    models = {c.model for c in used}
    assert models == {"SIMPLE_RADIAL"}, f"脚本只处理 SIMPLE_RADIAL,实际={models}"
    w = int(used[0].width); h = int(used[0].height)
    assert all(int(c.width) == w and int(c.height) == h for c in used), "图像尺寸不一致"

    params = np.array([c.params for c in used], dtype=np.float64)  # [N,4]=f,cx,cy,k1
    med = np.median(params, axis=0)
    print(f"used cameras={len(used)} model=SIMPLE_RADIAL size={w}x{h}")
    print(f"  f:    median={med[0]:.4f}  min={params[:,0].min():.4f} max={params[:,0].max():.4f}")
    print(f"  cx:   median={med[1]:.4f}  cy: median={med[2]:.4f}")
    print(f"  k1:   median={med[3]:.6f}  min={params[:,3].min():.6f} max={params[:,3].max():.6f}")

    # 3DGRUT 的 ColmapDataset 只吃无畸变相机(PINHOLE/SIMPLE_PINHOLE/OPENCV_FISHEYE)。
    # 我们 k1 极小(median≈0.0048),直接转 PINHOLE 丢 k1,畸变可忽略。
    Camera = type(used[0])  # namedtuple Camera
    f, cx, cy = float(med[0]), float(med[1]), float(med[2])
    shared = Camera(id=1, model="PINHOLE", width=w, height=h,
                    params=np.array([f, f, cx, cy], dtype=np.float64))
    print(f"  -> 共享相机 PINHOLE: fx=fy={f:.4f} cx={cx:.4f} cy={cy:.4f} (丢弃 k1)")
    new_images = [im._replace(camera_id=1) for im in scene.images]
    new_scene = P.ColmapScene(cameras=[shared], images=new_images)

    # 写出
    (dst / "images").mkdir(parents=True, exist_ok=True)
    dst_sparse = dst / "sparse/0"
    dst_sparse.mkdir(parents=True, exist_ok=True)
    P.write_colmap_cameras(dst_sparse / "cameras.bin", new_scene.cameras)
    P.write_colmap_images(dst_sparse / "images.bin", new_scene.images)
    # points3D 不引用 camera_id,直接软链
    p3d = dst_sparse / "points3D.bin"
    if p3d.exists() or p3d.is_symlink():
        p3d.unlink()
    p3d.symlink_to((sparse_dir / "points3D.bin").resolve())
    # 图像软链(basename)
    for im in new_scene.images:
        link = dst / "images" / P.image_basename(im)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(P.source_image_path(img_dir, im).resolve())

    # 自检:断言现在能过 shared_camera_intrinsics
    check = P.read_colmap_scene(dst_sparse)
    intr = P.shared_camera_intrinsics(check)
    print("OK shared intrinsics:", intr)
    print("wrote:", dst)


if __name__ == "__main__":
    main()
