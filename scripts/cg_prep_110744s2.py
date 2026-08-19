"""把 110744s2 会话(510帧 COLMAP位姿+ZED度量深度)转成 ConceptGraphs Replica 格式输入。
产出 outputs/cg_lab110744s2/lab1/results/frame*.jpg + depth*.png + traj.txt + dataconfig yaml。
位姿: cameras.npz extrinsic=world2cam(COLMAP/OpenCV约定) → c2w=逆 → traj.txt(每行16数)。
深度: pointcloud npz xyz[:,:,2] (ZED相机系z前向, mm) → uint16 PNG, png_depth_scale=1000。
用法: ~/micromamba/envs/cg/bin/python scripts/cg_prep_110744s2.py
"""
import os, shutil
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS = os.path.join(ROOT, "outputs", "vggto_colmapjoint_110744s2")
PCD = os.path.join(ROOT, "data", "110744", "pointcloud", "zed")
OUT = os.path.join(ROOT, "outputs", "cg_lab110744s2", "lab1")
RESULTS = os.path.join(OUT, "results")
# COLMAP任意尺度 → 米制world 的刚体变换(与align_gs_world同式, PREUNTWIST版与c2w严格一致1e-7)
SIMWORLD = os.path.join(ROOT, "outputs", "sim_world_colmapjoint_110744s2_PREUNTWIST.npz")


def main():
    cam = np.load(os.path.join(SESS, "cameras.npz"))
    w2c_all, K = cam["extrinsic"], cam["intrinsic"][0]
    names = sorted(os.listdir(os.path.join(SESS, "images")))
    assert len(names) == len(w2c_all), (len(names), len(w2c_all))

    sw = np.load(SIMWORLD)
    k_m = float(sw["k"]); Rz = sw["Rz"]; t_w = sw["t_world"]
    center = sw["center"]; s_norm = float(sw["s_norm"])
    print(f"[cg_prep] 米制变换 k={k_m:.3f} s_norm={s_norm:.3f} (PREUNTWIST刚体)")

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(RESULTS)

    traj = []
    n_missing = 0
    for i, name in enumerate(names):
        ts = name.split("_")[1].split(".")[0]
        npz = os.path.join(PCD, ts + ".npz")
        if not os.path.exists(npz):
            n_missing += 1
            continue
        idx = len(traj)
        # color
        shutil.copy(os.path.join(SESS, "images", name),
                    os.path.join(RESULTS, f"frame{idx:04d}.jpg"))
        # depth: z in mm -> uint16 png
        xyz = np.load(npz)["xyzrgba"].astype(np.float32)
        z = xyz[:, :, 2]
        z[~np.isfinite(z)] = 0.0
        z = np.clip(z, 0, 65000).astype(np.uint16)
        Image.fromarray(z).save(os.path.join(RESULTS, f"depth{idx:04d}.png"))
        # pose: world2cam -> c2w -> 米制world (t: k*(Rz@((t_c-center)/s_norm))+t_w, R: Rz@R_c)
        w2c = np.eye(4)
        w2c[:3] = w2c_all[i]
        c2w = np.linalg.inv(w2c)
        R_c, t_c = c2w[:3, :3], c2w[:3, 3]
        c2w_m = np.eye(4)
        c2w_m[:3, :3] = Rz @ R_c
        c2w_m[:3, 3] = k_m * (Rz @ ((t_c - center) / s_norm)) + t_w
        traj.append(" ".join(f"{v:.8f}" for v in c2w_m.flatten()))

    with open(os.path.join(OUT, "traj.txt"), "w") as f:
        f.write("\n".join(traj) + "\n")
    print(f"[cg_prep] {len(traj)}帧写出, 缺深度{n_missing}帧, K0 fx={K[0,0]:.2f} cx={K[0,2]} cy={K[1,2]}")

    yaml = f"""dataset_name: 'replica'
camera_params:
  image_height: 720
  image_width: 1280
  fx: {K[0,0]:.4f}
  fy: {K[1,1]:.4f}
  cx: {K[0,2]:.4f}
  cy: {K[1,2]:.4f}
  png_depth_scale: 1000.0
  crop_edge: 0
"""
    cfg_dir = os.path.join(ROOT, "third_party", "concept-graphs", "conceptgraph",
                           "dataset", "dataconfigs", "lab")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "lab110744s2.yaml"), "w") as f:
        f.write(yaml)
    print(f"[cg_prep] dataconfig -> {cfg_dir}/lab110744s2.yaml")


if __name__ == "__main__":
    main()
