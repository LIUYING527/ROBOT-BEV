# robot_bev_sim

基于现有 RGB-D 采集数据搭建 BEV 仿真测试环境，并设计基于任务的双层导航框架
（`Rule + VLM` → `DiffusionDrive`）。

> 详细设计、任务分类表、汇报清单见 [`docs/PROJECT_NOTES.md`](docs/PROJECT_NOTES.md)。

## 环境安装

```bash
cd robot_bev_sim
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备

原始 RGB-D 数据在**另一台机器**上，用 `rsync` 拷到本地（断点续传、增量同步，
比 `scp` 更适合上万个小 PNG；拷到本地盘而非 NFS 挂载，保证 Open3D 反投影的 I/O）：

```bash
# 把 <user>@<host>:<远程数据路径> 换成实际值
rsync -avhP --info=progress2 \
    <user>@<host>:/远程/path/to/color/  data/color/
rsync -avhP --info=progress2 \
    <user>@<host>:/远程/path/to/depth/  data/depth/

# （可选）位姿与避障传感器数据
rsync -avhP <user>@<host>:/远程/path/to/poses.txt   data/
rsync -avhP <user>@<host>:/远程/path/to/sensors/    data/sensors/
```

目录约定：

```
data/color/000000.png, 000001.png, ...   # RGB 逐帧
data/depth/000000.png, 000001.png, ...   # 深度逐帧
data/poses.txt                           # （可选）每帧位姿
data/sensors/                            # （可选）避障传感器
```

## 运行流程

```bash
# 1) 数据格式探测（拿到数据后第一时间跑，确认 P0 阻塞项）
python scripts/01_check_data.py

# 2) 单帧 RGB-D → 点云（需先在 configs/camera.py 填入相机内参）
python scripts/02_single_frame_pcd.py 0

# 3) 多帧拼接成完整场地点云（有 poses.txt 用位姿，否则回退 ICP）
python scripts/03_merge_pcd.py 5

# 4) 点云 → BEV 占据栅格（产出 bev_map.npy / bev_map.png）
python scripts/04_pcd_to_bev.py

# 5) Pygame 箭头仿真（方向键控制，展示 x/y/θ 轨迹）
python scripts/05_sim_env.py
```

## ⚠️ 待确认（P0 阻塞项）

跑 `01_check_data.py` 能自动探测的：深度图 dtype/单位、RGB-Depth 分辨率对应、总帧数。
仍需向学长确认的：**相机内参 fx/fy/cx/cy**、是否有 **poses.txt**、避障传感器格式与对齐方式。
详见 `docs/PROJECT_NOTES.md` 第五节。

## 目录结构

```
robot_bev_sim/
├── configs/camera.py       # 相机内参 / BEV 参数（待填真实内参）
├── scripts/01..05          # 数据探测 → 点云 → BEV → 仿真
├── sim/                    # env.py / robot.py / renderer.py
├── data/                  # 原始数据（git 忽略）
├── outputs/               # 中间产物（git 忽略）
└── docs/PROJECT_NOTES.md  # 完整项目文档
```
