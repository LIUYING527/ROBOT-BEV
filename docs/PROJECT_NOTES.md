# 机器人巡检项目 — 测试环境搭建与任务导航框架

> **项目背景**：基于现有 RGB-D 采集数据，搭建 BEV 仿真测试环境，并设计基于任务的双层导航框架（Rule + VLM → DiffusionDrive）。
>
> **下次汇报**：周二（向学长 VaE 汇报）；周三技术链路验证会议；周五向王老师团队汇报阶段性成果。

---

## 目录

- [一、项目目标与上下文](#一项目目标与上下文)
- [二、整体技术方案](#二整体技术方案)
- [三、任务拆分与模型设计](#三任务拆分与模型设计)
- [四、BEV 测试环境搭建（核心工作量）](#四bev-测试环境搭建核心工作量)
- [五、待确认事项（阻塞点）](#五待确认事项阻塞点)
- [六、项目目录结构](#六项目目录结构)
- [七、代码骨架](#七代码骨架)
- [八、周二汇报清单](#八周二汇报清单)
- [九、参考资料](#九参考资料)
- [十、时间表与 TODO](#十时间表与-todo)

---

## 一、项目目标与上下文

### 1.1 学长（VaE）的核心指令

来自语音与文字消息的整理：

1. **基于现有数据搭建 BEV 仿真测试环境**
   - 输入数据：RGB 视频（逐帧 PNG）、深度图、避障传感器数据
   - 形式：俯视 2D 平面图，标注**线（护栏、车道线、黄线）、墙、目标点**
   - 流程：先用 RGB-D 粗建 → 再用传感器距离数据精调
   - 车体简化为**箭头**（无机械臂），状态量 `(x, y, θ)`
   - 测试关注：箭头的**朝向变化**与**位移轨迹**

2. **设计基于任务的导航框架**
   - 架构：`Rule + VLM`（上层）→ `DiffusionDrive`（下层）
   - 详细区分任务：进入什么环境 → 执行什么任务
   - 这是论文创新点所在

3. **数据标注（最后做）**
   - **设计好框架再去标，不然浪费时间**
   - 标注 schema 取决于任务分类表

### 1.2 学长原话摘录（重要约束）

> "这个得抓紧做，没有什么创新点上的问题，就是工作量。"
> → 测试环境部分**不要纠结技术深度**，怼工作量、快速交付。

> "周二前给我汇报一次：一测试环境的搭建，二基于任务的导航框架怎么实现，然后再是标数据。"

> "设计好了再去标，不然浪费时间。"

### 1.3 学长推荐阅读（待补充要点）

- 《端到端 VLN 技术布局图，数据瓶颈藏不住了》
- 《从指令看端到端 VLN：细碎指令没意义》

阅读后需要回答：**我们的框架如何回应这两篇文章指出的痛点？**
（猜测：用任务级标签代替 step-level 指令；用仿真增强缓解数据瓶颈。）

---

## 二、整体技术方案

### 2.1 双层模型架构

```
┌─────────────────────────────────────────────┐
│  上层：任务识别层（Rule + VLM）              │
│  - Rule：硬触发（黄线、边界、固定间隔观测）  │
│  - VLM：语义识别（电箱、护栏、转向点）       │
│  输出：任务 token / sub-goal                 │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  下层：轨迹生成层（DiffusionDrive）          │
│  - 条件：上层任务 token + 当前观测           │
│  - 输出：未来 N 步轨迹 (x, y, θ)             │
│  - 评分：仿真 rollout → 多轨迹评分筛选       │
└─────────────────────────────────────────────┘
```

### 2.2 数据流水线

```
真实采集数据 (RGB + Depth + 避障传感器)
         │
         ▼
 RGB-D 反投影 → 3D 点云
         │
         ▼
 多帧拼接（位姿融合）
         │
         ▼
 Z 轴投影 → BEV 占据栅格
         │
         ▼
 传感器数据校准（精修几何）
         │
         ▼
 2D 仿真环境 (箭头 + 地图)
         │
         ▼
 模型训练 / 评估
```

---

## 三、任务拆分与模型设计

### 3.1 任务分类表（核心产出，周二要展示）

| 环境场景 | 任务类型 | 触发条件 | 期望行为 |
|---------|---------|---------|---------|
| 直行通道（有护栏） | 沿护栏巡航 | 检测到两侧护栏 | 保持中线 + 周期观测 |
| 接近电箱 | 定点观测 | VLM 识别电箱 | 停车 + 朝向调整 |
| 黄线区域 | 阶段切换 | Rule 检测黄线 | 触发横向移动 |
| 路口/转角 | 90° 转向 | VLM 识别转向特征 | 触发转向动作 |
| 顶部边界 | 回程 | Rule 检测边界 | 调头 |

> 这张表是后续标注 schema 和 Rule 设计的基础，**必须先定下来**。

### 3.2 视觉特征提取策略

- **连续片段识别**：用 10 秒（或更短，需压缩）连续视觉片段作为特征基准，提升辨识度
  - ⚠️ 注意：30fps × 10s = 300 帧，维度爆炸，需要时序压缩（Video-MAE / TimeSformer）或关键帧采样
- **空间先验**：基于点云重建 3D 环境，结合护栏连续性等先验优化观测点

### 3.3 动态观测策略

- 默认按护栏间隔 2 米固定观测频率
- 遇障碍物自动跳过观测
- **改进方向**（暂不实现）：基于 VLM 置信度的不确定性驱动观测

### 3.4 已知技术风险

| 风险点 | 缓解方案 |
|-------|---------|
| 模仿学习模式坍塌 | 引入 Diffusion Policy / ACT 建模多模态分布 |
| Sim-to-Real Gap | 仿真评分加保守机制；传感器数据校准几何 |
| 硬编码先验脆弱 | 将规则作为辅助监督信号，而非执行逻辑 |
| 数据量不足 | 仿真环境数据增强预训练 |

---

## 四、BEV 测试环境搭建（核心工作量）

### 4.1 数据现状

- **数据格式**：逐帧 PNG
  - `data/color/000000.png, 000001.png, ...`
  - `data/depth/000000.png, 000001.png, ...`
- **其余信息待确认**（见第五节）

### 4.2 技术路径

**阶段 1**：单帧 RGB-D → 点云（验证 pipeline）
**阶段 2**：多帧拼接成完整场地点云
**阶段 3**：点云沿 Z 轴投影 → BEV 占据栅格
**阶段 4**：（可选）用避障传感器数据校准几何
**阶段 5**：Pygame 渲染 + 箭头智能体 + 运动学

### 4.3 关键技术参数（参考值，需根据实际数据调整）

```python
# BEV 地图参数
RESOLUTION = 0.05        # 每格 5cm
X_RANGE = (-10, 10)      # 米
Y_RANGE = (-10, 10)
Z_RANGE = (0.1, 1.5)     # 只保留车体高度范围内的点

# 深度图参数
DEPTH_SCALE = 1000.0     # 毫米→米
DEPTH_TRUNC = 10.0       # 超过 10 米丢弃

# 运动学（差速模型）
DT = 0.1                 # 仿真步长
```

### 4.4 简化决策（节省工作量）

| 项目 | 决策 |
|-----|------|
| 仿真引擎 | Pygame（不上 Gazebo / Isaac Sim） |
| 物理模型 | 单车 / 差速模型，不做物理仿真 |
| 碰撞检测 | 第一版可跳过 |
| 传感器校准 | 第一版可跳过，预留接口 |
| 视觉观测 | BEV 局部切片（车前方扇形/矩形） |

---

## 五、待确认事项（阻塞点）—— ✅ 数据已到并探明（2026-06-14）

> ✅ **数据已传到 GPU 服务器并彻底探明。原假设（逐帧 PNG）全错，实际是多传感器 ROS2 采集。**
> 详见 `工作总结与技术详解.md`。原阻塞项更新如下：

### 5.1 数据相关

- [x] **深度图格式**：⚠️ depth/*.jpg 是**伪彩有损可视化（3通道JPEG）不可用**；真 3D 在 ZED 点云 `.npz`（`xyzrgba` float16，单位毫米）
- [x] **RGB-深度对应**：时间戳（纳秒）命名，需按时间戳对齐；ZED 系一致 1280×720
- [x] **总帧数**：每 session ZED ~1880 帧（~2fps），共 6 个 session
- [x] **采集帧率**：ZED ~2fps；动作/IMU 高频；单 session 约 15.7 分钟
- [x] **采集场地范围**：由动作回放估算约 45×48m（巡检环路）

### 5.2 相机相关

- [x] **相机型号**：**ZED 立体相机 + RealSense D405** 双相机（不是单 RGB-D）
- [x] **相机内参**：ZED 点云已是米坐标，**无需内参反投影**（直接用 npz）
- [x] **图像分辨率**：1280×720

### 5.3 位姿与传感器

- [x] **每帧位姿**：🔴 **全程无位姿/odom/tf/姿态四元数**。已用 `/svtrobot_cmd`(Twist) dead-reckoning 还原轨迹；后续可 ICP/轮速里程计提精度
- [x] **避障传感器数据**：`distance.jsonl`（前/右/后/左距离）+ IMU + 4轮编码器，均按时间戳对齐

### 5.4 行动项

1. ✅ 搭好项目目录骨架
2. ⤳ `01_check_data.py` 作废（数据非 PNG）；改用 sqlite 读 db3 + ZED npz 探测
3. [ ] 读学长推荐的两篇文章（**未做**）
4. [ ] 完善任务分类表（**待用真实场景细化**）

---

## 六、项目目录结构

```
robot_bev_sim/
├── data/                       # 原始数据（或软链接到原始位置）
│   ├── color/                  # RGB 逐帧 PNG
│   ├── depth/                  # 深度逐帧 PNG
│   ├── poses.txt               # （可选）每帧位姿
│   └── sensors/                # （可选）避障传感器数据
│
├── configs/
│   └── camera.py               # 相机内参 / BEV 参数
│
├── scripts/
│   ├── 01_check_data.py        # 数据格式自动探测
│   ├── 02_single_frame_pcd.py  # 单帧深度图 → 点云
│   ├── 03_merge_pcd.py         # 多帧拼接
│   ├── 04_pcd_to_bev.py        # 点云 → BEV 占据栅格
│   └── 05_sim_env.py           # Pygame 箭头仿真
│
├── sim/                        # 仿真环境模块
│   ├── env.py                  # 主环境类（gym 风格接口）
│   ├── robot.py                # 箭头智能体（运动学）
│   └── renderer.py             # Pygame 可视化
│
├── outputs/                    # 中间产物
│   ├── pointcloud.ply
│   ├── bev_map.npy
│   ├── bev_map.png
│   └── data_check.png
│
├── docs/
│   └── PROJECT_NOTES.md        # 本文档
│
├── requirements.txt
└── README.md
```

### 依赖列表（`requirements.txt`）

```
open3d>=0.17
numpy
opencv-python
matplotlib
pygame
scipy
```

---

## 七、代码骨架

### 7.1 数据探测脚本（`scripts/01_check_data.py`）

> **现在就可以跑**，跑完把输出贴出来，决定后续代码细节。

```python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

COLOR_DIR = "data/color"
DEPTH_DIR = "data/depth"

color_files = sorted(os.listdir(COLOR_DIR))
depth_files = sorted(os.listdir(DEPTH_DIR))
print(f"[INFO] Color frames: {len(color_files)}")
print(f"[INFO] Depth frames: {len(depth_files)}")
print(f"[INFO] First color: {color_files[0]}")
print(f"[INFO] First depth: {depth_files[0]}")

color = cv2.imread(os.path.join(COLOR_DIR, color_files[0]), cv2.IMREAD_UNCHANGED)
depth = cv2.imread(os.path.join(DEPTH_DIR, depth_files[0]), cv2.IMREAD_UNCHANGED)

print("\n=== Color ===")
print(f"  shape: {color.shape}, dtype: {color.dtype}")
print(f"  range: [{color.min()}, {color.max()}]")

print("\n=== Depth ===")
print(f"  shape: {depth.shape}, dtype: {depth.dtype}")
print(f"  range: [{depth.min()}, {depth.max()}]")
print(f"  非零像素数: {(depth > 0).sum()}")
if (depth > 0).any():
    print(f"  非零深度均值: {depth[depth>0].mean():.2f}")

os.makedirs("outputs", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
if color.ndim == 3:
    axes[0].imshow(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
else:
    axes[0].imshow(color, cmap='gray')
axes[0].set_title("Color")
axes[1].imshow(depth, cmap='viridis')
axes[1].set_title(f"Depth ({depth.dtype})")
plt.savefig("outputs/data_check.png", dpi=120)
plt.show()
```

### 7.2 单帧点云脚本（`scripts/02_single_frame_pcd.py`）

> **依赖**：相机内参确认后填入。

```python
import open3d as o3d
import numpy as np

# === 相机内参（待替换为真实值） ===
WIDTH, HEIGHT = 640, 480
FX, FY = 525.0, 525.0
CX, CY = 319.5, 239.5

color_raw = o3d.io.read_image("data/color/000000.png")
depth_raw = o3d.io.read_image("data/depth/000000.png")

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw,
    depth_scale=1000.0,
    depth_trunc=10.0,
    convert_rgb_to_intensity=False
)

intrinsic = o3d.camera.PinholeCameraIntrinsic(WIDTH, HEIGHT, FX, FY, CX, CY)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

o3d.io.write_point_cloud("outputs/single_frame.ply", pcd)
o3d.visualization.draw_geometries([pcd])
```

### 7.3 点云 → BEV 脚本（`scripts/04_pcd_to_bev.py`）

```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

RESOLUTION = 0.05
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = -10, 10
Z_MIN, Z_MAX = 0.1, 1.5

pcd = o3d.io.read_point_cloud("outputs/scene.ply")
points = np.asarray(pcd.points)

mask = (points[:, 2] > Z_MIN) & (points[:, 2] < Z_MAX)
points = points[mask]

W = int((X_MAX - X_MIN) / RESOLUTION)
H = int((Y_MAX - Y_MIN) / RESOLUTION)
grid = np.zeros((H, W), dtype=np.uint8)

ix = ((points[:, 0] - X_MIN) / RESOLUTION).astype(int)
iy = ((points[:, 1] - Y_MIN) / RESOLUTION).astype(int)
valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
grid[iy[valid], ix[valid]] = 1

np.save("outputs/bev_map.npy", grid)
plt.imshow(grid, cmap='gray_r', origin='lower')
plt.title("BEV Occupancy Grid")
plt.savefig("outputs/bev_map.png", dpi=150)
plt.show()
```

### 7.4 Pygame 仿真（`scripts/05_sim_env.py`）

```python
import pygame
import numpy as np

grid = np.load("outputs/bev_map.npy")
H, W = grid.shape
SCALE = 4

pygame.init()
screen = pygame.display.set_mode((W * SCALE, H * SCALE))
clock = pygame.time.Clock()

state = {"x": W / 2, "y": H / 2, "theta": 0.0}

def draw_grid():
    surf = pygame.surfarray.make_surface(
        np.stack([(1 - grid.T) * 220] * 3, axis=-1).astype(np.uint8)
    )
    surf = pygame.transform.scale(surf, (W * SCALE, H * SCALE))
    screen.blit(surf, (0, 0))

def draw_arrow(x, y, theta):
    cx, cy = x * SCALE, y * SCALE
    L = 20
    tip = (cx + L * np.cos(theta), cy + L * np.sin(theta))
    left = (cx + 10 * np.cos(theta + 2.5), cy + 10 * np.sin(theta + 2.5))
    right = (cx + 10 * np.cos(theta - 2.5), cy + 10 * np.sin(theta - 2.5))
    pygame.draw.polygon(screen, (255, 0, 0), [tip, left, right])

def step(v, omega, dt=0.1):
    state["theta"] += omega * dt
    state["x"] += v * np.cos(state["theta"]) * dt
    state["y"] += v * np.sin(state["theta"]) * dt

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    keys = pygame.key.get_pressed()
    v, omega = 0, 0
    if keys[pygame.K_UP]:    v = 5
    if keys[pygame.K_DOWN]:  v = -5
    if keys[pygame.K_LEFT]:  omega = -1
    if keys[pygame.K_RIGHT]: omega = 1
    step(v, omega)
    draw_grid()
    draw_arrow(state["x"], state["y"], state["theta"])
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
```

---

## 八、周二汇报清单

### 8.1 必须交付（P0）

- [x] **一张 BEV 地图截图** — ✅ 真实 ZED 点云重建 `outputs/bev_105731_0000.png`（重力对齐+去地面，通道两侧有墙）
- [x] **一段箭头跑动的 demo** — ✅ 两版：① 真实动作回放 `outputs/traj_105731.gif`；② 干净 2D 仿真 `outputs/sim2d_demo.gif`（推荐汇报用，一眼看懂）
- [x] **双层架构图** — ✅ 见本文档第 2.1 节
- [ ] **任务分类表** — 📄 第 3.1 节有初版，**待用真实场景细化**（建议汇报前定稿）

### 8.2 加分项（P1）

- [~] 多帧拼接的完整 BEV 地图 — 已实现 `zed_stitch_icp.py`，**当前会漂移/糊，待改进**
- [x] 干净 2D 仿真测试环境 — ✅ `sim/world2d.py`+`render2d.py`（棋盘地面+规整墙+箭头+分类目标，对齐参考图）
- [ ] 箭头的简单碰撞检测 — 未做（`sim/env.py` 留了 `is_collision` 接口）
- [ ] 标注 schema 设计草案 — 未做
- [ ] 推荐文章的要点摘录 + 我们的应对 — 未做

### 8.3 汇报话术建议（已按真实数据更新）

开场：
> "学长，数据我全摸清了 —— 是 ZED+D405 双相机加 38 个 rosbag 的多传感器采集，比当初设想的复杂。ZED 直接给稠密点云，我重建了 BEV；又发现相机俯视 40° 装的，做了重力对齐才正。全程没有位姿，我从 svtrobot_cmd 的速度指令 dead-reckoning 还原了 940 秒的真实巡检轨迹。"

中段（demo）：
> "箭头 demo 有两版：一版是真实动作回放，箭头按真实采集轨迹跑；另一版我搭了个干净的 2D 仿真测试环境（规整墙、门洞、分类目标、箭头），用来承载基于任务的导航测试。"

结尾（关于框架与标注）：
> "下一步把真实场地的墙提炼进这个仿真环境，再把任务分类表对应到目标语义。标注 schema 先不动，等任务分类表定稿再开标，避免返工。"

---

## 九、参考资料

### 9.1 技术参考

- **Open3D RGBD 教程**：https://www.open3d.org/docs/release/tutorial/geometry/rgbd_image.html
- **Open3D 重建系统**：https://www.open3d.org/docs/release/tutorial/reconstruction_system/index.html
- **BEV 占据栅格示例项目**：https://github.com/harrylal/simulation-of-birds-eye-view-map-generation-from-rgbd-data
- **Pygame 差速驱动机器人**：https://github.com/SurabhiGupta17/DifferentialDriveSim
- **ORB-SLAM3（备选 SLAM 方案）**：https://github.com/UZ-SLAMLab/ORB_SLAM3

### 9.2 论文方向

- **Hierarchical VLA**：OpenVLA、RDT-1B 的分层变体
- **长程任务 sub-goal 自动发现**：LISA、HiP
- **巡检 / 导航 benchmark**：RoboTHOR、Habitat (PointNav / ObjectNav)
- **多模态动作建模**：Diffusion Policy、Action Chunking Transformer (ACT)

### 9.3 待精读

- 端到端 VLN 技术布局图，数据瓶颈藏不住了（学长推荐）
- 从指令看端到端 VLN：细碎指令没意义（学长推荐）

---

## 十、时间表与 TODO

### 10.1 时间表

| 时间 | 任务 |
|-----|-----|
| 周五-周六 | 搭项目骨架；跑 `01_check_data.py`；问学长拿内参/位姿 |
| 周日 | 单帧 → 点云跑通；读两篇推荐文章 |
| 周一 | 多帧拼接 + BEV 生成；Pygame 箭头跑通；准备 slides |
| 周二上午 | 跑一遍完整 demo；汇报 |
| 周三 | 技术链路验证会议 |
| 周四 | 修正问题；细化框架 |
| 周五 | 向王老师团队汇报 |

### 10.2 TODO List（2026-06-14 更新）

**P0 — 阻塞项**（数据已到，全部解决/失效）
- [x] 确认深度图格式 → depth.jpg 伪彩不可用，真 3D 在 ZED npz
- [x] 相机内参 → ZED 点云已是米坐标，无需内参
- [x] RGB-Depth 对应 → 时间戳对齐
- [x] 位姿数据 → 确认无，已用 dead-reckoning 还原

**P0 — 必做**
- [x] 点云 → BEV 跑通（ZED npz 版，含重力对齐+去地面）
- [x] 箭头跑动 demo 跑通（真实回放 + 干净 2D 仿真两版，headless GIF）
- [ ] 定稿任务分类表（待用真实场景细化）

**P1 — 争取**
- [~] 多帧拼接（已实现，待改进漂移）
- [x] 干净 2D 仿真环境（墙+箭头+分类目标）
- [ ] 把真实数据提炼的地图接进 2D 仿真（「两者结合」后半步）
- [ ] 读完两篇推荐文章
- [ ] 标注 schema 草案
- [ ] 准备汇报 slides（5–8 页）

**P2 — 后续**
- [ ] 感知射线（ray-cast 打墙/目标）
- [ ] 4WS 底盘运动学（真实是四轮独立转向，非差速）
- [ ] 位姿提精度（ICP / 轮速里程计 / IMU 融合）
- [ ] VLM 接入接口设计
- [ ] DiffusionDrive 集成方案（动作标签 = svtrobot_cmd Twist）
- [ ] 轨迹评分机制

---

## 附录：关键决策记录

| 日期 | 决策 | 原因 |
|-----|------|-----|
| 初版 | 用 Pygame 而非 Gazebo/Isaac Sim | 学长明确说"只是工作量"，不需要物理仿真 |
| 初版 | 车体简化为箭头 (x, y, θ) | 学长指示，无机械臂 |
| 初版 | BEV 用占据栅格而非矢量地图 | 实现简单，足够支撑模型测试 |
| 初版 | 数据标注延后 | 学长明确指示，等框架定稿 |
| 06-14 | 直接用 ZED 点云 npz，跳过 RGB-D 反投影 | ZED 已输出稠密米坐标点云，省一大步 |
| 06-14 | BEV 前做 RANSAC 重力对齐 + 去地面 | 实测 ZED 相机俯视 40° 安装，不对齐则投影糊成一团 |
| 06-14 | 位姿用 dead-reckoning（svtrobot_cmd 积分） | 数据全程无 odom/tf/姿态，先用动作积分兜底 |
| 06-14 | 仿真测试环境改用**干净矢量地图**（render2d 用 matplotlib 渲染，非 Pygame 窗口） | 服务器 headless 无显示；且用户要"游戏感、能一眼看懂"的环境，而非噪点 BEV |
| 06-14 | 墙的来源"两者结合"：先手工干净地图，后接真实数据 | 先把仿真+目标+路线跑通看效果，再换真实场地底图 |

