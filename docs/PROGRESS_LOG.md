# 进展日志（PROGRESS LOG）

> 本文件每次工作后追加一条，倒序排列（最新在最上）。
> 格式固定：**做了什么 / 产出 / 结论 / 下一步 / 卡点**。
> 全过程叙事见 [`RECONSTRUCTION_JOURNEY.md`]，本文件只记“每次”的增量。

---

## 2026-06-20（周六，下午）—— 汇报 PPT 视觉素材

**做了什么**
- PPT 文档 `docs/汇报_0619_PPT要点.md`:加 5 部分浓缩结构(① 巡检场景 ② 真实采数 ③ 仿真环境 ④ 双脑架构 ⑤ 闭环展望);P5 补完整 real2sim 流水线;P8 改"照片级仿真已落地"。
- 架构图 `scripts/draw_arch.py` → `outputs/arch_diagram.png`:DiffusionDrive 图(a)风格,重点双脑,可直接用(字体/框线不重叠、无"自行插入"提示、不标创新点;三路输入用箭头颜色区分)。
- 轨迹叠图 `outputs/traj_overlay_114830.png`:在 `data/114830/.../1781149750414963.jpg` 上画多模路线(蓝/青/黄,只要路线、去掉红椭圆/红箭头),贴架构图"多模轨迹"框。
- 修 mp4 不可播:`pip install imageio-ffmpeg` → H.264/yuv420p/faststart;`sim_walk_discoverse.py` 已改。

**坑**:matplotlib 中文用 `Noto Sans CJK JP`;cv2 mp4v 浏览器播不了,要 libx264。

---

## 2026-06-20（周六）—— 打通 VGGT-Ω→3DGS→真实仿真器(能走进去)

**做了什么**(链路三段全跑通,见 plan `starry-twirling-ripple.md`)
- **A. VGGT-Ω→3DGS**:`scripts/train_3dgs_vggt.py`(改自 train_3dgs.py):读 `cameras.npz`(world2cam+K,对应688×384)、`recon.ply`(未对齐,与cameras同帧)降采样40万初始化、`frames_zed` resize到688×384监督 → 训出 `outputs/gs_vggto_114830.ply`(63.5万高斯,loss0.06)。自检从训练位姿渲=照片级室内走廊。
- **对齐**:`scripts/align_gs_world.py`:复用 bev_topdown 重力三件套 + `gaussian_renderer.transform_gs_model.transform_gaussian`,把归一化ply烘焙成**世界帧**(重力z-up + 用相机离地~1.2m锚定米制 k=9.92 + 地面落z=0)→ `gs_vggto_114830_world.ply` + `sim_world_114830.npz`(轨迹waypoints,跨度13×34.5m)。
- **B. 真实DISCOVERSE仿真器**:`scripts/sim_walk_common.py`(WalkEnv:动态MJCF=background体+差速机器人jx/jy/jyaw+车载fpv相机+地面+重力;headless EGL)、`scripts/sim_walk_discoverse.py`:机器人沿真实轨迹(去程,返程3DGS前视欠观测会糊)穿行 → `outputs/sim_walk_114830.mp4`(180帧照片级走廊穿行)。
- **C. 浏览器亲手驱动**:`scripts/sim_walk_server.py`:MJPEG流+stdlib http.server(无websocket依赖),`/stream`持续step仿真渲帧推流、`/cmd`收WASD设(v,ω)。实测44帧/2s(~22fps),机器人物理步进、位姿推进、实时画面=走廊FPV。

**关键坑/解法**
- MuJoCo headless EGL 报 `PLATFORM_DEVICE`:必须在 import mujoco/OpenGL **之前**设 `MUJOCO_GL=egl`+`MUJOCO_EGL_DEVICE_ID=0`+`PYOPENGL_PLATFORM=egl`(已写进 sim_walk_common.py 顶部)。
- 坐标系一致性:gs训练必须用 recon.ply(未对齐)配 cameras.npz,**不能用 recon_aligned.ply**(被重力旋转过,与外参不同帧)。
- GSRendererMuJoCo 只按 body pos/quat 摆高斯**不缩放**→ 米制缩放必须烘焙进ply。
- imageio无ffmpeg→视频用 cv2.VideoWriter(mp4v)。

**怎么用(用户亲手走)**
```
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
PYTHONPATH=third_party/discoverse ~/discoverse_venv/bin/python scripts/sim_walk_server.py 114830 --port 8000
# laptop: ssh -L 8000:localhost:8000 ...@10.0.0.20 ,浏览器开 http://localhost:8000 ,WASD 开机器人走
```

**下一步**:更密帧VGGT-Ω提画质;返程糊→可只发布去程或补反向帧重训;接 DiffusionDrive/VLM 把仿真器当训练台。

---

## 2026-06-19（周五）—— 0619汇报PPT要点 + DISCOVERSE 导入自有3DGS资产跑通

**做了什么**
- 写 0619 向王老师汇报的 **PPT 要点**（`docs/汇报_0619_PPT要点.md`）。用户两次修正定调：**主线 = 「真实巡检场景 → (A)任务驱动双层架构 + (B)真实数据仿真环境」的因果逻辑**；采数据这段讲"场景刻画驱动设计"，**不是**讲"缺了哪些数据"（缺数据弯路挪到备份页）。P4 是核心页。
- 复盘分析了全部 6 个 session 的真实结构（见下"数据事实"）。
- **任务2：试用"上一代平台" DISCOVERSE 导入我们自己的资产**——确认 GS-Playground README 明写其上一代 = DISCOVERSE；clone（GitHub 直连被墙，用 `https://ghfast.top/` 镜像）；建 venv（`~/discoverse_venv`，`--system-site-packages` 复用系统 torch2.5.1+cu124）；装 `gaussian_renderer 0.2.0 + mujoco 3.9.0`。
- 写两个脚本把我们的 `gs_111450.ply` 喂进 DISCOVERSE 渲染栈（headless，纯 gsplat 不碰 MuJoCo/显示）：
  - `scripts/discoverse_render_gs.py`：环绕视角 → **噪点**（印证记忆#14：3DGS 只在训练视角附近有效）。
  - `scripts/discoverse_render_drivethrough.py`：**用真实 COLMAP 训练位姿** → **照片级穿行帧**（路面/黄标线/建筑/行人清晰）。

**产出**（`outputs/discoverse_gs_111450/`）
- `drivethrough.gif`（40帧）+ `drivethrough_sheet.png`（3×3实景拼图，**汇报可用**）、`contact_sheet.png`（环绕噪点，反面对照）。

**结论**
- ✅ **我们能把自己重建的场景当资产放进上一代 DISCOVERSE 渲染器并 headless 渲出照片级帧**——这就是"把自己东西放进去变资产"的可行性验证（GS-Playground 的端到端导入仍未开源，DISCOVERSE 这条先通）。
- `gaussian_renderer.core.batch_render(gs, cam_pos, cam_xmat, H, W, fovy, y_up)` 是纯函数渲染接口；约定：`Tmat=[cam_xmat|cam_pos]`=cam→world，`viewmats=inv(Tmat)`。要对上 gsplat/COLMAP OpenCV world2cam：**`cam_pos=cn=(c-center)/s`、`cam_xmat=R.T`、`y_up=False`**。
- DISCOVERSE 包 PEP660 装不了 editable → 不装它本体，**PYTHONPATH 指仓库根 + 只装运行依赖**即可（纯 Python 包）。

**下一步（明天）**
- 可选：把场景接进 DISCOVERSE 的 MuJoCo Simulator（建名为 `background` 的 body + `gs_model_dict={"background":绝对路径ply}` + headless EGL），放机器人/相机做交互。
- 用更干净的 114830 / VGGT-Ω 结果重复同一导入。

---

## 2026-06-18（周四）—— VGGT-Ω 跑通 + 调研 GS-Playground

**做了什么**
- 解决昨天的卡点：拿到 VGGT-Ω 的 HF gated 权重（`vggt_omega_1b_512.pt`，4.5GB）。
- 在 session **114830** 上跑通 VGGT-Ω 重建：图像 → 点云 + 相机位姿 → 重力轴对齐 → BEV 俯视。
- 写了三个脚本：`run_vggto.py`（重建）、`bev_topdown.py`（出 BEV）、`render_views.py`（出多视角 3D 图）。
- 调研真实感仿真框架 **GS-Playground**（清华 DISCOVERSE 团队，RSS 2026）：clone + 建好 venv，torch/motrixsim 可 import。

**产出**（`outputs/vggto_114830/`）
- `recon.ply` / `recon_aligned.ply`（重力对齐后点云）、`cameras.npz`、`recon.glb`
- `bev_topdown_rgb.png`、`bev_topdown_height.png`、`bev_occupancy.png`、`views_3d.png`
- 效果：连贯的走廊俯视，**显著干净于**之前的 COLMAP / octomap 结果。

**结论**
- VGGT-Ω 这条路对**单段室内走廊**有效，是目前最干净的重建，可作为汇报主线。
- GS-Playground 是**软件框架**（物理引擎 MotrixSim + 自研 3DGS 渲染器 GaussianRenderer），不是单独 app；自带的是机械臂/四足预制场景。
- 想用它跑**我们自己的数据**，需走 `gs-real2sim` 把采集重建成 3DGS 资产再导入，而“资产打包导入”官方**未开源** → 它是**后续方向**，非当前可交付。
- 本机 headless（无 `DISPLAY`），它的实时窗口 demo 无法直接弹窗，需 Xvfb 或离屏渲染才能看。

**模型架构讨论（已落文档）**
- 确定双层双脑架构，写入 [`MODEL_ARCHITECTURE.md`] v0.1：
  - 上层 VLM(Qwen3-3B) 场景识别 → **离散任务 id**；下层 **DiffusionDrive** 任务条件轨迹。
  - 核心 framing = 把 DiffusionDrive 从车端 AD 迁移到任务驱动型机器人导航；创新点 = VLM 任务 id 注入级联解码器。
  - 4 迁移难点 + 两阶段训练 + Orin 部署约束，详见该文档 §9 待办表。

**下一步（待定）**
- 等用户从**工控机**拷数据过来 → 填架构里 🟡 待定项（底盘/控制接口/传感器/Orin型号/任务标签集/已有软件栈）。
- 备选：重跑更密帧 VGGT-Ω（~150 帧/2s, 约 5-10 分钟）；仿真线 GS-Playground 仍暂停。

**卡点 / 风险**
- GS-Playground Real2Sim 导入未开源；headless 渲染未验证。
- 明天（06-19 周五）向王老师汇报 → 主线用 VGGT-Ω 结果，GS-Playground 仅作一页“后续方向”。

---

## 之前里程碑（简记，详见 RECONSTRUCTION_JOURNEY.md）

- **06-17（周三）**：决定从 COLMAP/octomap 路线转向 **VGGT-Ω + DISCOVERSE 3D 路线**；d405 是手臂相机弃用，场景用 zed；环境/runner/抽帧就绪，卡在 HF gated 权重。
- **06-16（周二）**：ROS2/RoboStack 装 octomap 对比，从前视数据出占据图“一坨”，证伪“换标准工具就干净”；COLMAP CPU 匹配崩溃修复（`OPENBLAS_NUM_THREADS=1`+brute force）。
- **06-15（周日）**：重建实战发现——只有 142522/114830 适合做室内地图；111450 无里程计只能轨迹回放；朝向须 gyro 投影到重力轴；distance 传感器无效。早期用 **gsplat** 训过 3DGS（`gs_111450.ply`）。
- **06-14**：搭 `robot_bev_sim` 脚手架；发现真实数据是 ROS2/RealSense D405（rosbag+JPG+imu），与文档假设的逐帧 PNG 完全不同。
