# 本地仿真器安装与运行（笔记本，带显示器 + NVIDIA 显卡）

在自己的电脑上弹出一个桌面窗口，**亲手开着机器人在真实重建的走廊里走**（照片级 3DGS 场景 + MuJoCo 物理）。
重活（VGGT 重建、训 3DGS、以后训模型）在服务器；这里只是把训好的**场景资产**拷下来，本地交互。

---

## 0. 前提

- **NVIDIA 显卡 + 驱动**（3DGS 渲染靠 `gsplat`，必须 CUDA；Mac/核显/A 卡跑不了照片级版）。
- 有显示器（这就是你要的"能看、能操作"）。
- 系统 Linux 或 Windows 均可；Python **3.10**。
- 磁盘：依赖约 3–4 G + 场景资产约 70 M。

---

## 1. 拿代码

```bash
git clone https://github.com/LIUYING527/ROBOT-BEV.git robot_bev_sim
cd robot_bev_sim
```
> 仓库里**不含** `third_party/`、`data/`、`outputs/` 大文件（已 .gitignore）。下面分别补上。

## 2. Python 环境 + 依赖

```bash
python3.10 -m venv .venv
# Linux: source .venv/bin/activate     Windows: .venv\Scripts\activate

# (a) torch (CUDA 12.4 版)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# (b) gsplat —— 用预编译 wheel,免本机 nvcc
pip install gsplat==1.5.3 -f https://docs.gsplat.studio/whl/pt25cu124.html

# (c) 渲染器(带 mujoco extra,会一并装 mujoco/glfw) + 窗口/IO
pip install "gaussian_renderer[mujoco]==0.2.0" pygame opencv-python screeninfo scipy trimesh plyfile
```

## 3. 补 DISCOVERSE 源码（提供 `discoverse` 包，走 PYTHONPATH，不用 pip 装本体）

```bash
git clone https://github.com/discoverse-dev/discoverse third_party/discoverse
# 若 GitHub 直连慢/被墙,用镜像:
# git clone https://ghfast.top/https://github.com/discoverse-dev/discoverse third_party/discoverse
```

## 4. 拷场景资产（从服务器，仅两个文件）

```bash
mkdir -p outputs
scp <你>@10.0.0.20:/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/gs_vggto_114830c_world.ply outputs/
scp <你>@10.0.0.20:/data/DongBaorong/BEVLOOK/robot_bev_sim/outputs/sim_world_114830c.npz       outputs/
```
> `gs_vggto_114830c_world.ply`(~70M) = 去人版走廊的 3DGS 场景；`sim_world_114830c.npz` = 起点/轨迹。
> 想换别的场景，把对应 session 的这两个文件拷过来即可（命名 `gs_vggto_<s>_world.ply` + `sim_world_<s>.npz`）。

## 5. 跑！

```bash
# Linux
PYTHONPATH=third_party/discoverse python scripts/sim_walk_viewer.py 114830c
# Windows (PowerShell): $env:PYTHONPATH="third_party/discoverse"; python scripts/sim_walk_viewer.py 114830c
```

弹出窗口后：

| 键 | 作用 |
|---|---|
| **W / S** | 前进 / 后退 |
| **A / D** | 左转 / 右转 |
| **Shift** | 加速 |
| **R** | 回到起点 |
| **Esc / 关窗** | 退出 |

> 刚开窗若正对一片黑（起点朝向了没重建的方向），按 **A/D** 转一下就进走廊了。

---

## 常见问题

- **`Cannot initialize EGL` / 渲染报错**：脚本默认 `MUJOCO_GL=egl`（离屏，NVIDIA 上通用）。若本地不行，改用桌面 GL：
  `MUJOCO_GL=glfw PYTHONPATH=third_party/discoverse python scripts/sim_walk_viewer.py 114830c`
- **`gsplat` 装不上 / 编译报错**：务必用第 2(b) 步的 `-f ...whl/pt25cu124.html` 预编译 wheel；torch 必须是 cu124 版，别让它装 CPU 版。
- **`No module named discoverse`**：第 3 步没做，或没带 `PYTHONPATH=third_party/discoverse`。
- **画面有重影/半透明影子**：3DGS 对走廊强顶灯+反光地面的已知伪影，不是 bug；后续可在服务器端重训时做浮点抑制再出新资产。
- **想要无窗口的视频版**（服务器/无显示器）：用 `scripts/sim_walk_discoverse.py <s>`（出 mp4）或 `scripts/sim_walk_server.py <s>`（浏览器 WASD，SSH 端口转发）。

---

## 分工速记

| 机器 | 干什么 |
|---|---|
| 服务器 A100 | VGGT-Ω 重建、训 3DGS、以后训 DiffusionDrive 快脑 + VLM 慢脑 |
| 你的 N 卡笔记本 | 跑本仿真器亲手走、看效果、采轨迹、做 demo |

代码两边一致（都是 `sim_walk_common.py`）；服务器无头 EGL、笔记本桌面窗口，区别只是"屏幕在哪"。
