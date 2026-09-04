# 真机部署 README —— 怎么部署上去、怎么启动

> 命令速查版。完整流程、每步预期效果、判据见 `docs/真机部署操作手册_0904.md`。
> 本文件只回答两件事：①哪些代码/资产要搬到哪台机器 ②每个进程怎么启动、怎么确认它活了。

---

## 0. 组件清单总览

| 组件 | 文件 | 跑在哪台机器 | 启动命令 | 健康检查 |
|---|---|---|---|---|
| 机器人系统（底盘/rosbridge/web/ZED） | `robot_control_current/` | Orin（已部署） | `bash src/start_all.sh` | `ros2 topic list` 见 `/svtrobot_cmd` |
| 基线 DiffusionDrive | `robot_orig_code/infer.sh` | Orin（已部署） | `bash infer.sh` | 打印轨迹非 NaN、无 shm 超时 |
| RELOC 服务端 | `scripts/reloc_server.py` | GPU 服务器 | 见 §2.1 | `curl :8321/health` → `frames>0` |
| RELOC 客户端 | `scripts/reloc_client.py` | Orin | 见 §3.1 | 返回纠正位姿 json |
| VLM 服务 | `scripts/vlm_server_local.py` | GPU 服务器 | 见 §2.2 | `curl :8152/health` → `ok` |
| 建图流水线 | `scripts/auto_map_pipeline.sh` | GPU 服务器 | 见 §2.3 | 产物 VERDICT.txt |

---

## 1. 要搬到 Orin 的资产（4 样，都很小）

```bash
# 从服务器把下面 4 样 scp 到 Orin（IP 换成实际值）
# ① 快脑 ckpt（M0 用 R2R 版 / M2 用 lab 版，二选一按模式）
scp robot_bev_sim/goalnav/runs/goalnav_r2r_dense_anchor24.pt  <orin-user>@<orin-ip>:~/ckpt/
scp robot_bev_sim/goalnav/runs/goalnav_gate_lab.pt            <orin-user>@<orin-ip>:~/ckpt/
# ② RELOC 客户端
scp robot_bev_sim/scripts/reloc_client.py                     <orin-user>@<orin-ip>:~/deploy/
# ③ targets 站坐标 json（M2 才需要）
scp robot_bev_sim/outputs/*_targets_*.json                    <orin-user>@<orin-ip>:~/deploy/
# ④ 离线占据栅格 npz（M2 BEV 裁窗用，auto_map_pipeline 产出）
scp robot_bev_sim/outputs/occ_bev_*.npy                       <orin-user>@<orin-ip>:~/deploy/
```

> Orin 上 `reloc_client.py` 只依赖 `numpy + requests`（机器人环境若没 requests：`pip install requests`）。

---

## 2. 服务器端启动（先起服务，Orin 才能连）

### 2.1 RELOC 精定位服务

```bash
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
# 旧 lab 图（110744s2）尺度 0.70× 债 → OBS_SCALE=0.70；新图重 align 后 → 1.0
OBS_SCALE=0.70 CUDA_VISIBLE_DEVICES=4 ~/discoverse_venv/bin/python scripts/reloc_server.py --port 8321
```

- 默认加载地图 `colmapjoint_110744s2`（raw=`data/lab/110744`）；换场景加 `--session <sess> --raw <rawdir>`。
- 接口：`POST /reloc?x=&y=&yaw=&pitch_deg=&cam_h=`，body = npz 字节流（`xyzrgba` 720×1280×4 float16 mm）；`GET /health`。
- 判据：`curl -m5 http://10.0.0.20:8321/health` 返回 `{"ok":true,"frames":<N>,"obs_scale":0.70}`，`N>0`。

### 2.2 VLM 服务（本地 Qwen，自研 10.0.0.18 不在线时用）

```bash
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
CUDA_VISIBLE_DEVICES=3 /home/DongBaorong/micromamba/envs/artifixer/bin/python scripts/vlm_server_local.py --port 8152
```

- 接口：`POST /api/generate`（multipart images + prompt/max_new_tokens/do_sample）；`GET /health`。
- 判据：`curl -m5 http://10.0.0.20:8152/health` 返回 `{"status":"ok","model":"Qwen3.5-4B"}`。

### 2.3 建图流水线（M1 数据 → M2 地图，无人值守）

```bash
cd /data/DongBaorong/BEVLOOK/robot_bev_sim
# ⚠ cam_height 必须卷尺实测（ZED 离地 ~1.71m），脚本拒绝默认值
bash scripts/auto_map_pipeline.sh <session> --cam_height 1.71 --untwist --gpu 4
```

- 输入：`data/<session>/images/zed/*.jpg` + `data/<session>/pointcloud/zed/*.npz`。
- 产出：`gs_vggto_*_world.ply`、`sim_world_*.npz`、`occ_bev_*.npy`、`nav_ds_*/`。
- 判据：`outputs/automap_<session>_VERDICT.txt` 出现 `ALL AUTO STAGES DONE`；中间每阶段 GATE PASS/FAIL 都在里面。

---

## 3. Orin 端启动

### 3.1 机器人系统

```bash
cd ~/svtrobo_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
bash src/start_all.sh          # 底盘控制 → rosbridge → Web 服务（systemd 亦可）
```

### 3.2 基线 DiffusionDrive 干跑（P0 冒烟，只打印不发指令）

```bash
cd ~/robot_orig_code && bash infer.sh    # 已配 --dry-run + 限速 max-vx 0.1 / max-wz 0.05
```

### 3.3 RELOC 客户端（真机取当前 zed 点云求纠正）

```bash
cd ~/deploy
python reloc_client.py --x 0 --y 0 --yaw 0              # 真机 live_grab（需在 robot ws 环境）
python reloc_client.py --npz <某帧.npz> --x 1.0 --y 0.5 --yaw 30 --cam_h 1.71   # 离线回放
```

- 判据：返回 `{"ok":true,...,"accepted":true,...}`，打印 `纠正量 Δx/Δy/Δyaw` 和纠正后位姿。

---

## 4. 推荐启动顺序（一条龙）

1. **服务器**：先起 §2.1 reloc_server + §2.2 vlm_server，各 `curl /health` 确认活。
2. **Orin**：起 §3.1 机器人系统，确认 `/svtrobot_cmd` 在。
3. **Orin**：`curl` 服务器两个 health，确认网络通。
4. **Orin**：Web 点录制 10s，确认 `pointcloud/zed/*.npz` 格式对。
5. **Orin**：§3.2 基线 DD 干跑，确认特征链+模型+控制环三段通。
6. **服务器**：§3.3 RELOC `--selftest`，确认重定位精度 <10cm/<2°。

> P1+ 完整任务链需等 `run_lab_mission_real.py` 状态机移植完（见操作手册 §6 缺口），今天先做上面 6 步。
