# 用 DiffusionDrive 实现移动 / 路径规划

> 更新 2026/07/07。本文讲**我们后面怎么用 DiffusionDrive（快脑）来做机器人的移动路径规划**：它到底输出什么、怎么把输出变成机器人能执行的运动指令、我们项目分几步落地。
>
> 全部对齐上车真实代码：模型 `robot_orig_code/DiffusionDrive-main/navsim/agents/diffusiondrive/`，闭环推理节点 `robot_orig_code/svtrobo_ws-master/src/chassis_control/scripts/run_diffusiondrive_policy.py`。数据/BEV 见 [数据采集规范_0707.md](数据采集规范_0707.md) 与 memory `real-navsim-vs-zed-bev-0706`。

---

## 一、一句话定位

DiffusionDrive **不是**输出"下一步速度"的控制器，而是一个**条件轨迹生成器**：给它当前的相机+BEV+任务状态，它一次**扩散采样出未来 4 秒的一整条轨迹**（8 个路点）。**路径规划 = 这条轨迹；运动控制 = 把轨迹跟踪成 `[vx, vy, wz]`**。两件事在上车代码里是分开的两层。

```
感知输入 ──► DiffusionDrive ──► 未来轨迹(8×[x,y,yaw]) ──► 轨迹跟踪 ──► [vx,vy,wz] ──► 底盘
(cam+BEV+status)   (快脑/本文)      (路径规划结果)        (控制层)      每2Hz重规划
```

---

## 二、DiffusionDrive 输出什么（锚定扩散轨迹）

核心是 `transfuser_model_v2.py::TrajectoryHead`（DiffusionDrive 的招牌 = **anchored truncated diffusion**）：

- **预测目标**：`TrajectorySampling(time_horizon=4s, interval_length=0.5s)` → **8 个未来路点**，每点 3 维 `(x, y, yaw)`，在 **ego 车体系**（x 前、y 左）。
- **锚点先验**：`plan_anchor = kmeans_navsim_traj_20.npy` → **20 条聚类锚点轨迹**。扩散不是从纯噪声起步，而是**在 20 条锚点附近做截断扩散去噪**，这就是它比普通回归/纯扩散又快又稳的原因。
- **多模态 + 打分**：输出 `plan_reg (B, mode, 8, 3)` 一批候选轨迹 + `plan_cls` 每条的分数；推理时 `pred_idx = argmax(mode_probs)` **选最高分那条**（`run_diffusiondrive_policy.py:1284`）。
- 拿到 `pred["trajectory"][0]` = **(8, 3)** 一条轨迹（`:1318`）。

> 直觉：模型看一眼场景，从"20 种常见走法"里挑一种并微调，给出未来 4s 该怎么走。

---

## 三、从轨迹到运动指令（上车真实两种跟踪 + 安全层）

拿到 (8,3) 轨迹后，`run_diffusiondrive_policy.py` 把它转成底盘 `Twist=[vx,vy,wz]`，有两条路径：

### 1. 分段差分（legacy，`_traj_to_segment_velocities` :1517）
相邻两路点求差，投影回车体系：
```
dx,dy,dth = pose[i+1] - pose[i]
vx = ( dx*cos + dy*sin) / dt      # dt = traj-interval-dt = 0.5s (须等于训练 interval)
vy = (-dx*sin + dy*cos) / dt
wz =  dth / dt
```
得到 8 段"每段常值"的速度序列。

### 2. 样条路径跟踪（default，`_tracker_compute_cmd` :785，推荐）
把 8 路点拟合成样条参考路径，用**带反馈**的跟踪律消除累积误差：
```
vx = ref_vx + kx * e_x        # e_* = 车体系下相对参考路径的横/纵/航向误差
vy = ref_vy + ky * e_y
wz = ref_wz + ktheta * e_theta
```
比纯 open-loop 分段差分更稳（不会因积分漂移越跟越偏）。

### 3. Receding horizon（滚动重规划）
- **不是**把 8 段全执行完才重想。`--execute-segments N`：只执行前 N 段就**重新推理**（如 N=2 + dt=0.5 → 1s 视界、`--loop-hz 2` 即 2Hz 重规划）。
- `--replan-transition-ramp-s` 在新旧规划之间平滑过渡，避免重规划抖动。

### 4. 安全 / 限幅层（模型之外的硬保护）
- **深度急停**：前方中心条深度 < `--safety-stop-depth-m`(0.8m) → 停/重规划；右侧条可单独设阈值提前躲墙。
- **限幅**：`--max-vx/vy/wz` 夹速度；`--forbid-model-reverse` 禁模型倒车；`--right-turn-only` 等策略护栏。
- 这层**不信任模型**，是兜底安全，防止轨迹异常直接撞。

---

## 四、输入三件套（喂给快脑的，回链采集/BEV）

| 输入 | 形状 | 来源 | 说明 |
|---|---|---|---|
| `camera_feature` | 3×256×1024 | ZED 左/(拼接)图 | 上车单前视；NAVSIM 是三目拼接 |
| `lidar_feature`（BEV） | 1×256×256 | 点云俯视直方图 | 我们用 **ZED 深度点云**过同一函数（见 BEV 对比）；⚠️只有前向锥，非环视 |
| `status_feature` | 8 | `[driving_command(4), vel(2), acc(2)]` | 学长把前 4 维当 `cruise/turn/inspect/exit` one-hot。**⚠️ 我们不用这套 4 状态编码**（见 §六）；跑学长 ckpt 时这 4 维喂固定/零值只为让它能跑 |

> ⚠️ **快慢脑接缝不走这 4 维**。我们的 VLM 慢脑条件注入 DiffusionDrive **扩散解码器的 `global_cond`**（§六），不是学长那套 status 4-mode。

---

## 五、我们项目怎么落地（分步）

现状：闭环测试台已通（memory `closed-loop-env-working-0625`）——`obs={fpv,bev,pose}` + `step(v,ω)`，pure-pursuit 能跑；就差把**真模型**插进 `step`。

**Step 1｜跑通闭环管线（用学长 ckpt，无 VLM 条件）**
- BEV 保持 **±32m**（对齐 ckpt 训练分布），`status` 的 4 维 driving_command 喂**固定/零值**（只为让 ckpt 能跑，**不当作我们的条件设计**）。
- 把 `run_diffusiondrive_policy.py` 的"轨迹→样条跟踪→vx,vy,wz"搬进仿真器 `step`，验证 obs→模型→轨迹→控制这条链通。
- 预期有 domain gap（前向锥 BEV vs 车端环视 LiDAR、室内 vs 车道），轨迹可能偏——这步是**管线打通**不求好。

**Step 2｜BEV 对齐 / 洗干净**
- 用重建的**多帧融合占据栅格**（`sim_bev.py::OccBEV`，log-odds）补前向锥"后方全黑"，喂更接近环视的干净 BEV。
- 决策点：继续 ±32m 复用 ckpt，还是转 ±4~6m 室内尺度（后者**必须重训**）。

**Step 3｜VLM 慢脑条件注入解码器（我们的接缝，见 §六）**
- 慢脑（Qwen2.5-VL）从 FPV 出**语义条件向量** → 注入 DiffusionDrive **扩散解码器的 `global_cond`** → **重训**让快脑响应。
- ⚠️（memory `arch-e2e-folder-and-vlm-correction`）：**不走学长的 4 状态 mode**；我们的语义条件是另一层新通路。

**Step 4｜重训快脑**
- 用 [数据采集规范_0707.md](数据采集规范_0707.md) 采的新数据（更好的 BEV 覆盖 + 准确轨迹 GT + 按需的语义条件标注，**非 4 状态**）重训，让快脑吃我们的 BEV 分布 + VLM 语义条件。

---

## 六、VLM 接口到 DiffusionDrive（⭐注入扩散解码器，**不用学长 4 状态**）

> 双脑架构的关键接缝。**明确前提：我们不用学长那套 status 4 状态编码**（cruise/turn/inspect/exit）。我们的 VLM 条件走 DiffusionDrive **扩散解码器内部的 `global_cond` FiLM 挂钩**并重训。下面先给代码实证的入口，再给做法。对齐 `transfuser_model_v2.py`。

### 6.1 我们要用的注入点：解码器 `global_cond` FiLM 挂钩（代码实证）

`ModulationLayer`（`:258-297`）的 `forward(traj_feature, time_embed, global_cond=None, global_img=None)` 有个 **`global_cond` 形参目前恒传 None**（`:366`）。它在**扩散解码器每一层内部**（`CustomTransformerDecoderLayer.time_modulation`, `:336 ModulationLayer(d_model, 256)`）对轨迹特征做 `traj = traj*(1+scale)+shift` 的 FiLM 调制。**这就是"把 VLM 弄进解码器"的现成入口**——条件直接作用在去噪过程内部，表达最强，最贴"语义条件的扩散策略"。

> 对比（**我们不走的**）：学长的条件插槽是 `status_feature[:4]` 4 状态 one-hot → `_status_encoding` → 拼进 decoder keyval → 影响 `ego_query`（`model_v2.py:115,117,130`）；模型还有个 `ModeHead`（`:153`）从 BEV 自推 mode。**这套 4 状态我们弃用**，只在跑学长 ckpt 打通管线时给它喂固定/零值。

### 6.2 做法：VLM 语义向量 → `global_cond` → 重训

- VLM 慢脑从 FPV 出**语义条件**（结构化 JSON 或投影 embedding）→ 编码成 `vlm_vec` → 塞进每层 `ModulationLayer.global_cond` → 重训让扩散头学会响应。
- **改动点（具体到代码）**：
  1. `ModulationLayer.__init__` 的 `condition_dims` 从 `256`（=time_embed）改为 `256 + vlm_dim`；
  2. `_trajectory_head.forward → CustomTransformerDecoder → CustomTransformerDecoderLayer.forward` 一路把 `global_cond=vlm_vec` 透传（链路已存在，只是现在传 None）；
  3. 加一个 `VLM → vlm_vec` 投影头（VLM 隐层/文本 embedding → `vlm_dim`）；
  4. 重训（或微调）。
- **退路备选**：若嫌改解码器重，可把 `vlm_vec` 拼进 `status` 扩 `_status_encoding` 输入维度（走 keyval→ego_query 老路径），改动更小但注入较浅。**仍然不是 4 状态 one-hot**，是连续语义向量。

### 6.3 分阶段

- **Phase 1（管线验证，无 VLM 条件）**：用学长 ckpt 跑闭环，`global_cond=None`、status 4 维喂零值，只验证 obs→模型→轨迹→控制通（见 §五 Step 1）。
- **Phase 2（我们的目标，重训）**：VLM 语义 → `global_cond` 注入解码器 + 重训。这是"把 VLM 弄进解码器"的落地。

### 6.4 VLM 输出契约

- **结构化 JSON** 的语义条件（如场景类型/目标物/任务/方向…），而非裸 embedding——可解释、可调试、可映射到小 embedding 表；再由投影头编码成 `vlm_vec` 喂 `global_cond`。
- **不是** `{"mode":"cruise|turn|inspect|exit"}` 那种 4 类离散标签。

---

## 七、关键参数对照（训练↔部署必须一致）

| 参数 | 值 | 说明 |
|---|---|---|
| `time_horizon` / `interval_length` | 4s / 0.5s | → **8 路点**；部署 `--traj-interval-dt` 必须 =0.5 |
| plan anchors | 20 (`kmeans_navsim_traj_20.npy`) | 锚点扩散先验，部署要带同一文件 |
| 轨迹维度 | (8, 3) = (x,y,yaw) ego 系 | x 前 y 左 |
| BEV 尺度 | ±32m / 4px每米 / 256² | 复用 ckpt 须保持；改则重训 |
| replan | `--loop-hz 2`, `--execute-segments 2` | 2Hz、1s 滚动视界 |
| 安全停 | `--safety-stop-depth-m 0.8` | 模型外硬保护 |

---

## 八、卡点 / 风险

- **域差**：前向锥 BEV vs 车端 360° 环视 LiDAR（真实数据实测差一量级）→ 复用 ckpt 轨迹可能偏，Step 2 的 BEV 洗干净是关键。
- **室内尺度**：±32m 对室内过大，点云挤中间；换 ±6m 提升表达但触发重训。
- **mode 语义错位**：学长 4 运动模式 ≠ 我们语义任务条件，慢脑接入需重训快脑。
- **控制稳定性**：优先用样条跟踪（带反馈）而非分段差分（open-loop 易漂）；receding horizon + 过渡 ramp 防抖。

---

*模型细节对齐 `transfuser_model_v2.py` / `transfuser_config.py`；部署细节对齐 `run_diffusiondrive_policy.py`（轨迹→速度 :733/:798、模式选择 :1284、安全层 CLI）。*
