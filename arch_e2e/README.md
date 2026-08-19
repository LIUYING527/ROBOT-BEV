# arch_e2e —— 双脑巡检架构 端到端流水线（实时更新）

> 本文件夹把「看图 → 决策 → 动」整条链**从学长的 `robot_orig_code` 里解耦出来自成一体**：
> 需要的学长代码/权重都**复制**进 `vendor/`、`weights/`，脚本里**没有任何路径指向 `robot_orig_code`**。
> 只有 Stage A 的 3DGS 渲染器复用【我们自己的】`repo/third_party/discoverse`（那不是学长的东西）。
>
> 本 README 与 `outputs/bev_gallery/README.md` 一样**随进展实时更新**——每次改架构就来改这里。

**最后更新：2026-07-03** ｜ 状态：快脑链路已解耦跑通；**VLM 慢脑 = 桩（待用户接入）**。

---

## 一、这是什么（大白话）

双脑巡检机器人（**我们的架构**，`model-architecture-diffusiondrive`）：
- **VLM 慢脑**（Qwen-VL 系，验证中）：看 FPV → **语义场景/任务理解**（要的是场景间**区分度**），
  输出"当前任务/场景"。**不是**学长那 4 个运动模式。
- **DiffusionDrive 快脑**：原版只条件于「感知 BEV + ego」，我们的**创新 = 多加一路 VLM 任务条件，
  注入级联解码器** → 从"沿路开"变"按任务选轨迹模式"。

> ⚠️ **两套接口别混**：
> - **学长的**（`robot_orig_code`，我们现在 arch_e2e 里跑的 ckpt）：VLM→`cruise/turn/inspect/exit`
>   4 个**运动基元** one-hot，塞进 `status_feature` 前 4 位（复用他自带的 `mode_head` 槽）。本质是运动
>   基元、靠运动学启发式标的，**不是语义理解**。
> - **我们的**：VLM 做**语义场景/任务**识别，经**新的任务条件注入通路**进快脑。这个接口 ≠ 学长的 4 位槽，
>   所以**快脑必须按新通路重训**——学长 ckpt 喂不进我们的语义条件。

三输入（Stage B 现用学长 ckpt 做烟雾测试，故 status 仍是他的 8 维格式）：
| 输入 | 形状 | 来源 |
|---|---|---|
| `camera_feature` (FPV) | `[1,3,256,1024]` | Stage A 3DGS 渲染 → resize |
| `lidar_feature` (BEV) | `[1,1,256,256]` | 官方点直方图 `sim_bev.official_lidar_feature`（=`points_to_bev_tensor`，零手搓）|
| `status_feature` | `[1,8]` | `[mode one-hot ×4 \| vx,vy,·,·]`。⚠️前 4 位是**学长的 4 运动模式槽**，非我们慢脑接口 |

## 二、两阶段（绕开 env 冲突：渲染器 vs 模型不同 python 环境）

```
Stage A  (discoverse_venv, 有 3DGS 渲染器, 需 GPU+EGL)
  sim 沿参考轨迹渲 FPV + 存位姿  ──►  runs/obs_seq_<session>/{NNN.jpg, poses.npz}
                                          │
Stage B  (artifixer env, 只用 vendor+weights, 不碰 renderer / 不碰 robot_orig_code)
  逐帧: FPV+BEV+status ─► DiffusionDrive ─► 4s轨迹 ─► 静态占据里展开算(碰撞/进度)
                                          └►  runs/eval_nonreactive_<session>.{mp4,json}
```

## 三、目录

```
arch_e2e/
  pipeline/
    _paths.py            统一路径 + inject_vendor()（唯一 import 注入点）
    sim_bev.py           官方 BEV 生成件（从 vendor/robot_data_prep 引官方直方图）
    stage_a_gen_obs.py   Stage A：渲染观测
    stage_b_eval.py      Stage B：快脑评测（用学长 ckpt 烟雾测试；含他的 4 模式槽 mode_onehot_for_ckpt）
    stage_vlm_slowbrain.py  ⭐我们的 VLM 慢脑：语义场景/任务验证（独立，不接快脑）
  vendor/                ⭐从学长处复制来的解耦依赖（脚本只认这里）
    navsim/              DiffusionDrive 模型代码（agents/diffusiondrive + common）
    robot_data_prep/     官方 BEV 直方图 bev.py
    nuplan/              _nuplan_shim（免装整套 nuplan）
  weights/               ⭐复制来的权重
    best-val-loss.ckpt   学长 ckpt（仅烟雾测试快脑，非最终；重训后替换即可）
    kmeans_navsim_traj_20.npy   轨迹锚点
    resnet34.a1_in1k/    图像 backbone 预训练权重
  runs/                  本架构自己的产物（obs / 视频 / 指标 / 日志）
  vlm/                   ⭐VLM 慢脑权重放这（./download_vlm.sh 下 Qwen-VL 到这）
  run_stage_a.sh  run_stage_b.sh  download_vlm.sh
```

## 四、怎么跑

```bash
# Stage A（渲染观测；已有 runs/obs_seq_colmapjoint_all 则可跳过）
./run_stage_a.sh colmapjoint_all 60 <gpu>

# Stage B（模型评测；不需要渲染器）
./run_stage_b.sh colmapjoint_all <gpu> cruise
# 产物: runs/eval_nonreactive_colmapjoint_all.mp4 / .json
```

## 五、⭐ VLM 慢脑（我们的架构 —— 当前：先只跑慢脑验证）

用户拍板：**先只跑慢脑验证，暂不接快脑**（接快脑必然要按我们的新任务条件通路重训；且 0617 说话人4
担心 3B 幻觉，先 few-shot 不训看区分度够不够）。

- **模型 = `Qwen/Qwen3.5-4B`**（2026.03 原生多模态，同规模强于 Qwen3-VL，transformers 5.5.0 支持 `qwen3_5`；
  ⚠️ Qwen3.6 更新但只有 27B/35B 大杯且 transformers 5.5.0 不认，会牵动 ArtiFixer 栈，故不选）。
  下载：`./download_vlm.sh`（走 ModelScope 绕防火墙 → `vlm/Qwen3.5-4B`）。
- **跑慢脑验证**：`pipeline/stage_vlm_slowbrain.py` —— 对每帧 FPV 输出「场景 + 该做的巡检任务(自拟) + 关键物体」，
  产出 `runs/vlm_slowbrain_<session>/index.html` 图库，浏览器打开**眼看不同场景是否被区分开**。
- **注意**：这里 VLM 输出的是**语义任务/场景**，不是学长的 4 运动模式。验证 OK 后再拍板注入形式
  （离散语义任务集 / 语言 embedding）并重训快脑 —— 那才是接进 Stage B 的时刻。

> `stage_b_eval.py` 里的 `mode_onehot_for_ckpt` 是**学长 ckpt 的 4 模式槽**，与本节的语义慢脑无关，别混。

## 六、进展日志（倒序，实时追加）

- **2026-07-03（傍晚）**：⭐**慢脑验证通过**。Qwen3.5-4B 对 40 帧 FPV 逐帧输出语义场景/任务，
  解析 40/40，12 种任务，区分度好且接地（#20 认出"带红色指示灯的配电柜"、#15 阀门、#39 绿色安全
  出口指示灯、#00 全黑判"暂停巡检"）。few-shot 不训即可，0617 担心的 3B 幻觉基本未现。副产品:
  几帧 off-path 糊画面被判"等待画面恢复"→印证皮肤线(ArtiFixer)对慢脑也有用。图库
  `runs/vlm_slowbrain_colmapjoint_all/index.html`。下一步:拍板注入形式+重训快脑接语义条件。
- **2026-07-03（下午·续）**：定 VLM 慢脑型号 = **Qwen3.5-4B**（查证:无"Qwen3.7";真实谱系 Qwen3-VL/2025.10 →
  Qwen3.5/2026.02 原生多模态含2B/4B/9B小杯 → Qwen3.6/2026.04 只有27B/35B大杯; transformers5.5.0 认
  qwen2_5_vl/qwen3_vl/qwen3_5, 不认 qwen3_6）。Qwen3.5-4B 原生多模态吃图、同规模强于 Qwen3-VL、能上 Orin。
  ModelScope 下载中(~9.3G)。stage_vlm_slowbrain 加载类兼容 + 关 thinking 直出 JSON。
- **2026-07-03（下午）**：**纠正架构归属**——之前把 VLM 写成学长的"4 选 1 运动模式"是错的。我们的
  VLM 慢脑 = 语义场景/任务理解。加 `stage_vlm_slowbrain.py`（慢脑验证，出 HTML 图库看区分度）+
  `download_vlm.sh`（ModelScope 下 Qwen2.5-VL-3B）。stage_b 的 4 模式函数改名 `mode_onehot_for_ckpt`
  并标注"仅为学长 ckpt"。用户拍板：**先只跑慢脑验证，暂不接快脑（接需重训）**。待用户下 VLM。
- **2026-07-03**：新建 arch_e2e，从 `robot_orig_code` 解耦（vendor 复制 navsim/bev/nuplan-shim + 权重）。
  三脚本改写为零 `robot_orig_code` 引用。Stage B 用现有 40 帧观测跑一遍验证解耦。
  ⚠️ **VLM 仍是桩**，`status_feature` 前 4 位写死 cruise —— 所谓"端到端"目前只是**快脑半条链**，
  双脑要等 VLM 接入才算完整。轨迹质量受 sim↔真实 domain gap 影响（预期，非架构 bug）。

## 七、已知边界 / 待办

- [ ] **VLM 慢脑验证**（进行中）：用户下 Qwen-VL → 跑 `stage_vlm_slowbrain.py` → 看图库区分度。
- [ ] 验证 OK 后**拍板任务条件注入形式**（离散语义任务集 / 语言 embedding），改快脑解码器加新通路。
- [ ] **重训快脑**接我们的语义任务条件（学长 ckpt 只有 4 运动模式槽，喂不进）。
- [ ] BEV 室内重训（±4~6m range + ego 坐标 + 真轨迹标签）后替换 `weights/`。
- [ ] FPV 皮肤 domain gap：ArtiFixer 修正线（皮肤层）与本架构并行，见 `outputs/artifixer`。
