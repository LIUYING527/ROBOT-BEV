# VLM+Diffusion 语义可控多模态长程路径规划 —— Novelty 撞车分析与 SOTA 全景

> 编制:2026-07-08。来源=两轮深度调研(deep-research workflow 103 agent + 针对性 workflow 9 agent + Qwen-RobotNav 专项深读),对抗式多票核实。
> 用途:论文 related work 底稿 + 定位决策依据。
> 目标定位:冲 CVPR/ICLR,真车(巡检机器人,DiffusionDrive 已上车)作 sim-to-real 验证轨道。
> ⚠️ 全文 benchmark 数字多为**竞品论文自报表**——本环境网络策略挡了官方 leaderboard 服务器(eval.ai / huggingface space / paperswithcode 均不可达),正式投稿前每个榜首数字须再上官方服务器核。

---

## 0. 一句话结论(verdict)

**"VLM 慢脑 + Diffusion 快脑"作为通用架构主张,novelty 已经没有了**——DualVLN(ICLR 2026)几乎原样做了且全开源。**但课题不死**:三个更细的角度经逐篇核实仍是空白/半空白,论文脊柱须从"提出架构"改为"**度量并实现语义可控的多模态轨迹分布**"。最强车道 =(a)Diffusion 多模态快脑 +(b)语义可控性新评测协议;真车 sim-to-real +几何真实(生成只进皮肤层)作差异化护城河。

---

## 1. Novelty 撞车全景

当前(2025末—2026中)前沿**一致**收敛到"VLM 慢脑 + 快速低层控制器"的分层范式——这印证了我们双脑方向可行,但也意味着**架构本身零新意**。逐个点名:

| 工作 | 出处 | 慢脑/快脑 | 与本课题重合 | 留给我们的缝隙 |
|---|---|---|---|---|
| **DualVLN**(最致命) | arXiv 2512.08186,上海AI Lab,**ICLR 2026**,全开源 | 7B Qwen-VL-2.5 @2Hz + **Diffusion Transformer @30Hz** | 架构几乎一模一样,含 Diffusion 快脑;有真机 | 慢脑只出**像素目标**非语言分解;无语义可控性评测;无多模态多样性度量;无巡检/任务模式;**输入是 RGB(-D) 无 BEV/占据图** |
| **Qwen-RobotNav** | arXiv 2606.18112,阿里通义,2026-06,**闭源** | Qwen3.6-Plus 规划器 + Qwen3-VL 执行器 | 两层+**自然语言中途切任务模式**(撞巡检模式思路);导航 SOTA | 快脑是 **4层 MLP 回归,零 Diffusion/零多模态**;91.4 PDMS 重度依赖 GT ego 先验(去掉掉到 79.5);任务模式是导航族(VLN/PointNav/ObjNav/Tracking)非语义行为;无可控性度量;真机是通用 demo 非巡检 |
| **CLASH** | arXiv 2512.10360,同济,预印本(TRO 在审) | 大 VLM 反思器 + <2B 反应式小模型 | 大-小分层 | 小模型是学出的 VLN 策略,**非 Diffusion**;无可控性 |
| **VLingNav** | arXiv 2601.08665,2026 初 | VLM + AdaCoT 快慢思考 | 快慢切换 | ObjectNav-HM3D 榜首但**无 Diffusion**(LLM 出离散动作 token) |
| **FantasyVLN** | arXiv 2601.13976,阿里 AMAP,**CVPR 2026** | VLM + 视觉自回归 CoT(VAR) | 长程 LH-VLN 当前最佳 | **非 Diffusion**(自回归图像 token);无可控性;LH-VLN 分数极低=远未解决 |
| manipulation 一大片 | π0 / CogACT / GR00T / FiS-VLA(2506.01953)/ VLM-TDP(2507.04524) | VLM Sys2 + Diffusion 动作头 Sys1 | 范式早已烂大街 | **全是操作(manipulation)不是导航**;长程语义导航这套仍空 |

**要害**:两层 VLM 规划器、LLM 中途切任务模式、长程语言记忆——**这些都已经有人做了,不能当我们的新意**。可 over-claim 的红线见 §9。

---

## 2. DualVLN 逐条对照(我们最大的威胁,已精读全文焊缝)

**它是什么**:异步双系统 VLN 基础模型。System-2 = 7B Qwen-VL-2.5 @2Hz,图像接地推理出**2D 像素目标坐标** + 4个可学习 latent query + 离散视角调整/STOP;训练后**冻结**。System-1 = 轻量 **Diffusion Transformer**(hidden 384 / 12 层 / 6 头)@30Hz,吃 latent goal(3584→768 投影)+ 高频 RGB(DepthAnythingV2-Small 编码,Q-Former 压到32 token),**并行 0.03s 出 32 条稠密轨迹**(flow/速度预测)。真机:Turtlebot4 / Unitree Go2 / Unitree G1 + RealSense D455,远程 4090。

| 维度 | DualVLN 做了吗 | 证据 | 我们的切入 |
|---|---|---|---|
| 语言级任务分解 + 语义条件注入 | **没有** | 慢脑只输出像素坐标+latent+视角动作;指令只用来预测"下一个像素路点" | ✅ 语言级分解 + 语义条件注入进快脑 |
| 语义可控性(换指令→轨迹**分布**变) | **没有** | 仅做"像素目标↔轨迹一致性"(Fig10,1000样本量角度偏差)=目标跟随,不是分布变化 | ✅✅ 最干净的独占车道 |
| 多模态多样性度量 | **没有** | 32 条轨迹只当"快速候选集"求平滑实时,从不分析 mode 覆盖/熵/多样性 | ✅ 把多模态当**语义特征**去度量 |
| BEV/占据图输入 | **没有** | System-1 输入是 RGB(-D) egocentric,无 BEV/几何图 | ✅ 我们 DiffusionDrive-BEV 直方图占据图输入=不同赛道 |
| 巡检/任务模式切换 | **没有** | 只有慢脑内部"出像素目标 vs 视角调整 vs STOP"的感知子决策 | ✅ 语义巡检行为模式 |
| 真实巡检机器人 + sim-to-real 定量协议 | **没有** | 通用指令跟随导航,无巡检、无 sim-to-real 评测协议 | ✅ 真车巡检 + 几何真实(生成只进皮肤层) |
| Benchmark | R2R-CE Val-Unseen(RGB-only)**NE4.05/OSR70.7/SR64.3/SPL58.5** 新 SOTA;RxR-CE SR51.8;VLN-PE 零样本迁移超所有 baseline;自建 Social-VLN(加人体碰撞率 HCR) | — | 对标点 |
| 自曝短板 | System-1 对"大的/语义错误的像素目标"不鲁棒(近障碍时);System-1 数据 1% 即饱和(瓶颈在 System-2) | Fig8 + 数据缩放实验 | 可作为"确定性目标接地脆弱"的攻击点 |

**结论**:DualVLN 占了"架构 + Diffusion 快脑"这两格,但**语义可控性、多模态多样性度量、语言级条件注入、BEV 输入、巡检真车**这五格全空——我们的贡献焊得住。

---

## 3. Qwen-RobotNav 逐条对照(导航 SOTA,撞任务模式但无 Diffusion)

**要点**:两层 agentic。上层 Qwen3.6-Plus **用自然语言**分解目标、每段发 `(子目标指令, 任务模式, 观测配置)`;下层 Qwen3-VL 骨干当反应式执行器,真机 ~5Hz。

- **快脑 = 4层 MLP + MSE,单条确定性 8 路点 (x,y,θ)**,通篇零 Diffusion/零生成。Diffusion 只出现在两处非竞争位置:数据增强的图像风格迁移、以及把 **DiffusionDrive 列为它 NAVSIM 上打败的 baseline**。**从不讨论 mode averaging**=毫无防御。
- **91.4 PDMS 灌水**:须喂前 3 帧 GT ego 轨迹先验;去掉(‡行)4B/8B 掉到 **79.5**,比所有驾驶 baseline 还低(连 NavFoM 84.3 都不如)。且 91.4 是 **4B 非 8B**。
- **任务模式切换**:模式是导航任务族 VLN/PointNav/ObjNav/Tracking,切的主要是**观测编码/上下文策略**(token 预算、时间衰减、相机权重),**不是轨迹分布**;快脑跨模式同一个、确定性;它自己说贡献是"观测上下文建模而非架构"。**没有 inspect 这种语义行为模式**。
- **语义可控性**:无(确定性头结构上不可能出分布);只有"让它停床左侧就停左侧"的定性轶事,无指标。
- **真机**:Unitree Go2 四足 + Jetson Thor,零样本 demo,**非巡检、无 sim-to-real 定量协议**。且用 Qwen-Image-Edit **生成式改写每帧观测**——正是我们记忆里警惕的"生成进几何层=幻觉"风险,我们"生成只进皮肤层"反着打。
- **闭源**:代码/权重/数据均不放(明确无放权重计划)→ 没法复现,但也当不了"人人可比"的强 baseline。
- **无独立 Limitations 节**(结论全正面)——本身可作为审稿面的软肋点出。

**结论**:不杀 novelty。(a)Diffusion 多模态快脑 = 全开;(b)语义可控性 = 全开。**必须收敛**:两层分层、LLM 切模式、长程语言记忆都是它已做的,别当新意。

---

## 4. 各 Benchmark 真实 SOTA(截至 2026-07-08,**自报表为主,官方服务器未达**)

| Benchmark | 榜首(自报) | 数值 | 用 VLM/Diffusion/分层? | 开源 |
|---|---|---|---|---|
| **VLN-CE R2R-CE** | CLASH(claim eval.ai#719 test 第一);DualVLN(RGB-only val 最强);ETP-R1(2512.20940)也 claim 榜首 | CLASH test-unseen SR66/SPL58;DualVLN val-unseen SR64.3/SPL58.5;前任 SOTA g3D-LF val SR61/SPL52 | 前沿**一致** VLM 慢脑+快控制器分层(CLASH 无扩散,DualVLN 有扩散) | DualVLN/g3D-LF/StreamVLN 开源;CLASH 待放 |
| **ObjectNav HM3D** | VLingNav(2601.08665) | v1 SR79.1/SPL42.9;v2 SR83.0/SPL40.5(v2 SPL 仍略低于 FiLM-Nav 41.3);零样本人类参照 SR~93.7/SPL~42.5 | VLM + AdaCoT 快慢,**无 Diffusion** | VLingNav 未确认开源 |
| **LHPR-VLN(长程)** | FantasyVLN(2601.13976,CVPR2026,阿里AMAP) | 极低:SR2.44/ISR11.01/CSR9.64/CGT8.99;原 MGDM 基线 ISR4.69/CSR3.30/CGT5.83(3-4子任务档);2-3子任务档全 0 | VLM+视觉自回归 CoT,**无 Diffusion**;无分层 | benchmark 开源(HCPLab-SYSU/LH-VLN);FantasyVLN 有 repo |
| **NAVSIM PDMS** | 极挤且多>91 | Qwen-RobotNav-4B 91.4(灌先验);DiffusionDriveV2 91.2;ReCogDrive 90.8;ReflectDrive-2 91.0;AutoVLA 89.1;DiffusionDrive 88.1;更高的 DriveSuprim~93.5/TransDiffuser~94.85 | ReCogDrive/ReflectDrive=VLM+Diffusion;AutoVLA=VLM 自回归无扩散;DiffusionDrive=纯扩散无VLM | DiffusionDrive/ReCogDrive/AutoVLA 开源 |

**战略读数**:
- **LHPR-VLN 远未解决(ISR~11)= 头部空间最大**,且**目前榜上没人用 Diffusion**→ 若主打长程,这里扩散多模态+可控性最有戏;但 FantasyVLN(CVPR2026 阿里)刚占最佳,要正面比。
- **所有被查的头部方法,无一评测"指令→轨迹分布可控性"**——§7 白空白坐实。

---

## 5. Diffusion 是否必要(回应第二质疑)

- **有实证**:Diffusion Policy(2303.04137,RSS23)Push-T——DP 约 95% 成功从左右两侧接触,LSTM-GMM~20%、IBC 0%、BET 无法定于一个 mode。DiffusionDrive(2411.15139,CVPR25 Highlight)实测 vanilla 扩散在驾驶上 mode collapse,用 mIoU 多样性分 D 量化,截断扩散 D 高 64%,能出回归 baseline 没有的变道/超车。
- **但**"回归**必然**灾难性 mode averaging"这种**绝对说法在对抗核实中被否(1-2)**。→ 论文措辞必须收敛为"**案例中实证的优势**",不能写成定理,否则被一枪打死。
- 反例风险:Qwen-RobotNav 用 MLP 回归就拿了导航 SOTA(虽灌先验)→ 说明"回归够用"在某些设定成立,我们必须靠**语义可控性/多模态**这个回归结构上做不到的点来证明扩散必要,而非单纯刷分。

---

## 6. 语义可控性评测 = 真空白(我们的独门贡献候选)

两轮调研 + 逐篇核实,**没有任何现成的"指令→路径分布可控改变"评测协议/指标/benchmark**。可复用的最近组件只有 DiffusionDrive 的 mIoU 多样性分 D(测轨迹集多样性,不测指令条件下的分布位移)。LHPR-VLN 的 ISR/CSR/CGT 只测成功不测可控性。

→ **需自建**,且"自建一个可信的语义可控性协议"本身就能当一个方法贡献。建议构成:
1. **指令条件分布位移**:同起终点,指令 A vs B,生成轨迹分布的 KL/Wasserstein 距离(要显著>同指令内部方差)。
2. **指令-路径匹配打分**:VLM/人工判"这条路符合指令吗"。
3. **多模态覆盖率**:生成的 N 条候选是否覆盖多个合理 mode(借 mIoU-D 扩展)。
4. **有效性约束**:可控的同时碰撞率/可行性不塌。

---

## 7. 事实核查清账(⚠️ 修正上一轮误判:4 个"存疑"引用**全是真的**)

上一轮因"没搜到"判为"大概率幻觉",本轮针对性核实**全部证实存在且数值准确**,仅出处/ID 需修正。可放心引用:

| 引用 | 真伪 | 正确出处 | 数值核实 |
|---|---|---|---|
| DiffusionDrive 88.1 PDMS | ✅真 | arXiv 2411.15139,CVPR2025 Highlight | 准确(ResNet-34 aligned) |
| LHPR-VLN | ✅真 | arXiv 2412.09082,CVPR2025,SYSU HCPLab | ISR/CSR/CGT 指标准确 |
| CLASH VLN-CE 榜首 | ✅真 | arXiv 2512.10360,同济(预印本/TRO在审) | claim 第一,官方服务器未独立核 |
| **WAM-Diff 91.0 PDMS** | ✅真(此前判幻觉=**错**) | **arXiv 2512.11872**,复旦生成视觉组(masked-diffusion VLA,非"world model") | NAVSIM-v1 91.0 PDMS + v2 89.7 EPDMS 属实;"Envision"仅见 Medium 报道非作者列表 |
| **PRIX 87.8 PDMS** | ✅真 | **arXiv 2507.17596**(ID 正确),KTH+Scania | 87.8 PDMS/37M参/57FPS,camera-only 扩散头 属实 |
| **RATE-Nav SR67.8/SPL31.3** | ✅真 | **arXiv 2506.02354**,ACL2025 Findings(2025.findings-acl.341) | HM3D SR67.8/SPL31.3 属实 |
| **FOM-Nav +11.8 SR** | ✅真 | **arXiv 2512.01009**(ID 正确),Chabal 等 | HM3D v2 vs VLFM +11.8 SR/+14.3 SPL 属实 |

**教训**:昨天那份"评价标准与SOTA"文档的引用**质量其实不错**(ID 基本对,只有 WAM-Diff 需补 2512.11872、RATE-Nav 补 2506.02354)。此前的"幻觉"警告作废。

---

## 8. 推荐定位与实验骨架

**论文一句话定位(经得起 DualVLN/Qwen 的存在):**
> 现有两层导航基础模型(DualVLN 出像素目标+扩散跟随;Qwen-RobotNav 切任务模式重配观测但只吐**单条确定性回归轨迹**)均**不建模指令条件下的轨迹分布**。我们保留 VLM 慢脑做**语言级语义任务分解与条件注入**,加一个 **Diffusion 快脑把每个语义条件变成可控的多模态轨迹分布**,提出**语义可控性评测协议**度量之,并在**真实巡检机器人上做 sim-to-real 验证(几何真实、生成只进皮肤层)**。

**消融矩阵(回应三质疑):**

| 方法 | VLM | 轨迹生成 | 验证命题 |
|---|---|---|---|
| 传统规划器(A*/RRT*+语义地图) | ✗ | 经典 | 学习方法是否必要 |
| 回归 baseline(仿 Qwen 头) | ✓ | MLP 回归单轨迹 | **Diffusion 是否必要**(看 mode averaging + 可控性塌) |
| w/o VLM(语义地图+最近邻) | ✗ | Diffusion | **VLM 是否必要** |
| **完整** | ✓ | Diffusion | 主结果 |
| 完整 + 不同语义指令 | ✓ | Diffusion | **语义可控性**(只有完整方法路径分布可控变) |

**建议主战场**:LHPR-VLN(长程,头部空间大,榜上无扩散)作可比数字锚点 + VLN-CE 作 grounding 对照;真车作 sim-to-real 轨道。

---

## 9. 红线:别 over-claim(DualVLN/Qwen 已做,不能当新意)

1. 两层 VLM 规划器 + 反应式执行器、自然语言通信 —— 已做。
2. LLM 中途切任务模式 —— Qwen 已做(重定位为"语义行为模式 + 改分布的条件注入")。
3. VLM 慢脑 + Diffusion 快脑架构 —— DualVLN 已做(重定位为"可控多模态 + 语言条件 + BEV 输入 + 巡检真车")。
4. 长程语言记忆 —— Qwen 已做。
5. "回归必然 mode averaging" —— 措辞收敛为案例实证优势,非定理。

---

## 10. 待办 / 风险

- [ ] **官方 leaderboard 复核**:本轮官方服务器(eval.ai#719 / HF NAVSIM space / paperswithcode)被网络策略挡,所有榜首数字为自报表,投稿前须换可访问环境上官方服务器核实。
- [ ] **精读新冒出的直接竞品**:VLingNav(2601.08665)、FantasyVLN(2601.13976,尤其它在 LHPR-VLN 的做法)、ReCogDrive/ReflectDrive(VLM+Diffusion 驾驶,离我们最近)。
- [ ] **DualVLN 代码级确认**:InternNav 仓库的 System-1 是否真无任何可控/多样性接口(HF/GitHub README 本轮 JS-gated 未取全)。
- [ ] 风险:领域月度刷新极快(NAVSIM 18 个月 88.1→94.85;LHPR-VLN 刚被 FantasyVLN 占)。novelty 窗口按月计,宜尽快锁定 (b) 语义可控性协议这个别人不碰的方向立桩。

---

### 附:核心引用(已核实)
- DualVLN "Ground Slow, Move Fast": arXiv 2512.08186(ICLR 2026)| 代码 InternRobotics/InternNav | 权重 InternVLA-N1-DualVLN
- Qwen-RobotNav: arXiv 2606.18112(闭源)
- CLASH: arXiv 2512.10360 | VLingNav: arXiv 2601.08665 | FantasyVLN: arXiv 2601.13976(CVPR2026)
- DiffusionDrive: arXiv 2411.15139(CVPR2025)| DiffusionDriveV2: 2512.07745 | ReCogDrive: 2506.08052 | ReflectDrive-2: 2605.04647
- Diffusion Policy: arXiv 2303.04137(RSS2023)
- LHPR-VLN: arXiv 2412.09082(CVPR2025)| WAM-Diff: 2512.11872 | PRIX: 2507.17596 | RATE-Nav: 2506.02354(ACL2025 Findings)| FOM-Nav: 2512.01009
- manipulation 双系统: FiS-VLA 2506.01953 | VLM-TDP 2507.04524
