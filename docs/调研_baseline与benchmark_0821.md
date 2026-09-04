# 论文 Baseline 与 Benchmark 调研（0821，deep-research 对抗验证版）

> 调研方式：deep-research workflow × 3 轮（共 314 agent / 68 源 / 319 声明抽取 / 75 条对抗验证 → 69 条通过、6 条被否决）。
> 轮次：R1=benchmark+系统/扩散baseline；R2=RELOC+场景图；R3=语义可控性协议先例。
> 验证日期：2026/08/21。所有数字除非注明"官方榜单"，均为**作者自报**，投稿前须复核。
> 承接：[[各模块baseline对照_0819.md]]（模块对照总表），本文聚焦"实验用什么 benchmark、对比哪些 baseline、协议怎么设计"。

---

## 一、公开 Benchmark 选型（已核实）

### 1. LH-VLN（长程 VLN，首选入场券）✅

- **命名澄清（重要）**：一手来源（CVPR 2025 论文 arXiv:2412.09082 + 主办方 contest 页 + metrics.md）明确 **benchmark 叫 LHPR-VLN、任务/赛事叫 LH-VLN**，两种命名在官方材料并存。0819 笔记记"是 LH-VLN 不是 LHPR-VLN"与一手来源相反——**写作时用 "LHPR-VLN benchmark for the LH-VLN task" 并首现处注明**，否则审稿人查原文对不上。
- 出品：Song et al., SYSU HCP Lab，CVPR 2025；仿真器 **Habitat**；3260 任务 × 216 个 HM3D 场景，平均每任务 **150 步**；另有 MMSP 2025 Embodied AI Challenge 官方赛事渠道。
- 协议：每任务链 2-4 个顺序子任务；agent 从 +60°/0°/-60° 三视点观测；离散动作 {turn left, move forward, turn right, stop}；子任务成功 = 停在目标 **1m 测地线内且目标在 60° 水平视场内**。
- 指标：**SR / ISR / CSR / CGT**（协议另定义 SPL/OSR/NE，但 FantasyVLN 未报 SPL）；赛事总分 = 0.4·TAR + 0.2·ISR + 0.2·CSR + 0.2·CGT。
- 当前可核实最好成绩：**FantasyVLN SR=2.44 / ISR=11.01 / CSR=9.64 / CGT=8.99**（arXiv:2601.13976v2 表 2 逐字核对；其余对比方法 Random/GLM-4v/GPT-4+NaviLLM/MGDM/CoT-VLA/WorldVLA 全 SR=0，Aux-Think 0.65）。
  → **长程多阶段导航远未解决**是我们动机的硬证据，且与我们"多模态轨迹分布"卖点互补。
- ⚠ 注意：LH-VLN 的 ISR/CSR/CGT 与 VLN-CE 式 SR/SPL **不能直接横比**（口径不同）。

### 2. GOAT-Bench（终身/多目标导航，次选）✅

- CVPR 2024，代码开源 `github.com/Ram81/goat-bench`。
- 181 个 HM3DSem 场景（145 训 / 36 验），312 类物体，**~680k episode**（论文内部不一致：145×5k=725k 训练集 > 摘要 680k 总数，引用带 "~"）。
- 每 episode 5-10 个顺序子目标；目标可用**类别名/语言描述/图像**三种模态指定（与我们语义条件天然契合）。
- 协议：SR + SPL；终身链式变体中每个子任务最短路径从上一个子任务终点起算。
- 远未饱和：物体目标上 Modular GOAT 29.4% vs SenseAct-NN Skill Chain 25.8% vs Monolithic 25.7%；最佳非 oracle baseline **SR ~29.5% / SPL 17.2%**。

### 3. VLN-CE（R2R/RxR）⚠ 数字待复核

- ❌ **DualVLN 的 VLN-CE 自报数字（R2R val-unseen SR 64.3 等）被对抗验证 1-2 否决**——投稿前若引用必须查官方 leaderboard。
- 待办：VLN-CE 官方榜单 2026 可引用 SOTA 需另行核实。

### 4. HM3D ObjectNav / MP3D（VLingNav 口径）

- VLingNav 报数口径：HM3Dv1 SR 79.1 / SPL 42.9，HM3Dv2 83.0 / 40.5，MP3D 58.9 / 26.5，Instance ImageNav 60.8 / 37.4。
- ⚠ "全 SOTA" 表述过松：HM3Dv2 SPL 40.5 落后 FiLM-Nav 41.3，SOTA 仅 SR 成立；引用须精确到指标。

---

## 二、Baseline 清单（按模块，已核实部分）

### ① 整系统

| 方法 | 开源可复现性 | 结论 |
|---|---|---|
| **DualVLN** (InternVLA-N1) | ✅ InternNav 代码 **MIT**；2025/09 真机部署代码、2025/12 训练代码+文档；权重在 HF（InternRobotics/InternVLA-N1-DualVLN） | **首选整系统 baseline，完全可复现**；但 InternData-N1 数据 CC BY-NC-SA 非商用 |
| **VLingNav** (2601.08665) | VLA 架构（LLaVA-Video-7B + AdaCoT + VLingMem），动作头 MLP 高斯投影头 | **不是扩散快脑竞品**（论文明确自我区别于扩散/流匹配）；不评 VLN-CE/LH-VLN/GOAT，只在 ObjNav 系报数 |

### ② 扩散快脑

| 方法 | 开源可复现性 | 可比口径 |
|---|---|---|
| **NavDP** (ICRA 2026, 2505.08712) | ✅ `InternRobotics/NavDP` 官方开源（~759★）；已集成 InternNav 统一评测 harness | **首选扩散 baseline**；同 harness 可比 NoMaD/GNM/ViNT/iPlanner/ViPlanner/DD-PPO。ClutteredEnv SR/SPL：NavDP 89.8/87.7 vs iPlanner 84.8/83.6 vs ViPlanner 72.4/72.3；InternScenes：65.7/60.7 vs 48.8/46.7 vs 54.3/52.5。评测栈 IsaacSim 4.2.0/IsaacLab 1.2.0。⚠ 权重/数据 CC BY-NC-SA 且 ckpt 需填表获取；⚠ 同实验室 X-NavDP 报基座 NavDP 真机 hard-case SR 仅 10%（RL 后训练→65%），引用零样本真机能力须带此 caveat |
| **NoMaD/ViNT/GNM** (Berkeley BAIR) | ✅ `robodhruv/visualnav-transformer` MIT 官方 repo，训练代码+ROS 部署脚本+Google Drive 权重；NoMaD 依赖 real-stanford/diffusion_policy | NoMaD 论文本身就是 "Goal Masked Diffusion Policies" = 现成扩散 baseline。⚠ 无 VLN-CE/Habitat 协议，跨设置比较需自建；训练数据含未公开部分（从头复训不 bit-exact，作 baseline 无碍） |
| **NaviDiffusor** (ICRA 2025, 2504.10003) | 自建协议：Stanford 2D-3D-S + Gazebo Citysim，每任务 50 trials，指标 Length/Collision/Success | **与 VLN-CE/ObjectNav 不可直接比**；但与 NoMaD/ViNT 同训练数据混合（RECON/SCAND/GoStanford/SACSoN，单卡 RTX TITAN ~39h）且论文重训了二者 → 适合作**同数据下扩散 vs 回归公平对照**。推理期可微代价梯度引导与我们"免重训条件化"概念亲缘（控制源不同：代价梯度 vs 语义条件） |

### ③ RELOC 重定位（第二轮已核实，23 条 3-0 通过）

**谱系总览（按协议分两类）：**
- **refinement-only（需初始位姿，render-and-compare 迭代精化）**：iNeRF → **iComMa** → SGLoc-refine 类。协议 = 给初始扰动，测成功率/误差。
- **one-shot 全局定位（无需初始位姿）**：**GSVisLoc**、iGaussian（前馈）。

| 方法 | 精度数字（全自报） | 代码 | 关键 caveat |
|---|---|---|---|
| **iComMa** (2312.09031, **WACV 2025** 非 2024) | Table 4 消融：±[30°,50°] 旋转扰动下**去 matching 模块失败率 ~9%→38-38.5%**，去 comparing ~14% | ✅ `YuanSun-XJTU/iComMa` 官方开源 | ⚠ Issue #17/#13 有用户报告**自定义数据复现失败**（喂 GT 位姿仍发散到 25.2°）→ 引用注明"官方协议内有效，自定义数据有社区报告的坑" |
| **GSplatLoc 2412.20056** ("Ultra-Precise", Zeller, 东南大学成贤学院) | Replica 平移 ~0.016cm/旋转 ~0.009°（最优序列才 0.01cm）；TUM RGB-D 0.81cm/0.98° | `AtticusZeller/GsplatLoc` | ⚠⚠ **非同行评审 preprint**；**depth-based** 且 **tracking 用上一帧 GT 位姿初始化** → 与 RGB-only 的 iComMa 协议**不可直接比**，引用必须带 caveat |
| ⚠ **同名混淆实锤**：`haksorus/gsplatloc` 301 重定向到 `be2rlab/gsplatloc` = **另一篇** GSplatLoc（2409.16502, IROS 2025, ITMO, "Grounding Keypoint Descriptors"），Apache-2.0 148★ 可跑。**引用时极易张冠李戴；GSVisLoc Table 1 里对比的 "GSplatLoc" 是这篇**。 | | | |
| **GSVisLoc** (2508.18242, ICCVW 2025) | 7-Scenes 中值 **1.3cm/0.33°**，recall@5cm/5°=88.0%；ScanNet++ 跨场景零样本 1.94cm/2.59°（但有一个场景崩 21.65cm/37.17°） | one-shot：不需初始位姿/不检索/不改 3DGS 模型 | ⚠ 它把 iComMa/MCLoc 归类为 "refinement-only"；非 3DGS 系方法（DSAC* 1.1/0.3 等）仍更优，引用只能说"优于 3DGS 系 baseline" |
| **iGaussian** (2511.14149, IROS 2025) | 前馈反演，中值旋转 0.2°，移动机器人 2.87 FPS（vs iComMa 3.95s/帧）；协议 = ±[20°,40°]/[40°,80°]/[80°,180°] 三档扰动，成功率判据 rot<5°&trans<5cm：**iNeRF 62.9%→17.5%→0%、iComMa 78.7%→45.4%→0%、iGaussian 稳 ~82%** | ✅ `pythongod-exe/iGaussian`（API 核实结构完整） | v1 全自报无独立复现；2.87 FPS 对闭环控制仍非实时 |
| **SGLoc** (2507.12027, IROS 2025, SJTU) | 7Scenes（**只报 4/7 场景**）中值 0.5cm/0.15°、12Scenes（3/12）0.3cm/0.2° | ❌ **代码至今空 README**（承诺一年未兑现，2026-08 核实） | ⚠ 其消融（Table IV）"render-compare 精化接别的粗位姿会灾难发散(224cm/67°)"是**自家模块特性非普适定律**——同文 Table II 里 ACE+GSLoc 同样 ACE 初始化却 0.7cm/0.21°，引用勿过度泛化 |

**对我们的意义**：
1. **iComMa Table 4 + iGaussian 三档扰动成功率 = 两篇独立铁证**"纯 render-compare 对大扰动崩"→ 直接支撑我们"RELOC 必须在收敛域内周期触发 + 双闸门"的设计叙事；我们实测收敛域 ±0.5m/25° 正是该甜区。
2. 差异化仍是**周期纠偏闭环 + 无里程计 motivation + 真机 zed 域**——他们全是"一次性定位" benchmark 设定。
3. Baseline 实操：iComMa 官方 repo 可跑（注意 Issue #17 的坑）；GSVisLoc/iGaussian 可作 one-shot 对照；SGLoc 只能引数字不能复现。

### ④ 开放词汇场景图（第二轮已核实）

| 方法 | 开源 | 评测协议与数字 |
|---|---|---|
| **ConceptGraphs** (2309.16650, ICRA 2024) | （代码可用性本轮未单独核实，0729 我们已跑通全链） | Replica（room0-2, office0-3）+ **AMT 人工评测**：节点 caption 精度 **~71%**（0.65-0.78）、边精度 **~88%**、每场景重复检测 0-5 个；CG-D 变体 0.61；误差主要来自 LVLM。后续 RAG-3DSG(2601.10168) 复现同协议报 0.68 独立佐证 |
| **HOV-SG** (2403.17846, RSS 2024) | ✅ `hovsg/HOV-SG` 官方开源（MIT 带"学术用途"限定语），依赖 habitat-sim + OpenCLIP(ViT-H-14) + SAM | HM3DSem：object-level 开放词汇 AUC_top-k **84.88%**（1624 类）vs ConceptGraphs 84.07% vs VLMaps ~56%（**差距仅 0.81pt**）；层级检索（object-room-floor top-5）**+11.69pt** vs 增强版 CG baseline |
| **DovSG** (2410.11989, **RA-L 2025**) | ✅ `BJHYZJ/DovSG` 官方开源（conda env + demo.py 预录数据 + 真机说明 xARM6/Ranger Mini 3/D455；管线含 DROID-SLAM 位姿 + ACE 重定位） | 动态场景图 + 长程语言引导移动操作；建基于 OK-Robot/RoboEXP/ConceptGraphs（0819 表里的"导航 95.1% vs OK-Robot 82.8%"沿用） |

**对我们的意义**：场景图模块的对比叙事 = 我们**不做通用建图贡献**，场景图只当 RAG 几何挂载服务精定位链；ConceptGraphs 的 71% 节点精度（LVLM 误差主导）也印证我们 0729 "自研 VLM 纯文本 JSON 废 → RAM 多数票 fallback" 的工程判断。

---

## 三、语义可控性指标（独门卖点，第三轮已核实）

**总判断：学术界有现成的"积木"，但没人在导航/轨迹领域把它们组装成协议 → 空白真实存在，自建协议有明确先例可援引。**

### 3.1 可直接移植的协议骨架（可控生成领域标准做法）

**双轴分解**：样本质量（FID 类）与**条件遵循度（property-controllability）**分开度量。条件遵循度的惯用协议 = **"属性回读"两步法**：用在条件属性上训练的分类器/提取器对生成结果重新打分，与目标条件比对。

- **可控生成综述**（Wang et al., ACM Computing Surveys, 2207.09542）Table 2 逐行给出公式：连续属性用 MSE/MAE/cosine/L2、条件对数似然 CLL、**属性均值概率 AMP**（需为 M 个属性训 M 个分类器）；离散属性用 accuracy/precision/recall/F1/BCE；另有 A/B 检验 p 值判断属性分布是否显著不同。→ **整套可直接移植为轨迹的"语义可控性"协议**。
- **分类器回读范式的成熟先例**：
  - 可控文本生成：正/负样本训属性分类器（sentiment/topic/style），对生成文本报 accuracy 是标准协议（CTG 综述 2201.05337；多维版 CoDI-Eval 系用 RoBERTa 分类器+规则+Perspective API，六任务平均 accuracy）；
  - 可控图像生成（**ControlNet++** 2404.07987 → SCAR CVPR2026 等复用）：用预训练提取器从生成图**重新提取控制条件**（Canny/depth/normal）与输入条件比对，按条件类型配指标（Canny 用 F1、Depth 用 RMSE、HED 用 SSIM）。**= 我们"对生成轨迹回读语义条件"的直接学术先例**。
- **PIE-Bench**（指令编辑评测）：示范"一个条件遵循 claim → 多个正交子指标"的设计手法（结构一致性 DINO cosine / 背景保持 PSNR-LPIPS / CLIP 对齐分列）。→ 照搬到轨迹：任务达成度、轨迹分歧度、条件回读准确率分列。

### 3.2 多模态轨迹分布的证明（分布级指标线）

- **distribution-level Precision/Recall**：Sajjadi et al. (NeurIPS 2018, 1806.00035) 首提——precision=样本质量、recall=分布覆盖（mode dropping 检测），动机正是 FID 一维标量"无法区分失败情形"；**Kynkäänniemi et al. (NeurIPS 2019, 1904.06991)** 给出 k-NN 流形版精确数学协议（k=3 超球，Eq.1-2），并指出 Sajjadi 版在 mode collapse 时失效；**Naeem et al. (ICML 2020, 2002.09797)** Density/Coverage 修离群点脆弱性，**代码公开**（clovaai/generative-evaluation-prdc）。适用于"扩散采样轨迹 vs GT 轨迹分布"，**前提：有合理轨迹特征空间 + GT 分布本身多模态**（否则 recall 高也证明不了多模态——此点对抗验证 2-1 分票，须谨慎）。
- **Vendi Score**（Friedman & Dieng, **ICLR 2023, arXiv:2210.02410** ⚠注意正确编号，有张冠李戴误引 2302.08666=无关量子论文）：相似度矩阵特征值的 Shannon 熵指数 = "有效不同样本数"，无参考分布即可测各条件下轨迹多样性。⚠ "reference-free 适用任意域"强说法被 1-2 否决，用前需自验核选择。
- **PCE/RCE/RE**（AAAI 2025, 2412.14340）：信息论三维指标（fidelity + 类间/类内多样性），支持 mode 级分解。实验均在图像域，轨迹应用属域迁移。

### 3.3 ⭐ 协议设计范式：ICML 2025 立场论文（2505.22450, vanderschaarlab, 代码开源）

- 证明**所有现行 fidelity/diversity 指标都挂掉大量 sanity check**（Improved P/R、Density/Coverage、α-precision/β-recall、FID 系）——用任何指标前必须自验。
- 贡献可借鉴的方法论：**先定义 desiderata + 针对已知失败模式设计合成 sanity checks**（程序化 pass/fail 判定，挂 D1b 目的/D2 超参/D3 数据/D4 界/D5 不变性）。
- → **我们的协议设计模板**：为语义可控性指标定义 desiderata（换条件轨迹必须分开 / 同条件轨迹必须聚拢 / **回归 mode-averaging 必须被抓出**），再让指标过检。这是把"自建指标"写得有审稿说服力的关键引用。

### 3.4 空白证据（指令条件策略领域现状）

- **CALVIN**（RA-L 2022，语言条件操纵标准 benchmark）：只用任务链成功率（1000 条 5 连指令链，最佳 48.9/12.9/2.6/0.5/0.08%），**无任何轨迹分歧度或条件敏感性指标**；对"红/蓝块指令混淆"这类条件敏感性失败只有定性诊断。
- **Diffusion-ES**（CVPR 2024）：测指令可控性 = 8 个 nuPlan 指令任务的**逐指令成功率**（IDM 加噪 + 10 种子）——"换指令→换行为"评测的具体先例，但仍不是指令间行为分布散度。
- **结论**："同场景、最小差异指令 → 行为分布显著不同"在主流 benchmark 从未被量化 → 我们的指标组合（**条件回读准确率 + 条件间轨迹分布散度 + ICML2025 式 sanity checks**）有明确空白可填。

### 3.5 我们的协议草案（基于上述先例组装）

1. **条件回读准确率**（借 ControlNet++/CTG 范式）：训一个"轨迹→语义条件"分类器，对扩散采样轨迹回读条件，报 accuracy/F1；
2. **条件间分布散度**（借 P/R 线）：同一观测、不同条件下的采样轨迹集合，在轨迹特征空间算 P/R 或 Density/Coverage，证条件分离；同条件多次采样算聚拢度；
3. **回归对照抓 mode-averaging**：MLP 回归 baseline 在冲突条件数据上的输出应被 sanity check 抓出（既不分开也不覆盖）；
4. **sanity checks 先行**（借 ICML 2025 范式）：合成 case 验证指标本身（如人工构造两个已知 mode，看 recall 是否检出 mode dropping）。

### 3.6 ❌ 第三轮被否决声明

1. "Self-BLEU 可直接迁移证轨迹多模态" → 0-3 否决（勿用）；
2. "Vendi Score 无需参考分布即可用于任意域" → 1-2 否决（定义本身可用，强外推勿写）。

### 3.7 仍待精读（不阻塞协议设计）

- Diffusion Policy / DiffusionDrive / NoMaD 原文实验章节：它们具体用什么实验证明多模态（第三轮源已定位：2303.04137 / 2411.15139 / nomad 项目页，但声明未入验证批次）——引用"扩散多模态优于回归"时仍需逐篇精读（0819 已知 DiffusionDrive 自证普通扩散多样性仅 11%，靠锚点+截断才 74%）。
- VLN 原文有无 instruction-sensitivity 消融（换指令测路径分歧度）。

## 四、真机评测学术惯例

- 已核实先例（NaviDiffusor，ICRA 2025）：**真机部分仅定性 demo**（Diablo 轮腿 + Jackal，RealSense D435i 纯 RGB，**Jetson Orin AGX 端侧**），定量表只覆盖仿真 → **ICRA 级论文接受"仿真定量 + 真机定性"轻量协议**，且硬件与我们 Orin 全链同档。
- 更强协议（episode 数 / 成功率统计口径）待查 NoMaD/NavDP 真机章节。

---

## ❌ 本轮被对抗验证否决的声明（禁止引用）

1. **DualVLN 的 VLN-CE 自报数字**（R2R val-unseen NE 4.05/SR 64.3/SPL 58.5、RxR SR 61.4 等）→ 1-2 否决，须查官方 leaderboard。
2. **"LH-VLN 提出 ISR/CSR/CGT 三指标"** 的表述 → 0-3 否决（LH-VLN 协议另含 SPL/OSR/NE，ISR/CSR/CGT 非其独有提出）。

## ⚠ 数字使用纪律

- FantasyVLN 2.44%、VLingNav 全部数字、NavDP 的 ClutteredEnv/InternScenes 数字均为自报；NavDP 数字虽有两篇第三方论文复现（SanD-Planner 2602.00923 表 I、SIDP 2601.22965 表 II），但用的是同一 InternRobotics benchmark，非完全独立。
- 许可证：NavDP / InternData-N1 为 CC BY-NC-SA 4.0 非商用；DualVLN 代码 MIT 但数据非商用。

## 待补缺口（三轮后的剩余项）

1. ~~RELOC~~ ✅ / 2. ~~场景图~~ ✅ / 3. ~~语义可控性协议先例~~ ✅（见 §三）；
4. VLN-CE 官方 leaderboard 2026 可引用 SOTA（DualVLN 自报数字被否决后未补）；
5. NoMaD/NavDP 真机章节的定量协议细节（episode 数/统计口径）；
6. Diffusion Policy / DiffusionDrive / NoMaD 原文的多模态性实验精读（见 §3.7）；
7. 语义可控性"空白主张"投稿前需对 2025-2026 新作做一次查漏（本轮检索范围之外可能有新工作）。
