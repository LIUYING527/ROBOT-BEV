# 术语表:我们项目 × Benchmark × Baseline(0825)

> 用途:项目涉及的所有术语一张表。每条三列:**术语 | 是什么 | 在我们项目里的角色**。
> 定义只给到"够用",不知道的自己上网搜;重点在第三列——它在我们项目里干什么。
> 分六区:A 我们系统内部 / B Benchmark / C 指标 / D Baseline 与竞品 / E 数据·仿真·硬件 / F 评测方法论黑话。

---

## A. 我们系统内部

| 术语 | 是什么 | 在我们项目里的角色 |
|---|---|---|
| 双脑架构 (Dual-System) | 慢脑(语义决策)+快脑(反应式控制)的分层架构,源自 Kahneman 双系统理论 | **我们的总架构**。⚠️架构本身 novelty 已死(DualVLN 撞车),我们卖的是架构上的"语义可控性"机制 |
| 慢脑 (System 2 / Slow Brain) | 低频(~0.1-2Hz)、用语义模型做决策的一层 | VLM 担任:识别目标、给出 WHERE(去哪)和何时停 |
| 快脑 (System 1 / Fast Brain) | 高频(~10-30Hz)、直接从观测出运动的一层 | DiffusionDrive 担任:吃 BEV+图像+status,扩散出 8 点轨迹 |
| DiffusionDrive | 车端自动驾驶的扩散轨迹规划模型(NAVSIM 上 SOTA 之一) | 快脑的基座。我们把它从车端 AD 迁移到任务驱动机器人,权重 NAVSIM 暖启 |
| NAVSIM | 车端规划 benchmark/数据集 | 快脑预训练数据来源(ckpt 暖启);BEV 输入格式(±32m 点直方图)也从它继承 |
| Transfuser / transfuser_model_v2 | 图像+BEV 融合的 transformer 骨干,DiffusionDrive 的主干 | 快脑网络本体;v2 是我们改过的版本(加了 global_cond/status 通路) |
| BEV (Bird's Eye View) | 俯视图表示;这里特指**点云高度直方图**(非图像!) | 快脑两大输入之一。官方实现 `points_to_bev_tensor`(±32m,256×256)。⚠️±32m 是车端继承,室内过大,**没改,别声称改了** |
| status(8 维) | 快脑的条件向量:[0:2]=目标点 ego,[2:4]=目标朝向,[4]=vx,[5]=hist_w,[6]=gamma,[7]=预留 | 我们的"观察策略参数化"入口;真机部署时定义不同(真机是 onehot+vxvyaxay),是部署两大 gap 之一 |
| hist_w / gamma | BEV 时序历史:W=窗口长度,γ=逐帧衰减系数 | 我们的论文级创新点之一(BEV-only 时序记忆,官方架构零改动)。旧 ckpt 必须配 `--hist_w 0` |
| scene_gate(命令门控场景) | 零初始化门控:g=σ(gate(status)+4),让模型学会"原地转时关掉场景推力" | 语义可控性的机制铁证(转向门 0.98→0.046)= 论文卖点 |
| plan_anchor / kmeans anchors | 扩散轨迹的锚点先验(kmeans 聚类出的典型轨迹) | 我们把车端锚点换成机器人 kmeans 锚点(`robot_kmeans_anchors`) |
| traj_to_control | 轨迹→(vx,vy,wz) 的转换 | 对接我们 4 轮全向底座(svtrobot)的控制接口;横向分量当横移不当转弯 |
| WHERE / HOW 分离 | WHERE=去哪(语义+几何定位),HOW=怎么走(行为风格) | 我们任务分解的第一性原理。慢脑指位置+语义条件都给,只给位置就成 DualVLN 翻版 |
| 语义可控性 (Semantic Controllability) | 同一观测下换语义指令→轨迹分布可度量地不同 | **我们的核心 novelty**(赛道空白)。评测协议骨架:ControlNet++ 式回读 + P/R 线 |
| 多模态轨迹 (Multi-modal Trajectory) | 扩散模型对同一观测可采样出多条不同合法轨迹 | 快脑用扩散而非回归的核心理由(回归会 mode averaging 出鬼轨迹) |
| RELOC(重定位) | 用 3DGS 渲染对拍纠正累计位姿漂移 | 高精度导航支柱;真机无 odom 时周期纠偏正是卖点。双闸门:步长门+渲染验证 loss 门,GS 兜底 |
| 回归头 (Regression Head) | 直接回归精确站位/朝向的小网络( docking 1.18cm 级) | 精定位 20% 的候选方案;v4 双域训练(真 zed+sim) |
| 3DGS (3D Gaussian Splatting) | 用一堆 3D 高斯表示场景、可实时渲染新视角 | 我们场景的"皮肤"和 RELOC 的地图;gsplat 库 |
| 双层仿真器 | 干净几何物理层 + 3DGS 可切换皮肤层 | 我们的仿真哲学:几何进导航(可信),生成只进皮肤(喂 VLM),生成幻觉不进几何 |
| DISCOVERSE | 上一代 MuJoCo 仿真平台(可导入 3DGS 资产) | 实验室闭环仿真器,lab_mission 跑在里面 |
| lab_mission | 实验室任务执行器(状态机+RELOC+精对准) | 我们方法的"完整系统"形态;跑它必须 discoverse_venv + `--targets zed_front_perp_lvl.json` |
| locate_target / 场景图 RAG | ConceptGraphs 式节点语义图 + RAG 检索定位 | 替代手搓几何的定位路线(节点多视三角化精化);另一篇文章的料 |
| nav_ds_real | 从真实采集按真实运动分类的数据集(cruise+横扫) | 快脑训练数据铁律:GT 用真图真实未来轨迹,不许渲染/扰动造假 |
| DFS 骨架搜索 (search v5) | 死胡同回退 GT 的深度优先搜索行为数据 | 搜索/探索能力来源;R2R 慢脑"找不到目标"短板正需要它 |
| mode bit / 条件位 | status 里的模式开关位 | 教训:占比 13% 会"休眠",要 ×3 过采样;goal=0 本身就能当开关(位冗余) |
| NavCmd | 结构化参数包 nav(L,τ,Φ) | 论文改进点⑧:指令→参数化导航命令 |
| confirm 四段推理链 | 到位确认拆四段独立判断(不串联) | 论文⑩;配合⑪ key-frame 复核(≥2 帧 yes 才翻案)治 VLM 掷硬币 |
| our_server / run_r2r.py | 我们的标准 agent 服务器 + agent 无关 R2R 客户端 | "标准化"产物:InternNav 协议兼容,任何评测写薄客户端即可接入(见 D 区 InternNav) |
| 采集天花板 | 单前视 ZED 单程采集缺横向视角,重建糊是物理下限 | 重建线血泪结论:4 组受控实验证伪调参救赎,唯一解=重采绕拍 |

## B. Benchmark

| 术语 | 是什么 | 在我们项目里的角色 |
|---|---|---|
| LHPR-VLN (LH-VLN) | 长程多子任务拾取-放置 VLN benchmark(HM3D 场景,指令="把A拿到B再取C") | **评测主场一**(语义可控长程叙事对口)。SR 仅 2.44% 远未饱和 |
| VLN-CE | Vision-and-Language Navigation in Continuous Environments(连续空间 VLN,habitat 里跑) | **评测主场二**,DualVLN 的官方主场。用户原则:跟 X 比要去 X 参加过的评测 |
| R2R (Room-to-Room) | 最经典 VLN 数据集,英文路径指令,MP3D 场景 | VLN-CE 里的主赛道;我们对标 DualVLN 的战场(val_unseen 1839 条) |
| RxR | R2R 的多语言加大版(英/印地/泰卢固) | InternNav 也报了数字,备选第二赛道 |
| val_unseen / val_seen / test | VLN 标准划分:没见过/见过的场景/官方测试 | 我们跑 val_unseen(泛化能力标准照妖镜) |
| GOAT-Bench | 多模态目标导航 benchmark(类别/描述/图像目标) | 调研定的备选第三评测,未开工 |
| HM3D (Habitat-Matterport3D) | 1000 个真实房屋扫描,habitat 格式 | LHPR-VLN 的场景库(评测用 153 场景);免 token 镜像 `zhuhu00/hm3d` |
| MP3D (Matterport3D) | 90 个真实建筑扫描(老数据集,R2R 的场景) | VLN-CE R2R 的场景库;镜像 `Believe0029/MP3D_hab` |
| habitat-sim / habitat-lab | Meta 的仿真器(C++)与任务/评测框架(Python) | 两个 benchmark 的官方运行环境;lhvln env=sim 0.3.1+lab 0.3.0 |
| navmesh | 可行走面网格,测地距离/合法移动都靠它 | 评测地基:目标点 snap 上 navmesh 才算数;scene 00567 连通域断裂=GT inf |
| episode / subtask / stage | 一条任务 / LH-VLN 里的每个"取-放"小任务 / 仿真器里当前子任务序号 | 评测最小执行单元;我们 agent 按 stage 切目标 |
| Grab / Release 子任务 | LH-VLN 里的抓取/放下动作子任务 | ⚠️纯导航方法(我们和 DualVLN)天然做不了,官方 successes 只含 Move_to 导航子任务,同口径可比 |
| 测地距离 (geodesic distance) | 沿可行走面的最短路径长度(区别于直线欧氏距离) | LH-VLN 的 GT 路径与成功判定都用它 |
| 动作空间 (action space) | stop/forward 0.25m/turn ±15°或 ±30°/look up-down | 口径的一部分:VLN-CE=15°,LH-VLN 官方=30°(DualVLN 按 15° 训,跑 30° 吃亏是 caveat) |

## C. 指标(看懂榜单用)

| 术语 | 是什么 | 在我们项目里的角色 |
|---|---|---|
| SR (Success Rate) | 最终停在阈值内的 episode 比例(VLN-CE 阈 3m,LH-VLN 阈 1m+60° FoV) | 头号指标。LH-VLN 上全场≈0(FantasyVLN 2.44),**SR 破 0 即是卖点** |
| OS / OSR (Oracle Success) | 轨迹任意时刻进过阈值的比例(不要求停对) | 区分"找不到"与"找到但没停对";R2R 官方口径之一 |
| NE (Navigation Error) | 结束时到目标的测地距离(米) | 越低越好;DualVLN R2R 4.05 / LH-VLN 10.93 |
| SPL (Success weighted by Path Length) | 成功×(最短路径/实际路径),奖又快又准 | VLN 经典效率指标 |
| nDTW | 归一化动态时间规整,整条轨迹与参考路径的形状贴合度 | RxR 主指标(路径跟踪类任务用) |
| ISR (Independent SR) | 每个子任务独立算成功的比例 | LH-VLN 区分度指标(FantasyVLN 11.01 / DualVLN 2.98) |
| CSR (Conditional SR) | 考虑子任务顺序依赖的成功率(前面失败连坐) | LH-VLN 长程序贯性指标 |
| CGT | CSR 按 GT 路径长度加权版 | 同上,长任务权重更高 |
| TAR (Target Approach Rate) | 逐子任务"接近目标程度"(1-超出 1m 的误差占比) | LH-VLN 指标,没到达也有部分分;我们 smoke2 30.5% |
| PDMS | NAVSIM 的驾驶综合分(无碰撞/可行驶区/进度等加权) | 快脑预训练域(NAVSIM)的指标,论文背景板 |
| oracle / GT (ground truth) | 上帝视角真值(最短路径、真实目标位置) | 评测对照基准;⚠️真值口径三连修过(.scn 硬链/COLOR_0 提中心/z-up 转换) |

## D. Baseline 与竞品

| 术语 | 是什么 | 在我们项目里的角色 |
|---|---|---|
| DualVLN (InternVLA-N1) | 上海 AI Lab 双系统 VLN 基础模型(2Hz VLM 慢+30Hz DiT 快),VLN-CE SOTA,全开源 | **头号对标+架构撞车者**。我们已在它两个主场都有它的数字:LH-VLN SR0/ISR2.98,R2R SR≈60 |
| InternNav | DualVLN 的官方训练/评测平台 | 它的 AgentServer HTTP 协议被我们借来做标准化接口(our_server 兼容它) |
| FantasyVLN | 阿里 CVPR2026,LHPR-VLN 榜首(SR2.44/ISR11.01) | LH-VLN 上最强对手;ISR 11 说明长程远未解决 |
| MGDM | LH-VLN 论文作者自家模型 | 官方 baseline,SR=0,衬托赛道难度 |
| NavDP (Navigation Diffusion Policy) | 一个扩散导航策略(X-NavDP 真机 10→65%) | 可复现 baseline 备选;也是 InternVLA-N1 的 System1 变体 |
| Qwen-RobotNav | Qwen 系导航 SOTA(快脑是 MLP 回归非扩散,91.4 分灌 GT 先验,去掉掉 79.5) | 对照组:证明"回归做不到多模态"的论据来源 |
| CMA / Seq2Seq / RDP | VLN 经典三代 baseline(跨模态注意力/序列到序列/强化扩散策略) | VLN-PE 榜单背景板(SR 15-25 量级),衬托 SOTA 水位 |
| CLASH / VLingNav | 同期 VLN 竞品 | novelty 扫描里的赛道拥挤证据 |
| robo2vlm(Infinite,自研 VLM) | 我们自研远程 VLM(10.0.0.18:8152),0.42s/次 | **慢脑首选**(比 Qwen 快 4.3×、巡检召回高 18 点);⚠️别和本地 Qwen 混 |
| Qwen3.5-4B(本地) | 本地部署的小 VLM(127.0.0.1:8152) | 慢脑对照/兜底;对 prompt 极敏感(具体单名词是甜区) |

## E. 数据·仿真·硬件

| 术语 | 是什么 | 在我们项目里的角色 |
|---|---|---|
| ZED / ZED2 | Stereolabs 双目深度相机(逐帧度量点云 96-98% 有效) | 真机主力传感器;zed GT 点=定位真值铁标准 |
| COLMAP / VGGT | SfM 位姿估计两套(经典/学习) | 重建链前端;COLMAP 要单相机模式,VGGT 曾 roll 漂移要 untwist |
| ArtiFixer3D | NVIDIA 生成式扩散修 3DGS 拖糊(SIGGRAPH2026) | 皮肤修复工具(只进皮肤层);已蒸馏出锐利皮肤 |
| MMK2 / svtrobot | 实验室全向移动底座,(vx,vy,wz) 指令接口 | 真机本体;动作接口与 DiffusionDrive 完美对齐 |
| Orin | NVIDIA Jetson 端侧算力 | 部署目标平台(robot_orig 已在 Orin 跑通 DiffusionDrive) |
| GSplatLoc | 用 3DGS 做相机重定位的方法(两篇同名论文,已实锤混淆) | RELOC 的技术参照 |
| ogm / occ(占据栅格) | 2D 可通行占据图 | A* 规划与 bev_safe 急停用;要用 A3D 皮肤同源几何,别手搓 |
| 110744 / 111450 / 191715 等 | 采集场次编号(实验室/室外/crab 横移) | 数据资产索引;110744=实验室主场景,191715=crab 横向采集 |
| 684k ply | 定版 3DGS 皮肤(gs_vggto_colmapjoint_all_684k) | 重建永久保存版;教训:产物带版本号别覆盖 |

## F. 评测方法论黑话

| 术语 | 是什么 | 在我们项目里的角色 |
|---|---|---|
| zero-shot | 不在目标域训练直接评测 | 我们跑两个 benchmark 的姿态(ckpt 训练域=实验室走廊) |
| sim-to-real | 仿真训练→真机迁移 | 护城河叙事:别人没有的真机轨道 |
| 闭环 (closed-loop) | agent 动作改变环境再观测的评测 | 两个 benchmark 都是;⚠️历史教训:静态偏置好≠闭环好 |
| 开环 (open-loop) | 给定 GT 轨迹回放打分 | NAVSIM 车端常用;我们不用 |
| 口径 (protocol/caliber) | 评测的输入/动作/阈值/指标定义 | **用户两原则**:①跟 X 比要去 X 参加过的评测 ②用某评测就跟该评测已有数字的方法比 |
| 冒烟 (smoke test) | 小样本快速验证链路没断 | 每条链先 3-10 条冒烟再上大数;历史多次冒烟救场 |
| domain gap(域差) | 训练域与评测域分布差异 | 三大已知域差:HM3D 渲染 vs 实验室 GS、zed 真图 vs 合成、roll 扰动 |
| server/client 双进程 | 模型服务器+评测客户端分离架构 | 双 env 隔离(internvla/lhvln 依赖冲突)的标准解法;铁律:服务器已载模型再 init 必 OOM,换客户端先重启 server |
| warm-start(暖启) | 用预训练权重初始化再微调 | NAVSIM ckpt→快脑;零初始化层保证暖启无损 |
| OOD (out of distribution) | 分布外输入 | 已知雷区:±45° 转角 OOD 不可靠,回归头出域=纯 GS 兜底 |
| floater / 离群高斯 | 重建里飘在半空的垃圾高斯 | 裁剪(--crop 轨迹 bbox)治"乱" |

---

## 全景:我们的评测矩阵(0825 现状)

| Benchmark | 状态 | 对手数字(同口径) | 我们的数字 | 角色 |
|---|---|---|---|---|
| LHPR-VLN(444ep) | ✅ 链通,DualVLN 全量已出 | FantasyVLN SR2.44/ISR11.01;DualVLN SR0/ISR2.98;MGDM SR0 | smoke 级 ISR 11.1%(3ep,勿当正式) | 主场一:长程+语义可控叙事 |
| VLN-CE R2R val_unseen | ✅ 链通,DualVLN repro 中 | DualVLN 官方 SR64.3/NE4.05;repro ≈SR60 | v0 冒烟中(机制通,质量弱) | 主场二:对方主场公平对决 |
| GOAT-Bench | 未开工 | — | — | 备选第三评测 |
| 实验室闭环(三站任务) | ✅ 日常回归场 | — | 三站全 YES 级别 | 内部迭代场,非对外数字 |
| 真机(Orin) | 部署方案已落盘 | — | — | 护城河,非论文数字 |

> 文档维护:新术语入场时追加;数字以 `docs/评测_DualVLN_LHPR-VLN全量结果_0825.md` 与 memory 为准。
