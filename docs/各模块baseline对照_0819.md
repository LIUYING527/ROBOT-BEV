# 各模块 Baseline 对照表（2026-08-19，deep-research 106agent 核实版）

> 来源：deep-research 工作流（24 源 → 114 claims → 25 条三票对抗验证，23 过 2 杀）+ 补充搜索。
> 口径纪律：✅=三票通过可引用；⚠=存在但数字待核实；❌=被对抗验证否决**禁止引用**。

## 总表：我们的模块 ↔ 最新 baseline

| 模块 | 我们 | 最新 baseline | 对我们 novelty 的含义 |
|---|---|---|---|
| **①整体双系统** | VLM慢脑+扩散快脑+NavCmd | **DualVLN**(2512.08186, ICLR2026)✅；**Qwen-RobotNav**(2606.18112, 2026-06：规划LLM+Qwen3-VL+**4层MLP回归**8航点)✅；**VLingNav**(2601.08665, 字节/北大：AdaCoT单模型自适应快慢思维+VLingMem文本摘要记忆)✅；**FantasyVLN**(2601.13976, CVPR2026：VAR潜空间隐式CoT一体化7B)✅ | 三家最新竞品**全无扩散快脑、全无3D几何记忆、全无语义可控条件机制** → 差异化空间仍在 ✅ |
| **②扩散导航策略** | DiffusionDrive室内迁移+cond_embed+scene_gate | 谱系：**NoMaD**(2310.07896, ICRA2024最佳论文)→**NavDP**(2505.08712, 纯仿真RGB-D+critic特权信息轨迹筛选)→**X-NavDP**(2607.28560, 2026-07：GQRM扩散RL后训练，仿真61.2→84.3%，真机hard cases 10→65%)✅；正交竞品**NaviDiffusor**(2504.10003, ICRA2025：推理期可微代价梯度引导)✅；移动操作**M4Diffuser**(2509.14980, ICRA2026)✅ | NavDP的critic轨迹筛选可借鉴；NaviDiffusor是推理期引导vs我们训练期门控=正交互补可cite；X-NavDP是"快脑RL后训练"最新版 |
| **②基座** | DiffusionDrive | **DiffusionDrive**(2411.15139, CVPR2025, NAVSIM 88.1 PDMS, 截断扩散2步)✅；注意NAVSIM 2026已到~91-92(**DiffusionDriveV2** 2512.07745 91.2)⚠ | 引用"扩散多模态优于回归"必须带限定：论文自证**普通扩散多样性仅11%**，多模态优势靠锚点+截断设计才兑现(74%) |
| **③GS渲染-对比定位(RELOC)** | 可微3DGS+Adam+Charbonnier+双闸门+周期纠偏 | **iNeRF**(2021, 祖师爷)→**iComMa**(2312.09031, WACV2024：免训练反演3DGS，比iNeRF快~10×; **关键消融：纯render-compare对恶劣初始化脆弱，去匹配模块失败率~9%→38-40%**)✅；后续**GSplatLoc**(2412.20056 "Ultra-Precise")⚠、**GSVisLoc**(2508.18242 泛化版)⚠、**SGLoc**(2507.12027)⚠、**iGaussian**(2511.14149)⚠、CVPR2026 "Rethinking Pose Refinement in 3DGS under Pose Prior"⚠ | iComMa的消融直接支撑我们双闸门+收敛域设计；我们的差异化=**周期纠偏闭环+无里程计motivation+真机zed域**，他们全是"一次性定位"benchmark设定。后续各篇精度数字待逐篇核实（arxiv抓取被墙） |
| **④开放词汇3D场景图** | Grounded-SAM逐帧+跨帧合并185节点+CLIP指纹RAG | **ConceptGraphs**(ICRA2024)✅；后续：**HOV-SG**(RSS2024, 层次化楼层/房间/物体，长查询超CG~11.7%)✅；**DovSG**(2410.11989, 动态更新，导航95.1% vs OK-Robot 82.8%)✅；**OpenIN**(RA-L2025, 实例导向动态家居)；2026新潮：Hydra++(IROS2026)/PSG-Nav(ICML2026)/SuperMap(RSS2026)⚠ | 领域趋势=层次化+动态更新+概率化；我们的差异=场景图**只当RAG几何挂载**服务精定位链，不是通用建图贡献 |
| **⑤VLN benchmark** | 计划VLN-CE/LH-VLN | **VLingNav**: HM3Dv1 79.1SR/42.9SPL(超Uni-NaVid 73.7/37.1)、HM3Dv2 83.0SR、MP3D 58.9SR，零样本Go2真机~2.5FPS✅；**FantasyVLN**: LH-VLN SR 2.44/ISR 11.01（绝对SR仅2.44%=长程远未解决，印证我们判断）✅ | ⚠口径更正：基准名是**LH-VLN不是LHPR-VPN**（0806笔记误写）；HM3Dv2上VLingNav的SPL(40.5)其实略低于FiLM-Nav(41.3)，SOTA仅SR成立——引用须精确到指标 |
| **⑥精密对接** | min(深度,几何)伺服+EDGE对准 | 补调结果：**该赛道没有公认的学习类SOTA**，2025-26仍以经典视觉伺服/PID滑模为主（个别gantry docking、深度视觉伺服网络）| 对我们其实有利：厘米级对接+RELOC的组合在导航圈是稀缺能力，竞争不拥挤 |

## ❌ 被对抗验证否决的声明（透明记录，禁止引用）

1. **Qwen-RobotNav 的 VLN-CE SOTA 数字**（R2R 72.1% SR、RxR 76.5% 等）→ 0-3 否决。论文架构描述可信，**benchmark 数字勿引用**。
2. **"NoMaD 真机显著超越 5 种方法且模型更小"** → 0-3 否决。

## ⚠ 数字使用纪律

- VLingNav/X-NavDP/NaviDiffusor/M4Diffuser 的 SOTA 数字**全部为作者自报、无第三方复现**；X-NavDP 真机 10→65% 仅限 hard cases 子集；投稿前须复核官方榜单。
- FantasyVLN"延迟降一个数量级"实测约 5.4×（0.19→1.03 APS）。
- iComMa"快约10倍"实测 11-15×（保守方向，可引用）。

## 待补缺口

1. GSplatLoc/GSVisLoc/iGaussian/SGLoc 各自精度数字（arxiv 全文抓取被本机网络挡，需代理或本地 PDF）；
2. NAVSIM 2026 SOTA（DiffusionDriveV2 91.2）方法细节与我们基座的差距来源；
3. VTM-Nav(2607.14514) 是否在 HM3D ObjectNav 超 VLingNav（出现但无法核实）；
4. 场景图 2026 新潮（Hydra++/PSG-Nav/SuperMap）仅题名级，未逐篇核实。
