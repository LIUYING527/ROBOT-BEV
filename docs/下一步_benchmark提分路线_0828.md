# 下一步：benchmark 提分路线（0828）

## 现状盘点（诚实版）

| 战场 | 我们 | 对手 | 判读 |
|---|---|---|---|
| LH-VLN ISR | **4.25**(zero-shot) | FantasyVLN 11.01(训练) / Aux-Think 3.16 / MGDM 2.34 / GPT-4+NaviLLM 2.19 | **已是公开榜第二**（仅次于训练过的 FantasyVLN） |
| LH-VLN TAR vs ISR | 26.25 vs 4.25 | — | **22 点转化缺口=最大金矿**：目标已经到了边上，stop/confirm 通路没转化成成功 |
| R2R SR | 5.4(zero-shot+stop头) | DualVLN 59.1（在该任务训练） | 主瓶颈=实例接地，其次快脑域差 |

关键事实核查（0828）：
- LH-VLN 数据 `episode_task/batch_1-8` 共 **1052 任务**；我们评测的 444(batch_6/7/8)=held-out 测试集，**batch_1-5(608 任务)就是官方训练资源**。
- 每条任务带 **oracle 轨迹**(`task/batch_*/.../trial_*/task.json` 的 `trial.trial_*.pos` 全位置序列+逐子任务 target/room/Geo_dis)——和 R2R ref_path 完全同构，**adapt 管线直接复用**。
- FantasyVLN 的 ISR 11 是训练过的数字（其跨模态对齐消融 SR0/ISR2.39 证明有训练），对齐它=我们也用 batch_1-5 训头部模块，公平同台。

## 提分路线（按杠杆率排序）

### P0 — LH-VLN 转化通路（TAR 26→ISR 11+ 的最短路径）
1. **LH-VLN stop/confirm 头**：teleport 回放 batch_1-5 oracle 轨迹；正例=距子任务目标 <1m 且目标在视向 ±60°（对齐官方 1m+60°FoV 判据），硬负例=1.5-3m 对视目标。替换现在的 `vlm_confirm`（白桌掷硬币级不可靠）。
2. **到位后重确认逻辑**：confirm 否决≠丢弃目标重扫（现在一否就丢缓存），改为绕行小步逼近重试 2 次再放弃——直接抬 ISR/CSR。
3. 目标：ISR 4.25→≥8，CSR 3.03→≥6，咬住 FantasyVLN 的 11。

### P1 — R2R 快脑微调（域适配）
- 数据已在盘上（adapt_data/s0-s3, 2000ep 含 ref_path）；写 `train_goalnav` 同管线的微调脚本：goal_ego=前瞻路点，gt_traj=ref path 未来段。
- 预期 SR 5→10~15、NE 9.3→6~7；**治不了接地**（天花板仍在）。

### P2 — 实例接地（两战场共同天花板，最贵也最重要）
- 路线 A（轻）：多视消歧——锁定候选后绕 ±30° 两视角复核，不一致降级为"继续搜索"；
- 路线 B（重）：接地头——用 batch_1-5/R2R train 的 (instruction 子句, 目标对象真值, 轨迹) 训"当前视图是否存在 X"分类头，辅助 VLM bbox 的实例选择；
- 路线 C（借场景图）：ConceptGraphs 节点库做实例枚举+消歧（和实验室场景图线合流，但是另一篇的内容，注意边界）。

### 执行约束
- 全部需要 1 张卡（数据回放+训练+重评）；**6/7 暂不可用**，等放行或 0-5 空出。
- P0 工作量：数据回放脚本~半天（复用 gen_adapt_data 骨架）+训练~1h+重评 437ep~半天。

## 叙事定稿（两条评测线的写法）
- **LH-VLN=主攻**：zero-shot 已第二 → 训头部模块后对标 FantasyVLN（"唯一无训练任务级策略、只训轻量头即达 ISR X"）；
- **R2R=诚实对照**：暴露接地短板+证明头部模块可吸收任务数据（zero-shot 与 adapted 双数字都报）；
- 卖点不变：语义可控扩散快脑+几何接地+可训轻量头，**不靠训练任务级大策略**。
