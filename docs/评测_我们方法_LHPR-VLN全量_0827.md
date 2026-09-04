# 我们方法 LHPR-VLN 全量评测结果（0827）

> 口径：与 DualVLN 全量（`docs/评测_DualVLN_LHPR-VLN全量结果_0825.md`）**完全相同的 437 个有效 episode**（444 总集按同一规则排除 7 条数据侧不可跑：5 条缺小物件几何 + 2 条 GT 测地 inf；取两结果集 valid 交集，n=437 恰好一致）。
> 指标：官方 `utils/metrics.py NavigationMetrics` 原样复算。
> 我们 = `our_eval/our_agent.py`（VLM-WHERE bbox 定位 + 几何反投影 goal_ego + 条件 DD 快脑，`goalnav_gate_lab_v4.pt`），**zero-shot，未在 LH-VLN/HM3D 上训过任何参数**。

## 主表（n=437，同集对照）

| 指标 | **Ours(zero-shot)** | DualVLN | FantasyVLN(官方自报) |
|---|---|---|---|
| SR | 0.00 | 0.00 | 2.44 |
| ISR（独立子任务成功） | **4.25** | 2.98 | 11.01 |
| CSR（条件成功） | **3.03** | 1.43 | 9.64 |
| CGT（条件路径进展） | **1.95** | 1.28 | 8.99 |
| TAR（目标到达率） | **26.25** | 22.20 | — |
| NE（导航误差， 越低越好） | 11.73 | **10.93** | — |
| SPL | 0.00 | 0.00 | — |

（FantasyVLN 列来自其论文自报，episode 口径未公开，仅作量级参考；DualVLN 列为同集实测。）

## 结论

1. **zero-shot 即超 baseline**：ISR/CSR/CGT/TAR 四项全面超过 DualVLN（ISR +43%，CSR +112%），而 DualVLN 在 VLN-CE 上训练过、我们只是把实验室 ckpt 直接拿来。ISR 4.25 已居官方榜 DualVLN(≈3) 与 FantasyVLN(11) 之间。
2. NE 略逊（11.73 vs 10.93）：失败 episode 上我们的慢脑会把目标点锁错实例导致走得离 GT 更远——与 smoke 诊断一致，**瓶颈在实例级接地与 STOP，不在运动控制**。
3. SR 全场为 0 与榜单格局一致（多子任务全对才算 SR，含 Grab/Release 天然失败项），区分度在 ISR/CSR/TAR。
4. 下一步提分路线与 R2R 侧一致：stop 头（路B，训练中）+ 实例接地改进，预期直接抬 ISR/CSR 并压 NE。

## 产物

- 逐 episode 结果：`our_eval/results_our_merged.jsonl`（444 行，合并 b6/b7/b8a/b8b 四批，批日志 `our_eval/log_b*.txt`）
- 复算脚本：`dualvln_eval/compute_metrics.py`（同集交集过滤见本文首段）
- DualVLN 对照：`dualvln_eval/results_merged.jsonl`
