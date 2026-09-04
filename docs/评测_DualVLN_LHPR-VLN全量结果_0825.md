# DualVLN × LHPR-VLN 全量评测结果(0825 落盘)

> 评测链搭建见 0824 记录。本文件是全量跑分终版数字。
> 数据:`benchmarks/LH-VLN/dualvln_eval/results_full_b678.jsonl`(436 条有效)
> 复算脚本:`benchmarks/LH-VLN/dualvln_eval/compute_metrics.py`(官方 NavigationMetrics 原样复算)

## 覆盖情况(0825 补跑后终版)

- 计划 444 ep(batch_6 val 41 + batch_7 test 166 + batch_8 test 237),**444 全跑完**(merge: `results_merged.jsonl` = 全量 441 + 补跑 `run_topup.py` 8 条)
- 排除 7 条(数据侧不可跑,任何方法同):
  - **5 条目标对象真值缺失**: batch_6/15('box'), batch_7/21('lamp'), batch_7/63('device'), batch_8/60('pen'), batch_8/98('candle') —— 对象在 semantic.txt 有(hex 唯一),但 semantic.glb 顶点 COLOR_0 里**根本没有该颜色**(最近色差 9/27,非 sRGB 舍入)= HM3D 语义网格缺小物件几何, obj_centers 无法提取 → get_coord 空 → 官方代码 tuple 崩
  - **2 条 GT 测地 inf**: scene 00567-KjZrPggnHm8 两条,首子任务起点/目标不在同一 navmesh 连通域
- **最终有效 n=437**(val 40 / test 397)

## 最终指标(官方口径,百分制)

| 指标 | DualVLN(全437) | val(b6,n=40) | test(b7+8,n=397) | FantasyVLN(论文) | MGDM(作者自家) |
|---|---|---|---|---|---|
| SR | **0.00** | 0.00 | 0.00 | 2.44 | 0.00 |
| OSR | 0.00 | 0.00 | 0.00 | — | — |
| SPL | 0.00 | 0.00 | 0.00 | — | — |
| NE (m) | 10.93 | 11.45 | 10.87 | — | — |
| ISR | **2.98** | 6.54 | 2.62 | 11.01 | 低 |
| CSR | 1.43 | 2.27 | 1.34 | 9.64 | 低 |
| CGT | 1.28 | 2.56 | 1.15 | 8.99 | 低 |
| TAR | 22.20 | 24.35 | 21.98 | — | — |

## 口径 caveat(写作必须带)

1. **转角不匹配**: DualVLN 按 15° 转角训练,本评测保持 LH-VLN 官方 30° 动作空间(保与论文可比),转头过冲是预期性能损失。
2. **无操作能力**: LH-VLN 任务含 Grab/Release 子任务(共 1154 个),DualVLN 纯导航;successes 列表只含 Move_to 导航子任务(1170 个,ISR 分母即此),与官方 eval 代码同口径。
3. **SR=0 不丢人**: 该 benchmark 上除 FantasyVLN(2.44)外全部方法(含作者自家 MGDM)SR=0;区分度在 ISR/CSR/CGT/TAR/NE。
4. DualVLN ISR 2.99% << FantasyVLN 11.01% —— 长程多子任务序贯场景下 DualVLN(VLN-CE R2R 单指令风格)明显吃亏,这正是我们方法(语义可控长程)的对标空间。

## 结论

- DualVLN 在 LHPR-VLN 上 SR=0、ISR≈3%,与"全场 SR=0、FantasyVLN 最好 ISR=11"的榜单格局一致 → **评测链可信**(没有口径错误导致的异常好/坏)。
- 我们方法进场后的对标:第一格比 ISR/CSR/CGT(导航口径),SR 破 0 即是卖点。
- 下一步:我们方法移植(our_eval,另开文档)。
