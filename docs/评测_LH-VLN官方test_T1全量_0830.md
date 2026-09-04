# LH-VLN 官方 test(batch_7+8) 我们方法 T1 全量结果(0830)

**口径**: 官方 split(configs/lh_vln.yaml: train=batch_1-5 / val=batch_6 / **test=batch_7+8**)。
n=397(403 排除 4 条测地 tuple 崩溃 ep + 2 条测地 inf, DualVLN 同规则同集)。
复算 = 官方 NavigationMetrics 原样(dualvln_eval/compute_metrics.py 同路径)。

## 同集三方对照

| 指标 | **Ours T1 (zero-shot)** | DualVLN (VLN-CE训练) | FantasyVLN (训练, 官方) |
|---|---|---|---|
| SR | **0.25** | 0.00 | 2.44 |
| ISR | **3.55** | 2.62 | 11.01 |
| CSR | **2.81** | 1.34 | 9.64 |
| CGT | **1.42** | 1.15 | 8.99 |
| TAR | **22.15** | 21.98 | — |
| NE | 11.75 | **10.87** | — |

## 要点

- **zero-shot 在官方 test 上全面超 VLN-CE 训练过的 DualVLN**: ISR +35% / CSR 翻倍 / CGT +23% / TAR 持平略优 / SR 1:0。NE 略逊(11.75 vs 10.87) = 锁错实例走更远的老问题。
- 与 FantasyVLN(在 LH-VLN 训练)差距仍大(ISR 3.55 vs 11.01)——诚实对照, 它是训练过的数字。
- 本跑为 **T1 接地头集成版**(head 优先定位, VLM 仅 OOV 7% 兜底, Qwen 单后端全程=干净无混合)。0827 的 n=437 是 6/7/8 合并口径, 本文档是**官方 test 单一口径**, 论文报数用本文档。
- 失败结构(与 val 一致): 77% 子任务 NE>5m = 房屋级探索/实例接地仍是主瓶颈; NE<1m 转化率高(几何直停可靠)。

## 产物

- 结果: `our_eval/results_t1_b78.jsonl`(405 行: 399 有效+5 error+1 final_metrics)
- agent: T1v3(grounding_head_lhvln.pt + vantage 探索 + 近距 absent 信头 + 子任务清缓存)
- 运行: GPU2, ~15h(0830 02:40 收), Qwen(127.0.0.1:8152)单后端
