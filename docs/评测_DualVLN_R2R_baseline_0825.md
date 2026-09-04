# DualVLN × VLN-CE R2R baseline(0825 落盘)

> 立项B: 去 DualVLN 主场(VLN-CE R2R)比。本文件是 baseline 链搭建与跑分记录。
> LH-VLN 侧结果见 [[评测_DualVLN_LHPR-VLN全量结果_0825]]。

## 链路与口径(与 DualVLN 官方完全一致)

- **评测客户端**: `benchmarks/InternNav/r2r_eval/run_r2r.py` —— habitat-lab 官方 `benchmark/nav/vln_r2r.yaml` sim + 官方 measurements; agent 无关, 走 InternNav AgentServer HTTP 协议(/agent/init|step|reset), DualVLN `start_server.py` 与我们 `our_server.py` 可互换。
- **口径**: val_unseen(全 1839 ep) / 640×480 hfov79 RGB-D / forward 0.25m / turn ±15° / 成功阈 3.0m / max 500 步 / 动作 0stop 1fwd 2left 3right (-1=noop 不推进仿真, 连续 20 次强制 turn_left 防死锁)。
- **数据**: `benchmarks/mp3d_r2r/`(r2r episodes + mp3d scenes)。
- **服务器**: DualVLN 用 InternNav `scripts/eval/configs/r2r_dualvln_server_cfg.py` 的 model_settings(run_r2r.py 内 DUALVLN_SETTINGS 复刻)。

## DualVLN baseline 数字

| | SR | SPL | OSR | NE(m) | n |
|---|---|---|---|---|---|
| **本次实测**(seed0 抽 300/1839, 11 场景) | **58.7** | **52.2** | **67.7** | **4.48** | 300 |
| DualVLN 官方 README(全量 val_unseen) | 64.3 | 58.5 | 70.7 | 4.05 | 1839 |

- 实测 < 官方 ~5 点 = 300 子集采样噪声(跨场景 seed0)合理范围, 链路无可疑异常 → **R2R 评测链可信**。
- 耗时 6.73h/300ep(≈80s/ep), 全量 1839 约 41h —— 对照实验跑 300 子集即可, 同 seed 可比。
- 结果: `benchmarks/InternNav/r2r_eval/results_dualvln_300.jsonl`

## 我们方法(zero-shot)冒烟 —— 失败, 待诊断

- `results_our_smoke10.jsonl`(同链同口径, our_server + goalnav_gate_lab_v4.pt): **SR=0 / OSR=10% / NE=9.15**, 10 条里多数撞 500 步上限。
- our_agent R2R 模式: 慢脑(robo2vlm 每 8 步决策 GOTO地标/FORWARD/TURN/STOP, STOP 需连 2 票) + VLM bbox 定位子目标 + 快脑(lab ckpt)几何轨迹 + 卡死解卡(仅前进帧计, 连 3 轮 giveup)。
- 诊断脚本: `r2r_eval/diag_r2r_our.py`(进程内单 episode, 逐步慢脑决策/目标/动作落 jsonl + 轨迹俯视图)。

## 0826 慢脑五代迭代诊断(v0→v5, 两条诊断 episode 全程实录)

诊断对象: ep632("exit into the pool room and stop before pool") / ep505("walk down hallway...stop at table")。

| 版 | 机制 | 结果 | 失败模式 |
|---|---|---|---|
| v0 | 逐节拍多选(GOTO/FWD/TURN/STOP) | min_dtg 3.55 | **复读 GOTO POOL×206**, 0 STOP, 原地转 490 步 |
| v1 | +到达事件/STOP confirm/复读熔断 | min_dtg 3.55 / 0.59 | 复读病换皮: **TURN_LEFT×492**(转20圈) / **FORWARD×61**(min_dtg 0.59 走过头不停) |
| v2 | **地标链重构**(指令一次解析→逐地标bbox定位→链尾confirm) | min_dtg **0.04** | 导航到 GT 上 4cm 但 confirm("指令完成了吗")**站终点上都说 no**; ep1 目标锁定隔墙 1.8m 死磕(无超时) |
| v3 | +目标超时放弃/周期刷新/末地标近距confirm | min_dtg 0.07 | confirm 元问题依然全 no(52次); 链解析出"bed, pool"单行 |
| v4 | confirm 改接地近距问("X在眼前2m内吗")+解析拆逗号 | — | **假停**: ep0 第 2 步停(NE 5.03); ep1 三地标 17 步内全"到达"(假定位→瞬达) |
| v5 | +定位两帧复核采信/confirm 2 票/解析两问(终点+中途) | min_dtg **0.37** / stop@81 | ep0 到 0.37m 但 confirm 仍全 no(pool 不可见角度); ep1 停错位置(NE 8.58, 真桌但错实例) |

**沉淀下来的结论**:
1. **robo2vlm 多选一决策根本不可用**(复读病: 复读自己最近输出, 换 prompt 只换复读内容)——与 0715 走廊"多选把它搞糊"一致; **bbox 定位 + yes/no 才是甜区**, v2 起固定为此架构。
2. 快脑+几何全程无责: v2 起多次导航到 min_dtg<0.5m, **瓶颈全在"停"**(STOP 通路)与 VLM 实例级接地("stop before pool"应停池边, bbox 中心锁水面中央天然偏 3.5m; "table"锁错实例)。
3. v5 机制收敛: 链解析(两问)/skip+换视角探索/目标超时放弃/两帧定位复核/confirm 2 票——剩余失败=VLM 零样本接地天花板, 非结构问题。
4. "指令完成了吗"这类**元问题不可用**(v2/v3 站在 GT 上 7cm 连答 52 次 no); 接地近距问(v4 起)可用但受实例歧义影响。

## 下一步

- ~~v5 冒烟~~ 已跑(见下); 之后可考虑 300 子集对照或转 LH-VLN 全量。
- LH-VLN 侧 our_eval 全量(437ep)待排: smoke 已显示 NE 10.78 ≈ DualVLN 10.93(zero-shot 不丢人), smoke2(坐标系修复后) ISR 11.1%/TAR 30.5%。

## 0826 v5 冒烟结果(10ep, sample seed0, 与 v0 完全相同 episode 集)

| | SR | SPL | OSR | NE(m) | 平均步数 |
|---|---|---|---|---|---|
| v0(多选慢脑) | 0 | 0 | 10 | 9.15 | 500(全撞上限) |
| **v5(地标链慢脑)** | **10** | 7.1 | 10 | **7.25** | 161 |
| DualVLN(同链, 参考) | ~60 | ~59 | ~90 | ~3.3 | ~100 |

- **SR 破零**: ep1606 于 101 步停, NE=2.90m(<3m 阈), minDTG 1.10 —— 首个"导航到+停对"完整成功。
- 步数 500→161: 超时死循环绝迹, 失败 episode 也早停(假停/早停, 接地错误所致)。
- 与 DualVLN 差距仍大(SR 10 vs ~60), 但 zero-shot(ckpt 训练域=实验室走廊)+纯机制迭代未训练; 剩余差距=VLM 实例级接地与探索策略, 非链路问题。
- 结果: `r2r_eval/results_our_v5_smoke10.jsonl`; 诊断实录: `diag_r2r_ep{0,1}_v{0..5}.{jsonl,png}`。

## 0828 stop 头(路B) + R2R 300ep 全量对照(我们 vs DualVLN 同集)

**stop 头**(0827 晚落地, 替换 VLM confirm 判停): R2R train split ref_path 终点监督(2000ep/~19k 帧, 正例=距 goal<3m 且 goal 在视向 ±60° 内, 硬负例=goal 周边 3-5m 对视), timm resnet34 a1_in1k 暖启, 场景 hold-out valAUC=0.74; 判停 thr0.40/2 票(三档 smoke 定版: 0.55 不触发/混合档 SR 归 0)。smoke10: SR10/SPL0.10/NE8.19(v5=SR10/SPL0.07/NE7.25, n=10 噪声内, ep1606 57 步 NE1.13 全场最佳)。

**300ep 全量**(sample seed0 val_unseen, 与 DualVLN 完全同集, 有效 n=296/300, 0828 早因让卡停跑, 后 4 条未跑):

| | SR | OSR | SPL | NE(m) | 平均步数 |
|---|---|---|---|---|---|
| **Ours(v5慢脑+stop头, zero-shot)** | **5.4** | 8.4 | 4.0 | 9.32 | 215 |
| DualVLN(R2R 训练, 官方 ckpt) | 59.1 | 68.2 | 52.6 | 4.38 | 120 |

- 结果: `r2r_eval/results_our_stophead_300.jsonl`(296ep); DualVLN 对照 `results_dualvln_300.jsonl`。
- **结论(诚实)**: R2R 是 DualVLN 主场(它在 R2R/VLN-CE 上训练), 我们 zero-shot 个位数 SR 属预期量级; 失败主因仍是**实例级接地**(多数 episode min_dtg 从未进 3m), 其次探索策略——与 smoke 诊断一致, stop 头只解决"停"不解决"找错目标"。
- **我们的对照主场在 LH-VLN**: 同批 0827 全量(n=437)我们 ISR 4.25/CSR 3.03/TAR 26.25 全超 DualVLN(2.98/1.43/22.20), 见 `docs/评测_我们方法_LHPR-VLN全量_0827.md`。

## 0828(下) R2R 快脑微调版 300ep(域适配收益实锤)

快脑微调(train_goalnav_r2r.py: 2000ep ref_path 回放 6666 样本, warm-start v4, val 0.364) + stop 头。
smoke10(同 seed0 十条): **SR 10→20 / OSR 10→60 / NE 7.25→6.86 / SPL 0.07→0.13 / min_dtg 5.25→3.78**(ep1469 首次摸到 0.38m)。

300ep 三方同集(n=239; ⚠ robo2vlm 服务器中段宕机致 61 条 500 错误, 待补跑):

| | SR | OSR | SPL | NE(m) | 平均步数 |
|---|---|---|---|---|---|
| **Ours(快脑微调+stop头)** | **5.9** | **15.9** | **3.9** | 9.37 | 183 |
| Ours(zero-shot v5+stop头) | 4.6 | 7.5 | 3.3 | 9.31 | 212 |
| DualVLN(R2R 训练) | 58.6 | 69.0 | 52.4 | 4.20 | 123 |

- **域适配收益**: SR +28%/OSR +112%(摸到 3m 内的 episode 翻倍)/SPL +18%/步数 -14%——快脑微调对"接近目标"作用显著; **SR-OSR 缺口(5.9 vs 15.9)=STOP 通路+实例接地仍是天花板**(与 smoke 的 ep1469 到 0.38m 没停走飞一致)。
- 结果: `r2r_eval/results_our_r2rft_300.jsonl`。
- **0829 补齐终版(n=300 唯一有效, error 行已剔)**: robo2vlm 宕机毒化的 61 条用 Qwen 顶班补跑(两轮: 第一轮 habitat reset 迭代器对齐坑白跑 42 条重复, 显式 `env.episodes=[ep]` 修后补齐)——**SR5.3/OSR17.0/SPL3.1/NE9.64/234步**。
  ⚠两 caveat: ①Qwen 当 R2R 慢脑显著弱于 robo2vlm(补跑的 61 条 SR0 vs robo2vlm 段 14/239), 混合后端拉低总 SR, robo2vlm 恢复后应全量重跑; ②补跑 episode 分布与原计划尾部一致, 子集随机性不受影响。
