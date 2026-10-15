# r2r_eval — 快脑 R2R 评测/训练代码（留档）

**来源**
- `10.0.0.22:/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/*.py`
- `10.0.0.22:/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval/*.py` → `our_eval/`

**同步时间**：2026-10-15

**为什么要放这儿**：这批代码原本位于**上游开源仓库的 clone**（InternNav / LH-VLN）里、
且**未纳入版本控制**（`r2r_eval/` 在 InternNav 中是 untracked）。机器故障或误改就无处回溯。
本目录把它们纳入 `ROBOT-BEV` 仓库做版本留档。

## ⚠️ 这不是可直接运行的副本

代码内含**硬编码绝对路径**，指向上面的 22 机器位置，例如：

```python
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval')
```

**权威运行副本始终在 22 上**；本目录只做历史留档，不在本机执行。

## 1015 改动（快脑侧接入慢脑 latent）

- `eval_r2r_escape.py` — 新增 `PWS_HIDDEN_ONLY`（默认 `0` ⇒ 不设时逐位不变）
  - oracle 分支补发一次 HTTP、**只取 hidden**（目标仍走真值，隔离"推理时 latent 是否有用"）
  - 闭环分支仅在开关打开时给请求加 `want_hidden`（默认不加字段 ⇒ 请求体与原版逐字节相同）
  - `PWS_TEXT=1` 时用 hidden（base64/float16/2048 维）替代指令编码喂 `text_proj`；维度不符则回退并告警
  - 备份：`eval_r2r_escape.py.bak_1015_pre_hidden`
- `our_eval/our_agent.py` — `ck['trajectory_head']` 加载前加**形状过滤**（防 `size mismatch` 崩）
  - 必须位于 plan_anchor 先行重建**之后**，否则会误跳过 plan_anchor
  - 备份：`our_agent.py.bak_1015_pre_shapeskip`

**冒烟结果**（1015，`--oracle_goal 1 --sample 3 --n 3`，ckpt=`txtctx_ep3`）：**3/3 条轨迹分叉**，
日志 `[hid] 已接通: dim=2048 mean=-0.0053 std=0.4974`。

## 其他

- `*.bak_*` = 各次改动前的快照，用于回溯。
- **数据与日志不纳入**：同目录在 22 上还有 `detour_data/`（610G，21198 个子目录）和大量 `log_*.txt`。
