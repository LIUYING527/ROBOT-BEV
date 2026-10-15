"""生成最佳快脑制品的 README.md + MANIFEST.md。用法: python3 _mk_best_readme.py <PKG> <CKSUM>"""
import glob
import gzip
import json
import os
import subprocess
import sys

PKG, CKSUM = sys.argv[1], sys.argv[2]
R = '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval'
D = '/data/DongBaorong/BEVLOOK/benchmarks/mp3d_r2r/r2r/val_unseen/val_unseen.json.gz'
by = {str(e['episode_id']): e for e in json.load(gzip.open(D))['episodes']}


def yspan(e):
    v = [p[1] for p in e['reference_path']]
    return max(v) - min(v)


def load(pat):
    rs = []
    for f in sorted(glob.glob(R + '/' + pat)):
        rs += [json.loads(l) for l in open(f)]
    for r in rs:
        e = by.get(r['episode_id'])
        if e:
            r['ys'] = yspan(e)
    return {r['episode_id']: r for r in rs}


def stat(dd):
    rs = list(dd.values())
    n = len(rs)
    return (100 * sum(r['success'] for r in rs) / n,
            100 * sum(r['os'] for r in rs) / n,
            sum(r['ne'] for r in rs) / n, n)


CONFIGS = [
    ('np2full_s*.jsonl', '**最佳**: v3 + `--v3_noprog 2`'),
    ('recheck_v3_s*.jsonl', 'v3 默认 `--v3_noprog 4`'),
    ('full_goalmlp_ep5_escape2_s*.jsonl', '旧 v2 脱困(0911 基线)'),
    ('r2rft_v3_s*.jsonl', 'r2rft ckpt + v3(反例)'),
]
rows, main = [], None
for pat, lb in CONFIGS:
    try:
        dd = load(pat)
        if not dd:
            continue
        sr, osr, ne, n = stat(dd)
    except Exception:
        continue
    if main is None:
        main = dd
    rows.append('| %s | %.1f%% | %.1f | %.2f | %d |' % (lb, sr, osr, ne, n))

# 单层 / 多楼层
split_rows = []
for name, lo, hi in [('单层 yspan<1m', 0, 1.0), ('多楼层 yspan>=1m', 1.0, 1e9)]:
    if not main:
        break
    sub = {k: v for k, v in main.items() if lo <= v.get('ys', 0) < hi}
    if sub:
        sr, osr, ne, n = stat(sub)
        split_rows.append('| %s | %.1f%% | %.1f | %.2f | %d |' % (name, sr, osr, ne, n))

readme = """# 最佳快脑 —— 0913 定版

## 结论

**val_unseen 全量 1839 条: SR 79.6% / OSR 81.9 / NE 2.91**(单层子集 1545 条 **85.4%**)。

同一 ckpt、不加脱困时的裸成绩是 51.1% —— **这套配方把快脑拉高了 28.5 个百分点**,
且**全是推理侧改动, 零权重变更、零训练**。

## 怎么跑出来的

三样东西缺一不可:

1. **ckpt** `goalnav_r2r_dense_goalmlp_ep5.pt`(md5 `{cksum}`, 见 CKPT_MD5.txt)
2. **EscapeV3 脱困** —— 见 `escape_v3.py`。修的是 v2 的"未知方向被当成最空旷":
   累积点云只覆盖机器人**走过**的区域, 于是"身后(走过、确实空)"和"侧前方(没走过、无观测)"
   都得最大净空 → 实测失败 case 有 **94~95% 的时间选中正后方**, 越救越远。
   另外修了 level 饱和、无进展不自知、步长过短。
3. **`--v3_noprog 2`** —— 无进展重定位阈值从默认 4 降到 2, 全量实测 **+1.8pt**(配对净 +33 条)。

## 全量对照

| 配置 | SR | OSR | NE | n |
|---|---|---|---|---|
{rows}

按楼层拆分(最佳配置):

| 子集 | SR | OSR | NE | n |
|---|---|---|---|---|
{splits}

## 复现

```bash
bash run_best.sh          # 8 卡分片, 约 2~3 小时
```

## ⚠️ 已知限制(不要误读)

- **口径是 oracle-goal 探针**: goal 方向取自 GT 参考路径, stop 用 dtg<3m 判定,
  **不是 InternNav 标准协议**。79.6% 目前只在这个口径下成立。
- **EscapeV3 不在 `our_agent.py` 里**。标准协议链路(`run_r2r.py` → `OurAgent`)的卡死处理
  是另一套原始版(15 帧位移<0.08m 记一次, >=4 次直接 `giveup` 停)。要报标准协议数字,
  必须先把 `escape_v3.py` 接进 agent 的动作选择。
- **多楼层是架构限制**: yspan>=1m 的 294 条只有 49.0%。DiffusionDrive 是 2D 规划器,
  无竖直动作分量, 调脱困/anchor 都不会变。论文口径应同时报"全量"与"单层子集"。
- **`--turn_deg` 保持 30.0**: habitat 实际是 15°/步, 但全部历史数字在 30 下产生, 0913 实测
  30 优于 15(+6.0pt @100 条单层失败集)。escape 参数是在这个有效行为下调好的, 别当 bug 修。
- **SPL 缺列**: 本配置的这份结果产出时 SPL 计算 bug(取了一个不存在的 metric key)还没修,
  故 SPL 全为 0。同批的 ep9 / r2rft 已有(0.746 / 0.407)。
- **ckpt 选择**: 全量对比下 r2rft(62.8%)明显更差、ep9(78.8%)与 ep5 打平略低 ——
  **ep5 是这个配方下的最优 ckpt**。⚠️ 注意 r2rft 曾在"按 ep5 失败挑的 40 条子集"上领先,
  那是选择偏差, 全量一跑就反转了。
""".format(cksum=CKSUM, rows='\n'.join(rows), splits='\n'.join(split_rows))

open(os.path.join(PKG, 'README.md'), 'w').write(readme)

# MANIFEST
files = sorted(f for f in glob.glob(PKG + '/**/*', recursive=True) if os.path.isfile(f))
lines = ['# MANIFEST (md5)  %d files' % len(files), '']
for f in files:
    if os.path.basename(f) == 'MANIFEST.md':
        continue
    h = subprocess.check_output(['md5sum', f]).decode().split()[0]
    lines.append('%s  %s' % (h, os.path.relpath(f, PKG)))
open(os.path.join(PKG, 'MANIFEST.md'), 'w').write('\n'.join(lines) + '\n')
print('README + MANIFEST 完成 (%d 对照 / %d 拆分 / %d 文件)' % (len(rows), len(split_rows), len(files)))
