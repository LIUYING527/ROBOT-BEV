#!/usr/bin/env python3
"""1015 采数：dense_data 帧窗口 → 慢脑 latent（供训练 text_proj）。

**为什么需要它**：`text_proj`(2048→256) 要训，就得有 (输入帧窗口 → hidden) 的配对。
hidden 只能由慢脑 server 现算（它依赖图像），所以离线批量采。

**对齐训练口径**（必须一致，否则白训）：
- 帧取自 `meta.jsonl` 的 `frames`，**只保留 `vi >= 0`**（与 `collect_samples` 的过滤一致）
- 输入窗口 = 最近 `nframes` 帧（不足用最早帧补齐），帧序 **[旧...新]**（与评测 `fr_send` 一致）
- 输出 `.npy` (N,2048) float16 + `.index.jsonl`（key = `episode_id` + `fi`）

**纯 HTTP + I/O，不占 GPU**。多 server 并行采请用 `--shard-i/--shard-n` 分片。

用法示例（先小样验证）：
```
python collect_hidden_r2r.py --eps 3 --limit 30 --out /tmp/hid_smoke_train
```
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import requests


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval/dense_data')
    ap.add_argument('--shards', default='s0', help='逗号分隔, 如 s0,s1')
    ap.add_argument('--server', default='http://localhost:9450')
    ap.add_argument('--eps', type=int, default=100, help='每个分片最多取前 N 个 ep')
    ap.add_argument('--stride', type=int, default=1, help='每 stride 帧取 1(与训练 stride 对齐)')
    ap.add_argument('--nframes', type=int, default=8, help='发给 server 的帧数(必须与推理一致)')
    ap.add_argument('--out', default='/tmp/hid_train', help='输出前缀(生成 .npy + .index.jsonl)')
    ap.add_argument('--limit', type=int, default=0, help='>0 = 最多采这么多帧(调试)')
    ap.add_argument('--shard-i', type=int, default=0, help='并行分片号(0-based)')
    ap.add_argument('--shard-n', type=int, default=1, help='并行分片总数')
    ap.add_argument('--timeout', type=float, default=90.0)
    return ap.parse_args()


def load_eps(args):
    eps = []
    for sh in args.shards.split(','):
        mp = os.path.join(args.data_dir, sh, 'meta.jsonl')
        if not os.path.exists(mp):
            print('[warn] 缺 %s' % mp, file=sys.stderr, flush=True)
            continue
        with open(mp, encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    e = json.loads(ln)
                    e['_shard'] = sh
                    eps.append(e)
    return eps[:args.eps]


def main():
    args = parse_args()
    eps = load_eps(args)
    print('[collect] %d 个 ep | server=%s | nframes=%d | stride=%d | 分片 %d/%d'
          % (len(eps), args.server, args.nframes, args.stride, args.shard_i, args.shard_n), flush=True)

    hs, index, n = [], [], 0
    for ep in eps:
        epd = os.path.join(args.data_dir, ep['_shard'], 'ep%s' % ep['episode_id'])
        # 只保留 vi>=0 的帧, 与 collect_samples 完全一致。
        # ⚠️ meta 的 frames 元素【没有 fi 字段】, fi 是 enumerate 下标(见 train_goalnav_r2r_dense.py:201);
        #    文件名 f{fi:03d}.npz 用的就是它。
        fr = [dict(f, fi=i) for i, f in enumerate(ep['frames']) if f.get('vi', -1) >= 0]
        if not fr:
            continue
        rgbs = []
        ok = True
        for f in fr:
            p = os.path.join(epd, 'f%03d.npz' % f['fi'])
            if not os.path.exists(p):
                ok = False
                break
            with np.load(p) as z:
                rgbs.append(np.ascontiguousarray(z['rgb']))
        if not ok:
            print('[warn] %s 帧缺失, 跳过' % epd, flush=True)
            continue

        for fi in range(0, len(fr), args.stride):
            if args.shard_n > 1 and (n % args.shard_n) != args.shard_i:
                n += 1
                continue
            win = rgbs[max(0, fi - args.nframes + 1): fi + 1]
            if len(win) < args.nframes:                       # 开局补齐: 用最早帧重复
                win = [win[0]] * (args.nframes - len(win)) + win
            try:
                r = requests.post(args.server + '/locate', json={
                    'frames': [{'rgb': base64.b64encode(f.tobytes()).decode(),
                                'shape': list(f.shape)} for f in win],
                    'instruction': ep.get('instruction', ''),
                    'want_hidden': True}, timeout=args.timeout).json()
            except Exception as e:                            # noqa: BLE001
                print('[warn] ep%s f%03d HTTP 失败: %s' % (ep['episode_id'], fr[fi]['fi'], e), flush=True)
                n += 1
                continue
            h = r.get('hidden')
            if not h:
                print('[warn] ep%s f%03d 无 hidden 字段(server 未开 want_hidden?)'
                      % (ep['episode_id'], fr[fi]['fi']), flush=True)
                n += 1
                continue
            hs.append(np.frombuffer(base64.b64decode(h), np.float16))
            index.append({'shard': ep['_shard'], 'episode_id': str(ep['episode_id']),
                          'fi': int(fr[fi]['fi']), 'vi': int(fr[fi].get('vi', -1)),
                          'scene': ep.get('scene', '')})
            n += 1
            if n % 200 == 0:
                print('  ... %d 帧' % n, flush=True)
            if args.limit and len(hs) >= args.limit:
                break
        if args.limit and len(hs) >= args.limit:
            break

    if not hs:
        print('[done] 0 帧, 未写出', flush=True)
        return
    arr = np.stack(hs)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.save(args.out + '.npy', arr)
    with open(args.out + '.index.jsonl', 'w', encoding='utf-8') as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('[done] %d 帧 -> %s.npy shape=%s dim=%d  (均值 %.4f / 标准差 %.4f)'
          % (len(hs), args.out, arr.shape, arr.shape[1], float(arr.mean()), float(arr.std())), flush=True)


if __name__ == '__main__':
    main()
