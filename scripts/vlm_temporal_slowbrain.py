#!/usr/bin/env python3
"""
时序慢脑 (Temporal Slow-Brain) —— 发挥 Infinite VLM(自研, 10.0.0.18:8152) 的多帧时序+长记忆优势。

探测结论(0727, 走廊 nav_ds_real 实测):
  ✅ 多帧"ANY"检测(K≤6)可靠, ~1-2s; prompt 必须"visible in ANY of these frames"式
  ✅ 文本任务记忆注入(已检站点x位置+确认结果)可行
  ✅ 时序定位"which frame closest"近似正确(相位区对, 差一帧)
  ❌ 细粒度相位3分类(接近/正对/离开)全答LEAVING —— 不可用
  ❌ 长程枚举(16帧)退化复读+37s —— 不可用
  ❌ 流式输出: 服务器一次性返回(TTFB=总时长), 不支持; 用短答先出(verdict句首+小max_new_tokens)替代
  ⚠ "focus on the LAST frame"式窗口prompt会受上下文帧污染误报 —— 判决窗口必须同义对齐单帧判据

用法:
  sb = TemporalSlowBrain(target="a blue water pump")
  sb.push(frame_bgr, pose=(x, y))            # 每步推入当前感知帧
  det = sb.detect()                          # 滑窗检测: mode in {single, window_any, window_last}
  dec = sb.manager_decide(state="CRUISE")    # 带任务记忆的项目经理多选决策
"""
import os, time, json
import cv2
import requests

VLM_URL = os.environ.get("VLM_URL", "http://10.0.0.18:8152/api/generate")

# ---- prompt 配方(全部实测过) ----
def q_detect_single(target):
    return (f"Is there {target} right beside you, close and large in view? "
            f"Answer only yes or no.")

def q_detect_window_any(target, k):
    return (f"These {k} frames are a short time-ordered clip from a robot camera. "
            f"Is {target} visible in ANY of these frames, close and large? "
            f"Answer only yes or no.")

def q_detect_window_last(target, k):
    return (f"These {k} frames are in time order from a robot camera. "
            f"Considering only the LAST frame: is there {target} right beside the robot, "
            f"close and large in view? Answer only yes or no.")

def q_pick_closest(target, k):
    return (f"These {k} frames are in time order from a robot moving along a corridor. "
            f"Which frame number (1-{k}) shows {target} closest and most centered? "
            f"Answer with just the number.")


class TemporalSlowBrain:
    """滑窗时序 + 文本任务记忆 的 VLM 慢脑。"""

    def __init__(self, target="a blue water pump", url=VLM_URL, win=4, push_every=3):
        self.target = target
        self.url = url
        self.win = win                 # 滑窗帧数
        self.push_every = push_every   # 每N步采样一帧入窗
        self.buf = []                  # [(frame_bgr, x, y)] 滑窗
        self._step = 0
        self.memory = []               # 任务记忆: [{"x":.., "what":.., "verdict":..}]
        self.last_answer = ""
        self.last_latency = 0.0
        self.n_calls = 0
        self.total_latency = 0.0

    # ---------- 感知输入 ----------
    def push(self, frame_bgr, pose=None, force=False):
        """推入一帧。按 push_every 抽样入滑窗。"""
        self._step += 1
        if not force and (self._step % self.push_every != 0):
            return
        x = float(pose[0]) if pose is not None else None
        y = float(pose[1]) if pose is not None else None
        self.buf.append((frame_bgr.copy(), x, y))
        self.buf[:] = self.buf[-self.win:]

    # ---------- VLM 调用 ----------
    def _api(self, frames_bgr, prompt, max_new_tokens=8, timeout=90):
        """返回 (answer, ok)。ok=False=服务挂/超时(上层降级, 不吞错)。"""
        files = []
        for k, img in enumerate(frames_bgr):
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            files.append(("images", (f"f{k}.jpg", buf.tobytes(), "image/jpeg")))
        data = {"prompt": prompt, "max_new_tokens": str(max_new_tokens), "do_sample": "false"}
        t0 = time.time()
        try:
            r = requests.post(self.url, files=files, data=data, timeout=timeout)
            r.raise_for_status()
            ans = r.json().get("answer", "")
            self.last_latency = time.time() - t0
            self.n_calls += 1
            self.total_latency += self.last_latency
            self.last_answer = ans
            return ans, True
        except Exception as e:
            self.last_latency = time.time() - t0
            self.last_answer = f"__ERR__ {e}"
            return self.last_answer, False

    @staticmethod
    def _parse_yes(ans):
        a = (ans or "").strip().lower()
        return a.startswith("yes") or a == "y" or "yes" in a[:6]

    # ---------- 记忆 ----------
    def remember(self, x, what, verdict):
        """记下一条任务记忆(如: x=8 确认是蓝泵)。"""
        self.memory.append({"x": round(float(x), 1), "what": what, "verdict": verdict})

    def memory_text(self):
        if not self.memory:
            return "Mission memory: nothing inspected yet."
        items = "; ".join(f"x={m['x']}: {m['what']} ({m['verdict']})" for m in self.memory)
        return f"Mission memory (already checked): {items}."

    # ---------- 决策接口 ----------
    def detect(self, mode="window_any"):
        """滑窗检测目标是否在近旁。
        mode: single(基线) | window_any(窗内任一帧) | window_last(只看末帧,带上下文)
        返回 (yes, ok, latency)。"""
        if not self.buf:
            return False, False, 0.0
        if mode == "single" or len(self.buf) == 1:
            ans, ok = self._api([self.buf[-1][0]], q_detect_single(self.target))
        elif mode == "window_any":
            ans, ok = self._api([b[0] for b in self.buf], q_detect_window_any(self.target, len(self.buf)))
        elif mode == "window_last":
            ans, ok = self._api([b[0] for b in self.buf], q_detect_window_last(self.target, len(self.buf)))
        else:
            raise ValueError(mode)
        return self._parse_yes(ans), ok, self.last_latency

    def pick_closest_idx(self):
        """时序定位: 窗内哪帧目标最近最正。返回(0基idx or None, ok)。"""
        if len(self.buf) < 2:
            return None, False
        ans, ok = self._api([b[0] for b in self.buf], q_pick_closest(self.target, len(self.buf)))
        if not ok:
            return None, False
        digits = "".join(ch for ch in ans if ch.isdigit())
        if not digits:
            return None, True
        i = int(digits) - 1
        return (i if 0 <= i < len(self.buf) else None), True

    def classify_facing(self, x, use_memory=True):
        """geo_vlm转正后面向候选设备的分类: 滑窗时序(window_last判据) + 任务记忆注入。
        实测(0727走廊219帧): 比单帧精确率+5.8pt, FP-24%。返回 (yes, ok, latency)。"""
        frames = [b[0] for b in self.buf]
        if not frames:
            return False, False, 0.0
        mem = self.memory_text() if use_memory else ""
        q = (f"{mem}\nThe robot is at corridor position x={x:.0f} meters and has just turned "
             f"to face a candidate equipment beside it. Considering only the LAST frame: "
             f"is it {self.target}? Answer only yes or no.")
        ans, ok = self._api(frames, q)
        return self._parse_yes(ans), ok, self.last_latency

    def decide_with_memory(self, question_tail, options, max_new=12):
        """带任务记忆的多选决策(项目经理)。options=[(key,[大写别名...]),...]。
        返回 (key or None, raw, ok)。"""
        opt_str = "/".join(k for k, _ in options)
        prompt = (f"{self.memory_text()}\n"
                  f"Current camera frame attached. {question_tail} "
                  f"Answer with exactly one word: {opt_str}.")
        frames = [self.buf[-1][0]] if self.buf else []
        ans, ok = self._api(frames, prompt, max_new_tokens=max_new)
        if not ok:
            return None, ans, False
        a = ans.strip().upper()
        for key, aliases in options:
            if any(al in a for al in aliases):
                return key, ans, True
        return None, ans, True

    def stats(self):
        avg = self.total_latency / max(1, self.n_calls)
        return {"n_calls": self.n_calls, "avg_latency": round(avg, 2),
                "memory": list(self.memory)}
