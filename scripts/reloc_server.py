"""RELOC HTTP 服务(常驻GPU服务器): 预载GS地图, 收zed点云+believed位姿 → render-and-compare位姿优化 → 回纠正位姿。
真机架构: 机器人(Orin跑不了gsplat)只发点云, 重计算在本服务。

接口: POST /reloc?x=<m>&y=<m>&yaw=<deg>&pitch_deg=<deg,默认0>&cam_h=<m,默认1.2>
  body = npz字节流(xyzrgba, 720x1280x4 float16, 单位mm, 与采集同格式)
  坐标约定: x/y/yaw = 机器人believed位姿, 【真实米制】, 原点=采集起始点, 轴向与采集起始朝向对齐
  返回: {ok, x, y, yaw_deg (纠正后, 真实米制), loss0, loss1, accepted, nvalid, elapsed_s}
  ⚠内部自动做 真实↔世界(0.70×) 换算——客户端全程真实米制, 不用关心世界尺度。
健康检查: GET /health

用法: OBS_SCALE=0.70 CUDA_VISIBLE_DEVICES=4 ~/discoverse_venv/bin/python scripts/reloc_server.py --port 8321
"""
import os, sys, io, time, argparse, threading
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

app = None
try:
    from flask import Flask, request, jsonify
    app = Flask(__name__)
except ImportError:
    print("需要flask"); raise

G = None
LOCK = threading.Lock()


def cam_CR(x_w, y_w, yaw, pitch, cam_h_w):
    """世界系位姿 → (C_w, R_c2w)。OpenCV相机: x右 y下 z前。"""
    C_w = np.array([x_w, y_w, cam_h_w], np.float64)
    c, s = np.cos(yaw), np.sin(yaw); cp, sp = np.cos(pitch), np.sin(pitch)
    fwd = np.array([c * cp, s * cp, sp]); right = np.array([s, -c, 0.0])
    down = np.cross(fwd, right); down /= np.linalg.norm(down)
    return C_w, np.stack([right, down, fwd], axis=1)


@app.route("/health")
def health():
    return jsonify(ok=True, frames=len(G.names) if G else 0, obs_scale=G.obs_scale if G else None)


@app.route("/reloc", methods=["POST"])
def reloc():
    t0 = time.time()
    try:
        x = float(request.args["x"]); y = float(request.args["y"])
        yaw = np.radians(float(request.args["yaw"]))
        pitch = np.radians(float(request.args.get("pitch_deg", 0.0)))
        cam_h_real = float(request.args.get("cam_h", 1.2))
    except Exception as e:
        return jsonify(ok=False, err=f"参数解析: {e}"), 400
    try:
        pc = np.load(io.BytesIO(request.get_data()))["xyzrgba"]
    except Exception as e:
        return jsonify(ok=False, err=f"npz解析: {e}"), 400
    z_real = pc[..., 2].astype(np.float32) / 1000.0            # mm → 真实米
    s = G.obs_scale                                            # 真实→世界 0.70
    with LOCK:
        obs = {lv: G.obs_depth_from_z(z_real, lv[1], lv[0]) for lv in ((320, 180), (640, 360))}
        nvalid = int(obs[(640, 360)][1].sum())
        if nvalid < 3000:
            return jsonify(ok=False, err=f"有效深度点不足({nvalid})"), 422
        C_b, R_b = cam_CR(x * s, y * s, yaw, pitch, cam_h_real * s)
        C1, R1, trace = G.optimize(0, C_b, R_b, obs=obs)
    if C1 is None or not trace:
        return jsonify(ok=False, err="优化失败"), 500
    accepted = bool(trace[-1] <= 0.08)                          # 与lab_mission同门限
    yaw1 = float(np.degrees(np.arctan2(R1[1, 2], R1[0, 2])))
    out = dict(ok=True, accepted=accepted,
               x=float(C1[0] / s), y=float(C1[1] / s), yaw_deg=yaw1,
               dx=float(C1[0] / s - x), dy=float(C1[1] / s - y),
               dyaw_deg=float((yaw1 - np.degrees(yaw) + 180) % 360 - 180),
               loss0=round(trace[0], 4), loss1=round(trace[-1], 4),
               nvalid=nvalid, elapsed_s=round(time.time() - t0, 1))
    print(f"[reloc] bel({x:.2f},{y:.2f},{np.degrees(yaw):.0f}°) → ({out['x']:.2f},{out['y']:.2f},{yaw1:.1f}°) "
          f"loss {out['loss0']}→{out['loss1']} {'OK' if accepted else 'REJECT'} {out['elapsed_s']}s", flush=True)
    return jsonify(out)


def main():
    global G
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--session", default="colmapjoint_110744s2")
    ap.add_argument("--raw", default="data/lab/110744")
    args = ap.parse_args()
    from gs_relocate import GSReloc
    G = GSReloc(args.session, args.raw)
    print(f"[server] 地图已载 {args.session}, obs_scale={G.obs_scale}, port={args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
