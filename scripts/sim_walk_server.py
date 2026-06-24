"""Part C:浏览器亲手驱动真实 DISCOVERSE 仿真器。

服务器持有 Part B 的仿真器(MuJoCo 物理 + 差速机器人 + VGGT→3DGS 背景),
用 MJPEG 把仿真器渲的帧实时推到浏览器,浏览器 WASD → 机器人 (v,ω) 指令。
纯 stdlib http.server,无需 websocket 依赖。headless(EGL)。

用法:
  PYTHONPATH=third_party/discoverse ~/discoverse_venv/bin/python scripts/sim_walk_server.py [session] [--port 8000]
然后 laptop 浏览器开 http://<服务器IP>:8000  (或 ssh -L 8000:localhost:8000 转发)
键位:W/S 前后,A/D 左右转,Shift 加速,R 回到起点。
"""
import os
import sys
import time
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "discoverse"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from sim_walk_common import make_env

ENV = None
START = None
LOCK = threading.Lock()
SPEED = 1.2     # m/s
TURN = 1.2      # rad/s

HTML = """<!doctype html><html><head><meta charset=utf-8>
<title>VGGT→3DGS→DISCOVERSE walk</title>
<style>body{margin:0;background:#111;color:#ccc;font-family:sans-serif;text-align:center}
img{max-width:98vw;max-height:86vh;margin-top:6px;background:#000}
#hud{padding:6px;font-size:14px}kbd{background:#333;border-radius:4px;padding:1px 6px}</style></head>
<body>
<div id=hud>真实 DISCOVERSE 仿真器(双层:几何物理 + 3DGS皮肤) ·
<kbd>W</kbd>/<kbd>S</kbd> 前后 · <kbd>A</kbd>/<kbd>D</kbd> 转向 · <kbd>Shift</kbd> 加速 · <kbd>R</kbd> 回起点 ·
<kbd>T</kbd> 切换图层 · <button id=tg onclick="tog()">切换图层</button> ·
<span id=st>v=0 w=0</span> · <b><span id=ly>图层:3DGS皮肤</span></b></div>
<img id=v src="/stream">
<script>
let keys={};
function send(){
 let v=0,w=0,boost=keys['shift']?1.8:1;
 if(keys['w'])v+=1; if(keys['s'])v-=1;
 if(keys['a'])w+=1; if(keys['d'])w-=1;
 v*=boost;
 document.getElementById('st').textContent='v='+v.toFixed(1)+' w='+w.toFixed(1);
 fetch('/cmd?v='+v+'&w='+w);
}
function tog(){fetch('/toggle').then(r=>r.text()).then(t=>{
 document.getElementById('ly').textContent='图层:'+(t==='1'?'3DGS皮肤':'几何物理');});}
addEventListener('keydown',e=>{let k=e.key.toLowerCase();
 if(k==='r'){fetch('/cmd?reset=1');return;}
 if(k==='t'){tog();return;}
 if(!keys[k]){keys[k]=1;send();}});
addEventListener('keyup',e=>{keys[e.key.toLowerCase()]=0;send();});
addEventListener('blur',()=>{keys={};send();});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        elif u.path == "/toggle":
            global ENV
            with LOCK:
                ENV.show_gaussian_img = not ENV.show_gaussian_img
                if hasattr(ENV, "gs_renderer"):
                    ENV.gs_renderer.need_rerender = True
                cur = "1" if ENV.show_gaussian_img else "0"
            body = cur.encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
        elif u.path == "/cmd":
            q = parse_qs(u.query)
            if "reset" in q:
                with LOCK:
                    ENV.set_pose(START[0], START[1], START[2]); ENV.cmd[:] = 0
            else:
                v = float(q.get("v", ["0"])[0]) * SPEED
                w = float(q.get("w", ["0"])[0]) * TURN
                with LOCK:
                    ENV.cmd[:] = (v, w)
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        elif u.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    try:
                        with LOCK:
                            ENV.step()
                            img = ENV.frame()
                    except Exception as e:           # 渲染/切图层出错→跳过本帧,别挂流
                        sys.stderr.write(f"[stream] render err: {e}\n"); time.sleep(0.05); continue
                    if img is None:
                        time.sleep(0.03); continue
                    ok, jpg = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    time.sleep(0.03)   # ~30fps 上限
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", nargs="?", default="114830")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--skin_only", action="store_true",
                    help="跳过几何物理层(无墙/盒/碰撞),纯 3DGS 皮肤自由驾驶")
    args = ap.parse_args()

    global ENV, START
    ENV, W = make_env(args.session, width=args.width, height=args.height, fps=30,
                      skin_only=args.skin_only)
    wp, yaw = W["waypoints_xy"], W["yaw"]
    # 开局朝行进方向(录制的 yaw[0] 往往背对走廊→全黑):取第一个离起点≥1m 的前方路点定朝向
    fwd = wp[0]
    for j in range(1, len(wp)):
        if np.hypot(wp[j, 0] - wp[0, 0], wp[j, 1] - wp[0, 1]) >= 1.0:
            fwd = wp[j]; break
    start_yaw = float(np.arctan2(fwd[1] - wp[0, 1], fwd[0] - wp[0, 0]))
    START = (float(wp[0, 0]), float(wp[0, 1]), start_yaw)
    ENV.set_pose(*START)
    ENV.render()
    print(f"[server] 仿真器就绪。浏览器打开 http://<服务器IP>:{args.port}  (或 ssh -L {args.port}:localhost:{args.port})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
