import sys, io
import numpy as np
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/InternNav/r2r_eval')
sys.path.insert(0, '/data/DongBaorong/BEVLOOK/benchmarks/LH-VLN/our_eval')
from run_r2r import make_env
import requests
from PIL import Image

env = make_env('val_unseen', 640, 480, 79)
obs = env.reset()
rgb = np.ascontiguousarray(obs['rgb'][:, :, :3])
print('rgb', rgb.shape, rgb.dtype)

def ask(img, q, mx=48):
    b = io.BytesIO(); Image.fromarray(img).save(b, format='JPEG')
    r = requests.post('http://127.0.0.1:8152/api/generate',
                      files=[('images', ('a.jpg', b.getvalue(), 'image/jpeg'))],
                      data={'prompt': q, 'max_new_tokens': str(mx)}, timeout=180)
    return r.json()

# 1) 复刻 our_agent.vlm_locate 的真实提问
q1 = ('Detect the door. Output its bounding box as [x1,y1,x2,y2] normalized 0-1. '
      'If not visible, answer none.')
print('Q1(ans):', repr(ask(rgb, q1)['answer']))
# 2) 更宽松的视觉问题, 确认模型真的看得到图
print('Q2(ans):', repr(ask(rgb, 'Describe what you see in this image in one short sentence.')['answer']))
