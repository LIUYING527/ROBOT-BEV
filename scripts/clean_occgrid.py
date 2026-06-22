"""清理竖直跨度占据图:闭运算连断点 + 去小噪点斑 → 更实更可读的墙(标准 .pgm)。
适度即可(别过度连接=脑补不存在的墙)。
用法: python3 scripts/clean_occgrid.py <session>
输出: outputs/occ_vert_<s>_clean.pgm/.yaml + occ_vert_<s>_clean_view.png
"""
import os, sys, re, numpy as np, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.rcParams["font.sans-serif"]=["Noto Sans CJK JP"]; matplotlib.rcParams["axes.unicode_minus"]=False
SESS=sys.argv[1] if len(sys.argv)>1 else "111450"
CLOSE_K=3; CLOSE_IT=1; MIN_AREA=4   # 闭运算核3、去面积<4格的斑

occ=np.array(Image.open(f"outputs/occ_vert_{SESS}.pgm"))   # 已翻转存盘(行=y下到上反的);处理后同样翻回
y=open(f"outputs/occ_vert_{SESS}.yaml").read()
res=float(re.search(r"resolution: ([0-9.]+)",y).group(1))
ox,oy=[float(v) for v in re.search(r"origin: \[([-0-9.]+), ([-0-9.]+)",y).groups()]

occm=(occ==0).astype(np.uint8)
# 闭运算连断点
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(CLOSE_K,CLOSE_K))
closed=cv2.morphologyEx(occm,cv2.MORPH_CLOSE,k,iterations=CLOSE_IT)
# 去小连通斑(噪点)
n,lab,stats,_=cv2.connectedComponentsWithStats(closed,connectivity=8)
keep=np.zeros_like(closed)
for i in range(1,n):
    if stats[i,cv2.CC_STAT_AREA]>=MIN_AREA: keep[lab==i]=1
out=occ.copy(); out[(occ==0)&(keep==0)]=205   # 被删的噪点占据→未知
out[keep==1]=0                                 # 闭运算补的→占据
before=int((occ==0).sum()); after=int((out==0).sum())
print(f"[INFO] {SESS} 占据 {before}→{after} (闭运算核{CLOSE_K} 去斑<{MIN_AREA})",flush=True)

Image.fromarray(out).save(f"outputs/occ_vert_{SESS}_clean.pgm")
with open(f"outputs/occ_vert_{SESS}_clean.yaml","w") as f:
    f.write(f"image: occ_vert_{SESS}_clean.pgm\nmode: trinary\nresolution: {res}\n"
            f"origin: [{ox:.3f}, {oy:.3f}, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
print(f"[OK] occ_vert_{SESS}_clean.pgm/.yaml",flush=True)

H,W=occ.shape; ext=[ox,ox+W*res,oy,oy+H*res]
def show(ax,grid,title):
    disp=np.ones_like(grid); disp[grid==0]=0; disp[grid==254]=2  # 0占据,1未知,2自由
    ax.imshow(np.flipud(disp),cmap=mcolors.ListedColormap(["#222222","#bdbdbd","#ffffff"]),origin="lower",extent=ext,vmin=0,vmax=2)
    ax.set_aspect("equal"); ax.set_title(title)
fig,axs=plt.subplots(1,2,figsize=(22,9))
show(axs[0],occ,"清理前(竖直跨度占据)"); show(axs[1],out,f"清理后(闭运算+去斑)")
plt.tight_layout(); plt.savefig(f"outputs/occ_vert_{SESS}_clean_view.png",dpi=110)
print(f"[OK] occ_vert_{SESS}_clean_view.png",flush=True)
