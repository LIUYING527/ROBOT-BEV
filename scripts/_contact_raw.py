"""任意 session 原始 ZED 帧 contact sheet。用法: python scripts/_contact_raw.py <session> [every]"""
import os, sys, glob, numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sess = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 40
frames = sorted(glob.glob(os.path.join(ROOT, "data", sess, "images", "zed", "*.jpg")))
picks = list(range(0, len(frames), every))
cols = 6; rows = (len(picks) + cols - 1) // cols
th, tw = 150, 240
sheet = np.full((rows * (th + 22), cols * tw, 3), 30, np.uint8)
for j, fi in enumerate(picks):
    r, c = j // cols, j % cols
    im = cv2.resize(cv2.imread(frames[fi]), (tw, th))
    y0 = r * (th + 22); x0 = c * tw
    sheet[y0 + 22:y0 + 22 + th, x0:x0 + tw] = im
    cv2.putText(sheet, f"f{fi}", (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
out = os.path.join(ROOT, "outputs", f"_contact_raw_{sess}.png")
cv2.imwrite(out, sheet)
print("[ok]", out, "frames", len(frames), "picks", len(picks))
