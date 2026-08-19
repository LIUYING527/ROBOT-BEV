"""细看某段帧: python scripts/_contact_range.py <session> <start> <end> <every>"""
import os, sys, glob, numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sess, a, b, every = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
frames = sorted(glob.glob(os.path.join(ROOT, "data", sess, "images", "zed", "*.jpg")))
picks = list(range(a, min(b, len(frames)), every))
cols = 6; rows = (len(picks) + cols - 1) // cols
th, tw = 170, 270
sheet = np.full((rows * (th + 22), cols * tw, 3), 30, np.uint8)
for j, fi in enumerate(picks):
    r, c = j // cols, j % cols
    im = cv2.resize(cv2.imread(frames[fi]), (tw, th))
    y0 = r * (th + 22); x0 = c * tw
    sheet[y0 + 22:y0 + 22 + th, x0:x0 + tw] = im
    cv2.putText(sheet, f"f{fi}", (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
out = os.path.join(ROOT, "outputs", f"_contact_range_{sess}.png")
cv2.imwrite(out, sheet)
print("[ok]", out, "picks", len(picks))
