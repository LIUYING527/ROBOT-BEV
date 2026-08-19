"""105731 密集contact sheet: 每~30帧一张缩略, 找配电箱位置+评估画质/有没有人。"""
import os, glob, numpy as np, cv2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frames = sorted(glob.glob(os.path.join(ROOT, "outputs", "vggto_colmapjoint_105731", "frames_zed", "*.jpg")))
picks = list(range(0, len(frames), 30))
cols = 6; rows = (len(picks) + cols - 1) // cols
th, tw = 150, 240
sheet = np.full((rows * (th + 22), cols * tw, 3), 30, np.uint8)
for j, fi in enumerate(picks):
    r, c = j // cols, j % cols
    im = cv2.resize(cv2.imread(frames[fi]), (tw, th))
    y0 = r * (th + 22); x0 = c * tw
    sheet[y0 + 22:y0 + 22 + th, x0:x0 + tw] = im
    cv2.putText(sheet, f"f{fi}", (x0 + 3, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
out = os.path.join(ROOT, "outputs", "_contact_105731.png")
cv2.imwrite(out, sheet)
print("[ok]", out, "picks", len(picks))
