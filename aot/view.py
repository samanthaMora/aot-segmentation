"""Review helper: one frame (or a crop of it) with a coordinate grid in full-frame pixels.

  view.py STEM out.jpg [--crop x0,y0,x1,y1] [--labels DIR] [--inst TEACHER_DIR] [--plain]
--labels draws the label contours (titan red, human blue, void magenta); --inst draws every teacher
instance outline with its id; --plain adds the untouched image next to it.
"""
import argparse, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose import PAL, load_inst  # noqa

COLS = [(0, 255, 0), (0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0), (128, 255, 128), (128, 128, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem"); ap.add_argument("out")
    ap.add_argument("--images", default="data/train/images")
    ap.add_argument("--crop"); ap.add_argument("--labels"); ap.add_argument("--inst")
    ap.add_argument("--plain", action="store_true"); ap.add_argument("--fill", action="store_true")
    ap.add_argument("--width", type=int, default=1500)
    a = ap.parse_args()
    im = cv2.imread(f"{a.images}/{a.stem}.png")
    H, W = im.shape[:2]
    x0, y0, x1, y1 = map(int, a.crop.split(",")) if a.crop else (0, 0, W, H)
    o = im.copy()
    if a.labels and os.path.exists(f"{a.labels}/{a.stem}.png"):
        lab = cv2.imread(f"{a.labels}/{a.stem}.png", -1)
        if a.fill:
            o = (0.6 * im + 0.4 * PAL[lab]).astype(np.uint8); o[lab == 0] = im[lab == 0]
        for c in (1, 2, 255):
            cn, _ = cv2.findContours((lab == c).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cn, -1, tuple(int(v) for v in PAL[c]), 2)
    sc = min(a.width / (x1 - x0), 1000 / (y1 - y0))
    marks = []
    if a.inst:
        masks, meta = load_inst(a.inst, a.stem)
        for e in meta:
            m = masks[e["id"]]
            if not m[y0:y1, x0:x1].any():
                continue
            col = COLS[e["id"] % len(COLS)]
            cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cn, -1, col, 2)
            ys, xs = np.where(m[y0:y1, x0:x1])
            marks.append((e["id"], e.get("kind", "body")[0], int(np.median(xs)), int(np.median(ys)), col))

    def prep(t):
        t = cv2.resize(t[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        step = 100 if (x1 - x0) > 900 else (50 if (x1 - x0) > 400 else 20)
        for gx in range((x0 // step + 1) * step, x1, step):
            X = int((gx - x0) * sc); cv2.line(t, (X, 0), (X, t.shape[0]), (255, 255, 255), 1, cv2.LINE_4)
            cv2.putText(t, str(gx), (X + 2, 12), 0, 0.4, (0, 0, 0), 3); cv2.putText(t, str(gx), (X + 2, 12), 0, 0.4, (255, 255, 255), 1)
        for gy in range((y0 // step + 1) * step, y1, step):
            Y = int((gy - y0) * sc); cv2.line(t, (0, Y), (t.shape[1], Y), (255, 255, 255), 1, cv2.LINE_4)
            cv2.putText(t, str(gy), (2, Y - 3), 0, 0.4, (0, 0, 0), 3); cv2.putText(t, str(gy), (2, Y - 3), 0, 0.4, (255, 255, 255), 1)
        return t

    t = prep(o)
    for i, k, mx, my, col in marks:
        q = (int(mx * sc), int(my * sc)); s = f"{i}{'' if k == 'b' else k}"
        cv2.putText(t, s, q, 0, 0.7, (0, 0, 0), 5); cv2.putText(t, s, q, 0, 0.7, col, 2)
    if a.plain:
        t = np.vstack([cv2.resize(im[y0:y1, x0:x1], (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA), t])
    cv2.imwrite(a.out, t, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
