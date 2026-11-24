"""Grid of original train frames for review: python aot/grid.py out.jpg 1 2 5 8"""
import sys, cv2, numpy as np
out, ids = sys.argv[1], sys.argv[2:]
ims = []
for i in ids:
    im = cv2.resize(cv2.imread(f"data/train/images/aot_train_{int(i):04d}.png"), (960, 540))
    cv2.putText(im, f"#{int(i):04d}", (10, 45), 0, 1.5, (0, 0, 0), 8); cv2.putText(im, f"#{int(i):04d}", (10, 45), 0, 1.5, (0, 255, 255), 3)
    ims.append(im)
while len(ims) % 2: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(out, np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, len(ims), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
