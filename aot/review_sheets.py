"""Contact sheets of auto-label overlays that need a human look (titans or horses present).
python aot/review_sheets.py <labels_dir> <review_dir> <out_dir> <list.txt> [already_reviewed.txt]"""
import json, os, sys
import cv2, numpy as np
lab_dir, rev_dir, out, lst = sys.argv[1:5]
done = {l.strip() for l in open(sys.argv[5])} if len(sys.argv) > 5 and os.path.exists(sys.argv[5]) else set()
st = json.load(open(f"{lab_dir}/stats.json"))
os.makedirs(out, exist_ok=True)
stems = [l.strip() for l in open(lst) if l.strip() in st and l.strip() not in done]
need = [s for s in stems if st[s]["titan"] > 0.002 or st[s]["p_titan"] > 0.01 or st[s]["horse"] > 0.002]
print(f"{len(stems)} new labelled, {len(need)} need review, {len(stems) - len(need)} human/background only")
cols, per = 4, 24
for k in range(0, len(need), per):
    ims = []
    for s in need[k:k + per]:
        im = cv2.imread(f"{rev_dir}/{s}.jpg")
        t = s[3:] + (" H" if st[s]["horse"] > 0.002 else "")
        cv2.putText(im, t, (5, 24), 0, 0.75, (0, 0, 0), 5); cv2.putText(im, t, (5, 24), 0, 0.75, (0, 255, 255), 2)
        ims.append(im)
    while len(ims) % cols: ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f"{out}/r{k // per:02d}.jpg", np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
open(f"{out}/shown.txt", "w").write("\n".join(need) + "\n")
print("sheets:", (len(need) + per - 1) // per)
