"""Compose teacher instances into label maps, apply manual review overrides,
optionally evaluate against GT and render numbered overlays for review.

overrides.json format: {"<stem>": {"<inst id>": 0|1|2, ...}, ...}
  0 = drop (background), 1 = Pure_Titan, 2 = Human
"""
import argparse, glob, json, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics.segmentation_metrics import aggregate_dataset_metrics, compute_segmentation_metrics  # noqa

PAL = np.zeros((256, 3), np.uint8)
PAL[1] = (60, 20, 220)   # BGR crimson
PAL[2] = (255, 191, 0)   # BGR sky blue
PAL[255] = (255, 0, 255)


def load_inst(d, stem):
    z = np.load(f"{d}/inst/{stem}.npz")
    shp = tuple(z["shape"])
    m = np.unpackbits(z["masks"], axis=-1)[..., :shp[-1]].astype(bool) if shp[0] else np.zeros(shp, bool)
    return m, json.load(open(f"{d}/meta/{stem}.json"))


def auto_class(e, det_thr, margin):
    if e["det"] < det_thr:
        return 0
    pt, ph, po = e["p_titan"], e["p_human"], e["p_other"]
    if po > max(pt, ph):
        return 0
    if pt > ph + margin:
        return 1
    return 2


def compose(masks, meta, ov, det_thr, margin, auto_ignore=0.0):
    H, W = masks.shape[1:] if len(masks) else (1080, 1920)
    lab = np.zeros((H, W), np.uint8)
    ov = ov or {}
    cls = {e["id"]: (0 if e.get("kind") in ("gear", "horse") else auto_class(e, det_thr, margin)) for e in meta}
    if "_all" in ov:  # frame-level: every kept body instance gets this class
        cls = {k: (ov["_all"] if v else 0) for k, v in cls.items()}
    for k, v in ov.items():
        if not k.startswith("_"):
            cls[int(k)] = v
    # gear (3DMG boxes/blades) -> Human only when it touches a Human body mask
    hum = np.zeros((H, W), bool)
    for e in meta:
        if e.get("kind") != "gear" and cls[e["id"]] == 2:
            hum |= masks[e["id"]]
    hum_d = cv2.dilate(hum.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    for e in meta:
        if e.get("kind") == "gear" and str(e["id"]) not in ov:
            m = masks[e["id"]]
            x0, y0, x1, y1 = e["box"]
            fill = m.sum() / max(1.0, (x1 - x0) * (y1 - y0))
            cls[e["id"]] = 2 if (m & hum_d).any() and fill < 0.9 and e["sam_iou"] > 0.7 else 0
    # dedupe: drop instances mostly contained in a bigger same-class instance -> harmless anyway
    order = sorted(meta, key=lambda e: -e["area"])  # big first, small painted on top
    for e in order:
        c = cls[e["id"]]
        if c:
            lab[masks[e["id"]]] = c
    if auto_ignore:  # unreviewed frames: what the teacher is unsure about becomes Void instead of a guess
        for e in meta:
            if e.get("kind", "body") != "body" or str(e["id"]) in ov:
                continue
            pt, ph, po = e["p_titan"], e["p_human"], e["p_other"]
            weak = e["det"] < det_thr and po <= max(pt, ph) and e["det"] >= det_thr - 0.08
            torn = e["det"] >= det_thr and po <= max(pt, ph) and abs(pt - ph) < auto_ignore
            if torn:
                lab[masks[e["id"]]] = 255
            elif weak:
                lab[masks[e["id"]] & (lab == 0)] = 255
    if ov.get("_titan_ignore"):  # shifter titans etc.: not Pure_Titan, but not safe to call background either
        lab[lab == 1] = 255
    hm = np.zeros((H, W), bool)
    for e in meta:  # horses are background; where a horse mask overlaps a person/titan mask the pixel is ambiguous
        if e.get("kind") == "horse":
            hm |= masks[e["id"]]
    if hm.any():
        lab[hm & (lab != 0)] = 255
    for x0, y0, x1, y1 in ov.get("_ignore", []):  # boxes in 960x540 review coords
        lab[2 * y0:2 * y1, 2 * x0:2 * x1] = 255
    return lab, cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out-labels")
    ap.add_argument("--gt")
    ap.add_argument("--overrides")
    ap.add_argument("--review")
    ap.add_argument("--horses", help="dir with horse masks; used for frames whose override has _horse_bg")
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--auto-ignore", type=float, default=0.0,
                    help="unreviewed frames: titan/human probability gap under this -> Void (0 = off)")
    ap.add_argument("--list", help="only these stems")
    ap.add_argument("--manual", help="manual.json with hand corrections (see manual.py)")
    ap.add_argument("--manual-cache", default="work/manual_train")
    ap.add_argument("--fill-small", type=int, default=0, help="fill enclosed background holes up to this many px inside a figure")
    a = ap.parse_args()
    man = json.load(open(a.manual)) if a.manual else {}
    ovs = json.load(open(a.overrides)) if a.overrides and os.path.exists(a.overrides) else {}
    if a.out_labels: os.makedirs(a.out_labels, exist_ok=True)
    if a.review: os.makedirs(a.review, exist_ok=True)
    fm = {}
    only = {l.strip() for l in open(a.list)} if a.list else None
    for p in sorted(glob.glob(f"{a.images}/*.png")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if ovs.get(stem, {}).get("_skip") or (only is not None and stem not in only):
            continue
        if not os.path.exists(f"{a.teacher}/meta/{stem}.json"):
            continue
        masks, meta = load_inst(a.teacher, stem)
        lab, cls = compose(masks, meta, ovs.get(stem), a.det_thr, a.margin, a.auto_ignore)
        if a.horses and ovs.get(stem, {}).get("_horse_bg"):
            hm = cv2.imread(f"{a.horses}/{stem}.png", 0)
            lab[(hm > 0) & (lab == 2)] = 0
        if stem in man:
            from manual import apply_manual
            lab = apply_manual(lab, man[stem], a.manual_cache, stem, masks)
        if a.fill_small:
            from manual import apply_manual
            lab = apply_manual(lab, [{"fill": 1, "max": a.fill_small}, {"fill": 2, "max": a.fill_small}], a.manual_cache, stem, masks)
        if a.out_labels:
            cv2.imwrite(f"{a.out_labels}/{stem}.png", lab)
        if a.gt:
            gt = cv2.imread(f"{a.gt}/{stem}.png", cv2.IMREAD_UNCHANGED)
            r = compute_segmentation_metrics(lab, gt)
            fm[stem] = r
            print(stem, r.mean_iou, r.iou_per_class)
        if a.review:
            img = cv2.imread(p)
            ovl = (0.55 * img + 0.45 * PAL[lab]).astype(np.uint8)
            for e in meta:
                x0, y0, x1, y1 = map(int, e["box"])
                c = cls[e["id"]]
                if e.get("kind") == "gear" and c == 0:
                    continue
                col = (0, 255, 255) if c == 0 else tuple(int(v) for v in PAL[c])
                if c == 0 and e["det"] < a.det_thr:
                    col = (120, 120, 120)
                cv2.rectangle(ovl, (x0, y0), (x1, y1), col, 3)
                txt = f"{e['id']}"
                cx, cy = (x0 + x1) // 2 - 20, (y0 + y1) // 2
                cv2.putText(ovl, txt, (cx, cy), 0, 2.2, (0, 0, 0), 9)
                cv2.putText(ovl, txt, (cx, cy), 0, 2.2, (255, 255, 255), 3)
            cv2.imwrite(f"{a.review}/{stem}.jpg", cv2.resize(ovl, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 80])
    if fm:
        agg = aggregate_dataset_metrics(fm)
        print("AGG fg micro-mIoU", agg.mean_iou, agg.iou_per_class)


if __name__ == "__main__":
    main()
