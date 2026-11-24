"""Label unreviewed frames: SAM instance masks (teacher) + class voted by the trained student.

CLIP's zero-shot titan/human decision is too noisy without manual review (0.44 fg-mIoU on val), so
the class of every teacher instance is decided by the mean student probability inside its mask.
Anything the two disagree on becomes Void (255); detected horses are forced to background.

Titan labels are only trusted on frames a human has looked at: by default every Pure_Titan pixel of an
unreviewed frame becomes Void (the student calls shifter titans and backlit faces "titan").
Per-frame flags in --overrides: {"<stem>": {"_skip": 1, "_titan_ok": 1, "_horse_ok": 1, "_all": 2}}
  _horse_ok: the CLIP-confirmed horse masks of this frame were checked -> they are forced to background
  "<instance id>": 0|1|2|255  hand decision for one teacher instance (ids are drawn by --ids)
  _titan_ok: the titan labels of this frame were checked;  _all: every figure in the frame is this class
"""
import argparse, json, os, sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose import PAL, load_inst  # noqa
from train import MEAN, STD, build_model  # noqa


@torch.no_grad()
def student_prob(model, rgb, sizes=((448, 800), (544, 960))):
    out = 0
    for h, w in sizes:
        x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)
        lg = model(x) + model(x.flip(-1)).flip(-1)
        out = out + F.interpolate(lg / 2, size=rgb.shape[:2], mode="bilinear", align_corners=False).softmax(1)
    return (out / len(sizes))[0].numpy()  # [3, H, W]


def label_frame(masks, meta, prob, flags, void_horses=False):
    H, W = prob.shape[1:]
    lab = np.zeros((H, W), np.uint8)
    # horses confirmed by CLIP (verify_horses.py) are background, whatever the student thinks of them
    hm = np.zeros((H, W), bool)
    checked = bool(flags.get("_horse_ok"))
    if checked or void_horses:   # checked outlines -> background; unchecked CLIP-confirmed horses -> Void (never Human)
        for e in meta:
            if e.get("kind") == "horse" and e.get("p_horse", 0) > 0.5:
                hm |= masks[e["id"]]
    rider_on_horse = np.zeros((H, W), bool)
    forced = np.zeros((H, W), bool)
    body = sorted([e for e in meta if e.get("kind", "body") == "body" and e["area"] > 0], key=lambda e: -e["area"])
    for e in body:  # big first, small painted on top
        m = masks[e["id"]]
        in_horse = (m & hm).sum() / m.sum()
        if in_horse >= 0.5:      # this "person" is mostly a horse: only what sticks out of the horse can be a rider
            m = m & ~hm
            if m.sum() < max(200, 0.1 * e["area"]):
                continue
        pt, ph = float(prob[1][m].mean()), float(prob[2][m].mean())
        fg = pt + ph
        clip_other = e["p_other"] > max(e["p_titan"], e["p_human"]) and in_horse < 0.5
        if str(e["id"]) in flags:      # decided by hand
            c = flags[str(e["id"])]
            lab[m] = c
            forced[m] = True
            continue
        if fg < 0.3:
            c = 0 if (clip_other or e["det"] < 0.3) else 255
        elif flags.get("_all"):
            c = flags["_all"]
        elif clip_other:  # the student sees a figure, CLIP sees an object: do not teach either
            c = 255
        else:
            c = (1 if pt > ph else 2) if (max(pt, ph) / fg >= 0.8 and fg >= 0.5) else 255
        if c:
            lab[m] = c
            if in_horse < 0.5:
                rider_on_horse |= m & hm   # e.g. a rider's leg in front of the horse: ambiguous
    hum_d = cv2.dilate((lab == 2).astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    for e in meta:  # 3DMG gear touching a human is Human
        if e.get("kind") == "gear":
            m = masks[e["id"]]
            x0, y0, x1, y1 = e["box"]
            if (m & hum_d).any() and m.sum() / max(1.0, (x1 - x0) * (y1 - y0)) < 0.9 and e["sam_iou"] > 0.7:
                lab[m & (lab == 0) & ~hm] = 2
    # the student sees a figure that no instance covers: unknown, not background
    lab[((prob[1] + prob[2]) > 0.5) & (lab == 0) & ~hm & ~forced] = 255
    lab[hm] = 0 if checked else 255   # checked horse masks win, even under a rider's mask (a leg at most)
    if not (flags.get("_titan_ok") or flags.get("_all")):
        lab[(lab == 1) & ~forced] = 255
    return lab, hm, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--review", required=True); ap.add_argument("--overrides")
    ap.add_argument("--list", required=True, help="stems, one per line")
    ap.add_argument("--ids", action="store_true", help="960x540 review overlays with instance ids")
    ap.add_argument("--horse-out", help="write horse instance masks of _horse_ok frames here")
    ap.add_argument("--void-horses", action="store_true", help="unchecked CLIP-confirmed horses become Void")
    ap.add_argument("--gt", help="evaluate against GT masks instead of only writing labels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.review, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = build_model(ck["arch"], ck["encoder"], weights=None); model.load_state_dict(ck["state"]); model.eval()
    ovs = json.load(open(a.overrides)) if a.overrides and os.path.exists(a.overrides) else {}
    stats_p = f"{a.out}/stats.json"
    stats = json.load(open(stats_p)) if os.path.exists(stats_p) else {}
    fm = {}
    for stem in [l.strip() for l in open(a.list) if l.strip()]:
        if not os.path.exists(f"{a.teacher}/meta/{stem}.json"):
            continue
        flags = ovs.get(stem, {})
        if flags.get("_skip"):
            if os.path.exists(f"{a.out}/{stem}.png"): os.remove(f"{a.out}/{stem}.png")
            continue
        pp = f"{a.out}/prob/{stem}.npy"
        bgr = cv2.imread(f"{a.images}/{stem}.png")
        if os.path.exists(pp):
            prob = np.load(pp).astype(np.float32) / 255
        else:
            prob = student_prob(model, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            os.makedirs(f"{a.out}/prob", exist_ok=True)
            np.save(pp, (prob * 255).astype(np.uint8))
        masks, meta = load_inst(a.teacher, stem)
        lab, hm, _ = label_frame(masks, meta, prob, flags, a.void_horses)
        cv2.imwrite(f"{a.out}/{stem}.png", lab)
        if a.horse_out and flags.get("_horse_ok"):   # instance-id horse masks for the copy-paste hard negatives
            os.makedirs(a.horse_out, exist_ok=True)
            hid = np.zeros(lab.shape, np.uint8)
            for k, e in enumerate([e for e in meta if e.get("kind") == "horse" and e.get("p_horse", 0) > 0.5]):
                hid[masks[e["id"]]] = k + 1
            cv2.imwrite(f"{a.horse_out}/{stem}.png", hid)
        n = lab.size
        stats[stem] = dict(titan=round(float((lab == 1).sum() / n), 4), human=round(float((lab == 2).sum() / n), 4),
                           void=round(float((lab == 255).sum() / n), 4), horse=round(float(hm.sum() / n), 4),
                           p_titan=round(float((prob[1] > 0.5).mean()), 4))
        # unreviewed frames: show the titan candidates (red) so they can be approved, although they are saved as Void
        show = lab if flags else label_frame(masks, meta, prob, {"_titan_ok": 1}, a.void_horses)[0]
        ovl = (0.5 * bgr + 0.5 * PAL[show]).astype(np.uint8)
        ovl[show == 0] = bgr[show == 0]
        cnt, _ = cv2.findContours(hm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ovl, cnt, -1, (0, 255, 0), 5)   # horses outlined in green
        if a.ids:
            big = cv2.resize(ovl, (960, 540), interpolation=cv2.INTER_AREA)
            for e in meta:
                if e.get("kind", "body") != "body" or e["area"] < 1500:
                    continue
                ys, xs = np.where(masks[e["id"]][::8, ::8])
                if len(xs):
                    q = (int(np.median(xs) * 4) - 8, int(np.median(ys) * 4) + 6)
                    cv2.putText(big, str(e["id"]), q, 0, 0.55, (0, 0, 0), 4); cv2.putText(big, str(e["id"]), q, 0, 0.55, (255, 255, 255), 1)
            cv2.putText(big, stem[3:], (6, 26), 0, 0.8, (0, 0, 0), 5); cv2.putText(big, stem[3:], (6, 26), 0, 0.8, (0, 255, 255), 2)
            cv2.imwrite(f"{a.review}/{stem}.jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:
            cv2.imwrite(f"{a.review}/{stem}.jpg", cv2.resize(ovl, (480, 270), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])
        if a.gt:
            from src.metrics.segmentation_metrics import compute_segmentation_metrics
            gt = cv2.imread(f"{a.gt}/{stem}.png", cv2.IMREAD_UNCHANGED)
            pred = lab.copy(); gt = gt.copy(); gt[pred == 255] = 255; pred[pred == 255] = 0
            fm[stem] = compute_segmentation_metrics(pred, gt)
            print(stem, round(float(fm[stem].mean_iou), 3), "void frac", stats[stem]["void"], flush=True)
    json.dump(stats, open(stats_p, "w"), indent=0)
    if fm:
        from src.metrics.segmentation_metrics import aggregate_dataset_metrics
        r = aggregate_dataset_metrics(fm)
        print("AGG fg micro-mIoU (Void excluded)", r.mean_iou, r.iou_per_class)
    print("labelled", len(stats))


if __name__ == "__main__":
    main()
