"""Hand corrections on top of the composed teacher labels (used by compose.py --manual).

manual.json: {"<stem>": [op, ...]}   ops are applied in order, coordinates in full-frame pixels
  {"sam": {"box": [x0,y0,x1,y1], "pos": [[x,y],...], "neg": [[x,y],...]}, "cls": c}   SAM 2.1 mask from prompts
  {"poly": [[x,y],...], "cls": c}        polygon painted directly
  {"inst": id, "cls": c}                 a teacher instance mask
  {"fill": c, "max": area}               holes (label 0, not touching the border) enclosed by class c, up to `max` px
  optional on sam/poly/inst:  "only": [labels that may be overwritten]   "erode"/"dilate": px
Running this file computes and caches the SAM masks (<cache>/<stem>/<hash>.png); compose only reads them.
"""
import argparse, hashlib, json, os, sys

import cv2
import numpy as np


def _key(sam):
    return hashlib.md5(json.dumps(sam, sort_keys=True).encode()).hexdigest()[:12]


def apply_manual(lab, ops, cache, stem, masks=None):
    H, W = lab.shape
    for op in ops:
        if "fill" in op:
            c = op["fill"]
            n, cc, st, _ = cv2.connectedComponentsWithStats((lab == 0).astype(np.uint8), connectivity=4)
            ring = cv2.dilate((lab == c).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            other = cv2.dilate(((lab != c) & (lab != 0)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            for i in range(1, n):
                x, y, w, h, area = st[i]
                if area > op.get("max", 5000) or x == 0 or y == 0 or x + w == W or y + h == H:
                    continue
                m = cc == i
                if ring[m].any() and not other[m].any():
                    lab[m] = c
            continue
        if "sam" in op:
            p = f"{cache}/{stem}/{_key(op['sam'])}.png"
            if not os.path.exists(p):
                raise SystemExit(f"{stem}: SAM mask not cached, run aot/manual.py first ({op['sam']})")
            m = cv2.imread(p, 0) > 0
        elif "poly" in op:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.array(op["poly"], np.int32)], 1)
            m = m.astype(bool)
        else:
            m = masks[op["inst"]].copy()
        if op.get("erode"):
            m = cv2.erode(m.astype(np.uint8), np.ones((2 * op["erode"] + 1,) * 2, np.uint8)).astype(bool)
        if op.get("dilate"):
            m = cv2.dilate(m.astype(np.uint8), np.ones((2 * op["dilate"] + 1,) * 2, np.uint8)).astype(bool)
        if "only" in op:
            m &= np.isin(lab, op["only"])
        lab[m] = op["cls"]
    return lab


def main():
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--cache", required=True); ap.add_argument("--stems", nargs="*")
    a = ap.parse_args()
    man = json.load(open(a.manual))
    todo = []
    for stem, ops in man.items():
        if a.stems and stem[-4:] not in a.stems and stem not in a.stems:
            continue
        for op in ops:
            if "sam" in op and not os.path.exists(f"{a.cache}/{stem}/{_key(op['sam'])}.png"):
                todo.append((stem, op["sam"]))
    if not todo:
        print("nothing to do"); return
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    sam_id = "facebook/sam2.1-hiera-large"
    proc = Sam2Processor.from_pretrained(sam_id); sam = Sam2Model.from_pretrained(sam_id).to(dev).eval()
    for stem, s in todo:
        img = Image.open(f"{a.images}/{stem}.png").convert("RGB")
        kw = {}
        if "box" in s:
            kw["input_boxes"] = [[[float(v) for v in s["box"]]]]
        pts = [(p, 1) for p in s.get("pos", [])] + [(p, 0) for p in s.get("neg", [])]
        if pts:
            kw["input_points"] = [[[[float(p[0]), float(p[1])] for p, _ in pts]]]
            kw["input_labels"] = [[[l for _, l in pts]]]
        with torch.no_grad():
            si = proc(images=img, return_tensors="pt", **kw).to(dev)
            so = sam(**si, multimask_output=False)
            m = proc.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0]
        m = m[0, 0].numpy().astype(np.uint8) * 255
        os.makedirs(f"{a.cache}/{stem}", exist_ok=True)
        cv2.imwrite(f"{a.cache}/{stem}/{_key(s)}.png", m)
        print(stem, _key(s), "area", int((m > 0).sum()), "iou", round(float(so.iou_scores.reshape(-1)[0]), 3), flush=True)


if __name__ == "__main__":
    main()
