"""Teacher: Grounding DINO (detect) + CLIP (titan vs human) + SAM2.1 (masks).

Writes, per image:
  <out>/inst/<stem>.npz   instance masks (bool, N x H x W), boxes, det scores, clip probs
  <out>/meta/<stem>.json  instance list (box, det score, p_titan, p_human, p_other, area)
Composition into a label map is done by compose.py so that manual review edits
(class flips / drops) can be applied without re-running the heavy models.
"""
import argparse, glob, json, os, sys

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                          CLIPModel, CLIPProcessor, Sam2Model, Sam2Processor)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"

TITAN_PROMPTS = [
    "an anime illustration of a naked giant titan",
    "a nude giant humanoid monster with no clothes",
    "the face of a naked giant with a creepy grin",
    "bare skin of a huge naked body",
]
HUMAN_PROMPTS = [
    "an anime illustration of a soldier wearing a uniform and jacket",
    "a clothed person wearing a cape and harness",
    "an anime character wearing clothes",
    "a person in a brown jacket with straps and gear",
]
OTHER_PROMPTS = [
    "a horse", "a building", "a tree", "a wall", "the sky with clouds",
    "a roof with tiles", "smoke", "a wooden cart", "grass field", "a sword blade",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-thr", type=float, default=0.22)
    ap.add_argument("--list", help="text file with stems to process, in priority order (default: all, sorted)")
    ap.add_argument("--horses", action="store_true", help="also detect horses (saved as kind='horse' instances)")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/inst", exist_ok=True)
    os.makedirs(f"{args.out}/meta", exist_ok=True)

    gd_id = "IDEA-Research/grounding-dino-base"
    gd_proc = AutoProcessor.from_pretrained(gd_id)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(DEV).eval()
    clip_id = "openai/clip-vit-large-patch14"
    cl_proc = CLIPProcessor.from_pretrained(clip_id)
    cl = CLIPModel.from_pretrained(clip_id).to(DEV).eval()
    sam_id = "facebook/sam2.1-hiera-large"
    sam_proc = Sam2Processor.from_pretrained(sam_id)
    sam = Sam2Model.from_pretrained(sam_id).to(DEV).eval()

    prompts = TITAN_PROMPTS + HUMAN_PROMPTS + OTHER_PROMPTS
    with torch.no_grad():
        t = cl_proc(text=prompts, return_tensors="pt", padding=True).to(DEV)
        tf = cl.get_text_features(**t)
        tf = getattr(tf, "pooler_output", tf)
        tf = tf / tf.norm(dim=-1, keepdim=True)
    nT, nH = len(TITAN_PROMPTS), len(HUMAN_PROMPTS)

    paths = sorted(glob.glob(os.path.join(args.images, "*.png")))
    if args.list:
        paths = [os.path.join(args.images, f"{l.strip()}.png") for l in open(args.list) if l.strip()]
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if args.list and os.path.exists(f"{args.out}/meta/{stem}.json"):
            continue
        img = Image.open(p).convert("RGB")
        W, H = img.size
        boxes, scores, kinds = [], [], []
        # several text queries -> union of boxes; "gear" queries catch 3DMG equipment
        for text, kind, thr in [("person.", 0, args.box_thr), ("naked giant.", 0, args.box_thr),
                                ("human body. face.", 0, args.box_thr),
                                ("metal box. mechanical device.", 1, 0.28)]:
            with torch.no_grad():
                inp = gd_proc(images=img, text=text, return_tensors="pt").to(DEV)
                out = gd(**inp)
            res = gd_proc.post_process_grounded_object_detection(
                out, inp.input_ids, threshold=thr, text_threshold=0.2,
                target_sizes=[(H, W)])[0]
            boxes.append(res["boxes"].float().cpu())
            scores.append(res["scores"].float().cpu())
            kinds.append(torch.full((len(res["scores"]),), kind))
        boxes = torch.cat(boxes); scores = torch.cat(scores); kinds = torch.cat(kinds)
        if len(boxes):
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W - 1)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H - 1)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            ok = (area > 400) & (area < 0.97 * W * H) & ((kinds == 0) | (area < 0.08 * W * H))
            boxes, scores, kinds = boxes[ok], scores[ok], kinds[ok]
            keep = nms(boxes, scores + 10 * (kinds == 0).float(), 0.6)  # person boxes win ties
            boxes, scores, kinds = boxes[keep], scores[keep], kinds[keep]
        if args.horses:  # horses are background: kept as their own instances so compose can protect them
            with torch.no_grad():
                inp = gd_proc(images=img, text="horse.", return_tensors="pt").to(DEV)
                res = gd_proc.post_process_grounded_object_detection(
                    gd(**inp), inp.input_ids, threshold=0.35, text_threshold=0.25, target_sizes=[(H, W)])[0]
            hb, hs = res["boxes"].float().cpu(), res["scores"].float().cpu()
            if len(hb):
                hb[:, 0::2] = hb[:, 0::2].clamp(0, W - 1); hb[:, 1::2] = hb[:, 1::2].clamp(0, H - 1)
                ar = (hb[:, 2] - hb[:, 0]) * (hb[:, 3] - hb[:, 1])
                ok = (ar > 600) & (ar < 0.6 * W * H)
                hb, hs = hb[ok], hs[ok]
                k = nms(hb, hs, 0.6); hb, hs = hb[k], hs[k]
                boxes = torch.cat([boxes, hb]); scores = torch.cat([scores, hs])
                kinds = torch.cat([kinds, torch.full((len(hs),), 2)])

        meta, masks = [], []
        if len(boxes):
            # CLIP on padded crops
            crops = []
            for b in boxes.tolist():
                x0, y0, x1, y1 = b
                pw, ph = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
                crops.append(img.crop((max(0, x0 - pw), max(0, y0 - ph),
                                       min(W, x1 + pw), min(H, y1 + ph))))
            with torch.no_grad():
                ci = cl_proc(images=crops, return_tensors="pt").to(DEV)
                f = cl.get_image_features(**ci)
                f = getattr(f, "pooler_output", f)
                f = f / f.norm(dim=-1, keepdim=True)
                logits = 100 * f @ tf.T
                pr = logits.softmax(-1).cpu().numpy()
            pt = pr[:, :nT].sum(1); ph_ = pr[:, nT:nT + nH].sum(1); po = pr[:, nT + nH:].sum(1)

            # SAM2 masks, batched box prompts
            with torch.no_grad():
                si = sam_proc(images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to(DEV)
                so = sam(**si, multimask_output=False)
                m = sam_proc.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0]
            m = m[:, 0].numpy().astype(bool)
            iou = so.iou_scores.cpu().numpy().reshape(-1)
            for i, b in enumerate(boxes.tolist()):
                meta.append(dict(id=i, kind=("body", "gear", "horse")[int(kinds[i])], box=[round(v, 1) for v in b], det=round(float(scores[i]), 3),
                                 p_titan=round(float(pt[i]), 3), p_human=round(float(ph_[i]), 3),
                                 p_other=round(float(po[i]), 3), sam_iou=round(float(iou[i]), 3),
                                 area=int(m[i].sum())))
            masks = m
        else:
            masks = np.zeros((0, H, W), bool)
        np.savez_compressed(f"{args.out}/inst/{stem}.npz", masks=np.packbits(masks, axis=-1),
                            shape=np.array(masks.shape))
        json.dump(meta, open(f"{args.out}/meta/{stem}.json", "w"), indent=1)
        print(stem, len(meta), flush=True)


if __name__ == "__main__":
    main()
