"""Second opinion on the teacher's horse detections: Grounding DINO with the single prompt "horse." also fires
on whatever is salient (faces, titans). CLIP on the masked crop decides; writes p_horse into the meta files."""
import argparse, glob, json, os, sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compose import load_inst  # noqa

PROMPTS = ["an anime illustration of a horse", "a brown horse with a saddle", "the head of a horse",
           "an anime illustration of a person", "the face of an anime character", "a person wearing a cape or a hood",
           "a naked giant humanoid monster", "a building or a wall", "trees and grass", "smoke, clouds or sky"]
N_HORSE = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--images", required=True)
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cl = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    with torch.no_grad():
        tf = cl.get_text_features(**proc(text=PROMPTS, return_tensors="pt", padding=True).to(dev))
        tf = getattr(tf, "pooler_output", tf); tf = tf / tf.norm(dim=-1, keepdim=True)
    n = k = 0
    for mp in sorted(glob.glob(f"{a.teacher}/meta/*.json")):
        stem = os.path.basename(mp)[:-5]
        try:
            masks, meta = load_inst(a.teacher, stem)
        except Exception:
            continue
        hs = [e for e in meta if e.get("kind") == "horse"]
        if not hs:
            continue
        img = np.array(Image.open(f"{a.images}/{stem}.png").convert("RGB"))
        crops = []
        for e in hs:
            x0, y0, x1, y1 = [int(v) for v in e["box"]]
            c = img[y0:y1 + 1, x0:x1 + 1].copy()
            m = masks[e["id"]][y0:y1 + 1, x0:x1 + 1]
            c[~m] = (0.35 * c[~m] + 0.65 * 127).astype(np.uint8)   # fade what is outside the mask
            crops.append(Image.fromarray(c))
        with torch.no_grad():
            f = cl.get_image_features(**proc(images=crops, return_tensors="pt").to(dev))
            f = getattr(f, "pooler_output", f); f = f / f.norm(dim=-1, keepdim=True)
            pr = (100 * f @ tf.T).softmax(-1).cpu().numpy()
        for e, p in zip(hs, pr):
            e["p_horse"] = round(float(p[:N_HORSE].sum()), 3)
            n += 1; k += e["p_horse"] > 0.5
        json.dump(meta, open(mp, "w"), indent=1)
    print(f"horse detections: {n}, confirmed by CLIP: {k}")


if __name__ == "__main__":
    main()
