"""Detect horses (Grounding DINO) and segment them (SAM2.1) -> binary horse masks, used as hard negatives."""
import argparse, glob, os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.35)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gp = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    gd = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(DEV).eval()
    sp = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(DEV).eval()
    for p in sorted(glob.glob(f"{a.images}/*.png")):
        img = Image.open(p).convert("RGB"); W, H = img.size
        with torch.no_grad():
            inp = gp(images=img, text="horse.", return_tensors="pt").to(DEV)
            res = gp.post_process_grounded_object_detection(gd(**inp), inp.input_ids, threshold=a.thr,
                                                            text_threshold=0.25, target_sizes=[(H, W)])[0]
        b = res["boxes"].float().cpu()
        if len(b):
            ar = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = b[(ar > 600) & (ar < 0.6 * W * H)]
        out = np.zeros((H, W), np.uint8)
        if len(b):
            with torch.no_grad():
                si = sp(images=img, input_boxes=[b.tolist()], return_tensors="pt").to(DEV)
                so = sam(**si, multimask_output=False)
                m = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0][:, 0].numpy().astype(bool)
            for i in range(len(m)):
                out[m[i]] = i + 1
        cv2.imwrite(f"{a.out}/{os.path.basename(p)}", out)
        print(os.path.basename(p), len(b), [round(float(s), 2) for s in res["scores"]], flush=True)


if __name__ == "__main__":
    main()
