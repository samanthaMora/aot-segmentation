"""Run an exported ONNX on a folder, report official metrics and save pred|GT overlays."""
import argparse, glob, os, sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics.segmentation_metrics import aggregate_dataset_metrics, compute_segmentation_metrics  # noqa

PAL = np.zeros((256, 3), np.uint8)
PAL[1] = (60, 20, 220); PAL[2] = (255, 191, 0); PAL[255] = (255, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", default="data/val_in_domain/images")
    ap.add_argument("--gt", default="data/val_in_domain/masks")
    ap.add_argument("--out")
    a = ap.parse_args()
    s = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    fm, rows = {}, []
    for p in sorted(glob.glob(f"{a.images}/*.png")):
        stem = os.path.basename(p)
        bgr = cv2.imread(p)
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        pred = s.run(None, {"image": np.ascontiguousarray(x)})[0][0].astype(np.uint8)
        vis = [(0.5 * bgr + 0.5 * PAL[pred]).astype(np.uint8)]
        gp = f"{a.gt}/{stem}" if a.gt else None
        if gp and os.path.exists(gp):
            gt = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
            r = compute_segmentation_metrics(pred, gt); fm[stem] = r
            print(stem, r.mean_iou, r.iou_per_class)
            vis.append((0.5 * bgr + 0.5 * PAL[gt]).astype(np.uint8))
        rows.append(cv2.resize(np.hstack(vis), (480 * len(vis), 270)))
    if fm:
        agg = aggregate_dataset_metrics(fm)
        print("AGG fg micro-mIoU", agg.mean_iou, agg.iou_per_class)
    if a.out:
        cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
