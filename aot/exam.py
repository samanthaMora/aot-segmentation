"""Private exam: score ONNX models on held-out frames that are never used for training.

python aot/exam.py model1.onnx [model2.onnx ...]
Reports the official fg micro-mIoU on work/exam (Void ignored), the same on the kit's val GT, and the share of
verified horse pixels that the model paints as Titan/Human (the error the public audit showed)."""
import os, sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics.segmentation_metrics import aggregate_dataset_metrics, compute_segmentation_metrics  # noqa


def run(sess, path):
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return sess.run(None, {"image": rgb.transpose(2, 0, 1)[None].copy()})[0][0]


def main():
    stems = [l.strip() for l in open(f"{ROOT}/work/exam/list.txt") if l.strip()]
    vals = sorted(f[:-4] for f in os.listdir(f"{ROOT}/data/val_in_domain/masks") if f.endswith(".png"))
    for mp in sys.argv[1:]:
        s = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        fm, hp, hn = {}, 0, 0
        for st in stems:
            pr = run(s, f"{ROOT}/data/extra/images/{st}.png")
            gt = cv2.imread(f"{ROOT}/work/exam/labels/{st}.png", cv2.IMREAD_UNCHANGED)
            fm[st] = compute_segmentation_metrics(pr, gt)
            hpth = f"{ROOT}/work/exam/horses/{st}.png"
            if os.path.exists(hpth):
                hm = cv2.imread(hpth, 0) > 0
                hp += int((pr[hm] > 0).sum()); hn += int(hm.sum())
        r = aggregate_dataset_metrics(fm)
        vm = {st: compute_segmentation_metrics(run(s, f"{ROOT}/data/val_in_domain/images/{st}.png"),
                                               cv2.imread(f"{ROOT}/data/val_in_domain/masks/{st}.png", cv2.IMREAD_UNCHANGED)) for st in vals}
        v = aggregate_dataset_metrics(vm)
        per = r.iou_per_class
        print(f"{os.path.basename(mp):26s} EXAM {r.mean_iou:.4f} (titan {per['Pure_Titan']:.3f} human {per['Human']:.3f})"
              f" | horse painted as figure {100 * hp / max(1, hn):5.1f}% | VAL {v.mean_iou:.4f} (frame5 {vm[vals[4]].mean_iou:.2f})", flush=True)


if __name__ == "__main__":
    main()
