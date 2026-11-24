"""Export a trained student checkpoint to the contest ONNX contract and benchmark it on CPU."""
import argparse, os, sys, time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model  # noqa


class AoTOnnxWrapper(nn.Module):
    def __init__(self, seg_model, h, w, fast_up=False, pre="bilinear"):
        super().__init__()
        self.fast_up, self.pre = fast_up, pre
        self.seg_model = seg_model.eval()
        self.h, self.w = h, w
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, image):
        x = image.to(torch.float32)
        if self.pre == "pool2":   # exact 2x area average first: kills aliasing when downscaling >2x
            x = F.avg_pool2d(x, 2)
        x = F.interpolate(x, size=(self.h, self.w), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        # DinoSeg: take the 1/4-resolution logits directly (one upsample instead of two)
        logits = self.seg_model.logits4(x) if hasattr(self.seg_model, "logits4") else self.seg_model(x)
        if self.fast_up:
            # bilinear to 540x960, argmax, then nearest x2 (<=1px error, well inside 3px tolerance)
            logits = F.interpolate(logits, size=(540, 960), mode="bilinear", align_corners=False)
            m = torch.argmax(logits, dim=1, keepdim=True).to(torch.uint8)
            return F.interpolate(m, scale_factor=2.0, mode="nearest")[:, 0]
        logits = F.interpolate(logits, size=(1080, 1920), mode="bilinear", align_corners=False)
        return torch.argmax(logits, dim=1).to(torch.uint8)


class FoldedSegformer(nn.Module):
    """Mathematically equivalent SegFormer with the decoder folded.

    The stock decoder upsamples four 256-ch maps to 1/4 res, concatenates them and runs a
    1024->256 1x1 conv there (~20% of CPU time). MLP, bilinear upsample, 1x1 conv and BN are all
    linear, so they commute: project each stage straight to the fused 256-ch space at its own
    (low) resolution, upsample, and sum.
    """

    def __init__(self, m):
        super().__init__()
        self.encoder, self.head = m.encoder, m.segmentation_head
        conv, bn = m.decoder.fuse_stage[0], m.decoder.fuse_stage[1]
        W = conv.weight[:, :, 0, 0].double()                      # [256, 4*256]
        scale = bn.weight.double() / torch.sqrt(bn.running_var.double() + bn.eps)
        shift = bn.bias.double() - bn.running_mean.double() * scale
        if conv.bias is not None:
            shift = shift + conv.bias.double() * scale
        S = W.shape[0]
        self.proj = nn.ModuleList()
        for i, mlp in enumerate(m.decoder.mlp_stage):
            Wi = W[:, i * S:(i + 1) * S] * scale[:, None]
            lin = mlp.linear
            c = nn.Conv2d(lin.in_features, S, 1)
            c.weight.data = (Wi @ lin.weight.double()).float()[:, :, None, None]
            c.bias.data = (Wi @ lin.bias.double() + (shift if i == 0 else 0)).float()
            self.proj.append(c)

    def forward(self, x):
        f = self.encoder(x)
        size = [d // 4 for d in f[0].shape[2:]]
        f = (f[2:] if f[1].size(1) == 0 else f[1:])[::-1]
        out = None
        for p, t in zip(self.proj, f):
            y = p(t)
            if list(y.shape[2:]) != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            out = y if out is None else out + y
        return self.head(F.relu(out))


def bench(path, n=10):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.randint(0, 255, (1, 3, 1080, 1920), np.uint8)
    s.run(None, {"image": x})
    ts = []
    for _ in range(n):
        t = time.perf_counter(); s.run(None, {"image": x}); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--out", required=True)
    ap.add_argument("--arch"); ap.add_argument("--encoder"); ap.add_argument("--h", type=int); ap.add_argument("--w", type=int)
    ap.add_argument("--fast-up", action="store_true")
    ap.add_argument("--pre", default="bilinear", choices=["bilinear", "pool2"])
    ap.add_argument("--fold", action="store_true", help="fold the SegFormer decoder (exact, faster)")
    a = ap.parse_args()
    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu")
        model = build_model(ck["arch"], ck["encoder"], weights=None)
        model.load_state_dict(ck["state"]); h, w = a.h or ck["h"], a.w or ck["w"]
        if a.fold:
            ref = model.eval(); model = FoldedSegformer(ref).eval()
            with torch.no_grad():
                x = torch.randn(1, 3, h, w)
                r, o = ref(x), model(x)
            print(f"fold check: max|dlogit| {(r - o).abs().max():.2e}, argmax diff px {(r.argmax(1) != o.argmax(1)).sum().item()}")
    elif a.arch == "identity":  # wrapper overhead only
        model = nn.Conv2d(3, 3, 1); h, w = a.h, a.w
    else:  # latency probe with random weights
        model = build_model(a.arch, a.encoder, weights=None); h, w = a.h, a.w
    wrapper = AoTOnnxWrapper(model, h, w, a.fast_up, a.pre).cpu().eval()
    dummy = torch.zeros((1, 3, 1080, 1920), dtype=torch.uint8)
    kw = dict(export_params=True, opset_version=17, do_constant_folding=True,
              input_names=["image"], output_names=["mask"])
    try:
        torch.onnx.export(wrapper, (dummy,), a.out, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(wrapper, (dummy,), a.out, **kw)
    print(f"{a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB, CPU p50 {bench(a.out):.0f} ms")


if __name__ == "__main__":
    main()
