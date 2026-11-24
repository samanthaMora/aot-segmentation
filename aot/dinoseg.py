"""DinoSeg: a DINOv3 ViT decides *what* each region is, a small conv branch decides *where* its edges are.

With ~70 labelled frames a backbone fine-tuned end to end learns shortcuts (big figure = titan, horse = human).
A linear probe on frozen DINOv3 ViT-S/16 features already separates titan / human / horse on unseen episodes
better than the fully trained SegFormer, so the ViT is kept (almost) as pretrained and run on the whole frame
at reduced resolution; full-resolution detail comes from three stride-2 convs on the input image.

encoder string: "<timm vit name>[|s<vit downscale>][|L<a+b+c>][|w<width>][|b<blocks kept>][|c<d4,d8,f8,f4 channels>]"
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class DinoSeg(nn.Module):
    def __init__(self, encoder="vit_small_patch16_dinov3.lvd1689m", pretrained=True, classes=3):
        super().__init__()
        parts = encoder.split("|")
        opt = {p[0]: p[1:] for p in parts[1:]}
        self.scale = int(opt.get("s", 2))
        w = int(opt.get("w", 64))
        self.vit = timm.create_model(parts[0], pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        if "b" in opt:
            self.vit.blocks = self.vit.blocks[:int(opt["b"])]
        n = len(self.vit.blocks)
        self.idx = [int(v) - 1 for v in opt.get("L", f"{n // 4}+{n // 2}+{3 * n // 4}+{n}").split("+")]
        d = self.vit.embed_dim
        self.proj = nn.ModuleList([nn.Conv2d(d, w, 1) for _ in self.idx])
        self.sem = cbr(w, w)
        c4, c8, o8, o4 = (int(v) for v in opt.get("c", f"32,64,{w},48").split(","))   # the two fusion convs are ~15% of CPU time
        self.d2, self.d4, self.d8 = cbr(3, 16, s=2), cbr(16, c4, s=2), cbr(c4, c8, s=2)
        self.f8 = cbr(w + c8, o8)
        self.f4 = cbr(o8 + c4, o4)
        self.head = nn.Conv2d(o4, classes, 1)
        self.frozen = False

    def freeze_vit(self):
        self.frozen = True
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def logits4(self, x):
        """x: normalised image, H and W multiples of 16 * scale. Returns logits at 1/4 of the input."""
        v = F.avg_pool2d(x, self.scale) if self.scale > 1 else x
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            fs = self.vit.forward_intermediates(v, indices=self.idx, norm=True, output_fmt="NCHW", intermediates_only=True)
        s = None
        for p, f in zip(self.proj, fs):
            s = p(f) if s is None else s + p(f)
        s = self.sem(s)
        d4 = self.d4(self.d2(x)); d8 = self.d8(d4)
        y = self.f8(torch.cat([F.interpolate(s, size=d8.shape[2:], mode="bilinear", align_corners=False), d8], 1))
        y = self.f4(torch.cat([F.interpolate(y, size=d4.shape[2:], mode="bilinear", align_corners=False), d4], 1))
        return self.head(y)

    def forward(self, x):
        return F.interpolate(self.logits4(x), size=x.shape[2:], mode="bilinear", align_corners=False)
