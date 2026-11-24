"""Train the lightweight student segmenter on pseudo-labels (+ optional GT frames).

Model runs at --h x --w (default 544x960); the ONNX wrapper handles resize.
"""
import argparse, glob, math, os, random, sys, time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics.segmentation_metrics import aggregate_dataset_metrics, compute_segmentation_metrics  # noqa

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


ADE_HUB = {"mit_b0": "smp-hub/segformer-b0-512x512-ade-160k", "mit_b1": "smp-hub/segformer-b1-512x512-ade-160k",
           "mit_b2": "smp-hub/segformer-b2-512x512-ade-160k"}


def build_model(arch, encoder, weights="imagenet"):
    if arch == "dinoseg":  # DINOv3 ViT for semantics + small conv branch for edges (see dinoseg.py)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dinoseg import DinoSeg
        return DinoSeg(encoder, pretrained=weights is not None)
    if weights == "ade":  # encoder + decoder pretrained on ADE20K segmentation, fresh 3-class head
        m = smp.from_pretrained(ADE_HUB[encoder])
        old = m.segmentation_head[0]
        m.segmentation_head[0] = torch.nn.Conv2d(old.in_channels, 3, old.kernel_size, padding=old.padding)
        return m
    A = {"unet": smp.Unet, "fpn": smp.FPN, "dlv3p": smp.DeepLabV3Plus, "segformer": smp.Segformer,
         "upernet": smp.UPerNet}[arch]
    return A(encoder_name=encoder, encoder_weights=weights, classes=3)


def load_pairs(img_dir, lab_dir, store=1.0):
    """store < 1 keeps frames downscaled in RAM (enough when training below that scale; workers copy the dataset)."""
    out = []
    for p in sorted(glob.glob(f"{img_dir}/*.png")):
        q = f"{lab_dir}/{os.path.basename(p)}"
        if os.path.exists(q):
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            lb = cv2.imread(q, cv2.IMREAD_UNCHANGED)
            if store != 1.0:
                sz = (int(round(im.shape[1] * store)), int(round(im.shape[0] * store)))
                im = cv2.resize(im, sz, interpolation=cv2.INTER_AREA)
                lb = cv2.resize(lb, sz, interpolation=cv2.INTER_NEAREST)
            out.append((im, lb))
    return out


class LazyFrames:
    """(image, banded label, raw label) read from disk on access: spawned DataLoader workers each get a copy of the
    dataset, so hundreds of 1080p frames kept in RAM do not fit."""
    def __init__(self, items, band, store=1.0):
        self.items, self.band, self.store = list(items), band, store

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, q = self.items[i]
        im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        lb = cv2.imread(q, cv2.IMREAD_UNCHANGED)
        if self.store != 1.0:
            sz = (int(round(im.shape[1] * self.store)), int(round(im.shape[0] * self.store)))
            im = cv2.resize(im, sz, interpolation=cv2.INTER_AREA); lb = cv2.resize(lb, sz, interpolation=cv2.INTER_NEAREST)
        return im, (ignore_band(lb, self.band) if self.band else lb), lb


def lazy_items(img_dir, lab_dir):
    return [(p, f"{lab_dir}/{os.path.basename(p)}") for p in sorted(glob.glob(f"{img_dir}/*.png"))
            if os.path.exists(f"{lab_dir}/{os.path.basename(p)}")]


def ignore_band(lb, px=3):
    """Mark a thin band around class boundaries as 255 (matches metric tolerance, hides label noise)."""
    lb = lb.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    band = np.zeros(lb.shape, bool)
    for c in (1, 2):
        m = (lb == c).astype(np.uint8)
        if m.any():
            band |= cv2.dilate(m, k) != cv2.erode(m, k)
    lb[band] = 255
    return lb


def horse_objects(img_dir, lab_dir, horse_dir, store=1.0):
    """Horse (+ its rider) cut-outs: (image crop, label crop, object mask). Horse pixels keep label 0."""
    out = []
    for p in sorted(glob.glob(f"{horse_dir}/*.png")):
        b = os.path.basename(p)
        im = cv2.cvtColor(cv2.imread(f"{img_dir}/{b}"), cv2.COLOR_BGR2RGB)
        lb = cv2.imread(f"{lab_dir}/{b}", cv2.IMREAD_UNCHANGED); hm = cv2.imread(p, 0)
        n_, cc = cv2.connectedComponents((lb == 2).astype(np.uint8))
        for k in range(1, int(hm.max()) + 1):
            m = hm == k
            if m.sum() < 300:
                continue
            near = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
            obj = m | np.isin(cc, [c for c in np.unique(cc[near]) if c])
            ys, xs = np.where(obj)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            l = lb[y0:y1, x0:x1].copy(); l[l == 1] = 255
            o = (im[y0:y1, x0:x1].copy(), l, obj[y0:y1, x0:x1].copy())
            if store != 1.0:  # same scale as the stored frames
                sz = (max(4, int(round(o[0].shape[1] * store))), max(4, int(round(o[0].shape[0] * store))))
                o = (cv2.resize(o[0], sz, interpolation=cv2.INTER_AREA), cv2.resize(o[1], sz, interpolation=cv2.INTER_NEAREST),
                     cv2.resize(o[2].astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST).astype(bool))
            out.append(o)
    return out


class DS(torch.utils.data.Dataset):
    def __init__(self, pairs, h, w, n, band, crop=(384, 384), horses=(), zoom=(0.65, 1.6), haze=0.0, closeup=0.0):
        self.horses = list(horses)
        self.p = pairs if isinstance(pairs, LazyFrames) else [(im, ignore_band(lb, band) if band else lb, lb) for im, lb in pairs]
        (self.h, self.w), self.n = crop, n
        self.base = h / 1080.0  # inference scale factor
        self.zoom, self.haze = zoom, haze
        # close-ups: figures of each class with a few of their own pixels as candidate crop centres
        self.closeup, self.blobs = closeup, {1: [], 2: []}
        if closeup:
            rs = np.random.RandomState(0)
            self.hw = []
            for fi, (_, _, raw) in enumerate(self.p):
                self.hw.append(raw.shape)
                for c in (1, 2):
                    n_, cc, st, _ = cv2.connectedComponentsWithStats((raw == c).astype(np.uint8))
                    for k in range(1, n_):
                        if st[k, cv2.CC_STAT_AREA] >= 2500 * (raw.shape[0] / 1080.0) ** 2:  # noqa
                            ys, xs = np.where(cc == k)
                            j = rs.randint(0, len(ys), 32)
                            self.blobs[c].append((fi, int(st[k, cv2.CC_STAT_HEIGHT]), ys[j], xs[j]))
            self.blobs = {c: b for c, b in self.blobs.items() if b}

    def _closeup(self):
        """Crop window in which one figure (or part of it) fills the view, as in the anime's face shots.
        Random crops at the inference scale never show a Human that large, so size alone would say 'Titan'."""
        fi, bh, ys, xs = random.choice(self.blobs[random.choice(list(self.blobs))])
        H, W = self.hw[fi] if hasattr(self, "hw") else self.p[fi][1].shape
        k = H / 1080.0
        ch = bh * random.uniform(0.4, 1.2)
        ch = min(H, max(ch, self.h * k / 2.5))            # never enlarge more than 2.5x beyond the source pixels
        cw = min(W, ch * self.w / self.h); ch = min(ch, cw * self.h / self.w)
        ch, cw = int(round(ch)), int(round(cw))
        j = random.randrange(len(ys))
        y0 = int(min(max(ys[j] - ch // 2, 0), H - ch)); x0 = int(min(max(xs[j] - cw // 2, 0), W - cw))
        return fi, y0, x0, ch, cw, self.h * k / ch

    def _crop_hw(self, H=1080, W=1920):
        lo, hi = self.zoom   # zoom jitter around inference scale; log-uniform so wide ranges do not favour close-ups
        sf = self.base * (math.exp(random.uniform(math.log(lo), math.log(hi))) if hi / lo > 2 else random.uniform(lo, hi))
        # the source crop must fit in the frame, so the scale cannot go below crop/frame:
        # a 384x384 crop never sees scales under 384/1080 -- use a short crop to train at low inference scales
        sf = max(sf, self.h / 1080.0, self.w / 1920.0)
        k = H / 1080.0  # frames may be stored downscaled
        return min(H, int(round(self.h / sf * k))), min(W, int(round(self.w / sf * k))), sf

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.closeup and self.blobs and random.random() < self.closeup:
            fi, y0, x0, ch, cw, s = self._closeup()
            im, lb, raw = self.p[fi]
        else:
            im, lb, raw = self.p[random.randrange(len(self.p))]
            H, W = lb.shape
            # random square crop at (jittered) inference scale
            ch, cw, s = self._crop_hw(H, W)
            y0, x0 = random.randint(0, H - ch), random.randint(0, W - cw)
        im = im[y0:y0 + ch, x0:x0 + cw]; lb = lb[y0:y0 + ch, x0:x0 + cw]; raw = raw[y0:y0 + ch, x0:x0 + cw]
        # copy-paste: paste a foreground instance blob from another frame
        if random.random() < 0.5:
            im, lb = im.copy(), lb.copy()
            sim, slb, sraw = self.p[random.randrange(len(self.p))]
            fg = (sraw == 1) | (sraw == 2)
            if fg.any():
                n_, cc = cv2.connectedComponents(fg.astype(np.uint8))
                k = random.randint(1, n_ - 1)
                ys, xs = np.where(cc == k)
                by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                sc = random.uniform(0.3, 1.0) * s
                ph, pw = max(8, int((by1 - by0) * sc / s)), max(8, int((bx1 - bx0) * sc / s))
                ph, pw = min(ph, ch), min(pw, cw)
                pim = cv2.resize(sim[by0:by1, bx0:bx1], (pw, ph))
                pmk = cv2.resize((cc[by0:by1, bx0:bx1] == k).astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
                plb = cv2.resize(slb[by0:by1, bx0:bx1], (pw, ph), interpolation=cv2.INTER_NEAREST)
                ty, tx = random.randint(0, ch - ph), random.randint(0, cw - pw)
                im[ty:ty + ph, tx:tx + pw][pmk] = pim[pmk]
                lb[ty:ty + ph, tx:tx + pw][pmk] = plb[pmk]
        # hard negative: paste a horse (background) together with its rider (Human)
        if self.horses and random.random() < 0.3:
            im, lb = im.copy(), lb.copy()
            him, hlb, hmk = self.horses[random.randrange(len(self.horses))]
            u = random.uniform(0.6, 2.0)
            ph, pw = min(ch, max(8, int(hmk.shape[0] * u))), min(cw, max(8, int(hmk.shape[1] * u)))
            pim = cv2.resize(him, (pw, ph)); plb = cv2.resize(hlb, (pw, ph), interpolation=cv2.INTER_NEAREST)
            pmk = cv2.resize(hmk.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
            if random.random() < 0.5:
                pim, plb, pmk = pim[:, ::-1], plb[:, ::-1], pmk[:, ::-1]
            ty, tx = random.randint(0, ch - ph), random.randint(0, cw - pw)
            im[ty:ty + ph, tx:tx + pw][pmk] = pim[pmk]
            lb[ty:ty + ph, tx:tx + pw][pmk] = plb[pmk]
        im = cv2.resize(im, (self.w, self.h), interpolation=cv2.INTER_AREA if im.shape[0] >= self.h else cv2.INTER_LINEAR)
        lb = cv2.resize(lb, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if random.random() < 0.5:
            im, lb = im[:, ::-1], lb[:, ::-1]
        im = im.astype(np.float32)
        # photometric
        if random.random() < 0.8:
            hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
            hsv[..., 1] *= random.uniform(0.6, 1.4)
            im = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            im = im * random.uniform(0.6, 1.35) + random.uniform(-25, 25)
            im = im * np.random.uniform(0.85, 1.15, (1, 1, 3)).astype(np.float32)
        if random.random() < 0.2:
            im = cv2.GaussianBlur(im, (0, 0), random.uniform(0.5, 2.0))
        if random.random() < 0.15:
            im = np.repeat(im.mean(2, keepdims=True), 3, 2)
        if random.random() < self.haze:  # mist / backlight: blend towards a light tint along a random gradient
            col = np.array(random.choice([(255, 235, 200), (235, 240, 255), (255, 200, 140), (205, 220, 235)]),
                           np.float32) + np.random.uniform(-20, 20, 3).astype(np.float32)
            yy, xx = np.mgrid[0:im.shape[0], 0:im.shape[1]].astype(np.float32)
            th = random.uniform(0, 2 * math.pi)
            ramp = (xx / im.shape[1] - 0.5) * math.cos(th) + (yy / im.shape[0] - 0.5) * math.sin(th) + 0.5
            al = (random.uniform(0.15, 0.55) * (0.4 + 0.6 * np.clip(ramp, 0, 1)))[..., None]
            im = im * (1 - al) + col * al
        im = (np.clip(im, 0, 255) - MEAN) / STD
        return torch.from_numpy(im.transpose(2, 0, 1).copy()), torch.from_numpy(lb.astype(np.int64).copy())


def dice_loss(logits, y, ignore=255):
    valid = (y != ignore)
    p = logits.softmax(1)
    oh = F.one_hot(y.clamp(0, 2), 3).permute(0, 3, 1, 2).float()
    v = valid[:, None].float()
    p, oh = p * v, oh * v
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    d = 1 - (2 * inter + 1) / (den + 1)
    return d[1:].mean()


@torch.no_grad()
def predict(model, im, h, w, flip=False):
    x = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(DEV)
    lg = model(x)
    if flip:
        lg = lg + model(x.flip(-1)).flip(-1)
    lg = F.interpolate(lg, size=im.shape[:2], mode="bilinear", align_corners=False)
    return lg.argmax(1)[0].cpu().numpy().astype(np.uint8)


def evaluate(model, pairs, h, w):
    model.eval()
    fm = {i: compute_segmentation_metrics(predict(model, im, h, w), gt) for i, (im, gt) in enumerate(pairs)}
    model.train()
    return aggregate_dataset_metrics(fm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="img_dir:lab_dir pairs")
    ap.add_argument("--val", default="data/val_in_domain/images:data/val_in_domain/masks")
    ap.add_argument("--arch", default="unet"); ap.add_argument("--encoder", default="tu-convnext_tiny")
    ap.add_argument("--h", type=int, default=544); ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--band", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="imagenet", choices=["imagenet", "ade"])
    ap.add_argument("--horses", nargs="*", default=[], help="img_dir:lab_dir:horse_mask_dir")
    ap.add_argument("--crop-h", type=int, default=384); ap.add_argument("--crop-w", type=int, default=384)
    ap.add_argument("--zoom", default="0.65,1.6", help="min,max zoom jitter relative to the inference scale")
    ap.add_argument("--haze", type=float, default=0.0, help="probability of the mist/backlight augmentation")
    ap.add_argument("--resume", help="checkpoint to fine-tune from")
    ap.add_argument("--store", type=float, default=1.0, help="keep training frames at this scale in RAM")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--closeup", type=float, default=0.0, help="probability of a figure-filling close-up crop")
    ap.add_argument("--enc-mult", type=float, default=0.5, help="backbone lr = lr * this; 0 freezes the backbone")
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lazy", action="store_true", help="read training frames from disk on access (low RAM)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    pairs = []
    for t in a.train:  # "img_dir:lab_dir[:repeat]"
        parts = t.split(":")
        rep = int(parts[2]) if len(parts) > 2 else 1
        pairs += (lazy_items(parts[0], parts[1]) if a.lazy else load_pairs(parts[0], parts[1], a.store)) * rep
    if a.lazy:
        pairs = LazyFrames(pairs, a.band, a.store)
    vpairs = load_pairs(*a.val.split(":")) if a.val and a.val != "none" else []
    print(f"train frames {len(pairs)} val frames {len(vpairs)}", flush=True)
    horses = [o for hspec in a.horses for o in horse_objects(*hspec.split(":"), store=a.store)]
    print(f"horse objects {len(horses)}", flush=True)
    ds = DS(pairs, a.h, a.w, a.iters * a.bs, a.band, crop=(a.crop_h, a.crop_w), horses=horses,
            zoom=tuple(float(z) for z in a.zoom.split(",")), haze=a.haze, closeup=a.closeup)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, persistent_workers=True, drop_last=True)
    model = build_model(a.arch, a.encoder, a.init)
    if a.resume:
        model.load_state_dict(torch.load(a.resume, map_location="cpu")["state"])
    is_enc = lambda n: n.startswith("encoder") or n.startswith("vit.")
    if a.enc_mult == 0:
        if hasattr(model, "freeze_vit"):
            model.freeze_vit()
        else:
            [p.requires_grad_(False) for n, p in model.named_parameters() if is_enc(n)]
    model = model.to(DEV).train()
    enc = [p for n, p in model.named_parameters() if is_enc(n) and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not is_enc(n)]
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": enc, "lr": a.lr * a.enc_mult}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(i, a.iters) / a.iters)))
    cw = torch.tensor([1.0, 1.5, 2.0], device=DEV)
    t0 = time.time(); best = -1
    for it, (x, y) in enumerate(dl):
        x, y = x.to(DEV), y.to(DEV)
        with torch.autocast("mps", dtype=torch.float16, enabled=DEV == "mps"):
            lg = model(x)
        lg = lg.float()
        loss = F.cross_entropy(lg, y, weight=cw, ignore_index=255) + dice_loss(lg, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 50 == 0:
            print(f"it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if vpairs and ((it + 1) % a.eval_every == 0 or it + 1 == a.iters):
            r = evaluate(model, vpairs, a.h, a.w)
            print(f"EVAL it {it + 1} fg-mIoU {r.mean_iou} {r.iou_per_class}", flush=True)
        if (it + 1) % a.eval_every == 0 or it + 1 == a.iters:
            torch.save({"arch": a.arch, "encoder": a.encoder, "h": a.h, "w": a.w, "state": model.state_dict()}, a.out)
    print("done", a.out)


if __name__ == "__main__":
    main()
