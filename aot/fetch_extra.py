"""Download extra anime screenshots (per-episode image categories of a public MediaWiki) as 1920x1080 PNGs.

Only 16:9 images at least --min-w wide are kept; everything is decoded and re-encoded, nothing is stored raw.
"""
import argparse, json, os, ssl, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"}


CTX = ssl.create_default_context(cafile="/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else None)


def get(url, tries=3):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30, context=CTX) as r:
                return r.read()
        except Exception as e:
            err = e; time.sleep(1 + 2 * i)
    print("fetch failed:", url[:90], type(err).__name__, err, flush=True)
    return None


def list_files(api, cat):
    out, cont = [], {}
    while True:
        q = dict(action="query", generator="categorymembers", gcmtitle=f"Category:{cat}", gcmtype="file", gcmlimit="100",
                 prop="imageinfo", iiprop="url|size|mime", format="json", **cont)
        d = json.loads(get(f"{api}?{urllib.parse.urlencode(q)}") or b"{}")
        for p in (d.get("query", {}).get("pages", {}) or {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("url"):
                out.append((p["title"], ii["url"], ii.get("width", 0), ii.get("height", 0), ii.get("mime", "")))
        if "continue" not in d:
            return out
        cont = d["continue"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="https://attackontitan.fandom.com/api.php")
    ap.add_argument("--episodes", default="1-59"); ap.add_argument("--out", required=True)
    ap.add_argument("--min-w", type=int, default=1280)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = map(int, a.episodes.split("-"))
    jobs, seen = [], set()
    for ep in range(lo, hi + 1):
        fs = list_files(a.api, f"Episode {ep} images")
        keep = [f for f in fs if f[2] >= a.min_w and f[3] and abs(f[2] / f[3] - 16 / 9) < 0.02
                and f[4] in ("image/png", "image/jpeg") and f[1] not in seen]
        seen.update(f[1] for f in keep)
        jobs += [(ep, i, f) for i, f in enumerate(keep)]
        print(f"ep {ep}: {len(fs)} files, {len(keep)} usable", flush=True)

    def dl(job):
        ep, i, (title, url, w, h, _) = job
        dst = f"{a.out}/x_e{ep:02d}_{i:02d}.png"
        if os.path.exists(dst):
            return 1
        b = get(url)
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) if b else None
        if im is None or im.ndim != 3:
            return 0
        if im.shape[:2] != (1080, 1920):
            im = cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA if im.shape[1] > 1920 else cv2.INTER_CUBIC)
        cv2.imwrite(dst, im)
        return 1

    with ThreadPoolExecutor(4) as ex:
        ok = sum(ex.map(dl, jobs))
    json.dump([{"file": f"x_e{ep:02d}_{i:02d}.png", "title": f[0], "w": f[2], "h": f[3]} for ep, i, f in jobs],
              open(f"{a.out}/../extra_index.json", "w"), indent=0)
    print(f"downloaded {ok}/{len(jobs)}")


if __name__ == "__main__":
    main()
