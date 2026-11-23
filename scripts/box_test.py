"""Test one reviewer-flagged box against calibration (numbers only; no image is written).

The box is compared with every same-sized window (half-overlapping) of: the known-text control,
the blank control, and the rest of the same surface. For each metric the box's percentile rank in
each reference set is reported, so "text-like" means "ranks like known-text windows do".

Metrics per window (on CT-covered pixels, mask eroded 40 px):
  agree_r   mean pairwise Pearson r between the ensemble members
  depth_r   mean Pearson r between the default-window map and the +-3 layer shifted maps
  ct_r      Pearson r between ink map and CT intensity (high = copying CT structure)
  ink_frac  fraction of p' > 0.5 in the ensemble mean
  contrast  p95 - p50 of the ensemble mean (strength of the strongest marks above background)

Usage: python box_test.py SPEC_JSON
  SPEC_JSON: {"box": {"set": "cand", "y": 1600, "x": 2800, "h": 800, "w": 800},
              "sets": {"cand": {...}, "pos": {...}, "neg": {...}}}
  where each set is {"render": zarr, "mean": tif, "shallow": tif, "deep": tif, "members": [tif, ...]}
"""
import json, sys
import numpy as np
import zarr
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None


def q_of(p, shape=None):
    x = np.asarray(Image.open(p)).astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    x = np.clip((x - 0.25) / 0.5, 0, 1)
    if shape is not None and x.shape != shape:
        x = ndi.zoom(x, (shape[0] / x.shape[0], shape[1] / x.shape[1]), order=1)
    return x


def r(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def load_set(s):
    m0 = q_of(s["mean"]); S = m0.shape
    out = dict(m0=m0, sh=q_of(s["shallow"], S), dp=q_of(s["deep"], S), mem=[q_of(p, S) for p in s["members"]])
    g = zarr.open(s["render"], mode="r"); v = np.asarray(g if hasattr(g, "shape") else g["0"])
    v = np.moveaxis(v, int(np.argmin(v.shape)), 0); c = v.shape[0] // 2
    ct = v[c - 2:c + 3].astype(np.float32).mean(0)
    if ct.shape != S:
        ct = ndi.zoom(ct, (S[0] / ct.shape[0], S[1] / ct.shape[1]), order=1)
    out["ct"] = ct; out["valid"] = ndi.binary_erosion(ct > 0, iterations=40)
    out["roi"] = s.get("roi")   # optional [y0, y1, x0, x1]: reference windows only inside it (e.g. a text line)
    return out


def metrics(D, y, x, h, w):
    m = D["valid"][y:y + h, x:x + w]
    if m.sum() < 0.5 * h * w:
        return None
    sl = lambda a: a[y:y + h, x:x + w][m]
    mem = [sl(a) for a in D["mem"]]; m0 = sl(D["m0"])
    return dict(
        agree_r=np.nanmean([r(mem[i], mem[j]) for i in range(len(mem)) for j in range(i + 1, len(mem))]),
        depth_r=np.nanmean([r(m0, sl(D["sh"])), r(m0, sl(D["dp"]))]),
        ct_r=r(m0, sl(D["ct"])), ink_frac=float((m0 > 0.5).mean()),
        contrast=float(np.percentile(m0, 95) - np.percentile(m0, 50)))


def windows(D, h, w):
    H, W = D["m0"].shape; res = []
    y0, y1, x0, x1 = D.get("roi") or (0, H, 0, W)
    y1, x1 = min(y1, H), min(x1, W)
    for y in range(y0, y1 - h + 1, h // 2):
        for x in range(x0, x1 - w + 1, w // 2):
            mt = metrics(D, y, x, h, w)
            if mt:
                res.append(((y, x), mt))
    return res


spec = json.load(open(sys.argv[1])); b = spec["box"]
sets = {k: load_set(v) for k, v in spec["sets"].items()}
box = metrics(sets[b["set"]], b["y"], b["x"], b["h"], b["w"])
if box is None:
    sys.exit("box has too little CT coverage")
refs = {k: windows(D, b["h"], b["w"]) for k, D in sets.items()}
# Exclude windows overlapping the box from its own surface's reference set.
refs[b["set"]] = [(p, mt) for p, mt in refs[b["set"]]
                  if abs(p[0] - b["y"]) >= b["h"] or abs(p[1] - b["x"]) >= b["w"]]
keys = ["agree_r", "depth_r", "ct_r", "ink_frac", "contrast"]
print(f"box {b}: " + ", ".join(f"{k}={box[k]:.3f}" for k in keys))
print("reference set (n windows): median of each metric | box percentile rank within the set")
for name, lst in refs.items():
    cells = []
    for k in keys:
        vals = np.array([mt[k] for _, mt in lst if np.isfinite(mt[k])])
        if len(vals):
            cells.append(f"{k} med {np.median(vals):.3f} rank {100 * np.mean(vals < box[k]):3.0f}%")
    print(f"  {name:5s} (n={len(lst):3d}): " + " | ".join(cells))
