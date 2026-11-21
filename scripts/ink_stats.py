"""Aggregate statistics of ink-prediction TIFFs (no images are shown or written).

ink_9um outputs are label-smoothed: "no ink" sits near 0.25, so probabilities are rescaled
with p' = clip((p - 0.25) / 0.5, 0, 1) (scrollprize/ink_9um model card). Pixels are counted
only where the paired surface volume has CT data (centre layer nonzero), if given.

Usage: python ink_stats.py PRED.tif [--surface RENDER.zarr] [--compare OTHER.tif]
"""
import argparse
import numpy as np
import tifffile
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("--surface"); ap.add_argument("--compare")
a = ap.parse_args()


def load_prob(path):
    try:
        p = tifffile.imread(path).astype(np.float32)
    except ValueError:                      # LZW etc. without imagecodecs: Pillow decodes it
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        p = np.asarray(Image.open(path)).astype(np.float32)
    if p.ndim == 3:
        p = p[0] if p.shape[0] < p.shape[-1] else p[..., 0]
    if p.max() > 1.5:
        p /= 255.0 if p.max() <= 255 else 65535.0
    return p


p = load_prob(a.pred)
valid = np.ones_like(p, bool)
if a.surface:
    import zarr
    g = zarr.open(a.surface, mode="r")
    v = g["0"] if not hasattr(g, "shape") else g
    v = np.asarray(v); ax = int(np.argmin(v.shape)); c = np.take(v, v.shape[ax] // 2, axis=ax)
    if c.shape != p.shape:
        zy, zx = p.shape[0] / c.shape[0], p.shape[1] / c.shape[1]
        c = ndi.zoom(c, (zy, zx), order=0)
    valid = c > 0
q = np.clip((p - 0.25) / 0.5, 0, 1)
qv = q[valid]
print(f"{a.pred}")
print(f"  shape={p.shape} raw p range=({p.min():.3f},{p.max():.3f})  CT-covered px={valid.sum()} ({valid.mean():.2f})")
print(f"  rescaled p': mean={qv.mean():.4f}  p50={np.percentile(qv,50):.3f}  p99={np.percentile(qv,99):.3f}  "
      f"p99.9={np.percentile(qv,99.9):.3f}")
for t in (0.3, 0.5, 0.7):
    m = (q > t) & valid
    lab, n = ndi.label(m)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    print(f"  p'>{t}: {m.sum()/max(valid.sum(),1)*100:.3f}% of covered px; components={n}; "
          f"largest={sizes.max()} px; components>=200px={int((sizes>=200).sum())}")
if a.compare:
    o = load_prob(a.compare)
    if o.shape != p.shape:
        o = ndi.zoom(o, (p.shape[0] / o.shape[0], p.shape[1] / o.shape[1]), order=1)
    oq = np.clip((o - 0.25) / 0.5, 0, 1)
    r = np.corrcoef(q[valid], oq[valid])[0, 1]
    print(f"  vs {a.compare.split('/')[-1][:60]}...: pearson r={r:.3f} (on covered px), "
          f"other mean p'={oq[valid].mean():.4f}")
