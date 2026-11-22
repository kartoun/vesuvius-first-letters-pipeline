"""Region-level ink sanity metrics (numbers only), for calibrated comparison of a candidate area
with known text and blank papyrus processed the same way.

For each region (connected CT-covered "blocks" of a surface, or fixed windows with --windows N):
  agree_r   mean pairwise Pearson r between the ensemble members' maps
  depth_r   mean Pearson r between the default-window ensemble map and the maps from layer windows
            shifted 3 layers shallower and deeper (real ink should survive a small shift)
  ct_r      Pearson r between the ensemble map and the CT surface intensity (high = the model is
            copying CT structure such as fibres or cracks rather than ink)
  ink_frac  fraction of pixels with p' > 0.5 in the ensemble mean
  line      text-line periodicity score (screen_metrics.line_score) and its lag in px
Usage: python block_metrics.py NAME RENDER.zarr MEAN.tif SHALLOW.tif DEEP.tif MEMBER.tif [MEMBER.tif ...]
       [--windows N] [--min-area PX]
"""
import argparse, os, sys
import numpy as np
import zarr
from PIL import Image
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_metrics import line_score

Image.MAX_IMAGE_PIXELS = None
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("render"); ap.add_argument("mean"); ap.add_argument("shallow")
ap.add_argument("deep"); ap.add_argument("members", nargs="+")
ap.add_argument("--windows", type=int, default=0); ap.add_argument("--min-area", type=int, default=400000)
a = ap.parse_args()


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


m0 = q_of(a.mean); S = m0.shape
sh, dp = q_of(a.shallow, S), q_of(a.deep, S)
mem = [q_of(p, S) for p in a.members]
g = zarr.open(a.render, mode="r"); v = np.asarray(g if hasattr(g, "shape") else g["0"])
v = np.moveaxis(v, int(np.argmin(v.shape)), 0); c = v.shape[0] // 2
ct = v[c - 2:c + 3].astype(np.float32).mean(0)
if ct.shape != S:
    ct = ndi.zoom(ct, (S[0] / ct.shape[0], S[1] / ct.shape[1]), order=1)
valid = ndi.binary_erosion(ct > 0, iterations=40)

regions = []
if a.windows:
    w = a.windows
    for y in range(0, S[0] - w + 1, w // 2):
        for x in range(0, S[1] - w + 1, w // 2):
            m = np.zeros(S, bool); m[y:y + w, x:x + w] = True; m &= valid
            if m.sum() > 0.5 * w * w:
                regions.append((f"{a.name}@{y}_{x}", m))
else:
    lab, n = ndi.label(valid)
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= a.min_area:
            ys, xs = np.nonzero(m)
            regions.append((f"{a.name}#block{k}(rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()}, "
                            f"centre {int(ys.mean())},{int(xs.mean())})", m))
print("region,px,agree_r,depth_r,ct_r,ink_frac,line,line_lag")
for nm, m in regions:
    ag = np.nanmean([r(mem[i][m], mem[j][m]) for i in range(len(mem)) for j in range(i + 1, len(mem))])
    dr = np.nanmean([r(m0[m], sh[m]), r(m0[m], dp[m])])
    cr = r(m0[m], ct[m])
    ys, xs = np.nonzero(m); sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    ls, lag, _ = line_score(m0[sl], m[sl])
    print(f"{nm},{int(m.sum())},{ag:.3f},{dr:.3f},{cr:.3f},{(m0[m] > 0.5).mean():.3f},{ls:.3f},{lag}")
