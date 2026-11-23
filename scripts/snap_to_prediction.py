"""Snap a tifxyz surface onto the midline of the m7 surface-prediction sheet mask.

For every valid grid vertex, the binary prediction (0/255) is sampled along the vertex normal over
+-SEARCH vx (trilinear, 0.5 vx steps). The sheet run nearest the vertex is located; its midline is the
midpoint of the two 50% crossings (sub-voxel). The per-vertex offset field is cleaned (median 3x3, gaps
filled from neighbours, Gaussian sigma SMOOTH grid cells) and the vertex moved along its normal.
--stats only reports the offset distribution (no output surface).
Usage: python snap_to_prediction.py PRED_LEVEL0.zarr SRC_TIFXYZ [DST_TIFXYZ] [--search 12] [--smooth 1.0] [--stats]
"""
import argparse, glob, os, shutil, numpy as np, zarr
from PIL import Image
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("src"); ap.add_argument("dst", nargs="?")
ap.add_argument("--search", type=float, default=12); ap.add_argument("--smooth", type=float, default=1.0)
ap.add_argument("--stats", action="store_true")
a = ap.parse_args()
sd = a.src if os.path.exists(os.path.join(a.src, "x.tif")) else os.path.dirname(glob.glob(os.path.join(a.src, "**", "x.tif"), recursive=True)[0])
P = np.stack([np.asarray(Image.open(f"{sd}/{c}.tif")).astype(np.float64) for c in "xyz"], -1)
v = (P[..., 2] > 0) & np.isfinite(P).all(-1)
dr, dc = np.gradient(P, axis=(0, 1)); n = np.cross(dr, dc); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
v &= np.isfinite(n).all(-1)
Z = zarr.open(a.pred, mode="r")
t = np.arange(-a.search, a.search + 1e-6, 0.5); S = int(np.ceil(a.search)) + 2
idx = np.argwhere(v); off = np.full(v.shape, np.nan)
zs = P[v][:, 2]
for z0 in range(int(zs.min()) // 64 * 64, int(zs.max()) + 64, 64):
    sel = (zs >= z0) & (zs < z0 + 64)
    if not sel.any(): continue
    ij = idx[sel]; X = P[ij[:, 0], ij[:, 1]]; N = n[ij[:, 0], ij[:, 1]]
    lo = np.floor(X.min(0) - S).astype(int); hi = np.ceil(X.max(0) + S).astype(int)
    lo = np.maximum(lo, 0)
    blk = np.asarray(Z[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]).astype(np.float32) / 255.0
    pts = X[:, None, :] + t[None, :, None] * N[:, None, :] - lo                        # (n, T, xyz) in block coords
    prof = ndi.map_coordinates(blk, [pts[..., 2].ravel(), pts[..., 1].ravel(), pts[..., 0].ravel()], order=1).reshape(len(X), len(t))
    for k, p in enumerate(prof):
        inside = p >= 0.5
        if not inside.any(): continue
        lab, nr = ndi.label(inside); best = None
        for r in range(1, nr + 1):
            w = np.nonzero(lab == r)[0]; a0, a1 = w[0], w[-1]
            if a0 == 0 or a1 == len(t) - 1: continue                             # run cut by the search window
            e0 = t[a0 - 1] + (0.5 - p[a0 - 1]) / (p[a0] - p[a0 - 1] + 1e-9) * 0.5     # 50% crossings
            e1 = t[a1] + (p[a1] - 0.5) / (p[a1] - p[a1 + 1] + 1e-9) * 0.5
            mid = 0.5 * (e0 + e1)
            if best is None or abs(mid) < abs(best): best = mid
        if best is not None: off[ij[k, 0], ij[k, 1]] = best
found = np.isfinite(off) & v
o = off[found]
print(f"vertices {v.sum()}, midline found {found.sum() / v.sum():.0%}; offset to midline: median {np.median(o):+.2f} vx, "
      f"sd {o.std():.2f}, IQR {np.percentile(o, 25):+.2f}..{np.percentile(o, 75):+.2f}, |off|>3: {(np.abs(o) > 3).mean():.0%}")
# local roughness: offset minus its smooth trend (what a constant depth shift cannot absorb)
fill = np.where(found, off, 0.0); w = ndi.gaussian_filter(found.astype(float), 3)
trend = ndi.gaussian_filter(fill, 3) / np.maximum(w, 1e-6)
print(f"  local (detrended, sigma 3 cells) sd {np.std((off - trend)[found]):.2f} vx")
if a.stats or not a.dst:
    raise SystemExit
med = ndi.generic_filter(np.where(found, off, np.nan), np.nanmedian, size=3, mode="nearest")
med = np.where(np.isfinite(med), med, 0.0); have = np.isfinite(ndi.generic_filter(np.where(found, off, np.nan), np.nanmedian, size=3, mode="nearest"))
w = ndi.gaussian_filter(have.astype(float), a.smooth); d = ndi.gaussian_filter(np.where(have, med, 0.0), a.smooth) / np.maximum(w, 1e-6)
d = np.where(w > 0.05, d, 0.0)
Q = P + np.where(v[..., None], d[..., None] * n, 0)
if os.path.exists(a.dst): shutil.rmtree(a.dst)
shutil.copytree(sd, a.dst)
for k, c in enumerate("xyz"):
    Image.fromarray(np.where(v, Q[..., k], P[..., k]).astype(np.float32)).save(f"{a.dst}/{c}.tif")
print(f"wrote {a.dst}: applied shift median {np.median(d[v]):+.2f} vx, sd {d[v].std():.2f}")
