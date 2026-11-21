"""Numeric (image-free) quality checks for a rendered surface volume (vc_render_tifxyz zarr).

Prints only aggregate numbers, so renders of eligible scrolls never need to be viewed or
shared to judge whether a mesh sits on a papyrus sheet:
  1. coverage: fraction of render pixels with CT data (nonzero) in the centre layer
  2. depth profile: mean intensity per layer; a mesh on a sheet gives a peak near the
     centre layer and lower values on both sides (air/neighbouring gaps)
  3. fibre orientation: structure-tensor orientation histogram of the centre layer,
     reporting energy near 0 and 90 degrees (papyrus horizontal/vertical fibres) vs. a
     uniform baseline, and the mean coherence

Usage: python surface_quality.py RENDER.zarr [RENDER2.zarr ...]
"""
import sys
import numpy as np
import zarr
from scipy import ndimage as ndi


def load(path):
    g = zarr.open(path, mode="r")
    if hasattr(g, "shape"):
        return np.asarray(g)
    for k in ("0", 0):
        try:
            return np.asarray(g[k])
        except Exception:
            pass
    keys = list(g.array_keys()) if hasattr(g, "array_keys") else list(g.keys())
    return np.asarray(g[sorted(keys)[0]])


def orientation_stats(img, valid):
    f = img.astype(np.float32)
    gx, gy = ndi.sobel(f, 1), ndi.sobel(f, 0)
    s = 3.0
    jxx, jyy, jxy = (ndi.gaussian_filter(a, s) for a in (gx * gx, gy * gy, gx * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)          # gradient direction
    lam_diff = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = lam_diff / (jxx + jyy + 1e-6)
    m = valid & (jxx + jyy > np.percentile(jxx + jyy, 20))
    ang = np.degrees(np.mod(theta[m], np.pi))              # 0..180
    w = coh[m]
    hist, _ = np.histogram(ang, bins=36, range=(0, 180), weights=w)
    hist = hist / hist.sum()
    near = lambda c: hist[[int(((c + d) % 180) // 5) for d in (-5, 0, 5)]].sum()
    axis_energy = near(0) + near(90)
    return axis_energy, 6 / 36, float(np.median(coh[m]))


for path in sys.argv[1:]:
    v = load(path)
    if v.ndim != 3:
        print(path, "unexpected shape", v.shape); continue
    # Layer axis is the smallest dimension.
    ax = int(np.argmin(v.shape)); v = np.moveaxis(v, ax, 0)
    L = v.shape[0]; c = L // 2
    valid = v[c] > 0
    cov = valid.mean()
    prof = [float(v[i][valid].mean()) if valid.any() else 0.0 for i in range(L)]
    peak = int(np.argmax(prof))
    contrast = (max(prof) - 0.5 * (prof[0] + prof[-1])) / (max(prof) + 1e-6)
    ae, base, coh = orientation_stats(v[c], valid)
    print(f"{path}")
    print(f"  shape (layers,H,W)={v.shape}  coverage(centre layer nonzero)={cov:.3f}")
    print(f"  depth profile (mean per layer): " + " ".join(f"{p:.0f}" for p in prof))
    print(f"  peak layer={peak} of 0..{L-1} (centre {c})  peak-vs-ends contrast={contrast:.3f}")
    print(f"  fibre-axis energy (0/90 deg +-7.5)={ae:.3f} vs uniform {base:.3f}  median coherence={coh:.3f}")
    # 4. Per-tile depth profiles: on a sheet-following mesh, local brightness peaks cluster
    #    at the centre layer; on a wandering mesh they spread over all layers.
    T = 100; peaks, contr = [], []
    H, W = v.shape[1:]
    for y in range(0, H - T + 1, T):
        for x in range(0, W - T + 1, T):
            ok = valid[y:y + T, x:x + T]
            if ok.mean() < 0.9:
                continue
            p = v[:, y:y + T, x:x + T].reshape(L, -1).mean(1)
            p = ndi.gaussian_filter1d(p, 1.0)
            peaks.append(int(np.argmax(p)))
            contr.append(float((p.max() - np.median(p)) / (p.max() + 1e-6)))
    peaks, contr = np.array(peaks), np.array(contr)
    if len(peaks):
        h = np.bincount(peaks, minlength=L)
        near = np.mean(np.abs(peaks - c) <= 3)
        print(f"  tiles={len(peaks)} ({T}x{T} px, >=90% on CT); peak within +-3 of centre: {near:.2f} "
              f"(uniform would be {7/L:.2f}); median tile contrast={np.median(contr):.3f}")
        print(f"  tile peak-layer histogram: " + " ".join(str(int(x)) for x in h))
