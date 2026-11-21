"""Image-free per-patch screening metrics for First Letters candidates.

For a rendered surface volume and its ink prediction, compute numbers only:
  surf_centre  fraction of 100-px tiles whose depth-profile peak is within +-3 of centre
  surf_contr   median tile depth-profile contrast
  ink_frac     fraction of CT-covered pixels with p' > 0.5   (p' = (p - 0.25) / 0.5)
  comps200     connected p'>0.5 components of >= 200 px per million covered px
  blob_frac    fraction of predicted pixels in components > 20,000 px
  line_score   text-line periodicity of the p' map: for each image axis, the mean
               profile across the other axis is autocorrelated; the score is the
               highest peak over lags 250-700 px (~2.3-6.6 mm at 9.36 um, the range of
               Greek line spacings) minus the median autocorrelation over those lags.
               Edge lags are excluded so a window boundary cannot create a peak.
  line_lag / line_axis   where that peak is

Windows mode (--windows N) instead scores non-overlapping N x N sub-windows of a single
large render, to calibrate the scores on known-text and blank segments at patch size.

Usage:
  python screen_metrics.py SURFACE.zarr PRED.tif [--name NAME]
  python screen_metrics.py SURFACE.zarr PRED.tif --windows 1040
"""
import argparse
import numpy as np
import zarr
from scipy import ndimage as ndi
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
LAG_LO, LAG_HI = 250, 700


def load_vol(path):
    g = zarr.open(path, mode="r")
    v = np.asarray(g if hasattr(g, "shape") else g["0"])
    return np.moveaxis(v, int(np.argmin(v.shape)), 0)


def load_pred(path, shape):
    p = np.asarray(Image.open(path)).astype(np.float32)
    if p.max() > 1.5:
        p /= 255.0
    q = np.clip((p - 0.25) / 0.5, 0, 1)
    if q.shape != shape:
        q = ndi.zoom(q, (shape[0] / q.shape[0], shape[1] / q.shape[1]), order=1)
    return q


def line_score(q, valid):
    best = (-1.0, -1, -1)
    if valid.sum() < 1000:
        return best
    qq = np.where(valid, q - q[valid].mean(), 0.0)
    for axis in (0, 1):
        n = valid.sum(axis=1 - axis).astype(float)
        prof = np.where(n > 0.5 * n.max(), qq.sum(axis=1 - axis) / np.maximum(n, 1), 0.0)
        hi = min(LAG_HI, len(prof) // 2)
        if hi <= LAG_LO + 10:
            continue
        ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        seg = ac[LAG_LO:hi]
        k = int(np.argmax(seg))
        s = float(seg[k] - np.median(seg))
        if s > best[0]:
            best = (s, LAG_LO + k, axis)
    return best


def surf_metrics(v, valid):
    L = v.shape[0]; c = L // 2; T = 100; peaks, contr = [], []
    for y in range(0, v.shape[1] - T + 1, T):
        for x in range(0, v.shape[2] - T + 1, T):
            if valid[y:y + T, x:x + T].mean() < 0.9:
                continue
            p = ndi.gaussian_filter1d(v[:, y:y + T, x:x + T].reshape(L, -1).mean(1).astype(float), 1.0)
            peaks.append(int(np.argmax(p))); contr.append((p.max() - np.median(p)) / (p.max() + 1e-6))
    if not peaks:
        return float("nan"), float("nan"), 0
    return float(np.mean(np.abs(np.array(peaks) - c) <= 3)), float(np.median(contr)), len(peaks)


def ink_metrics(q, valid):
    m = (q > 0.5) & valid
    cov = max(int(valid.sum()), 1)
    lab, n = ndi.label(m)
    areas = np.bincount(lab.ravel())[1:] if n else np.array([0])
    return (m.sum() / cov, (areas >= 200).sum() / (cov / 1e6),
            areas[areas > 20000].sum() / max(m.sum(), 1))


def score(v, q, valid):
    sc, st, nt = surf_metrics(v, valid)
    f, c2, bf = ink_metrics(q, valid)
    ls, lag, ax = line_score(q, valid)
    return dict(cov=float(valid.mean()), tiles=nt, surf_centre=sc, surf_contr=st, ink_frac=float(f),
                comps200=float(c2), blob_frac=float(bf), line_score=ls, line_lag=lag, line_axis=ax)


def fmt(name, d):
    return (f"{name},{d['cov']:.3f},{d['tiles']},{d['surf_centre']:.3f},{d['surf_contr']:.3f},"
            f"{d['ink_frac']:.4f},{d['comps200']:.1f},{d['blob_frac']:.2f},{d['line_score']:.3f},"
            f"{d['line_lag']},{d['line_axis']}")


HEADER = "name,cov,tiles,surf_centre,surf_contr,ink_frac,comps200_per_Mpx,blob_frac,line_score,line_lag,line_axis"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("surface"); ap.add_argument("pred"); ap.add_argument("--name", default=None)
    ap.add_argument("--windows", type=int, default=0); ap.add_argument("--header", action="store_true")
    a = ap.parse_args()
    v = load_vol(a.surface); valid = v[v.shape[0] // 2] > 0
    q = load_pred(a.pred, valid.shape)
    if a.header:
        print(HEADER)
    name = a.name or a.surface.rstrip("/").split("/")[-1]
    if a.windows:
        W = a.windows
        for y in range(0, valid.shape[0] - W + 1, W // 2):
            for x in range(0, valid.shape[1] - W + 1, W // 2):
                vv = valid[y:y + W, x:x + W]
                if vv.mean() < 0.5:
                    continue
                print(fmt(f"{name}@{y},{x}", score(v[:, y:y + W, x:x + W], q[y:y + W, x:x + W], vv)))
    else:
        print(fmt(name, score(v, q, valid)))
