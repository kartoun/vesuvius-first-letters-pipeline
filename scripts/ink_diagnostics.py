"""Image-free diagnostics comparing surfaces and their ink predictions.

For each (surface volume, prediction) pair prints:
  A. input statistics on the layers the model reads (centre 17): mean, std, percentiles of CT
     intensity, and a high-pass texture energy (std of image minus 5-px Gaussian blur);
  B. shape statistics of predicted-ink components (p' > 0.5, area >= 50 px): median area,
     median stroke width (2 x median distance-transform ridge), median elongation, and
     the fraction of predicted pixels inside components larger than 20,000 px ("blobs");
  C. row structure: the strongest peak of the vertical autocorrelation of the p' map between
     200 and 900 px lags (text lines would give a periodic peak; blobs give none).

Usage: python ink_diagnostics.py SURFACE.zarr PRED.tif [SURFACE2.zarr PRED2.tif ...]
"""
import sys
import numpy as np
import zarr
from scipy import ndimage as ndi
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_vol(path):
    g = zarr.open(path, mode="r")
    v = np.asarray(g if hasattr(g, "shape") else g["0"])
    return np.moveaxis(v, int(np.argmin(v.shape)), 0)


def load_pred(path):
    p = np.asarray(Image.open(path)).astype(np.float32)
    if p.max() > 1.5:
        p /= 255.0
    return np.clip((p - 0.25) / 0.5, 0, 1)


args = sys.argv[1:]
for sv, pr in zip(args[0::2], args[1::2]):
    v = load_vol(sv); L = v.shape[0]; c = L // 2
    sel = v[max(0, c - 8):c + 9].astype(np.float32)
    valid = v[c] > 0
    x = sel[:, valid]
    mid = v[c].astype(np.float32)
    hp = mid - ndi.gaussian_filter(mid, 5)
    q = load_pred(pr)
    if q.shape != valid.shape:
        q = ndi.zoom(q, (valid.shape[0] / q.shape[0], valid.shape[1] / q.shape[1]), order=1)
    m = (q > 0.5) & valid
    lab, n = ndi.label(m)
    areas = np.bincount(lab.ravel())[1:]
    keep = np.nonzero(areas >= 50)[0] + 1
    widths, elong = [], []
    dt = ndi.distance_transform_edt(m)
    objs = ndi.find_objects(lab)
    for k in keep[:2000]:
        sl = objs[k - 1]; comp = lab[sl] == k
        widths.append(2 * np.median(dt[sl][comp & (dt[sl] >= np.percentile(dt[sl][comp], 75))]))
        yy, xx = np.nonzero(comp)
        if len(yy) > 2:
            ev = np.linalg.eigvalsh(np.cov(np.stack([yy, xx])))
            elong.append(np.sqrt(ev[1] / max(ev[0], 1e-6)))
    blob = areas[areas > 20000].sum() / max(m.sum(), 1)
    # vertical autocorrelation of p' (rows of text -> periodic peak)
    qq = np.where(valid, q - q[valid].mean(), 0)
    prof = qq.mean(1)
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = 200, min(900, len(ac) - 1)
    lag = int(np.argmax(ac[lo:hi]) + lo) if hi > lo else -1
    print(sv.rstrip("/").split("/")[-1][:48])
    print(f"  A. CT (centre 17 layers): mean={x.mean():.1f} std={x.std():.1f} p5={np.percentile(x,5):.0f} "
          f"p50={np.percentile(x,50):.0f} p95={np.percentile(x,95):.0f}  texture(high-pass std)={hp[valid].std():.2f}")
    print(f"  B. ink comps>=50px: n={len(keep)} median area={np.median(areas[keep-1]) if len(keep) else 0:.0f}px "
          f"median width={np.median(widths) if widths else 0:.1f}px median elongation={np.median(elong) if elong else 0:.2f} "
          f"blob fraction={blob:.2f}")
    print(f"  C. vertical autocorr: best peak lag {lag}px value {ac[lag]:.3f}" if lag > 0 else "  C. n/a")
