"""Stroke-versus-cloud metrics for ink probability maps (numbers only, no images written).

Ink-count metrics (ink fraction, connected components) cannot tell letters from soft "clouds":
broad, low-frequency blobs that an ink model can produce on unfamiliar papyrus. Letters are thin
strokes. These metrics measure shape and scale instead of amount:

  fine_coarse  threshold-free scale ratio: variance of the map's band-pass at stroke scale
               (difference of Gaussians, sigma 4 -> 16 px) over its variance at cloud scale
               (sigma 16 -> 64 px). Strokes raise it; clouds lower it.
  width_med    median local stroke width (px): 2 x distance-to-edge, sampled on the skeleton of
               the thresholded map. Text strokes at 9 um are tens of px; clouds are hundreds.
  thin_frac    fraction of thresholded ink area in components whose median width <= THIN px.
  skel_per_k   skeleton length per 1000 px of thresholded ink (higher = thinner structures).
Each width metric is computed at a fixed threshold (p' > 0.5, suffix _t50) and at an adaptive
threshold (top 15% of valid pixels, suffix _p85), so maps with little confident ink are still
measured. Smoothing uses normalised convolution inside the valid mask, and the mask is eroded so
the bright rim of a rendered patch does not count as ink.

Usage: python stroke_metrics.py MANIFEST.tsv OUT.csv
  MANIFEST.tsv lines: name<TAB>label<TAB>surface.zarr<TAB>ink.tif<TAB>window_px (0 = whole map;
  otherwise half-overlapping window_px x window_px windows)
"""
import csv, sys
import numpy as np
import zarr
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

Image.MAX_IMAGE_PIXELS = None
THIN, ERODE, MIN_AREA = 45, 40, 100


def load_valid(path, shape):
    g = zarr.open(path, mode="r")
    v = np.asarray(g if hasattr(g, "shape") else g["0"])
    v = np.moveaxis(v, int(np.argmin(v.shape)), 0)
    m = v.max(0) > 0
    if m.shape != shape:
        m = ndi.zoom(m.astype(np.uint8), (shape[0] / m.shape[0], shape[1] / m.shape[1]), order=0) > 0
    return ndi.binary_erosion(m, iterations=ERODE)


def load_q(path):
    p = np.asarray(Image.open(path)).astype(np.float32)
    if p.max() > 1.5:
        p /= 255.0
    return np.clip((p - 0.25) / 0.5, 0, 1)


def nsmooth(x, valid, s):
    w = ndi.gaussian_filter(valid.astype(np.float32), s)
    return ndi.gaussian_filter(x * valid, s) / np.maximum(w, 1e-3)


def widths(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return np.nan, np.nan, np.nan
    areas = np.bincount(lab.ravel())
    keep = areas >= MIN_AREA; keep[0] = False
    mask = keep[lab]
    if not mask.any():
        return np.nan, np.nan, np.nan
    edt = ndi.distance_transform_edt(mask)
    sk = skeletonize(mask)
    w = 2 * edt[sk]
    lab_sk = lab[sk]
    total = mask.sum()
    thin_area = 0
    for c in np.unique(lab_sk):
        if np.median(w[lab_sk == c]) <= THIN:
            thin_area += areas[c]
    return float(np.median(w)), thin_area / total, 1000.0 * sk.sum() / total


def region_metrics(q, valid):
    out = {"valid_px": int(valid.sum())}
    if valid.sum() < 20000:
        return None
    f = nsmooth(q, valid, 4) - nsmooth(q, valid, 16)
    c = nsmooth(q, valid, 16) - nsmooth(q, valid, 64)
    out["fine_coarse"] = float(np.mean(f[valid] ** 2) / max(np.mean(c[valid] ** 2), 1e-9))
    out["ink_frac"] = float(((q > 0.5) & valid).sum() / valid.sum())
    for tag, thr in (("t50", 0.5), ("p85", float(np.percentile(q[valid], 85)))):
        wm, tf, sp = widths((q > thr) & valid)
        out[f"width_med_{tag}"], out[f"thin_frac_{tag}"], out[f"skel_per_k_{tag}"] = wm, tf, sp
    return out


def main():
    man, outp = sys.argv[1], sys.argv[2]
    cols = ["name", "label", "valid_px", "ink_frac", "fine_coarse", "width_med_t50", "thin_frac_t50",
            "skel_per_k_t50", "width_med_p85", "thin_frac_p85", "skel_per_k_p85"]
    with open(outp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for line in open(man):
            if not line.strip() or line.startswith("#"):
                continue
            name, label, vol, ink, win = line.rstrip("\n").split("\t")
            q = load_q(ink); valid = load_valid(vol, q.shape); win = int(win)
            regions = [(name, slice(None), slice(None))] if win == 0 else [
                (f"{name}@{y}_{x}", slice(y, y + win), slice(x, x + win))
                for y in range(0, q.shape[0] - win + 1, win // 2) for x in range(0, q.shape[1] - win + 1, win // 2)]
            for rn, sy, sx in regions:
                m = region_metrics(q[sy, sx], valid[sy, sx])
                if m:
                    w.writerow({"name": rn, "label": label, **{k: (f"{v:.4g}" if isinstance(v, float) else v)
                                                                for k, v in m.items()}})
            fh.flush()
            print("done", name, flush=True)


if __name__ == "__main__":
    main()
