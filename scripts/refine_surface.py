"""Per-pixel layer alignment of a rendered surface volume ("best-layer selection").

A grown or fitted surface can sit a few voxels off the papyrus sheet, and the offset drifts
across the patch. Ink models are trained on renders centred on the sheet, so a drifting offset
blurs or hides ink. This script estimates the sheet position per pixel from the render itself
and resamples a new, sheet-centred render.

Steps:
  1. Smooth the render in-plane (Gaussian, sigma XY px) and along depth (sigma 1 layer).
  2. Per pixel, find the brightest layer within +-SEARCH layers of the current centre.
     Restricting the search keeps the neighbouring sheet from capturing the peak.
  3. Smooth the resulting offset map (median, then Gaussian) so it follows the sheet, not noise.
  4. Resample OUT_LAYERS layers centred on (centre + offset) with linear interpolation.

Pixels with no CT (all zero along depth) stay zero. Output is a plain uint8 zarr array
(layers, rows, cols) that the ink_detection inference reads directly.

Usage: python refine_surface.py IN.zarr OUT.zarr [--out-layers 28] [--search 8] [--xy 12]
Prints image-free summary numbers only (offset statistics).
"""
import argparse
import numpy as np
import zarr
from scipy import ndimage as ndi


def load(path):
    g = zarr.open(path, mode="r")
    v = np.asarray(g if hasattr(g, "shape") else g["0"])
    return np.moveaxis(v, int(np.argmin(v.shape)), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--out-layers", type=int, default=28)
    ap.add_argument("--search", type=int, default=8)
    ap.add_argument("--xy", type=float, default=12.0, help="in-plane smoothing sigma (px) for peak finding")
    ap.add_argument("--map-sigma", type=float, default=25.0, help="smoothing sigma (px) of the offset map")
    a = ap.parse_args()

    v = load(a.inp).astype(np.float32)
    L, H, W = v.shape
    c = L // 2
    valid = v.max(0) > 0
    s = ndi.gaussian_filter(v, (1.0, a.xy, a.xy))
    lo, hi = max(c - a.search, 0), min(c + a.search + 1, L)
    off = (np.argmax(s[lo:hi], axis=0) + lo - c).astype(np.float32)
    off[~valid] = 0
    # Fill invalid pixels before smoothing so CT gaps do not drag the map toward zero.
    if valid.any() and (~valid).any():
        idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        off = off[tuple(idx)]
    off = ndi.gaussian_filter(ndi.median_filter(off, size=31), a.map_sigma)

    half = a.out_layers // 2
    zz = c + off[None] + (np.arange(a.out_layers) - half)[:, None, None]
    z0 = np.clip(np.floor(zz).astype(np.int32), 0, L - 1)
    z1 = np.clip(z0 + 1, 0, L - 1)
    w = (zz - np.floor(zz)).astype(np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = (1 - w) * v[z0, yy[None], xx[None]] + w * v[z1, yy[None], xx[None]]
    out[:, ~valid] = 0
    # Layers that fall outside the source render are not real data.
    edge = (zz < 0) | (zz > L - 1)
    out[edge] = 0
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)

    z = zarr.open_array(a.out, mode="w", shape=out.shape, chunks=(out.shape[0], 128, 128), dtype="u1")
    z[:] = out
    ov = off[valid]
    print(f"refined {a.inp} -> {a.out}: layers {L}->{a.out_layers}, offset median {np.median(ov):+.2f}, "
          f"p5 {np.percentile(ov, 5):+.2f}, p95 {np.percentile(ov, 95):+.2f}, "
          f"|off|>3 {np.mean(np.abs(ov) > 3):.2f}, clipped-layer frac {edge[:, valid].mean():.3f}")


if __name__ == "__main__":
    main()
