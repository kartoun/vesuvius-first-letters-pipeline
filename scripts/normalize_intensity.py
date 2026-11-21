"""Map a rendered surface volume's intensities onto a reference scroll's distribution.

Tests whether an ink model's broad response on a new scroll is partly an intensity-domain
shift. Two modes:
  hist    quantile (histogram) matching of CT-covered voxels to the reference distribution
  zscore  linear: match mean and standard deviation of CT-covered voxels to the reference
Statistics come from the centre CENTRE layers (what the ink model reads); the mapping is
applied to all layers. Voxels with value 0 (no CT) stay 0.

Usage: python normalize_intensity.py REF.zarr IN.zarr OUT.zarr [--mode hist|zscore] [--centre 17]
Prints image-free summary numbers only.
"""
import argparse
import numpy as np
import zarr


def load(path):
    g = zarr.open(path, mode="r")
    v = np.asarray(g if hasattr(g, "shape") else g["0"])
    return np.moveaxis(v, int(np.argmin(v.shape)), 0)


def centre(v, n):
    c = v.shape[0] // 2
    return v[max(c - n // 2, 0): c - n // 2 + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--mode", choices=("hist", "zscore"), default="hist")
    ap.add_argument("--centre", type=int, default=17)
    a = ap.parse_args()

    r = centre(load(a.ref), a.centre); r = r[r > 0]
    v = load(a.inp); s = centre(v, a.centre); s = s[s > 0]
    if a.mode == "hist":
        hs = np.bincount(s.ravel(), minlength=256).astype(np.float64)
        hr = np.bincount(r.ravel(), minlength=256).astype(np.float64)
        hs[0] = hr[0] = 0
        cs, cr = np.cumsum(hs) / hs.sum(), np.cumsum(hr) / hr.sum()
        lut = np.interp(cs, cr, np.arange(256))
    else:
        lut = (np.arange(256) - s.mean()) / (s.std() + 1e-6) * r.std() + r.mean()
    lut = np.clip(np.rint(lut), 1, 255).astype(np.uint8)
    lut[0] = 0
    out = lut[v]
    z = zarr.open_array(a.out, mode="w", shape=out.shape, chunks=(out.shape[0], 128, 128), dtype="u1")
    z[:] = out
    o = centre(out, a.centre); o = o[o > 0]
    print(f"{a.mode}: in mean/sd {s.mean():.1f}/{s.std():.1f} -> out {o.mean():.1f}/{o.std():.1f}; "
          f"ref {r.mean():.1f}/{r.std():.1f}")


if __name__ == "__main__":
    main()
