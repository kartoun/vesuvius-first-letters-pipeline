"""How closely does a tifxyz surface follow the surface-prediction volume it was grown on?

For a random sample of valid grid points, reads a small cube of the (local) prediction zarr
around each point and reports the distance to the nearest predicted-surface voxel (value > 0),
capped at R voxels. A surface that tracks the prediction has most points within ~2 voxels.
Usage: python surface_on_prediction.py PRED.zarr SURFACE_DIR [SURFACE_DIR ...] [--n 1500] [--r 12]
"""
import argparse, glob, os
import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("pred"); ap.add_argument("surfaces", nargs="+")
ap.add_argument("--n", type=int, default=1500); ap.add_argument("--r", type=int, default=12)
a = ap.parse_args()
g = zarr.open(a.pred, mode="r"); P = g if hasattr(g, "shape") else g["0"]
rng = np.random.default_rng(0)
for sd in a.surfaces:
    p = os.path.dirname(glob.glob(os.path.join(sd, "**", "x.tif"), recursive=True)[0])
    X = np.stack([tifffile.imread(os.path.join(p, f"{k}.tif")) for k in "xyz"], -1)
    pts = X[X[..., 0] > 0]
    pts = pts[rng.choice(len(pts), min(a.n, len(pts)), replace=False)]
    r = a.r; d = []
    for x, y, z in pts:
        xi, yi, zi = int(round(x)), int(round(y)), int(round(z))
        if zi - r < 0 or yi - r < 0 or xi - r < 0:
            continue
        cube = np.asarray(P[zi - r:zi + r + 1, yi - r:yi + r + 1, xi - r:xi + r + 1]) > 0
        if not cube.any():
            d.append(np.inf); continue
        edt = ndi.distance_transform_edt(~cube)
        d.append(edt[r, r, r])
    d = np.array(d); f = np.isfinite(d)
    print(f"{os.path.basename(os.path.normpath(sd)):28s} n={len(d)} on-pred(<=2vx) {np.mean(d <= 2):.2f}  "
          f"<=5vx {np.mean(d <= 5):.2f}  none within {r}vx {np.mean(~f):.2f}  "
          f"median(finite) {np.median(d[f]) if f.any() else float('nan'):.1f}")
