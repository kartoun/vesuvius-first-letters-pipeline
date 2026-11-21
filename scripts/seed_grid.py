"""Grid of GrowPatch seeds across one slice: every SPACING px inside the papyrus mask, snapped
to the nearest predicted-surface voxel (prediction > 0) within SNAP px. Prints "x y z" lines.

Usage: python seed_grid.py CT_ZARR PRED_ZARR Z [--spacing 300] [--snap 12] [--margin 150]
"""
import argparse, json, os
import numpy as np
import numcodecs
from scipy import ndimage as ndi


def read_slice(zarr_root, z):
    meta = json.load(open(os.path.join(zarr_root, "0", ".zarray")))
    Z, Y, X = meta["shape"]; cz, cy, cx = meta["chunks"]
    comp = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
    zc, zo = divmod(z, cz)
    out = np.zeros((Y, X), np.uint8)
    for j in range(-(-Y // cy)):
        for i in range(-(-X // cx)):
            p = os.path.join(zarr_root, "0", str(zc), str(j), str(i))
            if not os.path.exists(p):
                continue
            b = open(p, "rb").read()
            a = np.frombuffer(comp.decode(b) if comp else b, np.uint8).reshape(cz, cy, cx)[zo]
            out[j*cy:(j+1)*cy, i*cx:(i+1)*cx] = a[:min(cy, Y-j*cy), :min(cx, X-i*cx)]
    return out


ap = argparse.ArgumentParser()
ap.add_argument("ct"); ap.add_argument("pred"); ap.add_argument("z", type=int)
ap.add_argument("--spacing", type=int, default=300); ap.add_argument("--snap", type=int, default=12)
ap.add_argument("--margin", type=int, default=150)
a = ap.parse_args()
ct = read_slice(a.ct, a.z) > 0
ct = ndi.binary_erosion(ndi.binary_fill_holes(ct), iterations=a.margin)   # stay off the outer edge
pred = read_slice(a.pred, a.z) > 0
dist, (iy, ix) = ndi.distance_transform_edt(~pred, return_indices=True)
n = 0
for y in range(a.spacing // 2, ct.shape[0], a.spacing):
    for x in range(a.spacing // 2, ct.shape[1], a.spacing):
        if not ct[y, x] or dist[y, x] > a.snap:
            continue
        print(f"{ix[y, x]} {iy[y, x]} {a.z}"); n += 1
import sys; print(f"# {n} seeds", file=sys.stderr)
