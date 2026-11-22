"""Which papyrus face of a tifxyz surface faces the scroll centre (where the text is)?
Prints "fwd" or "rev" for vesuvius ink inference --direction, plus the share of grid points whose
normal points toward the centre. Convention measured on the PHerc0211 known-text segment: forward
reads the ink when the grid normal cross(d/drow, d/dcol) points toward the scroll centre.
Usage: python orient_surface.py SURFACE_DIR VOLUME_S3_PATH
"""
import os, sys
import numpy as np
import zarr
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_consistency import load

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
grp = zarr.open_group(S3 + sys.argv[2], mode="r"); LV, F = 4, 16; cache = {}


def centre(z):
    k = int(z // F)
    if k not in cache:
        sl = ndi.binary_fill_holes(ndi.binary_closing(np.asarray(grp[str(LV)][k]) > 0, iterations=3))
        ys, xs = np.nonzero(sl); cache[k] = np.array([xs.mean() * F, ys.mean() * F])
    return cache[k]


_, X, v = load(sys.argv[1]); H, W = v.shape; s = []
for j in range(2, H - 2, 3):
    for i in range(2, W - 2, 3):
        if not (v[j, i] and v[j + 1, i] and v[j - 1, i] and v[j, i + 1] and v[j, i - 1]):
            continue
        n = np.cross(X[j + 1, i] - X[j - 1, i], X[j, i + 1] - X[j, i - 1]); n /= np.linalg.norm(n) + 1e-9
        c = centre(X[j, i, 2]); t = np.array([c[0] - X[j, i, 0], c[1] - X[j, i, 1], 0.0])
        s.append(np.dot(n, t / (np.linalg.norm(t) + 1e-9)))
s = np.array(s)
print(("fwd" if np.median(s) >= 0 else "rev") + f" {np.mean(s > 0):.2f} {np.median(s):+.2f}")
