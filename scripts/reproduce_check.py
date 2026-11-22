"""Does an ink mark reproduce on an independently grown surface? (numbers only)

Surface A (the reviewed one) and surface B (grown from a different seed) are tifxyz grids with
20 px of render canvas per grid cell (vc_render_tifxyz --scale 1). For canvas pixels of both
surfaces within RADIUS vx (3D) of CENTRE, pixels of B are paired with the nearest pixel of A in 3D
(kept if closer than PAIR vx). Reports the Pearson r of the two ensemble-mean ink maps over the
pairs, the share of paired pixels, and how far those pixels lie from each surface's edge, so a
mark sitting on A's edge can be checked in the interior of B.
Usage: python reproduce_check.py A_DIR A_INK.tif B_DIR B_INK.tif CX CY CZ [RADIUS] [PAIR]
"""
import glob, os, sys
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None


def grid(d):
    p = os.path.dirname(glob.glob(os.path.join(d, "**", "x.tif"), recursive=True)[0])
    X = np.stack([tifffile.imread(os.path.join(p, f"{k}.tif")) for k in "xyz"], -1).astype(np.float32)
    return X, X[..., 0] > 0


def q_of(p):
    a = np.asarray(Image.open(p)).astype(np.float32)
    if a.max() > 1.5:
        a /= 255.0
    return np.clip((a - 0.25) / 0.5, 0, 1)


def pixels_near(X, v, ink, centre, radius, step=4):
    """3D coords (bilinear on the grid), ink value and edge distance for canvas pixels near centre."""
    H, W = ink.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    gy, gx = ys / 20.0, xs / 20.0
    j0 = np.clip(np.floor(gy).astype(int), 0, v.shape[0] - 2); i0 = np.clip(np.floor(gx).astype(int), 0, v.shape[1] - 2)
    fy, fx = (gy - j0)[..., None], (gx - i0)[..., None]
    ok = v[j0, i0] & v[j0 + 1, i0] & v[j0, i0 + 1] & v[j0 + 1, i0 + 1]
    P = ((1 - fy) * (1 - fx) * X[j0, i0] + fy * (1 - fx) * X[j0 + 1, i0]
         + (1 - fy) * fx * X[j0, i0 + 1] + fy * fx * X[j0 + 1, i0 + 1])
    edge = ndi.distance_transform_edt(v)[j0, i0] * 20.0          # px to the surface edge
    near = ok & (np.linalg.norm(P - centre, axis=-1) < radius)
    return P[near], ink[ys[near], xs[near]], edge[near]


A_dir, A_ink, B_dir, B_ink = sys.argv[1:5]
centre = np.array([float(c) for c in sys.argv[5:8]], np.float32)
radius = float(sys.argv[8]) if len(sys.argv) > 8 else 400.0
pair = float(sys.argv[9]) if len(sys.argv) > 9 else 4.0
XA, vA = grid(A_dir); XB, vB = grid(B_dir)
PA, qa, ea = pixels_near(XA, vA, q_of(A_ink), centre, radius)
PB, qb, eb = pixels_near(XB, vB, q_of(B_ink), centre, radius)
d, idx = cKDTree(PA).query(PB, distance_upper_bound=pair)
m = np.isfinite(d)
if m.sum() < 100:
    sys.exit(f"too few paired pixels ({m.sum()}): surface B is not on A's sheet near the centre")
a, b = qa[idx[m]], qb[m]
r = float(np.corrcoef(a, b)[0, 1])
print(f"paired {m.sum()} of {len(PB)} B pixels within {radius:.0f} vx of the centre (median 3D gap {np.median(d[m]):.1f} vx)")
print(f"ink r(A,B) = {r:.3f};  ink>0.5: A {np.mean(a > 0.5):.3f}, B {np.mean(b > 0.5):.3f}")
print(f"edge distance of paired pixels: in A median {np.median(ea[idx[m]]):.0f} px, in B median {np.median(eb[m]):.0f} px")
