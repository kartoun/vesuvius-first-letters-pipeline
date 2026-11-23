"""Helpers for the PHerc0211 surface-quality test (public data).
  smooth SRC DST SIGMA         : Gaussian-smooth a tifxyz grid (valid points only), write DST tifxyz
  dir SURF REF REF_DIR         : inference direction for SURF reading the same face as REF read with REF_DIR
  score SURF INK               : r vs the official-segment ensemble ink at on-sheet points (|off|<3 vx), n
"""
import glob, json, os, shutil, sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
Image.MAX_IMAGE_PIXELS = None
rd = lambda d, c: np.asarray(Image.open(os.path.join(d, f"{c}.tif"))).astype(np.float64)
def tdir(d):
    return d if os.path.exists(os.path.join(d, "x.tif")) else os.path.dirname(glob.glob(os.path.join(d, "**", "x.tif"), recursive=True)[0])
def grid(d):
    d = tdir(d); S = np.stack([rd(d, c) for c in "xyz"], -1); return S, (S[..., 2] > 0) & np.isfinite(S).all(-1), d
def normals(P):
    dr, dc = np.gradient(P, axis=(0, 1)); n = np.cross(dr, dc); return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
cmd = sys.argv[1]
if cmd == "smooth":
    src, dst, sg = sys.argv[2], sys.argv[3], float(sys.argv[4])
    S, v, sd = grid(src); w = ndi.gaussian_filter(v.astype(float), sg)
    out = S.copy()
    for k in range(3):
        out[..., k] = np.where(v, ndi.gaussian_filter(np.where(v, S[..., k], 0), sg) / np.maximum(w, 1e-6), S[..., k])
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(sd, dst)
    for k, c in enumerate("xyz"):
        a = np.where(v, out[..., k], S[..., k]).astype(np.float32)
        Image.fromarray(a).save(os.path.join(dst, f"{c}.tif"))
    print("smoothed", dst, "max shift %.2f vx" % np.linalg.norm((out - S)[v], axis=1).max())
elif cmd == "dir":
    A, va, _ = grid(sys.argv[2]); B, vb, _ = grid(sys.argv[3]); refdir = sys.argv[4]
    NA, NB = normals(A), normals(B); ok = va & np.isfinite(NA).all(-1); okb = vb & np.isfinite(NB).all(-1)
    T = cKDTree(B[okb]); d, i = T.query(A[ok]); m = d < 10
    s = np.median(np.einsum("ij,ij->i", NA[ok][m], NB[okb][i[m]]))
    print(refdir if s > 0 else ("reverse" if refdir == "forward" else "forward"))
elif cmd == "score":
    S, v, sd = grid(sys.argv[2]); ink = sys.argv[3]
    sc = json.load(open(os.path.join(sd, "meta.json")))["scale"][0]
    M = glob.glob("/mnt/d/DBBun/data/vesuvius/PHerc0211/segments/*/mesh/*.tifxyz/")[0]
    P = np.stack([rd(M, c) for c in "xyz"], -1); ok = P[..., 2] > 0
    gy, gx = np.gradient(P, axis=(0, 1)); N = np.cross(gx, gy); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    ok &= np.isfinite(N).all(-1); idx = np.argwhere(ok); T = cKDTree(P[ok])
    yy, xx = np.nonzero(v); Q = S[v]; dist, i = T.query(Q); a, b = idx[i].T
    off = np.einsum("ij,ij->i", Q - P[a, b], N[a, b]); tan = Q - P[a, b] - off[:, None] * N[a, b]
    sol = np.einsum("nij,nj->ni", np.linalg.pinv(np.stack([gx[a, b], gy[a, b]], -1)), tan)
    oc, orow = (b + sol[:, 0]) / 0.05, (a + sol[:, 1]) / 0.05
    sel = (np.abs(off) < 3) & (np.abs(sol) < 1).all(1)
    sm = lambda p: ndi.gaussian_filter(np.asarray(Image.open(p)).astype(np.float32), 6)
    R = sm("/root/work/ensemble_cal/POS_mean.tif"); A = sm(ink)
    fy, fx = A.shape[0] / (S.shape[0] / sc), A.shape[1] / (S.shape[1] / sc)
    ay = np.clip((yy[sel] / sc * fy).astype(int), 0, A.shape[0] - 1); ax = np.clip((xx[sel] / sc * fx).astype(int), 0, A.shape[1] - 1)
    ry = np.clip(orow[sel].astype(int), 0, R.shape[0] - 1); rx = np.clip(oc[sel].astype(int), 0, R.shape[1] - 1)
    print(f"r_vs_official {np.corrcoef(A[ay, ax], R[ry, rx])[0, 1]:+.3f} n {sel.sum()} offset_sd {np.std(off[sel]):.2f}")
