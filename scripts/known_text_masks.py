"""Public control scroll: per-surface masks in the UPRIGHT canvas. Usage: masks_up.py SAMPLE s1 s2 ...
on  = automatic-surface pixel lies on the official known-text segment (within 15 vx laterally, 6 vx along the normal)
val = pixel has a valid surface vertex. Saves ~/work/SAMPLE/upright/masks_<s>.npz.
Env TEXT_ROWS="r0 r1": only official-segment rows r0..r1 (ink-canvas px) count as text (PHerc0343: the visible lines)."""
import glob, json, os, sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
Image.MAX_IMAGE_PIXELS = None
rd = lambda d, c: np.asarray(Image.open(f"{d}/{c}.tif")).astype(np.float64)
SM = sys.argv[1]; W = f"/root/work/{SM}"; U = W + "/upright"
M = glob.glob(f"/mnt/d/DBBun/data/vesuvius/{SM}/segments/*/mesh/*.tifxyz")[0]
P = np.stack([rd(M, c) for c in "xyz"], -1); ok = P[..., 2] > 0
gy, gx = np.gradient(P, axis=(0, 1)); N = np.cross(gx, gy); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
ok &= np.isfinite(N).all(-1)
if os.environ.get("TEXT_ROWS"):
    r0, r1 = map(int, os.environ["TEXT_ROWS"].split()); sc = json.load(open(M + "/meta.json"))["scale"][0]
    rows = np.arange(P.shape[0])[:, None] / sc; ok &= (rows >= r0) & (rows <= r1); print("text rows", r0, r1, "mesh rows", int(r0 * sc), int(r1 * sc))
pts, nrm = P[ok], N[ok]; T = cKDTree(pts)
for s in sys.argv[2:]:
    d = f"{W}/scan/{s}/large"; S = np.stack([rd(d, c) for c in "xyz"], -1)
    v = (S[..., 2] > 0) & np.isfinite(S).all(-1); on = np.zeros(v.shape, np.float32)
    dist, i = T.query(S[v]); off = np.einsum("ij,ij->i", S[v] - pts[i], nrm[i])
    on[v] = (np.sqrt(np.maximum(dist ** 2 - off ** 2, 0)) < 15) & (np.abs(off) < 6)
    m = json.load(open(f"{U}/{s}.zarr.json")); H, Wd = m["orig_shape"]
    onO = ndi.zoom(on, (H / on.shape[0], Wd / on.shape[1]), order=0); vO = ndi.zoom(v.astype(np.float32), (H / v.shape[0], Wd / v.shape[1]), order=0)
    onU = ndi.rotate(onO, m["angle_deg"], reshape=True, order=0) > 0.5; vU = ndi.rotate(vO, m["angle_deg"], reshape=True, order=0) > 0.5
    assert list(onU.shape) == m["rot_shape"], (onU.shape, m["rot_shape"])
    np.savez_compressed(f"{U}/masks_{s}.npz", on=onU, val=vU)
    print(SM, s, f"angle {m['angle_deg']:+.0f}  on-text {onU.mean():.3f}  valid {vU.mean():.3f}", flush=True)
