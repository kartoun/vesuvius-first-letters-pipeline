"""Line-spacing gate for one automatic surface (numbers only; env VOX = voxel size in um, default 8.64), settings validated on public data:
windows 11.2 x 3.75 mm, >= 95% on the surface after a 48-px erosion, lag searched 2.3-6.5 mm, in band 4.0-5.6 mm.
Public reference at these settings: known text 79-94%, auto right face on text 80% (wrong face 40%), off-text 18-36%, blank 0%.
Usage: line_gate.py NAME SURFACE_TIFXYZ UP_DIR   (UP_DIR has render_up.zarr.json, ink/NAME_mean.tif, ink_wrong/NAME_mean.tif)"""
import csv, json, os, sys, numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
VOX = float(os.environ.get("VOX", "8.64")); LAG_MM, BAND_MM, WIN_MM = (2.34, 6.55), (4.0, 5.6), (11.2, 3.75); VMIN, ERODE = 0.95, 48
nm, surf, up = sys.argv[1:4]; meta = json.load(open(f"{up}/render_up.zarr.json"))
def load(p):
    a = np.asarray(Image.open(p)).astype(np.float32); return a / 255.0 if a.max() > 1.5 else a
Z = np.asarray(Image.open(f"{surf}/z.tif")); v = ndi.binary_fill_holes(Z > 0).astype(np.float32); H, Wd = meta["orig_shape"]
zmm = (Z[Z > 0].max() - Z[Z > 0].min()) * VOX / 1000
vO = ndi.zoom(v, (H / v.shape[0], Wd / v.shape[1]), order=0)
V = ndi.binary_erosion(ndi.rotate(vO, meta["angle_deg"], reshape=True, order=0) > 0.5, iterations=ERODE)
def lag_peak(q, m):
    lo, hi = int(LAG_MM[0] * 1000 / VOX), int(LAG_MM[1] * 1000 / VOX)
    qq = np.where(m, q - q[m].mean(), 0.0); n = m.sum(axis=1).astype(float)
    prof = np.where(n > 0.5 * n.max(), qq.sum(axis=1) / np.maximum(n, 1), 0.0); hi = min(hi, len(prof) // 2)
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    if ac[0] <= 0 or hi <= lo + 10: return np.nan, np.nan
    seg = (ac / ac[0])[lo:hi]; k = int(np.argmax(seg)); return (lo + k) * VOX / 1000, float(seg[k] - np.median(seg))
h, w = int(WIN_MM[0] * 1000 / VOX), int(WIN_MM[1] * 1000 / VOX); Vf = ndi.uniform_filter(V.astype(np.float32), (h, w))
pos = [(y, x) for y in range(0, V.shape[0] - h + 1, h // 4) for x in range(0, V.shape[1] - w + 1, w // 2) if Vf[y + h // 2, x + w // 2] >= VMIN]
res = {}; rows = []
for face, d in (("right", "ink"), ("wrong", "ink_wrong")):
    A = load(f"{up}/{d}/{nm}_mean.tif")
    if A.shape != V.shape: A = ndi.zoom(A, (V.shape[0] / A.shape[0], V.shape[1] / A.shape[1]), order=1)
    L = []
    for y, x in pos:
        lag, st = lag_peak(A[y:y + h, x:x + w], V[y:y + h, x:x + w]); L.append(lag)
        rows.append(dict(face=face, y=y, x=x, lag_mm=lag, strength=st))
    L = np.array(L, float); res[face] = float(np.mean((L >= BAND_MM[0]) & (L <= BAND_MM[1]))) if len(L) else np.nan
with open(f"{up}/linetall.csv", "w", newline="") as fh:
    if rows: wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
# Best 2 x 2 cm block (the prize area), stride 1 cm: windows whose centre lies in the block; >= 6 windows.
B = int(20e3 / VOX); best = {"right": (np.nan, None, 0), "wrong": (np.nan, None, 0)}; perb = []
lagf = {f: {(r["y"], r["x"]): r["lag_mm"] for r in rows if r["face"] == f} for f in ("right", "wrong")}
for by in range(0, V.shape[0] - B // 2, B // 2):
    for bx in range(0, V.shape[1] - B // 2, B // 2):
        inb = [(y, x) for y, x in pos if by <= y + h // 2 < by + B and bx <= x + w // 2 < bx + B]
        if len(inb) < 6: continue
        sh = {f: float(np.mean([BAND_MM[0] <= lagf[f][k] <= BAND_MM[1] for k in inb])) for f in ("right", "wrong")}
        perb.append((sh["right"], sh["wrong"], len(inb), by, bx))
br = max(perb, default=None); bw = max(perb, key=lambda t: t[1], default=None)
blk = (f" | best 2x2 cm block right {br[0]:.0%} (its wrong {br[1]:.0%}, n {br[2]}, at {br[3]},{br[4]}); best wrong-face block {bw[1]:.0%}; blocks {len(perb)}"
       if br else " | no 2x2 cm block with >= 6 windows")
gate = "PASS" if len(pos) >= 8 and res["right"] >= 0.6 and res["right"] - res["wrong"] >= 0.2 else "no"
print(f"height {zmm:.1f} mm, windows {len(pos)}, in band right {res['right']:.0%} wrong {res['wrong']:.0%} -> gate {gate}{blk}")
