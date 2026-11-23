"""Line-spacing validation on UPRIGHT public maps, in millimetres (numbers only).
Pre-registered from PHerc0211 (2026-11-22): a window 'shows text lines' if the row-profile autocorrelation peak
(lag searched 2.3-6.5 mm) falls at 4.0-5.6 mm. Windows are tall strips 11.2 x 3.7 mm (1200 x 400 px at 9.36 um).
Groups per sample: on-text right face / wrong face (same windows), off-text right / wrong, official segment, blank.
Usage: line_spacing_eval.py  (paths in SETS/BLANK below; reads <work>/<sample>/upright/{ink,ink_wrong}, masks_<s>.npz from known_text_masks.py)"""
import csv, os, numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
LAG_MM, BAND_MM, WIN_MM = (2.34, 6.55), (4.0, 5.6), (11.2, 3.75)
VMIN, ERODE = float(os.environ.get("VMIN", "0.95")), int(os.environ.get("ERODE", "48"))
SETS = {"PHerc0211": dict(vox=9.362, surf=("s1", "s2", "s3"), official="/root/work/ensemble_cal/POS_mean.tif", off_rows=None),
        "PHerc0343": dict(vox=8.64, surf=("t02", "t08", "t12", "t14"), official="/root/work/PHerc0343/pc/ink/d_mean.tif", off_rows=(400, 1300))}
BLANK = dict(vox=8.64, path="/root/work/ensemble_cal/NEG_mean.tif")
def load(p):
    a = np.asarray(Image.open(p)).astype(np.float32); return a / 255.0 if a.max() > 1.5 else a
def lag_peak(q, v, vox):
    lo, hi = int(LAG_MM[0] * 1000 / vox), int(LAG_MM[1] * 1000 / vox)
    qq = np.where(v, q - q[v].mean(), 0.0); n = v.sum(axis=1).astype(float)
    prof = np.where(n > 0.5 * n.max(), qq.sum(axis=1) / np.maximum(n, 1), 0.0); hi = min(hi, len(prof) // 2)
    ac = np.correlate(prof, prof, "full")[len(prof) - 1:]
    if ac[0] <= 0 or hi <= lo + 10: return np.nan, np.nan
    seg = (ac / ac[0])[lo:hi]; k = int(np.argmax(seg)); return (lo + k) * vox / 1000, float(seg[k] - np.median(seg))
def windows(A, valid, vox, on=None):
    h, w = int(WIN_MM[0] * 1000 / vox), int(WIN_MM[1] * 1000 / vox); out = []
    V = ndi.uniform_filter(valid.astype(np.float32), (h, w)); O = ndi.uniform_filter(on.astype(np.float32), (h, w)) if on is not None else None
    for y in range(0, A.shape[0] - h + 1, h // 4):
        for x in range(0, A.shape[1] - w + 1, w // 2):
            c = (y + h // 2, x + w // 2)
            if V[c] < VMIN: continue
            lag, st = lag_peak(A[y:y + h, x:x + w], valid[y:y + h, x:x + w], vox)
            out.append(dict(y=y, x=x, onfrac=float(O[c]) if O is not None else np.nan, lag_mm=lag, strength=st))
    return out
rows = []
for SM, cfg in SETS.items():
    U = f"/root/work/{SM}/upright"
    for s in cfg["surf"]:
        mk = np.load(f"{U}/masks_{s}.npz"); on, val = mk["on"], ndi.binary_erosion(mk["val"], iterations=ERODE)
        for face, d in (("right", "ink"), ("wrong", "ink_wrong")):
            for r in windows(load(f"{U}/{d}/{s}_mean.tif"), val, cfg["vox"], on): rows.append(dict(sample=SM, surf=s, group=face, **r))
    A = load(cfg["official"]); v = ndi.binary_erosion(A > 0, iterations=ERODE)
    if cfg["off_rows"]: m = np.zeros_like(v); m[cfg["off_rows"][0]:cfg["off_rows"][1]] = True; on = m
    else: on = np.ones_like(v)
    for r in windows(A, v, cfg["vox"], on): rows.append(dict(sample=SM, surf="official", group="official", **r))
A = load(BLANK["path"])
for r in windows(A, ndi.binary_erosion(A > 0, iterations=ERODE), BLANK["vox"]): rows.append(dict(sample="PHerc0800", surf="blank", group="blank", **r))
with open(f"/root/work/upmetric2_v{VMIN:.2f}.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
def summ(name, L):
    if not L: print(f"    {name:26s} n   0"); return
    lag = np.array([r["lag_mm"] for r in L], float); st = np.array([r["strength"] for r in L], float)
    band = (lag >= BAND_MM[0]) & (lag <= BAND_MM[1])
    surfs = sorted({r["surf"] for r in L})
    print(f"    {name:26s} n {len(L):3d}  in band {band.mean():4.0%}  band & strength>0.2 {(band & (st > 0.2)).mean():4.0%}"
          f"  lag mm quartiles {np.nanpercentile(lag, [25, 50, 75]).round(2)}  surfaces {','.join(surfs)}")
for SM in SETS:
    R = [r for r in rows if r["sample"] == SM]; print(f"\n{SM}")
    summ("on-text, right face", [r for r in R if r["group"] == "right" and r["onfrac"] >= 0.6])
    summ("on-text, wrong face", [r for r in R if r["group"] == "wrong" and r["onfrac"] >= 0.6])
    summ("off-text, right face", [r for r in R if r["group"] == "right" and r["onfrac"] < 0.05])
    summ("off-text, wrong face", [r for r in R if r["group"] == "wrong" and r["onfrac"] < 0.05])
    summ("official, text rows", [r for r in R if r["group"] == "official" and r["onfrac"] >= 0.6])
    summ("official, other rows", [r for r in R if r["group"] == "official" and r["onfrac"] < 0.05])
print("\nPHerc0800"); summ("blank", [r for r in rows if r["group"] == "blank"])
print("UPMETRIC2 DONE")
