"""Choose, per grid cell, which side of the papyrus the ink model should read, and merge the
forward and reverse cell scans accordingly.

Convention (measured on the PHerc0211 known-text segment): --direction forward reads the ink when
the surface grid normal, cross(d/drow, d/dcol), points toward the scroll centre. The centre at a
given z is the centroid of the filled papyrus mask of a coarse pyramid level (public S3 volume).
That holds for the official surface volumes. vc_render_tifxyz renders stack layers the other way, so
for our renders pass the cells read with --direction reverse as FWD_CELLS and forward as REV_CELLS.
Writes NAME_cells_sided.csv (cell, side, toward, class, metrics) and prints the green cells.
Usage: python pick_side.py SURFACE_DIR VOLUME_S3_PATH FWD_CELLS.csv REV_CELLS.csv OUT.csv [GRID_PX]
"""
import csv, os, sys
import numpy as np
import zarr
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_consistency import load

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
sdir, vpath, fwd, rev, out = sys.argv[1:6]
G = int(sys.argv[6]) if len(sys.argv) > 6 else 400
grp = zarr.open_group(S3 + vpath, mode="r"); LV = 4; F = 2 ** LV; cache = {}


def centre(z):
    k = int(z // F)
    if k not in cache:
        sl = ndi.binary_fill_holes(ndi.binary_closing(np.asarray(grp[str(LV)][k]) > 0, iterations=3))
        ys, xs = np.nonzero(sl); cache[k] = np.array([xs.mean() * F, ys.mean() * F])
    return cache[k]


_, X, v = load(sdir); H, W = v.shape
step = G // 20                                     # grid cells per canvas cell (20 px per grid cell)
read = lambda p: {r["cell"]: r for r in csv.DictReader(open(p))}
F_, R_ = read(fwd), read(rev)
rows = []
for cell, rf in F_.items():
    j = ord(cell[0]) - 65; i = int(cell[1:]) - 1
    tw = []
    for jj in range(j * step, min((j + 1) * step, H - 1)):
        for ii in range(i * step, min((i + 1) * step, W - 1)):
            if jj < 1 or ii < 1 or not (v[jj, ii] and v[jj + 1, ii] and v[jj - 1, ii] and v[jj, ii + 1] and v[jj, ii - 1]):
                continue
            n = np.cross(X[jj + 1, ii] - X[jj - 1, ii], X[jj, ii + 1] - X[jj, ii - 1]); n /= np.linalg.norm(n) + 1e-9
            c = centre(X[jj, ii, 2]); t = np.array([c[0] - X[jj, ii, 0], c[1] - X[jj, ii, 1], 0.0])
            tw.append(np.dot(n, t / (np.linalg.norm(t) + 1e-9)))
    toward = float(np.median(tw)) if tw else float("nan")
    side = "fwd" if (np.isnan(toward) or toward >= 0) else "rev"
    r = rf if side == "fwd" else R_[cell]
    rows.append(dict(cell=cell, side=side, toward=f"{toward:.2f}", **{k: r[k] for k in r if k != "cell"}))
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
use = [r for r in rows if r["class"] != "grey"]
g = [f"{r['cell']}({r['side']})" for r in use if r["class"] == "green"]
print(f"{os.path.basename(out)}: usable {len(use)}, green {len(g)} ({len(g) / max(len(use), 1):.0%}); "
      f"cells read forward {sum(r['side'] == 'fwd' for r in use)}, reverse {sum(r['side'] == 'rev' for r in use)}")
print("  green:", " ".join(g) if g else "none")
