"""Reproduction test for chosen cells of a surface (numbers only; outputs PRIVATE).

For each cell: seed short bounded GrowPatch surfaces near the cell centre, keep those on the same
sheet (median gap < 3 vx near the cell and passing within 3 vx of their seed), render, align, run the
ensemble reading the SAME papyrus face as the reviewed surface A, and correlate the two ink maps at
matching 3D points within 150 vx of the cell centre (reproduce_check.py).
Same face: B uses the --direction that produced A's ink at that cell, inverted when B's grid normal is
opposite to A's (both are vc_render_tifxyz renders, so the layer-order convention cancels).
CELLS.csv columns: cell, dir (A's actual --direction there: forward|reverse), role (candidate|baseline),
max_ok (independent surfaces wanted). LAYERS env var is passed to run_ensemble.sh (A's layer window).
Usage: python repro_cells.py SAMPLE VOLID VOLNAME ZMIN ZMAX VOXEL A_SURFACE A_INK.tif CELLS.csv OUT_DIR
"""
import csv, os, subprocess, sys
import numpy as np
from scipy.spatial import cKDTree
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from sheet_consistency import load, densify

(sample, volid, volname, zmin, zmax, voxel, A, A_ink, cells_csv, out) = sys.argv[1:11]
R = f"/mnt/d/DBBun/data/vesuvius/{sample}"
pred = f"{R}/representations/predictions/surfaces/{volid}-surface-20260413222639-surface-m7-L0-th0.2"
vol = f"{R}/volumes/{volname}"
PY = os.path.expanduser("~/vc/villa/spiral-fitting/.venv/bin/python")
os.makedirs(out, exist_ok=True)
params = os.path.join(out, "params.json")
open(params, "w").write(
    f'{{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": 30, "thread_limit": 16, '
    f'"normal_grid_path": "{pred}.normal-grids", "voxelsize": {voxel}, "z_min": {zmin}, "z_max": {zmax}}}')
_, XA, vA = load(A); treeA = cKDTree(densify(XA, vA))


def normal(X, v, c):
    P = X[v]; idx = np.argwhere(v); j, i = idx[np.argmin(np.linalg.norm(P - c, axis=1))]
    n = np.cross(X[j + 1, i] - X[j - 1, i], X[j, i + 1] - X[j, i - 1]); return n / (np.linalg.norm(n) + 1e-9)


def run(cmd, log):
    with open(log, "w") as fh:
        return subprocess.run(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT).returncode


cells = list(csv.DictReader(open(cells_csv)))
res = []
for r in cells:
    cell, dirA, role, max_ok = r["cell"], r["dir"], r["role"], int(r["max_ok"])
    j = (ord(cell[0]) - 65) * 20 + 10; i = (int(cell[1:]) - 1) * 20 + 10
    if not vA[j, i]:
        ok = np.argwhere(vA[j - 10:j + 10, i - 10:i + 10])
        if not len(ok):
            continue
        j, i = j - 10 + ok[len(ok) // 2][0], i - 10 + ok[len(ok) // 2][1]
    c = XA[j, i]; nA = normal(XA, vA, c)
    rs = []
    for dj, di in ((0, 6), (6, 0), (0, -6), (-6, 0), (5, 5), (-5, -5)):
        if len(rs) >= max_ok:
            break
        jj, ii = j + dj, i + di
        if not (0 <= jj < vA.shape[0] and 0 <= ii < vA.shape[1] and vA[jj, ii]):
            continue
        name = f"{cell}_{dj}_{ii - i}"; d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        s = XA[jj, ii]
        if run(f"vc_grow_seg_from_seed -v {pred}.zarr -p {params} -t {d} --segment-name {name} "
               f"--seed {s[0]:.1f} {s[1]:.1f} {s[2]:.1f} --rng-seed 1", d + ".grow.log"):
            continue
        try:
            _, XB, vB = load(d)
        except IndexError:
            continue
        PB = densify(XB, vB); near = PB[np.linalg.norm(PB - c, axis=1) < 150]
        if len(near) < 50 or np.median(treeA.query(near)[0]) > 3 or np.min(np.linalg.norm(PB - s, axis=1)) > 3:
            continue
        flip = np.dot(nA, normal(XB, vB, c)) < 0
        dirB = ("reverse" if dirA == "forward" else "forward") if flip else dirA
        rz = d + ".render.zarr"
        run(f"vc_render_tifxyz --volume {vol} --group-idx 0 --scale 1 --segmentation {d} --num-slices 48 "
            f"--slice-step 1 --cache-gb 32 --zarr-output {d}.r48.zarr", d + ".render.log")
        run(f"{PY} {here}/refine_surface.py {d}.r48.zarr {rz} --out-layers 28 --search 8", d + ".refine.log")
        run(f"rm -rf {d}.r48.zarr", "/dev/null")
        run(f"bash {here}/run_ensemble.sh {rz} {out}/ink {name} {dirB}", d + ".ens.log")
        chk = subprocess.run(f"{PY} {here}/reproduce_check.py {A} {A_ink} {d} "
                             f"{out}/ink/{name}_mean.tif {c[0]} {c[1]} {c[2]} 150 4",
                             shell=True, capture_output=True, text=True).stdout
        line = [l for l in chk.splitlines() if l.startswith("ink r(A,B)")]
        if line:
            rs.append(float(line[0].split("=")[1].split(";")[0]))
    res.append((cell, role, dirA, rs))
    print(f"{cell} ({role}, A {dirA}): independent surfaces {len(rs)}, r = {', '.join(f'{x:.2f}' for x in rs) or 'n/a'}", flush=True)
with open(os.path.join(out, "repro_cells.csv"), "w") as fh:
    fh.write("cell,role,dir,n,r_values,r_mean\n")
    for cell, role, dirA, rs in res:
        fh.write(f"{cell},{role},{dirA},{len(rs)},{' '.join(f'{x:.3f}' for x in rs)},{np.mean(rs) if rs else float('nan'):.3f}\n")
print("REPRO CELLS DONE")
