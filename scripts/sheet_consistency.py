"""Check whether a large grown surface stays on one papyrus sheet, using independent short surfaces.

Short GrowPatch surfaces seeded at points along the large (target) surface grow independently. Where
the target stays on one sheet, the short surfaces that cover a target point lie on top of it (well
under the ~13-14 voxel sheet spacing). Where the target has jumped to a neighbouring sheet, short
surfaces grown from the other side of the jump disagree by about one sheet spacing.

Distances are point-to-surface: each reference grid (20 vx spacing) is densified by bilinear
interpolation to SUB vx before nearest-neighbour search.

Modes:
  seeds   python sheet_consistency.py seeds TARGET_DIR STRIDE > seeds.txt
          prints "x y z" seeds at valid target grid points on a STRIDE x STRIDE lattice
  check   python sheet_consistency.py check TARGET_DIR OUT_PREFIX [--seeds=SEEDS.txt] REF_DIR [REF_DIR ...]
          (with --seeds, references that do not pass within 3 vx of their own seed are discarded)
          writes OUT_PREFIX_agree.npy (per target grid point: fraction of covering references
          that agree, NaN if fewer than 2 cover it), OUT_PREFIX_cover.npy, and a trimmed tifxyz
          copy OUT_PREFIX_trimmed/ in which points with agreement < 0.5 are set invalid (-1).
Numbers only; no image is written.
"""
import glob, json, os, shutil, sys
import numpy as np
import tifffile
from scipy.spatial import cKDTree

SAME, COVER, SUB, SEED_OK = 5.0, 30.0, 4, 3.0


def load(d):
    x = glob.glob(os.path.join(d, "**", "x.tif"), recursive=True)
    p = os.path.dirname(x[0])
    X = np.stack([tifffile.imread(os.path.join(p, f"{k}.tif")) for k in "xyz"], -1).astype(np.float32)
    return p, X, X[..., 0] > 0


def densify(X, v, sub=SUB):
    pts = [X[v]]
    H, W = v.shape
    for fy in np.linspace(0, 1, sub, endpoint=False)[1:]:
        ok = v[:-1, :] & v[1:, :]
        pts.append(((1 - fy) * X[:-1, :] + fy * X[1:, :])[ok])
    for fx in np.linspace(0, 1, sub, endpoint=False)[1:]:
        ok = v[:, :-1] & v[:, 1:]
        pts.append(((1 - fx) * X[:, :-1] + fx * X[:, 1:])[ok])
    for fy in np.linspace(0, 1, sub, endpoint=False)[1:]:
        for fx in np.linspace(0, 1, sub, endpoint=False)[1:]:
            ok = v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
            q = ((1 - fy) * (1 - fx) * X[:-1, :-1] + fy * (1 - fx) * X[1:, :-1]
                 + (1 - fy) * fx * X[:-1, 1:] + fy * fx * X[1:, 1:])
            pts.append(q[ok])
    return np.concatenate(pts)


def main():
    mode = sys.argv[1]
    if mode == "seeds":
        _, X, v = load(sys.argv[2]); stride = int(sys.argv[3])
        for j in range(stride // 2, v.shape[0], stride):
            for i in range(stride // 2, v.shape[1], stride):
                if v[j, i]:
                    x, y, z = X[j, i]
                    print(f"{x:.0f} {y:.0f} {z:.0f}")
        return
    tdir, out = sys.argv[2], sys.argv[3]
    refs = sys.argv[4:]
    seeds = None
    if refs and refs[0].startswith("--seeds="):
        # GrowPatch does not pass through its seed: its initial optimisation can move the surface
        # up to ~20 vx along the normal, i.e. onto a neighbouring sheet. Only references that pass
        # within SEED_OK vx of their own seed (which lies on the target) are evidence about the
        # target's sheet. References are matched to seeds by their sorted order (r001, r002, ...).
        seeds = np.loadtxt(refs[0].split("=", 1)[1]).reshape(-1, 3); refs = refs[1:]
    tp, X, v = load(tdir)
    tq = X[v]
    agree_n = np.zeros(len(tq)); cover_n = np.zeros(len(tq)); dists = []
    used = rejected = 0
    for k, rd in enumerate(sorted(refs)):
        try:
            _, R, rv = load(rd)
        except IndexError:
            continue
        if rv.sum() < 20:
            continue
        dense = densify(R, rv)
        if seeds is not None:
            if np.min(np.linalg.norm(dense - seeds[k], axis=1)) > SEED_OK:
                rejected += 1
                continue
        used += 1
        d, _ = cKDTree(dense).query(tq, distance_upper_bound=COVER)
        cov = np.isfinite(d)
        cover_n += cov; agree_n += cov & (d < SAME)
        dists.append(d[cov])
    frac = np.where(cover_n >= 2, agree_n / np.maximum(cover_n, 1), np.nan)
    A = np.full(v.shape, np.nan, np.float32); A[v] = frac
    C = np.zeros(v.shape, np.float32); C[v] = cover_n
    np.save(out + "_agree.npy", A); np.save(out + "_cover.npy", C)
    alld = np.concatenate(dists) if dists else np.array([np.nan])
    checked = np.isfinite(frac)
    bad = checked & (frac < 0.5)
    print(f"references used {used}, rejected for leaving their seed {rejected}")
    print(f"target points {v.sum()}, checked (>=2 refs) {checked.sum()} ({checked.mean():.0%}), "
          f"disagree {bad.sum()} ({bad.sum() / max(checked.sum(), 1):.1%} of checked)")
    print("ref-to-target distance (vx) in covered points: "
          + ", ".join(f"p{q}={np.percentile(alld, q):.1f}" for q in (25, 50, 75, 90, 95)))
    print("fraction of covered distances: <5 vx %.2f, 5-9 vx %.2f, 9-20 vx %.2f, >20 vx %.2f" % (
        (alld < 5).mean(), ((alld >= 5) & (alld < 9)).mean(), ((alld >= 9) & (alld < 20)).mean(), (alld >= 20).mean()))
    # Trimmed copy: drop points that most covering references place on another sheet.
    td = out + "_trimmed"
    shutil.rmtree(td, ignore_errors=True); shutil.copytree(tp, td)
    drop = np.zeros(v.shape, bool); drop[v] = bad
    for k in "xyz":
        a = tifffile.imread(os.path.join(td, f"{k}.tif")); a[drop] = -1
        tifffile.imwrite(os.path.join(td, f"{k}.tif"), a)
    m = json.load(open(os.path.join(td, "meta.json"))); m["sheet_consistency_trimmed_points"] = int(drop.sum())
    json.dump(m, open(os.path.join(td, "meta.json"), "w"), indent=1)
    print("trimmed copy:", td, "dropped", int(drop.sum()))


if __name__ == "__main__":
    main()
