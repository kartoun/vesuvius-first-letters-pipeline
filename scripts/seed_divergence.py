"""Divergence profile: how far short reference surfaces drift from a target surface as a function
of distance from their seed (which lies on the target). Uses only references that pass within
3 vx of their own seed, so every reference starts on the target's sheet.

A surface tracer that follows one sheet should keep the references on the target (distance near 0)
far from the seed; drift to a neighbouring sheet shows up as distances near the sheet spacing
(about 13-14 vx at 9 um).
Usage: python seed_divergence.py TARGET_DIR SEEDS.txt REF_DIR [REF_DIR ...]
"""
import glob, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheet_consistency import load, densify, SEED_OK

BINS = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 450)]
tdir, seedf, refs = sys.argv[1], sys.argv[2], sorted(sys.argv[3:])
_, X, v = load(tdir); T = cKDTree(densify(X, v))
S = np.loadtxt(seedf).reshape(-1, 3)
prof = {b: [] for b in BINS}; used = rej = 0
for k, rd in enumerate(refs):
    try:
        _, R, rv = load(rd)
    except IndexError:
        continue
    P = densify(R, rv); ds = np.linalg.norm(P - S[k], axis=1)
    if ds.min() > SEED_OK:
        rej += 1; continue
    used += 1
    d = T.query(P, distance_upper_bound=40)[0]
    for lo, hi in BINS:
        m = (ds >= lo) & (ds < hi) & np.isfinite(d)
        if m.sum() > 30:
            prof[(lo, hi)].append(np.median(d[m]))
print(f"references used {used}, rejected for leaving their seed {rej} ({rej / max(used + rej, 1):.0%})")
print("distance from seed (vx) | refs | median of per-ref median distance to target | share of refs within 5 vx")
summary = [str(used), str(rej)]
for (lo, hi), a in prof.items():
    a = np.array(a)
    if len(a):
        print(f"{lo:4d}-{hi:<4d}               | {len(a):4d} | {np.median(a):6.1f} vx | {np.mean(a < 5):.2f}")
        summary.append(f"{np.mean(a < 5):.3f}")
    else:
        summary.append("nan")
# Machine-readable line: used, rejected, share within 5 vx for each distance bin.
print("SUMMARY," + ",".join(summary))
