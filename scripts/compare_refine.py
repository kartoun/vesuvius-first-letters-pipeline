"""Compare screen metrics before and after per-pixel layer alignment (numbers only).

Usage: python compare_refine.py SCREEN.csv REFINED.csv CALIBRATION.csv POS_REFINED_WINDOWS.csv
"""
import csv, statistics as st, sys

K = ("surf_centre", "surf_contr", "ink_frac", "comps200_per_Mpx")


def rows(p):
    return {r["name"]: r for r in csv.DictReader(open(p)) if r.get("cov") not in (None, "", "grow_failed")}


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float("nan")


scr, ref, cal, pos = (rows(p) for p in sys.argv[1:5])
print("PHerc0125 patches: before -> after")
print("name," + ",".join(f"{k}_before,{k}_after" for k in K))
d = {k: [] for k in K}
for n, r in ref.items():
    b = scr.get(n[:-4]) if n.endswith("_ref") else None
    if not b:
        continue
    print(n[:-4] + "," + ",".join(f"{f(b, k):.3f},{f(r, k):.3f}" for k in K))
    for k in K:
        d[k].append(f(r, k) - f(b, k))
for k in K:
    if d[k]:
        print(f"  median change {k:17s} {st.median(d[k]):+.3f}  (improved in {sum(x > 0 for x in d[k])}/{len(d[k])})")

print("\nPHerc0211 positive control, 1040-px windows: original -> refined (medians)")
orig = [r for n, r in cal.items() if n.startswith("POS")]
for k in K:
    a = [f(r, k) for r in orig]; b = [f(r, k) for r in pos.values()]
    if a and b:
        print(f"  {k:17s} {st.median(a):.3f} -> {st.median(b):.3f}  (windows {len(a)} / {len(b)})")
