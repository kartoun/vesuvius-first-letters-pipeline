"""Rank screen.csv patches for First Letters follow-up (numbers only).

Filters: enough CT tiles, surface on-sheet, little blob-shaped output. Score: geometric mean of
ink_frac and comps200 relative to the PHerc0211 known-text calibration medians (0.10, 22), so
1.0 means "like known text" and ~0.2 means "like blank papyrus" (PHerc0800: 0.02, ~6).

Usage: python rank_screen.py screen.csv [--top 15]
"""
import argparse, csv, math, statistics as st

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--top", type=int, default=15)
a = ap.parse_args()
rows = [r for r in csv.DictReader(open(a.csv)) if r.get("cov") not in (None, "", "grow_failed")]
for r in rows:
    for k in ("cov", "surf_centre", "surf_contr", "ink_frac", "comps200_per_Mpx", "blob_frac"):
        r[k] = float(r[k])
    r["tiles"] = int(r["tiles"])
    r["score"] = math.sqrt(max(r["ink_frac"], 1e-6) / 0.10 * max(r["comps200_per_Mpx"], 0.1) / 22.0)
    r["ok"] = r["tiles"] >= 30 and r["surf_centre"] >= 0.40 and r["blob_frac"] <= 0.10
print(f"patches scored: {len(rows)}; passing quality filters: {sum(r['ok'] for r in rows)}")
for k in ("surf_centre", "ink_frac", "comps200_per_Mpx", "score"):
    v = [r[k] for r in rows]
    print(f"  {k:17s} median={st.median(v):.3f}  p90={sorted(v)[int(0.9*len(v))-1]:.3f}  max={max(v):.3f}")
print("  score bands (all patches): " + ", ".join(
    f"{lo}-{hi}: {sum(lo <= r['score'] < hi for r in rows)}" for lo, hi in ((0, .3), (.3, .6), (.6, 1), (1, 9))))
print(f"\nTOP {a.top} passing filters (score 1.0 ~ known-text windows, ~0.2 ~ blank):")
print("rank,name,score,ink_frac,comps200,blob,surf_centre,surf_contr,tiles")
good = sorted([r for r in rows if r["ok"]], key=lambda r: -r["score"])
for i, r in enumerate(good[:a.top], 1):
    print(f"{i},{r['name']},{r['score']:.2f},{r['ink_frac']:.3f},{r['comps200_per_Mpx']:.1f},"
          f"{r['blob_frac']:.2f},{r['surf_centre']:.2f},{r['surf_contr']:.3f},{r['tiles']}")
