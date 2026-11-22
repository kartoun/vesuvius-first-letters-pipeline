"""Score every grid cell of a candidate surface against known-text and blank calibration, and
write a PRIVATE colour overlay (never upload it: First Letters confidentiality).

Each cell (GRID px, aligned with grid_views.py: rows A.., columns 1..) gets the box_test.py metrics.
Thresholds come from the known-text (pos) window distribution:
  red     CT-structure copying (ct_r outside the pos 5-95% range), blob (ink_frac > pos p95),
          or member agreement / depth persistence below the blank (neg) median
  green   agree, depth and contrast all >= pos p25, ct_r inside the pos 5-95% range,
          ink_frac <= pos p95
  yellow  everything else;  grey = too little CT coverage
The same classifier is applied to the pos and neg windows themselves, so a surface's green rate can
be read against "text" and "blank" green rates.
Usage: python cell_scan.py SPEC_JSON OUT_DIR NAME [GRID]
  (SPEC_JSON as for box_test.py; uses sets "cand", "pos", "neg")
"""
import json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "box_test.py")).read().split("\nspec = json.load")[0]
ns = {}; exec(compile(src, "box_test_helpers", "exec"), ns)
spec_path, out, name = sys.argv[1], sys.argv[2], sys.argv[3]
G = int(sys.argv[4]) if len(sys.argv) > 4 else 400
spec = json.load(open(spec_path)); os.makedirs(out, exist_ok=True)
sets = {k: ns["load_set"](spec["sets"][k]) for k in ("cand", "pos", "neg")}
pos = [m for _, m in ns["windows"](sets["pos"], G, G)]
neg = [m for _, m in ns["windows"](sets["neg"], G, G)]
P = lambda k, q: float(np.nanpercentile([m[k] for m in pos], q))
N = lambda k, q: float(np.nanpercentile([m[k] for m in neg], q))
th = dict(agree=P("agree_r", 25), depth=P("depth_r", 25), contrast=P("contrast", 25),
          ct_lo=P("ct_r", 5), ct_hi=P("ct_r", 95), ink_hi=P("ink_frac", 95),
          agree_neg=N("agree_r", 50), depth_neg=N("depth_r", 50))


def classify(m):
    if m is None:
        return "grey"
    if (not th["ct_lo"] <= m["ct_r"] <= th["ct_hi"]) or m["ink_frac"] > th["ink_hi"] \
            or m["agree_r"] < th["agree_neg"] or m["depth_r"] < th["depth_neg"]:
        return "red"
    if m["agree_r"] >= th["agree"] and m["depth_r"] >= th["depth"] and m["contrast"] >= th["contrast"]:
        return "green"
    return "yellow"


rate = lambda lst: np.mean([classify(m) == "green" for m in lst])
D = sets["cand"]; H, W = D["m0"].shape
rows, cols = (H + G - 1) // G, (W + G - 1) // G
cells = {}
for j in range(rows):
    for i in range(cols):
        h, w = min(G, H - j * G), min(G, W - i * G)
        m = ns["metrics"](D, j * G, i * G, h, w) if h >= G // 2 and w >= G // 2 else None
        cells[(j, i)] = (classify(m), m)
cnt = {c: sum(1 for v in cells.values() if v[0] == c) for c in ("green", "yellow", "red", "grey")}
usable = cnt["green"] + cnt["yellow"] + cnt["red"]
print(f"{name}: {usable} usable cells: green {cnt['green']} ({cnt['green'] / max(usable, 1):.0%}), "
      f"yellow {cnt['yellow']}, red {cnt['red']}; grey (no data) {cnt['grey']}")
print(f"  classifier green rate on calibration windows: known text {rate(pos):.0%}, blank {rate(neg):.0%}")
rn = lambda j: chr(65 + j) if j < 26 else chr(65 + j // 26 - 1) + chr(65 + j % 26)
greens = [f"{rn(j)}{i + 1}" for (j, i), (c, _) in sorted(cells.items()) if c == "green"]
print("  green cells:", " ".join(greens) if greens else "none")
with open(os.path.join(out, f"{name}_cells.csv"), "w") as fh:
    fh.write("cell,class,agree_r,depth_r,ct_r,ink_frac,contrast\n")
    for (j, i), (c, m) in sorted(cells.items()):
        vals = ",".join(f"{m[k]:.3f}" for k in ("agree_r", "depth_r", "ct_r", "ink_frac", "contrast")) if m else ",,,,"
        fh.write(f"{rn(j)}{i + 1},{c},{vals}\n")
# Private overlay: ink mean (x1/2) with coloured cell frames and names.
s = 2; im = Image.fromarray((D["m0"] * 255).astype(np.uint8)).convert("RGB")
im = im.resize((W // s, H // s), Image.BILINEAR); d = ImageDraw.Draw(im)
col = {"green": (0, 220, 0), "yellow": (255, 210, 0), "red": (230, 40, 40), "grey": (90, 90, 90)}
f = ImageFont.load_default(size=16)
for (j, i), (c, _) in cells.items():
    x0, y0 = i * G // s, j * G // s; x1, y1 = x0 + G // s - 1, y0 + G // s - 1
    d.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], outline=col[c], width=4 if c == "green" else 2)
    d.text((x0 + 5, y0 + 4), f"{rn(j)}{i + 1}", fill=col[c], font=f)
im.save(os.path.join(out, f"{name}_cell_scores.png"))
