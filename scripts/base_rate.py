"""Base rate of "text-like" windows: how often does a window pass a criterion in known text,
blank papyrus, and a candidate surface? Guards against picking one lucky cell out of hundreds.
Criterion (default): agree_r >= A, depth_r >= DR, contrast >= C, ct_r <= CT.
Usage: python base_rate.py SPEC_JSON SIZE [--agree 0.61] [--depth 0.93] [--contrast 0.34] [--ct 0.05]
       (SPEC_JSON as for box_test.py; the box entry is ignored except for its "set" name)
"""
import argparse, json, os, sys
import numpy as np
sys.argv, _argv = sys.argv[:1] + ["/dev/null"], sys.argv   # import box_test helpers without running it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec_bt = importlib.util.spec_from_file_location("bt_helpers", os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_test.py"))
src = open(spec_bt.origin).read().split("\nspec = json.load")[0]
ns = {}; exec(compile(src, "box_test_helpers", "exec"), ns)
sys.argv = _argv
ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("size", type=int)
ap.add_argument("--agree", type=float, default=0.61); ap.add_argument("--depth", type=float, default=0.93)
ap.add_argument("--contrast", type=float, default=0.34); ap.add_argument("--ct", type=float, default=0.05)
a = ap.parse_args()
spec = json.load(open(a.spec))
for name, s in spec["sets"].items():
    D = ns["load_set"](s); W = ns["windows"](D, a.size, a.size)
    ok = [p for p, m in W if m["agree_r"] >= a.agree and m["depth_r"] >= a.depth
          and m["contrast"] >= a.contrast and m["ct_r"] <= a.ct]
    print(f"{name:5s}: {len(ok):3d} of {len(W):3d} windows pass ({len(ok) / max(len(W), 1):.0%})"
          + (f"; e.g. {ok[:8]}" if name == spec['box']['set'] else ""))
