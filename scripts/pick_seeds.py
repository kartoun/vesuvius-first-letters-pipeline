"""Pick GrowPatch seed points on the flat sides of a flattened scroll from the surface
prediction (zarr v2, blosc), printing coordinates only (no images).

At slice Z, the papyrus mask (CT > 0, from the local CT zarr) is summarised by its principal
axes. Seeds are placed along the minor axis through the umbilicus, at fractions of the way
from the umbilicus to the edge on both sides, then snapped to the nearest predicted-surface
voxel (prediction > 0) within a small radius.

Usage: python pick_seeds.py CT_ZARR PRED_ZARR UMBILICUS_JSON Z [--fracs 0.25,0.45,0.65]
"""
import argparse, json, os
import numpy as np
import numcodecs


def read_slice(zarr_root, z):
    meta = json.load(open(os.path.join(zarr_root, "0", ".zarray")))
    Z, Y, X = meta["shape"]; cz, cy, cx = meta["chunks"]
    comp = numcodecs.get_codec(meta["compressor"]) if meta["compressor"] else None
    zc, zo = divmod(z, cz)
    out = np.zeros((Y, X), np.uint8)
    for j in range(-(-Y // cy)):
        for i in range(-(-X // cx)):
            p = os.path.join(zarr_root, "0", str(zc), str(j), str(i))
            if not os.path.exists(p):
                continue
            b = open(p, "rb").read()
            a = np.frombuffer(comp.decode(b) if comp else b, np.uint8).reshape(cz, cy, cx)[zo]
            out[j*cy:(j+1)*cy, i*cx:(i+1)*cx] = a[:min(cy, Y-j*cy), :min(cx, X-i*cx)]
    return out


ap = argparse.ArgumentParser()
ap.add_argument("ct"); ap.add_argument("pred"); ap.add_argument("umb"); ap.add_argument("z", type=int)
ap.add_argument("--fracs", default="0.25,0.45,0.65")
a = ap.parse_args()
pts = json.load(open(a.umb))["control_points"]
zs = np.array([p["z"] for p in pts]); o = np.argsort(zs)
ux = np.interp(a.z, zs[o], [pts[i]["x"] for i in o]); uy = np.interp(a.z, zs[o], [pts[i]["y"] for i in o])
ct = read_slice(a.ct, a.z); pred = read_slice(a.pred, a.z)
yy, xx = np.nonzero(ct[::4, ::4]); yy, xx = yy * 4.0, xx * 4.0
c = np.cov(np.stack([xx - xx.mean(), yy - yy.mean()]))
w, v = np.linalg.eigh(c); minor = v[:, 0]                    # (dx, dy) of the minor axis
print(f"z={a.z} umbilicus=({ux:.0f},{uy:.0f}) papyrus px={ct.astype(bool).sum()} "
      f"pred>0 px={(pred > 0).sum()} minor axis=({minor[0]:+.3f},{minor[1]:+.3f}) "
      f"axis ratio={np.sqrt(w[1]/w[0]):.2f}")
py, px = np.nonzero(pred > 0)
for side in (+1, -1):
    d = side * minor
    # distance from umbilicus to the papyrus edge along this direction
    t = 0
    while True:
        x, y = int(ux + d[0] * (t + 1)), int(uy + d[1] * (t + 1))
        if not (0 <= y < ct.shape[0] and 0 <= x < ct.shape[1]) or ct[y, x] == 0:
            break
        t += 1
    for f in map(float, a.fracs.split(",")):
        x0, y0 = ux + d[0] * t * f, uy + d[1] * t * f
        k = np.argmin((px - x0) ** 2 + (py - y0) ** 2)
        dist = np.hypot(px[k] - x0, py[k] - y0)
        print(f"seed side={side:+d} frac={f:.2f} edge_dist={t} -> x={px[k]} y={py[k]} z={a.z} "
              f"(snap {dist:.1f} px, pred={pred[py[k], px[k]]})")
