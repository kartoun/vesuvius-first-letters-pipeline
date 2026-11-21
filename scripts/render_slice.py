"""Render one z slice of a local uncompressed level-0 zarr v2 volume (128^3 uint8 chunks)
to a downsampled PNG, with the umbilicus point marked. Used to read off spiral sense.

Usage: python render_slice.py VOLUME_ZARR UMBILICUS_JSON Z OUT_PNG [--step 4]
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("zarr"); ap.add_argument("umbilicus"); ap.add_argument("z", type=int); ap.add_argument("out")
ap.add_argument("--step", type=int, default=4)
a = ap.parse_args()

meta = json.load(open(os.path.join(a.zarr, "0", ".zarray")))
Z, Y, X = meta["shape"]; c = meta["chunks"][0]
zc, zo = divmod(a.z, c)
ny, nx = -(-Y // c), -(-X // c)
img = np.zeros((Y, X), np.uint8)
for j in range(ny):
    for i in range(nx):
        p = os.path.join(a.zarr, "0", str(zc), str(j), str(i))
        if not os.path.exists(p):
            continue
        blk = np.fromfile(p, np.uint8).reshape(c, c, c)[zo]
        img[j*c:(j+1)*c, i*c:(i+1)*c] = blk[:min(c, Y-j*c), :min(c, X-i*c)]
small = Image.fromarray(img[::a.step, ::a.step]).convert("RGB")
pts = json.load(open(a.umbilicus))["control_points"]
pz = min(pts, key=lambda p: abs(p["z"] - a.z))
d = ImageDraw.Draw(small)
ux, uy = pz["x"] / a.step, pz["y"] / a.step
d.ellipse([ux-6, uy-6, ux+6, uy+6], outline=(255, 0, 0), width=2)
small.save(a.out)
print(f"slice z={a.z} saved {a.out} size={small.size} umbilicus(x,y)=({pz['x']},{pz['y']}) at z={pz['z']}")
