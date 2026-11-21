"""Render one z slice of a public Vesuvius zarr volume straight from S3, fetching
only the chunks that slice touches. Works for uncompressed uint8 zarr v2 levels.

Usage: python remote_slice.py SCROLL VOLUME_LONG_ID LEVEL Z_FRACTION OUT_PNG [--crop y0,y1,x0,x1]
  Z_FRACTION: 0..1 of the level's z extent (0.5 = mid-height)
"""
import argparse, json, urllib.request, concurrent.futures as cf
import numpy as np
from PIL import Image

S3 = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
ap = argparse.ArgumentParser()
ap.add_argument("scroll"); ap.add_argument("volume"); ap.add_argument("level", type=int)
ap.add_argument("zfrac", type=float); ap.add_argument("out")
ap.add_argument("--crop", default=None)
a = ap.parse_args()
base = f"{S3}{a.scroll}/volumes/{a.volume}/{a.level}/"
meta = json.loads(urllib.request.urlopen(base + ".zarray", timeout=60).read())
assert meta["compressor"] is None and meta["dtype"] == "|u1", meta
Z, Y, X = meta["shape"]; cz, cy, cx = meta["chunks"]; sep = meta.get("dimension_separator", ".")
z = int(a.zfrac * (Z - 1)); zc, zo = divmod(z, cz)
y0, y1, x0, x1 = (0, Y, 0, X) if not a.crop else map(int, a.crop.split(","))
img = np.zeros((y1 - y0, x1 - x0), np.uint8)

def get(j, i):
    url = base + sep.join(map(str, (zc, j, i)))
    try:
        b = urllib.request.urlopen(url, timeout=120).read()
    except Exception:
        return j, i, None
    return j, i, np.frombuffer(b, np.uint8).reshape(cz, cy, cx)[zo]

jobs = [(j, i) for j in range(y0 // cy, -(-y1 // cy)) for i in range(x0 // cx, -(-x1 // cx))]
with cf.ThreadPoolExecutor(32) as ex:
    for j, i, blk in ex.map(lambda t: get(*t), jobs):
        if blk is None:
            continue
        ys, xs = j * cy, i * cx
        sy0, sx0 = max(ys, y0), max(xs, x0)
        sy1, sx1 = min(ys + cy, y1, Y), min(xs + cx, x1, X)
        img[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = blk[sy0 - ys:sy1 - ys, sx0 - xs:sx1 - xs]
Image.fromarray(img).save(a.out)
print(f"{a.scroll} level {a.level} z={z}/{Z} (full-res z~{z * 2**a.level}) shape {img.shape} -> {a.out}")
