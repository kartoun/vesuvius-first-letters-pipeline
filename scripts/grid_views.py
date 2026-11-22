"""Write PRIVATE review images of one surface with an identical labelled grid, so a reviewer can
name a location (e.g. "E4") and compare the same spot across maps: ensemble mean, each member,
depth-shifted runs, and the CT surface. Never upload these images (First Letters confidentiality).

Usage: python grid_views.py OUT_DIR NAME RENDER.zarr GRID_PX SCALE LABEL=MAP.tif [LABEL=MAP.tif ...]
  GRID_PX  grid spacing in full-resolution pixels (e.g. 400, about one letter height at 8.6 um)
  SCALE    downscale factor for the written PNGs (e.g. 2)
Each MAP.tif is an ink probability map (rescaled p' = (p-0.25)/0.5). A "ct" image is always written.
"""
import os, sys
import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
out, name, rz, grid, scale = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.makedirs(out, exist_ok=True)


def q_of(p):
    a = np.asarray(Image.open(p)).astype(np.float32)
    if a.max() > 1.5:
        a /= 255.0
    return np.clip((a - 0.25) / 0.5, 0, 1)


MARGIN = 60   # px of labelled border around the image


def row_name(j):
    return chr(65 + j) if j < 26 else chr(65 + j // 26 - 1) + chr(65 + j % 26)


def write(img01, label):
    im = Image.fromarray((img01 * 255).astype(np.uint8)).convert("RGB")
    W, H = im.size
    im = im.resize((W // scale, H // scale), Image.BILINEAR)
    w, h = im.size; g = grid // scale
    canvas = Image.new("RGB", (w + 2 * MARGIN, h + 2 * MARGIN), (255, 255, 255))
    canvas.paste(im, (MARGIN, MARGIN))
    d = ImageDraw.Draw(canvas)
    big = ImageFont.load_default(size=30); small = ImageFont.load_default(size=16)
    cols = list(range(0, w, g)); rows = list(range(0, h, g))
    for x in cols + [w]:
        d.line([(MARGIN + x, MARGIN), (MARGIN + x, MARGIN + h)], fill=(255, 200, 0), width=2)
    for y in rows + [h]:
        d.line([(MARGIN, MARGIN + y), (MARGIN + w, MARGIN + y)], fill=(255, 200, 0), width=2)
    # Columns are numbered along x, rows lettered along y, on all four edges.
    for i, x in enumerate(cols):
        cx = MARGIN + x + min(g, w - x) // 2 - 10
        d.text((cx, 12), str(i + 1), fill=(0, 0, 0), font=big)
        d.text((cx, MARGIN + h + 12), str(i + 1), fill=(0, 0, 0), font=big)
    for j, y in enumerate(rows):
        cy = MARGIN + y + min(g, h - y) // 2 - 16
        d.text((14, cy), row_name(j), fill=(0, 0, 0), font=big)
        d.text((MARGIN + w + 14, cy), row_name(j), fill=(0, 0, 0), font=big)
    # Small cell name in the corner of every cell, e.g. "C7".
    for j, y in enumerate(rows):
        for i, x in enumerate(cols):
            d.text((MARGIN + x + 4, MARGIN + y + 3), f"{row_name(j)}{i + 1}", fill=(255, 200, 0), font=small)
    d.text((MARGIN, MARGIN + h + 40 - 6), f"{name} {label}: rows A.. (y), columns 1.. (x), "
           f"cell = {grid} px", fill=(0, 0, 0), font=small)
    canvas.save(os.path.join(out, f"{name}_grid_{label}.png"))


g = zarr.open(rz, mode="r"); v = np.asarray(g if hasattr(g, "shape") else g["0"])
v = np.moveaxis(v, int(np.argmin(v.shape)), 0); c = v.shape[0] // 2
ct = v[c - 2:c + 3].astype(np.float32).mean(0); m = ct > 0
lo, hi = np.percentile(ct[m], [1, 99]) if m.any() else (0, 255)
write(np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1) * m, "ct")
for spec in sys.argv[6:]:
    label, path = spec.split("=", 1)
    write(q_of(path), label)
    print("wrote", label)
print(f"grid: {grid} px cells, rows A.. (y), columns 1.. (x); images downscaled x{scale}")
