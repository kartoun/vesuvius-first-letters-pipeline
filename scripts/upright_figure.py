"""Public figure (PHerc0211 only): the same automatic GrowPatch surface, inferred three ways, shown in the same
upright view and crop around the known-text segment.
  1. old convention: inference direction taken as for official surface volumes (reads the wrong face on our renders)
  2. corrected face, inferred on the GrowPatch grid canvas (arbitrary in-plane rotation)
  3. corrected face, inferred on the upright canvas (upright_render.py), rotated back
Usage: upright_figure.py SURFACE_TIFXYZ META.json MASKS.npz OUT.png INK1 INK2 INK3 [R1 R2 R3]"""
import json, sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
surf, meta, masks, out = sys.argv[1:5]; inks = sys.argv[5:8]; labels = sys.argv[8:11] or ["", "", ""]
m = json.load(open(meta)); ang = m["angle_deg"]; on = np.load(masks)["on"]
ys, xs = np.nonzero(ndi.binary_dilation(on, iterations=150)); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
tiles = []
for p in inks:
    a = np.asarray(Image.open(p)).astype(np.float32); a = a / 255.0 if a.max() > 1.5 else a
    u = ndi.rotate(a, ang, reshape=True, order=1)[y0:y1, x0:x1]
    lo, hi = np.percentile(u[u > 0.01], [2, 99.5]) if (u > 0.01).any() else (0, 1)
    tiles.append((np.clip((u - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8))
s = 0.5; f = ImageFont.load_default(size=26); H = int(tiles[0].shape[0] * s); W = int(tiles[0].shape[1] * s)
canvas = Image.new("L", (W, (H + 44) * 3), 255); d = ImageDraw.Draw(canvas)
for k, (t, lab) in enumerate(zip(tiles, labels)):
    d.text((6, k * (H + 44) + 8), lab, fill=0, font=f)
    canvas.paste(Image.fromarray(t).resize((W, H), Image.BILINEAR), (0, k * (H + 44) + 40))
canvas.save(out); print("wrote", out, canvas.size)
