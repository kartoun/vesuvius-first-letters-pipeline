"""Export a large candidate surface for the author's PRIVATE visual review, in the same format as
the public known-text reference (PHerc0211 winning segment, ensemble mean), so the two can be
compared at the same scale. Output stays local; never upload it (First Letters confidentiality).

Writes OUT_DIR/NAME_compare.png: candidate ensemble-mean ink map (downscaled 1/3), candidate CT
surface (1/3), and the PHerc0211 known-text ensemble map (1/3) underneath, each with a 1 cm bar;
plus NAME_ink_full.png (full resolution ink map) for zooming.
Usage: python export_overview.py OUT_DIR NAME RENDER.zarr INK_MEAN.tif REF_INK_MEAN.tif
"""
import os, sys
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
out, name, rz, ink, ref = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
PPCM = 10000 / float(os.environ.get("VOXEL_UM", "9.362"))   # candidate's pixels per cm


def q_of(p):
    a = np.asarray(Image.open(p)).astype(np.float32)
    if a.max() > 1.5:
        a /= 255.0
    return np.clip((a - 0.25) / 0.5, 0, 1)


def small(a, f=3):
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(
        (max(a.shape[1] // f, 1), max(a.shape[0] // f, 1)), Image.BILINEAR)) / 255.0


q = q_of(ink); r = q_of(ref)
g = zarr.open(rz, mode="r"); v = np.asarray(g if hasattr(g, "shape") else g["0"])
v = np.moveaxis(v, int(np.argmin(v.shape)), 0); c = v.shape[0] // 2
ct = v[c - 2:c + 3].astype(np.float32).mean(0)
m = ct > 0
lo, hi = np.percentile(ct[m], [1, 99]) if m.any() else (0, 255)
ct = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1) * m
Image.fromarray((q * 255).astype(np.uint8)).save(os.path.join(out, f"{name}_ink_full.png"))

PPCM_REF = 10000 / 9.362                                     # PHerc0211 reference is 9.362 um
panels = [(small(q), f"{name}: ensemble mean ink (1/3 scale)", PPCM),
          (small(ct), f"{name}: CT surface (1/3 scale)", PPCM),
          (small(r), "REFERENCE PHerc0211 known text: ensemble mean ink (1/3 scale)", PPCM_REF)]
wmax = max(p.shape[1] for p, _, _ in panels)
fig, axs = plt.subplots(3, 1, figsize=(16, 16 * sum(p.shape[0] for p, _, _ in panels) / wmax + 2))
for a, (img, t, pc) in zip(axs, panels):
    a.imshow(img, cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=11)
    a.set_xlim(0, wmax); a.axis("off")
    a.plot([20, 20 + pc / 3], [img.shape[0] - 15] * 2, color="yellow", lw=3)
    a.text(20, img.shape[0] - 25, "1 cm", color="yellow", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(out, f"{name}_compare.png"), dpi=110)
print("wrote", name, "canvas", q.shape, "CT-covered fraction %.2f" % m.mean())
