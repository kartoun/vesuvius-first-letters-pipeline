"""Reference figure: what known ink looks like, from public data only (no eligible scroll).

Row 1  Fragment Frag1 (PHercParis2Fr47, 3.24 um): infrared photo, hand ink labels, CT surface.
Row 2  PHerc0211 First Letters winning segment (9.36 um): official ink_9um prediction, our 3-model
       ensemble mean, and the CT surface, each for the whole segment.
Row 3  PHerc0211 zooms: a letter-rich strip at full resolution, and one 1040 px patch-sized crop
       (the size of a screen patch), which is how small windows of real text look.
Data: Vesuvius Challenge open data (CC-BY-NC 4.0); EduceLab-Scrolls fragments (arXiv:2304.02084).
Usage: python reference_ink_figure.py OUT.png
"""
import glob, sys
import numpy as np
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
D = "/mnt/d/DBBun/data/vesuvius"
F = f"{D}/Frag1"
SEG = f"{D}/PHerc0211/segments/20261028201525-crop1711-lines123-v9"


def stretch(a, lo=1, hi=99, m=None):
    v = a[m] if m is not None else a
    l, h = np.percentile(v, [lo, hi])
    return np.clip((a - l) / (h - l + 1e-9), 0, 1)


def q_of(path):
    p = np.asarray(Image.open(path)).astype(np.float32)
    if p.max() > 1.5:
        p /= 255.0
    return np.clip((p - 0.25) / 0.5, 0, 1)


def scalebar(ax, px_per_cm, h, w, label="1 cm"):
    x0, y0 = 0.05 * w, 0.93 * h
    ax.plot([x0, x0 + px_per_cm], [y0, y0], color="yellow", lw=3)
    ax.text(x0, y0 - 0.03 * h, label, color="yellow", fontsize=9)


# Fragment crop: densest 2400 px window of the ink labels (about 7.8 mm at 3.24 um).
y, x, s = 2700, 1200, 2400
ir = np.asarray(Image.open(f"{F}/ir.png").convert("L"))[y:y + s, x:x + s].astype(np.float32)
lab = np.asarray(Image.open(f"{F}/inklabels.png").convert("L"))[y:y + s, x:x + s] > 0
ct = np.mean([np.asarray(Image.open(f"{F}/surface_volume/{k:02d}.tif"))[y:y + s, x:x + s].astype(np.float32)
              for k in range(30, 35)], axis=0)

# PHerc0211 segment.
off = q_of(glob.glob(f"{SEG}/ink-detection/*.tif")[0])
ens = q_of("/root/work/ensemble_cal/POS_mean.tif")
g = zarr.open(glob.glob(f"{SEG}/surface-volumes/*.zarr")[0], mode="r")
v = np.asarray(g if hasattr(g, "shape") else g["0"])
v = np.moveaxis(v, int(np.argmin(v.shape)), 0)
c = v.shape[0] // 2
seg_ct = v[c - 2:c + 3].astype(np.float32).mean(0)
seg_ct = stretch(seg_ct, m=seg_ct > 0)
H, W = ens.shape
ppcm9 = 10000 / 9.362          # pixels per cm at 9.362 um
ppcm3 = 10000 / 3.24

fig, axs = plt.subplots(3, 3, figsize=(18, 14), gridspec_kw={"height_ratios": [1.6, 0.9, 1.0]})
for a in axs.ravel():
    a.axis("off")
rows = [
    (ir / 255, "Frag1: infrared photo (ink is visible to the eye)", ppcm3),
    (lab.astype(float), "Frag1: hand-drawn ink labels (ground truth)", ppcm3),
    (stretch(ct), "Frag1: CT surface at 3.24 um (ink barely visible)", ppcm3),
]
for a, (img, t, pc) in zip(axs[0], rows):
    a.imshow(img, cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=11)
    scalebar(a, pc * 0.5, *img.shape, label="5 mm")
for a, (img, t) in zip(axs[1], [(off, "PHerc0211 (9.36 um): official ink_9um prediction, whole segment"),
                               (ens, "PHerc0211: our 3-model ensemble mean, whole segment"),
                               (seg_ct, "PHerc0211: CT surface, whole segment")]):
    a.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="equal"); a.set_title(t, fontsize=11)
    scalebar(a, ppcm9, *img.shape)
zy, zx = slice(1050, 1720), slice(3600, 5600)
axs[2, 0].imshow(off[zy, zx], cmap="gray", vmin=0, vmax=1)
axs[2, 0].set_title("Zoom (official prediction): letter-rich strip, full resolution", fontsize=11)
scalebar(axs[2, 0], ppcm9 * 0.5, 670, 2000, label="5 mm")
axs[2, 1].imshow(ens[zy, zx], cmap="gray", vmin=0, vmax=1)
axs[2, 1].set_title("Same strip, our ensemble mean", fontsize=11)
scalebar(axs[2, 1], ppcm9 * 0.5, 670, 2000, label="5 mm")
axs[2, 2].imshow(ens[0:1040, 1000:2040], cmap="gray", vmin=0, vmax=1)
axs[2, 2].set_title("One 1040 px crop of KNOWN text (= one screen patch)", fontsize=11)
scalebar(axs[2, 2], ppcm9 * 0.5, 1040, 1040, label="5 mm")
fig.suptitle("What known ink looks like: fragment ground truth (top) vs. 9 um scroll model output (bottom). "
             "Public data only; Vesuvius Challenge open data, CC-BY-NC 4.0.", fontsize=13)
fig.tight_layout()
fig.savefig(sys.argv[1], dpi=110)
print("saved", sys.argv[1])
