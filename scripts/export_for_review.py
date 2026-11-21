"""Export candidate patches as PNGs for the AUTHOR'S OWN local visual review.
These images are PRIVATE (First Letters confidentiality): they are written to a local folder
and must not be uploaded anywhere, including to AI services.

For each patch writes:  NAME_ink.png  (rescaled ink probability p' = (p-0.25)/0.5, 8-bit)
                        NAME_ct.png   (mean of the centre 5 render layers, percentile-stretched)
Usage: python export_for_review.py OUT_DIR SCREEN_DIR NAME [NAME ...]
"""
import os, sys
import numpy as np
import zarr
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
out, screen = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for n in sys.argv[3:]:
    p = np.asarray(Image.open(os.path.join(screen, "ink", n + ".tif"))).astype(np.float32)
    if p.max() > 1.5:
        p /= 255.0
    q = np.clip((p - 0.25) / 0.5, 0, 1)
    Image.fromarray((q * 255).astype(np.uint8)).save(os.path.join(out, n + "_ink.png"))
    g = zarr.open(os.path.join(screen, "renders", n + ".zarr"), mode="r")
    v = np.asarray(g if hasattr(g, "shape") else g["0"]); v = np.moveaxis(v, int(np.argmin(v.shape)), 0)
    c = v.shape[0] // 2
    m = v[c - 2:c + 3].astype(np.float32).mean(0)
    lo, hi = np.percentile(m[m > 0], [1, 99]) if (m > 0).any() else (0, 255)
    Image.fromarray((np.clip((m - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(out, n + "_ct.png"))
    print("wrote", n)
