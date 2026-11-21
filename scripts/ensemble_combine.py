"""Combine per-model ink probability TIFFs into ensemble maps.

  mean  average of the members' probabilities (the PHerc. 343-style ensemble average)
  min   per-pixel minimum, a strict "every member agrees" rule in the spirit of the
        PHerc. 306B product rule, which suppresses ink claimed by only one model
Output TIFFs hold raw probabilities scaled to 0-255, like the members, so screen_metrics.py
reads them unchanged. Also prints the mean pairwise Pearson r between members (agreement).

Usage: python ensemble_combine.py OUT_PREFIX MEMBER.tif [MEMBER.tif ...]
"""
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load(p):
    a = np.asarray(Image.open(p)).astype(np.float32)
    return a / 255.0 if a.max() > 1.5 else a


out, members = sys.argv[1], sys.argv[2:]
P = np.stack([load(m) for m in members])
for name, arr in (("mean", P.mean(0)), ("min", P.min(0))):
    Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8)).save(f"{out}_{name}.tif")
flat = P.reshape(len(members), -1)
sel = flat.max(0) > 0
r = np.corrcoef(flat[:, sel]) if sel.sum() > 10 else np.full((len(members),) * 2, np.nan)
iu = np.triu_indices(len(members), 1)
print(f"{out}: members={len(members)} mean_pairwise_r={np.nanmean(r[iu]):.3f}")
