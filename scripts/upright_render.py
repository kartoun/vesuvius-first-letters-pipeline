"""Rotate a rendered surface volume so the scroll axis (z) runs down the canvas, as in the official segment
renders (papyrus fibres and text lines horizontal), and rotate ink maps back afterwards.

GrowPatch grids have an arbitrary in-plane rotation; the ink models see official-style upright renders.
  rotate SURFACE_TIFXYZ RENDER.zarr OUT.zarr   -> writes OUT.zarr (+ OUT.zarr.json with angle and shapes)
  back   OUT.zarr.json INK.tif INK_BACK.tif     -> ink map in the original canvas
  fwd    OUT.zarr.json MAP.tif MAP_UP.tif       -> any original-canvas map in the upright canvas (viewing)
The angle is the median direction of the z gradient over the valid grid (one global rotation per surface).
"""
import json, os, sys, numpy as np, zarr
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
cmd = sys.argv[1]
if cmd == "rotate":
    sd, rz, out = sys.argv[2:5]
    Z = np.asarray(Image.open(f"{sd}/z.tif")).astype(np.float64); v = Z > 0
    # Pick the angle empirically: rotate the z map itself with the same function and keep the angle that makes
    # z increase straight down the rows (handles mirrored grids, where the sign of the rotation flips).
    zf = np.where(v, Z, 0.0); vf = v.astype(float); best = None
    for ang_c in np.arange(-180, 180, 1.0):
        zr = ndi.rotate(zf, ang_c, reshape=True, order=1); vr = ndi.rotate(vf, ang_c, reshape=True, order=1) > 0.99
        gy, gx = np.gradient(np.where(vr, zr, np.nan)); m = np.isfinite(gy) & np.isfinite(gx)
        if m.sum() < 50: continue
        down, side = np.median(gy[m]), np.median(np.abs(gx[m]))
        score = down - 2 * side
        if best is None or score > best[0]: best = (score, ang_c, down, side)
    ang = float(best[1])
    print(f"z per grid row after rotation {best[2]:+.2f} vx, |z per column| {best[3]:.2f} vx")
    g = zarr.open(rz, mode="r"); a = g if hasattr(g, "shape") else g["0"]
    L, H, W = a.shape
    first = ndi.rotate(np.asarray(a[0]), ang, reshape=True, order=1)
    o = zarr.open_array(out, mode="w", shape=(L,) + first.shape, chunks=(L, 128, 128), dtype=a.dtype)
    for k in range(L):
        o[k] = ndi.rotate(np.asarray(a[k]), ang, reshape=True, order=1)
    json.dump({"angle_deg": float(ang), "orig_shape": [H, W], "rot_shape": list(first.shape)}, open(out + ".json", "w"))
    print(f"rotated by {ang:+.1f} deg: {H}x{W} -> {first.shape[0]}x{first.shape[1]}")
elif cmd == "fwd":     # any 2-D map in the original canvas -> upright canvas (for viewing)
    meta, src, dst = sys.argv[2:5]; m = json.load(open(meta))
    a = np.asarray(Image.open(src)); order = 0 if a.dtype == np.uint8 and a.max() < 2 else 1
    Image.fromarray(ndi.rotate(a, m["angle_deg"], reshape=True, order=order)).save(dst)
elif cmd == "back":
    meta, ink, dst = sys.argv[2:5]; m = json.load(open(meta))
    a = np.asarray(Image.open(ink)).astype(np.float32)
    if list(a.shape) != m["rot_shape"]:
        a = ndi.zoom(a, (m["rot_shape"][0] / a.shape[0], m["rot_shape"][1] / a.shape[1]), order=1)
    b = ndi.rotate(a, -m["angle_deg"], reshape=True, order=1)
    H, W = m["orig_shape"]; y0, x0 = (b.shape[0] - H) // 2, (b.shape[1] - W) // 2
    Image.fromarray(np.clip(b[y0:y0 + H, x0:x0 + W], 0, 255).astype(np.uint8)).save(dst)
    print(f"rotated back to {H}x{W}")
