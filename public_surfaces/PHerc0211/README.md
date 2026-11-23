# Three automatically grown PHerc0211 surfaces (public data)

These are the surfaces s1, s2 and s3 used in README sections 10-12 and in ScrollPrize/villa PR #2664 (upright rotation before flat ink inference). Each one was grown with `vc_grow_seg_from_seed` on the public PHerc0211 First Letters area, with no hand editing.

| Surface | Seed (x, y, z) | Generations | Area |
|---|---|---|---|
| s1 | 3744, 3428, 8357 | 20 + 120 | 7.2 cm² |
| s2 | 4035, 4178, 8514 | 20 + 120 | 6.3 cm² |
| s3 | 2877, 3944, 8435 | 20 + 120 | 8.3 cm² |

- Volume: `PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr`, 9.362 µm (public).
- Surface prediction: `PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr` and its `.normal-grids` (public).
- Growth: `sN.params_small.json` (20 generations) and then `sN.params_large.json` (resume, 120 generations), `--rng-seed 1`. Growth ran with 12 threads, so a regrow will not be bit-identical; use these files to reproduce the inference results.
- Grid scale 0.05 (one grid cell = 20 voxels), as written by `vc_grow_seg_from_seed`.

Reproduce the PR #2664 comparison (villa at the PR commit):

```bash
vc_render_tifxyz --volume <PHerc0211 volume> --segmentation s2.tifxyz --zarr-output s2.zarr \
  --scale 1 --group-idx 0 --num-slices 28 --voxel-size 9.362 --voxel-unit micrometer --flip-normals
python -m vesuvius.ink_detection.inference.infer s2.zarr hybrid_3d2d-seed42/step-075000.pth s2_before.tif --direction forward
python -m vesuvius.ink_detection.inference.upright rotate s2.tifxyz s2.zarr s2_up.zarr
python -m vesuvius.ink_detection.inference.infer s2_up.zarr hybrid_3d2d-seed42/step-075000.pth s2_up.tif --direction forward
python -m vesuvius.ink_detection.inference.upright back s2_up.zarr s2_up.tif s2_after.tif
```

`--direction` was `reverse` for s1 and `forward` for s2 and s3; these read the inked face with `--flip-normals` renders.
