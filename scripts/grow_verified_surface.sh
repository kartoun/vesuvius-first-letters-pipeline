#!/usr/bin/env bash
# Grow a large surface inside the data band from a seed in a stable region, verify it with the
# seeded divergence test, then render, align, run the ensemble and export a PRIVATE review image
# next to the public known-text reference.
#   1. bounded GrowPatch (z_min/z_max), 20 generations (non-resumed bounded growth stalls there)
#   2. resume the same surface with the same bounds up to GEN generations (--resume-opt skip)
#   3. run_sheet_check.sh + seed_divergence.py on the large surface
#   4. render 48 layers, refine_surface.py -> 28, run_ensemble.sh, export_overview.py
# Usage: bash grow_verified_surface.sh SAMPLE VOLUME_ID NAME X Y Z ZMIN ZMAX GEN PRIVATE_DIR
set -uo pipefail
SAMPLE="$1"; VOLID="$2"; NM="$3"; X="$4"; Y="$5"; Z="$6"; ZMIN="$7"; ZMAX="$8"; GEN="$9"; PRIV="${10}"
R=/mnt/d/DBBun/data/vesuvius/$SAMPLE
PRED=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
VOXEL="${VOXEL:-9.362}"                                   # um; 8.64 for the 8.64 um scans
VOL=$R/volumes/${VOLNAME:-$VOLID-9.362um-1.2m-113keV-masked.zarr}
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
OUT=$HOME/work/$SAMPLE/verified/$NM; mkdir -p "$OUT"/{logs,ink}
p() { cat > "$OUT/params_$1.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $2, "thread_limit": 16,
 "normal_grid_path": "$GRIDS", "voxelsize": $VOXEL, "z_min": $ZMIN, "z_max": $ZMAX}
EOF
}
T0=$(date +%s)
p small 20; mkdir -p "$OUT/small"
vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params_small.json" -t "$OUT/small" --segment-name small \
  --seed "$X" "$Y" "$Z" --rng-seed 1 > "$OUT/logs/small.log" 2>&1 || { echo "small grow failed"; exit 1; }
p large "$GEN"; mkdir -p "$OUT/large"
vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params_large.json" -t "$OUT/large" --segment-name large \
  --resume "$OUT/small" --resume-opt skip --rng-seed 1 > "$OUT/logs/large.log" 2>&1 || { echo "resume failed"; exit 1; }
echo "grown in $(( $(date +%s) - T0 )) s: $(grep -o 'generated surface.*' "$OUT/logs/large.log" | tail -1)"
$PY $S/surface_on_prediction.py "$PRED" "$OUT/large" --n 800 2>/dev/null
bash $S/run_sheet_check.sh "$SAMPLE" "$VOLID" "$OUT/large" "$OUT/check" 15 20 12 2>/dev/null | grep -E "seeds|grown|references used"
$PY $S/seed_divergence.py "$OUT/large" "$OUT/check/seeds.txt" "$OUT"/check/refs/r* 2>/dev/null
rm -rf "$OUT/check/refs"
vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$OUT/large" --num-slices 48 \
  --slice-step 1 --cache-gb 48 --zarr-output "$OUT/r48.zarr" > "$OUT/logs/render.log" 2>&1 || { echo "render failed"; exit 1; }
$PY $S/refine_surface.py "$OUT/r48.zarr" "$OUT/render.zarr" --out-layers 28 --search 8 | tail -1
rm -rf "$OUT/r48.zarr"
bash $S/run_ensemble.sh "$OUT/render.zarr" "$OUT/ink" "$NM" 2>/dev/null | tail -1
$PY $S/screen_metrics.py "$OUT/render.zarr" "$OUT/ink/${NM}_mean.tif" --name "$NM" 2>/dev/null | grep "^$NM"
VOXEL_UM=$VOXEL $PY $S/export_overview.py "$PRIV" "$NM" "$OUT/render.zarr" "$OUT/ink/${NM}_mean.tif" "$HOME/work/ensemble_cal/POS_mean.tif" 2>/dev/null
echo "VERIFIED DONE in $(( $(date +%s) - T0 )) s"
