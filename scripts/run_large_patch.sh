#!/usr/bin/env bash
# Grow one large GrowPatch surface (several cm along the winding) from a seed, render 48 layers,
# align the sheet per pixel, run the 3-member ensemble with mirror TTA, and write the mean/min
# maps. Outputs are PRIVATE under $OUT (eligible scrolls: First Letters confidentiality).
# Usage: bash run_large_patch.sh SAMPLE VOLUME_ID GENERATIONS NAME X Y Z
set -uo pipefail
SAMPLE="$1"; VOLID="$2"; GEN="$3"; NM="$4"; X="$5"; Y="$6"; Z="$7"
R=/mnt/d/DBBun/data/vesuvius/$SAMPLE
OUT="${OUT:-$HOME/work/$SAMPLE/large}"; mkdir -p "$OUT"/{patches,r48,renders,ink,logs}
PRED=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
VOL=$R/volumes/$VOLID-9.362um-1.2m-113keV-masked.zarr
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
N="${NM}_g$GEN"
cat > "$OUT/params_g$GEN.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $GEN,
 "thread_limit": 12, "normal_grid_path": "$GRIDS", "voxelsize": 9.362}
EOF
T0=$(date +%s); mkdir -p "$OUT/patches/$N"
vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params_g$GEN.json" -t "$OUT/patches/$N" --segment-name "$N" \
  --seed "$X" "$Y" "$Z" --rng-seed 1 > "$OUT/logs/$N.grow.log" 2>&1 || { echo "$N grow failed"; exit 1; }
echo "$N grown in $(( $(date +%s) - T0 )) s: $(grep -o 'generated surface[^,]*' "$OUT/logs/$N.grow.log" | tail -1)"
vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$OUT/patches/$N" --num-slices 48 \
  --slice-step 1 --cache-gb 32 --zarr-output "$OUT/r48/$N.zarr" > "$OUT/logs/$N.render.log" 2>&1 || { echo "$N render failed"; exit 1; }
$PY $S/refine_surface.py "$OUT/r48/$N.zarr" "$OUT/renders/$N.zarr" --out-layers 28 --search 8 | tail -1
bash $S/run_ensemble.sh "$OUT/renders/$N.zarr" "$OUT/ink" "$N" 2>/dev/null | tail -1
echo "$N done in $(( $(date +%s) - T0 )) s"
