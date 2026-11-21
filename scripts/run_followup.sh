#!/usr/bin/env bash
# Follow-up on a screen candidate: grow a larger GrowPatch surface, render 48 layers, align the
# sheet per pixel (refine_surface.py), match intensities to the PHerc0211 known-text segment
# (normalize_intensity.py, hist), run ink_9um s43-20k, and score. Writes two variants per seed:
#   NAME_ref   aligned only
#   NAME_hist  aligned + histogram matched
# Everything stays PRIVATE under $WORK/followup.
# Usage: bash run_followup.sh GENERATIONS "NAME X Y Z" ["NAME X Y Z" ...]
set -uo pipefail
GEN="${1:?generations}"; shift
R=/mnt/d/DBBun/data/vesuvius
WORK="${WORK:-$HOME/work/PHerc0125}"; OUT=$WORK/followup; mkdir -p "$OUT"/{patches,r48,renders,ink,logs}
PRED=$R/PHerc0125/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/PHerc0125/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
VOL=$R/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr
CKPT=$R/PHerc0125/models/ink_9um/hybrid_3d2d-seed43/step-020000.pth
REF=$R/PHerc0211/segments/20261028201525-crop1711-lines123-v9/surface-volumes/9.362um-1.2m-113keV-volume-20250821151803.zarr
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
export PYTHONPATH=$HOME/vc/villa/vesuvius/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
CSV=$OUT/followup.csv
[ -f "$CSV" ] || $PY -c "import sys; sys.path.insert(0,'$S'); from screen_metrics import HEADER; print(HEADER)" > "$CSV"
cat > "$OUT/params_g$GEN.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $GEN,
 "thread_limit": 8, "normal_grid_path": "$GRIDS", "voxelsize": 9.362}
EOF
infer() { ( cd $HOME/vc/villa && $PY -m vesuvius.ink_detection.inference.infer "$1" "$CKPT" "$2" --overlap 0.5 \
  --blend-mode hann --batch-size 8 --direction forward --loader-threads 8 --device cuda ) > "$3" 2>&1; }

for SPEC in "$@"; do
  read -r NM X Y Z <<< "$SPEC"; N="${NM}_g$GEN"
  echo "== $N"
  mkdir -p "$OUT/patches/$N"
  /usr/bin/time -f "grow %e s" vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params_g$GEN.json" -t "$OUT/patches/$N" \
    --segment-name "$N" --seed "$X" "$Y" "$Z" --rng-seed 1 > "$OUT/logs/$N.grow.log" 2>&1 || { echo "grow failed"; continue; }
  tail -1 "$OUT/logs/$N.grow.log"
  vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$OUT/patches/$N" --num-slices 48 \
    --slice-step 1 --cache-gb 48 --zarr-output "$OUT/r48/$N.zarr" > "$OUT/logs/$N.render.log" 2>&1 || { echo "render failed"; continue; }
  $PY $S/refine_surface.py "$OUT/r48/$N.zarr" "$OUT/renders/${N}_ref.zarr" --out-layers 28 --search 8
  $PY $S/normalize_intensity.py "$REF" "$OUT/renders/${N}_ref.zarr" "$OUT/renders/${N}_hist.zarr" --mode hist
  for V in ref hist; do
    infer "$OUT/renders/${N}_$V.zarr" "$OUT/ink/${N}_$V.tif" "$OUT/logs/${N}_$V.ink.log" || { echo "ink failed $V"; continue; }
    $PY $S/screen_metrics.py "$OUT/renders/${N}_$V.zarr" "$OUT/ink/${N}_$V.tif" --name "${N}_$V" 2>/dev/null \
      | grep "^${N}_$V," >> "$CSV"
  done
done
echo "FOLLOWUP DONE"
