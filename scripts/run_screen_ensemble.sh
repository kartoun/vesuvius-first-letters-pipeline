#!/usr/bin/env bash
# Whole-scroll ensemble screen. For every seed: grow a GrowPatch surface on the m7 prediction,
# render 48 layers, align the sheet per pixel (refine_surface.py -> 28 layers), run a 3-member
# checkpoint ensemble with mirror TTA (run_ensemble.sh), and score the ensemble mean and min maps
# (screen_metrics.py) plus member agreement. Resumable. All outputs PRIVATE under $OUT.
# Usage: bash run_screen_ensemble.sh SAMPLE VOLUME_ID SEEDS.txt [GENERATIONS] [WORKER NWORKERS]
#   Run NWORKERS copies with WORKER=0..NWORKERS-1 to process seeds in parallel; each worker takes
#   every NWORKERS-th seed, so patch names stay the same as in a single-worker run.
set -uo pipefail
SAMPLE="${1:?sample}"; VOLID="${2:?volume id}"; SEEDS="${3:?seeds}"; GEN="${4:-25}"
WK="${5:-0}"; NW="${6:-1}"
R=/mnt/d/DBBun/data/vesuvius/$SAMPLE
OUT="${OUT:-$HOME/work/$SAMPLE/ens_screen/g$GEN}"; mkdir -p "$OUT"/{patches,r48,renders,ink,logs}
PRED=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
VOL=$R/volumes/$VOLID-9.362um-1.2m-113keV-masked.zarr
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
CSV=$OUT/screen.csv
[ -f "$CSV" ] || { $PY -c "import sys; sys.path.insert(0,'$S'); from screen_metrics import HEADER; print(HEADER + ',agree_r')" > "$CSV"; }
cat > "$OUT/params.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $GEN,
 "thread_limit": 8, "normal_grid_path": "$GRIDS", "voxelsize": 9.362}
EOF
i=0; T0=$(date +%s)
while read -r X Y Z; do
  [[ "$X" =~ ^# ]] && continue
  i=$((i+1)); N="s$(printf %03d $i)_x${X}_y${Y}_z${Z}"
  [ $(( i % NW )) -eq "$WK" ] || continue
  grep -q "^${N}_mean," "$CSV" && continue
  grep -q "^$N,grow_failed" "$CSV" && continue
  mkdir -p "$OUT/patches/$N"
  vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params.json" -t "$OUT/patches/$N" --segment-name "$N" \
    --seed "$X" "$Y" "$Z" --rng-seed 1 > "$OUT/logs/$N.grow.log" 2>&1 || { echo "$N,grow_failed" >> "$CSV"; continue; }
  vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$OUT/patches/$N" --num-slices 48 \
    --slice-step 1 --cache-gb "${CACHE_GB:-12}" --zarr-output "$OUT/r48/$N.zarr" > "$OUT/logs/$N.render.log" 2>&1 \
    || { echo "$N,render_failed" >> "$CSV"; continue; }
  $PY $S/refine_surface.py "$OUT/r48/$N.zarr" "$OUT/renders/$N.zarr" --out-layers 28 --search 8 > "$OUT/logs/$N.refine.log" 2>&1 \
    || { echo "$N,refine_failed" >> "$CSV"; continue; }
  rm -rf "$OUT/r48/$N.zarr"
  AG=$(bash $S/run_ensemble.sh "$OUT/renders/$N.zarr" "$OUT/ink" "$N" 2>/dev/null | grep -o "mean_pairwise_r=[-0-9.na]*" | cut -d= -f2)
  for K in mean min; do
    L=$($PY $S/screen_metrics.py "$OUT/renders/$N.zarr" "$OUT/ink/${N}_$K.tif" --name "${N}_$K" 2>/dev/null | grep "^${N}_$K,")
    [ -n "$L" ] && echo "$L,${AG:-nan}" >> "$CSV" || echo "${N}_$K,metrics_failed" >> "$CSV"
  done
  echo "[$i] $N done ($(( $(date +%s) - T0 )) s elapsed)"
done < "$SEEDS"
echo "ENS SCREEN DONE worker $WK/$NW"
