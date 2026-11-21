#!/usr/bin/env bash
# Screen PHerc0125 for ink: for every seed, grow a GrowPatch surface, render it, run
# ink_9um (seed43 step 20k, validated), and append image-free metrics to a CSV.
# All surfaces, renders and predictions stay PRIVATE under $WORK/screen.
# Usage: bash run_screen.sh SEEDS.txt [GENERATIONS]
set -uo pipefail
SEEDS="${1:?seeds file}"; GEN="${2:-25}"
R=/mnt/d/DBBun/data/vesuvius/PHerc0125
WORK="${WORK:-$HOME/work/PHerc0125}"; OUT=$WORK/screen/g$GEN; mkdir -p "$OUT"/{patches,renders,ink,logs}
PRED=$R/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
VOL=$R/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr
CKPT=$R/models/ink_9um/hybrid_3d2d-seed43/step-020000.pth
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python
CSV=$OUT/screen.csv
[ -f "$CSV" ] || $PY - <<'EOF' >> "$CSV"
import sys; sys.path.insert(0, "/root/vc/scripts"); from screen_metrics import HEADER; print(HEADER)
EOF
cat > "$OUT/params.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $GEN,
 "thread_limit": 8, "normal_grid_path": "$GRIDS", "voxelsize": 9.362}
EOF
export PYTHONPATH=$HOME/vc/villa/vesuvius/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
i=0; T0=$(date +%s)
while read -r X Y Z; do
  [[ "$X" =~ ^# ]] && continue
  i=$((i+1)); N="s$(printf %03d $i)_x${X}_y${Y}_z${Z}"
  grep -q "^$N," "$CSV" && continue                              # resumable
  mkdir -p "$OUT/patches/$N"
  vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params.json" -t "$OUT/patches/$N" --segment-name "$N" \
    --seed "$X" "$Y" "$Z" --rng-seed 1 > "$OUT/logs/$N.grow.log" 2>&1 || { echo "$N,grow_failed" >> "$CSV"; continue; }
  vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$OUT/patches/$N" --num-slices 28 \
    --slice-step 1 --cache-gb 32 --zarr-output "$OUT/renders/$N.zarr" > "$OUT/logs/$N.render.log" 2>&1 \
    || { echo "$N,render_failed" >> "$CSV"; continue; }
  ( cd $HOME/vc/villa && $PY -m vesuvius.ink_detection.inference.infer "$OUT/renders/$N.zarr" "$CKPT" \
      "$OUT/ink/$N.tif" --overlap 0.5 --blend-mode hann --batch-size 8 --direction forward \
      --loader-threads 8 --device cuda ) > "$OUT/logs/$N.ink.log" 2>&1 || { echo "$N,ink_failed" >> "$CSV"; continue; }
  $PY $HOME/vc/scripts/screen_metrics.py "$OUT/renders/$N.zarr" "$OUT/ink/$N.tif" --name "$N" 2>/dev/null \
    | grep "^$N," >> "$CSV" || echo "$N,metrics_failed" >> "$CSV"
  echo "[$i] $N done ($(( $(date +%s) - T0 )) s elapsed)"
done < "$SEEDS"
echo "SCREEN DONE: $(($(wc -l < "$CSV") - 1)) rows in $CSV"
