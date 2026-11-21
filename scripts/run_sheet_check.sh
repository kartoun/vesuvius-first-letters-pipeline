#!/usr/bin/env bash
# Sheet-consistency check of a large surface: grow short independent reference surfaces
# (GrowPatch, GEN_REF generations) from a lattice of seeds on the target, in parallel, then run
# sheet_consistency.py check. Writes the agreement maps and a trimmed tifxyz copy.
# Usage: bash run_sheet_check.sh SAMPLE VOLUME_ID TARGET_DIR OUT_DIR [STRIDE] [GEN_REF] [JOBS]
set -uo pipefail
SAMPLE="$1"; VOLID="$2"; TGT="$3"; OUT="$4"; STRIDE="${5:-15}"; GREF="${6:-20}"; JOBS="${7:-8}"
R=/mnt/d/DBBun/data/vesuvius/$SAMPLE
PRED=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
mkdir -p "$OUT/refs" "$OUT/logs"
cat > "$OUT/params_ref.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $GREF,
 "thread_limit": 4, "normal_grid_path": "$GRIDS", "voxelsize": 9.362}
EOF
$PY $S/sheet_consistency.py seeds "$TGT" "$STRIDE" > "$OUT/seeds.txt"
echo "reference seeds: $(wc -l < "$OUT/seeds.txt")"
T0=$(date +%s)
i=0
while read -r X Y Z; do
  i=$((i+1)); N=$(printf "r%03d" $i)
  [ -d "$OUT/refs/$N" ] && ls "$OUT/refs/$N"/x.tif >/dev/null 2>&1 && continue
  mkdir -p "$OUT/refs/$N"
  echo "vc_grow_seg_from_seed -v $PRED -p $OUT/params_ref.json -t $OUT/refs/$N --segment-name $N --seed $X $Y $Z --rng-seed 1 > $OUT/logs/$N.log 2>&1"
done < "$OUT/seeds.txt" | xargs -P "$JOBS" -I{} bash -c "{}"
echo "references grown in $(( $(date +%s) - T0 )) s"
$PY $S/sheet_consistency.py check "$TGT" "$OUT/check" "--seeds=$OUT/seeds.txt" "$OUT"/refs/r*
