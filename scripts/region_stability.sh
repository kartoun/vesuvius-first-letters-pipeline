#!/usr/bin/env bash
# Map where GrowPatch holds one sheet. For each seed: grow a small target surface bounded to the
# data band (ZMIN..ZMAX), seed short reference surfaces on a lattice of target points, and record
# how many references stay within 5 vx of the target at increasing distance from their seed
# (seed_divergence.py). Resumable; one CSV row per region. Outputs are geometry only (no images).
# Usage: bash region_stability.sh SAMPLE VOLUME_ID SEEDS.txt OUT_DIR ZMIN ZMAX [STRIDE] [JOBS]
#   VOXEL env var: voxel size in um (default 9.362; 8.64 for the 8.64 um scans). Used for areas only.
set -uo pipefail
SAMPLE="$1"; VOLID="$2"; SEEDS="$3"; OUT="$4"; ZMIN="$5"; ZMAX="$6"; STRIDE="${7:-6}"; JOBS="${8:-12}"
VOXEL="${VOXEL:-9.362}"
R=/mnt/d/DBBun/data/vesuvius/$SAMPLE
PRED=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
mkdir -p "$OUT"
cat > "$OUT/params_target.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": 25, "thread_limit": 16,
 "normal_grid_path": "$GRIDS", "voxelsize": $VOXEL, "z_min": $ZMIN, "z_max": $ZMAX}
EOF
cat > "$OUT/params_ref.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": 15, "thread_limit": 2,
 "normal_grid_path": "$GRIDS", "voxelsize": $VOXEL}
EOF
CSV=$OUT/stability.csv
[ -f "$CSV" ] || echo "region,x,y,z,target_pts,on_pred2,refs_used,refs_rejected,share5_0_50,share5_50_100,share5_100_200,share5_200_300,share5_300_450" > "$CSV"
i=0; T0=$(date +%s)
while read -r X Y Z; do
  [[ "$X" =~ ^# ]] && continue
  i=$((i+1)); N=$(printf "g%03d" $i); D="$OUT/$N"
  grep -q "^$N," "$CSV" && continue
  mkdir -p "$D/target" "$D/refs" "$D/logs"
  vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params_target.json" -t "$D/target" --segment-name "$N" \
    --seed "$X" "$Y" "$Z" --rng-seed 1 > "$D/logs/target.log" 2>&1 || { echo "$N,$X,$Y,$Z,grow_failed" >> "$CSV"; continue; }
  TGT="$D/target"
  NP=$($PY -c "import sys; sys.path.insert(0,'$S'); from sheet_consistency import load; print(int(load('$TGT')[2].sum()))" 2>/dev/null)
  ONP=$($PY $S/surface_on_prediction.py "$PRED" "$TGT" --n 300 2>/dev/null | grep -o "on-pred(<=2vx) [0-9.]*" | cut -d' ' -f2)
  $PY $S/sheet_consistency.py seeds "$TGT" "$STRIDE" > "$D/seeds.txt" 2>/dev/null
  j=0
  while read -r a b c; do
    j=$((j+1)); M=$(printf "r%03d" $j); mkdir -p "$D/refs/$M"
    echo "vc_grow_seg_from_seed -v $PRED -p $OUT/params_ref.json -t $D/refs/$M --segment-name $M --seed $a $b $c --rng-seed 1 > $D/logs/$M.log 2>&1"
  done < "$D/seeds.txt" | xargs -P "$JOBS" -I{} bash -c "{}"
  SUM=$($PY $S/seed_divergence.py "$TGT" "$D/seeds.txt" "$D"/refs/r* 2>/dev/null | grep "^SUMMARY," | cut -d, -f2-)
  echo "$N,$X,$Y,$Z,${NP:-nan},${ONP:-nan},${SUM:-nan}" >> "$CSV"
  rm -rf "$D/refs"     # keep the target surface and logs; references are cheap to regrow
  echo "[$i] $N done ($(( $(date +%s) - T0 )) s): ${SUM:-nan}"
done < "$SEEDS"
echo "STABILITY DONE"
