#!/usr/bin/env bash
# Grow local surface patches with VC3D GrowPatch (default optimizer + normal grids) on the
# m7 surface prediction, from seed points (x y z, level-0 voxels). Patches are PRIVATE.
# Usage: bash grow_patches.sh GENERATIONS "x y z" ["x y z" ...]
set -uo pipefail
GEN="${1:?generations}"; shift
R="${R:-/mnt/d/DBBun/data/vesuvius/PHerc0125}"
WORK="${WORK:-$HOME/work/PHerc0125}"
PRED="$R/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr"
GRIDS="$R/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.normal-grids"
OUT="$WORK/growpatch/g$GEN"; mkdir -p "$OUT"
cat > "$OUT/params.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $GEN,
 "thread_limit": 8, "normal_grid_path": "$GRIDS", "voxelsize": 9.362}
EOF
i=0
for S in "$@"; do
  i=$((i+1)); read -r X Y Z <<< "$S"; NAME="seed${i}_x${X}_y${Y}_z${Z}"
  echo "== $NAME"
  # --segment-name writes into --target-dir itself, so give every seed its own target dir.
  mkdir -p "$OUT/$NAME"
  /usr/bin/time -f "   %e s, %M KB peak" vc_grow_seg_from_seed -v "$PRED" -p "$OUT/params.json" \
     -t "$OUT/$NAME" --segment-name "$NAME" --seed "$X" "$Y" "$Z" --rng-seed 1 > "$OUT/$NAME.log" 2>&1
  grep -E "generated surface|rror" "$OUT/$NAME.log" | cut -c1-200; tail -1 "$OUT/$NAME.log"
done
