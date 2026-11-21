#!/usr/bin/env bash
# Experiment: does matching PHerc0125 intensities to the PHerc0211 known-text segment change the
# ink model's response? Runs on the refined (layer-aligned) screen patches, plus the PHerc0800
# blank control matched the same way, so a normalisation that invents ink is caught.
# Numbers only; all volumes and predictions stay PRIVATE under $WORK/norm.
# Usage: bash run_norm_experiment.sh NAME [NAME ...]
set -uo pipefail
R=/mnt/d/DBBun/data/vesuvius
WORK="${WORK:-$HOME/work/PHerc0125}"; IN=$WORK/refine/refined; OUT=$WORK/norm; mkdir -p "$OUT"/{vol,ink,logs}
CKPT=$R/PHerc0125/models/ink_9um/hybrid_3d2d-seed43/step-020000.pth
REF=$R/PHerc0211/segments/20261028201525-crop1711-lines123-v9/surface-volumes/9.362um-1.2m-113keV-volume-20250821151803.zarr
NEG=$R/PHerc0800/segments/20251028213516-auto_grown_20251028213516907/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
export PYTHONPATH=$HOME/vc/villa/vesuvius/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
CSV=$OUT/norm.csv
[ -f "$CSV" ] || $PY -c "import sys; sys.path.insert(0,'$S'); from screen_metrics import HEADER; print(HEADER)" > "$CSV"

infer() { ( cd $HOME/vc/villa && $PY -m vesuvius.ink_detection.inference.infer "$1" "$CKPT" "$2" --overlap 0.5 \
  --blend-mode hann --batch-size 8 --direction forward --loader-threads 8 --device cuda ) > "$3" 2>&1; }

for M in hist zscore; do
  # Blank control first.
  if [ ! -f "$OUT/neg_${M}_windows.csv" ]; then
    $PY $S/normalize_intensity.py "$REF" "$NEG" "$OUT/vol/NEG_$M.zarr" --mode $M | tee "$OUT/logs/NEG_$M.norm.log"
    infer "$OUT/vol/NEG_$M.zarr" "$OUT/ink/NEG_$M.tif" "$OUT/logs/NEG_$M.ink.log"
    $PY $S/screen_metrics.py "$OUT/vol/NEG_$M.zarr" "$OUT/ink/NEG_$M.tif" --name "NEG_$M" --windows 1040 \
      2>/dev/null | grep "^NEG_" > "$OUT/neg_${M}_windows.csv"
  fi
  for N in "$@"; do
    grep -q "^${N}_$M," "$CSV" && continue
    $PY $S/normalize_intensity.py "$REF" "$IN/$N.zarr" "$OUT/vol/${N}_$M.zarr" --mode $M > "$OUT/logs/${N}_$M.norm.log"
    infer "$OUT/vol/${N}_$M.zarr" "$OUT/ink/${N}_$M.tif" "$OUT/logs/${N}_$M.ink.log" || { echo "$N $M ink_failed"; continue; }
    $PY $S/screen_metrics.py "$OUT/vol/${N}_$M.zarr" "$OUT/ink/${N}_$M.tif" --name "${N}_$M" 2>/dev/null \
      | grep "^${N}_$M," >> "$CSV"
    rm -rf "$OUT/vol/${N}_$M.zarr"
  done
done
echo "NORM DONE"
