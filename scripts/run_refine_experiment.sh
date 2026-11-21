#!/usr/bin/env bash
# Experiment: does per-pixel layer alignment (refine_surface.py) sharpen PHerc0125 surfaces,
# and does it leave known text intact? Numbers only; all renders and predictions stay PRIVATE.
#
# For each patch name: re-render with 48 layers, refine to 28 sheet-centred layers, run ink
# (s43-20k, forward), score with screen_metrics.py. Also refines the PHerc0211 positive control
# and scores it in 1040-px windows for comparison with screen_calibration.csv.
# Usage: bash run_refine_experiment.sh NAME [NAME ...]
set -uo pipefail
R=/mnt/d/DBBun/data/vesuvius
WORK="${WORK:-$HOME/work/PHerc0125}"; SC=$WORK/screen/g25; OUT=$WORK/refine; mkdir -p "$OUT"/{r48,refined,ink,logs}
VOL=$R/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr
CKPT=$R/PHerc0125/models/ink_9um/hybrid_3d2d-seed43/step-020000.pth
POS=$R/PHerc0211/segments/20261028201525-crop1711-lines123-v9/surface-volumes/9.362um-1.2m-113keV-volume-20250821151803.zarr
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
export PYTHONPATH=$HOME/vc/villa/vesuvius/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
CSV=$OUT/refined.csv
[ -f "$CSV" ] || $PY -c "import sys; sys.path.insert(0,'$S'); from screen_metrics import HEADER; print(HEADER)" > "$CSV"

infer() { ( cd $HOME/vc/villa && $PY -m vesuvius.ink_detection.inference.infer "$1" "$CKPT" "$2" --overlap 0.5 \
  --blend-mode hann --batch-size 8 --direction forward --loader-threads 8 --device cuda ) > "$3" 2>&1; }

# Positive control first: if refinement hurts known text, the PHerc0125 numbers mean nothing.
if [ ! -f "$OUT/ink/POS_refined.tif" ]; then
  $PY $S/refine_surface.py "$POS" "$OUT/refined/POS.zarr" --out-layers 28 --search 6 | tee "$OUT/logs/POS.refine.log"
  infer "$OUT/refined/POS.zarr" "$OUT/ink/POS_refined.tif" "$OUT/logs/POS.ink.log"
  $PY $S/screen_metrics.py "$OUT/refined/POS.zarr" "$OUT/ink/POS_refined.tif" --name POS_refined --windows 1040 \
    2>/dev/null > "$OUT/pos_refined_windows.csv"
fi

for N in "$@"; do
  grep -q "^${N}_ref," "$CSV" && continue
  vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$SC/patches/$N" --num-slices 48 \
    --slice-step 1 --cache-gb 32 --zarr-output "$OUT/r48/$N.zarr" > "$OUT/logs/$N.render.log" 2>&1 || { echo "$N,render_failed"; continue; }
  $PY $S/refine_surface.py "$OUT/r48/$N.zarr" "$OUT/refined/$N.zarr" --out-layers 28 --search 8 | tee "$OUT/logs/$N.refine.log"
  infer "$OUT/refined/$N.zarr" "$OUT/ink/$N.tif" "$OUT/logs/$N.ink.log" || { echo "$N,ink_failed"; continue; }
  $PY $S/screen_metrics.py "$OUT/refined/$N.zarr" "$OUT/ink/$N.tif" --name "${N}_ref" 2>/dev/null | grep "^${N}_ref," >> "$CSV"
done
echo "REFINE DONE"
