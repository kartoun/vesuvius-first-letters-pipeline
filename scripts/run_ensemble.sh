#!/usr/bin/env bash
# Run a checkpoint ensemble (with mirror TTA) on one surface volume and combine the members.
# Usage: bash run_ensemble.sh SURFACE.zarr OUT_DIR NAME [DIRECTION]
#   MEMBERS env var: space-separated checkpoint paths (default: 3 diverse 9 um members).
# Writes OUT_DIR/NAME__<member>.tif, OUT_DIR/NAME_mean.tif, OUT_DIR/NAME_min.tif. PRIVATE outputs.
set -uo pipefail
VOLZ="${1:?surface zarr}"; OUT="${2:?out dir}"; N="${3:?name}"; DIR="${4:-forward}"
M=/mnt/d/DBBun/data/vesuvius/PHerc0125/models
MEMBERS="${MEMBERS:-$M/ink_9um/hybrid_3d2d-seed43/step-020000.pth $M/ink_9um/hybrid_3d2d-seed42/step-075000.pth $M/ink9um-dense/dense9um-all7-step075000.pth}"
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python
export PYTHONPATH=$HOME/vc/villa/vesuvius/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
mkdir -p "$OUT/logs"; TIFS=()
for C in $MEMBERS; do
  TAG=$(basename "$(dirname "$C")")_$(basename "$C" .pth)
  T="$OUT/${N}__$TAG.tif"
  if [ ! -f "$T" ]; then
    ( cd $HOME/vc/villa && $PY -m vesuvius.ink_detection.inference.infer "$VOLZ" "$C" "$T" --overlap 0.5 \
        --blend-mode hann --batch-size 8 --direction "$DIR" --tta-mirror --loader-threads 8 --device cuda ) \
        > "$OUT/logs/${N}__$TAG.log" 2>&1 || { echo "FAILED $N $TAG"; continue; }
  fi
  TIFS+=("$T")
done
$PY $HOME/vc/scripts/ensemble_combine.py "$OUT/$N" "${TIFS[@]}"
