#!/usr/bin/env bash
# Zero-shot ink inference with scrollprize/ink_9um (hybrid_3d2d-seed42, step 75000) using
# villa's vesuvius.ink_detection.inference.infer (Aug 2026 workflow flags).
# Outputs on eligible scrolls are PRIVATE (stay under $WORK/ink).
# Usage: bash run_ink.sh OUT_DIR SURFACE.zarr [SURFACE.zarr ...]
set -uo pipefail
OUT="${1:?out dir}"; shift
CKPT="${CKPT:-/mnt/d/DBBun/data/vesuvius/PHerc0125/models/ink_9um/hybrid_3d2d-seed42/step-075000.pth}"
mkdir -p "$OUT"
cd ~/vc/villa
export PYTHONPATH=$HOME/vc/villa/vesuvius/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
for Z in "$@"; do
  n=$(basename "$Z" .zarr)
  echo "== $n"
  /usr/bin/time -f "   %e s, %M KB peak RSS" spiral-fitting/.venv/bin/python -m vesuvius.ink_detection.inference.infer \
    "$Z" "$CKPT" "$OUT/$n.tif" --overlap 0.5 --blend-mode hann --batch-size 4 --direction both \
    --loader-threads 8 --device cuda ${ENSEMBLE:+--ensemble-checkpoint $ENSEMBLE} > "$OUT/$n.log" 2>&1
  grep -v -E "SyntaxWarning|math:" "$OUT/$n.log" | tail -3 | cut -c1-200
done
ls -la "$OUT"
