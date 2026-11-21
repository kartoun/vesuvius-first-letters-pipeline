#!/usr/bin/env bash
# Render selected winding meshes from a spiral fit into 28-layer surface volumes with the
# official vc_render_tifxyz (settings from the Aug 2026 First Letters workflow).
# Renders of eligible scrolls are PRIVATE: they stay under $WORK/renders (never published).
# Usage: bash render_windings.sh FIT_DIR WINDING [WINDING ...]   e.g. w060_spliced
set -euo pipefail
FIT="${1:?fit dir}"; shift
WORK="${WORK:-$HOME/work/PHerc0125}"
VOL="${VOL:-/mnt/d/DBBun/data/vesuvius/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr}"
OUT="$WORK/renders/$(basename "$(dirname "$FIT")")"
mkdir -p "$OUT"
for W in "$@"; do
  SEG=$(find "$FIT" -maxdepth 3 -type d -name "$W" | head -1)
  [ -n "$SEG" ] || { echo "no mesh $W"; continue; }
  echo "== $W  ($SEG)"
  /usr/bin/time -f "%e s, %M KB peak" vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 \
    --segmentation "$SEG" --num-slices 28 --slice-step 1 --cache-gb 32 \
    --zarr-output "$OUT/$W.zarr" 2>&1 | tail -5
done
ls -la "$OUT"
