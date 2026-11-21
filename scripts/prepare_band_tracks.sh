#!/usr/bin/env bash
# Band-limited track preparation for PHerc0125: filter DBM -> packed store -> object store.
# Usage: bash prepare_band_tracks.sh Z_LO Z_HI
set -euo pipefail
ZLO="${1:-10100}"; ZHI="${2:-11400}"
VILLA="${VILLA_DIR:-$HOME/vc/villa}"
DATA="${DATA:-/mnt/d/DBBun/data/vesuvius/PHerc0125}"
WORK="${WORK:-$HOME/work/PHerc0125}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="PHerc0125_20250821151825_surface_m7_L0_th0.2_z${ZLO}-${ZHI}.dbm"
PY="$VILLA/spiral-fitting/.venv/bin/python"
mkdir -p "$WORK/tracks"
cd "$WORK"
echo "== filter $(date +%T)"
[ -f "tracks/$NAME" ] || "$PY" "$HERE/filter_tracks_dbm.py" \
  "$DATA/tracks/PHerc0125_20250821151825_surface_m7_L0_th0.2.dbm" "tracks/$NAME" "$ZLO" "$ZHI"
echo "== convert $(date +%T)"
( cd "$VILLA/spiral-fitting" && "$PY" convert_track_store.py --quiet "$WORK/tracks/$NAME" )
echo "== ingest $(date +%T)"
/usr/bin/time -v "$VILLA/build/vc_object_store/bin/vc_object_ingest" --db "$WORK/PHerc0125_z${ZLO}-${ZHI}.vcobj" \
  --author "Uri Kartoun" --volume-uuid 20250821151825 --scale-factor 1 --no-progress \
  --tracks "$WORK/tracks/$NAME.vctracks" 2>&1 | grep -E "Elapsed|Maximum resident|Exit status|rror" || true
echo "== done $(date +%T)"
