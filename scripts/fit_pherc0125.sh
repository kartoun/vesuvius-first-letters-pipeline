#!/usr/bin/env bash
# Spiral fit of a z band of PHerc0125 with minimal constraints (tracks + Lasagna normals),
# following scrollprize.org/tutorial_spiral (2026-09-16) and the Aug 2026 First Letters workflow.
#
# Usage: bash fit_pherc0125.sh SENSE STEPS [Z_BEGIN Z_END]
#   SENSE = CW | ACW   (PHerc0125's catalog z direction is null, so sense is not derivable)
# Env:  DATA (Windows-side downloads, default /mnt/d/DBBun/data/vesuvius/PHerc0125)
#       WORK (Linux-side work dir, default ~/work/PHerc0125)
set -euo pipefail
SENSE="${1:?CW or ACW}"; STEPS="${2:-3000}"; ZB="${3:-10300}"; ZE="${4:-11200}"
VILLA="${VILLA_DIR:-$HOME/vc/villa}"
DATA="${DATA:-/mnt/d/DBBun/data/vesuvius/PHerc0125}"
WORK="${WORK:-$HOME/work/PHerc0125}"
BIN="$VILLA/build/vc_object_store/bin"
VOLUME_ID=20250821151825
# Tracks restricted to z 10100-11400 by prepare_band_tracks.sh (filter -> convert -> ingest).
# Ingesting all 13.8M tracks runs at ~18k tracks/min (one transaction per track), ~13 h.
TBAND="${TBAND:-10100-11400}"
TRACKS="$WORK/tracks/PHerc0125_20250821151825_surface_m7_L0_th0.2_z${TBAND}.dbm.vctracks"
STORE="$WORK/PHerc0125_z${TBAND}.vcobj"
LAS_REL=representations/predictions/lasagna/20250821151825-lasagna-20260419180421
S3=s3://vesuvius-challenge-open-data/PHerc0125

mkdir -p "$WORK"
# 1. One-time ingest of the packed tracks. They were skeletonized from level-0 surface
#    predictions and store level-0 coordinates, so scale factor 1.
if [ ! -f "$STORE" ]; then
  "$BIN/vc_object_ingest" --db "$STORE" --author "Uri Kartoun" \
    --volume-uuid "$VOLUME_ID" --scale-factor 1 --tracks "$TRACKS"
fi

# 2. Input manifest (one per sense).
MAN="$WORK/spiral-inputs-$SENSE.json"
cat > "$MAN" <<EOF
{
  "schema_version": 1,
  "volume_uuid": "$VOLUME_ID",
  "scroll": {
    "name": "PHerc0125",
    "voxel_size_um": 9.362,
    "base_shape_zyx": [20840, 8387, 8387],
    "left_handed_coordinates": false,
    "spiral_outward_sense": "$SENSE"
  },
  "umbilicus": {"path": "$(ls "$DATA"/representations/umbilicus/*.json)", "coordinate_scale": 1},
  "lasagna": {
    "manifest": "$DATA/$LAS_REL/PHerc0125.lasagna.json",
    "original_url": "$S3/$LAS_REL/PHerc0125.lasagna.json"
  },
  "scroll_zarr": "$DATA/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/0"
}
EOF

# 3. Fit. Overrides from the official workflow: no patches, no outer shell, grad_mag spacing.
#    (The Aug 2026 Substack also sets loss_weight_shell_patch_radius, which villa 0bddf2757 rejects.)
OUT="$WORK/fits/${SENSE}_z${ZB}-${ZE}_s${STEPS}"
mkdir -p "$OUT"
export FIT_SPIRAL_OUT_DIR="$OUT"
export FIT_SPIRAL_CONFIG_OVERRIDES=$(cat <<EOF
{"z_begin": $ZB, "z_end": $ZE, "optimizer_num_training_steps": $STEPS,
 "input_disable_patches": true, "input_use_verified_patches": false,
 "input_use_tracks": true, "input_use_fibers": false,
 "input_use_pcl_absolute": false, "input_use_pcl_relative": false,
 "input_use_pcl_same_winding": false, "input_use_pcl_drawn_control_points": false,
 "input_use_patch_overlaps": false, "input_use_winding_inference": false,
 "input_use_outer_shell": false, "loss_weight_shell_outer": 0,
 "dense_spacing_mode": "grad_mag", "loss_weight_dense_spacing": 0}
EOF
)
# WSL2: fit_spiral.py defaults PYTORCH_CUDA_ALLOC_CONF to expandable_segments:True, which
# needs CUDA virtual-memory APIs WSL does not provide ("CUDA driver error: unknown error"
# on the first allocation). An explicit setting takes precedence over its setdefault.
if grep -qi microsoft /proc/version; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
fi
cd "$VILLA/spiral-fitting"
/usr/bin/time -v .venv/bin/python fit_spiral.py --inputs "$MAN" --object-store "$STORE" \
  --object-store-author "Uri Kartoun" --cache "$WORK/cache" 2>&1 | tee "$OUT/fit.log"
