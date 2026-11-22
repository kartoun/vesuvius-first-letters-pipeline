#!/usr/bin/env bash
# Automatic region scan with a gate before any human review. For one seed in a stable region:
#   grow a bounded large surface (small + resume), check sheet consistency, orient it, render and
#   align, run the 3-member ensemble on the face toward the scroll centre (default window and +-3
#   layers), score every grid cell against known-text / blank calibration (cell_scan.py), drop
#   cells whose local orientation disagrees (pick_side.py), and append one summary row.
#   Review images are exported (PRIVATE) only when the green-cell rate passes the gate.
# Usage: bash scan_region.sh SAMPLE VOLID VOLNAME VOXEL ZMIN ZMAX NAME X Y Z GEN OUT_ROOT PRIV_ROOT CAL_SPEC.json [GATE]
set -uo pipefail
SAMPLE="$1"; VOLID="$2"; VOLNAME="$3"; VOXEL="$4"; ZMIN="$5"; ZMAX="$6"; NM="$7"; X="$8"; Y="$9"; Z="${10}"
GEN="${11}"; ROOT="${12}"; PRIV="${13}"; CAL="${14}"; GATE="${15:-0.30}"
R=/mnt/d/DBBun/data/vesuvius/$SAMPLE
PRED=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.zarr
GRIDS=$R/representations/predictions/surfaces/$VOLID-surface-20260413222639-surface-m7-L0-th0.2.normal-grids
VOL=$R/volumes/$VOLNAME; S3V=$SAMPLE/volumes/$VOLNAME/
PY=$HOME/vc/villa/spiral-fitting/.venv/bin/python; S=$HOME/vc/scripts
M=/mnt/d/DBBun/data/vesuvius/PHerc0125/models
MEM="$M/ink_9um/hybrid_3d2d-seed43/step-020000.pth $M/ink_9um/hybrid_3d2d-seed42/step-075000.pth $M/ink9um-dense/dense9um-all7-step075000.pth"
export PYTHONPATH=$HOME/vc/villa/vesuvius/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
O=$ROOT/$NM; mkdir -p "$O"/{logs,ink}; T0=$(date +%s)
p() { cat > "$O/params_$1.json" <<EOF
{"mode": "seed", "step_size": 20, "min_area_cm": 0, "generations": $2, "thread_limit": 12,
 "normal_grid_path": "$GRIDS", "voxelsize": $VOXEL, "z_min": $ZMIN, "z_max": $ZMAX}
EOF
}
p small 20; p large "$GEN"; mkdir -p "$O/small" "$O/large"
vc_grow_seg_from_seed -v "$PRED" -p "$O/params_small.json" -t "$O/small" --segment-name small --seed "$X" "$Y" "$Z" --rng-seed 1 > "$O/logs/small.log" 2>&1 \
 && vc_grow_seg_from_seed -v "$PRED" -p "$O/params_large.json" -t "$O/large" --segment-name large --resume "$O/small" --resume-opt skip --rng-seed 1 > "$O/logs/large.log" 2>&1 \
 || { echo "$NM,grow_failed" >> "$ROOT/summary.csv"; exit 1; }
AREA=$(grep -o "([0-9.]* cm^2)" "$O/logs/large.log" | tail -1 | tr -d '()cm^2 ')
read -r SIDE TOWARD MED <<< "$($PY $S/orient_surface.py "$O/large" "$S3V" 2>/dev/null)"
DIV=$(bash $S/run_sheet_check.sh "$SAMPLE" "$VOLID" "$O/large" "$O/check" 20 20 12 > /dev/null 2>&1; \
      $PY $S/seed_divergence.py "$O/large" "$O/check/seeds.txt" "$O"/check/refs/r* 2>/dev/null | grep SUMMARY | cut -d, -f6-8)
rm -rf "$O/check/refs"
vc_render_tifxyz --volume "$VOL" --group-idx 0 --scale 1 --segmentation "$O/large" --num-slices 48 --slice-step 1 \
  --cache-gb 24 --zarr-output "$O/r48.zarr" > "$O/logs/render.log" 2>&1 || { echo "$NM,render_failed" >> "$ROOT/summary.csv"; exit 1; }
$PY $S/refine_surface.py "$O/r48.zarr" "$O/render.zarr" --out-layers 28 --search 8 > "$O/logs/refine.log" 2>&1; rm -rf "$O/r48.zarr"
DIR=$([ "$SIDE" = "rev" ] && echo reverse || echo forward)
for w in "d 6 23" "m3 3 20" "p3 9 26"; do
  read -r tag a b <<< "$w"; T=()
  for CK in $MEM; do
    o=$O/ink/${tag}__$(basename $(dirname $CK))_$(basename $CK .pth).tif
    ( cd ~/vc/villa && $PY -m vesuvius.ink_detection.inference.infer "$O/render.zarr" $CK $o --overlap 0.5 --blend-mode hann \
        --batch-size 8 --direction $DIR --tta-mirror --loader-threads 8 --device cuda --layer-start $a --layer-end $b ) > "$O/logs/ink_$tag.log" 2>&1
    T+=($o)
  done
  $PY $S/ensemble_combine.py "$O/ink/$tag" "${T[@]}" > /dev/null
done
python3 - <<EOF
import json
s = json.load(open("$CAL")); O = "$O/ink"
s["sets"]["cand"] = {"render": "$O/render.zarr", "mean": O + "/d_mean.tif", "shallow": O + "/m3_mean.tif", "deep": O + "/p3_mean.tif",
                     "members": [O + "/d__hybrid_3d2d-seed43_step-020000.tif", O + "/d__hybrid_3d2d-seed42_step-075000.tif",
                                 O + "/d__ink9um-dense_dense9um-all7-step075000.tif"]}
json.dump(s, open("$O/spec.json", "w"))
EOF
$PY $S/cell_scan.py "$O/spec.json" "$O" "$NM" 400 > "$O/logs/cellscan.log" 2>&1
# Keep only cells whose local face matches the side that was read: the other side's CSV is all grey.
python3 -c "
import csv; r=list(csv.DictReader(open('$O/${NM}_cells.csv')))
for x in r: x['class']='grey'
w=csv.DictWriter(open('$O/${NM}_other.csv','w',newline=''),fieldnames=list(r[0])); w.writeheader(); w.writerows(r)"
if [ "$SIDE" = "fwd" ]; then F="$O/${NM}_cells.csv"; Rv="$O/${NM}_other.csv"; else F="$O/${NM}_other.csv"; Rv="$O/${NM}_cells.csv"; fi
SID=$($PY $S/pick_side.py "$O/large" "$S3V" "$F" "$Rv" "$O/${NM}_sided.csv" 400 2>/dev/null | head -1)
USABLE=$(echo "$SID" | grep -o "usable [0-9]*" | cut -d' ' -f2); GREEN=$(echo "$SID" | grep -o "green [0-9]*" | cut -d' ' -f2)
RATE=$(python3 -c "print(f'{${GREEN:-0}/max(${USABLE:-0},1):.2f}')")
PASS=$(python3 -c "print('PASS' if ${USABLE:-0} >= 15 and ${RATE} >= $GATE else 'no')")
GREENS=$(python3 -c "
import csv; print(' '.join(r['cell'] for r in csv.DictReader(open('$O/${NM}_sided.csv')) if r['class']=='green'))")
[ -f "$ROOT/summary.csv" ] || echo "region,x,y,z,area_cm2,side,toward,div_100_200,div_200_300,div_300_450,usable,green,green_rate,gate,green_cells,minutes" > "$ROOT/summary.csv"
echo "$NM,$X,$Y,$Z,$AREA,$SIDE,$TOWARD,$(echo $DIV | tr ',' ',' | cut -d, -f1-3),$USABLE,$GREEN,$RATE,$PASS,$GREENS,$(( ($(date +%s) - T0) / 60 ))" >> "$ROOT/summary.csv"
if [ "$PASS" = "PASS" ]; then
  mkdir -p "$PRIV/$NM"
  VOXEL_UM=$VOXEL $PY $S/export_overview.py "$PRIV/$NM" "$NM" "$O/render.zarr" "$O/ink/d_mean.tif" "$HOME/work/ensemble_cal/POS_mean.tif" > /dev/null 2>&1
  $PY $S/grid_views.py "$PRIV/$NM" "$NM" "$O/render.zarr" 400 2 mean="$O/ink/d_mean.tif" > /dev/null 2>&1
  cp "$O/${NM}_cell_scores.png" "$O/${NM}_sided.csv" "$PRIV/$NM/" 2>/dev/null
fi
echo "$NM: side $SIDE ($TOWARD), usable $USABLE, green $GREEN ($RATE) -> $PASS, $(( ($(date +%s) - T0) / 60 )) min"
