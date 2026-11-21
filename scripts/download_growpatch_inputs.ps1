# Download PHerc0125 GrowPatch inputs: m7 surface prediction (level 0, z chunk rows 52-59 =
# slices 9984-11519, covering the 10100-11400 band) plus its normal grids (whole scroll).
$py = "D:\venvs\d2\Scripts\python.exe"
$f = Join-Path $PSScriptRoot "fetch_s3_prefix.py"
$S = "PHerc0125/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2"
$prefixes = @("$S.zarr/.zattrs", "$S.zarr/.zgroup", "$S.zarr/metadata.json", "$S.zarr/0/.zarray", "$S.zarr/0/.zattrs")
foreach ($r in 52..59) { $prefixes += "$S.zarr/0/$r/" }
$prefixes += "$S.normal-grids/"
& $py -I $f "D:\DBBun\data\vesuvius" @prefixes
