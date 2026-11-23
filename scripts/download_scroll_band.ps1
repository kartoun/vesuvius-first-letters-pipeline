# Download what the GrowPatch + ink route needs for one z band of a scroll:
#   CT level 0 (128-voxel chunk rows), the m7 surface prediction (192-voxel chunk rows, plus the
#   pyramid metadata GrowPatch reads), and the whole-scroll normal grids.
# Usage: .\download_scroll_band.ps1 -Sample PHerc0846B -Volume 20250804142305 -Z0 6784 -Z1 7807
#        -Scan "8.640um-1.2m-116keV" for the 8.64 um volumes (default "9.362um-1.2m-113keV")
param([string]$Sample, [string]$Volume, [int]$Z0, [int]$Z1, [string]$Root = "D:\DBBun\data\vesuvius",
      [string]$Scan = "9.362um-1.2m-113keV")
$py = "D:\venvs\d2\Scripts\python.exe"
$f = Join-Path $PSScriptRoot "fetch_s3_prefix.py"
$V = "$Sample/volumes/$Volume-$Scan-masked.zarr"
$S = "$Sample/representations/predictions/surfaces/$Volume-surface-20260413222639-surface-m7-L0-th0.2"
$p = @("$V/.zattrs", "$V/.zgroup", "$V/metadata.json", "$V/0/.zarray", "$V/0/.zattrs")
foreach ($r in [math]::Floor($Z0 / 128)..[math]::Floor($Z1 / 128)) { $p += "$V/0/$r/" }
$p += @("$S.zarr/.zattrs", "$S.zarr/.zgroup", "$S.zarr/metadata.json")
foreach ($l in 0..5) { $p += "$S.zarr/$l/.zarray" }
foreach ($r in [math]::Floor($Z0 / 192)..[math]::Floor($Z1 / 192)) { $p += "$S.zarr/0/$r/" }
$p += "$S.normal-grids/"
& $py -I $f $Root @p
# The prediction's metadata.json is a catalog record (no "type" field); vc_grow_seg_from_seed reads
# metadata.json as volume metadata and fails on it, so keep it under another name.
$m = Join-Path $Root "$S.zarr\metadata.json"
if ((Test-Path $m) -and -not ((Get-Content $m -Raw | ConvertFrom-Json).PSObject.Properties.Name -contains "type")) {
    Move-Item $m (Join-Path $Root "$S.zarr\metadata.catalog.json") -Force
}
