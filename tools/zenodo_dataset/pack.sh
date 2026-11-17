#!/bin/bash
# Pack the HAZBRA product into one zip per source x hazard for Zenodo (max 100 files per record).
#
#   bash tools/zenodo_dataset/pack.sh [PRODUCT_DIR] [STAGING_DIR]
#
# Reads PRODUCT_DIR read-only; paths inside the zips start at netcdf/ or municipalities/,
# so unpacking all zips in one folder rebuilds the original tree.
# Excluded on purpose: municipalities/maps (GeoPackage copies of the tables), municipalities/logs.
set -euo pipefail
SRC=${1:-/gws/ssde/j25a/craft/lgcatt/z_GIZ/HAZBRA}
OUT=${2:-/gws/ssde/j25a/craft/lgcatt/z_GIZ/HAZBRA_zenodo/staging}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
cd "$SRC"

pack() {   # pack <zip name> <path> [<path> ...]
    local name=$1; shift
    rm -f "$OUT/$name.tmp.zip"
    zip -r -1 -q "$OUT/$name.tmp.zip" "$@" -x '*.tmp*' '*/__pycache__/*'
    mv "$OUT/$name.tmp.zip" "$OUT/$name.zip"
    echo "$(date +%T) $name.zip  $(du -h "$OUT/$name.zip" | cut -f1)"
}

for hz in Rainfall Drought Heat Compound; do
    pack "netcdf_CLIMBRA_${hz}"           "netcdf/CLIMBRA/$hz"
    pack "netcdf_Observational_${hz}"     "netcdf/Observational/$hz"
    pack "municipalities_CLIMBRA_${hz}"   "municipalities/CLIMBRA/$hz"
    pack "municipalities_Observations_${hz}" "municipalities/Observations/$hz"
done
pack netcdf_outliers            netcdf/outliers
pack municipalities_geometry    municipalities/geometry
pack municipalities_weights     municipalities/weights

# loose documentation files go up unzipped
[ -f municipalities/README.md ] && cp municipalities/README.md "$OUT/README_municipalities.md"
[ -f "$HERE/README_dataset.md" ] && cp "$HERE/README_dataset.md" "$OUT/README.md"

( cd "$OUT" && md5sum *.zip > MD5SUMS.txt )
echo "files: $(ls "$OUT" | wc -l)  total: $(du -sh "$OUT" | cut -f1)"
