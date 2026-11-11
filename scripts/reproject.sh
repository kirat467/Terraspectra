#!/usr/bin/env bash
# Usage: bash scripts/reproject.sh input.tif output.tif
gdalwarp -t_srs EPSG:4326 -tr 0.0003 0.0003 -r bilinear -dstnodata -9999 "$1" "$2"
gdalinfo "$2" | head -40
