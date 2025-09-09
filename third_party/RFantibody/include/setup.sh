#!/bin/bash

set -Eeuo pipefail

SENTINEL_DEFAULT="/home/.rf_setup_done"
SENTINEL="${SENTINEL:-$SENTINEL_DEFAULT}"

echo "[setup.sh] using SENTINEL=$SENTINEL"

currdir=$(cd `dirname $0` && pwd) &&

mkdir -p $currdir/dgl &&
cd $currdir/dgl &&

wget https://data.dgl.ai/wheels/torch-2.3/cu118/dgl-2.4.0%2Bcu118-cp310-cp310-manylinux1_x86_64.whl &&

cd $currdir/USalign &&
make &&

cd /home &&
poetry install &&

echo "Setup successful."

mkdir -p "$(dirname "$SENTINEL")"
: > "$SENTINEL"
echo "[setup.sh] sentinel created."