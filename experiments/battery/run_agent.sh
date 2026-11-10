#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CHECKOUT="$HERE/work/Battery-Sim-Agent"
cd "$CHECKOUT"
exec "$HERE/work/.venv/bin/python" script/run_local_experiments.py "$@"
