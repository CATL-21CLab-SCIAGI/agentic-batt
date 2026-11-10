#!/bin/sh
# Prepare a pinned upstream checkout; does not run an experiment.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK_DIR="$HERE/work"
CHECKOUT="$WORK_DIR/Battery-Sim-Agent"
PYTHON_BIN=${PYTHON_BIN:-python3.11}
if [ -e "$CHECKOUT" ]; then
    printf 'Checkout already exists: %s\n' "$CHECKOUT" >&2
    exit 1
fi
mkdir -p "$WORK_DIR"
git clone --no-checkout https://github.com/opqrst-chen/Battery-Sim-Agent.git "$CHECKOUT"
git -C "$CHECKOUT" checkout --detach 354f2ddcbb8d14d6dfe9673aac55199976d2d42c
git -C "$CHECKOUT" apply --check "$HERE/reproduce/compatibility.patch"
git -C "$CHECKOUT" apply "$HERE/reproduce/compatibility.patch"
cp "$HERE/reproduce/run_local_experiments.py" "$HERE/reproduce/run_local_baselines.py" "$HERE/reproduce/replay_bo.py" "$CHECKOUT/script/"
cp "$HERE/reproduce/requirements-baselines.txt" "$CHECKOUT/"
"$PYTHON_BIN" -m venv "$WORK_DIR/.venv"
"$WORK_DIR/.venv/bin/python" -m pip install -r "$CHECKOUT/requirements-baselines.txt"
printf 'Prepared %s\nRun agent: sh %s/run_agent.sh\nRun baselines: sh %s/run_baselines.sh\n' "$CHECKOUT" "$HERE" "$HERE"
