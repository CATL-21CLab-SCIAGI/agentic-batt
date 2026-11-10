#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON=${PYTHON:-python3.11}
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
