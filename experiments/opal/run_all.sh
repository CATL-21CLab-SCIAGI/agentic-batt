#!/bin/sh
# Requires setup.sh and a running local Ollama with the selected model installed.
# Usage: sh run_all.sh [qwen|gemma] [new-output-directory]
# Override OLLAMA_MODEL to use an existing alias with the intended model weights.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
AGENT=${1:-qwen}
RUN_ROOT=${2:-local-runs/${AGENT}_mnist_$(date +%Y%m%d_%H%M%S)}
case "$AGENT" in
    qwen) MODEL=${OLLAMA_MODEL:-qwen3.8:27b-opal} ;;
    gemma) MODEL=${OLLAMA_MODEL:-gemma4:e2b-qat-google} ;;
    *) printf 'Expected qwen or gemma.\n' >&2; exit 1 ;;
esac
if [ -e "$RUN_ROOT" ]; then
    printf 'Refusing to overwrite existing run: %s\n' "$RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT/source"
cp run_mnist.py ollama_backend.py summarize.py requirements.txt "$RUN_ROOT/source/"
.venv/bin/python run_mnist.py download --out "$RUN_ROOT/setup"
.venv/bin/python run_mnist.py plain --out "$RUN_ROOT/plain" > "$RUN_ROOT/plain.log" 2>&1
.venv/bin/python run_mnist.py keep --out "$RUN_ROOT/keep3" > "$RUN_ROOT/keep.log" 2>&1
if [ "$AGENT" = qwen ]; then
    .venv/bin/python run_mnist.py ollama --model "$MODEL" \
        --reasoning-level medium --max-output-tokens 8192 \
        --model-timeout 900 --temperature 1.0 \
        --out "$RUN_ROOT/agent" > "$RUN_ROOT/agent.log" 2>&1
else
    .venv/bin/python run_mnist.py ollama --model "$MODEL" \
        --out "$RUN_ROOT/agent" > "$RUN_ROOT/agent.log" 2>&1
fi
for mode in plain keep3 agent; do
    .venv/bin/python run_mnist.py evaluate --out "$RUN_ROOT/$mode"
done
.venv/bin/python summarize.py "$RUN_ROOT" --agent-name agent
.venv/bin/opal report "$RUN_ROOT/agent/opal"
