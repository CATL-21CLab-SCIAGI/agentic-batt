# One-GPU matched battery campaign

`run_dlc.py` runs the matched benchmark for Gemma, Qwen, CMA-ES, and BO,
with 100 evaluation slots per method and seeds 0, 1, and 2. It requires a
prepared Linux environment and exits when the campaign finishes. Its internal
deadline, the outer GNU `timeout` command, and the DLC maximum running time
bound a job to 24 hours. A timeout is an incomplete experiment, not a result.

Prepare these inputs on a mounted persistent volume before allocating the GPU:

- This repository and the upstream checkout pinned and patched as described in
  `experiments/battery/matched/`.
- A Python 3.11 virtual environment with the matched `requirements.txt` installed.
  Install CPU PyTorch before the other requirements: BO runs on CPU; Ollama uses
  the GPU through its own runtime. Prepare the environment on the same Linux ABI
  as the DLC image.
- Ollama 0.33.2 with its companion libraries and the exact local model manifests
  and blobs. The launcher checks both expected model weight SHA-256 hashes.
- A GPU image with `nvidia-smi`, `findmnt`, and GNU `timeout`.

Fill the placeholders in `job.template.json` privately. The template requests
one worker and one GPU; all workspace, quota, network, storage, and image settings
are deployment inputs. Keep the private launch JSON outside version control.
The template is configuration only and does not submit a job.

The launch command takes paths as arguments; no credential environment variables
are required. For example, with your own persistent paths assigned to the shell
variables below:

```sh
timeout --signal=TERM --kill-after=90s 85890s \
  "$BENCHMARK_PYTHON" "$BENCHMARK_PROJECT/cloud/run_dlc.py" \
  --persistent-root "$BENCHMARK_NAMESPACE" \
  --project "$BENCHMARK_PROJECT" \
  --upstream "$BENCHMARK_UPSTREAM" \
  --python "$BENCHMARK_PYTHON" \
  --ollama "$BENCHMARK_OLLAMA" \
  --models "$BENCHMARK_MODELS" \
  --run-dir "$BENCHMARK_RUN" \
  --max-seconds 85800
```

All supplied directories and executable launch paths must be under the mounted
persistent namespace. A virtual environment's interpreter may link to the image's
base Python. Use a fresh run directory, or add `--resume` to continue that same
campaign's saved evaluations. Each invocation has its own wall-clock deadline.

Before spending any simulator evaluations, the controller checks the mounted
volume, exactly one visible GPU, model hashes, and both models' structured JSON
responses. It requires Ollama to report nonzero GPU memory for each model.
Ollama binds only to `127.0.0.1:11434`, with one model loaded at a time. Only the
Ollama child process receives a dedicated persistent home for its generated local
key; the calling shell's home is unchanged. Temporary files, caches, and logs use
the run directory.

Inspect `launcher_status.json`, `gpu_smoke.json`, `ollama.log`, `benchmark.log`,
and `benchmark/status.json` in the run directory. A completed launcher status
requires both a zero benchmark exit code and a completed benchmark status. On
failure or timeout it saves a terminal status, stops its child processes, and
returns a nonzero exit code. GPU smoke checks establish placement and basic
inference; only the saved benchmark evaluations establish optimizer results.
