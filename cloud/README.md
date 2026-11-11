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
  "$BENCHMARK_PYTHON" -I -B "$BENCHMARK_PROJECT/cloud/run_dlc.py" \
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

Before loading models, the controller starts the prepared Python in isolated mode
with inherited Python import overrides and native loader overrides removed.
In particular, a DLC image's `LD_LIBRARY_PATH` must not select libraries from a
different Python/Torch installation. The cleaned environment also reaches the
unchanged simulator worker processes. Ollama retains its GPU loader environment.
The preflight imports the actual numerical packages, checks their filesystem
origins and loaded libtorch paths, executes a CPU tensor operation, tests an
ordinary worker subprocess, and runs synthetic BO through GP acquisition and
pickle/resume. These synthetic checks do not enter the battery results.

The controller also checks the mounted volume, exactly one visible GPU, model
hashes, and both models' structured JSON responses. It requires Ollama to report
nonzero GPU memory for each model.
Ollama binds only to `127.0.0.1:11434`, with one model loaded at a time. Only the
Ollama child process receives a dedicated persistent home for its generated local
key; the calling shell's home is unchanged. Temporary files, caches, and logs use
the run directory.

Inspect `launcher_status.json`, `python_preflight.json`, `python_preflight.log`,
`gpu_smoke.json`, `ollama.log`, `benchmark.log`, and `benchmark/status.json`
in the run directory. A completed launcher status
requires both a zero benchmark exit code and a completed benchmark status. On
failure or timeout it saves a terminal status, stops its child processes, and
returns a nonzero exit code. GPU smoke checks establish placement and basic
inference; only the saved benchmark evaluations establish optimizer results.

## CPU-only BO recovery

`recover_bo.py` runs the original BO method for 100 evaluations and seeds 0, 1,
and 2 in three concurrent CPU processes. It needs no Ollama server or GPU.
Use the same prepared virtual environment, patched upstream checkout, and frozen
scientific runner as the original campaign:

```sh
timeout --signal=TERM --kill-after=90s 5490s \
  "$BENCHMARK_PYTHON" -I -B "$BENCHMARK_PROJECT/cloud/recover_bo.py" \
  --persistent-root "$BENCHMARK_NAMESPACE" \
  --project "$BENCHMARK_PROJECT" \
  --upstream "$BENCHMARK_UPSTREAM" \
  --python "$BENCHMARK_PYTHON" \
  --reference-benchmark "$ORIGINAL_BENCHMARK" \
  --run-dir "$RECOVERY_RUN" --max-seconds 5400
```

`ORIGINAL_BENCHMARK` is the original campaign's `benchmark/` directory, containing
its target, target metadata, provenance, spec, and evaluation records. The recovery
refuses changes to the runner, simulator worker, recorded upstream files, Git
revision, dependency versions, or selected target model/protocol settings. Before
starting BO, it repeats the runtime preflight and verifies the common initial
point and a saved, distinct noninitial point against the original metrics
(absolute and relative tolerance `1e-8`). These replay checks are separate from
optimizer observations and the 100-evaluation budget.

Each seed receives a hash-verified copy of the original `target.npz` and its own
output root, `seed0/`, `seed1/`, or `seed2/`; prior optimizer observations are not
replayed into BO. The immutable original experiment stays separate. Inspect the
recovery's `launcher_status.json`, `reference_replay/validation.json`, per-seed
logs, and `seedN/bo_seedN/status.json`. Each process uses two numerical-library
threads. Add `--resume` only for this same recovery directory; the deadline applies
to each invocation.

The cleanup, environment isolation, reference copy, and numerical replay contracts
can be checked without cloud access:

```sh
python3.11 -B -m unittest discover -s cloud -p 'test_*.py'
```
