# Reproducing the local pilots

Run the commands below from the repository root. Python 3.11 and Git are required for experiment setup. Ollama must serve the selected locally installed model at `http://127.0.0.1:11434`; setup scripts do not download model weights or configure paid services. Package installation and upstream cloning require internet access.

## Inspect existing measurements

```sh
python3.11 scripts/verify_results.py
```

This checks the recorded metrics, all 2,340 OPAL losses per run, matched controls, decision counts, source hashes, and battery evaluation trajectories without loading a model or dataset. The archived result summaries contain runtime, parameters, objectives, and limitations. The plots are reproducible from committed JSON/CSV records using the commands in the root README.

To install the plotting environment and regenerate all five figures:

```sh
python3.11 -m venv .venv-plots
.venv-plots/bin/python -m pip install -r requirements-plots.txt
.venv-plots/bin/python scripts/plot_results.py
```

The combined battery figure reports independent best-so-far sums of MAPE and RMSE on the single-parameter target. A method's best candidate may differ between panels. Each trajectory ends at its recorded budget; Gemma is not extrapolated beyond five evaluations. Initialization and feedback differ across methods: CMA-ES uses populations of six, BO starts with 18 Sobol evaluations, and Gemma receives physics guidance and sequential diagnostic feedback.

## Battery simulation

```sh
sh experiments/battery/setup.sh
sh experiments/battery/run_agent.sh
sh experiments/battery/run_baselines.sh
```

Setup creates an isolated `experiments/battery/work/` directory, checks out upstream revision `354f2ddcbb8d14d6dfe9673aac55199976d2d42c`, applies the compatibility patch, and installs the simulation and baseline dependencies. It refuses to overwrite an existing checkout. Set `PYTHON_BIN` to select another Python 3.11 executable.

`run_agent.sh` runs the single-parameter, multi-parameter, and degradation cases with `gemma4:e2b-qat-google`, five simulation evaluations each. The first evaluation uses the initial parameters; the remaining four follow model proposals. The first-cycle cases score cycle index 1; degradation scores indices 1, 4, and 9, using zero-based indexing. Model configurations are created locally with a dummy Ollama API key.

`run_baselines.sh` starts fresh CMA-ES and Bayesian-optimization runs with budgets of 30 and 25 evaluations. Fresh BO evaluates all 25 suggestions; its timing will therefore differ from the archived continuation, which reused 18 completed evaluations. To exercise the verified replay path instead:

```sh
sh experiments/battery/run_baselines.sh --baselines bayesian_optimization \
  --replay-records "$PWD/experiments/battery/results/baselines/bayesian_optimization/replayed_initial18.jsonl"
```

Replay checks the parameter names and values before reusing each saved result. All new battery outputs remain under the ignored `work/` directory. Do not compare elapsed time for a replay continuation with a fully fresh run as though the compute budgets were equal.

The compatibility patch fixes `test_id=0` handling, degradation prompt selection, returned per-cycle metrics, and bounded model-response handling. The baseline environment pins pandas 2.2.3 for Ax 0.5 compatibility. Upstream physical equations and the fixed classical optimizer implementations are retained. Source provenance is in [the manifest](../experiments/battery/provenance/manifest.json).

## OPAL with local Ollama

```sh
sh experiments/opal/setup.sh
```

This installs OPAL from revision `8b16759a3e9930916ed582e28068267859f935f8` with the pinned experiment dependencies. Set `PYTHON` to select another Python 3.11 executable. Each run downloads MNIST if it is not already cached locally.

For Qwen, the recorded runtime alias was created from an existing local alias, `qwen3.8:27b-harness-fixed`, with the same Q4_K_M weights:

```sh
ollama create qwen3.8:27b-opal -f experiments/opal/Qwen38.Modelfile
sh experiments/opal/run_all.sh qwen
```

That source alias is a local name, not a guaranteed public model tag. If your matching weights already have another alias, skip alias creation and use:

```sh
OLLAMA_MODEL=your-local-qwen-alias sh experiments/opal/run_all.sh qwen
```

The archived weight SHA-256 and Ollama version are in [OPAL provenance](../experiments/opal/provenance.json). Using different weights, quantization, inference settings, or software versions is a new experiment. Qwen uses medium reasoning, temperature 1.0, and an 8,192-token generation cap. The interrupted maximum-reasoning attempt is documented separately in [other attempts](../experiments/opal/results/other_attempts.json).

For the small-model comparison:

```sh
sh experiments/opal/run_all.sh gemma
```

Each command runs fresh plain and keep controls plus the selected agent, evaluates all saved checkpoints in separate processes, and generates a result summary and native OPAL report. New outputs are placed under ignored `experiments/opal/local-runs/`. Timings in the committed results cover training and agent waiting, excluding dataset loading and final test evaluation.

The applied Qwen intervention preserved the AdamW algorithm: betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, moments, and step counters. Its fixed schedule for subsequent updates was:

$$
\eta_t = \frac{0.001}{2}\left[1+\cos\left(\pi\frac{t-1200}{2340-1200}\right)\right].
$$

The adapter checks generated Python before OPAL applies it. The first scheduled Qwen proposal failed those checks; one later proposal was repaired and applied. The gate is an execution restriction, not a complete sandbox. The published code preserves the restrictions used in the recorded run.

## Validation and interpretation

During packaging, the relocated OPAL plain runner reproduced every original loss and the final parameter hash. The battery compatibility patch was checked against the pinned source and archived case snapshots. New agent inference was not run during packaging. Generated plots were visually inspected.

The MNIST result is one seed, with 13 additional correct test predictions and slightly higher test cross entropy. Battery cases are simulated targets, not validation on physical cells. A controlled next study needs matched objectives and evaluation budgets for battery fitting, and repeated seeds with quality-versus-compute measurements for OPAL.

The overhead figure assumes a fixed total wait of 917.96 seconds and computes `wait / (training_time + wait)`. It excludes new distributed coordination costs and per-step overhead from changed optimizer code. It is a projection, not evidence from distributed training.
