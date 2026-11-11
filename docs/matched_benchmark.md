# Matched-budget battery campaign

This fresh campaign tests whether additional evaluation budget improves agent fitting on the pilot's single simulated target. It preserves the existing pilot results. It is not a reproduction of the paper's multi-chemistry experiments or its reported 58–97% MAPE reduction.

All 12 runs completed 100 slots each: 1,150 valid simulations and 50 numerical solver failures. Measured trajectories, selected parameters, seed statistics, and coverage are in [the result summary](../experiments/battery/matched/results/summary.json); [the figure](../plots/battery_matched.png) shows independent best-so-far RMSE and MAPE. Both metrics select candidates independently. Seed ranges are descriptive, not confidence intervals. The [campaign manifest](../experiments/battery/matched/results/campaign_manifest.json) connects the original nine runs and separately recovered BO runs.

## Fixed protocol

| Setting | Value |
| --- | --- |
| Target | Upstream single-target configuration, test ID 2; PyBaMM DFN / Chen2020 |
| Scored cycle | Zero-based index 1, 0.2C charge/discharge CCCV |
| Methods | Gemma4 E2B Google QAT Q4_0; Qwen3.8 27B Q4_K_M; CMA-ES; BO |
| Budget | 100 simulation evaluation slots per method and seed |
| Seeds | 0, 1, 2; 1,200 total slots |
| Search space | Same nine physical parameters and bounds for every method |
| First evaluation | Same Chen2020 parameter vector, included in each budget |
| Objective | Minimize capacity RMSE + current RMSE + voltage RMSE |
| Secondary metric | Sum of capacity, current, and voltage MAPEs (%) |
| Failure policy | Failed/timed-out simulations consume a slot and receive objective penalty 1,000,000 |
| Runtime limit | Original campaign: one GPU, 24-hour cap; separate BO recovery: CPU only, 90-minute cap |

Exact bounds, initial values, model settings, and optimizer parameters are in [spec.json](../experiments/battery/matched/spec.json). RMSE sums mix physical units. Each metric selects its best observed candidate independently.

CMA-ES operates in the unit box with population 10 and initial standard deviation 0.2. Its RNG seed is the experiment seed plus one, because CMA treats seed zero as time-dependent. The shared initial evaluation sets the starting reference; 99 further candidates span nine complete generations and one partial generation.

BO uses Ax 0.5 with BoTorch on CPU. It observes the shared initial point, generates 18 Sobol points, then uses GP acquisition for the remaining 81 evaluations. The shared initial point is additional to the 18 generated Sobol points, making 19 initialization observations.

Both agents receive the same objective, bounds, authored upstream physics hints, best valid observation, last eight full diagnostic observations, and all scalar objective values. They do not receive the hidden target parameters. Gemma uses a 4,096-token output cap without thinking; Qwen uses medium reasoning and an 8,192-token cap. Both have a 32,768-token context. The two model configurations therefore have different inference costs, which are recorded separately.

Invalid model proposals receive up to three attempts before any simulation. Requests, responses, rejected attempts, and timings are saved. Exhausted attempts stop that run with a visible failure; they do not create invented simulation results. Interrupted simulator calls consume a reserved slot when resumed. Target-data generation and basic GPU inference checks are outside optimizer budgets and recorded separately.

Both agents return parameter values as regex-constrained plain decimal strings. The adapter checks their exact decimal values against the bounds before converting them to simulator floats; it never guesses exponents or clips proposals. This addresses an observed first-attempt formatting failure: Gemma explained `5.86e-06` in its rationale but emitted `5.86e-0` in a numeric JSON field. That attempt is retained separately and excluded from the fresh comparison. The correction changes the interface representation, not the physical bounds, objective, or evaluation allowance.

## Deployment and reproduction

The [one-GPU launcher](../cloud/README.md) verifies the persistent mount, exact model weight hashes, GPU placement, and successful inference before starting the campaign. Ollama serves only on localhost. Account, quota, network, and storage identifiers stay outside this public repository. Linux uses Python 3.11, PyBaMM 25.6.0, CPU PyTorch 2.8.0, and Ollama 0.33.2; the pilot used Ollama 0.32.13.

Prepare the upstream revision and compatibility patch as in [reproduction.md](reproduction.md), install [matched requirements](../experiments/battery/matched/requirements.txt), and run:

```sh
python experiments/battery/matched/run.py \
  --upstream /path/to/pinned-patched/Battery-Sim-Agent \
  --out /path/to/fresh/campaign \
  --budget 100 --seeds 0 1 2 --methods gemma qwen cma bo
```

Add `--resume` only for the same saved campaign and configuration. Each method/seed saves evaluation JSON, a trajectory, status, inference logs, and optimizer state. Resume state is trusted local pickle data and should not be loaded from untrusted sources. Raw checkpoints, model weights, environments, and private deployment records are excluded from Git.

[Local validation](../experiments/battery/matched/validation.json) covers budget accounting, proposal validation, real initial-point agreement, and BO's transition to GP acquisition. [Linux validation](../experiments/battery/matched/linux_validation.json) confirms the prepared environment's real initial scores and BO acquisition/checkpoint behavior. The [GPU adapter diagnostic](../experiments/battery/matched/gpu_adapter_validation.json) verifies one real proposal and simulation from each corrected agent. [Deployment status](../experiments/battery/matched/launch_status.json) reports the latest verified cloud observation. A running job or a GPU smoke check is not a completed optimizer result.

The comparison reports all three seeds, failed evaluations, and best-so-far RMSE/MAPE against evaluation count. Elapsed and component timings are retained in the summary; the separately recovered BO runs are not an equal-hardware timing comparison. Additional simulator budget alone cannot establish an agent advantage; any conclusion remains limited to this target, these models, and this protocol.

## Native-library failure and BO recovery

The original job completed all 900 Gemma, Qwen, and CMA-ES slots. BO failed before its first simulation: the container's `LD_LIBRARY_PATH` selected `libtorch_python.so` from Python 3.12 while the prepared interpreter was Python 3.11. Package metadata correctly reported the installed CPU Torch version, but did not reveal the incompatible native library. Removing Python import overrides alone did not fix the actual-image diagnostic; removing the native-loader overrides did.

The launcher now gives Python and its workers an isolated environment with those loader overrides removed. Ollama retains its CUDA environment. Before loading models, a preflight imports actual dependencies, checks their locations and loaded Torch libraries, executes a CPU tensor operation, exercises BO GP acquisition, and checks optimizer serialization. This detects the observed failure before an hours-long agent campaign.

The [CPU recovery entrypoint](../cloud/recover_bo.py) preserves the completed runs and original failed status. It uses the same scientific runner, worker, specification, upstream sources, and target bytes. Before BO starts, it re-evaluates the initial point and a saved noninitial point and checks all component metrics for numerical agreement. These diagnostics do not initialize BO or count toward its budget. Each seed gets a separate output directory and 100 fresh evaluation slots, with no optimizer observations reused.

Recovered BO results are combined with the original nine runs through separate provenance records; the original job is never relabeled successful. BO seeds run concurrently on a CPU allocation with the same container image tag. This supports an evaluation-budget comparison, not an equal-hardware or wall-time comparison. Individual wall times and component timings remain available in the records.

The original CMA runs contained 29 numerical failures (8, 5, and 16 by seed). Their logs report `IDA_LSETUP_FAIL` before the upstream `list index out of range` message. They are counted as failed simulator slots, not categorized as physically impossible parameter vectors. See the [record audit](../experiments/battery/matched/results/original_record_audit.json).

The CPU recovery completed all 300 BO slots, and the platform reported success. Its 21 numerical failures (7, 6, and 8 by seed) have the same solver signature. The [recovery audit](../experiments/battery/matched/results/recovery_record_audit.json) verifies every evaluation against its trajectory, bounds, objective, and budget, confirms matching source and target hashes, and records exact agreement for both reference replays.

Regenerate the matched summaries and plots from committed trajectories:

```sh
python scripts/summarize_matched.py --data experiments/battery/matched/results
python scripts/plot_matched.py \
  --summary experiments/battery/matched/results/summary.json \
  --out plots/battery_matched
```
