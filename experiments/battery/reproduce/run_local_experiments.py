"""Run a small, isolated Ollama pilot using the repository's agent pipelines."""
import argparse
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "single": ("simulated_data_setting_single_new_filtered.yaml", 2, "first_cycle_pipeline", [1]),
    "multi": ("simulated_data_setting_multi_new_filtered.yaml", 4, "first_cycle_pipeline", [1]),
    "degradation": ("simulated_data_setting_SEI.yaml", 0, "degradation_pipeline", [1, 4, 9]),
}
METRICS = ("Q_rmse", "Q_mape", "I_rmse", "I_mape", "V_rmse", "V_mape")


def metrics(row):
    return {key: float(row[key]) for key in METRICS}


def run_case(family, args, output):
    filename, test_id, pipeline, cycles = CASES[family]
    case = output / family
    case.mkdir()
    code = case / "battery_agent"
    shutil.copytree(ROOT / "battery_agent", code,
                    ignore=shutil.ignore_patterns("__pycache__", "llm.yaml"))
    exp_path = code / "configs/exp.yaml"
    exp = yaml.safe_load(exp_path.read_text())
    exp.update(RESULTS_DIR=str(case / "results"), SEARCH_ROUNDS=args.rounds,
               INDEX_TO_SEARCH=cycles, EXP_CYCLES=max(cycles) + 1,
               MAX_LLM_ATTEMPTS=3)
    exp_path.write_text(yaml.safe_dump(exp))
    llm = {
        "BASE_URL": args.base_url, "API_KEY": "ollama", "LLM_MODEL_NAME": args.model,
        "TEMPERATURE": 0.7, "TIMEOUT_SECONDS": 180, "HTTP_MAX_RETRIES": 0,
        "MAX_TOKENS": 4096,
        "PROMPT": {"KNOWLEDGE": "TEXT_KNOWLEDGE", "WITH_FIGURE": False,
                   "WITH_DETAIL_DIFFERENCE": True},
    }
    (code / "configs/llm.yaml").write_text(yaml.safe_dump(llm))
    settings = ROOT / "generate_simulated_data/output" / filename
    cmd = [sys.executable, "-u", str(code / "pipeline.py"),
           "--pipeline_name", pipeline, "--test_id", str(test_id),
           "--yaml_path", str(settings)]
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "matplotlib"),
               PYBAMM_DISABLE_TELEMETRY="true", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    started = time.time()
    info = {"family": family, "test_id": test_id, "cycle_indices_zero_based": cycles,
            "settings": yaml.safe_load(settings.read_text())[test_id],
            "command": cmd, "model": args.model, "rounds_requested": args.rounds,
            "prompt_knowledge": "TEXT_KNOWLEDGE", "status": "running"}
    (case / "status.json").write_text(json.dumps(info, indent=2))
    print(f"Starting {family}, test {test_id}, {args.rounds} evaluation rounds", flush=True)
    with (case / "console.log").open("w") as log:
        try:
            completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=args.timeout)
            info.update(returncode=completed.returncode,
                        status="completed" if completed.returncode == 0 else "failed")
        except subprocess.TimeoutExpired:
            info.update(returncode=None, status="timeout")
    info["elapsed_seconds"] = time.time() - started
    table = case / "results/plot/experiments.csv"
    rows = list(csv.DictReader(table.open())) if table.exists() else []
    rows = [row for row in rows if all(math.isfinite(float(row[k])) for k in METRICS)]
    info["valid_simulation_evaluations"] = len(rows)
    params_file = case / "results/params.jsonl"
    if params_file.exists():
        groups = [json.loads(line) for line in params_file.read_text().splitlines()]
        info["rounds_attempted"] = len(groups)
        info["simulation_evaluations_attempted"] = sum(map(len, groups))
        info["failed_simulation_evaluations"] = info["simulation_evaluations_attempted"] - len(rows)
    if rows:
        best = min(rows, key=lambda row: sum(float(row[k]) for k in ("Q_mape", "I_mape", "V_mape")))
        info.update(initial_metrics=metrics(rows[0]), best_metrics=metrics(best),
                    last_valid_metrics=metrics(rows[-1]), best_evaluation=best["experiment_id"],
                    best_score_definition="Q_mape + I_mape + V_mape; percentages, averaged over selected cycles",
                    best_parameters={k: float(v) for k, v in best.items()
                                     if k not in (*METRICS, "experiment_id", "notes")})
    if info["status"] == "completed":
        if info.get("rounds_attempted") != args.rounds or not rows:
            info["status"] = "incomplete"
        elif info.get("failed_simulation_evaluations", 0):
            info["status"] = "completed_with_simulation_failures"
    messages_file = case / "results/all_messages.jsonl"
    info["llm_responses"] = max(0, len(messages_file.read_text().splitlines()) - 1) if messages_file.exists() else 0
    (case / "status.json").write_text(json.dumps(info, indent=2))
    print(f"{family}: {info['status']}, {len(rows)} valid evaluations, {info['elapsed_seconds']:.1f}s", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4:e2b-qat-google")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--families", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per case")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("At least two rounds are required to evaluate an LLM suggestion")
    output = (args.output or ROOT / "runs" / ("local_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    (output / "local_changes.patch").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    shutil.copy2(__file__, output / "run_local_experiments.py")
    summary = {"upstream_commit": commit, "model": args.model, "output": str(output),
               "scope": "Short local pilot; not the paper's full benchmark or model",
               "cases": []}
    print(f"Output: {output}", flush=True)
    for family in args.families:
        summary["cases"].append(run_case(family, args, output))
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0 if all(case["status"].startswith("completed") for case in summary["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
