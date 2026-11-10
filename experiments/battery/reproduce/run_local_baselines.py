"""Run short single-case baseline pilots in independent working directories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(name, output, replay_records=None):
    code_dir = "BO" if name == "bayesian_optimization" else "cma_es"
    case = output / name
    shutil.copytree(ROOT / "baseline" / name, case,
                    ignore=shutil.ignore_patterns("__pycache__", "results", "runs", "logs"))
    settings = ROOT / "generate_simulated_data/output/simulated_data_setting_single_new_filtered.yaml"
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", MPLCONFIGDIR=str(case / "matplotlib"),
               PYBAMM_DISABLE_TELEMETRY="true", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    subprocess.run([sys.executable, "scripts/update_configs_sim_sim.py", "--index", "2",
                    "--setting_file", str(settings)], cwd=case, env=env, check=True)
    optimizer_file = case / "configs" / ("BO.yaml" if code_dir == "BO" else "cma_es.yaml")
    config = yaml.safe_load(optimizer_file.read_text())
    if code_dir == "BO":
        config["BO_search_trials"] = 25
        expected = 25
    else:
        config.update(search_trials=5, popsize=6)
        expected = 30
    optimizer_file.write_text(yaml.safe_dump(config))
    cmd = [sys.executable, "-u", f"{code_dir}/pipeline.py", "--config_dir", "configs"]
    if replay_records is not None and code_dir == "BO":
        shutil.copy2(ROOT / "script/replay_bo.py", case / "replay_bo.py")
        cmd = [sys.executable, "-u", "replay_bo.py", "--records", str(replay_records.resolve()),
               "--config_dir", "configs"]
    start = time.time()
    print(f"Starting {name}: {expected} evaluations", flush=True)
    info = {"baseline": name, "test_id": 2, "requested_evaluations": expected,
            "command": cmd, "working_directory": str(case), "status": "running"}
    (case / "status.json").write_text(json.dumps(info, indent=2))
    with (case / "console.log").open("w") as log:
        try:
            proc = subprocess.run(cmd, cwd=case, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=1800)
            info["returncode"] = proc.returncode
        except subprocess.TimeoutExpired:
            info["returncode"] = None
    records = list(case.glob("results/*/*_sim_vs_sim_records.jsonl"))
    rows = [json.loads(line) for path in records for line in path.read_text().splitlines()]
    best_files = list(case.glob("results/*/best_parameters_sim_vs_sim.json"))
    info.update(elapsed_seconds=time.time() - start, evaluations=len(rows),
                status="completed" if len(rows) == expected and best_files and info["returncode"] == 0 else "incomplete")
    info["failed_simulation_evaluations"] = sum(row["total_loss"] >= 1e6 for row in rows)
    if best_files:
        info["best_result"] = json.loads(best_files[0].read_text())
    validation = case / "replay_validation.json"
    if validation.exists():
        info["replay"] = json.loads(validation.read_text())
    (case / "status.json").write_text(json.dumps(info, indent=2))
    print(f"{name}: {info['status']}, {len(rows)} evaluations", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baselines", nargs="+", choices=["cma_es_single", "bayesian_optimization"],
                        default=["cma_es_single", "bayesian_optimization"])
    parser.add_argument("--replay-records", type=Path)
    args = parser.parse_args()
    output = (args.output or ROOT / "runs" / ("baselines_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "run_local_baselines.py")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: run(name, output, args.replay_records), args.baselines))
    (output / "summary.json").write_text(json.dumps(results, indent=2))
    return 0 if all(item["status"] == "completed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
