"""Validate and summarize measured matched-budget trajectories, without filling gaps.

Input: DATA/{gemma,qwen,cma,bo}_seed{0,1,2}/trajectory.jsonl, with optional
status.json, timing.json, and config.json beside each trajectory. A missing run
remains missing; failed simulator calls consume their recorded evaluation slots.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("gemma", "qwen", "cma", "bo")
LABELS = {"gemma": "Gemma4 E2B", "qwen": "Qwen3.8 27B", "cma": "CMA-ES", "bo": "BO"}
METRICS = ("sum_rmse", "sum_mape")
SCIENTIFIC_CONFIG_FIELDS = ("spec", "runner_sha256", "worker_sha256", "target_sha256",
                            "simulation_timeout", "mock")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number, got {value!r}")
    if value < 0:
        raise ValueError(f"{label}: negative value {value}")
    return float(value)


def timestamp(value):
    if value is None:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp needs a time zone: {value}")
    return parsed


def stats(values):
    values = [v for v in values if v is not None]
    return {"seed_count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def selected(record, scores):
    return {"evaluation": record["evaluation"], "parameters": record["parameters"],
            "metrics": record["metrics"], **scores}


def validate_scientific_configs(data, methods, seeds):
    """Compare available run configs; absence is reported, never treated as agreement."""
    available, missing, reference = [], [], None
    for method in methods:
        for seed in seeds:
            name = f"{method}_seed{seed}"
            path = data / name / "config.json"
            if not path.exists():
                missing.append(name)
                continue
            config = read_json(path)
            absent = [field for field in SCIENTIFIC_CONFIG_FIELDS if config.get(field) is None]
            if absent:
                raise ValueError(f"{name}: config lacks scientific fields: {', '.join(absent)}")
            if not isinstance(config["spec"], dict) or not config["spec"]:
                raise ValueError(f"{name}: scientific spec must be a nonempty object")
            if not isinstance(config["mock"], bool):
                raise ValueError(f"{name}: mock mode must be an explicit boolean")
            scientific = {field: config[field] for field in SCIENTIFIC_CONFIG_FIELDS}
            if reference is None:
                reference = scientific
            else:
                changed = [field for field in SCIENTIFIC_CONFIG_FIELDS if scientific[field] != reference[field]]
                if changed:
                    raise ValueError(f"{name}: scientific config differs from {available[0]}: {', '.join(changed)}")
            available.append(name)
    return {"expected_config_count": len(methods) * len(seeds), "available_config_count": len(available),
            "available_config_runs": available, "missing_config_runs": missing,
            "checked_fields": list(SCIENTIFIC_CONFIG_FIELDS),
            "all_available_configs_match": True if available else None,
            "scope": "Only available configs are checked; missing configs do not establish comparability."}


def summarize_run(data, method, seed, budget):
    directory = data / f"{method}_seed{seed}"
    source = directory / "trajectory.jsonl"
    status = read_json(directory / "status.json")
    config = read_json(directory / "config.json")
    timing = read_json(directory / "timing.json")
    for metadata in (status, config):
        for key, expected in (("method", method), ("seed", seed), ("budget", budget)):
            if key in metadata and metadata[key] != expected:
                raise ValueError(f"{directory.name}: {key} disagrees with expected {expected!r}")
    if config.get("mock"):
        raise ValueError(f"{directory.name}: synthetic runs are not battery evidence")
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()] if source.exists() else []
    if len(records) > budget or [r["evaluation"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError(f"{directory.name}: slots must be unique and contiguous from 1, within budget")
    if any(isinstance(r["evaluation"], bool) or not isinstance(r["evaluation"], int) for r in records):
        raise ValueError(f"{directory.name}: evaluation numbers must be integers")
    start_text = timing.get("started_utc", status.get("started_utc"))
    if timing.get("started_utc") and status.get("started_utc") and timing["started_utc"] != status["started_utc"]:
        raise ValueError(f"{directory.name}: conflicting start timestamps")
    started, updated = timestamp(start_text), timestamp(status.get("updated_utc"))
    best, curve = {metric: None for metric in METRICS}, []
    simulation, proposal, valid_count, interrupted = 0., 0., 0, 0
    simulation_count = proposal_count = 0
    last_utc = None
    warnings = []
    for record in records:
        index = record["evaluation"]
        if record["status"] == "valid":
            valid_count += 1
            scores = {}
            for metric in METRICS:
                suffix = metric.removeprefix("sum_")
                values = [record.get("metrics", {}).get(f"{c}_{suffix}") for c in "QIV"]
                if all(value is not None for value in values):
                    scores[metric] = sum(finite_number(value, f"{directory.name}/{index}/{metric}") for value in values)
                elif metric == "sum_rmse":
                    raise ValueError(f"{directory.name}/{index}: valid record lacks RMSE components")
                else:
                    scores[metric] = None
            objective = finite_number(record["objective"], f"{directory.name}/{index}/objective")
            if not math.isclose(objective, scores["sum_rmse"], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{directory.name}/{index}: objective differs from summed RMSE")
            for metric, value in scores.items():
                if value is not None and (best[metric] is None or value < best[metric][metric]):
                    best[metric] = selected(record, scores)
        interrupted += record["status"] == "interrupted_simulation"
        if "simulation_seconds" in record:
            simulation += finite_number(record["simulation_seconds"], f"{directory.name}/{index}/simulation_seconds")
            simulation_count += 1
        if "proposal_seconds" in record.get("proposal", {}):
            proposal += finite_number(record["proposal"]["proposal_seconds"], f"{directory.name}/{index}/proposal_seconds")
            proposal_count += 1
        elif index == 1:  # The shared initial vector needs no proposal generation.
            proposal_count += 1
        utc = timestamp(record.get("utc"))
        if utc is not None and last_utc is not None and utc < last_utc:
            raise ValueError(f"{directory.name}/{index}: evaluation timestamps are not monotonic")
        if utc is not None:
            last_utc = utc
        elapsed = (utc - started).total_seconds() if utc is not None and started is not None else None
        if elapsed is not None and elapsed < 0:
            raise ValueError(f"{directory.name}/{index}: evaluation predates run start")
        curve.append({"evaluation": index, "status": record["status"],
                      **{f"best_{metric}": best[metric][metric] if best[metric] else None for metric in METRICS},
                      "elapsed_wall_seconds_since_start": elapsed,
                      "recorded_simulation_seconds_cumulative": simulation,
                      "recorded_proposal_seconds_cumulative": proposal})
    for key, observed in (("evaluations", len(records)), ("valid_evaluations", valid_count),
                          ("failed_evaluations", len(records) - valid_count)):
        if key in status and status[key] != observed:
            raise ValueError(f"{directory.name}: status {key}={status[key]} disagrees with trajectory {observed}")
    elapsed_at_last = curve[-1]["elapsed_wall_seconds_since_start"] if curve else None
    elapsed_at_status = (updated - started).total_seconds() if updated and started else None
    reported_elapsed = status.get("elapsed_wall_seconds_including_resume_gaps")
    elapsed_verified = None
    if reported_elapsed is not None and elapsed_at_status is not None:
        elapsed_verified = math.isclose(finite_number(reported_elapsed, "reported elapsed"), elapsed_at_status, abs_tol=1.)
        if not elapsed_verified:
            raise ValueError(f"{directory.name}: reported wall time disagrees with UTC timestamps")
    if updated and last_utc and updated < last_utc:
        raise ValueError(f"{directory.name}: status update predates its last evaluation")
    active_fits_elapsed = simulation + proposal <= elapsed_at_last + 1. if elapsed_at_last is not None else None
    if active_fits_elapsed is False:
        raise ValueError(f"{directory.name}: recorded component time exceeds elapsed wall time")
    if simulation_count != len(records) or proposal_count != len(records):
        warnings.append("Some evaluation timing fields are absent; component sums cover recorded fields only.")
    if interrupted:
        warnings.append("Interrupted simulator slots count against budget; their lost work may be absent from component timing.")
    if any(r["status"] == "valid" and not all(f"{c}_mape" in r.get("metrics", {}) for c in "QIV") for r in records):
        warnings.append("Some valid evaluations lack MAPE; its minimum includes only available valid MAPE observations.")
    return {"method": method, "seed": seed, "budget": budget,
            "reported_status": status.get("status"),
            "coverage": {"recorded_slots": len(records), "valid_evaluations": valid_count,
                         "failed_evaluations": len(records) - valid_count, "interrupted_slots": interrupted,
                         "budget_complete": len(records) == budget, "unrecorded_slots": budget - len(records)},
            "source": {"trajectory": source.relative_to(data).as_posix(),
                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None},
            "initial_parameters": records[0]["parameters"] if records else None,
            "best_by_metric": best, "curve": curve,
            "milestones": {str(n): {metric: curve[n - 1][f"best_{metric}"] for metric in METRICS}
                           for n in sorted({5, 30, budget}) if n <= len(curve)},
            "last_evaluation": {key: records[-1][key] for key in ("evaluation", "status", "objective", "metrics")
                                if key in records[-1]} if records else None,
            "inference_attempts": status.get("inference_attempts"),
            "inference_rejected_or_failed_attempts": status.get("inference_rejected_or_failed_attempts"),
            "timing": {"started_utc": start_text, "last_evaluation_utc": last_utc.isoformat() if last_utc else None,
                       "status_updated_utc": status.get("updated_utc"),
                       "elapsed_wall_seconds_at_last_evaluation": elapsed_at_last,
                       "elapsed_wall_seconds_at_status_update": elapsed_at_status,
                       "reported_elapsed_wall_seconds": reported_elapsed,
                       "reported_elapsed_matches_utc": elapsed_verified,
                       "recorded_simulation_seconds": simulation, "recorded_proposal_seconds": proposal,
                       "recorded_active_seconds": simulation + proposal,
                       "component_time_fits_elapsed": active_fits_elapsed,
                       "simulation_timing_records": simulation_count, "proposal_timing_records": proposal_count},
            "warnings": warnings}


def summarize(data, methods=METHODS, seeds=(0, 1, 2), budget=100):
    if not data.is_dir():
        raise ValueError(f"Input directory does not exist: {data}")
    if not methods or len(set(methods)) != len(methods) or not seeds or len(set(seeds)) != len(seeds) or budget < 1:
        raise ValueError("Use nonempty methods/seeds, unique seeds and a positive budget")
    provenance = read_json(data / "provenance.json")
    if provenance.get("mock"):
        raise ValueError("Synthetic provenance is not battery evidence")
    config_validation = validate_scientific_configs(data, methods, seeds)
    groups = {}
    for method in methods:
        runs = [summarize_run(data, method, seed, budget) for seed in seeds]
        completed = [r for r in runs if r["coverage"]["budget_complete"]]
        aggregate = []
        for index in range(budget):
            available = [run["curve"][index] for run in runs if len(run["curve"]) > index]
            point = {"evaluation": index + 1, "recorded_seed_count": len(available)}
            for metric in METRICS:
                values = [r[f"best_{metric}"] for r in available]
                full = len(values) == len(seeds) and all(v is not None for v in values)
                point[metric] = stats(values) if full else None
            aggregate.append(point)
        groups[method] = {"label": LABELS[method], "runs": runs,
                          "completed_seed_count": len(completed),
                          "recorded_seed_count": sum(bool(run["curve"]) for run in runs),
                          "recorded_slots": sum(r["coverage"]["recorded_slots"] for r in runs),
                          "failed_evaluations": sum(r["coverage"]["failed_evaluations"] for r in runs),
                          "at_budget": {metric: stats([r["best_by_metric"][metric][metric] if r["best_by_metric"][metric] else None
                                                       for r in completed]) for metric in METRICS},
                          "at_budget_timing": {key: stats([r["timing"][key] for r in completed]) for key in
                                               ("elapsed_wall_seconds_at_last_evaluation", "recorded_active_seconds",
                                                "recorded_simulation_seconds", "recorded_proposal_seconds")},
                          "latest_observed": {metric: stats([r["best_by_metric"][metric][metric] if r["best_by_metric"][metric] else None
                                                             for r in runs]) for metric in METRICS},
                          "milestones": {str(n): {metric: stats([r["curve"][n - 1][f"best_{metric}"] for r in runs if len(r["curve"]) >= n])
                                                   for metric in METRICS} for n in sorted({5, 30, budget}) if n <= budget},
                          "aggregate_curve": aggregate}
    all_runs = [run for group in groups.values() for run in group["runs"]]
    initial = [r["curve"][0] for r in all_runs if r["curve"]]
    initial_scores = [p["best_sum_rmse"] for p in initial]
    common_initial_score = bool(initial) and all(v is not None and math.isclose(v, initial_scores[0], rel_tol=1e-9, abs_tol=1e-12)
                                               for v in initial_scores)
    initial_parameters = [r["initial_parameters"] for r in all_runs if r["curve"]]
    common_initial_parameters = bool(initial_parameters) and all(p == initial_parameters[0] for p in initial_parameters)
    return {"schema_version": 1, "budget_per_run": budget, "expected_methods": list(methods), "expected_seeds": list(seeds),
            "scientific_config_validation": config_validation,
            "selection": {"sum_rmse": "Minimum valid Q_rmse + I_rmse + V_rmse up to each recorded slot; mixed physical units.",
                          "sum_mape": "Independent minimum valid Q_mape + I_mape + V_mape up to each recorded slot; percent.",
                          "failed_slots": "Consume budget; excluded from both minima.",
                          "aggregate_curves": "Median and seed min/max only where every expected seed has a recorded slot and metric.",
                          "at_budget": "Statistics over seeds with the full stated budget only; seed_count is explicit.",
                          "timing": "UTC start to final observed evaluation includes setup, checkpointing and resume gaps; component sums cover recorded evaluations."},
            "coverage": {"expected_runs": len(all_runs), "completed_runs": sum(r["coverage"]["budget_complete"] for r in all_runs),
                         "expected_slots": len(all_runs) * budget,
                         "recorded_slots": sum(r["coverage"]["recorded_slots"] for r in all_runs),
                         "valid_evaluations": sum(r["coverage"]["valid_evaluations"] for r in all_runs),
                         "failed_evaluations": sum(r["coverage"]["failed_evaluations"] for r in all_runs),
                         "initial_parameters_agree_across_recorded_runs": common_initial_parameters,
                         "initial_rmse_agrees_across_recorded_runs": common_initial_score},
            "methods": groups,
            "limitations": ["One simulated target; three seeds do not establish transfer across cells or chemistries.",
                            "RMSE and MAPE choose their best candidates independently; every method optimizes RMSE.",
                            "Seed bands are descriptive minima/maxima, not confidence intervals.",
                            "Missing slots are never extrapolated; incomplete methods cannot support a full-budget comparison."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Default: DATA/summary.json")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=100)
    args = parser.parse_args()
    result = summarize(args.data, args.methods, args.seeds, args.budget)
    output = args.out or args.data / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
