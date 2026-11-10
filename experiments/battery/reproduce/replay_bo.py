"""Replay matching recorded BO evaluations before continuing a failed run."""
import argparse
import json
import math
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--records", required=True, type=Path)
args, remainder = parser.parse_known_args()
rows = {row["exp_idx"]: row for row in map(json.loads, args.records.read_text().splitlines())}
sys.path.insert(0, str(Path.cwd() / "BO"))
sys.argv = ["BO/pipeline.py", *remainder]
import pipeline

original = pipeline.black_box_function
replayed = []


def evaluate(exp_idx, configs, timestamp, parameters, mapping, scale_factor, baseline_data):
    if exp_idx in rows:
        row = rows[exp_idx]
        actual = {key: value / scale_factor for key, value in parameters.items()}
        expected = row["parameters"]
        if actual.keys() != expected.keys() or not all(
            math.isclose(actual[key], expected[key], rel_tol=1e-10, abs_tol=1e-16)
            for key in actual
        ):
            raise RuntimeError(f"Cannot replay evaluation {exp_idx}: generated parameters do not match the saved record")
        replayed.append(exp_idx)
        Path("replay_validation.json").write_text(json.dumps({
            "source": str(args.records.resolve()), "matched_evaluations": replayed,
            "validation": "All parameter names and values match before each reuse"
        }, indent=2))
        print(f"Reused verified evaluation {exp_idx}", flush=True)
        objective = {key: tuple(value) if isinstance(value, list) else value
                     for key, value in row["objective_value"].items()}
        return objective, row["loss_list"], row["total_loss"]
    return original(exp_idx, configs, timestamp, parameters, mapping, scale_factor, baseline_data)


pipeline.black_box_function = evaluate
pipeline.main()
