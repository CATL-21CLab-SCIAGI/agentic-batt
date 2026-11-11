"""Plot measured matched-budget seed traces and fully covered seed aggregates."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = {"gemma": "#19846a", "qwen": "#8a56a6", "cma": "#2563a6", "bo": "#d66b20"}


def plot(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    handles = []
    expected = len(summary["expected_seeds"])
    for method in summary["expected_methods"]:
        group = summary["methods"][method]
        color = COLORS[method]
        label = f'{group["label"]} ({group["completed_seed_count"]}/{expected} complete)'
        if not group["recorded_slots"]:
            label = f'{group["label"]}: no observations'
        handles.append(Line2D([0], [0], color=color, lw=2.5,
                              linestyle="-" if group["recorded_slots"] else ":", label=label))
        for ax, metric in zip(axes, ("sum_rmse", "sum_mape")):
            for run in group["runs"]:
                points = [p for p in run["curve"] if p[f"best_{metric}"] is not None]
                if not points:
                    continue
                values = np.array([p[f"best_{metric}"] for p in points])
                if np.any(values <= 0):
                    raise ValueError("A log plot cannot represent zero or negative scores; change the axis explicitly")
                ax.step([p["evaluation"] for p in points], values, where="post", color=color, lw=.9, alpha=.3)
            points = [p for p in group["aggregate_curve"] if p[metric] is not None]
            if points:
                x = np.array([p["evaluation"] for p in points])
                med = np.array([p[metric]["median"] for p in points])
                lower = np.array([p[metric]["min"] for p in points])
                upper = np.array([p[metric]["max"] for p in points])
                ax.fill_between(x, lower, upper, step="post", color=color, alpha=.12, linewidth=0)
                ax.step(x, med, where="post", color=color, lw=2.4)
                ax.scatter(x[-1], med[-1], color=color, s=24, zorder=3)
    axes[0].set(title="Optimized metric: summed RMSE", ylabel="Best sum of RMSEs (mixed physical units)")
    axes[1].set(title="Secondary metric: summed MAPE", ylabel="Best sum of MAPEs (%)")
    for ax in axes:
        ax.set(yscale="log", xlabel="Simulation evaluation slot", xlim=(1, summary["budget_per_run"]))
        ax.grid(which="major", alpha=.15)
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, .88))
    coverage = summary["coverage"]
    complete = coverage["completed_runs"] == coverage["expected_runs"]
    title = f'Matched battery fitting: {summary["budget_per_run"]} slots per method and seed'
    status = f'{coverage["completed_runs"]}/{coverage["expected_runs"]} runs complete; {coverage["recorded_slots"]:,}/{coverage["expected_slots"]:,} slots recorded'
    fig.suptitle(title + ("" if complete else " — partial results") + "\n" + status, fontsize=14)
    seeds = ", ".join(map(str, summary["expected_seeds"]))
    fig.supxlabel(f"Seeds {seeds}. Solid: median; shading: seed min–max; faint: individual seeds. Failed simulations consume slots.\n"
                  "Each panel selects its own best metric. Aggregates require all seeds; curves stop at recorded coverage. One simulated target.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = output.with_suffix("." + extension)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", metadata={"Creator": "agentic-batt"})
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "plots/battery_matched_convergence")
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.out)


if __name__ == "__main__":
    main()
