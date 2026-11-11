"""Plot measured matched-budget seed traces and fully covered seed aggregates."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

COLORS = {"gemma": "#19846a", "qwen": "#8a56a6", "cma": "#2563a6", "bo": "#d66b20"}
CHANNEL_PANELS = (("Summed RMSE", "Mixed units"), ("Capacity RMSE", "Ah"),
                  ("Current RMSE", "A"), ("Voltage RMSE", "V"))
CHANNEL_STYLES = {
    "gemma": ("Gemma 4 E2B (agent)", "#007d68", "-", 2.8, "s"),
    "qwen": ("Qwen3.8 27B (agent)", "#733b98", "-", 2.8, "o"),
    "cma": ("CMA-ES", "#2563a6", "--", 2.1, "^"),
    "bo": ("BO", "#b45316", "-.", 2.1, "D"),
}


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
    save_figure(fig, output)


def rmse_incumbents(run, data):
    """Read one run's total and component RMSEs from its best fit at each slot."""
    expected = run["curve"]
    values = np.full((len(expected), len(CHANNEL_PANELS)), np.nan)
    if not expected:
        return values
    source = data / run["source"]["trajectory"]
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != run["source"]["sha256"]:
        raise ValueError(f"{source}: trajectory differs from summary; regenerate the summary")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(records) != len(expected):
        raise ValueError(f"{source}: slot count differs from summary")
    best = None
    for index, (record, point) in enumerate(zip(records, expected)):
        if record["evaluation"] != index + 1 or record["evaluation"] != point["evaluation"]:
            raise ValueError(f"{source}: evaluation order differs from summary")
        if record["status"] == "valid":
            terms = tuple(record["metrics"][f"{channel}_rmse"] for channel in "QIV")
            total = sum(terms)
            if best is None or total < best[0]:
                best = (total, *terms)
        if best is not None:
            values[index] = best
    return values


def plot_channels(summary, data, output):
    budget = summary["budget_per_run"]
    seeds = summary["expected_seeds"]
    x = np.arange(1, budget + 1)
    coverage = summary["coverage"]
    complete = coverage["completed_runs"] == coverage["expected_runs"]

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"}):
        fig, grid = plt.subplots(2, 2, figsize=(10.4, 7), sharex=True)
        fig.subplots_adjust(left=.087, right=.982, bottom=.158, top=.79,
                            wspace=.23, hspace=.28)
        axes = grid.ravel()
        handles = []
        has_zero = np.zeros(len(CHANNEL_PANELS), dtype=bool)

        for method in summary["expected_methods"]:
            group = summary["methods"][method]
            label, color, style, width, marker = CHANNEL_STYLES[method]
            if not group["recorded_slots"]:
                label += " · no observations"
            handles.append(Line2D([0], [0], color=color, linestyle=style, lw=width,
                                  marker=marker, markersize=5, label=label))
            traces = np.full((len(seeds), budget, len(CHANNEL_PANELS)), np.nan)
            for seed_index, run in enumerate(group["runs"]):
                values = rmse_incumbents(run, data)
                traces[seed_index, :len(values)] = values
            has_zero |= (traces == 0).any(axis=(0, 1))

            for column, ax in enumerate(axes):
                scores = traces[:, :, column]
                full = np.isfinite(scores).all(axis=0)
                for values in scores:
                    if np.isfinite(values).any():
                        ax.step(x, values, where="post", color=color, linestyle=style,
                                lw=.9, alpha=.22)
                if not full.any():
                    continue
                med = np.full(budget, np.nan)
                med[full] = np.median(scores[:, full], axis=0)
                ax.fill_between(x[full], scores[:, full].min(axis=0),
                                scores[:, full].max(axis=0), step="post", color=color,
                                alpha=.10, linewidth=0)
                ax.step(x, med, where="post", color=color, lw=width, linestyle=style,
                        zorder=4)
                ax.scatter(x[full][-1], med[full][-1], marker=marker, color=color,
                           s=28, zorder=5)

        ticks = MaxNLocator(nbins=5, integer=True).tick_values(1, budget)
        axes[-1].set_xticks([1, *(tick for tick in ticks if 1 < tick <= budget)])
        for ax, (title, unit), zero in zip(axes, CHANNEL_PANELS, has_zero):
            ax.set(title=title, ylabel=unit, yscale="linear" if zero else "log",
                   xlim=(max(0, 1 - budget * .015), budget * 1.025))
            ax.grid(which="major", alpha=.20)
            ax.set_axisbelow(True)
        title = "Battery parameter fitting" + ("" if complete else " · partial results")
        fig.suptitle(title, y=.980, fontsize=21, fontweight="bold")
        fig.text(.5, .915, f"One simulated DFN target · {len(seeds)} seeds per method · Lower is better",
                 ha="center", fontsize=12, color="#384251")
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .889),
                   ncol=min(4, len(handles)), frameon=False, fontsize=12, handlelength=2.3)
        fig.supxlabel("Simulation evaluation slot", y=.091, fontsize=12)
        fig.text(.5, .037, "Bold: seed median · Shading: seed range · Faint: individual seeds\n"
                 "All components come from each seed’s lowest summed RMSE fit so far.",
                 ha="center", fontsize=10.5, color="#384251")
        save_figure(fig, output)


def save_figure(fig, output):
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
    parser.add_argument("--channels", action="store_true",
                        help="Plot total and component RMSEs of the best summed RMSE fit")
    parser.add_argument("--data", type=Path, help="Trajectory root for --channels (default: summary directory)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    if args.channels:
        plot_channels(summary, args.data or args.summary.parent,
                      args.out or ROOT / "plots/battery_matched_channels")
    else:
        if args.data is not None:
            parser.error("--data requires --channels")
        plot(summary, args.out or ROOT / "plots/battery_matched_convergence")


if __name__ == "__main__":
    main()
