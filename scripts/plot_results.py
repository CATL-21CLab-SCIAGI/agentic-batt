"""Regenerate publication figures from the committed, measured result records."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / 'plots'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
BLUE, ORANGE, GREEN, GREY = '#2563a6', '#d66b20', '#19846a', '#78828e'


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight',
                    facecolor='white', metadata={'Creator': 'agentic-batt'})
    plt.close(fig)


def opal():
    root = ROOT / 'experiments/opal/results'
    summary = json.loads((root / 'summary.json').read_text())
    results = summary['results']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained',
                             gridspec_kw={'width_ratios': [1.6, 1, 1]})
    for name, label, color in [('plain', 'Plain / keep / Gemma (identical)', BLUE),
                                ('qwen', 'Qwen3.8-27B', ORANGE)]:
        rows = [json.loads(s) for s in (root / name / 'training.jsonl').read_text().splitlines()]
        loss = np.array([r['loss'] for r in rows])
        mean = np.convolve(loss, np.ones(100) / 100, mode='valid')
        axes[0].plot(np.arange(100, len(loss) + 1), mean, color=color, label=label, lw=2)
    axes[0].axvline(1200, color=GREY, ls='--', lw=1)
    axes[0].text(1240, .25, 'Cosine restart\napplied', color=ORANGE, fontsize=10)
    axes[0].set(yscale='log', xlabel='Training update', ylabel='Training cross entropy',
                title='Measured learning curves')
    axes[0].legend(frameon=False, fontsize=9)
    labels = ['Plain', 'OPAL keep', 'Gemma', 'Qwen']
    keys = ['plain', 'keep', 'gemma', 'ollama']
    colors = [BLUE, GREY, GREEN, ORANGE]
    for i, (key, color) in enumerate(zip(keys, colors)):
        acc = results[key]['test_accuracy'] * 100
        axes[1].scatter(acc, i, color=color, s=75, zorder=3)
        axes[1].annotate(f'{acc:.2f}%', (acc, i), xytext=(8, 0),
                         textcoords='offset points', va='center', fontsize=10)
    axes[1].set(yticks=range(4), yticklabels=labels, xlim=(98.35, 98.78),
                xlabel='Test accuracy (%)', title='Held-out test accuracy')
    axes[1].invert_yaxis()
    times = [results[k]['training_seconds'] for k in keys]
    axes[2].barh(labels, times, color=colors, height=.55)
    for i, t in enumerate(times):
        axes[2].text(t * 1.15, i, f'{t:.1f}s', va='center', fontsize=10)
    axes[2].set(xscale='log', xlim=(1, 4200), xlabel='Training wall time (s, log scale)',
                title='Agent waiting dominates this pilot')
    axes[2].invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('OPAL on MNIST: one seed, five epochs, three agent opportunities', fontsize=15)
    fig.supxlabel('100-update trailing mean at left; accuracy from 10,000 test images. No repeated-seed uncertainty estimate.', fontsize=10)
    save(fig, 'opal_results')

    wait = results['ollama']['agent_wait_seconds']
    hours = np.geomspace(1 / 60, 240, 400)
    fraction = 100 * wait / (3600 * hours + wait)
    fig, ax = plt.subplots(figsize=(8.5, 4.7), layout='constrained')
    ax.plot(hours, fraction, color=ORANGE, lw=2.5)
    for h in [1, 24, 168]:
        f = 100 * wait / (3600 * h + wait)
        ax.scatter([h], [f], color=BLUE, zorder=3)
        ax.annotate(f'{h}h: {f:.2f}%', (h, f), xytext=(7, 7), textcoords='offset points')
    ax.set(xscale='log', yscale='log', xlabel='Assumed training compute duration (hours)',
           ylabel='Agent wait / total elapsed time (%)', title='Illustrative amortization of a fixed agent budget')
    ax.grid(which='both', alpha=.15)
    fig.supxlabel(f'Projection, not a distributed benchmark: fixed {wait:.1f}s of agent waiting, blocking execution, no extra coordination cost.', fontsize=9)
    save(fig, 'agent_overhead_projection')


def battery():
    root = ROOT / 'experiments/battery/results'
    with (root / 'agent/trajectory.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    for ax, family, title in zip(axes, ['single', 'multi', 'degradation'],
                                 ['Single-parameter target', 'Multi-parameter target', 'Degradation target']):
        group = [r for r in rows if r['family'] == family]
        x = [int(r['evaluation']) for r in group]
        y = [float(r['sum_mape']) for r in group]
        best = [float(r['best_so_far_sum_mape']) for r in group]
        ax.plot(x, y, 'o--', color=GREY, label='Each evaluation', alpha=.8)
        ax.step(x, best, where='post', color=GREEN, lw=2.5, label='Best so far')
        ax.scatter([x[-1]], [best[-1]], color=GREEN, zorder=3)
        ax.set(title=title, xticks=x, xlabel='Simulation evaluation',
               ylabel='Sum of upstream MAPEs (%)')
        ax.grid(alpha=.15)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Battery-Sim-Agent with Gemma: five evaluations per case', fontsize=15)
    fig.supxlabel('Evaluation 1 is the initial configuration; evaluations 2–5 follow model proposals. Selected-cycle means; lower is better.', fontsize=9)
    save(fig, 'battery_agent_convergence')

    with (root / 'baselines/trajectory.csv').open() as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8.5, 4.7), layout='constrained')
    methods = list(dict.fromkeys(r['method'] for r in rows))
    for method, color in zip(methods, [BLUE, ORANGE]):
        group = [r for r in rows if r['method'] == method]
        x = [int(r['evaluation']) for r in group]
        y = [float(r['best_so_far_sum_rmse']) for r in group]
        label = 'CMA-ES (30 evaluations)' if 'cma' in method.lower() else 'BO (18 replayed + 7 new evaluations)'
        ax.step(x, y, where='post', label=label, color=color, lw=2)
    ax.set(title='Battery baselines on the single-parameter target', xlabel='Simulation evaluation',
           ylabel='Best sum of RMSEs (mixed physical units)', yscale='log')
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    fig.supxlabel('Upstream objective sums capacity, current and voltage RMSEs. Different objective and budget from the agent pilots.', fontsize=9)
    save(fig, 'battery_baseline_convergence')


if __name__ == '__main__':
    opal()
    battery()
