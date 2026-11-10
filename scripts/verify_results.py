"""Check published records using only the Python standard library."""
import csv
import json
import math
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'experiments/opal/verify_results.py'), run_name='__main__')
battery = ROOT / 'experiments/battery/results'
summary = json.loads((battery / 'summary.json').read_text())


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


agent = rows(battery / 'agent/trajectory.csv')
assert len(agent) == 15
for case in summary['cases']:
    group = [r for r in agent if r['family'] == case['family']]
    assert case['status'] == 'completed' and len(group) == 5
    assert case['llm_responses'] == 4
    assert [int(r['evaluation']) for r in group] == list(range(1, 6))
    best = float('inf')
    for row in group:
        values = [float(row[f'{channel}_mape']) for channel in ['Q', 'I', 'V']]
        assert all(math.isfinite(x) for x in values)
        score = sum(values)
        assert math.isclose(score, float(row['sum_mape']))
        best = min(best, score)
        assert math.isclose(best, float(row['best_so_far_sum_mape']))
    assert math.isclose(best, sum(case['best_metrics'][f'{c}_mape'] for c in ['Q', 'I', 'V']))

baselines = rows(battery / 'baselines/trajectory.csv')
for method, count in [('cma_es_single', 30), ('bayesian_optimization', 25)]:
    group = [r for r in baselines if r['method'] == method]
    assert len(group) == count
    best = float('inf')
    for row in group:
        score = sum(float(row[f'{c}_rmse']) for c in ['Q', 'I', 'V'])
        assert math.isfinite(score) and math.isclose(score, float(row['sum_rmse']))
        best = min(best, score)
        assert math.isclose(best, float(row['best_so_far_sum_rmse']))
    record = next(r for r in summary['baselines'] if r['baseline'] == method)
    assert record['status'] == 'completed'
    assert math.isclose(best, record['best_result']['best_total_loss'])
assert sum(r['replayed'].lower() == 'true' for r in baselines) == 18

comparison = json.loads((battery / 'agent_vs_baselines.json').read_text())
groups = [('Gemma agent', [r for r in agent if r['family'] == 'single']),
          ('CMA-ES', [r for r in baselines if r['method'] == 'cma_es_single']),
          ('Bayesian optimization', [r for r in baselines if r['method'] == 'bayesian_optimization'])]
for method, group in groups:
    report = comparison['methods'][method]
    assert report['evaluations'] == len(group)
    for metric in ['mape', 'rmse']:
        scores = [sum(float(r[f'{c}_{metric}']) for c in ['Q', 'I', 'V']) for r in group]
        assert math.isclose(report[f'best_first5_sum_{metric}'], min(scores[:5]))
        assert math.isclose(report[f'best_full_sum_{metric}'], min(scores))

for name in ['opal_results', 'battery_agent_convergence', 'battery_baseline_convergence', 'agent_overhead_projection', 'battery_all_methods']:
    for ext in ['png', 'svg']:
        assert (ROOT / 'plots' / f'{name}.{ext}').stat().st_size > 1000
for doc in [ROOT / 'README.md', ROOT / 'THIRD_PARTY_NOTICES.md', ROOT / 'docs/reproduction.md']:
    for link in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
        if '://' not in link and not link.startswith('#'):
            assert (doc.parent / link.split('#')[0]).exists(), (doc.name, link)
print('Verified 15 battery agent evaluations, 55 baseline records, objective sums, replay counts, plots, and documentation links.')
