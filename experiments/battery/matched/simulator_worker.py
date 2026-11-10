"""One isolated PyBaMM operation, with upstream metric definitions unchanged."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'evaluate'])
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--request', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY', 'true')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import numpy as np
    import pybamm
    import yaml
    sys.path.insert(0, str(a.upstream / 'battery_agent'))
    from utils.data import load_simulated_battery_data
    spec = json.loads(a.spec.read_text())
    settings = yaml.safe_load((a.upstream / spec['target_file']).read_text())[spec['test_id']]
    if a.action == 'prepare':
        initial = pybamm.ParameterValues(settings['param_name'])
        assert all(float(initial[k]) == v for k, v in spec['initial_parameters'].items()), 'Initial point differs from selected parameter set'
        data = load_simulated_battery_data(settings, cycle_len=max(spec['cycle_indices_zero_based']) + 1)
        packed = {f'c{i}__{k}': v for i, cycle in enumerate(data) for k, v in cycle.items()}
        temporary_target = a.target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary_target, **packed)
        os.replace(temporary_target, a.target)
        answer = {'status': 'prepared', 'settings': settings, 'cycles': len(data)}
    else:
        from pybamm_runner import simulate_capacity
        payload = json.loads(a.request.read_text())
        with np.load(a.target, allow_pickle=False) as cache:
            data = [{k.split('__', 1)[1]: cache[k] for k in cache.files if k.startswith(f'c{i}__')}
                    for i in range(max(spec['cycle_indices_zero_based']) + 1)]
        ok, feedback, capacity, detail, metrics = simulate_capacity(
            payload['parameters'], data, param_name=settings['param_name'], model_name=settings['model_name'],
            cycle_idxs=spec['cycle_indices_zero_based'], is_plot=False, is_save=False,
            save_dir=str(a.out.parent), id=str(payload['evaluation']), return_detail=True,
            return_loss=True, ori_settings=settings, search_keys=list(spec['bounds']))
        if ok:
            means = {k: float(np.mean([r[k] for r in metrics])) for k in metrics[0]}
            if not all(np.isfinite(v) for v in means.values()):
                raise ValueError('Simulator returned non-finite metrics')
            answer = {'status': 'valid', 'metrics': means, 'objective': sum(means[f'{c}_rmse'] for c in 'QIV'),
                      'feedback': feedback, 'capacity': float(capacity)}
        else:
            answer = {'status': 'simulation_failed', 'objective': spec['simulation_failure_penalty'], 'error': str(feedback)[:3000]}
    a.out.write_text(json.dumps(answer, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
