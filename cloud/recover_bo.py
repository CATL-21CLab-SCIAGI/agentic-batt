"""Bounded CPU-only recovery of the unchanged matched BO runs, one process per seed."""
import argparse
import datetime as dt
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

# Explicit file loading also works when this entry point is started with -I.
_helper_spec = importlib.util.spec_from_file_location('battery_cloud_launcher', Path(__file__).with_name('run_dlc.py'))
launch = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(launch)


def copy_reference_target(source, destination, expected):
    if launch.digest(source) != expected:
        raise RuntimeError('Original target does not match its recorded SHA-256')
    if destination.exists():
        if launch.digest(destination) != expected:
            raise RuntimeError('Recovery output contains another target')
        return
    temporary = destination.with_suffix('.tmp.npz')
    shutil.copyfile(source, temporary)
    if launch.digest(temporary) != expected:
        raise RuntimeError('Copied target SHA-256 mismatch')
    temporary.replace(destination)


def reference_records(reference, initial_parameters):
    selected = {}
    for directory in sorted(reference.glob('*_seed*')):
        for path in sorted((directory / 'evaluations').glob('*.json')):
            record = json.loads(path.read_text())
            if record.get('status') != 'valid':
                continue
            if record['evaluation'] == 1 and record['parameters'] == initial_parameters:
                selected.setdefault('initial', (path, record))
            elif record['parameters'] != initial_parameters:
                selected.setdefault('noninitial', (path, record))
            if len(selected) == 2:
                return selected
    raise RuntimeError('Reference needs a valid common initial point and a distinct valid noninitial point')


def check_equivalent(reference, observed):
    if observed.get('status') != 'valid':
        raise RuntimeError('Reference replay simulation did not return valid metrics')
    differences = {}
    for key in ['objective'] + list(reference['metrics']):
        expected = reference[key] if key == 'objective' else reference['metrics'][key]
        actual = observed[key] if key == 'objective' else observed['metrics'][key]
        if not math.isfinite(actual) or not math.isclose(expected, actual, rel_tol=1e-8, abs_tol=1e-8):
            raise RuntimeError(f'Reference replay mismatch for {key}: expected={expected}, actual={actual}')
        differences[key] = abs(actual - expected)
    return differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['persistent-root', 'project', 'upstream', 'python', 'reference-benchmark', 'run-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2])
    p.add_argument('--max-seconds', type=int, default=5400)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 300 <= a.max_seconds <= 86400 or len(set(a.seeds)) != len(a.seeds):
        p.error('Use a unique seed list and max-seconds in 300..86400')
    root = a.persistent_root.resolve()
    volume = launch.mounted_namespace(root)
    for name in ['project', 'upstream', 'reference_benchmark', 'run_dir']:
        setattr(a, name, launch.confined(getattr(a, name), root))
    a.python = launch.confined(a.python.parent, root) / a.python.name
    if not a.python.is_file() or not os.access(a.python, os.X_OK):
        p.error('Prepared interpreter is missing')
    if a.run_dir.exists() and not a.resume:
        p.error('Run directory exists; use a fresh directory or explicitly --resume')
    a.run_dir.mkdir(parents=True, exist_ok=True)
    for name in launch.RUNTIME_DIRECTORIES:
        launch.confined(a.run_dir / name, root).mkdir(exist_ok=True)
    started = time.monotonic()
    deadline = started + a.max_seconds
    state = {'status': 'starting', 'phase': 'reference_validation', 'requested_gpus': 0,
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'mount': volume,
             'methods': ['bo'], 'seeds': a.seeds, 'budget_each': 100,
             'max_seconds': a.max_seconds, 'parallel_seed_processes': len(a.seeds),
             'reference_replay_counts_toward_budget': False, 'optimizer_observations_reused': False}
    children, logs = {}, []
    def update(**changes):
        state.update(changes, elapsed_seconds=time.monotonic() - started,
                     updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        launch.write_json(a.run_dir / 'launcher_status.json', state)
    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Recovery wall-clock budget exhausted')
        return value
    def interrupted(signum, _frame):
        raise InterruptedError(f'Received signal {signum}')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    update()
    try:
        runner = a.project / 'experiments/battery/matched/run.py'
        worker = runner.with_name('simulator_worker.py')
        spec = json.loads(runner.with_name('spec.json').read_text())
        reference_spec = json.loads((a.reference_benchmark / 'spec.json').read_text())
        provenance = json.loads((a.reference_benchmark / 'provenance.json').read_text())
        if spec != reference_spec or launch.digest(runner) != provenance['runner_sha256']:
            raise RuntimeError('Recovery must use the original immutable runner and specification')
        for name, expected in provenance['source_sha256'].items():
            source = launch.confined(a.upstream / 'battery_agent' / name, a.upstream)
            if launch.digest(source) != expected:
                raise RuntimeError(f'Original upstream source changed: {name}')
        records = reference_records(a.reference_benchmark, spec['initial_parameters'])
        config = json.loads((records['initial'][0].parent.parent / 'config.json').read_text())
        if launch.digest(worker) != config['worker_sha256']:
            raise RuntimeError('Recovery simulator worker differs from the original run')
        target = a.reference_benchmark / 'target.npz'
        target_hash = config['target_sha256']
        if launch.digest(target) != target_hash:
            raise RuntimeError('Reference target changed after the original run')
        env = launch.python_environment(a.run_dir, a.python)
        update(phase='python_preflight', runner_sha256=provenance['runner_sha256'], target_sha256=target_hash)
        runtime = launch.python_preflight(a.python, a.project, a.run_dir, env, min(600, remaining()))
        versions = {item['distribution']: item['version'] for item in runtime['modules'].values()}
        for name, expected in provenance['packages'].items():
            if versions.get(name) != expected:
                raise RuntimeError(f'Original dependency version changed: {name}')
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.upstream, env=env, text=True).strip()
        if revision != provenance['upstream_revision']:
            raise RuntimeError('Original upstream Git revision changed')
        settings_code = ('import json,sys,yaml; '
                         'print(json.dumps(yaml.safe_load(open(sys.argv[1]))[int(sys.argv[2])]))')
        settings = json.loads(subprocess.check_output(launch.python_command(a.python) +
                              ['-c', settings_code, str(a.upstream / spec['target_file']), str(spec['test_id'])],
                              env=env, cwd=a.run_dir, text=True, timeout=min(30, remaining())))
        target_metadata = json.loads((a.reference_benchmark / 'target_metadata.json').read_text())
        if settings != target_metadata['settings'] or target_metadata['cycles'] != max(spec['cycle_indices_zero_based']) + 1:
            raise RuntimeError('Original target model/protocol settings changed')
        update(phase='reference_replay', python_preflight_passed=True)
        replay_root = a.run_dir / 'reference_replay'
        replay_root.mkdir(exist_ok=True)
        replay = {}
        for label, (original_path, record) in records.items():
            work = replay_root / label
            work.mkdir(exist_ok=True)
            request, answer = work / 'request.json', work / 'answer.json'
            launch.write_json(request, {'parameters': record['parameters'], 'evaluation': record['evaluation']})
            command = launch.python_command(a.python) + [str(worker), 'evaluate', '--upstream', str(a.upstream),
                       '--spec', str(a.reference_benchmark / 'spec.json'), '--target', str(target),
                       '--request', str(request), '--out', str(answer)]
            with (work / 'simulator.log').open('w') as log:
                subprocess.run(command, env=env, cwd=a.project, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=min(600, remaining()))
            observed = json.loads(answer.read_text())
            replay[label] = {'status': 'passed', 'reference_record': str(original_path.relative_to(a.reference_benchmark)),
                             'reference_objective': record['objective'], 'observed_objective': observed['objective'],
                             'absolute_differences': check_equivalent(record, observed)}
            launch.write_json(replay_root / 'validation.json', replay)
        update(phase='benchmark', status='running', reference_replay_passed=True)
        for seed in a.seeds:
            out = a.run_dir / f'seed{seed}'
            out.mkdir(exist_ok=True)
            copy_reference_target(target, out / 'target.npz', target_hash)
            command = launch.python_command(a.python) + [str(runner), '--upstream', str(a.upstream),
                       '--out', str(out), '--budget', '100', '--methods', 'bo', '--seeds', str(seed)]
            if a.resume:
                command.append('--resume')
            log = (a.run_dir / f'bo_seed{seed}.log').open('a')
            logs.append(log)
            children[seed] = subprocess.Popen(command, env=env, cwd=a.project, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
        while any(child.poll() is None for child in children.values()):
            statuses = {}
            for seed, child in children.items():
                path = a.run_dir / f'seed{seed}/bo_seed{seed}/status.json'
                statuses[str(seed)] = json.loads(path.read_text()) if path.exists() else {'status': 'starting'}
                if child.poll() is not None and child.returncode != 0:
                    raise RuntimeError(f'BO seed {seed} exited with status {child.returncode}')
            update(results=statuses)
            time.sleep(min(10, remaining()))
        results = {}
        for seed, child in children.items():
            result = json.loads((a.run_dir / f'seed{seed}/bo_seed{seed}/status.json').read_text())
            if child.returncode != 0 or result.get('status') != 'completed' or result.get('evaluations') != 100:
                raise RuntimeError(f'BO seed {seed} did not complete all 100 evaluations')
            results[str(seed)] = result
        update(status='completed', phase='finished', results=results)
        code = 0
    except BaseException as exc:
        status = 'timed_out' if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else 'interrupted' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'failed'
        update(status=status, error=f'{type(exc).__name__}: {exc}')
        print(state['error'], file=sys.stderr, flush=True)
        code = 124 if status == 'timed_out' else 130 if status == 'interrupted' else 1
    finally:
        errors = launch.cleanup_children([(f'bo_seed{seed}', child) for seed, child in children.items()])
        for log in logs:
            log.close()
        if errors:
            update(status='failed', phase='cleanup', cleanup_errors=errors)
            code = 1
    return code


if __name__ == '__main__':
    raise SystemExit(main())
