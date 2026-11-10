"""Fresh matched-budget fitting, durable observations, and resumable optimizer state.

Every method spends evaluation 1 on the same initial point. Failed or timed-out
simulations consume a slot. Invalid LLM proposals are repaired before simulation
and are counted separately; exhausted repair attempts stop that run visibly.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, data):
    atomic(path, (json.dumps(data, indent=2, allow_nan=False) + '\n').encode())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_params(params, spec):
    if not isinstance(params, dict) or set(params) != set(spec['bounds']):
        raise ValueError('Return all nine allowed parameters, without extra keys')
    result = {}
    for key, (low, high) in spec['bounds'].items():
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        if not low <= value <= high:
            raise ValueError(f'{key} must lie in [{low}, {high}]')
        result[key] = float(value)
    return result


def unit_vector(params, spec):
    return [(params[k] - lo) / (hi - lo) for k, (lo, hi) in spec['bounds'].items()]


def physical(vector, spec):
    values = {k: lo + min(1., max(0., float(x))) * (hi - lo)
              for x, (k, (lo, hi)) in zip(vector, spec['bounds'].items())}
    return valid_params(values, spec)


class CMA:
    def __init__(self, spec, seed):
        import cma
        self.spec = spec
        # cma seed=0 means time-based randomness; use the recorded seed+1 mapping.
        self.es = cma.CMAEvolutionStrategy(unit_vector(spec['initial_parameters'], spec), spec['cma_sigma_unit_box'],
                    {'bounds': [0., 1.], 'popsize': spec['cma_population'], 'seed': seed + 1,
                     'verbose': -9, 'verb_log': 0, 'CMA_stds': [1.] * len(spec['bounds'])})
        self.population, self.scores, self.index = [], [], 0

    def ask(self, records, **unused):
        if not self.population:
            self.population = self.es.ask()
            self.index, self.scores = 0, []
        params = physical(self.population[self.index], self.spec)
        return params, {'generation': self.es.countiter + 1, 'member': self.index + 1}

    def observe(self, record):
        if record['evaluation'] == 1:
            return
        self.scores.append(record['objective'])
        self.index += 1
        if self.index == len(self.population):
            self.es.tell(self.population, self.scores)
            self.population = []


class BO:
    def __init__(self, spec, seed):
        import torch
        from ax.service.ax_client import AxClient, ObjectiveProperties
        from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
        from ax.modelbridge.registry import Models
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        self.spec = spec
        strategy = GenerationStrategy(steps=[
            GenerationStep(model=Models.SOBOL, num_trials=spec['bo_sobol_generated_trials'], model_kwargs={'seed': seed}),
            GenerationStep(model=Models.BOTORCH_MODULAR, num_trials=-1, max_parallelism=1,
                           model_kwargs={'torch_device': torch.device('cpu')})])
        self.client = AxClient(generation_strategy=strategy, random_seed=seed, verbose_logging=False)
        self.client.create_experiment(name=f'battery_matched_seed{seed}',
            parameters=[{'name': f'p{i}', 'type': 'range', 'bounds': [0., 1.], 'value_type': 'float'} for i in range(len(spec['bounds']))],
            objectives={'loss': ObjectiveProperties(minimize=True)})
        self.trial = None

    def ask(self, records, **unused):
        params, self.trial = self.client.get_next_trial()
        stage = 'sobol' if len(records) <= self.spec['bo_sobol_generated_trials'] else 'gp_acquisition'
        return physical([params[f'p{i}'] for i in range(len(self.spec['bounds']))], self.spec), {'ax_trial': self.trial, 'stage': stage}

    def observe(self, record):
        if record['evaluation'] == 1:
            _, self.trial = self.client.attach_trial(parameters={f'p{i}': x for i, x in enumerate(unit_vector(record['parameters'], self.spec))})
        self.client.complete_trial(trial_index=self.trial, raw_data={'loss': (record['objective'], 0.0)})


class Agent:
    def __init__(self, spec, seed, model, base_url, upstream, timeout):
        self.spec, self.seed, self.model, self.base_url = spec, seed, model, base_url
        self.timeout = timeout
        import yaml
        prompt = yaml.safe_load((upstream / 'battery_agent/prompt.yaml').read_text())
        # Only authored domain knowledge, no hidden target parameter values.
        self.knowledge = prompt['FIRST_CYCLE_PROMPT']['TEXT_KNOWLEDGE'].split('The params should just')[0]

    def observe(self, record):
        pass

    def ask(self, records, directory, evaluation):
        best = min((r for r in records if r['status'] == 'valid'), key=lambda r: r['objective'], default=None)
        feedback = lambda r: {k: r[k] for k in ['evaluation', 'parameters', 'objective', 'status', 'metrics', 'feedback', 'error'] if k in r}
        context = {'protocol': 'Chen2020 DFN, 0.2C charge/discharge CCCV, score zero-based cycle 1',
                   'objective': self.spec['objective'], 'bounds': self.spec['bounds'],
                   'best_observation': feedback(best) if best else None,
                   'recent_observations': [feedback(r) for r in records[-8:]],
                   'objective_history': [[r['evaluation'], r['objective'], r['status']] for r in records]}
        system = ('You propose physical parameters for battery fitting. Minimize exactly the sum of capacity, current and voltage RMSE. '
                  'MAPE is supplementary feedback, not the objective. Return a JSON object with rationale and parameters. '
                  'parameters must include all nine names in bounds, finite numeric values, and stay inside each bound. '
                  'Use the best candidate and diagnostic feedback; failed simulations are penalized. Do not repeat an unchanged '
                  'candidate without a reason. No simulator tools are available to you.\nDomain knowledge:\n' + self.knowledge)
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(context)}]
        properties = {k: {'type': 'number', 'minimum': limits[0], 'maximum': limits[1]} for k, limits in self.spec['bounds'].items()}
        schema = {'type': 'object', 'properties': {'rationale': {'type': 'string'},
                 'parameters': {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}},
                 'required': ['rationale', 'parameters'], 'additionalProperties': False}
        opts = self.spec['models'][self.model]
        request_dir = directory / 'inference' / f'{evaluation:04d}'
        request_dir.mkdir(parents=True, exist_ok=True)
        # A process may stop during proposal generation, before the simulator
        # slot is reserved. Preserve every archived attempt when it is resumed.
        archived_attempts = []
        for path in request_dir.glob('*.json'):
            kind, _, suffix = path.stem.partition('-')
            if kind in {'request', 'response', 'error'} and suffix.isdigit():
                archived_attempts.append(int(suffix))
        archive_offset = max(archived_attempts, default=0)
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        errors = []
        for attempt in range(1, 4):
            archive_attempt = archive_offset + attempt
            body = {'model': opts['name'], 'messages': messages, 'stream': False, 'think': opts['think'], 'format': schema,
                    'options': {'seed': self.seed * 100000 + evaluation, 'temperature': opts['temperature'],
                                'num_ctx': self.spec['agent_context_tokens'], 'num_predict': opts['num_predict']},
                    'keep_alive': '30m'}
            write_json(request_dir / f'request-{archive_attempt}.json', body)
            start = time.monotonic()
            try:
                req = urllib.request.Request(self.base_url + '/api/chat', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                with http.open(req, timeout=self.timeout) as response:
                    answer = json.loads(response.read(4_000_000))
                write_json(request_dir / f'response-{archive_attempt}.json', answer)
                proposal = json.loads(answer['message']['content'])
                if set(proposal) != {'parameters', 'rationale'} or not isinstance(proposal['rationale'], str):
                    raise ValueError('Return exactly parameters and rationale')
                params = valid_params(proposal['parameters'], self.spec)
                return params, {'rationale': proposal['rationale'], 'inference_attempts': attempt,
                                'inference_seconds_last_attempt': time.monotonic() - start,
                                'inference_errors': errors, 'model': opts['name']}
            except Exception as exc:
                error = {'attempt': attempt, 'archive_attempt': archive_attempt,
                         'error': str(exc)[:2000], 'seconds': time.monotonic() - start}
                write_json(request_dir / f'error-{archive_attempt}.json', error)
                errors.append(error)
                messages.append({'role': 'user', 'content': 'Proposal could not be used: ' + str(exc)[:1000] + '. Return complete valid JSON within the supplied bounds.'})
        raise RuntimeError(f'Model proposal failed after 3 attempts before simulation: {errors[-1]["error"]}')


def load_records(directory):
    return [json.loads(p.read_text()) for p in sorted((directory / 'evaluations').glob('*.json'))]


def refresh_outputs(directory, config, status='running', error=None):
    records = load_records(directory)
    atomic(directory / 'trajectory.jsonl', ''.join(json.dumps(r, allow_nan=False) + '\n' for r in records).encode())
    valid = [r for r in records if r['status'] == 'valid']
    summary = {'status': status, 'method': config['method'], 'seed': config['seed'], 'budget': config['budget'],
               'evaluations': len(records), 'valid_evaluations': len(valid),
               'failed_evaluations': len(records) - len(valid), 'objective': config['spec']['objective'],
               'interrupted_budget_slots': sum(r['status'] == 'interrupted_simulation' for r in records),
               'inference_attempts': len(list((directory / 'inference').glob('*/request-*.json'))),
               'inference_rejected_or_failed_attempts': len(list((directory / 'inference').glob('*/error-*.json'))),
               'best_objective': min((r['objective'] for r in valid), default=None),
               'best_parameters': min(valid, key=lambda r: r['objective'])['parameters'] if valid else None,
               'milestones': {str(n): min((r['objective'] for r in valid if r['evaluation'] <= n), default=None)
                              for n in [5, 30, 100] if len(records) >= n},
               'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    if records:
        summary['simulation_wall_seconds'] = sum(r.get('simulation_seconds', 0) for r in records)
        summary['proposal_wall_seconds'] = sum(r.get('proposal', {}).get('proposal_seconds', 0) for r in records)
    timing_file = directory / 'timing.json'
    if timing_file.exists():
        started = json.loads(timing_file.read_text())['started_utc']
        summary['started_utc'] = started
        summary['elapsed_wall_seconds_including_resume_gaps'] = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(started)).total_seconds()
    if error:
        summary['error'] = error
    write_json(directory / 'status.json', summary)
    return summary


def persist(directory, state):
    state['python_rng'] = random.getstate()
    import numpy as np
    state['numpy_rng'] = np.random.get_state()
    try:
        import torch
        state['torch_rng'] = torch.get_rng_state()
    except ImportError:
        pass
    atomic(directory / 'optimizer_state.pkl', pickle.dumps(state, protocol=5))


def run_method(args, spec, method, seed, target):
    directory = args.out / f'{method}_seed{seed}'
    directory.mkdir(exist_ok=True)
    (directory / 'evaluations').mkdir(exist_ok=True)
    config = {'method': method, 'seed': seed, 'budget': args.budget, 'spec': spec,
              'runner_sha256': digest(__file__), 'worker_sha256': digest(HERE / 'simulator_worker.py'),
              'target_sha256': digest(target), 'simulation_timeout': args.simulation_timeout,
              'model_timeout': args.model_timeout, 'mock': args.mock}
    config_file = directory / 'config.json'
    if config_file.exists():
        if json.loads(config_file.read_text()) != config:
            raise ValueError(f'Resume config/source/target mismatch: {directory.name}')
        if not args.resume:
            raise ValueError(f'Output exists; use --resume: {directory.name}')
    else:
        write_json(config_file, config)
    if not (directory / 'timing.json').exists():
        write_json(directory / 'timing.json', {'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
    checkpoint = directory / 'optimizer_state.pkl'
    if checkpoint.exists():
        # Only load checkpoints from this explicitly selected, locally owned run directory.
        state = pickle.loads(checkpoint.read_bytes())
        random.setstate(state['python_rng'])
        import numpy as np
        np.random.set_state(state['numpy_rng'])
        if 'torch_rng' in state:
            import torch
            torch.set_rng_state(state['torch_rng'])
    else:
        random.seed(seed)
        import numpy as np
        np.random.seed(seed)
        strategy = CMA(spec, seed) if method == 'cma' else BO(spec, seed) if method == 'bo' else Agent(spec, seed, method, args.base_url, args.upstream, args.model_timeout)
        state = {'strategy': strategy, 'completed': 0, 'pending': None}
        persist(directory, state)
    try:
        while state['completed'] < args.budget:
            index = state['completed'] + 1
            if state['pending'] is None:
                if index == 1:
                    params, meta = spec['initial_parameters'], {'stage': 'shared_initial_point'}
                else:
                    proposal_started = time.monotonic()
                    params, meta = state['strategy'].ask(load_records(directory), directory=directory, evaluation=index)
                    meta['proposal_seconds'] = time.monotonic() - proposal_started
                state['pending'] = {'evaluation': index, 'parameters': valid_params(params, spec), 'proposal': meta, 'started': False}
                persist(directory, state)
            pending = state['pending']
            result_file = directory / 'evaluations' / f'{index:04d}.json'
            if result_file.exists():
                record = json.loads(result_file.read_text())
            else:
                record = {k: pending[k] for k in ['evaluation', 'parameters', 'proposal']}
                start = time.monotonic()
                if pending['started']:
                    record.update(status='interrupted_simulation', objective=spec['simulation_failure_penalty'],
                                  error='Process stopped after reserving this simulator slot; conservatively counted, not rerun')
                else:
                    pending['started'] = True
                    persist(directory, state)
                    if args.mock:
                        score = sum((x - .3) ** 2 for x in unit_vector(pending['parameters'], spec))
                        record.update(status='valid', objective=score, metrics={f'{c}_rmse': score / 3 for c in 'QIV'}, feedback='Synthetic test evaluator, not battery evidence')
                    else:
                        work = directory / 'work' / f'{index:04d}'
                        work.mkdir(parents=True, exist_ok=True)
                        request, answer = work / 'request.json', work / 'answer.json'
                        write_json(request, record)
                        cmd = [sys.executable, str(HERE / 'simulator_worker.py'), 'evaluate', '--upstream', str(args.upstream),
                               '--spec', str(args.out / 'spec.json'), '--target', str(target), '--request', str(request), '--out', str(answer)]
                        with (work / 'simulator.log').open('w') as log:
                            try:
                                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=args.simulation_timeout, env=worker_env(args.out))
                                if process.returncode != 0 or not answer.exists():
                                    raise RuntimeError(f'Simulator worker exit {process.returncode}; see simulator.log')
                                record.update(json.loads(answer.read_text()))
                            except Exception as exc:
                                record.update(status='simulation_failed', objective=spec['simulation_failure_penalty'], error=str(exc))
                record.update(simulation_seconds=time.monotonic() - start, utc=dt.datetime.now(dt.timezone.utc).isoformat())
                write_json(result_file, record)
            state['strategy'].observe(record)
            state['completed'], state['pending'] = index, None
            persist(directory, state)
            summary = refresh_outputs(directory, config)
            print(json.dumps({'method': method, 'seed': seed, 'evaluation': index, 'status': record['status'],
                              'objective': record['objective'], 'best': summary['best_objective']}), flush=True)
        return refresh_outputs(directory, config, 'completed')
    except Exception as exc:
        refresh_outputs(directory, config, 'failed', str(exc))
        raise


def worker_env(out):
    env = os.environ.copy()
    env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(out / 'matplotlib'), PYBAMM_DISABLE_TELEMETRY='true',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2')
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--spec', type=Path, default=HERE / 'spec.json')
    p.add_argument('--methods', nargs='+', choices=['gemma', 'qwen', 'cma', 'bo'], default=['gemma', 'qwen', 'cma', 'bo'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--budget', type=int, default=100)
    p.add_argument('--base-url', default='http://127.0.0.1:11434')
    p.add_argument('--simulation-timeout', type=int, default=600)
    p.add_argument('--model-timeout', type=int, default=900)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--mock', action='store_true', help='Synthetic test objective; never counted as battery results')
    a = p.parse_args()
    a.upstream, a.out = a.upstream.resolve(), a.out.resolve()
    url = urllib.parse.urlparse(a.base_url)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'} or url.username or url.password:
        p.error('Only unauthenticated localhost Ollama is supported')
    a.base_url = a.base_url.rstrip('/')
    if a.budget < 1 or a.budget > 1000 or any(seed < 0 for seed in a.seeds):
        p.error('Budget must be 1..1000 and seeds nonnegative')
    spec = json.loads(a.spec.read_text())
    valid_params(spec['initial_parameters'], spec)
    actual_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.upstream, text=True).strip()
    if actual_revision != spec['upstream_revision']:
        raise ValueError('Pinned upstream revision mismatch')
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'spec.json').exists() and json.loads((a.out / 'spec.json').read_text()) != spec:
        raise ValueError('Existing output has another specification')
    write_json(a.out / 'spec.json', spec)
    source_files = ['pybamm_runner.py', 'utils/exp.py', 'utils/data.py', 'utils/plot.py', 'prompt.yaml']
    packages = {name: importlib.metadata.version(name) for name in ['numpy', 'pybamm', 'cma', 'ax-platform']}
    for name in ['torch', 'pandas', 'botorch', 'gpytorch', 'casadi', 'pybammsolvers']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    provenance = {'upstream_revision': actual_revision, 'source_sha256': {name: digest(a.upstream / 'battery_agent' / name) for name in source_files},
                  'runner_sha256': digest(__file__), 'python': sys.version, 'mock': a.mock,
                  'packages': packages,
                  'shared_initial_evaluation_counted': True, 'target_generation_counted_as_optimizer_evaluation': False,
                  'bo_setup': 'Common initial point is attached; then 18 generated Sobol trials followed by Ax BoTorch modular GP acquisition.',
                  'cma_setup': 'Normalized unit cube; population 10; sigma 0.2; RNG seed=experiment seed+1; final partial population is evaluated but not told.',
                  'replay_policy': 'No previous-pilot observations reused. Resume restores only this run; an uncertain interrupted simulator call consumes its reserved slot.'}
    if (a.out / 'provenance.json').exists() and json.loads((a.out / 'provenance.json').read_text()) != provenance:
        raise ValueError('Resume environment/source provenance mismatch')
    write_json(a.out / 'provenance.json', provenance)
    target = a.out / 'target.npz'
    if not target.exists():
        if a.mock:
            target.write_bytes(b'mock-target')
        else:
            cmd = [sys.executable, str(HERE / 'simulator_worker.py'), 'prepare', '--upstream', str(a.upstream),
                   '--spec', str(a.out / 'spec.json'), '--target', str(target), '--out', str(a.out / 'target_metadata.json')]
            with (a.out / 'target_prepare.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=a.simulation_timeout, env=worker_env(a.out))
    results = []
    failed = False
    for method in a.methods:  # Group seeds by model to avoid repeated GPU reloads.
        for seed in a.seeds:
            try:
                results.append(run_method(a, spec, method, seed, target))
            except Exception as exc:
                failed = True
                print(f'{method} seed {seed}: FAILED: {exc}', file=sys.stderr, flush=True)
                path = a.out / f'{method}_seed{seed}' / 'status.json'
                results.append(json.loads(path.read_text()) if path.exists() else {'method': method, 'seed': seed, 'status': 'failed', 'error': str(exc)})
            write_json(a.out / 'status.json', {'status': 'running', 'methods': a.methods, 'seeds': a.seeds, 'budget_each': a.budget, 'results': results})
    write_json(a.out / 'status.json', {'status': 'failed' if failed else 'completed', 'methods': a.methods, 'seeds': a.seeds, 'budget_each': a.budget, 'results': results})
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
