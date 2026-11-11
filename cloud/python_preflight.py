"""Verify actual prepared-runtime imports/native operations before any experiment."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import pickle
import site
import subprocess
import sys
import traceback


def require_owned(path, prefix, label):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(prefix):
        raise RuntimeError(f'{label} came from outside the prepared environment: {resolved}')
    return str(resolved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    report = {'status': 'running', 'python': sys.version, 'executable': sys.executable,
              'prefix': sys.prefix, 'isolated': sys.flags.isolated,
              'user_site_enabled': site.ENABLE_USER_SITE, 'modules': {},
              'smoke_objective': 'synthetic unit-box quadratic; no battery evaluations'}
    try:
        if sys.version_info[:2] != (3, 11) or Path(sys.prefix).resolve() != prefix:
            raise RuntimeError('Expected the prepared Python 3.11 virtual environment')
        if not sys.flags.isolated or site.ENABLE_USER_SITE:
            raise RuntimeError('Prepared Python must disable environment/user-site imports')
        for entry in sys.path:
            if 'site-packages' in Path(entry).parts or 'dist-packages' in Path(entry).parts:
                require_owned(entry, prefix, 'site-packages path')
        for module_name, distribution in [
            ('torch', 'torch'), ('numpy', 'numpy'), ('scipy', 'scipy'),
            ('ax', 'ax-platform'), ('botorch', 'botorch'), ('gpytorch', 'gpytorch'),
            ('pybamm', 'pybamm'), ('casadi', 'casadi'), ('pybammsolvers', 'pybammsolvers'),
            ('cma', 'cma'), ('pandas', 'pandas'), ('matplotlib', 'matplotlib'),
            ('yaml', 'PyYAML'), ('jinja2', 'Jinja2')]:
            module = importlib.import_module(module_name)
            report['modules'][module_name] = {
                'file': require_owned(module.__file__, prefix, module_name),
                'distribution': distribution,
                'version': importlib.metadata.version(distribution)}
        import torch
        if torch.version.cuda is not None:
            raise RuntimeError('Prepared BO runtime must use CPU-only PyTorch')
        torch.set_num_threads(2)
        if (torch.ones((2, 2)) @ torch.ones((2, 2))).sum().item() != 8:
            raise RuntimeError('Native CPU tensor smoke failed')
        require_owned(torch._C.__file__, prefix, 'torch._C')
        maps = Path('/proc/self/maps')
        if maps.exists():
            libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                                if '/libtorch' in line and line.split()[-1].startswith('/')})
            if not libraries:
                raise RuntimeError('No mapped libtorch libraries found after native tensor smoke')
            report['torch_native_libraries'] = [require_owned(path, prefix, 'loaded libtorch') for path in libraries]
        # Mirror the unchanged simulator worker's ordinary sys.executable launch;
        # its clean inherited environment must work even without -I.
        child_code = ('import json,sys,site,torch; '
                      'print(json.dumps({"prefix":sys.prefix,"torch":torch.__file__, '
                      '"user_site":site.ENABLE_USER_SITE,"safe_path":sys.flags.safe_path,'
                      '"tensor_sum":float(torch.ones(2).sum())}))')
        child = json.loads(subprocess.check_output([sys.executable, '-B', '-c', child_code], text=True, timeout=90))
        require_owned(child['torch'], prefix, 'worker torch')
        if Path(child['prefix']).resolve() != prefix or child['user_site'] or not child['safe_path'] or child['tensor_sum'] != 2:
            raise RuntimeError('Inherited worker runtime smoke failed')
        report['worker_runtime'] = child
        module_spec = importlib.util.spec_from_file_location('battery_preflight_runner', args.runner)
        runner = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = runner
        module_spec.loader.exec_module(runner)
        spec = json.loads(args.runner.with_name('spec.json').read_text())
        strategy = runner.BO(spec, 0)
        records = []
        def observe(parameters):
            score = sum((x - 0.3) ** 2 for x in runner.unit_vector(parameters, spec))
            record = {'evaluation': len(records) + 1, 'parameters': parameters, 'objective': score}
            strategy.observe(record)
            records.append(record)
        observe(spec['initial_parameters'])
        for _ in range(spec['bo_sobol_generated_trials'] + 1):
            parameters, metadata = strategy.ask(records)
            observe(parameters)
        if metadata['stage'] != 'gp_acquisition':
            raise RuntimeError('BO preflight did not reach GP acquisition')
        strategy = pickle.loads(pickle.dumps(strategy, protocol=5))
        parameters, metadata = strategy.ask(records)
        if metadata['stage'] != 'gp_acquisition' or not all(math.isfinite(x) for x in parameters.values()):
            raise RuntimeError('BO pickle/resume smoke failed')
        report.update(status='passed', cpu_tensor_smoke=True, bo_gp_acquisition=True,
                      bo_pickle_resume=True, synthetic_completed_observations=len(records))
        return 0
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
        return 1
    finally:
        temporary = args.out.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.out)


if __name__ == '__main__':
    raise SystemExit(main())
