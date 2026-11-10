"""Bounded one-GPU launcher. Stages no credentials and exposes no network service.

All paths are supplied privately at launch; the public repository contains no
cloud account, quota, VPC, storage identifiers, or private endpoints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


EXPECTED_WEIGHTS = {
    'gemma4:e2b-qat-google': 'fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634',
    'qwen3.8:27b-opal': '11fdb2dc2fdec62d4684b0a4b6f6e96da3f9f633de1fba0a78830dff12b54375',
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path, check_deadline=None):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            if check_deadline is not None:
                check_deadline()
            h.update(chunk)
    return h.hexdigest()


def confined(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path must be inside the persistent namespace: {path}')
    return path


def stop_child(child):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=10)
        return
    try:
        child.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # Waiting for the leader says nothing about surviving descendants. Always
    # try to kill the remaining group, including when its leader already exited.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=10)


def cleanup_children(children):
    errors = []
    for name, child in children:
        try:
            stop_child(child)
        except BaseException as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--persistent-root', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--ollama', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=86400)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if not 1800 <= a.max_seconds <= 86400:
        p.error('max-seconds must be 1800..86400')
    root = a.persistent_root.resolve()
    if not root.is_dir():
        p.error('Persistent namespace is missing')
    # Require an actual separately mounted volume, not a pod-root fallback.
    volume = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,FSTYPE', '-T', str(root)], text=True).strip()
    if not volume or volume.split()[0] == '/' or 'overlay' in volume.split()[1:]:
        p.error('Persistent namespace is not on a mounted data volume')
    for name in ['project', 'upstream', 'ollama', 'models', 'run_dir']:
        setattr(a, name, confined(getattr(a, name), root))
    # Preserve the venv launcher path: resolving its final symlink would select
    # the base interpreter and silently drop the prepared venv's packages.
    a.python = confined(a.python.parent, root) / a.python.name
    for name in ['project', 'upstream', 'models']:
        if not getattr(a, name).is_dir():
            p.error(f'{name} directory is missing')
    for name in ['python', 'ollama']:
        if not getattr(a, name).is_file() or not os.access(getattr(a, name), os.X_OK):
            p.error(f'{name} executable is missing')
    if a.run_dir.exists() and not a.resume:
        p.error('Run directory exists; choose another directory or explicitly --resume')
    a.run_dir.mkdir(parents=True, exist_ok=True)
    for name in ['tmp', 'cache', 'config', 'data', 'state', 'matplotlib', 'ollama-home']:
        confined(a.run_dir / name, root).mkdir(exist_ok=True)
    os.chmod(a.run_dir / 'ollama-home', 0o700)
    status_path = a.run_dir / 'launcher_status.json'
    started = time.monotonic()
    deadline = started + a.max_seconds
    state = {'status': 'starting', 'phase': 'preflight',
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'max_seconds': a.max_seconds, 'requested_gpus': 1,
             'methods': ['gemma', 'qwen', 'cma', 'bo'], 'seeds': [0, 1, 2],
             'evaluations_per_method_seed': 100, 'smoke_inference_counts_toward_budget': False}
    write_json(status_path, state)
    server = benchmark = None
    log_handles = []

    def update(**changes):
        state.update(changes, elapsed_seconds=time.monotonic() - started,
                     updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(status_path, state)

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError('Launcher wall-clock budget exhausted')
        return value

    def interrupted(signum, _frame):
        raise InterruptedError(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def api(path, payload=None, timeout=30):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request('http://127.0.0.1:11434' + path, data=data,
                                         headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=min(timeout, remaining())) as response:
            return json.load(response)

    try:
        gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                            '--format=csv,noheader,nounits'], text=True, timeout=30)
        gpus = [line.strip() for line in gpu_text.splitlines() if line.strip()]
        if len(gpus) != 1:
            raise RuntimeError(f'Expected exactly one visible GPU; found {len(gpus)}')
        update(gpu_inventory=gpus, mount=volume, phase='verifying_model_weights')
        spec = json.loads((a.project / 'experiments/battery/matched/spec.json').read_text())
        hashes = {}
        for settings in spec['models'].values():
            model = settings['name']
            expected = EXPECTED_WEIGHTS[model]
            family, tag = model.split(':', 1)
            manifest_path = a.models / 'manifests/registry.ollama.ai/library' / family / tag
            manifest = json.loads(manifest_path.read_text())
            weights = [layer for layer in manifest['layers'] if layer['mediaType'] == 'application/vnd.ollama.image.model']
            if len(weights) != 1 or weights[0]['digest'] != 'sha256:' + expected:
                raise RuntimeError(f'Wrong model manifest for {model}')
            blob = a.models / 'blobs' / ('sha256-' + expected)
            if blob.stat().st_size != weights[0]['size'] or digest(blob, remaining) != expected:
                raise RuntimeError(f'Wrong model weights for {model}')
            hashes[model] = expected
        update(weight_sha256=hashes, phase='starting_local_ollama')
        with socket.socket() as test:
            test.bind(('127.0.0.1', 11434))  # Do not attach to an unrelated service.
        child_env = os.environ.copy()
        child_env.update(TMPDIR=str(a.run_dir / 'tmp'), XDG_CACHE_HOME=str(a.run_dir / 'cache'),
                         XDG_CONFIG_HOME=str(a.run_dir / 'config'), XDG_DATA_HOME=str(a.run_dir / 'data'),
                         XDG_STATE_HOME=str(a.run_dir / 'state'),
                         MPLCONFIGDIR=str(a.run_dir / 'matplotlib'), PYTHONDONTWRITEBYTECODE='1',
                         PYBAMM_DISABLE_TELEMETRY='true', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                         MKL_NUM_THREADS='2', CUDA_CACHE_PATH=str(a.run_dir / 'cache/cuda'))
        server_env = child_env.copy()
        # This is only the child process's dedicated home; the calling shell and
        # controller HOME are unchanged. Ollama creates its own local key here.
        server_env.update(HOME=str(a.run_dir / 'ollama-home'), OLLAMA_MODELS=str(a.models),
                          OLLAMA_HOST='127.0.0.1:11434', OLLAMA_NO_CLOUD='1',
                          OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1',
                          OLLAMA_KEEP_ALIVE='10m', OLLAMA_LOAD_TIMEOUT='10m')
        server_log = (a.run_dir / 'ollama.log').open('a')
        log_handles.append(server_log)
        server = subprocess.Popen([str(a.ollama), 'serve'], env=server_env, cwd=a.run_dir,
                                  stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        ready_by = time.monotonic() + min(120, remaining())
        while True:
            if server.poll() is not None:
                raise RuntimeError('Ollama exited before becoming ready; inspect ollama.log')
            try:
                version = api('/api/version')['version']
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= ready_by:
                    raise TimeoutError('Ollama readiness deadline exceeded')
                time.sleep(1)
        update(ollama_version=version, phase='gpu_model_smoke')
        smoke = []
        for settings in spec['models'].values():
            t0 = time.monotonic()
            response = api('/api/chat', {
                'model': settings['name'], 'stream': False, 'think': settings['think'],
                'messages': [{'role': 'user', 'content': 'Return exactly the JSON object {"ready":true}.'}],
                'format': {'type': 'object', 'properties': {'ready': {'type': 'boolean'}}, 'required': ['ready']},
                'options': {'seed': 0, 'temperature': 0, 'num_ctx': spec['agent_context_tokens'],
                            'num_predict': settings['num_predict']},
            }, timeout=900)
            if json.loads(response['message']['content']) != {'ready': True}:
                raise RuntimeError(f'Model smoke response failed for {settings["name"]}')
            active = next((x for x in api('/api/ps')['models'] if x.get('name') == settings['name']), None)
            if not active or active.get('size_vram', 0) <= 0:
                raise RuntimeError(f'No GPU placement reported for {settings["name"]}')
            smoke.append({'model': settings['name'], 'seconds': time.monotonic() - t0,
                          'size_vram': active['size_vram'], 'eval_count': response.get('eval_count'), 'success': True})
            write_json(a.run_dir / 'gpu_smoke.json', smoke)
        update(phase='benchmark', status='running', gpu_smoke_passed=True)
        command = [str(a.python), '-u', str(a.project / 'experiments/battery/matched/run.py'),
                   '--upstream', str(a.upstream), '--out', str(a.run_dir / 'benchmark'),
                   '--budget', '100', '--seeds', '0', '1', '2',
                   '--methods', 'gemma', 'qwen', 'cma', 'bo', '--base-url', 'http://127.0.0.1:11434']
        if a.resume:
            command.append('--resume')
        benchmark_log = (a.run_dir / 'benchmark.log').open('a')
        log_handles.append(benchmark_log)
        benchmark = subprocess.Popen(command, env=child_env, cwd=a.project, stdout=benchmark_log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        while benchmark.poll() is None:
            if server.poll() is not None:
                raise RuntimeError('Ollama exited during the benchmark; inspect ollama.log')
            update(benchmark_pid=benchmark.pid)
            time.sleep(min(10, remaining()))
        code = benchmark.returncode
        final = json.loads((a.run_dir / 'benchmark/status.json').read_text())
        if code != 0 or final.get('status') != 'completed':
            raise RuntimeError(f'Benchmark did not complete: exit={code}, status={final.get("status")}')
        update(status='completed', phase='finished', benchmark_exit_code=code)
        exit_code = 0
    except BaseException as exc:
        terminal = 'timed_out' if isinstance(exc, TimeoutError) else 'interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed'
        update(status=terminal, error=f'{type(exc).__name__}: {exc}')
        print(f'{terminal}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        exit_code = 124 if terminal == 'timed_out' else 130 if terminal == 'interrupted' else 1
    finally:
        cleanup_errors = cleanup_children([('benchmark', benchmark), ('ollama', server)])
        for handle in log_handles:
            try:
                handle.close()
            except BaseException as exc:
                cleanup_errors.append(f'log close: {type(exc).__name__}: {exc}')
        if cleanup_errors:
            update(status='failed', phase='cleanup', cleanup_errors=cleanup_errors)
            print('Cleanup failed: ' + '; '.join(cleanup_errors), file=sys.stderr, flush=True)
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
