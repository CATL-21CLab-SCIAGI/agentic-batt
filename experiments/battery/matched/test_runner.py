"""Meaningful contract tests; synthetic evaluations never enter published results."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
module_spec = importlib.util.spec_from_file_location('battery_matched_runner', HERE / 'run.py')
runner = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = runner
module_spec.loader.exec_module(runner)
SPEC = json.loads((HERE / 'spec.json').read_text())


class Contracts(unittest.TestCase):
    def test_bounds_reject_invalid_and_preserve_roundtrip(self):
        valid = runner.valid_params(SPEC['initial_parameters'], SPEC)
        recovered = runner.physical(runner.unit_vector(valid, SPEC), SPEC)
        for key in valid:
            self.assertAlmostEqual(valid[key], recovered[key])
        for invalid in [float('nan'), float('inf'), True, -1]:
            candidate = dict(valid)
            candidate[next(iter(valid))] = invalid
            with self.assertRaises(ValueError):
                runner.valid_params(candidate, SPEC)
        with self.assertRaises(ValueError):
            runner.valid_params({}, SPEC)

    def test_agent_repairs_before_simulation(self):
        model = runner.Agent.__new__(runner.Agent)
        model.spec, model.seed, model.model = SPEC, 2, 'gemma'
        model.base_url, model.timeout, model.knowledge = 'http://127.0.0.1:11434', 10, 'domain guidance'
        payloads = [json.dumps({'message': {'content': 'invalid JSON'}}).encode(),
                    json.dumps({'message': {'content': json.dumps({'parameters': SPEC['initial_parameters'], 'rationale': 'test'})}}).encode()]
        sent = []
        class Reply:
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return self.payload
        class HTTP:
            def open(self, req, timeout):
                sent.append(json.loads(req.data))
                return Reply(payloads.pop(0))
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner.urllib.request, 'build_opener', return_value=HTTP()):
            params, metadata = model.ask([], Path(temporary), 2)
            self.assertEqual(params, SPEC['initial_parameters'])
            self.assertEqual(metadata['inference_attempts'], 2)
            self.assertEqual(len(metadata['inference_errors']), 1)
            self.assertEqual(sent[0]['options']['seed'], 200002)
            self.assertIn('MAPE is supplementary', sent[0]['messages'][0]['content'])
            self.assertNotIn('parameter_change', sent[0]['messages'][1]['content'])

    def test_interrupted_evaluation_consumes_slot_and_resumes_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'target'; target.write_bytes(b'mock')
            args = SimpleNamespace(out=root, budget=13, simulation_timeout=1, model_timeout=1,
                                   mock=True, resume=True, upstream=HERE, base_url='http://127.0.0.1:11434')
            real_write = runner.write_json
            stopped = False
            def interrupted(path, data):
                nonlocal stopped
                if str(path).endswith('evaluations/0002.json') and not stopped:
                    stopped = True
                    raise RuntimeError('injected crash after simulator reservation')
                real_write(path, data)
            with patch.object(runner, 'write_json', side_effect=interrupted):
                with self.assertRaises(RuntimeError):
                    runner.run_method(args, copy.deepcopy(SPEC), 'cma', 0, target)
            summary = runner.run_method(args, copy.deepcopy(SPEC), 'cma', 0, target)
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['evaluations'], 13)
            self.assertEqual(summary['failed_evaluations'], 1)
            records = runner.load_records(root / 'cma_seed0')
            self.assertEqual(records[1]['status'], 'interrupted_simulation')
            self.assertEqual([r['evaluation'] for r in records], list(range(1, 14)))
            first_bytes = (root / 'cma_seed0/trajectory.jsonl').read_bytes()
            runner.run_method(args, copy.deepcopy(SPEC), 'cma', 0, target)
            self.assertEqual(first_bytes, (root / 'cma_seed0/trajectory.jsonl').read_bytes())
            self.assertEqual(records[11]['proposal']['generation'], 2)

    def test_resumed_proposals_preserve_attempt_archives_and_counts(self):
        model = runner.Agent.__new__(runner.Agent)
        model.spec, model.seed, model.model = SPEC, 0, 'gemma'
        model.base_url, model.timeout, model.knowledge = 'http://127.0.0.1:11434', 10, 'domain guidance'
        invalid = json.dumps({'message': {'content': 'invalid JSON'}}).encode()
        valid = json.dumps({'message': {'content': json.dumps({
            'parameters': SPEC['initial_parameters'], 'rationale': 'resumed proposal'})}}).encode()
        payloads = [invalid, invalid, invalid, valid]
        sent = []
        class Reply:
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return self.payload
        class HTTP:
            def open(self, req, timeout):
                sent.append(req)
                return Reply(payloads.pop(0))
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner.urllib.request, 'build_opener', return_value=HTTP()):
            directory = Path(temporary)
            archive = directory / 'inference/0002'
            archive.mkdir(parents=True)
            # The second request was interrupted before response/error storage.
            previous = {name: ('previous ' + name).encode() for name in
                        ['request-1.json', 'response-1.json', 'error-1.json', 'request-2.json']}
            for name, content in previous.items():
                (archive / name).write_bytes(content)
            with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
                model.ask([], directory, 2)
            self.assertEqual(len(sent), 3)
            params, metadata = model.ask([], directory, 2)
            self.assertEqual(params, SPEC['initial_parameters'])
            self.assertEqual(metadata['inference_attempts'], 1)
            for name, content in previous.items():
                self.assertEqual((archive / name).read_bytes(), content)
            self.assertTrue((archive / 'response-6.json').exists())
            self.assertFalse((archive / 'response-2.json').exists())
            summary = runner.refresh_outputs(directory, {'method': 'gemma', 'seed': 0, 'budget': 100, 'spec': SPEC})
            self.assertEqual(summary['inference_attempts'], 6)
            self.assertEqual(summary['inference_rejected_or_failed_attempts'], 4)


if __name__ == '__main__':
    unittest.main()
