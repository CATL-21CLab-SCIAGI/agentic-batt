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
    def test_decimal_json_preserves_values_and_string_content(self):
        values = {'small': [5.86e-6, 2.5e-8, 5e-324], 'large': 1.7976931348623157e308,
                  'negative': -4.2e-12, 'zero': -0.0, 'flags': [True, False, None, 7],
                  'text': 'Keep literal 5.86e-06 and "quotes" unchanged'}
        encoded = runner.decimal_json(values)
        self.assertEqual(json.loads(encoded), values)
        self.assertIn('0.00000586', encoded)
        self.assertIn('0.000000025', encoded)
        self.assertIn('"zero": -0.0', encoded)
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError):
                runner.decimal_json({'nested': [value]})
        self.assertEqual(runner.decimal_feedback('Chen2020: radius=5.86e-06, shift=-4.2E-12 m'),
                         'Chen2020: radius=0.00000586, shift=-0.0000000000042 m')

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

    def test_decimal_parameter_strings_are_strict_and_range_checked(self):
        params = {key: runner.decimal_json(value) for key, value in SPEC['initial_parameters'].items()}
        self.assertEqual(runner.agent_params(params, SPEC), SPEC['initial_parameters'])
        key = 'Negative particle radius [m]'
        for invalid in ['5.86e-06', '5.86E-06', 0.00000586, True, None, 'NaN', 'Infinity',
                        '0.00000586 m', ' 0.00000586', '0.00000586\n', '+0.00000586', '.00000586',
                        '5.86', '0.0000024999999999999999999999999999',
                        '0.0000274000000000000000000000000001']:
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                runner.agent_params(dict(params, **{key: invalid}), SPEC)
        for index in [0, 1]:
            boundary = {key: runner.decimal_json(limits[index]) for key, limits in SPEC['bounds'].items()}
            self.assertEqual(runner.agent_params(boundary, SPEC), {key: limits[index] for key, limits in SPEC['bounds'].items()})

    def test_agent_repairs_before_simulation(self):
        model = runner.Agent.__new__(runner.Agent)
        model.spec, model.seed, model.model = SPEC, 2, 'gemma'
        model.base_url, model.timeout, model.knowledge = 'http://127.0.0.1:11434', 10, 'domain guidance'
        decimal_params = {key: runner.decimal_json(value) for key, value in SPEC['initial_parameters'].items()}
        invalid_params = dict(decimal_params)
        invalid_params['Negative particle radius [m]'] = '5.86'
        bad_proposal = json.dumps({'parameters': invalid_params, 'rationale': 'intended radius 5.86e-06'})
        payloads = [json.dumps({'message': {'content': bad_proposal}}).encode(),
                    json.dumps({'message': {'content': json.dumps({'parameters': decimal_params, 'rationale': 'test'})}}).encode()]
        sent = []
        raw_sent = []
        class Reply:
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return self.payload
        class HTTP:
            def open(self, req, timeout):
                sent.append(json.loads(req.data))
                raw_sent.append(req.data.decode())
                return Reply(payloads.pop(0))
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner.urllib.request, 'build_opener', return_value=HTTP()):
            records = [{'evaluation': 1, 'status': 'valid', 'objective': 3.77,
                        'parameters': SPEC['initial_parameters'], 'feedback': 'Current radius: 5.86e-06'}]
            params, metadata = model.ask(records, Path(temporary), 2)
            self.assertEqual(params, SPEC['initial_parameters'])
            self.assertEqual(metadata['inference_attempts'], 2)
            self.assertEqual(len(metadata['inference_errors']), 1)
            self.assertEqual(sent[0]['options']['seed'], 200002)
            self.assertIn('MAPE is supplementary', sent[0]['messages'][0]['content'])
            self.assertIn('plain decimal JSON string', sent[0]['messages'][0]['content'])
            self.assertNotIn('parameter_change', sent[0]['messages'][1]['content'])
            self.assertNotIn('5.86e-06', sent[0]['messages'][1]['content'])
            self.assertIn('0.00000586', sent[0]['messages'][1]['content'])
            property_schema = sent[0]['format']['properties']['parameters']['properties']['Positive particle radius [m]']
            self.assertEqual(property_schema['type'], 'string')
            self.assertEqual(property_schema['pattern'], runner.DECIMAL_PARAMETER_PATTERN)
            self.assertIn('[0.000000025, 0.00002]', property_schema['description'])
            self.assertIn('decoded as 5.86', sent[1]['messages'][-1]['content'])
            self.assertIn('[0.0000025, 0.0000274]', sent[1]['messages'][-1]['content'])
            self.assertEqual((Path(temporary) / 'inference/0002/request-1.json').read_text().strip(), raw_sent[0])

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
            'parameters': {key: runner.decimal_json(value) for key, value in SPEC['initial_parameters'].items()},
            'rationale': 'resumed proposal'})}}).encode()
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
