"""Verify the published numeric records without data, weights, or model inference."""
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / 'results/summary.json').read_text())
losses = {}
for label, folder in [('plain', 'plain'), ('keep', 'keep'), ('gemma', 'gemma'), ('ollama', 'qwen')]:
    directory = root / 'results' / folder
    recorded = json.loads((directory / 'result.json').read_text())
    recorded.update(json.loads((directory / 'evaluation.json').read_text()))
    for key, value in recorded.items():
        assert summary['results'][label][key] == value, (label, key)
    rows = [json.loads(line) for line in (directory / 'training.jsonl').read_text().splitlines()]
    assert len(rows) == recorded['steps'] == 2340
    assert [r['step'] for r in rows] == list(range(1, 2341))
    assert all(math.isfinite(r['loss']) for r in rows)
    losses[label] = rows
    events = directory / 'events.jsonl'
    if events.exists():
        events = [json.loads(line) for line in events.read_text().splitlines()]
        assert events[-1]['kind'] == 'end'
        if label == 'ollama':
            assert sum(e['kind'] == 'applied' for e in events) == 1
            assert sum(e['kind'] == 'kept' and e.get('failed', False) for e in events) == 1
            assert not any(e['kind'] == 'reverted' for e in events)
assert losses['plain'] == losses['keep'] == losses['gemma']
assert losses['plain'][:1200] == losses['ollama'][:1200]
assert len({summary['results'][k]['parameter_sha256'] for k in ['plain', 'keep', 'gemma']}) == 1
delta = summary['results']['ollama']['test_accuracy'] - summary['results']['plain']['test_accuracy']
assert math.isclose(delta * 100, summary['accuracy_delta_percentage_points'])
assert round(delta * 10000) == summary['net_additional_correct_test_predictions'] == 13
provenance = json.loads((root / 'provenance.json').read_text())
for name, expected in provenance['source_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('Verified all 2,340 losses per run, matching controls, held-out metrics, decision counts, and source hashes.')
