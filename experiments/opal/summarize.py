"""Collect matched MNIST pilot metrics and verify the no-agent control."""
import argparse
import collections
import json
import math
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--keep-name', default='keep3')
p.add_argument('--agent-name', default='ollama')
p.add_argument('--out', type=Path)
a = p.parse_args()
results = {}
for label, folder in [('plain', 'plain'), ('keep', a.keep_name), ('ollama', a.agent_name)]:
    d = a.root / folder
    result = json.loads((d / 'result.json').read_text())
    result.update(json.loads((d / 'evaluation.json').read_text()))
    rows = [json.loads(s) for s in (d / 'training.jsonl').read_text().splitlines()]
    assert len(rows) == result['steps'] and all(math.isfinite(r['loss']) for r in rows)
    if (d / 'opal/events.jsonl').exists():
        events = [json.loads(s) for s in (d / 'opal/events.jsonl').read_text().splitlines()]
        assert events[-1]['kind'] == 'end'
        result['event_counts'] = dict(collections.Counter(e['kind'] for e in events))
        result['agent_decisions'] = [e for e in events if e['kind'] in ('applied', 'kept', 'rejected', 'reverted', 'agent_error')]
    if (d / 'ollama_calls').exists():
        result['ollama_calls'] = [json.loads(f.read_text()) for f in sorted((d / 'ollama_calls').glob('*/result.json'))]
        result['ollama_rejected_attempts'] = [json.loads(f.read_text()) for f in sorted((d / 'ollama_calls').glob('*/error-*.json'))]
    results[label] = result
plain_losses = (a.root / 'plain/training.jsonl').read_bytes()
keep_losses = (a.root / a.keep_name / 'training.jsonl').read_bytes()
assert plain_losses == keep_losses
assert results['plain']['parameter_sha256'] == results['keep']['parameter_sha256']
summary = dict(results=results, controls_bitwise_identical=True,
               description='One-seed CPU MNIST pilot: upstream MLP/AdamW recipe, three local-model opportunities; not a reproduction of the published large-model experiments.',
               held_out_test='10,000 images evaluated in a separate process after training; no test feedback given to agent.',
               accuracy_delta_percentage_points=100*(results['ollama']['test_accuracy']-results['plain']['test_accuracy']))
(a.out or a.root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for label, r in results.items():
    print(f"{label:8} test accuracy {r['test_accuracy']:.4%}, loss {r['test_loss']:.6f}, training {r['training_seconds']:.2f}s, waiting {r['agent_wait_seconds']:.2f}s")
print('Control weights and all 2340 training losses identical:', summary['controls_bitwise_identical'])
print('Accuracy delta (percentage points):', summary['accuracy_delta_percentage_points'])
