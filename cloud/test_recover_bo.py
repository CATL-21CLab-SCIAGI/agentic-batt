"""Reference-data and equivalence contracts for BO-only recovery."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import recover_bo


class RecoveryTests(unittest.TestCase):
    def test_target_copy_preserves_bytes_and_rejects_changed_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / 'original.npz', root / 'target.npz'
            source.write_bytes(b'original target fixture')
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            recover_bo.copy_reference_target(source, target, expected)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            source.write_bytes(b'changed reference')
            with self.assertRaises(RuntimeError):
                recover_bo.copy_reference_target(source, target, expected)

    def test_reference_replay_checks_every_metric_and_objective(self):
        reference = {'objective': 3.7, 'metrics': {'Q_rmse': 0.2, 'V_mape': 1.5}}
        observed = {'status': 'valid', 'objective': 3.7 + 1e-12,
                    'metrics': {'Q_rmse': 0.2, 'V_mape': 1.5}}
        self.assertLess(recover_bo.check_equivalent(reference, observed)['objective'], 1e-8)
        observed['metrics']['V_mape'] = 1.6
        with self.assertRaises(RuntimeError):
            recover_bo.check_equivalent(reference, observed)


if __name__ == '__main__':
    unittest.main()
