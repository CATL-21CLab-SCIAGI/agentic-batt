"""CPU-only launcher cleanup checks; no cloud, GPU, or model calls."""
import signal
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import venv

import run_dlc


class CleanupTests(unittest.TestCase):
    def test_exited_leader_does_not_leave_descendants(self):
        child = mock.Mock(pid=1234)
        child.wait.return_value = 0
        with mock.patch.object(run_dlc.os, 'killpg') as killpg:
            run_dlc.stop_child(child)
        self.assertEqual(killpg.call_args_list, [
            mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL)])

    def test_unresponsive_leader_is_killed(self):
        child = mock.Mock(pid=1234)
        child.wait.side_effect = [subprocess.TimeoutExpired('child', 20), -9]
        with mock.patch.object(run_dlc.os, 'killpg') as killpg:
            run_dlc.stop_child(child)
        killpg.assert_called_with(1234, signal.SIGKILL)
        child.wait.assert_called_with(timeout=10)

    def test_gone_group_is_harmless(self):
        child = mock.Mock(pid=1234)
        with mock.patch.object(run_dlc.os, 'killpg', side_effect=ProcessLookupError):
            run_dlc.stop_child(child)
        child.wait.assert_called_once_with(timeout=10)

    def test_cleanup_failure_does_not_skip_second_group(self):
        first, second = mock.Mock(), mock.Mock()
        with mock.patch.object(run_dlc, 'stop_child', side_effect=[RuntimeError('failed'), None]) as stop:
            errors = run_dlc.cleanup_children([('benchmark', first), ('ollama', second)])
        self.assertEqual(stop.call_args_list, [mock.call(first), mock.call(second)])
        self.assertEqual(errors, ['benchmark: RuntimeError: failed'])


class PythonIsolationTests(unittest.TestCase):
    def test_poisoned_imports_are_removed_for_runner_and_ordinary_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = root / 'venv'
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / 'bin/python'
            package_dir = next((environment / 'lib').glob('python*/site-packages'))
            (package_dir / 'runtime_probe.py').write_text('marker = "prepared"\n')
            poison = root / 'image'; poison.mkdir()
            (poison / 'runtime_probe.py').write_text('marker = "image"\n')
            current = root / 'current'; current.mkdir()
            (current / 'runtime_probe.py').write_text('marker = "current"\n')
            neutral = root / 'neutral'; neutral.mkdir()
            inherited = dict(os.environ, PYTHONPATH=str(poison))
            probe = 'import runtime_probe; print(runtime_probe.marker)'
            self.assertEqual(subprocess.check_output([str(python), '-c', probe], env=inherited,
                                                    cwd=neutral, text=True).strip(), 'image')
            inherited.update(PYTHONHOME='/missing-image-python', PYTHONUSERBASE=str(poison),
                             LD_LIBRARY_PATH='/image/python3.12/torch/lib', LD_PRELOAD='/image/foreign.so')
            before = dict(inherited)
            clean = run_dlc.python_environment(root, python, inherited)
            self.assertEqual(inherited, before)
            self.assertNotIn('PYTHONPATH', clean)
            self.assertNotIn('PYTHONHOME', clean)
            self.assertNotIn('PYTHONUSERBASE', clean)
            for key in run_dlc.PYTHON_LOADER_OVERRIDES:
                self.assertNotIn(key, clean)
            server = run_dlc.runtime_environment(root, inherited)
            self.assertEqual(server['LD_LIBRARY_PATH'], inherited['LD_LIBRARY_PATH'])
            self.assertEqual(server['LD_PRELOAD'], inherited['LD_PRELOAD'])
            nested = ('import json,subprocess,sys,runtime_probe; '
                      'print(json.dumps({"parent":runtime_probe.marker,"child":subprocess.check_output('
                      '[sys.executable,"-c",' + repr(probe) + '],text=True).strip()}))')
            result = json.loads(subprocess.check_output(run_dlc.python_command(python) + ['-c', nested],
                                                       env=clean, cwd=current, text=True))
            self.assertEqual(result, {'parent': 'prepared', 'child': 'prepared'})


if __name__ == '__main__':
    unittest.main()
