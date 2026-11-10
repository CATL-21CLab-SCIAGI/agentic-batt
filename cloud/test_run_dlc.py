"""CPU-only launcher cleanup checks; no cloud, GPU, or model calls."""
import signal
import subprocess
import unittest
from unittest import mock

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


if __name__ == '__main__':
    unittest.main()
