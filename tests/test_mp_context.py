import os
import unittest
from unittest.mock import patch

from plastburstalign.task_queue import get_mp_context


class MultiprocessingContextTests(unittest.TestCase):
    def test_windows_context_defaults_to_spawn(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.platform", "win32"):
            ctx = get_mp_context()
            self.assertEqual(ctx.get_start_method(), "spawn")

    def test_darwin_context_defaults_to_spawn(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.platform", "darwin"):
            ctx = get_mp_context()
            self.assertEqual(ctx.get_start_method(), "spawn")

    def test_linux_context_defaults_to_fork(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.platform", "linux"):
            ctx = get_mp_context()
            self.assertEqual(ctx.get_start_method(), "fork")

    def test_environment_override(self):
        with patch.dict(os.environ, {"PBA_MP_CONTEXT": "spawn"}):
            ctx = get_mp_context()
            self.assertEqual(ctx.get_start_method(), "spawn")

    def test_spawn_execution_safety(self):
        """Verify that a task executes under spawn without pickling failures."""
        ctx = get_mp_context(force_context="spawn")
        with ctx.Pool(processes=1) as pool:
            result = pool.apply(abs, [-42])
            self.assertEqual(result, 42)


if __name__ == "__main__":
    unittest.main()
