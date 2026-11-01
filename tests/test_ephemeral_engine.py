from __future__ import annotations

import unittest
from pathlib import Path

from plastburstalign.alignment_tools import MAFFT
from plastburstalign.ephemeral_engine import EphemeralScratchManager


class TestEphemeralScratchEngine(unittest.TestCase):
    def test_scratch_allocates_and_cleans_up(self):
        mgr = EphemeralScratchManager()
        scratch_path = mgr.path
        self.assertTrue(scratch_path.exists())
        self.assertTrue(scratch_path.is_dir())

        # Subprocess environment must set TMPDIR
        env = mgr.get_subprocess_env()
        self.assertIn("TMPDIR", env)
        self.assertEqual(env["TMPDIR"], str(scratch_path))

        # Explicit cleanup removes scratch
        mgr.cleanup()
        self.assertFalse(scratch_path.exists())

    def test_custom_scratch_directory_override(self):
        with EphemeralScratchManager(custom_scratch_dir="/tmp") as mgr:
            self.assertEqual(mgr.path.parent, Path("/tmp").resolve())
            self.assertTrue(mgr.path.name.startswith("pba_"))
            self.assertTrue(mgr.path.exists())

    def test_mafft_aligner_passes_ephemeral_tmpdir(self):
        mafft = MAFFT()
        env = mafft._get_subprocess_env()
        self.assertIn("TMPDIR", env)
        self.assertTrue(Path(env["TMPDIR"]).exists())
        # Clean up
        mafft.scratch_mgr.cleanup()


if __name__ == "__main__":
    unittest.main()
