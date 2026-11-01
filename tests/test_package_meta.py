"""Package metadata, typing marker and import-order robustness."""

from __future__ import annotations

import importlib
import subprocess
import sys
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import plastburstalign
from plastburstalign import _meta

REPO_ROOT = Path(__file__).resolve().parents[1]


class PackageMetaTests(unittest.TestCase):
    def test_package_name_is_not_overridden(self) -> None:
        # The stray `__name__ = "plastburstalign"` assignment is gone; the module
        # name is simply the real import name.
        self.assertEqual(plastburstalign.__name__, "plastburstalign")
        source = (REPO_ROOT / "plastburstalign" / "__init__.py").read_text()
        self.assertNotIn("__name__ =", source)

    def test_metadata_attributes_are_strings(self) -> None:
        self.assertIsInstance(plastburstalign.__version__, str)
        self.assertIsInstance(plastburstalign.__author__, str)
        self.assertIsInstance(plastburstalign.__email__, str)
        self.assertTrue(plastburstalign.__version__)

    def test_version_fallback_is_a_string_when_not_installed(self) -> None:
        with patch("importlib.metadata.version", side_effect=PackageNotFoundError("x")):
            reloaded = importlib.reload(_meta)
            try:
                self.assertEqual(reloaded.__version__, "0+unknown")
                self.assertIsInstance(reloaded.__version__, str)
            finally:
                importlib.reload(_meta)

    def test_logging_ops_shares_the_package_metadata(self) -> None:
        from plastburstalign import logging_ops

        self.assertEqual(logging_ops.__version__, plastburstalign.__version__)
        self.assertEqual(logging_ops.__email__, plastburstalign.__email__)

    def test_logging_ops_can_be_imported_first(self) -> None:
        # Import order must not matter (previously `from . import __email__` relied
        # on __init__ having defined it before importing the submodule).
        code = (
            "import plastburstalign.logging_ops as l, plastburstalign as p;"
            "assert l.__version__ == p.__version__;"
            "print(l.logger.name)"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_existing_exports_are_kept(self) -> None:
        for name in (
            "PlastBurstAlign",
            "UserParameters",
            "PlastidData",
            "ExtractAndCollect",
            "DataCleaning",
            "AlignmentCoordination",
            "MAFFT",
            "ClustalOmega",
            "MUSCLE",
            "align",
            "autotune",
            "estimate",
            "AlignmentResult",
            "TuningPlan",
            "HostProfile",
            "detect_host",
        ):
            self.assertIn(name, plastburstalign.__all__)
            self.assertTrue(hasattr(plastburstalign, name))

    def test_py_typed_marker_ships_with_the_package(self) -> None:
        self.assertTrue((REPO_ROOT / "plastburstalign" / "py.typed").is_file())
        pyproject = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertIn("py.typed", pyproject)


if __name__ == "__main__":
    unittest.main()
