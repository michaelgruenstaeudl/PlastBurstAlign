"""Parity-gate wrapper and unit tests for ``benchmarking/golden_parity.py``.

The slow end-to-end part runs the real CLI on the Asteraceae subset for the
fastest mode only (``rna``, about 2 s). Set ``PBA_GOLDEN_MODES`` (for example
``"cds igs int rna"``) to widen it; the full four-mode gate takes about
1-2 minutes and is normally run by hand::

    python benchmarking/golden_parity.py check
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarking" / "golden_parity.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("golden_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # do not create benchmarking/__pycache__
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


gate = _load_gate()


class CanonicalizationTests(unittest.TestCase):
    """Fast, dataset-free checks of the canonicalization helpers."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_alignment_hash_ignores_row_order_and_case(self) -> None:
        a = [("t2", "ac-gt"), ("t1", "ACCGT")]
        b = [("t1", "accgt"), ("t2", "AC-GT")]
        self.assertEqual(
            gate.canonical_alignment(a)["sha256"],
            gate.canonical_alignment(b)["sha256"],
        )

    def test_alignment_hash_detects_content_change(self) -> None:
        a = [("t1", "ACCGT"), ("t2", "AC-GT")]
        b = [("t1", "ACCGT"), ("t2", "ACTGT")]
        fa, fb = gate.canonical_alignment(a), gate.canonical_alignment(b)
        self.assertNotEqual(fa["sha256"], fb["sha256"])
        self.assertNotEqual(fa["ungapped_sha256"], fb["ungapped_sha256"])

    def test_realignment_keeps_ungapped_hash(self) -> None:
        a = [("t1", "AC-GT"), ("t2", "ACCGT")]
        b = [("t1", "ACG-T"), ("t2", "ACCGT")]
        fa, fb = gate.canonical_alignment(a), gate.canonical_alignment(b)
        self.assertNotEqual(fa["sha256"], fb["sha256"])
        self.assertEqual(fa["ungapped_sha256"], fb["ungapped_sha256"])

    def test_read_fasta_wrapped_lines(self) -> None:
        path = self._write("w.fasta", ">a desc\nAC\nGT\n\n>b\nTT\n")
        self.assertEqual(gate.read_fasta(path), [("a", "ACGT"), ("b", "TT")])

    def test_concat_hash_independent_of_partition_order(self) -> None:
        # Two partitions: g1 (3 columns), g2 (2 columns); same data, swapped order.
        fa1 = self._write("c1.fasta", ">t1\nAAACC\n>t2\nGGGTT\n")
        p1 = self._write("c1.txt", "DNA, g1 = 1-3\nDNA, g2 = 4-5\n")
        fa2 = self._write("c2.fasta", ">t2\nTTGGG\n>t1\nCCAAA\n")
        p2 = self._write("c2.txt", "DNA, g2 = 1-2\nDNA, g1 = 3-5\n")
        c1 = gate.canonical_concat(fa1, p1)
        c2 = gate.canonical_concat(fa2, p2)
        self.assertEqual(c1["sha256"], c2["sha256"])
        self.assertNotEqual(c1["ordered_sha256"], c2["ordered_sha256"])
        self.assertEqual(c1["partitions"], {"g1": 3, "g2": 2})
        self.assertTrue(c1["partitions_cover_matrix"])

    def test_diff_mode_reports_alignment_only_difference(self) -> None:
        base = {
            "returncode": 0,
            "locus_files": ["nucl_a.aligned.fasta"],
            "nexus_files": [],
            "unalign_files": [],
            "loci": {
                "nucl_a.aligned.fasta": {
                    "sha256": "x",
                    "ungapped_sha256": "u",
                    "ntax": 3,
                    "nchar": 10,
                }
            },
            "concat": None,
        }
        cur = {
            **base,
            "loci": {
                "nucl_a.aligned.fasta": {
                    "sha256": "y",
                    "ungapped_sha256": "u",
                    "ntax": 3,
                    "nchar": 10,
                }
            },
        }
        problems = gate.diff_mode("cds", base, cur)
        self.assertEqual(len(problems), 1)
        self.assertIn("ALIGNMENT differs", problems[0])
        self.assertEqual(gate.diff_mode("cds", base, base), [])


@unittest.skipUnless(shutil.which("mafft"), "mafft not available")
@unittest.skipUnless(gate.DEFAULT_DATASET.is_dir(), "Asteraceae dataset not present")
@unittest.skipUnless(gate.DEFAULT_GOLDEN.is_file(), "golden parity file missing")
class GoldenParityCheckTests(unittest.TestCase):
    """Runs the real CLI for the fastest mode(s) and diffs against the golden."""

    def test_check_fast_modes(self) -> None:
        modes = os.environ.get("PBA_GOLDEN_MODES", "rna").split()
        exit_code = gate.main(["check", "--modes", *modes])
        self.assertEqual(exit_code, 0, "golden parity check failed; see output")


if __name__ == "__main__":
    unittest.main()
