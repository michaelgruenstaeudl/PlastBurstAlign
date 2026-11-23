from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock

from plastburstalign.alignment_ops import AlignmentCoordination


class TestLPTScheduling(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = self.temp_dir.name

        # Mock PlastidData and user_params
        self.mock_plastid_data = MagicMock()
        self.mock_plastid_data.nucleotides = {
            "small_gene": None,
            "huge_gene": None,
            "medium_gene": None,
            "equal_b": None,
            "equal_a": None,
        }
        self.user_params = {"out_dir": self.out_dir, "num_threads": 4}

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_unaligned_file(self, name: str, size_bytes: int):
        path = os.path.join(self.out_dir, f"nucl_{name}.unalign.fasta")
        with open(path, "wb") as f:
            f.write(b"A" * size_bytes)

    def test_lpt_sorts_descending_by_file_size(self):
        # Create test unaligned files with distinct sizes
        self._create_unaligned_file("small_gene", 100)
        self._create_unaligned_file("medium_gene", 5_000)
        self._create_unaligned_file("huge_gene", 100_000)

        coord = AlignmentCoordination(self.mock_plastid_data, self.user_params)
        sorted_tasks = coord._get_lpt_sorted_regions(prefix="nucl")

        # huge_gene must come first, followed by medium, then small
        self.assertEqual(sorted_tasks[0], "huge_gene")
        self.assertEqual(sorted_tasks[1], "medium_gene")
        self.assertEqual(sorted_tasks[2], "small_gene")

    def test_deterministic_alphabetical_tie_breaking(self):
        # Two files of identical size must tie-break alphabetically
        self._create_unaligned_file("equal_b", 2_000)
        self._create_unaligned_file("equal_a", 2_000)

        self.mock_plastid_data.nucleotides = {
            "equal_b": None,
            "equal_a": None,
        }

        coord = AlignmentCoordination(self.mock_plastid_data, self.user_params)
        sorted_tasks = coord._get_lpt_sorted_regions(prefix="nucl")

        self.assertEqual(sorted_tasks, ["equal_a", "equal_b"])

    def test_missing_files_placed_at_tail_without_error(self):
        # huge_gene exists; missing_gene does not exist on disk
        self._create_unaligned_file("huge_gene", 50_000)
        self.mock_plastid_data.nucleotides = {
            "missing_gene": None,
            "huge_gene": None,
        }

        coord = AlignmentCoordination(self.mock_plastid_data, self.user_params)
        sorted_tasks = coord._get_lpt_sorted_regions(prefix="nucl")

        self.assertEqual(sorted_tasks[0], "huge_gene")
        self.assertEqual(sorted_tasks[1], "missing_gene")


if __name__ == "__main__":
    unittest.main()
