"""Short-sequence path of ``MAFFT.align`` with the real MAFFT.

PartTree aborts on sequences shorter than 6 residues. When a PartTree tier is
chosen and such sequences exist, the others are aligned with the tier and the
short ones are added with ``--addfragments`` (``--add`` above
``addfragments_max_n`` sequences). The output must keep the original input row
order and headers, every ungapped row must equal its input sequence, and all
rows must have the same length. The PartTree tier is forced here through the
``tier`` override (the data are tiny).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

from plastburstalign._simulate import simulate_locus
from plastburstalign.alignment_tools import MAFFT

HAS_MAFFT = shutil.which("mafft") is not None


def read_fasta(path: str) -> list[tuple[str, str]]:
    """``[(header, sequence), ...]`` in file order."""
    records: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                records.append((line[1:], []))
            elif line and records:
                records[-1][1].append(line)
    return [(h, "".join(p)) for h, p in records]


def write_fasta(path: str, records: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(f">{header}\n{seq}\n")


@unittest.skipUnless(HAS_MAFFT, "mafft not available")
class FragmentPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.aligners: list[MAFFT] = []

    def aligner(self, **tool_params: Any) -> MAFFT:
        params = {"alignment_tool_config": {"tier": "fast_tree", **tool_params}}
        mafft = MAFFT(params)
        self.aligners.append(mafft)
        self.addCleanup(mafft.scratch_mgr.cleanup)
        return mafft

    def path(self, name: str) -> str:
        return os.path.join(self.tmp.name, name)

    def locus(
        self, n_long: int = 24, alphabet: str = "nucl", seed: int = 3
    ) -> list[tuple[str, str]]:
        recs = simulate_locus(
            n_long,
            140 if alphabet == "nucl" else 80,
            alphabet=alphabet,
            divergence=0.15,
            seed=seed,
        )
        return [(f"{name} some description", seq) for name, seq in recs]

    def assert_alignment_of(
        self, source: list[tuple[str, str]], output: str, *, case: bool = False
    ) -> list[tuple[str, str]]:
        rows = read_fasta(output)
        self.assertEqual([h for h, _ in rows], [h for h, _ in source])  # order + ids
        self.assertEqual(len({len(s) for _, s in rows}), 1, "rows differ in length")
        for (header, row), (_, original) in zip(rows, source):
            ungapped = row.replace("-", "")
            if case:
                self.assertEqual(ungapped, original, header)
            else:
                self.assertEqual(ungapped.upper(), original.upper(), header)
        return rows

    def test_short_sequences_are_added_and_order_is_preserved(self) -> None:
        records = self.locus()
        # short sequences of 1..5 residues at the front, the middle and the end
        records.insert(0, ("short1 first", "A"))
        records.insert(9, ("short3 middle", "ACG"))
        records.insert(15, ("short5 middle", "TTGCA"))
        records.append(("short2 last", "GT"))
        src = self.path("in.fa")
        out = self.path("out.fa")
        write_fasta(src, records)
        mafft = self.aligner()
        with patch.object(
            MAFFT, "_build_command", side_effect=AssertionError("single-command path")
        ):
            self.assertTrue(mafft.align(src, out, threads=2))
        rows = self.assert_alignment_of(records, out)
        self.assertEqual(len(rows), len(records))
        self.assertGreater(len(rows[1][1]), 100)  # a real alignment, not a stub

    def test_uses_the_chosen_tier_and_addfragments(self) -> None:
        records = self.locus(12)
        records.append(("tiny", "ACG"))
        src = self.path("in.fa")
        write_fasta(src, records)
        mafft = self.aligner(quiet=True)
        commands: list[list[str]] = []
        original = MAFFT._run_to_file

        def recording(
            self_: MAFFT, cmd: list[str], output_file: str, env=None, **kwargs
        ) -> bool:
            commands.append(list(cmd))
            return original(self_, cmd, output_file, env=env, **kwargs)

        with patch.object(MAFFT, "_run_to_file", recording):
            self.assertTrue(mafft.align(src, self.path("out.fa"), threads=3))
        self.assertEqual(len(commands), 2)
        main_cmd, add_cmd = commands
        self.assertEqual(
            main_cmd[1:8],
            ["--parttree", "--retree", "2", "--maxiterate", "0", "--thread", "3"],
        )
        self.assertNotIn("--auto", main_cmd)
        self.assertEqual(add_cmd[1], "--addfragments")
        self.assertIn("--thread", add_cmd)
        self.assertEqual(add_cmd[add_cmd.index("--thread") + 1], "3")
        self.assertNotIn("--keeplength", add_cmd)  # would delete residues

    def test_add_mode_is_used_above_the_addfragments_cap(self) -> None:
        records = self.locus(12)
        records.insert(4, ("tiny", "ACG"))
        src = self.path("in.fa")
        out = self.path("out.fa")
        write_fasta(src, records)
        mafft = self.aligner(addfragments_max_n=5)
        commands: list[list[str]] = []
        original = MAFFT._run_to_file

        def recording(
            self_: MAFFT, cmd: list[str], output_file: str, env=None, **kwargs
        ) -> bool:
            commands.append(list(cmd))
            return original(self_, cmd, output_file, env=env, **kwargs)

        with patch.object(MAFFT, "_run_to_file", recording):
            self.assertTrue(mafft.align(src, out, threads=1))
        self.assertEqual(commands[1][1], "--add")
        self.assert_alignment_of(records, out)

    def test_empty_sequences_become_gap_rows(self) -> None:
        records = self.locus(10)
        records.insert(3, ("empty one", ""))
        records.insert(5, ("tiny", "AC"))
        records.append(("empty two", ""))
        src = self.path("in.fa")
        out = self.path("out.fa")
        write_fasta(src, records)
        self.assertTrue(self.aligner().align(src, out, threads=1))
        rows = self.assert_alignment_of(records, out)
        self.assertEqual(set(rows[3][1]), {"-"})
        self.assertEqual(set(rows[-1][1]), {"-"})

    def test_empty_sequences_only_need_no_second_mafft_run(self) -> None:
        records = self.locus(8)
        records.append(("empty", ""))
        src = self.path("in.fa")
        write_fasta(src, records)
        commands: list[list[str]] = []
        original = MAFFT._run_to_file

        def recording(
            self_: MAFFT, cmd: list[str], output_file: str, env=None, **kwargs
        ) -> bool:
            commands.append(list(cmd))
            return original(self_, cmd, output_file, env=env, **kwargs)

        with patch.object(MAFFT, "_run_to_file", recording):
            self.assertTrue(self.aligner().align(src, self.path("out.fa")))
        self.assertEqual(len(commands), 1)

    def test_protein_alphabet_by_prefix_and_explicitly(self) -> None:
        records = self.locus(14, alphabet="prot")
        records.insert(2, ("tinyprot", "MKV"))
        records.append(("tinyprot2", "WY"))
        for name, kwargs in (("prot_x.fa", {}), ("other.fa", {"alphabet": "prot"})):
            with self.subTest(file=name):
                src = self.path(name)
                out = self.path("out_" + name)
                write_fasta(src, records)
                self.assertTrue(self.aligner().align(src, out, threads=1, **kwargs))
                self.assert_alignment_of(records, out)

    def test_hints_are_used_without_scanning(self) -> None:
        records = self.locus(10)
        records.append(("tiny", "ACG"))
        src = self.path("in.fa")
        out = self.path("out.fa")
        write_fasta(src, records)
        mafft = self.aligner()
        with patch.object(MAFFT, "_scan_fasta_meta", side_effect=AssertionError):
            self.assertTrue(
                mafft.align(
                    src, out, threads=1, sequence_count=11, max_len=150, min_len=3
                )
            )
        self.assert_alignment_of(records, out)

    def test_output_is_deterministic_and_thread_independent(self) -> None:
        records = self.locus(30, seed=9)
        records.insert(7, ("tiny", "ACG"))
        records.insert(20, ("tiny2", "T"))
        src = self.path("in.fa")
        write_fasta(src, records)
        outputs = []
        for index, threads in enumerate((1, 1, 2)):
            out = self.path(f"out{index}.fa")
            self.assertTrue(self.aligner().align(src, out, threads=threads))
            with open(out, "rb") as handle:
                outputs.append(handle.read())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_no_stray_files_are_left_in_scratch(self) -> None:
        records = self.locus(8)
        records.append(("tiny", "AC"))
        src = self.path("in.fa")
        write_fasta(src, records)
        mafft = self.aligner()
        self.assertTrue(mafft.align(src, self.path("out.fa")))
        self.assertEqual(os.listdir(mafft.scratch_mgr.path), [])

    def test_accurate_tier_keeps_the_single_command_auto_path(self) -> None:
        # With --auto (small W, below the N cap) short sequences are fine.
        records = self.locus(10)
        records.append(("tiny", "ACG"))
        src = self.path("in.fa")
        out = self.path("out.fa")
        write_fasta(src, records)
        mafft = MAFFT()
        self.addCleanup(mafft.scratch_mgr.cleanup)
        with patch.object(
            MAFFT, "_align_with_fragments", side_effect=AssertionError("fragments")
        ):
            self.assertTrue(mafft.align(src, out, threads=1))
        self.assert_alignment_of(records, out)

    def test_without_two_long_sequences_small_loci_fall_back_to_auto(self) -> None:
        records = [("a", "ACG"), ("b", "AC"), ("c", "ACGTACGTAC"), ("d", "T")]
        src = self.path("in.fa")
        out = self.path("out.fa")
        write_fasta(src, records)
        self.assertTrue(self.aligner().align(src, out, threads=1))
        self.assert_alignment_of(records, out)

    def test_without_two_long_sequences_large_n_is_a_clear_error(self) -> None:
        records = [("a", "ACG"), ("b", "AC"), ("c", "ACGTACGTAC"), ("d", "T")]
        src = self.path("in.fa")
        write_fasta(src, records)
        mafft = self.aligner(n_force_parttree=3)
        with self.assertLogs("plastburstalign.logging_ops", level="ERROR") as logs:
            self.assertFalse(mafft.align(src, self.path("out.fa"), threads=1))
        self.assertTrue(any("PartTree needs at least 2" in m for m in logs.output))
        self.assertFalse(os.path.exists(self.path("out.fa")))

    def test_failing_main_alignment_returns_false(self) -> None:
        records = self.locus(6)
        records.append(("tiny", "AC"))
        src = self.path("in.fa")
        write_fasta(src, records)
        mafft = self.aligner()
        with patch.object(MAFFT, "_run_to_file", return_value=False):
            self.assertFalse(mafft.align(src, self.path("out.fa")))
        self.assertEqual(os.listdir(mafft.scratch_mgr.path), [])

    def test_larger_parttree_sized_set_with_short_sequences(self) -> None:
        # a few hundred taxa through the default (work/N-cap driven) path
        records = [
            (f"{n} d", s) for n, s in simulate_locus(300, 120, divergence=0.2, seed=5)
        ]
        records.insert(100, ("tiny", "ACG"))
        records.insert(250, ("tiny2", "GATTA"))
        src = self.path("in.fa")
        out = self.path("out.fa")
        write_fasta(src, records)
        mafft = self.aligner(tier=None, n_force_parttree=100)  # N cap drives the tier
        self.assertTrue(mafft.align(src, out, threads=2))
        self.assert_alignment_of(records, out)


if __name__ == "__main__":
    unittest.main()
