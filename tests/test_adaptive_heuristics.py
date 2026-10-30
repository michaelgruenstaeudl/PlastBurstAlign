"""MAFFT command construction with the work-based adaptive strategy.

The algorithm is a pure function of ``(N, L_min, L_max, alphabet)`` and explicit
overrides (see ``strategy.py``); the pre-v3 N-count gates are gone. Unit tests
of the strategy model itself live in ``tests/test_strategy.py``; real-MAFFT
runs of the short-sequence path in ``tests/test_mafft_fragments.py``.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from typing import Any

from plastburstalign import strategy as strategy_module
from plastburstalign.alignment_tools import MAFFT
from plastburstalign.strategy import TIER_MAFFT_ARGS, AlignmentTier

PROTEIN = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEH"


def algorithm_args(cmd: list[str]) -> list[str]:
    """The algorithm part of a command (between ``mafft`` and ``--thread``)."""
    return cmd[1 : cmd.index("--thread")]


class TestAdaptiveHeuristics(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.in_file = os.path.join(self.temp_dir.name, "test.fasta")
        strategy_module._reset_deprecation_warnings()
        self.addCleanup(strategy_module._reset_deprecation_warnings)

    def _create_fasta(self, n_seq: int, seq: str = "ATGCATGC") -> None:
        with open(self.in_file, "w") as f:
            for i in range(n_seq):
                f.write(f">seq_{i}\n{seq}\n")

    def _mafft(self, **tool_params: Any) -> MAFFT:
        params: dict[str, Any] = {}
        if tool_params:
            params = {"alignment_tool_config": tool_params}
        return MAFFT(params)

    # --- work-based tiers ------------------------------------------------

    def test_small_dataset_uses_auto(self) -> None:
        self._create_fasta(50)
        cmd = MAFFT()._build_command(self.in_file)
        self.assertIn("--auto", cmd)
        self.assertNotIn("--parttree", cmd)

    def test_n_count_gates_are_gone(self) -> None:
        # The old code switched to PartTree at N >= 1000 whatever the length;
        # 1200 tiny sequences are cheap and keep the accurate algorithm.
        self._create_fasta(1200)
        cmd = MAFFT()._build_command(self.in_file)
        self.assertIn("--auto", cmd)
        self.assertNotIn("--parttree", cmd)

    def test_fixed_n_cap_forces_parttree_for_memory_safety(self) -> None:
        # N >= N_FORCE_PARTTREE (5000): O(N^2) distance matrix. Short sequences
        # keep W small, so the cap alone selects the lightest PartTree tier.
        self._create_fasta(5500)
        cmd = MAFFT()._build_command(self.in_file)
        self.assertEqual(
            algorithm_args(cmd), list(TIER_MAFFT_ARGS[AlignmentTier.FAST_TREE])
        )
        self.assertNotIn("--auto", cmd)

    def test_work_selects_every_tier(self) -> None:
        mafft = MAFFT()
        cases = [
            (50, 3000, AlignmentTier.ACCURATE),
            (733, 600, AlignmentTier.ACCURATE),
            (156, 3500, AlignmentTier.FAST_TREE),
            (733, 2500, AlignmentTier.HIGH_SCALE),
            (700, 3500, AlignmentTier.EXTREME_SCALE),
            (13724, 1500, AlignmentTier.EXTREME_SCALE),
        ]
        for n, length, tier in cases:
            with self.subTest(n=n, length=length):
                cmd = mafft._build_command(
                    "dummy.fasta", sequence_count=n, max_len=length, min_len=length
                )
                self.assertEqual(algorithm_args(cmd), list(TIER_MAFFT_ARGS[tier]))

    def test_sequence_count_and_max_len_passed_avoid_scan(self) -> None:
        # Passing the statistics directly avoids reading (a missing) input_file
        mafft = MAFFT()
        cmd = mafft._build_command(
            "non_existent_file.fasta", sequence_count=13724, max_len=1500
        )
        self.assertEqual(
            algorithm_args(cmd), list(TIER_MAFFT_ARGS[AlignmentTier.EXTREME_SCALE])
        )

    def test_sequence_count_without_length_applies_the_n_cap_only(self) -> None:
        cmd = MAFFT()._build_command("non_existent_file.fasta", sequence_count=13724)
        self.assertIn("--parttree", cmd)

    def test_flop_complexity_triggers_parttree_for_mid_sized_long_locus(self) -> None:
        # N=733 with L=2500: W = 1.05e11 -> HIGH_SCALE (PartTree2, no FFT)
        cmd = MAFFT()._build_command("dummy.fasta", sequence_count=733, max_len=2500)
        self.assertEqual(
            algorithm_args(cmd), list(TIER_MAFFT_ARGS[AlignmentTier.HIGH_SCALE])
        )
        self.assertIn("--nofft", cmd)
        self.assertEqual(cmd[cmd.index("--retree") + 1], "2")

    def test_flop_complexity_retains_auto_for_mid_sized_short_locus(self) -> None:
        cmd = MAFFT()._build_command("dummy.fasta", sequence_count=733, max_len=600)
        self.assertIn("--auto", cmd)
        self.assertNotIn("--parttree", cmd)

    def test_synthetic_monocot_scale_gets_a_high_tier(self) -> None:
        cmd = MAFFT()._build_command("dummy.fasta", sequence_count=700, max_len=3500)
        self.assertIn("--parttree", cmd)
        self.assertIn("--nofft", cmd)

    # --- determinism -----------------------------------------------------

    def test_algorithm_is_independent_of_the_thread_count(self) -> None:
        mafft = MAFFT()
        for n, length in ((50, 3000), (733, 2500), (13724, 1500)):
            reference = algorithm_args(
                mafft._build_command(
                    "d.fa", threads=1, sequence_count=n, max_len=length
                )
            )
            for threads in (2, 8, 64):
                cmd = mafft._build_command(
                    "d.fa", threads=threads, sequence_count=n, max_len=length
                )
                self.assertEqual(algorithm_args(cmd), reference)
                self.assertEqual(cmd[cmd.index("--thread") + 1], str(threads))

    # --- explicit configuration -----------------------------------------

    def test_explicit_algorithm_overrides_heuristics(self) -> None:
        self._create_fasta(5500)
        cmd = self._mafft(algorithm="localpair")._build_command(self.in_file)
        self.assertIn("--localpair", cmd)
        self.assertNotIn("--parttree", cmd)
        self.assertNotIn("--auto", cmd)

    def test_explicit_retree_and_maxiterate(self) -> None:
        mafft = self._mafft(algorithm="retree", retree=2)
        cmd = mafft._build_command("d.fa", sequence_count=13724, max_len=1500)
        self.assertEqual(cmd[1:3], ["--retree", "2"])
        mafft = self._mafft(algorithm="maxiterate", maxiterate=1000)
        cmd = mafft._build_command("d.fa", sequence_count=13724, max_len=1500)
        self.assertEqual(cmd[1:3], ["--maxiterate", "1000"])

    def test_disabled_heuristics_falls_back_to_auto(self) -> None:
        self._create_fasta(5500)
        mafft = MAFFT({"adaptive_heuristics": False})
        cmd = mafft._build_command(self.in_file)
        self.assertIn("--auto", cmd)
        self.assertNotIn("--parttree", cmd)

    def test_disabled_heuristics_via_tool_config(self) -> None:
        cmd = self._mafft(adaptive_heuristics=False)._build_command(
            "d.fa", sequence_count=13724, max_len=1500
        )
        self.assertEqual(algorithm_args(cmd), ["--auto"])

    def test_explicit_work_thresholds_are_honoured(self) -> None:
        # pre-v3 names still work: parttree = first boundary, retree1 = extreme
        mafft = self._mafft(work_parttree_threshold=1.0e8, work_retree1_threshold=1.0e9)
        cmd = mafft._build_command("d.fa", sequence_count=733, max_len=1000)
        self.assertEqual(
            algorithm_args(cmd), list(TIER_MAFFT_ARGS[AlignmentTier.EXTREME_SCALE])
        )
        mafft = self._mafft(
            work_fast_tree_threshold=1.0e8, work_high_scale_threshold=1e12
        )
        cmd = mafft._build_command("d.fa", sequence_count=100, max_len=1000)
        self.assertEqual(
            algorithm_args(cmd), list(TIER_MAFFT_ARGS[AlignmentTier.FAST_TREE])
        )
        mafft = self._mafft(tier="high_scale")
        cmd = mafft._build_command("d.fa", sequence_count=5, max_len=100)
        self.assertEqual(
            algorithm_args(cmd), list(TIER_MAFFT_ARGS[AlignmentTier.HIGH_SCALE])
        )

    def test_legacy_count_thresholds_only_act_when_explicit(self) -> None:
        self._create_fasta(20)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cmd = self._mafft(
                parttree_threshold=10, retree1_threshold=100
            )._build_command(self.in_file)
        self.assertEqual(
            algorithm_args(cmd), list(TIER_MAFFT_ARGS[AlignmentTier.HIGH_SCALE])
        )
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "explicit legacy keys must warn",
        )
        cmd = self._mafft()._build_command(self.in_file)
        self.assertEqual(algorithm_args(cmd), ["--auto"])

    # --- alphabet --------------------------------------------------------

    def test_alphabet_defaults_to_filename_prefix(self) -> None:
        mafft = self._mafft(adjustdirection=True)
        nucl = mafft._build_command("nucl_x.fa", sequence_count=5, max_len=50)
        prot = mafft._build_command("prot_x.fa", sequence_count=5, max_len=50)
        self.assertIn("--adjustdirection", nucl)
        self.assertNotIn("--adjustdirection", prot)

    def test_explicit_alphabet_wins_over_the_filename(self) -> None:
        mafft = self._mafft(adjustdirection=True)
        cmd = mafft._build_command(
            "whatever.fa", sequence_count=5, max_len=50, alphabet="prot"
        )
        self.assertNotIn("--adjustdirection", cmd)
        cmd = mafft._build_command(
            "prot_x.fa", sequence_count=5, max_len=50, alphabet="nucl"
        )
        self.assertIn("--adjustdirection", cmd)
        with self.assertRaises(ValueError):
            mafft._build_command(
                "x.fa", sequence_count=5, max_len=50, alphabet="klingon"
            )

    def test_command_layout(self) -> None:
        mafft = self._mafft(quiet=True)
        cmd = mafft._build_command(
            "in.fa", threads=3, sequence_count=5, max_len=50, alphabet="nucl"
        )
        self.assertEqual(cmd, ["mafft", "--auto", "--thread", "3", "--quiet", "in.fa"])

    # --- short-sequence guard -------------------------------------------

    def test_short_sequence_flags_the_fragment_path_for_parttree_tiers(self) -> None:
        with open(self.in_file, "w") as f:
            for i in range(5000):
                f.write(f">seq_{i}\n{PROTEIN}\n")
            # one short 5-residue sequence ending with a stop *
            f.write(">short_seq\nMVLSS*\n")
        mafft = MAFFT()
        resolved = mafft._resolve_strategy(self.in_file, None, None, None, None)
        self.assertEqual(resolved.n_seq, 5001)
        self.assertEqual(resolved.min_len, 5)
        self.assertTrue(resolved.strategy.uses_parttree)
        self.assertTrue(resolved.strategy.needs_fragment_split)

    def test_single_command_view_bypasses_to_auto_for_short_sequences(self) -> None:
        # A single command cannot express "PartTree + --addfragments"; the
        # bypass is the legacy single-command view. align() uses the real path
        # (see tests/test_mafft_fragments.py).
        with open(self.in_file, "w") as f:
            for i in range(5000):
                f.write(f">seq_{i}\n{PROTEIN}\n")
            f.write(">short_seq\nMVLSS*\n")
        cmd = MAFFT()._build_command(self.in_file)
        self.assertNotIn("--parttree", cmd)
        self.assertIn("--auto", cmd)

    def test_parttree_is_kept_when_all_sequences_are_long_enough(self) -> None:
        with open(self.in_file, "w") as f:
            for i in range(5000):
                f.write(f">seq_{i}\n{PROTEIN}\n")
        resolved = MAFFT()._resolve_strategy(self.in_file, None, None, None, None)
        self.assertTrue(resolved.strategy.uses_parttree)
        self.assertFalse(resolved.strategy.needs_fragment_split)

    def test_no_short_sequence_lookup_for_the_accurate_tier(self) -> None:
        # min_len is only needed (and scanned for) when a PartTree tier is chosen
        self._create_fasta(10)
        scans: list[str] = []
        original = MAFFT._scan_fasta_meta

        def counting_scan(path: str) -> tuple[int, int, int]:
            scans.append(path)
            return original(path)

        MAFFT._scan_fasta_meta = staticmethod(counting_scan)  # type: ignore[method-assign]
        try:
            resolved = MAFFT()._resolve_strategy(self.in_file, 10, 8, None, None)
        finally:
            MAFFT._scan_fasta_meta = staticmethod(original)  # type: ignore[method-assign]
        self.assertEqual(scans, [])
        self.assertFalse(resolved.strategy.needs_fragment_split)


if __name__ == "__main__":
    unittest.main()
