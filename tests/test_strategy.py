"""Unit tests for the work-based strategy engine (strategy.py, strategy_constants.py)."""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from plastburstalign import strategy as strategy_module
from plastburstalign.strategy import (
    TIER_MAFFT_ARGS,
    AlignmentStrategy,
    AlignmentTier,
    LocusProfile,
    determine_alignment_args,
    determine_alignment_strategy,
    estimate_work,
    estimate_work_terms,
    normalize_alphabet,
)
from plastburstalign.strategy_constants import (
    DEFAULT_CONSTANTS,
    N_FORCE_PARTTREE,
    AlphabetModel,
    StrategyConstants,
)

A = AlignmentTier.ACCURATE
F = AlignmentTier.FAST_TREE
H = AlignmentTier.HIGH_SCALE
E = AlignmentTier.EXTREME_SCALE


class TierTableTests(unittest.TestCase):
    """Representative loci must land in sensible tiers (nucleotides)."""

    def test_tiers_of_representative_loci(self) -> None:
        # (N, L_max, expected tier(s), why)
        table: list[tuple[int, int, set[AlignmentTier], str]] = [
            (50, 1000, {A}, "small, short"),
            (50, 3000, {A}, "50 taxa x 3 kb (blueprint wrongly made it extreme)"),
            (100, 1000, {A}, "100 taxa x 1 kb (blueprint wrongly gave --nofft)"),
            (100, 2000, {A}, "100 taxa x 2 kb"),
            (100, 3000, {A}, "100 taxa x 3 kb stays --auto"),
            (156, 2000, {A}, "Asteraceae-sized dataset, 2 kb"),
            (733, 600, {A}, "mid N, short locus"),
            (1200, 1000, {A}, "old N>=1000 gate no longer applies"),
            (4999, 300, {A}, "just below the fixed N cap"),
            (156, 3500, {F}, "long locus: first heuristic band"),
            (733, 2500, {H}, "mid N, long locus"),
            (700, 3500, {H, E}, "synthetic Monocot IGS: a high tier"),
            (13724, 1500, {E}, "13.7k taxa x 1.5 kb"),
            (13724, 100, {F, H, E}, "N cap: never --auto"),
            (5000, 10, {F, H, E}, "N cap reached at N = 5000"),
            (100, 8000, {E}, "very long locus"),
        ]
        for n, length, allowed, why in table:
            with self.subTest(n=n, length=length, why=why):
                got = determine_alignment_strategy(n, length, length)
                self.assertIn(got.tier, allowed, got.reason)

    def test_tier_arguments_match_the_blueprint(self) -> None:
        self.assertEqual(TIER_MAFFT_ARGS[A], ("--auto",))
        self.assertEqual(
            TIER_MAFFT_ARGS[F], ("--parttree", "--retree", "2", "--maxiterate", "0")
        )
        self.assertEqual(
            TIER_MAFFT_ARGS[H],
            ("--parttree", "--retree", "2", "--maxiterate", "0", "--nofft"),
        )
        self.assertEqual(
            TIER_MAFFT_ARGS[E],
            ("--parttree", "--retree", "1", "--maxiterate", "0", "--nofft"),
        )
        for tier in AlignmentTier:
            self.assertEqual(
                determine_alignment_strategy(
                    2, 10, 10, overrides={"tier": tier}
                ).mafft_args,
                TIER_MAFFT_ARGS[tier],
            )

    def test_small_n_count_gates_are_gone(self) -> None:
        # The old code used PartTree at N >= 1000 and retree 1 at N >= 5000
        # whatever the length. Now only W (and the fixed cap) decide.
        self.assertIs(determine_alignment_strategy(1000, 300, 300).tier, A)
        self.assertIs(determine_alignment_strategy(4999, 300, 300).tier, A)
        self.assertIsNot(
            determine_alignment_strategy(N_FORCE_PARTTREE, 300, 300).tier, A
        )

    def test_n_cap_forces_at_least_fast_tree_without_inflating_it(self) -> None:
        got = determine_alignment_strategy(N_FORCE_PARTTREE, 100, 100)
        self.assertIs(got.tier, F)
        self.assertIn("N_FORCE_PARTTREE", got.reason)
        self.assertTrue(got.uses_parttree)

    def test_tier_never_decreases_with_n_or_length(self) -> None:
        for length in (50, 300, 1000, 1500, 2500, 4000, 8000):
            previous = A
            for n in (2, 10, 50, 100, 300, 700, 1500, 3000, 4999, 5000, 9000, 20000):
                tier = determine_alignment_strategy(n, length, length).tier
                self.assertGreaterEqual(tier, previous, (n, length))
                previous = tier
        for n in (10, 100, 700, 5000):
            previous = A
            for length in (50, 300, 999, 1000, 2499, 2500, 5000, 20000):
                tier = determine_alignment_strategy(n, length, length).tier
                self.assertGreaterEqual(tier, previous, (n, length))
                previous = tier

    def test_empty_locus_is_accurate(self) -> None:
        for n, length in ((0, 0), (0, 500), (5, 0)):
            self.assertIs(determine_alignment_strategy(n, length).tier, A)
            self.assertEqual(estimate_work(n, length), 0.0)

    def test_protein_alphabet_uses_its_own_constants(self) -> None:
        nucl = determine_alignment_strategy(700, 2500, 2500, alphabet="nucl")
        prot = determine_alignment_strategy(700, 2500, 2500, alphabet="prot")
        self.assertEqual(nucl.tier, prot.tier)  # provisional constants are equal
        custom = dataclasses.replace(
            DEFAULT_CONSTANTS, prot=AlphabetModel(scale=100.0, beta=0.5)
        )
        self.assertEqual(
            estimate_work(700, 2500, alphabet="nucl", constants=custom),
            estimate_work(700, 2500, alphabet="nucl"),
        )
        self.assertGreater(
            estimate_work(700, 2500, alphabet="prot", constants=custom),
            50 * estimate_work(700, 2500, alphabet="prot"),
        )
        self.assertIs(
            determine_alignment_strategy(
                700, 1000, 1000, alphabet="prot", constants=custom
            ).tier,
            E,
        )

    def test_alphabet_aliases(self) -> None:
        for name in ("nucl", "NUC", "dna", "RNA", "nucleotide"):
            self.assertEqual(normalize_alphabet(name), "nucl")
        for name in ("prot", "Protein", "aa"):
            self.assertEqual(normalize_alphabet(name), "prot")
        with self.assertRaises(ValueError):
            normalize_alphabet("morse")
        with self.assertRaises(ValueError):
            determine_alignment_strategy(10, 10, alphabet="morse")


class WorkModelTests(unittest.TestCase):
    def test_primary_term_uses_the_blueprint_exponents(self) -> None:
        # alpha: 2.0 below 1 kb, 2.2 below 2.5 kb, 2.4 otherwise
        for length, alpha in ((999, 2.0), (1000, 2.2), (2499, 2.2), (2500, 2.4)):
            terms = estimate_work_terms(1, length)
            self.assertAlmostEqual(terms.primary, float(length) ** alpha, places=3)
        terms = estimate_work_terms(733, 2500)
        self.assertAlmostEqual(terms.primary, 733 * 2500**2.4, delta=1.0)

    def test_pairwise_term_is_beta_n_squared_l(self) -> None:
        beta = DEFAULT_CONSTANTS.nucl.beta
        terms = estimate_work_terms(13724, 100)
        self.assertAlmostEqual(terms.pairwise, beta * 13724**2 * 100, delta=1.0)
        self.assertAlmostEqual(terms.total, terms.primary + terms.pairwise)
        self.assertEqual(estimate_work(13724, 100), terms.primary + terms.pairwise)

    def test_pairwise_term_dominates_huge_n_short_l(self) -> None:
        # A bare N*L^alpha would call 13.7k x 100 bp trivial (1.4e8).
        terms = estimate_work_terms(13724, 100)
        self.assertGreater(terms.pairwise, 10 * terms.primary)

    def test_pairwise_term_can_change_the_tier_on_its_own(self) -> None:
        # With a large beta the N^2*L term alone moves a short locus out of
        # --auto even below the N cap (default beta keeps this for N >= cap).
        heavy = dataclasses.replace(DEFAULT_CONSTANTS, nucl=AlphabetModel(beta=50.0))
        n, length = 4000, 600
        self.assertIs(determine_alignment_strategy(n, length, length).tier, A)
        with_pairwise = determine_alignment_strategy(n, length, length, constants=heavy)
        self.assertIsNot(with_pairwise.tier, A)
        self.assertGreater(
            estimate_work_terms(n, length, constants=heavy).pairwise,
            estimate_work_terms(n, length, constants=heavy).primary,
        )

    def test_locus_profile_matches_the_functions(self) -> None:
        profile = LocusProfile(n_seq=700, min_len=3000, max_len=3500, alphabet="nucl")
        self.assertEqual(profile.work(), estimate_work(700, 3500))
        self.assertEqual(
            profile.strategy(), determine_alignment_strategy(700, 3500, 3000)
        )


class DeterminismTests(unittest.TestCase):
    def test_repeated_calls_are_identical(self) -> None:
        first = determine_alignment_strategy(700, 3500, 20, alphabet="nucl")
        for _ in range(5):
            self.assertEqual(determine_alignment_strategy(700, 3500, 20), first)
        self.assertEqual(
            json.dumps(first.to_dict(), sort_keys=True),
            json.dumps(
                determine_alignment_strategy(700, 3500, 20).to_dict(), sort_keys=True
            ),
        )

    def test_result_does_not_depend_on_the_host(self) -> None:
        reference = [
            determine_alignment_strategy(n, length, length)
            for n, length in ((100, 3000), (700, 3500), (13724, 1500))
        ]
        with (
            patch("os.cpu_count", return_value=1),
            patch("os.sched_getaffinity", return_value={0}, create=True),
            patch("shutil.disk_usage", side_effect=OSError),
        ):
            again = [
                determine_alignment_strategy(n, length, length)
                for n, length in ((100, 3000), (700, 3500), (13724, 1500))
            ]
        self.assertEqual(reference, again)

    def test_module_imports_no_hardware_or_clock_apis(self) -> None:
        tree = ast.parse(Path(strategy_module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
        forbidden = {
            "os",
            "psutil",
            "time",
            "datetime",
            "multiprocessing",
            "shutil",
            "platform",
            "subprocess",
            "random",
            "socket",
        }
        self.assertEqual(imported & forbidden, set())


class FragmentSplitFlagTests(unittest.TestCase):
    def test_short_sequence_only_matters_for_parttree_tiers(self) -> None:
        self.assertFalse(determine_alignment_strategy(50, 100, 3).needs_fragment_split)
        got = determine_alignment_strategy(13724, 1500, 3)
        self.assertTrue(got.needs_fragment_split)
        self.assertIn("--addfragments", got.reason)
        self.assertFalse(
            determine_alignment_strategy(13724, 1500, 6).needs_fragment_split
        )
        self.assertFalse(
            determine_alignment_strategy(13724, 1500, None).needs_fragment_split
        )
        self.assertTrue(
            determine_alignment_strategy(13724, 1500, 0).needs_fragment_split
        )

    def test_strategy_is_a_frozen_dataclass(self) -> None:
        got = determine_alignment_strategy(10, 10, 10)
        self.assertIsInstance(got, AlignmentStrategy)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            got.work = 1.0  # type: ignore[misc]
        self.assertEqual(
            set(got.to_dict()),
            {
                "tier",
                "mafft_args",
                "work",
                "reason",
                "needs_fragment_split",
            },
        )

    def test_args_wrapper_returns_a_list(self) -> None:
        args = determine_alignment_args(13724, 1500, 1500)
        self.assertIsInstance(args, list)
        self.assertEqual(args, list(TIER_MAFFT_ARGS[E]))
        self.assertEqual(determine_alignment_args(10, 100), ["--auto"])


class OverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        strategy_module._reset_deprecation_warnings()
        self.addCleanup(strategy_module._reset_deprecation_warnings)

    def test_forced_tier(self) -> None:
        for spelling in ("extreme_scale", "EXTREME-SCALE", E, 3):
            got = determine_alignment_strategy(
                5, 100, 100, overrides={"tier": spelling}
            )
            self.assertIs(got.tier, E)
            self.assertIn("forced", got.reason)
        with self.assertRaises(ValueError):
            determine_alignment_strategy(5, 100, overrides={"tier": "warp"})

    def test_unknown_override_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            determine_alignment_strategy(5, 100, overrides={"paralleltree": 1})

    def test_new_work_thresholds_move_the_boundaries(self) -> None:
        n, length = 100, 1000  # W ~ 4.0e8
        self.assertIs(determine_alignment_strategy(n, length, length).tier, A)
        cases = {
            "work_fast_tree_threshold": 1.0e8,
            "work_parttree_threshold": 1.0e8,  # pre-v3 name of the first boundary
        }
        for key, value in cases.items():
            got = determine_alignment_strategy(
                n, length, length, overrides={key: value}
            )
            self.assertIs(got.tier, F, key)
        high = determine_alignment_strategy(
            n,
            length,
            length,
            overrides={
                "work_fast_tree_threshold": 1e8,
                "work_high_scale_threshold": 2e8,
            },
        )
        self.assertIs(high.tier, H)
        extreme = determine_alignment_strategy(
            n,
            length,
            length,
            overrides={
                "work_fast_tree_threshold": 1e8,
                "work_high_scale_threshold": 2e8,
                "work_extreme_scale_threshold": 3e8,
            },
        )
        self.assertIs(extreme.tier, E)
        self.assertIs(
            determine_alignment_strategy(
                n,
                length,
                length,
                overrides={
                    "work_retree1_threshold": 1e8,
                    "work_fast_tree_threshold": 1e7,
                    "work_high_scale_threshold": 5e7,
                },
            ).tier,
            E,
        )  # pre-v3 name of the extreme-scale boundary

    def test_inconsistent_boundaries_are_made_monotone(self) -> None:
        # first boundary above the second: the second is lifted to it, so a W
        # in between does not skip a tier backwards
        got = determine_alignment_strategy(
            100, 1000, 1000, overrides={"work_fast_tree_threshold": 1e12}
        )
        self.assertIs(got.tier, A)
        got = determine_alignment_strategy(
            100, 8000, 8000, overrides={"work_fast_tree_threshold": 1e12}
        )
        self.assertIs(got.tier, A)

    def test_n_cap_override(self) -> None:
        self.assertIs(
            determine_alignment_strategy(
                300, 100, 100, overrides={"n_force_parttree": 200}
            ).tier,
            F,
        )
        self.assertIs(
            determine_alignment_strategy(
                20000, 100, 100, overrides={"n_force_parttree": 10**9}
            ).tier,
            A,
        )

    def test_legacy_n_count_gates_work_only_when_explicit_and_warn_once(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(determine_alignment_strategy(1200, 300, 300).tier, A)
            self.assertEqual(caught, [])  # not set: no warning, no gate
            got = determine_alignment_strategy(
                1200, 300, 300, overrides={"parttree_threshold": 1000}
            )
            self.assertIs(got.tier, H)  # the pre-v3 PartTree2 --nofft tier
            self.assertEqual(got.mafft_args, TIER_MAFFT_ARGS[H])
            determine_alignment_strategy(
                1200, 300, 300, overrides={"parttree_threshold": 1000}
            )
            deprecations = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]
            self.assertEqual(len(deprecations), 1)  # one-time
            self.assertIn("parttree_threshold", str(deprecations[0].message))

            got = determine_alignment_strategy(
                6000, 300, 300, overrides={"retree1_threshold": 5000}
            )
            self.assertIs(got.tier, E)
            both = determine_alignment_strategy(
                1200,
                300,
                300,
                overrides={"parttree_threshold": 1000, "retree1_threshold": 5000},
            )
            self.assertIs(both.tier, H)  # N below the retree1 gate
            deprecations = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]
            self.assertEqual(len(deprecations), 2)  # + retree1_threshold, once each

    def test_legacy_gate_never_lowers_a_tier(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            got = determine_alignment_strategy(
                700, 3500, 3500, overrides={"parttree_threshold": 100}
            )
        self.assertIs(got.tier, E)


class ConstantsTests(unittest.TestCase):
    def test_constants_are_one_frozen_instance_with_provenance(self) -> None:
        self.assertIsInstance(DEFAULT_CONSTANTS, StrategyConstants)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_CONSTANTS.calibrated = True  # type: ignore[misc]
        for name in ("calibrated", "source", "mafft_version", "cpu", "date"):
            self.assertTrue(hasattr(DEFAULT_CONSTANTS, name), name)
        self.assertFalse(DEFAULT_CONSTANTS.calibrated)  # provisional until calibrated
        json.dumps(DEFAULT_CONSTANTS.to_dict())

    def test_boundaries_are_ordered_and_not_the_blueprint_literals(self) -> None:
        c = DEFAULT_CONSTANTS
        self.assertLess(c.work_fast_tree, c.work_high_scale)
        self.assertLess(c.work_high_scale, c.work_extreme_scale)
        for literal in (5.0e8, 8.0e7, 1.5e7):
            self.assertNotIn(
                literal, (c.work_fast_tree, c.work_high_scale, c.work_extreme_scale)
            )
        # a 100-taxon 1-kb locus must be far below the first boundary
        self.assertGreater(c.work_fast_tree / estimate_work(100, 1000), 10)
        self.assertEqual(c.n_force_parttree, N_FORCE_PARTTREE)
        self.assertEqual(c.min_parttree_seq_len, 6)

    def test_alphabet_model_exponents(self) -> None:
        model = AlphabetModel()
        self.assertEqual(model.alpha_for(10), 2.0)
        self.assertEqual(model.alpha_for(999), 2.0)
        self.assertEqual(model.alpha_for(1000), 2.2)
        self.assertEqual(model.alpha_for(2499), 2.2)
        self.assertEqual(model.alpha_for(2500), 2.4)


if __name__ == "__main__":
    unittest.main()
