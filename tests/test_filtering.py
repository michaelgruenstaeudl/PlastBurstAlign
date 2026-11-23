"""Tests for ``plastburstalign.filtering`` (site statistics, TrimSpec, gappyout)."""

from __future__ import annotations

import math
import tracemalloc
import unittest
from collections import Counter

import numpy as np

from plastburstalign.filtering import (
    CLASS_AMBIGUOUS,
    CLASS_GAP,
    CLASS_LUT,
    CLASS_MISSING,
    CLASS_N,
    TrimSpec,
    cell_class_counts,
    compute_keep_mask,
    evaluate_site_occupancy,
    gappyout_cutoff,
    site_statistics,
)
from plastburstalign.matrix_ops import DenseLocusMatrix

# 6 taxa x 9 columns with hand-computed statistics (see the expected arrays).
#   col: 0 1 2 3 4 5 6 7 8
ROWS = {
    "t1": "AAAA-ANRA",
    "t2": "AAAC-ANRC",
    "t3": "AAAG-ANAA",
    "t4": "AGAT--NAC",
    "t5": "AGA--?NAN",
    "t6": "AGC---NA-",
}
H_ONE_OF_SIX = -(5 / 6) * math.log2(5 / 6) - (1 / 6) * math.log2(1 / 6)
EXPECTED_ENTROPY = [0, 1, H_ONE_OF_SIX, 2, 0, 0, 0, 0, 1]
EXPECTED_VARIABLE = [0, 1, 1, 1, 0, 0, 0, 0, 1]
EXPECTED_INFORMATIVE = [0, 1, 0, 0, 0, 0, 0, 0, 1]
EXPECTED_GAP = [0, 0, 0, 2, 6, 2, 0, 0, 1]
EXPECTED_MISSING = [0, 0, 0, 0, 0, 1, 0, 0, 0]
EXPECTED_OCCUPIED = {
    "none": [6, 6, 6, 4, 0, 3, 0, 4, 4],
    "iupac": [6, 6, 6, 4, 0, 3, 0, 6, 4],
    "all": [6, 6, 6, 4, 0, 3, 6, 6, 5],
}


def make_matrix(rows: dict[str, str] | None = None) -> DenseLocusMatrix:
    return DenseLocusMatrix.from_sequences(rows or ROWS)


class SiteStatisticsTests(unittest.TestCase):
    def test_hand_computed_statistics(self):
        stats = site_statistics(make_matrix())
        self.assertEqual(stats.nrows, 6)
        self.assertEqual(stats.nchar, 9)
        np.testing.assert_array_equal(stats.occupied, EXPECTED_OCCUPIED["none"])
        np.testing.assert_allclose(stats.entropy, EXPECTED_ENTROPY, atol=1e-12)
        np.testing.assert_array_equal(stats.gap, EXPECTED_GAP)
        np.testing.assert_array_equal(stats.missing, EXPECTED_MISSING)
        np.testing.assert_array_equal(stats.variable, np.array(EXPECTED_VARIABLE, bool))
        np.testing.assert_array_equal(
            stats.informative, np.array(EXPECTED_INFORMATIVE, bool)
        )
        np.testing.assert_allclose(
            stats.occupancy, np.array(EXPECTED_OCCUPIED["none"]) / 6
        )
        np.testing.assert_array_equal(
            stats.gap_counts, 6 - np.array(EXPECTED_OCCUPIED["none"])
        )

    def test_ambiguity_parameter_changes_only_occupancy(self):
        base = site_statistics(make_matrix())
        for mode, expected in EXPECTED_OCCUPIED.items():
            with self.subTest(mode=mode):
                stats = site_statistics(make_matrix(), ambiguity_occupied=mode)  # type: ignore[arg-type]
                np.testing.assert_array_equal(stats.occupied, expected)
                np.testing.assert_allclose(stats.entropy, base.entropy)
                np.testing.assert_array_equal(stats.informative, base.informative)

    def test_invalid_ambiguity_mode(self):
        with self.assertRaises(ValueError):
            site_statistics(make_matrix(), ambiguity_occupied="bogus")  # type: ignore[arg-type]

    def test_case_insensitive_and_u_equals_t(self):
        ref = site_statistics(make_matrix())
        lower = site_statistics(make_matrix({k: v.lower() for k, v in ROWS.items()}))
        rna = site_statistics(
            make_matrix({k: v.replace("T", "U") for k, v in ROWS.items()})
        )
        for other in (lower, rna):
            np.testing.assert_array_equal(other.occupied, ref.occupied)
            np.testing.assert_allclose(other.entropy, ref.entropy)
            np.testing.assert_array_equal(other.variable, ref.variable)
            np.testing.assert_array_equal(other.informative, ref.informative)

    def test_dot_is_a_gap_and_question_mark_missing(self):
        m = make_matrix({"a": "A.?-N", "b": "AAAAA"})
        stats = site_statistics(m)
        np.testing.assert_array_equal(stats.occupied, [2, 1, 1, 1, 1])
        counts = cell_class_counts(m)
        self.assertEqual(counts[CLASS_GAP], 2)
        self.assertEqual(counts[CLASS_MISSING], 1)
        self.assertEqual(counts[CLASS_N], 1)

    def test_occupancy_only_mode(self):
        stats = site_statistics(make_matrix(), composition=False)
        self.assertIsNone(stats.entropy)
        self.assertIsNone(stats.gap)
        self.assertFalse(stats.has_composition)
        np.testing.assert_array_equal(stats.occupied, EXPECTED_OCCUPIED["none"])

    def test_raw_array_input_and_validation(self):
        arr = make_matrix().matrix
        np.testing.assert_array_equal(
            site_statistics(arr).occupied, EXPECTED_OCCUPIED["none"]
        )
        with self.assertRaises(ValueError):
            site_statistics(arr.astype(np.int32))
        with self.assertRaises(ValueError):
            site_statistics(arr.ravel())

    def test_empty_shapes(self):
        stats = site_statistics(np.zeros((3, 0), dtype=np.uint8))
        self.assertEqual(stats.nchar, 0)
        stats = site_statistics(np.zeros((0, 4), dtype=np.uint8))
        np.testing.assert_array_equal(stats.occupied, [0, 0, 0, 0])
        np.testing.assert_array_equal(stats.occupancy, [0.0] * 4)

    def test_informative_needs_two_states_in_two_taxa_each(self):
        m = make_matrix({"a": "AAAA", "b": "AAAA", "c": "CCAA", "d": "CGAA"})
        # col0: 2xA 2xC -> informative; col1: A,A,C,G -> variable only;
        # cols 2,3 monomorphic
        stats = site_statistics(m)
        np.testing.assert_array_equal(stats.variable, [True, True, False, False])
        np.testing.assert_array_equal(stats.informative, [True, False, False, False])

    def test_block_size_does_not_change_results(self):
        rng = np.random.default_rng(3)
        alphabet = np.frombuffer(b"ACGTacgtNRY-?.", dtype=np.uint8)
        arr = rng.choice(alphabet, size=(37, 501))
        ref = site_statistics(arr, ambiguity_occupied="iupac")
        for block_bytes in (1, 4 * 37, 4 * 37 * 7, 10_000, 1 << 30):
            small = site_statistics(
                arr, ambiguity_occupied="iupac", block_bytes=block_bytes
            )
            np.testing.assert_array_equal(small.occupied, ref.occupied)
            np.testing.assert_array_equal(small.entropy, ref.entropy)
            np.testing.assert_array_equal(small.informative, ref.informative)
            counts = cell_class_counts(arr, block_bytes=block_bytes)
            np.testing.assert_array_equal(counts, cell_class_counts(arr))

    def test_temporaries_stay_within_block_budget(self):
        arr = np.full((1000, 20000), ord("A"), dtype=np.uint8)
        arr[::3, ::2] = ord("C")
        arr[::7, ::5] = ord("-")
        block_bytes = 2 * 1024 * 1024
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            site_statistics(arr, block_bytes=block_bytes)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            cell_class_counts(arr, block_bytes=block_bytes)
            _, peak_counts = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # 20 MB matrix; temporaries (plus 14 B/column of outputs) must stay near the budget
        self.assertLess(peak, 2 * block_bytes + 1_000_000, f"peak={peak}")
        self.assertLess(peak_counts, 2 * block_bytes + 1_000_000, f"peak={peak_counts}")

    def test_cell_class_counts_match_python_counter(self):
        m = make_matrix()
        counts = cell_class_counts(m)
        text = "".join(ROWS.values())
        expected = Counter(int(CLASS_LUT[ord(ch)]) for ch in text)
        for cls in range(8):
            self.assertEqual(counts[cls], expected.get(cls, 0), f"class {cls}")
        self.assertEqual(counts.sum(), 54)
        self.assertEqual(CLASS_LUT[ord("R")], CLASS_AMBIGUOUS)
        self.assertEqual(cell_class_counts(np.zeros((0, 0), dtype=np.uint8)).sum(), 0)


class EvaluateSiteOccupancyTests(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        self.assertTrue(evaluate_site_occupancy("AC--", 0.5))
        self.assertFalse(evaluate_site_occupancy("AC---", 0.5))
        self.assertTrue(evaluate_site_occupancy("A-", 0.5))
        self.assertTrue(evaluate_site_occupancy("A", 1.0))
        self.assertTrue(evaluate_site_occupancy("----", 0.0))

    def test_default_threshold_is_half(self):
        self.assertTrue(evaluate_site_occupancy("AA--"))
        self.assertFalse(evaluate_site_occupancy("A---"))

    def test_input_types_agree(self):
        text = "aC-?nT"
        results = {
            evaluate_site_occupancy(text, 0.5),
            evaluate_site_occupancy(text.encode(), 0.5),
            evaluate_site_occupancy(list(text), 0.5),
            evaluate_site_occupancy(np.frombuffer(text.encode(), np.uint8), 0.5),
            evaluate_site_occupancy(bytearray(text.encode()), 0.5),
        }
        self.assertEqual(results, {True})  # 3 of 6 are A/C/T -> exactly 0.5

    def test_ambiguity_parameter(self):
        self.assertFalse(evaluate_site_occupancy("NNNA", 0.5))
        self.assertTrue(evaluate_site_occupancy("NNNA", 0.5, ambiguity_occupied="all"))
        self.assertFalse(
            evaluate_site_occupancy("NRRA", 0.9, ambiguity_occupied="iupac")
        )
        self.assertTrue(
            evaluate_site_occupancy("RRYA", 0.9, ambiguity_occupied="iupac")
        )

    def test_empty_column_is_not_occupied(self):
        self.assertFalse(evaluate_site_occupancy("", 0.0))
        self.assertFalse(evaluate_site_occupancy([], 0.5))


class TrimSpecTests(unittest.TestCase):
    def test_none_variants(self):
        for text in (None, "", "none", " NONE ", "  "):
            spec = TrimSpec.parse(text)
            self.assertTrue(spec.is_noop, text)
            self.assertEqual(str(spec), "none")

    def test_single_criteria_and_roundtrip(self):
        cases = {
            "occupancy=0.5": TrimSpec(min_occupancy=0.5),
            "entropy=0.1": TrimSpec(entropy_min=0.1),
            "entropy=0.1:1.5": TrimSpec(entropy_min=0.1, entropy_max=1.5),
            "entropy=:1.5": TrimSpec(entropy_max=1.5),
            "informative-only": TrimSpec(informative_only=True),
            "gappyout": TrimSpec(gappyout=True),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                spec = TrimSpec.parse(text)
                self.assertEqual(spec, expected)
                self.assertEqual(str(spec), text)
                self.assertEqual(TrimSpec.parse(str(spec)), spec)
                self.assertFalse(spec.is_noop)

    def test_combination_is_order_independent(self):
        a = TrimSpec.parse("occupancy=0.5+entropy=0.1:1.5+informative-only+gappyout")
        b = TrimSpec.parse(
            "gappyout + Informative_Only + ENTROPY=0.1:1.5 + occupancy=0.5"
        )
        self.assertEqual(a, b)
        self.assertEqual(
            a,
            TrimSpec(
                min_occupancy=0.5,
                entropy_min=0.1,
                entropy_max=1.5,
                informative_only=True,
                gappyout=True,
            ),
        )
        self.assertEqual(
            str(a), "occupancy=0.5+entropy=0.1:1.5+informative-only+gappyout"
        )
        self.assertTrue(a.needs_composition)
        self.assertFalse(TrimSpec.parse("occupancy=0.5+gappyout").needs_composition)

    def test_passthrough_and_ambiguity_override(self):
        spec = TrimSpec(min_occupancy=0.3)
        self.assertIs(TrimSpec.parse(spec), spec)
        spec2 = TrimSpec.parse("occupancy=0.3", ambiguity_occupied="all")
        self.assertEqual(spec2.ambiguity_occupied, "all")
        self.assertEqual(
            TrimSpec.parse(None, ambiguity_occupied="iupac").ambiguity_occupied, "iupac"
        )

    def test_invalid_specs(self):
        bad = [
            "bogus",
            "occupancy",
            "occupancy=abc",
            "occupancy=1.5",
            "occupancy=-0.1",
            "occupancy=nan",
            "entropy=",
            "entropy=:",
            "entropy=2:1",
            "entropy=-1",
            "entropy=a:b",
            "none+gappyout",
            "gappyout+gappyout",
            "occupancy=0.5+occupancy=0.6",
            "gappyout++informative-only",
            "informative-only=3",
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                TrimSpec.parse(text)
        with self.assertRaises(ValueError):
            TrimSpec(ambiguity_occupied="nope")  # type: ignore[arg-type]


class KeepMaskTests(unittest.TestCase):
    def mask(self, spec: str, **kwargs):
        return compute_keep_mask(make_matrix(), TrimSpec.parse(spec, **kwargs)).tolist()

    def test_noop_keeps_everything(self):
        self.assertEqual(self.mask("none"), [True] * 9)

    def test_occupancy(self):
        t, f = True, False
        self.assertEqual(self.mask("occupancy=0.5"), [t, t, t, t, f, t, f, t, t])
        self.assertEqual(self.mask("occupancy=0.7"), [t, t, t, f, f, f, f, f, f])
        self.assertEqual(self.mask("occupancy=0"), [True] * 9)
        self.assertEqual(self.mask("occupancy=1"), [t, t, t, f, f, f, f, f, f])
        self.assertEqual(
            self.mask("occupancy=0.5", ambiguity_occupied="all"),
            [t, t, t, t, f, t, t, t, t],
        )

    def test_entropy_bounds(self):
        t, f = True, False
        self.assertEqual(self.mask("entropy=0.5"), [f, t, t, t, f, f, f, f, t])
        self.assertEqual(self.mask("entropy=0.5:1.5"), [f, t, t, f, f, f, f, f, t])
        self.assertEqual(self.mask("entropy=:0.7"), [t, f, t, f, t, t, t, t, f])
        self.assertEqual(self.mask("entropy=1:1"), [f, t, f, f, f, f, f, f, t])

    def test_informative_only(self):
        t, f = True, False
        self.assertEqual(self.mask("informative-only"), [f, t, f, f, f, f, f, f, t])

    def test_combination_is_a_conjunction(self):
        t, f = True, False
        self.assertEqual(
            self.mask("occupancy=0.9+informative-only"), [f, t, f, f, f, f, f, f, f]
        )
        self.assertEqual(
            self.mask("occupancy=0.5+entropy=0.5"), [f, t, t, t, f, f, f, f, t]
        )

    def test_gappyout_on_matrix(self):
        # gap counts [0,0,0,2,6,3,6,2,2]: knee at 0 gaps; the all-gap columns go too
        t, f = True, False
        self.assertEqual(self.mask("gappyout"), [t, t, t, f, f, f, f, f, f])

    def test_precomputed_stats_are_reused_and_validated(self):
        m = make_matrix()
        spec = TrimSpec.parse("occupancy=0.5")
        stats = site_statistics(m, composition=False)
        self.assertEqual(
            compute_keep_mask(m, spec, stats=stats).tolist(),
            compute_keep_mask(m, spec).tolist(),
        )
        with self.assertRaises(ValueError):
            compute_keep_mask(
                m, TrimSpec.parse("entropy=0.5"), stats=stats
            )  # no composition
        with self.assertRaises(ValueError):
            compute_keep_mask(
                m,
                TrimSpec.parse("occupancy=0.5", ambiguity_occupied="all"),
                stats=stats,
            )
        with self.assertRaises(ValueError):
            compute_keep_mask(make_matrix({"a": "AC"}), spec, stats=stats)

    def test_zero_row_matrix(self):
        empty = np.zeros((0, 3), dtype=np.uint8)
        self.assertEqual(
            compute_keep_mask(empty, TrimSpec.parse("occupancy=0.5")).tolist(),
            [False] * 3,
        )
        self.assertEqual(compute_keep_mask(empty, TrimSpec()).tolist(), [True] * 3)

    def test_keep_mask_drives_select_columns(self):
        m = make_matrix()
        kept = m.select_columns(
            compute_keep_mask(m, TrimSpec.parse("informative-only"))
        )
        self.assertEqual(kept.to_dict()["t1"], "AA")  # cols 1 and 8 of t1
        self.assertEqual(kept.to_dict()["t4"], "GC")


def _gap_profile_matrix(counts: list[int], ntax: int, seed: int = 0) -> np.ndarray:
    """Build a matrix whose column ``i`` has exactly ``counts[i]`` gaps."""
    rng = np.random.default_rng(seed)
    arr = np.full((ntax, len(counts)), ord("A"), dtype=np.uint8)
    for col, n_gaps in enumerate(counts):
        rows = rng.choice(ntax, size=n_gaps, replace=False)
        arr[rows, col] = ord("-")
    return arr


class GappyoutTests(unittest.TestCase):
    BIMODAL = (
        [0] * 60
        + [1] * 20
        + [2] * 10
        + [15] * 2
        + [16] * 3
        + [17] * 2
        + [18] * 2
        + [19]
    )

    def test_bimodal_profile_cuts_between_the_modes(self):
        self.assertEqual(gappyout_cutoff(self.BIMODAL, 20), 2)

    def test_bimodal_profile_through_the_mask(self):
        counts = self.BIMODAL
        order = np.random.default_rng(5).permutation(len(counts))
        shuffled = [counts[i] for i in order]
        arr = _gap_profile_matrix(shuffled, ntax=20)
        keep = compute_keep_mask(arr, TrimSpec.parse("gappyout"))
        np.testing.assert_array_equal(keep, np.array(shuffled) <= 2)
        self.assertEqual(int(keep.sum()), 90)

    def test_cutoff_is_order_independent(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            shuffled = rng.permutation(self.BIMODAL).tolist()
            self.assertEqual(gappyout_cutoff(shuffled, 20), 2)

    def test_plateau_ties_keep_the_whole_plateau(self):
        # 2 gap-free columns, 10 with 3 gaps, 2 with 9 gaps (ntax=10): the knee
        # is the end of the 3-gap plateau and every column of it is retained
        counts = [0] * 2 + [3] * 10 + [9] * 2
        self.assertEqual(gappyout_cutoff(counts, 10), 3)

    def test_all_gap_columns_always_rejected(self):
        counts = [0] * 10 + [5] * 3 + [20] * 4  # ntax=20: four all-gap columns
        cutoff = gappyout_cutoff(counts, 20)
        self.assertLess(cutoff, 20)
        arr = _gap_profile_matrix(counts, 20)
        keep = compute_keep_mask(arr, TrimSpec.parse("gappyout"))
        self.assertFalse(keep[-4:].any())
        self.assertTrue(keep[:10].all())

    def test_only_all_gap_columns(self):
        self.assertEqual(gappyout_cutoff([6, 6, 6], 6), -1)
        arr = np.full((6, 4), ord("-"), dtype=np.uint8)
        self.assertFalse(compute_keep_mask(arr, TrimSpec.parse("gappyout")).any())
        self.assertEqual(gappyout_cutoff([], 6), -1)

    def test_no_gaps_keeps_everything(self):
        self.assertEqual(gappyout_cutoff([0] * 12, 8), 0)
        arr = np.full((8, 12), ord("A"), dtype=np.uint8)
        self.assertTrue(compute_keep_mask(arr, TrimSpec.parse("gappyout")).all())

    def test_single_taxon(self):
        m = DenseLocusMatrix.from_sequences({"only": "AC-?-A"})
        keep = compute_keep_mask(m, TrimSpec.parse("gappyout"))
        self.assertEqual(keep.tolist(), [True, True, False, False, False, True])
        self.assertEqual(gappyout_cutoff([0, 0, 1, 1], 1), 0)

    def test_uniform_ramp_has_no_knee(self):
        counts = list(range(20))  # one column for each gap count 0..19 (ntax=20)
        self.assertEqual(gappyout_cutoff(counts, 20), 19)

    def test_tiny_profiles(self):
        self.assertEqual(gappyout_cutoff([0, 5], 10), 5)  # fewer than 3 columns
        self.assertEqual(gappyout_cutoff([7], 10), 7)
        # a two-valued profile is cut below its upper plateau
        self.assertEqual(gappyout_cutoff([0, 5, 5, 0], 10), 0)

    def test_min_deviation_controls_sensitivity(self):
        counts = [0] * 50 + list(range(1, 51))  # mild knee at the plateau end
        strict = gappyout_cutoff(counts, 60, min_deviation=0.9)
        lenient = gappyout_cutoff(counts, 60, min_deviation=0.0)
        self.assertEqual(strict, 50)
        self.assertLess(lenient, strict)

    def test_ambiguity_convention_feeds_gap_counts(self):
        # N-only columns count as gaps by default but as data with ambiguity='all'
        m = DenseLocusMatrix.from_sequences(
            {"a": "ANNA", "b": "ANNA", "c": "A-NA", "d": "A-NA"}
        )
        default = compute_keep_mask(m, TrimSpec.parse("gappyout"))
        self.assertEqual(default.tolist(), [True, False, False, True])
        everything = compute_keep_mask(
            m, TrimSpec.parse("gappyout", ambiguity_occupied="all")
        )
        self.assertTrue(everything[2])


if __name__ == "__main__":
    unittest.main()
