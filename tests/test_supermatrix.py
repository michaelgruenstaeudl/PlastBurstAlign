"""Tests for ``plastburstalign.supermatrix`` (builder, writers, parity, scale)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from mafft_fixtures import (
    align_with_mafft,
    legacy_concat,
    mafft_available,
    simulate_locus,
)

from plastburstalign.filtering import TrimSpec
from plastburstalign.nexus_ops import (
    ConcatenationError,
    fast_write_fasta,
    fast_write_nexus,
    nexus_data_header,
    nexus_row_prefix,
)
from plastburstalign.supermatrix import (
    LocusSource,
    Supermatrix,
    SupermatrixBuilder,
    _ordered_map,
    legacy_concat_rename,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def strip_prefix(locus: str, raw_id: str) -> str:
    """Rename ``<locus>_x`` to ``x`` (used where short taxon names read better)."""
    prefix = f"{locus}_"
    return raw_id[len(prefix) :] if raw_id.startswith(prefix) else raw_id


def put(directory: Path, name: str, content: str | bytes | None) -> Path:
    """Write ``content`` as ``nucl_<name>.aligned.fasta`` (``None`` = leave missing)."""
    path = directory / f"nucl_{name}.aligned.fasta"
    if content is not None:
        path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def sources_for(
    directory: Path, loci: dict[str, str | bytes | None], kind: str = "nucl"
) -> list[tuple[str, Path]]:
    return [(name, put(directory, name, text)) for name, text in loci.items()]


def build(
    pairs: list[tuple[str, Path]],
    kind: str = "nucl",
    codon_aligned: bool = False,
    **kwargs,
) -> Supermatrix:
    srcs = [LocusSource(n, p, kind, codon_aligned) for n, p in pairs]
    kwargs.setdefault("rename", legacy_concat_rename)
    kwargs.setdefault("buffer", "ram")
    return SupermatrixBuilder(srcs, **kwargs).build()


# ----------------------------------------------------------------------
# parity: new builder == legacy accumulator + fast_write_* (byte for byte)
# ----------------------------------------------------------------------
PARITY_CASES: dict[str, dict[str, str | bytes | None]] = {
    "disjoint_taxa": {
        "g1": ">g1_a\nACGT\n>g1_b\nAC-T\n",
        "g2": ">g2_c\nGGGGG\n>g2_d\nGG-GG\n",
        "g3": ">g3_a\nTT\n>g3_d\nT-\n",
    },
    "reversed_plain_and_foreign_ids": {
        "g1": ">_R_g1_x\nACGTAC\n>g1_y desc here\nACG-AC\n>other_z\nACGTAC\n",
        "g2": ">g2_x\nAAAA\n>_R_g2_y\nA-AA\n>g2_g2_w\nAAAA\n",
    },
    "single_locus": {"only": ">only_t1\nACGTACGT\n>only_t2\nAC--ACGT\n"},
    "lowercase_wrapped_crlf": {
        "g1": ">g1_a\r\nacgt\r\nacgt\r\n\r\n>g1_b\r\nac-t\r\nacgn\r\n",
        "g2": ">g2_b\nacg\ntac\n>g2_c\nac-\ntnc",
    },
    "iupac_missing_dots": {
        "g1": ">g1_a\nRYKMN?-.\n>g1_b\nacgtn?-.\n",
        "g2": ">g2_a\nSWBD\n>g2_c\nHVN-\n",
    },
    "empty_missing_and_failing_loci": {
        "g1": ">g1_a\nACGT\n>g1_b\nAC-T\n",
        "empty_file": b"",
        "missing_file": None,
        "header_only": ">header_only_a\n>header_only_b\n",
        "bad_header": ">\nACGT\n>bad_header_b\nACGT\n",
        "g2": ">g2_b\nGGG\n>g2_c\nGG-\n",
    },
    "taxon_order_is_first_appearance": {
        "g1": ">g1_z\nAA\n>g1_m\nCC\n",
        "g2": ">g2_a\nGG\n>g2_m\nTT\n>g2_z\nAA\n",
        "g3": ">g3_q\nCC\n>g3_a\nAA\n",
    },
}

FILE_KEYS = ("fasta", "nexus", "raxml", "nexus_partitions")


class LegacyParityTests(unittest.TestCase):
    def assert_byte_parity(
        self, pairs: list[tuple[str, Path]], tmp: Path, **kwargs
    ) -> Supermatrix:
        legacy = legacy_concat(pairs, tmp / "legacy" / "nucl_Nconcat")
        sm = build(pairs, normalize_case=False, trim=None, **kwargs)
        written = sm.write_all(tmp / "new" / "nucl_Nconcat")
        for key in FILE_KEYS:
            self.assertEqual(
                legacy[key].read_bytes(),
                written[key].read_bytes(),
                f"{key} differs ({kwargs})",
            )
        return sm

    def test_hand_built_edge_cases(self):
        for label, loci in PARITY_CASES.items():
            for buffer in ("ram", "memmap"):
                for workers in (1, 3):
                    with (
                        self.subTest(case=label, buffer=buffer, workers=workers),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        tmp_path = Path(tmp)
                        pairs = sources_for(tmp_path, loci)
                        self.assert_byte_parity(
                            pairs,
                            tmp_path,
                            buffer=buffer,
                            max_workers=workers,
                            scratch_dir=tmp_path,
                        )

    def test_failing_loci_are_reported(self):
        loci = PARITY_CASES["empty_missing_and_failing_loci"]
        with tempfile.TemporaryDirectory() as tmp:
            sm = build(sources_for(Path(tmp), loci), normalize_case=False)
        self.assertEqual(
            sorted(sm.failed_loci),
            ["bad_header", "empty_file", "header_only", "missing_file"],
        )
        self.assertEqual(sm.partitions.names, ("g1", "g2"))
        self.assertEqual(sm.taxa, ("concat_a", "concat_b", "concat_c"))

    def test_taxon_order_and_padding(self):
        loci = PARITY_CASES["taxon_order_is_first_appearance"]
        with tempfile.TemporaryDirectory() as tmp:
            sm = build(sources_for(Path(tmp), loci), normalize_case=False)
        self.assertEqual(sm.taxa, ("concat_z", "concat_m", "concat_a", "concat_q"))
        self.assertEqual(
            sm.to_dict(),
            {
                "concat_z": "AAAA??",
                "concat_m": "CCTT??",
                "concat_a": "??GGAA",
                "concat_q": "????CC",
            },
        )

    @unittest.skipUnless(mafft_available(), "mafft is not installed")
    def test_real_mafft_loci(self):
        rng = random.Random(777)
        taxa = [f"taxon{i:02d}" for i in range(14)]
        plans = (
            ("atpA", taxa[:10], False),
            ("rbcL", taxa[4:], True),
            ("matK", taxa[2:12], False),
            ("trnLF", taxa[:3] + taxa[9:], False),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = []
            for locus, subset, adjust in plans:
                seqs = simulate_locus(
                    rng, subset, 150, reversed_taxa=subset[1:3] if adjust else ()
                )
                pairs.append(
                    (locus, align_with_mafft(locus, seqs, tmp_path / "fasta", adjust))
                )
            raw_rbcl = pairs[1][1].read_bytes()
            self.assertIn(b"_R_", raw_rbcl)
            self.assertTrue(any(c in raw_rbcl for c in b"acgt"))
            digests = set()
            for buffer in ("ram", "memmap"):
                for workers in (1, 4):
                    sm = self.assert_byte_parity(
                        pairs,
                        tmp_path / f"{buffer}{workers}",
                        buffer=buffer,
                        max_workers=workers,
                        scratch_dir=tmp_path,
                    )
                    digests.add(sm.sha256())
            self.assertEqual(len(digests), 1)
            self.assertEqual(sm.ntax, len(taxa))
            print(
                f"\n[parity] MAFFT loci: {sm.summary()} -- byte-identical to legacy "
                f"(fasta, nexus, raxml, nexus partitions)"
            )

    def test_normalized_output_differs_only_by_case_and_dots(self):
        loci = PARITY_CASES["iupac_missing_dots"]
        with tempfile.TemporaryDirectory() as tmp:
            pairs = sources_for(Path(tmp), loci)
            raw = build(pairs, normalize_case=False)
            norm = build(pairs, normalize_case=True)
        expected = {t: s.upper().replace(".", "-") for t, s in raw.to_dict().items()}
        self.assertEqual(norm.to_dict(), expected)
        self.assertEqual(raw.taxa, norm.taxa)

    def test_nexus_writer_matches_legacy_function_with_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairs = sources_for(tmp_path, PARITY_CASES["disjoint_taxa"])
            sm = build(pairs)  # normalize_case=True
            sm.write_nexus(tmp_path / "new.nex")
            fast_write_nexus(
                tmp_path / "old.nex",
                sm.to_dict(),
                charpartitions=sm.partitions.to_charpartitions(),
            )
            fast_write_fasta(tmp_path / "old.fasta", sm.to_dict(), list(sm.taxa))
            sm.write_fasta(tmp_path / "new.fasta")
            self.assertEqual(
                (tmp_path / "new.nex").read_bytes(), (tmp_path / "old.nex").read_bytes()
            )
            self.assertEqual(
                (tmp_path / "new.fasta").read_bytes(),
                (tmp_path / "old.fasta").read_bytes(),
            )

    def test_nexus_helpers_reproduce_legacy_layout(self):
        self.assertEqual(
            nexus_data_header(3, 7),
            "#NEXUS\n\nBEGIN DATA;\n  DIMENSIONS NTAX=3 NCHAR=7;\n"
            "  FORMAT DATATYPE=DNA GAP=- MISSING=?;\n  MATRIX\n",
        )
        self.assertEqual(nexus_row_prefix("ab", 5), "    ab" + " " * 7)


# ----------------------------------------------------------------------
# hand-computed trimming, statistics and partitions
# ----------------------------------------------------------------------
G1 = ">g1_a\nATGAAACCC\n>g1_b\nATGAA-CCC\n>g1_c\nACG---CC-\n>g1_d\nGCG---C--\n"
G2 = ">g2_b\nACGTA\n>g2_c\nAC-TA\n>g2_e\nA--C-\n"


class TrimmingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.pairs = sources_for(self.tmp, {"g1": G1, "g2": G2})

    def make(self, trim, **kwargs) -> Supermatrix:
        srcs = [
            LocusSource("g1", self.pairs[0][1], "cds", codon_aligned=True),
            LocusSource("g2", self.pairs[1][1], "igs"),
        ]
        kwargs.setdefault("buffer", "ram")
        kwargs.setdefault("rename", strip_prefix)
        return SupermatrixBuilder(srcs, trim=trim, **kwargs).build()

    def test_untrimmed_baseline(self):
        sm = self.make(None)
        self.assertEqual(sm.taxa, ("a", "b", "c", "d", "e"))
        self.assertEqual((sm.ntax, sm.nchar), (5, 14))
        self.assertEqual(sm.to_dict()["a"], "ATGAAACCC" + "?????")
        self.assertEqual(sm.partitions["g1"].end, 9)

    def test_occupancy_trim_matrix_stats_and_partitions(self):
        for buffer in ("ram", "memmap"):
            with self.subTest(buffer=buffer):
                sm = self.make("occupancy=0.5", buffer=buffer, scratch_dir=self.tmp)
                self.assertEqual(
                    sm.to_dict(),
                    {
                        "a": "ATGAACCC" + "????",
                        "b": "ATGAACCC" + "ACTA",
                        "c": "ACG--CC-" + "ACTA",
                        "d": "GCG--C--" + "????",
                        "e": "????????" + "A-C-",
                    },
                )
                self.assertEqual((sm.nchar, sm.ntax), (12, 5))
                p1, p2 = sm.partitions["g1"], sm.partitions["g2"]
                self.assertEqual((p1.start, p1.end, p2.start, p2.end), (1, 8, 9, 12))
                self.assertEqual(p1.columns.tolist(), [0, 1, 2, 3, 4, 6, 7, 8])
                self.assertEqual(p2.columns.tolist(), [0, 1, 3, 4])
                self.assertEqual((p1.codon_phase, p2.codon_phase), (0, None))

                g1, g2 = sm.locus_stats("g1"), sm.locus_stats("g2")
                self.assertEqual(
                    (g1.ntax_present, g1.nchar_in, g1.nchar_kept), (4, 9, 8)
                )
                self.assertEqual(
                    (g1.variable_in, g1.informative_in, g1.variable, g1.informative),
                    (2, 1, 2, 1),
                )
                self.assertAlmostEqual(g1.missing_fraction, 8 / 40)
                self.assertAlmostEqual(g1.gap_fraction, 7 / 40)
                self.assertEqual(g1.occupied_cells, 25)
                self.assertEqual((g1.start, g1.end), (1, 8))
                self.assertEqual(
                    (g2.ntax_present, g2.nchar_in, g2.nchar_kept), (3, 5, 4)
                )
                self.assertEqual((g2.variable, g2.informative), (1, 0))
                self.assertAlmostEqual(g2.missing_fraction, 8 / 20)
                self.assertAlmostEqual(g2.gap_fraction, 2 / 20)
                self.assertEqual(g2.occupied_cells, 10)
                self.assertEqual(sm.variable_sites, 3)
                self.assertEqual(sm.informative_sites, 1)
                self.assertAlmostEqual(sm.density, 35 / 60)
                self.assertEqual(sm.failed_loci, {})

    def test_partition_files_after_trimming_include_codon_runs(self):
        sm = self.make("occupancy=0.5")
        with tempfile.TemporaryDirectory() as out:
            plain = sm.write_partitions(Path(out) / "p.txt", "raxml")
            codon = sm.write_partitions(Path(out) / "c.txt", "iqtree", codon=True)
            nex = sm.write_partitions(Path(out) / "p.nex", "nexus", codon=True)
            self.assertEqual(plain.read_text(), "DNA, g1 = 1-8\nDNA, g2 = 9-12\n")
            self.assertEqual(
                codon.read_text(),
                "DNA, g1_pos1 = 1-4\\3, 6-6\n"
                "DNA, g1_pos2 = 2-5\\3, 7-7\n"
                "DNA, g1_pos3 = 3-3, 8-8\n"
                "DNA, g2 = 9-12\n",
            )
            self.assertIn("CHARSET g1_pos1 = 1-4\\3 6-6;", nex.read_text())
            self.assertIn("CHARSET g2 = 9-12;", nex.read_text())

    def test_informative_only_drops_a_locus_and_its_exclusive_taxa(self):
        for buffer in ("ram", "memmap"):
            with self.subTest(buffer=buffer):
                sm = self.make("informative-only", buffer=buffer, scratch_dir=self.tmp)
                self.assertEqual(sm.to_dict(), {"a": "T", "b": "T", "c": "C", "d": "C"})
                self.assertEqual(sm.partitions.names, ("g1",))
                self.assertEqual(sm.nchar, 1)
                self.assertEqual(
                    sm.failed_loci, {"g2": "all columns were removed by trimming"}
                )
                self.assertEqual(sm.locus_stats("g2").nchar_kept, 0)
                self.assertEqual(
                    (sm.locus_stats("g2").start, sm.locus_stats("g2").end), (0, 0)
                )

    def test_entropy_trim(self):
        sm = self.make("entropy=0.9")
        self.assertEqual(
            sm.to_dict(),
            {"a": "T?", "b": "TT", "c": "CT", "d": "C?", "e": "?C"},
        )
        self.assertEqual(sm.partitions["g1"].columns.tolist(), [1])
        self.assertEqual(sm.partitions["g2"].columns.tolist(), [3])
        sm2 = self.make(TrimSpec.parse("entropy=0.9:0.95"))
        self.assertEqual(sm2.nchar, 1)  # only g2's 0.918-bit column survives

    def test_codon_flag_ignored_for_non_multiple_of_three(self):
        pairs = sources_for(self.tmp, {"odd": ">odd_a\nACGTA\n>odd_b\nACGTA\n"})
        sm = build(pairs, kind="cds", codon_aligned=True)
        self.assertIsNone(sm.partitions["odd"].codon_phase)

    def test_codon_phase_only_for_codon_aligned_cds(self):
        pairs = sources_for(self.tmp, {"c": ">c_a\nACGTAC\n"})
        self.assertEqual(build(pairs, "cds", True).partitions["c"].codon_phase, 0)
        self.assertIsNone(build(pairs, "cds", False).partitions["c"].codon_phase)
        self.assertIsNone(build(pairs, "igs", True).partitions["c"].codon_phase)

    def test_gappyout_through_the_builder(self):
        counts = [0] * 60 + [1] * 20 + [2] * 10 + [15] * 2 + [16] * 3 + [17] * 2
        counts += [18] * 2 + [19]
        rng = np.random.default_rng(11)
        cols = rng.permutation(counts)
        grid = np.full((20, len(cols)), ord("A"), dtype=np.uint8)
        for j, n_gaps in enumerate(cols):
            grid[rng.choice(20, size=n_gaps, replace=False), j] = ord("-")
        text = "".join(f">t{i:02d}\n{grid[i].tobytes().decode()}\n" for i in range(20))
        pairs = sources_for(self.tmp, {"prof": text})
        sm = build(pairs, trim="gappyout", rename=None)
        self.assertEqual(sm.nchar, 90)
        self.assertEqual(
            sm.partitions["prof"].columns.tolist(), np.flatnonzero(cols <= 2).tolist()
        )

    def test_trim_string_and_spec_objects_are_equivalent(self):
        a = self.make("occupancy=0.5+informative-only")
        b = self.make(TrimSpec(min_occupancy=0.5, informative_only=True))
        self.assertEqual(a.sha256(), b.sha256())


# ----------------------------------------------------------------------
# result object, writers
# ----------------------------------------------------------------------
class ResultApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        pairs = sources_for(self.tmp, {"g1": G1, "g2": G2})
        self.sm = build(
            pairs,
            kind="cds",
            codon_aligned=True,
            trim="occupancy=0.5",
            rename=strip_prefix,
        )

    def test_matrix_is_read_only_view(self):
        view = self.sm.matrix
        self.assertEqual(view.shape, (5, 12))
        self.assertEqual(view.dtype, np.uint8)
        with self.assertRaises(ValueError):
            view[0, 0] = 1
        self.assertTrue(view[0].flags["C_CONTIGUOUS"])

    def test_sha256_definitions(self):
        digest = hashlib.sha256(self.sm.matrix.tobytes()).hexdigest()
        self.assertEqual(self.sm.sha256(), digest)
        path = self.sm.write_fasta(self.tmp / "out.fasta")
        self.assertEqual(
            self.sm.sha256(include_taxa=True),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertNotEqual(self.sm.sha256(), self.sm.sha256(include_taxa=True))

    def test_fasta_and_phylip_exact_bytes(self):
        fasta = self.sm.write_fasta(self.tmp / "x" / "m.fasta")
        self.assertEqual(
            fasta.read_text(),
            ">a\nATGAACCC????\n>b\nATGAACCCACTA\n>c\nACG--CC-ACTA\n"
            ">d\nGCG--C--????\n>e\n????????A-C-\n",
        )
        phy = self.sm.write_phylip(self.tmp / "m.phy")
        self.assertEqual(
            phy.read_text(),
            " 5 12\na  ATGAACCC????\nb  ATGAACCCACTA\nc  ACG--CC-ACTA\n"
            "d  GCG--C--????\ne  ????????A-C-\n",
        )
        strict = self.sm.write_phylip(self.tmp / "s.phy", relaxed=False)
        self.assertEqual(strict.read_text().splitlines()[1], "a         ATGAACCC????")

    def test_phylip_rejects_bad_names(self):
        pairs = sources_for(self.tmp, {"x": ">a_very_long_taxon_name\nACGT\n"})
        sm = build(pairs, rename=None)
        sm.write_phylip(self.tmp / "ok.phy")  # relaxed handles long names
        self.assertIn("a_very_long_taxon_name  ACGT", (self.tmp / "ok.phy").read_text())
        with self.assertRaises(ValueError):
            sm.write_phylip(self.tmp / "bad.phy", relaxed=False)
        spaced = Supermatrix(
            ("has space",), np.zeros((1, 2), np.uint8) + 65, self.sm.partitions, []
        )
        with self.assertRaises(ValueError):
            spaced.write_phylip(self.tmp / "sp.phy")

    def test_nexus_partition_formats(self):
        with_sets = self.sm.write_nexus(self.tmp / "a.nex").read_text()
        self.assertTrue(
            with_sets.startswith("#NEXUS\n\nBEGIN DATA;\n  DIMENSIONS NTAX=5 NCHAR=12;")
        )
        self.assertIn("    a    ATGAACCC????\n", with_sets)
        self.assertTrue(
            with_sets.endswith(
                "  ;\nEND;\n\nBEGIN SETS;\n  CHARSET g1 = 1-8;\n  CHARSET g2 = 9-12;\n"
                "  CHARPARTITION combined = g1:g1, g2:g2;\nEND;\n"
            )
        )
        none = self.sm.write_nexus(
            self.tmp / "b.nex", partition_format="none"
        ).read_text()
        self.assertTrue(none.endswith("  ;\nEND;\n"))
        self.assertNotIn("SETS", none)
        codon = self.sm.write_nexus(
            self.tmp / "c.nex", "nexus", codon_partitions=True
        ).read_text()
        self.assertIn("CHARSET g1_pos2 = 2-5\\3 7-7;", codon)
        with self.assertRaises(ValueError):
            self.sm.write_nexus(self.tmp / "d.nex", "bogus")  # type: ignore[arg-type]

    def test_write_partitions_all_formats(self):
        for fmt, needle in (
            ("raxml", "DNA, g1 = 1-8"),
            ("iqtree", "DNA, g2 = 9-12"),
            ("nexus", "CHARPARTITION combined"),
            ("partitionfinder", "[data_blocks]"),
        ):
            with self.subTest(fmt=fmt):
                path = self.sm.write_partitions(self.tmp / f"{fmt}.out", fmt)
                self.assertIn(needle, path.read_text())
        with self.assertRaises(ValueError):
            self.sm.write_partitions(self.tmp / "x", "bogus")  # type: ignore[arg-type]

    def test_write_all_names_and_keys(self):
        prefix = self.tmp / "out" / "nucl_2concat"
        written = self.sm.write_all(prefix)
        self.assertEqual(set(written), {"fasta", "nexus", "raxml", "nexus_partitions"})
        self.assertEqual(
            sorted(p.name for p in written.values()),
            [
                "nucl_2concat.aligned.fasta",
                "nucl_2concat.aligned.nexus",
                "nucl_2concat.partitions.nex",
                "nucl_2concat.partitions.txt",
            ],
        )
        full = self.sm.write_all(
            self.tmp / "full",
            phylip=True,
            partitions=("raxml", "nexus", "iqtree", "partitionfinder"),
            codon_partitions=True,
        )
        self.assertEqual(
            set(full),
            {
                "fasta",
                "nexus",
                "phylip",
                "raxml",
                "nexus_partitions",
                "iqtree",
                "partitionfinder",
            },
        )
        self.assertIn(
            "alignment = full.aligned.phy;", full["partitionfinder"].read_text()
        )
        self.assertIn("g1_pos3", full["raxml"].read_text())
        minimal = self.sm.write_all(
            self.tmp / "min", fasta=False, nexus=False, partitions=()
        )
        self.assertEqual(minimal, {})

    def test_close_and_context_manager(self):
        with self.sm as sm:
            self.assertEqual(sm.ntax, 5)
        self.assertIn("5 taxa x 12 sites", repr(self.sm))
        with self.assertRaises(KeyError):
            self.sm.locus_stats("nope")

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            Supermatrix(("a",), np.zeros((2, 2), np.uint8), self.sm.partitions, [])
        with self.assertRaises(ValueError):
            Supermatrix(("a",), np.zeros((1, 2), np.int32), self.sm.partitions, [])


# ----------------------------------------------------------------------
# builder options, failures, buffers
# ----------------------------------------------------------------------
class BuilderBehaviourTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_rename_callable_receives_locus_and_raw_id(self):
        calls: list[tuple[str, str]] = []

        def rename(locus: str, raw: str) -> str:
            calls.append((locus, raw))
            return raw.split("|")[0]

        pairs = sources_for(self.tmp, {"g1": ">sp1|x\nAC\n>sp2|y\nAC\n"})
        sm = build(pairs, rename=rename)
        self.assertEqual(sm.taxa, ("sp1", "sp2"))
        self.assertIn(("g1", "sp1|x"), calls)

    def test_no_rename_keeps_ids(self):
        pairs = sources_for(self.tmp, {"g1": ">g1_a\nAC\n"})
        self.assertEqual(build(pairs, rename=None).taxa, ("g1_a",))
        self.assertEqual(build(pairs).taxa, ("concat_a",))

    def test_legacy_rename_rule(self):
        self.assertEqual(legacy_concat_rename("atpA", "_R_atpA_x"), "concat_x")
        self.assertEqual(legacy_concat_rename("atpA", "atpA_x_y"), "concat_x_y")
        self.assertEqual(legacy_concat_rename("atpA", "rbcL_x"), "rbcL_x")
        self.assertEqual(legacy_concat_rename("atpA", "_R_rbcL_x"), "_R_rbcL_x")
        self.assertEqual(legacy_concat_rename("a", "_R_a_a_a"), "concat_a_a")

    def test_duplicate_ids_policy(self):
        text = ">g_a\nAC\n>g_b\nAC\n>g_a\nGT\n"
        pairs = sources_for(self.tmp, {"g": text, "h": ">h_a\nTT\n"})
        sm = build(pairs)
        self.assertEqual(list(sm.failed_loci), ["g"])
        self.assertIn("duplicate", sm.failed_loci["g"])
        sm2 = build(pairs, on_duplicate="suffix")
        self.assertEqual(sm2.failed_loci, {})
        self.assertEqual(sm2.taxa, ("concat_a", "concat_b", "concat_a__dup2"))

    def test_ragged_locus_is_skipped_not_fatal(self):
        pairs = sources_for(
            self.tmp,
            {
                "ok": ">ok_a\nACGT\n",
                "rag": ">rag_a\nACGT\n>rag_b\nACG\n",
                "ok2": ">ok2_b\nGG\n",
            },
        )
        sm = build(pairs)
        self.assertEqual(sorted(sm.failed_loci), ["rag"])
        self.assertIn("concat_b", sm.failed_loci["rag"])
        self.assertEqual(sm.to_dict(), {"concat_a": "ACGT??", "concat_b": "????GG"})

    def test_all_loci_failing_raises(self):
        pairs = sources_for(self.tmp, {"a": b"", "b": None})
        with self.assertRaises(ConcatenationError):
            build(pairs)
        with self.assertRaises(ConcatenationError):
            SupermatrixBuilder([]).build()

    def test_constructor_validation(self):
        pairs = sources_for(self.tmp, {"a": ">a_x\nAC\n"})
        src = LocusSource("a", pairs[0][1])
        with self.assertRaises(ValueError):
            SupermatrixBuilder([src, src])
        with self.assertRaises(ValueError):
            SupermatrixBuilder([src], buffer="disk")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            SupermatrixBuilder([src], trim="nonsense")
        self.assertTrue(SupermatrixBuilder([src], trim="none").trim.is_noop)

    def test_memmap_buffer_matches_ram_and_leaves_no_file(self):
        rng = np.random.default_rng(2)
        loci: dict[str, str | bytes | None] = {}
        for i in range(5):
            ids = rng.choice(30, size=20, replace=False)
            seqs = rng.choice(list("ACGT-"), size=(20, 40))
            loci[f"g{i}"] = "".join(
                f">g{i}_t{ids[j]:02d}\n{''.join(seqs[j])}\n" for j in range(20)
            )
        pairs = sources_for(self.tmp, loci)
        scratch = self.tmp / "scratch"
        ram = build(pairs, buffer="ram", trim="occupancy=0.5")
        mm = build(pairs, buffer="memmap", scratch_dir=scratch, trim="occupancy=0.5")
        self.assertIsInstance(mm.matrix, np.memmap)
        self.assertNotIsInstance(ram.matrix, np.memmap)
        self.assertEqual(ram.sha256(), mm.sha256())
        self.assertEqual(ram.taxa, mm.taxa)
        self.assertEqual(ram.partitions.entries, mm.partitions.entries)
        if os.name == "posix":
            self.assertEqual(os.listdir(scratch), [])  # unlinked right after mapping
        written = mm.write_all(self.tmp / "mm_out")
        self.assertTrue(written["fasta"].read_bytes().startswith(b">concat_t"))

    def test_auto_buffer_follows_the_memory_budget(self):
        pairs = sources_for(self.tmp, {"g": ">g_a\nACGT\n>g_b\nACGT\n"})
        scratch = self.tmp / "s"
        tiny = build(pairs, buffer="auto", mem_budget_bytes=1, scratch_dir=scratch)
        roomy = build(
            pairs, buffer="auto", mem_budget_bytes=10**12, scratch_dir=scratch
        )
        self.assertIsInstance(tiny.matrix, np.memmap)
        self.assertNotIsInstance(roomy.matrix, np.memmap)
        self.assertEqual(tiny.sha256(), roomy.sha256())

    def test_files_modified_between_passes_are_skipped_not_fatal(self):
        class Tamper(SupermatrixBuilder):
            def _process_locus(self, idx):
                loaded = super()._process_locus(idx)
                name = self._sources[idx].name
                if name == "grew":
                    loaded.nchar_in += 1
                elif name == "renamed":
                    loaded.taxa = tuple(f"new_{t}" for t in loaded.taxa)
                return loaded

        pairs = sources_for(
            self.tmp,
            {
                "ok": ">ok_a\nACGT\n",
                "grew": ">grew_a\nAC\n",
                "renamed": ">renamed_a\nAC\n",
            },
        )
        sm = Tamper(
            [LocusSource(n, p) for n, p in pairs], rename=legacy_concat_rename
        ).build()
        self.assertEqual(sorted(sm.failed_loci), ["grew", "renamed"])
        self.assertIn("width changed", sm.failed_loci["grew"])
        self.assertIn("not seen during the header scan", sm.failed_loci["renamed"])
        self.assertEqual(sm.to_dict(), {"concat_a": "ACGT"})

    def test_pass2_failure_drops_exclusive_taxa(self):
        class Flaky(SupermatrixBuilder):
            def _process_locus(self, idx):
                loaded = super()._process_locus(idx)
                if self._sources[idx].name == "bad":
                    loaded.error = "simulated I/O failure"
                return loaded

        pairs = sources_for(
            self.tmp,
            {"ok": ">ok_a\nACGT\n>ok_b\nACGT\n", "bad": ">bad_z\nGG\n>bad_a\nGG\n"},
        )
        for buffer in ("ram", "memmap"):
            sm = Flaky(
                [LocusSource(n, p) for n, p in pairs],
                rename=legacy_concat_rename,
                buffer=buffer,
                scratch_dir=self.tmp,
            ).build()
            self.assertEqual(sm.failed_loci, {"bad": "simulated I/O failure"})
            self.assertEqual(sm.taxa, ("concat_a", "concat_b"))
            self.assertEqual(sm.to_dict(), {"concat_a": "ACGT", "concat_b": "ACGT"})


# ----------------------------------------------------------------------
# determinism and bounded windows
# ----------------------------------------------------------------------
class DelayedBuilder(SupermatrixBuilder):
    """Builder whose workers finish in reverse locus order."""

    def __init__(self, *args, delays, **kwargs):
        super().__init__(*args, **kwargs)
        self._delays = delays
        self.completion: list[int] = []
        self._lock = threading.Lock()

    def _scan_locus(self, idx):
        time.sleep(self._delays[idx] / 2)
        return super()._scan_locus(idx)

    def _process_locus(self, idx):
        time.sleep(self._delays[idx])
        result = super()._process_locus(idx)
        with self._lock:
            self.completion.append(idx)
        return result


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        rng = np.random.default_rng(99)
        loci: dict[str, str | bytes | None] = {}
        for i in range(8):
            ids = rng.choice(25, size=15, replace=False)
            seqs = rng.choice(list("ACGTacgt-?"), size=(15, 30 + 3 * i))
            loci[f"loc{i}"] = "".join(
                f">loc{i}_sp{ids[j]:02d} extra\n{''.join(seqs[j])}\n" for j in range(15)
            )
        self.pairs = sources_for(self.tmp, loci)
        self.sources = [LocusSource(n, p, "cds", True) for n, p in self.pairs]

    def test_identical_sha_across_workers_windows_and_buffers(self):
        digests = {}
        for workers in (1, 2, 4, 8):
            for window in (1, 3, None):
                for buffer in ("ram", "memmap"):
                    sm = SupermatrixBuilder(
                        self.sources,
                        rename=legacy_concat_rename,
                        trim="occupancy=0.4",
                        max_workers=workers,
                        window=window,
                        buffer=buffer,
                        scratch_dir=self.tmp,
                    ).build()
                    digests[(workers, window, buffer)] = (
                        sm.sha256(include_taxa=True),
                        tuple(sm.taxa),
                        sm.partitions.entries,
                    )
        reference = next(iter(digests.values()))
        for key, value in digests.items():
            self.assertEqual(value, reference, key)

    def test_shuffled_completion_does_not_change_the_result(self):
        reference = SupermatrixBuilder(
            self.sources, rename=legacy_concat_rename, max_workers=1
        ).build()
        delays = [
            0.0 if i == len(self.sources) - 1 else 0.04 * (len(self.sources) - i)
            for i in range(len(self.sources))
        ]
        builder = DelayedBuilder(
            self.sources,
            delays=delays,
            rename=legacy_concat_rename,
            max_workers=len(self.sources),
            window=len(self.sources),
        )
        shuffled = builder.build()
        self.assertNotEqual(builder.completion, sorted(builder.completion))
        self.assertIn(
            builder.completion[0], (len(self.sources) - 1, len(self.sources) - 2)
        )
        self.assertEqual(
            shuffled.sha256(include_taxa=True), reference.sha256(include_taxa=True)
        )
        self.assertEqual(shuffled.partitions.entries, reference.partitions.entries)

    def test_ordered_map_bounds_the_in_flight_window(self):
        started: list[int] = []
        lock = threading.Lock()

        def work(i: int) -> int:
            with lock:
                started.append(i)
            time.sleep(0.005 * (10 - i % 5))
            return i * i

        with ThreadPoolExecutor(max_workers=6) as pool:
            window = 3
            out = []
            for item, result in _ordered_map(pool, work, list(range(12)), window):
                out.append((item, result))
                with lock:
                    self.assertLessEqual(len(started), item + window)
        self.assertEqual(out, [(i, i * i) for i in range(12)])

    def test_ordered_map_cancels_pending_on_early_exit(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            gen = _ordered_map(pool, lambda i: i, list(range(50)), 4)
            self.assertEqual(next(gen), (0, 0))
            gen.close()


# ----------------------------------------------------------------------
# scale / memory (run in a fresh interpreter so RSS is meaningful)
# ----------------------------------------------------------------------
SCALE_SCRIPT = r"""
import json, resource, sys, time
from pathlib import Path
import numpy as np
from plastburstalign.supermatrix import LocusSource, SupermatrixBuilder, legacy_concat_rename

ntax, nloci, nchar, wrap = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), 60
out = Path(sys.argv[1])
rng = np.random.default_rng(1)
alphabet = np.frombuffer(b"ACGTacgt-", dtype=np.uint8)
sources = []
for k in range(nloci):
    keep = np.sort(rng.choice(ntax + ntax // 10, size=ntax, replace=False))  # ~91% overlap
    data = rng.choice(alphabet, size=(ntax, nchar))
    parts = []
    for j in range(ntax):
        row = data[j].tobytes()
        body = b"\n".join(row[c:c + wrap] for c in range(0, nchar, wrap))
        parts.append(b">loc%d_sp%05d\n" % (k, keep[j]) + body + b"\n")
    path = out / f"nucl_loc{k}.aligned.fasta"
    path.write_bytes(b"".join(parts))
    sources.append(LocusSource(f"loc{k}", path, "cds", True))
del data, parts
_scale = 1 if sys.platform == "darwin" else 1024
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _scale
t0 = time.perf_counter()
sm = SupermatrixBuilder(sources, rename=legacy_concat_rename, trim=sys.argv[5] if sys.argv[5] != "none" else None,
                        buffer=sys.argv[6], scratch_dir=out, max_workers=4).build()
elapsed = time.perf_counter() - t0
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _scale
print(json.dumps({"ntax": sm.ntax, "nchar": sm.nchar, "nbytes": sm.nbytes,
                  "seconds": elapsed, "rss_before": rss0, "rss_after": rss1,
                  "informative": sm.informative_sites, "density": sm.density,
                  "memmap": type(sm.matrix).__name__}))
"""


class ScaleTests(unittest.TestCase):
    def run_scale(self, trim: str, buffer: str, ntax=3000, nloci=30, nchar=500) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    SCALE_SCRIPT,
                    tmp,
                    str(ntax),
                    str(nloci),
                    str(nchar),
                    trim,
                    buffer,
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                timeout=300,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_3000_taxa_30_loci_peak_rss_stays_near_supermatrix_size(self):
        res = self.run_scale("none", "ram")
        mb = 1024 * 1024
        delta = res["rss_after"] - res["rss_before"]
        print(
            f"\n[scale] {res['ntax']} taxa x {res['nchar']} sites "
            f"({res['nbytes'] / mb:.0f} MB) built in {res['seconds']:.2f} s; "
            f"peak RSS growth {delta / mb:.0f} MB = {delta / res['nbytes']:.2f} x matrix; "
            f"{res['informative']} informative sites, density {res['density']:.3f}"
        )
        self.assertEqual(res["nchar"], 30 * 500)
        self.assertGreaterEqual(res["ntax"], 3000)
        self.assertLess(delta, 2.5 * res["nbytes"] + 40 * mb)
        self.assertLess(res["seconds"], 60.0)

    def test_memmap_build_at_scale(self):
        res = self.run_scale("occupancy=0.5", "memmap", ntax=1500, nloci=12)
        self.assertEqual(res["memmap"], "memmap")
        self.assertEqual(
            res["nchar"], 12 * 500
        )  # random data: every column > 0.5 occupied
        self.assertLess(res["seconds"], 60.0)


if __name__ == "__main__":
    unittest.main()
