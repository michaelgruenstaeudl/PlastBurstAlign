"""Unit and parity tests for ``plastburstalign.matrix_ops`` (dense engine)."""

from __future__ import annotations

import io
import random
import tempfile
import time
import unittest
from functools import partial
from pathlib import Path

import numpy as np
from mafft_fixtures import (
    align_with_mafft,
    legacy_load_fasta,
    mafft_available,
    simulate_locus,
)

from plastburstalign.matrix_ops import (
    CANONICAL_LUT,
    GAP_LUT,
    IDENTITY_LUT,
    UPPER_LUT,
    AlignmentFormatError,
    DenseLocusMatrix,
    make_lut,
    scan_fasta,
)
from plastburstalign.supermatrix import legacy_concat_rename


class LutTests(unittest.TestCase):
    def test_tables_have_256_entries(self):
        for table in (IDENTITY_LUT, UPPER_LUT, GAP_LUT, CANONICAL_LUT):
            self.assertEqual(len(table), 256)

    def test_upper_and_gap_mapping(self):
        self.assertEqual(b"acgtn-.?RYry".translate(UPPER_LUT), b"ACGTN-.?RYRY")
        self.assertEqual(b"acgt-.?".translate(GAP_LUT), b"acgt--?")
        self.assertEqual(b"acgt-.?".translate(CANONICAL_LUT), b"ACGT--?")
        self.assertEqual(bytes(range(256)).translate(IDENTITY_LUT), bytes(range(256)))
        self.assertEqual(make_lut(upper=False, gaps=False), IDENTITY_LUT)

    def test_non_letters_untouched_by_upper(self):
        data = bytes(range(256))
        up = data.translate(UPPER_LUT)
        for code in range(256):
            expected = code - 32 if ord("a") <= code <= ord("z") else code
            self.assertEqual(up[code], expected)


class ParserTests(unittest.TestCase):
    def test_wrapped_multiline(self):
        raw = b">a\nACGT\nACGT\nAC\n>b\nAC-T\nACGT\nA-\n"
        m = DenseLocusMatrix.from_fasta_bytes(raw)
        self.assertEqual(m.taxa, ("a", "b"))
        self.assertEqual((m.ntax, m.nchar), (2, 10))
        self.assertEqual(m.to_dict(), {"a": "ACGTACGTAC", "b": "AC-TACGTA-"})

    def test_layout_properties(self):
        m = DenseLocusMatrix.from_fasta_bytes(b">a\nACG\n>b\nACT\n")
        self.assertEqual(m.matrix.dtype, np.uint8)
        self.assertTrue(m.matrix.flags["C_CONTIGUOUS"])
        self.assertEqual(m.matrix.shape, (2, 3))
        self.assertEqual(m.nbytes, 6)
        self.assertEqual(m.ntax, 2)
        self.assertIn("ntax=2", repr(m))

    def test_crlf_blank_lines_and_missing_trailing_newline(self):
        crlf = b">a\r\nAC\r\nGT\r\n\r\n>b\r\nA-\r\n-T"
        m = DenseLocusMatrix.from_fasta_bytes(crlf)
        self.assertEqual(m.to_dict(), {"a": "ACGT", "b": "A--T"})
        blank = b"\n\n>a\n\nAC\n\n\nGT\n\n>b\nA--T\n\n\n"
        self.assertEqual(
            DenseLocusMatrix.from_fasta_bytes(blank).to_dict(), m.to_dict()
        )

    def test_header_description_and_rename(self):
        raw = b">locus_a some description\nACGT\n>_R_locus_b|x y\nAC-T\n>c\nACGT\n"
        m = DenseLocusMatrix.from_fasta_bytes(
            raw, rename=partial(legacy_concat_rename, "locus")
        )
        self.assertEqual(m.taxa, ("concat_a", "concat_b|x", "c"))

    def test_normalize_case_flags(self):
        raw = b">a\nacgtn.\n>b\nACGTN-\n"
        norm = DenseLocusMatrix.from_fasta_bytes(raw)
        self.assertEqual(norm.to_dict(), {"a": "ACGTN-", "b": "ACGTN-"})
        verbatim = DenseLocusMatrix.from_fasta_bytes(raw, normalize_case=False)
        self.assertEqual(verbatim.to_dict(), {"a": "acgtn.", "b": "ACGTN-"})
        upper_only = DenseLocusMatrix.from_fasta_bytes(raw, normalize_gaps=False)
        self.assertEqual(upper_only.to_dict()["a"], "ACGTN.")
        gaps_only = DenseLocusMatrix.from_fasta_bytes(
            raw, normalize_case=False, normalize_gaps=True
        )
        self.assertEqual(gaps_only.to_dict()["a"], "acgtn-")

    def test_iupac_and_missing_are_lossless(self):
        raw = b">a\nRYKMSWBDHVNn?-\n"
        m = DenseLocusMatrix.from_fasta_bytes(raw, normalize_case=False)
        self.assertEqual(m.to_dict()["a"], "RYKMSWBDHVNn?-")

    def test_inner_whitespace_is_removed(self):
        m = DenseLocusMatrix.from_fasta_bytes(b">a\n AC GT \t\n>b\nACGT\n")
        self.assertEqual(m.to_dict(), {"a": "ACGT", "b": "ACGT"})

    def test_empty_and_blank_inputs(self):
        for raw in (b"", b"\n\n  \n", b"\r\n"):
            m = DenseLocusMatrix.from_fasta_bytes(raw)
            self.assertEqual((m.ntax, m.nchar), (0, 0))
        zero = DenseLocusMatrix.from_fasta_bytes(b">a\n>b\n")
        self.assertEqual((zero.ntax, zero.nchar), (2, 0))

    def test_duplicate_ids_raise_by_default(self):
        with self.assertRaises(AlignmentFormatError) as ctx:
            DenseLocusMatrix.from_fasta_bytes(b">a\nAC\n>b\nAC\n>a\nGT\n")
        self.assertIn("'a'", str(ctx.exception))

    def test_duplicate_ids_suffix_is_deterministic(self):
        raw = b">a\nAA\n>a\nCC\n>a\nGG\n>a__dup2\nTT\n"
        m = DenseLocusMatrix.from_fasta_bytes(raw, on_duplicate="suffix")
        self.assertEqual(m.taxa, ("a", "a__dup2", "a__dup3", "a__dup2__dup2"))
        again = DenseLocusMatrix.from_fasta_bytes(raw, on_duplicate="suffix")
        self.assertEqual(m.taxa, again.taxa)
        self.assertEqual(m.to_dict()["a__dup3"], "GG")

    def test_duplicates_detected_after_rename(self):
        raw = b">loc_x\nAC\n>_R_loc_x\nGT\n"
        with self.assertRaises(AlignmentFormatError):
            DenseLocusMatrix.from_fasta_bytes(
                raw, rename=partial(legacy_concat_rename, "loc")
            )

    def test_ragged_names_the_offending_id(self):
        with self.assertRaises(AlignmentFormatError) as ctx:
            DenseLocusMatrix.from_fasta_bytes(b">a\nACGT\n>bad\nACG\n>c\nACGT\n")
        self.assertIn("'bad'", str(ctx.exception))
        with self.assertRaises(AlignmentFormatError) as ctx2:
            scan_fasta(b">a\nACGT\n>bad\nACG\n", validate_lengths=True)
        self.assertIn("'bad'", str(ctx2.exception))

    def test_malformed_headers_and_preamble(self):
        with self.assertRaises(AlignmentFormatError):
            DenseLocusMatrix.from_fasta_bytes(b"ACGT\n>a\nACGT\n")
        with self.assertRaises(AlignmentFormatError):
            DenseLocusMatrix.from_fasta_bytes(b">\nACGT\n")
        with self.assertRaises(AlignmentFormatError):
            DenseLocusMatrix.from_fasta_bytes(b">a\nAC\n> \nAC\n")

    def test_rename_returning_invalid_id_raises(self):
        with self.assertRaises(AlignmentFormatError):
            DenseLocusMatrix.from_fasta_bytes(b">a\nAC\n", rename=lambda _: "")
        with self.assertRaises(AlignmentFormatError):
            DenseLocusMatrix.from_fasta_bytes(b">a\nAC\n", rename=lambda _: "x y")

    def test_scan_matches_parse(self):
        raw = b">a d\nAC\nGT\n>b\nA-\n-T\n"
        scan = scan_fasta(raw, validate_lengths=True)
        self.assertEqual(scan.ids, ("a", "b"))
        self.assertEqual(scan.nchar, 4)
        self.assertEqual(len(scan.spans), 2)

    def test_scan_validation_counts_crlf_and_spaces(self):
        raw = b">a x y\r\nAC GT\r\nAC\r\n>b\r\nACGTAC\r\n>c\nA C G T A C"
        scan = scan_fasta(raw, validate_lengths=True)
        self.assertEqual((scan.ids, scan.nchar), (("a", "b", "c"), 6))

    def test_unicode_ids_roundtrip(self):
        raw = ">Ünïcode_taxon\nACGT\n".encode()
        m = DenseLocusMatrix.from_fasta_bytes(raw)
        self.assertEqual(m.taxa, ("Ünïcode_taxon",))
        self.assertEqual(m.to_fasta_bytes(), raw)


class PathAndMmapTests(unittest.TestCase):
    def test_path_bytes_and_mmap_agree(self):
        raw = b">a\nACGT\nac-t\n>b\nAC.T\nAC-T\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.fasta"
            p.write_bytes(raw)
            ref = DenseLocusMatrix.from_fasta_bytes(raw)
            for use_mmap in (False, True):
                m = DenseLocusMatrix.from_fasta_path(p, use_mmap=use_mmap)
                self.assertEqual(m.taxa, ref.taxa)
                self.assertTrue(np.array_equal(m.matrix, ref.matrix))
            # parsed arrays must not depend on the (closed) mapping
            m = DenseLocusMatrix.from_fasta_path(p, use_mmap=True)
            self.assertEqual(m.to_dict()["a"], "ACGTAC-T")

    def test_empty_file_with_mmap_and_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.fasta"
            p.write_bytes(b"")
            m = DenseLocusMatrix.from_fasta_path(p, use_mmap=True)
            self.assertEqual((m.ntax, m.nchar), (0, 0))
            p.write_bytes(b">loc_a\nAC\n")
            m = DenseLocusMatrix.from_fasta_path(
                p, rename=partial(legacy_concat_rename, "loc"), use_mmap=True
            )
            self.assertEqual(m.taxa, ("concat_a",))


class MatrixApiTests(unittest.TestCase):
    def setUp(self):
        self.m = DenseLocusMatrix.from_sequences(
            {"a": "AC-GT?", "b": "ACGG.T", "c": "NNNNNN"}
        )

    def test_constructor_validation(self):
        good = np.zeros((2, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            DenseLocusMatrix(("a",), good)
        with self.assertRaises(ValueError):
            DenseLocusMatrix(("a", "b"), good.astype(np.int32))
        with self.assertRaises(ValueError):
            DenseLocusMatrix(("a", "b"), good.ravel())
        with self.assertRaises(ValueError):
            DenseLocusMatrix(("a", "b"), good, nchar=4)
        self.assertEqual(DenseLocusMatrix(("a", "b"), good, nchar=3).nchar, 3)

    def test_non_contiguous_input_is_made_contiguous(self):
        base = np.zeros((3, 4), dtype=np.uint8)
        m = DenseLocusMatrix(("a", "b", "c", "d"), base.T)
        self.assertTrue(m.matrix.flags["C_CONTIGUOUS"])

    def test_row_index_and_ungapped(self):
        self.assertEqual(self.m.index_of("b"), 1)
        self.assertEqual(bytes(self.m.row("a")), b"AC-GT?")
        self.assertEqual(bytes(self.m.row(2)), b"NNNNNN")
        self.assertEqual(self.m.ungapped("a"), b"ACGT?")
        self.assertEqual(self.m.ungapped(0, strip_missing=True), b"ACGT")
        self.assertEqual(self.m.ungapped("b"), b"ACGGT")
        with self.assertRaises(KeyError):
            self.m.index_of("zzz")

    def test_select_columns_mask_and_indices(self):
        mask = np.array([True, False, True, False, False, True])
        sub = self.m.select_columns(mask)
        self.assertEqual(sub.to_dict(), {"a": "A-?", "b": "AGT", "c": "NNN"})
        self.assertTrue(sub.matrix.flags["C_CONTIGUOUS"])
        self.assertEqual(sub.taxa, self.m.taxa)
        by_idx = self.m.select_columns([5, 0, 0])
        self.assertEqual(by_idx.to_dict()["a"], "?AA")
        self.assertEqual(self.m.select_columns([-1]).to_dict()["b"], "T")
        self.assertEqual(self.m.select_columns([]).nchar, 0)
        # original is untouched
        self.assertEqual(self.m.to_dict()["a"], "AC-GT?")

    def test_select_columns_errors(self):
        with self.assertRaises(ValueError):
            self.m.select_columns(np.ones(3, dtype=bool))
        with self.assertRaises(TypeError):
            self.m.select_columns([0.5, 1.5])
        with self.assertRaises(IndexError):
            self.m.select_columns([99])

    def test_slice_columns_returns_independent_copy(self):
        sl = self.m.slice_columns(1, 4)
        self.assertEqual(sl.to_dict(), {"a": "C-G", "b": "CGG", "c": "NNN"})
        self.assertTrue(sl.matrix.flags["C_CONTIGUOUS"])
        sl.matrix[0, 0] = ord("X")
        self.assertEqual(self.m.to_dict()["a"], "AC-GT?")
        self.assertEqual(self.m.slice_columns(4).nchar, 2)

    def test_normalized_copy(self):
        m = DenseLocusMatrix.from_sequences({"a": "acg.", "b": "ACG-"})
        n = m.normalized()
        self.assertEqual(n.to_dict(), {"a": "ACG-", "b": "ACG-"})
        self.assertEqual(m.to_dict()["a"], "acg.")

    def test_from_sequences_ragged_and_empty(self):
        with self.assertRaises(AlignmentFormatError):
            DenseLocusMatrix.from_sequences({"a": "AC", "b": "A"})
        self.assertEqual(DenseLocusMatrix.from_sequences({}).ntax, 0)

    def test_fasta_output_roundtrip_wrapped_and_unwrapped(self):
        self.assertEqual(
            self.m.to_fasta_bytes(), b">a\nAC-GT?\n>b\nACGG.T\n>c\nNNNNNN\n"
        )
        self.assertEqual(
            self.m.to_fasta_bytes(wrap=4),
            b">a\nAC-G\nT?\n>b\nACGG\n.T\n>c\nNNNN\nNN\n",
        )
        back = DenseLocusMatrix.from_fasta_bytes(
            self.m.to_fasta_bytes(wrap=4), normalize_case=False
        )
        self.assertEqual(back.to_dict(), self.m.to_dict())
        with self.assertRaises(ValueError):
            self.m.write_fasta(io.BytesIO(), wrap=0)

    def test_write_fasta_to_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sub" / "out.fasta"
            self.m.write_fasta(p)
            self.assertEqual(p.read_bytes(), self.m.to_fasta_bytes())


# ----------------------------------------------------------------------
# parity against the legacy dict loader
# ----------------------------------------------------------------------
EDGE_CASES: dict[str, bytes] = {
    "plain": b">loc_a\nACGT\n>loc_b\nAC-T\n",
    "wrapped_lower": b">loc_a\nacg\ntac\n>_R_loc_b\nac-\nt-c\n>loc_c d e\nacgtac\n",
    "crlf": b">loc_a\r\nACGT\r\n>loc_b\r\nACGA\r\n",
    "blank_lines": b"\n>loc_a\n\nACGT\n\n>loc_b\nAC\n\nGT\n\n",
    "no_trailing_newline": b">loc_a\nACGT\n>b\nACGT",
    "foreign_ids": b">other_a\nACGT\n>x\nA-GT\n>loc_loc_z\nACGT\n",
    "iupac_missing": b">loc_a\nRYKM?-NN\n>loc_b\nacgtn?-.\n",
    "single_taxon": b">loc_only\nACGTACGT\n",
}


class LegacyParityTests(unittest.TestCase):
    """New parser == legacy ``_load_alignment_fast`` FASTA loader (normalization off)."""

    def _assert_parity(self, locus: str, raw: bytes) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / f"nucl_{locus}.aligned.fasta"
            p.write_bytes(raw)
            legacy = legacy_load_fasta(locus, p)
            self.assertIsNotNone(legacy)
            for use_mmap in (False, True):
                new = DenseLocusMatrix.from_fasta_path(
                    p,
                    rename=partial(legacy_concat_rename, locus),
                    normalize_case=False,
                    use_mmap=use_mmap,
                )
                self.assertEqual(list(new.taxa), legacy.taxlabels)
                self.assertEqual(new.to_dict(), legacy.matrix)
                self.assertEqual(new.nchar, legacy.nchar)
            # with normalization the only difference is upper-casing / '.' -> '-'
            norm = DenseLocusMatrix.from_fasta_path(
                p, rename=partial(legacy_concat_rename, locus)
            )
            expected = {
                k: v.upper().replace(".", "-") for k, v in legacy.matrix.items()
            }
            self.assertEqual(norm.to_dict(), expected)

    def test_hand_built_edge_cases(self):
        for label, raw in EDGE_CASES.items():
            with self.subTest(case=label):
                self._assert_parity("loc", raw)

    @unittest.skipUnless(mafft_available(), "mafft is not installed")
    def test_real_mafft_alignments(self):
        rng = random.Random(20260101)
        taxa = [f"taxon{i:02d}" for i in range(12)]
        with tempfile.TemporaryDirectory() as tmp:
            for locus, adjust in (("atpA", False), ("rbcL", False), ("matK", True)):
                reversed_taxa = taxa[2:4] if adjust else ()
                seqs = simulate_locus(rng, taxa, 140, reversed_taxa=reversed_taxa)
                path = align_with_mafft(locus, seqs, Path(tmp), adjust)
                raw = path.read_bytes()
                self.assertTrue(
                    any(c in raw for c in b"acgt"), "MAFFT output should be lowercase"
                )
                with self.subTest(locus=locus):
                    self._assert_parity(locus, raw)
                    m = DenseLocusMatrix.from_fasta_path(
                        path, rename=partial(legacy_concat_rename, locus)
                    )
                    self.assertEqual(m.ntax, len(taxa))
                    if adjust:
                        self.assertIn(
                            b"_R_", raw, "MAFFT should flag reversed sequences"
                        )
                    else:
                        for i, taxon in enumerate(taxa):
                            self.assertEqual(
                                m.ungapped(f"concat_{taxon}"), seqs[taxon].encode()
                            )
                            self.assertEqual(m.taxa[i], f"concat_{taxon}")


class PerformanceTests(unittest.TestCase):
    def test_parse_2000_taxa_x_3000_columns(self):
        ntax, nchar, wrap = 2000, 3000, 60
        rng = np.random.default_rng(7)
        alphabet = np.frombuffer(b"acgt-", dtype=np.uint8)
        data = rng.choice(alphabet, size=(ntax, nchar))
        parts: list[bytes] = []
        for i in range(ntax):
            row = data[i].tobytes()
            body = b"\n".join(row[c : c + wrap] for c in range(0, nchar, wrap))
            parts.append(b">taxon%05d description\n" % i + body + b"\n")
        raw = b"".join(parts)
        t0 = time.perf_counter()
        m = DenseLocusMatrix.from_fasta_bytes(raw)
        elapsed = time.perf_counter() - t0
        mb = len(raw) / 1e6
        print(
            f"\n[perf] parsed {ntax}x{nchar} ({mb:.1f} MB wrapped FASTA) in "
            f"{elapsed * 1000:.0f} ms = {mb / elapsed:.0f} MB/s"
        )
        self.assertEqual((m.ntax, m.nchar), (ntax, nchar))
        expected = np.frombuffer(data.tobytes().upper(), dtype=np.uint8)
        self.assertTrue(np.array_equal(m.matrix, expected.reshape(ntax, nchar)))
        self.assertEqual(m.taxa[1234], "taxon01234")
        self.assertLess(elapsed, 10.0)


if __name__ == "__main__":
    unittest.main()
