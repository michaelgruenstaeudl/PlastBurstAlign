"""Tests for the pure-Python center-star fallback aligner."""

from __future__ import annotations

import itertools
import os
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from collections import Counter
from math import comb
from pathlib import Path
from unittest import mock

from plastburstalign import builtin_aligner as ba
from plastburstalign.alignment_tools import AlignmentTool
from plastburstalign.logging_ops import logger as pba_log

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def read_fasta(path: str | Path) -> tuple[list[str], list[str]]:
    headers: list[str] = []
    seqs: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            headers.append(line[1:])
            seqs.append("")
        elif line:
            seqs[-1] += line
    return headers, seqs


def write_fasta(
    path: str | Path, records: list[tuple[str, str]], width: int = 70
) -> str:
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")
            if not seq:
                fh.write("\n")
    return str(path)


Residues = list[tuple[str, int]]  # (base, ancestral residue id)


def _evolve(
    rng: random.Random,
    parent: Residues,
    sub: float,
    indel: float,
    ids: "itertools.count[int]",
) -> Residues:
    out: Residues = []
    for base, rid in parent:
        r = rng.random()
        if r < indel / 2:
            continue  # deletion
        base2 = rng.choice("ACGT") if rng.random() < sub else base
        out.append((base2, rid))
        if indel / 2 <= r < indel:  # insertion of 1-4 brand new residues
            out.extend(
                (rng.choice("ACGT"), next(ids)) for _ in range(rng.randint(1, 4))
            )
    return out


def simulate_clade(seed: int, length: int = 300, depth: int = 3) -> list[Residues]:
    """Binary-tree simulation (2**depth leaves) tracking true homology ids."""
    rng = random.Random(seed)
    ids = itertools.count(10_000)
    root: Residues = [(rng.choice("ACGT"), i) for i in range(length)]
    level = [root]
    for _ in range(depth):
        level = [
            _evolve(rng, parent, sub=0.04, indel=0.012, ids=ids)
            for parent in level
            for _ in range(2)
        ]
    return level


def sp_scores(rows: list[str], truth: list[Residues]) -> tuple[float, float]:
    """Homology-based SP recall and precision of an alignment (rows vs truth ids)."""
    column_ids: list[list[int]] = [[] for _ in range(len(rows[0]))]
    for row, residues in zip(rows, truth):
        it = iter(residues)
        for col, ch in enumerate(row):
            if ch != "-":
                column_ids[col].append(next(it)[1])
    true_pairs = sum(
        comb(c, 2) for c in Counter(rid for res in truth for _b, rid in res).values()
    )
    hit = sum(comb(k, 2) for col in column_ids for k in Counter(col).values())
    aligned = sum(comb(len(col), 2) for col in column_ids)
    return hit / max(true_pairs, 1), hit / max(aligned, 1)


def mean_identity(rows: list[str]) -> float:
    """Mean pairwise identity over co-aligned residues."""
    total = same = 0
    for a, b in itertools.combinations(rows, 2):
        for x, y in zip(a, b):
            if x != "-" and y != "-":
                total += 1
                same += x == y
    return same / max(total, 1)


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="pba_builtin_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        ba.BuiltinAligner.reset_quality_warning()
        self.addCleanup(ba.BuiltinAligner.reset_quality_warning)

    def aligner(self, **config: object) -> ba.BuiltinAligner:
        params = {
            "alignment_tool_config": dict(config),
            "scratch_dir": str(self.tmp / "scratch"),
        }
        return ba.BuiltinAligner(params)

    def run_align(
        self,
        records: list[tuple[str, str]],
        name: str = "nuc_locus.fasta",
        **config: object,
    ) -> tuple[bool, str]:
        src = write_fasta(self.tmp / name, records)
        out = str(self.tmp / f"aligned_{name}")
        ok = self.aligner(**config).align(src, out)
        return ok, out


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #


class TestContract(_Case):
    def test_is_an_alignment_tool_with_builtin_exec_path(self) -> None:
        tool = self.aligner()
        self.assertIsInstance(tool, AlignmentTool)
        self.assertEqual(tool.exec_path, "builtin")
        self.assertEqual(tool.max_useful_threads, 1)
        self.assertTrue(tool.is_fallback)

    def test_tool_path_is_ignored_with_warning(self) -> None:
        with self.assertLogs(pba_log, level="WARNING") as cm:
            tool = ba.BuiltinAligner({}, tool_path="/usr/bin/whatever")
        self.assertEqual(tool.exec_path, "builtin")
        self.assertTrue(any("ignores tool_path" in m for m in cm.output))

    def test_execute_alias_and_threads_ignored(self) -> None:
        records = [(f"s{i}", "ACGTACGTAC" + "A" * i) for i in range(4)]
        src = write_fasta(self.tmp / "nuc_a.fasta", records)
        out = str(self.tmp / "out.fasta")
        self.assertTrue(self.aligner().execute(src, out, threads=8, sequence_count=4))
        self.assertEqual(len(read_fasta(out)[1]), 4)

    def test_missing_input_returns_false(self) -> None:
        self.assertFalse(
            self.aligner().align(str(self.tmp / "nope.fasta"), str(self.tmp / "o"))
        )


class TestInvariants(_Case):
    def setUp(self) -> None:
        super().setUp()
        self.truth = simulate_clade(seed=11, length=300)
        self.records = [
            (
                f"taxon_{i:02d} Genus species{i} [voucher {i}]",
                "".join(b for b, _ in res),
            )
            for i, res in enumerate(self.truth)
        ]

    def test_core_invariants(self) -> None:
        ok, out = self.run_align(self.records)
        self.assertTrue(ok)
        headers, rows = read_fasta(out)
        self.assertEqual(
            headers, [h for h, _ in self.records]
        )  # ids + descriptions + order
        self.assertEqual(len({len(r) for r in rows}), 1)  # equal lengths
        for (_h, original), row in zip(self.records, rows):
            self.assertEqual(row.replace("-", ""), original)  # ungapped == input
        self.assertGreaterEqual(len(rows[0]), max(len(s) for _h, s in self.records))

    def test_determinism_across_runs_and_instances(self) -> None:
        digests = set()
        for run in range(3):
            ok, out = self.run_align(self.records, name=f"nuc_run{run}.fasta")
            self.assertTrue(ok)
            digests.add(Path(out).read_bytes())
        self.assertEqual(len(digests), 1)

    def test_output_wrapping_options(self) -> None:
        ok, out = self.run_align(self.records)
        lines = Path(out).read_text().splitlines()
        self.assertTrue(
            all(len(line) <= 60 for line in lines if not line.startswith(">"))
        )
        ok, out = self.run_align(self.records, name="nuc_unwrapped.fasta", line_width=0)
        self.assertTrue(ok)
        text_lines = [
            ln for ln in Path(out).read_text().splitlines() if not ln.startswith(">")
        ]
        self.assertEqual(len(text_lines), len(self.records))  # one line per sequence

    def test_reverse_input_order_gives_equivalent_alignment(self) -> None:
        ok, fwd = self.run_align(self.records, name="nuc_fwd.fasta")
        ok2, rev = self.run_align(self.records[::-1], name="nuc_rev.fasta")
        self.assertTrue(ok and ok2)
        h1, r1 = read_fasta(fwd)
        h2, r2 = read_fasta(rev)
        self.assertEqual(h2, h1[::-1])  # ids follow the input order
        by_id_fwd, by_id_rev = dict(zip(h1, r1)), dict(zip(h2, r2))
        for h in h1:  # same ungapped sequence regardless of order
            self.assertEqual(
                by_id_fwd[h].replace("-", ""), by_id_rev[h].replace("-", "")
            )
        # exact equality is NOT required (documented); lengths must be close
        self.assertLess(abs(len(r1[0]) - len(r2[0])) / len(r1[0]), 0.10)
        self.assertLess(abs(mean_identity(r1) - mean_identity(r2)), 0.05)

    def test_shuffled_order_preserves_sequences(self) -> None:
        shuffled = self.records[:]
        random.Random(5).shuffle(shuffled)
        ok, out = self.run_align(shuffled)
        self.assertTrue(ok)
        headers, rows = read_fasta(out)
        self.assertEqual(headers, [h for h, _ in shuffled])
        for (_h, original), row in zip(shuffled, rows):
            self.assertEqual(row.replace("-", ""), original)

    def test_quality_vs_truth_and_mafft(self) -> None:
        """Report SP-score/identity vs MAFFT on closely related simulated data."""
        ok, out = self.run_align(self.records)
        self.assertTrue(ok)
        _h, rows = read_fasta(out)
        recall, precision = sp_scores(rows, self.truth)
        ident = mean_identity(rows)
        report = [
            f"builtin: SP recall={recall:.3f} precision={precision:.3f} identity={ident:.3f} cols={len(rows[0])}"
        ]
        # loose absolute bounds: closely related sequences must align well
        self.assertGreater(recall, 0.75)
        self.assertGreater(precision, 0.75)
        self.assertGreater(ident, 0.85)
        mafft = shutil.which("mafft")
        if mafft:
            src = write_fasta(self.tmp / "nuc_for_mafft.fasta", self.records)
            proc = subprocess.run(
                [mafft, "--auto", "--quiet", "--thread", "1", src],
                capture_output=True,
                text=True,
                check=True,
            )
            m_rows = [
                "".join(block.splitlines()[1:]).lower().upper()
                for block in proc.stdout.split(">")[1:]
            ]
            m_recall, m_precision = sp_scores([r.upper() for r in m_rows], self.truth)
            m_ident = mean_identity(m_rows)
            report.append(
                f"mafft  : SP recall={m_recall:.3f} precision={m_precision:.3f} identity={m_ident:.3f} cols={len(m_rows[0])}"
            )
            # builtin may be worse than MAFFT but must stay in the same ballpark
            self.assertGreater(recall, m_recall - 0.15)
            self.assertGreater(ident, m_ident - 0.10)
            self.assertLess(len(rows[0]), 1.25 * len(m_rows[0]))
        print("\n" + "\n".join(report))


class TestEdgeCases(_Case):
    def test_case_iupac_and_stop_preserved_and_gaps_removed(self) -> None:
        records = [
            ("a", "ACGTRYacgtNNACGT"),
            ("b", "ACG-TRYacgt..NNACGT"),  # input gaps are stripped
            ("c", "ACGTRYacgtNACGT"),
        ]
        ok, out = self.run_align(records)
        self.assertTrue(ok)
        _h, rows = read_fasta(out)
        self.assertEqual(rows[0].replace("-", ""), "ACGTRYacgtNNACGT")
        self.assertEqual(rows[1].replace("-", ""), "ACGTRYacgtNNACGT")
        self.assertEqual(len({len(r) for r in rows}), 1)

    def test_rna_u_aligns_to_t(self) -> None:
        records = [("dna", "ACGTACGTACGTAAAT"), ("rna", "ACGUACGUACGUAAAU")]
        ok, out = self.run_align(records)
        self.assertTrue(ok)
        _h, rows = read_fasta(out)
        self.assertEqual([len(r) for r in rows], [16, 16])  # U matches T, no gaps

    def test_empty_sequence_becomes_all_gap_row(self) -> None:
        records = [("a", "ACGTACGTAC"), ("empty", ""), ("b", "ACGTACGTAC")]
        ok, out = self.run_align(records)
        self.assertTrue(ok)
        _h, rows = read_fasta(out)
        self.assertEqual(set(rows[1]), {"-"})
        self.assertEqual(len(rows[1]), len(rows[0]))

    def test_all_empty_and_single_sequence(self) -> None:
        ok, out = self.run_align([("a", ""), ("b", "")])
        self.assertTrue(ok)
        ok, out = self.run_align([("only", "ACGTTGCA")], name="nuc_single.fasta")
        self.assertTrue(ok)
        self.assertEqual(read_fasta(out), (["only"], ["ACGTTGCA"]))

    def test_identical_sequences_no_gaps(self) -> None:
        ok, out = self.run_align([(f"s{i}", "ACGTAGCTAGCTAGGATC") for i in range(5)])
        self.assertTrue(ok)
        _h, rows = read_fasta(out)
        self.assertEqual(set(rows), {"ACGTAGCTAGCTAGGATC"})

    def test_duplicate_ids_are_kept_in_order(self) -> None:
        records = [("dup", "ACGTACGTAC"), ("dup", "ACGTACGAAC"), ("x", "ACGTACGTAC")]
        ok, out = self.run_align(records)
        self.assertTrue(ok)
        headers, rows = read_fasta(out)
        self.assertEqual(headers, ["dup", "dup", "x"])
        self.assertEqual(rows[1].replace("-", ""), "ACGTACGAAC")

    def test_invalid_inputs(self) -> None:
        empty = self.tmp / "nuc_empty.fasta"
        empty.write_text("")
        self.assertFalse(self.aligner().align(str(empty), str(self.tmp / "o1")))
        bad = self.tmp / "nuc_bad.fasta"
        bad.write_text("ACGT\n>late header\nACGT\n")
        self.assertFalse(self.aligner().align(str(bad), str(self.tmp / "o2")))
        self.assertFalse((self.tmp / "o2").exists())

    def test_output_directory_is_created_and_write_is_atomic(self) -> None:
        records = [("a", "ACGTACGT"), ("b", "ACGTACGT")]
        src = write_fasta(self.tmp / "nuc_x.fasta", records)
        out = self.tmp / "deep" / "dir" / "aligned.fasta"
        self.assertTrue(self.aligner().align(src, str(out)))
        self.assertTrue(out.exists())
        with mock.patch.object(ba, "_render_rows", side_effect=RuntimeError("boom")):
            out2 = self.tmp / "deep" / "dir" / "fail.fasta"
            self.assertFalse(self.aligner().align(src, str(out2)))
        self.assertFalse(out2.exists())
        leftovers = [p.name for p in (self.tmp / "deep" / "dir").iterdir()]
        self.assertEqual(leftovers, ["aligned.fasta"])  # no temp files left behind


class TestProtein(_Case):
    PROTS = [
        ("p1", "MKVLAAGIVGLLLAQPSTEEW"),
        ("p2", "MKVLAAGIVGLLLAQPSTEEW"),
        ("p3", "MKVLAGIVGLLLAQPSTEW"),
        ("p4", "MKVIAAGIVALLLAQPTEEW"),
        ("p5", "MKVLAAGIVGLLLAQPSTEEWRR"),
    ]

    def test_prot_prefix_selects_blosum(self) -> None:
        ok, out = self.run_align(self.PROTS, name="prot_gene.fasta")
        self.assertTrue(ok)
        _h, rows = read_fasta(out)
        self.assertEqual(len({len(r) for r in rows}), 1)
        for (_h, original), row in zip(self.PROTS, rows):
            self.assertEqual(row.replace("-", ""), original)
        # identical sequences p1/p2 stay gap-free relative to each other
        self.assertEqual(rows[0], rows[1])
        # the shared prefix must align without internal gaps in p4 (conservative I/L)
        self.assertEqual(rows[3][:4], "MKVI")

    def test_explicit_alphabet_kwarg_and_stop_codon(self) -> None:
        records = [("a", "MKVLAAG*"), ("b", "MKVLAG*"), ("c", "MKVLAAG*")]
        src = write_fasta(self.tmp / "genefile.fasta", records)
        out = str(self.tmp / "o.fasta")
        self.assertTrue(self.aligner().align(src, out, alphabet="protein"))
        _h, rows = read_fasta(out)
        self.assertEqual(rows[0].replace("-", ""), "MKVLAAG*")
        self.assertEqual(len({len(r) for r in rows}), 1)
        self.assertTrue(rows[0].endswith("*") and rows[1].endswith("*"))

    def test_protein_vs_nucleotide_scoring_differ(self) -> None:
        seqs = ["MKVLAAGIVGLLL", "MKVLAGIVGLLL", "MKVLAAGIVGLLL"]
        as_prot = ba.center_star_align(seqs, protein=True)
        self.assertEqual(as_prot[1].replace("-", ""), seqs[1])
        self.assertEqual(len({len(r) for r in as_prot}), 1)


class TestGuards(_Case):
    RECORDS = [(f"s{i}", "ACGTACGTAC" * 5) for i in range(6)]

    def _assert_refused(self, **config: object) -> None:
        src = write_fasta(self.tmp / "nuc_g.fasta", self.RECORDS)
        out = self.tmp / "g_out.fasta"
        with self.assertLogs(pba_log, level="ERROR") as cm:
            ok = self.aligner(**config).align(src, str(out))
        self.assertFalse(ok)
        self.assertFalse(out.exists())
        message = " ".join(cm.output)
        self.assertIn("refuses", message)
        self.assertIn("conda install -c bioconda mafft", message)
        self.assertIn("brew install mafft", message)

    def test_max_sequences(self) -> None:
        self._assert_refused(max_sequences=3)

    def test_max_total_cells(self) -> None:
        self._assert_refused(max_total_cells=1000)

    def test_max_pair_cells(self) -> None:
        self._assert_refused(max_pair_cells=100)

    def test_sequence_count_hint_fails_fast(self) -> None:
        src = write_fasta(self.tmp / "nuc_h.fasta", self.RECORDS)
        with mock.patch.object(
            ba, "_read_fasta", side_effect=AssertionError("must not read")
        ):
            with self.assertLogs(pba_log, level="ERROR"):
                ok = self.aligner(max_sequences=10).align(
                    src, str(self.tmp / "h.fasta"), sequence_count=5000
                )
        self.assertFalse(ok)

    def test_kwargs_override_config(self) -> None:
        src = write_fasta(self.tmp / "nuc_k.fasta", self.RECORDS)
        tool = self.aligner(max_sequences=2)
        with self.assertLogs(pba_log, level="ERROR"):
            self.assertFalse(tool.align(src, str(self.tmp / "k1.fasta")))
        self.assertTrue(tool.align(src, str(self.tmp / "k2.fasta"), max_sequences=100))

    def test_defaults_allow_realistic_small_loci(self) -> None:
        cfg = ba.BuiltinAlignerConfig()
        self.assertGreaterEqual(cfg.max_sequences, 1000)
        self.assertGreater(cfg.max_total_cells, 1e9)

    def test_center_star_align_raises_size_guard(self) -> None:
        with self.assertRaises(ba.SizeGuardError):
            ba.center_star_align(
                ["ACGT"] * 5, config=ba.BuiltinAlignerConfig(max_sequences=2)
            )

    def test_config_parsing_is_tolerant(self) -> None:
        cfg = ba.BuiltinAlignerConfig.from_params(
            {
                "max_sequences": "12",
                "max_total_cells": "1e6",
                "center_sample_size": 0,
                "gap_open": "-8",
            },
            {"alphabet": "PROTEIN"},
        )
        self.assertEqual((cfg.max_sequences, cfg.max_total_cells), (12, 1e6))
        self.assertEqual(cfg.center_sample_size, 1)
        self.assertEqual((cfg.gap_open, cfg.alphabet), (-8.0, "protein"))


class TestQualityWarning(_Case):
    def test_loud_warning_emitted_once_per_process(self) -> None:
        records = [("a", "ACGTACGT"), ("b", "ACGTACGA")]
        with self.assertLogs(pba_log, level="WARNING") as cm:
            for i in range(3):
                src = write_fasta(self.tmp / f"nuc_w{i}.fasta", records)
                self.assertTrue(self.aligner().align(src, str(self.tmp / f"w{i}.out")))
        warnings = [m for m in cm.output if "FALLBACK ALIGNER" in m]
        self.assertEqual(len(warnings), 1)
        self.assertIn("LOWER ACCURACY", warnings[0])
        self.assertIn("conda install -c bioconda mafft", warnings[0])


class TestDirectApiAndSpeed(_Case):
    def test_center_star_align_direct(self) -> None:
        rows = ba.center_star_align(
            ["ACGTACGT", "ACGACGT", "ACGTTACGT"], ids=["x", "y", "z"]
        )
        self.assertEqual(
            [r.replace("-", "") for r in rows], ["ACGTACGT", "ACGACGT", "ACGTTACGT"]
        )
        self.assertEqual(len({len(r) for r in rows}), 1)
        self.assertEqual(ba.center_star_align([]), [])

    def test_center_choice_is_input_order_independent_for_distinct_lengths(
        self,
    ) -> None:
        seqs = ["ACGTACGTAC", "ACGTACGTACGG", "ACGTACGTTC", "ACGAACGTAC", "TTTTACGTAC"]
        ids = [f"s{i}" for i in range(5)]
        fwd = ba.center_star_align(seqs, ids=ids)
        rev = ba.center_star_align(seqs[::-1], ids=ids[::-1])
        self.assertEqual(len(fwd[0]), len(rev[0]))
        self.assertEqual([r.replace("-", "") for r in rev], seqs[::-1])

    def test_moderate_locus_runs_in_reasonable_time(self) -> None:
        truth = simulate_clade(seed=3, length=1200, depth=4)  # 16 leaves, ~1.2 kb
        seqs = ["".join(b for b, _ in res) for res in truth]
        start = time.perf_counter()
        rows = ba.center_star_align(seqs)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 30.0)
        self.assertEqual(len({len(r) for r in rows}), 1)
        recall, _precision = sp_scores(rows, truth)
        self.assertGreater(recall, 0.7)


class TestNoExternalBinaryNeeded(_Case):
    def test_works_when_no_aligner_on_path(self) -> None:
        records = [("a", "ACGTACGTAC"), ("b", "ACGTACGAAC")]
        src = write_fasta(self.tmp / "nuc_np.fasta", records)
        with mock.patch.dict(os.environ, {"PATH": str(self.tmp / "empty")}):
            self.assertTrue(self.aligner().align(src, str(self.tmp / "np.out")))


if __name__ == "__main__":
    unittest.main()
