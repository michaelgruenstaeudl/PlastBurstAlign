"""Tests of the anchor-based sub-locus chunker (chunking.py).

The hard invariants are checked with the real MAFFT on simulated loci:
every spliced row ungaps to the exact input sequence, all rows have the same
length, row order and headers are preserved, and the plan depends only on the
data and the configuration (not on the host, the hash seed or the input
container).
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from plastburstalign._simulate import (
    reverse_complement,
    simulate_locus,
    write_fasta,
)
from plastburstalign.alignment_tools import MAFFT
from plastburstalign.chunking import (
    ChunkConfig,
    ChunkPlan,
    ChunkSpliceError,
    _is_low_complexity,
    plan_chunks,
    plan_chunks_with_reason,
    splice_alignments,
)

HAS_MAFFT = shutil.which("mafft") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]

SMALL = ChunkConfig(min_locus_len=1000, target_chunk_len=600)
SMALL_PROT = ChunkConfig(min_locus_len_prot=500, target_chunk_len_prot=250)


def occurrences(seq: str, kmer: str) -> int:
    """Number of (possibly overlapping) occurrences of ``kmer`` in ``seq``."""
    return len(re.findall(f"(?={re.escape(kmer)})", seq))


def read_fasta(path: str | os.PathLike[str]) -> list[tuple[str, str]]:
    records: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                records.append((line[1:], []))
            elif line and records:
                records[-1][1].append(line)
    return [(h, "".join(p)) for h, p in records]


def simulated(
    n: int = 10, length: int = 1800, seed: int = 7, **kwargs: Any
) -> list[tuple[str, str]]:
    kwargs.setdefault("divergence", 0.2)
    kwargs.setdefault("conserved_fraction", 0.4)
    return simulate_locus(n, length, seed=seed, **kwargs)


class PlanPropertyTests(unittest.TestCase):
    def check_plan(self, plan: ChunkPlan, seqs: list[str], k: int) -> None:
        """Properties every plan must have."""
        self.assertEqual(plan.n_seq, len(seqs))
        self.assertEqual(plan.n_chunks, len(plan.anchors) + 1)
        for row, seq in enumerate(seqs):
            pieces = [plan.chunk_piece(row, c) for c in range(plan.n_chunks)]
            self.assertEqual("".join(pieces), seq)  # a partition of the sequence
            # strictly collinear (and non-overlapping) anchor chain
            starts = [a.positions[row] for a in plan.anchors]
            self.assertEqual(starts, sorted(set(starts)))
            self.assertTrue(all(b - a >= k for a, b in zip(starts, starts[1:])))
        for anchor in plan.anchors:
            self.assertEqual(len(anchor.kmer), k)
            self.assertFalse(_is_low_complexity(anchor.kmer, plan.config))
            for row, seq in enumerate(seqs):
                upper = seq.upper()
                self.assertEqual(occurrences(upper, anchor.kmer), 1, (anchor, row))
                self.assertEqual(upper.find(anchor.kmer), anchor.positions[row])
                # the anchor k-mer starts the right-hand chunk
                piece = plan.chunk_piece(row, anchor.index + 1)
                self.assertTrue(piece.upper().startswith(anchor.kmer))

    def test_plan_properties_nucleotides(self) -> None:
        records = simulated()
        plan = plan_chunks(records, "nucl", SMALL, name="locus")
        assert plan is not None
        self.assertGreaterEqual(plan.n_chunks, 2)
        self.check_plan(plan, [s for _, s in records], 15)
        self.assertEqual(plan.alphabet, "nucl")
        self.assertEqual(plan.ids, tuple(n for n, _ in records))

    def test_plan_properties_protein(self) -> None:
        records = simulated(8, 700, seed=3, alphabet="prot", conserved_fraction=0.5)
        plan = plan_chunks(records, "prot", SMALL_PROT, name="prot_locus")
        assert plan is not None
        self.assertEqual(plan.alphabet, "prot")
        self.check_plan(plan, [s for _, s in records], 6)

    def test_default_config_on_a_long_locus(self) -> None:
        records = simulated(12, 8000, seed=11, divergence=0.1)
        plan = plan_chunks(records, "nucl")  # ChunkConfig() defaults
        assert plan is not None
        self.assertGreaterEqual(plan.n_chunks, 3)
        self.check_plan(plan, [s for _, s in records], 15)
        lengths = [plan.chunk_piece(0, c) for c in range(plan.n_chunks)]
        self.assertTrue(
            all(1500 < len(p) < 3600 for p in lengths), [len(p) for p in lengths]
        )

    def test_chunk_specs_describe_the_pieces(self) -> None:
        records = simulated()
        plan = plan_chunks(records, "nucl", SMALL)
        assert plan is not None
        for spec in plan.chunks:
            pieces = [len(plan.chunk_piece(r, spec.index)) for r in range(plan.n_seq)]
            non_empty = [p for p in pieces if p]
            self.assertEqual(spec.n_seq, len(non_empty))
            self.assertEqual(spec.n_empty, plan.n_seq - len(non_empty))
            self.assertEqual(spec.max_len, max(non_empty))
            self.assertEqual(spec.min_len, min(non_empty))
            self.assertEqual(spec.needs_alignment, len(non_empty) >= 2)

    def test_reference_is_the_median_length_sequence_with_id_tiebreak(self) -> None:
        plan = plan_chunks(simulated(5, 1500, seed=2, divergence=0.02), "nucl", SMALL)
        assert plan is not None
        lengths = [len(s) for s in plan.sequences]
        order = sorted(range(5), key=lambda i: (lengths[i], plan.ids[i], i))
        self.assertEqual(plan.reference_index, order[2])
        # identical lengths: the smallest id (then index) among the median group
        base = simulated(1, 1500, seed=2, divergence=0.02)[0][1]
        tied = plan_chunks(
            [("z", base), ("b", base), ("a", base), ("c", base)], "nucl", SMALL
        )
        assert tied is not None
        self.assertEqual(tied.ids[tied.reference_index], "b")  # sorted a,b,c,z; index 1

    def test_repeated_kmers_are_not_anchors(self) -> None:
        rng = random.Random(5)
        base = "".join(rng.choice("ACGT") for _ in range(1800))
        records = [(f"s{i}", base) for i in range(4)]
        first = plan_chunks(records, "nucl", SMALL)
        assert first is not None
        # duplicate the region around the first anchor into one sequence
        pos = first.anchors[0].positions[3]
        copy = base[pos - 60 : pos + 75]
        mutated = base + copy
        records[3] = ("s3", mutated)
        second = plan_chunks(records, "nucl", SMALL)
        assert second is not None
        self.assertNotEqual(first.anchors[0].kmer, second.anchors[0].kmer)
        self.check_plan(second, [s for _, s in records], 15)

    def test_swapped_blocks_keep_the_chain_collinear(self) -> None:
        records = simulated(8, 1800, seed=21, divergence=0.05, conserved_fraction=0.7)
        name, seq = records[4]
        records[4] = (name, seq[900:1500] + seq[600:900] + seq[:600] + seq[1500:])
        plan = plan_chunks(records, "nucl", SMALL)
        if plan is not None:  # fewer anchors are fine, a non-collinear chain is not
            self.check_plan(plan, [s for _, s in records], 15)

    def test_low_complexity_kmers(self) -> None:
        cfg = ChunkConfig()
        for kmer in (
            "AAAAAAAAAAAAAAA",
            "ACACACACACACACA",
            "ACGACGACGACGACG",
            "AAAAAGCTTAGCTTA",
            "ACACACACACACACG",
        ):
            self.assertTrue(_is_low_complexity(kmer, cfg), kmer)
        for kmer in ("ACGTTGCAAGCTTAG", "GATTACAGATTACAG"[:15], "CCATTGACGGTAACT"):
            self.assertFalse(_is_low_complexity(kmer, cfg), kmer)

    def test_lowercase_input_is_matched_and_kept(self) -> None:
        records = [(n, s.lower()) for n, s in simulated()]
        plan = plan_chunks(records, "nucl", SMALL)
        assert plan is not None
        self.assertEqual(plan.sequences[0], records[0][1])  # original case kept
        self.check_plan(plan, [s for _, s in records], 15)


class NoPlanTests(unittest.TestCase):
    def test_short_locus(self) -> None:
        records = simulated(10, 900)
        plan, reason = plan_chunks_with_reason(records, "nucl", SMALL)
        self.assertIsNone(plan)
        self.assertIn("min_locus_len", reason)
        self.assertIsNone(plan_chunks(simulated(6, 4000), "nucl"))  # < 5 kb default

    def test_too_few_sequences(self) -> None:
        plan, reason = plan_chunks_with_reason(simulated(1, 2000), "nucl", SMALL)
        self.assertIsNone(plan)
        self.assertIn("few sequences", reason)
        self.assertIsNone(plan_chunks([], "nucl", SMALL))

    def test_adjust_direction_disables_chunking(self) -> None:
        records = simulated()
        cfg = ChunkConfig(
            min_locus_len=1000, target_chunk_len=600, adjust_direction=True
        )
        plan, reason = plan_chunks_with_reason(records, "nucl", cfg)
        self.assertIsNone(plan)
        self.assertIn("adjust_direction", reason)
        self.assertIsNotNone(plan_chunks(records, "nucl", SMALL))

    def test_reverse_complemented_member_breaks_the_chain(self) -> None:
        records = simulated()
        name, seq = records[3]
        records[3] = (name, reverse_complement(seq))
        plan, reason = plan_chunks_with_reason(records, "nucl", SMALL)
        self.assertIsNone(plan)
        self.assertIn("no collinear anchor chain", reason)

    def test_unrelated_sequences_have_no_anchor(self) -> None:
        rng = random.Random(0)
        records = [
            (f"r{i}", "".join(rng.choice("ACGT") for _ in range(2500)))
            for i in range(4)
        ]
        self.assertIsNone(plan_chunks(records, "nucl", SMALL))

    def test_ambiguity_codes_cannot_be_anchored(self) -> None:
        records = [(f"n{i}", "N" * 2000) for i in range(3)]
        self.assertIsNone(plan_chunks(records, "nucl", SMALL))


@unittest.skipUnless(HAS_MAFFT, "mafft not available")
class SpliceWithMafftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # a fixed fast algorithm keeps the tests quick; `--auto` is exercised too
        self.mafft = MAFFT(
            {"alignment_tool_config": {"algorithm": "retree", "retree": 2}}
        )
        self.addCleanup(self.mafft.scratch_mgr.cleanup)

    def align_chunks(
        self, plan: ChunkPlan, alphabet: str = "nucl"
    ) -> list[Path | None]:
        paths = plan.write_chunk_fastas(self.root / "chunks")
        aligned: list[Path | None] = []
        for spec, path in zip(plan.chunks, paths):
            if not spec.needs_alignment:
                aligned.append(None)
                continue
            out = path.with_suffix(".aligned.fasta")
            ok = self.mafft.align(
                str(path),
                str(out),
                threads=1,
                sequence_count=spec.n_seq,
                max_len=spec.max_len,
                min_len=spec.min_len,
                alphabet=alphabet,
            )
            self.assertTrue(ok)
            aligned.append(out)
        return aligned

    def assert_invariants(
        self, plan: ChunkPlan, records: list[tuple[str, str]], out: Path
    ) -> list[tuple[str, str]]:
        rows = read_fasta(out)
        self.assertEqual([h for h, _ in rows], [h for h, _ in records])  # ids/order
        self.assertEqual(len({len(s) for _, s in rows}), 1)  # equal length
        for (header, row), (_, original) in zip(rows, records):
            self.assertEqual(row.replace("-", ""), original, header)  # exact content
        return rows

    def test_spliced_alignment_satisfies_the_invariants(self) -> None:
        records = simulated(10, 1800)
        plan = plan_chunks(records, "nucl", SMALL, name="nucl_locus")
        assert plan is not None
        aligned = self.align_chunks(plan)
        out = self.root / "spliced.aligned.fasta"
        result = splice_alignments(plan, aligned, out)
        rows = self.assert_invariants(plan, records, out)
        self.assertEqual(result.n_seq, 10)
        self.assertEqual(result.width, len(rows[0][1]))
        self.assertEqual(sum(result.chunk_widths), result.width)
        # every chunk contributes its own alignment columns
        self.assertTrue(
            all(w >= s.max_len for w, s in zip(result.chunk_widths, plan.chunks))
        )

    def test_default_auto_algorithm_also_works(self) -> None:
        records = simulated(8, 1500, seed=2)
        plan = plan_chunks(records, "nucl", SMALL, name="nucl_auto")
        assert plan is not None
        self.mafft = MAFFT()
        self.addCleanup(self.mafft.scratch_mgr.cleanup)
        aligned = self.align_chunks(plan)
        out = self.root / "auto.fasta"
        splice_alignments(plan, aligned, out)
        self.assert_invariants(plan, records, out)

    def test_headers_and_wrapping_follow_the_input_fasta(self) -> None:
        records = [
            (f"{n} locus=ycf1 taxon={n}", s) for n, s in simulated(6, 1600, seed=4)
        ]
        src = self.root / "in.fasta"
        write_fasta(src, records)
        plan = plan_chunks(src, "nucl", SMALL, name="nucl_hdr")
        assert plan is not None
        self.assertEqual(plan.headers, tuple(h for h, _ in records))
        self.assertEqual(plan.ids, tuple(h.split()[0] for h, _ in records))
        out = self.root / "out.fasta"
        splice_alignments(plan, self.align_chunks(plan), out, line_width=60)
        self.assert_invariants(plan, records, out)
        longest = max(
            len(line)
            for line in out.read_text().splitlines()
            if not line.startswith(">")
        )
        self.assertLessEqual(longest, 60)
        flat = self.root / "flat.fasta"
        splice_alignments(plan, self.align_chunks(plan), flat, line_width=0)
        self.assert_invariants(plan, records, flat)

    def test_lowercase_input_is_reproduced_exactly(self) -> None:
        records = [(n, s.lower()) for n, s in simulated(8, 1600, seed=6)]
        plan = plan_chunks(records, "nucl", SMALL)
        assert plan is not None
        out = self.root / "lower.fasta"
        splice_alignments(plan, self.align_chunks(plan), out)
        self.assert_invariants(plan, records, out)

    def test_mixed_case_input_is_reproduced_exactly(self) -> None:
        records = [
            (n, s[:300].lower() + s[300:]) for n, s in simulated(8, 1600, seed=6)
        ]
        plan = plan_chunks(records, "nucl", SMALL)
        assert plan is not None
        out = self.root / "mixed.fasta"
        splice_alignments(plan, self.align_chunks(plan), out)
        self.assert_invariants(plan, records, out)

    def test_protein_loci(self) -> None:
        records = simulated(8, 700, seed=3, alphabet="prot", conserved_fraction=0.5)
        plan = plan_chunks(records, "prot", SMALL_PROT, name="prot_locus")
        assert plan is not None
        out = self.root / "prot.aligned.fasta"
        splice_alignments(plan, self.align_chunks(plan, "prot"), out)
        self.assert_invariants(plan, records, out)

    def test_empty_pieces_become_gap_rows_and_stay_out_of_the_chunk(self) -> None:
        records = simulated(8, 1800, seed=7)
        first = plan_chunks(records, "nucl", SMALL)
        assert first is not None
        # the shortest sequence starts exactly at the first anchor (it stays the
        # shortest, so the reference and therefore the anchor do not change)
        row = min(range(8), key=lambda i: (len(records[i][1]), i))
        cut = first.anchors[0].positions[row]
        records[row] = (records[row][0], records[row][1][cut:])
        plan = plan_chunks(records, "nucl", SMALL)
        assert plan is not None
        self.assertEqual(plan.chunk_piece(row, 0), "")
        self.assertEqual(plan.chunks[0].n_empty, 1)
        chunk_fasta = plan.write_chunk_fastas(self.root / "peek")[0]
        self.assertNotIn(
            f">s{row}\n", chunk_fasta.read_text()
        )  # not given to the aligner
        out = self.root / "empty.fasta"
        result = splice_alignments(plan, self.align_chunks(plan), out)
        rows = self.assert_invariants(plan, records, out)
        width0 = result.chunk_widths[0]
        self.assertEqual(set(rows[row][1][:width0]), {"-"})
        self.assertNotEqual(set(rows[row][1][width0:]), {"-"})

    def test_chunk_with_a_single_sequence_needs_no_aligner(self) -> None:
        # reference A (median length) keeps all its residues; the four others
        # start at the first anchor: two shorter ones, two longer (3' junk) ones
        raw = simulated(5, 1800, seed=5, divergence=0.03, conserved_fraction=0.7)
        junk = "".join(random.Random(1).choice("ACGT") for _ in range(800))
        records = [
            raw[0],
            (raw[1][0], raw[1][1][:1100]),
            (raw[2][0], raw[2][1][:1100]),
            (raw[3][0], raw[3][1] + junk),
            (raw[4][0], raw[4][1] + junk),
        ]
        first = plan_chunks(records, "nucl", SMALL)
        assert first is not None and first.reference_index == 0
        modified = [records[0]]
        for row in range(1, 5):
            cut = first.anchors[0].positions[row]
            modified.append((records[row][0], records[row][1][cut:]))
        plan = plan_chunks(modified, "nucl", SMALL)
        assert plan is not None
        self.assertEqual(plan.reference_index, 0)
        self.assertEqual(plan.anchors[0].kmer, first.anchors[0].kmer)
        self.assertEqual((plan.chunks[0].n_seq, plan.chunks[0].n_empty), (1, 4))
        self.assertFalse(plan.chunks[0].needs_alignment)
        aligned = self.align_chunks(plan)
        self.assertIsNone(aligned[0])
        out = self.root / "single.fasta"
        result = splice_alignments(plan, aligned, out)
        rows = self.assert_invariants(plan, modified, out)
        width0 = result.chunk_widths[0]
        self.assertEqual(width0, len(plan.chunk_piece(0, 0)))
        for row in range(1, 5):
            self.assertEqual(set(rows[row][1][:width0]), {"-"})
        self.assertEqual(rows[0][1][:width0], modified[0][1][:width0])

    def test_inconsistent_aligned_chunks_are_rejected(self) -> None:
        records = simulated(8, 1500, seed=8)
        plan = plan_chunks(records, "nucl", SMALL)
        assert plan is not None
        aligned = self.align_chunks(plan)
        out = self.root / "bad.fasta"
        first = aligned[0]
        assert first is not None
        good = first.read_text()

        # missing record
        first.write_text(good.split(">s1\n")[0])
        with self.assertRaises(ChunkSpliceError):
            splice_alignments(plan, aligned, out)
        # changed residues
        first.write_text(good.replace("a", "g", 1))
        with self.assertRaises(ChunkSpliceError):
            splice_alignments(plan, aligned, out)
        # rows of unequal length
        lines = good.splitlines()
        first.write_text("\n".join(lines[:-1] + [lines[-1] + "-"]) + "\n")
        with self.assertRaises(ChunkSpliceError):
            splice_alignments(plan, aligned, out)
        # wrong number of chunk paths / missing path
        first.write_text(good)
        with self.assertRaises(ChunkSpliceError):
            splice_alignments(plan, aligned[:-1], out)
        with self.assertRaises(ChunkSpliceError):
            splice_alignments(plan, [None, *aligned[1:]], out)
        self.assertFalse(out.exists())  # nothing written on failure
        splice_alignments(plan, aligned, out)  # the intact files splice fine
        self.assert_invariants(plan, records, out)


class DeterminismTests(unittest.TestCase):
    def fingerprint(self, records: Any) -> str:
        plan = plan_chunks(records, "nucl", SMALL, name="locus")
        assert plan is not None
        return plan.fingerprint

    def test_repeated_plans_are_identical(self) -> None:
        records = simulated()
        plans = [plan_chunks(records, "nucl", SMALL, name="x") for _ in range(3)]
        assert all(p is not None for p in plans)
        first = plans[0]
        assert first is not None
        for other in plans[1:]:
            assert other is not None
            self.assertEqual(first, other)
            self.assertEqual(first.to_dict(), other.to_dict())

    def test_container_type_does_not_matter(self) -> None:
        records = simulated()
        tuples = self.fingerprint(records)
        seqrecords = self.fingerprint(
            [SeqRecord(Seq(s), id=n, description=n) for n, s in records]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.fasta"
            write_fasta(path, records, width=70)
            from_file = self.fingerprint(str(path))
            from_pathlike = self.fingerprint(path)
        self.assertEqual({tuples, seqrecords, from_file, from_pathlike}, {tuples})

    def test_plan_is_independent_of_the_host(self) -> None:
        records = simulated()
        reference = plan_chunks(records, "nucl", SMALL)
        with (
            patch("os.cpu_count", return_value=1),
            patch("os.sched_getaffinity", return_value={0}, create=True),
        ):
            again = plan_chunks(records, "nucl", SMALL)
        self.assertEqual(reference, again)

    def test_plan_is_identical_across_hash_seeds(self) -> None:
        script = (
            "import json, sys\n"
            "from plastburstalign._simulate import simulate_locus\n"
            "from plastburstalign.chunking import ChunkConfig, plan_chunks\n"
            "r = simulate_locus(10, 1800, seed=7, divergence=0.2, conserved_fraction=0.4)\n"
            "p = plan_chunks(r, 'nucl', ChunkConfig(min_locus_len=1000, target_chunk_len=600), name='locus')\n"
            "print(json.dumps(p.to_dict(), sort_keys=True))\n"
        )
        expected_plan = plan_chunks(simulated(), "nucl", SMALL, name="locus")
        assert expected_plan is not None
        expected = json.dumps(expected_plan.to_dict(), sort_keys=True)
        env_base = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
        for seed in ("0", "1", "4242"):
            with self.subTest(hashseed=seed):
                done = subprocess.run(
                    [sys.executable, "-c", script],
                    cwd=REPO_ROOT,
                    env={**env_base, "PYTHONHASHSEED": seed},
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(done.stdout.strip().splitlines()[-1], expected)

    def test_manifest_dict_is_json_serializable(self) -> None:
        plan = plan_chunks(simulated(), "nucl", SMALL, name="nucl_ycf1")
        assert plan is not None
        data = json.loads(json.dumps(plan.to_dict()))
        self.assertEqual(data["n_chunks"], plan.n_chunks)
        self.assertEqual(len(data["anchors"]), plan.n_chunks - 1)
        self.assertEqual(len(data["chunks"]), plan.n_chunks)
        self.assertEqual(data["fingerprint"], plan.fingerprint)
        self.assertEqual(data["name"], "nucl_ycf1")
        self.assertEqual(data["config"]["k_nucl"], 15)

    def test_chunk_files_are_written_deterministically(self) -> None:
        records = simulated()
        plan = plan_chunks(records, "nucl", SMALL, name="nucl_x")
        assert plan is not None
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            paths_a = plan.write_chunk_fastas(a)
            paths_b = plan.write_chunk_fastas(b)
            self.assertEqual([p.name for p in paths_a], [p.name for p in paths_b])
            self.assertEqual(paths_a[0].name, "nucl_x.chunk00.unalign.fasta")
            for pa, pb in zip(paths_a, paths_b):
                self.assertEqual(pa.read_bytes(), pb.read_bytes())


@unittest.skipUnless(HAS_MAFFT, "mafft not available")
@unittest.skipIf(os.environ.get("PBA_SKIP_TIMING"), "timing report disabled")
class ChunkingTimingReport(unittest.TestCase):
    """Reports (does not assert) chunked vs unchunked timing of an ~8 kb locus."""

    def test_report_chunked_versus_unchunked_alignment_of_an_8kb_locus(self) -> None:
        records = simulate_locus(
            8, 8000, divergence=0.25, conserved_fraction=0.3, seed=4
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mafft = MAFFT(
                {"alignment_tool_config": {"algorithm": "retree", "retree": 2}}
            )
            self.addCleanup(mafft.scratch_mgr.cleanup)
            src = root / "nucl_long.unalign.fasta"
            write_fasta(src, records)
            started = time.perf_counter()
            self.assertTrue(mafft.align(str(src), str(root / "full.aln"), threads=1))
            unchunked = time.perf_counter() - started

            plan = plan_chunks(records, "nucl", name="nucl_long")
            assert plan is not None
            paths = plan.write_chunk_fastas(root / "chunks")
            aligned: list[Path | None] = []
            durations: list[float] = []
            for spec, path in zip(plan.chunks, paths):
                out = path.with_suffix(".aligned.fasta")
                started = time.perf_counter()
                self.assertTrue(
                    mafft.align(
                        str(path),
                        str(out),
                        threads=1,
                        sequence_count=spec.n_seq,
                        max_len=spec.max_len,
                        min_len=spec.min_len,
                    )
                )
                durations.append(time.perf_counter() - started)
                aligned.append(out)
            started = time.perf_counter()
            splice_alignments(plan, aligned, root / "spliced.aln")
            splice = time.perf_counter() - started

            rows = read_fasta(root / "spliced.aln")
            for (_, row), (_, original) in zip(rows, records):
                self.assertEqual(row.replace("-", ""), original)
            sys.stderr.write(
                "\n[chunking timing] N=8, L~8 kb, MAFFT --retree 2 (1 thread): "
                f"unchunked {unchunked:.2f}s; {plan.n_chunks} chunks "
                f"{', '.join(f'{d:.2f}' for d in durations)}s = {sum(durations):.2f}s serial, "
                f"critical path {max(durations):.2f}s with {plan.n_chunks} cores; "
                f"splice {splice:.3f}s\n"
            )


if __name__ == "__main__":
    unittest.main()
