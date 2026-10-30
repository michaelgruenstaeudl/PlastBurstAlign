"""Regression tests for the alignment coordination fixes of milestone M0.

Covers: cheap emptiness checks (no `SeqRecord` hydration), MAFFT
`max_len`/`min_len` plumbing, the Slurm worker functions (`supports_max_len`
NameError, `BackTranslation` usage, export-from-store when the unaligned FASTA
is missing, output location).
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from Bio import SeqIO
from Bio.Seq import Seq

from plastburstalign.alignment_ops import AlignmentCoordination, BackTranslation
from plastburstalign.alignment_tools import MAFFT
from plastburstalign.seqfeature_ops import PlastidData
from plastburstalign.storage import DiskBackedFeatureStore, FeatureStoreView
from plastburstalign.user_parameters import UserParameters


class RecordingAligner:
    """Copies input to output (a stand-in for an aligner) and records calls."""

    exec_path = "recording-aligner"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        max_len: int | None = None,
        min_len: int | None = None,
    ) -> bool:
        with open(input_file, encoding="utf-8") as handle:
            content = handle.read()
        with self._lock:
            self.calls.append(
                {
                    "input": input_file,
                    "output": output_file,
                    "input_existed": True,
                    "content": content,
                    "threads": threads,
                    "sequence_count": sequence_count,
                    "max_len": max_len,
                    "min_len": min_len,
                }
            )
        shutil.copyfile(input_file, output_file)
        return True


class LegacySignatureAligner:
    """An aligner that predates `max_len`/`min_len` (and has no **kwargs)."""

    exec_path = "legacy-aligner"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
    ) -> bool:
        self.calls.append((os.path.basename(input_file), sequence_count))
        shutil.copyfile(input_file, output_file)
        return True


class AlignmentFixtureMixin:
    tmp: tempfile.TemporaryDirectory
    root: Path

    def make_data(
        self, mode: str = "rna", threads: int = 2, **kwargs: Any
    ) -> tuple[PlastidData, UserParameters]:
        in_dir = self.root / "in"
        in_dir.mkdir(exist_ok=True)
        params = UserParameters(
            in_dir=str(in_dir),
            out_dir=str(self.root / "out"),
            select_mode=mode,
            num_threads=threads,
            min_seq_length=1,
            **kwargs,
        )
        data = PlastidData(params)
        return data, params

    @staticmethod
    def add_nucleotide_loci(data: PlastidData) -> None:
        """Three equal-length loci (A, B, C), three taxa each."""
        for region, base in (("locA", "ACGT"), ("locB", "TTGA"), ("locC", "CCAG")):
            for i, taxon in enumerate(("t1", "t2", "t3")):
                seq = (base * 4)[: 12 + i]  # lengths 12, 13, 14
                data.store.add_record(
                    region, taxon, "nucl", f"{region}_{taxon}", "", seq, file_index=i
                )

    @staticmethod
    def add_coding_loci(data: PlastidData) -> None:
        """Two CDS loci with consistent nucleotide and protein records."""
        genes = {
            "psbA": ["ATGAAATTTGGGCCCTAA", "ATGAAGTTTGGGCCATAA", "ATGAAATTCGGGCCCTAG"],
            "rbcL": [
                "ATGTCACCACAAACAGAGTAA",
                "ATGTCACCGCAAACAGAGTAA",
                "ATGTCACCACAGACAGAGTGA",
            ],
        }
        for region, seqs in genes.items():
            for i, nuc in enumerate(seqs):
                taxon = f"t{i + 1}"
                prot = str(Seq(nuc).translate(table=11))
                data.store.add_record(
                    region, taxon, "nucl", f"{region}_{taxon}", "", nuc, file_index=i
                )
                data.store.add_record(
                    region, taxon, "prot", f"{region}_{taxon}", "", prot, file_index=i
                )

    def coordination(self, data: PlastidData, params: UserParameters, aligner: Any):
        with patch(
            "plastburstalign.alignment_ops.get_alignment_tool", return_value=aligner
        ):
            return AlignmentCoordination(data, params)


class NoHydrationTests(AlignmentFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _guards(self, data: PlastidData):
        boom = AssertionError("SeqRecords were hydrated")
        return (
            patch.object(DiskBackedFeatureStore, "get_records", side_effect=boom),
            patch.object(FeatureStoreView, "items", side_effect=boom),
            patch.object(FeatureStoreView, "values", side_effect=boom),
            patch.object(FeatureStoreView, "__getitem__", side_effect=boom),
        )

    def test_nucleotide_msa_does_not_hydrate_records(self) -> None:
        data, params = self.make_data("rna")
        self.addCleanup(data.store.close)
        self.add_nucleotide_loci(data)
        aligner = RecordingAligner()
        coord = self.coordination(data, params, aligner)
        g1, g2, g3, g4 = self._guards(data)
        with g1 as get_records, g2, g3, g4:
            coord.perform_MSA()
        get_records.assert_not_called()
        self.assertEqual(len(aligner.calls), 3)
        self.assertEqual({n for n, _ in coord.success_list}, {"locA", "locB", "locC"})

    def test_protein_msa_does_not_hydrate_records(self) -> None:
        data, params = self.make_data("cds")
        self.addCleanup(data.store.close)
        self.add_coding_loci(data)
        aligner = RecordingAligner()
        coord = self.coordination(data, params, aligner)
        g1, g2, g3, g4 = self._guards(data)
        with g1 as get_records, g2, g3, g4:
            coord.perform_MSA()
        get_records.assert_not_called()
        self.assertEqual({n for n, _ in coord.success_list}, {"psbA", "rbcL"})

    def test_empty_store_still_raises(self) -> None:
        data, params = self.make_data("rna")
        self.addCleanup(data.store.close)
        coord = self.coordination(data, params, RecordingAligner())
        with self.assertRaises(RuntimeError):
            coord._nuc_MSA()
        with self.assertRaises(RuntimeError):
            coord._prot_MSA()


class MafftLengthPlumbingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fasta = os.path.join(self.tmp.name, "locus.fasta")
        self.mafft = MAFFT()
        self.mafft.exec_path = "mafft"

    def _write(self, seqs: list[str]) -> None:
        with open(self.fasta, "w") as handle:
            for i, seq in enumerate(seqs):
                handle.write(f">s{i}\n{seq}\n")

    def test_supplied_min_len_is_used_instead_of_hardcoded_six(self) -> None:
        # N=5000 qualifies for PartTree; min_len < 6 must bypass it ...
        cmd = self.mafft._build_command(
            "missing.fasta", sequence_count=5000, max_len=100, min_len=3
        )
        self.assertIn("--auto", cmd)
        self.assertNotIn("--parttree", cmd)
        # ... while a long enough minimum keeps PartTree enabled.
        cmd = self.mafft._build_command(
            "missing.fasta", sequence_count=5000, max_len=100, min_len=40
        )
        self.assertIn("--parttree", cmd)
        self.assertNotIn("--auto", cmd)

    def test_min_len_is_scanned_when_counts_are_supplied_without_it(self) -> None:
        self._write(["ACGTACGTAC"] * 20 + ["ACG"])  # one 3-residue sequence
        cmd = self.mafft._build_command(self.fasta, sequence_count=5000, max_len=10)
        self.assertNotIn("--parttree", cmd)
        self.assertIn("--auto", cmd)

    def test_scan_is_skipped_when_everything_is_supplied(self) -> None:
        with patch.object(MAFFT, "_scan_fasta_meta", side_effect=AssertionError):
            self._write(["ACGTACGTAC"] * 3)
            cmd = self.mafft._build_command(
                self.fasta, sequence_count=5000, max_len=10, min_len=10
            )
        self.assertIn("--parttree", cmd)

    def test_unknown_min_len_and_missing_file_keeps_parttree(self) -> None:
        cmd = self.mafft._build_command(
            "missing.fasta", sequence_count=13724, max_len=1500
        )
        self.assertIn("--parttree", cmd)

    def test_align_forwards_max_len_and_min_len(self) -> None:
        self._write(["ACGTACGTAC"] * 3)
        out = os.path.join(self.tmp.name, "out.fasta")
        captured: dict[str, Any] = {}

        def fake_build(input_file: str, **kwargs: Any) -> list[str]:
            captured.update(kwargs)
            return ["true"]

        with (
            patch.object(self.mafft, "_build_command", side_effect=fake_build),
            patch("plastburstalign.alignment_tools.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            self.assertTrue(
                self.mafft.align(
                    self.fasta,
                    out,
                    threads=2,
                    sequence_count=3,
                    max_len=10,
                    min_len=7,
                )
            )
        self.assertEqual(captured["sequence_count"], 3)
        self.assertEqual(captured["max_len"], 10)
        self.assertEqual(captured["min_len"], 7)
        self.assertEqual(captured["threads"], 2)


class MinLenCoordinationTests(AlignmentFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_summary_lengths_are_plumbed_to_aligners_that_accept_them(self) -> None:
        data, params = self.make_data("rna")
        self.addCleanup(data.store.close)
        self.add_nucleotide_loci(data)
        aligner = RecordingAligner()
        coord = self.coordination(data, params, aligner)
        coord.perform_MSA()
        summary = data.store.get_region_summary("nucl")
        self.assertEqual(len(aligner.calls), 3)
        for call in aligner.calls:
            region = os.path.basename(call["input"]).split("_", 1)[1].split(".")[0]
            self.assertEqual(call["sequence_count"], summary[region]["count"])
            self.assertEqual(call["max_len"], summary[region]["max_len"])
            self.assertEqual(call["min_len"], summary[region]["min_len"])
            self.assertEqual(call["min_len"], 12)

    def test_aligners_without_min_len_are_not_given_it(self) -> None:
        data, params = self.make_data("rna")
        self.addCleanup(data.store.close)
        self.add_nucleotide_loci(data)
        aligner = LegacySignatureAligner()
        coord = self.coordination(data, params, aligner)
        coord.perform_MSA()  # would raise TypeError inside the workers if forced
        self.assertEqual(len(aligner.calls), 3)
        self.assertTrue(all(count == 3 for _, count in aligner.calls))
        self.assertEqual(len(coord.success_list), 3)


class WorkerSubsetTests(AlignmentFixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_nuc_subset_aligns_only_targets_from_store_without_shared_files(
        self,
    ) -> None:
        data, params = self.make_data("rna")
        self.addCleanup(data.store.close)
        self.add_nucleotide_loci(data)
        aligner = RecordingAligner()
        coord = self.coordination(data, params, aligner)
        out = Path(params["out_dir"])
        self.assertEqual(list(out.glob("*.unalign.fasta")), [])

        # Used to raise NameError (`supports_max_len`), swallowed by future.result().
        coord._nuc_MSA_subset({"locA", "locC"})

        aligned = {Path(c["input"]).name for c in aligner.calls}
        self.assertEqual(
            aligned, {"nucl_locA.unalign.fasta", "nucl_locC.unalign.fasta"}
        )
        for call in aligner.calls:
            self.assertEqual(call["max_len"], 14)
            self.assertEqual(call["min_len"], 12)
            self.assertEqual(call["sequence_count"], 3)
            # The unaligned FASTA was exported from the store, in file order ...
            ids = [r.id for r in SeqIO.parse(io.StringIO(call["content"]), "fasta")]
            self.assertEqual([i.rsplit("_", 1)[-1] for i in ids], ["t1", "t2", "t3"])
            # ... into scratch, not into the (shared) output directory.
            self.assertFalse(call["input"].startswith(str(out) + os.sep))
            self.assertFalse(os.path.exists(call["input"]))  # and cleaned up
        # Outputs land in out_dir/fasta like in the monolithic path.
        self.assertEqual(
            sorted(p.name for p in (out / "fasta").glob("*.aligned.fasta")),
            ["nucl_locA.aligned.fasta", "nucl_locC.aligned.fasta"],
        )
        self.assertEqual(list(out.glob("*.unalign.fasta")), [])

    def test_existing_unaligned_file_is_used_without_exporting(self) -> None:
        data, params = self.make_data("rna")
        self.addCleanup(data.store.close)
        self.add_nucleotide_loci(data)
        out = Path(params["out_dir"])
        shared = out / "nucl_locB.unalign.fasta"
        shared.write_text(">locB_t1\nAAAAAAAAAAAA\n>locB_t2\nCCCCCCCCCCCC\n")
        aligner = RecordingAligner()
        coord = self.coordination(data, params, aligner)
        with patch.object(
            DiskBackedFeatureStore, "export_fasta", side_effect=AssertionError
        ):
            coord._nuc_MSA_subset({"locB"})
        self.assertEqual(aligner.calls[0]["input"], str(shared))
        self.assertTrue(shared.exists())  # shared inputs are never deleted

    def test_align_partition_regions_returns_regions_and_writes_nexus(self) -> None:
        data, params = self.make_data("rna")
        self.addCleanup(data.store.close)
        self.add_nucleotide_loci(data)
        coord = self.coordination(data, params, RecordingAligner())
        done = coord.align_partition_regions(["locC", "locA"])
        self.assertEqual(done, ["locC", "locA"])
        out = Path(params["out_dir"])
        for region in ("locA", "locC"):
            self.assertTrue((out / "nexus" / f"nucl_{region}.aligned.nexus").exists())
        self.assertFalse((out / "fasta" / "nucl_locB.aligned.fasta").exists())

    def test_prot_subset_uses_correct_backtranslation_signature(self) -> None:
        data, params = self.make_data("cds")
        self.addCleanup(data.store.close)
        self.add_coding_loci(data)
        aligner = RecordingAligner()
        coord = self.coordination(data, params, aligner)
        out = Path(params["out_dir"])
        seen: dict[str, Any] = {}

        def fake_translation(*args: Any, **kwargs: Any) -> MagicMock:
            seen["args"] = args
            seen["kwargs"] = kwargs
            # the unaligned nucleotide FASTA must be readable when it is needed
            seen["nuc_exists"] = os.path.exists(args[2])
            instance = MagicMock()
            instance.backtranslate.side_effect = lambda: Path(args[3]).write_text(
                ">x\nATG\n"
            )
            seen["instance"] = instance
            return instance

        with patch(
            "plastburstalign.alignment_ops.BackTranslation",
            side_effect=fake_translation,
        ):
            coord._prot_MSA_subset({"psbA"})

        args = seen["args"]
        # (prot_name, prot_align_file, nuc_fasta_file, nuc_align_file, table)
        self.assertEqual(len(args), 5)
        self.assertEqual(seen["kwargs"], {})
        self.assertEqual(args[0], "psbA")
        self.assertEqual(args[1], str(out / "prot_psbA.aligned.fasta"))
        self.assertTrue(seen["nuc_exists"])
        self.assertEqual(args[3], str(out / "fasta" / "nucl_psbA.aligned.fasta"))
        self.assertEqual(args[4], 11)
        seen["instance"].backtranslate.assert_called_once_with()
        self.assertEqual(len(aligner.calls), 1)  # only the targeted protein locus
        self.assertEqual(
            Path(aligner.calls[0]["input"]).name, "prot_psbA.unalign.fasta"
        )
        self.assertEqual(aligner.calls[0]["max_len"], 6)  # 5 aa + stop

    def test_prot_subset_real_backtranslation_roundtrip(self) -> None:
        data, params = self.make_data("cds")
        self.addCleanup(data.store.close)
        self.add_coding_loci(data)
        coord = self.coordination(data, params, RecordingAligner())
        done = coord.align_partition_regions(["psbA", "rbcL"])
        self.assertEqual(sorted(done), ["psbA", "rbcL"])
        out = Path(params["out_dir"])
        for region in ("psbA", "rbcL"):
            records = list(
                SeqIO.parse(out / "fasta" / f"nucl_{region}.aligned.fasta", "fasta")
            )
            self.assertEqual(len(records), 3)
            expected = {
                r.id: str(r.seq) for r in data.store.get_records(region, "nucl")
            }
            self.assertEqual({r.id: str(r.seq) for r in records}, expected)
        self.assertEqual(list(out.glob("*.unalign.fasta")), [])

    def test_finalize_falls_back_to_fasta_without_individual_nexus(self) -> None:
        data, params = self.make_data("rna", individual_nexus=False)
        self.addCleanup(data.store.close)
        for region, base in (("locA", "ACGT"), ("locB", "TTGA")):
            for i, taxon in enumerate(("t1", "t2", "t3")):
                data.store.add_record(
                    region,
                    taxon,
                    "nucl",
                    f"{region}_{taxon}",
                    "",
                    base * 3,
                    file_index=i,
                )
        worker = self.coordination(data, params, RecordingAligner())
        done = worker.align_partition_regions(["locA", "locB"])
        self.assertEqual(done, ["locA", "locB"])
        self.assertFalse((Path(params["out_dir"]) / "nexus").exists())

        finalizer = self.coordination(data, params, RecordingAligner())
        finalizer.collect_partition_nexus(done)
        self.assertEqual(
            [Path(path).suffix for _, path in finalizer.success_list],
            [".fasta", ".fasta"],
        )
        finalizer.concat_MSAs()
        concat = Path(params["out_dir"]) / "nucl_2concat.aligned.fasta"
        rows = list(SeqIO.parse(concat, "fasta"))
        self.assertEqual({r.id for r in rows}, {"concat_t1", "concat_t2", "concat_t3"})
        self.assertEqual({len(r.seq) for r in rows}, {24})

    def test_backtranslation_class_signature(self) -> None:
        """`BackTranslation(prot_name, prot_align, nuc_fasta, nuc_align, table)`."""
        translator = BackTranslation("g", "p.fa", "n.fa", "o.fa", 11)
        self.assertEqual(translator.prot_name, "g")
        self.assertEqual(translator.prot_align_file, "p.fa")
        self.assertEqual(translator.nuc_fasta_file, "n.fa")
        self.assertEqual(translator.nuc_align_file, "o.fa")
        self.assertEqual(translator.table, 11)


if __name__ == "__main__":
    unittest.main()
