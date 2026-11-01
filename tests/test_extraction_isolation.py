"""Per-feature, per-file, and per-worker isolation during extraction."""

from __future__ import annotations

import os
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from plastburstalign.extraction_ops import (
    ChunkResult,
    ExtractAndCollect,
    _unpack_worker_result,
    log,
)
from plastburstalign.extraction_report import ExtractionError, ExtractionReport
from plastburstalign.genbank_scanner import fast_read_genbank
from plastburstalign.seqfeature_ops import (
    GeneFeature,
    GeneNameCleaner,
    IntergenicFeature,
    IntronFeature,
    PlastidData,
    PlastidDict,
    PlastidFeature,
)
from plastburstalign.task_queue import get_mp_context
from plastburstalign.user_parameters import UserParameters
from tests.fuzz_genbank import missing_origin, write_flatfile
from tests.synth import make_plastome, make_template, write_genbank_dataset
from tests.test_synth import _regions_by_taxon, _taxon_of


def _extractor(source: str = "genome.gb") -> tuple[ExtractAndCollect, ExtractionReport]:
    report = ExtractionReport()
    ext = ExtractAndCollect.__new__(ExtractAndCollect)
    ext.exclude_cds = ["rps12"]
    ext.exclude_region = []
    ext.include_trna = True
    ext.include_rrna = True
    ext.min_seq_length = 3
    ext._report = report
    ext._source_file = source
    return ext, report


def _cds_names(record) -> list[tuple[str, str]]:
    """(raw qualifier, cleaned name) for CDS features that extraction keeps."""
    cleaner = GeneNameCleaner()
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for feature in record.features:
        if feature.type != "CDS":
            continue
        raw = (feature.qualifiers.get("gene") or [""])[0]
        clean = cleaner.get_safe_gene(feature)
        if not clean or clean == "rps12" or clean in seen:
            continue
        seen.add(clean)
        found.append((raw, clean))
    return found


def _run(in_dir: Path, out_dir: Path, **overrides):
    values = {
        "in_dir": str(in_dir),
        "out_dir": str(out_dir),
        "select_mode": "cds",
        "num_threads": 2,
        "min_seq_length": 3,
        "min_num_taxa": 1,
        "order": "alpha",
    }
    values.update(overrides)
    params = UserParameters(**values)
    data = PlastidData(params)
    try:
        ExtractAndCollect(data, params).extract()
    except ExtractionError:
        raise
    finally:
        report = data.extraction_report
        by_taxon = _regions_by_taxon(data)
        seqs = {
            region: tuple(
                sorted((_taxon_of(r.id), str(r.seq)) for r in data.nucleotides[region])
            )
            for region in data.nucleotides
        }
        data.store.close()
    return report, by_taxon, seqs


class ChunkResultTests(unittest.TestCase):
    def test_unpacks_as_three_tuple_and_pickles_the_report(self) -> None:
        report = ExtractionReport()
        report.add_file_failure("/data/bad.gb", "parse", "no ORIGIN", 3)
        result = ChunkResult(
            {"atpA": [("id", "", "ATG", 0)]}, None, {"a_b": "b_a"}, report
        )
        nuc, prot, inv = result
        self.assertEqual(len(result), 3)
        self.assertEqual(set(nuc), {"atpA"})
        self.assertIsNone(prot)
        self.assertEqual(inv["a_b"], "b_a")
        self.assertIs(result.report, report)

        restored = pickle.loads(pickle.dumps(result))
        nuc2, prot2, inv2 = restored
        self.assertEqual(nuc2, nuc)
        self.assertIsNone(prot2)
        self.assertEqual(inv2, inv)
        self.assertEqual(restored.report.failed_files, ["/data/bad.gb"])
        self.assertEqual(restored.report.failure_for("bad.gb").stage, "parse")

        plain_nuc, plain_prot, plain_inv, plain_report = _unpack_worker_result(
            ({"x": []}, None, {})
        )
        self.assertIsNone(plain_report)
        self.assertEqual(set(plain_nuc), {"x"})
        self.assertIsNone(plain_prot)
        self.assertEqual(plain_inv, {})

        two_nuc, two_prot, two_inv, two_report = _unpack_worker_result(
            ({"y": []}, None)
        )
        self.assertIsNone(two_report)
        self.assertEqual(set(two_nuc), {"y"})
        self.assertIsNone(two_prot)
        self.assertEqual(two_inv, {})


class FeatureIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.template = make_template(3, 12)
        cls.genome = Path(cls._tmp.name) / "one.gb"
        cls.genome.write_text(
            make_plastome(1, "SYN000", cls.template), encoding="ascii"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _record(self):
        return fast_read_genbank(self.genome)

    def test_bad_cds_does_not_drop_the_genome(self) -> None:
        record = self._record()
        named = _cds_names(record)
        self.assertGreaterEqual(len(named), 2)
        victim_raw, victim = next(pair for pair in named if pair[1] == "psbA")
        _, keeper = next(pair for pair in named if pair[1] != "psbA")
        baseline_ext, _ = _extractor()
        baseline = PlastidDict()
        baseline_ext._extract_cds(self._record(), baseline, PlastidDict())
        orig = GeneFeature.__init__

        def boom(self, rec, feature):
            raw = (feature.qualifiers.get("gene") or [""])[0]
            if raw == victim_raw:
                raise RuntimeError("corrupt cds")
            orig(self, rec, feature)

        ext, report = _extractor(self.genome.name)
        patched = PlastidDict()
        with patch.object(GeneFeature, "__init__", boom):
            ext._extract_cds(self._record(), patched, PlastidDict())
        self.assertIn(keeper, patched)
        self.assertNotIn(victim, patched)
        self.assertEqual(set(patched), set(baseline) - {victim})
        self.assertTrue(
            any(victim_raw in failure.feature for failure in report.feature_failures)
        )
        self.assertTrue(
            all(failure.stage == "feature:cds" for failure in report.feature_failures)
        )

    def test_bad_rna_does_not_drop_the_genome(self) -> None:
        record = self._record()
        rnas = [
            f
            for f in record.features
            if f.type in ("tRNA", "rRNA") and f.qualifiers.get("gene")
        ]
        self.assertGreaterEqual(len(rnas), 2)
        victim_raw = rnas[0].qualifiers["gene"][0]
        orig = PlastidFeature.__init__

        def boom(self, rec, feature):
            raw = (feature.qualifiers.get("gene") or [""])[0]
            if raw == victim_raw:
                raise RuntimeError("corrupt rna")
            orig(self, rec, feature)

        ext, report = _extractor()
        baseline = PlastidDict()
        ext._extract_rna(self._record(), baseline)
        patched_ext, report = _extractor()
        patched = PlastidDict()
        with patch.object(PlastidFeature, "__init__", boom):
            patched_ext._extract_rna(self._record(), patched)
        self.assertTrue(set(patched))
        self.assertLess(
            sum(len(v) for v in patched.values()),
            sum(len(v) for v in baseline.values()),
        )
        self.assertTrue(set(patched) <= set(baseline))
        self.assertTrue(any("corrupt rna" in f.reason for f in report.feature_failures))

    def test_bad_igs_anchor_does_not_drop_the_genome(self) -> None:
        record = self._record()
        genes = [
            (f.qualifiers.get("gene") or [""])[0]
            for f in record.features
            if f.type == "gene" and f.qualifiers.get("gene")
        ]
        victim_raw = next(g for g in genes if g and "rps12" not in g.casefold())
        orig = IntergenicFeature.__init__

        def boom(self, rec, current, subsequent):
            raw = (current.qualifiers.get("gene") or [""])[0]
            if raw == victim_raw:
                raise RuntimeError("corrupt igs")
            orig(self, rec, current, subsequent)

        base_ext, _ = _extractor()
        baseline = PlastidDict()
        base_ext._extract_igs(self._record(), baseline)
        ext, report = _extractor()
        patched = PlastidDict()
        with patch.object(IntergenicFeature, "__init__", boom):
            ext._extract_igs(self._record(), patched)
        self.assertTrue(set(baseline))
        self.assertTrue(set(patched))
        self.assertTrue(set(patched) < set(baseline))
        self.assertTrue(any(f.stage == "feature:igs" for f in report.feature_failures))

    def test_bad_intron_does_not_drop_the_genome(self) -> None:
        record = self._record()
        cleaner = GeneNameCleaner()
        victims = []
        for feature in record.features:
            parts = getattr(feature.location, "parts", ())
            if feature.type not in ("CDS", "tRNA") or len(parts) not in (2, 3):
                continue
            raw = (feature.qualifiers.get("gene") or [""])[0]
            clean = cleaner.get_safe_gene(feature)
            if clean and clean != "rps12" and clean not in victims:
                victims.append(clean)
                if len(victims) == 1:
                    victim_raw, victim = raw, clean
        self.assertGreaterEqual(len(victims), 2, victims)
        orig = IntronFeature.__init__

        def boom(self, rec, feature, offset=0):
            raw = (feature.qualifiers.get("gene") or [""])[0]
            if raw == victim_raw:
                raise RuntimeError("corrupt intron")
            orig(self, rec, feature, offset)

        base_ext, _ = _extractor()
        baseline = PlastidDict()
        base_ext._extract_int(self._record(), baseline)
        ext, report = _extractor()
        patched = PlastidDict()
        with patch.object(IntronFeature, "__init__", boom):
            ext._extract_int(self._record(), patched)
        self.assertTrue(set(baseline))
        self.assertIn(f"{victim}_intron1", baseline)
        self.assertNotIn(f"{victim}_intron1", patched)
        self.assertTrue(set(patched) < set(baseline))
        self.assertTrue(
            any("corrupt intron" in f.reason for f in report.feature_failures)
        )


class FileAndWorkerIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.manifest = write_genbank_dataset(
            Path(cls._tmp.name) / "good",
            n_taxa=6,
            n_genes=8,
            seed=11,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _clean(self):
        out = Path(tempfile.mkdtemp(dir=self._tmp.name))
        return _run(self.manifest.out_dir, out, order="seq")

    def test_bad_file_is_named_and_good_genomes_match(self) -> None:
        mixed = Path(self._tmp.name) / "mixed"
        mixed.mkdir(exist_ok=True)
        for src in self.manifest.files:
            dest = mixed / src.name
            if not dest.exists():
                os.link(src, dest)
        bad = mixed / "zzz_bad.gb"
        write_flatfile(
            bad,
            missing_origin(
                self.manifest.files[0].read_text(encoding="ascii"), random.Random(0)
            ),
        )
        clean_report, _, clean_seqs = self._clean()
        self.assertTrue(clean_report.ok)

        out = Path(self._tmp.name) / "mixed_out"
        with patch.object(log, "warning", wraps=log.warning) as warning:
            report, _, seqs = _run(mixed, out, order="seq")
        good = set(self.manifest.taxa)
        self.assertEqual(
            {
                k: tuple(row for row in rows if row[0] in good)
                for k, rows in seqs.items()
            },
            {
                k: tuple(row for row in rows if row[0] in good)
                for k, rows in clean_seqs.items()
            },
        )
        self.assertIsNotNone(report.failure_for(bad.name))
        self.assertEqual(report.failure_for(bad.name).stage, "parse")
        self.assertTrue(
            all(
                Path(item.file).name == bad.name
                for item in (*report.file_failures, *report.feature_failures)
            )
        )
        logged = " ".join(
            str(arg) for call in warning.call_args_list for arg in call.args
        )
        self.assertIn("extraction report", logged)
        self.assertIn(bad.name, logged)

    def test_all_corrupt_raises_actionable_error(self) -> None:
        src = Path(self._tmp.name) / "allbad"
        src.mkdir(exist_ok=True)
        for i, path in enumerate(self.manifest.files[:4]):
            write_flatfile(src / f"bad{i}.gb", "")
        out = Path(self._tmp.name) / "allbad_out"
        with self.assertRaises(ExtractionError) as caught:
            _run(src, out)
        message = str(caught.exception)
        self.assertIn("No items in main dictionary", message)
        self.assertIn("No usable records", message)
        self.assertIn("bad0.gb", message)
        self.assertGreaterEqual(caught.exception.report.files_failed, 4)
        self.assertEqual(caught.exception.report.files_with_regions, 0)

    def _mixed_crash_dir(self) -> tuple[Path, set[str]]:
        """Six good genomes plus two filenames a patched worker treats as fatal.

        Sorted order puts each bad file in a chunk with a good file when the
        run uses two threads (chunks of two).
        """
        dest_root = Path(self._tmp.name) / "crash_in"
        if dest_root.exists():
            return dest_root, set(self.manifest.taxa)
        dest_root.mkdir()
        for i, src in enumerate(self.manifest.files):
            os.link(src, dest_root / f"g{i}.gb")
        (dest_root / "g2_BAD.gb").write_text("not a genbank", encoding="ascii")
        (dest_root / "g5_BAD.gb").write_text("not a genbank", encoding="ascii")
        return dest_root, set(self.manifest.taxa)

    def _assert_recovered(self, seqs, good: set[str]) -> None:
        _, _, clean = self._clean()
        got = {
            k: tuple(row for row in rows if row[0] in good) for k, rows in seqs.items()
        }
        exp = {
            k: tuple(row for row in rows if row[0] in good) for k, rows in clean.items()
        }
        self.assertEqual(got, exp)

    @unittest.skipUnless(
        get_mp_context().get_start_method() == "fork",
        "the patched worker is visible to the pool only when it forks",
    )
    def test_chunk_exception_is_rerun_per_file(self) -> None:
        original = ExtractAndCollect._extract_recs_static

        def fake(files, *args):
            if any("BAD" in os.path.basename(f) for f in files):
                if len(files) > 1:
                    raise RuntimeError("chunk blew up")
                raise RuntimeError("file blew up")
            return original(files, *args)

        in_dir, good = self._mixed_crash_dir()
        out = Path(self._tmp.name) / "exc_out"
        with patch.object(
            ExtractAndCollect, "_extract_recs_static", staticmethod(fake)
        ):
            report, _, seqs = _run(in_dir, out, num_threads=2, order="alpha")
        self._assert_recovered(seqs, good)
        self.assertGreaterEqual(report.chunks_retried, 1)
        self.assertEqual(report.worker_crashes, 0)
        named = {Path(f.file).name for f in report.file_failures}
        self.assertEqual(named, {"g2_BAD.gb", "g5_BAD.gb"})
        self.assertTrue(all(f.stage == "worker" for f in report.file_failures))

    @unittest.skipUnless(
        get_mp_context().get_start_method() == "fork",
        "a crashing worker is visible to the parent only when the pool forks",
    )
    def test_worker_crash_is_rerun_per_file(self) -> None:
        original = ExtractAndCollect._extract_recs_static

        def fake(files, *args):
            if any("BAD" in os.path.basename(f) for f in files):
                if len(files) > 1:
                    os._exit(97)
                raise RuntimeError("file blew up")
            return original(files, *args)

        in_dir, good = self._mixed_crash_dir()
        out = Path(self._tmp.name) / "crash_out"
        with patch.object(
            ExtractAndCollect, "_extract_recs_static", staticmethod(fake)
        ):
            report, _, seqs = _run(in_dir, out, num_threads=2, order="alpha")
        self._assert_recovered(seqs, good)
        self.assertGreaterEqual(report.worker_crashes, 1)
        self.assertGreaterEqual(report.chunks_retried, 1)
        named = {Path(f.file).name for f in report.file_failures}
        self.assertEqual(named, {"g2_BAD.gb", "g5_BAD.gb"})


if __name__ == "__main__":
    unittest.main()
