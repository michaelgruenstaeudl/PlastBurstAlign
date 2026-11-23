"""Extraction stays up when synthetic plastomes are corrupted.

A bad file is named in the report and cannot change records that came from a
good file. The only exception extraction is allowed to raise here is
:class:`~plastburstalign.extraction_report.ExtractionError`, and only when
nothing usable was read.
"""

from __future__ import annotations

import os
import random
import signal
import tempfile
import time
import unittest
from pathlib import Path

from plastburstalign.extraction_ops import ExtractAndCollect
from plastburstalign.extraction_report import ExtractionError
from plastburstalign.seqfeature_ops import PlastidData
from plastburstalign.user_parameters import UserParameters
from tests.fuzz_genbank import (
    ALL_CORRUPTIONS,
    FAILS_THE_FILE,
    apply,
    mutate_bytes,
    write_flatfile,
)
from tests.synth import make_plastome, make_template, write_genbank_dataset
from tests.test_synth import _taxon_of

_N_GOOD = 6
_N_GENES = 8
_SEED = 19


def _good_seqs(data: PlastidData, taxa: set[str]) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for region in data.nucleotides:
        rows = [
            (_taxon_of(rec.id), str(rec.seq))
            for rec in data.nucleotides[region]
            if _taxon_of(rec.id) in taxa
        ]
        if rows:
            out[region] = tuple(sorted(rows))
    return out


def _blamed(report) -> set[str]:
    names = {Path(item.file).name for item in report.file_failures}
    names.update(Path(item.file).name for item in report.feature_failures)
    names.update(Path(item.file).name for item in report.file_warnings)
    return names


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __enter__(self):
        def handle(signum, frame):
            raise TimeoutError(f"extraction exceeded {self.seconds}s")

        self._previous = signal.signal(signal.SIGALRM, handle)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *args):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._previous)


class FuzzGenBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.manifest = write_genbank_dataset(
            cls.root / "good",
            n_taxa=_N_GOOD,
            n_genes=_N_GENES,
            seed=_SEED,
        )
        cls.template = make_template(_SEED, _N_GENES)
        cls.bad_text = [
            make_plastome(100 + i, f"BAD{i}", cls.template) for i in range(2)
        ]
        cls.taxa = set(cls.manifest.taxa)
        cls.clean_seqs = _extract(
            cls.manifest.out_dir, cls.root / "clean_out", cls.taxa
        )[1]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_each_corruption_keeps_good_genomes(self) -> None:
        self.assertEqual(len(ALL_CORRUPTIONS), 22)
        self.assertTrue(FAILS_THE_FILE <= {name for name, _ in ALL_CORRUPTIONS})
        for name, fn in ALL_CORRUPTIONS:
            with self.subTest(corruption=name):
                in_dir = self.root / f"in_{name}"
                in_dir.mkdir()
                for src in self.manifest.files:
                    os.link(src, in_dir / src.name)
                rng = random.Random(name)
                bad_names = []
                for i, text in enumerate(self.bad_text):
                    bad_name = f"zzz_bad{i}.gb"
                    bad_names.append(bad_name)
                    write_flatfile(in_dir / bad_name, fn(text, rng))
                report, seqs = _extract(in_dir, self.root / f"out_{name}", self.taxa)
                self.assertEqual(seqs, self.clean_seqs)
                self.assertEqual(report.files_total, _N_GOOD + 2)
                blamed = _blamed(report)
                self.assertTrue(
                    blamed <= set(bad_names),
                    f"{name} blamed {blamed - set(bad_names)}",
                )
                if name in FAILS_THE_FILE:
                    failed = {Path(item.file).name for item in report.file_failures}
                    self.assertEqual(failed, set(bad_names), name)

    def test_all_corrupt_is_fatal(self) -> None:
        in_dir = self.root / "all_bad"
        in_dir.mkdir()
        rng = random.Random(0)
        for i in range(4):
            text = make_plastome(200 + i, f"DEAD{i}", self.template)
            write_flatfile(
                in_dir / f"dead{i}.gb", apply(text, rng, "missing_origin")[1]
            )
        params = UserParameters(
            in_dir=str(in_dir),
            out_dir=str(self.root / "all_bad_out"),
            select_mode="cds",
            num_threads=2,
            min_seq_length=3,
            order="seq",
        )
        data = PlastidData(params)
        try:
            with self.assertRaises(ExtractionError) as caught:
                ExtractAndCollect(data, params).extract()
        finally:
            data.store.close()
        message = str(caught.exception)
        self.assertIn("No items in main dictionary", message)
        self.assertIn("No usable records", message)
        self.assertIn("dead0.gb", message)
        self.assertGreaterEqual(caught.exception.report.files_failed, 4)
        self.assertEqual(caught.exception.report.files_with_regions, 0)
        self.assertIn("GenBank", message)

    def test_seeded_mutations_do_not_escape_or_hang(self) -> None:
        base = make_plastome(7, "MUT000", self.template)
        rng = random.Random(20261030)
        path = self.root / "mut.gb"
        started = time.perf_counter()
        for i in range(300):
            _, text = apply(base, rng)
            if rng.random() < 0.5:
                text = mutate_bytes(text, rng, rng.randint(1, 4))
            write_flatfile(path, text)
            try:
                with _Deadline(8):
                    result = ExtractAndCollect._extract_recs_static(
                        [str(path)],
                        "SelectMode.CDS",
                        True,
                        ["rps12"],
                        True,
                        True,
                        3,
                        False,
                        0,
                    )
            except ExtractionError:
                self.fail("the worker must not raise the fatal error; extract() does")
            nuc, prot, inv = result
            self.assertEqual(len(result), 3, i)
            self.assertIsInstance(inv, dict)
            self.assertTrue(nuc is None or isinstance(nuc, dict))
            self.assertTrue(prot is None or isinstance(prot, dict))
        self.assertLess(time.perf_counter() - started, 45)


def _extract(in_dir: Path, out_dir: Path, taxa: set[str]):
    params = UserParameters(
        in_dir=str(in_dir),
        out_dir=str(out_dir),
        select_mode="cds",
        num_threads=2,
        min_seq_length=3,
        min_num_taxa=1,
        order="seq",
    )
    data = PlastidData(params)
    try:
        ExtractAndCollect(data, params).extract()
        return data.extraction_report, _good_seqs(data, taxa)
    finally:
        data.store.close()


if __name__ == "__main__":
    unittest.main()
