"""Synthetic plastomes parse and extract to the manifest's expected regions."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation, SeqFeature

from plastburstalign.extraction_ops import ExtractAndCollect
from plastburstalign.genbank_scanner import fast_read_genbank
from plastburstalign.seqfeature_ops import GeneNameCleaner, PlastidData
from plastburstalign.user_parameters import UserParameters
from tests.synth import make_template, spellings_for, write_genbank_dataset

_TARGET = frozenset({"CDS", "gene", "tRNA", "rRNA"})
_MODES = ("cds", "igs", "int", "rna")


def _taxon_of(seq_id: str) -> str:
    return seq_id.rsplit("_", 1)[-1]


def _regions_by_taxon(data: PlastidData) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for region in data.nucleotides:
        for rec in data.nucleotides[region]:
            found.setdefault(_taxon_of(rec.id), set()).add(region)
    return found


class SynthPlastomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.manifest = write_genbank_dataset(
            Path(cls._tmp.name) / "genomes",
            n_taxa=3,
            n_genes=12,
            seed=7,
            keep_truth=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_spellings_clean_to_canonical_names(self) -> None:
        cleaner = GeneNameCleaner()
        ftype = {"cds": "CDS", "trna": "tRNA", "rrna": "rRNA"}
        template = make_template(0, 80)
        self.assertGreaterEqual(len(template.specs), 24)
        for spec in template.specs.values():
            self.assertEqual(spec.spellings, spellings_for(spec.name, spec.kind))
            for spelling in spec.spellings:
                feature = SeqFeature(
                    FeatureLocation(0, 3),
                    type=ftype[spec.kind],
                    qualifiers={"gene": [spelling]},
                )
                self.assertEqual(
                    cleaner.get_safe_gene(feature),
                    spec.name,
                    f"{spec.kind} spelling {spelling!r} did not clean to {spec.name}",
                )

    def test_fast_scanner_matches_seqio(self) -> None:
        for path in self.manifest.files:
            with self.subTest(path=path.name):
                fast = fast_read_genbank(path)
                bio = SeqIO.read(path, "genbank")
                self.assertEqual(fast.id, bio.id)
                self.assertEqual(fast.name, bio.name)
                self.assertEqual(str(fast.seq), str(bio.seq))
                self.assertNotIn("pba_skipped_features", fast.annotations)
                fast_feats = [f for f in fast.features if f.type in _TARGET]
                bio_feats = [f for f in bio.features if f.type in _TARGET]
                self.assertEqual(len(fast_feats), len(bio_feats))
                for got, exp in zip(fast_feats, bio_feats):
                    self.assertEqual(got.type, exp.type)
                    self.assertEqual(str(got.location), str(exp.location))
                    self.assertEqual(
                        got.qualifiers.get("gene"), exp.qualifiers.get("gene")
                    )

    def test_extraction_matches_expected_regions(self) -> None:
        root = Path(self._tmp.name)
        for mode in _MODES:
            with self.subTest(mode=mode):
                out_dir = root / f"out_{mode}"
                params = UserParameters(
                    in_dir=str(self.manifest.out_dir),
                    out_dir=str(out_dir),
                    select_mode=mode,
                    num_threads=2,
                    min_seq_length=3,
                    min_num_taxa=1,
                    order="seq",
                )
                data = PlastidData(params)
                try:
                    ExtractAndCollect(data, params).extract()
                    report = data.extraction_report
                    self.assertTrue(
                        report.ok,
                        f"clean synthetic input was not clean:\n{report.summary()}",
                    )
                    self.assertEqual(report.files_failed, 0)
                    self.assertEqual(report.features_skipped, 0)
                    self.assertEqual(report.files_total, self.manifest.n_taxa)
                    self.assertEqual(report.files_with_regions, self.manifest.n_taxa)
                    self.assertEqual(
                        data.store.get_active_regions("seq"),
                        self.manifest.regions(mode),
                    )
                    counts = {
                        region: len(data.nucleotides[region])
                        for region in data.nucleotides
                    }
                    self.assertEqual(counts, self.manifest.expected[mode])
                    by_taxon = _regions_by_taxon(data)
                    for taxon, per_mode in self.manifest.per_taxon.items():
                        self.assertEqual(
                            by_taxon.get(taxon, set()),
                            set(per_mode[mode]),
                            taxon,
                        )
                    if mode == "cds":
                        self.assertEqual(set(data.nucleotides), set(data.proteins))
                        excluded = set(self.manifest.exclude_fullcds)
                        for taxon, genes in self.manifest.truth_cds.items():
                            for gene, seq in genes.items():
                                if gene in excluded:
                                    continue
                                recs = [
                                    r
                                    for r in data.nucleotides[gene]
                                    if _taxon_of(r.id) == taxon
                                ]
                                self.assertEqual(len(recs), 1, f"{gene} {taxon}")
                                self.assertEqual(str(recs[0].seq), seq)
                finally:
                    data.store.close()

    def test_generator_speed(self) -> None:
        n_taxa = 40
        with tempfile.TemporaryDirectory() as tmp:
            started = time.perf_counter()
            manifest = write_genbank_dataset(
                Path(tmp) / "speed", n_taxa=n_taxa, n_genes=24, seed=1
            )
            elapsed = time.perf_counter() - started
        self.assertEqual(manifest.n_taxa, n_taxa)
        rate = n_taxa / elapsed
        # The generator is around a few hundred genomes/s. The bound is loose
        # so a busy laptop does not flake; the measured rate is what we report.
        self.assertGreater(rate, 20, f"{rate:.1f} genomes/s over {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
