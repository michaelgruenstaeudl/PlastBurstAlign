import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.extraction_ops import DataCleaning, ExtractAndCollect, RecordFeatures
from plastburstalign.plast_burst_align import PlastBurstAlign
from plastburstalign.seqfeature_ops import GeneNameCleaner, PlastidData
from plastburstalign.user_parameters import SelectMode, UserParameters, UserParametersScript


class CopyAligner:
    """Test double for already aligned, equal-length fixture sequences."""

    exec_path = "test-aligner"

    def align(self, input_file, output_file):
        shutil.copyfile(input_file, output_file)


class RNAModeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()

    def parameters(self, **overrides):
        values = {
            "in_dir": str(self.input_dir),
            "out_dir": str(self.root / "output"),
            "select_mode": "rna",
            "num_threads": 1,
            "min_seq_length": 1,
            "min_num_taxa": 1,
        }
        values.update(overrides)
        return UserParameters(**values)

    @staticmethod
    def record(name="taxon1"):
        record = SeqRecord(
            Seq("AACGTACGTTAGCATGACCTGATCGTAGCTAGCATGCAACGTTAGCTACGTACGATCGTAC"),
            id=name,
            name=name,
            annotations={"molecule_type": "DNA"},
        )
        record.features = [
            SeqFeature(FeatureLocation(0, 7), type="gene", qualifiers={"gene": ["trnH-GUG"]}),
            SeqFeature(FeatureLocation(0, 7), type="tRNA", qualifiers={"gene": ["trnH-GUG"]}),
            SeqFeature(
                FeatureLocation(7, 18, strand=-1), type="rRNA", qualifiers={"gene": ["rrn16"]}
            ),
            SeqFeature(
                CompoundLocation([FeatureLocation(20, 24), FeatureLocation(28, 32)]),
                type="tRNA",
                qualifiers={"gene": ["trnK-UUU"]},
            ),
            SeqFeature(FeatureLocation(32, 39), type="tRNA", qualifiers={"gene": ["trnH-gug"]}),
            SeqFeature(FeatureLocation(39, 48), type="CDS", qualifiers={"gene": ["psbA"]}),
            SeqFeature(FeatureLocation(48, 52), type="tRNA"),
            SeqFeature(FeatureLocation(52, 56), type="rRNA", qualifiers={"gene": ["orf123"]}),
        ]
        return record

    def extract_record(self, **kwargs) -> tuple[PlastidData, dict]:
        import tempfile

        # Give every test isolation a fresh tempdir to store the db
        temp_out = tempfile.mkdtemp()
        params = self.parameters(**kwargs)
        params["out_dir"] = temp_out

        data = PlastidData(params)
        data.files = [str(self.input_dir / "taxon1.gb")]
        data.num_taxa = 1
        ExtractAndCollect(data, params)._extract_rna(self.record(), data.nucleotides)
        return data, params

    def test_cli_and_yaml_accept_rna(self):
        base_args = ["plastburstalign", "-i", str(self.input_dir), "-o", str(self.root / "output")]
        with patch.object(sys, "argv", base_args + ["-s", "RNA"]):
            params = UserParametersScript()
        self.assertIs(params["select_mode"], SelectMode.RNA)
        self.assertFalse(params["select_mode"].collects_proteins())
        config = self.root / "config.yaml"
        config.write_text('select_mode: "rna"\n', encoding="ascii")
        with patch.object(sys, "argv", base_args + ["--config", str(config)]):
            params = UserParametersScript()
        self.assertIs(params["select_mode"], SelectMode.RNA)
        self.assertIn("rna", UserParametersScript._get_parser().format_help())

    def test_extracts_untrimmed_strand_correct_spliced_rnas(self):
        data, _ = self.extract_record()
        self.assertIsNone(data.proteins)
        self.assertEqual(list(data.nucleotides), ["trnH_GUG", "rrn16", "trnK_UUU"])
        self.assertEqual(len(data.nucleotides["trnH_GUG"]), 1)
        self.assertEqual(str(data.nucleotides["trnH_GUG"][0].seq), "AACGTAC")
        record = self.record()
        self.assertEqual(
            data.nucleotides["rrn16"][0].seq,
            record.seq[7:18].reverse_complement(),
        )
        self.assertEqual(
            data.nucleotides["trnK_UUU"][0].seq,
            record.seq[20:24] + record.seq[28:32],
        )
        self.assertEqual(data.nucleotides["rrn16"][0].id, "rrn16_taxon1")

    def test_rrna_names_are_standardized_to_canonical_gene_names(self):
        cleaner = GeneNameCleaner()
        aliases = {
            "rrn16": (
                "rrn16",
                "rrn16S",
                "16S",
                "16S rRNA",
                "16S ribosomal RNA",
                "rRNA16",
                "rRNA16S",
                "rns",
                "SSU",
                "SSU rRNA",
                "SSU rDNA",
            ),
            "rrn23": (
                "rrn23",
                "rrn23S",
                "23S",
                "23S rRNA",
                "23S ribosomal RNA",
                "rRNA23",
                "rRNA23S",
                "rnl",
                "LSU",
                "LSU rRNA",
                "LSU rDNA",
            ),
            "rrn4.5": (
                "rrn4.5",
                "rrn4.5S",
                "4.5S",
                "4.5S rRNA",
                "4.5S ribosomal RNA",
                "rRNA4.5",
                "rRNA4.5S",
                "rrn4_5",
                "rrn4_5S",
            ),
            "rrn5": (
                "rrn5",
                "rrn5S",
                "5S",
                "5S rRNA",
                "5S ribosomal RNA",
                "rRNA5",
                "rRNA5S",
                "5S rDNA",
            ),
        }
        for expected, annotations in aliases.items():
            for annotation in annotations:
                with self.subTest(annotation=annotation):
                    feature = SeqFeature(
                        FeatureLocation(0, 1),
                        type="rRNA",
                        qualifiers={"gene": [annotation]},
                    )
                    self.assertEqual(cleaner.get_safe_gene(feature), expected)

    def test_trna_name_cleaning_remains_separate(self):
        cleaner = GeneNameCleaner()
        self.assertEqual(cleaner.clean_trna_gene("trnH-gug"), "trnH_GUG")

    def test_rna_toggles_select_independent_types(self):
        for trna, rrna, expected in (
            (True, True, ["trnH_GUG", "rrn16", "trnK_UUU"]),
            (True, False, ["trnH_GUG", "trnK_UUU"]),
            (False, True, ["rrn16"]),
        ):
            with self.subTest(include_trna=trna, include_rrna=rrna):
                data, _ = self.extract_record(include_trna=trna, include_rrna=rrna)
                self.assertEqual(list(data.nucleotides), expected)

    def test_cli_rna_toggles_override_yaml_only_when_explicit(self):
        config = self.root / "config.yaml"
        base_args = [
            "plastburstalign",
            "-i",
            str(self.input_dir),
            "-o",
            str(self.root / "output"),
            "--config",
            str(config),
        ]
        for trna, rrna in ((True, True), (True, False), (False, True)):
            config.write_text(
                f"select_mode: rna\ninclude_trna: {str(trna).lower()}\n"
                f"include_rrna: {str(rrna).lower()}\n",
                encoding="ascii",
            )
            for flags, expected in (
                ([], (trna, rrna)),
                (["--trna", "--no-rrna"], (True, False)),
                (["--no-trna", "--rrna"], (False, True)),
                (["--trna", "--rrna"], (True, True)),
            ):
                with self.subTest(config=(trna, rrna), flags=flags):
                    with patch.object(sys, "argv", base_args + flags):
                        params = UserParametersScript()
                    self.assertEqual((params["include_trna"], params["include_rrna"]), expected)

    def test_rna_toggle_validation(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.parameters(include_trna=False, include_rrna=False)
        for option in ("include_trna", "include_rrna"):
            for invalid in ("false", 0, None):
                with self.subTest(option=option, invalid=invalid):
                    with self.assertRaises(TypeError):
                        self.parameters(**{option: invalid})
        for mode in ("cds", "int", "igs"):
            with self.subTest(mode=mode):
                self.parameters(select_mode=mode, include_trna=False, include_rrna=False)
        with patch.object(
            sys,
            "argv",
            [
                "plastburstalign",
                "-s",
                "rna",
                "--no-trna",
                "--no-rrna",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "at least one"):
                UserParametersScript()

    def test_gene_and_region_exclusions(self):
        data, params = self.extract_record(exclude_fullcds=["trnH_GUG"], exclude_region=["rrn16"])
        self.assertNotIn("trnH_GUG", data.nucleotides)
        DataCleaning(data, params).clean()
        self.assertEqual(list(data.nucleotides), ["trnK_UUU"])

    def test_length_and_taxon_thresholds(self):
        data, params = self.extract_record(min_seq_length=8)
        DataCleaning(data, params).clean()
        self.assertEqual(list(data.nucleotides), ["rrn16", "trnK_UUU"])
        for threshold in (2, 0.75):
            with self.subTest(min_num_taxa=threshold):
                data, params = self.extract_record(min_num_taxa=threshold)
                data.num_taxa = 2
                DataCleaning(data, params).clean()
                self.assertEqual(len(data.nucleotides), 0)

    def test_existing_feature_selection_is_unchanged(self):
        features = RecordFeatures(self.record())
        self.assertEqual([f.type for f in features.cds_feats], ["CDS"])
        self.assertEqual([f.type for f in features.int_feats], ["tRNA", "tRNA", "tRNA", "CDS"])
        self.assertEqual([f.type for f in features.igs_feats], ["gene"])
        for mode in SelectMode:
            with self.subTest(mode=mode):
                data = PlastidData(self.parameters(select_mode=mode))
                self.assertEqual(data.proteins is not None, mode == SelectMode.CDS)

    def test_pipeline_writes_nucleotide_alignments_and_concatenation(self):
        for name in ("taxon1", "taxon2"):
            SeqIO.write(self.record(name), self.input_dir / f"{name}.gb", "genbank")
        params = self.parameters(min_num_taxa=2, concat=True, order="alpha")
        pipeline = PlastBurstAlign(params)
        with (
            patch("plastburstalign.alignment_ops.get_alignment_tool", return_value=CopyAligner()),
            patch(
                "plastburstalign.extraction_ops.ProteinFeature",
                side_effect=AssertionError("RNA translated"),
            ),
            patch(
                "plastburstalign.alignment_ops.BackTranslation",
                side_effect=AssertionError("RNA back-translated"),
            ),
        ):
            pipeline.execute()
        output = Path(params["out_dir"])
        expected_regions = ["rrn16", "trnH_GUG", "trnK_UUU"]
        for region in expected_regions:
            with (output / f"nucl_{region}.unalign.fasta").open() as handle:
                unaligned = list(SeqIO.parse(handle, "fasta"))
            fasta = AlignIO.read(output / "fasta" / f"nucl_{region}.aligned.fasta", "fasta")
            nexus = AlignIO.read(output / "nexus" / f"nucl_{region}.aligned.nexus", "nexus")
            self.assertEqual(len(unaligned), 2)
            self.assertEqual(len(fasta), 2)
            self.assertEqual([str(r.seq) for r in fasta], [str(r.seq) for r in unaligned])
            self.assertEqual([str(r.seq) for r in nexus], [str(r.seq) for r in unaligned])
        concatenated = AlignIO.read(output / "nucl_3concat.aligned.fasta", "fasta")
        self.assertEqual({r.id for r in concatenated}, {"concat_taxon1", "concat_taxon2"})
        self.assertEqual(concatenated.get_alignment_length(), 26)
        self.assertEqual(
            str(concatenated[0].seq),
            "".join(
                str(pipeline.plastid_data.nucleotides[region][0].seq) for region in expected_regions
            ),
        )
        self.assertEqual(len(AlignIO.read(output / "nucl_3concat.aligned.nexus", "nexus")), 2)
        self.assertEqual(list(output.rglob("prot_*")), [])

    def test_pipeline_does_not_align_disabled_rna_types(self):
        for name in ("taxon1", "taxon2"):
            SeqIO.write(self.record(name), self.input_dir / f"{name}.gb", "genbank")
        for trna, rrna, expected in (
            (True, False, {"trnH_GUG", "trnK_UUU"}),
            (False, True, {"rrn16"}),
        ):
            with self.subTest(include_trna=trna, include_rrna=rrna):
                params = self.parameters(
                    include_trna=trna,
                    include_rrna=rrna,
                    out_dir=str(self.root / f"output_{trna}_{rrna}"),
                    min_num_taxa=2,
                )
                aligner = CopyAligner()
                with (
                    patch("plastburstalign.alignment_ops.get_alignment_tool", return_value=aligner),
                    patch.object(aligner, "align", wraps=aligner.align) as align,
                ):
                    pipeline = PlastBurstAlign(params)
                    pipeline.execute()
                self.assertEqual(set(pipeline.plastid_data.nucleotides), expected)
                self.assertEqual(align.call_count, len(expected))
                self.assertEqual(
                    {Path(call.args[0]).name for call in align.call_args_list},
                    {f"nucl_{region}.unalign.fasta" for region in expected},
                )
                output = Path(params["out_dir"])
                self.assertEqual(
                    {path.name for path in (output / "fasta").glob("*.aligned.fasta")},
                    {f"nucl_{region}.aligned.fasta" for region in expected},
                )
                self.assertEqual(
                    {path.name for path in (output / "nexus").glob("*.aligned.nexus")},
                    {f"nucl_{region}.aligned.nexus" for region in expected},
                )

    def test_early_filtering_during_extraction(self):
        # 1. exclude_region is skipped directly during extraction (before clean())
        data, params = self.extract_record(exclude_region=["rrn16"])
        self.assertNotIn("rrn16", data.nucleotides)
        self.assertIn("trnK_UUU", data.nucleotides)
        self.assertIn("trnH_GUG", data.nucleotides)

        # 2. min_seq_length filters out short features directly during extraction (before clean())
        # trnH_GUG length is 7 bp; min_seq_length=8 should discard it immediately
        data, params = self.extract_record(min_seq_length=8)
        self.assertNotIn("trnH_GUG", data.nucleotides)
        self.assertIn("rrn16", data.nucleotides)
        self.assertIn("trnK_UUU", data.nucleotides)


if __name__ == "__main__":
    unittest.main()
