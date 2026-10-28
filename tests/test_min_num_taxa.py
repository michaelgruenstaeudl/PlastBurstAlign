import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.extraction_ops import DataCleaning, ExtractAndCollect
from plastburstalign.seqfeature_ops import PlastidData
from plastburstalign.user_parameters import UserParameters, UserParametersScript


class MinNumTaxaTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()

    def record(self, name="taxon1"):
        record = SeqRecord(
            Seq("AACGTACGTTAGCATGACCTGATCGTAGCTAGCATGCAACGTTAGCTACGTACGATCGTAC"),
            id=name,
            name=name,
            annotations={"molecule_type": "DNA"},
        )
        record.features = [
            SeqFeature(FeatureLocation(0, 10), type="gene", qualifiers={"gene": ["trnH-GUG"]}),
            SeqFeature(FeatureLocation(0, 10), type="tRNA", qualifiers={"gene": ["trnH-GUG"]}),
            SeqFeature(FeatureLocation(10, 25), type="gene", qualifiers={"gene": ["psbA"]}),
            SeqFeature(FeatureLocation(10, 25), type="CDS", qualifiers={"gene": ["psbA"]}),
        ]
        return record

    def test_cli_parsing_types_and_values(self):
        parse = UserParametersScript._parse_min_num_taxa

        # Integers must return int
        self.assertEqual(parse("1"), 1)
        self.assertIs(type(parse("1")), int)
        self.assertEqual(parse("5"), 5)
        self.assertIs(type(parse("5")), int)

        # Floats, including 1.0, must return float
        self.assertEqual(parse("1.0"), 1.0)
        self.assertIs(type(parse("1.0")), float)
        self.assertEqual(parse("0.5"), 0.5)
        self.assertIs(type(parse("0.5")), float)
        self.assertEqual(parse("0.1"), 0.1)
        self.assertIs(type(parse("0.1")), float)

        # Invalid strings
        for invalid in ("abc", "1.0.0", "nan", "-nan", "inf", "-inf"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse(invalid)

    def test_user_parameters_validation(self):
        base_args = {
            "in_dir": str(self.input_dir),
            "out_dir": str(self.root / "output"),
            "select_mode": "cds",
        }

        # Valid values
        for val in (1, 5, 0.1, 0.5, 1.0):
            with self.subTest(val=val):
                p = UserParameters(min_num_taxa=val, **base_args)
                self.assertEqual(p["min_num_taxa"], val)
                self.assertIs(type(p["min_num_taxa"]), type(val))

        # Invalid integers (< 1)
        for invalid_int in (0, -1, -10):
            with self.subTest(invalid_int=invalid_int):
                with self.assertRaises(ValueError):
                    UserParameters(min_num_taxa=invalid_int, **base_args)

        # Invalid floats (<= 0 or > 1 or NaN)
        for invalid_float in (0.0, -0.1, 1.0001, 2.0, float("nan")):
            with self.subTest(invalid_float=invalid_float):
                with self.assertRaises(ValueError):
                    UserParameters(min_num_taxa=invalid_float, **base_args)

        # Invalid types (booleans, strings, None)
        for invalid_type in (True, False, "1", None, [1]):
            with self.subTest(invalid_type=invalid_type):
                with self.assertRaises(TypeError):
                    UserParameters(min_num_taxa=invalid_type, **base_args)

    def test_cli_integration_coercion_preservation(self):
        argv = [
            "plastburstalign",
            "-i",
            str(self.input_dir),
            "-o",
            str(self.root / "output"),
            "-t",
            "1.0",
        ]
        with patch.object(sys, "argv", argv):
            params = UserParametersScript()
            self.assertEqual(params["min_num_taxa"], 1.0)
            self.assertIs(type(params["min_num_taxa"]), float)

    def test_frequency_filtering_logic_regression(self):
        out_dir = self.root / "output"
        params_float = UserParameters(
            in_dir=str(self.input_dir),
            out_dir=str(out_dir / "float"),
            select_mode="cds",
            min_num_taxa=1.0,
        )
        params_int = UserParameters(
            in_dir=str(self.input_dir),
            out_dir=str(out_dir / "int"),
            select_mode="cds",
            min_num_taxa=1,
        )

        # Single taxon record extracted
        data_float = PlastidData(params_float)
        data_float.num_taxa = 2  # Total dataset has 2 taxa
        ExtractAndCollect(data_float, params_float)._extract_cds(
            self.record("taxon1"), data_float.nucleotides, data_float.proteins
        )
        self.assertIn("psbA", data_float.nucleotides)

        # With min_num_taxa=1.0 (float) and count=1 over 2 taxa (1/2 < 1.0), it must be removed
        DataCleaning(data_float, params_float).clean()
        self.assertNotIn("psbA", data_float.nucleotides)

        # With min_num_taxa=1 (int) and count=1 (1 >= 1), it must be retained
        data_int = PlastidData(params_int)
        data_int.num_taxa = 2
        ExtractAndCollect(data_int, params_int)._extract_cds(
            self.record("taxon1"), data_int.nucleotides, data_int.proteins
        )
        self.assertIn("psbA", data_int.nucleotides)
        DataCleaning(data_int, params_int).clean()
        self.assertIn("psbA", data_int.nucleotides)

    def test_frequency_filtering_num_taxa_guard(self):
        params = UserParameters(
            in_dir=str(self.input_dir),
            out_dir=str(self.root / "guard_test"),
            select_mode="cds",
            min_num_taxa=0.5,
        )

        for invalid_taxa in (None, 0, -1):
            with self.subTest(num_taxa=invalid_taxa):
                data = PlastidData(params)
                data.num_taxa = invalid_taxa
                ExtractAndCollect(data, params)._extract_cds(self.record(), data.nucleotides, data.proteins)
                with self.assertRaises(AttributeError):
                    DataCleaning(data, params).clean()