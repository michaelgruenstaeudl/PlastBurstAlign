import io
import unittest

from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation

from plastburstalign.genbank_scanner import fast_read_genbank


class TestFastGenBankScanner(unittest.TestCase):
    """Parity and robustness tests for FastGenBankScanner."""

    def test_basic_features_and_headers_parity(self):
        seq = "atgcatgcat" * 150  # exactly 1500 bp matching LOCUS
        origin_lines = "\n".join(
            f"{i + 1:>9} " + " ".join(seq[i + j : i + j + 10] for j in range(0, 60, 10))
            for i in range(0, 1500, 60)
        )
        gb_data = f"""LOCUS       NC_012345               1500 bp    DNA     circular PLN 10-OCT-2020
DEFINITION  Chloroplast genome of Arabidopsis thaliana.
ACCESSION   NC_012345
VERSION     NC_012345.1
FEATURES             Location/Qualifiers
     source          1..1500
                     /organism="Arabidopsis thaliana"
     gene            10..100
                     /gene="psbA"
     CDS             10..100
                     /gene="psbA"
                     /product="photosystem II protein D1"
     gene            join(200..300,400..500)
                     /gene="rps12"
     CDS             join(200..300,
                     400..500)
                     /gene="rps12"
                     /trans_splicing
     tRNA            complement(600..670)
                     /gene="trnH-GUG"
     rRNA            700..1200
                     /gene="rrn16"
ORIGIN
{origin_lines}
//
"""
        rec_bio = SeqIO.read(io.StringIO(gb_data), "genbank")
        rec_fast = fast_read_genbank(io.StringIO(gb_data))

        self.assertEqual(rec_fast.id, rec_bio.id)
        self.assertEqual(rec_fast.name, rec_bio.name)
        self.assertEqual(rec_fast.description, rec_bio.description)
        self.assertEqual(str(rec_fast.seq), str(rec_bio.seq))

        bio_targets = [
            f for f in rec_bio.features if f.type in {"CDS", "gene", "tRNA", "rRNA"}
        ]
        self.assertEqual(len(rec_fast.features), len(bio_targets))

        for f_b, f_f in zip(bio_targets, rec_fast.features):
            self.assertEqual(f_f.type, f_b.type)
            self.assertEqual(str(f_f.location), str(f_b.location))
            self.assertEqual(f_f.qualifiers.get("gene"), f_b.qualifiers.get("gene"))

        # Verify valueless boolean qualifier (/trans_splicing)
        trans_feat = [
            f
            for f in rec_fast.features
            if f.type == "CDS" and "rps12" in f.qualifiers.get("gene", [])
        ][0]
        self.assertIn("trans_splicing", trans_feat.qualifiers)
        self.assertTrue(bool(trans_feat.qualifiers.get("trans_splicing")))

    def test_multiline_qualifiers_and_locations(self):
        gb_data = """LOCUS       TEST_REC                 500 bp    DNA     linear   PLN 01-JAN-2022
DEFINITION  Multi-line qualifier and coordinate test.
ACCESSION   TEST_ACC
VERSION     TEST_ACC.2
FEATURES             Location/Qualifiers
     CDS             join(10..50,
                     60..100,
                     150..200)
                     /gene="test_gene"
                     /product="a long product description that spans
                     across multiple lines of annotations"
                     /note="another note"
ORIGIN
        1 acgtacgtac gtacgtacgt acgtacgtac gtacgtacgt
//
"""
        rec = fast_read_genbank(io.StringIO(gb_data))
        self.assertEqual(rec.id, "TEST_ACC.2")
        self.assertEqual(rec.name, "TEST_REC")
        self.assertEqual(len(rec.features), 1)

        cds = rec.features[0]
        self.assertIsInstance(cds.location, CompoundLocation)
        self.assertEqual(len(cds.location.parts), 3)
        self.assertEqual(cds.qualifiers["gene"], ["test_gene"])
        self.assertEqual(
            cds.qualifiers["product"],
            [
                "a long product description that spans across multiple lines of annotations"
            ],
        )

    def test_complement_compound_location(self):
        gb_data = """LOCUS       COMP_REC                 500 bp    DNA     linear   PLN 01-JAN-2022
ACCESSION   COMP_ACC
VERSION     COMP_ACC.1
FEATURES             Location/Qualifiers
     CDS             complement(join(10..50,100..150))
                     /gene="comp_gene"
ORIGIN
        1 aaaaaa
//
"""
        rec = fast_read_genbank(io.StringIO(gb_data))
        cds = rec.features[0]
        self.assertEqual(cds.location.strand, -1)
        self.assertEqual(len(cds.location.parts), 2)

    def test_fallback_on_unusual_record(self):
        gb_data = """LOCUS       WEIRD_REC
FEATURES             Location/Qualifiers
     CDS             10..20
                     /gene="weird"
ORIGIN
        1 aaaaaa
//
"""
        rec = fast_read_genbank(io.StringIO(gb_data))
        self.assertIsNotNone(rec)
        self.assertIn(
            "weird",
            [
                f.qualifiers.get("gene", [""])[0]
                for f in rec.features
                if f.type == "CDS"
            ],
        )
