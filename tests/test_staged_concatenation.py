import io
import unittest

from Bio.Nexus.Nexus import Nexus as NexusAlignment
from Bio.Nexus.Nexus import combine as nexus_combine

from plastburstalign.nexus_ops import ConcatenationError, IncrementalNexusAccumulator


class StagedConcatenationTests(unittest.TestCase):
    def test_accumulator_byte_identity_with_nexus_combine(self):
        nexus_a = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=6;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon1 ATGCAT\n"
            "concat_taxon2 ATGC--\n"
            ";\n"
            "end;\n"
        )
        nexus_b = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=4;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon1 GGGG\n"
            "concat_taxon2 CCCC\n"
            ";\n"
            "end;\n"
        )

        m_a1 = NexusAlignment(nexus_a)
        m_b1 = NexusAlignment(nexus_b)
        combined_batch = nexus_combine([("geneA", m_a1), ("geneB", m_b1)])

        m_a2 = NexusAlignment(nexus_a)
        m_b2 = NexusAlignment(nexus_b)
        acc = IncrementalNexusAccumulator()
        acc.add("geneA", m_a2)
        acc.add("geneB", m_b2)
        combined_inc = acc.finalize()

        buf_batch = io.StringIO()
        combined_batch.write_nexus_data(buf_batch)

        buf_inc = io.StringIO()
        combined_inc.write_nexus_data(buf_inc)

        self.assertEqual(buf_batch.getvalue(), buf_inc.getvalue())

    def test_accumulator_disjoint_taxa_padding(self):
        nexus_a = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=3;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon1 ATG\n"
            "concat_taxon2 ATG\n"
            ";\n"
            "end;\n"
        )
        nexus_b = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=2 nchar=3;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_taxon2 CCC\n"
            "concat_taxon3 CCC\n"
            ";\n"
            "end;\n"
        )

        acc = IncrementalNexusAccumulator()
        acc.add("gene1", NexusAlignment(nexus_a))
        acc.add("gene2", NexusAlignment(nexus_b))
        res = acc.finalize()

        self.assertEqual(res.ntax, 3)
        self.assertEqual(res.nchar, 6)
        # taxon1 missing gene2 -> ???
        self.assertEqual(str(res.matrix["concat_taxon1"]), "ATG???")
        # taxon3 missing gene1 -> ???
        self.assertEqual(str(res.matrix["concat_taxon3"]), "???CCC")
        # taxon2 present in both
        self.assertEqual(str(res.matrix["concat_taxon2"]), "ATGCCC")

    def test_accumulator_empty_and_single(self):
        acc = IncrementalNexusAccumulator()
        self.assertIsNone(acc.finalize())

        nexus_single = (
            "#NEXUS\n"
            "begin data;\n"
            "dimensions ntax=1 nchar=4;\n"
            "format datatype=dna gap=- missing=?;\n"
            "matrix\n"
            "concat_t1 AAAA\n"
            ";\n"
            "end;\n"
        )
        acc.add("g1", NexusAlignment(nexus_single))
        res = acc.finalize()
        self.assertEqual(res.nchar, 4)
        self.assertEqual(str(res.matrix["concat_t1"]), "AAAA")

    def test_concatenation_error_inheritance(self):
        err = ConcatenationError("Test error")
        self.assertIsInstance(err, RuntimeError)
        self.assertEqual(str(err), "Test error")

    def test_accumulator_empty_finalize_returns_none(self):
        acc = IncrementalNexusAccumulator()
        self.assertIsNone(acc.finalize())

    def test_accumulator_datatype_mismatch_and_charlabels(self):
        nex1_str = """#NEXUS
BEGIN DATA;
    DIMENSIONS NTAX=2 NCHAR=3;
    FORMAT DATATYPE=DNA GAP=- MISSING=? MATCHCHAR=.;
    MATRIX
    TaxonA ATG
    TaxonB ACG
    ;
END;"""
        nex2_str = """#NEXUS
BEGIN DATA;
    DIMENSIONS NTAX=2 NCHAR=3;
    FORMAT DATATYPE=PROTEIN GAP=- MISSING=? MATCHCHAR=.;
    MATRIX
    TaxonA MET
    TaxonB THR
    ;
END;"""
        n1 = NexusAlignment(nex1_str)
        n1.charsets = {"codon1": [0, 1]}
        n1.taxsets = {"ingroup": ["TaxonA"]}

        n2 = NexusAlignment(nex2_str)
        n2.charsets = {"site1": [0]}
        n2.taxsets = {"outgroup": ["TaxonB"]}
        n2.charlabels = {0: "lbl1", 1: "lbl2"}

        acc = IncrementalNexusAccumulator()
        acc.add("geneA", n1)
        acc.add("geneB", n2)
        res = acc.finalize()

        self.assertIsNotNone(res)
        assert res is not None
        self.assertEqual(res.datatype, "None")
        self.assertEqual(res.nchar, 6)
        self.assertIn("geneA.codon1", res.charsets)
        self.assertIn("geneB.site1", res.charsets)
        self.assertEqual(res.charsets["geneB.site1"], [3])
        self.assertIn("geneB.outgroup", res.taxsets)
        self.assertEqual(res.charlabels[3], "lbl1")


if __name__ == "__main__":
    unittest.main()
