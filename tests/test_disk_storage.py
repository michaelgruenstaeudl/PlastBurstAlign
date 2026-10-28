"""Tests for the indexed disk-backed feature storage engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from plastburstalign.storage import DiskBackedFeatureStore, FeatureStoreView


class DiskBackedFeatureStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_features.db"
        self.store = DiskBackedFeatureStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.temp_dir.cleanup()

    def test_record_insertion_and_ordering(self) -> None:
        rec_a = SeqRecord(Seq("ATGC"), id="TaxA_psbA", description="psbA gene")
        rec_b = SeqRecord(Seq("ATGG"), id="TaxB_psbA", description="psbA gene")
        rec_c = SeqRecord(Seq("TTTT"), id="TaxA_matK", description="matK gene")

        self.assertTrue(self.store.add_seqrecord("psbA", "TaxonA", "nucl", rec_a))
        self.assertTrue(self.store.add_seqrecord("psbA", "TaxonB", "nucl", rec_b))
        self.assertTrue(self.store.add_seqrecord("matK", "TaxonA", "nucl", rec_c))

        # Duplicate insertion for same taxon and region must return False
        self.assertFalse(self.store.add_seqrecord("psbA", "TaxonA", "nucl", rec_a))

        # Check sequence order
        self.assertEqual(self.store.get_active_regions("seq"), ["psbA", "matK"])
        self.assertEqual(self.store.get_active_regions("alpha"), ["matK", "psbA"])

    def test_inverted_repeat_deduplication(self) -> None:
        # First orientation: psbA_trnH
        self.assertTrue(
            self.store.add_record(
                region_name="psbA_trnH",
                taxon_name="Taxon1",
                seq_type="nucl",
                seq_id="Taxon1_psbA_trnH",
                description="",
                sequence="AACCGG",
                inv_name="trnH_psbA",
            )
        )
        # Attempt to insert same taxon under inverted name: trnH_psbA
        self.assertFalse(
            self.store.add_record(
                region_name="trnH_psbA",
                taxon_name="Taxon1",
                seq_type="nucl",
                seq_id="Taxon1_trnH_psbA",
                description="",
                sequence="GGCCAA",
                inv_name="psbA_trnH",
            )
        )

    def test_get_region_summary_aggregation(self) -> None:
        self.store.add_record("rpoB", "TaxA", "nucl", "id1", "", "ATG")
        self.store.add_record("rpoB", "TaxB", "nucl", "id2", "", "ATGCCC")
        self.store.add_record("atpA", "TaxA", "nucl", "id3", "", "AT")

        summary = self.store.get_region_summary("nucl")
        self.assertEqual(summary["rpoB"]["count"], 2)
        self.assertEqual(summary["rpoB"]["max_len"], 6)
        self.assertEqual(summary["atpA"]["count"], 1)
        self.assertEqual(summary["atpA"]["max_len"], 2)

    def test_streaming_fasta_export(self) -> None:
        self.store.add_record("rbcL", "TaxA", "nucl", "TaxA_rbcL", "TaxA_rbcL desc", "ATGCATGC")
        self.store.add_record("rbcL", "TaxB", "nucl", "TaxB_rbcL", "TaxB_rbcL desc", "ATGCATGG")

        out_fasta = Path(self.temp_dir.name) / "nucl_rbcL.unalign.fasta"
        count = self.store.export_fasta("rbcL", "nucl", out_fasta)
        self.assertEqual(count, 2)

        # Verify output via Biopython SeqIO
        records = list(SeqIO.parse(out_fasta, "fasta"))
        self.assertEqual(len(records), 2)
        self.assertEqual(str(records[0].seq), "ATGCATGC")
        self.assertEqual(str(records[1].seq), "ATGCATGG")

    def test_feature_store_view_interface(self) -> None:
        self.store.add_record("psaA", "TaxA", "nucl", "id1", "", "AAA")
        self.store.add_record("psaB", "TaxA", "nucl", "id2", "", "CCC")

        view = FeatureStoreView(self.store, "nucl")
        self.assertEqual(len(view), 2)
        self.assertIn("psaA", view)
        self.assertNotIn("unknown", view)
        self.assertEqual(view.keys(), ["psaA", "psaB"])

        # Test hydration on access
        records = view["psaA"]
        self.assertEqual(len(records), 1)
        self.assertEqual(str(records[0].seq), "AAA")

        # Deletion
        del view["psaA"]
        self.assertNotIn("psaA", view)
        self.assertEqual(len(view), 1)
        self.assertIsNone(view.get("psaA"))

    def test_context_manager(self) -> None:
        with DiskBackedFeatureStore(Path(self.temp_dir.name) / "ctx.db") as store:
            self.assertTrue(store.add_record("matK", "TaxonA", "nucl", "id1", "", "ATGC"))
            self.assertTrue(store.has_region("matK"))

    def test_feature_store_view_edge_cases(self) -> None:
        self.store.add_record("ycf1", "TaxA", "nucl", "id1", "", "ATG")
        view = FeatureStoreView(self.store, "nucl")

        # Test KeyError on missing key
        with self.assertRaises(KeyError):
            _ = view["non_existent_gene"]

        # Test items and values
        items = view.items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][0], "ycf1")

        values = view.values()
        self.assertEqual(len(values), 1)
        self.assertEqual(len(values[0]), 1)


if __name__ == "__main__":
    unittest.main()
