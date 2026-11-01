"""Tests for ``plastburstalign.partitions`` (tables, codon tracking, writers)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from plastburstalign.nexus_ops import (
    fast_write_nexus_partitions,
    fast_write_raxml_partitions,
)
from plastburstalign.partitions import PartitionEntry, PartitionTable, stride_runs


def two_locus_table() -> PartitionTable:
    table = PartitionTable()
    table.append_locus("atpA", 500, "cds")
    table.append_locus("rbcL", 1428, "cds")
    return table


class StrideRunTests(unittest.TestCase):
    def test_runs(self):
        self.assertEqual(stride_runs([]), [])
        self.assertEqual(stride_runs([5]), [(5, 5)])
        self.assertEqual(stride_runs([1, 2, 3, 7, 8, 12]), [(1, 3), (7, 8), (12, 12)])
        self.assertEqual(stride_runs([1, 4, 7, 19, 22], 3), [(1, 7), (19, 22)])
        self.assertEqual(stride_runs([1, 4, 5, 8], 3), [(1, 4), (5, 8)])


class TableTests(unittest.TestCase):
    def test_coordinates_are_one_based_inclusive_and_contiguous(self):
        table = two_locus_table()
        self.assertEqual(len(table), 2)
        self.assertEqual(table.names, ("atpA", "rbcL"))
        self.assertEqual((table[0].start, table[0].end), (1, 500))
        self.assertEqual((table["rbcL"].start, table["rbcL"].end), (501, 1928))
        self.assertEqual(table.nchar, 1928)
        self.assertEqual(table[0].width, 500)
        self.assertEqual([e.name for e in table], ["atpA", "rbcL"])
        self.assertIn("2 partitions", repr(table))
        with self.assertRaises(KeyError):
            table["nope"]

    def test_legacy_charpartitions_structure(self):
        table = two_locus_table()
        legacy = table.to_charpartitions()
        self.assertEqual(list(legacy), ["combined"])
        self.assertEqual(legacy["combined"]["atpA"], list(range(0, 500)))
        self.assertEqual(legacy["combined"]["rbcL"], list(range(500, 1928)))

    def test_validation(self):
        table = PartitionTable()
        table.append_locus("a", 5)
        with self.assertRaises(ValueError):
            table.append_locus("a", 5)  # duplicate name
        with self.assertRaises(ValueError):
            table.append_locus("b", 0)
        with self.assertRaises(ValueError):
            PartitionTable([PartitionEntry("a", 1, 5), PartitionEntry("b", 7, 9)])
        with self.assertRaises(ValueError):
            PartitionEntry("x", 0, 3)
        with self.assertRaises(ValueError):
            PartitionEntry("x", 5, 4)
        with self.assertRaises(ValueError):
            PartitionEntry("x", 1, 3, codon_phase=3)
        with self.assertRaises(ValueError):
            PartitionEntry("x", 1, 3, columns=np.array([0, 1]))  # wrong length

    def test_entry_equality_hash_and_immutability(self):
        a = PartitionEntry("x", 1, 3, "cds", None, 0)
        b = PartitionEntry("x", 1, 3, "cds", np.array([0, 1, 2]), 0)
        c = PartitionEntry("x", 1, 3, "cds", np.array([0, 1, 4]), 0)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, "x")
        with self.assertRaises(ValueError):
            c.columns[0] = 9  # type: ignore[index]

    def test_empty_table(self):
        table = PartitionTable()
        self.assertEqual(table.nchar, 0)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(table.write_raxml(Path(tmp) / "p.txt"))
            self.assertIsNone(table.write_nexus_sets(Path(tmp) / "p.nex"))
            self.assertIsNone(table.write_partitionfinder(Path(tmp) / "p.cfg"))
            self.assertEqual(list(Path(tmp).iterdir()), [])


class LegacyByteCompatibilityTests(unittest.TestCase):
    """Without codon mode the output equals the legacy writers byte for byte."""

    def test_raxml_and_nexus_files_equal_legacy_writers(self):
        table = PartitionTable()
        table.append_locus("atpA", 500, "cds")
        table.append_locus("rbcL", 1428, "cds", codon_phase=0)  # codon off -> same
        table.append_locus("trnL-trnF", 77, "igs")
        legacy_parts = table.to_charpartitions()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for datatype in ("dna", "DNA", "nucl", "protein"):
                new = tmp_path / f"new_{datatype}.txt"
                old = tmp_path / f"old_{datatype}.txt"
                table.write_raxml(new, datatype=datatype)
                fast_write_raxml_partitions(old, legacy_parts, datatype=datatype)
                self.assertEqual(new.read_bytes(), old.read_bytes(), datatype)
            new_nex = tmp_path / "new.nex"
            old_nex = tmp_path / "old.nex"
            table.write_nexus_sets(new_nex)
            fast_write_nexus_partitions(old_nex, legacy_parts)
            self.assertEqual(new_nex.read_bytes(), old_nex.read_bytes())

    def test_exact_strings(self):
        table = two_locus_table()
        self.assertEqual(
            "".join(table.iter_raxml_lines()),
            "DNA, atpA = 1-500\nDNA, rbcL = 501-1928\n",
        )
        self.assertEqual(
            "".join(table.iter_raxml_lines("protein")),
            "AUTO, atpA = 1-500\nAUTO, rbcL = 501-1928\n",
        )
        self.assertEqual(
            "".join(table.iter_raxml_lines(model="GTR+G")),
            "GTR+G, atpA = 1-500\nGTR+G, rbcL = 501-1928\n",
        )
        self.assertEqual(
            "".join(table.iter_nexus_sets_lines()),
            "  CHARSET atpA = 1-500;\n"
            "  CHARSET rbcL = 501-1928;\n"
            "  CHARPARTITION combined = atpA:atpA, rbcL:rbcL;\n",
        )


class CodonPartitionTests(unittest.TestCase):
    def test_untrimmed_codon_aligned_locus(self):
        table = PartitionTable()
        table.append_locus("atpA", 12, "cds", codon_phase=0)
        table.append_locus("trnL", 5, "int")
        table.append_locus("rbcL", 9, "cds", codon_phase=0)
        self.assertEqual(
            "".join(table.iter_raxml_lines(codon=True)),
            "DNA, atpA_pos1 = 1-10\\3\n"
            "DNA, atpA_pos2 = 2-11\\3\n"
            "DNA, atpA_pos3 = 3-12\\3\n"
            "DNA, trnL = 13-17\n"
            "DNA, rbcL_pos1 = 18-24\\3\n"
            "DNA, rbcL_pos2 = 19-25\\3\n"
            "DNA, rbcL_pos3 = 20-26\\3\n",
        )
        # codon mode off ignores the phase information entirely
        self.assertEqual(
            "".join(table.iter_raxml_lines()),
            "DNA, atpA = 1-12\nDNA, trnL = 13-17\nDNA, rbcL = 18-26\n",
        )

    def test_trimming_the_ends_keeps_codon_positions_correct(self):
        # original columns 1..10 survive: 0 and 11 were trimmed away
        table = PartitionTable()
        table.append_locus("g", 10, "cds", columns=np.arange(1, 11), codon_phase=0)
        entry = table[0]
        self.assertEqual(
            entry.codon_positions().tolist(), [2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
        )
        self.assertEqual(
            "".join(table.iter_raxml_lines(codon=True)),
            "DNA, g_pos1 = 3-9\\3\nDNA, g_pos2 = 1-10\\3\nDNA, g_pos3 = 2-8\\3\n",
        )

    def test_interior_trimming_breaks_the_stride_into_runs(self):
        # original column 3 (codon position 1) was removed from a 13-column locus
        cols = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        table = PartitionTable()
        table.append_locus("atpA", 12, "cds", columns=cols, codon_phase=0)
        table.append_locus("trnL", 5, "int")
        self.assertEqual(
            "".join(table.iter_raxml_lines(codon=True)),
            "DNA, atpA_pos1 = 1-1, 6-12\\3\n"
            "DNA, atpA_pos2 = 2-2, 4-10\\3\n"
            "DNA, atpA_pos3 = 3-3, 5-11\\3\n"
            "DNA, trnL = 13-17\n",
        )
        self.assertEqual(
            "".join(table.iter_nexus_sets_lines(codon=True)),
            "  CHARSET atpA_pos1 = 1-1 6-12\\3;\n"
            "  CHARSET atpA_pos2 = 2-2 4-10\\3;\n"
            "  CHARSET atpA_pos3 = 3-3 5-11\\3;\n"
            "  CHARSET trnL = 13-17;\n"
            "  CHARPARTITION combined = atpA_pos1:atpA_pos1, atpA_pos2:atpA_pos2, "
            "atpA_pos3:atpA_pos3, trnL:trnL;\n",
        )

    def test_codon_phase_offset(self):
        # alignment starts at the second codon position
        table = PartitionTable()
        table.append_locus("g", 6, "cds", codon_phase=1)
        self.assertEqual(table[0].codon_positions().tolist(), [2, 3, 1, 2, 3, 1])
        self.assertEqual(
            "".join(table.iter_raxml_lines(codon=True)),
            "DNA, g_pos1 = 3-6\\3\nDNA, g_pos2 = 1-4\\3\nDNA, g_pos3 = 2-5\\3\n",
        )

    def test_missing_codon_positions_are_not_emitted(self):
        table = PartitionTable()
        table.append_locus("short", 2, "cds", codon_phase=0)
        self.assertEqual(
            "".join(table.iter_raxml_lines(codon=True)),
            "DNA, short_pos1 = 1-1\nDNA, short_pos2 = 2-2\n",
        )
        # only codon position 1 survives, but those columns are now adjacent
        t2 = PartitionTable()
        t2.append_locus("g", 4, "cds", columns=[0, 3, 6, 9], codon_phase=0)
        self.assertEqual(t2[0].codon_positions().tolist(), [1, 1, 1, 1])
        self.assertEqual(
            "".join(t2.iter_raxml_lines(codon=True)),
            "DNA, g_pos1 = 1-1, 2-2, 3-3, 4-4\n",
        )

    def test_non_codon_loci_unaffected_by_codon_flag(self):
        table = PartitionTable()
        table.append_locus("igs", 7, "igs")
        self.assertIsNone(table[0].codon_positions())
        self.assertEqual(table[0].codon_runs(), {})
        self.assertEqual(
            "".join(table.iter_raxml_lines(codon=True)), "DNA, igs = 1-7\n"
        )

    def test_original_columns_helper(self):
        entry = PartitionEntry("x", 4, 6, "cds", np.array([2, 5, 8]), 0)
        self.assertEqual(entry.original_columns().tolist(), [2, 5, 8])
        self.assertEqual(
            PartitionEntry("x", 4, 6).original_columns().tolist(), [0, 1, 2]
        )
        self.assertEqual(entry.codon_runs(), {3: [(4, 4), (5, 5), (6, 6)]})


class WriterFileTests(unittest.TestCase):
    def test_standalone_nexus_and_partitionfinder_files(self):
        table = PartitionTable()
        table.append_locus("atpA", 6, "cds", codon_phase=0)
        table.append_locus("trnL", 4, "int")
        with tempfile.TemporaryDirectory() as tmp:
            nex = Path(tmp) / "sub" / "p.nex"
            table.write_nexus_sets(nex, codon=True)
            self.assertEqual(
                nex.read_text(),
                "#NEXUS\n\nBEGIN SETS;\n"
                "  CHARSET atpA_pos1 = 1-4\\3;\n"
                "  CHARSET atpA_pos2 = 2-5\\3;\n"
                "  CHARSET atpA_pos3 = 3-6\\3;\n"
                "  CHARSET trnL = 7-10;\n"
                "  CHARPARTITION combined = atpA_pos1:atpA_pos1, atpA_pos2:atpA_pos2, "
                "atpA_pos3:atpA_pos3, trnL:trnL;\n"
                "END;\n",
            )
            cfg = Path(tmp) / "pf.cfg"
            table.write_partitionfinder(cfg, "super.phy", codon=True, models="all")
            self.assertEqual(
                cfg.read_text(),
                "## ALIGNMENT FILE ##\n"
                "alignment = super.phy;\n"
                "\n## BRANCHLENGTHS: linked | unlinked ##\n"
                "branchlengths = linked;\n"
                "\n## MODELS OF EVOLUTION ##\n"
                "models = all;\n"
                "model_selection = aicc;\n"
                "\n## DATA BLOCKS ##\n"
                "[data_blocks]\n"
                "atpA_pos1 = 1-4\\3;\n"
                "atpA_pos2 = 2-5\\3;\n"
                "atpA_pos3 = 3-6\\3;\n"
                "trnL = 7-10;\n"
                "\n## SCHEMES ##\n"
                "[schemes]\n"
                "search = greedy;\n",
            )
            raxml = Path(tmp) / "p.txt"
            table.write_raxml(raxml, codon=True)
            self.assertEqual(
                raxml.read_text(),
                "DNA, atpA_pos1 = 1-4\\3\nDNA, atpA_pos2 = 2-5\\3\n"
                "DNA, atpA_pos3 = 3-6\\3\nDNA, trnL = 7-10\n",
            )


if __name__ == "__main__":
    unittest.main()
