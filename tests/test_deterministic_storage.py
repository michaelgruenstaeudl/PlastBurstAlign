"""Regression tests for deterministic record order and the feature-store schema.

Covers: `file_index` column + migration, `ORDER BY file_index, id`, 3/4-tuple
`add_pdict`, `min_len` in the region summary, `ordered_bounded_map`, sorted
`PlastidData.files`, and extraction that is independent of worker completion
order and of `num_threads`.
"""

from __future__ import annotations

import os
import random
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.extraction_ops import ExtractAndCollect
from plastburstalign.helpers import ordered_bounded_map
from plastburstalign.seqfeature_ops import PlastidData
from plastburstalign.storage import DiskBackedFeatureStore
from plastburstalign.task_queue import get_mp_context
from plastburstalign.user_parameters import UserParameters


class FileIndexSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "store.db"

    def _columns(self, path: Path) -> list[str]:
        conn = sqlite3.connect(path)
        try:
            return [r[1] for r in conn.execute("PRAGMA table_info(sequence_records)")]
        finally:
            conn.close()

    def test_new_database_has_file_index_column_and_order_index(self) -> None:
        with DiskBackedFeatureStore(self.db_path):
            pass
        self.assertIn("file_index", self._columns(self.db_path))
        conn = sqlite3.connect(self.db_path)
        try:
            names = {r[1] for r in conn.execute("PRAGMA index_list(sequence_records)")}
        finally:
            conn.close()
        self.assertIn("idx_records_order", names)

    def test_legacy_database_is_migrated_in_place(self) -> None:
        # Schema of a pre-file_index database, with data in it.
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE regions (
                region_name TEXT PRIMARY KEY,
                inv_name TEXT,
                insertion_order INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE sequence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_name TEXT NOT NULL,
                taxon_name TEXT NOT NULL,
                seq_type TEXT NOT NULL,
                seq_id TEXT NOT NULL,
                description TEXT,
                sequence TEXT NOT NULL,
                seq_len INTEGER NOT NULL,
                FOREIGN KEY (region_name) REFERENCES regions(region_name)
                    ON DELETE CASCADE,
                UNIQUE(region_name, taxon_name, seq_type)
            );
            INSERT INTO regions VALUES ('geneA', NULL, 1, 1);
            INSERT INTO sequence_records
                (region_name, taxon_name, seq_type, seq_id, description, sequence,
                 seq_len)
                VALUES ('geneA', 'tB', 'nucl', 'tB', '', 'ACGT', 4),
                       ('geneA', 'tA', 'nucl', 'tA', '', 'ACGA', 4);
            """
        )
        conn.commit()
        conn.close()
        self.assertNotIn("file_index", self._columns(self.db_path))

        with DiskBackedFeatureStore(self.db_path) as store:
            # Legacy rows keep their data and default to file_index 0 => id order.
            self.assertEqual([r.id for r in store.get_records("geneA")], ["tB", "tA"])
            store.add_record("geneA", "tC", "nucl", "tC", "", "AAAA", file_index=3)
        self.assertIn("file_index", self._columns(self.db_path))

        # Re-opening an already migrated database must be a no-op.
        with DiskBackedFeatureStore(self.db_path) as store:
            self.assertEqual(
                [r.id for r in store.get_records("geneA")], ["tB", "tA", "tC"]
            )
            conn = store._conn
            values = [
                tuple(r)
                for r in conn.execute(
                    "SELECT taxon_name, file_index FROM sequence_records ORDER BY id"
                )
            ]
        self.assertEqual(values, [("tB", 0), ("tA", 0), ("tC", 3)])


class OrderingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = DiskBackedFeatureStore(Path(self.tmp.name) / "store.db")
        self.addCleanup(self.store.close)

    def test_get_records_and_export_order_by_file_index_then_id(self) -> None:
        # Inserted in "completion order" (3, 1, 2, 1) but must read back by file.
        self.store.add_record("g", "t3", "nucl", "t3", "", "AAAA", file_index=3)
        self.store.add_record("g", "t1a", "nucl", "t1a", "", "CCCC", file_index=1)
        self.store.add_record("g", "t2", "nucl", "t2", "", "GGGG", file_index=2)
        self.store.add_record("g", "t1b", "nucl", "t1b", "", "TTTT", file_index=1)
        expected = ["t1a", "t1b", "t2", "t3"]
        self.assertEqual([r.id for r in self.store.get_records("g")], expected)
        out = Path(self.tmp.name) / "g.fasta"
        self.assertEqual(self.store.export_fasta("g", "nucl", out), 4)
        self.assertEqual([r.id for r in SeqIO.parse(out, "fasta")], expected)

    def test_add_pdict_accepts_three_and_four_tuples_and_seqrecords(self) -> None:
        pdict = {
            "g": [
                ("tA", "descA", "ACGT"),  # legacy 3-tuple -> default file_index
                ("tB", "descB", "ACGA", 5),  # 4-tuple carries the file index
                SeqRecord(Seq("ACGG"), id="tC", description=""),
            ]
        }
        self.store.add_pdict(pdict, "nucl", file_index=2)
        rows = [
            tuple(r)
            for r in self.store._conn.execute(
                "SELECT taxon_name, file_index FROM sequence_records ORDER BY id"
            )
        ]
        self.assertEqual(rows, [("tA", 2), ("tB", 5), ("tC", 2)])
        self.assertEqual(
            [r.id for r in self.store.get_records("g")], ["tA", "tC", "tB"]
        )

    def test_region_summary_has_min_len_and_keeps_existing_keys(self) -> None:
        self.store.add_record("g", "t1", "nucl", "t1", "", "ACGTACGT")
        self.store.add_record("g", "t2", "nucl", "t2", "", "ACG")
        self.store.add_record("p", "t1", "prot", "t1", "", "MKVL*")
        self.store.add_record("p", "t2", "prot", "t2", "", "MKVLAAA")
        self.store.register_region("empty")
        nucl = self.store.get_region_summary("nucl")
        self.assertEqual(nucl["g"], {"count": 2, "max_len": 8, "min_len": 3})
        # A region without records of this type reports zeros.
        self.assertEqual(nucl["empty"], {"count": 0, "max_len": 0, "min_len": 0})
        prot = self.store.get_region_summary("prot")
        # `max_len` keeps counting the stop; `min_len` ignores one trailing '*'
        # (the shortest protein, MKVL*, is four residues).
        self.assertEqual(prot["p"], {"count": 2, "max_len": 7, "min_len": 4})


class OrderedBoundedMapTests(unittest.TestCase):
    def test_results_are_yielded_in_submission_order(self) -> None:
        delays = [0.15, 0.05, 0.0, 0.1, 0.0]

        def work(i: int) -> int:
            time.sleep(delays[i])
            return i * 10

        with ThreadPoolExecutor(max_workers=4) as pool:
            out = [
                (item, fut.result())
                for item, fut in ordered_bounded_map(pool, work, range(5), window=4)
            ]
        self.assertEqual(out, [(i, i * 10) for i in range(5)])

    def test_window_bounds_outstanding_tasks(self) -> None:
        lock = threading.Lock()
        started: list[int] = []

        def work(i: int) -> int:
            with lock:
                started.append(i)
            return i

        with ThreadPoolExecutor(max_workers=8) as pool:
            gen = ordered_bounded_map(pool, work, range(20), window=3)
            next(gen)  # first result consumed -> at most 3 submitted so far (+1 refill)
            time.sleep(0.2)
            with lock:
                self.assertLessEqual(len(started), 4)
            self.assertEqual([item for item, _ in gen], list(range(1, 20)))

    def test_exception_surfaces_at_its_position_and_empty_input(self) -> None:
        def work(i: int) -> int:
            if i == 2:
                raise ValueError("boom")
            return i

        with ThreadPoolExecutor(max_workers=2) as pool:
            seen = []
            with self.assertRaises(ValueError):
                for item, fut in ordered_bounded_map(pool, work, range(5), window=2):
                    seen.append(item)
                    fut.result()
            self.assertEqual(seen, [0, 1, 2])
            self.assertEqual(list(ordered_bounded_map(pool, work, [], window=2)), [])


class SortedFilesTests(unittest.TestCase):
    def test_plastid_data_files_are_sorted_regardless_of_listdir_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            names = ["c.gb", "a.gb", "b.gb", "notes.txt", "d.gb"]
            for name in names:
                (in_dir / name).write_text("x")
            params = UserParameters(in_dir=str(in_dir), out_dir=str(Path(tmp) / "out"))
            real_listdir = os.listdir
            for order in (list(names), list(reversed(names))):
                with patch(
                    "os.listdir",
                    side_effect=lambda p, o=order: (
                        o if str(p) == str(in_dir) else real_listdir(p)
                    ),
                ):
                    data = PlastidData(params)
                    self.assertEqual(
                        [os.path.basename(f) for f in data.files],
                        ["a.gb", "b.gb", "c.gb", "d.gb"],
                    )
                    data.store.close()

    def test_missing_input_dir_is_tolerated_only_for_cluster_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = str(Path(tmp) / "out")
            previous_cwd = os.getcwd()
            os.chdir(tmp)  # like a Slurm worker run from the output dir: no ./input
            self.addCleanup(os.chdir, previous_cwd)
            # Workers and the finalizer only need the feature store: the default
            # `./input` need not exist, neither for UserParameters nor PlastidData.
            for flags in ({"align_partition": 0}, {"finalize_slurm": True}):
                with self.subTest(flags=flags):
                    params = UserParameters(out_dir=out_dir, **flags)
                    data = PlastidData(params)
                    self.assertEqual(data.files, [])
                    data.store.close()
            # A regular run must still fail loudly.
            with self.assertRaises(FileNotFoundError):
                UserParameters(out_dir=out_dir)
            regular = UserParameters(in_dir=tmp, out_dir=out_dir)
            regular["in_dir"] = str(Path(tmp) / "does-not-exist")
            with self.assertRaises(OSError):
                PlastidData(regular)


def _make_genome(name: str, index: int, rng: random.Random) -> SeqRecord:
    """Small synthetic genome: three shared tRNAs plus one tRNA unique to this genome.

    The unique tRNA creates a region that is first seen in this file only, so the
    region registration order (used for concatenation order) depends on the order
    in which extraction results are committed.
    """
    gene_names = ["trnH-GUG", "trnK-GUG", "trnL-GUG", f"trnX{index:02d}"]
    pieces: list[str] = []
    features: list[SeqFeature] = []
    pos = 0
    for gene in gene_names:
        spacer = "".join(rng.choice("ACGT") for _ in range(rng.randint(5, 12)))
        pieces.append(spacer)
        pos += len(spacer)
        body = "".join(rng.choice("ACGT") for _ in range(rng.randint(30, 50)))
        pieces.append(body)
        features.append(
            SeqFeature(
                FeatureLocation(pos, pos + len(body), strand=1),
                type="tRNA",
                qualifiers={"gene": [gene]},
            )
        )
        pos += len(body)
    record = SeqRecord(
        Seq("".join(pieces)),
        id=name,
        name=name,
        description=f"synthetic {name}",
        annotations={"molecule_type": "DNA"},
    )
    record.features = features
    return record


class ExtractionDeterminismTests(unittest.TestCase):
    NUM_FILES = 11

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.in_dir = self.root / "in"
        self.in_dir.mkdir()
        rng = random.Random(42)
        # Names are deliberately not in creation order.
        order = list(range(self.NUM_FILES))
        rng.shuffle(order)
        for i in order:
            SeqIO.write(
                _make_genome(f"TAX{i:02d}", i, rng),
                self.in_dir / f"TAX{i:02d}.gb",
                "genbank",
            )

    def _extract(self, label: str, num_threads: int, order: str = "seq") -> list:
        out_dir = self.root / f"out_{label}"
        params = UserParameters(
            in_dir=str(self.in_dir),
            out_dir=str(out_dir),
            select_mode="rna",
            num_threads=num_threads,
            min_seq_length=1,
            order=order,
        )
        data = PlastidData(params)
        try:
            ExtractAndCollect(data, params).extract()
            rows = [
                tuple(r)
                for r in data.store._conn.execute(
                    "SELECT region_name, seq_id, file_index, seq_type FROM"
                    " sequence_records ORDER BY region_name, file_index, id"
                )
            ]
            regions = data.store.get_active_regions("seq")
            read_back = {
                region: [r.id for r in data.nucleotides[region]] for region in regions
            }
            return [rows, regions, read_back]
        finally:
            data.store.close()

    def test_record_order_is_file_order(self) -> None:
        rows, regions, read_back = self._extract("plain", 2)
        # Regions are registered in sorted-file order: shared ones first, then
        # the tRNA unique to file 0, file 1, ...
        self.assertEqual(
            regions,
            ["trnH_GUG", "trnK_GUG", "trnL_GUG"]
            + [f"trnX{i:02d}" for i in range(self.NUM_FILES)],
        )
        for region in ("trnH_GUG", "trnK_GUG", "trnL_GUG"):
            self.assertEqual(
                [i.rsplit("_", 1)[-1] for i in read_back[region]],
                [f"TAX{i:02d}" for i in range(self.NUM_FILES)],
                f"records of {region} are not in sorted file order",
            )
        # file_index equals the position of the file in the sorted file list.
        for _region, seq_id, file_index, _type in rows:
            self.assertEqual(int(seq_id.rsplit("TAX", 1)[1]), file_index)

    def test_identical_across_runs_and_thread_counts(self) -> None:
        baseline = self._extract("n1", 1)
        for label, threads in (("n2", 2), ("n2b", 2), ("n3", 3), ("n5", 5)):
            with self.subTest(num_threads=threads, run=label):
                self.assertEqual(self._extract(label, threads), baseline)

    def test_alpha_order_has_same_record_order(self) -> None:
        seq_rows = self._extract("seq", 2)[0]
        alpha_rows = self._extract("alpha", 2, order="alpha")[0]
        self.assertEqual(seq_rows, alpha_rows)

    @unittest.skipUnless(
        get_mp_context().get_start_method() == "fork",
        "completion-order stress needs fork so the patched worker is inherited",
    )
    def test_commit_order_ignores_completion_order(self) -> None:
        """Make early chunks finish *last*; the store order must not change."""
        baseline = self._extract("baseline", 3)
        original = ExtractAndCollect._extract_recs_static

        def slow_for_early_chunks(files, *args):
            # args[-1] is `start_index`: the earlier the chunk, the longer it sleeps
            start_index = args[-1]
            time.sleep(max(0.0, 0.6 - 0.12 * start_index))
            return original(files, *args)

        with patch.object(
            ExtractAndCollect,
            "_extract_recs_static",
            staticmethod(slow_for_early_chunks),
        ):
            stressed = self._extract("stressed", 3)
        self.assertEqual(stressed, baseline)


if __name__ == "__main__":
    unittest.main()
