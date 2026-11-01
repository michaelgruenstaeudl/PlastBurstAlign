"""Indexed disk-backed storage engine for extracted sequence features.

Replaces in-memory accumulation of SeqRecord instances with indexed SQLite
storage,
enabling O(1) memory footprint during extraction, cleaning, and export.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class DiskBackedFeatureStore:
    """SQLite-backed sequence feature store with indexed query and streaming capabilities."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else ":memory:"
        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._in_transaction = False
        self._registered_regions: set[str] = set()
        self._init_db()
        self._local = threading.local()

    def _init_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute("PRAGMA cache_size = -131072;")  # 128 MB RAM page cache
        cursor.execute(
            "PRAGMA temp_store = MEMORY;"
        )  # Keep temp tables and indices in RAM
        cursor.execute(
            "PRAGMA mmap_size = 3221225472;"
        )  # 3 GB memory-mapped I/O direct access

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regions (
                region_name TEXT PRIMARY KEY,
                inv_name TEXT,
                insertion_order INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sequence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_name TEXT NOT NULL,
                taxon_name TEXT NOT NULL,
                seq_type TEXT NOT NULL,
                seq_id TEXT NOT NULL,
                description TEXT,
                sequence TEXT NOT NULL,
                seq_len INTEGER NOT NULL,
                file_index INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (region_name) REFERENCES regions(region_name) ON DELETE CASCADE,
                UNIQUE(region_name, taxon_name, seq_type)
            );
            """)
        self._migrate_schema(cursor)
        # `idx_records_order` also serves (region_name, seq_type) lookups, so the
        # former `idx_records_lookup` index is no longer created for new databases.
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_taxon ON"
            " sequence_records(taxon_name, region_name);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_order ON"
            " sequence_records(region_name, seq_type, file_index, id);"
        )
        self._conn.commit()

        # Pre-populate registered regions set from existing table
        cursor.execute("SELECT region_name FROM regions;")
        self._registered_regions = {row[0] for row in cursor.fetchall()}

    @staticmethod
    def _migrate_schema(cursor: sqlite3.Cursor) -> None:
        """Bring databases created by older versions up to the current schema.

        Version history:
          * `file_index` (source GenBank file position, default 0) was added so
            that record order is independent of extraction completion order.
        """
        cursor.execute("PRAGMA table_info(sequence_records);")
        columns = {row[1] for row in cursor.fetchall()}
        if "file_index" not in columns:
            cursor.execute(
                "ALTER TABLE sequence_records ADD COLUMN file_index INTEGER NOT NULL"
                " DEFAULT 0;"
            )

    @contextmanager
    def transaction(self) -> Iterator[Self]:
        """Context manager for batching multiple operations in a single atomic transaction."""
        if self._in_transaction:
            yield self
            return

        self._in_transaction = True
        try:
            yield self
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._in_transaction = False

    def register_region(self, region_name: str, inv_name: str | None = None) -> None:
        """Register a region maintaining first-seen insertion order with O(1) set caching."""
        if region_name in self._registered_regions:
            return

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO regions (region_name, inv_name, insertion_order, is_active)
            VALUES (
                :name,
                :inv,
                (SELECT COALESCE(MAX(insertion_order), 0) + 1 FROM regions),
                1
            );
            """,
            {"name": region_name, "inv": inv_name},
        )
        self._registered_regions.add(region_name)
        if not self._in_transaction:
            self._conn.commit()

    def add_record(
        self,
        region_name: str,
        taxon_name: str,
        seq_type: str,
        seq_id: str,
        description: str,
        sequence: str,
        inv_name: str | None = None,
        file_index: int = 0,
    ) -> bool:
        """Insert a sequence record for a region and taxon.

        Enforces uniqueness per (region, taxon, seq_type). Handles inverted repeat
        deduplication when `inv_name` is provided. `file_index` is the position of
        the source file in the ordered input file list; records are read back
        ordered by (file_index, id). Returns True if inserted, False if skipped.
        """
        self.register_region(region_name, inv_name=inv_name)

        if inv_name:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT 1 FROM sequence_records
                WHERE region_name = :inv_name AND taxon_name = :taxon_name AND seq_type = :seq_type
                LIMIT 1;
                """,
                {
                    "inv_name": inv_name,
                    "taxon_name": taxon_name,
                    "seq_type": seq_type,
                },
            )
            if cursor.fetchone() is not None:
                return False

        seq_len = len(sequence)
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO sequence_records (
                region_name, taxon_name, seq_type, seq_id, description, sequence,
                seq_len, file_index
            ) VALUES (
                :region, :taxon, :type, :id, :desc, :seq, :length, :file_index
            );
            """,
            {
                "region": region_name,
                "taxon": taxon_name,
                "type": seq_type,
                "id": seq_id,
                "desc": description,
                "seq": sequence,
                "length": seq_len,
                "file_index": int(file_index),
            },
        )
        inserted = cursor.rowcount > 0
        if not self._in_transaction:
            self._conn.commit()
        return inserted

    def add_pdict(
        self,
        pdict: Any,
        seq_type: str,
        inv_map: dict[str, str] | None = None,
        file_index: int = 0,
    ) -> None:
        """Batch-insert sequence records from an extracted dictionary into SQLite.

        Registers all regions upfront and bulk-inserts rows via executemany
        to eliminate per-record SQL compile and Python method overhead.

        Records are either `SeqRecord` instances or flat tuples
        `(taxon, description, sequence)` / `(taxon, description, sequence,
        file_index)`. The optional fourth tuple element is the position of the
        source file in the ordered input file list; when absent, the
        `file_index` argument is used.
        """
        if not pdict:
            return

        inv_map = inv_map or {}

        # 1. Register all unique regions upfront (O(num_regions) instead of O(num_records))
        for feat_name in pdict.keys():
            self.register_region(feat_name, inv_name=inv_map.get(feat_name))

        # 2. Vectorize rows into flat tuples
        rows: list[tuple[str, str, str, str, str, str, int, int]] = []
        cursor = self._conn.cursor()

        for feat_name, records in pdict.items():
            inv_name = inv_map.get(feat_name)
            for rec in records:
                rec_file_index = file_index
                if isinstance(rec, tuple):
                    taxon_name, desc, seq_str = rec[0], rec[1], rec[2]
                    if len(rec) > 3:
                        rec_file_index = int(rec[3])
                    seq_id = taxon_name
                else:
                    taxon_name = str(rec.id)
                    seq_id = str(rec.id) if rec.id is not None else ""
                    desc = rec.description or ""
                    seq_str = str(rec.seq)

                if inv_name:
                    cursor.execute(
                        """
                        SELECT 1 FROM sequence_records
                        WHERE region_name = ? AND taxon_name = ? AND seq_type = ?
                        LIMIT 1;
                        """,
                        (inv_name, taxon_name, seq_type),
                    )
                    if cursor.fetchone() is not None:
                        continue

                rows.append(
                    (
                        feat_name,
                        taxon_name,
                        seq_type,
                        seq_id,
                        desc,
                        seq_str,
                        len(seq_str),
                        rec_file_index,
                    )
                )

        # 3. High-throughput bulk insert
        if rows:
            cursor.executemany(
                """
                INSERT OR IGNORE INTO sequence_records (
                    region_name, taxon_name, seq_type, seq_id, description, sequence,
                    seq_len, file_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            if not self._in_transaction:
                self._conn.commit()

    def add_seqrecord(
        self,
        region_name: str,
        taxon_name: str,
        seq_type: str,
        record: SeqRecord,
        inv_name: str | None = None,
        file_index: int = 0,
    ) -> bool:
        """Add a Biopython SeqRecord into the database."""
        return self.add_record(
            region_name=region_name,
            taxon_name=taxon_name,
            seq_type=seq_type,
            seq_id=str(record.id) if record.id is not None else "",
            description=record.description or "",
            sequence=str(record.seq),
            inv_name=inv_name,
            file_index=file_index,
        )

    def has_region(self, region_name: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM regions WHERE region_name = :name AND is_active = 1"
            " LIMIT 1;",
            {"name": region_name},
        )
        return cursor.fetchone() is not None

    def get_active_regions(self, order: str = "seq") -> list[str]:
        """Return list of active region names ordered by insertion ('seq') or alphabetical ('alpha')."""
        order_clause = "insertion_order ASC" if order == "seq" else "region_name ASC"
        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT region_name FROM regions WHERE is_active = 1 ORDER BY"
            f" {order_clause};"
        )
        return [row[0] for row in cursor.fetchall()]

    def remove_region(self, region_name: str) -> None:
        """Mark a region as inactive."""
        self._conn.execute(
            "UPDATE regions SET is_active = 0 WHERE region_name = :name;",
            {"name": region_name},
        )
        if not self._in_transaction:
            self._conn.commit()

    def get_records(self, region_name: str, seq_type: str = "nucl") -> list[SeqRecord]:
        """Hydrate SeqRecord instances on demand for a single region."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT seq_id, description, sequence
            FROM sequence_records
            WHERE region_name = :region AND seq_type = :type
            ORDER BY file_index ASC, id ASC;
            """,
            {"region": region_name, "type": seq_type},
        )
        return [
            SeqRecord(
                Seq(row["sequence"]),
                id=row["seq_id"],
                description=row["description"] or "",
            )
            for row in cursor.fetchall()
        ]

    def get_region_summary(self, seq_type: str = "nucl") -> dict[str, dict[str, int]]:
        """Return aggregated summary per active region.

        Keys per region: `count` (number of taxa), `max_len` (longest stored
        sequence) and `min_len` (shortest sequence, not counting one trailing
        stop `*`, which aligners do not treat as a residue). Regions without
        records of `seq_type` report zeros.
        """
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT r.region_name,
                   COUNT(s.id) AS taxon_count,
                   COALESCE(MAX(s.seq_len), 0) AS max_len,
                   COALESCE(MIN(
                       s.seq_len - (CASE WHEN substr(s.sequence, -1, 1) = '*'
                                         THEN 1 ELSE 0 END)
                   ), 0) AS min_len
            FROM regions r
            LEFT JOIN sequence_records s
              ON r.region_name = s.region_name AND s.seq_type = :type
            WHERE r.is_active = 1
            GROUP BY r.region_name;
            """,
            {"type": seq_type},
        )
        return {
            row["region_name"]: {
                "count": row["taxon_count"],
                "max_len": row["max_len"],
                "min_len": row["min_len"],
            }
            for row in cursor.fetchall()
        }

    def _get_read_conn(self) -> sqlite3.Connection:
        """Thread-safe connection accessor for parallel reads under WAL mode."""
        if self.db_path == ":memory:":
            return self._conn
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def export_fasta(
        self, region_name: str, seq_type: str, out_path: str | Path
    ) -> int:
        """Stream FASTA file straight from disk cursor without hydrating SeqRecord instances."""
        conn = self._get_read_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT seq_id, description, sequence
            FROM sequence_records
            WHERE region_name = :region AND seq_type = :type
            ORDER BY file_index ASC, id ASC;
            """,
            {"region": region_name, "type": seq_type},
        )

        count = 0
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", buffering=1024 * 1024) as hndl:
            for row in cursor:
                count += 1
                desc = row["description"]
                header = (
                    desc
                    if desc and desc.startswith(row["seq_id"])
                    else f"{row['seq_id']} {desc}".strip()
                )
                hndl.write(f">{header}\n{row['sequence']}\n")
        cursor.close()
        return count

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class FeatureStoreView:
    """Dict-compatible view over DiskBackedFeatureStore for a specific sequence type ('nucl' or 'prot')."""

    def __init__(self, store: DiskBackedFeatureStore, seq_type: str) -> None:
        self._store = store
        self._seq_type = seq_type

    def __iter__(self) -> Iterator[str]:
        return iter(self._store.get_active_regions())

    def __len__(self) -> int:
        return len(self._store.get_active_regions())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._store.has_region(key)

    def __getitem__(self, region_name: str) -> list[SeqRecord]:
        if not self._store.has_region(region_name):
            raise KeyError(region_name)
        return self._store.get_records(region_name, self._seq_type)

    def __delitem__(self, region_name: str) -> None:
        self._store.remove_region(region_name)

    def get(self, region_name: str, default: Any = None) -> Any:
        if not self._store.has_region(region_name):
            return default
        return self._store.get_records(region_name, self._seq_type)

    def keys(self) -> list[str]:
        return self._store.get_active_regions()

    def items(self) -> list[tuple[str, list[SeqRecord]]]:
        return [(k, self[k]) for k in self.keys()]

    def values(self) -> list[list[SeqRecord]]:
        return [self[k] for k in self.keys()]

    def add_feature(self, feature: Any) -> None:
        """Compatibility method for tests and direct extraction injection."""
        rec = feature.get_record()
        self._store.add_seqrecord(
            region_name=feature.feat_name,
            taxon_name=str(rec.id) if rec.id else "",
            seq_type=self._seq_type,
            record=rec,
            inv_name=getattr(feature, "inv_feat_name", None),
        )
