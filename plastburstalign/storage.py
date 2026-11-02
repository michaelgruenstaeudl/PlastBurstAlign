"""Indexed disk-backed storage engine for extracted sequence features.

Replaces in-memory accumulation of SeqRecord instances with indexed SQLite storage,
enabling O(1) memory footprint during extraction, cleaning, and export.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
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
        self._init_db()

    def _init_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS regions (
                region_name TEXT PRIMARY KEY,
                inv_name TEXT,
                insertion_order INTEGER NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sequence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region_name TEXT NOT NULL,
                taxon_name TEXT NOT NULL,
                seq_type TEXT NOT NULL,
                seq_id TEXT NOT NULL,
                description TEXT,
                sequence TEXT NOT NULL,
                seq_len INTEGER NOT NULL,
                FOREIGN KEY (region_name) REFERENCES regions(region_name) ON DELETE CASCADE,
                UNIQUE(region_name, taxon_name, seq_type)
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_lookup ON sequence_records(region_name, seq_type);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_taxon ON sequence_records(taxon_name, region_name);"
        )
        self._conn.commit()

    def register_region(self, region_name: str, inv_name: str | None = None) -> None:
        """Register a region maintaining first-seen insertion order."""
        with self._conn:
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

    def add_record(
        self,
        region_name: str,
        taxon_name: str,
        seq_type: str,
        seq_id: str,
        description: str,
        sequence: str,
        inv_name: str | None = None,
    ) -> bool:
        """Insert a sequence record for a region and taxon.

        Enforces uniqueness per (region, taxon, seq_type). Handles inverted repeat
        deduplication when `inv_name` is provided. Returns True if inserted, False if skipped.
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
                {"inv_name": inv_name, "taxon_name": taxon_name, "seq_type": seq_type},
            )
            if cursor.fetchone() is not None:
                return False

        seq_len = len(sequence)
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sequence_records (
                        region_name, taxon_name, seq_type, seq_id, description, sequence, seq_len
                    ) VALUES (
                        :region, :taxon, :type, :id, :desc, :seq, :length
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
                    },
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def add_seqrecord(
        self,
        region_name: str,
        taxon_name: str,
        seq_type: str,
        record: SeqRecord,
        inv_name: str | None = None,
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
        )

    def has_region(self, region_name: str) -> bool:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT 1 FROM regions WHERE region_name = :name AND is_active = 1 LIMIT 1;",
            {"name": region_name},
        )
        return cursor.fetchone() is not None

    def get_active_regions(self, order: str = "seq") -> list[str]:
        """Return list of active region names ordered by insertion ('seq') or alphabetical ('alpha')."""
        order_clause = "insertion_order ASC" if order == "seq" else "region_name ASC"
        cursor = self._conn.cursor()
        cursor.execute(
            f"SELECT region_name FROM regions WHERE is_active = 1 ORDER BY {order_clause};"
        )
        return [row[0] for row in cursor.fetchall()]

    def remove_region(self, region_name: str) -> None:
        """Mark a region as inactive."""
        with self._conn:
            self._conn.execute(
                "UPDATE regions SET is_active = 0 WHERE region_name = :name;",
                {"name": region_name},
            )

    def get_records(self, region_name: str, seq_type: str = "nucl") -> list[SeqRecord]:
        """Retrieve hydrated SeqRecord list for a region."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT seq_id, description, sequence
            FROM sequence_records
            WHERE region_name = :region AND seq_type = :type
            ORDER BY id ASC;
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
        """Return aggregated summary per active region: taxon_count and max_seq_len."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT r.region_name,
                   COUNT(s.id) AS taxon_count,
                   COALESCE(MAX(s.seq_len), 0) AS max_len
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
            }
            for row in cursor.fetchall()
        }

    def export_fasta(self, region_name: str, seq_type: str, out_path: str | Path) -> int:
        """Stream FASTA file straight from disk cursor without hydrating SeqRecord instances."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT seq_id, description, sequence
            FROM sequence_records
            WHERE region_name = :region AND seq_type = :type
            ORDER BY id ASC;
            """,
            {"region": region_name, "type": seq_type},
        )

        count = 0
        with open(out_path, "w") as hndl:
            for row in cursor:
                count += 1
                desc = row["description"]
                header = (
                    desc
                    if desc and desc.startswith(row["seq_id"])
                    else f"{row['seq_id']} {desc}".strip()
                )
                hndl.write(f">{header}\n{row['sequence']}\n")
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
