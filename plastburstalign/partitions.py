"""Partition tables and writers for concatenated (super-)matrices.

Coordinates are **1-based and inclusive** and refer to the final super-matrix,
i.e. they are computed *after* column trimming. Without codon splitting the
RAxML-NG/IQ-TREE and NEXUS outputs are byte-compatible with the legacy
:func:`~plastburstalign.nexus_ops.fast_write_raxml_partitions` and
:func:`~plastburstalign.nexus_ops.fast_write_nexus_partitions`.

Codon partitions
----------------
A coding locus whose alignment is codon-aligned (e.g. a protein alignment
back-translated to nucleotides) carries ``codon_phase`` = the codon position
(0, 1 or 2) of its *original* in-locus column 0, and ``columns`` = the
original in-locus indices of the kept columns. The codon position of kept
column ``i`` is therefore ``(columns[i] + codon_phase) % 3 + 1`` regardless of
which columns trimming removed. Each codon position becomes a partition
``<name>_pos1`` .. ``<name>_pos3``; its columns are written as stride-3 runs
``a-b\\3`` (a single run when the kept columns are contiguous, otherwise a
comma-separated list of maximal stride-3 runs, single sites as ``a-a``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "CODON_STRIDE",
    "PartitionEntry",
    "PartitionTable",
    "stride_runs",
]

IntArray: TypeAlias = npt.NDArray[np.int64]
Run: TypeAlias = tuple[int, int, int]

CODON_STRIDE = 3


def stride_runs(positions: npt.ArrayLike, stride: int = 1) -> list[tuple[int, int]]:
    """Split ascending ``positions`` into maximal arithmetic runs of step ``stride``.

    Returns:
        ``(first, last)`` pairs; single sites are ``(p, p)``.
    """
    pos = np.asarray(positions, dtype=np.int64).ravel()
    if pos.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(pos) != stride) + 1
    firsts = np.concatenate(([0], breaks))
    lasts = np.concatenate((breaks, [pos.size])) - 1
    return [(int(pos[a]), int(pos[b])) for a, b in zip(firsts, lasts, strict=True)]


def _format_run(first: int, last: int, stride: int) -> str:
    if stride == 1 or first == last:
        return f"{first}-{last}"
    return f"{first}-{last}\\{stride}"


@dataclass(frozen=True, eq=False)
class PartitionEntry:
    """One locus (or codon-position) block of the super-matrix.

    Attributes:
        name: partition name (locus name).
        start: first super-matrix column (1-based).
        end: last super-matrix column (1-based, inclusive).
        kind: locus kind, e.g. ``"cds"``, ``"igs"``, ``"int"``, ``"rna"``.
        columns: original (pre-trimming) 0-based in-locus index of every kept
            column, aligned with super-matrix columns ``start..end``; ``None``
            means nothing was trimmed (identity).
        codon_phase: codon position (0, 1 or 2) of original column 0, or
            ``None`` if the locus is not codon-aligned (no codon splitting).
    """

    name: str
    start: int
    end: int
    kind: str = "nucl"
    columns: IntArray | None = None
    codon_phase: int | None = None

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError(
                f"invalid partition range {self.start}-{self.end} for {self.name!r}"
            )
        if self.codon_phase not in (None, 0, 1, 2):
            raise ValueError(f"codon_phase must be 0, 1, 2 or None: {self.codon_phase}")
        if self.columns is not None:
            cols = np.asarray(self.columns, dtype=np.int64)
            if cols.shape != (self.width,):
                raise ValueError(
                    f"{self.name!r}: {cols.shape[0]} column indices for width "
                    f"{self.width}"
                )
            cols.flags.writeable = False
            object.__setattr__(self, "columns", cols)

    @property
    def width(self) -> int:
        """Number of super-matrix columns covered."""
        return self.end - self.start + 1

    def original_columns(self) -> IntArray:
        """Original in-locus column index of every column of the block."""
        if self.columns is None:
            return np.arange(self.width, dtype=np.int64)
        return self.columns

    def codon_positions(self) -> IntArray | None:
        """Codon position (1, 2 or 3) of every column, or ``None`` if not coding."""
        if self.codon_phase is None:
            return None
        return (self.original_columns() + self.codon_phase) % 3 + 1

    def codon_runs(self) -> dict[int, list[tuple[int, int]]]:
        """Stride-3 runs of super-matrix columns for each present codon position."""
        positions = self.codon_positions()
        if positions is None:
            return {}
        runs: dict[int, list[tuple[int, int]]] = {}
        for codon_pos in (1, 2, 3):
            idx = np.flatnonzero(positions == codon_pos)
            if idx.size:
                runs[codon_pos] = stride_runs(self.start + idx, CODON_STRIDE)
        return runs

    def _scalar_key(self) -> tuple[object, ...]:
        return (self.name, self.start, self.end, self.kind, self.codon_phase)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PartitionEntry):
            return NotImplemented
        return self._scalar_key() == other._scalar_key() and bool(
            np.array_equal(self.original_columns(), other.original_columns())
        )

    def __hash__(self) -> int:
        return hash(self._scalar_key())


class PartitionTable:
    """Ordered, contiguous, non-overlapping partitions of a super-matrix."""

    __slots__ = ("_entries",)

    def __init__(self, entries: Iterable[PartitionEntry] = ()) -> None:
        self._entries: list[PartitionEntry] = []
        for entry in entries:
            self._append(entry)

    def _append(self, entry: PartitionEntry) -> None:
        expected = self._entries[-1].end + 1 if self._entries else None
        if expected is not None and entry.start != expected:
            raise ValueError(
                f"partition {entry.name!r} starts at {entry.start}, expected {expected}"
            )
        if any(e.name == entry.name for e in self._entries):
            raise ValueError(f"duplicate partition name {entry.name!r}")
        self._entries.append(entry)

    def append_locus(
        self,
        name: str,
        width: int,
        kind: str = "nucl",
        columns: npt.ArrayLike | None = None,
        codon_phase: int | None = None,
    ) -> PartitionEntry:
        """Append a block of ``width`` columns directly after the last one."""
        if width < 1:
            raise ValueError(f"partition {name!r} must span at least one column")
        start = self._entries[-1].end + 1 if self._entries else 1
        cols = None if columns is None else np.asarray(columns, dtype=np.int64)
        entry = PartitionEntry(name, start, start + width - 1, kind, cols, codon_phase)
        self._append(entry)
        return entry

    # container protocol ------------------------------------------------
    @property
    def entries(self) -> tuple[PartitionEntry, ...]:
        """All entries in matrix order."""
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[PartitionEntry]:
        return iter(self._entries)

    def __getitem__(self, key: int | str) -> PartitionEntry:
        if isinstance(key, int):
            return self._entries[key]
        for entry in self._entries:
            if entry.name == key:
                return entry
        raise KeyError(key)

    def __repr__(self) -> str:
        return f"PartitionTable({len(self)} partitions, nchar={self.nchar})"

    @property
    def names(self) -> tuple[str, ...]:
        """Partition names in matrix order."""
        return tuple(e.name for e in self._entries)

    @property
    def nchar(self) -> int:
        """Total number of covered columns."""
        return self._entries[-1].end if self._entries else 0

    def to_charpartitions(self) -> dict[str, dict[str, list[int]]]:
        """Legacy ``{"combined": {name: [0-based column, ...]}}`` structure."""
        return {
            "combined": {e.name: list(range(e.start - 1, e.end)) for e in self._entries}
        }

    # rendering ---------------------------------------------------------
    def _blocks(self, codon: bool) -> Iterator[tuple[str, list[Run]]]:
        """Yield ``(label, runs)``; runs are ``(first, last, stride)``."""
        for entry in self._entries:
            if codon and entry.codon_phase is not None:
                for codon_pos, runs in entry.codon_runs().items():
                    yield (
                        f"{entry.name}_pos{codon_pos}",
                        [(a, b, CODON_STRIDE) for a, b in runs],
                    )
            else:
                yield entry.name, [(entry.start, entry.end, 1)]

    @staticmethod
    def _model_for(datatype: str, model: str | None) -> str:
        if model is not None:
            return model
        return "DNA" if str(datatype).lower() in ("dna", "nucl") else "AUTO"

    def iter_raxml_lines(
        self, datatype: str = "dna", codon: bool = False, model: str | None = None
    ) -> Iterator[str]:
        """RAxML-NG / IQ-TREE lines ``DNA, name = a-b`` (newline terminated)."""
        prefix = self._model_for(datatype, model)
        for label, runs in self._blocks(codon):
            spec = ", ".join(_format_run(a, b, s) for a, b, s in runs)
            yield f"{prefix}, {label} = {spec}\n"

    def iter_nexus_sets_lines(self, codon: bool = False) -> Iterator[str]:
        """Body of a NEXUS ``SETS`` block: ``CHARSET`` lines and ``CHARPARTITION``."""
        labels: list[str] = []
        for label, runs in self._blocks(codon):
            spec = " ".join(_format_run(a, b, s) for a, b, s in runs)
            labels.append(label)
            yield f"  CHARSET {label} = {spec};\n"
        part_list = ", ".join(f"{label}:{label}" for label in labels)
        yield f"  CHARPARTITION combined = {part_list};\n"

    def iter_partitionfinder_lines(
        self,
        alignment: str = "alignment.phy",
        codon: bool = False,
        models: str = "GTR+G",
        search: str = "greedy",
        branchlengths: str = "linked",
        model_selection: str = "aicc",
    ) -> Iterator[str]:
        """PartitionFinder 2 style ``.cfg`` file lines (newline terminated)."""
        yield "## ALIGNMENT FILE ##\n"
        yield f"alignment = {alignment};\n"
        yield "\n## BRANCHLENGTHS: linked | unlinked ##\n"
        yield f"branchlengths = {branchlengths};\n"
        yield "\n## MODELS OF EVOLUTION ##\n"
        yield f"models = {models};\n"
        yield f"model_selection = {model_selection};\n"
        yield "\n## DATA BLOCKS ##\n"
        yield "[data_blocks]\n"
        for label, runs in self._blocks(codon):
            spec = ", ".join(_format_run(a, b, s) for a, b, s in runs)
            yield f"{label} = {spec};\n"
        yield "\n## SCHEMES ##\n"
        yield "[schemes]\n"
        yield f"search = {search};\n"

    # writers -----------------------------------------------------------
    @staticmethod
    def _write(path: str | os.PathLike[str], lines: Iterable[str]) -> Path:
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", buffering=1024 * 1024) as fh:
            for line in lines:
                fh.write(line)
        return dest

    def write_raxml(
        self,
        path: str | os.PathLike[str],
        datatype: str = "dna",
        codon: bool = False,
        model: str | None = None,
    ) -> Path | None:
        """Write RAxML-NG/IQ-TREE partitions; no file is written for an empty table."""
        if not self._entries:
            return None
        return self._write(path, self.iter_raxml_lines(datatype, codon, model))

    def write_nexus_sets(
        self, path: str | os.PathLike[str], codon: bool = False
    ) -> Path | None:
        """Write a standalone NEXUS ``SETS`` block (MrBayes, PAUP*, ...)."""
        if not self._entries:
            return None

        def lines() -> Iterator[str]:
            yield "#NEXUS\n\nBEGIN SETS;\n"
            yield from self.iter_nexus_sets_lines(codon)
            yield "END;\n"

        return self._write(path, lines())

    def write_partitionfinder(
        self,
        path: str | os.PathLike[str],
        alignment: str = "alignment.phy",
        codon: bool = False,
        **options: str,
    ) -> Path | None:
        """Write a PartitionFinder-style ``.cfg``; see :meth:`iter_partitionfinder_lines`."""
        if not self._entries:
            return None
        return self._write(
            path, self.iter_partitionfinder_lines(alignment, codon, **options)
        )
