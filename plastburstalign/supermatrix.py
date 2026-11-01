"""Streaming, deterministic supermatrix construction.

:class:`SupermatrixBuilder` concatenates per-locus aligned FASTA files into ONE
``(ntax, nchar)`` ``uint8`` buffer, optionally trimming columns in flight:

1. **Pass 1 (scan)** reads every locus, extracts the (renamed) taxon ids and
   the alignment width, and fully validates the FASTA layout with C-level
   byte counts (no sequence copies). It fixes the global taxon order (first
   appearance across loci, in locus order) and sizes the buffer.
2. **Pass 2 (load)** parses loci in parallel threads with a bounded window of
   in-flight loci, computes site statistics and the trimming mask, and
   *commits in locus order*, so column offsets (after trimming) are
   deterministic and independent of ``max_workers`` or completion order. Peak
   RAM is the buffer plus ``window x`` one locus matrix and its temporaries.
3. The buffer is compacted in place to the final ``(ntax, nchar)`` shape.

The buffer lives in RAM, or in an unlinked ``np.memmap`` file in
``scratch_dir`` when it would exceed ``mem_budget_bytes`` (or when
``buffer="memmap"``).

Semantics shared with the legacy accumulator
--------------------------------------------
* locus order = order of ``sources``; taxon row order = first appearance;
* taxa absent from a locus are padded with ``?``;
* FASTA/NEXUS outputs are byte-identical to
  :func:`~plastburstalign.nexus_ops.fast_write_fasta` /
  :func:`~plastburstalign.nexus_ops.fast_write_nexus` for the same data when
  ``normalize_case=False`` and ``trim=None``.

Deliberate differences: a locus that cannot be parsed (missing/empty file,
ragged lengths, duplicate ids, rename errors, zero width) is skipped with a
warning and recorded in :attr:`Supermatrix.failed_loci` (the legacy loader
silently accepted ragged rows); loci whose columns are all trimmed away are
also listed there; taxa that only occur in loci that contributed no columns
are dropped from the result.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import weakref
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

from .filtering import (
    DEFAULT_BLOCK_BYTES,
    TrimSpec,
    compute_keep_mask,
    site_statistics,
)
from .logging_ops import logger as log
from .matrix_ops import (
    MISSING,
    AlignmentFormatError,
    DenseLocusMatrix,
    DuplicatePolicy,
    scan_fasta,
)
from .nexus_ops import (
    NEXUS_DATA_FOOTER,
    ConcatenationError,
    nexus_data_header,
    nexus_row_prefix,
)
from .partitions import PartitionTable

__all__ = [
    "BufferMode",
    "LocusSource",
    "LocusStats",
    "PartitionFileFormat",
    "Supermatrix",
    "SupermatrixBuilder",
    "legacy_concat_rename",
]

U8Array: TypeAlias = npt.NDArray[np.uint8]
IntArray: TypeAlias = npt.NDArray[np.int64]
BufferMode: TypeAlias = Literal["auto", "ram", "memmap"]
PartitionFileFormat: TypeAlias = Literal["raxml", "iqtree", "nexus", "partitionfinder"]
RenameFn: TypeAlias = Callable[[str, str], str]

_T = TypeVar("_T")
_R = TypeVar("_R")

_WRITE_BUFFER = 4 * 1024 * 1024


def legacy_concat_rename(locus: str, raw_id: str) -> str:
    """Legacy taxon renaming of ``AlignmentCoordination.concat_MSAs``.

    A leading ``_R_<locus>_`` (MAFFT ``--adjustdirection`` marker) or
    ``<locus>_`` is replaced by ``concat_``; other ids are unchanged.
    """
    reversed_prefix = f"_R_{locus}_"
    if raw_id.startswith(reversed_prefix):
        return "concat_" + raw_id[len(reversed_prefix) :]
    plain_prefix = f"{locus}_"
    if raw_id.startswith(plain_prefix):
        return "concat_" + raw_id[len(plain_prefix) :]
    return raw_id


@dataclass(frozen=True)
class LocusSource:
    """One aligned locus to concatenate.

    Attributes:
        name: locus name (becomes the partition name; must be unique).
        path: aligned FASTA file (wrapped lines and lowercase are fine).
        kind: locus kind, e.g. ``"cds"``, ``"igs"``, ``"int"``, ``"rna"``.
        codon_aligned: the alignment is codon-aligned and starts at codon
            position 1 (e.g. a back-translated protein alignment). Only then
            (and only for ``kind == "cds"``) codon partitions are available.
    """

    name: str
    path: str | os.PathLike[str]
    kind: str = "nucl"
    codon_aligned: bool = False


@dataclass(frozen=True)
class LocusStats:
    """Per-locus summary (statistics refer to the columns that were kept).

    Attributes:
        name, kind: from the :class:`LocusSource`.
        ntax_present: taxa present in this locus.
        nchar_in: alignment width before trimming.
        nchar_kept: columns contributed to the supermatrix.
        variable, informative: variable / parsimony-informative sites among
            the kept columns (definitions in :mod:`plastburstalign.filtering`).
        variable_in, informative_in: same counts before trimming.
        missing_fraction: fraction of the locus' supermatrix block (ALL taxa x
            kept columns) that is ``?`` (absent-taxon padding or explicit ``?``).
        gap_fraction: fraction of that block made of gap characters.
        occupied_cells: block cells that are neither gap nor ``?``.
        start, end: 1-based supermatrix columns (both 0 if no column was kept).
    """

    name: str
    kind: str
    ntax_present: int
    nchar_in: int
    nchar_kept: int
    variable: int
    informative: int
    variable_in: int
    informative_in: int
    missing_fraction: float
    gap_fraction: float
    occupied_cells: int
    start: int
    end: int


# ----------------------------------------------------------------------
# result object
# ----------------------------------------------------------------------
class Supermatrix:
    """A finished supermatrix: read-only matrix, taxa, partitions, statistics.

    Attributes:
        taxa: row labels (first appearance order).
        partitions: :class:`~plastburstalign.partitions.PartitionTable` in
            final (post-trimming) coordinates.
        stats: per-locus :class:`LocusStats` in locus order.
        failed_loci: ``{locus: reason}`` for loci that contributed no columns.
        datatype: NEXUS datatype label (``"dna"``).
    """

    __slots__ = (
        "taxa",
        "partitions",
        "stats",
        "failed_loci",
        "datatype",
        "_matrix",
        "_owner",
        "_taxa_bytes",
    )

    def __init__(
        self,
        taxa: Sequence[str],
        matrix: U8Array,
        partitions: PartitionTable,
        stats: Sequence[LocusStats],
        failed_loci: dict[str, str] | None = None,
        datatype: str = "dna",
        owner: object | None = None,
    ) -> None:
        if matrix.ndim != 2 or matrix.dtype != np.uint8:
            raise ValueError("matrix must be a 2-D uint8 array")
        if matrix.shape[0] != len(taxa):
            raise ValueError("taxa do not match matrix rows")
        view = matrix.view()
        view.flags.writeable = False
        self.taxa: tuple[str, ...] = tuple(taxa)
        self._matrix: U8Array = view
        self.partitions = partitions
        self.stats: tuple[LocusStats, ...] = tuple(stats)
        self.failed_loci: dict[str, str] = dict(failed_loci or {})
        self.datatype = datatype
        self._owner = owner
        self._taxa_bytes: list[bytes] | None = None

    # -- basic properties ------------------------------------------------
    @property
    def matrix(self) -> U8Array:
        """Read-only ``(ntax, nchar)`` view of the raw ASCII bytes."""
        return self._matrix

    @property
    def ntax(self) -> int:
        """Number of taxa (rows)."""
        return len(self.taxa)

    @property
    def nchar(self) -> int:
        """Number of columns."""
        return int(self._matrix.shape[1])

    @property
    def nbytes(self) -> int:
        """Size of the matrix in bytes."""
        return int(self._matrix.nbytes)

    @property
    def variable_sites(self) -> int:
        """Total variable sites among kept columns."""
        return sum(s.variable for s in self.stats)

    @property
    def informative_sites(self) -> int:
        """Total parsimony-informative sites among kept columns."""
        return sum(s.informative for s in self.stats)

    @property
    def density(self) -> float:
        """Fraction of cells that are neither gap nor ``?`` (0 if empty)."""
        cells = self.ntax * self.nchar
        if cells == 0:
            return 0.0
        return sum(s.occupied_cells for s in self.stats) / cells

    def locus_stats(self, name: str) -> LocusStats:
        """Statistics of one locus by name."""
        for entry in self.stats:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def to_dict(self) -> dict[str, str]:
        """Legacy-style ``{taxon: sequence}`` mapping (for tests/interop)."""
        return {
            t: self._matrix[i].tobytes().decode("ascii")
            for i, t in enumerate(self.taxa)
        }

    def summary(self) -> str:
        """One-line human readable summary."""
        return (
            f"{self.ntax} taxa x {self.nchar} sites in {len(self.partitions)} loci "
            f"({self.informative_sites} informative, density {self.density:.3f}, "
            f"{len(self.failed_loci)} skipped)"
        )

    def close(self) -> None:
        """Release the backing buffer (a no-op for RAM buffers beyond GC)."""
        self._owner = None

    def __enter__(self) -> Supermatrix:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Supermatrix({self.summary()})"

    # -- hashing ---------------------------------------------------------
    def sha256(self, include_taxa: bool = False) -> str:
        """SHA-256 hex digest of the canonical matrix bytes.

        The canonical bytes are all rows concatenated in row order (C order,
        ``ntax * nchar`` bytes, no separators). With ``include_taxa=True`` the
        digest is that of the unwrapped FASTA text (``>taxon\\nrow\\n`` per
        row), i.e. of the file produced by :meth:`write_fasta`.
        """
        digest = hashlib.sha256()
        labels = self._labels() if include_taxa else None
        for i in range(self.ntax):
            if labels is not None:
                digest.update(b">" + labels[i] + b"\n")
            digest.update(self._matrix[i])
            if labels is not None:
                digest.update(b"\n")
        return digest.hexdigest()

    # -- writers ---------------------------------------------------------
    def _labels(self) -> list[bytes]:
        if self._taxa_bytes is None:
            self._taxa_bytes = [t.encode("utf-8") for t in self.taxa]
        return self._taxa_bytes

    @staticmethod
    def _open(path: str | os.PathLike[str]) -> BinaryIO:
        dest = Path(path)
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        return open(dest, "wb", buffering=_WRITE_BUFFER)

    def write_fasta(self, path: str | os.PathLike[str]) -> Path:
        """Stream an unwrapped FASTA file (legacy ``fast_write_fasta`` layout)."""
        labels = self._labels()
        with self._open(path) as fh:
            for i, label in enumerate(labels):
                fh.write(b">" + label + b"\n")
                fh.write(self._matrix[i])
                fh.write(b"\n")
        return Path(path)

    def write_phylip(self, path: str | os.PathLike[str], relaxed: bool = True) -> Path:
        """Stream a sequential PHYLIP file.

        Args:
            path: output file.
            relaxed: relaxed PHYLIP (names of any length, padded to the longest
                name plus two spaces). Strict PHYLIP pads/limits names to 10
                characters.

        Raises:
            ValueError: a name contains whitespace, or (strict) exceeds 10 chars.
        """
        for taxon in self.taxa:
            if re.search(r"\s", taxon):
                raise ValueError(
                    f"taxon {taxon!r} contains whitespace; unusable in PHYLIP"
                )
            if not relaxed and len(taxon) > 10:
                raise ValueError(
                    f"taxon {taxon!r} exceeds 10 characters; use relaxed=True"
                )
        width = max((len(t) for t in self.taxa), default=0) + 2 if relaxed else 10
        with self._open(path) as fh:
            fh.write(f" {self.ntax} {self.nchar}\n".encode("ascii"))
            for i, taxon in enumerate(self.taxa):
                fh.write(taxon.ljust(width).encode("utf-8"))
                fh.write(self._matrix[i])
                fh.write(b"\n")
        return Path(path)

    def write_nexus(
        self,
        path: str | os.PathLike[str],
        partition_format: Literal["nexus", "none"] = "nexus",
        *,
        codon_partitions: bool = False,
        gap: str = "-",
        missing: str = "?",
    ) -> Path:
        """Stream a non-interleaved NEXUS file (legacy ``fast_write_nexus`` layout).

        Args:
            path: output file.
            partition_format: ``"nexus"`` appends a ``SETS`` block with
                ``CHARSET``/``CHARPARTITION``; ``"none"`` omits it.
            codon_partitions: split codon-aligned CDS loci into ``_pos1..3``.
            gap, missing: characters announced in the ``FORMAT`` line.
        """
        if partition_format not in ("nexus", "none"):
            raise ValueError(
                f"partition_format must be 'nexus' or 'none': {partition_format!r}"
            )
        label_width = max((len(t) for t in self.taxa), default=0)
        with self._open(path) as fh:
            fh.write(
                nexus_data_header(
                    self.ntax, self.nchar, self.datatype, gap, missing
                ).encode("utf-8")
            )
            for i, taxon in enumerate(self.taxa):
                fh.write(nexus_row_prefix(taxon, label_width).encode("utf-8"))
                fh.write(self._matrix[i])
                fh.write(b"\n")
            fh.write(NEXUS_DATA_FOOTER.encode("ascii"))
            if partition_format == "nexus" and len(self.partitions):
                fh.write(b"\nBEGIN SETS;\n")
                for line in self.partitions.iter_nexus_sets_lines(codon_partitions):
                    fh.write(line.encode("utf-8"))
                fh.write(b"END;\n")
        return Path(path)

    def write_partitions(
        self,
        path: str | os.PathLike[str],
        fmt: PartitionFileFormat = "raxml",
        *,
        codon: bool = False,
        alignment_name: str = "alignment.phy",
    ) -> Path | None:
        """Write a partition file.

        Args:
            path: output file.
            fmt: ``"raxml"`` (RAxML-NG; also read by IQ-TREE), ``"iqtree"``
                (alias of ``"raxml"``: IQ-TREE reads the same ``DNA, name =
                a-b`` lines), ``"nexus"`` (standalone ``SETS`` block) or
                ``"partitionfinder"`` (``.cfg`` referring to ``alignment_name``).
            codon: split codon-aligned CDS loci into ``_pos1..3`` partitions.
            alignment_name: alignment file name written into the PartitionFinder cfg.

        Returns:
            The written path, or ``None`` if there are no partitions.
        """
        if fmt in ("raxml", "iqtree"):
            return self.partitions.write_raxml(path, self.datatype, codon)
        if fmt == "nexus":
            return self.partitions.write_nexus_sets(path, codon)
        if fmt == "partitionfinder":
            return self.partitions.write_partitionfinder(path, alignment_name, codon)
        raise ValueError(f"unknown partition format {fmt!r}")

    def write_all(
        self,
        prefix: str | os.PathLike[str],
        *,
        fasta: bool = True,
        nexus: bool = True,
        phylip: bool = False,
        partitions: Sequence[PartitionFileFormat] = ("raxml", "nexus"),
        partition_format: Literal["nexus", "none"] = "nexus",
        codon_partitions: bool = False,
    ) -> dict[str, Path]:
        """Write the standard output set next to ``prefix``.

        With ``prefix="<out_dir>/nucl_<N>concat"`` the file names equal the
        legacy ones: ``<prefix>.aligned.fasta``, ``.aligned.nexus``,
        ``.partitions.txt`` (RAxML-NG/IQ-TREE) and ``.partitions.nex``.
        Additional: ``.aligned.phy`` (``phylip=True``), ``.partitions.iqtree.txt``
        (``"iqtree"``), ``.partitionfinder.cfg`` (``"partitionfinder"``).

        Returns:
            ``{"fasta"|"nexus"|"phylip"|"raxml"|"nexus_partitions"|"iqtree"|
            "partitionfinder": path}`` for the files written.
        """
        base = str(prefix)
        written: dict[str, Path] = {}
        if fasta:
            written["fasta"] = self.write_fasta(f"{base}.aligned.fasta")
        if nexus:
            written["nexus"] = self.write_nexus(
                f"{base}.aligned.nexus",
                partition_format,
                codon_partitions=codon_partitions,
            )
        if phylip:
            written["phylip"] = self.write_phylip(f"{base}.aligned.phy")
        names: dict[str, tuple[str, str]] = {
            "raxml": ("raxml", ".partitions.txt"),
            "nexus": ("nexus_partitions", ".partitions.nex"),
            "iqtree": ("iqtree", ".partitions.iqtree.txt"),
            "partitionfinder": ("partitionfinder", ".partitionfinder.cfg"),
        }
        for fmt in partitions:
            key, suffix = names[fmt]
            dest = self.write_partitions(
                f"{base}{suffix}",
                fmt,
                codon=codon_partitions,
                alignment_name=Path(f"{base}.aligned.phy").name,
            )
            if dest is not None:
                written[key] = dest
        return written


# ----------------------------------------------------------------------
# buffer management
# ----------------------------------------------------------------------
def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class _MatrixBuffer:
    """One ``ntax x capacity`` ``uint8`` buffer in RAM or an unlinked memmap."""

    def __init__(
        self,
        ntax: int,
        capacity: int,
        mode: Literal["ram", "memmap"],
        scratch_dir: str | os.PathLike[str] | None,
    ) -> None:
        self.ntax = ntax
        self.capacity = capacity
        self.mode: Literal["ram", "memmap"] = mode
        size = ntax * capacity
        self._flat: U8Array
        if mode == "memmap" and size > 0:
            if scratch_dir is not None:
                os.makedirs(scratch_dir, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix="pba_supermatrix_", suffix=".u8", dir=scratch_dir
            )
            os.close(fd)
            self._flat = np.memmap(name, dtype=np.uint8, mode="w+", shape=(size,))
            if os.name == "posix":
                # Data stays reachable through the mapping; the kernel reclaims
                # the file as soon as the mapping is dropped, even after a crash.
                _unlink_quiet(name)
            else:  # pragma: no cover - Windows cannot unlink a mapped file
                weakref.finalize(self, _unlink_quiet, name)
        else:
            self.mode = "ram"
            self._flat = np.empty(size, dtype=np.uint8)
        self._view2d: U8Array | None = self._flat.reshape(ntax, capacity)

    def write_locus(
        self, rows: npt.NDArray[np.intp], block: U8Array, offset: int
    ) -> None:
        """Store ``block`` in the given rows at ``offset``; ``?`` for other rows."""
        view = self._view2d
        assert view is not None
        width = block.shape[1]
        cols = slice(offset, offset + width)
        if rows.size == self.ntax and bool((rows == np.arange(self.ntax)).all()):
            view[:, cols] = block
            return
        view[rows, cols] = block
        if rows.size < self.ntax:
            absent = np.ones(self.ntax, dtype=np.bool_)
            absent[rows] = False
            view[absent, cols] = MISSING

    def finalize(self, width: int, keep_rows: IntArray | None) -> U8Array:
        """Compact in place to ``(len(keep_rows) or ntax, width)`` and return it."""
        cap = self.capacity
        rows: Sequence[int] = (
            range(self.ntax) if keep_rows is None else keep_rows.tolist()
        )
        nout = len(rows)
        self._view2d = None
        flat = self._flat
        if width != cap or nout != self.ntax:
            # Ascending in-place moves: destination offsets never exceed the
            # source offsets, and never overwrite rows that are still to be read.
            for new_i, old_i in enumerate(rows):
                if new_i * width != old_i * cap:
                    flat[new_i * width : (new_i + 1) * width] = flat[
                        old_i * cap : old_i * cap + width
                    ]
            if self.mode == "ram":
                flat.resize(nout * width, refcheck=False)
        return flat[: nout * width].reshape(nout, width)


def _default_budget() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available * 0.5)
    except Exception:
        return 1 << 62


def _ordered_map(
    pool: ThreadPoolExecutor,
    fn: Callable[[_T], _R],
    items: Sequence[_T],
    window: int,
) -> Iterator[tuple[_T, _R]]:
    """Map ``fn`` over ``items`` in threads, yielding results IN ORDER.

    At most ``window`` tasks are submitted ahead of the consumer, which bounds
    the number of results held in memory (head-of-line blocking included).
    """
    pending: deque[Future[_R]] = deque()
    submitted = 0
    total = len(items)
    try:
        for i in range(total):
            while submitted < total and len(pending) < window:
                pending.append(pool.submit(fn, items[submitted]))
                submitted += 1
            yield items[i], pending.popleft().result()
    finally:
        for fut in pending:
            fut.cancel()


# ----------------------------------------------------------------------
# builder
# ----------------------------------------------------------------------
@dataclass
class _Scan:
    ids: tuple[str, ...] = ()
    nchar: int = 0
    error: str | None = None


@dataclass
class _Loaded:
    taxa: tuple[str, ...] = ()
    block: U8Array | None = None
    nchar_in: int = 0
    columns: IntArray | None = None
    variable_in: int = 0
    informative_in: int = 0
    variable: int = 0
    informative: int = 0
    gap_cells: int = 0
    missing_cells: int = 0
    data_cells: int = 0
    error: str | None = None


class SupermatrixBuilder:
    """Concatenate aligned per-locus FASTA files into a :class:`Supermatrix`.

    Args:
        sources: ordered loci; the order defines column order.
        rename: ``(locus_name, raw_id) -> taxon id``; default keeps ids
            unchanged. Use :func:`legacy_concat_rename` for the legacy rule.
        trim: :class:`~plastburstalign.filtering.TrimSpec`, its string form, or
            ``None`` (no trimming). Applied per locus before concatenation.
        normalize_case: upper-case letters and map ``.`` to ``-``. ``False``
            keeps bytes verbatim (required for byte parity with the legacy path).
        buffer: ``"ram"``, ``"memmap"`` or ``"auto"`` (memmap when the buffer
            plus the in-flight window would exceed ``mem_budget_bytes``).
        scratch_dir: directory for the memmap file (default: system temp dir).
        mem_budget_bytes: RAM budget for ``buffer="auto"``; defaults to half of
            the currently available memory (via ``psutil``).
        max_workers: parser threads.
        window: loci allowed in flight, i.e. parsed or being parsed but not yet
            committed (default ``max_workers + 2``).
        on_duplicate: ``"error"`` (locus skipped) or ``"suffix"`` for repeated
            ids inside one locus.
        datatype: NEXUS datatype label.
        block_bytes: total bound for filtering temporaries; each worker gets
            ``block_bytes // max_workers`` (default 64 MiB in total, at least 4 MiB
            per worker).
    """

    def __init__(
        self,
        sources: Sequence[LocusSource],
        *,
        rename: RenameFn | None = None,
        trim: TrimSpec | str | None = None,
        normalize_case: bool = True,
        buffer: BufferMode = "auto",
        scratch_dir: str | os.PathLike[str] | None = None,
        mem_budget_bytes: int | None = None,
        max_workers: int = 4,
        window: int | None = None,
        on_duplicate: DuplicatePolicy = "error",
        datatype: str = "dna",
        block_bytes: int | None = None,
    ) -> None:
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate locus names: {dup}")
        if buffer not in ("auto", "ram", "memmap"):
            raise ValueError(f"buffer must be 'auto', 'ram' or 'memmap': {buffer!r}")
        self._sources = list(sources)
        self._rename: RenameFn | None = rename
        self._spec = TrimSpec.parse(trim)
        self._normalize_case = normalize_case
        self._buffer: BufferMode = buffer
        self._scratch_dir = scratch_dir
        self._mem_budget = mem_budget_bytes
        self._max_workers = max(1, int(max_workers))
        self._window = max(1, int(window)) if window else self._max_workers + 2
        self._on_duplicate: DuplicatePolicy = on_duplicate
        self._datatype = datatype
        total = DEFAULT_BLOCK_BYTES if block_bytes is None else int(block_bytes)
        self._block_bytes = max(4 * 1024 * 1024, total // self._max_workers)

    # -- helpers ---------------------------------------------------------
    @property
    def trim(self) -> TrimSpec:
        """The effective trimming specification."""
        return self._spec

    def _renamer(self, name: str) -> Callable[[str], str] | None:
        if self._rename is None:
            return None
        return partial(self._rename, name)

    def _decide_buffer(
        self, ntax: int, capacity: int, max_locus_cells: int
    ) -> Literal["ram", "memmap"]:
        if self._buffer == "ram":
            return "ram"
        if self._buffer == "memmap":
            return "memmap"
        budget = self._mem_budget if self._mem_budget is not None else _default_budget()
        # window x (raw bytes + parsed matrix + trimmed copy / temporaries)
        needed = ntax * capacity + self._window * max_locus_cells * 3
        return "memmap" if needed > budget else "ram"

    # -- pass 1 ----------------------------------------------------------
    def _scan_locus(self, idx: int) -> _Scan:
        src = self._sources[idx]
        try:
            raw = Path(src.path).read_bytes()
        except OSError as err:
            return _Scan(error=f"cannot read {src.path}: {err}")
        try:
            scan = scan_fasta(
                raw,
                self._renamer(src.name),
                on_duplicate=self._on_duplicate,
                validate_lengths=True,
            )
        except (AlignmentFormatError, ValueError) as err:
            return _Scan(error=str(err))
        except Exception as err:
            return _Scan(error=f"unexpected error while scanning: {err!r}")
        if not scan.ids:
            return _Scan(error="alignment file is empty (no sequences)")
        if scan.nchar == 0:
            return _Scan(error="alignment has zero columns")
        return _Scan(ids=scan.ids, nchar=scan.nchar)

    # -- pass 2 ----------------------------------------------------------
    def _process_locus(self, idx: int) -> _Loaded:
        """Parse, analyse and trim one locus (runs in a worker thread)."""
        src = self._sources[idx]
        try:
            mat = DenseLocusMatrix.from_fasta_path(
                src.path,
                self._renamer(src.name),
                self._normalize_case,
                on_duplicate=self._on_duplicate,
            )
            stats = site_statistics(
                mat,
                ambiguity_occupied=self._spec.ambiguity_occupied,
                composition=True,
                block_bytes=self._block_bytes,
            )
            keep = compute_keep_mask(
                mat, self._spec, stats=stats, block_bytes=self._block_bytes
            )
            assert stats.variable is not None and stats.informative is not None
            assert stats.gap is not None and stats.missing is not None
            kept_all = bool(keep.all())
            columns: IntArray | None = None
            if kept_all:
                block = mat.matrix
            else:
                columns = np.flatnonzero(keep).astype(np.int64)
                block = np.ascontiguousarray(mat.matrix[:, keep])
            gap = int(stats.gap[keep].sum(dtype=np.int64))
            miss = int(stats.missing[keep].sum(dtype=np.int64))
            return _Loaded(
                taxa=mat.taxa,
                block=block,
                nchar_in=mat.nchar,
                columns=columns,
                variable_in=int(stats.variable.sum()),
                informative_in=int(stats.informative.sum()),
                variable=int(stats.variable[keep].sum()),
                informative=int(stats.informative[keep].sum()),
                gap_cells=gap,
                missing_cells=miss,
                data_cells=mat.ntax * int(block.shape[1]) - gap - miss,
            )
        except Exception as err:
            return _Loaded(error=f"{type(err).__name__}: {err}")

    # -- driver ----------------------------------------------------------
    def build(self) -> Supermatrix:
        """Run both passes and return the :class:`Supermatrix`.

        Raises:
            ConcatenationError: no locus contributed any column.
        """
        sources = self._sources
        failed: dict[str, str] = {}

        def skip(name: str, reason: str) -> None:
            failed[name] = reason
            log.warning(f"locus `{name}` skipped: {reason}")

        index: dict[str, int] = {}
        active: list[int] = []
        widths: dict[int, int] = {}
        cells: dict[int, int] = {}
        buf: _MatrixBuffer | None = None

        with ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="pba-supermatrix"
        ) as pool:
            # ---- pass 1: scan headers / widths, fix taxon order and size ----
            for i, scan in _ordered_map(
                pool, self._scan_locus, range(len(sources)), self._window
            ):
                if scan.error is not None:
                    skip(sources[i].name, scan.error)
                    continue
                for taxon in scan.ids:
                    index.setdefault(taxon, len(index))
                active.append(i)
                widths[i] = scan.nchar
                cells[i] = len(scan.ids) * scan.nchar
            if not active:
                raise ConcatenationError(
                    "no alignment could be loaded: "
                    + "; ".join(f"{k}: {v}" for k, v in failed.items())
                )

            ntax = len(index)
            capacity = sum(widths.values())
            mode = self._decide_buffer(ntax, capacity, max(cells.values()))
            log.debug(
                f"supermatrix buffer: {ntax} taxa x {capacity} columns "
                f"({ntax * capacity / 1e6:.1f} MB) in {mode}"
            )
            buf = _MatrixBuffer(ntax, capacity, mode, self._scratch_dir)

            # ---- pass 2: parallel load, commit in locus order ----
            table = PartitionTable()
            stats_list: list[LocusStats] = []
            used = np.zeros(ntax, dtype=np.bool_)
            width = 0
            for i, loaded in _ordered_map(
                pool, self._process_locus, active, self._window
            ):
                src = sources[i]
                if loaded.error is not None or loaded.block is None:
                    skip(src.name, loaded.error or "unknown error")
                    continue
                block = loaded.block
                kept = int(block.shape[1])
                present = len(loaded.taxa)
                if loaded.nchar_in != widths[i]:
                    skip(
                        src.name,
                        f"alignment width changed between passes "
                        f"({widths[i]} -> {loaded.nchar_in}); file modified?",
                    )
                    continue
                if kept == 0:
                    skip(src.name, "all columns were removed by trimming")
                    stats_list.append(
                        LocusStats(
                            name=src.name,
                            kind=src.kind,
                            ntax_present=present,
                            nchar_in=loaded.nchar_in,
                            nchar_kept=0,
                            variable=0,
                            informative=0,
                            variable_in=loaded.variable_in,
                            informative_in=loaded.informative_in,
                            missing_fraction=0.0,
                            gap_fraction=0.0,
                            occupied_cells=0,
                            start=0,
                            end=0,
                        )
                    )
                    continue
                try:
                    rows = np.fromiter(
                        (index[t] for t in loaded.taxa), dtype=np.intp, count=present
                    )
                except KeyError as err:
                    skip(
                        src.name,
                        f"taxon {err} was not seen during the header scan; "
                        "file modified?",
                    )
                    continue
                buf.write_locus(rows, block, width)
                used[rows] = True
                codon_phase: int | None = None
                if src.kind == "cds" and src.codon_aligned:
                    if loaded.nchar_in % 3 == 0:
                        codon_phase = 0
                    else:
                        log.warning(
                            f"locus `{src.name}` is flagged codon-aligned but its "
                            f"width {loaded.nchar_in} is not a multiple of 3; "
                            "codon partitions disabled for it"
                        )
                entry = table.append_locus(
                    src.name, kept, src.kind, loaded.columns, codon_phase
                )
                block_cells = ntax * kept
                missing = loaded.missing_cells + (ntax - present) * kept
                stats_list.append(
                    LocusStats(
                        name=src.name,
                        kind=src.kind,
                        ntax_present=present,
                        nchar_in=loaded.nchar_in,
                        nchar_kept=kept,
                        variable=loaded.variable,
                        informative=loaded.informative,
                        variable_in=loaded.variable_in,
                        informative_in=loaded.informative_in,
                        missing_fraction=missing / block_cells,
                        gap_fraction=loaded.gap_cells / block_cells,
                        occupied_cells=loaded.data_cells,
                        start=entry.start,
                        end=entry.end,
                    )
                )
                width += kept
                loaded.block = None  # release before the next commit

        assert buf is not None
        if width == 0:
            raise ConcatenationError(
                "no alignment contributed any column: "
                + "; ".join(f"{k}: {v}" for k, v in failed.items())
            )
        taxa = list(index)
        keep_rows: IntArray | None = None
        if not bool(used.all()):
            keep_rows = np.flatnonzero(used).astype(np.int64)
            dropped = [taxa[j] for j in np.flatnonzero(~used).tolist()]
            log.warning(
                f"dropping {len(dropped)} taxa that occur only in skipped loci "
                f"(e.g. {dropped[:3]})"
            )
            taxa = [taxa[j] for j in keep_rows.tolist()]
        matrix = buf.finalize(width, keep_rows)
        result = Supermatrix(
            taxa, matrix, table, stats_list, failed, self._datatype, owner=buf
        )
        log.info(f"supermatrix built: {result.summary()} [{mode}]")
        return result
