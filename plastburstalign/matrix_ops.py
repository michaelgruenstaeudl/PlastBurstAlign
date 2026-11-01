"""Dense ``uint8`` locus matrices and a high-throughput FASTA parser.

A :class:`DenseLocusMatrix` stores one aligned locus as a taxa tuple plus a
C-contiguous 2-D ``uint8`` array of raw ASCII bytes (one byte per alignment
cell). Raw bytes keep IUPAC codes, lowercase letters and gap characters
lossless; case/gap normalization is an explicit, LUT-driven, optional step.

Parsing never loops over characters in Python. Per FASTA record the parser
uses ``bytes.find`` to locate line boundaries and one C-level
``bytes.translate`` call (which deletes line breaks/whitespace and, optionally,
applies the normalization LUT in the same pass) before copying the cleaned
bytes straight into one preallocated array.

Legacy-compatible conventions (see ``alignment_ops._load_alignment_fast``):

* the taxon id is the first whitespace-delimited token of the header, passed
  through an optional ``rename`` callable;
* wrapped sequence lines, CRLF line endings, blank lines and a missing trailing
  newline are accepted.

Deliberate differences from the legacy dict loader (all raise
:class:`AlignmentFormatError` instead of silently producing bad data):

* ragged records (any record whose length differs from the first) are
  rejected, the offending id is named in the message;
* duplicate ids (after ``rename``) are rejected unless ``on_duplicate="suffix"``
  is requested, which renames the k-th repeat to ``<id>__dup<k>``;
* non-whitespace text before the first ``>`` header, or an empty header, is
  rejected.
"""

from __future__ import annotations

import io
import mmap
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from .logging_ops import logger as log

__all__ = [
    "BASE_N",
    "CANONICAL_LUT",
    "DOT",
    "GAP",
    "GAP_LUT",
    "IDENTITY_LUT",
    "MISSING",
    "UPPER_LUT",
    "AlignmentFormatError",
    "DenseLocusMatrix",
    "FastaScan",
    "make_lut",
    "scan_fasta",
]

U8Array: TypeAlias = npt.NDArray[np.uint8]
RawFasta: TypeAlias = bytes | mmap.mmap
DuplicatePolicy: TypeAlias = Literal["error", "suffix"]

GAP: int = ord("-")
"""ASCII code of the canonical gap character ``-``."""
DOT: int = ord(".")
"""ASCII code of ``.``, an alternative gap character normalized to ``-``."""
MISSING: int = ord("?")
"""ASCII code of the missing-data character ``?``."""
BASE_N: int = ord("N")
"""ASCII code of the fully ambiguous nucleotide ``N``."""

# Bytes deleted from sequence bodies: space, tab, CR, LF, VT, FF.
_WS: bytes = b" \t\r\n\x0b\x0c"


class AlignmentFormatError(ValueError):
    """Raised when an aligned FASTA input is malformed (ragged, duplicates, ...)."""


def make_lut(*, upper: bool = True, gaps: bool = True) -> bytes:
    """Build a 256-entry translation table for ``bytes.translate``.

    Args:
        upper: map ``a-z`` to ``A-Z``.
        gaps: map ``.`` to ``-``.

    Returns:
        A 256-byte table; the identity table when both flags are false.
    """
    table = bytearray(range(256))
    if upper:
        for code in range(ord("a"), ord("z") + 1):
            table[code] = code - 32
    if gaps:
        table[DOT] = GAP
    return bytes(table)


IDENTITY_LUT: bytes = make_lut(upper=False, gaps=False)
UPPER_LUT: bytes = make_lut(upper=True, gaps=False)
GAP_LUT: bytes = make_lut(upper=False, gaps=True)
CANONICAL_LUT: bytes = make_lut(upper=True, gaps=True)

_WS_SET: frozenset[int] = frozenset(_WS)


def _count_in(raw: RawFasta, sub: bytes, start: int, end: int) -> int:
    """``raw[start:end].count(sub)`` without copying when ``raw`` is ``bytes``."""
    if isinstance(raw, bytes):
        return raw.count(sub, start, end)
    return raw[start:end].count(sub)


class FastaScan:
    """Record layout of a FASTA buffer (ids and sequence byte spans).

    Attributes:
        ids: renamed, duplicate-resolved taxon ids in file order.
        nchar: alignment width (cleaned length of the first record).
        spans: ``(start, end)`` byte offsets of each record's raw sequence body.
    """

    __slots__ = ("ids", "nchar", "spans")

    def __init__(
        self, ids: tuple[str, ...], nchar: int, spans: list[tuple[int, int]]
    ) -> None:
        self.ids = ids
        self.nchar = nchar
        self.spans = spans


def scan_fasta(
    raw: RawFasta,
    rename: Callable[[str], str] | None = None,
    *,
    on_duplicate: DuplicatePolicy = "error",
    validate_lengths: bool = False,
) -> FastaScan:
    """Locate FASTA records and extract (renamed) ids without copying sequences.

    One ``find`` per record locates the next ``\\n>`` boundary; sequence bodies
    are never copied except the first one (needed for the alignment width).

    Args:
        raw: file contents (``bytes`` or a read-only ``mmap``).
        rename: optional ``raw_id -> id`` callable applied to each header token.
        on_duplicate: ``"error"`` rejects repeated ids; ``"suffix"`` renames the
            k-th repeat of ``x`` to ``x__dup<k>`` (deterministic, collision-free).
        validate_lengths: also verify, with C-level ``count`` calls, that every
            record has the same cleaned length as the first (cheap header-only
            style validation used by the supermatrix pre-scan).

    Returns:
        A :class:`FastaScan`; empty (``ids == ()``) for empty/blank input.

    Raises:
        AlignmentFormatError: malformed input.
    """
    n = len(raw)
    pos = 0
    while pos < n and raw[pos] in _WS_SET:
        pos += 1
    if pos >= n:
        return FastaScan((), 0, [])
    if raw[pos] != 0x3E:  # '>'
        raise AlignmentFormatError("input does not start with a FASTA header ('>')")

    starts: list[int] = [pos]
    find = raw.find
    while True:
        nxt = find(b"\n>", pos)
        if nxt < 0:
            break
        pos = nxt + 1
        starts.append(pos)
    starts.append(n)

    ids: list[str] = []
    spans: list[tuple[int, int]] = []
    seen: dict[str, int] = {}
    nrec = len(starts) - 1
    for k in range(nrec):
        rec_start = starts[k]
        rec_end = starts[k + 1]
        header_end = find(b"\n", rec_start, rec_end)
        if header_end < 0:
            header_end = rec_end
        parts = raw[rec_start + 1 : header_end].split(None, 1)
        if not parts:
            raise AlignmentFormatError(f"record {k + 1} has an empty FASTA header")
        token = parts[0].decode("utf-8", errors="replace")
        rid = rename(token) if rename is not None else token
        if not isinstance(rid, str) or not rid or rid.split() != [rid]:
            raise AlignmentFormatError(
                f"rename produced an invalid id {rid!r} for header {token!r}"
            )
        if rid in seen:
            if on_duplicate == "error":
                raise AlignmentFormatError(f"duplicate sequence id {rid!r}")
            count = seen[rid]
            candidate = f"{rid}__dup{count + 1}"
            while candidate in seen:
                count += 1
                candidate = f"{rid}__dup{count + 1}"
            seen[rid] = count + 1
            seen[candidate] = 1
            rid = candidate
        else:
            seen[rid] = 1
        ids.append(rid)
        spans.append((min(header_end + 1, rec_end), rec_end))

    first_start, first_end = spans[0]
    nchar = len(raw[first_start:first_end].translate(None, _WS))
    if validate_lengths and nrec > 1:
        present = [bytes([c]) for c in _WS if find(bytes([c])) >= 0]
        for rid, (b0, b1) in zip(ids[1:], spans[1:], strict=True):
            width = (b1 - b0) - sum(_count_in(raw, ch, b0, b1) for ch in present)
            if width != nchar:
                raise AlignmentFormatError(
                    f"ragged alignment: record {rid!r} has {width} columns, "
                    f"expected {nchar} (first record {ids[0]!r})"
                )
    return FastaScan(tuple(ids), nchar, spans)


class DenseLocusMatrix:
    """One aligned locus as raw ASCII bytes in a C-contiguous ``uint8`` matrix.

    Attributes:
        taxa: taxon ids, one per matrix row.
        matrix: ``(ntax, nchar)`` C-contiguous ``uint8`` array.
        nchar: alignment width (``matrix.shape[1]``).
    """

    __slots__ = ("taxa", "matrix", "nchar", "_index")

    taxa: tuple[str, ...]
    matrix: U8Array
    nchar: int

    def __init__(
        self, taxa: Sequence[str], matrix: U8Array, nchar: int | None = None
    ) -> None:
        if matrix.ndim != 2:
            raise ValueError("matrix must be 2-dimensional")
        if matrix.dtype != np.uint8:
            raise ValueError(f"matrix must be uint8, got {matrix.dtype}")
        if len(taxa) != matrix.shape[0]:
            raise ValueError(
                f"{len(taxa)} taxa do not match {matrix.shape[0]} matrix rows"
            )
        if nchar is not None and nchar != matrix.shape[1]:
            raise ValueError(f"nchar={nchar} does not match shape {matrix.shape}")
        self.taxa = tuple(taxa)
        self.matrix = np.ascontiguousarray(matrix)
        self.nchar = int(self.matrix.shape[1])
        self._index: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # basic properties
    # ------------------------------------------------------------------
    @property
    def ntax(self) -> int:
        """Number of taxa (rows)."""
        return len(self.taxa)

    @property
    def nbytes(self) -> int:
        """Size of the matrix payload in bytes."""
        return int(self.matrix.nbytes)

    def __repr__(self) -> str:
        return f"DenseLocusMatrix(ntax={self.ntax}, nchar={self.nchar})"

    def index_of(self, taxon: str) -> int:
        """Row index of ``taxon`` (lazily built map); raises ``KeyError`` if absent."""
        if self._index is None:
            self._index = {t: i for i, t in enumerate(self.taxa)}
        return self._index[taxon]

    def row(self, key: int | str) -> U8Array:
        """Row view addressed by index or taxon id."""
        idx = self.index_of(key) if isinstance(key, str) else key
        return self.matrix[idx]

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_fasta_bytes(
        cls,
        raw: RawFasta,
        rename: Callable[[str], str] | None = None,
        normalize_case: bool = True,
        *,
        normalize_gaps: bool | None = None,
        on_duplicate: DuplicatePolicy = "error",
    ) -> DenseLocusMatrix:
        """Parse (wrapped, multi-line) aligned FASTA into one preallocated array.

        Args:
            raw: file contents (``bytes`` or ``mmap``).
            rename: optional ``raw_id -> id`` callable.
            normalize_case: upper-case all letters (via a 256-entry LUT). When
                false (and ``normalize_gaps`` is false) bytes are kept verbatim
                apart from the deleted line breaks/whitespace.
            normalize_gaps: map ``.`` to ``-``; defaults to ``normalize_case``.
            on_duplicate: ``"error"`` or ``"suffix"`` (see :func:`scan_fasta`).

        Returns:
            The parsed matrix; ``ntax == 0`` and ``nchar == 0`` for empty input.

        Raises:
            AlignmentFormatError: ragged lengths (offending id named), duplicate
                ids, empty headers or data before the first header.
        """
        gaps = normalize_case if normalize_gaps is None else normalize_gaps
        table: bytes | None = None
        if normalize_case or gaps:
            table = make_lut(upper=normalize_case, gaps=gaps)
        scan = scan_fasta(raw, rename, on_duplicate=on_duplicate)
        ntax = len(scan.ids)
        nchar = scan.nchar
        mat = np.empty((ntax, nchar), dtype=np.uint8)
        for i, (b0, b1) in enumerate(scan.spans):
            clean = raw[b0:b1].translate(table, _WS)
            if len(clean) != nchar:
                raise AlignmentFormatError(
                    f"ragged alignment: record {scan.ids[i]!r} has {len(clean)} "
                    f"columns, expected {nchar} (first record {scan.ids[0]!r})"
                )
            if nchar:
                mat[i] = np.frombuffer(clean, dtype=np.uint8)
        return cls(scan.ids, mat)

    @classmethod
    def from_fasta_path(
        cls,
        path: str | os.PathLike[str],
        rename: Callable[[str], str] | None = None,
        normalize_case: bool = True,
        *,
        use_mmap: bool = False,
        normalize_gaps: bool | None = None,
        on_duplicate: DuplicatePolicy = "error",
    ) -> DenseLocusMatrix:
        """Parse an aligned FASTA file; ``use_mmap`` avoids reading it into RAM."""
        p = Path(path)
        if use_mmap and p.stat().st_size > 0:
            with (
                p.open("rb") as fh,
                mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm,
            ):
                log.debug(f"parsing {p} via mmap ({len(mm)} bytes)")
                return cls.from_fasta_bytes(
                    mm,
                    rename,
                    normalize_case,
                    normalize_gaps=normalize_gaps,
                    on_duplicate=on_duplicate,
                )
        return cls.from_fasta_bytes(
            p.read_bytes(),
            rename,
            normalize_case,
            normalize_gaps=normalize_gaps,
            on_duplicate=on_duplicate,
        )

    @classmethod
    def from_sequences(
        cls, sequences: Mapping[str, str] | Sequence[tuple[str, str]]
    ) -> DenseLocusMatrix:
        """Build a matrix from ``{taxon: sequence}`` (all sequences equal length)."""
        items = list(sequences.items() if isinstance(sequences, Mapping) else sequences)
        if not items:
            return cls((), np.empty((0, 0), dtype=np.uint8))
        width = len(items[0][1])
        mat = np.empty((len(items), width), dtype=np.uint8)
        for i, (taxon, seq) in enumerate(items):
            if len(seq) != width:
                raise AlignmentFormatError(
                    f"ragged alignment: record {taxon!r} has {len(seq)} columns, "
                    f"expected {width}"
                )
            if width:
                mat[i] = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        return cls(tuple(t for t, _ in items), mat)

    # ------------------------------------------------------------------
    # transformations
    # ------------------------------------------------------------------
    def normalized(self, *, upper: bool = True, gaps: bool = True) -> DenseLocusMatrix:
        """Return a copy with case/gap normalization applied through a LUT."""
        lut = np.frombuffer(make_lut(upper=upper, gaps=gaps), dtype=np.uint8)
        return DenseLocusMatrix(self.taxa, lut[self.matrix])

    def select_columns(self, mask_or_indices: npt.ArrayLike) -> DenseLocusMatrix:
        """Return a new matrix holding the selected columns (order preserved).

        Args:
            mask_or_indices: boolean mask of length ``nchar`` or integer column
                indices (negative indices allowed).
        """
        sel = np.asarray(mask_or_indices)
        if sel.dtype == np.bool_:
            if sel.shape != (self.nchar,):
                raise ValueError(
                    f"column mask has shape {sel.shape}, expected ({self.nchar},)"
                )
        elif sel.size == 0:
            sel = sel.astype(np.intp)
        elif not np.issubdtype(sel.dtype, np.integer):
            raise TypeError("column selection must be a bool mask or integer indices")
        return DenseLocusMatrix(self.taxa, self.matrix[:, sel])

    def slice_columns(self, start: int, stop: int | None = None) -> DenseLocusMatrix:
        """Return a new (copied, contiguous) matrix for columns ``[start:stop)``."""
        return DenseLocusMatrix(self.taxa, self.matrix[:, start:stop])

    # ------------------------------------------------------------------
    # inspection helpers
    # ------------------------------------------------------------------
    def ungapped(self, row: int | str, *, strip_missing: bool = False) -> bytes:
        """Row bytes with gap characters (``-`` and ``.``) removed.

        Args:
            row: row index or taxon id.
            strip_missing: also remove ``?``.
        """
        delete = b"-.?" if strip_missing else b"-."
        return self.row(row).tobytes().translate(None, delete)

    def to_dict(self) -> dict[str, str]:
        """Legacy-style ``{taxon: sequence}`` mapping (for tests and interop)."""
        return {
            t: self.matrix[i].tobytes().decode("ascii") for i, t in enumerate(self.taxa)
        }

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def write_fasta(
        self, dest: str | os.PathLike[str] | BinaryIO, wrap: int | None = None
    ) -> None:
        """Stream the matrix as FASTA, row by row, without building one big string.

        Args:
            dest: path or binary file object.
            wrap: line width; ``None`` writes unwrapped sequences (legacy format).
        """
        if wrap is not None and wrap < 1:
            raise ValueError("wrap must be a positive integer or None")
        if isinstance(dest, (str, os.PathLike)):
            path = Path(dest)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb", buffering=4 * 1024 * 1024) as fh:
                self._write_fasta(fh, wrap)
        else:
            self._write_fasta(dest, wrap)

    def _write_fasta(self, fh: BinaryIO, wrap: int | None) -> None:
        for i, taxon in enumerate(self.taxa):
            fh.write(b">" + taxon.encode("utf-8") + b"\n")
            row = self.matrix[i]
            if wrap is None or self.nchar <= wrap:
                fh.write(row)
                fh.write(b"\n")
            else:
                for c0 in range(0, self.nchar, wrap):
                    fh.write(row[c0 : c0 + wrap])
                    fh.write(b"\n")

    def to_fasta_bytes(self, wrap: int | None = None) -> bytes:
        """Serialize to FASTA bytes (convenience; prefer :meth:`write_fasta`)."""
        buf = io.BytesIO()
        self.write_fasta(buf, wrap)
        return buf.getvalue()
