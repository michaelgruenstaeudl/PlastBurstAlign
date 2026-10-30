"""Vectorized per-site statistics and column filters for nucleotide alignments.

All statistics work on a :class:`~plastburstalign.matrix_ops.DenseLocusMatrix`
(or any 2-D ``uint8`` array of raw ASCII bytes), are **case-insensitive**, and
are computed over column blocks so that temporaries stay bounded
(``block_bytes``, default 64 MiB) regardless of alignment size.

Character classes
-----------------
Every byte is mapped (via one 256-entry LUT) to a class:

=========  =============================================  ==================
class      bytes                                          meaning
=========  =============================================  ==================
``A..T``   ``A C G T U`` (and lowercase; ``U`` counts as   unambiguous bases
           ``T``)
ambiguity  IUPAC partial ambiguity ``R Y K M S W B D H V``  ambiguity codes
           and every other byte not listed in this table
``N``      ``N`` and ``X``                                 fully unknown
gap        ``-`` and ``.``                                 gap
missing    ``?``                                           missing data
=========  =============================================  ==================

Definitions
-----------
* **Occupied cell**: a cell that is an unambiguous base. Gaps and ``?`` are
  never occupied. Whether ambiguity codes count as occupied is controlled by
  ``ambiguity_occupied``: ``"none"`` (default; only ``A C G T U`` are
  occupied, so ``N`` runs from assembly gaps behave like missing data),
  ``"iupac"`` (also ``R Y K M S W B D H V`` and other non-N bytes), or
  ``"all"`` (also ``N``/``X``).
* **Occupancy** of a column: occupied cells / number of rows of the matrix
  passed in. For a locus matrix the denominator is the taxa *present in that
  locus*, never the global supermatrix taxon count.
* **Entropy**: Shannon entropy in bits (log2) of the base frequencies of the
  four unambiguous states in the column; ambiguity/gap/missing cells are
  ignored. 0 for monomorphic or empty columns, at most 2.
* **Variable site**: at least two distinct unambiguous states.
* **Parsimony-informative site**: at least two distinct unambiguous states,
  each present in at least two rows.
* **gappyout**: see :func:`gappyout_cutoff` (NOT verified bit-identical to
  trimAl).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from .matrix_ops import DenseLocusMatrix

__all__ = [
    "CLASS_A",
    "CLASS_AMBIGUOUS",
    "CLASS_C",
    "CLASS_G",
    "CLASS_GAP",
    "CLASS_LUT",
    "CLASS_MISSING",
    "CLASS_N",
    "CLASS_T",
    "DEFAULT_BLOCK_BYTES",
    "AmbiguityMode",
    "SiteStats",
    "TrimSpec",
    "cell_class_counts",
    "compute_keep_mask",
    "evaluate_site_occupancy",
    "gappyout_cutoff",
    "site_statistics",
]

U8Array: TypeAlias = npt.NDArray[np.uint8]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
AmbiguityMode: TypeAlias = Literal["none", "iupac", "all"]

CLASS_A = 0
CLASS_C = 1
CLASS_G = 2
CLASS_T = 3
CLASS_AMBIGUOUS = 4
CLASS_N = 5
CLASS_GAP = 6
CLASS_MISSING = 7
_NUM_CLASSES = 8

DEFAULT_BLOCK_BYTES: int = 64 * 1024 * 1024
"""Upper bound for per-block temporaries (see ``site_statistics``)."""

# Conservative bytes of temporaries per matrix cell inside one block: the class
# code array (1 B) plus up to two boolean comparison arrays (1 B each) plus slack.
_BYTES_PER_CELL = 4

_COUNT_CHUNK_BYTES = 4 * 1024 * 1024

_OCCUPIED_MAX_CLASS: dict[str, int] = {
    "none": CLASS_T,
    "iupac": CLASS_AMBIGUOUS,
    "all": CLASS_N,
}


def _build_class_lut() -> U8Array:
    lut = np.full(256, CLASS_AMBIGUOUS, dtype=np.uint8)
    for chars, cls in (
        ("Aa", CLASS_A),
        ("Cc", CLASS_C),
        ("Gg", CLASS_G),
        ("TtUu", CLASS_T),
        ("NnXx", CLASS_N),
        ("-.", CLASS_GAP),
        ("?", CLASS_MISSING),
    ):
        for ch in chars:
            lut[ord(ch)] = cls
    return lut


CLASS_LUT: U8Array = _build_class_lut()
"""256-entry ``byte -> class`` table (see module docstring)."""


def _as_array(matrix: DenseLocusMatrix | U8Array) -> U8Array:
    arr = matrix.matrix if isinstance(matrix, DenseLocusMatrix) else matrix
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError("expected a 2-D uint8 matrix")
    return arr


def _check_mode(mode: str) -> int:
    try:
        return _OCCUPIED_MAX_CLASS[mode]
    except KeyError:
        raise ValueError(
            f"ambiguity_occupied must be one of {sorted(_OCCUPIED_MAX_CLASS)}, "
            f"got {mode!r}"
        ) from None


def _cols_per_block(nrows: int, block_bytes: int) -> int:
    return max(1, block_bytes // (max(1, nrows) * _BYTES_PER_CELL))


@dataclass(frozen=True)
class SiteStats:
    """Per-column statistics of one matrix.

    Attributes:
        nrows: number of rows (taxa) the statistics were computed over.
        ambiguity_occupied: the occupancy convention used.
        occupied: ``int32`` count of occupied cells per column.
        entropy: ``float64`` Shannon entropy in bits per column, or ``None``
            when computed with ``composition=False``.
        variable: boolean variable-site flags (or ``None``).
        informative: boolean parsimony-informative flags (or ``None``).
        gap: ``int32`` count of gap cells (``-`` and ``.``) per column (or ``None``).
        missing: ``int32`` count of ``?`` cells per column (or ``None``).
    """

    nrows: int
    ambiguity_occupied: AmbiguityMode
    occupied: npt.NDArray[np.int32]
    entropy: npt.NDArray[np.float64] | None = None
    variable: BoolArray | None = None
    informative: BoolArray | None = None
    gap: npt.NDArray[np.int32] | None = None
    missing: npt.NDArray[np.int32] | None = None

    @property
    def nchar(self) -> int:
        """Number of columns."""
        return int(self.occupied.shape[0])

    @property
    def occupancy(self) -> npt.NDArray[np.float64]:
        """Occupied fraction per column (0 for a matrix with no rows)."""
        if self.nrows == 0:
            return np.zeros(self.nchar, dtype=np.float64)
        return self.occupied / np.float64(self.nrows)

    @property
    def gap_counts(self) -> npt.NDArray[np.int32]:
        """Unoccupied (gap-like) cells per column: ``nrows - occupied``."""
        return (self.nrows - self.occupied).astype(np.int32)

    @property
    def has_composition(self) -> bool:
        """Whether entropy/variable/informative arrays are available."""
        return self.entropy is not None


def site_statistics(
    matrix: DenseLocusMatrix | U8Array,
    *,
    ambiguity_occupied: AmbiguityMode = "none",
    composition: bool = True,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
) -> SiteStats:
    """Compute per-column statistics in column blocks.

    Args:
        matrix: locus matrix or raw ``uint8`` array (rows = taxa).
        ambiguity_occupied: occupancy convention, see the module docstring.
        composition: also compute entropy, variable and informative flags and
            the per-column gap / ``?`` counts. ``False`` computes occupancy
            only (cheaper).
        block_bytes: bound for per-block temporaries; the block width is
            ``block_bytes // (4 * nrows)`` columns (at least 1).

    Returns:
        A :class:`SiteStats` with one entry per column.
    """
    arr = _as_array(matrix)
    max_class = _check_mode(ambiguity_occupied)
    nrows, ncols = arr.shape
    step = _cols_per_block(nrows, block_bytes)

    occupied = np.zeros(ncols, dtype=np.int32)
    entropy = np.zeros(ncols, dtype=np.float64) if composition else None
    variable = np.zeros(ncols, dtype=np.bool_) if composition else None
    informative = np.zeros(ncols, dtype=np.bool_) if composition else None
    gap = np.zeros(ncols, dtype=np.int32) if composition else None
    missing = np.zeros(ncols, dtype=np.int32) if composition else None

    for c0 in range(0, ncols, step):
        c1 = min(ncols, c0 + step)
        codes = CLASS_LUT[arr[:, c0:c1]]
        occupied[c0:c1] = (codes <= max_class).sum(axis=0, dtype=np.int32)
        if not composition:
            continue
        assert entropy is not None
        assert variable is not None
        assert informative is not None
        assert gap is not None
        assert missing is not None
        gap[c0:c1] = (codes == CLASS_GAP).sum(axis=0, dtype=np.int32)
        missing[c0:c1] = (codes == CLASS_MISSING).sum(axis=0, dtype=np.int32)
        counts = np.empty((4, c1 - c0), dtype=np.int32)
        for state in range(4):
            counts[state] = (codes == state).sum(axis=0, dtype=np.int32)
        nstates = (counts > 0).sum(axis=0)
        total = counts.sum(axis=0, dtype=np.int64)
        cf = counts.astype(np.float64)
        safe = np.where(counts > 0, counts, 1).astype(np.float64)
        xlogx = (cf * np.log2(safe)).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = np.where(
                nstates > 1,
                np.log2(np.maximum(total, 1)) - xlogx / np.maximum(total, 1),
                0.0,
            )
        entropy[c0:c1] = np.maximum(ent, 0.0)
        variable[c0:c1] = nstates >= 2
        informative[c0:c1] = (counts >= 2).sum(axis=0) >= 2

    return SiteStats(
        nrows=nrows,
        ambiguity_occupied=ambiguity_occupied,
        occupied=occupied,
        entropy=entropy,
        variable=variable,
        informative=informative,
        gap=gap,
        missing=missing,
    )


def cell_class_counts(
    matrix: DenseLocusMatrix | U8Array, *, block_bytes: int = DEFAULT_BLOCK_BYTES
) -> npt.NDArray[np.int64]:
    """Total number of cells per character class (length-8 ``int64`` vector).

    Index with the ``CLASS_*`` constants; processed in row blocks so that the
    temporaries stay within ``min(block_bytes, 4 MiB)``.
    """
    arr = _as_array(matrix)
    nrows, ncols = arr.shape
    totals = np.zeros(_NUM_CLASSES, dtype=np.int64)
    if arr.size == 0:
        return totals
    # bincount casts to intp (8 bytes/cell) on top of the 1-byte class codes.
    rows = max(1, min(block_bytes, _COUNT_CHUNK_BYTES) // (max(1, ncols) * 9))
    for r0 in range(0, nrows, rows):
        codes = CLASS_LUT[arr[r0 : r0 + rows]]
        totals += np.bincount(codes.ravel(), minlength=_NUM_CLASSES)
    return totals


def evaluate_site_occupancy(
    column: str | bytes | bytearray | Sequence[str] | U8Array,
    min_occupancy: float = 0.5,
    *,
    ambiguity_occupied: AmbiguityMode = "none",
) -> bool:
    """Whether one alignment column is occupied enough to be kept.

    Args:
        column: the cells of one column (string, bytes, sequence of 1-char
            strings, or ``uint8`` array); case-insensitive.
        min_occupancy: minimum occupied fraction, inclusive (``>=``).
        ambiguity_occupied: occupancy convention (module docstring).

    Returns:
        ``True`` if ``occupied / len(column) >= min_occupancy``; ``False`` for an
        empty column.
    """
    max_class = _check_mode(ambiguity_occupied)
    if isinstance(column, str):
        data = np.frombuffer(column.encode("latin-1", errors="replace"), np.uint8)
    elif isinstance(column, (bytes, bytearray)):
        data = np.frombuffer(bytes(column), np.uint8)
    elif isinstance(column, np.ndarray):
        data = column.astype(np.uint8, copy=False).ravel()
    else:
        joined = "".join(column)
        data = np.frombuffer(joined.encode("latin-1", errors="replace"), np.uint8)
    if data.size == 0:
        return False
    occupied = int((CLASS_LUT[data] <= max_class).sum())
    return occupied >= min_occupancy * data.size - 1e-9


def gappyout_cutoff(
    gap_counts: npt.ArrayLike, ntax: int, *, min_deviation: float = 0.1
) -> int:
    """Automatic gap threshold from the sorted per-column gap-count curve.

    **Re-implementation in the spirit of trimAl's ``-gappyout``; it is NOT
    verified to be bit-identical to trimAl** (trimAl was not available to diff
    against). What is implemented, exactly:

    1. ``gap_counts[i]`` is the number of unoccupied (gap/missing, and per the
       ambiguity convention possibly ambiguous) cells of column ``i``.
    2. Columns whose gap count equals ``ntax`` (entirely gap/missing) are
       always rejected and excluded from the threshold search.
    3. The remaining ``m`` counts are sorted ascending, giving the monotone
       "gap profile" ``s[0..m-1]`` (flat where columns are well covered,
       climbing where they are gappy).
    4. The profile's chord joins ``(0, s[0])`` and ``(m-1, s[m-1])``. The
       **knee** is the rank ``k`` where the profile falls furthest below the
       chord, i.e. where its local slope crosses the chord's mean slope --
       the slope-change point between the well-covered plateau and the
       gappy tail. Ties pick the last index of a plateau (the first maximum
       of the deviation), so all columns sharing the knee's gap count are
       kept.
    5. If the largest deviation is below ``min_deviation * (s[m-1] - s[0])``
       (no pronounced knee, e.g. a uniform ramp) or all counts are equal,
       no threshold is applied: every non-all-gap column is kept.

    Args:
        gap_counts: per-column gap counts.
        ntax: number of rows the counts refer to.
        min_deviation: minimal knee depth as a fraction of the gap range.

    Returns:
        The maximum gap count a retained column may have; ``-1`` if every
        column is entirely gap/missing (nothing can be retained).
    """
    counts = np.asarray(gap_counts, dtype=np.int64).ravel()
    eligible = counts[counts < ntax]
    m = int(eligible.size)
    if m == 0:
        return -1
    ordered = np.sort(eligible)
    low = int(ordered[0])
    high = int(ordered[-1])
    if high == low or m < 3:
        return high
    ranks = np.arange(m, dtype=np.float64)
    chord = low + (high - low) * ranks / (m - 1)
    deviation = chord - ordered
    knee = int(np.argmax(deviation))
    if deviation[knee] < min_deviation * (high - low):
        return high
    return int(ordered[knee])


_TOKEN_OCC = re.compile(r"^(?:occupancy|min-occupancy)=(?P<v>.+)$")
_TOKEN_ENT = re.compile(r"^entropy=(?P<lo>[^:]*)(?::(?P<hi>.*))?$")


@dataclass(frozen=True)
class TrimSpec:
    """A conjunction of column-retention criteria.

    Grammar of the string form (case-insensitive, criteria joined by ``+``)::

        none
        occupancy=<fraction>          keep columns with occupancy >= fraction
        entropy=<min>[:<max>]         keep columns with min <= H <= max (bits);
                                      ``entropy=:<max>`` sets only the upper bound
        informative-only              keep parsimony-informative columns only
        gappyout                      automatic gap threshold (gappyout_cutoff)

    ``occupancy=0.5+entropy=0.1+informative-only`` keeps a column only if all
    three criteria hold. ``none`` (the empty spec) cannot be combined.
    """

    min_occupancy: float | None = None
    entropy_min: float | None = None
    entropy_max: float | None = None
    informative_only: bool = False
    gappyout: bool = False
    ambiguity_occupied: AmbiguityMode = "none"

    def __post_init__(self) -> None:
        _check_mode(self.ambiguity_occupied)
        if self.min_occupancy is not None and not 0.0 <= self.min_occupancy <= 1.0:
            raise ValueError(
                f"occupancy must be within [0, 1], got {self.min_occupancy}"
            )
        for label, value in (("min", self.entropy_min), ("max", self.entropy_max)):
            if value is not None and (math.isnan(value) or value < 0):
                raise ValueError(f"entropy {label} must be >= 0, got {value}")
        if (
            self.entropy_min is not None
            and self.entropy_max is not None
            and self.entropy_min > self.entropy_max
        ):
            raise ValueError(
                f"entropy min ({self.entropy_min}) exceeds max ({self.entropy_max})"
            )

    @classmethod
    def parse(
        cls,
        text: str | TrimSpec | None,
        *,
        ambiguity_occupied: AmbiguityMode | None = None,
    ) -> TrimSpec:
        """Parse the string form; ``None``/``""``/``"none"`` give the empty spec.

        Args:
            text: string spec, an existing :class:`TrimSpec`, or ``None``.
            ambiguity_occupied: override of the occupancy convention.

        Raises:
            ValueError: unknown token, repeated criterion, bad number, or
                ``none`` combined with other criteria.
        """
        if isinstance(text, TrimSpec):
            spec = text
        else:
            spec = cls._parse_text(text)
        if ambiguity_occupied is not None:
            spec = replace(spec, ambiguity_occupied=ambiguity_occupied)
        return spec

    @classmethod
    def _parse_text(cls, text: str | None) -> TrimSpec:
        if text is None or not text.strip():
            return cls()
        tokens = [t.strip().lower().replace("_", "-") for t in text.split("+")]
        if "none" in tokens:
            if len(tokens) != 1:
                raise ValueError("'none' cannot be combined with other criteria")
            return cls()
        fields: dict[str, object] = {}

        def put(key: str, value: object) -> None:
            if key in fields:
                raise ValueError(f"criterion {key!r} given more than once in {text!r}")
            fields[key] = value

        for token in tokens:
            if not token:
                raise ValueError(f"empty criterion in trim spec {text!r}")
            occ = _TOKEN_OCC.match(token)
            ent = _TOKEN_ENT.match(token)
            if occ:
                put("occupancy", cls._number(occ["v"], token))
            elif ent:
                lo, hi = ent["lo"], ent["hi"]
                if not lo and not hi:
                    raise ValueError(f"entropy needs a min and/or max: {token!r}")
                put(
                    "entropy",
                    (
                        cls._number(lo, token) if lo else None,
                        cls._number(hi, token) if hi else None,
                    ),
                )
            elif token == "informative-only":
                put("informative", True)
            elif token == "gappyout":
                put("gappyout", True)
            else:
                raise ValueError(
                    f"unknown trim criterion {token!r}; expected none, "
                    "occupancy=<f>, entropy=<min>[:<max>], informative-only, gappyout"
                )
        entropy = fields.get("entropy", (None, None))
        assert isinstance(entropy, tuple)
        occupancy = fields.get("occupancy")
        return cls(
            min_occupancy=occupancy if isinstance(occupancy, float) else None,
            entropy_min=entropy[0],
            entropy_max=entropy[1],
            informative_only=bool(fields.get("informative", False)),
            gappyout=bool(fields.get("gappyout", False)),
        )

    @staticmethod
    def _number(raw: str, token: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"invalid number in trim criterion {token!r}") from None
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"invalid number in trim criterion {token!r}")
        return value

    @property
    def is_noop(self) -> bool:
        """``True`` if the spec keeps every column."""
        return (
            self.min_occupancy is None
            and self.entropy_min is None
            and self.entropy_max is None
            and not self.informative_only
            and not self.gappyout
        )

    @property
    def needs_composition(self) -> bool:
        """Whether entropy/informative statistics are required."""
        return (
            self.entropy_min is not None
            or self.entropy_max is not None
            or self.informative_only
        )

    def __str__(self) -> str:
        parts: list[str] = []
        if self.min_occupancy is not None:
            parts.append(f"occupancy={self.min_occupancy:g}")
        if self.entropy_min is not None or self.entropy_max is not None:
            lo = "" if self.entropy_min is None else f"{self.entropy_min:g}"
            hi = "" if self.entropy_max is None else f":{self.entropy_max:g}"
            parts.append(f"entropy={lo}{hi}")
        if self.informative_only:
            parts.append("informative-only")
        if self.gappyout:
            parts.append("gappyout")
        return "+".join(parts) if parts else "none"


def compute_keep_mask(
    matrix: DenseLocusMatrix | U8Array,
    spec: TrimSpec,
    *,
    stats: SiteStats | None = None,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
) -> BoolArray:
    """Boolean column mask (``True`` = keep) implementing ``spec``.

    Args:
        matrix: locus matrix or ``uint8`` array.
        spec: the trimming criteria (all must hold).
        stats: optionally pre-computed statistics to avoid a second pass; they
            must use ``spec.ambiguity_occupied`` and, for entropy/informative
            criteria, include composition.
        block_bytes: bound for temporaries (see :func:`site_statistics`).
    """
    arr = _as_array(matrix)
    nrows, ncols = arr.shape
    if spec.is_noop:
        return np.ones(ncols, dtype=np.bool_)
    if nrows == 0:
        return np.zeros(ncols, dtype=np.bool_)

    if stats is None:
        stats = site_statistics(
            arr,
            ambiguity_occupied=spec.ambiguity_occupied,
            composition=spec.needs_composition,
            block_bytes=block_bytes,
        )
    else:
        if stats.ambiguity_occupied != spec.ambiguity_occupied:
            raise ValueError(
                "stats were computed with a different ambiguity convention"
            )
        if stats.nchar != ncols or stats.nrows != nrows:
            raise ValueError("stats do not match the matrix shape")
        if spec.needs_composition and not stats.has_composition:
            raise ValueError("spec needs composition statistics; recompute stats")

    keep = np.ones(ncols, dtype=np.bool_)
    if spec.min_occupancy is not None:
        keep &= stats.occupied >= spec.min_occupancy * nrows - 1e-9
    if spec.entropy_min is not None or spec.entropy_max is not None:
        assert stats.entropy is not None
        if spec.entropy_min is not None:
            keep &= stats.entropy >= spec.entropy_min - 1e-12
        if spec.entropy_max is not None:
            keep &= stats.entropy <= spec.entropy_max + 1e-12
    if spec.informative_only:
        assert stats.informative is not None
        keep &= stats.informative
    if spec.gappyout:
        cutoff = gappyout_cutoff(stats.gap_counts, nrows)
        keep &= stats.gap_counts <= cutoff
        keep &= stats.gap_counts < nrows
    return keep
