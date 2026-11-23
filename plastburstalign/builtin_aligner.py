"""Pure-Python fallback aligner (deterministic *center-star*), no external binary.

PlastBurstAlign bundles no aligner executables. When the resolver
(:mod:`plastburstalign.aligner_resolver`) finds no working MAFFT/MUSCLE/Clustal,
``BuiltinAligner`` keeps the pipeline usable on small-to-medium loci. It is a
**lower-accuracy** stand-in and says so loudly (once per process).

Algorithm
---------
1. Parse FASTA (headers and input order preserved; alignment gaps ``-``/``.``
   and whitespace are removed from the input, everything else - case, IUPAC
   codes, ``*`` - is kept and reproduced verbatim in the output).
2. Pick the *center* deterministically: take an order-independent sample of up
   to ``center_sample_size`` sequences (evenly spaced ranks by ``(length, id)``),
   score all sample pairs (global, affine gaps) and choose the highest summed
   length-normalised similarity; ties break on the FASTA id, then the index.
3. Globally align every sequence to the center with Biopython's C
   ``Bio.Align.PairwiseAligner`` (NUC.4.4 for nucleotides, BLOSUM62 for
   proteins, affine gaps).
4. Merge the pairwise alignments by the classic *once a gap, always a gap* rule:
   each inter-center slot is as wide as the longest insertion any sequence
   needs there (insertions are left-justified inside their slot).

Properties: the ungapped output row equals the (gap-stripped) input, all rows
have equal length, ids and order are preserved, and the result is bit-for-bit
reproducible. Exact permutation invariance is *not* guaranteed - reversing the
input order may change the center only in tie situations and the left-justified
slot layout can differ slightly, so alignment lengths agree only approximately.

Hard size guards (``alignment_tool_config``: ``max_sequences``,
``max_total_cells``, ``max_pair_cells``) make ``align`` return ``False`` with an
actionable message instead of running for hours.
"""

from __future__ import annotations

import functools
import itertools
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from Bio.Align import PairwiseAligner, substitution_matrices

from .alignment_tools import AlignmentTool
from .logging_ops import logger as log

__all__ = [
    "BuiltinAligner",
    "BuiltinAlignerConfig",
    "SizeGuardError",
    "center_star_align",
]

_GAP_BYTE = ord("-")
_INSTALL_HINT = (
    "Install MAFFT for accurate alignments: `conda install -c bioconda mafft`, "
    "`apt install mafft` or `brew install mafft`."
)
_GAP_CHARS = b"-. \t\r\n"

#: Score-only alignment is roughly this many times faster than traceback alignment.
_SCORE_SPEEDUP = 3.0

_QUALITY_WARNED = False

IntArray = npt.NDArray[np.int64]


class SizeGuardError(RuntimeError):
    """Raised internally when an input exceeds a configured size guard."""


@dataclass(frozen=True)
class BuiltinAlignerConfig:
    """Tunables (all overridable through ``alignment_tool_config``).

    PROVISIONAL defaults sized so one locus finishes in about a minute on a
    laptop (Biopython aligns about 9e7 DP cells/s with traceback).

    Attributes:
        max_sequences: Refuse loci with more sequences.
        max_total_cells: Refuse loci whose summed pairwise DP cells
            (``sum_i (len(center)+1) * (len(seq_i)+1)``) exceed this.
        max_pair_cells: Refuse a single pair above this many cells
            (traceback memory is about 3 bytes per cell).
        center_sample_size: Sequences considered when choosing the center.
        gap_open / gap_extend: Affine gap scores (``None`` = alphabet default).
        line_width: FASTA line width of the output (``0`` = unwrapped).
        alphabet: ``"auto"`` (``prot_`` filename prefix => protein), ``"nucleotide"``
            or ``"protein"``.
    """

    max_sequences: int = 2000
    max_total_cells: float = 5.0e9
    max_pair_cells: float = 6.0e7
    center_sample_size: int = 8
    gap_open: float | None = None
    gap_extend: float | None = None
    line_width: int = 60
    alphabet: str = "auto"

    @classmethod
    def from_params(
        cls, params: dict[str, Any], overrides: dict[str, Any] | None = None
    ) -> BuiltinAlignerConfig:
        """Build from an ``alignment_tool_config`` dict plus per-call overrides."""
        merged = {**params, **(overrides or {})}
        kwargs: dict[str, Any] = {}
        for field_name, caster in (
            ("max_sequences", int),
            ("max_total_cells", float),
            ("max_pair_cells", float),
            ("center_sample_size", int),
            ("line_width", int),
        ):
            if merged.get(field_name) is not None:
                kwargs[field_name] = caster(merged[field_name])
        for field_name in ("gap_open", "gap_extend"):
            if merged.get(field_name) is not None:
                kwargs[field_name] = float(merged[field_name])
        if merged.get("alphabet"):
            kwargs["alphabet"] = str(merged["alphabet"]).lower()
        if int(kwargs.get("center_sample_size", 1)) < 1:
            kwargs["center_sample_size"] = 1
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Scoring setup
# --------------------------------------------------------------------------- #


class _Scoring(NamedTuple):
    aligner: Any  # Bio.Align.PairwiseAligner
    table: bytes  # 256-entry translation: raw byte -> scoring symbol


@functools.lru_cache(maxsize=2)
def _load_matrix(protein: bool) -> tuple[Any, bytes]:
    """Substitution matrix and the byte -> scoring-symbol translation table."""
    matrix = substitution_matrices.load("BLOSUM62" if protein else "NUC.4.4")
    alphabet = set(str(matrix.alphabet))
    table = bytearray(256)
    for byte in range(256):
        ch = chr(byte).upper()
        if ch in alphabet:
            sym = ch
        elif protein:
            sym = {"U": "C", "O": "K"}.get(ch, "X")
        else:
            sym = "T" if ch == "U" else "N"
        table[byte] = ord(sym if sym in alphabet else ("X" if protein else "N"))
    return matrix, bytes(table)


def _build_scoring(protein: bool, cfg: BuiltinAlignerConfig) -> _Scoring:
    matrix, table = _load_matrix(protein)
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = matrix
    default_open, default_extend = (-11.0, -1.0) if protein else (-10.0, -0.5)
    aligner.open_gap_score = cfg.gap_open if cfg.gap_open is not None else default_open
    aligner.extend_gap_score = (
        cfg.gap_extend if cfg.gap_extend is not None else default_extend
    )
    return _Scoring(aligner, table)


# --------------------------------------------------------------------------- #
# Core algorithm
# --------------------------------------------------------------------------- #


class _Pairwise(NamedTuple):
    """Alignment of one sequence to the center (all arrays int64)."""

    match_t: IntArray  # center start of each aligned block
    match_q: IntArray  # query start of each aligned block
    match_len: IntArray
    ins_slot: IntArray  # center position the insertion precedes
    ins_q: IntArray
    ins_len: IntArray


_EMPTY = np.zeros(0, dtype=np.int64)
_EMPTY_PAIRWISE = _Pairwise(_EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY, _EMPTY)


def _pairwise_from_coordinates(coords: Any) -> _Pairwise:
    """Convert ``alignment.coordinates`` (2 x k path) into blocks and insertions."""
    path = np.asarray(coords, dtype=np.int64)
    t, q = path[0], path[1]
    dt, dq = np.diff(t), np.diff(q)
    diag = (dt > 0) & (dq > 0)
    ins = (dt == 0) & (dq > 0)
    return _Pairwise(
        t[:-1][diag], q[:-1][diag], dt[diag], t[:-1][ins], q[:-1][ins], dq[ins]
    )


def _pick_center(
    scoring_seqs: list[str],
    ids: list[bytes],
    lengths: list[int],
    scoring: _Scoring,
    sample_size: int,
) -> int:
    """Deterministic center: highest summed normalised similarity on a sample."""
    nonempty = [i for i, n in enumerate(lengths) if n > 0]
    if not nonempty:
        return 0
    ranked = sorted(nonempty, key=lambda i: (lengths[i], ids[i], i))
    if len(ranked) <= sample_size:
        sample = ranked
    else:
        step = len(ranked) / sample_size
        sample = [ranked[int(k * step + step / 2)] for k in range(sample_size)]
        sample = list(dict.fromkeys(sample))
    if len(sample) == 1:
        return sample[0]
    totals = {i: 0.0 for i in sample}
    for a, b in itertools.combinations(sample, 2):
        score = float(scoring.aligner.score(scoring_seqs[a], scoring_seqs[b]))
        sim = score / max(lengths[a], lengths[b], 1)
        totals[a] += sim
        totals[b] += sim
    return min(sample, key=lambda i: (-round(totals[i], 9), ids[i], i))


@dataclass(frozen=True)
class _StarPlan:
    """Everything needed to render the merged alignment row by row."""

    raw: list[bytes]
    center: int
    pairwise: list[_Pairwise]
    slot_start: IntArray
    center_col: IntArray
    total_cols: int


def _plan_center_star(
    raw: list[bytes],
    ids: list[bytes],
    scoring: _Scoring,
    cfg: BuiltinAlignerConfig,
) -> _StarPlan:
    n = len(raw)
    lengths = [len(s) for s in raw]
    if n > cfg.max_sequences:
        raise SizeGuardError(f"{n} sequences exceed max_sequences={cfg.max_sequences}")
    positive = sorted(x for x in lengths if x > 0)
    median = positive[len(positive) // 2] if positive else 0
    sample = min(cfg.center_sample_size, max(1, len(positive)))
    sample_cost = sample * (sample - 1) / 2 * (median + 1) ** 2 / _SCORE_SPEEDUP
    rough = (n - 1) * (median + 1) ** 2 + sample_cost
    if rough > cfg.max_total_cells:
        raise SizeGuardError(
            f"estimated {rough:.2e} DP cells for {n} sequences (median length "
            f"{median}) exceed max_total_cells={cfg.max_total_cells:.2e}"
        )
    scoring_seqs = [s.translate(scoring.table).decode("ascii") for s in raw]
    center = _pick_center(scoring_seqs, ids, lengths, scoring, cfg.center_sample_size)
    lc = lengths[center]

    exact = sum((lc + 1) * (lengths[i] + 1) for i in range(n) if i != center)
    worst = max(
        ((lc + 1) * (lengths[i] + 1) for i in range(n) if i != center), default=0
    )
    if exact > cfg.max_total_cells:
        raise SizeGuardError(
            f"{exact:.2e} DP cells (center length {lc}) exceed "
            f"max_total_cells={cfg.max_total_cells:.2e}"
        )
    if worst > cfg.max_pair_cells:
        raise SizeGuardError(
            f"a single pairwise alignment needs {worst:.2e} cells "
            f"(> max_pair_cells={cfg.max_pair_cells:.2e}; traceback memory about "
            f"{worst * 3 / 2**30:.1f} GiB)"
        )

    pairwise: list[_Pairwise] = []
    max_ins = np.zeros(lc + 1, dtype=np.int64)
    for i in range(n):
        if i == center or lengths[i] == 0:
            pw = _EMPTY_PAIRWISE  # center row / empty sequence: no pairwise work
        else:
            alignments = scoring.aligner.align(scoring_seqs[center], scoring_seqs[i])
            best = next(iter(alignments))
            pw = _pairwise_from_coordinates(best.coordinates)
        if pw.ins_slot.size:
            max_ins[pw.ins_slot] = np.maximum(max_ins[pw.ins_slot], pw.ins_len)
        pairwise.append(pw)

    inc = max_ins[:lc] + 1
    slot_start = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(inc)])
    center_col = slot_start[:lc] + max_ins[:lc]
    total = int(slot_start[lc] + max_ins[lc])
    return _StarPlan(raw, center, pairwise, slot_start, center_col, total)


def _render_row(plan: _StarPlan, index: int) -> bytes:
    """Gapped row of sequence ``index`` (ungapped it equals the input)."""
    row = np.full(plan.total_cols, _GAP_BYTE, dtype=np.uint8)
    residues = np.frombuffer(plan.raw[index], dtype=np.uint8)
    if index == plan.center:
        if residues.size:
            row[plan.center_col] = residues
        return row.tobytes()
    pw = plan.pairwise[index]
    for t, q, length in zip(
        pw.match_t.tolist(), pw.match_q.tolist(), pw.match_len.tolist()
    ):
        row[plan.center_col[t : t + length]] = residues[q : q + length]
    for slot, q, length in zip(
        pw.ins_slot.tolist(), pw.ins_q.tolist(), pw.ins_len.tolist()
    ):
        start = int(plan.slot_start[slot])
        row[start : start + length] = residues[q : q + length]
    return row.tobytes()


def _render_rows(plan: _StarPlan) -> Iterator[bytes]:
    for index in range(len(plan.raw)):
        yield _render_row(plan, index)


def _strip_gaps(seq: bytes) -> bytes:
    return seq.translate(None, _GAP_CHARS)


def center_star_align(
    sequences: Sequence[str],
    *,
    protein: bool = False,
    config: BuiltinAlignerConfig | None = None,
    ids: Sequence[str] | None = None,
) -> list[str]:
    """Align in-memory sequences with the builtin center-star algorithm.

    Args:
        sequences: Unaligned sequences (gap characters are removed).
        protein: Use BLOSUM62 instead of the nucleotide matrix.
        config: Tunables / guards (defaults otherwise).
        ids: Optional identifiers used only for deterministic tie-breaking.

    Returns:
        Gapped sequences, same order, all of equal length.

    Raises:
        SizeGuardError: If a size guard is exceeded.
    """
    cfg = config or BuiltinAlignerConfig()
    raw = [_strip_gaps(s.encode("latin-1")) for s in sequences]
    id_bytes = [
        (ids[i] if ids is not None else str(i)).encode("latin-1")
        for i in range(len(raw))
    ]
    if not raw:
        return []
    plan = _plan_center_star(raw, id_bytes, _build_scoring(protein, cfg), cfg)
    return [row.decode("latin-1") for row in _render_rows(plan)]


# --------------------------------------------------------------------------- #
# FASTA I/O
# --------------------------------------------------------------------------- #


def _read_fasta(path: str) -> tuple[list[bytes], list[bytes]]:
    """Return ``(headers, gap-stripped sequences)`` preserving file order."""
    headers: list[bytes] = []
    chunks: list[list[bytes]] = []
    with open(path, "rb") as handle:
        for line in handle:
            line = line.rstrip(b"\r\n")
            if line.startswith(b">"):
                headers.append(line[1:])
                chunks.append([])
            elif line.strip():
                if not headers:
                    raise ValueError("sequence data before the first FASTA header")
                chunks[-1].append(line)
    return headers, [_strip_gaps(b"".join(c)) for c in chunks]


def _fasta_id(header: bytes) -> bytes:
    parts = header.split(None, 1)
    return parts[0] if parts else b""


def _write_fasta(
    path: str, headers: list[bytes], rows: Iterator[bytes], width: int
) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".pba_builtin_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as out:
            for header, row in zip(headers, rows):
                out.write(b">" + header + b"\n")
                if width and width > 0:
                    for start in range(0, len(row), width):
                        out.write(row[start : start + width] + b"\n")
                    if not row:
                        out.write(b"\n")
                else:
                    out.write(row + b"\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------- #
# AlignmentTool implementation
# --------------------------------------------------------------------------- #


class BuiltinAligner(AlignmentTool):
    """Deterministic center-star aligner used when no external tool is available.

    Same ``align()`` contract as the other tools. ``threads`` is ignored: the
    algorithm is single-threaded (see :attr:`max_useful_threads`, a hint for the
    scheduler's core leasing).
    """

    #: Scheduler hint: leasing more than one core to this tool is wasted.
    max_useful_threads = 1

    #: Marker so callers can distinguish the fallback from real aligners.
    is_fallback = True

    def __init__(
        self,
        user_params: dict[str, Any] | None = None,
        tool_path: str | None = None,
    ) -> None:
        if tool_path:
            log.warning(
                f"BuiltinAligner ignores tool_path={tool_path!r} (it is pure Python)"
            )
        super().__init__(user_params, None)

    def _find_tool(self) -> None:
        """The builtin aligner is always available."""
        self.exec_path = "builtin"

    @classmethod
    def reset_quality_warning(cls) -> None:
        """Re-arm the one-time quality warning (used by tests)."""
        global _QUALITY_WARNED
        _QUALITY_WARNED = False

    @staticmethod
    def _warn_quality_once() -> None:
        global _QUALITY_WARNED
        if _QUALITY_WARNED:
            return
        _QUALITY_WARNED = True
        log.warning(
            "USING THE BUILT-IN FALLBACK ALIGNER (center-star): no working MAFFT, "
            "MUSCLE or Clustal Omega was found. Alignments are deterministic but "
            f"LOWER ACCURACY than MAFFT and are limited in size. {_INSTALL_HINT}"
        )

    def config_for(self, **overrides: Any) -> BuiltinAlignerConfig:
        """Effective config: defaults < ``alignment_tool_config`` < ``overrides``."""
        return BuiltinAlignerConfig.from_params(self.tool_params, overrides)

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        **kwargs: Any,
    ) -> bool:
        """Align ``input_file`` to ``output_file`` (aligned FASTA).

        Args:
            input_file: Unaligned FASTA.
            output_file: Destination FASTA (written atomically).
            threads: Ignored (single-threaded).
            sequence_count: Optional hint; used to fail fast on huge inputs.
            **kwargs: ``alphabet``, ``max_sequences``, ``max_total_cells``,
                ``max_pair_cells``, ``center_sample_size``, ``gap_open``,
                ``gap_extend``, ``line_width`` override the configuration;
                ``max_len``/``min_len`` (store hints) are accepted and ignored.

        Returns:
            ``True`` on success; ``False`` (with an explanatory log message)
            on invalid input, a tripped size guard or any error.
        """
        if not self._validate_input(input_file, output_file):
            return False
        self._warn_quality_once()
        config_keys = {
            "alphabet",
            "max_sequences",
            "max_total_cells",
            "max_pair_cells",
            "center_sample_size",
            "gap_open",
            "gap_extend",
            "line_width",
        }
        overrides = {k: v for k, v in kwargs.items() if k in config_keys}
        cfg = self.config_for(**overrides)
        if self.tool_params.get("adjustdirection"):
            log.warning(
                "builtin aligner does not support adjustdirection; sequences are "
                "aligned in the orientation given"
            )
        if threads and threads > 1:
            log.debug(f"builtin aligner is single-threaded; ignoring threads={threads}")
        name = os.path.basename(input_file)
        try:
            if cfg.alphabet in {"protein", "prot", "aa"}:
                protein = True
            elif cfg.alphabet in {"nucleotide", "dna", "rna", "nuc"}:
                protein = False
            else:
                protein = name.startswith("prot_")
            if sequence_count is not None and sequence_count > cfg.max_sequences:
                raise SizeGuardError(
                    f"{sequence_count} sequences exceed max_sequences={cfg.max_sequences}"
                )
            headers, raw = _read_fasta(input_file)
            if not raw:
                log.error(f"builtin aligner: no sequences in {name}")
                return False
            if len(raw) == 1:
                _write_fasta(output_file, headers, iter(raw), cfg.line_width)
                return True
            ids = [_fasta_id(h) for h in headers]
            plan = _plan_center_star(raw, ids, _build_scoring(protein, cfg), cfg)
            _write_fasta(output_file, headers, _render_rows(plan), cfg.line_width)
            log.info(
                f"      aligned {name} with the builtin center-star aligner "
                f"({len(raw)} sequences x {plan.total_cols} columns)"
            )
            return True
        except SizeGuardError as exc:
            log.error(
                f"builtin fallback aligner refuses {name}: {exc}. This fallback is "
                f"only meant for small loci. {_INSTALL_HINT} Alternatively raise the "
                "limit via alignment_tool_config (max_sequences / max_total_cells)."
            )
            return False
        except Exception as exc:
            log.error(f"builtin aligner failed on {name}: {type(exc).__name__}: {exc}")
            return False
