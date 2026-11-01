"""Pre-flight estimates: what will this run cost *before* it starts?

:func:`preflight_estimate` predicts, from cheap inputs only (the per-locus
``count / min_len / max_len`` summary of the feature store, the tuning plan and
the mode), the algorithm tier, single-core seconds, wall time, peak RAM, disk
footprint and scratch need of an alignment run, and warns about the usual ways
a run goes wrong (memory envelope, tiny RAM-disk, multi-hour wall time, ...).

.. warning::
   **Every number is an estimate from an uncalibrated model.** All constants are
   PROVISIONAL analytic placeholders in the frozen dataclasses
   :class:`TimeModel` and :class:`FootprintModel` (``calibrated=False``); the
   memory side reuses :class:`~plastburstalign.autotune.MemoryModel`. Results
   are labelled ``"estimate (model not yet calibrated)"`` until a calibration
   run supplies measured constants (pass them with ``constants=`` / ``footprint=``;
   see :class:`TimeModel` for exactly what has to be measured).

Pipeline of the prediction
--------------------------
1. **Per locus** the alignment alphabet and length are fixed (``cds`` aligns
   *proteins*, length ``max_len // 3`` unless ``prot_summary`` is given; every
   other mode aligns nucleotides), then
   :func:`~plastburstalign.strategy.determine_alignment_strategy` picks the
   tier and work ``W`` exactly as the real run will (pure function of the data).
2. **Single-core seconds** come from :meth:`TimeModel.align_seconds`: a
   piecewise-linear, *continuous and monotone* function of ``W`` whose marginal
   cost per work unit drops from tier to tier (cheaper heuristics), plus fixed
   per-locus overhead, I/O and (cds) back-translation.
3. **Chunk fan-out**: with ``subdivide=True`` a locus whose alignment length
   reaches :class:`~plastburstalign.chunking.ChunkConfig`'s ``min_locus_len``
   (nucleotide 5000, protein 1700) is predicted to split into
   ``max(2, round(L / target_chunk_len))`` equal chunks with their own tier,
   plus a planning unit and a splice finalizer (deterministic, data only; the
   real plan can have fewer chunks if no anchor chain exists).
4. **Wall time** is the makespan of
   :func:`~plastburstalign.scheduler_sim.simulate` (``locus_scheduler`` policy)
   over those tasks on the plan's cores (Amdahl speedup per task), plus the
   serial supermatrix concatenation and a fixed start-up cost. If memory limits
   the plan to fewer workers than cores, the simulation is repeated with
   ``workers`` cores and the larger makespan wins.
5. **Peak RAM** is the autotune blueprint ``max_i(MAFFT_i) + p x parser_buffer
   + supermatrix`` with ``p = plan.workers`` (the pessimistic all-concurrent
   bound is reported too).
6. **Disk** and **scratch**: unaligned/aligned FASTA, per-locus NEXUS, SQLite
   store and concatenated outputs; scratch need is ``3 x`` the in-flight
   unaligned bytes (the scratch engine's rule) for the ``workers`` largest loci.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .autotune import (
    MIN_RAM_SCRATCH_FREE_BYTES,
    MemoryModel,
    TuningPlan,
    est_mem_bytes,
    estimate_mafft_peak_bytes,
    estimate_parser_buffer_bytes,
    estimate_supermatrix_bytes,
)
from .chunking import ChunkConfig
from .ephemeral_engine import DEFAULT_MIN_HEADROOM_BYTES, SCRATCH_BYTES_FACTOR
from .hardware import format_bytes
from .scaling import format_duration
from .scheduler_sim import Policy, SimTask, simulate
from .strategy import AlignmentTier, determine_alignment_strategy
from .strategy_constants import DEFAULT_CONSTANTS, StrategyConstants

if TYPE_CHECKING:
    from rich.console import Console

__all__ = [
    "DEFAULT_FOOTPRINT_MODEL",
    "DEFAULT_TIME_MODEL",
    "LABEL_UNCALIBRATED",
    "DiskFootprint",
    "EstimateWarning",
    "FootprintModel",
    "LocusEstimate",
    "PreflightEstimate",
    "ScratchEstimate",
    "TimeModel",
    "predicted_chunks",
    "preflight_estimate",
    "render_preflight",
]

#: Label attached to every estimate while any model constant is provisional.
LABEL_UNCALIBRATED = "estimate (model not yet calibrated)"
_LABEL_CALIBRATED = "estimate (calibrated model)"

_MIB = 1024**2
_NUCL = "nucl"
_PROT = "prot"

#: Warn when the predicted peak RAM exceeds this fraction of the memory envelope.
RAM_WARN_FRACTION: float = 0.85
#: Warn when the predicted wall time exceeds this many seconds.
LONG_WALL_SECONDS: float = 2.0 * 3600.0


# ---------------------------------------------------------------------------
# models (PROVISIONAL constants)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeModel:
    """Throughput model of the alignment run (PROVISIONAL until ``calibrated``).

    **Alignment time.** For a locus with work ``W`` (the strategy's
    ``N*L^alpha + beta*N^2*L``) and tier ``t`` with lower edge ``B_t``
    (``B_0 = 0``; ``B_1..B_3`` are the strategy's tier boundaries)::

        T1(W) = sum_{j<t} c_j * (B_{j+1} - B_j)  +  c_t * (W - B_t)      # seconds, 1 core

    i.e. a continuous piecewise-linear function whose slope ``c_j``
    (``sec_per_work[j]``) shrinks for the cheaper heuristic tiers, so a bigger
    locus is never predicted to be faster. Protein loci multiply it by
    ``protein_factor``. With ``p`` threads a task of tier ``t`` runs
    ``T(p) = T1 * (s + (1 - s) / min(p, max_threads))`` (Amdahl, ``s =
    serial_fraction[t]``; the fixed/I-O parts are serial too).

    **Derivation of the provisional numbers.** The calibration target of the
    strategy model is "a locus at the upper edge of its tier takes ~45 s on 8
    threads": ``sec_per_work = (5.0e-9, 1.5e-9, 7.0e-10, 3.0e-10)`` with
    ``serial_fraction = (0.20, 0.35, 0.35, 0.45)`` reproduces that for the
    default boundaries ``(3e10, 7e10, 1.5e11)`` (the extreme tier is anchored at
    ``W = 3e11``). Real PartTree runs at 10k+ taxa are likely slower than this
    optimistic anchor, which is why ``uncertainty`` is wide.

    **What the calibration phase has to measure** (serially, quiet machine):

    1. ``sec_per_work`` / ``protein_factor``: single-thread MAFFT wall time per
       tier over an ``(N, L)`` grid, nucleotide and protein; least-squares fit
       of the piecewise slopes (``benchmarking/calibrate_strategy.py --grid``
       already records these timings).
    2. ``serial_fraction`` / ``max_useful_threads``: the same loci at 1, 2, 4, 8
       (and 12) threads; fit Amdahl's ``s`` per tier (Karp-Flatt) and the thread
       count after which the speedup stops improving.
    3. ``fixed_overhead_s``: MAFFT wall time on a tiny input (process start,
       FASTA export, parse). ``io_s_per_cell`` / ``backtranslate_s_per_cell``:
       export + parse + NEXUS conversion (and back-translation) time per
       ``N x L`` cell.
    4. ``chunk_plan_s_per_cell`` / ``chunk_splice_s_per_cell``: time of
       ``plan_chunks`` and ``splice_alignments`` per cell.
    5. ``concat_s_per_cell``: ``SupermatrixBuilder`` + writers per ``ntax x
       nchar`` cell. ``startup_s``: host detection + scheduler/scratch start.
    6. ``uncertainty``: the observed 5th/95th percentile of predicted/measured.

    Attributes:
        calibrated: ``False`` while these are analytic placeholders.
        provenance: Free-text origin of the constants.
        sec_per_work: Marginal single-core seconds per work unit, per tier
            ``(ACCURATE, FAST_TREE, HIGH_SCALE, EXTREME_SCALE)``.
        serial_fraction: Amdahl serial fraction per tier.
        protein_factor: Multiplier of the alignment time for protein loci.
        max_useful_threads: Threads beyond this add nothing (per task).
        min_work_per_thread_s: A task only uses a further thread per this many
            seconds of alignment work (tiny loci stay single-threaded).
        fixed_overhead_s: Per-locus (and per-chunk) start-up/export/parse cost.
        io_s_per_cell: Per ``N x L`` aligned cell: parse, convert, write.
        backtranslate_s_per_cell: Per nucleotide cell of a ``cds`` locus.
        chunk_plan_s_per_cell / chunk_splice_s_per_cell: Per cell of a chunked locus.
        concat_s_per_cell: Per supermatrix cell (serial).
        startup_s: Fixed run start-up cost.
        uncertainty: ``(low, high)`` multipliers of the predicted wall time that
            bracket the plausible outcome.
    """

    calibrated: bool = False
    provenance: str = (
        "PROVISIONAL analytic placeholders (45 s @ 8 threads at each tier edge); "
        "2026-11-01 reduced-grid ACCURATE kappa (seconds/W @8t) spanned ~3e-10..1e-7 "
        "(median ~1.2e-8) vs sec_per_work[0]=5e-9 — not trusted enough to calibrate; "
        "see docs/calibration.md"
    )
    sec_per_work: tuple[float, float, float, float] = (5.0e-9, 1.5e-9, 7.0e-10, 3.0e-10)
    serial_fraction: tuple[float, float, float, float] = (0.20, 0.35, 0.35, 0.45)
    protein_factor: float = 1.5
    max_useful_threads: int = 8
    min_work_per_thread_s: float = 2.0
    fixed_overhead_s: float = 0.15
    io_s_per_cell: float = 1.0e-8
    backtranslate_s_per_cell: float = 2.0e-8
    chunk_plan_s_per_cell: float = 5.0e-8
    chunk_splice_s_per_cell: float = 3.0e-8
    concat_s_per_cell: float = 4.0e-9
    startup_s: float = 0.5
    uncertainty: tuple[float, float] = (0.5, 3.0)

    def align_seconds(
        self,
        work: float,
        tier: int | AlignmentTier,
        *,
        alphabet: str = _NUCL,
        boundaries: tuple[float, float, float] = (
            DEFAULT_CONSTANTS.work_fast_tree,
            DEFAULT_CONSTANTS.work_high_scale,
            DEFAULT_CONSTANTS.work_extreme_scale,
        ),
    ) -> float:
        """Single-core seconds of the alignment step (see the class docstring).

        A locus forced into a tier above its natural one (``work`` below the
        tier's lower edge) costs ``sec_per_work[tier] * work``.
        """
        if work <= 0:
            return 0.0
        t = min(max(int(tier), 0), len(self.sec_per_work) - 1)
        edges = (0.0, *boundaries)
        start = edges[t]
        if work < start:
            seconds = self.sec_per_work[t] * work
        else:
            seconds = sum(
                self.sec_per_work[j] * (edges[j + 1] - edges[j]) for j in range(t)
            )
            seconds += self.sec_per_work[t] * (work - start)
        return seconds * (self.protein_factor if alphabet == _PROT else 1.0)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (for the manifest and for calibration files)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TimeModel:
        """Rebuild a model from :meth:`to_dict` output (unknown keys are ignored)."""
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        for key in ("sec_per_work", "serial_fraction", "uncertainty"):
            if key in kwargs:
                kwargs[key] = tuple(float(x) for x in kwargs[key])
        return cls(**kwargs)


@dataclass(frozen=True)
class FootprintModel:
    """Byte model of the on-disk outputs (PROVISIONAL until ``calibrated``).

    Attributes:
        calibrated: ``False`` while these are analytic placeholders.
        header_bytes: ``>name\\n`` bytes per sequence in a FASTA file.
        nexus_name_bytes: Taxon label + padding bytes per NEXUS matrix row.
        sqlite_row_overhead: SQLite bytes per stored sequence besides the text
            (ids, description, indices).
        sqlite_page_factor: Multiplier for page slack / free pages.
        concat_name_bytes: Taxon label bytes per row of a concatenated output.
    """

    calibrated: bool = False
    header_bytes: int = 40
    nexus_name_bytes: int = 30
    sqlite_row_overhead: int = 220
    sqlite_page_factor: float = 1.25
    concat_name_bytes: int = 40

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return asdict(self)


#: Time model used when callers pass none.
DEFAULT_TIME_MODEL = TimeModel()
#: Footprint model used when callers pass none.
DEFAULT_FOOTPRINT_MODEL = FootprintModel()


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EstimateWarning:
    """A machine-readable warning (``code``) with its human text."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class LocusEstimate:
    """Prediction for one locus.

    Attributes:
        name: Region name.
        n_seq, min_len, max_len: Taxa and nucleotide lengths from the summary.
        alphabet: Alphabet that is aligned (``"nucl"`` or ``"prot"``).
        align_len: Longest sequence of the aligned alphabet.
        tier: Strategy tier name (``ACCURATE`` ... ``EXTREME_SCALE``).
        work: Strategy work ``W`` of the whole locus.
        chunks: Predicted number of chunks (0 = aligned monolithically).
        single_core_seconds: Total predicted single-core seconds of the locus
            (planning + chunks + finalizer when chunked).
        mem_bytes: Predicted per-task peak RSS (admission estimate).
        unaligned_bytes: Unaligned FASTA bytes (nucleotide + protein for cds).
        aligned_bytes: Aligned FASTA bytes (nucleotide + protein for cds).
    """

    name: str
    n_seq: int
    min_len: int
    max_len: int
    alphabet: str
    align_len: int
    tier: str
    work: float
    chunks: int
    single_core_seconds: float
    mem_bytes: int
    unaligned_bytes: int
    aligned_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "name": self.name,
            "n_seq": self.n_seq,
            "min_len": self.min_len,
            "max_len": self.max_len,
            "alphabet": self.alphabet,
            "align_len": self.align_len,
            "tier": self.tier,
            "work": self.work,
            "chunks": self.chunks,
            "single_core_seconds": round(self.single_core_seconds, 6),
            "mem_bytes": self.mem_bytes,
            "unaligned_bytes": self.unaligned_bytes,
            "aligned_bytes": self.aligned_bytes,
        }


@dataclass(frozen=True)
class DiskFootprint:
    """Predicted bytes written to disk (``total`` honours ``keep_unaligned``)."""

    unaligned_fasta: int
    aligned_fasta: int
    nexus_per_locus: int
    sqlite_db: int
    concatenated: int
    keep_unaligned: bool

    @property
    def total(self) -> int:
        """Sum of all outputs (unaligned FASTA only if kept)."""
        kept = self.unaligned_fasta if self.keep_unaligned else 0
        return (
            kept
            + self.aligned_fasta
            + self.nexus_per_locus
            + self.sqlite_db
            + self.concatenated
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        data = asdict(self)
        data["total"] = self.total
        return data


@dataclass(frozen=True)
class ScratchEstimate:
    """Scratch recommendation and the predicted need.

    Attributes:
        tier: ``"ram"``, ``"disk"`` or ``"system"`` (no usable location found).
        path: Primary scratch path (``None`` for ``system``).
        fstype: Filesystem type of the primary tier, if known.
        budget_bytes: Bytes of the primary tier the run may use.
        need_bytes: ``3 x`` the in-flight unaligned bytes (``workers`` largest loci).
        fits: ``need_bytes`` (plus headroom) fits the budget of a RAM tier;
            always ``True`` for disk tiers.
        fallback_path: Disk tier used when the RAM-disk is full.
    """

    tier: str
    path: str | None
    fstype: str | None
    budget_bytes: int | None
    need_bytes: int
    fits: bool
    fallback_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return asdict(self)


@dataclass(frozen=True)
class PreflightEstimate:
    """Everything :func:`preflight_estimate` predicts (see the module docstring).

    ``wall_seconds`` is ``startup + simulated alignment makespan + concatenation``;
    ``wall_range`` brackets it with the model's uncertainty factors.
    """

    mode: str
    label: str
    calibrated: bool
    ntax: int
    n_loci: int
    n_skipped: int
    n_chunked: int
    tier_counts: dict[str, int]
    cores: int
    workers: int
    total_work: float
    single_core_seconds: float
    align_seconds: float
    concat_seconds: float
    wall_seconds: float
    wall_range: tuple[float, float]
    parallel_efficiency: float
    straggler_tax: float
    critical_locus: str | None
    peak_ram_bytes: int
    peak_ram_concurrent_bytes: int
    max_mafft_bytes: int
    parser_buffer_bytes: int
    supermatrix_bytes: int
    envelope_bytes: int
    largest_memory_locus: str | None
    disk: DiskFootprint
    scratch: ScratchEstimate
    warnings: tuple[EstimateWarning, ...]
    heaviest: tuple[LocusEstimate, ...]
    loci: tuple[LocusEstimate, ...] = field(repr=False, default=())

    @property
    def ram_fraction(self) -> float:
        """Predicted peak RAM as a fraction of the memory envelope."""
        return self.peak_ram_bytes / self.envelope_bytes if self.envelope_bytes else 0.0

    @property
    def warning_codes(self) -> tuple[str, ...]:
        """Codes of all warnings."""
        return tuple(w.code for w in self.warnings)

    def to_dict(self, include_loci: bool = False) -> dict[str, Any]:
        """JSON-friendly view; ``include_loci`` adds every per-locus prediction."""
        data: dict[str, Any] = {
            "mode": self.mode,
            "label": self.label,
            "calibrated": self.calibrated,
            "ntax": self.ntax,
            "n_loci": self.n_loci,
            "n_skipped": self.n_skipped,
            "n_chunked": self.n_chunked,
            "tier_counts": dict(self.tier_counts),
            "cores": self.cores,
            "workers": self.workers,
            "total_work": self.total_work,
            "single_core_seconds": round(self.single_core_seconds, 6),
            "align_seconds": round(self.align_seconds, 6),
            "concat_seconds": round(self.concat_seconds, 6),
            "wall_seconds": round(self.wall_seconds, 6),
            "wall_range": [round(x, 6) for x in self.wall_range],
            "parallel_efficiency": round(self.parallel_efficiency, 6),
            "straggler_tax": round(self.straggler_tax, 6),
            "critical_locus": self.critical_locus,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_ram_concurrent_bytes": self.peak_ram_concurrent_bytes,
            "max_mafft_bytes": self.max_mafft_bytes,
            "parser_buffer_bytes": self.parser_buffer_bytes,
            "supermatrix_bytes": self.supermatrix_bytes,
            "envelope_bytes": self.envelope_bytes,
            "ram_fraction": round(self.ram_fraction, 6),
            "largest_memory_locus": self.largest_memory_locus,
            "disk": self.disk.to_dict(),
            "scratch": self.scratch.to_dict(),
            "warnings": [{"code": w.code, "message": w.message} for w in self.warnings],
            "heaviest": [loc.to_dict() for loc in self.heaviest],
        }
        if include_loci:
            data["loci"] = [loc.to_dict() for loc in self.loci]
        return data

    def _tier_line(self) -> str:
        return ", ".join(f"{k} {v}" for k, v in self.tier_counts.items()) or "none"

    def _rows(self) -> list[tuple[str, str]]:
        lo, hi = self.wall_range
        s = self.scratch
        d = self.disk
        scratch_where = (f"{s.tier} {s.path}" if s.path else "system temp dir") + (
            f" ({s.fstype})" if s.fstype else ""
        )
        rows = [
            (
                "Workload",
                f"{self.n_loci} loci, up to {self.ntax} taxa ({self.mode})"
                + (f", {self.n_skipped} empty skipped" if self.n_skipped else ""),
            ),
            ("Algorithm tiers", self._tier_line()),
        ]
        if self.n_chunked:
            rows.append(
                ("Sub-locus chunking", f"{self.n_chunked} loci split into chunks")
            )
        rows.extend(
            [
                (
                    "Predicted wall time",
                    f"~{format_duration(self.wall_seconds)} "
                    f"(range {format_duration(lo)} - {format_duration(hi)}) "
                    f"on {self.cores} cores / {self.workers} workers",
                ),
                (
                    "Predicted CPU time",
                    f"~{format_duration(self.single_core_seconds)} single-core",
                ),
                (
                    "Predicted efficiency",
                    f"{100 * self.parallel_efficiency:.0f}%, straggler tax "
                    f"{100 * self.straggler_tax:.0f}%"
                    + (
                        f", critical locus {self.critical_locus}"
                        if self.critical_locus
                        else ""
                    ),
                ),
                (
                    "Predicted peak RAM",
                    f"{format_bytes(self.peak_ram_bytes)} = "
                    f"{100 * self.ram_fraction:.0f}% of the {format_bytes(self.envelope_bytes)} envelope "
                    f"(max MAFFT {format_bytes(self.max_mafft_bytes)} + {self.workers} x parser "
                    f"{format_bytes(self.parser_buffer_bytes)} + supermatrix "
                    f"{format_bytes(self.supermatrix_bytes)})",
                ),
                (
                    "Predicted disk",
                    f"{format_bytes(d.total)} (aligned FASTA {format_bytes(d.aligned_fasta)}, "
                    f"NEXUS {format_bytes(d.nexus_per_locus)}, SQLite {format_bytes(d.sqlite_db)}, "
                    f"concatenated {format_bytes(d.concatenated)}"
                    + (
                        f", unaligned {format_bytes(d.unaligned_fasta)}"
                        if d.keep_unaligned
                        else ""
                    )
                    + ")",
                ),
                (
                    "Scratch",
                    f"{scratch_where}, budget {format_bytes(s.budget_bytes)}, "
                    f"need ~{format_bytes(s.need_bytes)}",
                ),
            ]
        )
        return rows

    def to_text(self) -> str:
        """Plain-text report (no ``rich``)."""
        rows = self._rows()
        width = max(len(k) for k, _ in rows)
        lines = [f"PlastBurstAlign pre-flight {self.label}"]
        lines.extend(f"  {k:<{width}}  {v}" for k, v in rows)
        if self.heaviest:
            lines.append("  Heaviest loci:")
            for loc in self.heaviest[:5]:
                lines.append(
                    f"    {loc.name:<28} {loc.n_seq:>6} x {loc.align_len:<7} "
                    f"{loc.tier:<14} ~{format_duration(loc.single_core_seconds)}"
                    + (f"  ({loc.chunks} chunks)" if loc.chunks else "")
                )
        lines.extend(f"  ! [{w.code}] {w.message}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def predicted_chunks(
    n_seq: int,
    align_len: int,
    alphabet: str = _NUCL,
    config: ChunkConfig | None = None,
    *,
    subdivide: bool = True,
) -> int:
    """Predicted chunk count of a locus (0 = not chunked), from data only.

    Mirrors the criteria of :func:`~plastburstalign.chunking.plan_chunks`:
    enough sequences, a longest sequence of at least ``min_locus_len`` and not
    ``adjust_direction`` (nucleotides), then ``max(2, round(L / target))``.
    """
    cfg = config or ChunkConfig()
    if not subdivide or n_seq < max(2, cfg.min_sequences):
        return 0
    if cfg.adjust_direction and alphabet == _NUCL:
        return 0
    min_locus, target, _k = cfg.for_alphabet(alphabet)
    if align_len < min_locus:
        return 0
    return max(2, round(align_len / target))


def _entry_int(entry: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in entry and entry[key] is not None:
            try:
                return max(0, int(entry[key]))
            except (TypeError, ValueError):
                return 0
    return 0


def _normalize_mode(mode: object) -> str:
    text = str(getattr(mode, "value", mode)).strip().lower()
    if not text:
        raise ValueError("mode must be a non-empty string such as 'cds' or 'igs'")
    if "+" in text or "," in text:
        raise ValueError(
            f"mode {text!r} combines several modes; estimate one mode per call"
        )
    return text


@dataclass
class _Plan:
    """Per-locus intermediate (unit seconds are single-core)."""

    est: LocusEstimate
    task: SimTask
    mafft_bytes: int
    parser_bytes: int
    inflight_bytes: int
    nuc_cols: int
    sqlite_rows_bytes: int
    nexus_bytes: int


def _threads_for(seconds: float, cap: int, model: TimeModel) -> int:
    per = max(1e-9, model.min_work_per_thread_s)
    return max(1, min(cap, int(seconds // per)))


def _eff_serial(
    tier: int, align_s: float, serial_extra: float, model: TimeModel
) -> float:
    total = align_s + serial_extra
    if total <= 0:
        return 1.0
    s = model.serial_fraction[min(max(tier, 0), len(model.serial_fraction) - 1)]
    return min(1.0, max(0.0, (s * align_s + serial_extra) / total))


def preflight_estimate(
    region_summary: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
    ntax: int | None = None,
    plan: TuningPlan,
    constants: TimeModel | None = None,
    subdivide: bool = True,
    prot_summary: Mapping[str, Mapping[str, Any]] | None = None,
    keep_unaligned: bool = True,
    footprint: FootprintModel | None = None,
    strategy_constants: StrategyConstants | None = None,
    strategy_overrides: Mapping[str, Any] | None = None,
    chunk_config: ChunkConfig | None = None,
    long_wall_seconds: float = LONG_WALL_SECONDS,
    ram_warn_fraction: float = RAM_WARN_FRACTION,
    disk_free_bytes: int | None = None,
    heaviest: int = 10,
) -> PreflightEstimate:
    """Predict time, memory, disk and scratch of a run before it starts.

    Args:
        region_summary: ``{region: {"count": N, "max_len": L, "min_len": l}}`` of
            the *nucleotide* store (``store.get_region_summary("nucl")``).
            ``n_seq`` is accepted as an alias of ``count``. Regions with no
            sequences are skipped.
        mode: One mode (``cds``/``igs``/``int``/``rna``). ``cds`` aligns
            proteins; the others align nucleotides.
        ntax: Taxa in the supermatrix (default: the largest locus ``count``).
        plan: The :class:`~plastburstalign.autotune.TuningPlan` (cores, workers,
            memory envelope and model, scratch).
        constants: :class:`TimeModel` (default :data:`DEFAULT_TIME_MODEL`,
            provisional). Override to inject calibrated numbers.
        subdivide: Predict sub-locus chunking for long loci.
        prot_summary: Protein summary for ``cds`` (exact aa lengths); default
            ``max_len // 3``.
        keep_unaligned: Count the unaligned FASTA files in the disk total.
        footprint: :class:`FootprintModel` override.
        strategy_constants, strategy_overrides: Passed to the strategy decision
            (tiers are a pure function of the data and these).
        chunk_config: Chunking criteria (default :class:`ChunkConfig`).
        long_wall_seconds: Warn when the predicted wall time exceeds this.
        ram_warn_fraction: Warn when predicted peak RAM exceeds this share of
            the memory envelope.
        disk_free_bytes: Free space of the output volume, if known (enables the
            ``disk_space`` warning).
        heaviest: How many of the costliest loci to keep in ``heaviest``.

    Returns:
        A :class:`PreflightEstimate`, labelled "estimate (model not yet
        calibrated)" while any model is provisional.

    Raises:
        ValueError: ``mode`` is empty or combines several modes.
    """
    tm = constants or DEFAULT_TIME_MODEL
    fm = footprint or DEFAULT_FOOTPRINT_MODEL
    sc = strategy_constants or DEFAULT_CONSTANTS
    cfg = chunk_config or ChunkConfig()
    mem_model = plan.memory_model
    mode_name = _normalize_mode(mode)
    is_cds = mode_name == "cds"
    alphabet = _PROT if is_cds else _NUCL
    mem_alpha = "protein" if is_cds else "nucleotide"
    boundaries = (sc.work_fast_tree, sc.work_high_scale, sc.work_extreme_scale)
    cores = max(1, int(plan.cpu_budget))
    workers = max(1, int(plan.workers))
    cap_threads = max(1, min(int(plan.threads_per_task_cap), tm.max_useful_threads))
    exp = mem_model.alignment_expansion

    plans: list[_Plan] = []
    skipped = 0
    for name in sorted(region_summary):
        entry = region_summary[name]
        n_seq = _entry_int(entry, "count", "n_seq", "n")
        max_len = _entry_int(entry, "max_len", "length")
        min_len = _entry_int(entry, "min_len")
        if n_seq <= 0 or max_len <= 0:
            skipped += 1
            continue
        if is_cds:
            pentry = (prot_summary or {}).get(name)
            align_len = (
                _entry_int(pentry, "max_len") if pentry is not None else max_len // 3
            )
            align_len = max(1, align_len)
            align_min = (
                _entry_int(pentry, "min_len")
                if pentry is not None
                else (min_len // 3 if min_len else 0)
            )
        else:
            align_len, align_min = max_len, min_len
        plans.append(
            _plan_locus(
                name,
                n_seq,
                min_len,
                max_len,
                align_len,
                align_min,
                alphabet,
                mem_alpha,
                is_cds,
                tm,
                fm,
                sc,
                boundaries,
                strategy_overrides,
                cfg,
                subdivide,
                cap_threads,
                mem_model,
                exp,
            )
        )

    tasks = [p.task for p in plans]
    sim_wall = 0.0
    efficiency = 0.0
    tax = 0.0
    critical: str | None = None
    notes: list[EstimateWarning] = []
    if tasks:
        result = simulate(tasks, cores, Policy.LOCUS_SCHEDULER, use_chunks=subdivide)
        sim_wall = result.makespan
        efficiency = result.metrics.parallel_efficiency
        tax = result.metrics.straggler_tax
        critical = result.critical_task_id
        if workers < cores:
            limited = simulate(
                tasks, workers, Policy.LOCUS_SCHEDULER, use_chunks=subdivide
            )
            if limited.makespan > sim_wall:
                sim_wall = limited.makespan
                efficiency = limited.metrics.parallel_efficiency * workers / cores
                tax = limited.metrics.straggler_tax
                critical = limited.critical_task_id
                notes.append(
                    EstimateWarning(
                        "workers_limited",
                        f"only {workers} of {cores} cores can run concurrent tasks "
                        "(memory model); wall time predicted with that concurrency",
                    )
                )

    ntax_eff = (
        int(ntax) if ntax is not None else max((p.est.n_seq for p in plans), default=0)
    )
    total_cols = sum(p.nuc_cols for p in plans)
    concat_s = tm.concat_s_per_cell * ntax_eff * total_cols
    wall = (tm.startup_s + sim_wall + concat_s) if plans else tm.startup_s
    lo_f, hi_f = tm.uncertainty
    wall_range = (wall * lo_f, wall * hi_f)

    # ---- memory -------------------------------------------------------------
    max_mafft = max((p.mafft_bytes for p in plans), default=0)
    parser_max = max((p.parser_bytes for p in plans), default=0)
    sup_bytes = estimate_supermatrix_bytes(ntax_eff, total_cols, model=mem_model)
    peak = max_mafft + workers * parser_max + sup_bytes
    top_mafft = sorted((p.mafft_bytes for p in plans), reverse=True)[:workers]
    peak_conc = sum(top_mafft) + workers * parser_max + sup_bytes
    envelope = int(plan.memory_envelope_bytes)
    largest_mem = (
        max(plans, key=lambda p: (p.mafft_bytes, p.est.name)).est.name
        if plans
        else None
    )

    # ---- disk & scratch -------------------------------------------------------
    unaligned = sum(p.est.unaligned_bytes for p in plans)
    aligned = sum(p.est.aligned_bytes for p in plans)
    nexus = sum(p.nexus_bytes for p in plans)
    sqlite = int(sum(p.sqlite_rows_bytes for p in plans) * fm.sqlite_page_factor)
    concat_bytes = 2 * ntax_eff * (total_cols + fm.concat_name_bytes) if plans else 0
    disk = DiskFootprint(
        unaligned, aligned, nexus, sqlite, concat_bytes, keep_unaligned
    )

    inflight = sum(sorted((p.inflight_bytes for p in plans), reverse=True)[:workers])
    need = SCRATCH_BYTES_FACTOR * inflight
    tier = plan.scratch_tier
    if tier is None:
        scratch = ScratchEstimate(
            "system", None, None, None, need, True, plan.scratch_fallback_path
        )
    else:
        budget = plan.scratch_budget_bytes
        is_ram = tier.is_ram
        fits = (not is_ram) or (
            budget is None or need + DEFAULT_MIN_HEADROOM_BYTES <= budget
        )
        scratch = ScratchEstimate(
            "ram" if is_ram else "disk",
            tier.path,
            tier.fstype,
            budget,
            need,
            fits,
            plan.scratch_fallback_path,
        )

    # ---- warnings ---------------------------------------------------------------
    warns: list[EstimateWarning] = list(notes)
    if envelope > 0 and peak > envelope:
        warns.append(
            EstimateWarning(
                "ram_over_envelope",
                f"predicted peak RAM {format_bytes(peak)} exceeds the "
                f"{format_bytes(envelope)} memory envelope; expect admission "
                "serialisation, swapping or an OOM kill. Raise --memory-limit, use "
                "fewer workers or a trimmed/smaller dataset.",
            )
        )
    elif envelope > 0 and peak > ram_warn_fraction * envelope:
        warns.append(
            EstimateWarning(
                "ram_high",
                f"predicted peak RAM {format_bytes(peak)} is {100 * peak / envelope:.0f}% "
                f"of the {format_bytes(envelope)} memory envelope "
                f"(threshold {100 * ram_warn_fraction:.0f}%)",
            )
        )
    if envelope > 0 and sup_bytes > envelope // 2 and plans:
        warns.append(
            EstimateWarning(
                "supermatrix_memmap",
                f"the supermatrix ({format_bytes(sup_bytes)}) exceeds half of the "
                "memory envelope; the builder will spill it to a scratch memmap",
            )
        )
    oversize = [p for p in plans if envelope > 0 and p.est.mem_bytes > envelope]
    if oversize:
        worst = max(oversize, key=lambda p: (p.est.mem_bytes, p.est.name)).est
        warns.append(
            EstimateWarning(
                "oversize_locus",
                f"{len(oversize)} locus/loci need more memory than the whole envelope "
                f"(largest: {worst.name}, ~{format_bytes(worst.mem_bytes)}); they will "
                "run alone",
            )
        )
    for ram in plan.host.ram_tiers:
        free = ram.free_bytes
        if free is not None and free < MIN_RAM_SCRATCH_FREE_BYTES:
            where = scratch.path if scratch.tier != "ram" else scratch.fallback_path
            warns.append(
                EstimateWarning(
                    "tiny_ramdisk",
                    f"RAM-disk {ram.path} has only {format_bytes(free)} free "
                    f"(< {format_bytes(MIN_RAM_SCRATCH_FREE_BYTES)}); scratch will use "
                    f"{where or 'the system temp dir'} instead. Mount a larger tmpfs "
                    "or set PBA_SCRATCH for faster FASTA I/O.",
                )
            )
    if scratch.tier == "ram" and not scratch.fits:
        warns.append(
            EstimateWarning(
                "ramdisk_budget_low",
                f"predicted in-flight scratch need {format_bytes(need)} exceeds the "
                f"RAM-disk budget {format_bytes(scratch.budget_bytes)}; tasks will "
                "fall back to disk scratch individually",
            )
        )
    extreme = [
        p.est.name for p in plans if p.est.tier == AlignmentTier.EXTREME_SCALE.name
    ]
    if extreme:
        warns.append(
            EstimateWarning(
                "extreme_tier",
                f"{len(extreme)} locus/loci use the EXTREME_SCALE algorithm "
                "(--parttree --retree 1 --nofft): fastest tier, lowest accuracy "
                f"(e.g. {extreme[0]})",
            )
        )
    if wall > long_wall_seconds:
        warns.append(
            EstimateWarning(
                "wall_long",
                f"predicted wall time ~{format_duration(wall)} exceeds "
                f"{format_duration(long_wall_seconds)}; consider --subdivide, more "
                "cores, a Slurm export, or aligning fewer loci",
            )
        )
    if disk_free_bytes is not None and disk.total > 0.9 * disk_free_bytes:
        warns.append(
            EstimateWarning(
                "disk_space",
                f"predicted output size {format_bytes(disk.total)} is close to the "
                f"{format_bytes(disk_free_bytes)} free on the output volume",
            )
        )
    for message in plan.warnings:
        warns.append(EstimateWarning("host_plan", message))

    ests = [p.est for p in plans]
    ranked = sorted(ests, key=lambda e: (-e.single_core_seconds, e.name))
    calibrated = bool(tm.calibrated and fm.calibrated and mem_model.calibrated)
    counts = Counter(e.tier for e in ests)
    tier_counts = {t.name: counts[t.name] for t in AlignmentTier if counts[t.name]}
    return PreflightEstimate(
        mode=mode_name,
        label=_LABEL_CALIBRATED if calibrated else LABEL_UNCALIBRATED,
        calibrated=calibrated,
        ntax=ntax_eff,
        n_loci=len(ests),
        n_skipped=skipped,
        n_chunked=sum(1 for e in ests if e.chunks),
        tier_counts=tier_counts,
        cores=cores,
        workers=workers,
        total_work=sum(e.work for e in ests),
        single_core_seconds=sum(e.single_core_seconds for e in ests),
        align_seconds=sim_wall,
        concat_seconds=concat_s,
        wall_seconds=wall,
        wall_range=wall_range,
        parallel_efficiency=efficiency,
        straggler_tax=tax,
        critical_locus=critical,
        peak_ram_bytes=peak,
        peak_ram_concurrent_bytes=peak_conc,
        max_mafft_bytes=max_mafft,
        parser_buffer_bytes=parser_max,
        supermatrix_bytes=sup_bytes,
        envelope_bytes=envelope,
        largest_memory_locus=largest_mem,
        disk=disk,
        scratch=scratch,
        warnings=tuple(warns),
        heaviest=tuple(ranked[: max(0, int(heaviest))]),
        loci=tuple(ests),
    )


def _plan_locus(
    name: str,
    n_seq: int,
    min_len: int,
    max_len: int,
    align_len: int,
    align_min: int,
    alphabet: str,
    mem_alpha: str,
    is_cds: bool,
    tm: TimeModel,
    fm: FootprintModel,
    sc: StrategyConstants,
    boundaries: tuple[float, float, float],
    overrides: Mapping[str, Any] | None,
    cfg: ChunkConfig,
    subdivide: bool,
    cap_threads: int,
    mem_model: MemoryModel,
    exp: float,
) -> _Plan:
    """Predict one locus (strategy, seconds, memory, bytes) and build its sim task."""
    strat = determine_alignment_strategy(
        n_seq,
        align_len,
        align_min or None,
        alphabet=alphabet,
        overrides=overrides,
        constants=sc,
    )
    tier = int(strat.tier)
    nuc_cols = math.ceil(max_len * exp)
    nuc_cells = n_seq * nuc_cols
    io_s = tm.io_s_per_cell * nuc_cells
    bt_s = tm.backtranslate_s_per_cell * nuc_cells if is_cds else 0.0
    n_chunks = predicted_chunks(n_seq, align_len, alphabet, cfg, subdivide=subdivide)

    chunk_tasks: tuple[SimTask, ...] = ()
    plan_s = fin_s = 0.0
    mono_aln = tm.align_seconds(
        strat.work, strat.tier, alphabet=alphabet, boundaries=boundaries
    )
    mono_serial = tm.fixed_overhead_s + io_s + bt_s
    mono_total = mono_aln + mono_serial
    unit_len = align_len
    unit_parttree = strat.uses_parttree
    max_threads = _threads_for(mono_aln, cap_threads, tm)
    if n_chunks:
        unit_len = math.ceil(align_len / n_chunks)
        cstrat = determine_alignment_strategy(
            n_seq, unit_len, None, alphabet=alphabet, overrides=overrides, constants=sc
        )
        unit_parttree = cstrat.uses_parttree
        c_aln = tm.align_seconds(
            cstrat.work, cstrat.tier, alphabet=alphabet, boundaries=boundaries
        )
        c_serial = tm.fixed_overhead_s + tm.io_s_per_cell * n_seq * math.ceil(
            unit_len * exp
        )
        c_total = c_aln + c_serial
        c_threads = _threads_for(c_aln, cap_threads, tm)
        chunk_tasks = tuple(
            SimTask(
                task_id=f"c{j:03d}",
                work=c_total,
                serial_fraction=_eff_serial(int(cstrat.tier), c_aln, c_serial, tm),
                max_threads=c_threads,
                n_seq=n_seq,
                max_len=unit_len,
            )
            for j in range(n_chunks)
        )
        plan_s = tm.fixed_overhead_s + tm.chunk_plan_s_per_cell * n_seq * align_len
        fin_s = tm.chunk_splice_s_per_cell * nuc_cells + io_s + bt_s
        single = plan_s + n_chunks * c_total + fin_s
    else:
        single = mono_total
    tier_name = strat.tier.name

    task = SimTask(
        task_id=name,
        work=mono_total,
        serial_fraction=_eff_serial(tier, mono_aln, mono_serial, tm),
        max_threads=max_threads,
        n_seq=n_seq,
        max_len=align_len,
        chunks=chunk_tasks,
        plan_work=plan_s,
        finalize_work=fin_s,
    )

    task_threads = max(max_threads, 1)
    mafft = estimate_mafft_peak_bytes(
        n_seq,
        unit_len,
        parttree=unit_parttree,
        alphabet=mem_alpha,
        threads=task_threads,
        model=mem_model,
    )
    parser = estimate_parser_buffer_bytes(n_seq, max_len, model=mem_model)
    mem = est_mem_bytes(
        n_seq,
        unit_len,
        parttree=unit_parttree,
        alphabet=mem_alpha,
        threads=task_threads,
        model=mem_model,
    )
    mid_len = (min_len + max_len) // 2 if min_len > 0 else max_len
    hdr = fm.header_bytes
    unaligned = n_seq * (mid_len + hdr)
    aligned = n_seq * (nuc_cols + hdr)
    sqlite_rows = n_seq * (mid_len + fm.sqlite_row_overhead)
    if is_cds:
        prot_mid = max(1, mid_len // 3)
        unaligned += n_seq * (prot_mid + hdr)
        aligned += n_seq * (math.ceil(align_len * exp) + hdr)
        sqlite_rows += n_seq * (prot_mid + fm.sqlite_row_overhead)
    est = LocusEstimate(
        name=name,
        n_seq=n_seq,
        min_len=min_len,
        max_len=max_len,
        alphabet=alphabet,
        align_len=align_len,
        tier=tier_name,
        work=strat.work,
        chunks=n_chunks,
        single_core_seconds=single,
        mem_bytes=mem,
        unaligned_bytes=unaligned,
        aligned_bytes=aligned,
    )
    return _Plan(
        est=est,
        task=task,
        mafft_bytes=mafft,
        parser_bytes=parser,
        inflight_bytes=unaligned,
        nuc_cols=nuc_cols,
        sqlite_rows_bytes=sqlite_rows,
        nexus_bytes=n_seq * (nuc_cols + fm.nexus_name_bytes),
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def render_preflight(est: PreflightEstimate, console: Console | None = None) -> None:
    """Print the pre-flight report as a ``rich`` panel (table + warnings)."""
    from rich.console import Console as _Console
    from rich.console import Group, RenderableType
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if console is None:
        console = _Console(stderr=True)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    for key, value in est._rows():
        grid.add_row(key, Text(value))
    parts: list[RenderableType] = [grid]
    if est.heaviest:
        table = Table(
            title="Heaviest loci",
            title_justify="left",
            show_edge=False,
            header_style="bold",
        )
        table.add_column("Locus", overflow="fold")
        table.add_column("N x L", justify="right")
        table.add_column("Tier")
        table.add_column("Chunks", justify="right")
        table.add_column("~1-core time", justify="right")
        for loc in est.heaviest[:5]:
            table.add_row(
                Text(loc.name),
                f"{loc.n_seq:,} x {loc.align_len:,}",
                loc.tier,
                str(loc.chunks) if loc.chunks else "",
                format_duration(loc.single_core_seconds),
            )
        parts.append(table)
    for w in est.warnings:
        parts.append(Text(f"! [{w.code}] {w.message}", style="yellow"))
    parts.append(
        Text(
            est.label,
            style="dim italic" if not est.calibrated else "dim",
        )
    )
    console.print(
        Panel(
            Group(*parts),
            title=f"Pre-flight estimate | {escape(est.mode)}",
            title_align="left",
            border_style="yellow" if est.warnings else "cyan",
        )
    )
