"""Auto-tuning: turn a :class:`~plastburstalign.hardware.HostProfile` into a plan.

``autotune()`` is the single place where *hardware* is translated into
*scheduling decisions* (never into algorithm choices - those stay a pure
function of the data, see the v3 design principles). It returns an immutable
:class:`TuningPlan` holding the CPU budget, the memory envelope, the scratch
recommendation and a human-readable rationale (:meth:`TuningPlan.explain`).

Memory model
------------
The peak-RSS model follows the blueprint::

    M_peak(p) = max_i(mafft_i) + p * parser_buffer + supermatrix_buffer

where ``p`` is the number of parallel workers. :func:`plan_parallelism` lowers
``p`` until ``M_peak(p)`` fits the memory envelope (never below 1; if even a
single worker does not fit, the plan carries a loud warning instead of failing).
The per-locus estimates are also exposed through :func:`est_mem_bytes` for the
scheduler's *memory admission control*.

.. warning::
   Every constant in the "PROVISIONAL" block below is an analytic placeholder.
   A dedicated serial calibration phase measures MAFFT/parser RSS with
   ``RUSAGE_CHILDREN`` and the SMT-vs-physical throughput on the target
   machine, then finalizes these values (and flips ``calibrated`` to ``True``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

from .hardware import (
    HostProfile,
    StorageTier,
    detect_host,
    format_bytes,
    parse_size,
)
from .logging_ops import logger as log
from .topology import CoreInfo, CpuTopology

__all__ = [
    "DEFAULT_MEMORY_MODEL",
    "DEFAULT_USE_SMT",
    "ENVELOPE_FRACTION",
    "MIN_DISK_SCRATCH_FREE_BYTES",
    "MIN_RAM_SCRATCH_FREE_BYTES",
    "RAM_SCRATCH_ENVELOPE_FRACTION",
    "THREADS_PER_TASK_SOFT_CAP",
    "UNKNOWN_MEMORY_FALLBACK_BYTES",
    "LocusWorkload",
    "MemoryModel",
    "ParallelismPlan",
    "TaskMemory",
    "ScratchChoice",
    "TuningPlan",
    "autotune",
    "est_mem_bytes",
    "estimate_mafft_peak_bytes",
    "estimate_parser_buffer_bytes",
    "estimate_supermatrix_bytes",
    "estimate_task_memory",
    "plan_parallelism",
    "recommend_scratch",
]

_MIB = 1024**2
_GIB = 1024**3

# ============================================================================ #
# PROVISIONAL constants - finalized by the serial calibration phase.
# Everything tunable lives in this block (plus MemoryModel below).
# ============================================================================ #

#: Use SMT siblings by default? PROVISIONAL: physical cores only. 2026-11-01
#: on a Ryzen 5650U: 6 vs 12 workers on a 24-taxon CDS set gave only ~1.16×
#: wall (1.92 s → 1.65 s); keep physical-only until a clearer win. Users
#: override per run with ``smt=True/False``.
DEFAULT_USE_SMT: bool = False

#: Fraction of the available memory (min of host available, cgroup headroom,
#: user limit) that may be committed by the run.
ENVELOPE_FRACTION: float = 0.85

#: Envelope used (with a warning) when no memory information is available.
UNKNOWN_MEMORY_FALLBACK_BYTES: int = 2 * _GIB

#: Optional ceiling on threads leased to one MAFFT task (``None`` = only the
#: NUMA-node size limits it). Calibration may set this from MAFFT scaling data.
THREADS_PER_TASK_SOFT_CAP: int | None = None

#: A RAM (tmpfs) scratch tier needs at least this much free space to be the
#: primary recommendation; below it the best disk tier is recommended and the
#: scheduler still falls back task by task.
MIN_RAM_SCRATCH_FREE_BYTES: int = 128 * _MIB

#: tmpfs pages are charged to the same memory (and cgroup) budget as the
#: workers, so a RAM scratch tier may use at most this fraction of the memory
#: envelope; the rest of the envelope is what tasks may plan with.
RAM_SCRATCH_ENVELOPE_FRACTION: float = 0.25

#: A disk scratch tier should have at least this much free space.
MIN_DISK_SCRATCH_FREE_BYTES: int = 1 * _GIB

_DEVICE_CLASS_RANK = {
    "nvme": 0,
    "ssd": 1,
    "unknown": 2,
    "hdd": 3,
    "network": 4,
    "ram": 5,
}


@dataclass(frozen=True)
class MemoryModel:
    """Analytic memory model (PROVISIONAL until ``calibrated`` is ``True``).

    Attributes:
        calibrated: ``False`` while the constants are untested guesses.
        provenance: Free-text origin of the constants.
        mafft_base_bytes: Fixed MAFFT process overhead (binary, tables).
        mafft_bytes_per_residue: Bytes per (sequence x column) cell: input copy,
            aligned copy and working profile.
        protein_factor: Multiplier on the per-residue cost for amino acids
            (larger scoring/profile tables).
        alignment_expansion: Aligned length relative to ``max_len`` (gaps).
        mafft_bytes_per_sequence: Per-sequence bookkeeping (headers, pointers).
        distance_bytes_per_pair: Bytes per ``N^2`` distance cell (float32); only
            charged for non-PartTree tiers.
        dp_bytes_per_cell: Bytes per ``L^2`` DP cell and thread; only charged
            for non-PartTree tiers.
        parser_bytes_per_cell: Python-side buffer per aligned cell while a
            finished locus is read (raw FASTA bytes + uint8 matrix).
        parser_bytes_per_sequence: Header bytes retained per sequence.
        supermatrix_bytes_per_cell: Bytes per supermatrix cell (uint8 = 1).
        safety_factor: Global multiplier applied to every estimate.
    """

    calibrated: bool = False
    provenance: str = (
        "PROVISIONAL analytic estimates; 2026-11-01 reduced-grid ACCURATE RSS "
        "@8t was ~0.6–2.7× this model (200–460 MB observed) — not refit; "
        "see docs/calibration.md"
    )
    mafft_base_bytes: int = 64 * _MIB
    mafft_bytes_per_residue: float = 8.0
    protein_factor: float = 1.5
    alignment_expansion: float = 1.25
    mafft_bytes_per_sequence: int = 512
    distance_bytes_per_pair: float = 4.0
    dp_bytes_per_cell: float = 4.0
    parser_bytes_per_cell: float = 2.05
    parser_bytes_per_sequence: int = 128
    supermatrix_bytes_per_cell: float = 1.0
    safety_factor: float = 1.0


#: Default model instance used when callers pass none.
DEFAULT_MEMORY_MODEL = MemoryModel()


# ============================================================================ #
# Memory estimates
# ============================================================================ #


class LocusWorkload(NamedTuple):
    """Lightweight per-locus description (plain 3-tuples are accepted too)."""

    n_seq: int
    max_len: int
    parttree: bool = False
    alphabet: str = "nucleotide"


class TaskMemory(NamedTuple):
    """Memory breakdown of one locus task."""

    mafft_bytes: int
    parser_bytes: int
    peak_bytes: int


def _is_protein(alphabet: str) -> bool:
    return alphabet.lower() in {"protein", "prot", "aa", "amino", "amino_acid"}


def estimate_mafft_peak_bytes(
    n_seq: int,
    max_len: int,
    *,
    parttree: bool = False,
    alphabet: str = "nucleotide",
    threads: int = 1,
    model: MemoryModel = DEFAULT_MEMORY_MODEL,
) -> int:
    """Estimate the peak RSS of one MAFFT run (PROVISIONAL model).

    ``O(N*L)`` profile/alignment storage plus, for non-PartTree tiers, the
    ``O(N^2)`` distance matrix (and ``O(L^2)`` DP scratch per thread).
    PartTree tiers avoid both quadratic terms.
    """
    n = max(0, int(n_seq))
    length = max(0, int(max_len))
    per_residue = model.mafft_bytes_per_residue * (
        model.protein_factor if _is_protein(alphabet) else 1.0
    )
    total = float(model.mafft_base_bytes)
    total += n * length * model.alignment_expansion * per_residue
    total += n * model.mafft_bytes_per_sequence
    if not parttree:
        total += model.distance_bytes_per_pair * n * n
        total += model.dp_bytes_per_cell * length * length * max(1, int(threads))
    return int(total * model.safety_factor)


def estimate_parser_buffer_bytes(
    n_seq: int, max_len: int, *, model: MemoryModel = DEFAULT_MEMORY_MODEL
) -> int:
    """Estimate the per-worker buffer while parsing one aligned locus."""
    n = max(0, int(n_seq))
    length = max(0, int(max_len))
    cells = n * length * model.alignment_expansion
    total = cells * model.parser_bytes_per_cell + n * model.parser_bytes_per_sequence
    return int(total * model.safety_factor)


def estimate_supermatrix_bytes(
    ntax: int, total_cols: int, *, model: MemoryModel = DEFAULT_MEMORY_MODEL
) -> int:
    """Estimate the in-RAM supermatrix buffer (``ntax x total_cols`` cells)."""
    cells = max(0, int(ntax)) * max(0, int(total_cols))
    return int(cells * model.supermatrix_bytes_per_cell * model.safety_factor)


def estimate_task_memory(
    n_seq: int,
    max_len: int,
    *,
    parttree: bool = False,
    alphabet: str = "nucleotide",
    threads: int = 1,
    model: MemoryModel = DEFAULT_MEMORY_MODEL,
) -> TaskMemory:
    """Return the MAFFT/parser breakdown and the per-task peak.

    A task aligns first (MAFFT child) and parses afterwards (parent buffer), so
    its own peak is the larger of the two, not their sum.
    """
    mafft = estimate_mafft_peak_bytes(
        n_seq,
        max_len,
        parttree=parttree,
        alphabet=alphabet,
        threads=threads,
        model=model,
    )
    parser = estimate_parser_buffer_bytes(n_seq, max_len, model=model)
    return TaskMemory(mafft, parser, max(mafft, parser))


def est_mem_bytes(
    n_seq: int,
    max_len: int,
    *,
    parttree: bool = False,
    alphabet: str = "nucleotide",
    threads: int = 1,
    model: MemoryModel = DEFAULT_MEMORY_MODEL,
) -> int:
    """Predicted per-task peak RSS in bytes, for the scheduler's memory admission.

    Pass the result as ``est_mem_bytes`` of the scheduler's task description.
    """
    return estimate_task_memory(
        n_seq,
        max_len,
        parttree=parttree,
        alphabet=alphabet,
        threads=threads,
        model=model,
    ).peak_bytes


def _normalize_workload(
    workload: Iterable[Sequence[Any]] | None,
) -> list[LocusWorkload]:
    loci: list[LocusWorkload] = []
    for item in workload or ():
        if isinstance(item, LocusWorkload):
            loci.append(item)
            continue
        if len(item) < 2:
            raise ValueError(
                f"workload item needs (n_seq, max_len[, parttree]): {item!r}"
            )
        loci.append(
            LocusWorkload(
                int(item[0]),
                int(item[1]),
                bool(item[2]) if len(item) > 2 else False,
                str(item[3]) if len(item) > 3 else "nucleotide",
            )
        )
    return loci


@dataclass(frozen=True)
class ParallelismPlan:
    """Result of :func:`plan_parallelism`.

    Attributes:
        workers: Chosen number of parallel workers ``p`` (>= 1).
        requested_workers: Upper bound that was asked for.
        m_peak_bytes: Blueprint ``M_peak`` at ``workers``.
        m_peak_concurrent_bytes: Pessimistic bound where the ``workers`` largest
            MAFFT runs coexist (sum of the top-p MAFFT estimates).
        max_mafft_bytes / parser_buffer_bytes / supermatrix_bytes: Components.
        envelope_bytes: The memory envelope the plan was fitted to.
        fits: ``False`` if even ``workers == 1`` exceeds the envelope.
        limited_by_memory: ``True`` if ``workers < requested_workers``.
        warnings: Human-readable problems (loud when ``fits`` is ``False``).
    """

    workers: int
    requested_workers: int
    m_peak_bytes: int
    m_peak_concurrent_bytes: int
    max_mafft_bytes: int
    parser_buffer_bytes: int
    supermatrix_bytes: int
    envelope_bytes: int
    fits: bool
    limited_by_memory: bool
    warnings: tuple[str, ...] = ()


def plan_parallelism(
    workload_loci: Iterable[Sequence[Any]],
    envelope: int,
    *,
    max_workers: int | None = None,
    ntax: int | None = None,
    total_cols: int | None = None,
    conservative: bool = False,
    model: MemoryModel = DEFAULT_MEMORY_MODEL,
) -> ParallelismPlan:
    """Pick the number of workers so that ``M_peak <= envelope``.

    Implements ``M_peak(p) = max_i(mafft_i) + p * parser_buffer + supermatrix``
    (``parser_buffer`` is the worst per-locus buffer) and returns the largest
    ``p <= max_workers`` that fits. ``p`` is never below 1; if ``p == 1`` still
    exceeds ``envelope`` the plan is returned with ``fits=False`` and a loud
    warning (the run is allowed to proceed - admission control then serialises
    work and the OS decides).

    Args:
        workload_loci: ``(n_seq, max_len, parttree[, alphabet])`` per locus.
        envelope: Memory budget in bytes.
        max_workers: CPU-side upper bound (default: number of loci).
        ntax: Taxa in the supermatrix (default: the largest ``n_seq``).
        total_cols: Pre-trim supermatrix columns (default: ``sum(max_len)``).
        conservative: Fit the *concurrent* bound (sum of the p largest MAFFT
            runs) instead of the blueprint's ``max_i``.
        model: Memory model.
    """
    loci = _normalize_workload(workload_loci)
    envelope = max(0, int(envelope))
    upper = max(1, int(max_workers)) if max_workers is not None else max(1, len(loci))
    warnings: list[str] = []
    if not loci:
        return ParallelismPlan(
            workers=upper,
            requested_workers=upper,
            m_peak_bytes=0,
            m_peak_concurrent_bytes=0,
            max_mafft_bytes=0,
            parser_buffer_bytes=0,
            supermatrix_bytes=0,
            envelope_bytes=envelope,
            fits=True,
            limited_by_memory=False,
        )
    mafft = sorted(
        (
            estimate_mafft_peak_bytes(
                loc.n_seq,
                loc.max_len,
                parttree=loc.parttree,
                alphabet=loc.alphabet,
                model=model,
            )
            for loc in loci
        ),
        reverse=True,
    )
    parser = max(
        estimate_parser_buffer_bytes(loc.n_seq, loc.max_len, model=model)
        for loc in loci
    )
    taxa = ntax if ntax is not None else max(loc.n_seq for loc in loci)
    cols = total_cols if total_cols is not None else sum(loc.max_len for loc in loci)
    sup = estimate_supermatrix_bytes(taxa, cols, model=model)

    def m_peak(p: int) -> int:
        return mafft[0] + p * parser + sup

    def m_concurrent(p: int) -> int:
        return sum(mafft[:p]) + p * parser + sup

    bound = m_concurrent if conservative else m_peak
    chosen = 1
    for p in range(min(upper, len(loci)), 0, -1):
        if bound(p) <= envelope:
            chosen = p
            break
    # more workers than loci never helps, but report the CPU-side request
    fits = bound(chosen) <= envelope
    if not fits:
        warnings.append(
            f"MEMORY ENVELOPE TOO SMALL: even one worker needs about "
            f"{format_bytes(bound(1))} but the envelope is {format_bytes(envelope)}; "
            "continuing with 1 worker - expect swapping or an out-of-memory kill. "
            "Free memory, raise --memory-limit, or use fewer/smaller loci."
        )
    limited = chosen < min(upper, len(loci))
    if limited:
        warnings.append(
            f"workers reduced from {min(upper, len(loci))} to {chosen} to fit the "
            f"{format_bytes(envelope)} memory envelope "
            f"(predicted M_peak {format_bytes(bound(chosen))})"
        )
    return ParallelismPlan(
        workers=chosen,
        requested_workers=upper,
        m_peak_bytes=m_peak(chosen),
        m_peak_concurrent_bytes=m_concurrent(chosen),
        max_mafft_bytes=mafft[0],
        parser_buffer_bytes=parser,
        supermatrix_bytes=sup,
        envelope_bytes=envelope,
        fits=fits,
        limited_by_memory=limited,
        warnings=tuple(warnings),
    )


# ============================================================================ #
# Scratch recommendation
# ============================================================================ #


def _tier_usable(tier: StorageTier) -> bool:
    return tier.writable is not False and tier.free_bytes is not None


class ScratchChoice(NamedTuple):
    """Outcome of :func:`recommend_scratch`.

    Attributes:
        primary: Recommended scratch tier (``None`` -> use the system temp dir).
        fallback: Best non-RAM tier for per-task overflow when tmpfs fills.
        budget_bytes: Bytes of ``primary`` the run may use (RAM tiers are capped
            by the memory envelope, see ``RAM_SCRATCH_ENVELOPE_FRACTION``).
        reason: Human-readable justification.
    """

    primary: StorageTier | None
    fallback: StorageTier | None
    budget_bytes: int | None
    reason: str


def recommend_scratch(host: HostProfile, envelope: int | None = None) -> ScratchChoice:
    """Recommend scratch tiers (RAM first when roomy, else the best disk tier).

    Primary is a RAM tier (tmpfs) whose *budget* - ``min(free, 25% of envelope)``
    when ``envelope`` is given - is at least :data:`MIN_RAM_SCRATCH_FREE_BYTES`;
    otherwise the best disk tier: ``$SLURM_TMPDIR`` first, then NVMe > SSD >
    unknown > HDD, network filesystems last. ``fallback`` is the best non-RAM
    tier (the scheduler falls back to it task by task when tmpfs fills).
    """
    usable = [t for t in host.storage if _tier_usable(t)]

    def ram_budget(t: StorageTier) -> int:
        free = t.free_bytes or 0
        if envelope is None:
            return free
        return min(free, int(envelope * RAM_SCRATCH_ENVELOPE_FRACTION))

    ram = sorted(
        (t for t in usable if t.is_ram and ram_budget(t) >= MIN_RAM_SCRATCH_FREE_BYTES),
        key=lambda t: (-ram_budget(t), t.path),
    )

    def disk_key(t: StorageTier) -> tuple[int, int, int]:
        slurm = 0 if "env:SLURM_TMPDIR" in t.labels and not t.is_network else 1
        enough = 0 if (t.free_bytes or 0) >= MIN_DISK_SCRATCH_FREE_BYTES else 1
        return (enough, slurm, _DEVICE_CLASS_RANK.get(t.device_class, 2))

    disks = sorted(
        (t for t in usable if not t.is_ram),
        key=lambda t: (*disk_key(t), -(t.free_bytes or 0), t.path),
    )
    fallback = disks[0] if disks else None
    if ram:
        primary = ram[0]
        budget = ram_budget(primary)
        reason = (
            f"RAM tier {primary.path} ({format_bytes(primary.free_bytes)} free, budget "
            f"{format_bytes(budget)} = min(free, "
            f"{RAM_SCRATCH_ENVELOPE_FRACTION:g} x envelope)); tmpfs preferred for FASTA I/O"
        )
        return ScratchChoice(primary, fallback, budget, reason)
    small_ram = [t for t in usable if t.is_ram]
    if fallback is not None:
        reason = f"disk tier {fallback.path} ({fallback.device_class}, {fallback.fstype or '?'})"
        if small_ram:
            best = max(small_ram, key=ram_budget)
            reason += (
                f"; RAM tier {best.path} offers only {format_bytes(ram_budget(best))} "
                f"(< {format_bytes(MIN_RAM_SCRATCH_FREE_BYTES)})"
            )
        return ScratchChoice(fallback, fallback, fallback.free_bytes, reason)
    return ScratchChoice(
        None,
        None,
        None,
        "no usable scratch location detected; the system temp dir will be used",
    )


# ============================================================================ #
# Tuning plan
# ============================================================================ #


@dataclass(frozen=True)
class TuningPlan:
    """Immutable scheduling decisions derived from a :class:`HostProfile`.

    Attributes:
        cpu_budget: Total threads the run may use (what ``num_threads`` resolves to).
        workers: Concurrent tasks; equals ``cpu_budget`` unless the memory model
            forces fewer (``parallelism.limited_by_memory``).
        threads_per_task_cap: Most threads one task may lease (bounded by the
            largest NUMA node's share of the budget).
        use_smt: Whether SMT siblings are part of the budget.
        topology: The CPUs the run may use. Without SMT it contains one logical
            CPU per physical core, so ``topology.logical_count == cpu_budget``.
        memory_envelope_bytes: ``0.85 x min(host available, cgroup headroom, user limit)``.
        task_memory_envelope_bytes: The part of the envelope tasks may plan with
            (the envelope minus the RAM-scratch budget when scratch is tmpfs).
        scratch_tier / scratch_path: Primary scratch recommendation.
        scratch_budget_bytes: Bytes of the primary tier the run may use.
        scratch_fallback_path: Best disk tier for per-task tmpfs overflow.
        pinning: PROVISIONAL recommendation to pin workers to cores (Linux, no CPU
            quota, and multiple NUMA nodes or a dedicated SLURM allocation with idle SMT).
        warnings: Everything the user should know (clamps, tiny envelope, ...).
        parallelism: Memory-model result when a workload was supplied.
        memory_model: The (provisional) model used for estimates.
        host: The profile the plan was derived from.
        reasons: Ordered rationale lines used by :meth:`explain`.
    """

    cpu_budget: int
    workers: int
    threads_per_task_cap: int
    use_smt: bool
    topology: CpuTopology
    memory_envelope_bytes: int
    task_memory_envelope_bytes: int
    scratch_tier: StorageTier | None
    scratch_path: str | None
    scratch_budget_bytes: int | None
    scratch_fallback_path: str | None
    pinning: bool
    warnings: tuple[str, ...]
    parallelism: ParallelismPlan | None
    memory_model: MemoryModel
    host: HostProfile
    reasons: tuple[str, ...] = ()

    def est_mem_bytes(
        self,
        n_seq: int,
        max_len: int,
        *,
        parttree: bool = False,
        alphabet: str = "nucleotide",
        threads: int = 1,
    ) -> int:
        """Per-task peak estimate using this plan's memory model."""
        return est_mem_bytes(
            n_seq,
            max_len,
            parttree=parttree,
            alphabet=alphabet,
            threads=threads,
            model=self.memory_model,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary (host profile included)."""
        return dataclasses.asdict(self)

    def explain(self) -> str:
        """Human-readable rationale for every decision in the plan."""
        lines = ["PlastBurstAlign tuning plan"]
        lines.extend(f"  - {reason}" for reason in self.reasons)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        if not self.memory_model.calibrated:
            lines.append(
                "Note: memory model and SMT default are PROVISIONAL (not yet calibrated)."
            )
        return "\n".join(lines)


def _parse_num_threads(num_threads: int | str, warnings: list[str]) -> int | None:
    """Return ``None`` for ``"auto"`` else the requested positive int."""
    if isinstance(num_threads, bool):
        raise ValueError(f"invalid num_threads: {num_threads!r}")
    if isinstance(num_threads, str):
        token = num_threads.strip().lower()
        if token == "auto":
            return None
        try:
            value = int(token)
        except ValueError:
            raise ValueError(
                f"num_threads must be 'auto' or a positive integer, got {num_threads!r}"
            ) from None
    else:
        value = int(num_threads)
    if value < 1:
        warnings.append(f"num_threads={value} is invalid; using 1")
        return 1
    return value


def _restrict_topology(
    topology: CpuTopology, budget: int, use_smt: bool
) -> CpuTopology:
    """Topology of the CPUs actually used for ``budget`` threads."""
    cores = topology.cores
    if not use_smt or topology.smt_width == 1:
        chosen = cores[:budget]
        return CpuTopology(
            cores=tuple(
                CoreInfo(i, c.cpu_ids[:1], c.numa_node) for i, c in enumerate(chosen)
            )
        )
    total = topology.logical_count
    if budget >= total:
        return topology
    keep: list[list[int]] = [[] for _ in cores]
    remaining = budget
    depth = 0
    while remaining > 0:
        progressed = False
        for idx, core in enumerate(cores):
            if depth < len(core.cpu_ids) and remaining > 0:
                keep[idx].append(core.cpu_ids[depth])
                remaining -= 1
                progressed = True
        if not progressed:
            break
        depth += 1
    selected = [(c, sorted(k)) for c, k in zip(cores, keep) if k]
    return CpuTopology(
        cores=tuple(
            CoreInfo(i, tuple(ids), c.numa_node) for i, (c, ids) in enumerate(selected)
        )
    )


def autotune(
    host: HostProfile | None = None,
    *,
    num_threads: int | str = "auto",
    smt: bool | None = None,
    memory_limit: int | str | None = None,
    workload: Iterable[Sequence[Any]] | None = None,
    supermatrix_shape: tuple[int, int] | None = None,
    memory_model: MemoryModel | None = None,
) -> TuningPlan:
    """Derive a :class:`TuningPlan` for this host (or a supplied profile).

    Args:
        host: Host profile; ``None`` re-detects the live machine (memory
            availability is volatile, so the cache is refreshed).
        num_threads: ``"auto"`` = usable *physical* cores (SMT siblings only
            with ``smt=True``). An int is honored but clamped to the usable
            set with a warning.
        smt: ``True``/``False`` forces SMT use; ``None`` uses the (provisional)
            default :data:`DEFAULT_USE_SMT`.
        memory_limit: User cap on memory (bytes or ``"8G"``-style string).
        workload: Optional ``(n_seq, max_len, parttree[, alphabet])`` per locus;
            enables memory-aware worker downscaling.
        supermatrix_shape: Optional ``(ntax, total_cols)`` override for the
            supermatrix buffer estimate.
        memory_model: Override for the (provisional) memory model.

    Returns:
        An immutable plan. Never raises for host-detection problems; invalid
        arguments (a non-integer ``num_threads`` string, unparsable
        ``memory_limit``) raise ``ValueError``.
    """
    if host is None:
        host = detect_host(refresh=True)
    model = memory_model or DEFAULT_MEMORY_MODEL
    warnings: list[str] = []
    reasons: list[str] = []
    topo = host.topology

    # ---- CPU budget ---------------------------------------------------------
    requested = _parse_num_threads(num_threads, warnings)
    smt_default = smt is None
    want_smt = DEFAULT_USE_SMT if smt is None else bool(smt)
    if want_smt and topo.smt_width == 1:
        if smt:
            warnings.append(
                "smt=True requested but the usable CPU set has no SMT siblings"
            )
        want_smt = False
    physical, logical = topo.physical_count, topo.logical_count
    if requested is None:
        use_smt = want_smt
        budget = logical if use_smt else physical
        origin = "provisional default" if smt_default else "by request"
        if use_smt:
            detail = f"all {logical} usable logical CPUs (SMT on, {origin})"
        elif topo.smt_width > 1:
            detail = f"{physical} usable physical cores (SMT siblings idle, {origin})"
        else:
            detail = f"{physical} usable cores (no SMT)"
        reasons.append(f"CPU budget {budget}: num_threads='auto' -> {detail}")
    else:
        budget = requested
        if budget > logical:
            warnings.append(
                f"num_threads={requested} exceeds the {logical} usable CPUs "
                f"({host.host_logical_cpus or '?'} on the host); clamped to {logical}"
            )
            budget = logical
        if budget > physical and smt is False:
            warnings.append(
                f"num_threads={requested} exceeds the {physical} physical cores but "
                f"smt=False; clamped to {physical}"
            )
            budget = physical
        use_smt = budget > physical
        if use_smt and smt is None:
            warnings.append(
                f"num_threads={budget} > {physical} physical cores: SMT siblings will be used"
            )
        reasons.append(f"CPU budget {budget}: honoring num_threads={requested}")
    limit_notes = "; ".join(host.cpu_limit_notes) or "no external limit"
    reasons.append(
        f"usable CPUs: {logical} logical / {physical} physical "
        f"(host {host.host_logical_cpus or '?'} logical); limits: {limit_notes}"
    )
    plan_topology = _restrict_topology(topo, budget, use_smt)

    # ---- threads per task cap (a task never spans NUMA nodes) --------------------
    node_capacity = max(
        (
            sum(len(c.cpu_ids) for c in plan_topology.cores_on_node(n))
            for n in plan_topology.numa_nodes
        ),
        default=budget,
    )
    cap = max(1, min(budget, node_capacity))
    if THREADS_PER_TASK_SOFT_CAP is not None:
        cap = max(1, min(cap, THREADS_PER_TASK_SOFT_CAP))
    reasons.append(
        f"threads per task capped at {cap}"
        + (
            f" (largest NUMA node share of {len(plan_topology.numa_nodes)} nodes)"
            if len(plan_topology.numa_nodes) > 1
            else " (single NUMA node)"
        )
    )

    # ---- memory envelope ------------------------------------------------------
    mem = host.memory
    user_limit: int | None = None
    if memory_limit is not None:
        user_limit = parse_size(memory_limit)
        if user_limit <= 0:
            raise ValueError("memory_limit must be positive")
    candidates: list[tuple[int, str]] = []
    if mem.effective_bytes is not None:
        candidates.append(
            (
                mem.effective_bytes,
                f"host/cgroup effective {format_bytes(mem.effective_bytes)}",
            )
        )
    if user_limit is not None:
        candidates.append((user_limit, f"user limit {format_bytes(user_limit)}"))
        if mem.effective_bytes is not None and user_limit > mem.effective_bytes:
            warnings.append(
                f"memory_limit {format_bytes(user_limit)} exceeds the {format_bytes(mem.effective_bytes)} "
                "currently available; the smaller value is used"
            )
    if candidates:
        base, base_label = min(candidates)
        envelope = int(base * ENVELOPE_FRACTION)
        parts = ", ".join(label for _v, label in candidates)
        reasons.append(
            f"memory envelope {format_bytes(envelope)} = {ENVELOPE_FRACTION:g} x "
            f"min({parts}) [binding: {base_label}]"
        )
    else:
        envelope = int(UNKNOWN_MEMORY_FALLBACK_BYTES * ENVELOPE_FRACTION)
        warnings.append(
            "memory availability could not be determined; assuming "
            f"{format_bytes(UNKNOWN_MEMORY_FALLBACK_BYTES)} (use memory_limit to set it)"
        )
        reasons.append(
            f"memory envelope {format_bytes(envelope)} (fallback, host memory unknown)"
        )
    if mem.limit_sources and mem.effective_bytes is not None:
        reasons.append("memory bound by: " + ", ".join(mem.limit_sources))

    # ---- scratch (tmpfs competes with the tasks for the same memory) --------
    choice = recommend_scratch(host, envelope)
    primary, fallback = choice.primary, choice.fallback
    reasons.append(f"scratch: {choice.reason}")
    if primary is not None and primary.is_network:
        warnings.append(
            f"the best scratch location {primary.path} is on a network filesystem "
            f"({primary.fstype}); set $SLURM_TMPDIR/$PBA_SCRATCH to node-local storage"
        )
    task_envelope = envelope
    if primary is not None and primary.is_ram and choice.budget_bytes:
        task_envelope = max(envelope - choice.budget_bytes, envelope // 2)
        reasons.append(
            f"task memory envelope {format_bytes(task_envelope)} "
            f"(envelope minus {format_bytes(choice.budget_bytes)} of RAM scratch)"
        )

    # ---- parallelism vs. memory ---------------------------------------------
    parallelism: ParallelismPlan | None = None
    workers = budget
    loci = _normalize_workload(workload)
    if loci:
        ntax, cols = (
            supermatrix_shape if supermatrix_shape is not None else (None, None)
        )
        parallelism = plan_parallelism(
            loci,
            task_envelope,
            max_workers=budget,
            ntax=ntax,
            total_cols=cols,
            model=model,
        )
        workers = (
            min(budget, parallelism.workers)
            if parallelism.limited_by_memory
            else budget
        )
        warnings.extend(parallelism.warnings)
        reasons.append(
            f"memory model ({'calibrated' if model.calibrated else 'PROVISIONAL'}): "
            f"M_peak(p={parallelism.workers}) = max MAFFT "
            f"{format_bytes(parallelism.max_mafft_bytes)} + {parallelism.workers} x parser "
            f"{format_bytes(parallelism.parser_buffer_bytes)} + supermatrix "
            f"{format_bytes(parallelism.supermatrix_bytes)} = {format_bytes(parallelism.m_peak_bytes)}"
            f" (pessimistic all-concurrent {format_bytes(parallelism.m_peak_concurrent_bytes)})"
        )
        if not parallelism.fits:
            workers = 1
    reasons.append(f"{workers} concurrent task(s) over {budget} thread(s)")

    # ---- pinning (PROVISIONAL rule) ---------------------------------------------
    # Pin on multi-NUMA hosts (locality) and, on dedicated SLURM allocations, to
    # keep workers off idle SMT siblings. Never under a CPU quota (affinity is
    # meaningless there) and never on shared desktops, where the OS scheduler
    # already balances load better than static pinning.
    multi_numa = len(plan_topology.numa_nodes) > 1
    hidden_smt = topo.smt_width > 1 and not use_smt
    quota_limited = host.cpu_quota is not None
    pinning = (
        host.system == "Linux"
        and workers >= 2
        and not quota_limited
        and (multi_numa or (hidden_smt and host.slurm_job))
    )
    if pinning:
        why = (
            "multiple NUMA nodes"
            if multi_numa
            else "dedicated allocation with idle SMT siblings"
        )
        reasons.append(f"pinning recommended ({why})")
    else:
        reasons.append("pinning not needed")

    for w in warnings:
        log.debug(f"autotune: {w}")
    return TuningPlan(
        cpu_budget=budget,
        workers=max(1, workers),
        threads_per_task_cap=cap,
        use_smt=use_smt,
        topology=plan_topology,
        memory_envelope_bytes=envelope,
        task_memory_envelope_bytes=task_envelope,
        scratch_tier=primary,
        scratch_path=primary.path if primary else None,
        scratch_budget_bytes=choice.budget_bytes,
        scratch_fallback_path=fallback.path if fallback else None,
        pinning=pinning,
        warnings=tuple(dict.fromkeys(warnings)),
        parallelism=parallelism,
        memory_model=model,
        host=host,
        reasons=tuple(reasons),
    )
