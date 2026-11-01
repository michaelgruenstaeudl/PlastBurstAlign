"""Post-flight scaling report: *why* did the run take as long as it did?

:meth:`ScalingReport.from_scheduler` condenses a
:class:`~plastburstalign.scheduler.SchedulerReport` (plus, optionally, the
supermatrix statistics and the scratch-engine accounting) into a handful of
numbers that answer the usual HPC questions:

* **wall time / CPU time** -- the makespan and ``sum(duration x threads)``.
* **parallel efficiency** ``E = busy / (cores x makespan)`` and the measured
  speedup ``S = cpu_seconds / makespan`` (the mean number of busy cores).
* **Karp-Flatt serial fraction** ``e = (1/S - 1/p) / (1 - 1/p)`` with ``p``
  cores. Because ``S`` is *measured work-conserving concurrency*, ``e`` is an
  **upper bound** of the algorithmic serial fraction: it also absorbs the
  straggler tail, memory/scratch admission waits and scheduling gaps. It is
  ``None`` for ``p == 1`` (0/0) and for a degenerate ``S``; otherwise it is
  clamped to ``[0, 1]`` (``S`` slightly above ``p`` can occur through timer
  granularity and would give a meaningless negative value).
* **Amdahl straggler tax** -- the fraction of the makespan during which fewer
  than 50 % of the cores were busy. It is *not* re-derived here: the value is
  taken from the scheduler's own :func:`~plastburstalign.scheduler.compute_schedule_metrics`
  (re-evaluated with that same function only if ``cores`` differs from the
  scheduler's core count).
* **critical path** -- the locus whose completion fixed the makespan, its wall
  time and its share of the run, plus the five longest loci.
* **idle core-seconds** ``cores x makespan - cpu_seconds``.
* **data density** (supermatrix): non-gap fraction, missing fraction, informative
  sites per locus.
* **scratch usage**: RAM-disk vs disk tickets and peak bytes.

Diagnosis rules (first match wins; thresholds are the module constants)
-----------------------------------------------------------------------
1. nothing executed -> "No alignment tasks were executed."
2. run cancelled -> "Run was cancelled ..."
3. at least one task exceeded the memory envelope and ran alone ->
   "Memory-bound: ..."
4. ``cores > 1`` and ``straggler_tax >= STRAGGLER_TAX_THRESHOLD`` and the
   critical locus spans ``>= CRITICAL_SHARE_THRESHOLD`` of the makespan and at
   least ``SHORT_LOCUS_SECONDS`` (a sub-second tail is not actionable) ->
   "Straggler-bound: locus X held N cores idle for Y s; consider enabling
   subdivide" (or, when it was already chunked, a hint to shrink the chunks).
5. ``E < LOW_EFFICIENCY_THRESHOLD`` and the median locus is shorter than
   ``SHORT_LOCUS_SECONDS`` and there are many loci -> "Overhead-bound: ...".
6. ``E < LOW_EFFICIENCY_THRESHOLD`` otherwise -> "Under-utilised: ...".
7. else -> "Well balanced: ...".

This module does not import ``rich`` at import time (only inside
:meth:`ScalingReport.render`), so it is cheap to import in library use.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .hardware import format_bytes
from .scheduler import (
    ScheduleMetrics,
    SchedulerReport,
    TaskRecord,
    compute_schedule_metrics,
)

if TYPE_CHECKING:
    from rich.console import Console, RenderableType

__all__ = [
    "CRITICAL_SHARE_THRESHOLD",
    "LOW_EFFICIENCY_THRESHOLD",
    "SHORT_LOCUS_SECONDS",
    "STRAGGLER_TAX_THRESHOLD",
    "DataDensity",
    "LocusTiming",
    "ScalingReport",
    "ScratchUsage",
    "format_duration",
    "karp_flatt",
    "summarize_supermatrix",
]

#: ``straggler_tax`` at/above which a run may be called straggler-bound.
STRAGGLER_TAX_THRESHOLD: float = 0.20
#: ...provided the critical locus spans at least this share of the makespan.
CRITICAL_SHARE_THRESHOLD: float = 0.35
#: Parallel efficiency below which a run is called under-utilised.
LOW_EFFICIENCY_THRESHOLD: float = 0.60
#: A median locus wall time below this (seconds) makes low efficiency "overhead".
SHORT_LOCUS_SECONDS: float = 2.0
#: How many of the longest loci are listed.
TOP_N_LOCI: int = 5


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def format_duration(seconds: float | None) -> str:
    """Human duration: ``"0.42 s"``, ``"12.3 s"``, ``"4m05s"``, ``"1h02m"``, ``"2d03h"``."""
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    s = max(0.0, float(seconds))
    if s < 1.0:
        return f"{s:.2f} s"
    if s < 60.0:
        return f"{s:.1f} s"
    total = int(round(s))
    minutes, sec = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def karp_flatt(speedup: float | None, procs: int) -> float | None:
    """Karp-Flatt experimentally determined serial fraction.

    ``e = (1/S - 1/p) / (1 - 1/p)``. Returns ``None`` when it is undefined:
    ``p <= 1`` or a missing / non-finite / non-positive speedup. The result is
    clamped to ``[0, 1]``.
    """
    if procs <= 1 or speedup is None or not math.isfinite(speedup) or speedup <= 0:
        return None
    e = (1.0 / speedup - 1.0 / procs) / (1.0 - 1.0 / procs)
    return min(1.0, max(0.0, e))


def _num(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


# ---------------------------------------------------------------------------
# supermatrix summary
# ---------------------------------------------------------------------------


def summarize_supermatrix(sm: Any) -> dict[str, Any]:
    """Plain-dict summary of a :class:`~plastburstalign.supermatrix.Supermatrix`.

    Duck-typed (needs ``ntax``, ``nchar``, ``stats``, ``failed_loci`` and the
    ``density`` / ``informative_sites`` / ``variable_sites`` properties) so this
    module does not import numpy. Keys:

    ``ntax``, ``nchar``, ``n_loci``, ``n_failed_loci``, ``informative_sites``,
    ``variable_sites``, ``density`` (cells that are neither gap nor ``?``),
    ``gap_fraction``, ``missing_fraction`` (nchar-weighted over loci),
    ``informative_mean`` / ``informative_median`` (per locus).
    """
    stats = list(getattr(sm, "stats", ()) or ())
    nchar = int(getattr(sm, "nchar", 0))
    weight = sum(int(s.nchar_kept) for s in stats) or 0
    gap = sum(float(s.gap_fraction) * int(s.nchar_kept) for s in stats)
    missing = sum(float(s.missing_fraction) * int(s.nchar_kept) for s in stats)
    informative = [int(s.informative) for s in stats]
    return {
        "ntax": int(getattr(sm, "ntax", 0)),
        "nchar": nchar,
        "n_loci": len(stats),
        "n_failed_loci": len(getattr(sm, "failed_loci", {}) or {}),
        "informative_sites": int(getattr(sm, "informative_sites", sum(informative))),
        "variable_sites": int(getattr(sm, "variable_sites", 0)),
        "density": float(getattr(sm, "density", 0.0)),
        "gap_fraction": gap / weight if weight else 0.0,
        "missing_fraction": missing / weight if weight else 0.0,
        "informative_mean": statistics.fmean(informative) if informative else 0.0,
        "informative_median": float(statistics.median(informative))
        if informative
        else 0.0,
    }


@dataclass(frozen=True)
class DataDensity:
    """Supermatrix density figures used by the report.

    Attributes:
        ntax, nchar, n_loci: Matrix shape and number of loci kept.
        non_gap_fraction: Fraction of cells holding real data (neither gap nor
            ``?``); the supermatrix' ``density``.
        missing_fraction: Fraction of ``?`` cells (absent taxa / explicit ``?``).
        gap_fraction: Fraction of alignment gap cells.
        informative_sites: Parsimony-informative sites kept.
        informative_per_locus: Mean informative sites per kept locus.
    """

    ntax: int
    nchar: int
    n_loci: int
    non_gap_fraction: float | None
    missing_fraction: float | None
    gap_fraction: float | None
    informative_sites: int
    informative_per_locus: float | None

    @classmethod
    def from_summary(cls, summary: Mapping[str, Any]) -> DataDensity:
        """Build from the dict produced by :func:`summarize_supermatrix` (or alike)."""
        gap = _num(summary, "gap_fraction")
        missing = _num(summary, "missing_fraction")
        dens = _num(summary, "density")
        if dens is None:
            dens = _num(summary, "non_gap_fraction")
        if dens is None and gap is not None and missing is not None:
            dens = max(0.0, 1.0 - gap - missing)
        n_loci = int(_num(summary, "n_loci") or 0)
        informative = int(_num(summary, "informative_sites") or 0)
        per_locus = _num(summary, "informative_mean")
        if per_locus is None and n_loci > 0:
            per_locus = informative / n_loci
        return cls(
            ntax=int(_num(summary, "ntax") or 0),
            nchar=int(_num(summary, "nchar") or 0),
            n_loci=n_loci,
            non_gap_fraction=dens,
            missing_fraction=missing,
            gap_fraction=gap,
            informative_sites=informative,
            informative_per_locus=per_locus,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "ntax": self.ntax,
            "nchar": self.nchar,
            "n_loci": self.n_loci,
            "non_gap_fraction": self.non_gap_fraction,
            "missing_fraction": self.missing_fraction,
            "gap_fraction": self.gap_fraction,
            "informative_sites": self.informative_sites,
            "informative_per_locus": self.informative_per_locus,
        }


# ---------------------------------------------------------------------------
# scratch usage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScratchUsage:
    """Scratch tier usage of the run.

    Attributes:
        source: ``"engine"`` (``ScratchStats``), ``"records"`` (counted from the
            task records' ``scratch_tier``) or ``"none"``.
        ramdisk_tickets, disk_tickets: Per-task scratch directories by tier.
        fallbacks: Tasks that wanted the RAM-disk but fell back to disk.
        ramdisk_peak_bytes: Largest simultaneous RAM-disk reservation.
        peak_bytes: Largest simultaneous reservation over both tiers.
        ramdisk_budget_bytes: Byte budget of the RAM-disk tier.
    """

    source: str = "none"
    ramdisk_tickets: int = 0
    disk_tickets: int = 0
    fallbacks: int = 0
    ramdisk_peak_bytes: int | None = None
    peak_bytes: int | None = None
    ramdisk_budget_bytes: int | None = None

    @property
    def tickets(self) -> int:
        """Total tickets handed out."""
        return self.ramdisk_tickets + self.disk_tickets

    @property
    def ramdisk_fraction(self) -> float | None:
        """Share of tickets that lived on the RAM-disk (``None`` without tickets)."""
        return self.ramdisk_tickets / self.tickets if self.tickets else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "source": self.source,
            "ramdisk_tickets": self.ramdisk_tickets,
            "disk_tickets": self.disk_tickets,
            "fallbacks": self.fallbacks,
            "ramdisk_peak_bytes": self.ramdisk_peak_bytes,
            "peak_bytes": self.peak_bytes,
            "ramdisk_budget_bytes": self.ramdisk_budget_bytes,
        }


def _scratch_usage(
    records: Sequence[TaskRecord], stats: Mapping[str, Any] | None
) -> ScratchUsage:
    if stats:

        def opt(key: str) -> int | None:
            v = _num(stats, key)
            return None if v is None else int(v)

        return ScratchUsage(
            source="engine",
            ramdisk_tickets=opt("ramdisk_tickets") or 0,
            disk_tickets=opt("disk_tickets") or 0,
            fallbacks=opt("fallback_count") or 0,
            ramdisk_peak_bytes=opt("ramdisk_peak_bytes"),
            peak_bytes=opt("peak_bytes_reserved"),
            ramdisk_budget_bytes=opt("ramdisk_budget_bytes"),
        )
    ram = sum(1 for r in records if r.scratch_tier == "ramdisk")
    disk = sum(1 for r in records if r.scratch_tier == "disk")
    if ram + disk == 0:
        return ScratchUsage()
    return ScratchUsage(source="records", ramdisk_tickets=ram, disk_tickets=disk)


# ---------------------------------------------------------------------------
# per-locus timing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocusTiming:
    """Wall and CPU time of one top-level locus (all its units combined).

    Attributes:
        name: Top-level task id.
        wall_seconds: First unit start to last unit end.
        cpu_seconds: ``sum(duration x threads)`` over its units.
        threads_peak: Widest lease of any unit.
        n_units: Executed units (1, or planner + chunks + finalizer).
        chunked: ``True`` if the locus fanned out into sub-locus chunks.
        ok: Whether the locus succeeded.
        share_of_makespan: ``wall_seconds / makespan`` (0 when the makespan is 0).
    """

    name: str
    wall_seconds: float
    cpu_seconds: float
    threads_peak: int
    n_units: int
    chunked: bool
    ok: bool
    share_of_makespan: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "name": self.name,
            "wall_seconds": round(self.wall_seconds, 6),
            "cpu_seconds": round(self.cpu_seconds, 6),
            "threads_peak": self.threads_peak,
            "n_units": self.n_units,
            "chunked": self.chunked,
            "ok": self.ok,
            "share_of_makespan": round(self.share_of_makespan, 6),
        }


def _locus_timings(
    report: SchedulerReport, executed: Sequence[TaskRecord], makespan: float
) -> list[LocusTiming]:
    groups: dict[str, list[TaskRecord]] = defaultdict(list)
    for r in executed:
        groups[r.root_id].append(r)
    out: list[LocusTiming] = []
    for name, recs in groups.items():
        wall = max(r.end for r in recs) - min(r.start for r in recs)
        if name in report.loci:
            ok = report.loci[name][0]
        else:
            ok = all(r.ok for r in recs)
        out.append(
            LocusTiming(
                name=name,
                wall_seconds=max(0.0, wall),
                cpu_seconds=sum(r.duration * r.threads for r in recs),
                threads_peak=max(r.threads for r in recs),
                n_units=len(recs),
                chunked=any(r.kind == "subtask" for r in recs),
                ok=ok,
                share_of_makespan=(wall / makespan) if makespan > 0 else 0.0,
            )
        )
    out.sort(key=lambda t: (-t.wall_seconds, t.name))
    return out


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingReport:
    """Post-flight scaling report (see the module docstring for definitions).

    Build it with :meth:`from_scheduler`; render it with :meth:`to_text`
    (plain), :meth:`render` (rich) or serialise it with :meth:`to_dict`.
    """

    cores: int
    wall_seconds: float
    cpu_seconds: float
    parallel_efficiency: float
    speedup: float | None
    karp_flatt: float | None
    straggler_tax: float
    idle_core_seconds: float
    critical_locus: str | None
    critical_path: tuple[str, ...]
    critical_seconds: float
    critical_share: float
    top_loci: tuple[LocusTiming, ...]
    n_loci: int
    n_ok: int
    n_failed: int
    n_chunked: int
    n_units: int
    failed_loci: tuple[str, ...]
    cancelled: bool
    peak_memory_bytes: int
    oversize_tasks: tuple[str, ...]
    median_locus_seconds: float
    scratch: ScratchUsage
    data: DataDensity | None
    diagnosis: str
    notes: tuple[str, ...] = field(default=())

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_scheduler(
        cls,
        report: SchedulerReport,
        *,
        cores: int,
        supermatrix_summary: Mapping[str, Any] | None = None,
        scratch_stats: Mapping[str, Any] | None = None,
    ) -> ScalingReport:
        """Condense a scheduler report.

        Args:
            report: The :class:`SchedulerReport` of the run.
            cores: Core budget the efficiency refers to (normally
                ``report.total_cores``).
            supermatrix_summary: Dict from :func:`summarize_supermatrix` (or any
                mapping with the same keys); ``None`` omits the density section.
            scratch_stats: ``ScratchStats.as_dict()`` of the scratch engine;
                ``None`` counts tiers from the task records instead.
        """
        p = max(1, int(cores))
        executed = [r for r in report.records if r.status in ("ok", "failed")]
        spans = [(r.start, r.end, r.threads) for r in executed]
        metrics: ScheduleMetrics = (
            report.metrics
            if p == report.total_cores
            else compute_schedule_metrics(spans, p, report.started_at, report.ended_at)
        )
        makespan = metrics.makespan
        cpu = metrics.total_cpu_seconds
        speedup = cpu / makespan if makespan > 0 and cpu > 0 else None
        idle = max(0.0, p * makespan - cpu)

        timings = _locus_timings(report, executed, makespan)
        by_name = {t.name: t for t in timings}
        critical_id = report.critical_task_id
        crit = by_name.get(critical_id) if critical_id is not None else None
        walls = [t.wall_seconds for t in timings]
        median_wall = float(statistics.median(walls)) if walls else 0.0
        n_loci = report.n_loci or len(timings)
        n_ok = report.n_ok if report.loci else sum(1 for t in timings if t.ok)
        n_failed = n_loci - n_ok if report.loci else sum(1 for t in timings if not t.ok)

        last_threads = 1
        if crit is not None:
            chain = [r for r in executed if r.root_id == crit.name]
            if chain:
                last_threads = max(
                    1, max(chain, key=lambda r: (r.end, r.order)).threads
                )

        draft = cls(
            cores=p,
            wall_seconds=makespan,
            cpu_seconds=cpu,
            parallel_efficiency=metrics.parallel_efficiency,
            speedup=speedup,
            karp_flatt=karp_flatt(speedup, p),
            straggler_tax=metrics.straggler_tax,
            idle_core_seconds=idle,
            critical_locus=critical_id,
            critical_path=tuple(report.critical_path),
            critical_seconds=crit.wall_seconds if crit else 0.0,
            critical_share=crit.share_of_makespan if crit else 0.0,
            top_loci=tuple(timings[:TOP_N_LOCI]),
            n_loci=n_loci,
            n_ok=n_ok,
            n_failed=n_failed,
            n_chunked=sum(1 for t in timings if t.chunked),
            n_units=len(executed),
            failed_loci=tuple(report.failed_loci),
            cancelled=report.cancelled,
            peak_memory_bytes=int(report.peak_memory_bytes),
            oversize_tasks=tuple(report.oversize_task_ids),
            median_locus_seconds=median_wall,
            scratch=_scratch_usage(report.records, scratch_stats),
            data=(
                DataDensity.from_summary(supermatrix_summary)
                if supermatrix_summary
                else None
            ),
            diagnosis="",
        )
        diagnosis = draft._diagnose(
            critical_chunked=bool(crit and crit.chunked),
            critical_threads=last_threads,
        )
        notes: list[str] = []
        if draft.n_failed:
            shown = ", ".join(draft.failed_loci[:5])
            more = f" (+{draft.n_failed - 5} more)" if draft.n_failed > 5 else ""
            notes.append(f"{draft.n_failed} locus/loci failed: {shown}{more}")
        if draft.scratch.fallbacks:
            notes.append(
                f"{draft.scratch.fallbacks} task(s) fell back from the RAM-disk to disk scratch"
            )
        return replace(draft, diagnosis=diagnosis, notes=tuple(notes))

    # -- diagnosis --------------------------------------------------------------

    def _diagnose(self, *, critical_chunked: bool, critical_threads: int) -> str:
        if self.n_units == 0:
            return "No alignment tasks were executed."
        if self.cancelled:
            return (
                "Run was cancelled before all loci finished; the figures cover the "
                "executed part only."
            )
        if self.oversize_tasks:
            return (
                f"Memory-bound: {len(self.oversize_tasks)} task(s) exceeded the memory "
                f"envelope and ran alone (peak {format_bytes(self.peak_memory_bytes)}); "
                "raise --memory-limit or free memory."
            )
        tail = self.straggler_tax * self.wall_seconds
        if (
            self.cores > 1
            and self.straggler_tax >= STRAGGLER_TAX_THRESHOLD
            and self.critical_share >= CRITICAL_SHARE_THRESHOLD
            and self.critical_seconds >= SHORT_LOCUS_SECONDS
        ):
            idle_cores = max(1, self.cores - critical_threads)
            advice = (
                "it was already subdivided; consider smaller chunks or a faster "
                "algorithm tier"
                if critical_chunked
                else "consider enabling subdivide (--subdivide auto|force)"
            )
            return (
                f"Straggler-bound: locus {self.critical_locus} held {idle_cores} of "
                f"{self.cores} cores idle for {format_duration(tail)} "
                f"({100.0 * self.critical_share:.0f}% of wall time); {advice}."
            )
        if self.parallel_efficiency < LOW_EFFICIENCY_THRESHOLD:
            if (
                self.median_locus_seconds < SHORT_LOCUS_SECONDS
                and self.n_loci >= 2 * self.cores
            ):
                return (
                    f"Overhead-bound: many short loci (median "
                    f"{format_duration(self.median_locus_seconds)}); per-task startup "
                    f"and I/O dominate at {_pct(self.parallel_efficiency)} efficiency."
                )
            if self.n_loci < self.cores:
                return (
                    f"Under-utilised: only {self.n_loci} loci for {self.cores} cores "
                    f"({_pct(self.parallel_efficiency)} efficiency); fewer cores would "
                    "give the same wall time."
                )
            return (
                f"Under-utilised: {_pct(self.parallel_efficiency)} efficiency without a "
                "dominant straggler; tasks may be waiting for memory or scratch "
                "admission, or on serial phases."
            )
        return (
            f"Well balanced: {_pct(self.parallel_efficiency)} efficiency, straggler tax "
            f"{_pct(self.straggler_tax)}."
        )

    # -- serialisation -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (for the run manifest)."""
        return {
            "cores": self.cores,
            "wall_seconds": round(self.wall_seconds, 6),
            "cpu_seconds": round(self.cpu_seconds, 6),
            "parallel_efficiency": round(self.parallel_efficiency, 6),
            "speedup": None if self.speedup is None else round(self.speedup, 6),
            "karp_flatt": None
            if self.karp_flatt is None
            else round(self.karp_flatt, 6),
            "straggler_tax": round(self.straggler_tax, 6),
            "idle_core_seconds": round(self.idle_core_seconds, 6),
            "critical_locus": self.critical_locus,
            "critical_path": list(self.critical_path),
            "critical_seconds": round(self.critical_seconds, 6),
            "critical_share": round(self.critical_share, 6),
            "top_loci": [t.to_dict() for t in self.top_loci],
            "n_loci": self.n_loci,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_chunked": self.n_chunked,
            "n_units": self.n_units,
            "failed_loci": list(self.failed_loci),
            "cancelled": self.cancelled,
            "peak_memory_bytes": self.peak_memory_bytes,
            "oversize_tasks": list(self.oversize_tasks),
            "median_locus_seconds": round(self.median_locus_seconds, 6),
            "scratch": self.scratch.to_dict(),
            "data": None if self.data is None else self.data.to_dict(),
            "diagnosis": self.diagnosis,
            "notes": list(self.notes),
        }

    def _rows(self) -> list[tuple[str, str]]:
        """Label/value rows shared by the text and rich renderings."""
        rows = [
            (
                "Wall time",
                f"{format_duration(self.wall_seconds)} on {self.cores} cores",
            ),
            ("CPU time", f"{format_duration(self.cpu_seconds)} core-time"),
            (
                "Parallel efficiency",
                f"{_pct(self.parallel_efficiency)}"
                + (f" (speedup {self.speedup:.2f}x)" if self.speedup else ""),
            ),
            (
                "Serial fraction (Karp-Flatt)",
                "n/a" if self.karp_flatt is None else f"{self.karp_flatt:.3f}",
            ),
            (
                "Straggler tax",
                f"{_pct(self.straggler_tax)} of wall time below 50% cores busy",
            ),
            ("Idle capacity", f"{format_duration(self.idle_core_seconds)} core-time"),
            (
                "Loci",
                f"{self.n_ok}/{self.n_loci} ok"
                + (f", {self.n_failed} failed" if self.n_failed else "")
                + (f", {self.n_chunked} chunked" if self.n_chunked else ""),
            ),
        ]
        if self.critical_locus is not None:
            rows.append(
                (
                    "Critical path",
                    f"{self.critical_locus}: {format_duration(self.critical_seconds)} "
                    f"({_pct(self.critical_share)} of wall time)",
                )
            )
        if self.peak_memory_bytes:
            rows.append(("Peak admitted memory", format_bytes(self.peak_memory_bytes)))
        s = self.scratch
        if s.tickets:
            frac = s.ramdisk_fraction
            rows.append(
                (
                    "Scratch",
                    f"{s.ramdisk_tickets} RAM-disk / {s.disk_tickets} disk tickets"
                    + (f" ({_pct(frac)} RAM)" if frac is not None else "")
                    + (
                        f", peak {format_bytes(s.peak_bytes)}"
                        if s.peak_bytes is not None
                        else ""
                    ),
                )
            )
        d = self.data
        if d is not None:
            rows.append(
                ("Matrix", f"{d.ntax} taxa x {d.nchar} sites in {d.n_loci} loci")
            )
            rows.append(
                (
                    "Data density",
                    f"{_pct(d.non_gap_fraction)} data, {_pct(d.missing_fraction)} missing, "
                    f"{_pct(d.gap_fraction)} gaps",
                )
            )
            if d.informative_per_locus is not None:
                rows.append(
                    (
                        "Informative sites",
                        f"{d.informative_sites} ({d.informative_per_locus:.1f} per locus)",
                    )
                )
        return rows

    def to_text(self, title: str = "PlastBurstAlign scaling report") -> str:
        """Plain-text report (no ``rich``)."""
        rows = self._rows()
        width = max(len(k) for k, _ in rows)
        lines = [title]
        lines.extend(f"  {k:<{width}}  {v}" for k, v in rows)
        if self.top_loci:
            lines.append(f"  Longest loci (top {len(self.top_loci)}):")
            for t in self.top_loci:
                lines.append(
                    f"    {t.name:<28} {format_duration(t.wall_seconds):>9}  "
                    f"{_pct(t.share_of_makespan):>6} of wall  "
                    f"{t.threads_peak:>3} thr" + ("  chunked" if t.chunked else "")
                )
        lines.append(f"  Diagnosis: {self.diagnosis}")
        lines.extend(f"  Note: {n}" for n in self.notes)
        return "\n".join(lines)

    def renderable(self, title: str = "Scaling report") -> RenderableType:
        """The report as a ``rich`` renderable (panel with tables)."""
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column()
        for key, value in self._rows():
            grid.add_row(key, value)
        parts: list[RenderableType] = [grid]
        if self.top_loci:
            table = Table(
                title=f"Longest loci (top {len(self.top_loci)})",
                title_justify="left",
                show_edge=False,
                header_style="bold",
            )
            table.add_column("Locus", overflow="fold")
            table.add_column("Wall", justify="right")
            table.add_column("% of run", justify="right")
            table.add_column("Threads", justify="right")
            table.add_column("Chunked", justify="center")
            for t in self.top_loci:
                table.add_row(
                    t.name,
                    format_duration(t.wall_seconds),
                    _pct(t.share_of_makespan),
                    str(t.threads_peak),
                    "yes" if t.chunked else "",
                )
            parts.append(table)
        healthy = self.diagnosis.startswith("Well balanced")
        parts.append(
            f"[{'green' if healthy else 'yellow'}]Diagnosis:[/] "
            f"{_escape(self.diagnosis)}"
        )
        for note in self.notes:
            parts.append(f"[dim]Note: {_escape(note)}[/]")
        return Panel(
            Group(*parts),
            title=title,
            title_align="left",
            border_style="green" if healthy else "yellow",
        )

    def render(self, console: Console | None = None) -> None:
        """Print the report to ``console`` (default: a new stderr console)."""
        if console is None:
            from rich.console import Console as _Console

            console = _Console(stderr=True)
        console.print(self.renderable())


def _escape(text: str) -> str:
    from rich.markup import escape

    return escape(text)
