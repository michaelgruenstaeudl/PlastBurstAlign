"""Dynamic locus scheduler: LPT queue, memory admission, straggler promotion, fan-out.

``LocusScheduler`` replaces "submit every locus to a ``ThreadPoolExecutor`` and let
each worker block on a ``CoreBudgetPool`` lease" with a single priority queue that
worker threads pull from:

* **LPT priority** -- the heaviest ready task (by estimated ``work``) is always
  started first; ties break by submission order, so scheduling is deterministic.
* **CPU-set leases** -- every started task owns a concrete :class:`CoreLease`
  (physical cores first, NUMA best-fit) and learns it via :class:`TaskContext`
  (``ctx.threads``, ``ctx.cpu_ids``, optional ``taskset`` pinning).
* **Memory admission** -- a task starts only if its predicted RSS fits the
  remaining :class:`MemoryBudget`; a task larger than the whole envelope runs
  alone (with a warning) and can never deadlock the run. A bounded number of
  smaller tasks may *backfill* past a memory-blocked head so cores do not idle,
  after which the head is waited for (no starvation).
* **Straggler promotion** -- when fewer tasks are ready than cores are idle, the
  remaining tasks lease all idle cores up to their ``max_threads``
  (:func:`lease_width`).
* **Fan-out / fan-in** -- a task may return :class:`FanOut` (or a list of
  :class:`SchedTask`); the subtasks enter the same queue, and a finalizer runs
  once all of them completed, so one huge locus can be split into chunks that
  absorb idle cores.
* **Failure isolation, bounded memory** -- an exception in a task is recorded and
  never aborts other tasks (unless ``fail_fast``); no per-task futures exist, so
  13k+ tasks cost a few small objects each.

The pure helpers (:class:`ReadyQueue`, :func:`lease_width`,
:func:`compute_schedule_metrics`) are shared with ``scheduler_sim`` so the policy
that is simulated is the policy that ships.
"""

from __future__ import annotations

import heapq
import math
import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from .events import (
    LocusFinished,
    LocusQueued,
    LocusStarted,
    PhaseChanged,
    ProgressEvent,
    ProgressSink,
    WarningEvent,
    safe_emit,
)
from .logging_ops import logger as log
from .task_queue import CoreLease, CpuSetAllocator, pin_command
from .topology import CpuTopology

__all__ = [
    "FanOut",
    "LocusScheduler",
    "MemoryBudget",
    "ReadyQueue",
    "ScheduleMetrics",
    "SchedTask",
    "SchedulerReport",
    "ScratchTicketLike",
    "TaskContext",
    "TaskRecord",
    "compute_schedule_metrics",
    "lease_width",
]

T = TypeVar("T")
R = TypeVar("R")

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Pure building blocks (shared with the simulator)
# ---------------------------------------------------------------------------


class ReadyQueue(Generic[T]):
    """Priority queue with deterministic LPT order and bounded backfill.

    Items are ordered by ``key`` (smallest first). :meth:`pop_next` hands each
    candidate, in order, to an ``admit`` callback; the first item the callback
    accepts is returned and the rejected ones stay queued. After a *blocked head*
    at most ``max_backfill`` later items may be admitted before the queue holds
    back everything until the head fits (so a big task is never starved), and at
    most ``max_scan`` candidates are examined per call (O(1) with 13k items).
    """

    def __init__(self, max_backfill: int = 16, max_scan: int = 64) -> None:
        self._heap: list[tuple[tuple[Any, ...], int, float, T]] = []
        self._serial = 0
        self._count = 0
        self._weight = 0.0
        self._streak = 0
        self.max_backfill = max(0, int(max_backfill))
        self.max_scan = max(1, int(max_scan))

    def __len__(self) -> int:
        return self._count

    @property
    def weight(self) -> float:
        """Sum of the weights of all queued items."""
        return self._weight if self._count else 0.0

    def push(self, key: tuple[Any, ...], item: T, weight: float = 0.0) -> None:
        """Queue ``item`` under ``key`` (``weight`` feeds promotion shares)."""
        heapq.heappush(self._heap, (key, self._serial, weight, item))
        self._serial += 1
        self._count += 1
        self._weight += weight

    def push_many(self, entries: Iterable[tuple[tuple[Any, ...], T, float]]) -> None:
        """Bulk-load entries (``heapify`` once; O(n))."""
        for key, item, weight in entries:
            self._heap.append((key, self._serial, weight, item))
            self._serial += 1
            self._count += 1
            self._weight += weight
        heapq.heapify(self._heap)

    def pop_next(
        self, admit: Callable[[T, int, float], R | None]
    ) -> tuple[T, R] | None:
        """Pop the first admissible item.

        ``admit(item, n_others, others_weight)`` returns a non-``None`` decision to
        accept; ``n_others``/``others_weight`` describe the other queued items.
        """
        examined: list[tuple[tuple[Any, ...], int, float, T]] = []
        result: tuple[T, R] | None = None
        entry: tuple[tuple[Any, ...], int, float, T] | None = None
        try:
            while self._heap:
                if examined and (
                    self._streak >= self.max_backfill or len(examined) >= self.max_scan
                ):
                    break
                entry = heapq.heappop(self._heap)
                _key, _serial, weight, item = entry
                decision = admit(item, self._count - 1, self._weight - weight)
                if decision is not None:
                    result = (item, decision)
                    self._count -= 1
                    self._weight -= weight
                    self._streak = self._streak + 1 if examined else 0
                    entry = None
                    break
                examined.append(entry)
                entry = None
        finally:
            if entry is not None:  # admit raised: keep the item queued
                examined.append(entry)
            for e in examined:
                heapq.heappush(self._heap, e)
        return result

    def remove_where(self, predicate: Callable[[T], bool]) -> list[T]:
        """Remove (and return, in priority order) every item matching ``predicate``."""
        keep: list[tuple[tuple[Any, ...], int, float, T]] = []
        gone: list[tuple[tuple[Any, ...], int, float, T]] = []
        for e in self._heap:
            (gone if predicate(e[3]) else keep).append(e)
        if gone:
            self._heap = keep
            heapq.heapify(self._heap)
            self._count = len(keep)
            self._weight = sum(e[2] for e in keep)
        gone.sort(key=lambda e: (e[0], e[1]))
        return [e[3] for e in gone]

    def drain(self) -> list[T]:
        """Remove and return all items in priority order."""
        return self.remove_where(lambda _item: True)


def lease_width(
    *,
    work: float,
    preferred: int,
    min_threads: int,
    max_threads: int,
    idle: int,
    n_others: int,
    others_work: float,
    running: int,
    promote: bool = True,
) -> int | None:
    """How many cores should a task take right now? ``None`` means "wait".

    * Base width is ``preferred`` clamped to ``[min_threads, max_threads]``; the
      task accepts as few as half of it (never fewer than ``min_threads``) so a
      multi-core request does not wait for a perfect gap.
    * **Straggler promotion**: if fewer tasks are ready than cores are idle
      (``n_others + 1 < idle``) the task takes its work-proportional share of
      the idle cores (rounded up, leaving one core for every other ready task),
      up to ``max_threads``.
    * When nothing is running the task always starts, so the queue cannot stall.
    """
    if idle < 1:
        return None
    maxt = max(1, int(max_threads))
    mint = min(max(1, int(min_threads)), maxt)
    pref = min(max(int(preferred), mint), maxt)
    want = pref
    n_queue = n_others + 1
    if promote and maxt > pref and n_queue < idle:
        total = work + others_work
        share_f = idle * (work / total) if total > 0 else idle / n_queue
        share = max(1, math.ceil(share_f - _EPS))
        cap = idle - n_others
        want = max(pref, min(maxt, share, cap))
    need = max(mint, (pref + 1) // 2)
    if running == 0:
        need = min(need, idle)
    if idle < need:
        return None
    return min(want, idle)


@dataclass(frozen=True)
class ScheduleMetrics:
    """Utilization metrics of one schedule (shared by the scheduler and simulator).

    Attributes:
        makespan: Wall time of the schedule.
        total_cpu_seconds: ``sum(duration * threads)`` over all executed units.
        parallel_efficiency: ``busy / (cores * makespan)`` in ``[0, 1]``.
        straggler_tax: Fraction of the makespan during which fewer than 50 % of the
            cores were busy (the tail where a few stragglers run alone).
        mean_busy_cores: Time-averaged number of busy cores.
    """

    makespan: float
    total_cpu_seconds: float
    parallel_efficiency: float
    straggler_tax: float
    mean_busy_cores: float


def compute_schedule_metrics(
    intervals: Iterable[tuple[float, float, int]],
    total_cores: int,
    t_start: float | None = None,
    t_end: float | None = None,
) -> ScheduleMetrics:
    """Compute :class:`ScheduleMetrics` from ``(start, end, threads)`` intervals.

    The window defaults to ``[min(start), max(end)]``; pass ``t_start``/``t_end``
    to measure over the scheduler's own wall-clock window instead.
    """
    cores = max(1, int(total_cores))
    spans = [
        (float(s), float(e), int(th)) for s, e, th in intervals if e > s and th > 0
    ]
    if not spans and (t_start is None or t_end is None):
        return ScheduleMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    t0 = t_start if t_start is not None else min(s for s, _, _ in spans)
    t1 = t_end if t_end is not None else max(e for _, e, _ in spans)
    makespan = max(0.0, t1 - t0)
    busy = sum(
        (min(e, t1) - max(s, t0)) * th for s, e, th in spans if e > t0 and s < t1
    )
    if makespan <= 0:
        return ScheduleMetrics(0.0, busy, 0.0, 0.0, 0.0)

    points: list[tuple[float, int]] = []
    for s, e, th in spans:
        points.append((s, th))
        points.append((e, -th))
    points.sort()
    threshold = 0.5 * cores
    cur = 0
    last = t0
    low = 0.0
    for t, delta in points:
        t = min(max(t, t0), t1)
        if t > last:
            if cur < threshold - _EPS:
                low += t - last
            last = t
        cur += delta
    if t1 > last and cur < threshold - _EPS:
        low += t1 - last
    return ScheduleMetrics(
        makespan=makespan,
        total_cpu_seconds=busy,
        parallel_efficiency=min(1.0, busy / (cores * makespan)),
        straggler_tax=min(1.0, low / makespan),
        mean_busy_cores=busy / makespan,
    )


class MemoryBudget:
    """Thread-safe RSS admission envelope.

    ``total_bytes=None`` means "unlimited" (usage is still tracked). A request is
    granted when it fits the remaining envelope; a request *larger than the whole
    envelope* is granted only when nothing else holds memory (it then runs alone,
    see :meth:`is_oversize`), so an oversized task can never deadlock the run.
    Zero-byte requests are always granted.
    """

    def __init__(self, total_bytes: int | None = None) -> None:
        self.total: int | None = (
            None if total_bytes is None else max(0, int(total_bytes))
        )
        self._lock = threading.Lock()
        self._used = 0
        self._peak = 0
        self._oversize = 0

    @property
    def used(self) -> int:
        """Bytes currently reserved."""
        with self._lock:
            return self._used

    @property
    def peak(self) -> int:
        """Largest ``used`` value observed."""
        with self._lock:
            return self._peak

    @property
    def remaining(self) -> int | None:
        """Bytes still free (negative while an oversize task runs); ``None`` if unlimited."""
        with self._lock:
            return None if self.total is None else self.total - self._used

    @property
    def oversize_grants(self) -> int:
        """Number of oversize requests granted so far."""
        with self._lock:
            return self._oversize

    def is_oversize(self, nbytes: int) -> bool:
        """True if ``nbytes`` can never fit the envelope (needs exclusive access)."""
        return self.total is not None and nbytes > self.total

    def try_acquire(self, nbytes: int) -> bool:
        """Reserve ``nbytes`` if admissible; never blocks."""
        n = max(0, int(nbytes))
        if n == 0:
            return True
        with self._lock:
            if self.total is not None and self._used + n > self.total:
                if not (n > self.total and self._used == 0):
                    return False
                self._oversize += 1
            self._used += n
            self._peak = max(self._peak, self._used)
            return True

    def force_acquire(self, nbytes: int) -> None:
        """Reserve unconditionally (deadlock guard of last resort)."""
        n = max(0, int(nbytes))
        with self._lock:
            self._used += n
            self._peak = max(self._peak, self._used)

    def release(self, nbytes: int) -> None:
        """Return ``nbytes`` to the envelope."""
        n = max(0, int(nbytes))
        with self._lock:
            self._used = max(0, self._used - n)


# ---------------------------------------------------------------------------
# Public task / context / report types
# ---------------------------------------------------------------------------


class ScratchTicketLike(Protocol):
    """Structural type of ``ephemeral_engine.ScratchTicket``."""

    @property
    def path(self) -> Path: ...

    @property
    def tier(self) -> str: ...

    def release(self) -> None: ...


class ScratchProvider(Protocol):
    """Structural type of ``ephemeral_engine.ScratchEngine`` (byte admission)."""

    def reserve(self, nbytes: int) -> ScratchTicketLike: ...


@dataclass(frozen=True)
class TaskContext:
    """What a running task learns about its resources.

    Attributes:
        task_id: Full task id (``parent/child`` for subtasks).
        threads: Threads to use (the lease size; pass to ``aligner.align(threads=)``).
        cpu_ids: OS CPU ids owned by the lease.
        lease: The full :class:`CoreLease`.
        scratch: Per-task scratch ticket when ``scratch_bytes > 0`` and the
            scheduler has a scratch provider, else ``None``.
        parent_id: Owning task for subtasks / finalizers.
        kind: ``"task"``, ``"subtask"`` or ``"finalize"``.
        pinning: Whether :meth:`pin` actually prefixes ``taskset``.
        cancel_event: Set when the run is cancelled; long tasks may poll it.
    """

    task_id: str
    threads: int
    cpu_ids: tuple[int, ...]
    lease: CoreLease
    scratch: ScratchTicketLike | None = None
    parent_id: str | None = None
    kind: str = "task"
    pinning: bool = False
    cancel_event: threading.Event = field(
        default_factory=threading.Event, compare=False, repr=False
    )

    @property
    def cancelled(self) -> bool:
        """True once the run was cancelled (Ctrl-C, ``fail_fast``)."""
        return self.cancel_event.is_set()

    def pin(self, cmd: Sequence[str]) -> list[str]:
        """``cmd`` prefixed with ``taskset -c <cpus>`` if pinning is on and available."""
        if not self.pinning:
            return list(cmd)
        return pin_command(cmd, self.cpu_ids)

    def subprocess_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Environment for a child process: ``TMPDIR`` -> this task's scratch dir."""
        env = dict(base if base is not None else os.environ)
        if self.scratch is not None:
            env["TMPDIR"] = str(self.scratch.path)
        return env


@dataclass
class SchedTask:
    """One unit of schedulable work.

    Attributes:
        task_id: Unique id among top-level tasks (subtask ids only need to be
            unique within their parent; the full id is ``parent/child``).
        fn: Callable receiving a :class:`TaskContext`. It may return a
            :class:`FanOut` (or a non-empty list of :class:`SchedTask`) to split
            itself into subtasks; any other value is the task's result.
        work: Estimated work; sets LPT priority and promotion shares.
        est_mem_bytes: Predicted peak RSS for memory admission.
        preferred_threads: Width requested when the queue is deep.
        max_threads: Largest useful width (straggler promotion cap). Defaults to
            ``preferred_threads``, i.e. promotion is opt-in per task.
        min_threads: Smallest acceptable width.
        scratch_bytes: Scratch bytes to reserve (``ctx.scratch``); 0 = none.
        n_seq, max_len: Descriptive sizes carried into progress events.
        priority: Overrides ``work`` for queue ordering when given.
        finalizer: Default fan-in callback ``(ctx, subtask_results) -> result`` used
            when the task fans out without supplying its own.
        meta: Free-form metadata (not interpreted).
    """

    task_id: str
    fn: Callable[[TaskContext], Any]
    work: float = 1.0
    est_mem_bytes: int = 0
    preferred_threads: int = 1
    max_threads: int | None = None
    min_threads: int = 1
    scratch_bytes: int = 0
    n_seq: int = 0
    max_len: int = 0
    priority: float | None = None
    finalizer: Callable[[TaskContext, list[Any]], Any] | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FanOut:
    """Return value of a task function that splits itself into subtasks.

    Attributes:
        subtasks: Tasks entering the same priority queue.
        finalizer: ``(ctx, results_in_subtask_order) -> result``; runs once, as its
            own highest-priority single-thread unit, after all subtasks succeeded.
            If ``None`` the parent's result is the list of subtask results.
        finalize_work: Estimated work of the finalizer.
        finalize_est_mem_bytes: Predicted RSS of the finalizer.
    """

    subtasks: Sequence[SchedTask]
    finalizer: Callable[[TaskContext, list[Any]], Any] | None = None
    finalize_work: float = 0.0
    finalize_est_mem_bytes: int = 0


@dataclass(frozen=True)
class TaskRecord:
    """Outcome of one executed (or cancelled) scheduling unit."""

    task_id: str
    parent_id: str | None
    root_id: str
    kind: str
    order: tuple[int, ...]
    work: float
    est_mem_bytes: int
    threads: int
    cpu_ids: tuple[int, ...]
    ready: float
    start: float
    end: float
    ok: bool
    status: str
    error: str | None = None
    scratch_tier: str | None = None
    oversize: bool = False

    @property
    def duration(self) -> float:
        """Run time of the unit in seconds."""
        return max(0.0, self.end - self.start)

    @property
    def wait(self) -> float:
        """Time between becoming ready and starting."""
        return max(0.0, self.start - self.ready)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "work": self.work,
            "threads": self.threads,
            "cpu_ids": list(self.cpu_ids),
            "wait": round(self.wait, 6),
            "duration": round(self.duration, 6),
            "ok": self.ok,
            "status": self.status,
            "error": self.error,
            "scratch_tier": self.scratch_tier,
        }


@dataclass
class SchedulerReport:
    """Result of :meth:`LocusScheduler.run`.

    ``records`` is in deterministic *submission* order (top-level task order, each
    task followed by its subtasks and finalizer), independent of completion order.
    """

    records: list[TaskRecord]
    total_cores: int
    started_at: float
    ended_at: float
    cancelled: bool = False
    loci: dict[str, tuple[bool, str | None]] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    peak_memory_bytes: int = 0
    oversize_task_ids: tuple[str, ...] = ()

    @property
    def wall_seconds(self) -> float:
        """Wall time of the run."""
        return max(0.0, self.ended_at - self.started_at)

    @property
    def failed(self) -> list[TaskRecord]:
        """Unit records that raised."""
        return [r for r in self.records if r.status == "failed"]

    @property
    def n_loci(self) -> int:
        """Number of top-level tasks submitted."""
        return len(self.loci)

    @property
    def n_ok(self) -> int:
        """Number of top-level tasks that completed successfully (after fan-in)."""
        return sum(1 for ok, _ in self.loci.values() if ok)

    @property
    def n_failed_roots(self) -> int:
        """Number of top-level tasks that failed or were cancelled."""
        return self.n_loci - self.n_ok

    @property
    def failed_loci(self) -> list[str]:
        """Ids of top-level tasks that failed or were cancelled (submission order)."""
        return [rid for rid, (ok, _) in self.loci.items() if not ok]

    @property
    def ok(self) -> bool:
        """True if every top-level task succeeded and the run was not cancelled."""
        return not self.cancelled and self.n_failed_roots == 0

    def record(self, task_id: str) -> TaskRecord | None:
        """Look up a record by full task id."""
        for r in self.records:
            if r.task_id == task_id:
                return r
        return None

    @cached_property
    def metrics(self) -> ScheduleMetrics:
        """Makespan, CPU-seconds, parallel efficiency and straggler tax."""
        spans = [
            (r.start, r.end, r.threads)
            for r in self.records
            if r.status in ("ok", "failed")
        ]
        return compute_schedule_metrics(
            spans, self.total_cores, self.started_at, self.ended_at
        )

    @property
    def makespan(self) -> float:
        """Wall time of the run (alias of ``metrics.makespan``)."""
        return self.metrics.makespan

    @property
    def total_cpu_seconds(self) -> float:
        """``sum(duration * threads)`` over all executed units."""
        return self.metrics.total_cpu_seconds

    @property
    def parallel_efficiency(self) -> float:
        """``busy / (cores * makespan)``."""
        return self.metrics.parallel_efficiency

    @property
    def straggler_tax(self) -> float:
        """Fraction of the makespan with fewer than 50 % of the cores busy."""
        return self.metrics.straggler_tax

    @cached_property
    def critical_path(self) -> tuple[str, ...]:
        """Unit ids of the chain that finished last (root, latest subtask, finalizer)."""
        executed = [r for r in self.records if r.status in ("ok", "failed")]
        if not executed:
            return ()
        by_parent: dict[str, list[TaskRecord]] = defaultdict(list)
        by_id: dict[str, TaskRecord] = {}
        for r in executed:
            by_id[r.task_id] = r
            if r.parent_id is not None:
                by_parent[r.parent_id].append(r)
        last = max(executed, key=lambda r: (r.end, r.order))
        root_id = last.root_id

        def chain(task_id: str) -> list[str]:
            kids = [k for k in by_parent.get(task_id, []) if k.kind == "subtask"]
            out = [task_id]
            if kids:
                latest = max(kids, key=lambda k: (k.end, k.order))
                out.extend(chain(latest.task_id))
                fin = [k for k in by_parent.get(task_id, []) if k.kind == "finalize"]
                out.extend(k.task_id for k in fin)
            return out

        return tuple(chain(root_id))

    @property
    def critical_task_id(self) -> str | None:
        """Top-level task whose completion determined the makespan."""
        path = self.critical_path
        return path[0] if path else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary (for the run manifest / scaling report)."""
        m = self.metrics
        return {
            "total_cores": self.total_cores,
            "makespan": round(m.makespan, 6),
            "total_cpu_seconds": round(m.total_cpu_seconds, 6),
            "parallel_efficiency": round(m.parallel_efficiency, 6),
            "straggler_tax": round(m.straggler_tax, 6),
            "critical_task_id": self.critical_task_id,
            "critical_path": list(self.critical_path),
            "cancelled": self.cancelled,
            "n_tasks": self.n_loci,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed_roots,
            "failed_loci": self.failed_loci,
            "n_units": len(self.records),
            "peak_memory_bytes": self.peak_memory_bytes,
            "oversize_task_ids": list(self.oversize_task_ids),
        }

    def summary(self) -> str:
        """One-paragraph human summary."""
        m = self.metrics
        crit = self.critical_task_id or "-"
        return (
            f"{self.n_ok}/{self.n_loci} loci ok "
            f"in {m.makespan:.2f}s on {self.total_cores} cores; "
            f"efficiency {100 * m.parallel_efficiency:.1f}%, "
            f"straggler tax {100 * m.straggler_tax:.1f}%, critical path `{crit}`"
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class _Node:
    """Internal bookkeeping for one scheduling unit."""

    __slots__ = (
        "task",
        "node_id",
        "kind",
        "order",
        "parent",
        "root",
        "idx",
        "ready_at",
        "state",
        "pref",
        "mint",
        "maxt",
        "mem_held",
        "oversize",
        "first_start",
        "threads_peak",
        "children",
        "child_results",
        "pending_children",
        "finalizer",
        "finalize_work",
        "finalize_mem",
        "finalize_node",
        "serial",
    )

    def __init__(
        self,
        task: SchedTask,
        node_id: str,
        kind: str,
        order: tuple[int, ...],
        parent: _Node | None,
        idx: int,
        serial: int,
    ) -> None:
        self.task = task
        self.node_id = node_id
        self.kind = kind
        self.order = order
        self.parent = parent
        self.root: _Node = parent.root if parent is not None else self
        self.idx = idx
        self.serial = serial
        self.ready_at = 0.0
        self.state = "new"  # new|queued|running|waiting|done
        self.pref = 1
        self.mint = 1
        self.maxt = 1
        self.mem_held = 0
        self.oversize = False
        self.first_start: float | None = None
        self.threads_peak = 0
        self.children: list[_Node] = []
        self.child_results: list[Any] = []
        self.pending_children = 0
        self.finalizer: Callable[[TaskContext, list[Any]], Any] | None = None
        self.finalize_work = 0.0
        self.finalize_mem = 0
        self.finalize_node: _Node | None = None


@dataclass
class _Assignment:
    node: _Node
    lease: CoreLease
    start: float
    wait: float
    events: list[ProgressEvent]


@dataclass(frozen=True)
class _Admission:
    threads: int
    oversize: bool


def _is_subtask_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, SchedTask) for v in value)
    )


class LocusScheduler:
    """Work-stealing-style dynamic scheduler for loci (see module docstring).

    Args:
        total_cores: Core budget. With ``topology`` it restricts the usable slots;
            without it a flat topology is assumed. Defaults to ``os.cpu_count()``.
        topology: CPU topology for NUMA/physical-core aware leases.
        use_smt: Treat SMT siblings as separate lease units.
        memory_budget: ``MemoryBudget`` (shared) or an int of bytes, ``None`` = unlimited.
        scratch: Optional scratch provider (``ScratchEngine``); tasks with
            ``scratch_bytes > 0`` get a ticket in ``ctx.scratch``.
        sink: Progress sink receiving ``LocusQueued/Started/Finished`` events.
        pin: Let ``ctx.pin(cmd)`` prefix ``taskset -c``.
        fail_fast: Cancel everything after the first failure (default: isolate).
        max_workers: Worker thread count (default: number of core slots).
        promote_stragglers: Enable straggler promotion.
        max_backfill, max_scan: See :class:`ReadyQueue`.
        keep_results: Keep top-level task results in ``report.results``.
        shutdown_timeout: Seconds to wait for running tasks after an interrupt of
            the calling thread (``None`` = wait for them).
        clock: Monotonic clock (injectable for tests).
    """

    def __init__(
        self,
        total_cores: int | None = None,
        *,
        topology: CpuTopology | None = None,
        use_smt: bool = False,
        memory_budget: MemoryBudget | int | None = None,
        scratch: ScratchProvider | None = None,
        sink: ProgressSink | None = None,
        pin: bool = False,
        fail_fast: bool = False,
        max_workers: int | None = None,
        promote_stragglers: bool = True,
        max_backfill: int = 16,
        max_scan: int = 64,
        keep_results: bool = False,
        shutdown_timeout: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if topology is None:
            cores = int(total_cores) if total_cores else (os.cpu_count() or 1)
            topology = CpuTopology.flat(max(1, cores))
            max_slots = None
        else:
            max_slots = int(total_cores) if total_cores else None
        self._topology = topology
        self._use_smt = bool(use_smt)
        self._max_slots = max_slots
        probe = CpuSetAllocator(topology, use_smt=self._use_smt, max_slots=max_slots)
        self.total_cores = probe.total_slots
        self._max_workers = max(
            1, int(max_workers) if max_workers else self.total_cores
        )
        self._memory_arg = memory_budget
        self._scratch = scratch
        self._sink = sink
        self._pin = bool(pin)
        self._fail_fast = bool(fail_fast)
        self._promote = bool(promote_stragglers)
        self._max_backfill = max_backfill
        self._max_scan = max_scan
        self._keep_results = bool(keep_results)
        self._shutdown_timeout = shutdown_timeout
        self._clock = clock

        self._run_lock = threading.Lock()
        self._cond = threading.Condition(threading.RLock())
        self._cancel_event = threading.Event()
        self._done_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self.last_report: SchedulerReport | None = None
        self._reset()

    # -- public API ----------------------------------------------------------

    @property
    def memory_budget(self) -> MemoryBudget:
        """The memory envelope of the current/last run."""
        return self._mem

    def alive_workers(self) -> int:
        """Number of worker threads still alive (0 after a clean ``run``)."""
        return sum(1 for t in self._threads if t.is_alive())

    def cancel(self) -> None:
        """Stop starting new tasks; running tasks finish (safe from any thread)."""
        with self._cond:
            self._cancel_locked()

    def run(
        self,
        tasks: Iterable[SchedTask],
        *,
        phase: str | None = None,
        on_tick: Callable[[], None] | None = None,
        tick_seconds: float = 0.05,
    ) -> SchedulerReport:
        """Execute ``tasks`` to completion and return the report.

        The calling thread blocks here. ``on_tick`` is invoked in the calling
        thread every ``tick_seconds`` (UI refresh hook). If the calling thread is
        interrupted (``KeyboardInterrupt`` or any exception from ``on_tick``) the
        scheduler cancels, waits for running tasks, partially fills
        :attr:`last_report` and re-raises.
        """
        with self._run_lock:
            self._reset()
            roots = list(tasks)
            seen: set[str] = set()
            for t in roots:
                if not isinstance(t, SchedTask):
                    raise TypeError(f"expected SchedTask, got {type(t).__name__}")
                if t.task_id in seen:
                    raise ValueError(f"duplicate task id: {t.task_id!r}")
                seen.add(t.task_id)

            self._t_start = self._clock()
            if phase:
                safe_emit(self._sink, PhaseChanged(phase))
            if not roots:
                return self._build_report()

            events: list[ProgressEvent] = []
            entries: list[tuple[tuple[Any, ...], _Node, float]] = []
            with self._cond:
                now = self._clock()
                for i, t in enumerate(roots):
                    self._root_order.append(t.task_id)
                    node = self._new_node(t, t.task_id, "task", (i,), None, 0)
                    node.ready_at = now
                    node.state = "queued"
                    entries.append(self._entry(node))
                    events.append(self._queued_event(node))
                self._ready.push_many(entries)
            for ev in events:
                safe_emit(self._sink, ev)

            n_workers = min(self._max_workers, max(1, self.total_cores))
            try:
                for i in range(n_workers):
                    th = threading.Thread(
                        target=self._worker_main, name=f"pba-sched-{i}", daemon=True
                    )
                    th.start()
                    self._threads.append(th)
                while not self._done_event.wait(tick_seconds):
                    if on_tick is not None:
                        on_tick()
            except BaseException:
                self.cancel()
                self._join_workers(self._shutdown_timeout)
                self.last_report = self._build_report()
                raise
            self._join_workers(None)
            report = self._build_report()
            if self._internal_error is not None:
                raise RuntimeError("scheduler internal error") from self._internal_error
            return report

    # -- state ---------------------------------------------------------------

    def _reset(self) -> None:
        self._alloc = CpuSetAllocator(
            self._topology, use_smt=self._use_smt, max_slots=self._max_slots
        )
        if isinstance(self._memory_arg, MemoryBudget):
            self._mem = self._memory_arg
        else:
            self._mem = MemoryBudget(self._memory_arg)
        self._ready: ReadyQueue[_Node] = ReadyQueue(self._max_backfill, self._max_scan)
        self._running = 0
        self._serial = 0
        self._records: list[TaskRecord] = []
        self._unfinished: dict[int, _Node] = {}
        self._results: dict[str, Any] = {}
        self._root_order: list[str] = []
        self._logical: dict[str, tuple[bool, str | None]] = {}
        self._internal_error: BaseException | None = None
        self._oversize_ids: list[str] = []
        self._cancelled = False
        self._cancel_event = threading.Event()
        self._done_event = threading.Event()
        self._threads = []
        self._t_start = self._clock()
        self._t_end = self._t_start

    def _new_node(
        self,
        task: SchedTask,
        node_id: str,
        kind: str,
        order: tuple[int, ...],
        parent: _Node | None,
        idx: int,
    ) -> _Node:
        node = _Node(task, node_id, kind, order, parent, idx, self._serial)
        self._serial += 1
        node.maxt = max(
            1,
            min(
                self.total_cores,
                int(
                    task.max_threads
                    if task.max_threads is not None
                    else task.preferred_threads
                ),
            ),
        )
        node.mint = min(max(1, int(task.min_threads)), node.maxt)
        node.pref = min(max(int(task.preferred_threads), node.mint), node.maxt)
        self._unfinished[node.serial] = node
        return node

    def _entry(self, node: _Node) -> tuple[tuple[Any, ...], _Node, float]:
        task = node.task
        prio = task.priority if task.priority is not None else task.work
        tier = 0 if node.kind == "finalize" else 1
        return ((tier, -float(prio), node.order), node, max(0.0, float(task.work)))

    def _queued_event(self, node: _Node) -> LocusQueued:
        t = node.task
        return LocusQueued(
            node.node_id,
            n_seq=t.n_seq,
            max_len=t.max_len,
            work=float(t.work),
            parent=node.parent.node_id if node.parent is not None else None,
        )

    def _cancel_locked(self) -> None:
        if not self._cancelled:
            self._cancelled = True
            self._cancel_event.set()
        self._cond.notify_all()

    def _finished(self) -> bool:
        return self._running == 0 and (self._cancelled or len(self._ready) == 0)

    def _join_workers(self, timeout: float | None) -> None:
        deadline = None if timeout is None else self._clock() + timeout
        for th in self._threads:
            remaining = None if deadline is None else max(0.0, deadline - self._clock())
            th.join(remaining)
        alive = self.alive_workers()
        if alive:
            log.warning(f"{alive} scheduler worker(s) still running after shutdown")

    # -- worker --------------------------------------------------------------

    def _worker_main(self) -> None:
        while True:
            with self._cond:
                assignment: _Assignment | None = None
                while True:
                    try:
                        assignment = self._try_assign()
                    except Exception as exc:
                        self._fail_internal(exc)
                        return
                    if assignment is not None:
                        break
                    if self._finished():
                        self._done_event.set()
                        self._cond.notify_all()
                        return
                    self._cond.wait(0.5)
            try:
                self._execute(assignment)
            except Exception as exc:  # bookkeeping bug: never hang the run
                with self._cond:
                    self._fail_internal(exc)
                return

    def _fail_internal(self, exc: BaseException) -> None:
        log.error(f"scheduler internal error: {exc!r}")
        self._internal_error = exc
        self._cancel_locked()
        self._done_event.set()

    def _admit(
        self, node: _Node, n_others: int, others_work: float
    ) -> _Admission | None:
        task = node.task
        idle = self._alloc.free_count
        threads = lease_width(
            work=float(task.work),
            preferred=node.pref,
            min_threads=node.mint,
            max_threads=node.maxt,
            idle=idle,
            n_others=n_others,
            others_work=others_work,
            running=self._running,
            promote=self._promote,
        )
        if threads is None:
            return None
        est = int(task.est_mem_bytes)
        oversize = self._mem.is_oversize(est)
        if not self._mem.try_acquire(est):
            return None
        return _Admission(threads, oversize)

    def _force_admit(self, node: _Node, _n: int, _w: float) -> _Admission:
        self._mem.force_acquire(int(node.task.est_mem_bytes))
        return _Admission(max(1, node.mint), False)

    def _try_assign(self) -> _Assignment | None:
        """Pick the next startable task (cond held)."""
        if self._cancelled or len(self._ready) == 0 or self._alloc.free_count <= 0:
            return None
        picked = self._ready.pop_next(self._admit)
        if picked is None and self._running == 0 and len(self._ready) > 0:
            log.error("scheduler found no admissible task while idle; forcing head")
            picked = self._ready.pop_next(self._force_admit)
        if picked is None:
            return None
        node, adm = picked
        lease = self._alloc.try_allocate(adm.threads)
        if lease is None:  # cannot happen (idle >= threads); stay safe
            self._mem.release(int(node.task.est_mem_bytes))
            self._ready.push(*self._entry(node))
            return None
        now = self._clock()
        node.mem_held = int(node.task.est_mem_bytes)
        node.oversize = adm.oversize
        node.state = "running"
        node.threads_peak = max(node.threads_peak, lease.threads)
        first = node.first_start is None
        if first:
            node.first_start = now
        root = node.root
        if root.first_start is None:
            root.first_start = now
        self._running += 1
        events: list[ProgressEvent] = []
        if adm.oversize:
            msg = (
                f"task `{node.node_id}` needs ~{node.mem_held / 2**20:.0f} MB, more than "
                "the whole memory envelope; running it alone"
            )
            log.warning(msg)
            events.append(WarningEvent(msg, locus=node.node_id, category="memory"))
            self._oversize_ids.append(node.node_id)
        t = node.task
        events.append(
            LocusStarted(
                node.node_id,
                n_seq=t.n_seq,
                max_len=t.max_len,
                threads=lease.threads,
                cpu_ids=lease.cpu_ids,
                work=float(t.work),
                parent=node.parent.node_id if node.parent is not None else None,
            )
        )
        return _Assignment(node, lease, now, max(0.0, now - node.ready_at), events)

    def _execute(self, a: _Assignment) -> None:
        for ev in a.events:
            safe_emit(self._sink, ev)
        node = a.node
        task = node.task
        ticket: ScratchTicketLike | None = None
        result: Any = None
        error: str | None = None
        ok = True
        try:
            if self._scratch is not None and task.scratch_bytes > 0:
                ticket = self._scratch.reserve(int(task.scratch_bytes))
            ctx = TaskContext(
                task_id=node.node_id,
                threads=a.lease.threads,
                cpu_ids=a.lease.cpu_ids,
                lease=a.lease,
                scratch=ticket,
                parent_id=node.parent.node_id if node.parent is not None else None,
                kind=node.kind,
                pinning=self._pin,
                cancel_event=self._cancel_event,
            )
            result = task.fn(ctx)
        except BaseException as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            log.error(f"task `{node.node_id}` failed: {error}")
        tier: str | None = None
        if ticket is not None:
            tier = getattr(ticket, "tier", None)
            try:
                ticket.release()
            except Exception as exc:
                log.debug(f"scratch release failed for `{node.node_id}`: {exc!r}")
        end = self._clock()
        events: list[ProgressEvent] = []
        with self._cond:
            self._finish_unit(a, ok, result, error, end, tier, events)
            self._cond.notify_all()
        for ev in events:
            safe_emit(self._sink, ev)

    # -- completion logic (cond held) ------------------------------------------

    def _finish_unit(
        self,
        a: _Assignment,
        ok: bool,
        result: Any,
        error: str | None,
        end: float,
        tier: str | None,
        events: list[ProgressEvent],
    ) -> None:
        node = a.node
        self._running -= 1
        self._alloc.release(a.lease)
        if node.mem_held:
            self._mem.release(node.mem_held)
            node.mem_held = 0
        self._records.append(
            TaskRecord(
                task_id=node.node_id,
                parent_id=node.parent.node_id if node.parent is not None else None,
                root_id=node.root.node_id,
                kind=node.kind,
                order=node.order,
                work=float(node.task.work),
                est_mem_bytes=int(node.task.est_mem_bytes),
                threads=a.lease.threads,
                cpu_ids=a.lease.cpu_ids,
                ready=node.ready_at,
                start=a.start,
                end=end,
                ok=ok,
                status="ok" if ok else "failed",
                error=error,
                scratch_tier=tier,
                oversize=node.oversize,
            )
        )
        if (
            node.kind == "finalize"
            and node.parent is not None
            and node.parent.state == "done"
        ):
            return  # owner already failed/cancelled
        if not ok:
            self._complete(node, False, None, error, end, events)
            if self._fail_fast:
                self._cancel_locked()
            return
        if node.kind != "finalize" and (
            isinstance(result, FanOut) or _is_subtask_list(result)
        ):
            self._start_fanout(node, result, end, events)
            return
        self._complete(node, True, result, None, end, events)

    def _start_fanout(
        self, node: _Node, result: Any, now: float, events: list[ProgressEvent]
    ) -> None:
        fan = result if isinstance(result, FanOut) else FanOut(subtasks=list(result))
        subs = list(fan.subtasks)
        ids = [s.task_id for s in subs]
        if not all(isinstance(s, SchedTask) for s in subs) or len(set(ids)) != len(ids):
            self._complete(
                node,
                False,
                None,
                "invalid fan-out: subtasks must be unique SchedTask objects",
                now,
                events,
            )
            return
        node.state = "waiting"
        node.finalizer = (
            fan.finalizer if fan.finalizer is not None else node.task.finalizer
        )
        node.finalize_work = float(fan.finalize_work)
        node.finalize_mem = int(fan.finalize_est_mem_bytes)
        node.children = []
        node.child_results = [None] * len(subs)
        node.pending_children = len(subs)
        entries: list[tuple[tuple[Any, ...], _Node, float]] = []
        for i, sub in enumerate(subs):
            child = self._new_node(
                sub,
                f"{node.node_id}/{sub.task_id}",
                "subtask",
                (*node.order, 1, i),
                node,
                i,
            )
            child.ready_at = now
            child.state = "queued"
            node.children.append(child)
            entries.append(self._entry(child))
            events.append(self._queued_event(child))
        for entry in entries:
            self._ready.push(*entry)
        if not subs:
            self._children_done(node, now, events)

    def _children_done(
        self, node: _Node, now: float, events: list[ProgressEvent]
    ) -> None:
        results = list(node.child_results)
        if node.finalizer is None:
            self._complete(node, True, results, None, now, events)
            return
        finalizer = node.finalizer
        owner_ctx_id = node.node_id

        def run_finalizer(ctx: TaskContext) -> Any:
            return finalizer(ctx, results)

        task = SchedTask(
            task_id="finalize",
            fn=run_finalizer,
            work=node.finalize_work,
            est_mem_bytes=node.finalize_mem,
        )
        fin = self._new_node(
            task, f"{owner_ctx_id}::finalize", "finalize", (*node.order, 2, 0), node, 0
        )
        fin.ready_at = now
        fin.state = "queued"
        node.finalize_node = fin
        self._ready.push(*self._entry(fin))
        events.append(self._queued_event(fin))

    def _complete(
        self,
        node: _Node,
        ok: bool,
        result: Any,
        error: str | None,
        now: float,
        events: list[ProgressEvent],
    ) -> None:
        """Logical completion of ``node`` (after fan-in for fan-out nodes)."""
        if node.state == "done":
            return
        node.state = "done"
        self._unfinished.pop(node.serial, None)
        start = node.first_start if node.first_start is not None else now
        if node.kind == "finalize":
            owner = node.parent
            assert owner is not None
            events.append(
                LocusFinished(
                    node.node_id,
                    ok,
                    seconds=max(0.0, now - start),
                    threads=node.threads_peak,
                    error=error,
                    parent=owner.node_id,
                )
            )
            owner.threads_peak = max(owner.threads_peak, node.threads_peak)
            self._complete(
                owner,
                ok,
                result,
                None if ok else f"finalizer failed: {error}",
                now,
                events,
            )
            return
        parent = node.parent
        events.append(
            LocusFinished(
                node.node_id,
                ok,
                seconds=max(0.0, now - start),
                threads=node.threads_peak,
                error=error,
                parent=parent.node_id if parent is not None else None,
            )
        )
        if parent is None:
            self._logical[node.node_id] = (ok, error)
            if ok and self._keep_results:
                self._results[node.node_id] = result
            return
        if parent.state == "done":
            return
        parent.threads_peak = max(parent.threads_peak, node.threads_peak)
        parent.child_results[node.idx] = result if ok else None
        parent.pending_children -= 1
        if not ok:
            self._abort_subtree(parent, now, events)
            self._complete(
                parent,
                False,
                None,
                f"subtask `{node.node_id}` failed: {error}",
                now,
                events,
            )
        elif parent.pending_children <= 0:
            self._children_done(parent, now, events)

    def _abort_subtree(
        self, owner: _Node, now: float, events: list[ProgressEvent]
    ) -> None:
        """Cancel every queued/waiting descendant of ``owner`` (running ones finish)."""
        doomed: set[int] = set()

        def collect(n: _Node) -> None:
            for c in list(n.children) + ([n.finalize_node] if n.finalize_node else []):
                if c.state == "queued":
                    doomed.add(c.serial)
                elif c.state == "waiting":
                    collect(c)
                    self._cancel_node(c, now, events, record=False)
                elif c.state == "running":
                    collect(c)

        collect(owner)
        if doomed:
            for gone in self._ready.remove_where(lambda x: x.serial in doomed):
                self._cancel_node(gone, now, events, record=True)

    def _cancel_node(
        self, node: _Node, now: float, events: list[ProgressEvent], record: bool
    ) -> None:
        if node.state == "done":
            return
        node.state = "done"
        self._unfinished.pop(node.serial, None)
        if record:
            self._records.append(self._cancelled_record(node, now))
        events.append(
            LocusFinished(
                node.node_id,
                False,
                seconds=0.0,
                threads=0,
                error="cancelled",
                parent=node.parent.node_id if node.parent is not None else None,
            )
        )

    def _cancelled_record(self, node: _Node, now: float) -> TaskRecord:
        return TaskRecord(
            task_id=node.node_id,
            parent_id=node.parent.node_id if node.parent is not None else None,
            root_id=node.root.node_id,
            kind=node.kind,
            order=node.order,
            work=float(node.task.work),
            est_mem_bytes=int(node.task.est_mem_bytes),
            threads=0,
            cpu_ids=(),
            ready=node.ready_at,
            start=now,
            end=now,
            ok=False,
            status="cancelled",
            error="cancelled",
        )

    # -- report ---------------------------------------------------------------

    def _build_report(self) -> SchedulerReport:
        now = self._clock()
        events: list[ProgressEvent] = []
        with self._cond:
            leftovers = sorted(self._unfinished.values(), key=lambda n: n.order)
            for node in leftovers:
                if node.state == "running":
                    continue
                # fan-out owners ("waiting") already have a record for their own call
                self._cancel_node(
                    node, now, events, record=node.state in ("new", "queued")
                )
            self._t_end = now
            records = sorted(self._records, key=lambda r: r.order)
            loci = {
                rid: self._logical.get(rid, (False, "cancelled"))
                for rid in self._root_order
            }
            report = SchedulerReport(
                records=records,
                total_cores=self.total_cores,
                started_at=self._t_start,
                ended_at=self._t_end,
                cancelled=self._cancelled,
                loci=loci,
                results=dict(self._results),
                peak_memory_bytes=self._mem.peak,
                oversize_task_ids=tuple(self._oversize_ids),
            )
        for ev in events:
            safe_emit(self._sink, ev)
        self.last_report = report
        return report
