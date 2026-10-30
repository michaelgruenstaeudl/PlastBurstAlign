"""Discrete-event simulator comparing locus scheduling policies.

Three policies are simulated on the same workload and core count:

``static_fifo``
    Naive baseline: every task leases a fixed number of threads (default 1) and
    tasks start in *input order*.

``lpt_waterfill``
    The pre-v3 policy: tasks sorted by decreasing size (LPT), per-task core
    preferences from :meth:`CoreBudgetPool.solve_water_filling_schedule`, and the
    exact lease decision of :meth:`CoreBudgetPool.lease_cores`
    (:func:`~plastburstalign.task_queue.compute_lease`). A worker that cannot get
    its lease holds its task while waiting, like the real blocking pool.

``locus_scheduler``
    The new :class:`~plastburstalign.scheduler.LocusScheduler` policy: a single
    LPT :class:`~plastburstalign.scheduler.ReadyQueue`, straggler promotion via
    :func:`~plastburstalign.scheduler.lease_width`, and chunk fan-out (a task with
    ``chunks`` runs a short planning unit, then its chunks as independent tasks in
    the same queue, then a finalizer).

The simulator reuses the production decision functions (``compute_lease``,
``lease_width``, ``ReadyQueue``, ``compute_schedule_metrics``), so a change to the
shipped policy changes the simulation too. Task run time follows an Amdahl model:
``duration(p) = work * (s + (1 - s) / min(p, max_threads))``.
"""

from __future__ import annotations

import heapq
import random
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from .scheduler import (
    ReadyQueue,
    ScheduleMetrics,
    compute_schedule_metrics,
    lease_width,
)
from .task_queue import CoreBudgetPool, compute_lease

__all__ = [
    "Policy",
    "SimRecord",
    "SimResult",
    "SimTask",
    "long_tail_workload",
    "simulate",
]


class Policy(str, Enum):
    """Scheduling policies that can be simulated."""

    STATIC_FIFO = "static_fifo"
    LPT_WATERFILL = "lpt_waterfill"
    LOCUS_SCHEDULER = "locus_scheduler"


@dataclass(frozen=True)
class SimTask:
    """A simulated locus.

    Attributes:
        task_id: Unique id.
        work: Seconds of work on one thread when run monolithically.
        serial_fraction: Amdahl serial fraction ``s`` (speedup saturates at ``1/s``).
        max_threads: Threads beyond this add nothing.
        n_seq, max_len: Sizes fed to the water-filling core estimator.
        chunks: Optional sub-locus tasks (used only by ``locus_scheduler``).
        plan_work: Single-thread seconds spent deciding the split.
        finalize_work: Single-thread seconds spent splicing chunk results.
    """

    task_id: str
    work: float
    serial_fraction: float = 0.15
    max_threads: int = 8
    n_seq: int = 0
    max_len: int = 0
    chunks: tuple[SimTask, ...] = ()
    plan_work: float = 0.0
    finalize_work: float = 0.0

    def duration(self, threads: int) -> float:
        """Run time with ``threads`` threads under the Amdahl model."""
        p = max(1, min(int(threads), max(1, self.max_threads)))
        s = min(max(self.serial_fraction, 0.0), 1.0)
        return self.work * (s + (1.0 - s) / p)


@dataclass(frozen=True)
class SimRecord:
    """One simulated execution interval."""

    task_id: str
    parent_id: str | None
    kind: str
    start: float
    end: float
    threads: int


@dataclass(frozen=True)
class SimResult:
    """Outcome of one simulation."""

    policy: Policy
    total_cores: int
    records: tuple[SimRecord, ...]
    metrics: ScheduleMetrics
    critical_task_id: str | None
    completion: dict[str, float] = field(default_factory=dict)

    @property
    def makespan(self) -> float:
        """Simulated wall time."""
        return self.metrics.makespan

    def format_row(self) -> str:
        """One aligned text row for logs and reports."""
        m = self.metrics
        return (
            f"{self.policy.value:<16} cores={self.total_cores:<3} "
            f"makespan={m.makespan:8.1f}s  cpu={m.total_cpu_seconds:8.1f}s  "
            f"eff={100 * m.parallel_efficiency:5.1f}%  "
            f"tail={100 * m.straggler_tax:5.1f}%  critical={self.critical_task_id}"
        )


def long_tail_workload(
    n_short: int = 40,
    n_long: int = 2,
    *,
    seed: int = 7,
    short_work: tuple[float, float] = (5.0, 20.0),
    long_work: float = 900.0,
    n_chunks: int = 16,
    chunk_efficiency: float = 0.7,
    plan_work: float = 2.0,
    finalize_work: float = 5.0,
) -> list[SimTask]:
    """Synthetic long-tail workload: many short loci plus a few huge ones.

    The long loci sit late in the input order (as an arbitrary alphabetical order
    would place them). ``chunk_efficiency`` is the total work of all chunks relative
    to the monolithic locus (< 1 because alignment cost is super-linear in length,
    so splitting is cheaper in total); a conservative 0.7 is the default.
    """
    rng = random.Random(seed)
    tasks: list[SimTask] = []
    for i in range(n_short):
        tasks.append(
            SimTask(
                task_id=f"short{i:03d}",
                work=rng.uniform(*short_work),
                serial_fraction=0.3,
                max_threads=2,
                n_seq=rng.randint(80, 150),
                max_len=rng.randint(500, 1100),
            )
        )
    chunk_work = long_work * chunk_efficiency / max(1, n_chunks)
    total = n_short + n_long
    positions = [
        min(total - 1, int(total * (0.55 + 0.3 * k / max(1, n_long))))
        for k in range(n_long)
    ]
    for k, pos in enumerate(positions):
        chunks = tuple(
            SimTask(
                task_id=f"chunk{j:02d}",
                work=chunk_work,
                serial_fraction=0.1,
                max_threads=2,
            )
            for j in range(n_chunks)
        )
        tasks.insert(
            pos,
            SimTask(
                task_id=f"long{k}",
                work=long_work,
                serial_fraction=0.15,
                max_threads=8,
                n_seq=600,
                max_len=3500,
                chunks=chunks,
                plan_work=plan_work,
                finalize_work=finalize_work,
            ),
        )
    return tasks


def simulate(
    tasks: Sequence[SimTask],
    total_cores: int,
    policy: Policy | str,
    *,
    static_threads: int = 1,
    promote: bool = True,
    use_chunks: bool = True,
    max_backfill: int = 16,
) -> SimResult:
    """Simulate ``tasks`` on ``total_cores`` cores under ``policy``.

    Args:
        tasks: Workload (input order matters for ``static_fifo``).
        total_cores: Core budget.
        policy: A :class:`Policy` (or its string value).
        static_threads: Fixed lease width of ``static_fifo``.
        promote: ``locus_scheduler`` only: enable straggler promotion.
        use_chunks: ``locus_scheduler`` only: honour ``SimTask.chunks`` fan-out.
        max_backfill: ``locus_scheduler`` only: backfill bound of the ready queue.
    """
    cores = max(1, int(total_cores))
    pol = Policy(policy)
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")
    completion: dict[str, float]
    if pol is Policy.STATIC_FIFO:
        records, completion = _sim_static(tasks, cores, static_threads)
    elif pol is Policy.LPT_WATERFILL:
        records, completion = _sim_waterfill(tasks, cores)
    else:
        records, completion = _sim_locus(
            tasks, cores, promote, use_chunks, max_backfill
        )
    end = max((r.end for r in records), default=0.0)
    metrics = compute_schedule_metrics(
        [(r.start, r.end, r.threads) for r in records], cores, 0.0, end
    )
    critical = max(completion, key=lambda k: (completion[k], k)) if completion else None
    return SimResult(
        policy=pol,
        total_cores=cores,
        records=tuple(records),
        metrics=metrics,
        critical_task_id=critical,
        completion=completion,
    )


# ---------------------------------------------------------------------------
# (a) static lease, FIFO
# ---------------------------------------------------------------------------


def _sim_static(
    tasks: Sequence[SimTask], cores: int, threads: int
) -> tuple[list[SimRecord], dict[str, float]]:
    width = max(1, min(int(threads), cores))
    slots = max(1, cores // width)
    queue = deque(tasks)
    running: list[tuple[float, int, str]] = []
    records: list[SimRecord] = []
    completion: dict[str, float] = {}
    t = 0.0
    seq = 0
    while queue or running:
        while queue and len(running) < slots:
            task = queue.popleft()
            dur = task.duration(width)
            heapq.heappush(running, (t + dur, seq, task.task_id))
            seq += 1
            records.append(SimRecord(task.task_id, None, "task", t, t + dur, width))
        end, _, tid = heapq.heappop(running)
        t = end
        completion[tid] = end
    return records, completion


# ---------------------------------------------------------------------------
# (b) LPT + CoreBudgetPool water filling
# ---------------------------------------------------------------------------


def _sim_waterfill(
    tasks: Sequence[SimTask], cores: int
) -> tuple[list[SimRecord], dict[str, float]]:
    meta = {t.task_id: (max(1, t.n_seq), max(1, t.max_len)) for t in tasks}
    schedule = CoreBudgetPool.solve_water_filling_schedule(meta, cores)
    pending = deque(sorted(tasks, key=lambda x: (-x.work, x.task_id)))
    blocked: list[SimTask] = []
    free_workers = cores
    avail = cores
    remaining = len(tasks)
    running: list[tuple[float, int, SimTask, int]] = []
    records: list[SimRecord] = []
    completion: dict[str, float] = {}
    t = 0.0
    seq = 0

    def lease(task: SimTask) -> int | None:
        n = compute_lease(schedule.get(task.task_id, 1), remaining, avail, cores, cores)
        return n if n is not None and n <= avail else None

    def begin(task: SimTask, n: int) -> None:
        nonlocal avail, seq
        avail -= n
        dur = task.duration(n)
        heapq.heappush(running, (t + dur, seq, task, n))
        seq += 1
        records.append(SimRecord(task.task_id, None, "task", t, t + dur, n))

    while True:
        progressed = True
        while progressed:
            progressed = False
            for task in list(blocked):
                n = lease(task)
                if n is not None:
                    blocked.remove(task)
                    begin(task, n)
                    progressed = True
            while free_workers > 0 and pending:
                task = pending.popleft()
                free_workers -= 1
                n = lease(task)
                if n is None:
                    blocked.append(task)
                else:
                    begin(task, n)
                progressed = True
        if not running:
            break
        end, _, task, n = heapq.heappop(running)
        t = end
        avail += n
        free_workers += 1
        remaining -= 1
        completion[task.task_id] = end
    if blocked or pending:
        raise RuntimeError("simulation deadlock in lpt_waterfill")
    return records, completion


# ---------------------------------------------------------------------------
# (c) LocusScheduler policy
# ---------------------------------------------------------------------------


class _SimNode:
    __slots__ = (
        "node_id",
        "task",
        "origin",
        "kind",
        "parent",
        "root_id",
        "weight",
        "pref",
        "maxt",
        "pending",
    )

    def __init__(
        self,
        node_id: str,
        task: SimTask,
        origin: SimTask,
        kind: str,
        parent: _SimNode | None,
        root_id: str,
        weight: float,
        pref: int,
        maxt: int,
    ) -> None:
        self.node_id = node_id
        self.task = task
        self.origin = origin
        self.kind = kind
        self.parent = parent
        self.root_id = root_id
        self.weight = weight
        self.pref = pref
        self.maxt = maxt
        self.pending = 0


def _sim_locus(
    tasks: Sequence[SimTask],
    cores: int,
    promote: bool,
    use_chunks: bool,
    max_backfill: int,
) -> tuple[list[SimRecord], dict[str, float]]:
    meta = {t.task_id: (max(1, t.n_seq), max(1, t.max_len)) for t in tasks}
    schedule = CoreBudgetPool.solve_water_filling_schedule(meta, cores)
    rq: ReadyQueue[_SimNode] = ReadyQueue(max_backfill=max_backfill)
    seq = 0

    def push(node: _SimNode, order: tuple[int, ...], tier: int = 1) -> None:
        rq.push((tier, -node.weight, order), node, node.weight)

    for i, task in enumerate(tasks):
        if use_chunks and task.chunks:
            planner = SimTask(
                task.task_id, task.plan_work, serial_fraction=1.0, max_threads=1
            )
            node = _SimNode(
                task.task_id, planner, task, "task", None, task.task_id, task.work, 1, 1
            )
        else:
            maxt = max(1, min(cores, task.max_threads))
            pref = min(schedule.get(task.task_id, 1), maxt)
            node = _SimNode(
                task.task_id,
                task,
                task,
                "task",
                None,
                task.task_id,
                task.work,
                pref,
                maxt,
            )
        push(node, (i,))

    running: list[tuple[float, int, _SimNode, int]] = []
    records: list[SimRecord] = []
    completion: dict[str, float] = {}
    avail = cores
    t = 0.0

    def start_ready() -> None:
        nonlocal avail, seq
        while avail > 0:

            def admit(node: _SimNode, n_others: int, others_w: float) -> int | None:
                return lease_width(
                    work=node.weight,
                    preferred=node.pref,
                    min_threads=1,
                    max_threads=node.maxt,
                    idle=avail,
                    n_others=n_others,
                    others_work=others_w,
                    running=len(running),
                    promote=promote,
                )

            got = rq.pop_next(admit)
            if got is None:
                return
            node, n = got
            avail -= n
            dur = node.task.duration(n)
            heapq.heappush(running, (t + dur, seq, node, n))
            seq += 1
            parent_id = node.parent.node_id if node.parent is not None else None
            records.append(SimRecord(node.node_id, parent_id, node.kind, t, t + dur, n))

    def done(node: _SimNode, order_hint: int) -> None:
        origin = node.origin
        if node.kind == "task" and use_chunks and origin.chunks:
            node.pending = len(origin.chunks)
            for j, chunk in enumerate(origin.chunks):
                child = _SimNode(
                    f"{node.node_id}/{chunk.task_id}",
                    chunk,
                    chunk,
                    "subtask",
                    node,
                    node.root_id,
                    chunk.work,
                    1,
                    max(1, min(cores, chunk.max_threads)),
                )
                push(child, (order_hint, 1, j))
            return
        if node.kind == "subtask":
            parent = node.parent
            assert parent is not None
            parent.pending -= 1
            if parent.pending == 0:
                fin_task = SimTask(
                    f"{parent.node_id}::finalize",
                    parent.origin.finalize_work,
                    serial_fraction=1.0,
                    max_threads=1,
                )
                fin = _SimNode(
                    fin_task.task_id,
                    fin_task,
                    parent.origin,
                    "finalize",
                    parent,
                    parent.root_id,
                    0.0,
                    1,
                    1,
                )
                push(fin, (order_hint, 2, 0), tier=0)
            return
        completion[node.root_id] = t

    order_of = {task.task_id: i for i, task in enumerate(tasks)}
    while True:
        start_ready()
        if not running:
            break
        end, _, node, n = heapq.heappop(running)
        t = end
        avail += n
        done(node, order_of[node.root_id])
    if len(rq):
        raise RuntimeError("simulation deadlock in locus_scheduler")
    return records, completion
