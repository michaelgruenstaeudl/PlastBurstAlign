"""Concurrency utilities, task queues, and dynamic core budget management.

Besides the legacy integer-based :class:`CoreBudgetPool`, this module provides
CPU-set leases: :class:`CoreLease` (which concrete cores/CPUs a task owns),
:class:`CpuSetAllocator` (topology-aware allocation: physical cores first,
best-fit inside one NUMA node, SMT siblings only when SMT use is enabled) and
:func:`pin_command` (optional ``taskset -c`` pinning of aligner subprocesses).
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import sys
import threading
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.context import BaseContext

from .topology import CpuTopology


def get_mp_context(force_context: str | None = None) -> BaseContext:
    """Returns an appropriate multiprocessing context based on platform and configuration."""
    if force_context:
        return multiprocessing.get_context(force_context)

    env_override = os.environ.get("PBA_MP_CONTEXT")
    if env_override:
        return multiprocessing.get_context(env_override)

    if sys.platform in ("win32", "darwin"):
        return multiprocessing.get_context("spawn")

    return multiprocessing.get_context("fork")


# ---------------------------------------------------------------------------
# CPU-set leases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreLease:
    """A concrete set of CPUs owned by one task.

    Attributes:
        core_ids: Dense physical core ids (``CoreInfo.core_id``) touched by the lease.
        cpu_ids: Sorted OS logical CPU ids the task may use (what ``taskset -c`` gets).
        numa_node: NUMA node holding most of the lease (lowest id on ties).
        threads: Number of worker threads the task should use. Equals
            ``len(cpu_ids)``: one logical CPU per thread.
        numa_nodes: All NUMA nodes touched (more than one only if the lease had to
            spill across nodes).
    """

    core_ids: tuple[int, ...]
    cpu_ids: tuple[int, ...]
    numa_node: int
    threads: int
    numa_nodes: tuple[int, ...] = ()

    @property
    def spans_numa(self) -> bool:
        """True if the lease had to spill across more than one NUMA node."""
        return len(self.numa_nodes) > 1


@dataclass(frozen=True)
class _Slot:
    """One allocatable logical CPU of the allocator."""

    cpu_id: int
    core_id: int
    node: int
    rank: int  # position of the CPU inside its physical core (0 = primary thread)


class CpuSetAllocator:
    """Thread-safe, topology-aware allocator of :class:`CoreLease` objects.

    Rules:

    * The allocation unit is the primary logical CPU of each physical core. SMT
      siblings are only allocatable when ``use_smt`` is true, so by default two
      leases never share a physical core.
    * Physical cores first: slots on fully idle cores are preferred (and, with
      SMT enabled, siblings of an idle core before siblings of a busy core).
    * Best fit inside one NUMA node: the node with the *fewest* free slots that
      still fits the request is chosen. Only when no single node fits does the
      lease spill across nodes (largest free node first, fewest nodes).
    * Allocation is all-or-nothing and never blocks; callers wait on their own
      condition and retry after a release.
    """

    def __init__(
        self,
        topology: CpuTopology,
        use_smt: bool = False,
        max_slots: int | None = None,
    ) -> None:
        self.topology = topology
        self.use_smt = bool(use_smt)
        slots: list[_Slot] = []
        for core in topology.cores:
            cpus = core.cpu_ids if core.cpu_ids else (core.core_id,)
            usable = cpus if self.use_smt else cpus[:1]
            for rank, cpu in enumerate(usable):
                slots.append(_Slot(cpu, core.core_id, core.numa_node, rank))
        if max_slots is not None and 0 < max_slots < len(slots):
            slots = sorted(slots, key=lambda s: (s.rank, s.node, s.core_id))[:max_slots]
        slots.sort(key=lambda s: (s.node, s.core_id, s.rank))
        if not slots:
            slots = [_Slot(0, 0, 0, 0)]
        self._slots = slots
        self._by_cpu = {s.cpu_id: i for i, s in enumerate(slots)}
        self._free: set[int] = set(range(len(slots)))
        self._busy_on_core: Counter[int] = Counter()
        self._lock = threading.Lock()

    @property
    def total_slots(self) -> int:
        """Number of allocatable logical CPUs (cores, or hardware threads with SMT)."""
        return len(self._slots)

    @property
    def free_count(self) -> int:
        """Number of currently unleased slots."""
        with self._lock:
            return len(self._free)

    def free_cpu_ids(self) -> tuple[int, ...]:
        """Sorted OS CPU ids that are currently free."""
        with self._lock:
            return tuple(sorted(self._slots[i].cpu_id for i in self._free))

    def try_allocate(self, threads: int) -> CoreLease | None:
        """Lease ``threads`` CPUs, or return ``None`` if not enough are free."""
        n = max(1, int(threads))
        with self._lock:
            if n > len(self._free):
                return None
            chosen = self._choose(n)
            for i in chosen:
                self._free.discard(i)
                self._busy_on_core[self._slots[i].core_id] += 1
            return self._make_lease(chosen)

    def release(self, lease: CoreLease) -> None:
        """Return a lease's CPUs to the free set (unknown/duplicate CPUs ignored)."""
        with self._lock:
            for cpu in lease.cpu_ids:
                i = self._by_cpu.get(cpu)
                if i is None or i in self._free:
                    continue
                self._free.add(i)
                core = self._slots[i].core_id
                self._busy_on_core[core] -= 1
                if self._busy_on_core[core] <= 0:
                    del self._busy_on_core[core]

    # -- internals (call with the lock held) -------------------------------

    def _pick_order(self, idx: int) -> tuple[bool, int, int]:
        slot = self._slots[idx]
        return (self._busy_on_core[slot.core_id] > 0, slot.rank, slot.core_id)

    def _choose(self, n: int) -> list[int]:
        by_node: dict[int, list[int]] = {}
        for i in sorted(self._free):
            by_node.setdefault(self._slots[i].node, []).append(i)
        for members in by_node.values():
            members.sort(key=self._pick_order)

        def idle_count(members: list[int]) -> int:
            return sum(
                1 for i in members if self._busy_on_core[self._slots[i].core_id] == 0
            )

        # 1) a single node whose fully-idle physical cores can hold the lease
        fits = [
            (idle_count(m), node) for node, m in by_node.items() if idle_count(m) >= n
        ]
        if fits:
            return by_node[min(fits)[1]][:n]
        # 2) a single node with enough free slots (may share cores with SMT)
        fits = [(len(m), node) for node, m in by_node.items() if len(m) >= n]
        if fits:
            return by_node[min(fits)[1]][:n]
        # 3) spill: fewest nodes, largest free node first
        chosen: list[int] = []
        need = n
        for _node, members in sorted(
            by_node.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            take = members[:need]
            chosen.extend(take)
            need -= len(take)
            if need <= 0:
                break
        return chosen

    def _make_lease(self, chosen: list[int]) -> CoreLease:
        slots = [self._slots[i] for i in chosen]
        nodes = Counter(s.node for s in slots)
        primary = min(nodes, key=lambda nd: (-nodes[nd], nd))
        return CoreLease(
            core_ids=tuple(sorted({s.core_id for s in slots})),
            cpu_ids=tuple(sorted(s.cpu_id for s in slots)),
            numa_node=primary,
            threads=len(slots),
            numa_nodes=tuple(sorted(nodes)),
        )


# ---------------------------------------------------------------------------
# Optional CPU pinning
# ---------------------------------------------------------------------------


def format_cpu_list(cpu_ids: Iterable[int]) -> str:
    """Format CPU ids as a ``taskset -c`` list, e.g. ``[0,1,2,3,8]`` -> ``"0-3,8"``."""
    ids = sorted({int(c) for c in cpu_ids})
    parts: list[str] = []
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        parts.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(parts)


def pin_command(
    cmd: Sequence[str],
    cpu_ids: Iterable[int] | None,
    *,
    platform: str | None = None,
    which: Callable[[str], str | None] | None = None,
) -> list[str]:
    """Prefix ``cmd`` with ``taskset -c <cpus>`` to pin a subprocess to ``cpu_ids``.

    The command is returned unchanged (as a new list) when there is nothing to
    pin to, when the platform is not Linux, when ``taskset`` is not installed,
    or when ``cmd`` is already a ``taskset`` invocation. ``platform`` and
    ``which`` are injectable for tests.
    """
    command = list(cmd)
    cpus = format_cpu_list(cpu_ids) if cpu_ids else ""
    if not command or not cpus:
        return command
    plat = platform if platform is not None else sys.platform
    if not plat.startswith("linux"):
        return command
    finder = which if which is not None else shutil.which
    taskset = finder("taskset")
    if not taskset or os.path.basename(command[0]) == "taskset":
        return command
    return [taskset, "-c", cpus, *command]


def compute_lease(
    preferred: int,
    remaining_tasks: int,
    available: int,
    total_cores: int,
    max_cores_per_task: int,
) -> int | None:
    """Pure decision logic of :meth:`CoreBudgetPool.lease_cores`.

    Returns the number of cores to take now, or ``None`` if the task has to wait.
    Shared with the discrete-event simulator so the policy simulated is the policy
    shipped.
    """
    # Solo straggler optimization: lone remaining locus claims all available capacity
    if remaining_tasks <= 1:
        return max(1, min(available, max_cores_per_task))

    # A preference larger than the whole pool could never be satisfied (the
    # "half the quota" escape hatch below would not fire on small pools).
    preferred = max(1, min(preferred, total_cores))

    # Dynamic fair share expands as remaining tasks drop below core count
    fair_share = max(1, total_cores // max(1, remaining_tasks))
    target = max(fair_share, preferred)

    # Standard tasks wait only when saturated (0 cores available). Multi-core
    # scheduled tasks wait until their quota (or at least half) is free.
    min_needed = target if (preferred > 1 and remaining_tasks >= total_cores) else 1
    if available < min_needed and not (
        preferred > 1 and available >= max(2, target // 2)
    ):
        return None
    return max(1, min(available, target, max_cores_per_task))


class CoreBudgetPool:
    """Thread-safe dynamic core budget manager with mathematical water-filling makespan balancing.

    Solves the Malleable Task Scheduling Problem (P | malleable | C_max) with non-linear
    Amdahl speedup S(p, L). Loci that would otherwise become critical-path tail stragglers
    are assigned multi-core leases to finish concurrently with the single-core pool,
    targeting near-100% core utilization across arbitrary hardware deployments.
    """

    def __init__(
        self,
        total_cores: int,
        total_tasks: int = 1,
        max_cores_per_task: int | None = None,
        schedule: dict[str, int] | None = None,
        topology: CpuTopology | None = None,
        use_smt: bool = False,
    ) -> None:
        """Create a pool.

        Args:
            total_cores: Core budget (integer API; unchanged semantics).
            total_tasks: Number of tasks that will eventually call ``task_completed``.
            max_cores_per_task: Upper bound for one lease (default ``total_cores``).
            schedule: Optional per-task preferred core counts (water-filling result).
            topology: Optional CPU topology enabling concrete CPU-set leases via
                :meth:`lease_cpuset`. ``total_cores`` is clamped to the number of
                allocatable slots. Without it a flat topology is built on demand.
            use_smt: Allow SMT siblings as separate lease units (default: physical
                cores only, never splitting a core between leases).
        """
        self.total_cores = max(1, int(total_cores))
        self._allocator: CpuSetAllocator | None = None
        if topology is not None:
            self._allocator = CpuSetAllocator(
                topology, use_smt=use_smt, max_slots=self.total_cores
            )
            self.total_cores = min(self.total_cores, self._allocator.total_slots)
        self._use_smt = use_smt
        self._remaining_tasks = max(1, int(total_tasks))
        self.max_cores_per_task = (
            max(1, int(max_cores_per_task))
            if max_cores_per_task is not None
            else self.total_cores
        )
        self._schedule = dict(schedule) if schedule else {}
        self._available_cores = self.total_cores
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    @classmethod
    def from_topology(
        cls,
        topology: CpuTopology,
        total_tasks: int = 1,
        max_cores_per_task: int | None = None,
        schedule: dict[str, int] | None = None,
        use_smt: bool = False,
    ) -> CoreBudgetPool:
        """Build a pool whose core budget is the whole (usable) ``topology``."""
        slots = CpuSetAllocator(topology, use_smt=use_smt).total_slots
        return cls(
            total_cores=slots,
            total_tasks=total_tasks,
            max_cores_per_task=max_cores_per_task,
            schedule=schedule,
            topology=topology,
            use_smt=use_smt,
        )

    @property
    def allocator(self) -> CpuSetAllocator:
        """The CPU-set allocator (a flat topology is created lazily if none was given)."""
        with self._lock:
            if self._allocator is None:
                self._allocator = CpuSetAllocator(
                    CpuTopology.flat(self.total_cores), use_smt=False
                )
            return self._allocator

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._remaining_tasks

    @property
    def available_cores(self) -> int:
        with self._lock:
            return self._available_cores

    def task_completed(self) -> None:
        with self._cond:
            self._remaining_tasks = max(0, self._remaining_tasks - 1)
            self._cond.notify_all()

    @staticmethod
    def solve_water_filling_schedule(
        tasks_meta: dict[str, tuple[int, int]],
        total_cores: int,
        alpha: float = 0.85,
    ) -> dict[str, int]:
        """Calculates optimal core allocation per locus via mathematical makespan balancing.

        Accounts for super-quadratic complexity and cache-miss overhead in ultra-long,
        divergent sequences (L >= 1800 aa), allocating cores to true critical stragglers
        while preserving single-core efficiency for standard loci.
        """
        if not tasks_meta or total_cores <= 1:
            return {k: 1 for k in tasks_meta}

        K = len(tasks_meta)
        C = total_cores

        work: dict[str, float] = {}
        p_max: dict[str, int] = {}

        for k, (n, length) in tasks_meta.items():
            n_eff = max(1, n)
            l_eff = max(1, length)

            # Super-quadratic complexity scaling for ultra-long unanchored loci
            if l_eff >= 1800:
                w = float(n_eff * (l_eff**2.4))
                ceiling = min(8, C)
            elif l_eff >= 1200:
                w = float(n_eff * (l_eff**2.1))
                ceiling = min(4, C)
            elif l_eff >= 600:
                w = float(n_eff * (l_eff**2.0))
                ceiling = min(2, C)
            else:
                w = float(n_eff * (l_eff**2.0))
                ceiling = 1
            work[k] = w
            p_max[k] = max(1, ceiling)

        total_work = sum(work.values())
        if total_work <= 0:
            return {k: 1 for k in tasks_meta}

        # Sparse Workload Regime (K < C)
        if K < C:
            work_scaled = {k: work[k] ** (1.0 / alpha) for k in work}
            sum_scaled = sum(work_scaled.values())
            allocations = {
                k: max(1, min(p_max[k], round((work_scaled[k] / sum_scaled) * C)))
                for k in work
            }
            assigned = sum(allocations.values())
            sorted_keys = sorted(work.keys(), key=lambda x: work[x], reverse=True)
            while assigned < C:
                progress = False
                for k in sorted_keys:
                    if allocations[k] < p_max[k] and assigned < C:
                        allocations[k] += 1
                        assigned += 1
                        progress = True
                if not progress:
                    break
            while assigned > C:
                for k in reversed(sorted_keys):
                    if allocations[k] > 1 and assigned > C:
                        allocations[k] -= 1
                        assigned -= 1
                        if assigned == C:
                            break
            return allocations

        # Dense Workload Regime (K >= C)
        ideal_makespan = total_work / C
        sorted_keys = sorted(work.keys(), key=lambda x: work[x], reverse=True)

        # True critical stragglers must significantly exceed ideal makespan
        outliers = [
            k for k in sorted_keys if work[k] > (1.2 * ideal_makespan) and p_max[k] > 1
        ]

        allocations = {k: 1 for k in work}
        if not outliers:
            return allocations

        # Allow critical stragglers to access full node capacity
        cores_remaining = C
        for k in outliers:
            wanted = p_max[k]
            grant = min(wanted, cores_remaining)
            if grant >= 8:
                p = 8
            elif grant >= 4:
                p = 4
            elif grant >= 2:
                p = 2
            else:
                p = 1

            allocations[k] = p
            cores_remaining = max(0, cores_remaining - p)
            if cores_remaining < 2:
                break

        return allocations

    @staticmethod
    def estimate_preferred_cores(file_size_bytes: int) -> int:
        """Fallback static sizing heuristic for standalone unit tests."""
        if file_size_bytes < 2 * 1024 * 1024:
            return 1
        elif file_size_bytes < 10 * 1024 * 1024:
            return 2
        elif file_size_bytes < 25 * 1024 * 1024:
            return 4
        else:
            return 16

    def _preferred(self, file_size: int, task_id: str | None) -> int:
        if task_id is not None and task_id in self._schedule:
            return self._schedule[task_id]
        return self.estimate_preferred_cores(file_size)

    def lease_cores(self, file_size: int = 0, task_id: str | None = None) -> int:
        """Block until cores are available and return how many were leased (int API)."""
        preferred = self._preferred(file_size, task_id)
        with self._cond:
            while True:
                allocated = compute_lease(
                    preferred,
                    self._remaining_tasks,
                    self._available_cores,
                    self.total_cores,
                    self.max_cores_per_task,
                )
                if allocated is not None:
                    break
                self._cond.wait()
            self._available_cores -= allocated
            return allocated

    def release_cores(self, count: int) -> None:
        with self._cond:
            self._available_cores = min(self.total_cores, self._available_cores + count)
            self._cond.notify_all()

    @contextmanager
    def lease(self, file_size: int = 0, task_id: str | None = None) -> Iterator[int]:
        allocated = self.lease_cores(file_size=file_size, task_id=task_id)
        try:
            yield allocated
        finally:
            self.release_cores(allocated)

    # -- CPU-set leases -----------------------------------------------------

    def lease_cpuset(
        self,
        file_size: int = 0,
        task_id: str | None = None,
        threads: int | None = None,
    ) -> CoreLease:
        """Like :meth:`lease_cores` but returns a concrete :class:`CoreLease`.

        The number of cores is decided by exactly the same policy as
        :meth:`lease_cores` (``threads`` overrides the preferred count). The CPUs
        come from the pool's topology: physical cores first, best-fit inside a
        single NUMA node when possible. Release with :meth:`release_lease`.
        """
        preferred = (
            max(1, int(threads))
            if threads is not None
            else self._preferred(file_size, task_id)
        )
        allocator = self.allocator
        with self._cond:
            while True:
                available = min(self._available_cores, allocator.free_count)
                wanted = compute_lease(
                    preferred,
                    self._remaining_tasks,
                    available,
                    self.total_cores,
                    self.max_cores_per_task,
                )
                if wanted is not None and available > 0:
                    lease = allocator.try_allocate(min(wanted, available))
                    if lease is not None:
                        self._available_cores -= lease.threads
                        return lease
                self._cond.wait()

    def release_lease(self, lease: CoreLease) -> None:
        """Return a lease obtained from :meth:`lease_cpuset`."""
        allocator = self.allocator
        with self._cond:
            allocator.release(lease)
            self._available_cores = min(
                self.total_cores, self._available_cores + lease.threads
            )
            self._cond.notify_all()

    @contextmanager
    def cpuset_lease(
        self,
        file_size: int = 0,
        task_id: str | None = None,
        threads: int | None = None,
    ) -> Iterator[CoreLease]:
        """Context manager around :meth:`lease_cpuset` / :meth:`release_lease`."""
        lease = self.lease_cpuset(file_size=file_size, task_id=task_id, threads=threads)
        try:
            yield lease
        finally:
            self.release_lease(lease)
