"""Shared CPU topology primitives.

This tiny module is the contract between the hardware profiler
(`hardware.py`, which *detects* topology) and the scheduler
(`scheduler.py` / `task_queue.py`, which *leases* cores). Keep it free of
imports from other package modules so both sides can depend on it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CoreInfo:
    """One physical core (with its SMT siblings) inside the usable CPU set.

    Attributes:
        core_id: Dense index (0..n-1) of this core within the usable set. It is
            *not* necessarily the OS core id; it is stable and deterministic.
        cpu_ids: Sorted OS logical CPU ids that belong to this physical core.
        numa_node: NUMA node index owning this core (0 on single-socket hosts).
    """

    core_id: int
    cpu_ids: tuple[int, ...]
    numa_node: int = 0


@dataclass(frozen=True)
class CpuTopology:
    """Immutable description of the CPUs this process may use."""

    cores: tuple[CoreInfo, ...]

    @property
    def physical_count(self) -> int:
        return len(self.cores)

    @property
    def logical_count(self) -> int:
        return sum(len(c.cpu_ids) for c in self.cores)

    @property
    def smt_width(self) -> int:
        """Maximum number of logical CPUs per physical core (1 if no SMT)."""
        return max((len(c.cpu_ids) for c in self.cores), default=1)

    @property
    def numa_nodes(self) -> tuple[int, ...]:
        return tuple(sorted({c.numa_node for c in self.cores}))

    def cores_on_node(self, node: int) -> tuple[CoreInfo, ...]:
        return tuple(c for c in self.cores if c.numa_node == node)

    def cpu_ids_for(
        self, core_ids: Iterable[int], include_smt_siblings: bool = False
    ) -> tuple[int, ...]:
        """Return sorted OS CPU ids for the given dense core ids.

        By default only the first (primary) logical CPU of each core is
        returned, so pinned workers do not share a physical core.
        """
        wanted = set(core_ids)
        out: list[int] = []
        for c in self.cores:
            if c.core_id in wanted and c.cpu_ids:
                out.extend(c.cpu_ids if include_smt_siblings else c.cpu_ids[:1])
        return tuple(sorted(out))

    @classmethod
    def synthetic(
        cls, physical_cores: int, smt: int = 1, numa_nodes: int = 1
    ) -> CpuTopology:
        """Build a deterministic synthetic topology (used by tests and fallbacks).

        Layout mimics Linux: primary threads are CPUs ``0..n-1`` and SMT
        siblings follow (``n..2n-1``). Cores are split evenly and contiguously
        across NUMA nodes.
        """
        n = max(1, int(physical_cores))
        smt = max(1, int(smt))
        nodes = max(1, min(int(numa_nodes), n))
        per_node = -(-n // nodes)  # ceil division
        cores = tuple(
            CoreInfo(
                core_id=i,
                cpu_ids=tuple(i + k * n for k in range(smt)),
                numa_node=min(i // per_node, nodes - 1),
            )
            for i in range(n)
        )
        return cls(cores=cores)

    @classmethod
    def flat(cls, cpus: int) -> CpuTopology:
        """A flat topology of ``cpus`` single-threaded cores on one NUMA node."""
        return cls.synthetic(cpus, smt=1, numa_nodes=1)
