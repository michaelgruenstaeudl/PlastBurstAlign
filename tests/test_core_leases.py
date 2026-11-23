from __future__ import annotations

import random
import threading
import unittest

from plastburstalign.task_queue import (
    CoreBudgetPool,
    CoreLease,
    CpuSetAllocator,
    compute_lease,
    format_cpu_list,
    pin_command,
)
from plastburstalign.topology import CpuTopology


class TestCpuSetAllocator(unittest.TestCase):
    def test_basic_lease_physical_cores_first(self):
        alloc = CpuSetAllocator(CpuTopology.synthetic(8))
        lease = alloc.try_allocate(3)
        assert lease is not None
        self.assertEqual(lease.core_ids, (0, 1, 2))
        self.assertEqual(lease.cpu_ids, (0, 1, 2))
        self.assertEqual(lease.numa_node, 0)
        self.assertEqual(lease.threads, 3)
        self.assertFalse(lease.spans_numa)
        self.assertEqual(alloc.free_count, 5)

    def test_smt_siblings_never_leased_by_default(self):
        topo = CpuTopology.synthetic(4, smt=2)
        alloc = CpuSetAllocator(topo)
        self.assertEqual(alloc.total_slots, 4)
        lease = alloc.try_allocate(4)
        assert lease is not None
        self.assertEqual(lease.cpu_ids, (0, 1, 2, 3))  # primary threads only
        self.assertIsNone(alloc.try_allocate(1))
        self.assertEqual(alloc.free_cpu_ids(), ())

    def test_smt_enabled_uses_physical_cores_before_siblings(self):
        topo = CpuTopology.synthetic(4, smt=2)
        alloc = CpuSetAllocator(topo, use_smt=True)
        self.assertEqual(alloc.total_slots, 8)
        first = alloc.try_allocate(4)
        assert first is not None
        self.assertEqual(first.cpu_ids, (0, 1, 2, 3))
        self.assertEqual(first.core_ids, (0, 1, 2, 3))
        second = alloc.try_allocate(4)
        assert second is not None
        self.assertEqual(second.cpu_ids, (4, 5, 6, 7))  # siblings, only now

    def test_smt_idle_cores_preferred_over_sharing_busy_cores(self):
        alloc = CpuSetAllocator(CpuTopology.synthetic(4, smt=2), use_smt=True)
        a = alloc.try_allocate(2)
        assert a is not None and a.cpu_ids == (0, 1)
        b = alloc.try_allocate(3)
        assert b is not None
        # fully idle cores 2 and 3 (incl. core 2's sibling) before cores 0/1's siblings
        self.assertEqual(b.cpu_ids, (2, 3, 6))

    def test_numa_best_fit_inside_one_node(self):
        topo = CpuTopology.synthetic(8, numa_nodes=2)  # node0: 0-3, node1: 4-7
        alloc = CpuSetAllocator(topo)
        a = alloc.try_allocate(3)
        assert a is not None
        self.assertEqual((a.numa_node, a.core_ids), (0, (0, 1, 2)))
        # node0 has 1 free, node1 has 4: best fit for 1 core is the nearly full node0
        b = alloc.try_allocate(1)
        assert b is not None
        self.assertEqual((b.numa_node, b.core_ids), (0, (3,)))
        c = alloc.try_allocate(4)
        assert c is not None
        self.assertEqual((c.numa_node, c.core_ids), (1, (4, 5, 6, 7)))
        self.assertFalse(c.spans_numa)

    def test_spill_across_nodes_only_when_needed(self):
        topo = CpuTopology.synthetic(8, numa_nodes=2)
        alloc = CpuSetAllocator(topo)
        a = alloc.try_allocate(3)
        assert a is not None  # node0 left with 1 free, node1 with 4
        spill = alloc.try_allocate(5)
        assert spill is not None
        self.assertTrue(spill.spans_numa)
        self.assertEqual(spill.numa_nodes, (0, 1))
        self.assertEqual(spill.numa_node, 1)  # holds 4 of the 5 cores
        self.assertEqual(spill.core_ids, (3, 4, 5, 6, 7))

    def test_all_or_nothing_and_release(self):
        alloc = CpuSetAllocator(CpuTopology.flat(4))
        a = alloc.try_allocate(3)
        assert a is not None
        self.assertIsNone(alloc.try_allocate(2))
        self.assertEqual(alloc.free_count, 1)  # failed request changed nothing
        alloc.release(a)
        alloc.release(a)  # idempotent
        self.assertEqual(alloc.free_count, 4)
        again = alloc.try_allocate(4)
        assert again is not None
        self.assertEqual(again.cpu_ids, (0, 1, 2, 3))

    def test_max_slots_restricts_usable_cores(self):
        alloc = CpuSetAllocator(CpuTopology.flat(8), max_slots=3)
        self.assertEqual(alloc.total_slots, 3)
        lease = alloc.try_allocate(3)
        assert lease is not None
        self.assertIsNone(alloc.try_allocate(1))

    def test_concurrent_leases_never_overlap(self):
        topo = CpuTopology.synthetic(12, smt=2, numa_nodes=2)
        alloc = CpuSetAllocator(topo, use_smt=True)
        in_use: set[int] = set()
        guard = threading.Lock()
        errors: list[str] = []

        def worker(seed: int) -> None:
            rng = random.Random(seed)
            for _ in range(300):
                lease = alloc.try_allocate(rng.randint(1, 6))
                if lease is None:
                    continue
                with guard:
                    overlap = in_use.intersection(lease.cpu_ids)
                    if overlap:
                        errors.append(f"overlap {sorted(overlap)}")
                    in_use.update(lease.cpu_ids)
                for _ in range(rng.randint(0, 400)):  # hold the lease for a moment
                    pass
                with guard:
                    in_use.difference_update(lease.cpu_ids)
                alloc.release(lease)

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(alloc.free_count, alloc.total_slots)


class TestPinCommand(unittest.TestCase):
    def test_format_cpu_list(self):
        self.assertEqual(format_cpu_list([0, 1, 2, 3, 8]), "0-3,8")
        self.assertEqual(format_cpu_list([5]), "5")
        self.assertEqual(format_cpu_list([4, 2, 3, 3, 7, 9, 10]), "2-4,7,9-10")
        self.assertEqual(format_cpu_list([]), "")

    def test_pins_with_taskset(self):
        out = pin_command(
            ["mafft", "--auto", "in.fa"],
            [0, 1, 2, 3, 8],
            platform="linux",
            which=lambda _name: "/usr/bin/taskset",
        )
        self.assertEqual(
            out, ["/usr/bin/taskset", "-c", "0-3,8", "mafft", "--auto", "in.fa"]
        )

    def test_unchanged_when_taskset_missing(self):
        cmd = ["mafft", "in.fa"]
        out = pin_command(cmd, [0, 1], platform="linux", which=lambda _n: None)
        self.assertEqual(out, cmd)
        self.assertIsNot(out, cmd)  # always a fresh list

    def test_unchanged_off_linux_or_without_cpus(self):
        cmd = ["mafft", "in.fa"]
        which = lambda _n: "/usr/bin/taskset"  # noqa: E731
        self.assertEqual(pin_command(cmd, [0], platform="darwin", which=which), cmd)
        self.assertEqual(pin_command(cmd, [0], platform="win32", which=which), cmd)
        self.assertEqual(pin_command(cmd, [], platform="linux", which=which), cmd)
        self.assertEqual(pin_command(cmd, None, platform="linux", which=which), cmd)

    def test_does_not_double_pin(self):
        cmd = ["/usr/bin/taskset", "-c", "0", "mafft"]
        out = pin_command(
            cmd, [1], platform="linux", which=lambda _n: "/usr/bin/taskset"
        )
        self.assertEqual(out, cmd)

    def test_default_discovery_never_raises(self):
        out = pin_command(["true"], [0])
        self.assertEqual(out[-1], "true")


class TestComputeLease(unittest.TestCase):
    def test_matches_documented_policy(self):
        # deep queue, single core preferred
        self.assertEqual(compute_lease(1, 100, 16, 16, 16), 1)
        # saturated -> wait
        self.assertIsNone(compute_lease(1, 100, 0, 16, 16))
        # tail fair share
        self.assertEqual(compute_lease(1, 2, 16, 16, 16), 8)
        # lone remaining task takes everything
        self.assertEqual(compute_lease(1, 1, 16, 16, 16), 16)
        # multi-core quota waits for at least half of the target
        self.assertIsNone(compute_lease(8, 100, 3, 16, 16))
        self.assertEqual(compute_lease(8, 100, 4, 16, 16), 4)

    def test_preference_larger_than_pool_cannot_deadlock(self):
        # A 25 MB file prefers 16 cores; a 4-core pool must still make progress.
        got = compute_lease(16, 50, 4, 4, 4)
        self.assertEqual(got, 4)


class TestPoolCpuSetLeases(unittest.TestCase):
    def test_lease_cpuset_and_release_keep_counters_consistent(self):
        topo = CpuTopology.synthetic(8, numa_nodes=2)
        pool = CoreBudgetPool.from_topology(topo, total_tasks=100)
        self.assertEqual(pool.total_cores, 8)
        lease = pool.lease_cpuset(file_size=1000)
        self.assertIsInstance(lease, CoreLease)
        self.assertEqual(lease.threads, 1)
        self.assertEqual(pool.available_cores, 7)
        pool.release_lease(lease)
        self.assertEqual(pool.available_cores, 8)
        self.assertEqual(pool.allocator.free_count, 8)

    def test_schedule_preference_gives_multi_core_lease(self):
        topo = CpuTopology.synthetic(8, numa_nodes=2)
        pool = CoreBudgetPool.from_topology(topo, total_tasks=100, schedule={"big": 4})
        with pool.cpuset_lease(task_id="big") as lease:
            self.assertEqual(lease.threads, 4)
            self.assertEqual(lease.core_ids, (0, 1, 2, 3))
            self.assertEqual(pool.available_cores, 4)
        self.assertEqual(pool.available_cores, 8)

    def test_threads_override_and_solo_straggler(self):
        pool = CoreBudgetPool.from_topology(CpuTopology.flat(6), total_tasks=1)
        lease = pool.lease_cpuset()
        self.assertEqual(lease.threads, 6)  # lone remaining task takes all cores
        pool.release_lease(lease)
        pool2 = CoreBudgetPool.from_topology(CpuTopology.flat(6), total_tasks=50)
        with pool2.cpuset_lease(threads=3) as lease3:
            self.assertEqual(lease3.threads, 3)

    def test_total_cores_clamped_to_topology(self):
        pool = CoreBudgetPool(16, topology=CpuTopology.flat(4))
        self.assertEqual(pool.total_cores, 4)
        restricted = CoreBudgetPool(2, topology=CpuTopology.flat(8))
        self.assertEqual(restricted.total_cores, 2)
        self.assertEqual(restricted.allocator.total_slots, 2)

    def test_int_api_unchanged_without_topology(self):
        pool = CoreBudgetPool(total_cores=4, total_tasks=10)
        self.assertEqual(pool.lease_cores(file_size=1), 1)
        # a flat topology is created lazily for cpuset leases
        lease = pool.lease_cpuset()
        self.assertEqual(lease.cpu_ids, (0,))
        self.assertEqual(pool.available_cores, 2)
        pool.release_lease(lease)
        pool.release_cores(1)
        self.assertEqual(pool.available_cores, 4)

    def test_blocked_cpuset_lease_wakes_on_release(self):
        pool = CoreBudgetPool.from_topology(CpuTopology.flat(2), total_tasks=10)
        a = pool.lease_cpuset()
        b = pool.lease_cpuset()
        got: list[CoreLease] = []
        started = threading.Event()

        def waiter() -> None:
            started.set()
            got.append(pool.lease_cpuset())

        t = threading.Thread(target=waiter)
        t.start()
        started.wait(5)
        pool.release_lease(a)
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].cpu_ids, a.cpu_ids)
        pool.release_lease(got[0])
        pool.release_lease(b)
        self.assertEqual(pool.available_cores, 2)


if __name__ == "__main__":
    unittest.main()
