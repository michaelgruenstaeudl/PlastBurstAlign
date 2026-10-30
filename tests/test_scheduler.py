from __future__ import annotations

import _thread
import signal
import threading
import unittest
from pathlib import Path
from typing import Any

from plastburstalign.events import (
    ListSink,
    LocusFinished,
    LocusQueued,
    LocusStarted,
    PhaseChanged,
    ProgressEvent,
    WarningEvent,
)
from plastburstalign.scheduler import (
    FanOut,
    LocusScheduler,
    MemoryBudget,
    ReadyQueue,
    SchedTask,
    SchedulerReport,
    TaskContext,
    compute_schedule_metrics,
    lease_width,
)
from plastburstalign.task_queue import pin_command
from plastburstalign.topology import CpuTopology


class _FakeInterrupt(BaseException):
    """Stands in for KeyboardInterrupt raised in the caller thread."""


def noop(ctx: TaskContext) -> None:
    return None


def run_guarded(
    sch: LocusScheduler, tasks: list[SchedTask], timeout: float = 60.0, **kw: Any
) -> SchedulerReport:
    """Run with a watchdog so a scheduler bug fails the test instead of hanging."""
    timer = threading.Timer(timeout, sch.cancel)
    timer.daemon = True
    timer.start()
    try:
        return sch.run(tasks, **kw)
    finally:
        timer.cancel()


class TestMemoryBudget(unittest.TestCase):
    def test_fit_release_and_peak(self):
        mb = MemoryBudget(100)
        self.assertTrue(mb.try_acquire(60))
        self.assertFalse(mb.try_acquire(50))
        self.assertTrue(mb.try_acquire(40))
        self.assertEqual((mb.used, mb.remaining, mb.peak), (100, 0, 100))
        mb.release(60)
        self.assertTrue(mb.try_acquire(50))
        self.assertEqual(mb.peak, 100)

    def test_oversize_only_when_nothing_else_holds_memory(self):
        mb = MemoryBudget(100)
        self.assertTrue(mb.is_oversize(500))
        self.assertTrue(mb.try_acquire(10))
        self.assertFalse(mb.try_acquire(500))  # someone holds memory -> wait
        mb.release(10)
        self.assertTrue(mb.try_acquire(500))  # alone -> admitted
        self.assertEqual(mb.oversize_grants, 1)
        self.assertFalse(mb.try_acquire(1))  # envelope blown: nobody else starts
        self.assertTrue(mb.try_acquire(0))  # zero-byte tasks never block
        mb.release(500)
        self.assertTrue(mb.try_acquire(1))

    def test_unlimited(self):
        mb = MemoryBudget(None)
        self.assertTrue(mb.try_acquire(10**15))
        self.assertIsNone(mb.remaining)
        self.assertFalse(mb.is_oversize(10**15))


class TestReadyQueue(unittest.TestCase):
    def _queue(self, **kw: int) -> ReadyQueue[str]:
        q: ReadyQueue[str] = ReadyQueue(**kw)
        for i, name in enumerate(["a", "b", "c", "d", "e"]):
            q.push((1, -float(10 - i), (i,)), name, weight=float(10 - i))
        return q

    def test_priority_order_and_counts(self):
        q = self._queue()
        self.assertEqual(len(q), 5)
        self.assertEqual(q.weight, 10 + 9 + 8 + 7 + 6)
        order = []
        while True:
            got = q.pop_next(lambda item, n, w: True)
            if got is None:
                break
            order.append(got[0])
        self.assertEqual(order, ["a", "b", "c", "d", "e"])
        self.assertEqual(len(q), 0)
        self.assertEqual(q.weight, 0.0)

    def test_admit_sees_other_items(self):
        q = self._queue()
        seen: list[tuple[str, int, float]] = []

        def admit(item: str, n: int, w: float) -> bool:
            seen.append((item, n, w))
            return True

        q.pop_next(admit)
        self.assertEqual(seen, [("a", 4, 9 + 8 + 7 + 6)])

    def test_backfill_is_bounded_so_blocked_head_cannot_starve(self):
        q = self._queue(max_backfill=2)
        blocked = {"a"}

        def admit(item: str, n: int, w: float) -> bool | None:
            return None if item in blocked else True

        got = [q.pop_next(admit) for _ in range(4)]
        self.assertEqual([g[0] if g else None for g in got], ["b", "c", None, None])
        blocked.clear()  # head fits now -> admitted first, streak resets
        first = q.pop_next(admit)
        assert first is not None
        self.assertEqual(first[0], "a")
        blocked.add("d")
        nxt = [q.pop_next(admit) for _ in range(2)]
        self.assertEqual([g[0] if g else None for g in nxt], ["e", None])

    def test_zero_backfill_waits_for_head(self):
        q = self._queue(max_backfill=0)
        self.assertIsNone(q.pop_next(lambda item, n, w: None if item == "a" else True))
        self.assertEqual(len(q), 5)

    def test_scan_is_bounded(self):
        q: ReadyQueue[int] = ReadyQueue(max_backfill=10**6, max_scan=3)
        for i in range(100):
            q.push((i,), i)
        calls: list[int] = []

        def admit(item: int, n: int, w: float) -> bool | None:
            calls.append(item)
            return None

        self.assertIsNone(q.pop_next(admit))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(q), 100)

    def test_remove_where_and_drain(self):
        q = self._queue()
        gone = q.remove_where(lambda item: item in {"b", "d"})
        self.assertEqual(gone, ["b", "d"])
        self.assertEqual(len(q), 3)
        self.assertEqual(q.drain(), ["a", "c", "e"])
        self.assertEqual(len(q), 0)

    def test_admit_exception_keeps_items_queued(self):
        q = self._queue()

        def boom(item: str, n: int, w: float) -> bool:
            raise RuntimeError("x")

        with self.assertRaises(RuntimeError):
            q.pop_next(boom)
        self.assertEqual(len(q), 5)


class TestLeaseWidth(unittest.TestCase):
    def kw(self, **over: Any) -> dict[str, Any]:
        base = dict(
            work=10.0,
            preferred=1,
            min_threads=1,
            max_threads=8,
            idle=8,
            n_others=0,
            others_work=0.0,
            running=0,
            promote=True,
        )
        base.update(over)
        return base

    def test_lone_task_takes_all_idle_cores_up_to_max(self):
        self.assertEqual(lease_width(**self.kw()), 8)
        self.assertEqual(lease_width(**self.kw(max_threads=4)), 4)

    def test_deep_queue_uses_preferred_width(self):
        self.assertEqual(lease_width(**self.kw(n_others=20, others_work=200.0)), 1)
        self.assertEqual(
            lease_width(**self.kw(n_others=20, others_work=200.0, preferred=4)), 4
        )

    def test_work_proportional_share_leaves_a_core_per_other_task(self):
        # heavy task vs. two light ones: heavy takes most idle cores
        w = lease_width(**self.kw(work=90.0, n_others=2, others_work=10.0))
        self.assertEqual(w, 6)  # cap = idle - n_others
        # equal tasks split evenly
        self.assertEqual(
            lease_width(**self.kw(work=10.0, n_others=1, others_work=10.0)), 4
        )

    def test_promotion_can_be_disabled(self):
        self.assertEqual(lease_width(**self.kw(promote=False)), 1)

    def test_waits_for_half_of_preferred_unless_idle_system(self):
        self.assertIsNone(
            lease_width(**self.kw(preferred=8, idle=3, n_others=30, running=2))
        )
        self.assertEqual(
            lease_width(**self.kw(preferred=8, idle=4, n_others=30, running=2)), 4
        )
        # nothing running: always start, even below half of the preference
        self.assertEqual(
            lease_width(**self.kw(preferred=8, idle=3, n_others=30, running=0)), 3
        )

    def test_no_idle_core_means_wait(self):
        self.assertIsNone(lease_width(**self.kw(idle=0)))

    def test_min_threads_respected(self):
        self.assertIsNone(
            lease_width(**self.kw(min_threads=4, preferred=4, idle=2, running=1))
        )


class TestScheduleMetrics(unittest.TestCase):
    def test_efficiency_and_straggler_tax(self):
        m = compute_schedule_metrics([(0, 10, 1), (0, 4, 1)], total_cores=4)
        self.assertAlmostEqual(m.makespan, 10.0)
        self.assertAlmostEqual(m.total_cpu_seconds, 14.0)
        self.assertAlmostEqual(m.parallel_efficiency, 14 / 40)
        # busy < 2 cores (50% of 4) during 4..10
        self.assertAlmostEqual(m.straggler_tax, 0.6)
        self.assertAlmostEqual(m.mean_busy_cores, 1.4)

    def test_threads_weight_cpu_seconds(self):
        m = compute_schedule_metrics([(0, 5, 4), (0, 5, 4)], total_cores=8)
        self.assertAlmostEqual(m.total_cpu_seconds, 40.0)
        self.assertAlmostEqual(m.parallel_efficiency, 1.0)
        self.assertAlmostEqual(m.straggler_tax, 0.0)

    def test_window_clipping_and_empty(self):
        m = compute_schedule_metrics([(2, 6, 2)], 2, t_start=0.0, t_end=8.0)
        self.assertAlmostEqual(m.makespan, 8.0)
        self.assertAlmostEqual(m.straggler_tax, 0.5)  # idle 0..2 and 6..8
        empty = compute_schedule_metrics([], 4)
        self.assertEqual((empty.makespan, empty.total_cpu_seconds), (0.0, 0.0))


class TestLocusSchedulerBasics(unittest.TestCase):
    def test_lpt_order_with_deterministic_tie_break(self):
        order: list[str] = []

        def rec(ctx: TaskContext) -> None:
            order.append(ctx.task_id)

        works = [3, 10, 1, 10, 5]
        tasks = [SchedTask(f"t{i}", rec, work=float(w)) for i, w in enumerate(works)]
        run_guarded(LocusScheduler(1), tasks)
        self.assertEqual(order, ["t1", "t3", "t4", "t0", "t2"])

    def test_priority_overrides_work(self):
        order: list[str] = []

        def rec(ctx: TaskContext) -> None:
            order.append(ctx.task_id)

        tasks = [
            SchedTask("heavy", rec, work=100.0),
            SchedTask("urgent", rec, work=1.0, priority=500.0),
        ]
        run_guarded(LocusScheduler(1), tasks)
        self.assertEqual(order, ["urgent", "heavy"])

    def test_report_records_in_submission_order_regardless_of_completion(self):
        n = 200
        tasks = [SchedTask(f"loc{i:03d}", noop, work=float(i % 13)) for i in range(n)]
        rep = run_guarded(LocusScheduler(4), tasks)
        self.assertEqual([r.task_id for r in rep.records], [t.task_id for t in tasks])
        self.assertTrue(rep.ok)
        self.assertEqual(rep.n_ok, n)
        for r in rep.records:
            self.assertGreaterEqual(r.wait, 0.0)
            self.assertGreaterEqual(r.threads, 1)
            self.assertTrue(r.cpu_ids)

    def test_empty_run_and_scheduler_reuse(self):
        sch = LocusScheduler(2)
        self.assertEqual(sch.run([]).records, [])
        r1 = run_guarded(sch, [SchedTask("a", noop)])
        r2 = run_guarded(sch, [SchedTask("a", noop), SchedTask("b", noop)])
        self.assertEqual((len(r1.records), len(r2.records)), (1, 2))
        self.assertEqual(sch.alive_workers(), 0)

    def test_validation(self):
        sch = LocusScheduler(2)
        with self.assertRaises(ValueError):
            sch.run([SchedTask("a", noop), SchedTask("a", noop)])
        with self.assertRaises(TypeError):
            sch.run([object()])  # type: ignore[list-item]

    def test_results_kept_on_request(self):
        sch = LocusScheduler(2, keep_results=True)
        rep = run_guarded(sch, [SchedTask("a", lambda ctx: 41 + 1)])
        self.assertEqual(rep.results, {"a": 42})
        rep2 = run_guarded(LocusScheduler(2), [SchedTask("a", lambda ctx: 1)])
        self.assertEqual(rep2.results, {})

    def test_context_provides_threads_cpus_and_pinning(self):
        seen: dict[str, Any] = {}

        def fn(ctx: TaskContext) -> None:
            seen["threads"] = ctx.threads
            seen["cpus"] = ctx.cpu_ids
            seen["pinned"] = ctx.pin(["mafft", "x.fa"])
            seen["plain"] = TaskContext.pin  # attribute exists
            seen["cancelled"] = ctx.cancelled

        sch = LocusScheduler(4, pin=True)
        run_guarded(sch, [SchedTask("a", fn, preferred_threads=2, max_threads=2)])
        self.assertEqual(seen["threads"], 2)
        self.assertEqual(len(seen["cpus"]), 2)
        self.assertEqual(seen["pinned"], pin_command(["mafft", "x.fa"], seen["cpus"]))
        self.assertFalse(seen["cancelled"])

        unpinned: dict[str, Any] = {}

        def fn2(ctx: TaskContext) -> None:
            unpinned["cmd"] = ctx.pin(["mafft"])

        run_guarded(LocusScheduler(2, pin=False), [SchedTask("a", fn2)])
        self.assertEqual(unpinned["cmd"], ["mafft"])

    def test_concurrent_leases_are_disjoint_on_numa_smt_topology(self):
        topo = CpuTopology.synthetic(8, smt=2, numa_nodes=2)
        barrier = threading.Barrier(4)
        held: dict[str, tuple[int, ...]] = {}
        lock = threading.Lock()

        def fn(ctx: TaskContext) -> None:
            with lock:
                held[ctx.task_id] = ctx.cpu_ids
            barrier.wait(10)  # all four hold their leases at the same time

        tasks = [
            SchedTask(f"t{i}", fn, work=5.0, preferred_threads=2, max_threads=2)
            for i in range(4)
        ]
        run_guarded(LocusScheduler(topology=topo), tasks)
        all_cpus = [c for cpus in held.values() for c in cpus]
        self.assertEqual(len(all_cpus), len(set(all_cpus)))
        self.assertEqual(len(held), 4)
        # physical cores only (SMT off): no sibling CPUs 8..15
        self.assertTrue(all(c < 8 for c in all_cpus))
        # two cores per task, best-fit inside a node: each lease on one NUMA node
        for cpus in held.values():
            self.assertEqual(len({c // 4 for c in cpus}), 1)


class TestMemoryAdmission(unittest.TestCase):
    def test_blocked_task_waits_for_release_while_smaller_task_backfills(self):
        a_finished = threading.Event()
        c_done = threading.Event()
        release_a = threading.Event()
        observed: dict[str, bool] = {}
        ticks = {"n": 0}

        def fn_a(ctx: TaskContext) -> None:
            release_a.wait(20)
            a_finished.set()

        def fn_b(ctx: TaskContext) -> None:
            observed["a_still_running_when_b_started"] = not a_finished.is_set()

        def fn_c(ctx: TaskContext) -> None:
            c_done.set()

        tasks = [
            SchedTask("A", fn_a, work=10.0, est_mem_bytes=80),
            SchedTask("B", fn_b, work=5.0, est_mem_bytes=50),
            SchedTask("C", fn_c, work=1.0, est_mem_bytes=20),
        ]

        def on_tick() -> None:
            if c_done.is_set():
                ticks["n"] += 1
                if ticks["n"] >= 4:  # B had ample opportunity to (wrongly) start
                    release_a.set()

        rep = run_guarded(
            LocusScheduler(4, memory_budget=100),
            tasks,
            on_tick=on_tick,
            tick_seconds=0.02,
        )
        self.assertTrue(rep.ok)
        self.assertFalse(observed["a_still_running_when_b_started"])
        a, b, c = (rep.record(x) for x in "ABC")
        assert a and b and c
        self.assertGreaterEqual(b.start, a.end)  # B waited for A's memory
        self.assertLess(c.start, a.end)  # C backfilled next to A
        self.assertLessEqual(rep.peak_memory_bytes, 100)

    def test_single_oversize_task_runs_alone_with_warning(self):
        running: set[str] = set()
        lock = threading.Lock()
        overlaps: list[tuple[str, tuple[str, ...]]] = []
        big_running = threading.Event()
        release_big = threading.Event()
        ticks = {"n": 0}

        def small(ctx: TaskContext) -> None:
            with lock:
                if running:
                    overlaps.append((ctx.task_id, tuple(sorted(running))))
                running.add(ctx.task_id)
            with lock:
                running.discard(ctx.task_id)

        def big(ctx: TaskContext) -> None:
            with lock:
                if running:
                    overlaps.append((ctx.task_id, tuple(sorted(running))))
                running.add("big")
            big_running.set()
            release_big.wait(20)
            with lock:
                running.discard("big")

        tasks = [SchedTask("big", big, work=100.0, est_mem_bytes=500)] + [
            SchedTask(f"s{i}", small, work=1.0, est_mem_bytes=10) for i in range(5)
        ]

        def on_tick() -> None:
            ticks["n"] += 1
            if big_running.is_set() and ticks["n"] >= 6:
                release_big.set()

        sink = ListSink()
        rep = run_guarded(
            LocusScheduler(4, memory_budget=100, sink=sink),
            tasks,
            on_tick=on_tick,
            tick_seconds=0.02,
        )
        self.assertTrue(rep.ok)  # no deadlock, everything ran
        self.assertEqual(overlaps, [])
        self.assertEqual(rep.oversize_task_ids, ("big",))
        warns = sink.of_type(WarningEvent)
        self.assertEqual(len(warns), 1)
        self.assertEqual(warns[0].category, "memory")
        self.assertEqual(warns[0].locus, "big")
        self.assertTrue(rep.record("big").oversize)  # type: ignore[union-attr]

    def test_oversize_task_queued_behind_small_ones_waits_then_runs_alone(self):
        gate = threading.Event()
        lock = threading.Lock()
        active = {"smalls": 0}
        seen_by_big: list[int] = []
        ticks = {"n": 0}

        def small(ctx: TaskContext) -> None:
            with lock:
                active["smalls"] += 1
            gate.wait(20)  # hold memory until the caller opens the gate
            with lock:
                active["smalls"] -= 1

        def big(ctx: TaskContext) -> None:
            with lock:
                seen_by_big.append(active["smalls"])

        tasks = [SchedTask("big", big, work=1.0, est_mem_bytes=10**9)] + [
            SchedTask(f"s{i}", small, work=50.0, est_mem_bytes=40) for i in range(12)
        ]

        def on_tick() -> None:
            ticks["n"] += 1
            if ticks["n"] >= 3:
                gate.set()

        rep = run_guarded(
            LocusScheduler(4, memory_budget=100),
            tasks,
            on_tick=on_tick,
            tick_seconds=0.02,
        )
        self.assertTrue(rep.ok)
        self.assertEqual(rep.n_ok, 13)
        self.assertEqual(
            seen_by_big, [0]
        )  # started only when no small task held memory

    def test_memory_never_exceeds_envelope_under_load(self):
        budget = MemoryBudget(1000)
        tasks = [
            SchedTask(
                f"t{i}", noop, work=float(i % 9), est_mem_bytes=100 + (i % 5) * 60
            )
            for i in range(300)
        ]
        rep = run_guarded(LocusScheduler(8, memory_budget=budget), tasks)
        self.assertTrue(rep.ok)
        self.assertLessEqual(budget.peak, 1000)
        self.assertEqual(budget.used, 0)


class TestFailureIsolation(unittest.TestCase):
    def test_exception_in_one_task_does_not_abort_others(self):
        def make(i: int) -> SchedTask:
            def fn(ctx: TaskContext) -> int:
                if i % 3 == 0:
                    raise ValueError(f"bad locus {i}")
                return i

            return SchedTask(f"t{i}", fn, work=float(i))

        rep = run_guarded(
            LocusScheduler(4, keep_results=True), [make(i) for i in range(30)]
        )
        self.assertFalse(rep.ok)
        self.assertEqual(rep.failed_loci, [f"t{i}" for i in range(30) if i % 3 == 0])
        self.assertEqual(rep.n_ok, 20)
        self.assertEqual(len(rep.results), 20)
        self.assertEqual(len(rep.failed), 10)
        self.assertIn("ValueError: bad locus 0", rep.failed[0].error or "")
        self.assertEqual(rep.failed[0].status, "failed")
        self.assertFalse(rep.cancelled)

    def test_systemexit_in_task_is_isolated(self):
        def fn(ctx: TaskContext) -> None:
            raise SystemExit(3)

        rep = run_guarded(LocusScheduler(2), [SchedTask("x", fn), SchedTask("y", noop)])
        self.assertEqual(rep.failed_loci, ["x"])

    def test_fail_fast_cancels_remaining_tasks(self):
        def boom(ctx: TaskContext) -> None:
            raise RuntimeError("stop")

        tasks = [SchedTask("first", boom, work=100.0)] + [
            SchedTask(f"later{i}", noop, work=1.0) for i in range(5)
        ]
        rep = run_guarded(LocusScheduler(1, fail_fast=True), tasks)
        self.assertTrue(rep.cancelled)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.record("first").status, "failed")  # type: ignore[union-attr]
        for i in range(5):
            self.assertEqual(rep.record(f"later{i}").status, "cancelled")  # type: ignore[union-attr]

    def test_failing_task_releases_cores_and_memory(self):
        budget = MemoryBudget(100)

        def boom(ctx: TaskContext) -> None:
            raise RuntimeError("x")

        tasks = [SchedTask(f"t{i}", boom, est_mem_bytes=100) for i in range(20)]
        rep = run_guarded(LocusScheduler(2, memory_budget=budget), tasks)
        self.assertEqual(len(rep.failed), 20)  # all ran: nothing leaked or deadlocked
        self.assertEqual(budget.used, 0)


class TestFanOut(unittest.TestCase):
    def test_fan_out_fan_in_result_and_record_order(self):
        def subtask(j: int) -> SchedTask:
            return SchedTask(f"c{j}", lambda ctx, j=j: j * 10, work=5.0)

        def root(ctx: TaskContext) -> FanOut:
            return FanOut(
                [subtask(j) for j in range(5)],
                finalizer=lambda ctx2, results: sum(results),
            )

        sink = ListSink()
        sch = LocusScheduler(4, keep_results=True, sink=sink)
        rep = run_guarded(
            sch, [SchedTask("locus", root, work=50.0), SchedTask("other", noop)]
        )
        self.assertTrue(rep.ok)
        self.assertEqual(rep.results["locus"], 100)
        ids = [r.task_id for r in rep.records]
        self.assertEqual(
            ids,
            ["locus"]
            + [f"locus/c{j}" for j in range(5)]
            + ["locus::finalize", "other"],
        )
        kinds = [r.kind for r in rep.records]
        self.assertEqual(kinds, ["task"] + ["subtask"] * 5 + ["finalize", "task"])
        # events: sub-units carry the parent, the locus finishes exactly once
        started = {e.locus: e for e in sink.of_type(LocusStarted)}
        self.assertEqual(started["locus/c3"].parent, "locus")
        self.assertIsNone(started["locus"].parent)
        fin = [e for e in sink.of_type(LocusFinished) if e.locus == "locus"]
        self.assertEqual(len(fin), 1)
        self.assertTrue(fin[0].ok)
        queued = {e.locus for e in sink.of_type(LocusQueued)}
        self.assertIn("locus/c0", queued)
        self.assertIn("locus::finalize", queued)

    def test_plain_list_of_tasks_is_fan_out_and_default_result_is_list(self):
        def root(ctx: TaskContext) -> list[SchedTask]:
            return [SchedTask(f"s{j}", lambda c, j=j: j) for j in range(3)]

        rep = run_guarded(LocusScheduler(2, keep_results=True), [SchedTask("r", root)])
        self.assertEqual(rep.results["r"], [0, 1, 2])
        self.assertNotIn("finalize", [r.kind for r in rep.records])

    def test_task_level_finalizer_and_empty_fan_out(self):
        def root(ctx: TaskContext) -> FanOut:
            return FanOut([])

        task = SchedTask("r", root, finalizer=lambda ctx, res: ("done", res))
        rep = run_guarded(LocusScheduler(2, keep_results=True), [task])
        self.assertEqual(rep.results["r"], ("done", []))

    def test_empty_list_return_is_just_a_result(self):
        rep = run_guarded(
            LocusScheduler(2, keep_results=True), [SchedTask("r", lambda ctx: [])]
        )
        self.assertEqual(rep.results["r"], [])

    def test_subtasks_share_the_priority_queue_with_other_loci(self):
        order: list[str] = []

        def rec(ctx: TaskContext) -> None:
            order.append(ctx.task_id)

        def root(ctx: TaskContext) -> FanOut:
            order.append(ctx.task_id)
            subs = [SchedTask(f"c{j}", rec, work=50.0) for j in range(2)]
            return FanOut(subs, finalizer=lambda c, r: order.append("R1::fin"))

        tasks = [
            SchedTask("R1", root, work=100.0),
            SchedTask("R2", rec, work=10.0),
            SchedTask("R3", rec, work=60.0),
        ]
        run_guarded(LocusScheduler(1), tasks)
        # chunks (50) rank below R3 (60) but above R2 (10); the finalizer jumps the queue
        self.assertEqual(order, ["R1", "R3", "R1/c0", "R1/c1", "R1::fin", "R2"])

    def test_subtask_failure_fails_parent_skips_finalizer_and_cancels_siblings(self):
        finalizer_ran = threading.Event()

        def bad(ctx: TaskContext) -> None:
            raise RuntimeError("chunk failed")

        def root(ctx: TaskContext) -> FanOut:
            subs = [SchedTask("c0", bad, work=9.0)] + [
                SchedTask(f"c{j}", noop, work=1.0) for j in range(1, 6)
            ]
            return FanOut(subs, finalizer=lambda c, r: finalizer_ran.set())

        sink = ListSink()
        rep = run_guarded(
            LocusScheduler(1, sink=sink), [SchedTask("R", root), SchedTask("ok", noop)]
        )
        self.assertFalse(finalizer_ran.is_set())
        self.assertEqual(rep.failed_loci, ["R"])
        self.assertEqual(rep.loci["ok"], (True, None))
        self.assertIn("subtask `R/c0` failed", rep.loci["R"][1] or "")
        self.assertEqual(rep.record("R/c0").status, "failed")  # type: ignore[union-attr]
        for j in range(1, 6):
            self.assertEqual(rep.record(f"R/c{j}").status, "cancelled")  # type: ignore[union-attr]
        fin = [e for e in sink.of_type(LocusFinished) if e.locus == "R"]
        self.assertEqual([e.ok for e in fin], [False])

    def test_finalizer_failure_fails_parent(self):
        def root(ctx: TaskContext) -> FanOut:
            def fin(c: TaskContext, r: list[Any]) -> None:
                raise ValueError("splice failed")

            return FanOut([SchedTask("c", noop)], finalizer=fin)

        rep = run_guarded(LocusScheduler(2), [SchedTask("R", root)])
        self.assertEqual(rep.failed_loci, ["R"])
        self.assertIn("finalizer failed", rep.loci["R"][1] or "")

    def test_nested_fan_out(self):
        def leaf(v: int) -> SchedTask:
            return SchedTask(f"l{v}", lambda ctx: v)

        def mid(ctx: TaskContext) -> FanOut:
            return FanOut([leaf(1), leaf(2)], finalizer=lambda c, r: sum(r))

        def root(ctx: TaskContext) -> FanOut:
            return FanOut(
                [SchedTask("m0", mid), SchedTask("m1", mid)],
                finalizer=lambda c, r: sum(r),
            )

        rep = run_guarded(LocusScheduler(3, keep_results=True), [SchedTask("R", root)])
        self.assertTrue(rep.ok)
        self.assertEqual(rep.results["R"], 6)
        self.assertIn("R/m1/l2", [r.task_id for r in rep.records])

    def test_invalid_fan_out_fails_the_task_only(self):
        def dup(ctx: TaskContext) -> FanOut:
            return FanOut([SchedTask("x", noop), SchedTask("x", noop)])

        rep = run_guarded(
            LocusScheduler(2), [SchedTask("R", dup), SchedTask("ok", noop)]
        )
        self.assertEqual(rep.failed_loci, ["R"])

    def test_critical_path_follows_the_chunk_chain(self):
        def root(ctx: TaskContext) -> FanOut:
            return FanOut(
                [SchedTask(f"c{j}", noop, work=2.0) for j in range(3)],
                finalizer=lambda c, r: None,
            )

        rep = run_guarded(LocusScheduler(2), [SchedTask("R", root, work=10.0)])
        path = rep.critical_path
        self.assertEqual(path[0], "R")
        self.assertEqual(path[-1], "R::finalize")
        self.assertEqual(rep.critical_task_id, "R")
        self.assertIn("critical_path", rep.to_dict())


class TestStragglerPromotion(unittest.TestCase):
    def test_two_stragglers_split_idle_cores(self):
        barrier = threading.Barrier(2)

        def fn(ctx: TaskContext) -> None:
            barrier.wait(10)

        tasks = [
            SchedTask(f"s{i}", fn, work=10.0, preferred_threads=1, max_threads=8)
            for i in range(2)
        ]
        rep = run_guarded(LocusScheduler(8), tasks)
        self.assertEqual(sorted(r.threads for r in rep.records), [4, 4])
        cpus = [c for r in rep.records for c in r.cpu_ids]
        self.assertEqual(len(cpus), len(set(cpus)))

    def test_lone_straggler_takes_every_core_up_to_its_cap(self):
        rep = run_guarded(
            LocusScheduler(8), [SchedTask("s", noop, work=9.0, max_threads=8)]
        )
        self.assertEqual(rep.records[0].threads, 8)
        rep2 = run_guarded(
            LocusScheduler(8), [SchedTask("s", noop, work=9.0, max_threads=3)]
        )
        self.assertEqual(rep2.records[0].threads, 3)

    def test_default_max_threads_makes_promotion_opt_in(self):
        rep = run_guarded(LocusScheduler(8), [SchedTask("s", noop, work=9.0)])
        self.assertEqual(rep.records[0].threads, 1)

    def test_deep_queue_is_not_promoted_and_switch_disables_promotion(self):
        gate = threading.Barrier(8)

        def first_wave(ctx: TaskContext) -> None:
            gate.wait(10)

        tasks = [
            SchedTask(
                f"w{i}", first_wave, work=10.0, preferred_threads=1, max_threads=4
            )
            for i in range(8)
        ] + [
            SchedTask(f"x{i}", noop, work=1.0, preferred_threads=1, max_threads=4)
            for i in range(8)
        ]
        rep = run_guarded(LocusScheduler(8), tasks)
        for r in rep.records:
            if r.task_id.startswith("w"):
                self.assertEqual(r.threads, 1)  # 15 other tasks queued >= 8 idle cores
        off = run_guarded(
            LocusScheduler(8, promote_stragglers=False),
            [SchedTask(f"y{i}", noop, work=1.0, max_threads=4) for i in range(3)],
        )
        self.assertEqual({r.threads for r in off.records}, {1})

    def test_promotion_never_oversubscribes_cores(self):
        cores = 8
        lock = threading.Lock()
        active = {"threads": 0, "peak": 0}

        def fn(ctx: TaskContext) -> None:
            with lock:
                active["threads"] += ctx.threads
                active["peak"] = max(active["peak"], active["threads"])
            with lock:
                active["threads"] -= ctx.threads

        tasks = [
            SchedTask(
                f"t{i}", fn, work=float(1 + i % 7), preferred_threads=1, max_threads=6
            )
            for i in range(60)
        ]
        rep = run_guarded(LocusScheduler(cores), tasks)
        self.assertTrue(rep.ok)
        self.assertLessEqual(active["peak"], cores)


class TestScale(unittest.TestCase):
    def test_2000_tiny_tasks_with_memory_admission_no_deadlock(self):
        n = 2000
        tasks = [
            SchedTask(
                f"t{i}", noop, work=float(i % 11), est_mem_bytes=50 + (i % 4) * 25
            )
            for i in range(n)
        ]
        sch = LocusScheduler(8, memory_budget=400)
        rep = run_guarded(sch, tasks, timeout=60)
        self.assertEqual(len(rep.records), n)
        self.assertTrue(rep.ok)
        self.assertLessEqual(rep.peak_memory_bytes, 400)
        self.assertEqual(sch.alive_workers(), 0)

    def test_13k_tasks_are_cheap(self):
        n = 13_000
        tasks = [SchedTask(f"loc{i}", noop, work=float(i % 97)) for i in range(n)]
        sch = LocusScheduler(8)
        rep = run_guarded(sch, tasks, timeout=120)
        self.assertEqual(rep.n_ok, n)
        self.assertEqual(sch.alive_workers(), 0)
        self.assertEqual(
            [r.task_id for r in rep.records[:3]], ["loc0", "loc1", "loc2"]
        )  # deterministic report order

    def test_fan_out_of_many_chunks(self):
        def root(ctx: TaskContext) -> FanOut:
            return FanOut(
                [SchedTask(f"c{j}", lambda c, j=j: 1, work=1.0) for j in range(500)],
                finalizer=lambda c, r: sum(r),
            )

        rep = run_guarded(
            LocusScheduler(8, keep_results=True), [SchedTask("big", root)]
        )
        self.assertEqual(rep.results["big"], 500)


class TestEventsAndScratch(unittest.TestCase):
    def test_lifecycle_events_and_failing_sink_is_harmless(self):
        class Exploding:
            def emit(self, event: ProgressEvent) -> None:
                raise RuntimeError("sink down")

        rep = run_guarded(
            LocusScheduler(2, sink=Exploding()),
            [SchedTask("a", noop), SchedTask("b", noop)],
        )
        self.assertTrue(rep.ok)

        sink = ListSink()
        tasks = [
            SchedTask(f"t{i}", noop, work=float(i), n_seq=10 + i, max_len=900)
            for i in range(6)
        ]
        run_guarded(LocusScheduler(3, sink=sink), tasks, phase="align")
        self.assertEqual([e.phase for e in sink.of_type(PhaseChanged)], ["align"])
        self.assertEqual(len(sink.of_type(LocusQueued)), 6)
        self.assertEqual(len(sink.of_type(LocusStarted)), 6)
        finished = sink.of_type(LocusFinished)
        self.assertEqual(len(finished), 6)
        self.assertTrue(all(e.ok for e in finished))
        st = {e.locus: e for e in sink.of_type(LocusStarted)}
        self.assertEqual(
            (st["t3"].n_seq, st["t3"].max_len, st["t3"].work), (13, 900, 3.0)
        )
        self.assertTrue(st["t3"].cpu_ids)

    def test_scratch_ticket_is_provided_and_released(self):
        released: list[str] = []

        class FakeTicket:
            def __init__(self, nbytes: int) -> None:
                self.path = Path("/tmp/fake-ticket")
                self.tier = "ramdisk" if nbytes < 1000 else "disk"

            def release(self) -> None:
                released.append(self.tier)

        class FakeEngine:
            def reserve(self, nbytes: int) -> FakeTicket:
                return FakeTicket(nbytes)

        seen: dict[str, Any] = {}

        def with_scratch(ctx: TaskContext) -> None:
            assert ctx.scratch is not None
            seen["tier"] = ctx.scratch.tier
            seen["tmpdir"] = ctx.subprocess_env({"A": "b"})

        def without(ctx: TaskContext) -> None:
            seen["none"] = ctx.scratch is None

        sch = LocusScheduler(2, scratch=FakeEngine())
        rep = run_guarded(
            sch,
            [
                SchedTask("a", with_scratch, scratch_bytes=10),
                SchedTask("b", without, scratch_bytes=0),
            ],
        )
        self.assertTrue(rep.ok)
        self.assertEqual(seen["tier"], "ramdisk")
        self.assertEqual(seen["tmpdir"], {"A": "b", "TMPDIR": "/tmp/fake-ticket"})
        self.assertTrue(seen["none"])
        self.assertEqual(released, ["ramdisk"])
        self.assertEqual(rep.record("a").scratch_tier, "ramdisk")  # type: ignore[union-attr]

    def test_scratch_released_even_when_task_fails(self):
        released: list[int] = []

        class T:
            path = Path("/tmp/x")
            tier = "disk"

            def release(self) -> None:
                released.append(1)

        class E:
            def reserve(self, nbytes: int) -> T:
                return T()

        def boom(ctx: TaskContext) -> None:
            raise RuntimeError("x")

        run_guarded(
            LocusScheduler(1, scratch=E()), [SchedTask("a", boom, scratch_bytes=5)]
        )
        self.assertEqual(released, [1])


class TestShutdown(unittest.TestCase):
    def test_caller_thread_interrupt_cancels_and_joins_cleanly(self):
        started = threading.Semaphore(0)
        counter = {"n": 0}
        lock = threading.Lock()

        def blocker(ctx: TaskContext) -> str:
            with lock:
                counter["n"] += 1
            started.release()
            ctx.cancel_event.wait(20)  # a cooperative task sees the cancellation
            return "stopped"

        tasks = [SchedTask(f"t{i}", blocker, work=float(100 - i)) for i in range(8)]

        def on_tick() -> None:
            with lock:
                if counter["n"] >= 2:
                    raise _FakeInterrupt()

        sch = LocusScheduler(2)
        with self.assertRaises(_FakeInterrupt):
            run_guarded(sch, tasks, on_tick=on_tick, tick_seconds=0.01)
        self.assertEqual(sch.alive_workers(), 0)  # clean shutdown, nothing leaked
        rep = sch.last_report
        assert rep is not None
        self.assertTrue(rep.cancelled)
        ran = [r for r in rep.records if r.status == "ok"]
        cancelled = [r for r in rep.records if r.status == "cancelled"]
        self.assertEqual(len(ran), 2)  # the two running tasks were allowed to finish
        self.assertEqual(len(cancelled), 6)  # queued ones never started
        self.assertFalse(rep.ok)
        self.assertEqual(len(rep.failed_loci), 6)

    @unittest.skipUnless(
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "SIGINT")
        and callable(signal.getsignal(signal.SIGINT)),
        "needs a Python-level SIGINT handler in the main thread",
    )
    def test_real_keyboard_interrupt_in_main_thread(self):
        fired = threading.Event()

        def trigger(ctx: TaskContext) -> None:
            if not fired.is_set():
                fired.set()
                _thread.interrupt_main()  # what Ctrl-C does
            ctx.cancel_event.wait(20)

        tasks = [SchedTask(f"t{i}", trigger, work=float(10 - i)) for i in range(6)]
        sch = LocusScheduler(2)
        with self.assertRaises(KeyboardInterrupt):
            run_guarded(sch, tasks, tick_seconds=0.01)
        self.assertEqual(sch.alive_workers(), 0)
        assert sch.last_report is not None
        self.assertTrue(sch.last_report.cancelled)

    def test_external_cancel(self):
        started = threading.Event()

        def blocker(ctx: TaskContext) -> None:
            started.set()
            ctx.cancel_event.wait(20)

        sch = LocusScheduler(1)
        tasks = [SchedTask(f"t{i}", blocker, work=float(5 - i)) for i in range(4)]
        threading.Thread(
            target=lambda: (started.wait(10), sch.cancel()), daemon=True
        ).start()
        rep = run_guarded(sch, tasks)
        self.assertTrue(rep.cancelled)
        self.assertEqual(sum(1 for r in rep.records if r.status == "ok"), 1)
        self.assertEqual(sum(1 for r in rep.records if r.status == "cancelled"), 3)


class TestReportMetrics(unittest.TestCase):
    def test_report_metrics_and_summary(self):
        tasks = [SchedTask(f"t{i}", noop, work=float(i + 1)) for i in range(10)]
        rep = run_guarded(LocusScheduler(4), tasks)
        self.assertEqual(rep.total_cores, 4)
        self.assertGreaterEqual(rep.makespan, 0.0)
        self.assertGreaterEqual(rep.total_cpu_seconds, 0.0)
        self.assertTrue(0.0 <= rep.parallel_efficiency <= 1.0)
        self.assertTrue(0.0 <= rep.straggler_tax <= 1.0)
        self.assertIn("10/10 loci ok", rep.summary())
        d = rep.to_dict()
        for key in (
            "makespan",
            "total_cpu_seconds",
            "parallel_efficiency",
            "straggler_tax",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["n_tasks"], 10)
        self.assertIsNotNone(rep.critical_task_id)

    def test_task_record_helpers(self):
        rep = run_guarded(LocusScheduler(2), [SchedTask("a", noop)])
        rec = rep.records[0]
        self.assertEqual(rec.to_dict()["task_id"], "a")
        self.assertGreaterEqual(rec.duration, 0.0)
        self.assertTrue(rec.ok)


if __name__ == "__main__":
    unittest.main()
