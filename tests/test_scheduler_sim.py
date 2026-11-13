"""Discrete-event simulation of the three scheduling policies.

Numbers are printed (run with ``-s``) and logged so the straggler claims are backed
by simulator output, not assertions alone. Only the *ordering* is asserted, with a
modest tolerance.
"""

from __future__ import annotations

import logging
import unittest

from plastburstalign.scheduler_sim import (
    Policy,
    SimResult,
    SimTask,
    long_tail_workload,
    simulate,
)

TOL = 0.02  # modest tolerance on makespan ordering
LOG = logging.getLogger("plastburstalign.test.scheduler_sim")


def emit(title: str, results: list[tuple[str, SimResult]]) -> None:
    """Print + log a comparison table (visible with ``pytest -s``)."""
    lines = [f"--- {title} ---"]
    for label, res in results:
        lines.append(f"  {res.format_row()}  [{label}]")
    text = "\n".join(lines)
    print(text)
    LOG.info(text)


def peak_busy(res: SimResult) -> int:
    """Largest number of simultaneously leased cores."""
    points: list[tuple[float, int]] = []
    for r in res.records:
        if r.end > r.start:
            points.append((r.start, r.threads))
            points.append((r.end, -r.threads))
    points.sort(key=lambda p: (p[0], p[1]))  # releases before acquisitions at ties
    cur = peak = 0
    for _, d in points:
        cur += d
        peak = max(peak, cur)
    return peak


class TestSimulatorBasics(unittest.TestCase):
    def test_amdahl_duration_model(self):
        t = SimTask("x", work=100.0, serial_fraction=0.2, max_threads=4)
        self.assertAlmostEqual(t.duration(1), 100.0)
        self.assertAlmostEqual(t.duration(4), 100 * (0.2 + 0.8 / 4))
        self.assertAlmostEqual(
            t.duration(64), t.duration(4)
        )  # saturates at max_threads

    def test_static_fifo_single_core_is_serial(self):
        tasks = [SimTask(f"t{i}", 10.0, max_threads=1) for i in range(5)]
        res = simulate(tasks, 1, Policy.STATIC_FIFO)
        self.assertAlmostEqual(res.makespan, 50.0)
        self.assertAlmostEqual(res.metrics.parallel_efficiency, 1.0)

    def test_every_policy_runs_every_task_exactly_once_without_oversubscription(self):
        wl = long_tail_workload()
        for cores in (4, 16, 64):
            for pol in Policy:
                res = simulate(wl, cores, pol)
                roots = [r.task_id for r in res.records if r.parent_id is None]
                self.assertEqual(
                    sorted(roots), sorted(t.task_id for t in wl), (pol, cores)
                )
                self.assertLessEqual(peak_busy(res), cores, (pol, cores))
                self.assertEqual(set(res.completion), {t.task_id for t in wl})
                self.assertGreater(res.makespan, 0)

    def test_deterministic(self):
        wl = long_tail_workload()
        a = simulate(wl, 16, Policy.LOCUS_SCHEDULER)
        b = simulate(wl, 16, Policy.LOCUS_SCHEDULER)
        self.assertEqual(a.records, b.records)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            simulate([SimTask("a", 1.0), SimTask("a", 1.0)], 4, Policy.STATIC_FIFO)

    def test_string_policy_names(self):
        wl = long_tail_workload(n_short=5, n_long=1)
        self.assertEqual(simulate(wl, 4, "static_fifo").policy, Policy.STATIC_FIFO)

    def test_chunked_locus_records_plan_chunks_and_finalizer(self):
        chunks = tuple(SimTask(f"c{j}", 10.0, max_threads=1) for j in range(3))
        task = SimTask("L", 100.0, chunks=chunks, plan_work=1.0, finalize_work=2.0)
        res = simulate([task], 4, Policy.LOCUS_SCHEDULER)
        kinds = [r.kind for r in res.records]
        self.assertEqual(kinds.count("subtask"), 3)
        self.assertEqual(kinds.count("finalize"), 1)
        # plan (1) -> 3 parallel chunks (10) -> finalize (2)
        self.assertAlmostEqual(res.makespan, 13.0)
        mono = simulate([task], 4, Policy.LOCUS_SCHEDULER, use_chunks=False)
        self.assertGreater(mono.makespan, res.makespan)


class TestLongTailOrdering(unittest.TestCase):
    """40 short loci + 2 very long ones: (c) <= (b) <= (a)."""

    def _compare(
        self,
        cores: int,
        workload: list[SimTask],
        title: str,
        b_beats_a: bool = True,
    ) -> dict[str, SimResult]:
        a = simulate(workload, cores, Policy.STATIC_FIFO)
        b = simulate(workload, cores, Policy.LPT_WATERFILL)
        c = simulate(workload, cores, Policy.LOCUS_SCHEDULER)
        c_nochunk = simulate(workload, cores, Policy.LOCUS_SCHEDULER, use_chunks=False)
        emit(
            f"{title} on {cores} cores",
            [
                ("a: naive static-lease FIFO", a),
                ("b: LPT + CoreBudgetPool water-filling", b),
                ("c: LocusScheduler (promotion + chunk fan-out)", c),
                ("c without chunk fan-out (promotion only)", c_nochunk),
            ],
        )
        self.assertLessEqual(
            c.makespan, b.makespan * (1 + TOL), f"(c) {c.makespan} vs (b) {b.makespan}"
        )
        self.assertLessEqual(
            c.makespan, a.makespan * (1 + TOL), f"(c) {c.makespan} vs (a) {a.makespan}"
        )
        if b_beats_a:
            self.assertLessEqual(
                b.makespan,
                a.makespan * (1 + TOL),
                f"(b) {b.makespan} vs (a) {a.makespan}",
            )
        return {"a": a, "b": b, "c": c, "c_nochunk": c_nochunk}

    def test_40_short_plus_2_long_on_16_cores(self):
        res = self._compare(16, long_tail_workload(40, 2), "40 short + 2 long")
        self.assertLess(res["c"].makespan, res["b"].makespan)  # chunking really helps
        self.assertLess(res["b"].makespan, res["a"].makespan)

    def test_40_short_plus_2_long_on_64_cores(self):
        res = self._compare(64, long_tail_workload(40, 2), "40 short + 2 long")
        self.assertLess(res["c"].makespan, res["b"].makespan)
        # with chunk fan-out the straggler tail is absorbed by idle cores
        self.assertLess(res["c"].metrics.straggler_tax, res["b"].metrics.straggler_tax)
        self.assertGreater(
            res["c"].metrics.parallel_efficiency, res["b"].metrics.parallel_efficiency
        )

    def test_chunk_fan_out_is_what_absorbs_idle_cores_at_64_cores(self):
        wl = long_tail_workload(40, 2)
        with_chunks = simulate(wl, 64, Policy.LOCUS_SCHEDULER)
        without = simulate(wl, 64, Policy.LOCUS_SCHEDULER, use_chunks=False)
        self.assertLess(with_chunks.makespan, 0.5 * without.makespan)

    def test_ordering_holds_even_without_superlinear_chunk_savings(self):
        # chunks cost the same total work as the monolithic locus
        wl = long_tail_workload(40, 2, chunk_efficiency=1.0)
        for cores in (16, 64):
            self._compare(cores, wl, "long tail, chunk_efficiency=1.0")

    def test_new_policy_dominates_for_other_tail_shapes(self):
        # Under saturation (100 short loci keep 8 cores busy) the old water-filling
        # policy can even lose to naive FIFO, because wide leases for the stragglers
        # waste CPU on Amdahl overhead; only (c) <= both baselines is asserted here.
        for n_short, n_long in ((10, 1), (100, 3)):
            wl = long_tail_workload(n_short, n_long, seed=3)
            for cores in (8, 32):
                self._compare(
                    cores, wl, f"{n_short} short + {n_long} long", b_beats_a=False
                )

    def test_critical_task_is_a_long_locus_for_the_baselines(self):
        wl = long_tail_workload(40, 2)
        a = simulate(wl, 16, Policy.STATIC_FIFO)
        self.assertTrue((a.critical_task_id or "").startswith("long"))

    def test_straggler_promotion_helps_in_the_sparse_regime(self):
        # few heavy loci, many idle cores: the queue drains below the idle-core count
        wl = [
            SimTask(
                f"loc{i}",
                60.0,
                serial_fraction=0.1,
                max_threads=8,
                n_seq=100,
                max_len=900,
            )
            for i in range(6)
        ]
        promoted = simulate(wl, 32, Policy.LOCUS_SCHEDULER)
        plain = simulate(wl, 32, Policy.LOCUS_SCHEDULER, promote=False)
        b = simulate(wl, 32, Policy.LPT_WATERFILL)
        emit(
            "6 heavy loci on 32 cores",
            [("c with promotion", promoted), ("c without promotion", plain), ("b", b)],
        )
        self.assertLess(promoted.makespan, plain.makespan)
        self.assertLessEqual(promoted.makespan, b.makespan * (1 + TOL))
        self.assertLessEqual(peak_busy(promoted), 32)

    def test_scales_to_thousands_of_tasks(self):
        wl = [
            SimTask(f"t{i}", 5.0 + (i % 17), max_threads=2, n_seq=100, max_len=800)
            for i in range(5000)
        ]
        results = [simulate(wl, 64, pol) for pol in Policy]
        self.assertLessEqual(
            results[2].makespan, results[0].makespan * (1 + TOL)
        )  # (c) not worse than naive FIFO on a flat workload
        for res in results:
            self.assertGreater(res.metrics.parallel_efficiency, 0.9)


if __name__ == "__main__":
    unittest.main()
