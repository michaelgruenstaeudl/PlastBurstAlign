"""Tests for ``plastburstalign.scaling`` (math on hand-built scheduler reports)."""

from __future__ import annotations

import io
import json
import unittest
from types import SimpleNamespace

from rich.console import Console

from plastburstalign.scaling import (
    DataDensity,
    ScalingReport,
    format_duration,
    karp_flatt,
    summarize_supermatrix,
)
from plastburstalign.scheduler import SchedulerReport, TaskRecord


def rec(
    task_id: str,
    start: float,
    end: float,
    threads: int = 1,
    *,
    kind: str = "task",
    parent: str | None = None,
    root: str | None = None,
    order: tuple[int, ...] = (0,),
    ok: bool = True,
    scratch_tier: str | None = None,
) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        parent_id=parent,
        root_id=root or task_id,
        kind=kind,
        order=order,
        work=1.0,
        est_mem_bytes=0,
        threads=threads,
        cpu_ids=tuple(range(threads)),
        ready=0.0,
        start=start,
        end=end,
        ok=ok,
        status="ok" if ok else "failed",
        error=None if ok else "boom",
        scratch_tier=scratch_tier,
    )


def make_report(
    records: list[TaskRecord],
    cores: int,
    *,
    ended_at: float | None = None,
    failed: tuple[str, ...] = (),
    **kwargs: object,
) -> SchedulerReport:
    roots = list(dict.fromkeys(r.root_id for r in records))
    end = (
        ended_at if ended_at is not None else max((r.end for r in records), default=0.0)
    )
    return SchedulerReport(
        records=records,
        total_cores=cores,
        started_at=0.0,
        ended_at=end,
        loci={r: (r not in failed, None) for r in roots},
        **kwargs,  # type: ignore[arg-type]
    )


def straggler_report() -> SchedulerReport:
    """4 cores: two full waves of 1-s tasks, then one 10-s straggler alone."""
    recs = [
        rec(f"s{i}", float(wave), wave + 1.0, order=(wave * 4 + i,))
        for wave in (0, 1)
        for i in range(4)
    ]
    recs.append(rec("long", 2.0, 12.0, order=(99,)))
    return make_report(recs, 4)


class TestMath(unittest.TestCase):
    def test_full_machine_has_unit_efficiency_and_zero_serial_fraction(self) -> None:
        rep = make_report([rec("a", 0, 10, 4)], 4)
        sr = ScalingReport.from_scheduler(rep, cores=4)
        self.assertAlmostEqual(sr.wall_seconds, 10.0)
        self.assertAlmostEqual(sr.cpu_seconds, 40.0)
        self.assertAlmostEqual(sr.parallel_efficiency, 1.0)
        self.assertAlmostEqual(sr.speedup or 0.0, 4.0)
        self.assertAlmostEqual(sr.karp_flatt or 0.0, 0.0)
        self.assertAlmostEqual(sr.idle_core_seconds, 0.0)
        self.assertTrue(sr.diagnosis.startswith("Well balanced"))

    def test_known_karp_flatt_case(self) -> None:
        # 4 one-thread tasks for 10 s on 8 cores: S = 4, E = 0.5, e = 1/7
        recs = [rec(f"t{i}", 0, 10, order=(i,)) for i in range(4)]
        sr = ScalingReport.from_scheduler(make_report(recs, 8), cores=8)
        self.assertAlmostEqual(sr.parallel_efficiency, 0.5)
        self.assertAlmostEqual(sr.speedup or 0.0, 4.0)
        self.assertAlmostEqual(sr.karp_flatt or 0.0, 1.0 / 7.0)
        self.assertAlmostEqual(sr.idle_core_seconds, 40.0)
        self.assertAlmostEqual(karp_flatt(4.0, 8), 1.0 / 7.0)

    def test_karp_flatt_guards(self) -> None:
        self.assertIsNone(karp_flatt(3.0, 1))  # p = 1: 0/0
        self.assertIsNone(karp_flatt(None, 8))
        self.assertIsNone(karp_flatt(0.0, 8))
        self.assertIsNone(karp_flatt(float("nan"), 8))
        self.assertEqual(karp_flatt(9.0, 8), 0.0)  # super-linear: clamped
        self.assertEqual(karp_flatt(0.5, 2), 1.0)  # slower than serial: clamped

    def test_straggler_tax_is_the_schedulers_own_value(self) -> None:
        rep = straggler_report()
        sr = ScalingReport.from_scheduler(rep, cores=4)
        self.assertEqual(sr.straggler_tax, rep.straggler_tax)
        self.assertAlmostEqual(sr.straggler_tax, 10.0 / 12.0)
        self.assertEqual(sr.parallel_efficiency, rep.parallel_efficiency)
        self.assertAlmostEqual(sr.idle_core_seconds, 4 * 12 - 18.0)
        # different core budget: recomputed with the same function
        half = ScalingReport.from_scheduler(rep, cores=8)
        self.assertAlmostEqual(half.parallel_efficiency, 18.0 / (8 * 12))

    def test_critical_locus_top_loci_and_diagnosis(self) -> None:
        sr = ScalingReport.from_scheduler(straggler_report(), cores=4)
        self.assertEqual(sr.critical_locus, "long")
        self.assertAlmostEqual(sr.critical_seconds, 10.0)
        self.assertAlmostEqual(sr.critical_share, 10.0 / 12.0)
        self.assertEqual(sr.top_loci[0].name, "long")
        self.assertEqual(len(sr.top_loci), 5)
        self.assertEqual([t.name for t in sr.top_loci[1:]], ["s0", "s1", "s2", "s3"])
        self.assertTrue(
            sr.diagnosis.startswith("Straggler-bound: locus long"), sr.diagnosis
        )
        self.assertIn("3 of 4 cores idle", sr.diagnosis)
        self.assertIn("subdivide", sr.diagnosis)

    def test_chunked_straggler_gets_chunk_advice(self) -> None:
        recs = [
            rec("big", 0, 1, order=(0,)),
            rec(
                "big/c0",
                1,
                11,
                kind="subtask",
                parent="big",
                root="big",
                order=(0, 1, 0),
            ),
            rec(
                "big/c1",
                1,
                5,
                kind="subtask",
                parent="big",
                root="big",
                order=(0, 1, 1),
            ),
            rec(
                "big::finalize",
                11,
                12,
                kind="finalize",
                parent="big",
                root="big",
                order=(0, 2, 0),
            ),
            rec("x", 0, 11, 2, order=(1,)),
        ]
        sr = ScalingReport.from_scheduler(make_report(recs, 4), cores=4)
        big = next(t for t in sr.top_loci if t.name == "big")
        self.assertTrue(big.chunked)
        self.assertEqual(big.n_units, 4)
        self.assertEqual(sr.n_chunked, 1)
        self.assertEqual(sr.critical_locus, "big")
        self.assertEqual(sr.critical_path, ("big", "big/c0", "big::finalize"))

    def test_single_core_run(self) -> None:
        rep = make_report([rec("a", 0, 3), rec("b", 3, 5, order=(1,))], 1)
        sr = ScalingReport.from_scheduler(rep, cores=1)
        self.assertIsNone(sr.karp_flatt)
        self.assertAlmostEqual(sr.parallel_efficiency, 1.0)
        self.assertAlmostEqual(sr.speedup or 0.0, 1.0)
        self.assertEqual(sr.straggler_tax, 0.0)
        self.assertTrue(sr.diagnosis.startswith("Well balanced"))
        self.assertIn("n/a", sr.to_text())

    def test_empty_report(self) -> None:
        rep = SchedulerReport(records=[], total_cores=4, started_at=0.0, ended_at=0.0)
        sr = ScalingReport.from_scheduler(rep, cores=4)
        self.assertEqual((sr.wall_seconds, sr.cpu_seconds, sr.n_loci), (0.0, 0.0, 0))
        self.assertIsNone(sr.speedup)
        self.assertIsNone(sr.karp_flatt)
        self.assertEqual(sr.top_loci, ())
        self.assertIsNone(sr.critical_locus)
        self.assertEqual(sr.diagnosis, "No alignment tasks were executed.")
        json.dumps(sr.to_dict())
        self.assertIn("scaling report", sr.to_text())
        sr.render(Console(file=io.StringIO(), width=80))

    def test_low_efficiency_diagnoses(self) -> None:
        # 4 one-thread loci on 8 cores: exactly 50 % busy (no tail), E = 0.5
        four = [rec(f"a{i}", 0, 10, order=(i,)) for i in range(4)]
        few = ScalingReport.from_scheduler(make_report(four, 8), cores=8)
        self.assertEqual(few.straggler_tax, 0.0)
        self.assertTrue(
            few.diagnosis.startswith("Under-utilised: only 4 loci"), few.diagnosis
        )
        # 128 half-second loci, only 16 running at a time on 64 cores (8 waves)
        waves = [
            rec(f"v{i}", 0.5 * (i // 16), 0.5 * (i // 16) + 0.5, order=(i,))
            for i in range(128)
        ]
        short = ScalingReport.from_scheduler(make_report(waves, 64), cores=64)
        self.assertAlmostEqual(short.parallel_efficiency, 0.25)
        self.assertTrue(short.diagnosis.startswith("Overhead-bound"), short.diagnosis)

    def test_failures_cancel_and_memory(self) -> None:
        recs = [rec("a", 0, 4, 2), rec("b", 0, 4, 2, order=(1,), ok=False)]
        sr = ScalingReport.from_scheduler(make_report(recs, 4, failed=("b",)), cores=4)
        self.assertEqual((sr.n_ok, sr.n_failed), (1, 1))
        self.assertEqual(sr.failed_loci, ("b",))
        self.assertTrue(any("failed: b" in n for n in sr.notes))
        cancelled = ScalingReport.from_scheduler(
            make_report(recs, 4, cancelled=True), cores=4
        )
        self.assertTrue(cancelled.diagnosis.startswith("Run was cancelled"))
        oversize = ScalingReport.from_scheduler(
            make_report(
                recs, 4, oversize_task_ids=("a",), peak_memory_bytes=3 * 1024**3
            ),
            cores=4,
        )
        self.assertTrue(
            oversize.diagnosis.startswith("Memory-bound"), oversize.diagnosis
        )
        self.assertIn("3.0 GiB", oversize.diagnosis)

    def test_scratch_usage_from_engine_stats_and_records(self) -> None:
        recs = [
            rec("a", 0, 1, scratch_tier="ramdisk"),
            rec("b", 0, 1, order=(1,), scratch_tier="disk"),
            rec("c", 1, 2, order=(2,), scratch_tier="ramdisk"),
        ]
        rep = make_report(recs, 2)
        from_records = ScalingReport.from_scheduler(rep, cores=2).scratch
        self.assertEqual(
            (
                from_records.source,
                from_records.ramdisk_tickets,
                from_records.disk_tickets,
            ),
            ("records", 2, 1),
        )
        stats = {
            "ramdisk_tickets": 7,
            "disk_tickets": 3,
            "fallback_count": 2,
            "ramdisk_peak_bytes": 5 * 1024**2,
            "peak_bytes_reserved": 9 * 1024**2,
            "ramdisk_budget_bytes": 64 * 1024**2,
        }
        sr = ScalingReport.from_scheduler(rep, cores=2, scratch_stats=stats)
        self.assertEqual(sr.scratch.source, "engine")
        self.assertEqual((sr.scratch.ramdisk_tickets, sr.scratch.disk_tickets), (7, 3))
        self.assertAlmostEqual(sr.scratch.ramdisk_fraction or 0.0, 0.7)
        self.assertEqual(sr.scratch.peak_bytes, 9 * 1024**2)
        self.assertTrue(any("fell back" in n for n in sr.notes))
        self.assertIn("7 RAM-disk / 3 disk", sr.to_text())

    def test_data_density_from_summary(self) -> None:
        summary = {
            "ntax": 10,
            "nchar": 400,
            "n_loci": 4,
            "informative_sites": 80,
            "density": 0.7,
            "gap_fraction": 0.2,
            "missing_fraction": 0.1,
        }
        sr = ScalingReport.from_scheduler(
            make_report([rec("a", 0, 1)], 1), cores=1, supermatrix_summary=summary
        )
        assert sr.data is not None
        self.assertAlmostEqual(sr.data.non_gap_fraction or 0.0, 0.7)
        self.assertAlmostEqual(sr.data.missing_fraction or 0.0, 0.1)
        self.assertAlmostEqual(sr.data.informative_per_locus or 0.0, 20.0)
        self.assertIn("70.0% data, 10.0% missing, 20.0% gaps", sr.to_text())
        derived = DataDensity.from_summary(
            {"gap_fraction": 0.25, "missing_fraction": 0.25}
        )
        self.assertAlmostEqual(derived.non_gap_fraction or 0.0, 0.5)
        self.assertIsNone(
            ScalingReport.from_scheduler(make_report([rec("a", 0, 1)], 1), cores=1).data
        )

    def test_summarize_supermatrix_duck_typing(self) -> None:
        def stat(nchar: int, inf: int, gap: float, miss: float) -> SimpleNamespace:
            return SimpleNamespace(
                nchar_kept=nchar,
                informative=inf,
                gap_fraction=gap,
                missing_fraction=miss,
            )

        sm = SimpleNamespace(
            ntax=5,
            nchar=300,
            stats=[stat(100, 10, 0.1, 0.0), stat(200, 40, 0.4, 0.3)],
            failed_loci={"z": "bad"},
            informative_sites=50,
            variable_sites=90,
            density=0.5,
        )
        out = summarize_supermatrix(sm)
        self.assertEqual(
            (out["ntax"], out["nchar"], out["n_loci"], out["n_failed_loci"]),
            (5, 300, 2, 1),
        )
        self.assertAlmostEqual(out["gap_fraction"], (0.1 * 100 + 0.4 * 200) / 300)
        self.assertAlmostEqual(out["missing_fraction"], 0.3 * 200 / 300)
        self.assertEqual(out["informative_median"], 25.0)
        self.assertEqual(summarize_supermatrix(SimpleNamespace())["n_loci"], 0)

    def test_serialisation_and_rendering(self) -> None:
        sr = ScalingReport.from_scheduler(
            straggler_report(),
            cores=4,
            supermatrix_summary={"ntax": 3, "nchar": 9, "n_loci": 2, "density": 1.0},
        )
        payload = json.loads(json.dumps(sr.to_dict()))
        self.assertEqual(payload["critical_locus"], "long")
        self.assertEqual(payload["top_loci"][0]["name"], "long")
        self.assertIn("Diagnosis: Straggler-bound", sr.to_text())
        console = Console(
            file=io.StringIO(), record=True, force_terminal=False, width=100
        )
        sr.render(console)
        text = console.export_text()
        self.assertIn("Scaling report", text)
        self.assertIn("Longest loci", text)
        self.assertIn("Straggler-bound", text)

    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(0.42), "0.42 s")
        self.assertEqual(format_duration(12.34), "12.3 s")
        self.assertEqual(format_duration(245), "4m05s")
        self.assertEqual(format_duration(3720), "1h02m")
        self.assertEqual(format_duration(2 * 86400 + 3 * 3600), "2d03h")
        self.assertEqual(format_duration(None), "n/a")
        self.assertEqual(format_duration(float("inf")), "n/a")


if __name__ == "__main__":
    unittest.main()
