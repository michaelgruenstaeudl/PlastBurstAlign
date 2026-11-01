"""Tests for ``plastburstalign.estimate`` (pre-flight predictions)."""

from __future__ import annotations

import dataclasses
import io
import json
import unittest
from importlib import import_module
from unittest import mock

from rich.console import Console

from plastburstalign.autotune import TuningPlan, autotune
from plastburstalign.chunking import ChunkConfig
from plastburstalign.estimate import (
    DEFAULT_TIME_MODEL,
    LABEL_UNCALIBRATED,
    TimeModel,
    predicted_chunks,
    preflight_estimate,
    render_preflight,
)
from plastburstalign.hardware import HostProfile, MemoryInfo, StorageTier
from plastburstalign.scheduler_sim import Policy, SimTask, simulate
from plastburstalign.topology import CpuTopology

# Package attribute ``plastburstalign.estimate`` is the public API callable;
# the implementation module lives in ``sys.modules``.
est_mod = import_module("plastburstalign.estimate")

GIB = 1024**3
MIB = 1024**2


def make_plan(
    cores: int = 8,
    avail_gib: float = 12.0,
    shm_free: int | None = 8 * GIB,
    **kwargs: object,
) -> TuningPlan:
    storage = [
        StorageTier(
            "/tmp",
            "local",
            fstype="ext4",
            device_class="ssd",
            total_bytes=500 * GIB,
            free_bytes=300 * GIB,
            writable=True,
        )
    ]
    if shm_free is not None:
        storage.append(
            StorageTier(
                "/dev/shm",
                "ram",
                fstype="tmpfs",
                device_class="ram",
                total_bytes=max(shm_free, 1),
                free_bytes=shm_free,
                writable=True,
            )
        )
    avail = int(avail_gib * GIB)
    host = HostProfile(
        system="Linux",
        machine="x86_64",
        topology=CpuTopology.flat(cores),
        host_logical_cpus=cores,
        host_physical_cores=cores,
        affinity_cpus=tuple(range(cores)),
        memory=MemoryInfo(
            total_bytes=avail, available_bytes=avail, effective_bytes=avail
        ),
        storage=tuple(storage),
    )
    return autotune(host, **kwargs)  # type: ignore[arg-type]


def summary(
    n_loci: int, count: int, max_len: int, min_len: int | None = None
) -> dict[str, dict[str, int]]:
    return {
        f"g{i:03d}": {
            "count": count,
            "max_len": max_len,
            "min_len": max_len if min_len is None else min_len,
        }
        for i in range(n_loci)
    }


class TestModels(unittest.TestCase):
    def test_provisional_and_overridable(self) -> None:
        self.assertFalse(DEFAULT_TIME_MODEL.calibrated)
        self.assertIn("PROVISIONAL", DEFAULT_TIME_MODEL.provenance)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_TIME_MODEL.calibrated = True  # type: ignore[misc]
        again = TimeModel.from_dict(
            json.loads(json.dumps(DEFAULT_TIME_MODEL.to_dict()))
        )
        self.assertEqual(again, DEFAULT_TIME_MODEL)

    def test_align_seconds_is_continuous_and_monotone(self) -> None:
        tm = DEFAULT_TIME_MODEL
        b1, b2, b3 = 3.0e10, 7.0e10, 1.5e11
        for edge, tier in ((b1, 1), (b2, 2), (b3, 3)):
            below = tm.align_seconds(edge, tier - 1)
            above = tm.align_seconds(edge, tier)
            self.assertAlmostEqual(below, above, places=6)
        grid = [1e6 * 10 ** (k / 4) for k in range(0, 4 * 7)]
        times = []
        for w in grid:
            tier = (w >= b1) + (w >= b2) + (w >= b3)
            times.append(tm.align_seconds(w, tier))
        self.assertEqual(times, sorted(times))
        self.assertEqual(len(set(times)), len(times))
        self.assertAlmostEqual(
            tm.align_seconds(1e9, 0, alphabet="prot"), 1.5 * tm.align_seconds(1e9, 0)
        )
        self.assertEqual(tm.align_seconds(0, 0), 0.0)

    def test_predicted_chunks_mirrors_chunking_criteria(self) -> None:
        self.assertEqual(predicted_chunks(100, 4999), 0)
        self.assertEqual(predicted_chunks(100, 5000), 2)
        self.assertEqual(predicted_chunks(100, 12000), 5)
        self.assertEqual(predicted_chunks(1, 12000), 0)  # too few sequences
        self.assertEqual(predicted_chunks(100, 12000, subdivide=False), 0)
        adj = ChunkConfig(adjust_direction=True)
        self.assertEqual(predicted_chunks(100, 12000, "nucl", adj), 0)
        self.assertEqual(predicted_chunks(100, 1699, "prot"), 0)
        self.assertEqual(predicted_chunks(100, 4000, "prot"), 5)


class TestSanity(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = make_plan()

    def test_tiny_summary_takes_seconds(self) -> None:
        est = preflight_estimate(summary(20, 30, 1000, 800), mode="igs", plan=self.plan)
        self.assertEqual(est.tier_counts, {"ACCURATE": 20})
        self.assertLess(est.wall_seconds, 30.0)
        self.assertGreater(est.wall_seconds, 0.0)
        self.assertEqual((est.n_loci, est.ntax, est.n_chunked), (20, 30, 0))
        self.assertEqual(est.label, LABEL_UNCALIBRATED)
        self.assertFalse(est.calibrated)
        self.assertEqual(est.warnings, ())

    def test_13_7k_taxa_is_extreme_and_slow(self) -> None:
        big = preflight_estimate(
            summary(20, 13_700, 1500, 1400), mode="igs", plan=self.plan
        )
        tiny = preflight_estimate(summary(20, 30, 1000), mode="igs", plan=self.plan)
        self.assertEqual(big.tier_counts, {"EXTREME_SCALE": 20})
        self.assertGreater(big.wall_seconds, 600.0)
        self.assertGreater(big.wall_seconds, 50 * tiny.wall_seconds)
        self.assertIn("extreme_tier", big.warning_codes)
        self.assertEqual(big.ntax, 13_700)

    def test_wall_time_is_monotone_in_n_and_l(self) -> None:
        # 16 identical loci on 8 cores = two full waves: no scheduling anomalies
        def wall(n: int, length: int, **kw: object) -> float:
            return preflight_estimate(
                summary(16, n, length),
                mode="igs",
                plan=self.plan,
                **kw,  # type: ignore[arg-type]
            ).wall_seconds

        by_n = [wall(n, 1500) for n in (5, 20, 80, 320, 1280, 5000, 13_700)]
        self.assertEqual(by_n, sorted(by_n))
        self.assertEqual(len(set(by_n)), len(by_n))
        by_l = [
            wall(100, L, subdivide=False)
            for L in (100, 300, 900, 2000, 4000, 8000, 16000)
        ]
        self.assertEqual(by_l, sorted(by_l))
        self.assertEqual(len(set(by_l)), len(by_l))
        below_chunking = [wall(100, L) for L in (100, 500, 1500, 3000, 4999)]
        self.assertEqual(below_chunking, sorted(below_chunking))
        # on one core every term is a plain sum: strictly monotone, tier edges included
        one = make_plan(cores=1)
        seq = [
            preflight_estimate(summary(3, n, 2500), mode="igs", plan=one).wall_seconds
            for n in (10, 100, 500, 1000, 2000, 4000, 8000)
        ]
        self.assertEqual(seq, sorted(seq))

    def test_output_is_deterministic(self) -> None:
        s = summary(40, 300, 6000)
        a = preflight_estimate(s, mode="cds", plan=self.plan).to_dict(include_loci=True)
        b = preflight_estimate(
            dict(reversed(list(s.items()))), mode="cds", plan=self.plan
        )
        self.assertEqual(a, b.to_dict(include_loci=True))
        self.assertEqual(
            json.dumps(a, sort_keys=True),
            json.dumps(b.to_dict(include_loci=True), sort_keys=True),
        )

    def test_chunk_fanout_is_predicted_and_helps(self) -> None:
        s = {"long": {"count": 300, "max_len": 12_000, "min_len": 11_000}}
        on = preflight_estimate(s, mode="igs", plan=self.plan, subdivide=True)
        off = preflight_estimate(s, mode="igs", plan=self.plan, subdivide=False)
        self.assertEqual((on.n_chunked, on.loci[0].chunks), (1, 5))
        self.assertEqual((off.n_chunked, off.loci[0].chunks), (0, 0))
        self.assertLess(on.wall_seconds, 0.5 * off.wall_seconds)
        # chunk plan + splice are charged: total CPU is not below the sum of chunks
        self.assertGreater(on.loci[0].single_core_seconds, 0.0)

    def test_cds_aligns_proteins(self) -> None:
        s = summary(4, 100, 3000, 2900)
        est = preflight_estimate(s, mode="cds", plan=self.plan)
        self.assertEqual(est.loci[0].alphabet, "prot")
        self.assertEqual(est.loci[0].align_len, 1000)
        exact = preflight_estimate(
            s,
            mode="cds",
            plan=self.plan,
            prot_summary={k: {"count": 100, "max_len": 950, "min_len": 900} for k in s},
        )
        self.assertEqual(exact.loci[0].align_len, 950)
        nucl = preflight_estimate(s, mode="igs", plan=self.plan)
        self.assertEqual(nucl.loci[0].alphabet, "nucl")
        self.assertGreater(
            est.disk.unaligned_fasta, nucl.disk.unaligned_fasta
        )  # + protein

    def test_input_validation_and_empty(self) -> None:
        with self.assertRaises(ValueError):
            preflight_estimate({}, mode="cds+igs", plan=self.plan)
        with self.assertRaises(ValueError):
            preflight_estimate({}, mode="", plan=self.plan)
        empty = preflight_estimate(
            {"a": {"count": 0, "max_len": 0}, "b": {"count": 1, "max_len": 500}},
            mode="int",
            plan=self.plan,
        )
        self.assertEqual((empty.n_loci, empty.n_skipped), (1, 1))
        none = preflight_estimate({}, mode="int", plan=self.plan)
        self.assertEqual(
            (none.n_loci, none.wall_seconds), (0, DEFAULT_TIME_MODEL.startup_s)
        )
        self.assertEqual(none.heaviest, ())

    def test_constants_are_overridable(self) -> None:
        s = summary(16, 200, 1500)
        base = preflight_estimate(s, mode="igs", plan=self.plan)
        slow = dataclasses.replace(
            DEFAULT_TIME_MODEL,
            sec_per_work=tuple(10 * x for x in DEFAULT_TIME_MODEL.sec_per_work),  # type: ignore[arg-type]
            calibrated=True,
        )
        got = preflight_estimate(s, mode="igs", plan=self.plan, constants=slow)
        self.assertGreater(got.align_seconds, 5 * base.align_seconds)
        # the memory model is still provisional: the label must stay honest
        self.assertEqual(got.label, LABEL_UNCALIBRATED)


class TestSimulatorAndFootprint(unittest.TestCase):
    def test_wall_time_comes_from_the_scheduler_simulator(self) -> None:
        plan = make_plan(cores=8)
        s = {
            "a": {"count": 200, "max_len": 12_000, "min_len": 11_000},
            "b": {"count": 50, "max_len": 1000, "min_len": 900},
        }
        with mock.patch.object(est_mod, "simulate", wraps=simulate) as spy:
            est = preflight_estimate(s, mode="igs", plan=plan)
        self.assertEqual(spy.call_count, 1)
        args, kwargs = spy.call_args
        tasks, cores, policy = args
        self.assertEqual((cores, policy), (8, Policy.LOCUS_SCHEDULER))
        self.assertEqual([t.task_id for t in tasks], ["a", "b"])
        self.assertTrue(all(isinstance(t, SimTask) for t in tasks))
        self.assertEqual(len(tasks[0].chunks), 5)
        self.assertGreater(tasks[0].plan_work, 0.0)
        self.assertGreater(tasks[0].finalize_work, 0.0)
        self.assertEqual(tasks[1].chunks, ())
        self.assertTrue(kwargs["use_chunks"])
        direct = simulate(tasks, 8, Policy.LOCUS_SCHEDULER)
        self.assertAlmostEqual(est.align_seconds, direct.makespan)
        self.assertAlmostEqual(
            est.parallel_efficiency, direct.metrics.parallel_efficiency
        )
        self.assertAlmostEqual(est.straggler_tax, direct.metrics.straggler_tax)
        self.assertEqual(est.critical_locus, direct.critical_task_id)
        self.assertGreater(est.wall_seconds, est.align_seconds)  # + start-up + concat

    def test_memory_limited_workers_lengthen_the_prediction(self) -> None:
        plan = make_plan(cores=8)
        limited = dataclasses.replace(plan, workers=2)
        s = summary(16, 100, 1500)
        full = preflight_estimate(s, mode="igs", plan=plan)
        slow = preflight_estimate(s, mode="igs", plan=limited)
        self.assertGreater(slow.wall_seconds, 2 * full.align_seconds)
        self.assertIn("workers_limited", slow.warning_codes)
        self.assertNotIn("workers_limited", full.warning_codes)

    def test_disk_and_memory_follow_n_times_l(self) -> None:
        plan = make_plan()
        s = summary(10, 100, 1000)  # no spread: every sequence is 1000 nt
        est = preflight_estimate(s, mode="igs", plan=plan)
        hdr = 40
        self.assertEqual(est.disk.unaligned_fasta, 10 * 100 * (1000 + hdr))
        exp = plan.memory_model.alignment_expansion
        self.assertEqual(est.disk.aligned_fasta, 10 * 100 * (int(1000 * exp) + hdr))
        self.assertGreater(est.disk.total, est.disk.unaligned_fasta)
        keep = preflight_estimate(
            s, mode="igs", plan=plan, keep_unaligned=True
        ).disk.total
        drop = preflight_estimate(
            s, mode="igs", plan=plan, keep_unaligned=False
        ).disk.total
        self.assertEqual(keep - drop, est.disk.unaligned_fasta)
        m = plan.memory_model
        self.assertEqual(
            est.peak_ram_bytes,
            est.max_mafft_bytes
            + plan.workers * est.parser_buffer_bytes
            + est.supermatrix_bytes,
        )
        self.assertGreaterEqual(est.peak_ram_concurrent_bytes, est.peak_ram_bytes)
        self.assertFalse(m.calibrated)
        self.assertEqual(est.envelope_bytes, plan.memory_envelope_bytes)
        self.assertAlmostEqual(
            est.ram_fraction, est.peak_ram_bytes / est.envelope_bytes
        )

    def test_scratch_tier_and_budget(self) -> None:
        ram = preflight_estimate(summary(8, 100, 1000), mode="igs", plan=make_plan())
        self.assertEqual(ram.scratch.tier, "ram")
        self.assertEqual(ram.scratch.path, "/dev/shm")
        self.assertTrue(ram.scratch.fits)
        self.assertEqual(ram.scratch.budget_bytes, make_plan().scratch_budget_bytes)
        self.assertGreater(ram.scratch.need_bytes, 0)
        disk = preflight_estimate(
            summary(8, 100, 1000), mode="igs", plan=make_plan(shm_free=None)
        )
        self.assertEqual(disk.scratch.tier, "disk")
        self.assertEqual(disk.scratch.path, "/tmp")
        self.assertNotIn("tiny_ramdisk", disk.warning_codes)


class TestWarnings(unittest.TestCase):
    def test_tiny_ramdisk_warns(self) -> None:
        plan = make_plan(shm_free=64 * MIB)  # the classic 64 MB /dev/shm
        est = preflight_estimate(summary(8, 100, 1000), mode="igs", plan=plan)
        self.assertIn("tiny_ramdisk", est.warning_codes)
        self.assertEqual(est.scratch.tier, "disk")
        msg = next(w.message for w in est.warnings if w.code == "tiny_ramdisk")
        self.assertIn("/dev/shm", msg)
        self.assertIn("64.0 MiB", msg)

    def test_ramdisk_budget_too_small_for_inflight_loci(self) -> None:
        plan = make_plan(shm_free=600 * MIB)  # usable but small
        s = summary(16, 4000, 3000, 2500)
        est = preflight_estimate(s, mode="igs", plan=plan)
        self.assertEqual(est.scratch.tier, "ram")
        self.assertFalse(est.scratch.fits)
        self.assertIn("ramdisk_budget_low", est.warning_codes)

    def test_oversized_workload_exceeds_the_envelope(self) -> None:
        plan = make_plan(avail_gib=1.0)  # ~870 MiB envelope
        est = preflight_estimate(summary(80, 13_700, 1500, 1400), mode="igs", plan=plan)
        self.assertGreater(est.peak_ram_bytes, est.envelope_bytes)
        self.assertIn("ram_over_envelope", est.warning_codes)
        self.assertIn("supermatrix_memmap", est.warning_codes)

    def test_ram_high_threshold(self) -> None:
        plan = make_plan(avail_gib=12.0)
        s = summary(8, 3000, 3000, 2500)
        probe = preflight_estimate(s, mode="igs", plan=plan)
        assert 0 < probe.ram_fraction < 1
        tight = preflight_estimate(
            s, mode="igs", plan=plan, ram_warn_fraction=probe.ram_fraction / 2
        )
        self.assertIn("ram_high", tight.warning_codes)
        self.assertNotIn("ram_over_envelope", tight.warning_codes)
        self.assertNotIn("ram_high", probe.warning_codes)

    def test_oversize_locus_with_forced_accurate_tier(self) -> None:
        plan = make_plan(avail_gib=1.0)
        est = preflight_estimate(
            {"huge": {"count": 13_000, "max_len": 1500, "min_len": 1400}},
            mode="igs",
            plan=plan,
            strategy_overrides={"tier": "accurate"},
        )
        self.assertEqual(est.loci[0].tier, "ACCURATE")
        self.assertIn("oversize_locus", est.warning_codes)

    def test_long_wall_and_disk_space(self) -> None:
        plan = make_plan()
        s = summary(80, 13_700, 1500, 1400)
        est = preflight_estimate(
            s, mode="igs", plan=plan, long_wall_seconds=600.0, disk_free_bytes=GIB
        )
        self.assertIn("wall_long", est.warning_codes)
        self.assertIn("disk_space", est.warning_codes)
        quick = preflight_estimate(summary(4, 20, 800), mode="igs", plan=plan)
        self.assertNotIn("wall_long", quick.warning_codes)

    def test_host_plan_warnings_are_passed_through(self) -> None:
        plan = make_plan(num_threads=1000)  # clamped with an autotune warning
        self.assertTrue(plan.warnings)
        est = preflight_estimate(summary(2, 10, 500), mode="igs", plan=plan)
        self.assertIn("host_plan", est.warning_codes)


class TestRendering(unittest.TestCase):
    def test_text_dict_and_rich_rendering(self) -> None:
        plan = make_plan(shm_free=64 * MIB)
        est = preflight_estimate(
            {**summary(6, 400, 6000), "x": {"count": 13_700, "max_len": 1500}},
            mode="cds",
            plan=plan,
            long_wall_seconds=1.0,
        )
        text = est.to_text()
        self.assertIn("estimate (model not yet calibrated)", text)
        self.assertIn("Predicted wall time", text)
        self.assertIn("[tiny_ramdisk]", text)
        payload = json.loads(json.dumps(est.to_dict(include_loci=True)))
        self.assertEqual(payload["label"], LABEL_UNCALIBRATED)
        self.assertEqual(payload["n_loci"], 7)
        self.assertEqual(len(payload["loci"]), 7)
        self.assertNotIn("loci", est.to_dict())
        console = Console(
            file=io.StringIO(), record=True, force_terminal=False, width=110
        )
        render_preflight(est, console)
        out = console.export_text()
        self.assertIn("Pre-flight estimate", out)
        self.assertIn("estimate (model not yet calibrated)", out)
        self.assertIn("Heaviest loci", out)
        self.assertIn("tiny_ramdisk", out)
        # an empty estimate renders too
        render_preflight(preflight_estimate({}, mode="igs", plan=plan), console)


if __name__ == "__main__":
    unittest.main()
