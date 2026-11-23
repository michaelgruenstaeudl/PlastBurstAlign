"""Tests for ``plastburstalign.autotune`` (plans built from fixture hosts)."""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import hostfs_builder as hb

from plastburstalign.hardware import HostProfile, MemoryInfo, detect_host
from plastburstalign.topology import CpuTopology

# Package attribute ``plastburstalign.autotune`` is the public API callable;
# the implementation module lives in ``sys.modules``.
at = import_module("plastburstalign.autotune")

GIB = 1024**3
MIB = 1024**2


def _du(table: dict[str, tuple[int, int]] | None = None):
    table = table or {}

    def usage(path: str) -> tuple[int, int, int]:
        total, free = table.get(path, (500 * GIB, 400 * GIB))
        return total, total - free, free

    return usage


class _HostCase(unittest.TestCase):
    def make_host(
        self,
        *,
        packages: int = 1,
        cores: int = 6,
        threads: int = 2,
        env: dict[str, str] | None = None,
        du: dict[str, tuple[int, int]] | None = None,
        mem_total_gb: int = 16,
        mem_avail_gb: int = 8,
        mountinfo: str = "",
        block_devices: dict[str, dict[str, str]] | None = None,
        cgroup: dict[str, str] | None = None,
        allowed: str | None = None,
    ) -> HostProfile:
        root = Path(tempfile.mkdtemp(prefix="pba_at_"))
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        hb.build_host_tree(
            root,
            packages=packages,
            cores_per_package=cores,
            threads_per_core=threads,
            mem_total_kb=mem_total_gb * 1024 * 1024,
            mem_avail_kb=mem_avail_gb * 1024 * 1024,
            mountinfo=mountinfo,
            block_devices=block_devices,
            allowed_list=allowed,
        )
        if cgroup is not None:
            hb.cgroup_v2(root, **cgroup)  # type: ignore[arg-type]
        return detect_host(
            root, env or {}, system="Linux", machine="x86_64", disk_usage=_du(du)
        )

    def epyc(self, **kwargs: object) -> HostProfile:
        return self.make_host(
            packages=2,
            cores=32,
            mem_total_gb=512,
            mem_avail_gb=400,
            mountinfo=hb.MOUNTINFO_EPYC,
            block_devices=hb.EPYC_BLOCK,
            du={"/dev/shm": (256 * GIB, 256 * GIB)},
            **kwargs,  # type: ignore[arg-type]
        )


class TestProvisionalConstants(unittest.TestCase):
    def test_everything_is_marked_provisional(self) -> None:
        self.assertFalse(at.DEFAULT_MEMORY_MODEL.calibrated)
        self.assertIn("PROVISIONAL", at.DEFAULT_MEMORY_MODEL.provenance)
        self.assertIs(at.DEFAULT_USE_SMT, False)  # physical-only until calibrated
        self.assertAlmostEqual(at.ENVELOPE_FRACTION, 0.85)

    def test_model_is_immutable(self) -> None:
        with self.assertRaises(dataclasses.FrozenInstanceError):
            at.DEFAULT_MEMORY_MODEL.calibrated = True  # type: ignore[misc]


class TestCpuBudget(_HostCase):
    def test_auto_is_physical_cores_on_epyc(self) -> None:
        plan = at.autotune(self.epyc())
        self.assertEqual(plan.cpu_budget, 64)
        self.assertEqual(plan.workers, 64)
        self.assertFalse(plan.use_smt)
        self.assertEqual(plan.topology.physical_count, 64)
        self.assertEqual(plan.topology.logical_count, 64)  # siblings excluded
        self.assertEqual(plan.topology.cores[0].cpu_ids, (0,))
        # two NUMA nodes of 32 cores: a task never spans nodes
        self.assertEqual(plan.threads_per_task_cap, 32)
        self.assertTrue(plan.pinning)
        self.assertEqual(plan.warnings, ())

    def test_smt_true_uses_all_logical_cpus(self) -> None:
        plan = at.autotune(self.epyc(), smt=True)
        self.assertTrue(plan.use_smt)
        self.assertEqual(plan.cpu_budget, 128)
        self.assertEqual(plan.topology.logical_count, 128)
        self.assertEqual(plan.topology.cores[0].cpu_ids, (0, 64))
        self.assertEqual(plan.threads_per_task_cap, 64)

    def test_explicit_int_is_clamped_with_warning(self) -> None:
        host = self.epyc()
        plan = at.autotune(host, num_threads=10_000)
        self.assertEqual(plan.cpu_budget, 128)
        self.assertTrue(any("clamped to 128" in w for w in plan.warnings))
        self.assertTrue(plan.use_smt)

    def test_explicit_int_within_physical_is_honored(self) -> None:
        plan = at.autotune(self.epyc(), num_threads=8)
        self.assertEqual((plan.cpu_budget, plan.workers), (8, 8))
        self.assertEqual(plan.topology.physical_count, 8)
        self.assertEqual(plan.warnings, ())
        self.assertEqual(at.autotune(self.epyc(), num_threads="12").cpu_budget, 12)

    def test_int_above_physical_uses_smt_with_note(self) -> None:
        plan = at.autotune(self.epyc(), num_threads=100)
        self.assertEqual(plan.cpu_budget, 100)
        self.assertTrue(plan.use_smt)
        self.assertEqual(plan.topology.logical_count, 100)
        self.assertTrue(any("SMT siblings will be used" in w for w in plan.warnings))

    def test_smt_false_clamps_int_to_physical(self) -> None:
        plan = at.autotune(self.epyc(), num_threads=100, smt=False)
        self.assertEqual(plan.cpu_budget, 64)
        self.assertFalse(plan.use_smt)
        self.assertTrue(any("smt=False" in w for w in plan.warnings))

    def test_invalid_thread_values(self) -> None:
        host = self.make_host()
        self.assertEqual(at.autotune(host, num_threads=0).cpu_budget, 1)
        self.assertTrue(at.autotune(host, num_threads=-3).warnings)
        with self.assertRaises(ValueError):
            at.autotune(host, num_threads="lots")
        with self.assertRaises(ValueError):
            at.autotune(host, num_threads=True)  # type: ignore[arg-type]

    def test_smt_requested_without_siblings_warns(self) -> None:
        host = self.make_host(cores=4, threads=1)
        plan = at.autotune(host, smt=True)
        self.assertFalse(plan.use_smt)
        self.assertEqual(plan.cpu_budget, 4)
        self.assertTrue(any("no SMT siblings" in w for w in plan.warnings))

    def test_laptop_single_node(self) -> None:
        plan = at.autotune(self.make_host())
        self.assertEqual((plan.cpu_budget, plan.threads_per_task_cap), (6, 6))
        self.assertEqual(plan.topology.numa_nodes, (0,))
        self.assertFalse(plan.pinning)  # single node, not a dedicated allocation

    def test_container_quota_gives_two_workers(self) -> None:
        host = self.make_host(cores=4, cgroup={"cpu_max": "250000 100000"})
        plan = at.autotune(host)
        self.assertEqual(plan.cpu_budget, 2)
        self.assertFalse(plan.pinning)

    def test_dedicated_slurm_allocation_pins_around_idle_smt(self) -> None:
        host = self.make_host(env={"SLURM_JOB_ID": "5"})
        self.assertTrue(at.autotune(host).pinning)

    def test_physical_first_topology_when_budget_below_logical(self) -> None:
        plan = at.autotune(self.make_host(cores=4, threads=2), num_threads=6)
        # 4 primary threads first, then 2 siblings: 6 cpus over 4 cores
        self.assertEqual(plan.topology.logical_count, 6)
        self.assertEqual(plan.topology.physical_count, 4)
        self.assertEqual(
            sorted(len(c.cpu_ids) for c in plan.topology.cores), [1, 1, 2, 2]
        )


class TestMemoryEnvelope(_HostCase):
    def test_envelope_is_085_of_min(self) -> None:
        host = self.make_host(mem_avail_gb=8)
        plan = at.autotune(host)
        self.assertEqual(plan.memory_envelope_bytes, int(8 * GIB * 0.85))

    def test_cgroup_headroom_binds(self) -> None:
        host = self.make_host(
            mem_avail_gb=8,
            cgroup={"memory_max": str(2 * GIB), "memory_current": str(GIB // 2)},
        )
        plan = at.autotune(host)
        self.assertEqual(plan.memory_envelope_bytes, int(1.5 * GIB * 0.85))
        self.assertIn("binding", plan.explain())

    def test_user_limit_lowers_and_warns_when_too_high(self) -> None:
        host = self.make_host(mem_avail_gb=8)
        plan = at.autotune(host, memory_limit="2G")
        self.assertEqual(plan.memory_envelope_bytes, int(2 * GIB * 0.85))
        plan = at.autotune(host, memory_limit=64 * GIB)
        self.assertEqual(plan.memory_envelope_bytes, int(8 * GIB * 0.85))
        self.assertTrue(any("exceeds" in w for w in plan.warnings))
        with self.assertRaises(ValueError):
            at.autotune(host, memory_limit="0")
        with self.assertRaises(ValueError):
            at.autotune(host, memory_limit="plenty")

    def test_unknown_memory_uses_fallback_with_warning(self) -> None:
        host = dataclasses.replace(self.make_host(), memory=MemoryInfo())
        plan = at.autotune(host)
        self.assertEqual(
            plan.memory_envelope_bytes, int(at.UNKNOWN_MEMORY_FALLBACK_BYTES * 0.85)
        )
        self.assertTrue(any("could not be determined" in w for w in plan.warnings))
        with_limit = at.autotune(host, memory_limit="4G")
        self.assertEqual(with_limit.memory_envelope_bytes, int(4 * GIB * 0.85))
        self.assertEqual(with_limit.warnings, ())


class TestMemoryModel(unittest.TestCase):
    def test_estimates_grow_with_size(self) -> None:
        small = at.estimate_mafft_peak_bytes(100, 1000)
        self.assertGreater(at.estimate_mafft_peak_bytes(200, 1000), small)
        self.assertGreater(at.estimate_mafft_peak_bytes(100, 2000), small)
        self.assertGreater(small, at.DEFAULT_MEMORY_MODEL.mafft_base_bytes)

    def test_parttree_drops_quadratic_distance_matrix(self) -> None:
        n, length = 20_000, 500
        full = at.estimate_mafft_peak_bytes(n, length, parttree=False)
        part = at.estimate_mafft_peak_bytes(n, length, parttree=True)
        model = at.DEFAULT_MEMORY_MODEL
        self.assertGreaterEqual(full - part, model.distance_bytes_per_pair * n * n)
        self.assertLess(part, full)
        # N^2 term: doubling N should more than double the non-PartTree estimate
        self.assertGreater(
            at.estimate_mafft_peak_bytes(2 * n, length) - model.mafft_base_bytes,
            2 * (full - model.mafft_base_bytes),
        )

    def test_protein_costs_more_than_nucleotide(self) -> None:
        self.assertGreater(
            at.estimate_mafft_peak_bytes(500, 800, alphabet="protein"),
            at.estimate_mafft_peak_bytes(500, 800, alphabet="nucleotide"),
        )

    def test_parser_and_supermatrix(self) -> None:
        self.assertEqual(at.estimate_supermatrix_bytes(1000, 100_000), 100_000_000)
        parser = at.estimate_parser_buffer_bytes(1000, 2000)
        self.assertGreater(parser, 1000 * 2000)  # at least one byte per cell
        self.assertEqual(at.estimate_parser_buffer_bytes(0, 100), 0)

    def test_task_memory_peak_is_max_not_sum(self) -> None:
        tm = at.estimate_task_memory(300, 1500, parttree=False)
        self.assertEqual(tm.peak_bytes, max(tm.mafft_bytes, tm.parser_bytes))
        self.assertEqual(at.est_mem_bytes(300, 1500), tm.peak_bytes)

    def test_safety_factor_scales(self) -> None:
        model = dataclasses.replace(at.DEFAULT_MEMORY_MODEL, safety_factor=2.0)
        base = at.est_mem_bytes(100, 1000)
        self.assertAlmostEqual(
            at.est_mem_bytes(100, 1000, model=model) / base, 2.0, places=2
        )


class TestParallelismPlanning(unittest.TestCase):
    LOCI = [(1000, 2000, False)] * 12

    def _components(self) -> tuple[int, int, int]:
        p1 = at.plan_parallelism(self.LOCI, 10**12, max_workers=1)
        return p1.max_mafft_bytes, p1.parser_buffer_bytes, p1.supermatrix_bytes

    def test_blueprint_formula_and_exact_boundary(self) -> None:
        mafft, parser, sup = self._components()
        plan = at.plan_parallelism(self.LOCI, mafft + 4 * parser + sup, max_workers=12)
        self.assertEqual(plan.workers, 4)
        self.assertEqual(plan.m_peak_bytes, mafft + 4 * parser + sup)
        self.assertTrue(plan.fits and plan.limited_by_memory)
        below = at.plan_parallelism(
            self.LOCI, mafft + 4 * parser + sup - 1, max_workers=12
        )
        self.assertEqual(below.workers, 3)
        self.assertTrue(any("reduced" in w for w in below.warnings))

    def test_plenty_of_memory_keeps_requested_workers(self) -> None:
        plan = at.plan_parallelism(self.LOCI, 10**12, max_workers=8)
        self.assertEqual(plan.workers, 8)
        self.assertFalse(plan.limited_by_memory)
        self.assertEqual(plan.warnings, ())

    def test_never_below_one_and_loud_warning(self) -> None:
        plan = at.plan_parallelism(self.LOCI, 1024, max_workers=8)
        self.assertEqual(plan.workers, 1)
        self.assertFalse(plan.fits)
        self.assertIn("MEMORY ENVELOPE TOO SMALL", plan.warnings[0])

    def test_conservative_bound_is_stricter(self) -> None:
        mafft, parser, sup = self._components()
        envelope = mafft + 6 * parser + sup  # blueprint allows 6, sum-of-top-6 does not
        loose = at.plan_parallelism(self.LOCI, envelope, max_workers=12)
        strict = at.plan_parallelism(
            self.LOCI, envelope, max_workers=12, conservative=True
        )
        self.assertEqual(loose.workers, 6)
        self.assertLess(strict.workers, loose.workers)
        self.assertGreater(loose.m_peak_concurrent_bytes, loose.m_peak_bytes)

    def test_empty_workload_and_workers_never_exceed_loci(self) -> None:
        empty = at.plan_parallelism([], 1, max_workers=4)
        self.assertEqual((empty.workers, empty.fits), (4, True))
        few = at.plan_parallelism([(10, 100, True)] * 2, 10**9, max_workers=8)
        self.assertEqual(few.workers, 2)

    def test_largest_locus_dominates_mafft_term(self) -> None:
        loci = [(10, 100, False), (5000, 3000, True), (50, 500, False)]
        plan = at.plan_parallelism(loci, 10**12)
        self.assertEqual(
            plan.max_mafft_bytes,
            at.estimate_mafft_peak_bytes(5000, 3000, parttree=True),
        )

    def test_bad_workload_item(self) -> None:
        with self.assertRaises(ValueError):
            at.plan_parallelism([(1,)], 10**9)

    def test_named_tuple_and_alphabet_item(self) -> None:
        plan = at.plan_parallelism(
            [
                at.LocusWorkload(100, 500, False, "protein"),
                (100, 500, False, "protein"),
            ],
            10**12,
        )
        self.assertEqual(
            plan.max_mafft_bytes,
            at.estimate_mafft_peak_bytes(100, 500, alphabet="protein"),
        )


class TestWorkloadAwareAutotune(_HostCase):
    def test_memory_downscales_workers_but_not_cpu_budget(self) -> None:
        host = self.make_host(cores=8, threads=1, mem_avail_gb=1)
        loci = [(4000, 3000, False)] * 20
        plan = at.autotune(host, workload=loci, memory_limit="700M")
        self.assertEqual(plan.cpu_budget, 8)
        self.assertLess(plan.workers, 8)
        self.assertGreaterEqual(plan.workers, 1)
        self.assertIsNotNone(plan.parallelism)
        self.assertLessEqual(plan.workers, plan.parallelism.workers)  # type: ignore[union-attr]
        self.assertEqual(
            plan.threads_per_task_cap, 8
        )  # fewer tasks may take more threads
        self.assertIn("reduced", " ".join(plan.warnings))
        self.assertIn("PROVISIONAL", plan.explain())

    def test_hopeless_envelope_warns_loudly_but_returns_plan(self) -> None:
        host = self.make_host(cores=8, threads=1, mem_avail_gb=1)
        plan = at.autotune(host, workload=[(50_000, 5000, False)], memory_limit="100M")
        self.assertEqual(plan.workers, 1)
        self.assertTrue(any("MEMORY ENVELOPE TOO SMALL" in w for w in plan.warnings))

    def test_small_workload_keeps_all_workers(self) -> None:
        host = self.make_host(cores=4, threads=1, mem_avail_gb=8)
        plan = at.autotune(host, workload=[(20, 800, False)] * 10)
        self.assertEqual(plan.workers, 4)
        self.assertEqual(plan.warnings, ())

    def test_plan_est_mem_uses_plan_model(self) -> None:
        plan = at.autotune(self.make_host())
        self.assertEqual(plan.est_mem_bytes(100, 1000), at.est_mem_bytes(100, 1000))


class TestScratchRecommendation(_HostCase):
    def test_large_tmpfs_is_primary_and_disk_is_fallback(self) -> None:
        plan = at.autotune(
            self.epyc(env={"SLURM_JOB_ID": "1", "SLURM_TMPDIR": "/local/j1"})
        )
        self.assertEqual(plan.scratch_path, "/dev/shm")
        self.assertEqual(plan.scratch_fallback_path, "/local/j1")  # SLURM_TMPDIR first
        self.assertIn("RAM tier", plan.explain())

    def test_ram_scratch_budget_is_capped_by_the_envelope(self) -> None:
        host = self.make_host(
            mem_avail_gb=4,
            mountinfo=hb.MOUNTINFO_EPYC,
            block_devices=hb.EPYC_BLOCK,
            du={"/dev/shm": (256 * GIB, 256 * GIB)},
        )
        plan = at.autotune(host)
        envelope = int(4 * GIB * 0.85)
        self.assertEqual(plan.scratch_path, "/dev/shm")
        self.assertEqual(plan.scratch_budget_bytes, int(envelope * 0.25))
        # tmpfs pages count against the same memory: tasks plan with the rest
        self.assertEqual(
            plan.task_memory_envelope_bytes, envelope - plan.scratch_budget_bytes
        )
        self.assertEqual(plan.memory_envelope_bytes, envelope)

    def test_workers_are_fitted_to_the_task_envelope(self) -> None:
        host = self.make_host(
            cores=8,
            threads=1,
            mem_avail_gb=2,
            mountinfo=hb.MOUNTINFO_EPYC,
            du={"/dev/shm": (256 * GIB, 256 * GIB)},
        )
        loci = [(4000, 3000, False)] * 20
        plan = at.autotune(host, workload=loci)
        assert plan.parallelism is not None
        self.assertEqual(
            plan.parallelism.envelope_bytes, plan.task_memory_envelope_bytes
        )
        self.assertLess(plan.task_memory_envelope_bytes, plan.memory_envelope_bytes)

    def test_tmpfs_too_small_after_cap_falls_back_to_disk(self) -> None:
        host = self.make_host(
            mem_avail_gb=1,  # envelope ~870 MiB -> RAM budget ~217 MiB ...
            mountinfo=hb.MOUNTINFO_EPYC,
            block_devices=hb.EPYC_BLOCK,
            du={"/dev/shm": (256 * GIB, 256 * GIB)},
        )
        self.assertEqual(at.autotune(host).scratch_path, "/dev/shm")
        tiny = at.autotune(host, memory_limit="300M")  # ... but ~64 MiB at 300 MiB
        self.assertEqual(tiny.scratch_path, "/local")
        self.assertEqual(tiny.task_memory_envelope_bytes, tiny.memory_envelope_bytes)

    def test_small_tmpfs_falls_back_to_best_disk(self) -> None:
        host = self.make_host(
            mountinfo=hb.MOUNTINFO_EPYC,
            block_devices=hb.EPYC_BLOCK,
            du={"/dev/shm": (64 * MIB, 64 * MIB)},
        )
        plan = at.autotune(host)
        # /local (xfs on NVMe) beats /scratch (lustre); the 64 MiB tmpfs is too small
        self.assertEqual(plan.scratch_path, "/local")
        self.assertIn("only", plan.explain())
        self.assertEqual(plan.warnings, ())

    def test_network_only_scratch_warns(self) -> None:
        mountinfo = "1 0 0:1 / / rw - lustre mds:/fs rw\n"
        host = self.make_host(mountinfo=mountinfo, env={"TMPDIR": "/work/tmp"})
        plan = at.autotune(host)
        self.assertEqual(plan.scratch_path, "/work/tmp")
        self.assertTrue(any("network filesystem" in w for w in plan.warnings))

    def test_read_only_tiers_are_ignored(self) -> None:
        mountinfo = (
            "1 0 8:1 / / ro - ext4 /dev/sda1 ro\n"
            "2 1 0:2 / /dev/shm ro - tmpfs tmpfs ro,size=1g\n"
        )
        host = self.make_host(mountinfo=mountinfo, env={"TMPDIR": "/x"})
        plan = at.autotune(host)
        self.assertIsNone(plan.scratch_path)
        self.assertIn("system temp", plan.explain())


class TestPlanObject(_HostCase):
    def test_frozen_and_serialisable(self) -> None:
        plan = at.autotune(self.make_host(), workload=[(10, 100, False)])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.workers = 99  # type: ignore[misc]
        payload = json.loads(json.dumps(plan.to_dict()))
        self.assertEqual(payload["cpu_budget"], plan.cpu_budget)
        self.assertIsInstance(plan.topology, CpuTopology)

    def test_explain_covers_every_decision(self) -> None:
        text = at.autotune(self.epyc(), num_threads=10_000).explain()
        for needle in (
            "CPU budget",
            "memory envelope",
            "scratch",
            "pinning",
            "Warnings",
            "NUMA",
        ):
            self.assertIn(needle, text)


class TestLive(unittest.TestCase):
    def test_live_autotune_smoke(self) -> None:
        plan = at.autotune()
        self.assertGreaterEqual(plan.cpu_budget, 1)
        self.assertGreaterEqual(plan.workers, 1)
        self.assertGreater(plan.memory_envelope_bytes, 0)
        self.assertEqual(plan.cpu_budget, plan.host.topology.physical_count)
        self.assertIsInstance(plan.explain(), str)


if __name__ == "__main__":
    unittest.main()
