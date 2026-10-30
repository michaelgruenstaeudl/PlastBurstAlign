"""Fixture-driven tests for ``plastburstalign.hardware`` (no live-host assumptions)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import hostfs_builder as hb

from plastburstalign.hardware import (
    HostProfile,
    classify_fstype,
    clear_host_cache,
    detect_host,
    format_bytes,
    parse_cpu_list,
    parse_size,
)

GIB = 1024**3
MIB = 1024**2


def _du(table: dict[str, tuple[int, int]]):
    """disk_usage stub: ``{path: (total, free)}`` with a generous default."""

    def usage(path: str) -> tuple[int, int, int]:
        total, free = table.get(path, (100 * GIB, 50 * GIB))
        return total, total - free, free

    return usage


class _TmpTree(unittest.TestCase):
    def make_tmp(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="pba_hostfs_"))
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def copy_fixture(self, name: str) -> Path:
        dest = self.make_tmp() / name
        shutil.copytree(hb.FIXTURE_DIR / name, dest)
        return dest


class TestParsers(unittest.TestCase):
    def test_parse_cpu_list(self) -> None:
        self.assertEqual(parse_cpu_list("0-3,8,10-11\n"), (0, 1, 2, 3, 8, 10, 11))
        self.assertEqual(parse_cpu_list(""), ())
        self.assertEqual(parse_cpu_list("5,5,4"), (4, 5))
        for bad in ("a-b", "3-1", "x"):
            with self.assertRaises(ValueError):
                parse_cpu_list(bad)

    def test_parse_size_and_format(self) -> None:
        self.assertEqual(parse_size("8G"), 8 * GIB)
        self.assertEqual(parse_size("512MiB"), 512 * MIB)
        self.assertEqual(parse_size("1.5 gb"), int(1.5 * GIB))
        self.assertEqual(parse_size("4000", "m"), 4000 * MIB)
        self.assertEqual(parse_size(1024), 1024)
        for bad in ("", "abc", "-1G", "3 parsecs"):
            with self.assertRaises(ValueError):
                parse_size(bad)
        self.assertEqual(format_bytes(None), "unknown")
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(3 * GIB), "3.0 GiB")

    def test_classify_fstype(self) -> None:
        for fs in ("nfs", "nfs4", "lustre", "gpfs", "beegfs", "cifs", "fuse.sshfs"):
            self.assertEqual(classify_fstype(fs), "network", fs)
        for fs in ("ext4", "xfs", "btrfs", "f2fs", "overlay"):
            self.assertEqual(classify_fstype(fs), "local", fs)
        self.assertEqual(classify_fstype("tmpfs"), "ram")
        self.assertEqual(classify_fstype("weirdfs"), "unknown")
        self.assertEqual(classify_fstype(None), "unknown")


class TestEpyc(_TmpTree):
    """Dual-socket EPYC: 2 x 32 cores, SMT2 => 64C/128T, 2 NUMA nodes."""

    def setUp(self) -> None:
        self.root = self.make_tmp()
        hb.build_host_tree(
            self.root,
            packages=2,
            cores_per_package=32,
            threads_per_core=2,
            flags=hb.X86_FLAGS_AVX2,
            mem_total_kb=512 * 1024 * 1024,
            mem_avail_kb=400 * 1024 * 1024,
            swap_total_kb=8 * 1024 * 1024,
            swap_free_kb=8 * 1024 * 1024,
            mountinfo=hb.MOUNTINFO_EPYC,
            block_devices=hb.EPYC_BLOCK,
        )
        hb.cgroup_v2(self.root, rel="/user.slice/job_1")
        self.env = {
            "PBA_SCRATCH": "/mnt/hdd data/pba",
            "TMPDIR": "/var/tmp/x",
        }
        self.host = detect_host(
            self.root,
            self.env,
            system="Linux",
            machine="x86_64",
            disk_usage=_du(
                {"/dev/shm": (256 * GIB, 256 * GIB), "/local": (3000 * GIB, 2800 * GIB)}
            ),
        )

    def test_cores_and_smt(self) -> None:
        topo = self.host.topology
        self.assertEqual(topo.physical_count, 64)
        self.assertEqual(topo.logical_count, 128)
        self.assertEqual(topo.smt_width, 2)
        self.assertEqual(self.host.host_logical_cpus, 128)
        self.assertEqual(self.host.host_physical_cores, 64)
        # Linux numbering: thread 0 of core i is cpu i, its sibling is cpu i + 64.
        self.assertEqual(topo.cores[0].cpu_ids, (0, 64))
        self.assertEqual(topo.cores[33].cpu_ids, (33, 97))
        # primary threads only when asked for physical-core pinning
        self.assertEqual(topo.cpu_ids_for([0, 33]), (0, 33))

    def test_numa_nodes(self) -> None:
        self.assertEqual(len(self.host.numa_nodes), 2)
        self.assertEqual(self.host.topology.numa_nodes, (0, 1))
        self.assertEqual(len(self.host.topology.cores_on_node(0)), 32)
        self.assertEqual(len(self.host.topology.cores_on_node(1)), 32)
        self.assertEqual(self.host.numa_nodes[1].cpus[0], 32)
        self.assertIsNotNone(self.host.numa_nodes[0].mem_total_bytes)

    def test_simd(self) -> None:
        simd = self.host.simd
        self.assertTrue(simd.avx2 and simd.avx and simd.sse4_2 and simd.fma)
        self.assertFalse(simd.avx512f)
        self.assertEqual(simd.level, "avx2")
        self.assertEqual(simd.machine, "x86_64")

    def test_memory(self) -> None:
        mem = self.host.memory
        self.assertEqual(mem.total_bytes, 512 * GIB)
        self.assertEqual(mem.available_bytes, 400 * GIB)
        self.assertEqual(mem.swap_total_bytes, 8 * GIB)
        self.assertEqual(mem.effective_bytes, 400 * GIB)
        self.assertIsNone(mem.cgroup_limit_bytes)

    def test_storage_tiers(self) -> None:
        tiers = {t.path: t for t in self.host.storage}
        shm = tiers["/dev/shm"]
        self.assertTrue(shm.is_ram)
        self.assertEqual(shm.free_bytes, 256 * GIB)
        # kernel-plumbing tmpfs mounts are not scratch candidates
        self.assertNotIn("/run/lock", tiers)
        self.assertNotIn("/sys/fs/cgroup", tiers)
        scratch = tiers["/scratch"]
        self.assertEqual((scratch.kind, scratch.device_class), ("network", "network"))
        self.assertEqual(scratch.fstype, "lustre")
        local = tiers["/local"]
        self.assertEqual((local.kind, local.device_class), ("local", "nvme"))
        self.assertEqual(local.device, "nvme0n1")
        self.assertFalse(local.rotational)
        hdd = tiers["/mnt/hdd data/pba"]  # octal-escaped mount point + env label
        self.assertEqual(hdd.device_class, "hdd")
        self.assertTrue(hdd.rotational)
        self.assertIn("env:PBA_SCRATCH", hdd.labels)
        var_tmp = tiers["/var/tmp/x"]  # on a device-mapper root -> nvme slave
        self.assertEqual(var_tmp.device_class, "nvme")
        self.assertEqual(var_tmp.fstype, "ext4")

    def test_to_dict_is_json_serialisable(self) -> None:
        payload = json.dumps(self.host.to_dict())
        self.assertIn('"cores"', payload)
        self.assertEqual(
            json.loads(payload)["topology"]["cores"][0]["cpu_ids"], [0, 64]
        )
        self.assertIsInstance(self.host.summary(), str)


class TestLaptop(_TmpTree):
    def test_single_socket_6c12t(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=6, threads_per_core=2)
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual((host.usable_physical, host.usable_logical), (6, 12))
        self.assertEqual(host.topology.numa_nodes, (0,))
        self.assertEqual(len(host.numa_nodes), 1)
        self.assertIsNone(host.cgroup_version)
        self.assertIsNone(host.cpu_quota)

    def test_adjacent_sibling_numbering(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(
            root,
            packages=1,
            cores_per_package=4,
            threads_per_core=2,
            numbering="adjacent",
        )
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual([c.cpu_ids for c in host.topology.cores][:2], [(0, 1), (2, 3)])

    def test_no_smt_host(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=4, threads_per_core=1)
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(
            (host.usable_physical, host.usable_logical, host.topology.smt_width),
            (4, 4, 1),
        )

    def test_avx512_and_aarch64(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, flags=hb.X86_FLAGS_AVX512)
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(host.simd.level, "avx512")
        arm = self.make_tmp()
        hb.build_host_tree(
            arm,
            packages=1,
            cores_per_package=4,
            threads_per_core=1,
            flags=hb.ARM_FEATURES,
            cpuinfo_key="Features",
        )
        host = detect_host(
            arm, {}, system="Linux", machine="aarch64", disk_usage=_du({})
        )
        self.assertTrue(host.simd.neon)
        self.assertEqual(host.simd.level, "neon")
        self.assertFalse(host.simd.avx2)


class TestCgroupV2Container(_TmpTree):
    def test_fractional_quota_and_memory_limit(self) -> None:
        root = self.copy_fixture("cgroup_v2_container")
        host = detect_host(
            root,
            {},
            system="Linux",
            machine="x86_64",
            disk_usage=_du({"/dev/shm": (64 * MIB, 64 * MIB)}),
        )
        self.assertEqual(host.cgroup_version, 2)
        self.assertEqual(host.cpu_quota, 2.5)
        # 2.5 CPUs floor to 2; spread policy keeps one thread on 2 distinct cores
        self.assertEqual((host.usable_physical, host.usable_logical), (2, 2))
        self.assertTrue(all(len(c.cpu_ids) == 1 for c in host.topology.cores))
        self.assertEqual(host.host_logical_cpus, 4)
        self.assertEqual(host.host_physical_cores, 2)
        mem = host.memory
        self.assertEqual(mem.cgroup_limit_bytes, 4 * GIB)
        self.assertIsNone(mem.cgroup_high_bytes)  # "max"
        self.assertEqual(mem.cgroup_usage_bytes, 1 * GIB)
        # headroom = limit - (current - inactive_file)
        self.assertEqual(mem.cgroup_headroom_bytes, 4 * GIB - (1 * GIB - 256 * MIB))
        self.assertEqual(mem.effective_bytes, mem.cgroup_headroom_bytes)
        self.assertEqual(mem.available_bytes, 40 * GIB)
        self.assertIn("docker", host.containers)
        self.assertEqual(host.simd.level, "avx512")

    def test_tmpfs_filtering_and_network_mount(self) -> None:
        root = self.copy_fixture("cgroup_v2_container")
        host = detect_host(
            root,
            {"TMPDIR": "/data/tmp"},
            system="Linux",
            machine="x86_64",
            disk_usage=_du({"/dev/shm": (64 * MIB, 60 * MIB)}),
        )
        tiers = {t.path: t for t in host.storage}
        self.assertEqual(tiers["/dev/shm"].free_bytes, 60 * MIB)
        self.assertNotIn("/proc/acpi", tiers)  # plumbing mount
        self.assertEqual(tiers["/data/tmp"].kind, "network")  # NFS4 mount

    def test_memory_high_is_a_soft_bound(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=2, threads_per_core=1)
        hb.cgroup_v2(
            root,
            memory_max=str(8 * GIB),
            memory_high=str(3 * GIB),
            memory_current=str(1 * GIB),
        )
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        mem = host.memory
        self.assertEqual(mem.cgroup_limit_bytes, 8 * GIB)
        self.assertEqual(mem.cgroup_high_bytes, 3 * GIB)
        self.assertEqual(mem.cgroup_headroom_bytes, 2 * GIB)
        self.assertEqual(mem.effective_bytes, 2 * GIB)
        self.assertIn("memory.high", mem.limit_sources[0])

    def test_nested_cgroup_uses_tightest_ancestor(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=8, threads_per_core=1)
        hb.cgroup_v2(root, rel="/job/step/task", memory_max="max")
        # parent (job) holds the real limits; the leaf has none
        hb.write(root, "sys/fs/cgroup/job/memory.max", f"{2 * GIB}\n")
        hb.write(root, "sys/fs/cgroup/job/cpu.max", "300000 100000\n")
        hb.write(root, "sys/fs/cgroup/job/step/task/memory.current", f"{GIB // 2}\n")
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(host.memory.cgroup_limit_bytes, 2 * GIB)
        self.assertEqual(host.cpu_quota, 3.0)
        self.assertEqual(host.usable_physical, 3)

    def test_tiny_quota_floors_to_one_core(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=4, threads_per_core=2)
        hb.cgroup_v2(root, cpu_max="50000 100000")
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(host.cpu_quota, 0.5)
        self.assertEqual((host.usable_physical, host.usable_logical), (1, 1))

    def test_cpuset_effective_restricts(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=4, threads_per_core=2)
        hb.cgroup_v2(root, cpuset="0-1,4-5")  # cores 0,1 with siblings 4,5
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual((host.usable_physical, host.usable_logical), (2, 4))
        self.assertEqual([c.cpu_ids for c in host.topology.cores], [(0, 4), (1, 5)])


class TestCgroupV1Container(_TmpTree):
    def test_v1_limits_and_sentinel(self) -> None:
        root = self.copy_fixture("cgroup_v1_container")
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(host.cgroup_version, 1)
        self.assertEqual(host.cpu_quota, 1.5)
        # cpuset 0-2 leaves cores (0,2),(1) -> quota floor(1.5)=1 core, one thread
        self.assertEqual((host.usable_physical, host.usable_logical), (1, 1))
        mem = host.memory
        self.assertEqual(mem.cgroup_limit_bytes, 2 * GIB)
        self.assertIsNone(mem.cgroup_high_bytes)  # 9223372036854771712 is "unlimited"
        self.assertEqual(mem.cgroup_headroom_bytes, 2 * GIB - (512 * MIB - 128 * MIB))
        # MemAvailable missing -> MemFree + Buffers + Cached
        self.assertEqual(mem.available_bytes, (4194304 + 262144 + 2097152) * 1024)
        self.assertEqual(mem.effective_bytes, mem.cgroup_headroom_bytes)
        self.assertIn("podman", host.containers)
        self.assertIn("docker", host.containers)  # from /proc/self/cgroup


class TestSlurm(_TmpTree):
    def _epyc(self, **kwargs: object) -> Path:
        root = self.make_tmp()
        hb.build_host_tree(
            root,
            packages=2,
            cores_per_package=32,
            threads_per_core=2,
            mem_total_kb=512 * 1024 * 1024,
            mem_avail_kb=400 * 1024 * 1024,
            **kwargs,  # type: ignore[arg-type]
        )
        return root

    def test_step_affinity_subset(self) -> None:
        # Slurm step bound to 8 cores (+ SMT siblings): cpus 0-7 and 64-71
        root = self._epyc(allowed_list="0-7,64-71")
        env = {
            "SLURM_JOB_ID": "1234",
            "SLURM_CPUS_PER_TASK": "16",
            "SLURM_MEM_PER_CPU": "4000",
        }
        host = detect_host(
            root, env, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertTrue(host.slurm_job)
        self.assertEqual((host.usable_physical, host.usable_logical), (8, 16))
        self.assertEqual(host.slurm_cpu_limit, 16)
        self.assertEqual(host.topology.cores[0].cpu_ids, (0, 64))
        self.assertEqual(host.host_logical_cpus, 128)
        self.assertEqual(host.memory.slurm_limit_bytes, 16 * 4000 * MIB)
        self.assertEqual(host.memory.effective_bytes, 16 * 4000 * MIB)
        self.assertEqual(host.memory.limit_sources, ("SLURM_MEM_PER_CPU",))

    def test_slurm_count_trims_unrestricted_affinity(self) -> None:
        root = self._epyc()
        env = {
            "SLURM_JOB_ID": "9",
            "SLURM_CPUS_ON_NODE": "4",
            "SLURM_MEM_PER_NODE": "16384",
        }
        host = detect_host(
            root, env, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual((host.usable_physical, host.usable_logical), (4, 4))
        self.assertEqual(host.memory.slurm_limit_bytes, 16 * GIB)
        # distinct physical cores, primary threads, deterministic
        self.assertEqual(
            [c.cpu_ids for c in host.topology.cores], [(0,), (1,), (2,), (3,)]
        )

    def test_job_cpus_per_node_forms(self) -> None:
        root = self._epyc(allowed_list="0-31,64-95")
        env = {"SLURM_JOB_ID": "9", "SLURM_JOB_CPUS_PER_NODE": "16(x2)"}
        host = detect_host(
            root, env, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(host.slurm_cpu_limit, 16)
        self.assertEqual(host.usable_logical, 16)
        env = {"SLURM_JOB_ID": "9", "SLURM_JOB_CPUS_PER_NODE": "48,8(x2)"}
        host = detect_host(
            root, env, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(
            host.slurm_cpu_limit, 8
        )  # heterogeneous: smallest, conservative

    def test_slurm_tmpdir_tier(self) -> None:
        root = self._epyc(mountinfo=hb.MOUNTINFO_EPYC, block_devices=hb.EPYC_BLOCK)
        env = {"SLURM_JOB_ID": "9", "SLURM_TMPDIR": "/local/job_9"}
        host = detect_host(
            root, env, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        tiers = {t.path: t for t in host.storage}
        self.assertIn("env:SLURM_TMPDIR", tiers["/local/job_9"].labels)
        self.assertEqual(tiers["/local/job_9"].device_class, "nvme")
        self.assertEqual(host.storage[0].path, "/local/job_9")  # env tiers come first

    def test_affinity_injection_overrides_status_file(self) -> None:
        root = self._epyc(allowed_list="0-127")
        host = detect_host(
            root,
            {},
            affinity=[0, 1, 64, 65],
            system="Linux",
            machine="x86_64",
            disk_usage=_du({}),
        )
        self.assertEqual((host.usable_physical, host.usable_logical), (2, 4))
        self.assertEqual(host.affinity_cpus, (0, 1, 64, 65))


class TestMacOS(_TmpTree):
    def test_apple_silicon_with_mock_sysctl(self) -> None:
        values = {
            "hw.physicalcpu": "8",
            "hw.logicalcpu": "8",
            "hw.memsize": str(16 * GIB),
        }
        calls: list[str] = []

        def fake_sysctl(name: str) -> str | None:
            calls.append(name)
            return values.get(name)

        root = self.make_tmp()  # empty: macOS has no /proc or /sys
        host = detect_host(
            root,
            {"TMPDIR": "/var/folders/xx/T"},
            system="Darwin",
            machine="arm64",
            sysctl=fake_sysctl,
            use_psutil=False,
            disk_usage=_du({}),
        )
        self.assertEqual((host.usable_physical, host.usable_logical), (8, 8))
        self.assertEqual(host.host_physical_cores, 8)
        self.assertEqual(host.memory.total_bytes, 16 * GIB)
        self.assertTrue(host.simd.neon)
        self.assertIn("hw.physicalcpu", calls)
        self.assertEqual(host.storage[0].path, "/var/folders/xx/T")
        self.assertIsNone(host.cgroup_version)

    def test_intel_mac_smt_adjacent_pairs(self) -> None:
        values = {
            "hw.physicalcpu": "6",
            "hw.logicalcpu": "12",
            "hw.memsize": str(32 * GIB),
            "machdep.cpu.features": "FPU SSE SSE2 SSE4.2 AVX1.0 FMA",
            "machdep.cpu.leaf7_features": "AVX2 BMI1",
        }
        host = detect_host(
            self.make_tmp(),
            {},
            system="Darwin",
            machine="x86_64",
            sysctl=values.get,
            use_psutil=False,
            disk_usage=_du({}),
        )
        self.assertEqual((host.usable_physical, host.usable_logical), (6, 12))
        self.assertEqual(host.topology.cores[1].cpu_ids, (2, 3))  # adjacent pairs
        self.assertEqual(host.simd.level, "avx2")
        self.assertTrue(host.simd.fma)

    def test_failing_sysctl_does_not_crash(self) -> None:
        def broken(name: str) -> str | None:
            raise RuntimeError("sysctl exploded")

        host = detect_host(
            self.make_tmp(),
            {},
            system="Darwin",
            machine="arm64",
            sysctl=broken,
            use_psutil=False,
            disk_usage=_du({}),
        )
        self.assertGreaterEqual(host.usable_physical, 1)
        self.assertTrue(host.warnings)


class TestDegradation(_TmpTree):
    def test_empty_root_degrades_to_psutil_and_os(self) -> None:
        host = detect_host(self.make_tmp(), {}, disk_usage=_du({}))
        self.assertIsInstance(host, HostProfile)
        self.assertGreaterEqual(host.usable_physical, 1)
        self.assertGreaterEqual(host.usable_logical, host.usable_physical)
        self.assertLessEqual(host.usable_logical, (os.cpu_count() or 1))
        self.assertTrue(host.warnings)
        self.assertEqual(host.numa_nodes, ())
        self.assertEqual(host.topology.numa_nodes, (0,))
        # memory comes from the psutil fallback
        self.assertIsNotNone(host.memory.total_bytes)
        self.assertIsNotNone(host.memory.effective_bytes)

    def test_empty_root_without_psutil_still_works(self) -> None:
        host = detect_host(self.make_tmp(), {}, use_psutil=False, disk_usage=_du({}))
        self.assertGreaterEqual(host.usable_physical, 1)
        self.assertIsNone(host.memory.total_bytes)
        self.assertIsNone(host.memory.effective_bytes)
        self.assertEqual(host.storage, ())
        self.assertTrue(any("RAM" in w or "CPU" in w for w in host.warnings))

    def test_windows_style_host_uses_psutil_and_adjacent_siblings(self) -> None:
        host = detect_host(
            self.make_tmp(), {}, system="Windows", machine="AMD64", disk_usage=_du({})
        )
        self.assertGreaterEqual(host.usable_physical, 1)
        self.assertEqual(host.system, "Windows")
        if host.topology.smt_width == 2:  # Windows enumerates siblings adjacently
            self.assertEqual(host.topology.cores[0].cpu_ids, (0, 1))
        # a missing /proc and /sys is normal on Windows: not a warning
        self.assertFalse(any("topology files missing" in w for w in host.warnings))

    def test_garbage_files_never_raise(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=2, threads_per_core=2)
        hb.write(root, "sys/devices/system/cpu/online", "not-a-list\n")
        hb.write(root, "sys/devices/system/node/node0/cpulist", "zzz\n")
        hb.write(
            root, "sys/devices/system/cpu/cpu1/topology/thread_siblings_list", "??\n"
        )
        hb.write(root, "proc/meminfo", "MemTotal: banana kB\n\x00\x01\n")
        hb.write(root, "proc/cpuinfo", "\x00\x00 garbage")
        hb.write(root, "proc/self/mountinfo", "this is not a mountinfo line\n- - -\n")
        hb.cgroup_v2(root, cpu_max="abc def", memory_max="lots", memory_high="-5")
        host = detect_host(
            root,
            {"SLURM_CPUS_PER_TASK": "many", "SLURM_MEM_PER_NODE": "??"},
            system="Linux",
            machine="x86_64",
            use_psutil=False,
            disk_usage=_du({}),
        )
        self.assertGreaterEqual(host.usable_physical, 1)
        self.assertIsNone(host.memory.cgroup_limit_bytes)
        self.assertEqual(host.simd.level, "unknown")
        self.assertTrue(host.warnings)

    def test_unreadable_status_falls_back_to_online_set(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=3, threads_per_core=1)
        (root / "proc/self/status").unlink()
        host = detect_host(
            root, {}, system="Linux", machine="x86_64", disk_usage=_du({})
        )
        self.assertEqual(host.usable_logical, 3)

    def test_slurm_job_flag_requires_job_id(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=2, threads_per_core=1)
        host = detect_host(
            root,
            {"SLURM_CPUS_PER_TASK": "2"},
            system="Linux",
            machine="x86_64",
            disk_usage=_du({}),
        )
        self.assertFalse(host.slurm_job)
        self.assertEqual(host.slurm_cpu_limit, 2)

    def test_container_env_hints(self) -> None:
        root = self.make_tmp()
        hb.build_host_tree(root, packages=1, cores_per_package=2, threads_per_core=1)
        host = detect_host(
            root,
            {"APPTAINER_CONTAINER": "/x.sif"},
            system="Linux",
            machine="x86_64",
            disk_usage=_du({}),
        )
        self.assertEqual(host.containers, ("apptainer",))


class TestLiveAndCache(unittest.TestCase):
    def tearDown(self) -> None:
        clear_host_cache()

    def test_live_detection_is_sane_and_cached(self) -> None:
        clear_host_cache()
        first = detect_host()
        self.assertGreaterEqual(first.usable_physical, 1)
        self.assertLessEqual(
            first.usable_logical, os.cpu_count() or first.usable_logical
        )
        self.assertIs(detect_host(), first)  # cached
        self.assertIsNot(detect_host(refresh=True), first)
        self.assertIs(detect_host(), detect_host())
        # a non-default call is never cached
        self.assertIsNot(detect_host(env={}), detect_host(env={}))

    def test_live_respects_current_affinity(self) -> None:
        if not hasattr(os, "sched_getaffinity"):
            self.skipTest("no sched_getaffinity on this platform")
        host = detect_host(refresh=True)
        self.assertEqual(set(host.affinity_cpus), set(os.sched_getaffinity(0)))
        self.assertLessEqual(host.usable_logical, len(os.sched_getaffinity(0)))


if __name__ == "__main__":
    unittest.main()
