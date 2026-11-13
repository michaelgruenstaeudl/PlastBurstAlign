from __future__ import annotations

import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from plastburstalign.ephemeral_engine import (
    DEFAULT_MIN_HEADROOM_BYTES,
    EphemeralScratchManager,
    ScratchEngine,
    ScratchEnvironment,
    _CleanupRegistry,
    discover_candidates,
    estimate_scratch_need,
    find_mount,
    is_ram_path,
    parse_mountinfo,
    parse_proc_mounts,
    read_mount_table,
    sweep_stale_dirs,
    tmpfs_capacity_ok,
)


def _can_self_signal() -> bool:
    """Some sandboxes forbid even ``raise_signal`` on the current process."""
    try:
        old = signal.signal(signal.SIGUSR2, lambda _s, _f: None)
        try:
            signal.raise_signal(signal.SIGUSR2)
        finally:
            signal.signal(signal.SIGUSR2, old)
        return True
    except (OSError, ValueError, AttributeError):
        return False


needs_signals = unittest.skipUnless(
    _can_self_signal(), "process may not signal itself in this sandbox"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
HEADROOM = 1000  # tiny headroom so tests can use tiny budgets


def mountinfo(*mounts: tuple[str, str]) -> str:
    """Fake ``/proc/self/mountinfo`` text for ``(mount_point, fs_type)`` pairs."""
    lines = [
        "20 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw",
    ]
    for i, (mp, fs) in enumerate(mounts):
        esc = mp.replace(" ", "\\040")
        lines.append(f"{30 + i} 20 0:{40 + i} / {esc} rw,nosuid,nodev - {fs} {fs} rw")
    return "\n".join(lines) + "\n"


class Host:
    """Mutable fake host handed to :class:`ScratchEnvironment` as callables."""

    def __init__(self, ram: int = 8 * GIB, free: dict[str, int] | None = None):
        self.ram = ram
        self.free = dict(free or {})
        self.default_free = 100 * GIB
        self.alive: set[int] = set()

    def disk_usage(self, path: str) -> tuple[int, int, int]:
        best = None
        for prefix in self.free:
            if (path == prefix or path.startswith(prefix.rstrip("/") + "/")) and (
                best is None or len(prefix) > len(best)
            ):
                best = prefix
        free = self.free[best] if best else self.default_free
        return (free + 1, 1, free)

    def available_ram(self) -> int:
        return self.ram

    def pid_alive(self, pid: int) -> bool:
        return pid in self.alive

    def env(self, **kw: object) -> ScratchEnvironment:
        kw.setdefault("platform", "linux")
        return ScratchEnvironment(
            disk_usage=self.disk_usage,
            available_ram=self.available_ram,
            pid_alive=self.pid_alive,
            **kw,  # type: ignore[arg-type]
        )


class TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory(prefix="pba_test_scratch_")
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(os.path.realpath(self._td.name))

    def mkdir(self, name: str) -> Path:
        p = self.tmp / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def engine(self, *args: object, **kw: object) -> ScratchEngine:
        kw.setdefault("min_headroom_bytes", HEADROOM)
        eng = ScratchEngine(*args, **kw)  # type: ignore[arg-type]
        self.addCleanup(eng.cleanup)
        return eng

    def ram_engine(
        self,
        free: int = 10_000,
        ram: int = 8 * GIB,
        **kw: object,
    ) -> tuple[ScratchEngine, Host, Path, Path]:
        """Engine whose RAM tier is a fake tmpfs dir and whose disk tier is another dir."""
        shm = self.mkdir("fake_tmpfs")
        disk = self.mkdir("node_local")
        host = Host(ram=ram, free={str(shm): free})
        env = host.env(
            mounts=mountinfo((str(shm), "tmpfs")),
            env={"SLURM_TMPDIR": str(disk)},
            platform="linux",
            uid=4242,
            temp_dir=str(self.mkdir("systemp")),
        )
        eng = self.engine(environment=env, **kw)
        return eng, host, shm, disk


class TestMountParsing(unittest.TestCase):
    MOUNTINFO = textwrap.dedent(
        """\
        36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
        699 684 0:30 / /dev/shm rw,nosuid,nodev - tmpfs tmpfs rw,inode64
        707 703 0:51 / /run/user/1000 rw,nosuid,nodev,relatime - tmpfs tmpfs rw,size=1566032k
        800 664 0:60 / /mnt/my\\040ram rw - ramfs ramfs rw
        1506 684 0:57 / /dev rw,nosuid - tmpfs none rw,size=64k
        1513 1506 0:68 / /dev/shm rw,nosuid,nodev,noexec,relatime - tmpfs none rw,size=65536k
        garbage line
        """
    )

    def test_parse_mountinfo(self):
        mounts = parse_mountinfo(self.MOUNTINFO)
        self.assertEqual(len(mounts), 6)
        self.assertEqual(mounts[0].mount_point, "/mnt2")
        self.assertEqual(mounts[0].fs_type, "ext3")
        self.assertEqual(mounts[0].source, "/dev/root")
        self.assertIn("noatime", mounts[0].options)
        self.assertEqual(mounts[3].mount_point, "/mnt/my ram")  # octal escape decoded
        self.assertTrue(mounts[3].is_ram_fs)
        self.assertFalse(mounts[0].is_ram_fs)

    def test_parse_proc_mounts(self):
        text = "tmpfs /dev/shm tmpfs rw,nosuid,nodev 0 0\n/dev/sda1 / ext4 ro,relatime 0 0\n"
        mounts = parse_proc_mounts(text)
        self.assertEqual([m.fs_type for m in mounts], ["tmpfs", "ext4"])
        self.assertTrue(mounts[0].is_ram_fs)
        self.assertTrue(mounts[1].read_only)

    def test_read_mount_table_from_injected_proc_root(self):
        with tempfile.TemporaryDirectory() as td:
            proc = Path(td)
            (proc / "self").mkdir()
            (proc / "self" / "mountinfo").write_text(self.MOUNTINFO)
            self.assertEqual(len(read_mount_table(proc)), 6)
        with tempfile.TemporaryDirectory() as td:  # falls back to /proc/mounts
            proc = Path(td)
            (proc / "mounts").write_text("tmpfs /dev/shm tmpfs rw 0 0\n")
            self.assertEqual(
                [m.mount_point for m in read_mount_table(proc)], ["/dev/shm"]
            )
        with tempfile.TemporaryDirectory() as td:  # nothing readable
            self.assertEqual(read_mount_table(td), [])

    def test_find_mount_longest_prefix_and_later_entry_wins(self):
        mounts = parse_mountinfo(self.MOUNTINFO)
        hit = find_mount("/dev/shm/pba_1_abc/t1", mounts)
        assert hit is not None
        self.assertEqual(
            (hit.mount_point, hit.source), ("/dev/shm", "none")
        )  # the later one
        self.assertEqual(find_mount("/dev/null", mounts).mount_point, "/dev")  # type: ignore[union-attr]
        self.assertIsNone(find_mount("/home/x", mounts))  # no "/" in this table
        # a path that merely shares a string prefix is not inside the mount
        self.assertEqual(find_mount("/dev/shmx", mounts).mount_point, "/dev")  # type: ignore[union-attr]

    def test_is_ram_path(self):
        mounts = parse_mountinfo(self.MOUNTINFO)
        self.assertTrue(is_ram_path("/dev/shm/work", mounts))
        self.assertTrue(is_ram_path("/mnt/my ram/deep/dir", mounts))
        self.assertTrue(is_ram_path("/run/user/1000/x", mounts))
        self.assertFalse(is_ram_path("/home/me/data", mounts))
        self.assertFalse(is_ram_path("/mnt2/data", mounts))

    @unittest.skipUnless(os.path.isdir("/dev/shm"), "needs /dev/shm")
    def test_legacy_tmpfs_probe_uses_the_real_mount_table(self):
        # the old statvfs.f_type probe could never work; /dev/shm is tmpfs on Linux
        if not sys.platform.startswith("linux"):
            self.skipTest("Linux only")
        self.assertTrue(EphemeralScratchManager._check_if_tmpfs(Path("/dev/shm")))
        self.assertTrue(EphemeralScratchManager._check_if_tmpfs(Path("/dev/shm/x/y")))
        self.assertFalse(EphemeralScratchManager._check_if_tmpfs(Path(os.getcwd())))


class TestCapacityRule(unittest.TestCase):
    def test_estimate_is_three_times_inflight(self):
        self.assertEqual(estimate_scratch_need(1000), 3000)
        self.assertEqual(estimate_scratch_need(-5), 0)

    def test_rule(self):
        need, head = 3000, 500
        self.assertTrue(tmpfs_capacity_ok(10_000, 10_000, need, head))
        self.assertTrue(tmpfs_capacity_ok(3500, 3500, need, head))  # exactly enough
        self.assertFalse(tmpfs_capacity_ok(3499, 10_000, need, head))  # tmpfs too small
        self.assertFalse(
            tmpfs_capacity_ok(10_000, 3499, need, head)
        )  # RAM can't absorb
        self.assertFalse(
            tmpfs_capacity_ok(None, 10_000, need, head)
        )  # unknown free space
        self.assertTrue(
            tmpfs_capacity_ok(10_000, None, need, head)
        )  # unknown RAM tolerated


class TestTierOrder(TmpCase):
    def test_discovery_order_on_linux(self):
        mounts = mountinfo(
            ("/mnt/ramdisk", "tmpfs"),
            ("/dev/shm", "tmpfs"),
            ("/run/user/1234", "tmpfs"),
            ("/sys/fs/cgroup", "tmpfs"),
            ("/run/lock", "tmpfs"),
            ("/data", "ext4"),
        )
        env = ScratchEnvironment(
            platform="linux",
            mounts=mounts,
            uid=1234,
            temp_dir="/systemp",
            env={
                "PBA_SCRATCH": "/explicit",
                "SLURM_TMPDIR": "/slurm",
                "TMPDIR": "/tmpdir",
            },
        )
        cands = discover_candidates("/custom", env)
        self.assertEqual(
            [(c.source, str(c.path)) for c in cands],
            [
                ("custom", "/custom"),
                ("PBA_SCRATCH", "/explicit"),
                ("tmpfs", "/dev/shm"),
                ("tmpfs", "/run/user/1234"),
                ("tmpfs", "/mnt/ramdisk"),
                ("SLURM_TMPDIR", "/slurm"),
                ("TMPDIR", "/tmpdir"),
                ("system-temp", "/systemp"),
            ],
        )
        self.assertTrue(
            cands[0].explicit and cands[1].explicit and not cands[2].explicit
        )
        self.assertEqual(
            [c.is_ram for c in cands],
            [False, False, True, True, True, False, False, False],
        )

    def test_discovery_without_custom_and_with_all_node_local_vars(self):
        env = ScratchEnvironment(
            platform="linux",
            mounts="",
            temp_dir="/systemp",
            env={
                "LOCAL_SCRATCH": "/local",
                "TMPDIR": "/tmpdir",
                "SLURM_TMPDIR": "/slurm",
            },
        )
        self.assertEqual(
            [c.source for c in discover_candidates(None, env)],
            ["SLURM_TMPDIR", "LOCAL_SCRATCH", "TMPDIR", "system-temp"],
        )

    def test_tmpfs_tmpdir_is_classified_as_ram(self):
        env = ScratchEnvironment(
            platform="linux",
            mounts=mountinfo(("/tmp", "tmpfs")),
            temp_dir="/tmp",
            env={"TMPDIR": "/tmp/x"},
        )
        cands = discover_candidates(None, env)
        self.assertTrue(
            all(c.is_ram for c in cands if c.source in ("TMPDIR", "system-temp"))
        )

    def test_macos_looks_for_ramdisk_volumes_and_env(self):
        volumes = self.mkdir("Volumes")
        (volumes / "RAMDisk").mkdir()
        (volumes / "ramdisk2").mkdir()
        (volumes / "Macintosh HD").mkdir()
        env = ScratchEnvironment(
            platform="darwin",
            mounts=mountinfo(("/dev/shm", "tmpfs")),  # must be ignored on macOS
            volumes_root=str(volumes),
            temp_dir="/systemp",
            env={"PBA_SCRATCH": "/explicit", "TMPDIR": "/var/folders/x"},
        )
        cands = discover_candidates(None, env)
        self.assertEqual(
            [(c.source, c.path.name) for c in cands],
            [
                ("PBA_SCRATCH", "explicit"),
                ("RAMDisk", "RAMDisk"),
                ("RAMDisk", "ramdisk2"),
                ("TMPDIR", "x"),
                ("system-temp", "systemp"),
            ],
        )
        self.assertTrue(cands[1].is_ram and not cands[3].is_ram)

    def test_windows_uses_plain_temp(self):
        env = ScratchEnvironment(
            platform="win32",
            mounts=mountinfo(("/dev/shm", "tmpfs")),
            temp_dir="C:/Temp",
            env={
                "TMPDIR": "C:/other",
                "SLURM_TMPDIR": "D:/slurm",
                "PBA_SCRATCH": "D:/fast",
            },
        )
        self.assertEqual(
            [c.source for c in discover_candidates(None, env)],
            ["PBA_SCRATCH", "system-temp"],
        )
        self.assertFalse(any(c.is_ram for c in discover_candidates(None, env)))

    def test_engine_prefers_custom_then_pba_scratch_then_tmpfs(self):
        shm = self.mkdir("fake_tmpfs")
        explicit = self.mkdir("explicit")
        pba = self.mkdir("pba_env")
        host = Host(free={str(shm): 10_000})
        mounts = mountinfo((str(shm), "tmpfs"))
        base_env = {"PBA_SCRATCH": str(pba)}

        e1 = self.engine(
            explicit,
            environment=host.env(mounts=mounts, env=base_env, temp_dir=str(self.tmp)),
        )
        self.assertEqual(e1.path.parent, explicit.resolve())
        self.assertFalse(e1.is_ramdisk)
        self.assertIsNone(e1.ram_path)  # user chose disk: no RAM tier

        e2 = self.engine(
            environment=host.env(mounts=mounts, env=base_env, temp_dir=str(self.tmp))
        )
        self.assertEqual(e2.path.parent, pba)

        e3 = self.engine(
            environment=host.env(mounts=mounts, env={}, temp_dir=str(self.tmp))
        )
        self.assertTrue(e3.is_ramdisk)
        self.assertEqual(e3.path.parent, shm)

    def test_custom_dir_on_tmpfs_is_honoured_without_capacity_rule(self):
        shm = self.mkdir("fake_tmpfs")
        host = Host(free={str(shm): 10})  # tiny, would fail the capacity rule
        env = host.env(
            mounts=mountinfo((str(shm), "tmpfs")), env={}, temp_dir=str(self.tmp)
        )
        eng = self.engine(shm, environment=env)
        self.assertTrue(eng.is_ramdisk)
        self.assertEqual(eng.path.parent, shm)
        self.assertEqual(eng.ramdisk_budget_bytes, 0)  # but nothing may be admitted
        self.assertEqual(eng.reserve(1).tier, "disk")

    def test_node_local_order_when_no_tmpfs_qualifies(self):
        slurm = self.mkdir("slurm")
        local = self.mkdir("local")
        tmpdir = self.mkdir("tmpdir")
        host = Host()
        for env_vars, expected in (
            (
                {
                    "SLURM_TMPDIR": str(slurm),
                    "LOCAL_SCRATCH": str(local),
                    "TMPDIR": str(tmpdir),
                },
                slurm,
            ),
            ({"LOCAL_SCRATCH": str(local), "TMPDIR": str(tmpdir)}, local),
            ({"TMPDIR": str(tmpdir)}, tmpdir),
            (
                {"SLURM_TMPDIR": str(self.tmp / "missing"), "TMPDIR": str(tmpdir)},
                tmpdir,
            ),
            ({}, self.tmp),
        ):
            env = host.env(
                mounts="", env=env_vars, temp_dir=str(self.tmp), platform="linux"
            )
            eng = self.engine(environment=env)
            self.assertEqual(eng.path.parent, expected, env_vars)
            self.assertFalse(eng.is_ramdisk)

    def test_tmpfs_tier_then_disk_fallback_tier(self):
        eng, _host, shm, disk = self.ram_engine()
        self.assertTrue(eng.is_ramdisk)
        self.assertEqual(eng.ram_path.parent, shm)  # type: ignore[union-attr]
        self.assertEqual(eng.disk_path.parent, disk)  # type: ignore[union-attr]
        self.assertEqual(eng.ramdisk_budget_bytes, 10_000 - HEADROOM)
        self.assertIn("ram=", eng.describe())

    def test_tmpfs_skipped_when_free_space_below_need_plus_headroom(self):
        shm = self.mkdir("fake_tmpfs")
        disk = self.mkdir("disk")
        host = Host(free={str(shm): 5_000})
        env = host.env(
            mounts=mountinfo((str(shm), "tmpfs")),
            env={"SLURM_TMPDIR": str(disk)},
            temp_dir=str(self.tmp),
        )
        small = self.engine(
            environment=env, expected_inflight_bytes=1000
        )  # need 3000+1000
        self.assertTrue(small.is_ramdisk)
        big = self.engine(
            environment=env, expected_inflight_bytes=2000
        )  # need 6000+1000
        self.assertFalse(big.is_ramdisk)
        self.assertEqual(big.path.parent, disk)
        self.assertIsNone(big.ram_path)

    def test_tmpfs_skipped_when_ram_cannot_absorb(self):
        shm = self.mkdir("fake_tmpfs")
        disk = self.mkdir("disk")
        host = Host(ram=3_000, free={str(shm): 1_000_000})
        env = host.env(
            mounts=mountinfo((str(shm), "tmpfs")),
            env={"SLURM_TMPDIR": str(disk)},
            temp_dir=str(self.tmp),
        )
        eng = self.engine(
            environment=env, expected_inflight_bytes=1000
        )  # needs 4000 RAM
        self.assertFalse(eng.is_ramdisk)
        self.assertEqual(eng.path.parent, disk)

    def test_unusable_candidates_are_skipped(self):
        ro_like = self.tmp / "does-not-exist"
        disk = self.mkdir("disk")
        env = Host().env(
            mounts=mountinfo((str(ro_like), "tmpfs")),
            env={"SLURM_TMPDIR": str(disk)},
            temp_dir=str(self.tmp),
        )
        eng = self.engine(environment=env)
        self.assertEqual(eng.path.parent, disk)

    def test_unreadable_proc_still_finds_dev_shm_blind_fallback(self):
        if not os.path.isdir("/dev/shm"):
            self.skipTest("no /dev/shm")
        env = ScratchEnvironment(
            platform="linux", proc_root=self.tmp / "no-proc", temp_dir=str(self.tmp)
        )
        sources = [(c.source, str(c.path)) for c in discover_candidates(None, env)]
        self.assertIn(("tmpfs", "/dev/shm"), sources)


class TestTicketAdmission(TmpCase):
    def test_ticket_basics_and_env(self):
        eng, _host, _shm, _disk = self.ram_engine()
        with eng.reserve(2_000) as t:
            self.assertTrue(t.path.is_dir())
            self.assertEqual(t.tier, "ramdisk")
            self.assertTrue(t.is_ramdisk)
            self.assertEqual(t.path.parent, eng.ram_path)
            env = eng.get_subprocess_env(t)
            self.assertEqual(env["TMPDIR"], str(t.path))
            self.assertEqual(eng.get_subprocess_env()["TMPDIR"], str(eng.path))
            self.assertEqual(eng.stats().ramdisk_bytes_reserved, 2_000)
        self.assertFalse(t.path.exists())  # unlinked immediately on exit
        self.assertTrue(t.released)
        self.assertEqual(eng.stats().bytes_reserved, 0)

    def test_budget_exhaustion_falls_back_to_disk_per_ticket_and_recovers(self):
        eng, _host, _shm, _disk = self.ram_engine()  # budget 9000
        a = eng.reserve(6_000)
        self.assertEqual(a.tier, "ramdisk")
        b = eng.reserve(4_000)  # 6000 + 4000 > 9000 -> disk
        self.assertEqual(b.tier, "disk")
        self.assertEqual(b.path.parent, eng.disk_path)
        c = eng.reserve(3_000)  # 6000 + 3000 == 9000 -> still fits RAM
        self.assertEqual(c.tier, "ramdisk")
        st = eng.stats()
        self.assertEqual(
            (st.ramdisk_tickets, st.disk_tickets, st.fallback_count), (2, 1, 1)
        )
        self.assertEqual(st.ramdisk_bytes_reserved, 9_000)
        self.assertEqual(st.bytes_reserved, 13_000)
        self.assertEqual(st.active_tickets, 3)
        a.release()  # budget returns immediately
        d = eng.reserve(5_000)
        self.assertEqual(d.tier, "ramdisk")
        for t in (b, c, d):
            t.release()
        st = eng.stats()
        self.assertEqual(
            (st.active_tickets, st.bytes_reserved, st.ramdisk_bytes_reserved), (0, 0, 0)
        )
        self.assertEqual(st.ramdisk_peak_bytes, 9_000)
        self.assertEqual(st.peak_bytes_reserved, 13_000)
        self.assertEqual(st.total_bytes_requested, 6_000 + 4_000 + 3_000 + 5_000)
        self.assertEqual(list(eng.ram_path.iterdir()), [])  # type: ignore[union-attr]
        self.assertEqual(list(eng.disk_path.iterdir()), [])  # type: ignore[union-attr]

    def test_oversize_ticket_goes_to_disk(self):
        eng, _host, _shm, _disk = self.ram_engine()
        with eng.reserve(10**9) as t:
            self.assertEqual(t.tier, "disk")
        self.assertEqual(eng.stats().fallback_count, 1)

    def test_low_ram_falls_back_then_recovers(self):
        eng, host, _shm, _disk = self.ram_engine(free=1_000_000)
        with eng.reserve(500) as t:
            self.assertEqual(t.tier, "ramdisk")
        host.ram = HEADROOM + 100  # not enough RAM to absorb 500 more bytes
        with eng.reserve(500) as t:
            self.assertEqual(t.tier, "disk")
        host.ram = 8 * GIB
        with eng.reserve(500) as t:
            self.assertEqual(t.tier, "ramdisk")
        self.assertEqual(eng.stats().fallback_count, 1)

    def test_live_free_space_check_protects_against_other_processes(self):
        eng, host, shm, _disk = self.ram_engine(free=1_000_000)
        host.free[str(shm)] = HEADROOM + 10  # someone else filled the tmpfs
        with eng.reserve(500) as t:
            self.assertEqual(t.tier, "disk")
        host.free[str(shm)] = 1_000_000
        with eng.reserve(500) as t:
            self.assertEqual(t.tier, "ramdisk")

    def test_live_check_can_be_disabled(self):
        eng, host, _shm, _disk = self.ram_engine(free=1_000_000, live_check=False)
        host.ram = 1
        with eng.reserve(500) as t:
            self.assertEqual(t.tier, "ramdisk")

    def test_ramdisk_directory_failure_falls_back_to_disk(self):
        eng, _host, _shm, _disk = self.ram_engine()
        shutil.rmtree(eng.ram_path)  # type: ignore[arg-type]
        with eng.reserve(100) as t:
            self.assertEqual(t.tier, "disk")
        st = eng.stats()
        self.assertEqual(
            (st.fallback_count, st.ramdisk_bytes_reserved, st.ramdisk_tickets),
            (1, 0, 0),
        )

    def test_engine_without_ram_tier_uses_disk_without_counting_fallbacks(self):
        disk = self.mkdir("disk")
        env = Host().env(
            mounts="", env={"SLURM_TMPDIR": str(disk)}, temp_dir=str(self.tmp)
        )
        eng = self.engine(environment=env)
        with eng.reserve(10**6) as t:
            self.assertEqual(t.tier, "disk")
            self.assertFalse(t.is_ramdisk)
        st = eng.stats()
        self.assertEqual(
            (st.disk_tickets, st.ramdisk_tickets, st.fallback_count), (1, 0, 0)
        )
        self.assertIsNone(st.ram_path)

    def test_release_is_idempotent_and_exception_safe(self):
        eng, _host, _shm, _disk = self.ram_engine()
        with self.assertRaises(RuntimeError):
            with eng.reserve(1_000) as t:
                raise RuntimeError("task failed")
        self.assertFalse(t.path.exists())
        t.release()
        t.release()
        self.assertEqual(eng.stats().ramdisk_bytes_reserved, 0)

    def test_negative_and_zero_byte_tickets(self):
        eng, _host, _shm, _disk = self.ram_engine()
        for n in (0, -5):
            with eng.reserve(n) as t:
                self.assertEqual(t.nbytes, 0)
                self.assertTrue(t.path.is_dir())

    def test_reserve_after_cleanup_raises(self):
        eng, _host, _shm, _disk = self.ram_engine()
        eng.cleanup()
        with self.assertRaises(RuntimeError):
            eng.reserve(1)
        with self.assertRaises(RuntimeError):
            _ = eng.path

    def test_concurrent_reservations_never_oversubscribe_tmpfs(self):
        eng, _host, _shm, _disk = self.ram_engine(free=1_000 + 1_000)  # budget 1000
        budget = eng.ramdisk_budget_bytes
        self.assertEqual(budget, 1_000)
        lock = threading.Lock()
        live = {"ram": 0, "peak": 0}
        errors: list[str] = []
        n_threads, per_thread = 12, 150

        def worker(seed: int) -> None:
            rng = random.Random(seed)
            for _ in range(per_thread):
                n = rng.randint(1, 400)
                t = eng.reserve(n)
                try:
                    if not t.path.is_dir():
                        errors.append("missing dir")
                    if t.is_ramdisk:
                        with lock:
                            live["ram"] += n
                            live["peak"] = max(live["peak"], live["ram"])
                            if live["ram"] > budget:
                                errors.append(
                                    f"oversubscribed {live['ram']} > {budget}"
                                )
                finally:
                    if t.is_ramdisk:
                        with lock:
                            live["ram"] -= n
                    t.release()

        threads = [threading.Thread(target=worker, args=(s,)) for s in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        st = eng.stats()
        self.assertLessEqual(st.ramdisk_peak_bytes, budget)
        self.assertEqual(st.ramdisk_tickets + st.disk_tickets, n_threads * per_thread)
        self.assertEqual(
            (st.active_tickets, st.bytes_reserved, st.ramdisk_bytes_reserved), (0, 0, 0)
        )
        self.assertGreater(st.ramdisk_tickets, 0)
        self.assertGreater(st.fallback_count, 0)  # contention forced some onto disk
        self.assertEqual(list(eng.ram_path.iterdir()), [])  # type: ignore[union-attr]
        self.assertEqual(list(eng.disk_path.iterdir()), [])  # type: ignore[union-attr]

    def test_stats_dict_is_json_friendly(self):
        import json

        eng, _host, _shm, _disk = self.ram_engine()
        with eng.reserve(10):
            pass
        d = eng.stats().as_dict()
        json.dumps(d)
        self.assertEqual(d["ramdisk_tickets"], 1)


class TestCleanup(TmpCase):
    def test_cleanup_idempotent_and_removes_active_tickets(self):
        eng, _host, _shm, _disk = self.ram_engine()
        t = eng.reserve(100)
        ram, disk = eng.ram_path, eng.disk_path
        assert ram is not None and disk is not None
        eng.cleanup()
        eng.cleanup()
        self.assertFalse(ram.exists())
        self.assertFalse(disk.exists())
        self.assertTrue(eng.closed)
        t.release()  # releasing a ticket of a closed engine must not raise

    def test_cleanup_never_raises_when_dirs_are_already_gone(self):
        eng, _host, _shm, _disk = self.ram_engine()
        shutil.rmtree(eng.ram_path)  # type: ignore[arg-type]
        shutil.rmtree(eng.disk_path)  # type: ignore[arg-type]
        eng.cleanup()
        eng.cleanup()

    def test_context_manager(self):
        env = Host().env(mounts="", env={}, temp_dir=str(self.tmp))
        with ScratchEngine(environment=env) as eng:
            run = eng.path
            self.assertTrue(run.is_dir())
        self.assertFalse(run.exists())

    def test_engine_registers_and_unregisters_owned_dirs(self):
        from plastburstalign import ephemeral_engine as ee

        eng, _host, _shm, _disk = self.ram_engine()
        paths = {str(eng.ram_path), str(eng.disk_path)}
        self.assertTrue(paths <= set(ee._REGISTRY.owned))
        eng.cleanup()
        self.assertFalse(paths & set(ee._REGISTRY.owned))

    def test_registry_only_removes_own_pid_and_own_naming_scheme(self):
        reg = _CleanupRegistry()
        mine = self.mkdir(f"pba_{os.getpid()}_deadbeef")
        foreign_pid = self.mkdir("pba_1_cafebabe")  # registered by "another process"
        bad_name = self.mkdir("precious_data")
        with reg._lock:
            reg._owned[str(mine)] = os.getpid()
            reg._owned[str(foreign_pid)] = os.getpid() + 1
            reg._owned[str(bad_name)] = os.getpid()
            reg._owned[str(self.tmp / "pba_1_00000000")] = os.getpid()  # already gone
        (mine / "file").write_text("x")
        reg.cleanup()
        reg.cleanup()  # idempotent
        self.assertFalse(mine.exists())
        self.assertTrue(
            foreign_pid.exists()
        )  # fork children never delete their parent's dirs
        self.assertTrue(bad_name.exists())  # never delete outside our naming scheme

    @needs_signals
    def test_signal_handler_chains_previous_callable_handler(self):
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("main thread only")
        got: list[int] = []
        original = signal.signal(signal.SIGUSR1, lambda s, f: got.append(s))
        reg = _CleanupRegistry()
        try:
            self.assertEqual(
                reg.install_signal_handlers([signal.SIGUSR1]), [signal.SIGUSR1]
            )
            self.assertEqual(
                reg.install_signal_handlers([signal.SIGUSR1]), []
            )  # idempotent
            d = self.mkdir(f"pba_{os.getpid()}_01234567")
            with reg._lock:
                reg._owned[str(d)] = os.getpid()
            signal.raise_signal(signal.SIGUSR1)
            for _ in range(100):
                if got:
                    break
                time.sleep(0.01)
            self.assertEqual(got, [signal.SIGUSR1])  # previous handler was chained
            self.assertTrue(
                d.exists()
            )  # callable handler decides; atexit cleans up later
        finally:
            reg.uninstall_signal_handlers()
            signal.signal(signal.SIGUSR1, original)

    def test_signal_handlers_not_installed_from_worker_threads(self):
        reg = _CleanupRegistry()
        result: list[list[int]] = []
        t = threading.Thread(
            target=lambda: result.append(reg.install_signal_handlers([signal.SIGUSR1]))
        )
        t.start()
        t.join()
        self.assertEqual(result, [[]])

    def test_ignored_signals_stay_ignored(self):
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("main thread only")
        original = signal.signal(signal.SIGUSR2, signal.SIG_IGN)
        reg = _CleanupRegistry()
        try:
            self.assertEqual(reg.install_signal_handlers([signal.SIGUSR2]), [])
            self.assertIs(signal.getsignal(signal.SIGUSR2), signal.SIG_IGN)
        finally:
            signal.signal(signal.SIGUSR2, original)

    def _child(self, body: str) -> subprocess.CompletedProcess[str]:
        script = textwrap.dedent(
            f"""
            import os, signal, sys
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from plastburstalign.ephemeral_engine import ScratchEngine
            eng = ScratchEngine(custom_dir={str(self.tmp)!r}, min_headroom_bytes=1000)
            t = eng.reserve(10)
            print(eng.path, flush=True)
            """
        ) + textwrap.dedent(body)
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

    def test_atexit_removes_directories_of_a_normal_exit(self):
        proc = self._child("")  # exits normally without any cleanup call
        run = Path(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(run.exists())

    @needs_signals
    def test_sigterm_removes_owned_directories_and_dies_by_the_signal(self):
        proc = self._child(
            "signal.raise_signal(signal.SIGTERM)\nimport time; time.sleep(5)"
        )
        run = Path(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(proc.returncode, -signal.SIGTERM, proc.stderr)
        self.assertFalse(run.exists())

    @needs_signals
    def test_sigint_unwinds_as_keyboard_interrupt_and_cleans_up(self):
        proc = self._child(
            "signal.raise_signal(signal.SIGINT)\nimport time; time.sleep(5)"
        )
        run = Path(proc.stdout.strip().splitlines()[-1])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("KeyboardInterrupt", proc.stderr)  # previous handler was chained
        self.assertFalse(run.exists())


class TestStaleSweep(TmpCase):
    def make_dirs(self) -> dict[str, Path]:
        base = self.mkdir("base")
        out = {
            "dead": base / "pba_4000001_abcdef01",
            "alive": base / "pba_4000002_abcdef02",
            "own": base / f"pba_{os.getpid()}_abcdef03",
            "bad_hex": base / "pba_4000003_xyz",
            "legacy": base / "pba_unaligned_abc123",
            "plain": base / "other_dir",
            "dead_nested": base / "pba_4000004_abcdef04",
        }
        for p in out.values():
            p.mkdir()
        (out["dead"] / "t1").mkdir()
        (out["dead"] / "t1" / "f.fa").write_text(">a\nACGT\n")
        (base / "pba_4000005_abcdef05").write_text("a file, not a dir")
        (base / "pba_4000006_abcdef06").symlink_to(
            out["plain"], target_is_directory=True
        )
        out["base"] = base
        return out

    def test_only_dead_matching_directories_are_removed(self):
        d = self.make_dirs()
        alive = {4000002}
        removed = sweep_stale_dirs(d["base"], pid_alive=lambda pid: pid in alive)
        self.assertEqual(
            sorted(Path(r).name for r in removed),
            ["pba_4000001_abcdef01", "pba_4000004_abcdef04"],
        )
        self.assertFalse(d["dead"].exists())
        self.assertFalse(d["dead_nested"].exists())
        for keep in ("alive", "own", "bad_hex", "legacy", "plain"):
            self.assertTrue(d[keep].exists(), keep)
        self.assertTrue((d["base"] / "pba_4000005_abcdef05").exists())  # file
        self.assertTrue((d["base"] / "pba_4000006_abcdef06").is_symlink())  # symlink
        self.assertTrue(d["plain"].exists())  # symlink target untouched

    def test_live_process_is_never_swept_with_the_real_probe(self):
        base = self.mkdir("base2")
        parent = base / f"pba_{os.getppid()}_abcdef07"
        parent.mkdir()
        removed = sweep_stale_dirs(base)  # real psutil/os.kill probe
        self.assertEqual(removed, [])
        self.assertTrue(parent.exists())

    def test_foreign_owned_directories_are_never_removed(self):
        if not hasattr(os, "getuid"):
            self.skipTest("POSIX only")
        d = self.make_dirs()
        with mock.patch(
            "plastburstalign.ephemeral_engine.os.getuid", return_value=os.getuid() + 1
        ):
            removed = sweep_stale_dirs(d["base"], pid_alive=lambda pid: False)
        self.assertEqual(removed, [])
        self.assertTrue(d["dead"].exists())

    def test_min_age_protects_recent_directories(self):
        d = self.make_dirs()
        old = time.time() - 7200
        os.utime(d["dead"], (old, old))
        removed = sweep_stale_dirs(
            d["base"], pid_alive=lambda pid: False, min_age_seconds=3600
        )
        self.assertEqual([Path(r).name for r in removed], ["pba_4000001_abcdef01"])
        self.assertTrue(d["dead_nested"].exists())  # too fresh

    def test_missing_base_is_fine(self):
        self.assertEqual(sweep_stale_dirs(self.tmp / "nope"), [])

    def test_engine_sweeps_its_tier_bases_on_construction(self):
        base = self.mkdir("custom_base")
        stale = base / "pba_4000001_abcdef01"
        stale.mkdir()
        (stale / "x").write_text("junk")
        live = base / "pba_4000002_abcdef02"
        live.mkdir()
        host = Host()
        host.alive = {4000002}
        env = host.env(mounts="", env={}, temp_dir=str(self.tmp))
        eng = self.engine(base, environment=env, stale_min_age_seconds=0.0)
        self.assertFalse(stale.exists())
        self.assertTrue(live.exists())
        self.assertEqual(eng.stats().stale_swept, 1)
        no_sweep = self.mkdir("other_base")
        stale2 = no_sweep / "pba_4000001_abcdef09"
        stale2.mkdir()
        self.engine(no_sweep, environment=env, sweep_stale=False)
        self.assertTrue(stale2.exists())


class TestLegacyManagerCompatibility(TmpCase):
    def test_default_manager_allocates_and_cleans_up(self):
        mgr = EphemeralScratchManager()
        path = mgr.path
        self.assertTrue(path.is_dir())
        self.assertEqual(mgr.get_subprocess_env()["TMPDIR"], str(path))
        self.assertEqual(mgr.scratch_path, path)
        self.assertEqual(mgr.min_headroom_bytes, DEFAULT_MIN_HEADROOM_BYTES)
        self.assertIsInstance(mgr.is_ramdisk, bool)
        mgr.cleanup()
        mgr.cleanup()
        self.assertFalse(path.exists())
        with self.assertRaises(RuntimeError):
            _ = mgr.path

    def test_positional_arguments_and_custom_dir(self):
        custom = self.mkdir("custom")
        with EphemeralScratchManager(custom, 123) as mgr:
            self.assertEqual(mgr.min_headroom_bytes, 123)
            self.assertEqual(mgr.path.parent, custom)
            self.assertTrue(mgr.path.name.startswith(f"pba_{os.getpid()}_"))
            self.assertFalse(mgr.is_ramdisk)
        self.assertFalse(mgr.scratch_path.exists())  # type: ignore[union-attr]

    def test_custom_dir_on_tmpfs_reports_ramdisk(self):
        shm = self.mkdir("fake_tmpfs")
        env = Host().env(
            mounts=mountinfo((str(shm), "tmpfs")), env={}, temp_dir=str(self.tmp)
        )
        with EphemeralScratchManager(shm, environment=env) as mgr:
            self.assertTrue(mgr.is_ramdisk)

    def test_manager_exposes_the_engine_for_ticket_admission(self):
        env = Host().env(mounts="", env={}, temp_dir=str(self.tmp))
        with EphemeralScratchManager(environment=env) as mgr:
            with mgr.engine.reserve(10) as t:
                self.assertTrue(t.path.is_dir())


if __name__ == "__main__":
    unittest.main()
