"""Host hardware profiling for PlastBurstAlign (autonomic profiler).

:func:`detect_host` inspects the machine (or a *fixture tree* standing in for
it) and returns an immutable :class:`HostProfile` describing everything the
scheduler and the auto-tuner need to size a run:

* **CPU**: logical vs. *physical* cores, the *usable* CPU set (OS affinity
  intersected with cgroup v1/v2 ``cpuset``/``cpu.max``/``cfs_quota`` and SLURM
  allocation variables) expressed as a :class:`~plastburstalign.topology.CpuTopology`,
  NUMA nodes, SIMD flags.
* **Memory**: total/available RAM, swap, cgroup v1/v2 limits resolved through
  ``/proc/self/cgroup`` (``memory.high`` respected as a soft hint), SLURM
  memory hints and the resulting *effective* RAM.
* **Storage**: tmpfs mounts, ``$SLURM_TMPDIR``/``$TMPDIR``/``$PBA_SCRATCH`` and
  local scratch candidates annotated with filesystem type (local vs. network)
  and device class (NVMe/SSD/HDD).
* **Container hints**: Docker/Podman/Kubernetes/Apptainer.

Design rules
------------
Every probe is individually guarded: a missing or malformed file never raises,
the affected field degrades to ``None``/a default and a message is appended to
``HostProfile.warnings``. All parsers read through an injectable file ``root``
(so tests use fixture trees for fake ``/proc``, ``/sys`` and cgroup files) and an
injectable environment mapping. The live (default) detection is cached; pass
``refresh=True`` to re-measure (memory availability in particular is volatile).
"""

from __future__ import annotations

import dataclasses
import math
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .logging_ops import logger as log
from .topology import CoreInfo, CpuTopology

__all__ = [
    "HostProfile",
    "MemoryInfo",
    "NumaNodeInfo",
    "SimdInfo",
    "StorageTier",
    "clear_host_cache",
    "classify_fstype",
    "detect_host",
    "format_bytes",
    "parse_cpu_list",
    "parse_size",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: cgroup "unlimited" sentinels are huge (v1: 9223372036854771712). Anything at
#: or above this value is treated as "no limit".
_UNLIMITED_THRESHOLD = 1 << 62

_RAM_FSTYPES = frozenset({"tmpfs", "ramfs"})
_NETWORK_FSTYPES = frozenset(
    {
        "nfs",
        "nfs3",
        "nfs4",
        "lustre",
        "gpfs",
        "beegfs",
        "cifs",
        "smb3",
        "smbfs",
        "ceph",
        "glusterfs",
        "panfs",
        "afs",
        "orangefs",
        "9p",
        "vboxsf",
        "virtiofs",
        "wekafs",
    }
)
_LOCAL_FSTYPES = frozenset(
    {
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "f2fs",
        "zfs",
        "jfs",
        "reiserfs",
        "ntfs",
        "ntfs3",
        "vfat",
        "exfat",
        "apfs",
        "hfs",
        "hfsplus",
        "overlay",
        "fuseblk",
        "bcachefs",
    }
)

#: tmpfs mounts below these prefixes are kernel/system plumbing, not scratch.
_IGNORED_TMPFS_PREFIXES = (
    "/sys",
    "/proc",
    "/run/lock",
    "/run/credentials",
    "/run/snapd",
    "/run/qemu",
    "/var/lib/docker",
)
#: ``/run`` itself is the system runtime directory (root-owned); per-user
#: runtime dirs below it (``/run/user/<uid>``) remain valid scratch candidates.
_IGNORED_TMPFS_EXACT = ("/run",)
_MIN_TMPFS_BYTES = 1 << 20  # ignore tmpfs mounts smaller than 1 MiB

_STANDARD_SCRATCH_CANDIDATES = (
    "/tmp",
    "/var/tmp",
    "/scratch",
    "/local",
    "/localscratch",
    "/lscratch",
)

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1 << 10,
    "kb": 1 << 10,
    "kib": 1 << 10,
    "m": 1 << 20,
    "mb": 1 << 20,
    "mib": 1 << 20,
    "g": 1 << 30,
    "gb": 1 << 30,
    "gib": 1 << 30,
    "t": 1 << 40,
    "tb": 1 << 40,
    "tib": 1 << 40,
}


# --------------------------------------------------------------------------- #
# Small public helpers
# --------------------------------------------------------------------------- #


def parse_cpu_list(text: str) -> tuple[int, ...]:
    """Parse a kernel CPU list such as ``"0-3,8,10-11"`` into sorted unique ids.

    Raises:
        ValueError: If a token is not an integer or a ``lo-hi`` range.
    """
    cpus: set[int] = set()
    for token in text.strip().split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo or lo < 0:
                raise ValueError(f"invalid CPU range {token!r}")
            cpus.update(range(lo, hi + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError(f"invalid CPU id {token!r}")
            cpus.add(value)
    return tuple(sorted(cpus))


def parse_size(text: str | int | float, default_unit: str = "") -> int:
    """Parse a human size (``"8G"``, ``"512MiB"``, ``"1.5 GB"``, ``1024``) to bytes.

    Units are binary (1 G = 1024**3), matching how SLURM and cgroups report
    memory. A bare number uses ``default_unit`` (bytes unless given).

    Raises:
        ValueError: If the text cannot be parsed or is negative.
    """
    if isinstance(text, bool):
        raise ValueError(f"invalid size {text!r}")
    if isinstance(text, (int, float)):
        if text < 0 or not math.isfinite(text):
            raise ValueError(f"invalid size {text!r}")
        return int(text * _SIZE_UNITS.get(default_unit.lower(), 1))
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*", str(text))
    if not match:
        raise ValueError(f"invalid size {text!r}")
    number = float(match.group(1))
    unit = match.group(2).lower() or default_unit.lower()
    if unit not in _SIZE_UNITS:
        raise ValueError(f"unknown size unit in {text!r}")
    return int(number * _SIZE_UNITS[unit])


def format_bytes(n: int | float | None) -> str:
    """Format a byte count for humans (``"14.9 GiB"``); ``None`` -> ``"unknown"``."""
    if n is None:
        return "unknown"
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"  # pragma: no cover - loop always returns


def classify_fstype(fstype: str | None) -> str:
    """Classify a filesystem type as ``"ram"``, ``"network"``, ``"local"`` or ``"unknown"``."""
    if not fstype:
        return "unknown"
    name = fstype.lower()
    if name in _RAM_FSTYPES:
        return "ram"
    if name in _NETWORK_FSTYPES:
        return "network"
    if name.startswith("fuse.") and name != "fuse.fuseblk":
        return "network"  # sshfs, s3fs, glusterfs, ... are remote by nature
    if name in _LOCAL_FSTYPES:
        return "local"
    return "unknown"


# --------------------------------------------------------------------------- #
# Public data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NumaNodeInfo:
    """One NUMA node: its OS CPU ids (all online CPUs, not only usable ones)."""

    node_id: int
    cpus: tuple[int, ...]
    mem_total_bytes: int | None = None


@dataclass(frozen=True)
class SimdInfo:
    """SIMD capability summary (from ``/proc/cpuinfo`` flags or the machine type)."""

    machine: str
    known: bool = False
    sse4_2: bool = False
    avx: bool = False
    avx2: bool = False
    avx512f: bool = False
    fma: bool = False
    neon: bool = False

    @property
    def level(self) -> str:
        """Best vector tier: ``avx512``/``avx2``/``avx``/``sse4.2``/``neon``/``none``/``unknown``."""
        if self.avx512f:
            return "avx512"
        if self.avx2:
            return "avx2"
        if self.avx:
            return "avx"
        if self.sse4_2:
            return "sse4.2"
        if self.neon:
            return "neon"
        return "none" if self.known else "unknown"


@dataclass(frozen=True)
class MemoryInfo:
    """Memory picture of the host and of the limits applied to this process.

    ``effective_bytes`` is ``min(host available, cgroup headroom, SLURM limit)``
    over the values that could be determined (``None`` if nothing is known).
    """

    total_bytes: int | None = None
    available_bytes: int | None = None
    swap_total_bytes: int | None = None
    swap_free_bytes: int | None = None
    cgroup_limit_bytes: int | None = None
    cgroup_high_bytes: int | None = None
    cgroup_usage_bytes: int | None = None
    cgroup_headroom_bytes: int | None = None
    slurm_limit_bytes: int | None = None
    effective_bytes: int | None = None
    limit_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class StorageTier:
    """A candidate scratch location with its filesystem and device characteristics."""

    path: str
    kind: str  # "ram" | "local" | "network" | "unknown"
    fstype: str | None = None
    mount_point: str | None = None
    device: str | None = None
    device_class: str = (
        "unknown"  # "ram" | "nvme" | "ssd" | "hdd" | "network" | "unknown"
    )
    rotational: bool | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    writable: bool | None = None
    labels: tuple[str, ...] = ()

    @property
    def is_ram(self) -> bool:
        return self.kind == "ram"

    @property
    def is_network(self) -> bool:
        return self.kind == "network"


@dataclass(frozen=True)
class HostProfile:
    """Immutable snapshot of the host as seen by this process.

    ``topology`` describes only the *usable* CPUs (see module docstring); host
    wide counts are kept separately for reporting.
    """

    system: str
    machine: str
    topology: CpuTopology
    host_logical_cpus: int | None
    host_physical_cores: int | None
    affinity_cpus: tuple[int, ...]
    cpu_quota: float | None = None
    slurm_cpu_limit: int | None = None
    cpu_limit_notes: tuple[str, ...] = ()
    numa_nodes: tuple[NumaNodeInfo, ...] = ()
    simd: SimdInfo = dataclasses.field(default_factory=lambda: SimdInfo(machine=""))
    memory: MemoryInfo = dataclasses.field(default_factory=MemoryInfo)
    storage: tuple[StorageTier, ...] = ()
    containers: tuple[str, ...] = ()
    cgroup_version: int | None = None
    slurm_job: bool = False
    warnings: tuple[str, ...] = ()
    root: str = "/"
    timestamp: float = 0.0

    @property
    def usable_logical(self) -> int:
        return self.topology.logical_count

    @property
    def usable_physical(self) -> int:
        return self.topology.physical_count

    @property
    def ram_tiers(self) -> tuple[StorageTier, ...]:
        return tuple(t for t in self.storage if t.is_ram)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary (for run manifests / ``--doctor --json``)."""
        return dataclasses.asdict(self)

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        topo = self.topology
        lines = [
            f"Host: {self.system} {self.machine}"
            + (f" [{', '.join(self.containers)}]" if self.containers else "")
            + (" [SLURM job]" if self.slurm_job else ""),
            f"CPU: host {_fmt_opt(self.host_logical_cpus)} logical / "
            f"{_fmt_opt(self.host_physical_cores)} physical; usable "
            f"{topo.logical_count} logical / {topo.physical_count} physical "
            f"(SMT width {topo.smt_width})",
        ]
        if self.cpu_limit_notes:
            lines.append("CPU limits: " + "; ".join(self.cpu_limit_notes))
        if self.numa_nodes:
            node_txt = ", ".join(
                f"node{n.node_id}:{len(n.cpus)} cpus" for n in self.numa_nodes
            )
            lines.append(f"NUMA: {len(self.numa_nodes)} node(s) ({node_txt})")
        else:
            lines.append("NUMA: unknown (assuming 1 node)")
        lines.append(f"SIMD: {self.simd.level}")
        m = self.memory
        mem = (
            f"Memory: total {format_bytes(m.total_bytes)}, available "
            f"{format_bytes(m.available_bytes)}, effective "
            f"{format_bytes(m.effective_bytes)}"
        )
        if m.cgroup_limit_bytes is not None:
            mem += f", cgroup limit {format_bytes(m.cgroup_limit_bytes)}"
        if m.slurm_limit_bytes is not None:
            mem += f", SLURM limit {format_bytes(m.slurm_limit_bytes)}"
        lines.append(mem)
        for tier in self.storage:
            lines.append(
                f"Storage: {tier.path} [{tier.kind}/{tier.device_class}, "
                f"{tier.fstype or '?'}] free {format_bytes(tier.free_bytes)}"
            )
        for w in self.warnings:
            lines.append(f"Warning: {w}")
        return "\n".join(lines)


def _fmt_opt(value: int | None) -> str:
    return "?" if value is None else str(value)


# --------------------------------------------------------------------------- #
# Probe context and low-level readers
# --------------------------------------------------------------------------- #


@dataclass
class _Ctx:
    """Mutable detection context shared by the probes."""

    root: str
    env: Mapping[str, str]
    system: str
    machine: str
    sysctl: Callable[[str], str | None]
    disk_usage: Callable[[str], tuple[int, int, int]]
    use_psutil: bool
    affinity_override: tuple[int, ...] | None
    warnings: list[str]

    @property
    def is_live(self) -> bool:
        return os.path.abspath(self.root) == os.path.abspath(os.sep)

    def path(self, p: str) -> str:
        return os.path.join(self.root, p.lstrip("/"))

    def read(self, p: str) -> str | None:
        """Read a text file below the root; ``None`` if missing/unreadable."""
        try:
            with open(self.path(p), encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except (OSError, ValueError):
            return None

    def listdir(self, p: str) -> list[str]:
        try:
            return sorted(os.listdir(self.path(p)))
        except OSError:
            return []

    def isdir(self, p: str) -> bool:
        return os.path.isdir(self.path(p))

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            log.debug(f"hardware: {message}")


def _psutil_module(ctx: _Ctx) -> Any | None:
    if not ctx.use_psutil:
        return None
    try:
        import psutil

        return psutil
    except ImportError:  # pragma: no cover - psutil is a hard dependency
        return None


def _int_or_none(text: str | None) -> int | None:
    """Parse an int; ``max``/``-1``/``unlimited``/huge sentinels become ``None``."""
    if text is None:
        return None
    token = text.strip().lower()
    if not token or token in {"max", "unlimited", "-1"}:
        return None
    try:
        value = int(token)
    except ValueError:
        return None
    if value < 0 or value >= _UNLIMITED_THRESHOLD:
        return None
    return value


def _default_sysctl(name: str) -> str | None:
    """Run ``sysctl -n <name>`` (macOS/BSD); ``None`` on any failure."""
    try:
        proc = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _make_disk_usage(root: str) -> Callable[[str], tuple[int, int, int]]:
    def _usage(path: str) -> tuple[int, int, int]:
        live = os.path.abspath(root) == os.path.abspath(os.sep)
        real = path if live else os.path.join(root, path.lstrip("/"))
        usage = shutil.disk_usage(real)
        return usage.total, usage.used, usage.free

    return _usage


def _sysctl_int(ctx: _Ctx, name: str) -> int | None:
    try:
        text = ctx.sysctl(name)
    except Exception as exc:  # injected callables may misbehave
        ctx.warn(f"sysctl {name} failed: {exc}")
        return None
    if text is None:
        return None
    try:
        return int(str(text).strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# CPU: online set, affinity, cgroup cpuset
# --------------------------------------------------------------------------- #


def _detect_online_cpus(ctx: _Ctx) -> tuple[int, ...]:
    text = ctx.read("/sys/devices/system/cpu/online")
    if text is not None:
        try:
            cpus = parse_cpu_list(text)
            if cpus:
                return cpus
        except ValueError:
            ctx.warn("could not parse /sys/devices/system/cpu/online")
    names = ctx.listdir("/sys/devices/system/cpu")
    ids = sorted(int(m.group(1)) for n in names if (m := re.fullmatch(r"cpu(\d+)", n)))
    if ids:
        return tuple(ids)
    return ()


def _count_logical_fallback(ctx: _Ctx) -> int | None:
    if ctx.system == "Darwin":
        n = _sysctl_int(ctx, "hw.logicalcpu")
        if n:
            return n
    ps = _psutil_module(ctx)
    if ps is not None:
        try:
            n = ps.cpu_count(logical=True)
            if n:
                return int(n)
        except Exception as exc:
            ctx.warn(f"psutil.cpu_count failed: {exc}")
    n = os.cpu_count()
    return int(n) if n else None


def _count_physical_fallback(ctx: _Ctx) -> int | None:
    if ctx.system == "Darwin":
        n = _sysctl_int(ctx, "hw.physicalcpu")
        if n:
            return n
    ps = _psutil_module(ctx)
    if ps is not None:
        try:
            n = ps.cpu_count(logical=False)
            if n:
                return int(n)
        except Exception as exc:
            ctx.warn(f"psutil.cpu_count(logical=False) failed: {exc}")
    return None


def _detect_affinity(ctx: _Ctx, online: tuple[int, ...]) -> tuple[int, ...]:
    """OS CPU affinity of this process (falls back to the online set)."""
    if ctx.affinity_override is not None:
        return ctx.affinity_override
    if ctx.is_live and hasattr(os, "sched_getaffinity"):
        try:
            cpus = tuple(sorted(os.sched_getaffinity(0)))
            if cpus:
                return cpus
        except OSError as exc:
            ctx.warn(f"sched_getaffinity failed: {exc}")
    status = ctx.read("/proc/self/status")
    if status:
        match = re.search(r"^Cpus_allowed_list:\s*(\S+)", status, re.MULTILINE)
        if match:
            try:
                cpus = parse_cpu_list(match.group(1))
                if cpus:
                    return cpus
            except ValueError:
                ctx.warn("could not parse Cpus_allowed_list")
    ps = _psutil_module(ctx)
    if ps is not None and ctx.is_live:
        try:
            cpus = tuple(sorted(ps.Process().cpu_affinity()))
            if cpus:
                return cpus
        except Exception:  # not implemented on macOS, etc.
            pass
    if online:
        return online
    n = _count_logical_fallback(ctx)
    if n:
        ctx.warn("no CPU topology files found; using logical CPU count fallback")
        return tuple(range(n))
    ctx.warn("could not determine CPU count; assuming 1 CPU")
    return (0,)


# --------------------------------------------------------------------------- #
# cgroups
# --------------------------------------------------------------------------- #


class _Cgroup:
    """Resolve this process's cgroup directories below an injectable root."""

    def __init__(self, ctx: _Ctx) -> None:
        self.ctx = ctx
        self.version: int | None = None
        self._paths: dict[str, str] = {}  # controller (v1) or "" (v2) -> rel path
        text = ctx.read("/proc/self/cgroup")
        has_v2_files = ctx.read("/sys/fs/cgroup/cgroup.controllers") is not None
        v1_lines: list[tuple[list[str], str]] = []
        v2_path: str | None = None
        if text:
            for line in text.splitlines():
                parts = line.split(":", 2)
                if len(parts) != 3:
                    continue
                _hier, ctrls, rel = parts
                if ctrls == "":
                    v2_path = rel
                else:
                    v1_lines.append((ctrls.split(","), rel))
        if has_v2_files:
            self.version = 2
            self._paths[""] = v2_path if v2_path is not None else "/"
        elif v1_lines:
            self.version = 1
            for names, v1_rel in v1_lines:
                for c in names:
                    self._paths.setdefault(c, v1_rel)
        elif v2_path is not None and ctx.isdir("/sys/fs/cgroup"):
            # v2 line but no cgroup.controllers (stripped fixture / odd mount)
            self.version = 2
            self._paths[""] = v2_path

    def chain(self, controller: str) -> list[str]:
        """Existing cgroup directories for ``controller``, leaf first, root last."""
        if self.version is None:
            return []
        if self.version == 2:
            rel = self._paths.get("", "/")
            base_candidates = ["/sys/fs/cgroup"]
        else:
            rel = self._paths.get(controller, "/")
            base_candidates = [f"/sys/fs/cgroup/{controller}"]
            if controller == "cpu":
                base_candidates += [
                    "/sys/fs/cgroup/cpu,cpuacct",
                    "/sys/fs/cgroup/cpuacct,cpu",
                ]
        base = next((b for b in base_candidates if self.ctx.isdir(b)), None)
        if base is None:
            return []
        parts = [p for p in rel.strip("/").split("/") if p and p != ".."]
        dirs: list[str] = []
        for k in range(len(parts), -1, -1):
            candidate = "/".join([base.rstrip("/"), *parts[:k]]) if k else base
            if self.ctx.isdir(candidate):
                dirs.append(candidate)
        return dirs

    def read(self, directory: str, name: str) -> str | None:
        return self.ctx.read(f"{directory}/{name}")


def _cgroup_cpu_quota(ctx: _Ctx, cg: _Cgroup) -> float | None:
    """Smallest CPU quota (in CPUs, possibly fractional) along the cgroup chain."""
    best: float | None = None
    for d in cg.chain("cpu"):
        quota: float | None = None
        if cg.version == 2:
            text = cg.read(d, "cpu.max")
            if text:
                parts = text.split()
                if len(parts) >= 1 and parts[0] != "max":
                    try:
                        period = float(parts[1]) if len(parts) > 1 else 100000.0
                        if period > 0:
                            quota = float(parts[0]) / period
                    except ValueError:
                        ctx.warn(f"could not parse {d}/cpu.max")
        else:
            q_txt = cg.read(d, "cpu.cfs_quota_us")
            p_txt = cg.read(d, "cpu.cfs_period_us")
            try:
                if q_txt is not None and int(q_txt.strip()) > 0:
                    period_us = int(p_txt.strip()) if p_txt else 100000
                    if period_us > 0:
                        quota = int(q_txt.strip()) / period_us
            except ValueError:
                ctx.warn(f"could not parse cfs quota in {d}")
        if quota is not None and quota > 0 and (best is None or quota < best):
            best = quota
    return best


def _cgroup_cpuset(ctx: _Ctx, cg: _Cgroup) -> tuple[int, ...] | None:
    """Effective cpuset of the nearest cgroup that defines one."""
    names = (
        ("cpuset.cpus.effective", "cpuset.cpus")
        if cg.version == 2
        else ("cpuset.effective_cpus", "cpuset.cpus")
    )
    for d in cg.chain("cpuset"):
        for name in names:
            text = cg.read(d, name)
            if text is None or not text.strip():
                continue
            try:
                cpus = parse_cpu_list(text)
            except ValueError:
                ctx.warn(f"could not parse {d}/{name}")
                continue
            if cpus:
                return cpus
    return None


def _cgroup_memory(
    ctx: _Ctx, cg: _Cgroup
) -> tuple[int | None, int | None, int | None, int | None]:
    """Return ``(hard_limit, high, usage, headroom)`` along the memory cgroup chain."""
    limit: int | None = None
    high: int | None = None
    usage: int | None = None
    headroom: int | None = None
    for i, d in enumerate(cg.chain("memory")):
        if cg.version == 2:
            lim = _int_or_none(cg.read(d, "memory.max"))
            hi = _int_or_none(cg.read(d, "memory.high"))
            cur = _int_or_none(cg.read(d, "memory.current"))
            stat_key = "inactive_file"
        else:
            lim = _int_or_none(cg.read(d, "memory.limit_in_bytes"))
            hi = _int_or_none(cg.read(d, "memory.soft_limit_in_bytes"))
            cur = _int_or_none(cg.read(d, "memory.usage_in_bytes"))
            stat_key = "total_inactive_file"
        if i == 0 and cur is not None:
            usage = cur
        if lim is not None and (limit is None or lim < limit):
            limit = lim
        if hi is not None and (high is None or hi < high):
            high = hi
        bound = min((x for x in (lim, hi) if x is not None), default=None)
        if bound is not None:
            used = cur if cur is not None else 0
            stat = cg.read(d, "memory.stat")
            if stat and cur is not None:
                m = re.search(rf"^{stat_key}\s+(\d+)", stat, re.MULTILINE)
                if m:  # page cache that can be dropped is not "used" headroom
                    used = max(0, used - int(m.group(1)))
            room = max(0, bound - used)
            headroom = room if headroom is None else min(headroom, room)
    return limit, high, usage, headroom


# --------------------------------------------------------------------------- #
# SLURM
# --------------------------------------------------------------------------- #


def _env_int(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_slurm_cpus_per_node(text: str) -> int | None:
    """Parse ``SLURM_JOB_CPUS_PER_NODE`` (``"32"``, ``"32(x2)"``, ``"16,8(x2)"``).

    For heterogeneous nodes the smallest count is returned (conservative).
    """
    counts: list[int] = []
    for token in text.split(","):
        m = re.match(r"\s*(\d+)", token)
        if m and int(m.group(1)) > 0:
            counts.append(int(m.group(1)))
    return min(counts) if counts else None


def _slurm_cpu_limit(ctx: _Ctx) -> tuple[int | None, str | None]:
    """Smallest CPU count advertised by SLURM variables (``None`` if no SLURM)."""
    candidates: list[tuple[int, str]] = []
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = _env_int(ctx.env, var)
        if value:
            candidates.append((value, var))
    raw = ctx.env.get("SLURM_JOB_CPUS_PER_NODE")
    if raw:
        value = _parse_slurm_cpus_per_node(raw)
        if value:
            candidates.append((value, "SLURM_JOB_CPUS_PER_NODE"))
    if not candidates:
        return None, None
    value, var = min(candidates)
    return value, var


def _slurm_mem_limit(ctx: _Ctx, cpus_hint: int) -> tuple[int | None, str | None]:
    """SLURM memory hint in bytes (``SLURM_MEM_PER_NODE`` / ``SLURM_MEM_PER_CPU``, MB)."""
    raw = ctx.env.get("SLURM_MEM_PER_NODE")
    if raw:
        try:
            return parse_size(raw, "m"), "SLURM_MEM_PER_NODE"
        except ValueError:
            ctx.warn(f"could not parse SLURM_MEM_PER_NODE={raw!r}")
    raw = ctx.env.get("SLURM_MEM_PER_CPU")
    if raw:
        try:
            return parse_size(raw, "m") * max(1, cpus_hint), "SLURM_MEM_PER_CPU"
        except ValueError:
            ctx.warn(f"could not parse SLURM_MEM_PER_CPU={raw!r}")
    return None, None


# --------------------------------------------------------------------------- #
# Topology construction
# --------------------------------------------------------------------------- #


def _sysfs_sibling_groups(
    ctx: _Ctx, cpus: Iterable[int]
) -> dict[int, tuple[int, ...]] | None:
    """Map cpu -> full sibling tuple using sysfs; ``None`` if no data at all."""
    groups: dict[int, tuple[int, ...]] = {}
    pkg_core: dict[int, tuple[int, int]] = {}
    for cpu in cpus:
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        sib = ctx.read(f"{base}/thread_siblings_list") or ctx.read(
            f"{base}/core_cpus_list"
        )
        if sib is not None:
            try:
                parsed = parse_cpu_list(sib)
                if parsed:
                    groups[cpu] = parsed
                    continue
            except ValueError:
                pass
        core = ctx.read(f"{base}/core_id")
        pkg = ctx.read(f"{base}/physical_package_id")
        if core is not None:
            try:
                pkg_core[cpu] = (int(pkg.strip()) if pkg else 0, int(core.strip()))
            except ValueError:
                pass
    if pkg_core:
        by_key: dict[tuple[int, int], list[int]] = {}
        for cpu, key in pkg_core.items():
            by_key.setdefault(key, []).append(cpu)
        for members in by_key.values():
            tup = tuple(sorted(members))
            for cpu in members:
                groups.setdefault(cpu, tup)
    return groups or None


def _numa_info(ctx: _Ctx) -> tuple[NumaNodeInfo, ...]:
    nodes: list[NumaNodeInfo] = []
    for name in ctx.listdir("/sys/devices/system/node"):
        m = re.fullmatch(r"node(\d+)", name)
        if not m:
            continue
        text = ctx.read(f"/sys/devices/system/node/{name}/cpulist")
        if text is None:
            continue
        try:
            cpus = parse_cpu_list(text)
        except ValueError:
            ctx.warn(f"could not parse NUMA cpulist for {name}")
            continue
        mem: int | None = None
        meminfo = ctx.read(f"/sys/devices/system/node/{name}/meminfo")
        if meminfo:
            mm = re.search(r"MemTotal:\s+(\d+)\s*kB", meminfo)
            if mm:
                mem = int(mm.group(1)) * 1024
        nodes.append(NumaNodeInfo(int(m.group(1)), cpus, mem))
    return tuple(sorted(nodes, key=lambda n: n.node_id))


def _layout_groups(
    cpus: tuple[int, ...], physical: int, logical: int, adjacent: bool
) -> dict[int, tuple[int, ...]]:
    """Infer sibling groups from counts only (no sysfs): adjacent or ``cpu % physical``."""
    if physical <= 0 or logical <= physical or logical % physical:
        return {cpu: (cpu,) for cpu in cpus}
    smt = logical // physical
    by_key: dict[int, list[int]] = {}
    for cpu in cpus:
        key = cpu // smt if adjacent else cpu % physical
        by_key.setdefault(key, []).append(cpu)
    out: dict[int, tuple[int, ...]] = {}
    for members in by_key.values():
        tup = tuple(sorted(members))
        for cpu in members:
            out[cpu] = tup
    return out


def _build_cores(
    usable: tuple[int, ...],
    groups: Mapping[int, tuple[int, ...]],
    cpu_to_node: Mapping[int, int],
) -> list[tuple[int, tuple[int, ...]]]:
    """Group usable CPUs into physical cores -> ``[(numa_node, cpu_ids), ...]`` (ordered)."""
    usable_set = set(usable)
    seen: set[tuple[int, ...]] = set()
    cores: list[tuple[int, tuple[int, ...]]] = []
    for cpu in usable:
        full = groups.get(cpu, (cpu,))
        if full in seen:
            continue
        seen.add(full)
        members = tuple(sorted(c for c in full if c in usable_set)) or (cpu,)
        node = cpu_to_node.get(members[0], 0)
        cores.append((node, members))
    cores.sort(key=lambda item: (item[0], item[1][0]))
    return cores


def _spread_trim(
    cores: list[tuple[int, tuple[int, ...]]], cap: int
) -> list[tuple[int, tuple[int, ...]]]:
    """Reduce to ``cap`` logical CPUs preferring one thread on each distinct core first."""
    total = sum(len(c[1]) for c in cores)
    if cap >= total or cap < 1:
        return cores
    chosen: list[list[int]] = [[] for _ in cores]
    remaining = cap
    depth = 0
    while remaining > 0:
        progressed = False
        for idx, (_node, cpu_ids) in enumerate(cores):
            if depth < len(cpu_ids) and remaining > 0:
                chosen[idx].append(cpu_ids[depth])
                remaining -= 1
                progressed = True
        if not progressed:
            break
        depth += 1
    return [(node, tuple(sorted(ch))) for (node, _ids), ch in zip(cores, chosen) if ch]


def _make_topology(cores: list[tuple[int, tuple[int, ...]]]) -> CpuTopology:
    return CpuTopology(
        cores=tuple(
            CoreInfo(core_id=i, cpu_ids=ids, numa_node=node)
            for i, (node, ids) in enumerate(cores)
        )
    )


# --------------------------------------------------------------------------- #
# SIMD
# --------------------------------------------------------------------------- #

_DARWIN_FEATURE_MAP = {
    "sse4.2": "sse4_2",
    "avx1.0": "avx",
    "avx2": "avx2",
    "avx512f": "avx512f",
    "fma": "fma",
}


def _detect_simd(ctx: _Ctx) -> SimdInfo:
    machine = ctx.machine
    flags: set[str] = set()
    known = False
    text = ctx.read("/proc/cpuinfo")
    if text:
        match = re.search(r"^(?:flags|Features)\s*:\s*(.*)$", text, re.MULTILINE)
        if match:
            flags = set(match.group(1).split())
            known = bool(flags)
    if not known and ctx.system == "Darwin":
        names: set[str] = set()
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            try:
                value = ctx.sysctl(key)
            except Exception:
                value = None
            if value:
                names.update(tok.lower() for tok in value.split())
        for tok, flag in _DARWIN_FEATURE_MAP.items():
            if tok in names:
                flags.add(flag)
        known = bool(names)
    arm = machine.lower() in {"arm64", "aarch64"} or machine.lower().startswith("arm")
    neon = ("asimd" in flags) or ("neon" in flags)
    if arm and machine.lower() in {"arm64", "aarch64"}:
        neon = True  # NEON/ASIMD is mandatory on AArch64
        known = True
    return SimdInfo(
        machine=machine,
        known=known,
        sse4_2="sse4_2" in flags,
        avx="avx" in flags,
        avx2="avx2" in flags,
        avx512f="avx512f" in flags,
        fma="fma" in flags,
        neon=neon,
    )


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


def _parse_meminfo(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"(\w[\w()]*):\s+(\d+)(?:\s*kB)?", line)
        if m:
            factor = 1024 if line.rstrip().endswith("kB") else 1
            out[m.group(1)] = int(m.group(2)) * factor
    return out


def _detect_memory(ctx: _Ctx, cg: _Cgroup, cpus_hint: int) -> MemoryInfo:
    total = available = swap_total = swap_free = None
    text = ctx.read("/proc/meminfo")
    if text:
        info = _parse_meminfo(text)
        total = info.get("MemTotal")
        available = info.get("MemAvailable")
        if available is None and total is not None:
            parts = [info.get(k) for k in ("MemFree", "Buffers", "Cached")]
            if all(p is not None for p in parts):
                available = sum(p for p in parts if p is not None)
        swap_total = info.get("SwapTotal")
        swap_free = info.get("SwapFree")
    if total is None and ctx.system == "Darwin":
        total = _sysctl_int(ctx, "hw.memsize")
    if total is None or available is None:
        ps = _psutil_module(ctx)
        if ps is not None:
            try:
                vm = ps.virtual_memory()
                total = total if total is not None else int(vm.total)
                available = available if available is not None else int(vm.available)
                if swap_total is None:
                    sm = ps.swap_memory()
                    swap_total, swap_free = int(sm.total), int(sm.free)
            except Exception as exc:
                ctx.warn(f"psutil memory query failed: {exc}")
        if total is None:
            ctx.warn("could not determine total RAM")
        elif text is None and ctx.system == "Linux":
            ctx.warn("memory figures taken from psutil fallback (no /proc/meminfo)")

    limit, high, usage, headroom = (None, None, None, None)
    try:
        limit, high, usage, headroom = _cgroup_memory(ctx, cg)
    except Exception as exc:
        ctx.warn(f"cgroup memory probe failed: {exc}")
    slurm_bytes, slurm_var = _slurm_mem_limit(ctx, cpus_hint)

    bounds: list[tuple[int, str]] = []
    if available is not None:
        bounds.append((available, "host MemAvailable"))
    if headroom is not None:
        src = (
            "cgroup memory.high headroom"
            if (high is not None and (limit is None or high < limit))
            else "cgroup memory limit headroom"
        )
        bounds.append((headroom, src))
    if slurm_bytes is not None and slurm_var:
        bounds.append((slurm_bytes, slurm_var))
    effective: int | None = None
    sources: tuple[str, ...] = ()
    if bounds:
        effective = min(b[0] for b in bounds)
        sources = tuple(name for value, name in bounds if value == effective)
    return MemoryInfo(
        total_bytes=total,
        available_bytes=available,
        swap_total_bytes=swap_total,
        swap_free_bytes=swap_free,
        cgroup_limit_bytes=limit,
        cgroup_high_bytes=high,
        cgroup_usage_bytes=usage,
        cgroup_headroom_bytes=headroom,
        slurm_limit_bytes=slurm_bytes,
        effective_bytes=effective,
        limit_sources=sources,
    )


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Mount:
    mount_point: str
    fstype: str
    source: str
    majmin: str
    options: tuple[str, ...]
    root: str = "/"


_OCTAL_ESCAPE = re.compile(r"\\([0-7]{3})")


def _unescape_mount(text: str) -> str:
    return _OCTAL_ESCAPE.sub(lambda m: chr(int(m.group(1), 8)), text)


def _parse_mountinfo(text: str) -> list[_Mount]:
    mounts: list[_Mount] = []
    for line in text.splitlines():
        fields = line.split()
        if "-" not in fields[6:]:
            continue
        sep = fields.index("-", 6)
        if len(fields) < sep + 3:
            continue
        mounts.append(
            _Mount(
                mount_point=_unescape_mount(fields[4]),
                fstype=fields[sep + 1],
                source=_unescape_mount(fields[sep + 2]),
                majmin=fields[2],
                options=tuple(fields[5].split(",")),
                root=_unescape_mount(fields[3]),
            )
        )
    return mounts


def _parse_proc_mounts(text: str) -> list[_Mount]:
    mounts: list[_Mount] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4:
            mounts.append(
                _Mount(
                    mount_point=_unescape_mount(fields[1]),
                    fstype=fields[2],
                    source=_unescape_mount(fields[0]),
                    majmin="",
                    options=tuple(fields[3].split(",")),
                )
            )
    return mounts


def _find_mount(mounts: list[_Mount], path: str) -> _Mount | None:
    """Longest-prefix mount containing ``path`` (later duplicates shadow earlier)."""
    norm = os.path.normpath(path)
    best: _Mount | None = None
    best_len = -1
    for mount in mounts:
        mp = os.path.normpath(mount.mount_point)
        inside = (
            mp == os.sep or norm == mp or norm.startswith(mp.rstrip(os.sep) + os.sep)
        )
        if inside and len(mp) >= best_len:
            best, best_len = mount, len(mp)
    return best


_PARTITION_PATTERNS = (
    re.compile(r"^((?:nvme\d+n\d+|mmcblk\d+|loop\d+|md\d+))p\d+$"),
    re.compile(r"^((?:sd|vd|xvd|hd)[a-z]+)\d+$"),
)


def _resolve_block_device(
    ctx: _Ctx, name: str, depth: int = 0
) -> tuple[str, bool | None] | None:
    """Resolve a kernel block device name to ``(physical device, rotational)``."""
    base = name
    if not ctx.isdir(f"/sys/block/{base}"):
        for pattern in _PARTITION_PATTERNS:
            m = pattern.match(name)
            if m:
                base = m.group(1)
                break
    if not ctx.isdir(f"/sys/block/{base}"):
        return None
    if depth < 4:
        for slave in ctx.listdir(f"/sys/block/{base}/slaves"):
            resolved = _resolve_block_device(ctx, slave, depth + 1)
            if resolved is not None:
                return resolved
    rot_txt = ctx.read(f"/sys/block/{base}/queue/rotational")
    rotational: bool | None = None
    if rot_txt is not None and rot_txt.strip() in {"0", "1"}:
        rotational = rot_txt.strip() == "1"
    return base, rotational


def _block_device_for_mount(ctx: _Ctx, mount: _Mount) -> tuple[str, bool | None] | None:
    names: list[str] = []
    if mount.majmin:
        link = ctx.path(f"/sys/dev/block/{mount.majmin}")
        if os.path.exists(link):
            names.append(os.path.basename(os.path.realpath(link)))
    if mount.source.startswith("/dev/"):
        base = os.path.basename(mount.source)
        if mount.source.startswith("/dev/mapper/"):
            for dm in ctx.listdir("/sys/block"):
                if dm.startswith("dm-"):
                    dm_name = ctx.read(f"/sys/block/{dm}/dm/name")
                    if dm_name and dm_name.strip() == base:
                        names.append(dm)
        names.append(base)
    for name in names:
        resolved = _resolve_block_device(ctx, name)
        if resolved is not None:
            return resolved
    return None


def _make_tier(
    ctx: _Ctx,
    path: str,
    mounts: list[_Mount],
    labels: tuple[str, ...],
) -> StorageTier:
    mount = _find_mount(mounts, path)
    fstype = mount.fstype if mount else None
    kind = classify_fstype(fstype)
    device: str | None = None
    rotational: bool | None = None
    if mount is not None and kind not in {"ram", "network"}:
        try:
            resolved = _block_device_for_mount(ctx, mount)
        except Exception as exc:
            ctx.warn(f"block device probe failed for {path}: {exc}")
            resolved = None
        if resolved is not None:
            device, rotational = resolved
    if kind == "ram":
        device_class = "ram"
    elif kind == "network":
        device_class = "network"
    elif rotational is True:
        device_class = "hdd"
    elif device is not None and device.startswith("nvme"):
        device_class = "nvme"
    elif rotational is False:
        device_class = "ssd"
    else:
        device_class = "unknown"
    total = free = None
    try:
        total, _used, free = ctx.disk_usage(path)
    except Exception:
        pass
    writable: bool | None = None
    if mount is not None and "ro" in mount.options:
        writable = False
    elif ctx.is_live:
        writable = os.access(path, os.W_OK)
    return StorageTier(
        path=path,
        kind=kind,
        fstype=fstype,
        mount_point=mount.mount_point if mount else None,
        device=device,
        device_class=device_class,
        rotational=rotational,
        total_bytes=total,
        free_bytes=free,
        writable=writable,
        labels=labels,
    )


def _detect_storage(ctx: _Ctx) -> tuple[StorageTier, ...]:
    text = ctx.read("/proc/self/mountinfo")
    mounts = _parse_mountinfo(text) if text else []
    if not mounts:
        text = ctx.read("/proc/mounts")
        mounts = _parse_proc_mounts(text) if text else []
    paths: dict[str, list[str]] = {}

    def add(path: str, label: str) -> None:
        paths.setdefault(os.path.normpath(path), []).append(label)

    for var in ("SLURM_TMPDIR", "TMPDIR", "PBA_SCRATCH", "TEMP", "TMP"):
        value = ctx.env.get(var)
        if value:
            add(value, f"env:{var}")
    ram_mounts = sorted(
        {
            m.mount_point
            for m in mounts
            if m.fstype in _RAM_FSTYPES
            and m.root == "/"  # skip bind mounts of sub-paths/files
            and not (ctx.is_live and not os.path.isdir(m.mount_point))
            and m.mount_point not in _IGNORED_TMPFS_EXACT
            and not any(
                m.mount_point == p or m.mount_point.startswith(p + "/")
                for p in _IGNORED_TMPFS_PREFIXES
            )
        }
    )
    for mp in ram_mounts:
        add(mp, "tmpfs")
    mount_points = {os.path.normpath(m.mount_point) for m in mounts}
    for cand in _STANDARD_SCRATCH_CANDIDATES:
        if ctx.isdir(cand) or cand in mount_points:
            add(cand, "candidate")
    if not paths and ctx.is_live and ctx.system in {"Windows", "Darwin"}:
        import tempfile

        add(tempfile.gettempdir(), "tempdir")

    tiers: list[StorageTier] = []
    for path, labels in paths.items():
        try:
            tier = _make_tier(ctx, path, mounts, tuple(dict.fromkeys(labels)))
        except Exception as exc:
            ctx.warn(f"storage probe failed for {path}: {exc}")
            continue
        tiny = tier.total_bytes is not None and tier.total_bytes < _MIN_TMPFS_BYTES
        if tier.kind == "ram" and tier.labels == ("tmpfs",) and tiny:
            continue  # kernel plumbing mounts (e.g. 4 KiB /run/lock), not scratch
        tiers.append(tier)
    return tuple(tiers)


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #


def _detect_containers(ctx: _Ctx) -> tuple[str, ...]:
    hints: list[str] = []
    if os.path.exists(ctx.path("/.dockerenv")):
        hints.append("docker")
    if os.path.exists(ctx.path("/run/.containerenv")):
        hints.append("podman")
    if any(ctx.env.get(v) for v in ("APPTAINER_CONTAINER", "APPTAINER_NAME")):
        hints.append("apptainer")
    if any(ctx.env.get(v) for v in ("SINGULARITY_CONTAINER", "SINGULARITY_NAME")):
        hints.append("singularity")
    if ctx.env.get("KUBERNETES_SERVICE_HOST"):
        hints.append("kubernetes")
    for proc_path in ("/proc/1/cgroup", "/proc/self/cgroup"):
        text = ctx.read(proc_path)
        if text:
            lowered = text.lower()
            if "kubepods" in lowered and "kubernetes" not in hints:
                hints.append("kubernetes")
            elif "docker" in lowered and "docker" not in hints:
                hints.append("docker")
            elif "containerd" in lowered and not hints:
                hints.append("containerd")
    return tuple(dict.fromkeys(hints))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

_CACHE_LOCK = threading.Lock()
_LIVE_CACHE: HostProfile | None = None


def clear_host_cache() -> None:
    """Drop the cached live :class:`HostProfile` (the next call re-detects)."""
    global _LIVE_CACHE
    with _CACHE_LOCK:
        _LIVE_CACHE = None


def detect_host(
    root: str | os.PathLike[str] = "/",
    env: Mapping[str, str] | None = None,
    *,
    refresh: bool = False,
    affinity: Iterable[int] | None = None,
    system: str | None = None,
    machine: str | None = None,
    sysctl: Callable[[str], str | None] | None = None,
    disk_usage: Callable[[str], tuple[int, int, int]] | None = None,
    use_psutil: bool = True,
) -> HostProfile:
    """Detect the host and return an immutable :class:`HostProfile`.

    Args:
        root: Filesystem root under which ``/proc`` and ``/sys`` are read. Tests
            pass fixture trees; the default is the live system.
        env: Environment mapping (``os.environ`` by default).
        refresh: Re-run the (cached) live detection. Only the default
            invocation (``root="/"``, no overrides) is cached.
        affinity: Override for the OS CPU affinity set (default: the live
            ``os.sched_getaffinity`` or ``Cpus_allowed_list`` below ``root``).
        system: Override for ``platform.system()`` (e.g. ``"Darwin"``).
        machine: Override for ``platform.machine()`` (e.g. ``"aarch64"``).
        sysctl: Callable ``name -> value`` replacing ``sysctl -n`` (macOS).
        disk_usage: Callable ``path -> (total, used, free)`` replacing
            ``shutil.disk_usage`` (fixture roots have no real filesystems).
        use_psutil: Allow the psutil fallback for missing ``/proc`` data.

    Returns:
        The usable-CPU topology, NUMA/SIMD/memory/storage/container facts and a
        ``warnings`` tuple naming every degraded field. Never raises for
        missing or malformed system files.
    """
    global _LIVE_CACHE
    root_str = os.fspath(root)
    cacheable = (
        os.path.abspath(root_str) == os.path.abspath(os.sep)
        and env is None
        and affinity is None
        and system is None
        and machine is None
        and sysctl is None
        and disk_usage is None
        and use_psutil
    )
    if cacheable and not refresh:
        with _CACHE_LOCK:
            if _LIVE_CACHE is not None:
                return _LIVE_CACHE
    profile = _detect_uncached(
        root_str,
        env if env is not None else os.environ,
        affinity=tuple(sorted(set(affinity))) if affinity is not None else None,
        system=system,
        machine=machine,
        sysctl=sysctl,
        disk_usage=disk_usage,
        use_psutil=use_psutil,
    )
    if cacheable:
        with _CACHE_LOCK:
            _LIVE_CACHE = profile
    return profile


def _detect_uncached(
    root: str,
    env: Mapping[str, str],
    *,
    affinity: tuple[int, ...] | None,
    system: str | None,
    machine: str | None,
    sysctl: Callable[[str], str | None] | None,
    disk_usage: Callable[[str], tuple[int, int, int]] | None,
    use_psutil: bool,
) -> HostProfile:
    warnings_list: list[str] = []
    ctx = _Ctx(
        root=root,
        env=env,
        system=system if system is not None else platform.system(),
        machine=machine if machine is not None else platform.machine(),
        sysctl=sysctl
        if sysctl is not None
        else (
            _default_sysctl
            if os.path.abspath(root) == os.path.abspath(os.sep)
            else (lambda _: None)
        ),
        disk_usage=disk_usage if disk_usage is not None else _make_disk_usage(root),
        use_psutil=use_psutil,
        affinity_override=affinity,
        warnings=warnings_list,
    )

    def guarded(label: str, fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception as exc:  # never crash on odd hosts
            ctx.warn(f"{label} probe failed: {type(exc).__name__}: {exc}")
            return default

    online: tuple[int, ...] = guarded(
        "online-cpus", lambda: _detect_online_cpus(ctx), ()
    )
    aff: tuple[int, ...] = guarded(
        "cpu-affinity",
        lambda: _detect_affinity(ctx, online),
        tuple(range(os.cpu_count() or 1)),
    )
    cg: _Cgroup = guarded("cgroup", lambda: _Cgroup(ctx), None) or _EmptyCgroup(ctx)

    notes: list[str] = []
    usable = aff
    if online:
        inter = tuple(c for c in usable if c in set(online))
        if inter:
            usable = inter
    notes.append(f"affinity: {len(usable)}/{len(online) or '?'} CPUs")
    cpuset = guarded("cgroup-cpuset", lambda: _cgroup_cpuset(ctx, cg), None)
    if cpuset:
        inter = tuple(c for c in usable if c in set(cpuset))
        if inter and len(inter) < len(usable):
            notes.append(f"cgroup cpuset: {len(inter)} CPUs")
            usable = inter
        elif not inter:
            ctx.warn("cgroup cpuset does not intersect the affinity mask; ignoring it")

    # Topology (physical cores)
    host_physical: int | None = None
    groups: dict[int, tuple[int, ...]] | None = guarded(
        "sysfs-topology",
        lambda: _sysfs_sibling_groups(ctx, online or usable),
        None,
    )
    host_logical: int | None = len(online) if online else None
    if groups is not None:
        host_physical = len({g for g in groups.values()})
    else:
        host_logical = host_logical or guarded(
            "logical-count", lambda: _count_logical_fallback(ctx), None
        )
        host_physical = guarded(
            "physical-count", lambda: _count_physical_fallback(ctx), None
        )
        if host_physical is None:
            ctx.warn("physical core count unknown; assuming no SMT")
            host_physical = host_logical
        elif ctx.system == "Linux":  # missing sysfs is normal on macOS/Windows
            ctx.warn(
                "CPU topology files missing; sibling layout inferred from core counts"
            )
        groups = _layout_groups(
            usable,
            host_physical or len(usable),
            host_logical or len(usable),
            adjacent=ctx.system in {"Darwin", "Windows"},
        )
    numa: tuple[NumaNodeInfo, ...] = guarded("numa", lambda: _numa_info(ctx), ())
    cpu_to_node: dict[int, int] = {}
    for node in numa:
        for cpu in node.cpus:
            cpu_to_node[cpu] = node.node_id
    cores = _build_cores(usable, groups, cpu_to_node)

    # Fractional cgroup quota and SLURM caps (counts of logical CPUs)
    quota = guarded("cgroup-cpu-quota", lambda: _cgroup_cpu_quota(ctx, cg), None)
    slurm_cap, slurm_var = guarded(
        "slurm-cpus", lambda: _slurm_cpu_limit(ctx), (None, None)
    )
    caps: list[int] = []
    if quota is not None:
        quota_cpus = max(1, math.floor(quota + 1e-9))
        caps.append(quota_cpus)
        notes.append(f"cgroup cpu quota: {quota:g} CPUs -> {quota_cpus}")
    if slurm_cap is not None:
        caps.append(slurm_cap)
        notes.append(f"{slurm_var}={slurm_cap}")
    if caps:
        cores = _spread_trim(cores, min(caps))
    topology = _make_topology(cores)
    if topology.physical_count == 0:  # defensive: never an empty topology
        topology = CpuTopology.synthetic(1)
        ctx.warn("empty usable CPU set; assuming a single core")

    simd = guarded("simd", lambda: _detect_simd(ctx), SimdInfo(machine=ctx.machine))
    memory: MemoryInfo = guarded(
        "memory",
        lambda: _detect_memory(ctx, cg, slurm_cap or topology.logical_count),
        MemoryInfo(),
    )
    storage: tuple[StorageTier, ...] = guarded(
        "storage", lambda: _detect_storage(ctx), ()
    )
    containers: tuple[str, ...] = guarded(
        "containers", lambda: _detect_containers(ctx), ()
    )
    slurm_job = any(k.startswith("SLURM_") for k in env) and bool(
        env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID") or env.get("SLURM_STEP_ID")
    )

    return HostProfile(
        system=ctx.system,
        machine=ctx.machine,
        topology=topology,
        host_logical_cpus=host_logical,
        host_physical_cores=host_physical,
        affinity_cpus=aff,
        cpu_quota=quota,
        slurm_cpu_limit=slurm_cap,
        cpu_limit_notes=tuple(notes),
        numa_nodes=numa,
        simd=simd,
        memory=memory,
        storage=storage,
        containers=containers,
        cgroup_version=cg.version,
        slurm_job=slurm_job,
        warnings=tuple(warnings_list),
        root=root,
        timestamp=time.time(),
    )


class _EmptyCgroup(_Cgroup):
    """A cgroup view with no resolvable directories (probe failure fallback)."""

    def __init__(self, ctx: _Ctx) -> None:
        self.ctx = ctx
        self.version = None
        self._paths = {}
