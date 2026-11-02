"""Build fake ``/proc`` + ``/sys`` + cgroup trees for hardware-profiler tests.

Large CPU layouts (a 128-thread dual-socket EPYC would need ~400 tiny files) are
generated into a temporary directory by :func:`build_host_tree`; small,
hand-written trees live as static fixtures under ``tests/fixtures/hostfs/``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hostfs"

X86_FLAGS_AVX2 = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 "
    "clflush mmx fxsr sse sse2 ht syscall nx lm constant_tsc rep_good nopl "
    "pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx "
    "f16c rdrand lahf_lm bmi1 avx2 bmi2"
)
X86_FLAGS_AVX512 = X86_FLAGS_AVX2 + " avx512f avx512dq avx512bw avx512vl"
ARM_FEATURES = "fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics cpuid"

SENTINEL_V1_UNLIMITED = 9223372036854771712


def write(root: Path, rel: str, text: str = "") -> Path:
    """Write ``text`` to ``root/rel`` creating parents; returns the path."""
    path = root / rel.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Format ids like the kernel does (``0-3,8``)."""
    ids = sorted(set(cpus))
    parts: list[str] = []
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        parts.append(str(ids[i]) if i == j else f"{ids[i]}-{ids[j]}")
        i = j + 1
    return ",".join(parts)


def meminfo_text(
    total_kb: int, available_kb: int, swap_total_kb: int = 0, swap_free_kb: int = 0
) -> str:
    return (
        f"MemTotal:       {total_kb} kB\n"
        f"MemFree:        {available_kb // 2} kB\n"
        f"MemAvailable:   {available_kb} kB\n"
        "Buffers:           10240 kB\n"
        "Cached:           102400 kB\n"
        f"SwapTotal:      {swap_total_kb} kB\n"
        f"SwapFree:       {swap_free_kb} kB\n"
    )


def build_host_tree(
    dest: Path,
    *,
    packages: int = 1,
    cores_per_package: int = 6,
    threads_per_core: int = 2,
    numbering: str = "linux",
    flags: str = X86_FLAGS_AVX2,
    cpuinfo_key: str = "flags",
    mem_total_kb: int = 16 * 1024 * 1024,
    mem_avail_kb: int = 8 * 1024 * 1024,
    swap_total_kb: int = 0,
    swap_free_kb: int = 0,
    numa_per_package: int = 1,
    allowed_list: str | None = None,
    mountinfo: str = "",
    cgroup_proc: str | None = None,
    block_devices: dict[str, dict[str, str]] | None = None,
    online: bool = True,
) -> dict[str, list[int]]:
    """Create a fake host tree below ``dest``; returns ``{"siblings": ...}`` info.

    ``numbering="linux"`` puts every core's first thread at ``0..C-1`` and SMT
    siblings after (``i`` and ``i + C``); ``"adjacent"`` pairs siblings
    (``2i``, ``2i + 1``) like Windows/macOS.
    """
    total_cores = packages * cores_per_package
    n_cpu = total_cores * threads_per_core

    def cpu_id(pkg: int, core: int, thread: int) -> int:
        idx = pkg * cores_per_package + core
        if numbering == "adjacent":
            return idx * threads_per_core + thread
        return thread * total_cores + idx

    node_cpus: dict[int, list[int]] = {}
    for pkg in range(packages):
        for core in range(cores_per_package):
            sibs = [cpu_id(pkg, core, t) for t in range(threads_per_core)]
            for t, cpu in enumerate(sibs):
                base = f"sys/devices/system/cpu/cpu{cpu}/topology"
                write(
                    dest, f"{base}/thread_siblings_list", format_cpu_list(sibs) + "\n"
                )
                write(dest, f"{base}/core_id", f"{core}\n")
                write(dest, f"{base}/physical_package_id", f"{pkg}\n")
                write(dest, f"{base}/core_cpus_list", format_cpu_list(sibs) + "\n")
                node = (
                    pkg * numa_per_package
                    + (core * numa_per_package) // cores_per_package
                )
                node_cpus.setdefault(node, []).append(cpu)
    if online:
        write(dest, "sys/devices/system/cpu/online", f"0-{n_cpu - 1}\n")
    for node, cpus in sorted(node_cpus.items()):
        write(
            dest,
            f"sys/devices/system/node/node{node}/cpulist",
            format_cpu_list(cpus) + "\n",
        )
        write(
            dest,
            f"sys/devices/system/node/node{node}/meminfo",
            f"Node {node} MemTotal:       {mem_total_kb // len(node_cpus)} kB\n",
        )

    cpuinfo = "".join(
        f"processor\t: {cpu}\nmodel name\t: Fake CPU\n{cpuinfo_key}\t\t: {flags}\n\n"
        for cpu in range(min(n_cpu, 2))
    )
    write(dest, "proc/cpuinfo", cpuinfo)
    write(
        dest,
        "proc/meminfo",
        meminfo_text(mem_total_kb, mem_avail_kb, swap_total_kb, swap_free_kb),
    )
    allowed = allowed_list if allowed_list is not None else f"0-{n_cpu - 1}"
    write(dest, "proc/self/status", f"Name:\tpython\nCpus_allowed_list:\t{allowed}\n")
    if mountinfo:
        write(dest, "proc/self/mountinfo", mountinfo)
    if cgroup_proc is not None:
        write(dest, "proc/self/cgroup", cgroup_proc)
    for dev, files in (block_devices or {}).items():
        for rel, text in files.items():
            write(dest, f"sys/block/{dev}/{rel}", text)
    return {"siblings": [cpu_id(0, 0, t) for t in range(threads_per_core)]}


def cgroup_v2(
    dest: Path,
    *,
    rel: str = "/",
    cpu_max: str | None = None,
    memory_max: str | None = None,
    memory_high: str | None = None,
    memory_current: str | None = None,
    inactive_file: int | None = None,
    cpuset: str | None = None,
) -> None:
    """Add cgroup-v2 files for the cgroup ``rel`` (also writes ``/proc/self/cgroup``)."""
    write(dest, "proc/self/cgroup", f"0::{rel}\n")
    write(dest, "sys/fs/cgroup/cgroup.controllers", "cpuset cpu io memory pids\n")
    leaf = "sys/fs/cgroup/" + rel.strip("/")
    if cpu_max is not None:
        write(dest, f"{leaf}/cpu.max", cpu_max + "\n")
    if memory_max is not None:
        write(dest, f"{leaf}/memory.max", memory_max + "\n")
    if memory_high is not None:
        write(dest, f"{leaf}/memory.high", memory_high + "\n")
    if memory_current is not None:
        write(dest, f"{leaf}/memory.current", memory_current + "\n")
    if inactive_file is not None:
        write(dest, f"{leaf}/memory.stat", f"anon 1\ninactive_file {inactive_file}\n")
    if cpuset is not None:
        write(dest, f"{leaf}/cpuset.cpus.effective", cpuset + "\n")


MOUNTINFO_EPYC = r"""22 1 252:1 / / rw,relatime shared:1 - ext4 /dev/mapper/vg0-root rw
30 22 0:25 / /dev/shm rw,nosuid,nodev shared:3 - tmpfs tmpfs rw,size=263431168k
31 22 0:26 / /run rw,nosuid,nodev shared:4 - tmpfs tmpfs rw,size=52686236k,mode=755
32 31 0:27 / /run/lock rw,nosuid,nodev,noexec,relatime shared:5 - tmpfs tmpfs rw,size=5120k
33 22 0:28 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:6 - tmpfs tmpfs ro,mode=755
40 22 259:1 / /local rw,relatime shared:10 - xfs /dev/nvme0n1p1 rw
41 22 0:40 / /scratch rw,relatime shared:11 - lustre 10.0.0.1@o2ib:/scratch rw,flock
42 22 0:41 / /home rw,relatime shared:12 - nfs4 fs1:/export/home rw,vers=4.2
43 22 8:1 / /mnt/hdd\040data rw,relatime shared:13 - ext4 /dev/sda1 rw
"""

EPYC_BLOCK = {
    "dm-0": {
        "queue/rotational": "0\n",
        "dm/name": "vg0-root\n",
        "slaves/nvme1n1p3": "",
    },
    "nvme0n1": {"queue/rotational": "0\n"},
    "nvme1n1": {"queue/rotational": "0\n"},
    "sda": {"queue/rotational": "1\n"},
}
