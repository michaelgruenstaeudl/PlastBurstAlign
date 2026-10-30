"""
Tiered scratch engine for PlastBurstAlign.

Aligners and parsers write large, short-lived intermediates (unaligned and
protein-aligned FASTA, MAFFT's ``TMPDIR`` spill). This module decides *where* those
files live and bounds how much RAM they may occupy:

* :class:`ScratchEngine` -- one shared engine per run. It picks a scratch location
  in tier order (explicit directory, ``PBA_SCRATCH``, tmpfs/RAM-disk when capacity
  allows, node-local ``SLURM_TMPDIR``/``LOCAL_SCRATCH``/``TMPDIR``, system temp),
  hands out per-task :class:`ScratchTicket` directories with **byte admission**
  (RAM-disk while the remaining RAM-disk budget allows, otherwise transparently the
  disk tier) and unlinks every ticket directory as soon as it is released.
* :class:`EphemeralScratchManager` -- the original single-directory manager, kept
  fully backward compatible (now a thin wrapper around an engine, with real tmpfs
  detection from the mount table instead of the broken ``statvfs.f_type`` probe).

Everything environment-dependent (mount table, free space, available RAM, env vars,
platform, uid, PID liveness) is injectable through :class:`ScratchEnvironment`, so
every tier is unit-testable with fake data. Cleanup is robust: ``atexit`` plus
chained ``SIGTERM``/``SIGINT``/``SIGHUP`` handlers remove only directories this
process created, a sweep removes leftovers of dead PIDs (``pba_<pid>_<hex8>``) and
nothing in the cleanup path ever raises.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any

from .logging_ops import logger as log

__all__ = [
    "DEFAULT_MIN_HEADROOM_BYTES",
    "SCRATCH_BYTES_FACTOR",
    "EphemeralScratchManager",
    "MountEntry",
    "ScratchCandidate",
    "ScratchEngine",
    "ScratchEnvironment",
    "ScratchStats",
    "ScratchTicket",
    "discover_candidates",
    "estimate_scratch_need",
    "find_mount",
    "is_ram_path",
    "parse_mountinfo",
    "parse_proc_mounts",
    "read_mount_table",
    "sweep_stale_dirs",
    "tmpfs_capacity_ok",
]

DEFAULT_MIN_HEADROOM_BYTES = 500 * 1024 * 1024  # 500 MB safe margin
SCRATCH_BYTES_FACTOR = 3  # unaligned + aligned + temp copy per in-flight byte
RAM_FS_TYPES = frozenset({"tmpfs", "ramfs"})
_RUN_DIR_RE = re.compile(r"^pba_(\d+)_[0-9a-f]{8}$")
# tmpfs mounts that are never useful scratch space
_SKIP_MOUNT_PREFIXES = (
    "/sys",
    "/proc",
    "/dev/pts",
    "/dev/mqueue",
    "/dev/hugepages",
    "/run/lock",
    "/run/credentials",
    "/snap",
)


# ---------------------------------------------------------------------------
# Mount table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MountEntry:
    """One line of the Linux mount table."""

    mount_point: str
    fs_type: str
    source: str = ""
    options: tuple[str, ...] = ()

    @property
    def is_ram_fs(self) -> bool:
        """True for RAM-backed filesystems (tmpfs, ramfs)."""
        return self.fs_type in RAM_FS_TYPES

    @property
    def read_only(self) -> bool:
        """True if mounted read-only."""
        return "ro" in self.options


def _unescape_mount(field_text: str) -> str:
    """Decode the octal escapes (``\\040`` for space, ...) used by the kernel."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field_text)


def parse_mountinfo(text: str) -> list[MountEntry]:
    """Parse ``/proc/self/mountinfo`` text into entries (table order preserved)."""
    out: list[MountEntry] = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        lf, rf = left.split(), right.split()
        if len(lf) < 6 or len(rf) < 2:
            continue
        out.append(
            MountEntry(
                mount_point=_unescape_mount(lf[4]),
                fs_type=rf[0],
                source=_unescape_mount(rf[1]),
                options=tuple(lf[5].split(",")),
            )
        )
    return out


def parse_proc_mounts(text: str) -> list[MountEntry]:
    """Parse ``/proc/mounts`` (``fstab`` format) text into entries."""
    out: list[MountEntry] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        out.append(
            MountEntry(
                mount_point=_unescape_mount(parts[1]),
                fs_type=parts[2],
                source=_unescape_mount(parts[0]),
                options=tuple(parts[3].split(",")),
            )
        )
    return out


def _parse_any_mount_text(text: str) -> list[MountEntry]:
    """Parse either mountinfo or fstab-style text (auto-detected)."""
    first = next((ln.split() for ln in text.splitlines() if ln.strip()), [])
    if first and first[0].isdigit() and " - " in text:
        return parse_mountinfo(text)
    return parse_proc_mounts(text)


def read_mount_table(proc_root: str | Path = "/proc") -> list[MountEntry]:
    """Read the mount table below ``proc_root`` (``self/mountinfo``, then ``mounts``).

    Returns an empty list when neither file is readable (non-Linux hosts).
    """
    root = Path(proc_root)
    for rel, parser in (
        ("self/mountinfo", parse_mountinfo),
        ("mounts", parse_proc_mounts),
    ):
        try:
            entries = parser((root / rel).read_text(errors="replace"))
        except OSError:
            continue
        if entries:
            return entries
    return []


def find_mount(path: str | Path, mounts: Sequence[MountEntry]) -> MountEntry | None:
    """Return the mount that contains ``path`` (longest mount point, later entries win)."""
    p = os.path.normpath(str(path))
    best: MountEntry | None = None
    best_len = -1
    for m in mounts:
        mp = os.path.normpath(m.mount_point)
        if mp == "/" or p == mp or p.startswith(mp.rstrip("/") + "/"):
            if len(mp) >= best_len:
                best, best_len = m, len(mp)
    return best


def is_ram_path(path: str | Path, mounts: Sequence[MountEntry]) -> bool:
    """True if ``path`` lies on a tmpfs/ramfs mount of ``mounts``."""
    p = str(path)
    try:
        if os.path.exists(p):
            p = os.path.realpath(p)
    except OSError:
        pass
    m = find_mount(p, mounts)
    return m is not None and m.is_ram_fs


# ---------------------------------------------------------------------------
# Environment abstraction (everything injectable)
# ---------------------------------------------------------------------------


def _default_available_ram() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _default_pid_alive(pid: int) -> bool:
    """Conservative liveness probe: unknown means *alive* (so nothing is deleted)."""
    if pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        pass
    if sys.platform == "win32":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


@dataclass
class ScratchEnvironment:
    """Injectable view of the host used to choose scratch tiers.

    All fields are optional; ``None`` means "ask the real system". Tests pass fake
    mount tables, free-space and RAM callables, environment variables and PID probes.

    Attributes:
        platform: ``sys.platform`` value (``"linux"``, ``"darwin"``, ``"win32"``).
        env: Environment variables (default: ``os.environ``).
        mounts: Mount table, as parsed entries or raw mountinfo / ``/proc/mounts`` text.
        proc_root: Where to read the real mount table from when ``mounts`` is ``None``.
        uid: Numeric user id (for ``/run/user/<uid>``).
        temp_dir: System temp directory.
        volumes_root: macOS volumes directory searched for ``RAMDisk*`` volumes.
        disk_usage: ``path -> (total, used, free)`` bytes.
        available_ram: Callable returning available RAM in bytes (default: psutil).
        pid_alive: Callable deciding whether a PID is alive.
    """

    platform: str | None = None
    env: dict[str, str] | None = None
    mounts: Sequence[MountEntry] | str | None = None
    proc_root: str | Path = "/proc"
    uid: int | None = None
    temp_dir: str | None = None
    volumes_root: str = "/Volumes"
    disk_usage: Callable[[str], tuple[int, int, int]] | None = None
    available_ram: Callable[[], int | None] | None = None
    pid_alive: Callable[[int], bool] | None = None
    _mount_cache: list[MountEntry] | None = field(default=None, init=False, repr=False)

    @property
    def sys_platform(self) -> str:
        """Platform string in effect."""
        return self.platform if self.platform is not None else sys.platform

    def getenv(self, key: str) -> str | None:
        """Look up an environment variable (empty string counts as unset)."""
        src = self.env if self.env is not None else os.environ
        val = src.get(key)
        return val if val else None

    def get_mounts(self) -> list[MountEntry]:
        """The mount table (parsed once and cached)."""
        if self._mount_cache is None:
            if self.mounts is None:
                table = (
                    read_mount_table(self.proc_root)
                    if self.sys_platform.startswith("linux")
                    else []
                )
            elif isinstance(self.mounts, str):
                table = _parse_any_mount_text(self.mounts)
            else:
                table = list(self.mounts)
            self._mount_cache = table
        return self._mount_cache

    def get_uid(self) -> int | None:
        """Numeric user id, or ``None`` where unavailable."""
        if self.uid is not None:
            return self.uid
        getter = getattr(os, "getuid", None)
        return int(getter()) if getter is not None else None

    def get_temp_dir(self) -> str:
        """System temporary directory."""
        return self.temp_dir if self.temp_dir is not None else tempfile.gettempdir()

    def free_bytes(self, path: str | Path) -> int | None:
        """Free bytes on the filesystem of ``path`` (``None`` if unknown)."""
        try:
            if self.disk_usage is not None:
                return int(self.disk_usage(str(path))[2])
            return int(shutil.disk_usage(str(path)).free)
        except (OSError, ValueError, IndexError):
            return None

    def ram_available(self) -> int | None:
        """Available RAM in bytes (``None`` if unknown)."""
        fn = (
            self.available_ram
            if self.available_ram is not None
            else _default_available_ram
        )
        try:
            val = fn()
        except Exception:
            return None
        return None if val is None else int(val)

    def is_pid_alive(self, pid: int) -> bool:
        """Whether ``pid`` is alive (unknown counts as alive)."""
        fn = self.pid_alive if self.pid_alive is not None else _default_pid_alive
        try:
            return bool(fn(pid))
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Tier discovery and capacity rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScratchCandidate:
    """A possible scratch base directory in tier order.

    Attributes:
        source: Where the candidate came from (``custom``, ``PBA_SCRATCH``,
            ``tmpfs``, ``RAMDisk``, ``SLURM_TMPDIR``, ``LOCAL_SCRATCH``, ``TMPDIR``,
            ``system-temp``).
        path: Base directory (the engine creates ``pba_<pid>_<hex>`` inside it).
        is_ram: Whether the directory lives on a RAM-backed filesystem.
        explicit: Chosen by the user (custom dir / ``PBA_SCRATCH``): honoured
            without the capacity rule.
    """

    source: str
    path: Path
    is_ram: bool
    explicit: bool = False


def estimate_scratch_need(inflight_unaligned_bytes: int) -> int:
    """RAM-disk bytes needed for ``inflight_unaligned_bytes`` of in-flight input (3x)."""
    return SCRATCH_BYTES_FACTOR * max(0, int(inflight_unaligned_bytes))


def tmpfs_capacity_ok(
    free_bytes: int | None,
    ram_available: int | None,
    need_bytes: int,
    headroom_bytes: int,
) -> bool:
    """Capacity rule for using a tmpfs tier.

    The filesystem must have ``need + headroom`` free *and* available RAM must be
    able to absorb ``need + headroom`` (tmpfs pages are RAM). Unknown free space
    disqualifies the tier; unknown RAM is not held against it.
    """
    wanted = max(0, need_bytes) + max(0, headroom_bytes)
    if free_bytes is None or free_bytes < wanted:
        return False
    return not (ram_available is not None and ram_available < wanted)


def discover_candidates(
    custom_dir: str | Path | None, env: ScratchEnvironment
) -> list[ScratchCandidate]:
    """List scratch candidates in tier order for the platform of ``env``.

    Order: explicit ``custom_dir`` -> ``PBA_SCRATCH`` -> tmpfs mounts (Linux:
    ``/dev/shm``, ``/run/user/<uid>``, other tmpfs; macOS: ``/Volumes/RAMDisk*``) ->
    ``SLURM_TMPDIR`` / ``LOCAL_SCRATCH`` / ``TMPDIR`` -> system temp. Windows only
    knows custom, ``PBA_SCRATCH`` and the plain temp directory.
    """
    plat = env.sys_platform
    mounts = env.get_mounts() if plat.startswith("linux") else []
    out: list[ScratchCandidate] = []
    seen: set[str] = set()

    def add(
        source: str, path: str | Path, is_ram: bool | None, explicit: bool = False
    ) -> None:
        p = Path(path).expanduser()
        key = os.path.normpath(str(p))
        if key in seen:
            return
        seen.add(key)
        ram = is_ram if is_ram is not None else is_ram_path(p, mounts)
        out.append(ScratchCandidate(source, p, ram, explicit))

    if custom_dir:
        add("custom", Path(custom_dir).expanduser().resolve(), None, explicit=True)
    pba = env.getenv("PBA_SCRATCH")
    if pba:
        add("PBA_SCRATCH", pba, None, explicit=True)

    if plat.startswith("linux"):
        # An unreadable real mount table must not hide the ubiquitous /dev/shm.
        blind = not mounts and env.mounts is None and os.path.isdir("/dev/shm")
        if blind or _is_ram_mount("/dev/shm", mounts):
            add("tmpfs", "/dev/shm", True)
        uid = env.get_uid()
        if uid is not None:
            run_user = f"/run/user/{uid}"
            if _is_ram_mount(run_user, mounts):
                add("tmpfs", run_user, True)
        others = sorted(
            {
                m.mount_point
                for m in mounts
                if m.is_ram_fs
                and not m.read_only
                and not m.mount_point.startswith(_SKIP_MOUNT_PREFIXES)
                and m.mount_point not in ("/dev", "/run")
            }
        )
        for mp in others:
            add("tmpfs", mp, True)
    elif plat == "darwin":
        try:
            names = sorted(os.listdir(env.volumes_root))
        except OSError:
            names = []
        for name in names:
            if name.lower().startswith("ramdisk"):
                add("RAMDisk", Path(env.volumes_root) / name, True)

    if plat != "win32":
        for var in ("SLURM_TMPDIR", "LOCAL_SCRATCH", "TMPDIR"):
            val = env.getenv(var)
            if val:
                add(var, val, None)
    add("system-temp", env.get_temp_dir(), None)
    return out


def _is_ram_mount(path: str, mounts: Sequence[MountEntry]) -> bool:
    """True if ``path`` is *itself* a tmpfs mount point (not merely inside one)."""
    target = os.path.normpath(path)
    hit = [m for m in mounts if os.path.normpath(m.mount_point) == target]
    return bool(hit) and hit[-1].is_ram_fs


# ---------------------------------------------------------------------------
# Stale-directory sweep
# ---------------------------------------------------------------------------


def sweep_stale_dirs(
    base: str | Path,
    pid_alive: Callable[[int], bool] | None = None,
    own_pid: int | None = None,
    min_age_seconds: float = 0.0,
    now: float | None = None,
) -> list[str]:
    """Remove leftover ``pba_<pid>_<hex8>`` directories of *dead* processes in ``base``.

    Live PIDs, this process, symlinks, foreign-owned directories and names that do
    not match the pattern exactly are never touched; with ``min_age_seconds`` a
    directory must also be that old (guards PID namespaces sharing one tmp dir).
    Never raises.
    """
    removed: list[str] = []
    probe = pid_alive if pid_alive is not None else _default_pid_alive
    me = own_pid if own_pid is not None else os.getpid()
    clock = now if now is not None else time.time()
    getuid = getattr(os, "getuid", None)
    try:
        with os.scandir(base) as it:
            entries = list(it)
    except OSError:
        return removed
    for entry in entries:
        m = _RUN_DIR_RE.match(entry.name)
        if m is None:
            continue
        pid = int(m.group(1))
        try:
            if pid == me or not entry.is_dir(follow_symlinks=False):
                continue
            st = entry.stat(follow_symlinks=False)
            if getuid is not None and st.st_uid != getuid():
                continue
            if min_age_seconds > 0 and clock - st.st_mtime < min_age_seconds:
                continue
            if probe(pid):
                continue
            shutil.rmtree(entry.path, ignore_errors=True)
            removed.append(entry.path)
        except Exception as exc:
            log.debug(f"stale scratch sweep skipped `{entry.path}`: {exc!r}")
    if removed:
        log.info(f"removed {len(removed)} stale scratch dir(s) in `{base}`")
    return removed


# ---------------------------------------------------------------------------
# Process-wide cleanup registry (atexit + chained signal handlers)
# ---------------------------------------------------------------------------


class _CleanupRegistry:
    """Tracks directories this process owns and removes them on exit / signals.

    * ``atexit`` removes everything still registered.
    * Signal handlers (``SIGTERM``, ``SIGINT``, ``SIGHUP``) are installed only from the
      main thread and only for signals whose current disposition is a Python
      callable or ``SIG_DFL``; ignored signals stay ignored. A callable previous
      handler (e.g. Python's ``KeyboardInterrupt`` raiser) is *chained* and
      cleanup is left to ``atexit`` so in-flight work can unwind. With ``SIG_DFL``
      (the process would die immediately without ``atexit``) the directories are
      removed first and the signal is re-delivered with the default action.
    * Directories are only removed by the PID that created them (``fork`` children
      inherit the registry but own nothing).
    """

    def __init__(self) -> None:
        self._lock = (
            threading.RLock()
        )  # re-entrant: signal handlers run in the main thread
        self._owned: dict[str, int] = {}
        self._prev: dict[int, Any] = {}
        self._atexit_done = False
        self._signals_done = False

    @property
    def owned(self) -> list[str]:
        """Paths currently registered (for tests / diagnostics)."""
        with self._lock:
            return sorted(self._owned)

    def register(self, path: str | Path) -> None:
        """Take ownership of ``path`` (removed on exit/signal)."""
        with self._lock:
            self._owned[str(path)] = os.getpid()
        self.ensure_hooks()

    def unregister(self, path: str | Path) -> None:
        """Forget ``path`` (it was removed by its owner)."""
        with self._lock:
            self._owned.pop(str(path), None)

    def ensure_hooks(self) -> None:
        """Install the ``atexit`` hook once and signal handlers when in the main thread."""
        with self._lock:
            if not self._atexit_done:
                atexit.register(self.cleanup)
                self._atexit_done = True
            need_signals = not self._signals_done
        if need_signals and threading.current_thread() is threading.main_thread():
            self.install_signal_handlers()

    def cleanup(self) -> int:
        """Remove every directory owned by this process. Never raises."""
        removed = 0
        try:
            me = os.getpid()
            with self._lock:
                mine = [p for p, pid in self._owned.items() if pid == me]
                for p in mine:
                    self._owned.pop(p, None)
            for p in mine:
                try:
                    if not _RUN_DIR_RE.match(os.path.basename(p)):
                        continue  # defensive: only ever delete our own naming scheme
                    shutil.rmtree(p, ignore_errors=True)
                    removed += 1
                except Exception:
                    pass
        except Exception:
            pass
        return removed

    def install_signal_handlers(
        self, signals: Iterable[int] | None = None
    ) -> list[int]:
        """Chain-install cleanup handlers; returns the signals actually installed."""
        if threading.current_thread() is not threading.main_thread():
            return []
        if signals is None:
            wanted = [
                getattr(signal, name)
                for name in ("SIGTERM", "SIGINT", "SIGHUP")
                if hasattr(signal, name)
            ]
        else:
            wanted = list(signals)
        installed: list[int] = []
        for sig in wanted:
            try:
                prev = signal.getsignal(sig)
                if prev is signal.SIG_IGN or prev is None or sig in self._prev:
                    continue
                self._prev[sig] = prev
                signal.signal(sig, self._on_signal)
                installed.append(sig)
            except (ValueError, OSError, RuntimeError) as exc:
                self._prev.pop(sig, None)
                log.debug(f"cannot install scratch cleanup handler for {sig}: {exc!r}")
        with self._lock:
            self._signals_done = True
        return installed

    def uninstall_signal_handlers(self) -> None:
        """Restore the handlers that were replaced (used by tests)."""
        if threading.current_thread() is not threading.main_thread():
            return
        for sig, prev in list(self._prev.items()):
            try:
                if signal.getsignal(sig) == self._on_signal:
                    signal.signal(sig, prev)
            except (ValueError, OSError, RuntimeError):
                pass
        self._prev.clear()
        with self._lock:
            self._signals_done = False

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        prev = self._prev.get(signum)
        if callable(prev):
            prev(signum, frame)  # chain (may raise KeyboardInterrupt / SystemExit)
            return
        self.cleanup()
        try:
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)  # re-deliver with the default action
        except Exception:
            pass
        os._exit(128 + int(signum))  # only reached if the re-delivery did not kill us


_REGISTRY = _CleanupRegistry()


# ---------------------------------------------------------------------------
# Tickets, stats, engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScratchStats:
    """Snapshot of engine accounting (for the final scaling report)."""

    ram_path: str | None
    disk_path: str | None
    ramdisk_budget_bytes: int
    bytes_reserved: int
    peak_bytes_reserved: int
    ramdisk_bytes_reserved: int
    ramdisk_peak_bytes: int
    total_bytes_requested: int
    ramdisk_tickets: int
    disk_tickets: int
    fallback_count: int
    active_tickets: int
    stale_swept: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return asdict(self)


class ScratchTicket:
    """A per-task scratch directory with a byte reservation.

    Use as a context manager (``with engine.reserve(n) as t:``) or call
    :meth:`release`. Releasing is idempotent, removes the directory immediately and
    returns the bytes to the RAM-disk budget.
    """

    def __init__(
        self, engine: ScratchEngine, path: Path, tier: str, nbytes: int, ticket_id: int
    ) -> None:
        self._engine = engine
        self.path = path
        self.tier = tier
        self.nbytes = nbytes
        self.ticket_id = ticket_id
        self._released = False
        self._lock = threading.Lock()

    @property
    def is_ramdisk(self) -> bool:
        """True if the directory lives on the RAM-disk tier."""
        return self.tier == "ramdisk"

    @property
    def released(self) -> bool:
        """True once :meth:`release` ran."""
        return self._released

    def release(self) -> None:
        """Remove the directory and free the reservation (idempotent, never raises)."""
        with self._lock:
            if self._released:
                return
            self._released = True
        self._engine._release_ticket(self)

    def __enter__(self) -> ScratchTicket:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"ScratchTicket(path={str(self.path)!r}, tier={self.tier!r}, nbytes={self.nbytes})"


class ScratchEngine:
    """Shared, thread-safe, tiered scratch manager with per-task byte admission.

    Args:
        custom_dir: Explicit scratch base directory (tier 1, honoured as given).
        min_headroom_bytes: Bytes of tmpfs/RAM that must stay free (default 500 MB).
        expected_inflight_bytes: Sum of in-flight unaligned bytes; the tmpfs capacity
            rule needs ``3 x`` this plus the headroom.
        ram_budget_bytes: Optional hard cap on RAM-disk bytes handed to tickets.
        environment: Injectable host view (:class:`ScratchEnvironment`).
        sweep_stale: Remove leftover directories of dead PIDs in the tier bases.
        stale_min_age_seconds: Only sweep directories at least this old (default 1 h;
            protects runs in other PID namespaces that share the tmp directory).
        live_check: Re-check free tmpfs space and available RAM at every reservation
            (guards against other processes filling the RAM-disk).
        register_cleanup: Register directories for ``atexit``/signal cleanup.
    """

    def __init__(
        self,
        custom_dir: str | Path | None = None,
        *,
        min_headroom_bytes: int = DEFAULT_MIN_HEADROOM_BYTES,
        expected_inflight_bytes: int = 0,
        ram_budget_bytes: int | None = None,
        environment: ScratchEnvironment | None = None,
        sweep_stale: bool = True,
        stale_min_age_seconds: float = 3600.0,
        live_check: bool = True,
        register_cleanup: bool = True,
    ) -> None:
        self.min_headroom_bytes = max(0, int(min_headroom_bytes))
        self._stale_min_age = float(stale_min_age_seconds)
        self.environment = (
            environment if environment is not None else ScratchEnvironment()
        )
        self._live_check = bool(live_check)
        self._register = bool(register_cleanup)
        self._lock = threading.Lock()
        self._closed = False
        self._seq = 0
        self._ram_run: Path | None = None
        self._disk_run: Path | None = None
        self._ram_budget = 0
        self._ram_used = 0
        self._ram_peak = 0
        self._used = 0
        self._peak = 0
        self._requested = 0
        self._ram_tickets = 0
        self._disk_tickets = 0
        self._fallbacks = 0
        self._active: dict[int, ScratchTicket] = {}
        self._stale = 0
        self._primary_tier = "disk"
        self._select_tiers(custom_dir, expected_inflight_bytes, ram_budget_bytes)
        if sweep_stale:
            for run in (self._ram_run, self._disk_run):
                if run is not None:
                    self._stale += len(
                        sweep_stale_dirs(
                            run.parent,
                            self.environment.pid_alive,
                            min_age_seconds=self._stale_min_age,
                        )
                    )

    # -- tier selection ------------------------------------------------------

    def _make_run_dir(self, base: Path, explicit: bool) -> Path | None:
        """Create ``<base>/pba_<pid>_<hex8>``; ``None`` if the base is unusable."""
        try:
            if explicit:
                base.mkdir(parents=True, exist_ok=True)
            elif not base.is_dir():
                return None
            run = base / f"pba_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            run.mkdir(mode=0o700)
        except OSError as exc:
            if explicit:
                log.warning(f"requested scratch directory `{base}` is unusable: {exc}")
            else:
                log.debug(f"scratch base `{base}` unusable: {exc}")
            return None
        if self._register:
            _REGISTRY.register(run)
        return run

    def _ram_budget_for(self, base: Path, cap: int | None) -> int:
        free = self.environment.free_bytes(base)
        ram = self.environment.ram_available()
        limits = [v for v in (free, ram) if v is not None]
        budget = max(0, min(limits) - self.min_headroom_bytes) if limits else 0
        return budget if cap is None else min(budget, max(0, int(cap)))

    def _select_tiers(
        self,
        custom_dir: str | Path | None,
        expected_inflight_bytes: int,
        ram_budget_bytes: int | None,
    ) -> None:
        cands = discover_candidates(custom_dir, self.environment)
        need = estimate_scratch_need(expected_inflight_bytes)
        primary_idx = -1
        for idx, cand in enumerate(cands):
            if not cand.explicit and not (
                cand.path.is_dir() and os.access(cand.path, os.W_OK | os.X_OK)
            ):
                log.debug(
                    f"scratch candidate `{cand.path}` is not a writable directory"
                )
                continue
            if cand.is_ram and not cand.explicit:
                free = self.environment.free_bytes(cand.path)
                ram = self.environment.ram_available()
                if not tmpfs_capacity_ok(free, ram, need, self.min_headroom_bytes):
                    log.info(
                        f"{cand.path} (tmpfs) skipped: free="
                        f"{'?' if free is None else free // 2**20} MB, "
                        f"RAM available={'?' if ram is None else ram // 2**20} MB, "
                        f"need={need // 2**20} MB + {self.min_headroom_bytes // 2**20} MB headroom"
                    )
                    continue
            run = self._make_run_dir(cand.path, cand.explicit)
            if run is None:
                continue
            if cand.is_ram:
                self._ram_run = run
                self._ram_budget = self._ram_budget_for(cand.path, ram_budget_bytes)
                self._primary_tier = "ramdisk"
            else:
                self._disk_run = run
                self._primary_tier = "disk"
            primary_idx = idx
            log.info(
                f"Allocated ephemeral scratch `{run}` "
                f"({'RAM-disk' if cand.is_ram else 'filesystem'} via {cand.source})"
            )
            break
        if primary_idx < 0:
            raise OSError("no usable scratch directory could be created")
        if self._ram_run is not None:
            for cand in cands[primary_idx + 1 :]:
                if cand.is_ram:
                    continue
                run = self._make_run_dir(cand.path, cand.explicit)
                if run is not None:
                    self._disk_run = run
                    break
            if self._disk_run is None:
                run = self._make_run_dir(Path(self.environment.get_temp_dir()), False)
                self._disk_run = run
            log.info(
                f"RAM-disk scratch budget {self._ram_budget / 2**20:.0f} MB; "
                f"disk fallback `{self._disk_run}`"
            )

    # -- properties ----------------------------------------------------------

    @property
    def closed(self) -> bool:
        """True after :meth:`cleanup`."""
        return self._closed

    @property
    def path(self) -> Path:
        """Primary run directory (RAM-disk if one was selected, else disk)."""
        run = self._ram_run if self._primary_tier == "ramdisk" else self._disk_run
        if run is None or self._closed:
            raise RuntimeError(
                "Scratch directory has not been initialized or was already cleaned up."
            )
        return run

    @property
    def is_ramdisk(self) -> bool:
        """True if the primary directory resides in RAM."""
        return self._primary_tier == "ramdisk"

    @property
    def ram_path(self) -> Path | None:
        """RAM-disk run directory, if the engine has a RAM tier."""
        return self._ram_run

    @property
    def disk_path(self) -> Path | None:
        """Disk-tier run directory."""
        return self._disk_run

    @property
    def ramdisk_budget_bytes(self) -> int:
        """Bytes of RAM-disk that tickets may reserve in total."""
        return self._ram_budget

    # -- reservations --------------------------------------------------------

    def _live_ok(self, nbytes: int) -> bool:
        if not self._live_check or self._ram_run is None:
            return True
        free = self.environment.free_bytes(self._ram_run)
        if free is not None and free - self.min_headroom_bytes < nbytes:
            return False
        ram = self.environment.ram_available()
        return not (ram is not None and ram - self.min_headroom_bytes < nbytes)

    def _make_ticket_dir(self, tier: str, seq: int) -> Path:
        run = self._ram_run if tier == "ramdisk" else self._disk_run
        if run is None:
            raise OSError(f"no {tier} scratch tier")
        return Path(tempfile.mkdtemp(prefix=f"t{seq}_", dir=run))

    def reserve(self, nbytes: int = 0) -> ScratchTicket:
        """Admit ``nbytes`` of scratch and return a ticket with its own directory.

        The directory is on the RAM-disk tier while the remaining RAM-disk budget
        (and live free space / RAM) allows, otherwise on the disk tier. Never
        blocks and never over-subscribes the RAM-disk.
        """
        n = max(0, int(nbytes))
        # Phase 1 (locked): admission decision and byte accounting.
        with self._lock:
            if self._closed:
                raise RuntimeError("scratch engine is closed")
            self._seq += 1
            seq = self._seq
            use_ram = (
                self._ram_run is not None
                and self._ram_used + n <= self._ram_budget
                and self._live_ok(n)
            )
            fell_back = self._ram_run is not None and not use_ram
            self._used += n
            self._peak = max(self._peak, self._used)
            if use_ram:
                self._ram_used += n
                self._ram_peak = max(self._ram_peak, self._ram_used)
        tier = "ramdisk" if use_ram else "disk"
        # Phase 2 (unlocked): create the directory; RAM-disk errors fall back to disk.
        try:
            try:
                path = self._make_ticket_dir(tier, seq)
            except OSError as exc:
                if not use_ram:
                    raise
                log.warning(
                    f"RAM-disk ticket failed ({exc}); falling back to disk tier"
                )
                with self._lock:
                    self._ram_used -= n
                use_ram, fell_back, tier = False, True, "disk"
                path = self._make_ticket_dir(tier, seq)
        except OSError:
            with self._lock:
                self._used -= n
            raise
        ticket = ScratchTicket(self, path, tier, n, seq)
        # Phase 3 (locked): publish the ticket.
        with self._lock:
            if self._closed:
                self._used -= n
                if use_ram:
                    self._ram_used -= n
                shutil.rmtree(path, ignore_errors=True)
                raise RuntimeError("scratch engine is closed")
            self._active[seq] = ticket
            self._requested += n
            if use_ram:
                self._ram_tickets += 1
            else:
                self._disk_tickets += 1
            if fell_back:
                self._fallbacks += 1
        return ticket

    def _release_ticket(self, ticket: ScratchTicket) -> None:
        try:
            shutil.rmtree(ticket.path, ignore_errors=True)
        except Exception as exc:
            log.debug(f"ticket cleanup failed for `{ticket.path}`: {exc!r}")
        with self._lock:
            if self._active.pop(ticket.ticket_id, None) is None:
                return
            self._used = max(0, self._used - ticket.nbytes)
            if ticket.tier == "ramdisk":
                self._ram_used = max(0, self._ram_used - ticket.nbytes)

    def stats(self) -> ScratchStats:
        """Accounting snapshot (bytes reserved, peaks, tier counts, fallbacks)."""
        with self._lock:
            return ScratchStats(
                ram_path=str(self._ram_run) if self._ram_run else None,
                disk_path=str(self._disk_run) if self._disk_run else None,
                ramdisk_budget_bytes=self._ram_budget,
                bytes_reserved=self._used,
                peak_bytes_reserved=self._peak,
                ramdisk_bytes_reserved=self._ram_used,
                ramdisk_peak_bytes=self._ram_peak,
                total_bytes_requested=self._requested,
                ramdisk_tickets=self._ram_tickets,
                disk_tickets=self._disk_tickets,
                fallback_count=self._fallbacks,
                active_tickets=len(self._active),
                stale_swept=self._stale,
            )

    # -- subprocess environment / cleanup -------------------------------------

    def get_subprocess_env(self, ticket: ScratchTicket | None = None) -> dict[str, str]:
        """Copy of ``os.environ`` with ``TMPDIR`` set to the ticket (or engine) directory."""
        env = os.environ.copy()
        env["TMPDIR"] = str(ticket.path if ticket is not None else self.path)
        return env

    def cleanup(self) -> None:
        """Remove all engine directories (idempotent, never raises)."""
        try:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                runs = [r for r in (self._ram_run, self._disk_run) if r is not None]
                self._active.clear()
                self._used = self._ram_used = 0
            for run in runs:
                try:
                    shutil.rmtree(run, ignore_errors=True)
                    log.debug(f"Successfully unlinked ephemeral scratch `{run}`.")
                except Exception as exc:
                    log.warning(f"Error cleaning ephemeral scratch `{run}`: {exc}")
                finally:
                    _REGISTRY.unregister(run)
        except Exception:
            pass

    def describe(self) -> str:
        """One-line description for logs and reports."""
        ram = (
            f"ram={self._ram_run} (budget {self._ram_budget / 2**20:.0f} MB)"
            if self._ram_run
            else "ram=none"
        )
        return f"ScratchEngine({ram}, disk={self._disk_run})"

    def __enter__(self) -> ScratchEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cleanup()


# ---------------------------------------------------------------------------
# Backward-compatible single-directory manager
# ---------------------------------------------------------------------------


class EphemeralScratchManager:
    """Manages one ephemeral scratch directory (RAM-disk when possible).

    Backward compatible with the original manager (constructor, ``path``,
    ``is_ramdisk``, ``get_subprocess_env()``, ``cleanup()``, context manager) but
    built on :class:`ScratchEngine`: real tmpfs detection, capacity rule, stale
    sweep and signal-safe cleanup. New code should share one :class:`ScratchEngine`
    per run (available as :attr:`engine`) instead of creating a manager per aligner.
    """

    def __init__(
        self,
        custom_scratch_dir: str | Path | None = None,
        min_headroom_bytes: int = DEFAULT_MIN_HEADROOM_BYTES,
        *,
        environment: ScratchEnvironment | None = None,
        expected_inflight_bytes: int = 0,
        sweep_stale: bool = True,
    ) -> None:
        self.min_headroom_bytes = min_headroom_bytes
        self.engine = ScratchEngine(
            custom_scratch_dir,
            min_headroom_bytes=min_headroom_bytes,
            expected_inflight_bytes=expected_inflight_bytes,
            environment=environment,
            sweep_stale=sweep_stale,
        )
        self.scratch_path: Path | None = self.engine.path
        self._is_ramdisk = self.engine.is_ramdisk
        self._cleaned = False

    @staticmethod
    def _check_if_tmpfs(path: Path) -> bool:
        """Whether ``path`` lies on a tmpfs/ramfs mount (parsed from the mount table)."""
        try:
            return is_ram_path(path, ScratchEnvironment().get_mounts())
        except Exception:
            return False

    @property
    def is_ramdisk(self) -> bool:
        """Returns True if the active scratch path resides in RAM."""
        return self._is_ramdisk

    @property
    def path(self) -> Path:
        """Returns the active Path object for scratch operations."""
        if self.scratch_path is None or self._cleaned:
            raise RuntimeError(
                "Scratch directory has not been initialized or was already cleaned up."
            )
        return self.scratch_path

    def get_subprocess_env(self) -> dict[str, str]:
        """Returns a copy of os.environ with TMPDIR redirected to ephemeral scratch."""
        env = os.environ.copy()
        env["TMPDIR"] = str(self.path)
        return env

    def cleanup(self) -> None:
        """Recursively removes the scratch directory tree."""
        if self._cleaned:
            return
        self._cleaned = True
        self.engine.cleanup()

    def __enter__(self) -> EphemeralScratchManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.cleanup()
