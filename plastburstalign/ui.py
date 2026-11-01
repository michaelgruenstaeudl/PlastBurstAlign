"""Terminal UI: the ``rich`` live dashboard, the sink factory and the run summary.

Three independent pieces live here:

* :class:`RichSink` -- a :class:`~plastburstalign.events.BaseSink` that turns the
  scheduler's progress events into a live dashboard (overall progress bar with
  ETA, instantaneous throughput as an EMA of loci/min, a table of the currently
  *active* loci with ``N x L``, algorithm tier, leased threads, CPU ids and
  elapsed time, and done/failed/chunked counters).
* :func:`make_sink` -- picks the right sink for the environment: rich on a real
  terminal, plain ``logging`` lines in CI / pipes, nothing for library use.
* :func:`render_run_summary` / :func:`format_run_summary_text` -- the end-of-run
  panel (and its plain-text twin) built from a
  :class:`~plastburstalign.scaling.ScalingReport`.

Contract of :class:`RichSink`
-----------------------------
* ``emit`` only updates lock-protected state; rendering happens on ``rich``'s
  refresh thread at <= 4 Hz and in :meth:`RichSink.close`. Neither ``emit`` nor
  ``close`` ever raises (``BaseException`` such as ``KeyboardInterrupt`` is not
  swallowed, as everywhere in the event bus).
* While the live view is active the package logger is routed through
  ``rich.logging.RichHandler`` (:func:`~plastburstalign.logging_ops.rich_logging`),
  so log lines scroll above the dashboard instead of corrupting it.
* The terminal is restored by :meth:`RichSink.close` (use the sink as a context
  manager, or call ``close()`` in a ``finally``); an ``atexit`` hook is a
  last-resort safety net (cursor visible, ``sys.stdout``/``sys.stderr``
  un-proxied, log handlers restored).
* The dashboard shows a tier only if one is *provided*: through
  :meth:`RichSink.annotate`, ``RunStarted.details["loci"]`` (``{locus: tier |
  {"tier": ...}}``), or a ``meta``/``details`` mapping (keys ``tier``,
  ``strategy_tier`` or ``strategy``) on the locus events.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Any, TextIO

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

from .events import (
    BaseSink,
    LocusFinished,
    LocusQueued,
    LocusStarted,
    LoggingSink,
    NullSink,
    PhaseChanged,
    ProgressEvent,
    ProgressSink,
    RunFinished,
    RunStarted,
    WarningEvent,
)
from .logging_ops import logger as log
from .logging_ops import rich_logging
from .scaling import (
    DataDensity,
    ScalingReport,
    format_duration,
    summarize_supermatrix,
)

__all__ = [
    "MAX_REFRESH_HZ",
    "DashboardSnapshot",
    "RichSink",
    "annotate_sink",
    "close_sink",
    "detect_plain_environment",
    "format_cpu_ids",
    "format_run_summary_text",
    "make_sink",
    "render_run_summary",
]

#: Upper bound of the dashboard refresh rate (frames per second).
MAX_REFRESH_HZ: float = 4.0

#: Smoothing factor of the loci/min EMA (per one-second sample).
_EMA_ALPHA = 0.3
#: Minimum spacing (seconds) between two throughput samples.
_RATE_SAMPLE_SECONDS = 1.0
#: Elapsed seconds after which an active locus is highlighted.
_SLOW_LOCUS_SECONDS = 120.0
#: How many recent warning messages the dashboard remembers.
_KEEP_WARNINGS = 3

_FALSEY = frozenset({"", "0", "false", "no", "off"})
_TIER_KEYS = ("tier", "strategy_tier", "strategy")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def format_cpu_ids(ids: Sequence[int], max_chars: int = 16) -> str:
    """Compact CPU-id list: ``(0, 1, 2, 3, 8)`` -> ``"0-3,8"`` (``"-"`` if empty).

    Text longer than ``max_chars`` is cut with an ellipsis.
    """
    if not ids:
        return "-"
    ordered = sorted(set(int(i) for i in ids))
    runs: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        runs.append((start, prev))
        start = prev = cpu
    runs.append((start, prev))
    text = ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
    if len(text) > max_chars:
        text = text[: max(1, max_chars - 1)] + "…"
    return text


def _tier_text(value: Any) -> str | None:
    """Normalise a tier given as enum, string or ``{"tier": ...}`` mapping."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in _TIER_KEYS:
            if key in value:
                tier = _tier_text(value[key])
                if tier:
                    return tier
        return None
    inner = getattr(value, "tier", None)  # an AlignmentStrategy-like object
    if inner is not None:
        return _tier_text(inner)
    name = getattr(value, "name", value)
    text = str(name).strip().lower()
    return text or None


def _event_tier(event: ProgressEvent) -> str | None:
    """Tier carried by an event's ``meta``/``details`` (or ``tier``), if any."""
    for attr in ("meta", "details"):
        mapping = getattr(event, attr, None)
        if isinstance(mapping, Mapping):
            for key in _TIER_KEYS:
                if key in mapping:
                    tier = _tier_text(mapping[key])
                    if tier:
                        return tier
    return _tier_text(getattr(event, "tier", None))


def _fmt_nl(n_seq: int, max_len: int) -> str:
    if n_seq <= 0 and max_len <= 0:
        return "-"
    return f"{n_seq:,} x {max_len:,}"


# ---------------------------------------------------------------------------
# dashboard state
# ---------------------------------------------------------------------------


@dataclass
class _Locus:
    """Mutable state of one active top-level locus (guarded by the sink lock)."""

    name: str
    n_seq: int = 0
    max_len: int = 0
    tier: str | None = None
    started_at: float = 0.0
    units: dict[str, tuple[int, tuple[int, ...]]] = field(default_factory=dict)
    units_total: int = 0
    units_done: int = 0

    @property
    def threads(self) -> int:
        return sum(t for t, _ in self.units.values())

    @property
    def cpu_ids(self) -> tuple[int, ...]:
        ids: set[int] = set()
        for _, cpus in self.units.values():
            ids.update(cpus)
        return tuple(sorted(ids))


@dataclass(frozen=True)
class ActiveRow:
    """One row of the active-loci table (immutable snapshot)."""

    name: str
    n_seq: int
    max_len: int
    tier: str | None
    threads: int
    cpu_ids: tuple[int, ...]
    elapsed: float
    units_done: int
    units_total: int


@dataclass(frozen=True)
class DashboardSnapshot:
    """Consistent copy of the dashboard state (also handy in tests).

    Attributes:
        total: Top-level loci expected.
        done: Loci finished successfully; ``failed``: loci that failed.
        chunked: Distinct loci that fanned out into sub-locus chunks.
        queued: Top-level loci seen in ``LocusQueued`` events.
        active: Rows of loci currently executing (longest running first).
        rate_per_min: Smoothed throughput (EMA of loci/min), ``None`` if unknown.
        eta_seconds: Remaining time at the current rate, ``None`` if unknown.
        elapsed: Seconds since ``RunStarted`` (or the first event).
        finished: ``RunFinished`` was received.
        warnings: Number of ``WarningEvent``s; ``last_warnings`` the latest texts.
    """

    phase: str
    mode: str
    threads: int
    total: int
    done: int
    failed: int
    chunked: int
    queued: int
    active: tuple[ActiveRow, ...]
    rate_per_min: float | None
    eta_seconds: float | None
    elapsed: float
    finished: bool
    ok: bool
    warnings: int
    last_warnings: tuple[str, ...]

    @property
    def completed(self) -> int:
        """Loci that are no longer pending (``done + failed``)."""
        return self.done + self.failed


# ---------------------------------------------------------------------------
# RichSink
# ---------------------------------------------------------------------------


class RichSink(BaseSink):
    """Live ``rich`` dashboard fed by progress events (see the module docstring).

    Args:
        console: Console to draw on (default: a stderr console).
        stream: Output stream when no ``console`` is given (default ``sys.stderr``).
        transient: Erase the dashboard on :meth:`close` (``rich`` ``Live`` semantics).
        refresh_per_second: Frame rate, clamped to :data:`MAX_REFRESH_HZ`.
        max_active_rows: Rows of the active-loci table; the rest is summarised.
        capture_logging: Route package logging through ``rich`` while live.
        stop_on_finish: Stop the live view on ``RunFinished`` (default: keep it
            until :meth:`close`, so multi-pass runs reuse one dashboard).
        live: ``False`` tracks state only (no ``Live`` display, no logging
            swap): poll :meth:`snapshot` / :meth:`renderable` yourself, e.g. from
            ``LocusScheduler.run(on_tick=...)`` or a notebook.
        clock: Monotonic clock (injectable for tests).
        title: Panel title prefix.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        stream: TextIO | None = None,
        transient: bool = False,
        refresh_per_second: float = MAX_REFRESH_HZ,
        max_active_rows: int = 12,
        capture_logging: bool = True,
        stop_on_finish: bool = False,
        live: bool = True,
        clock: Callable[[], float] = time.monotonic,
        title: str = "PlastBurstAlign",
    ) -> None:
        if console is None:
            console = (
                Console(file=stream) if stream is not None else Console(stderr=True)
            )
        self.console: Console = console
        self._transient = bool(transient)
        self._fps = max(0.1, min(float(refresh_per_second), MAX_REFRESH_HZ))
        self._max_rows = max(1, int(max_active_rows))
        self._capture_logging = bool(capture_logging)
        self._stop_on_finish = bool(stop_on_finish)
        self._use_live = bool(live)
        self._clock = clock
        self._title = title

        self._lock = threading.RLock()  # guards all state below
        self._render_lock = threading.Lock()  # serialises Progress updates
        self._reset_state()
        self._info: dict[str, dict[str, Any]] = {}

        self._live: Live | None = None
        self._stack: ExitStack | None = None
        self._started = False
        self._closed = False

        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[timing]}"),
            console=self.console,
            auto_refresh=False,
            expand=True,
        )
        self._task: TaskID = self._progress.add_task("loci", total=None, timing="")

        ref = weakref.ref(self)

        def _close_at_exit() -> None:
            sink = ref()
            if sink is not None:
                sink.close()

        self._atexit_cb = _close_at_exit
        atexit.register(_close_at_exit)

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> RichSink:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def is_live(self) -> bool:
        """True while the live view is drawing."""
        return self._live is not None

    def start(self) -> None:
        """Start the live view (idempotent; failures degrade to "no display")."""
        with self._lock:
            if self._started or self._closed:
                return
            self._started = True
        if not self._use_live:
            return
        stack = ExitStack()
        try:
            if self._capture_logging:
                stack.enter_context(rich_logging(self.console))
            live = Live(
                console=self.console,
                get_renderable=self.renderable,
                auto_refresh=True,
                refresh_per_second=self._fps,
                transient=self._transient,
            )
            live.start()
        except Exception as exc:  # e.g. another live display owns the console
            log.debug(f"dashboard could not start: {exc!r}")
            try:
                stack.close()
            except Exception as exc2:
                log.debug(f"dashboard cleanup failed: {exc2!r}")
            return
        with self._lock:
            if self._closed:  # closed concurrently while starting
                late = True
            else:
                late = False
                self._live, self._stack = live, stack
        if late:
            self._stop(live, stack)

    def close(self) -> None:
        """Stop the live view and restore the terminal and log handlers (idempotent)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            live, self._live = self._live, None
            stack, self._stack = self._stack, None
        try:
            atexit.unregister(self._atexit_cb)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug(f"atexit unregister failed: {exc!r}")
        self._stop(live, stack)

    @staticmethod
    def _stop(live: Live | None, stack: ExitStack | None) -> None:
        try:
            if live is not None:
                live.stop()
        except Exception as exc:
            log.debug(f"stopping the live view failed: {exc!r}")
        finally:
            if stack is not None:
                try:
                    stack.close()
                except Exception as exc:
                    log.debug(f"restoring logging failed: {exc!r}")

    # -- annotations -----------------------------------------------------------

    def annotate(self, locus: str, **info: Any) -> None:
        """Attach display information (``tier``, ``n_seq``, ``max_len``) to a locus.

        Call it before the locus starts (e.g. from the coordinator, once the
        strategy was decided); unknown keys are ignored.
        """
        with self._lock:
            self._info.setdefault(locus, {}).update(info)

    # -- event handling ----------------------------------------------------------

    def _reset_state(self) -> None:
        self._phase = ""
        self._mode = ""
        self._threads = 0
        self._total = 0
        self._done = 0
        self._failed = 0
        self._queued = 0
        self._chunked_roots: set[str] = set()
        self._active: dict[str, _Locus] = {}
        self._root_of: dict[str, str] = {}
        self._warning_count = 0
        self._last_warnings: deque[str] = deque(maxlen=_KEEP_WARNINGS)
        self._t_start: float | None = None
        self._finished = False
        self._run_ok = True
        self._rate_t: float | None = None
        self._rate_count = 0
        self._rate_ema: float | None = None
        self._first_done_t: float | None = None

    def handle(self, event: ProgressEvent) -> None:
        if self._closed:
            return
        if not self._started:
            self.start()
        now = self._clock()
        with self._lock:
            if self._t_start is None:
                self._t_start = now
            if isinstance(event, RunStarted):
                self._on_run_started(event, now)
            elif isinstance(event, PhaseChanged):
                self._phase = event.phase
            elif isinstance(event, LocusQueued):
                self._on_queued(event)
            elif isinstance(event, LocusStarted):
                self._on_started(event, now)
            elif isinstance(event, LocusFinished):
                self._on_finished(event, now)
            elif isinstance(event, WarningEvent):
                self._warning_count += 1
                where = f"[{event.locus}] " if event.locus else ""
                self._last_warnings.append(f"{where}{event.message}")
            elif isinstance(event, RunFinished):
                self._on_run_finished(event)
        if isinstance(event, RunFinished) and self._stop_on_finish:
            self.close()

    def _on_run_started(self, event: RunStarted, now: float) -> None:
        self._reset_state()  # annotations (``_info``) survive a new pass
        self._t_start = now
        self._total = int(event.total_loci)
        self._mode = event.mode
        self._threads = int(event.threads)
        self._phase = "align"
        loci = event.details.get("loci") if event.details else None
        if isinstance(loci, Mapping):
            for name, value in loci.items():
                tier = _tier_text(value)
                if tier:
                    self._info.setdefault(str(name), {})["tier"] = tier

    def _root(self, name: str, parent: str | None) -> str:
        if parent is None:
            return name
        return self._root_of.get(parent, parent)

    def _on_queued(self, event: LocusQueued) -> None:
        if event.parent is None:
            self._queued += 1
            self._total = max(self._total, self._queued)
            tier = _event_tier(event)
            if tier:
                self._info.setdefault(event.locus, {})["tier"] = tier
            return
        root = self._root(event.locus, event.parent)
        self._root_of[event.locus] = root
        self._chunked_roots.add(root)
        loc = self._active.get(root)
        if loc is not None:
            loc.units_total += 1

    def _on_started(self, event: LocusStarted, now: float) -> None:
        root = self._root(event.locus, event.parent)
        self._root_of[event.locus] = root
        loc = self._active.get(root)
        if loc is None:
            info = self._info.get(root, {})
            loc = _Locus(
                name=root,
                n_seq=int(info.get("n_seq", 0) or 0),
                max_len=int(info.get("max_len", 0) or 0),
                tier=_tier_text(info.get("tier")),
                started_at=now,
            )
            self._active[root] = loc
        if event.parent is None:
            loc.n_seq = event.n_seq or loc.n_seq
            loc.max_len = event.max_len or loc.max_len
        else:
            # a fanned-out locus: the planning unit has finished for good
            self._chunked_roots.add(root)
            loc.units.pop(root, None)
        tier = _event_tier(event)
        if tier:
            loc.tier = tier
        loc.units[event.locus] = (max(0, int(event.threads)), tuple(event.cpu_ids))

    def _on_finished(self, event: LocusFinished, now: float) -> None:
        root = self._root_of.pop(event.locus, self._root(event.locus, event.parent))
        if event.parent is not None:
            loc = self._active.get(root)
            if loc is not None:
                loc.units.pop(event.locus, None)
                loc.units_done += 1
            return
        self._active.pop(root, None)
        if event.ok:
            self._done += 1
        else:
            self._failed += 1
            self._run_ok = False
        if self._first_done_t is None:
            self._first_done_t = now
        stale = [k for k, v in self._root_of.items() if v == root]
        for key in stale:
            del self._root_of[key]

    def _on_run_finished(self, event: RunFinished) -> None:
        self._finished = True
        self._run_ok = bool(event.ok) and self._failed == 0
        self._done = max(self._done, int(event.loci_ok))
        self._failed = max(self._failed, int(event.loci_failed))
        if self._total <= 0:
            self._total = self._done + self._failed
        self._active.clear()
        self._phase = "done" if self._run_ok else "finished with errors"

    # -- snapshot / render -------------------------------------------------------

    def _sample_rate(self, now: float) -> None:
        finished = self._done + self._failed
        if self._rate_t is None:
            self._rate_t, self._rate_count = now, finished
            return
        dt = now - self._rate_t
        if dt < _RATE_SAMPLE_SECONDS:
            return
        if finished == 0 and self._rate_ema is None:
            self._rate_t = now
            return
        inst = (finished - self._rate_count) / dt
        self._rate_ema = (
            inst
            if self._rate_ema is None
            else _EMA_ALPHA * inst + (1.0 - _EMA_ALPHA) * self._rate_ema
        )
        self._rate_t, self._rate_count = now, finished

    def snapshot(self) -> DashboardSnapshot:
        """Consistent copy of the current state (advances the throughput EMA)."""
        with self._lock:
            now = self._clock()
            self._sample_rate(now)
            finished = self._done + self._failed
            elapsed = 0.0 if self._t_start is None else max(0.0, now - self._t_start)
            rate: float | None = self._rate_ema if self._rate_ema else None
            if rate is None and finished > 0 and self._first_done_t is not None:
                span = max(1e-9, now - (self._t_start or self._first_done_t))
                rate = finished / span
            remaining = max(0, self._total - finished)
            eta: float | None = None
            if self._finished:
                eta = 0.0
            elif rate and rate > 0:
                eta = remaining / rate
            rows = tuple(
                sorted(
                    (
                        ActiveRow(
                            name=loc.name,
                            n_seq=loc.n_seq,
                            max_len=loc.max_len,
                            tier=loc.tier,
                            threads=loc.threads,
                            cpu_ids=loc.cpu_ids,
                            elapsed=max(0.0, now - loc.started_at),
                            units_done=loc.units_done,
                            units_total=loc.units_total,
                        )
                        for loc in self._active.values()
                    ),
                    key=lambda r: (-r.elapsed, r.name),
                )
            )
            return DashboardSnapshot(
                phase=self._phase,
                mode=self._mode,
                threads=self._threads,
                total=self._total,
                done=self._done,
                failed=self._failed,
                chunked=len(self._chunked_roots),
                queued=self._queued,
                active=rows,
                rate_per_min=None if rate is None else rate * 60.0,
                eta_seconds=eta,
                elapsed=elapsed,
                finished=self._finished,
                ok=self._run_ok,
                warnings=self._warning_count,
                last_warnings=tuple(self._last_warnings),
            )

    def renderable(self) -> RenderableType:
        """The dashboard as a ``rich`` renderable (the ``Live`` callback).

        Never raises: a dead refresh thread would freeze the display.
        """
        try:
            return self._build(self.snapshot())
        except Exception as exc:
            log.debug(f"dashboard render failed: {exc!r}")
            return Text(f"(dashboard unavailable: {exc!r})")

    def _build(self, snap: DashboardSnapshot) -> RenderableType:
        timing = f"elapsed {format_duration(snap.elapsed)}"
        if snap.finished:
            timing += "  finished"
        elif snap.eta_seconds is not None:
            timing += f"  ETA {format_duration(snap.eta_seconds)}"
        else:
            timing += "  ETA --"
        with self._render_lock:
            self._progress.update(
                self._task,
                description=escape(snap.phase) or "loci",
                total=snap.total or None,
                completed=snap.completed,
                timing=timing,
            )
            bar: RenderableType = self._progress.get_renderable()

        rate = "--" if snap.rate_per_min is None else f"{snap.rate_per_min:.1f}"
        counters = Text.assemble(
            ("done ", "dim"),
            (str(snap.done), "bold green"),
            ("   failed ", "dim"),
            (str(snap.failed), "bold red" if snap.failed else "bold"),
            ("   chunked ", "dim"),
            (str(snap.chunked), "bold magenta" if snap.chunked else "bold"),
            ("   active ", "dim"),
            (str(len(snap.active)), "bold cyan"),
            ("   queued ", "dim"),
            (str(max(0, snap.total - snap.completed - len(snap.active))), "bold"),
        )
        counters.append("\n")
        counters.append("throughput ", style="dim")
        counters.append(f"{rate} loci/min", style="bold")
        if snap.warnings:
            counters.append(f"   warnings {snap.warnings}", style="yellow")

        table = Table(
            box=box.SIMPLE_HEAD,
            expand=True,
            show_edge=False,
            pad_edge=False,
            header_style="bold",
        )
        table.add_column("Active locus", no_wrap=True, overflow="ellipsis", ratio=3)
        table.add_column("N x L", justify="right", no_wrap=True)
        table.add_column("Tier", no_wrap=True)
        table.add_column("Thr", justify="right", no_wrap=True)
        table.add_column("CPUs", no_wrap=True)
        table.add_column("Elapsed", justify="right", no_wrap=True)
        for row in snap.active[: self._max_rows]:
            name = row.name
            if row.units_total:
                name += f" [{row.units_done}/{row.units_total} units]"
            style = "yellow" if row.elapsed >= _SLOW_LOCUS_SECONDS else None
            table.add_row(
                Text(name, style=style or ""),
                Text(_fmt_nl(row.n_seq, row.max_len)),
                Text(row.tier or "-"),
                Text(str(row.threads) if row.threads else "-"),
                Text(format_cpu_ids(row.cpu_ids)),
                Text(format_duration(row.elapsed)),
            )
        hidden = len(snap.active) - self._max_rows
        if hidden > 0:
            table.add_row(
                Text(f"+{hidden} more active", style="dim"), "", "", "", "", ""
            )

        parts: list[RenderableType] = [bar, counters, table]
        for message in snap.last_warnings[-1:]:
            parts.append(
                Text(f"! {message}", style="yellow", no_wrap=True, overflow="ellipsis")
            )
        head = " | ".join(
            p
            for p in (
                self._title,
                snap.mode,
                f"{snap.threads} threads" if snap.threads else "",
            )
            if p
        )
        if snap.finished:
            border = "green" if snap.ok else "red"
        else:
            border = "cyan"
        return Panel(
            Group(*parts), title=Text(head), title_align="left", border_style=border
        )


# ---------------------------------------------------------------------------
# sink factory
# ---------------------------------------------------------------------------


def detect_plain_environment(env: Mapping[str, str] | None = None) -> str | None:
    """Name of the variable that asks for plain output, or ``None``.

    ``CI`` / ``GITHUB_ACTIONS`` (any value except empty/0/false/no/off),
    ``NO_COLOR`` (any non-empty value, per no-color.org) and ``TERM=dumb``.
    """
    environ = os.environ if env is None else env
    for key in ("CI", "GITHUB_ACTIONS"):
        if environ.get(key, "").strip().lower() not in _FALSEY:
            return key
    if environ.get("NO_COLOR", "") != "":
        return "NO_COLOR"
    if environ.get("TERM", "").strip().lower() == "dumb":
        return "TERM=dumb"
    return None


def _is_tty(stream: object) -> bool:
    try:
        isatty = getattr(stream, "isatty", None)
        return bool(isatty()) if callable(isatty) else False
    except Exception:
        return False


def make_sink(
    mode: str = "auto",
    *,
    console: Console | None = None,
    stream: TextIO | None = None,
    transient: bool = False,
) -> ProgressSink:
    """Create the progress sink for ``mode``.

    * ``"off"``   -> :class:`~plastburstalign.events.NullSink`.
    * ``"plain"`` -> :class:`~plastburstalign.events.LoggingSink`.
    * ``"rich"``  -> :class:`RichSink` (even on a non-TTY; ``rich`` then only
      prints the final frame).
    * ``"auto"``  -> :class:`RichSink` only if the output is a real terminal
      (``console.is_terminal`` and not a dumb terminal, else ``stream.isatty()``,
      else ``sys.stderr``) and none of ``CI``, ``GITHUB_ACTIONS``, ``NO_COLOR``,
      ``TERM=dumb`` asks for plain output; otherwise ``LoggingSink``.

    Raises:
        ValueError: Unknown ``mode``.
    """
    key = str(mode).strip().lower()
    if key in ("off", "none", "null", "quiet"):
        return NullSink()
    if key in ("plain", "log", "logging"):
        return LoggingSink()
    if key == "rich":
        return RichSink(console, stream=stream, transient=transient)
    if key != "auto":
        raise ValueError(
            f"unknown dashboard mode {mode!r}; expected auto, rich, plain or off"
        )
    if detect_plain_environment() is not None:
        return LoggingSink()
    if console is not None:
        interactive = console.is_terminal and not console.is_dumb_terminal
    else:
        interactive = _is_tty(stream if stream is not None else sys.stderr)
    if not interactive:
        return LoggingSink()
    return RichSink(console, stream=stream, transient=transient)


def close_sink(sink: object | None) -> None:
    """Close ``sink`` if it has a ``close`` method; never raises."""
    closer = getattr(sink, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception as exc:
            log.debug(f"closing sink {type(sink).__name__} failed: {exc!r}")


def annotate_sink(sink: object | None, locus: str, **info: Any) -> None:
    """Call ``sink.annotate(locus, **info)`` if supported; never raises."""
    annotate = getattr(sink, "annotate", None)
    if callable(annotate):
        try:
            annotate(locus, **info)
        except Exception as exc:
            log.debug(f"annotating sink failed: {exc!r}")
        return
    for child in getattr(sink, "sinks", ()) or ():  # CompositeSink children
        annotate_sink(child, locus, **info)


# ---------------------------------------------------------------------------
# end-of-run summary
# ---------------------------------------------------------------------------


def _with_matrix(
    report: ScalingReport, supermatrix_summary: Mapping[str, Any] | object | None
) -> ScalingReport:
    if supermatrix_summary is None:
        return report
    summary = (
        supermatrix_summary
        if isinstance(supermatrix_summary, Mapping)
        else summarize_supermatrix(supermatrix_summary)
    )
    return replace(report, data=DataDensity.from_summary(summary))


def render_run_summary(
    console: Console | None,
    scaling_report: ScalingReport,
    supermatrix_summary: Mapping[str, Any] | object | None = None,
) -> None:
    """Print the end-of-run panel (scaling report + supermatrix density).

    ``supermatrix_summary`` is a dict (see
    :func:`~plastburstalign.scaling.summarize_supermatrix`) or a ``Supermatrix``;
    ``None`` keeps whatever the report already carries.
    """
    if console is None:
        console = Console(stderr=True)
    report = _with_matrix(scaling_report, supermatrix_summary)
    console.print(report.renderable(title="Run summary"))


def format_run_summary_text(
    scaling_report: ScalingReport,
    supermatrix_summary: Mapping[str, Any] | object | None = None,
) -> str:
    """Plain-text twin of :func:`render_run_summary` (CI logs, ``--no-dashboard``)."""
    return _with_matrix(scaling_report, supermatrix_summary).to_text(
        title="PlastBurstAlign run summary"
    )
