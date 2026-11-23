"""Typed, thread-safe progress events and sinks (the ``ProgressSink`` event bus).

Producers (the locus scheduler, the alignment coordinator, the public API) emit
small immutable event objects to a :class:`ProgressSink`. Consumers implement
the one-method protocol ``emit(event)``:

* :class:`NullSink`      -- library default, drops everything.
* :class:`LoggingSink`   -- plain ``logging`` output for CI / non-TTY runs.
* :class:`ListSink`      -- records events in memory (tests).
* :class:`CompositeSink` -- fans one event stream out to several sinks.
* ``rich`` live dashboards subclass :class:`BaseSink` (see ``ui.py``).

Design contract: **a sink never raises into its caller.** Every concrete sink
here derives from :class:`BaseSink`, whose ``emit`` wraps ``handle`` in a
``try/except Exception``; producers additionally go through :func:`safe_emit`
so that even a third-party object implementing the bare protocol cannot take a
worker thread down. ``BaseException`` (``KeyboardInterrupt``/``SystemExit``) is
deliberately *not* swallowed. All sinks in this module are thread-safe because
the scheduler emits from many worker threads at once.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from .logging_ops import logger as log

__all__ = [
    "BaseSink",
    "CompositeSink",
    "ListSink",
    "LocusFinished",
    "LocusQueued",
    "LocusStarted",
    "LoggingSink",
    "NullSink",
    "PhaseChanged",
    "ProgressEvent",
    "ProgressSink",
    "RunFinished",
    "RunStarted",
    "WarningEvent",
    "safe_emit",
]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressEvent:
    """Base class of all progress events.

    Attributes:
        ts: Wall-clock creation time (``time.time()``), keyword-only so that
            subclasses can keep natural positional fields.
    """

    ts: float = field(default_factory=time.time, kw_only=True)


@dataclass(frozen=True)
class RunStarted(ProgressEvent):
    """A run (one alignment pass) is about to start.

    Attributes:
        total_loci: Number of top-level loci that will be processed.
        mode: Run mode label (for example ``"cds"`` or ``"igs"``).
        threads: Total worker threads / cores granted to the run.
        details: Free-form extra information (host profile, scratch tier, ...).
    """

    total_loci: int
    mode: str = ""
    threads: int = 1
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseChanged(ProgressEvent):
    """The run moved to a new phase (``extract``, ``align``, ``concat``, ...)."""

    phase: str
    detail: str = ""


@dataclass(frozen=True)
class LocusQueued(ProgressEvent):
    """A locus (or sub-locus chunk when ``parent`` is set) entered the queue."""

    locus: str
    n_seq: int = 0
    max_len: int = 0
    work: float = 0.0
    parent: str | None = None


@dataclass(frozen=True)
class LocusStarted(ProgressEvent):
    """A locus (or chunk / finalizer unit) started executing.

    Attributes:
        locus: Task identifier.
        n_seq: Number of sequences (0 when unknown).
        max_len: Longest sequence length (0 when unknown).
        threads: Threads leased for the unit.
        cpu_ids: OS CPU ids of the lease (empty when unknown).
        work: Estimated work (scheduler priority units).
        parent: Parent locus for chunk/finalizer units, else ``None``.
    """

    locus: str
    n_seq: int = 0
    max_len: int = 0
    threads: int = 1
    cpu_ids: tuple[int, ...] = ()
    work: float = 0.0
    parent: str | None = None


@dataclass(frozen=True)
class LocusFinished(ProgressEvent):
    """A locus (or chunk / finalizer unit) finished.

    For a locus that fanned out into chunks this is emitted once, when the
    finalizer completed (or the locus failed), and ``seconds`` is the wall time
    since the locus first started.
    """

    locus: str
    ok: bool
    seconds: float = 0.0
    threads: int = 1
    error: str | None = None
    parent: str | None = None


@dataclass(frozen=True)
class WarningEvent(ProgressEvent):
    """A non-fatal condition worth surfacing in a UI (named to avoid ``Warning``)."""

    message: str
    locus: str | None = None
    category: str = ""


@dataclass(frozen=True)
class RunFinished(ProgressEvent):
    """The run finished; ``summary`` may carry scheduler/scratch statistics."""

    ok: bool = True
    seconds: float = 0.0
    loci_ok: int = 0
    loci_failed: int = 0
    summary: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------


@runtime_checkable
class ProgressSink(Protocol):
    """Anything that can receive progress events."""

    def emit(self, event: ProgressEvent) -> None:
        """Receive one event (must be thread-safe and must not raise)."""
        ...


def safe_emit(sink: ProgressSink | None, event: ProgressEvent) -> None:
    """Emit ``event`` to ``sink`` swallowing every ``Exception``.

    Producers call this instead of ``sink.emit`` so a faulty sink can never
    break a worker thread. ``None`` sinks are ignored.
    """
    if sink is None:
        return
    try:
        sink.emit(event)
    except Exception as exc:
        log.debug(f"progress sink {type(sink).__name__} failed: {exc!r}")


class BaseSink:
    """Convenience base class: ``emit`` never raises, subclasses override ``handle``."""

    def emit(self, event: ProgressEvent) -> None:
        """Dispatch ``event`` to :meth:`handle`, isolating any ``Exception``."""
        try:
            self.handle(event)
        except Exception as exc:
            log.debug(f"progress sink {type(self).__name__} failed: {exc!r}")

    def handle(self, event: ProgressEvent) -> None:
        """Process one event. Subclasses implement this."""

    def close(self) -> None:
        """Release resources (live displays, files). Idempotent; default no-op."""


class NullSink(BaseSink):
    """Drops every event (default for library use)."""

    def emit(self, event: ProgressEvent) -> None:
        return None


class CompositeSink(BaseSink):
    """Forwards each event to every child sink; one failing child never blocks others."""

    def __init__(self, sinks: Iterable[ProgressSink] = ()) -> None:
        self._lock = threading.Lock()
        self._sinks: list[ProgressSink] = list(sinks)

    @property
    def sinks(self) -> tuple[ProgressSink, ...]:
        """Snapshot of the current children."""
        with self._lock:
            return tuple(self._sinks)

    def add(self, sink: ProgressSink) -> None:
        """Append a child sink."""
        with self._lock:
            self._sinks.append(sink)

    def remove(self, sink: ProgressSink) -> None:
        """Remove a child sink (no error if absent)."""
        with self._lock:
            try:
                self._sinks.remove(sink)
            except ValueError:
                pass

    def handle(self, event: ProgressEvent) -> None:
        for sink in self.sinks:
            safe_emit(sink, event)

    def close(self) -> None:
        for sink in self.sinks:
            closer = getattr(sink, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as exc:
                    log.debug(f"closing sink {type(sink).__name__} failed: {exc!r}")


E = TypeVar("E", bound=ProgressEvent)


class ListSink(BaseSink):
    """Thread-safe in-memory recorder, mainly for tests."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._events: list[ProgressEvent] = []

    def handle(self, event: ProgressEvent) -> None:
        with self._cond:
            self._events.append(event)
            self._cond.notify_all()

    @property
    def events(self) -> list[ProgressEvent]:
        """Copy of the recorded events in arrival order."""
        with self._cond:
            return list(self._events)

    def of_type(self, kind: type[E]) -> list[E]:
        """All recorded events that are instances of ``kind``."""
        with self._cond:
            return [e for e in self._events if isinstance(e, kind)]

    def clear(self) -> None:
        """Forget all recorded events."""
        with self._cond:
            self._events.clear()

    def wait_for(
        self, predicate: Callable[[Sequence[ProgressEvent]], bool], timeout: float = 5.0
    ) -> bool:
        """Block until ``predicate(events)`` holds or ``timeout`` seconds pass."""
        with self._cond:
            return self._cond.wait_for(lambda: predicate(self._events), timeout)

    def __len__(self) -> int:
        with self._cond:
            return len(self._events)


class LoggingSink(BaseSink):
    """Plain-text progress through :mod:`logging` (CI, pipes, non-TTY).

    Per-locus start/finish lines are emitted at ``DEBUG``; failures and
    warnings at ``WARNING``; a compact ``loci done k/N`` line is emitted at
    ``level`` roughly every 10 % (or every ``progress_every`` top-level loci).
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        progress_every: int | None = None,
    ) -> None:
        self._log = logger if logger is not None else log
        self._level = level
        self._progress_every = progress_every
        self._lock = threading.Lock()
        self._total = 0
        self._done = 0
        self._failed = 0

    def _step(self) -> int:
        if self._progress_every is not None:
            return max(1, self._progress_every)
        return max(1, self._total // 10)

    def handle(self, event: ProgressEvent) -> None:
        lg = self._log
        if isinstance(event, RunStarted):
            with self._lock:
                self._total, self._done, self._failed = event.total_loci, 0, 0
            lg.log(
                self._level,
                f"run started: {event.total_loci} loci, mode={event.mode or '-'}, "
                f"{event.threads} threads",
            )
        elif isinstance(event, PhaseChanged):
            suffix = f" ({event.detail})" if event.detail else ""
            lg.log(self._level, f"phase: {event.phase}{suffix}")
        elif isinstance(event, LocusQueued):
            lg.debug(f"queued {event.locus} (work={event.work:.3g})")
        elif isinstance(event, LocusStarted):
            lg.debug(
                f"started {event.locus}: {event.n_seq} seq x {event.max_len}, "
                f"{event.threads} threads, cpus={list(event.cpu_ids)}"
            )
        elif isinstance(event, LocusFinished):
            self._on_finished(event)
        elif isinstance(event, WarningEvent):
            where = f"[{event.locus}] " if event.locus else ""
            lg.warning(f"{where}{event.message}")
        elif isinstance(event, RunFinished):
            lg.log(
                self._level,
                f"run finished: ok={event.ok}, {event.loci_ok} loci ok, "
                f"{event.loci_failed} failed, {event.seconds:.1f}s",
            )

    def _on_finished(self, event: LocusFinished) -> None:
        lg = self._log
        if not event.ok:
            lg.warning(
                f"locus {event.locus} FAILED after {event.seconds:.2f}s: {event.error}"
            )
        else:
            lg.debug(
                f"finished {event.locus} in {event.seconds:.2f}s "
                f"({event.threads} threads)"
            )
        if event.parent is not None:
            return
        with self._lock:
            self._done += 1
            if not event.ok:
                self._failed += 1
            done, total, failed = self._done, self._total, self._failed
            report = total > 0 and (done % self._step() == 0 or done == total)
        if report:
            lg.log(
                self._level,
                f"loci done {done}/{total} ({100.0 * done / total:.0f}%)"
                + (f", {failed} failed" if failed else ""),
            )
