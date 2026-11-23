"""Tests for ``plastburstalign.ui`` and ``logging_ops.rich_logging``."""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from rich.console import Console
from rich.logging import RichHandler

from plastburstalign import logging_ops
from plastburstalign.events import (
    CompositeSink,
    ListSink,
    LocusFinished,
    LocusQueued,
    LocusStarted,
    LoggingSink,
    NullSink,
    PhaseChanged,
    RunFinished,
    RunStarted,
    WarningEvent,
)
from plastburstalign.logging_ops import Logger, rich_logging
from plastburstalign.logging_ops import logger as pkg_log
from plastburstalign.scaling import ScalingReport
from plastburstalign.scheduler import FanOut, LocusScheduler, SchedTask
from plastburstalign.ui import (
    RichSink,
    annotate_sink,
    close_sink,
    detect_plain_environment,
    format_cpu_ids,
    format_run_summary_text,
    make_sink,
    render_run_summary,
)


def make_console(*, terminal: bool = True, width: int = 100) -> Console:
    # ``_environ={}``: ignore the host's TERM/NO_COLOR (a dumb TERM would make
    # ``rich`` skip all live-display control codes).
    return Console(
        file=io.StringIO(),
        record=True,
        force_terminal=terminal,
        width=width,
        _environ={},
    )


class _Strict(RichSink):
    """Records exceptions that ``BaseSink.emit`` would otherwise swallow."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.errors: list[BaseException] = []

    def emit(self, event: object) -> None:
        try:
            self.handle(event)  # type: ignore[arg-type]
        except Exception as exc:
            self.errors.append(exc)


class _FakeTTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def render_to_text(sink: RichSink, console: Console | None = None) -> str:
    console = console or make_console(terminal=False)
    console.print(sink.renderable())
    return console.export_text(clear=False)


class TestHelpers(unittest.TestCase):
    def test_format_cpu_ids(self) -> None:
        self.assertEqual(format_cpu_ids(()), "-")
        self.assertEqual(format_cpu_ids((3,)), "3")
        self.assertEqual(format_cpu_ids((0, 1, 2, 3, 8)), "0-3,8")
        self.assertEqual(format_cpu_ids((9, 1, 2, 2, 8)), "1-2,8-9")
        long = format_cpu_ids(tuple(range(0, 100, 2)), max_chars=12)
        self.assertEqual(len(long), 12)
        self.assertTrue(long.endswith("…"))

    def test_detect_plain_environment(self) -> None:
        self.assertIsNone(detect_plain_environment({}))
        self.assertEqual(detect_plain_environment({"CI": "true"}), "CI")
        self.assertEqual(detect_plain_environment({"CI": "1"}), "CI")
        self.assertIsNone(detect_plain_environment({"CI": "false"}))
        self.assertIsNone(detect_plain_environment({"CI": "0", "TERM": "xterm"}))
        self.assertEqual(
            detect_plain_environment({"GITHUB_ACTIONS": "true"}), "GITHUB_ACTIONS"
        )
        self.assertEqual(detect_plain_environment({"NO_COLOR": "1"}), "NO_COLOR")
        self.assertIsNone(detect_plain_environment({"NO_COLOR": ""}))
        self.assertEqual(detect_plain_environment({"TERM": "dumb"}), "TERM=dumb")


class TestRichSinkStream(unittest.TestCase):
    def test_1000_loci_from_8_threads(self) -> None:
        console = make_console()
        sink = _Strict(console)
        self.addCleanup(sink.close)
        n, workers = 1000, 8
        sink.emit(
            RunStarted(n, "igs", workers, details={"loci": {"l0001": "FAST_TREE"}})
        )
        sink.emit(PhaseChanged("align"))
        for i in range(n):
            sink.emit(LocusQueued(f"l{i:04d}", 50, 1000, 1e6))
        errors: list[BaseException] = []

        def worker(wid: int) -> None:
            try:
                for i in range(wid, n, workers):
                    name = f"l{i:04d}"
                    cpus = tuple(range(wid, wid + 1 + i % 4))
                    sink.emit(LocusStarted(name, 50, 1000, 1 + i % 4, cpus, 1e6))
                    if i % 50 == 0:  # fan out into 3 chunks and a finalizer
                        for c in range(3):
                            child = f"{name}/c{c}"
                            sink.emit(LocusQueued(child, 50, 400, 1.0, parent=name))
                            sink.emit(
                                LocusStarted(
                                    child, 50, 400, 1, (wid,), 1.0, parent=name
                                )
                            )
                            sink.emit(LocusFinished(child, True, 0.0, 1, parent=name))
                        fin = f"{name}::finalize"
                        sink.emit(LocusQueued(fin, parent=name))
                        sink.emit(LocusStarted(fin, 0, 0, 1, (wid,), parent=name))
                        sink.emit(LocusFinished(fin, True, 0.0, 1, parent=name))
                    ok = i % 97 != 0
                    sink.emit(
                        LocusFinished(name, ok, 0.01, 2, error=None if ok else "boom")
                    )
                    if i % 200 == 0:
                        sink.emit(WarningEvent(f"odd {name}", locus=name))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
        for t in threads:
            t.start()
        paints = 0
        while True:  # race the refresh thread: paint while the workers emit
            alive = any(t.is_alive() for t in threads)
            self.assertNotIn("unavailable", str(sink.renderable()))
            paints += 1
            if not alive:
                break
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        sink.emit(RunFinished(True, 1.0, 989, 11))
        snap = sink.snapshot()
        failed = len([i for i in range(n) if i % 97 == 0])
        self.assertEqual(failed, 11)
        self.assertEqual((snap.done, snap.failed), (n - failed, failed))
        self.assertEqual(snap.total, n)
        self.assertEqual(snap.chunked, len(range(0, n, 50)))
        self.assertEqual(snap.active, ())
        self.assertEqual(snap.queued, n)
        self.assertEqual(snap.warnings, len(range(0, n, 200)))
        self.assertTrue(snap.finished)
        self.assertFalse(snap.ok)
        self.assertEqual(sink._root_of, {})  # no bookkeeping leaks
        self.assertEqual(sink.errors, [])
        sink.close()
        out = console.export_text(clear=False)
        self.assertIn("1000/1000", out)
        self.assertIn("100%", out)
        self.assertIn("failed", out)
        self.assertGreaterEqual(paints, 1)

    def test_non_terminal_console_gets_only_the_final_plain_frame(self) -> None:
        console = make_console(terminal=False)
        sink = _Strict(console)
        sink.emit(RunStarted(2, "cds", 4))
        sink.emit(LocusStarted("rbcL", 20, 1400, 2, (0, 1)))
        sink.emit(LocusFinished("rbcL", True, 0.5, 2))
        sink.emit(LocusStarted("atpB", 20, 1500, 1, (2,)))
        sink.close()
        raw = console.file.getvalue()  # type: ignore[attr-defined]
        self.assertNotIn("\x1b", raw)
        self.assertEqual(raw.count("PlastBurstAlign"), 1)
        self.assertIn("1/2", raw)
        self.assertIn("atpB", raw)
        self.assertEqual(sink.errors, [])

    def test_close_restores_terminal_logging_and_is_idempotent(self) -> None:
        console = make_console()
        out, err = sys.stdout, sys.stderr
        handlers = list(pkg_log.handlers)
        level = pkg_log.level
        sink = _Strict(console)
        sink.emit(RunStarted(1, "igs", 2))
        self.assertTrue(sink.is_live)
        self.assertTrue(any(isinstance(h, RichHandler) for h in pkg_log.handlers))
        pkg_log.info("hello from inside the live view")
        sink.close()
        sink.close()
        self.assertFalse(sink.is_live)
        self.assertIs(sys.stdout, out)
        self.assertIs(sys.stderr, err)
        self.assertEqual(pkg_log.handlers, handlers)
        self.assertEqual(pkg_log.level, level)
        raw = console.file.getvalue()  # type: ignore[attr-defined]
        self.assertIn("\x1b[?25l", raw)  # cursor hidden while live ...
        self.assertTrue(
            raw.rstrip().endswith("\x1b[?25h") or "\x1b[?25h" in raw
        )  # ... shown again
        self.assertIn(
            "hello from inside the live view", console.export_text(clear=False)
        )
        sink.emit(LocusStarted("late", 1, 1))  # after close: ignored, no error
        sink.emit(RunFinished())
        self.assertEqual(sink.errors, [])

    def test_exception_inside_with_block_still_restores_everything(self) -> None:
        for exc_type in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exc=exc_type.__name__):
                console = make_console()
                out, err = sys.stdout, sys.stderr
                handlers = list(pkg_log.handlers)
                sink = _Strict(console)
                with self.assertRaises(exc_type):
                    with sink:
                        sink.emit(RunStarted(3, "igs", 2))
                        sink.emit(LocusStarted("a", 1, 1))
                        raise exc_type("boom")
                self.assertFalse(sink.is_live)
                self.assertIs(sys.stdout, out)
                self.assertIs(sys.stderr, err)
                self.assertEqual(pkg_log.handlers, handlers)

    def test_transient_dashboard_can_be_closed(self) -> None:
        console = make_console()
        sink = _Strict(console, transient=True)
        sink.emit(RunStarted(1, "igs", 1))
        sink.emit(LocusFinished("x", True))
        sink.close()
        self.assertFalse(sink.is_live)
        self.assertEqual(sink.errors, [])

    def test_refresh_rate_is_capped_at_4_hz(self) -> None:
        console = make_console()
        sink = _Strict(console, refresh_per_second=100)
        self.addCleanup(sink.close)
        sink.emit(RunStarted(1, "igs", 1))
        live = sink._live
        assert live is not None
        self.assertLessEqual(live.refresh_per_second, 4.0)
        self.assertEqual(RichSink(console, refresh_per_second=0.5)._fps, 0.5)

    def test_never_raises_into_callers(self) -> None:
        sink = RichSink(make_console(), live=False)
        sink.emit(object())  # type: ignore[arg-type]
        sink.emit(LocusFinished("never-started", False, error="x", parent="p"))
        with mock.patch.object(RichSink, "_build", side_effect=RuntimeError("render")):
            self.assertIn("unavailable", str(sink.renderable()))
        with mock.patch.object(RichSink, "handle", side_effect=ValueError("bad")):
            sink.emit(RunStarted(1))  # BaseSink.emit swallows
        sink.close()

    def test_second_run_resets_counters_but_keeps_annotations(self) -> None:
        sink = _Strict(make_console(), live=False)
        sink.annotate("a", tier="fast_tree")
        sink.emit(RunStarted(2, "cds", 2))
        sink.emit(LocusFinished("a", False))
        self.assertEqual(sink.snapshot().failed, 1)
        sink.emit(RunStarted(5, "igs", 4))
        snap = sink.snapshot()
        self.assertEqual(
            (snap.total, snap.done, snap.failed, snap.mode), (5, 0, 0, "igs")
        )
        sink.emit(LocusStarted("a", 3, 300, 1, (0,)))
        self.assertEqual(sink.snapshot().active[0].tier, "fast_tree")
        sink.close()


class TestDashboardContent(unittest.TestCase):
    def test_active_table_shows_shape_tier_threads_cpus_and_chunks(self) -> None:
        clock = [0.0]
        console = make_console(terminal=False, width=110)
        sink = _Strict(console, live=False, clock=lambda: clock[0])
        sink.annotate("ycf1", tier="HIGH_SCALE")
        sink.emit(
            RunStarted(3, "igs", 8, details={"loci": {"rpoC2": {"tier": "FAST_TREE"}}})
        )
        sink.emit(LocusStarted("ycf1", 120, 6000, 1, (0,)))  # planning unit
        sink.emit(LocusStarted("rpoC2", 80, 4000, 4, (2, 3, 4, 5)))
        sink.emit(LocusStarted("ycf1/c0", 120, 2500, 2, (6, 7), parent="ycf1"))
        sink.emit(LocusStarted("ycf1/c1", 120, 2500, 3, (8, 9, 10), parent="ycf1"))
        sink.emit(LocusStarted("plain", 10, 500, 1, (11,)))
        sink.emit(LocusStarted("with_meta", 1, 1, 1, (12,)))
        clock[0] = 65.0
        snap = sink.snapshot()
        rows = {r.name: r for r in snap.active}
        self.assertEqual(rows["ycf1"].threads, 5)  # planner unit replaced by its chunks
        self.assertEqual(rows["ycf1"].cpu_ids, (6, 7, 8, 9, 10))
        self.assertEqual(rows["ycf1"].tier, "high_scale")
        self.assertEqual(rows["rpoC2"].tier, "fast_tree")
        self.assertIsNone(rows["plain"].tier)
        self.assertEqual(snap.chunked, 1)
        self.assertEqual(len(snap.active), 4)
        text = render_to_text(sink, console)
        for needle in (
            "ycf1",
            "120 x 6,000",
            "high_scale",
            "fast_tree",
            "6-10",
            "1m05s",
            "rpoC2",
            "80 x 4,000",
        ):
            self.assertIn(needle, text)
        sink.close()

    def test_tier_from_event_meta_and_overflow_row(self) -> None:
        @__import__("dataclasses").dataclass(frozen=True)
        class StartedWithMeta(LocusStarted):
            meta: dict = __import__("dataclasses").field(default_factory=dict)

        sink = _Strict(make_console(terminal=False), live=False, max_active_rows=2)
        sink.emit(
            StartedWithMeta(
                "m", 5, 50, 1, (0,), meta={"strategy": {"tier": "EXTREME_SCALE"}}
            )
        )
        for i in range(3):
            sink.emit(LocusStarted(f"s{i}", 1, 1, 1, (i + 1,)))
        snap = sink.snapshot()
        self.assertEqual(snap.active[0].tier, "extreme_scale")
        self.assertIn("+2 more active", render_to_text(sink))
        sink.close()

    def test_rate_and_eta_follow_an_ema(self) -> None:
        clock = [0.0]
        sink = _Strict(make_console(), live=False, clock=lambda: clock[0])
        sink.emit(RunStarted(100, "igs", 4))
        self.assertIsNone(sink.snapshot().rate_per_min)  # baseline sample
        for i in range(5):
            sink.emit(LocusFinished(f"a{i}", True))
        clock[0] = 1.0
        snap = sink.snapshot()
        self.assertAlmostEqual(snap.rate_per_min or 0.0, 300.0)  # 5 loci/s
        self.assertAlmostEqual(snap.eta_seconds or 0.0, 95 / 5.0)
        for i in range(20):
            sink.emit(LocusFinished(f"b{i}", True))
        clock[0] = 2.0
        snap = sink.snapshot()
        self.assertAlmostEqual(snap.rate_per_min or 0.0, 60 * (0.3 * 20 + 0.7 * 5))
        self.assertAlmostEqual(snap.eta_seconds or 0.0, 75 / (0.3 * 20 + 0.7 * 5))
        text = render_to_text(sink)
        self.assertIn("loci/min", text)
        self.assertIn("ETA", text)
        self.assertIn("25/100", text)
        sink.emit(RunFinished(True, 2.0, 25, 0))
        self.assertEqual(sink.snapshot().eta_seconds, 0.0)
        sink.close()

    def test_warning_and_phase_are_shown(self) -> None:
        sink = _Strict(make_console(terminal=False), live=False)
        sink.emit(RunStarted(2, "igs", 2))
        sink.emit(PhaseChanged("align"))
        sink.emit(WarningEvent("[red]weird[/] markup", locus="loc"))
        text = render_to_text(sink)
        self.assertIn("align", text)
        self.assertIn("[loc] [red]weird[/] markup", text)  # not interpreted as markup
        self.assertIn("warnings 1", text)
        sink.close()


class TestSchedulerIntegration(unittest.TestCase):
    def test_real_scheduler_events_drive_the_dashboard_and_report(self) -> None:
        def work(ctx: object) -> str:
            time.sleep(0.005)
            return "ok"

        def fan(ctx: object) -> FanOut:
            return FanOut(
                subtasks=[SchedTask(f"c{j}", work, work=1.0) for j in range(3)],
                finalizer=lambda c, results: results,
            )

        tasks = [
            SchedTask(f"t{i}", work, work=10.0 - i, n_seq=10, max_len=500)
            for i in range(9)
        ]
        tasks.append(SchedTask("big", fan, work=100.0, n_seq=50, max_len=6000))
        sink = _Strict(make_console(), live=False)
        spy = ListSink()
        both = CompositeSink([sink, spy])
        both.emit(RunStarted(len(tasks), "igs", 4))
        report = LocusScheduler(total_cores=4, sink=both).run(tasks)
        snap = sink.snapshot()
        self.assertEqual((snap.done, snap.failed, snap.chunked), (10, 0, 1))
        self.assertEqual(snap.active, ())
        self.assertEqual(sink.errors, [])
        self.assertTrue(spy.of_type(LocusStarted))

        sr = ScalingReport.from_scheduler(
            report, cores=4, supermatrix_summary={"ntax": 4, "nchar": 40, "n_loci": 2}
        )
        self.assertEqual((sr.n_loci, sr.n_ok, sr.n_chunked), (10, 10, 1))
        text = format_run_summary_text(sr)
        self.assertIn("Diagnosis:", text)
        sink.close()


class TestMakeSink(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, *args: object, **kwargs: object) -> object:
        sink = make_sink(*args, **kwargs)  # type: ignore[arg-type]
        self.addCleanup(close_sink, sink)
        return sink

    def test_explicit_modes(self) -> None:
        self.assertIsInstance(self.make("off"), NullSink)
        self.assertIsInstance(self.make("plain"), LoggingSink)
        self.assertIsInstance(
            self.make("rich", console=make_console(terminal=False)), RichSink
        )
        self.assertIsInstance(self.make("RICH", stream=io.StringIO()), RichSink)
        with self.assertRaises(ValueError):
            make_sink("fancy")

    def test_auto_picks_rich_only_on_a_real_terminal(self) -> None:
        self.assertIsInstance(
            self.make("auto", console=make_console(terminal=True)), RichSink
        )
        self.assertIsInstance(
            self.make("auto", console=make_console(terminal=False)), LoggingSink
        )
        self.assertIsInstance(self.make(stream=_FakeTTY()), RichSink)
        self.assertIsInstance(self.make("auto", stream=io.StringIO()), LoggingSink)

    def test_auto_defaults_to_stderr(self) -> None:
        with mock.patch.object(sys, "stderr", _FakeTTY()):
            self.assertIsInstance(self.make("auto"), RichSink)
        with mock.patch.object(sys, "stderr", io.StringIO()):
            self.assertIsInstance(self.make("auto"), LoggingSink)

    def test_auto_respects_ci_no_color_and_dumb_terminals(self) -> None:
        tty = make_console(terminal=True)
        for env in (
            {"CI": "true"},
            {"GITHUB_ACTIONS": "true"},
            {"NO_COLOR": "1"},
            {"TERM": "dumb"},
        ):
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                self.assertIsInstance(self.make("auto", console=tty), LoggingSink)
                self.assertIsInstance(self.make("auto", stream=_FakeTTY()), LoggingSink)
                # an explicit request still wins
                self.assertIsInstance(self.make("rich", console=tty), RichSink)
        with mock.patch.dict(os.environ, {"CI": "false", "TERM": "xterm"}):
            self.assertIsInstance(self.make("auto", console=tty), RichSink)


class TestRichLogging(unittest.TestCase):
    def setUp(self) -> None:
        self.handlers = list(pkg_log.handlers)
        self.level = pkg_log.level
        self.logger_state = Logger._logger
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        for h in list(pkg_log.handlers):
            pkg_log.removeHandler(h)
        for h in self.handlers:
            pkg_log.addHandler(h)
        pkg_log.setLevel(self.level)
        Logger._logger = self.logger_state

    def test_round_trip_swaps_and_restores_handlers_and_level(self) -> None:
        console = make_console(terminal=False)
        with rich_logging(console) as lg:
            self.assertIs(lg, pkg_log)
            self.assertEqual(
                [type(h) for h in pkg_log.handlers if isinstance(h, RichHandler)],
                [RichHandler],
            )
            plain = [
                h
                for h in pkg_log.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            self.assertEqual(plain, [])
            pkg_log.info("hello rich")
            pkg_log.warning("careful [bold]x[/]")
        self.assertEqual(pkg_log.handlers, self.handlers)
        self.assertEqual(pkg_log.level, self.level)
        text = console.export_text(clear=False)
        self.assertIn("hello rich", text)
        self.assertIn("careful [bold]x[/]", text)  # markup is not interpreted

    def test_exception_inside_the_context_restores(self) -> None:
        console = make_console(terminal=False)
        with self.assertRaises(ValueError):
            with rich_logging(console, verbose=True):
                self.assertEqual(pkg_log.level, logging.DEBUG)
                raise ValueError("boom")
        self.assertEqual(pkg_log.handlers, self.handlers)
        self.assertEqual(pkg_log.level, self.level)

    def test_verbose_enables_debug_lines(self) -> None:
        console = make_console(terminal=False)
        with rich_logging(console, verbose=True):
            pkg_log.debug("dbg line")
        self.assertIn("dbg line", console.export_text(clear=False))
        self.assertIn("(process", console.export_text(clear=False))

    def test_reentrant_nesting(self) -> None:
        outer, inner = make_console(terminal=False), make_console(terminal=False)
        with rich_logging(outer):
            with rich_logging(inner):
                rich_handlers = [
                    h for h in pkg_log.handlers if isinstance(h, RichHandler)
                ]
                self.assertEqual(len(rich_handlers), 1)
                self.assertIs(rich_handlers[0].console, inner)
                pkg_log.info("to inner")
            rich_handlers = [h for h in pkg_log.handlers if isinstance(h, RichHandler)]
            self.assertEqual(len(rich_handlers), 1)
            self.assertIs(rich_handlers[0].console, outer)
            pkg_log.info("to outer")
        self.assertEqual(pkg_log.handlers, self.handlers)
        self.assertIn("to inner", inner.export_text(clear=False))
        self.assertNotIn("to inner", outer.export_text(clear=False))
        self.assertIn("to outer", outer.export_text(clear=False))

    def test_other_handlers_are_left_alone(self) -> None:
        extra = logging.NullHandler()
        pkg_log.addHandler(extra)
        late = logging.NullHandler()
        try:
            with rich_logging(make_console(terminal=False)):
                self.assertIn(extra, pkg_log.handlers)
                pkg_log.addHandler(late)  # attached while active (e.g. assertLogs)
            self.assertIn(extra, pkg_log.handlers)
            self.assertIn(late, pkg_log.handlers)
        finally:
            pkg_log.removeHandler(extra)
            pkg_log.removeHandler(late)

    def test_reinitialize_in_a_subprocess_drops_the_foreign_rich_handler(self) -> None:
        console = make_console(terminal=False)
        foreign = RichHandler(console=console)
        setattr(foreign, logging_ops._RICH_PID_ATTR, os.getpid() + 4242)
        pkg_log.addHandler(foreign)
        Logger.reinitialize_logger()  # what a worker process does
        self.assertNotIn(foreign, pkg_log.handlers)
        plain = [
            h
            for h in pkg_log.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(plain), 1)
        pkg_log.info("worker line")  # must not raise or reach the parent's console
        self.assertNotIn("worker line", console.export_text(clear=False))

    def test_logger_recreated_while_swapped_does_not_duplicate_handlers(self) -> None:
        console = make_console(terminal=False)
        with rich_logging(console):
            Logger.set_logger(False)  # installs a fresh coloredlogs handler
        plain = [
            h
            for h in pkg_log.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(plain), 1)
        self.assertFalse(any(isinstance(h, RichHandler) for h in pkg_log.handlers))


class TestSummary(unittest.TestCase):
    def report(self) -> ScalingReport:
        from test_scaling import make_report, rec

        return ScalingReport.from_scheduler(
            make_report([rec("a", 0, 4, 2), rec("b", 0, 2, 2, order=(1,))], 4), cores=4
        )

    def test_run_summary_text_and_panel(self) -> None:
        sr = self.report()
        stub = SimpleNamespace(
            ntax=7,
            nchar=1234,
            stats=[
                SimpleNamespace(
                    nchar_kept=1234,
                    informative=99,
                    gap_fraction=0.2,
                    missing_fraction=0.1,
                )
            ],
            failed_loci={},
            informative_sites=99,
            variable_sites=150,
            density=0.7,
        )
        text = format_run_summary_text(sr, stub)
        self.assertIn("PlastBurstAlign run summary", text)
        self.assertIn("7 taxa x 1234 sites", text)
        self.assertIn("70.0% data, 10.0% missing, 20.0% gaps", text)
        self.assertIn("Diagnosis:", text)
        console = make_console(terminal=False)
        render_run_summary(console, sr, {"ntax": 2, "nchar": 10, "n_loci": 1})
        out = console.export_text(clear=False)
        self.assertIn("Run summary", out)
        self.assertIn("2 taxa x 10 sites", out)
        render_run_summary(console, sr)  # keeps what the report carries

    def test_annotate_and_close_helpers(self) -> None:
        inner = RichSink(make_console(terminal=False), live=False)
        sink = CompositeSink([NullSink(), inner])
        annotate_sink(sink, "rbcL", tier="ACCURATE")
        annotate_sink(None, "x")
        annotate_sink(NullSink(), "x")
        inner.emit(LocusStarted("rbcL", 1, 1, 1, (0,)))
        self.assertEqual(inner.snapshot().active[0].tier, "accurate")
        close_sink(sink)
        close_sink(None)
        self.assertTrue(inner._closed)


if __name__ == "__main__":
    unittest.main()
