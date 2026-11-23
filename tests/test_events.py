from __future__ import annotations

import dataclasses
import logging
import threading
import unittest

from plastburstalign.events import (
    BaseSink,
    CompositeSink,
    ListSink,
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
    safe_emit,
)


class _Exploding(BaseSink):
    def __init__(self) -> None:
        self.calls = 0

    def handle(self, event: ProgressEvent) -> None:
        self.calls += 1
        raise RuntimeError("boom")


class _BareExploding:
    """Implements only the protocol (no BaseSink protection)."""

    def emit(self, event: ProgressEvent) -> None:
        raise ValueError("bare boom")


class _Interrupting:
    def emit(self, event: ProgressEvent) -> None:
        raise KeyboardInterrupt


class TestEvents(unittest.TestCase):
    def test_positional_construction_and_defaults(self):
        rs = RunStarted(100, "cds", 8)
        self.assertEqual((rs.total_loci, rs.mode, rs.threads), (100, "cds", 8))
        self.assertGreater(rs.ts, 0)
        self.assertEqual(PhaseChanged("align").detail, "")
        lq = LocusQueued("atpA", 120, 1500, 3.5)
        self.assertEqual(lq.parent, None)
        ls = LocusStarted("atpA", 120, 1500, 4, (0, 1, 2, 3), 3.5)
        self.assertEqual(ls.cpu_ids, (0, 1, 2, 3))
        lf = LocusFinished("atpA", True, 1.5, 4)
        self.assertTrue(lf.ok)
        self.assertIsNone(lf.error)
        self.assertEqual(WarningEvent("careful").locus, None)
        self.assertTrue(RunFinished().ok)

    def test_events_are_frozen(self):
        ev = LocusFinished("x", True, 0.1, 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ev.ok = False  # type: ignore[misc]

    def test_timestamp_is_keyword_only(self):
        ev = PhaseChanged("p", ts=12.5)
        self.assertEqual(ev.ts, 12.5)
        with self.assertRaises(TypeError):
            PhaseChanged("p", "d", 12.5)  # type: ignore[misc]


class TestSinks(unittest.TestCase):
    def test_protocol_membership(self):
        for sink in (NullSink(), ListSink(), CompositeSink(), LoggingSink()):
            self.assertIsInstance(sink, ProgressSink)

    def test_null_sink_drops(self):
        NullSink().emit(PhaseChanged("x"))

    def test_list_sink_records_and_filters(self):
        sink = ListSink()
        sink.emit(PhaseChanged("a"))
        sink.emit(LocusQueued("L1"))
        sink.emit(PhaseChanged("b"))
        self.assertEqual(len(sink), 3)
        self.assertEqual([e.phase for e in sink.of_type(PhaseChanged)], ["a", "b"])
        sink.clear()
        self.assertEqual(sink.events, [])

    def test_list_sink_thread_safety(self):
        sink = ListSink()
        n_threads, per_thread = 8, 500

        def work(i: int) -> None:
            for j in range(per_thread):
                sink.emit(LocusQueued(f"{i}-{j}"))

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(sink), n_threads * per_thread)
        self.assertEqual(len({e.locus for e in sink.of_type(LocusQueued)}), len(sink))

    def test_list_sink_wait_for(self):
        sink = ListSink()
        threading.Timer(0.01, lambda: sink.emit(PhaseChanged("go"))).start()
        self.assertTrue(sink.wait_for(lambda evs: len(evs) >= 1, timeout=5))
        self.assertFalse(sink.wait_for(lambda evs: len(evs) >= 99, timeout=0.01))

    def test_base_sink_never_raises(self):
        bad = _Exploding()
        bad.emit(PhaseChanged("x"))  # must not raise
        self.assertEqual(bad.calls, 1)

    def test_safe_emit_swallows_exceptions_but_not_interrupts(self):
        safe_emit(_BareExploding(), PhaseChanged("x"))
        safe_emit(None, PhaseChanged("x"))
        with self.assertRaises(KeyboardInterrupt):
            safe_emit(_Interrupting(), PhaseChanged("x"))

    def test_composite_isolates_failing_children(self):
        good, bad, bare = ListSink(), _Exploding(), _BareExploding()
        comp = CompositeSink([bad, bare, good])
        comp.emit(PhaseChanged("x"))
        comp.emit(PhaseChanged("y"))
        self.assertEqual(len(good), 2)
        self.assertEqual(bad.calls, 2)

    def test_composite_add_remove_close(self):
        a, b = ListSink(), ListSink()
        comp = CompositeSink()
        comp.add(a)
        comp.add(b)
        comp.emit(PhaseChanged("x"))
        comp.remove(b)
        comp.remove(b)  # absent: no error
        comp.emit(PhaseChanged("y"))
        self.assertEqual((len(a), len(b)), (2, 1))
        self.assertEqual(comp.sinks, (a,))
        comp.close()


class TestLoggingSink(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("pba-test-events")
        self.logger.setLevel(logging.DEBUG)

    def test_run_lifecycle_and_progress_milestones(self):
        sink = LoggingSink(self.logger, progress_every=2)
        with self.assertLogs(self.logger, level="INFO") as cm:
            sink.emit(RunStarted(4, "cds", 8))
            sink.emit(PhaseChanged("align", "mafft"))
            for i in range(4):
                sink.emit(LocusStarted(f"L{i}"))
                sink.emit(LocusFinished(f"L{i}", True, 0.1, 1))
            sink.emit(RunFinished(True, 1.0, 4, 0))
        text = "\n".join(cm.output)
        self.assertIn("run started: 4 loci", text)
        self.assertIn("phase: align (mafft)", text)
        self.assertIn("loci done 2/4", text)
        self.assertIn("loci done 4/4", text)
        self.assertNotIn("started L0", text)  # per-locus lines are DEBUG only
        self.assertIn("run finished", text)

    def test_failures_and_warnings_are_logged(self):
        sink = LoggingSink(self.logger)
        with self.assertLogs(self.logger, level="WARNING") as cm:
            sink.emit(LocusFinished("bad", False, 0.5, 2, error="ValueError: x"))
            sink.emit(WarningEvent("tmpfs full", locus="L9"))
        text = "\n".join(cm.output)
        self.assertIn("FAILED", text)
        self.assertIn("ValueError: x", text)
        self.assertIn("[L9] tmpfs full", text)

    def test_child_units_do_not_count_as_loci(self):
        sink = LoggingSink(self.logger, progress_every=1)
        with self.assertLogs(self.logger, level="INFO") as cm:
            sink.emit(RunStarted(1))
            sink.emit(LocusFinished("L/chunk0", True, 0.1, 1, parent="L"))
            sink.emit(LocusFinished("L", True, 0.3, 2))
        done_lines = [m for m in cm.output if "loci done" in m]
        self.assertEqual(len(done_lines), 1)
        self.assertIn("1/1", done_lines[0])


if __name__ == "__main__":
    unittest.main()
