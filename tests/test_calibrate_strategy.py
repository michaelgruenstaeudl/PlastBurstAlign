"""Tests of ``benchmarking/calibrate_strategy.py`` (fit logic, runner, smoke mode).

The full grid is never run here; only synthetic timing rows, the timer and the
tiny ``--smoke`` mode (a few seconds, real MAFFT) are exercised.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

from plastburstalign.strategy import AlignmentTier
from plastburstalign.strategy_constants import (
    DEFAULT_CONSTANTS,
    AlphabetModel,
    StrategyConstants,
)

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarking" / "calibrate_strategy.py"
HAS_MAFFT = shutil.which("mafft") is not None


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibrate_strategy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no benchmarking/__pycache__
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


cal = load_script()


def synthetic_rows(
    kappas: dict[AlignmentTier, float], threads: int = 8
) -> list[dict[str, Any]]:
    """Timings that follow exactly ``seconds = kappa * W`` of the default model."""
    rows: list[dict[str, Any]] = []
    for tier, k in kappas.items():
        for n in (50, 100, 200, 400, 800, 1600):
            for length in (400, 800, 1500, 2200, 3000, 4000):
                work = n * float(length) ** DEFAULT_CONSTANTS.nucl.alpha_for(length) + (
                    DEFAULT_CONSTANTS.nucl.beta * n * n * length
                )
                rows.append(
                    {
                        "n": n,
                        "length": length,
                        "alphabet": "nucl",
                        "tier": tier.name,
                        "threads": threads,
                        "seconds": k * work,
                        "returncode": 0,
                        "timed_out": False,
                        "skipped": False,
                        "rss_kb": 1000 + n * n // 10,
                        "mafft_version": "v7.test",
                    }
                )
    return rows


class FitTests(unittest.TestCase):
    def test_boundaries_follow_the_time_budget_of_every_tier(self) -> None:
        kappas = {
            AlignmentTier.ACCURATE: 2.0e-9,
            AlignmentTier.FAST_TREE: 1.0e-9,
            AlignmentTier.HIGH_SCALE: 5.0e-10,
            AlignmentTier.EXTREME_SCALE: 2.0e-10,
        }
        constants, notes = cal.fit_constants(
            synthetic_rows(kappas),
            target_seconds=45.0,
            target_threads=8,
            min_seconds=0.0,
        )
        self.assertTrue(constants.calibrated)
        self.assertIn("v7.test", constants.mafft_version)
        self.assertEqual(constants.target_threads, 8)
        # limits are target_seconds / kappa = 2.25e10, 4.5e10, 9e10 (model fit
        # is on a coarse grid, so allow some slack)
        for got, want in (
            (constants.work_fast_tree, 2.25e10),
            (constants.work_high_scale, 4.5e10),
            (constants.work_extreme_scale, 9.0e10),
        ):
            self.assertAlmostEqual(got / want, 1.0, delta=0.3, msg=notes)
        self.assertLessEqual(constants.work_fast_tree, constants.work_high_scale)
        self.assertLessEqual(constants.work_high_scale, constants.work_extreme_scale)
        self.assertEqual(constants.nucl.alphas, (2.0, 2.2, 2.4))

    def test_a_tier_that_is_not_faster_gets_an_empty_band(self) -> None:
        kappas = {
            AlignmentTier.ACCURATE: 1.0e-9,
            AlignmentTier.FAST_TREE: 3.0e-9,  # slower than --auto: dominated
            AlignmentTier.HIGH_SCALE: 5.0e-10,
            AlignmentTier.EXTREME_SCALE: 2.0e-10,
        }
        constants, notes = cal.fit_constants(synthetic_rows(kappas), min_seconds=0.0)
        self.assertEqual(constants.work_high_scale, constants.work_fast_tree)
        self.assertTrue(any("empty" in n for n in notes), notes)

    def test_fallback_to_available_threads_is_reported(self) -> None:
        constants, notes = cal.fit_constants(
            synthetic_rows({AlignmentTier.ACCURATE: 1e-9}, threads=2),
            target_threads=8,
            min_seconds=0.0,
        )
        self.assertEqual(constants.target_threads, 2)
        self.assertTrue(any("no 8-thread runs" in n for n in notes), notes)

    def test_block_is_valid_python_that_rebuilds_the_constants(self) -> None:
        constants, _ = cal.fit_constants(
            synthetic_rows(
                {AlignmentTier.ACCURATE: 2e-9, AlignmentTier.FAST_TREE: 1e-9}
            ),
            min_seconds=0.0,
        )
        block = cal.format_constants_block(constants)
        namespace: dict[str, Any] = {
            "StrategyConstants": StrategyConstants,
            "AlphabetModel": AlphabetModel,
        }
        exec(block, namespace)  # noqa: S102 - generated by this repo's own script
        rebuilt = namespace["DEFAULT_CONSTANTS"]
        self.assertEqual(rebuilt.nucl, constants.nucl)
        self.assertEqual(rebuilt.source, constants.source)
        self.assertEqual(rebuilt.calibrated, True)
        self.assertAlmostEqual(
            rebuilt.work_fast_tree / constants.work_fast_tree, 1.0, 2
        )

    def test_timeouts_and_skipped_cells_are_ignored(self) -> None:
        rows = synthetic_rows({AlignmentTier.ACCURATE: 2e-9})
        poisoned = [dict(r, timed_out=True, seconds=1e9) for r in rows[:5]]
        skipped = [dict(r, skipped=True, seconds=None) for r in rows[5:9]]
        a, _ = cal.fit_constants(rows, min_seconds=0.0)
        b, _ = cal.fit_constants(rows + poisoned + skipped, min_seconds=0.0)
        self.assertEqual(a.nucl, b.nucl)
        self.assertEqual(a.work_fast_tree, b.work_fast_tree)

    def test_no_rows_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            cal.fit_constants([])


class RunnerTests(unittest.TestCase):
    def test_timer_measures_and_enforces_the_timeout(self) -> None:
        probe = subprocess.Popen(["sleep", "2"], start_new_session=True)
        try:
            probe.kill()
        except PermissionError:
            self.skipTest("this environment forbids signalling child processes")
        finally:
            probe.wait()
        with tempfile.TemporaryDirectory() as tmp:
            seconds, rc, _rss, timed_out = cal.run_timed(["sleep", "0.2"], 5, tmp)
            self.assertFalse(timed_out)
            self.assertEqual(rc, 0)
            self.assertGreaterEqual(seconds, 0.2)
            seconds, rc, _rss, timed_out = cal.run_timed(["sleep", "6"], 0.3, tmp)
            self.assertTrue(timed_out)
            self.assertIsNone(rc)
            self.assertLess(seconds, 5)

    def test_tier_commands(self) -> None:
        self.assertEqual(
            cal.tier_command(AlignmentTier.HIGH_SCALE, 8, "x.fa"),
            [
                "mafft",
                "--parttree",
                "--retree",
                "2",
                "--maxiterate",
                "0",
                "--nofft",
                "--thread",
                "8",
                "x.fa",
            ],
        )

    def test_named_scenario_is_the_monocot_case(self) -> None:
        spec = cal.SCENARIOS["monocot"]
        self.assertEqual((spec["n"], spec["length"]), (700, 3500))

    @unittest.skipUnless(HAS_MAFFT, "mafft not available")
    def test_smoke_mode_runs_a_tiny_grid_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke.json"
            self.assertEqual(cal.main(["--smoke", "--out", str(out)]), 0)
            data = json.loads(out.read_text())
        rows = data["results"]
        self.assertEqual(len(rows), 2 * 2 * 2 * 2)  # N x L x tiers x threads
        self.assertTrue(all(r["returncode"] == 0 for r in rows))
        self.assertTrue(all(r["seconds"] > 0 for r in rows))
        self.assertTrue(all(r["rss_kb"] > 0 for r in rows))
        self.assertEqual({r["tier"] for r in rows}, {"ACCURATE", "FAST_TREE"})
        self.assertIn("cpu", data)
        # the fit accepts what the runner produced
        constants, _notes = cal.fit_constants(rows, min_seconds=0.0, target_threads=2)
        self.assertTrue(constants.calibrated)


if __name__ == "__main__":
    unittest.main()
