"""Tests for ``plastburstalign.doctor`` (the ``--doctor`` diagnosis)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import hostfs_builder as hb

from plastburstalign import aligner_resolver as ar
from plastburstalign import doctor
from plastburstalign.hardware import HostProfile, detect_host

GIB = 1024**3
MIB = 1024**2
MAFFT_OK = '#!/bin/sh\necho "v7.505 (2022/Apr/10)" 1>&2\n'
MUSCLE_OK = '#!/bin/sh\necho "muscle 5.1.linux64 []"\n'


def _du(path: str) -> tuple[int, int, int]:
    table = {"/dev/shm": (64 * MIB, 0, 64 * MIB)}
    return table.get(path, (100 * GIB, 50 * GIB, 50 * GIB))


class _DoctorCase(unittest.TestCase):
    def setUp(self) -> None:
        ar.clear_cache()
        self.addCleanup(ar.clear_cache)
        self.tmp = Path(tempfile.mkdtemp(prefix="pba_doc_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        patcher = mock.patch.object(
            ar, "sys", types.SimpleNamespace(prefix=str(self.tmp / "no_prefix"))
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def host(self, **kwargs: object) -> HostProfile:
        root = self.tmp / "hostroot"
        hb.build_host_tree(
            root,
            packages=1,
            cores_per_package=4,
            threads_per_core=2,
            mem_total_kb=16 * 1024 * 1024,
            mem_avail_kb=8 * 1024 * 1024,
            mountinfo=hb.MOUNTINFO_EPYC,
            block_devices=hb.EPYC_BLOCK,
        )
        env = kwargs.pop("env", {})
        return detect_host(root, env, system="Linux", machine="x86_64", disk_usage=_du)  # type: ignore[arg-type]

    def make_exe(self, name: str, body: str, directory: str = "bin") -> str:
        d = self.tmp / directory
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        path.write_text(body)
        path.chmod(0o755)
        return str(path)

    def env(self, **extra: str) -> dict[str, str]:
        return {"PATH": str(self.tmp / "emptybin"), **extra}


class TestReportContent(_DoctorCase):
    def test_no_aligners_recommends_install_and_mentions_fallback(self) -> None:
        report = doctor.run_doctor(env=self.env(), host=self.host())
        self.assertFalse(report.external_found)
        self.assertTrue(report.builtin_available)
        self.assertTrue(report.ok)
        recs = "\n".join(report.recommendations)
        self.assertIn("MAFFT not found: conda install -c bioconda mafft", recs)
        self.assertIn("builtin fallback will be used with reduced accuracy", recs)
        text = doctor.format_report(report)
        self.assertIn("NOT FOUND", text)
        self.assertIn("Builtin", text)
        self.assertIn("external aligner missing", text)

    def test_resolved_tools_are_listed_with_version_and_source(self) -> None:
        mafft = self.make_exe("mafft", MAFFT_OK)
        self.make_exe("muscle", MUSCLE_OK)
        env = self.env(PATH=str(self.tmp / "bin"))
        report = doctor.run_doctor(env=env, host=self.host())
        status = {r.kind: r for r in report.tools}
        assert status["mafft"].tool is not None
        self.assertEqual(status["mafft"].tool.path, mafft)
        self.assertEqual(status["mafft"].tool.version, "7.505")
        self.assertIsNone(status["clustalo"].tool)
        text = doctor.format_report(report)
        self.assertIn("v7.505", text)
        self.assertIn("[PATH]", text)
        recs = "\n".join(report.recommendations)
        self.assertNotIn("MAFFT not found", recs)
        self.assertIn("Optional: Clustal Omega not found", recs)
        self.assertNotIn("external aligner missing", text)

    def test_rejected_candidate_reason_is_reported(self) -> None:
        broken = self.make_exe("mafft", "#!/bin/sh\nexit 1\n")
        report = doctor.run_doctor(env=self.env(PBA_MAFFT=broken), host=self.host())
        recs = "\n".join(report.recommendations)
        self.assertIn("MAFFT not found", recs)
        self.assertIn(f"candidate {broken} (env:PBA_MAFFT) was rejected", recs)
        self.assertIn("exited with status 1", doctor.format_report(report))

    def test_too_old_version_is_reported(self) -> None:
        old = self.make_exe("mafft", '#!/bin/sh\necho "v7.100" 1>&2\n')
        report = doctor.run_doctor(env=self.env(PBA_MAFFT=old), host=self.host())
        self.assertIn("older than the required 7.310", doctor.format_report(report))

    def test_fallback_to_later_candidate_surfaces_warning(self) -> None:
        bad = self.make_exe("mafft", "#!/bin/sh\nexit 2\n", directory="bad")
        self.make_exe("mafft", MAFFT_OK, directory="good")
        path = f"{self.tmp / 'bad'}:{self.tmp / 'good'}"
        report = doctor.run_doctor(env=self.env(PATH=path), host=self.host())
        self.assertTrue(report.external_found)
        self.assertIn(f"candidate {bad} rejected", "\n".join(report.recommendations))

    def test_forced_builtin_is_explained(self) -> None:
        self.make_exe("mafft", MAFFT_OK)
        env = self.env(PATH=str(self.tmp / "bin"), PBA_FORCE_BUILTIN_ALIGNER="1")
        report = doctor.run_doctor(env=env, host=self.host())
        self.assertTrue(report.forced_builtin)
        self.assertFalse(report.external_found)
        self.assertIn(
            "PBA_FORCE_BUILTIN_ALIGNER is set", "\n".join(report.recommendations)
        )
        self.assertIn("skipped", doctor.format_report(report))

    def test_small_tmpfs_and_smt_recommendations(self) -> None:
        report = doctor.run_doctor(env=self.env(), host=self.host())
        recs = "\n".join(report.recommendations)
        self.assertIn("RAM-disk /dev/shm has only 64.0 MiB free", recs)
        self.assertIn("SMT detected", recs)
        self.assertIn("--smt", recs)
        smt_report = doctor.run_doctor(env=self.env(), host=self.host(), smt=True)
        self.assertNotIn("SMT detected", "\n".join(smt_report.recommendations))

    def test_tiny_memory_budget_is_called_out(self) -> None:
        report = doctor.run_doctor(
            env=self.env(), host=self.host(), memory_limit="512M"
        )
        self.assertIn("Only", "\n".join(report.recommendations))

    def test_format_contains_every_section(self) -> None:
        text = doctor.format_report(doctor.run_doctor(env=self.env(), host=self.host()))
        for heading in (
            "== Host ==",
            "== Tuning plan",
            "== Aligners ==",
            "== Scratch tiers ==",
            "== Dependencies ==",
            "== Recommendations ==",
            "Status:",
        ):
            self.assertIn(heading, text)
        self.assertIn("/dev/shm", text)
        self.assertIn("MAFFT", text)

    def test_report_serialises_to_json(self) -> None:
        report = doctor.run_doctor(env=self.env(), host=self.host())
        payload = json.loads(json.dumps(report.to_dict(), default=str))
        self.assertEqual(payload["builtin_available"], True)
        self.assertEqual(len(payload["tools"]), 3)


class TestRunDoctorRobustness(_DoctorCase):
    def test_empty_root_degrades_without_crashing(self) -> None:
        empty = self.tmp / "emptyroot"
        empty.mkdir()
        report = doctor.run_doctor(root=str(empty), env=self.env())
        self.assertTrue(report.ok)
        self.assertGreaterEqual(report.plan.cpu_budget, 1)
        self.assertIn("Host probe:", "\n".join(report.recommendations))
        self.assertIsInstance(doctor.format_report(report), str)

    def test_invalid_arguments_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            doctor.run_doctor(env=self.env(), host=self.host(), num_threads="many")


class TestMain(_DoctorCase):
    def _run(self, argv: list[str]) -> tuple[int, str]:
        buf = io.StringIO()
        code = doctor.main(argv, stdout=buf)
        return code, buf.getvalue()

    def _no_aligners_env(self) -> "contextlib.AbstractContextManager[object]":
        """Process env without aligners, even if CI exported the builtin switch."""
        stack = contextlib.ExitStack()
        stack.enter_context(
            mock.patch.dict("os.environ", {"PATH": str(self.tmp / "emptybin")})
        )
        os.environ.pop("PBA_FORCE_BUILTIN_ALIGNER", None)
        return stack

    def test_text_output_and_exit_code_with_builtin_only(self) -> None:
        empty = self.tmp / "emptyroot"
        empty.mkdir()
        with self._no_aligners_env():
            code, out = self._run(["--root", str(empty)])
        self.assertEqual(code, 0)
        self.assertIn("PlastBurstAlign doctor", out)
        self.assertIn("MAFFT not found", out)

    def test_strict_fails_without_external_aligner(self) -> None:
        empty = self.tmp / "emptyroot"
        empty.mkdir()
        with self._no_aligners_env():
            code, _ = self._run(["--root", str(empty), "--strict"])
        self.assertEqual(code, 1)

    def test_json_output_parses(self) -> None:
        empty = self.tmp / "emptyroot"
        empty.mkdir()
        with self._no_aligners_env():
            code, out = self._run(["--root", str(empty), "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIn("recommendations", payload)

    def test_bad_option_value_exits_2(self) -> None:
        empty = self.tmp / "emptyroot"
        empty.mkdir()
        with contextlib.redirect_stderr(io.StringIO()):
            code, _ = self._run(["--root", str(empty), "--num-threads", "lots"])
        self.assertEqual(code, 2)


class TestLiveSmoke(unittest.TestCase):
    def test_live_doctor_runs(self) -> None:
        ar.clear_cache()
        report = doctor.run_doctor()
        text = doctor.format_report(report)
        self.assertIn("== Host ==", text)
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
