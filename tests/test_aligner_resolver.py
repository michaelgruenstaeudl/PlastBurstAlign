"""Tests for ``plastburstalign.aligner_resolver`` using fake and real executables."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from plastburstalign import aligner_resolver as ar

MAFFT_OK = '#!/bin/sh\necho "v7.505 (2022/Apr/10)" 1>&2\n'
MUSCLE_OK = '#!/bin/sh\necho "muscle 5.1.linux64 []"\n'
CLUSTALO_OK = "#!/bin/sh\necho 1.2.4\n"


def make_exe(directory: Path, name: str, body: str, mode: int = 0o755) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    path.chmod(mode)
    return str(path)


class _ResolverCase(unittest.TestCase):
    def setUp(self) -> None:
        ar.clear_cache()
        self.addCleanup(ar.clear_cache)
        self.tmp = Path(tempfile.mkdtemp(prefix="pba_res_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # keep the dev env's real tools (sys.prefix/bin) out of unit scenarios
        fake_sys = types.SimpleNamespace(prefix=str(self.tmp / "no_prefix"))
        patcher = mock.patch.object(ar, "sys", fake_sys)
        patcher.start()
        self.addCleanup(patcher.stop)

    def env(self, **extra: str) -> dict[str, str]:
        base = {"PATH": str(self.tmp / "empty_bin")}
        base.update(extra)
        return base


class TestVersions(unittest.TestCase):
    def test_parse_and_compare(self) -> None:
        self.assertEqual(ar.parse_version("7.526"), (7, 526))
        self.assertEqual(ar.parse_version("v1.2.4"), (1, 2, 4))
        self.assertEqual(ar.parse_version([5, 3]), (5, 3))
        self.assertGreater(ar.parse_version("7.526"), ar.parse_version("7.310"))
        self.assertLess(ar.parse_version("7.99"), ar.parse_version("7.310"))
        self.assertLess(ar.parse_version("5.0"), ar.parse_version("5.0.1"))
        self.assertEqual(ar.format_version((7, 526)), "7.526")
        with self.assertRaises(ValueError):
            ar.parse_version("no digits")

    def test_kind_aliases(self) -> None:
        self.assertEqual(ar.normalize_kind(" MAFFT "), "mafft")
        for alias in ("clustal", "clustal-omega", "ClustalO", "clustalo"):
            self.assertEqual(ar.normalize_kind(alias), "clustalo")
        with self.assertRaises(ValueError):
            ar.normalize_kind("blast")

    def test_default_minimums(self) -> None:
        self.assertEqual(ar.DEFAULT_MIN_VERSIONS["mafft"], (7, 310))
        self.assertGreaterEqual(ar.DEFAULT_MIN_VERSIONS["muscle"], (5, 0))


class TestProbeAndValidation(_ResolverCase):
    def test_accepts_each_tool_format(self) -> None:
        cases = [
            ("mafft", MAFFT_OK, "7.505"),
            ("muscle", MUSCLE_OK, "5.1"),
            ("clustalo", CLUSTALO_OK, "1.2.4"),
        ]
        for kind, body, version in cases:
            with self.subTest(kind=kind):
                exe = make_exe(self.tmp / "bin", kind, body)
                tool = ar.resolve_aligner(kind, exe, env=self.env())
                assert tool is not None
                self.assertEqual(tool.version, version)
                self.assertEqual(tool.version_tuple, ar.parse_version(version))
                self.assertEqual(
                    (tool.kind, tool.path, tool.source), (kind, exe, "explicit")
                )
                self.assertEqual(tool.warnings, ())

    def test_muscle3_banner_is_rejected_by_min_version(self) -> None:
        exe = make_exe(
            self.tmp, "muscle", '#!/bin/sh\necho "MUSCLE v3.8.31 by Robert C. Edgar"\n'
        )
        self.assertIsNone(ar.resolve_aligner("muscle", exe, env=self.env()))
        res = ar.resolve_aligner_detailed("muscle", exe, env=self.env())
        self.assertIn("older than the required 5.0", res.attempts[0].reason)
        tool = ar.resolve_aligner("muscle", exe, env=self.env(), min_version="3.0")
        assert tool is not None
        self.assertEqual(tool.version, "3.8.31")

    def _reason(self, kind: str, path: str) -> str:
        res = ar.resolve_aligner_detailed(kind, path, env=self.env())
        self.assertIsNone(res.tool)
        self.assertEqual(len(res.attempts), 1)
        return res.attempts[0].reason

    def test_rejects_script_exiting_nonzero(self) -> None:
        exe = make_exe(self.tmp, "mafft", "#!/bin/sh\necho boom 1>&2\nexit 1\n")
        self.assertIn("exited with status 1", self._reason("mafft", exe))

    def test_rejects_non_executable(self) -> None:
        exe = make_exe(self.tmp, "mafft", MAFFT_OK, mode=0o644)
        self.assertFalse(os.access(exe, os.X_OK))
        self.assertEqual(self._reason("mafft", exe), "not executable")

    def test_rejects_missing_and_directory(self) -> None:
        self.assertEqual(
            self._reason("mafft", str(self.tmp / "nope")), "does not exist"
        )
        empty = self.tmp / "emptydir"
        empty.mkdir()
        self.assertIn("directory", self._reason("mafft", str(empty)))

    def test_rejects_bad_interpreter(self) -> None:
        exe = make_exe(
            self.tmp, "mafft", "#!/definitely/not/an/interpreter\necho v7.505\n"
        )
        self.assertIn("cannot execute", self._reason("mafft", exe))

    def test_rejects_wrong_architecture_binary(self) -> None:
        exe = make_exe(self.tmp, "mafft", "\x7fELF\x00\x01 not a real binary \x00\x00")
        self.assertIn("cannot execute", self._reason("mafft", exe))

    def test_rejects_crash_by_signal(self) -> None:
        exe = make_exe(self.tmp, "mafft", "#!/bin/sh\nkill -SEGV $$\n")
        self.assertIn("signal", self._reason("mafft", exe))

    def test_rejects_output_without_version(self) -> None:
        exe = make_exe(self.tmp, "mafft", "#!/bin/sh\necho hello world\n")
        self.assertIn("no version number", self._reason("mafft", exe))

    def test_timeout(self) -> None:
        exe = make_exe(self.tmp, "mafft", "#!/bin/sh\nsleep 5\n")
        res = ar.resolve_aligner_detailed("mafft", exe, env=self.env(), timeout=0.3)
        self.assertIsNone(res.tool)
        self.assertIn("timed out", res.attempts[0].reason)

    def test_min_version_enforcement_and_override(self) -> None:
        old = make_exe(
            self.tmp, "mafft", '#!/bin/sh\necho "v7.100 (2012/Jan/1)" 1>&2\n'
        )
        self.assertIsNone(ar.resolve_aligner("mafft", old, env=self.env()))
        self.assertIsNotNone(
            ar.resolve_aligner("mafft", old, env=self.env(), min_version=(7, 0))
        )
        self.assertIsNotNone(
            ar.resolve_aligner("mafft", old, env=self.env(), min_version="")
        )
        self.assertIsNone(
            ar.resolve_aligner("mafft", old, env=self.env(), min_version="7.200")
        )


class TestLadder(_ResolverCase):
    def test_explicit_beats_env_beats_path(self) -> None:
        explicit = make_exe(self.tmp / "explicit", "mafft", MAFFT_OK)
        override = make_exe(self.tmp / "override", "mafft", MAFFT_OK)
        on_path = make_exe(self.tmp / "pathbin", "mafft", MAFFT_OK)
        env = self.env(PATH=str(self.tmp / "pathbin"), PBA_MAFFT=override)
        tool = ar.resolve_aligner("mafft", explicit, env=env)
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (explicit, "explicit"))
        tool = ar.resolve_aligner("mafft", None, env=env)
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (override, "env:PBA_MAFFT"))
        tool = ar.resolve_aligner(
            "mafft", None, env=self.env(PATH=str(self.tmp / "pathbin"))
        )
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (on_path, "PATH"))

    def test_bad_explicit_falls_through_with_warning(self) -> None:
        bad = make_exe(self.tmp / "bad", "mafft", "#!/bin/sh\nexit 3\n")
        good = make_exe(self.tmp / "pathbin", "mafft", MAFFT_OK)
        tool = ar.resolve_aligner(
            "mafft", bad, env=self.env(PATH=str(self.tmp / "pathbin"))
        )
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (good, "PATH"))
        self.assertEqual(len(tool.warnings), 1)
        self.assertIn("explicit candidate", tool.warnings[0])
        self.assertIn("exited with status 3", tool.warnings[0])

    def test_strict_explicit_does_not_fall_through(self) -> None:
        bad = make_exe(self.tmp / "bad", "mafft", "#!/bin/sh\nexit 3\n")
        make_exe(self.tmp / "pathbin", "mafft", MAFFT_OK)
        env = self.env(PATH=str(self.tmp / "pathbin"))
        self.assertIsNone(
            ar.resolve_aligner("mafft", bad, env=env, strict_explicit=True)
        )
        self.assertIsNotNone(ar.resolve_aligner("mafft", bad, env=env))

    def test_broken_first_path_entry_skipped(self) -> None:
        make_exe(self.tmp / "first", "mafft", "#!/bin/sh\nexit 1\n")
        good = make_exe(self.tmp / "second", "mafft", MAFFT_OK)
        path = os.pathsep.join([str(self.tmp / "first"), str(self.tmp / "second")])
        tool = ar.resolve_aligner("mafft", env=self.env(PATH=path))
        assert tool is not None
        self.assertEqual(tool.path, good)
        self.assertEqual(len(tool.warnings), 1)

    def test_active_prefixes(self) -> None:
        conda = make_exe(self.tmp / "conda" / "bin", "mafft", MAFFT_OK)
        venv = make_exe(self.tmp / "venv" / "bin", "mafft", MAFFT_OK)
        env = self.env(
            CONDA_PREFIX=str(self.tmp / "conda"), VIRTUAL_ENV=str(self.tmp / "venv")
        )
        tool = ar.resolve_aligner("mafft", env=env)
        assert tool is not None
        self.assertEqual(
            (tool.path, tool.source), (venv, "VIRTUAL_ENV")
        )  # venv before conda
        tool = ar.resolve_aligner(
            "mafft", env=self.env(CONDA_PREFIX=str(self.tmp / "conda"))
        )
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (conda, "CONDA_PREFIX"))

    def test_sys_prefix_is_searched(self) -> None:
        prefix_tool = make_exe(self.tmp / "no_prefix" / "bin", "clustalo", CLUSTALO_OK)
        tool = ar.resolve_aligner("clustalo", env=self.env())
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (prefix_tool, "sys.prefix"))

    def test_hpc_module_variables(self) -> None:
        root = self.tmp / "apps" / "MAFFT" / "7.505"
        exe = make_exe(root / "bin", "mafft", MAFFT_OK)
        tool = ar.resolve_aligner("mafft", env=self.env(EBROOTMAFFT=str(root)))
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (exe, "module:EBROOTMAFFT"))
        other = make_exe(self.tmp / "mod2" / "bin", "muscle", MUSCLE_OK)
        tool = ar.resolve_aligner(
            "muscle", env=self.env(MUSCLE_HOME=str(self.tmp / "mod2"))
        )
        assert tool is not None
        self.assertEqual((tool.path, tool.source), (other, "module:MUSCLE_HOME"))
        # a *_BIN variable may point straight at the bin directory
        direct = make_exe(self.tmp / "mod3" / "bin", "mafft", MAFFT_OK)
        tool = ar.resolve_aligner(
            "mafft", env=self.env(MAFFT_BIN=str(self.tmp / "mod3" / "bin"))
        )
        assert tool is not None
        self.assertEqual(tool.path, direct)

    def test_directory_as_explicit_path(self) -> None:
        exe = make_exe(self.tmp / "toolsdir" / "bin", "mafft", MAFFT_OK)
        tool = ar.resolve_aligner("mafft", str(self.tmp / "toolsdir"), env=self.env())
        assert tool is not None
        self.assertEqual(tool.path, exe)

    def test_bare_name_explicit_resolved_on_path(self) -> None:
        exe = make_exe(self.tmp / "pathbin", "mafft", MAFFT_OK)
        tool = ar.resolve_aligner(
            "mafft", "mafft", env=self.env(PATH=str(self.tmp / "pathbin"))
        )
        assert tool is not None
        self.assertEqual(tool.path, exe)

    def test_symlink_aliases_are_probed_once(self) -> None:
        counter = self.tmp / "counter"
        exe = make_exe(
            self.tmp / "bin",
            "mafft",
            f'#!/bin/sh\necho x >> "{counter}"\necho "v7.505" 1>&2\n',
        )
        os.symlink(exe, self.tmp / "bin" / "mafft-alias")
        res = ar.resolve_aligner_detailed(
            "mafft",
            str(self.tmp / "bin" / "mafft-alias"),
            env=self.env(PATH=str(self.tmp / "bin")),
        )
        assert res.tool is not None
        self.assertEqual(len(counter.read_text().split()), 1)

    def test_nothing_found(self) -> None:
        for kind in ar.KINDS:
            self.assertIsNone(ar.resolve_aligner(kind, env=self.env()))
        res = ar.resolve_aligner_detailed("mafft", env=self.env())
        self.assertEqual(res.attempts, ())
        self.assertFalse(res.forced_builtin)


class TestForceBuiltin(_ResolverCase):
    def test_force_flag_skips_everything(self) -> None:
        exe = make_exe(self.tmp / "bin", "mafft", MAFFT_OK)
        base = self.env(PATH=str(self.tmp / "bin"), PBA_MAFFT=exe)
        self.assertIsNotNone(ar.resolve_aligner("mafft", exe, env=base))
        forced = dict(base, PBA_FORCE_BUILTIN_ALIGNER="1")
        self.assertIsNone(ar.resolve_aligner("mafft", exe, env=forced))
        self.assertTrue(ar.builtin_forced(forced))
        res = ar.resolve_aligner_detailed("mafft", exe, env=forced)
        self.assertTrue(res.forced_builtin)
        self.assertEqual(res.attempts, ())

    def test_falsey_values_do_not_force(self) -> None:
        exe = make_exe(self.tmp / "bin", "mafft", MAFFT_OK)
        for value in ("0", "", "no", "false"):
            env = self.env(PATH=str(self.tmp / "bin"), PBA_FORCE_BUILTIN_ALIGNER=value)
            self.assertFalse(ar.builtin_forced(env), value)
            self.assertIsNotNone(ar.resolve_aligner("mafft", env=env), value)
        self.assertTrue(ar.builtin_forced({"PBA_FORCE_BUILTIN_ALIGNER": "TRUE"}))
        self.assertTrue(exe)

    def test_force_flag_read_from_process_env(self) -> None:
        with mock.patch.dict(os.environ, {"PBA_FORCE_BUILTIN_ALIGNER": "1"}):
            self.assertTrue(ar.builtin_forced())
            self.assertIsNone(ar.resolve_aligner("mafft"))


class TestMemoization(_ResolverCase):
    def _counting(self) -> tuple[str, Path]:
        counter = self.tmp / "counter"
        exe = make_exe(
            self.tmp / "bin",
            "mafft",
            f'#!/bin/sh\necho x >> "{counter}"\necho "v7.505" 1>&2\n',
        )
        return exe, counter

    def test_probe_runs_once_per_key(self) -> None:
        exe, counter = self._counting()
        env = self.env()
        first = ar.resolve_aligner("mafft", exe, env=env)
        second = ar.resolve_aligner("mafft", exe, env=env)
        self.assertIs(first, second)
        self.assertEqual(len(counter.read_text().split()), 1)

    def test_clear_cache_forces_reprobe(self) -> None:
        exe, counter = self._counting()
        ar.resolve_aligner("mafft", exe, env=self.env())
        ar.clear_cache()
        ar.resolve_aligner("mafft", exe, env=self.env())
        self.assertEqual(len(counter.read_text().split()), 2)

    def test_changed_environment_is_not_stale(self) -> None:
        make_exe(self.tmp / "a", "mafft", MAFFT_OK)
        b = make_exe(self.tmp / "b", "mafft", '#!/bin/sh\necho "v7.600" 1>&2\n')
        first = ar.resolve_aligner("mafft", env=self.env(PATH=str(self.tmp / "a")))
        second = ar.resolve_aligner("mafft", env=self.env(PATH=str(self.tmp / "b")))
        assert first is not None and second is not None
        self.assertEqual(second.path, b)
        self.assertNotEqual(first.version, second.version)

    def test_negative_result_is_memoized_until_cleared(self) -> None:
        env = self.env(PATH=str(self.tmp / "later"))
        self.assertIsNone(ar.resolve_aligner("mafft", env=env))
        make_exe(self.tmp / "later", "mafft", MAFFT_OK)
        self.assertIsNone(ar.resolve_aligner("mafft", env=env))  # cached miss
        ar.clear_cache()
        self.assertIsNotNone(ar.resolve_aligner("mafft", env=env))

    def test_min_version_is_part_of_the_key(self) -> None:
        old = make_exe(self.tmp / "bin", "mafft", '#!/bin/sh\necho "v7.100" 1>&2\n')
        self.assertIsNone(ar.resolve_aligner("mafft", old, env=self.env()))
        self.assertIsNotNone(
            ar.resolve_aligner("mafft", old, env=self.env(), min_version=(7, 0))
        )


class TestRealTools(unittest.TestCase):
    """Real binaries from the active environment (skipped where not installed)."""

    def setUp(self) -> None:
        ar.clear_cache()
        self.addCleanup(ar.clear_cache)

    def _check(self, kind: str, exe: str, major: int) -> None:
        if not shutil.which(exe):
            self.skipTest(f"{exe} not installed")
        tool = ar.resolve_aligner(
            kind, env=dict(os.environ, PBA_FORCE_BUILTIN_ALIGNER="0")
        )
        assert tool is not None
        self.assertEqual(tool.kind, kind)
        self.assertTrue(os.access(tool.path, os.X_OK))
        self.assertGreaterEqual(tool.version_tuple, (major,))
        self.assertGreaterEqual(tool.version_tuple, ar.DEFAULT_MIN_VERSIONS[kind])

    def test_real_mafft(self) -> None:
        self._check("mafft", "mafft", 7)

    def test_real_muscle(self) -> None:
        self._check("muscle", "muscle", 5)

    def test_real_clustalo(self) -> None:
        self._check("clustalo", "clustalo", 1)

    def test_real_mafft_too_old_for_absurd_minimum(self) -> None:
        if not shutil.which("mafft"):
            self.skipTest("mafft not installed")
        env = dict(os.environ, PBA_FORCE_BUILTIN_ALIGNER="0")
        self.assertIsNone(ar.resolve_aligner("mafft", env=env, min_version="99.0"))

    def test_executable_bit_is_checked_on_real_files(self) -> None:
        exe = shutil.which("mafft")
        if not exe:
            self.skipTest("mafft not installed")
        self.assertTrue(stat.S_IMODE(os.stat(exe).st_mode) & stat.S_IXUSR)


if __name__ == "__main__":
    unittest.main()
