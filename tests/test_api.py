"""Public API (`align` / `estimate` / `autotune`) and CLI UI flag smoke tests."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from plastburstalign import (
    AlignmentResult,
    HostProfile,
    align,
    autotune,
    detect_host,
    estimate,
)
from plastburstalign.__main__ import main
from plastburstalign.api import (
    _api_concat_rename,
    _region_key_for_rename,
    parse_modes,
)
from plastburstalign.estimate import PreflightEstimate
from plastburstalign.events import NullSink
from plastburstalign.supermatrix import legacy_concat_rename
from tests.synth import write_genbank_dataset


class ParseModesTests(unittest.TestCase):
    def test_plus_and_comma(self) -> None:
        self.assertEqual(parse_modes("cds+igs+int"), ("cds", "igs", "int"))
        self.assertEqual(parse_modes("rna,cds"), ("rna", "cds"))

    def test_dedup_and_reject(self) -> None:
        self.assertEqual(parse_modes("cds+cds"), ("cds",))
        with self.assertRaises(ValueError):
            parse_modes("cds+bogus")
        with self.assertRaises(ValueError):
            parse_modes("")


class MultiModeRenameTests(unittest.TestCase):
    def test_region_key_strips_mode_prefix(self) -> None:
        self.assertEqual(_region_key_for_rename("cds_atpA"), "atpA")
        self.assertEqual(_region_key_for_rename("igs_atpF_rpoC1"), "atpF_rpoC1")
        self.assertEqual(_region_key_for_rename("atpA"), "atpA")

    def test_api_rename_collapses_mode_prefixed_loci(self) -> None:
        # Same biological taxon across mode-prefixed partitions must unify.
        self.assertEqual(
            _api_concat_rename("cds_atpA", "atpA_SYN000000"),
            "concat_SYN000000",
        )
        self.assertEqual(
            _api_concat_rename("igs_atpF_rpoC1", "atpF_rpoC1_SYN000000"),
            "concat_SYN000000",
        )
        # Single-mode / bare region still matches legacy behaviour.
        self.assertEqual(
            _api_concat_rename("atpA", "atpA_SYN000000"),
            legacy_concat_rename("atpA", "atpA_SYN000000"),
        )


class ApiE2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="pba_api_")
        cls.root = Path(cls._tmp.name)
        cls.n_taxa = 4
        cls.manifest = write_genbank_dataset(
            cls.root / "gb", n_taxa=cls.n_taxa, n_genes=12, seed=7
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_align_single_mode(self) -> None:
        out = self.root / "out_cds"
        if out.exists():
            shutil.rmtree(out)
        result = align(
            self.manifest.out_dir,
            mode="cds",
            out_dir=out,
            num_threads=2,
            keep_unaligned=False,
            dashboard="off",
            min_num_taxa=1,
            min_seq_length=3,
            individual_nexus=False,
        )
        self.assertIsInstance(result, AlignmentResult)
        self.assertEqual(result.modes, ("cds",))
        self.assertEqual(result.ntax, self.n_taxa)
        self.assertGreater(result.nchar, 0)
        self.assertGreaterEqual(result.informative_sites, 0)
        self.assertTrue(result.locus_report)
        self.assertEqual(result.matrix.shape, (result.ntax, result.nchar))
        self.assertFalse(result.matrix.flags.writeable)
        self.assertIn("taxa", result.summary().lower() + " " + result.summary())
        self.assertTrue(result.manifest_path and Path(result.manifest_path).is_file())
        self.assertTrue(all(t.startswith("concat_") for t in result.taxa))
        fasta = out / "api_export.fasta"
        written = result.write_fasta(fasta, partition_format="raxml")
        self.assertTrue(written.is_file())
        self.assertTrue(fasta.with_name("api_export.partitions.txt").is_file())

    def test_align_multi_mode_prefixes_partitions(self) -> None:
        out = self.root / "out_multi"
        if out.exists():
            shutil.rmtree(out)
        result = align(
            self.manifest.out_dir,
            mode="cds+igs",
            out_dir=out,
            num_threads=2,
            keep_unaligned=False,
            dashboard=False,
            min_num_taxa=1,
            min_seq_length=3,
            individual_nexus=False,
        )
        self.assertEqual(result.modes, ("cds", "igs"))
        self.assertGreater(result.nchar, 0)
        # Regression: mode-prefixed partition names must not inflate ntax to
        # n_taxa * n_loci (was 576 on a 16-taxon / 36-locus synth merge).
        self.assertEqual(result.ntax, self.n_taxa)
        self.assertEqual(len(result.taxa), self.n_taxa)
        self.assertTrue(all(t.startswith("concat_") for t in result.taxa))
        names = [s.name for s in result.locus_report]
        self.assertTrue(any(n.startswith("cds_") for n in names), names)
        self.assertTrue(any(n.startswith("igs_") for n in names), names)
        self.assertTrue((out / "cds").is_dir())
        self.assertTrue((out / "igs").is_dir())
        concat = list(out.glob("nucl_*concat.aligned.fasta"))
        self.assertTrue(concat, "merged concat FASTA missing at out_dir root")
        self.assertTrue(result.manifest_path and Path(result.manifest_path).is_file())

    def test_estimate_returns_preflight(self) -> None:
        est = estimate(
            self.manifest.out_dir,
            mode="igs",
            num_threads=2,
            min_num_taxa=1,
            min_seq_length=3,
        )
        self.assertIsInstance(est, PreflightEstimate)
        self.assertEqual(est.mode, "igs")
        self.assertGreaterEqual(est.n_loci, 1)
        self.assertIn("pre-flight", est.to_text().lower())

    def test_estimate_rejects_multi_mode(self) -> None:
        with self.assertRaises(ValueError):
            estimate(self.manifest.out_dir, mode="cds+igs")

    def test_autotune_and_host_exports(self) -> None:
        plan = autotune(num_threads=2)
        self.assertGreaterEqual(plan.cpu_budget, 1)
        self.assertIsInstance(detect_host(), HostProfile)


class CliFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="pba_cli_")
        cls.root = Path(cls._tmp.name)
        cls.manifest = write_genbank_dataset(
            cls.root / "gb", n_taxa=3, n_genes=10, seed=3
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_version(self) -> None:
        from contextlib import redirect_stderr

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            code = main(["--version"])
        self.assertEqual(code, 0)
        self.assertTrue(buf.getvalue().strip())

    def test_doctor(self) -> None:
        code = main(["--doctor"])
        self.assertEqual(code, 0)

    def test_estimate(self) -> None:
        out = self.root / "est_out"
        out.mkdir(exist_ok=True)
        from contextlib import redirect_stderr

        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            code = main(
                [
                    "--inpd",
                    str(self.manifest.out_dir),
                    "--outd",
                    str(out),
                    "--selectmode",
                    "rna",
                    "--numthreads",
                    "2",
                    "--estimate",
                    "--no-dashboard",
                ]
            )
        self.assertEqual(code, 0)
        text = buf.getvalue().lower()
        self.assertTrue(
            "pre-flight" in text or "wall" in text or "locus" in text,
            text[:500],
        )

    def test_no_dashboard_run(self) -> None:
        out = self.root / "dash_off"
        if out.exists():
            shutil.rmtree(out)
        # Force the builtin aligner so CI without MAFFT still works; prefer
        # real MAFFT when present.
        env = {**os.environ}
        buf = io.StringIO()
        with (
            redirect_stdout(buf),
            mock.patch.dict(os.environ, env, clear=False),
        ):
            code = main(
                [
                    "--inpd",
                    str(self.manifest.out_dir),
                    "--outd",
                    str(out),
                    "--selectmode",
                    "rna",
                    "--numthreads",
                    "2",
                    "--no-dashboard",
                    "--no-keep-unaligned",
                    "--no-individual-nexus",
                    "--concat",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("end of script", buf.getvalue())
        self.assertTrue(
            (out / "run_manifest.json").is_file() or list(out.glob("*.fasta"))
        )


class SinkWiringTests(unittest.TestCase):
    def test_align_accepts_progress_sink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_genbank_dataset(root / "gb", n_taxa=3, n_genes=8, seed=1)
            sink = NullSink()
            result = align(
                manifest.out_dir,
                mode="rna",
                out_dir=root / "out",
                num_threads=2,
                dashboard="off",
                progress_sink=sink,
                min_num_taxa=1,
                individual_nexus=False,
            )
            self.assertGreaterEqual(result.ntax, 1)


if __name__ == "__main__":
    unittest.main()
