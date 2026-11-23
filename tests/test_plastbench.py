"""Unit tests for ``benchmarking/plastbench/plastbench.py``.

Everything here is tiny, offline and fast: no network (``fetch`` is exercised
through ``file://`` URLs), no PlastBurstAlign run and no benchmark.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarking" / "plastbench" / "plastbench.py"
MANIFEST = REPO_ROOT / "benchmarking" / "plastbench" / "datasets.json"
ASTERACEAE_ARCHIVE = (
    REPO_ROOT / "benchmarking" / "benchmark_YangEtAl2022_Asteraceae.tar.gz"
)


def _load_module() -> Any:
    spec = importlib_util.spec_from_file_location("plastbench", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib_util.module_from_spec(spec)
    sys.modules["plastbench"] = module  # needed by dataclasses
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # keep benchmarking/ free of __pycache__
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


pb = _load_module()


def make_results(
    pba_wall: float | None = 2.0,
    pba_rss: int = 512 * 1024**2,
    mafft_wall: float | None = 8.0,
    n: int = 100,
    pba_status: str = "ok",
) -> dict[str, Any]:
    """A schema-valid results document with PBA and a MAFFT baseline."""

    def tool(status: str, wall: float | None, rss: int | None) -> dict[str, Any]:
        entry = {
            "status": status,
            "reason": None if status == "ok" else "boom",
            "command": ["x"],
            "threads": 1,
            "wall_seconds": wall,
            "peak_rss_bytes": rss,
            "phases": None,
            "n_loci": 10,
            "ntax": n,
            "nchar": 12345,
            "matrix_sha256": None,
            "tree": None,
        }
        return entry

    tools: dict[str, Any] = {
        "pba": tool(pba_status, pba_wall, pba_rss),
        "mafft": tool(
            "ok" if mafft_wall else "skipped",
            mafft_wall,
            256 * 1024**2 if mafft_wall else None,
        ),
    }
    tools["mafft"]["tree"] = {"method": "fasttree", "normalized_rf": 0.125}
    return {
        "schema_version": 1,
        "generator": "plastbench 1",
        "created_utc": "2026-01-01T00:00:00Z",
        "dataset": {
            "name": "benchmark_YangEtAl2022_Asteraceae",
            "tier": "smoke",
            "n_genomes": n,
            "seed": "plastbench-v1",
            "listing_sha256": None,
            "mode": "igs",
            "min_num_taxa": 5,
            "min_seq_length": 6,
        },
        "host": {"system": "Linux"},
        "tool_versions": {"python": "3.11"},
        "baseline": "mafft",
        "settings": {},
        "tools": tools,
        "notes": ["a note"],
    }


class TempDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


class SubsamplerTests(TempDirTestCase):
    NAMES = [f"GENOME_{i:03d}.gb" for i in range(60)]

    def test_same_seed_same_list_and_input_order_is_irrelevant(self) -> None:
        first = pb.select_subset(self.NAMES, 10, "s1")
        again = pb.select_subset(list(reversed(self.NAMES)), 10, "s1")
        self.assertEqual(first, again)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(set(first)), 10)

    def test_different_seed_gives_different_list(self) -> None:
        self.assertNotEqual(
            pb.select_subset(self.NAMES, 10, "seed-a"),
            pb.select_subset(self.NAMES, 10, "seed-b"),
        )

    def test_matches_the_documented_method(self) -> None:
        # sha1(filename + NUL + seed), ascending; independent re-implementation.
        seed = "plastbench-v1"
        expected = sorted(
            self.NAMES,
            key=lambda n: (hashlib.sha1(f"{n}\0{seed}".encode()).hexdigest(), n),
        )[:7]
        self.assertEqual(pb.select_subset(self.NAMES, 7), expected)
        self.assertEqual(pb.DEFAULT_SEED, seed)

    def test_subsets_are_nested(self) -> None:
        small = pb.select_subset(self.NAMES, 5)
        large = pb.select_subset(self.NAMES, 25)
        self.assertEqual(large[:5], small)

    def test_stable_across_pythonhashseed(self) -> None:
        src = self.root / "src"
        src.mkdir()
        for name in self.NAMES:
            (src / name).write_text(">x\nACGT\n")
        listings = []
        for hash_seed in ("0", "1", "4242"):
            out = self.root / f"out_{hash_seed}"
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "subsample", "--src", str(src)]
                + ["--out", str(out), "-n", "12"],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            listings.append(json.loads((out / "subset.json").read_text())["files"])
        self.assertEqual(listings[0], listings[1])
        self.assertEqual(listings[0], listings[2])
        self.assertEqual(listings[0], pb.select_subset(self.NAMES, 12))

    def test_write_subset_materializes_links_and_listing(self) -> None:
        src = self.root / "src"
        src.mkdir()
        for name in self.NAMES[:20]:
            (src / name).write_text(">x\nACGT\n")
        (src / "notes.txt").write_text("ignored")
        out = self.root / "subset"
        doc = pb.write_subset(src, out, 8)
        written = json.loads((out / "subset.json").read_text())
        self.assertEqual(doc, written)
        self.assertEqual(written["n_selected"], 8)
        self.assertEqual(written["n_available"], 20)
        self.assertEqual(written["listing_sha256"], pb.listing_sha256(written["files"]))
        self.assertEqual(
            sorted(p.name for p in out.glob("*.gb")), sorted(written["files"])
        )
        self.assertTrue(
            all((out / n).read_text().startswith(">x") for n in written["files"])
        )
        # subset.json carries no absolute paths: it is byte-reproducible.
        self.assertNotIn(str(self.root), (out / "subset.json").read_text())

    def test_refuses_too_few_files_and_non_empty_output(self) -> None:
        src = self.root / "src"
        src.mkdir()
        for name in self.NAMES[:5]:
            (src / name).write_text(">x\nA\n")
        with self.assertRaises(SystemExit):
            pb.write_subset(src, self.root / "o1", 6)
        doc = pb.write_subset(src, self.root / "o2", 6, allow_fewer=True)
        self.assertEqual(doc["n_selected"], 5)
        with self.assertRaises(SystemExit):
            pb.write_subset(src, self.root / "o2", 3)
        again = pb.write_subset(src, self.root / "o2", 3, force=True)
        self.assertEqual(again["n_selected"], 3)
        self.assertEqual(len(list((self.root / "o2").glob("*.gb"))), 3)


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = pb.load_manifest(MANIFEST)

    def test_shipped_manifest_is_valid(self) -> None:
        self.assertEqual(pb.validate_manifest(self.manifest), [])

    def test_every_entry_has_the_required_fields(self) -> None:
        for entry in self.manifest["datasets"]:
            for field in pb.REQUIRED_DATASET_FIELDS:
                self.assertIn(field, entry, entry["name"])

    def test_asteraceae_is_the_real_published_dataset(self) -> None:
        entry = pb.find_dataset(self.manifest, "benchmark_YangEtAl2022_Asteraceae")
        self.assertEqual(entry["zenodo_record"], "17137737")
        self.assertEqual(entry["status"], "published")
        self.assertIn("/records/17137737/files/", entry["url"])
        self.assertIn(entry["name"] + ".tar.gz", entry["url"])
        self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")

    @unittest.skipUnless(ASTERACEAE_ARCHIVE.is_file(), "local archive not downloaded")
    def test_recorded_sha256_matches_the_local_archive(self) -> None:
        entry = pb.find_dataset(self.manifest, "benchmark_YangEtAl2022_Asteraceae")
        self.assertEqual(entry["sha256"], pb.sha256_file(ASTERACEAE_ARCHIVE))

    def test_all_tiers_are_defined_and_unpublished_ones_have_todos(self) -> None:
        self.assertEqual(set(self.manifest["tiers"]), set(pb.TIERS))
        for tier, spec in self.manifest["tiers"].items():
            self.assertEqual(spec["n_genomes"], pb.TIERS[tier])
            if spec["status"] == "unpublished":
                self.assertTrue(spec["todo"], tier)
        self.assertEqual(self.manifest["tiers"]["smoke"]["status"], "published")

    def test_unpublished_datasets_have_no_checksum(self) -> None:
        for entry in self.manifest["datasets"]:
            if entry["status"] == "unpublished":
                self.assertIsNone(entry["sha256"])
                self.assertTrue(entry["todo"])

    def test_validator_reports_problems(self) -> None:
        broken = json.loads(json.dumps(self.manifest))
        del broken["datasets"][0]["url"]
        broken["datasets"][0]["sha256"] = "XYZ"
        broken["datasets"].append(dict(broken["datasets"][1]))  # duplicate name
        problems = "\n".join(pb.validate_manifest(broken))
        self.assertIn("missing field `url`", problems)
        self.assertIn("sha256", problems)
        self.assertIn("duplicate name", problems)
        self.assertTrue(pb.validate_manifest([]))
        self.assertTrue(pb.validate_manifest({"schema_version": 99}))


class ResultsAndBadgeTests(unittest.TestCase):
    def test_fixture_is_schema_valid_and_validation_catches_errors(self) -> None:
        self.assertEqual(pb.validate_results(make_results()), [])
        bad = make_results()
        bad["tools"]["pba"]["wall_seconds"] = -1
        bad["tools"]["mafft"]["status"] = "weird"
        problems = "\n".join(pb.validate_results(bad))
        self.assertIn("positive wall_seconds", problems)
        self.assertIn("status", problems)

    def test_speedup(self) -> None:
        self.assertAlmostEqual(pb.speedup(make_results(), "pba", "mafft"), 4.0)
        self.assertIsNone(pb.speedup(make_results(mafft_wall=None), "pba", "mafft"))
        self.assertIsNone(pb.speedup(make_results(pba_status="failed"), "pba", "mafft"))

    def test_badge_content(self) -> None:
        badges = pb.make_badges(make_results(pba_wall=2.0, mafft_wall=8.0))
        self.assertEqual(set(badges), {"speedup.json", "peak_memory.json"})
        speedup = badges["speedup.json"]
        self.assertEqual(speedup["schemaVersion"], 1)
        self.assertEqual(speedup["message"], "4.0x")
        self.assertEqual(speedup["color"], "green")
        self.assertIn("N=100", speedup["label"])
        memory = badges["peak_memory.json"]
        self.assertEqual(memory["schemaVersion"], 1)
        self.assertEqual(memory["message"], "512 MiB")
        self.assertEqual(memory["color"], "brightgreen")
        for badge in badges.values():
            self.assertEqual(set(badge), {"schemaVersion", "label", "message", "color"})

    def test_badge_colors_and_failure(self) -> None:
        fast = pb.make_badges(make_results(pba_wall=0.5, mafft_wall=8.0))
        self.assertEqual(fast["speedup.json"]["color"], "brightgreen")
        slow = pb.make_badges(make_results(pba_wall=10.0, mafft_wall=8.0))
        self.assertEqual(slow["speedup.json"]["color"], "orange")
        huge = pb.make_badges(make_results(pba_rss=20 * 1024**3))
        self.assertEqual(huge["peak_memory.json"]["message"], "20.00 GiB")
        self.assertEqual(huge["peak_memory.json"]["color"], "orange")
        failed = pb.make_badges(make_results(pba_status="failed", pba_wall=None))
        self.assertEqual(failed["speedup.json"]["message"], "n/a")
        self.assertEqual(failed["speedup.json"]["color"], "lightgrey")
        self.assertEqual(failed["peak_memory.json"]["message"], "n/a")

    def test_write_badges_uses_the_largest_tier_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = pb.write_badges(
                [make_results(n=100), make_results(n=1000, pba_wall=1.0)], Path(tmp)
            )
            self.assertEqual(
                {p.name for p in paths}, {"speedup.json", "peak_memory.json"}
            )
            payload = json.loads((Path(tmp) / "speedup.json").read_text())
            self.assertIn("N=1000", payload["label"])
            self.assertEqual(payload["message"], "8.0x")

    def test_report_renders_a_markdown_table_with_speedups(self) -> None:
        text = pb.render_report([make_results(n=1000), make_results(n=100)])
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("| Dataset |"))
        self.assertIn("Speedup vs mafft", lines[0])
        self.assertTrue(lines[1].startswith("|---"))
        rows = [line for line in lines if line.startswith("| smoke")]
        self.assertEqual(len(rows), 4)  # 2 documents x (pba, mafft)
        # sorted by N: the N=100 rows come first
        self.assertIn("| 100 |", rows[0])
        self.assertIn("| 1000 |", rows[2])
        pba_row, mafft_row = rows[0], rows[1]
        self.assertIn("PlastBurstAlign", pba_row)
        self.assertIn("4.00x", pba_row)
        self.assertIn("512 MiB", pba_row)
        self.assertIn("1.00x", mafft_row)
        self.assertIn("0.125", mafft_row)  # nRF of the baseline fixture
        self.assertIn("- a note", text)

    def test_report_marks_missing_baseline_as_na_and_honors_override(self) -> None:
        text = pb.render_report([make_results(mafft_wall=None)])
        pba_row = next(line for line in text.splitlines() if "PlastBurstAlign" in line)
        self.assertIn("n/a", pba_row)
        self.assertIn("skipped", text)
        other = make_results()
        other["tools"]["muscle"] = dict(other["tools"]["mafft"], wall_seconds=4.0)
        text = pb.render_report([other], baseline="muscle")
        pba_row = next(line for line in text.splitlines() if "PlastBurstAlign" in line)
        self.assertIn("2.00x", pba_row)

    def test_cli_report_and_badges_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "r.json"
            results.write_text(json.dumps(make_results()))
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                self.assertEqual(pb.main(["report", str(results)]), 0)
                self.assertEqual(
                    pb.main(
                        ["badges", str(results), "--out-dir", str(Path(tmp) / "b")]
                    ),
                    0,
                )
            self.assertIn("4.00x", buffer.getvalue())
            self.assertTrue((Path(tmp) / "b" / "speedup.json").is_file())
            results.write_text(json.dumps({"schema_version": 7}))
            with self.assertRaises(SystemExit):
                pb.main(["report", str(results)])


class PeakRssSamplerTests(unittest.TestCase):
    ALLOC = 48 * 1024 * 1024

    def test_samples_peak_rss_of_a_trivial_subprocess(self) -> None:
        code = f"data = b'a' * {self.ALLOC}; import time; time.sleep(0.6); print(len(data))"
        measured = pb.run_measured([sys.executable, "-c", code], interval=0.02)
        self.assertEqual(measured.returncode, 0)
        self.assertFalse(measured.timed_out)
        self.assertGreaterEqual(measured.wall_seconds, 0.6)
        self.assertGreater(measured.samples, 3)
        self.assertGreater(measured.peak_rss_bytes, self.ALLOC * 0.8)
        self.assertLess(measured.peak_rss_bytes, 4 * 1024**3)
        self.assertIn(str(self.ALLOC), measured.log_tail)

    def test_includes_grandchildren_in_the_tree(self) -> None:
        child = f"data = b'a' * {self.ALLOC}; import time; time.sleep(0.8)"
        parent = (
            f"import subprocess, sys;subprocess.run([sys.executable, '-c', {child!r}])"
        )
        measured = pb.run_measured([sys.executable, "-c", parent], interval=0.02)
        self.assertEqual(measured.returncode, 0)
        self.assertGreater(measured.peak_rss_bytes, self.ALLOC * 0.8)

    def test_timeout_kills_the_process_tree(self) -> None:
        started = time.perf_counter()
        measured = pb.run_measured(
            [sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.4
        )
        self.assertTrue(measured.timed_out)
        self.assertIsNone(measured.returncode)
        self.assertLess(time.perf_counter() - started, 20)

    def test_nonzero_exit_status_is_reported(self) -> None:
        measured = pb.run_measured([sys.executable, "-c", "import sys; sys.exit(3)"])
        self.assertEqual(measured.returncode, 3)


class ConcatenationTests(TempDirTestCase):
    def _write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text)
        return path

    def _fixtures(self) -> list[Path]:
        a = self._write(
            "nucl_geneA.aligned.fasta",
            ">geneA_t1\nAC-GT\n>geneA_t2\nACCGT\n",
        )
        b = self._write(
            "nucl_geneB.aligned.fasta",
            ">geneB_t2\nTT\n>geneB_t3\nTG\n",
        )
        return [a, b]

    def test_missing_taxa_are_gap_filled(self) -> None:
        out = self.root / "concat.fasta"
        ntax, nchar = pb.concatenate_alignments(self._fixtures(), out)
        self.assertEqual((ntax, nchar), (3, 7))
        self.assertEqual(
            pb.read_fasta(out),
            [
                ("concat_t1", "AC-GT--"),
                ("concat_t2", "ACCGTTT"),
                ("concat_t3", "-----TG"),
            ],
        )
        self.assertEqual(pb.fasta_dimensions(out), (3, 7))

    def test_locus_and_taxon_helpers(self) -> None:
        self.assertEqual(pb.locus_name("a/nucl_trnL_UAA.unalign.fasta"), "trnL_UAA")
        self.assertEqual(pb.locus_name("rbcL.aligned.fasta"), "rbcL")
        self.assertEqual(
            pb.taxon_from_header("trnL_UAA_NC_007578", "trnL_UAA"), "NC_007578"
        )
        self.assertEqual(pb.taxon_from_header("other desc", "trnL_UAA"), "other")

    @unittest.skipUnless(
        shutil.which("bash") and shutil.which("awk"), "bash and awk required"
    )
    def test_generated_shell_pipeline_matches_the_python_concatenation(self) -> None:
        unaligned = self.root / "unaligned"
        unaligned.mkdir()
        for path in self._fixtures():
            shutil.copy(path, unaligned / path.name.replace(".aligned", ""))
        script = pb.write_shell_pipeline(self.root / "pipeline.sh")
        text = script.read_text()
        self.assertIn("mafft --auto --thread 1", text)
        self.assertIn("PLASTBENCH_ALIGNER", text)
        out = self.root / "shell_out"
        # `cat` stands in for the aligner: the fixtures are already aligned.
        proc = subprocess.run(
            ["bash", str(script), str(unaligned), str(out)],
            capture_output=True,
            text=True,
            env=dict(os.environ, PLASTBENCH_ALIGNER="cat"),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        expected = self.root / "expected.fasta"
        pb.concatenate_alignments(sorted((out / "aligned").glob("*.fasta")), expected)
        self.assertEqual(
            (out / "concat.aligned.fasta").read_text(), expected.read_text()
        )

    @unittest.skipUnless(shutil.which("mafft"), "MAFFT not available")
    def test_mafft_baseline_loop_on_two_tiny_loci(self) -> None:
        unaligned = self.root / "unaligned"
        unaligned.mkdir()
        (unaligned / "geneA.fasta").write_text(
            ">geneA_t1\nACGTACGTAC\n>geneA_t2\nACGACGTAC\n>geneA_t3\nACGTACGTAC\n"
        )
        (unaligned / "geneB.fasta").write_text(
            ">geneB_t1\nGGGTTTAAAC\n>geneB_t2\nGGGTTAAAC\n"
        )
        summary = pb.baseline_loop("mafft", unaligned, self.root / "out")
        self.assertEqual(summary["n_loci"], 2)
        self.assertEqual(summary["n_failed_loci"], 0)
        self.assertEqual(summary["ntax"], 3)
        self.assertGreaterEqual(summary["nchar"], 20)
        self.assertEqual(
            pb.fasta_dimensions(self.root / "out" / "concat.aligned.fasta"),
            (summary["ntax"], summary["nchar"]),
        )


class NewickTests(unittest.TestCase):
    def test_identical_and_different_topologies(self) -> None:
        a = "((A,B),(C,D),(E,F));"
        self.assertEqual(pb.robinson_foulds(a, a)["rf"], 0)
        b = "((A,C),(B,D),(E,F));"
        result = pb.robinson_foulds(a, b)
        self.assertEqual(result["rf"], 4)  # (A,B)/(C,D) vs (A,C)/(B,D); (E,F) shared
        self.assertEqual(result["n_shared_taxa"], 6)
        self.assertAlmostEqual(result["normalized_rf"], 4 / 6)

    def test_rooted_and_unrooted_forms_of_a_tree_agree(self) -> None:
        rooted = "(((A:1,B:1):1,C:1):1,(D:1,(E:1,F:1):1):1);"
        unrooted = "((A,B),C,(D,(E,F)));"
        self.assertEqual(pb.robinson_foulds(rooted, unrooted)["rf"], 0)

    def test_labels_support_values_comments_and_concat_prefix(self) -> None:
        a = "((concat_A:0.1,'B':0.2)0.95:0.1,[note](C,D)90,E);"
        b = "((A,B),(C,D),E);"
        self.assertEqual(pb.robinson_foulds(a, b)["rf"], 0)

    def test_uses_only_shared_taxa_and_handles_tiny_trees(self) -> None:
        a = "((A,B),(C,D),(E,F),X);"
        b = "((A,B),(C,D),(E,F),Y);"
        result = pb.robinson_foulds(a, b)
        self.assertEqual(result["n_shared_taxa"], 6)
        self.assertEqual(result["rf"], 0)
        self.assertIsNone(pb.robinson_foulds("(A,B,C);", "(A,B,C);")["rf"])

    def test_polytomy_distance(self) -> None:
        resolved = "((A,B),(C,D),(E,F));"
        star = "(A,B,C,D,E,F);"
        result = pb.robinson_foulds(resolved, star)
        self.assertEqual(result["rf"], 3)
        self.assertAlmostEqual(result["normalized_rf"], 1.0)

    def test_deep_caterpillar_does_not_overflow_the_stack(self) -> None:
        n = 3000
        newick = "(" * (n - 1) + "t0"
        for i in range(1, n):
            newick += f",t{i})"
        newick += ";"
        tips, clades = pb.parse_newick(newick)
        self.assertEqual(len(tips), n)
        self.assertEqual(len(clades), n - 2)

    def test_unbalanced_input_raises(self) -> None:
        with self.assertRaises(ValueError):
            pb.parse_newick("((A,B);")


class FetchTests(TempDirTestCase):
    def _make_archive(self, members: dict[str, bytes], name: str = "toy") -> Path:
        archive = self.root / f"{name}_src.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for member, payload in members.items():
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        return archive

    def _entry(self, archive: Path, sha256: str | None) -> dict[str, Any]:
        return {
            "name": "toy",
            "n_genomes": 2,
            "source": "test",
            "zenodo_record": "0",
            "url": archive.resolve().as_uri(),
            "sha256": sha256,
            "status": "published",
        }

    def test_download_verify_extract_and_idempotent_refetch(self) -> None:
        archive = self._make_archive({"toy/a.gb": b"AAA", "toy/b.gb": b"BBB"})
        entry = self._entry(archive, pb.sha256_file(archive))
        dest = self.root / "data"
        path = pb.fetch_dataset(entry, dest)
        self.assertEqual(path, dest / "toy")
        self.assertEqual(sorted(p.name for p in path.iterdir()), ["a.gb", "b.gb"])
        self.assertTrue((dest / "toy.fetched.json").is_file())
        self.assertFalse(list(dest.glob("*.part")))
        self.assertFalse(list(dest.glob(".extract-*")))
        # second call is a no-op even if the source disappeared
        archive.unlink()
        self.assertEqual(pb.fetch_dataset(entry, dest), path)

    def test_checksum_mismatch_is_rejected_and_nothing_is_extracted(self) -> None:
        archive = self._make_archive({"toy/a.gb": b"AAA"})
        entry = self._entry(archive, "0" * 64)
        with self.assertRaises(SystemExit):
            pb.fetch_dataset(entry, self.root / "data")
        self.assertFalse((self.root / "data" / "toy").exists())

    def test_missing_checksum_needs_explicit_opt_in(self) -> None:
        archive = self._make_archive({"toy/a.gb": b"AAA"})
        entry = self._entry(archive, None)
        with self.assertRaises(SystemExit):
            pb.fetch_dataset(entry, self.root / "data")
        path = pb.fetch_dataset(entry, self.root / "data", allow_unverified=True)
        self.assertTrue((path / "a.gb").is_file())

    def test_md5_from_the_manifest_is_used_when_there_is_no_sha256(self) -> None:
        archive = self._make_archive({"toy/a.gb": b"AAA"})
        entry = self._entry(archive, None)
        entry["md5"] = pb.md5_file(archive)
        path = pb.fetch_dataset(entry, self.root / "data")
        self.assertTrue((path / "a.gb").is_file())
        entry["md5"] = "f" * 32
        with self.assertRaises(SystemExit):
            pb.fetch_dataset(entry, self.root / "data2")

    def test_path_traversal_archives_are_rejected(self) -> None:
        archive = self._make_archive({"../evil.txt": b"x"})
        entry = self._entry(archive, pb.sha256_file(archive))
        with self.assertRaises(SystemExit):
            pb.fetch_dataset(entry, self.root / "data")
        self.assertFalse((self.root / "evil.txt").exists())

    def test_corrupt_existing_archive_is_replaced(self) -> None:
        archive = self._make_archive({"toy/a.gb": b"AAA"})
        entry = self._entry(archive, pb.sha256_file(archive))
        dest = self.root / "data"
        dest.mkdir()
        (dest / "toy.tar.gz").write_bytes(b"truncated garbage")
        path = pb.fetch_dataset(entry, dest)
        self.assertTrue((path / "a.gb").is_file())

    def _serve(self, payload: bytes, support_range: bool) -> tuple[str, list[Any]]:
        """Serve ``payload`` on localhost (optionally with Range support)."""
        seen: list[Any] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                wanted = self.headers.get("Range")
                seen.append(wanted)
                body, status = payload, 200
                if wanted and support_range:
                    body, status = payload[int(wanted.split("=")[1][:-1]) :], 206
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:  # pragma: no cover - sandboxed environments
            self.skipTest(f"cannot bind a localhost socket: {exc}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/toy.tar.gz", seen

    def test_interrupted_download_is_resumed_with_a_range_request(self) -> None:
        archive = self._make_archive({"toy/a.gb": b"A" * 5000, "toy/b.gb": b"B" * 5000})
        payload = archive.read_bytes()
        url, seen = self._serve(payload, support_range=True)
        dest = self.root / "toy.tar.gz"
        (self.root / "toy.tar.gz.part").write_bytes(payload[:100])  # interrupted
        pb.download(url, dest)
        self.assertEqual(dest.read_bytes(), payload)
        self.assertEqual(seen, ["bytes=100-"])
        self.assertFalse((self.root / "toy.tar.gz.part").exists())

    def test_server_without_range_support_restarts_cleanly(self) -> None:
        archive = self._make_archive({"toy/a.gb": b"A" * 5000})
        payload = archive.read_bytes()
        url, seen = self._serve(payload, support_range=False)
        dest = self.root / "toy.tar.gz"
        (self.root / "toy.tar.gz.part").write_bytes(payload[:100])
        pb.download(url, dest)
        self.assertEqual(dest.read_bytes(), payload)  # no duplicated prefix
        self.assertEqual(seen, ["bytes=100-"])

    def test_unpublished_datasets_cannot_be_fetched(self) -> None:
        manifest = pb.load_manifest(MANIFEST)
        unpublished = [d for d in manifest["datasets"] if d["status"] == "unpublished"]
        for entry in unpublished:
            with self.assertRaises(SystemExit):
                pb.fetch_dataset(entry, self.root / "data")

    def test_pack_is_reproducible(self) -> None:
        src = self.root / "set"
        src.mkdir()
        (src / "b.gb").write_text("B")
        (src / "a.gb").write_text("A")
        first = pb.pack_directory(src, self.root / "one.tar.gz")
        os.utime(src / "a.gb", (1, 1))  # different mtime must not matter
        second = pb.pack_directory(src, self.root / "two.tar.gz")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["n_files"], 2)
        with tarfile.open(self.root / "one.tar.gz") as tar:
            self.assertEqual(tar.getnames(), ["set", "set/a.gb", "set/b.gb"])


class CommandTests(TempDirTestCase):
    def test_pba_command_uses_the_stable_cli_flags(self) -> None:
        cmd = pb.pba_command(
            "python",
            Path("in"),
            Path("out"),
            mode="igs",
            threads=4,
            min_taxa=5,
            min_len=6,
            concat=True,
            extra=["--trim", "none"],
        )
        self.assertEqual(cmd[:3], ["python", "-m", "plastburstalign"])
        for flag, value in (("-i", "in"), ("-o", "out"), ("-s", "igs"), ("-n", "4")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        self.assertIn("-c", cmd)
        self.assertEqual(cmd[-2:], ["--trim", "none"])

    def test_unaligned_and_matrix_discovery(self) -> None:
        out = self.root / "out"
        (out / "fasta").mkdir(parents=True)
        (out / "nucl_b.unalign.fasta").write_text(">b_t1\nA\n")
        (out / "nucl_a.unalign.fasta").write_text(">a_t1\nA\n")
        (out / "fasta" / "nucl_a.aligned.fasta").write_text(">a_t1\nA\n")
        (out / "nucl_2concat.aligned.fasta").write_text(">concat_t1\nAA\n")
        self.assertEqual(
            [p.name for p in pb.find_unaligned_fastas(out)],
            ["nucl_a.unalign.fasta", "nucl_b.unalign.fasta"],
        )
        matrix = pb.find_concat_matrix(out)
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.name, "nucl_2concat.aligned.fasta")
        self.assertEqual(pb.count_aligned_loci(out), 1)

    def test_host_profile_and_list_command(self) -> None:
        host = pb.host_profile()
        for key in (
            "system",
            "machine",
            "python",
            "cpu_count_logical",
            "ram_total_bytes",
        ):
            self.assertIn(key, host)
        self.assertGreater(host["ram_total_bytes"], 0)
        self.assertNotIn("hostname", host)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(pb.main(["list"]), 0)
        self.assertIn("benchmark_YangEtAl2022_Asteraceae", buffer.getvalue())
        self.assertIn("15k", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
