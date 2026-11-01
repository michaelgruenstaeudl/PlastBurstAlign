"""Slurm path tests without mocked aligners or hand-made unaligned files.

* Partition weights come from the store's region summary (N * L^alpha), with the
  file-size fallback preserved.
* `export -> worker -> finalize` runs end to end with the real MAFFT (skipped if
  it is missing), both in-process from a populated store and through the real
  command line exactly as the generated Slurm scripts invoke it (workers and the
  finalizer get neither `-i` nor `-s`), and the resulting supermatrix equals the
  one of the monolithic run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from plastburstalign.cluster_ops import ClusterManifest, ClusterPartitionEngine
from plastburstalign.plast_burst_align import PlastBurstAlign
from plastburstalign.user_parameters import SelectMode, UserParameters

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_gate():
    path = REPO_ROOT / "benchmarking" / "golden_parity.py"
    spec = importlib.util.spec_from_file_location("golden_parity_for_slurm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # do not create benchmarking/__pycache__
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


gate = _load_gate()
HAS_MAFFT = shutil.which("mafft") is not None


class PartitionWeightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out_dir = self.tmp.name

    def test_alpha_tiers(self) -> None:
        from plastburstalign.strategy import estimate_work

        work = ClusterPartitionEngine.estimate_alignment_work
        for n, length in ((10, 999), (10, 1000), (10, 2499), (10, 2500)):
            self.assertEqual(work(n, length), round(estimate_work(n, length)))
        self.assertEqual(work(0, 1000), 0)
        self.assertEqual(work(10, 0), 0)

    def test_summary_weights_need_no_unaligned_files(self) -> None:
        summary = {
            "huge": {"count": 500, "max_len": 3000, "min_len": 10},
            "mid1": {"count": 200, "max_len": 1500, "min_len": 10},
            "mid2": {"count": 200, "max_len": 1500, "min_len": 10},
            "tiny": {"count": 20, "max_len": 300, "min_len": 10},
        }
        regions = list(summary)
        self.assertEqual(os.listdir(self.out_dir), [])
        parts = ClusterPartitionEngine.partition_regions(
            regions, self.out_dir, 2, region_summary=summary
        )
        self.assertEqual({p.weight_kind for p in parts}, {"work"})
        by_region = {r: p.partition_id for p in parts for r in p.regions}
        # N * L^2.4 of the huge locus dominates: it must not share a partition
        # with the other heavy loci, and LPT puts everything else together.
        self.assertNotEqual(by_region["huge"], by_region["mid1"])
        self.assertEqual(by_region["mid1"], by_region["mid2"])
        work = ClusterPartitionEngine.estimate_alignment_work
        total = sum(work(v["count"], v["max_len"]) for v in summary.values())
        self.assertEqual(sum(p.estimated_bytes for p in parts), total)

    def test_incomplete_or_empty_summary_falls_back_to_file_sizes(self) -> None:
        for region, size in (("a", 300), ("b", 100), ("c", 100)):
            (Path(self.out_dir) / f"nucl_{region}.unalign.fasta").write_bytes(
                b"N" * size
            )
        incomplete = {"a": {"count": 5, "max_len": 50, "min_len": 5}}
        for summary in ({}, incomplete, None):
            with self.subTest(summary=summary):
                parts = ClusterPartitionEngine.partition_regions(
                    ["a", "b", "c"], self.out_dir, 2, region_summary=summary
                )
                self.assertEqual({p.weight_kind for p in parts}, {"bytes"})
                self.assertEqual(sum(p.estimated_bytes for p in parts), 500)
                weights = ClusterPartitionEngine.compute_region_weights(
                    ["a", "b", "c"], self.out_dir, region_summary=summary
                )
                self.assertEqual(weights, {"a": 300, "b": 100, "c": 100})

    def test_export_partitions_from_store_without_unaligned_files(self) -> None:
        in_dir = Path(self.out_dir) / "in"
        in_dir.mkdir()
        out_dir = Path(self.out_dir) / "run"
        params = UserParameters(
            in_dir=str(in_dir),
            out_dir=str(out_dir),
            select_mode="igs",
            export_slurm=True,
            slurm_partitions=2,
        )
        pba = PlastBurstAlign(params)
        self.addCleanup(pba.plastid_data.store.close)
        store = pba.plastid_data.store
        rng = random.Random(1)
        shapes = {
            "big": (30, 900),
            "small1": (4, 60),
            "small2": (4, 60),
            "small3": (4, 60),
        }
        for region, (n_taxa, length) in shapes.items():
            for i in range(n_taxa):
                seq = "".join(rng.choice("ACGT") for _ in range(length - i))
                store.add_record(region, f"t{i}", "nucl", f"{region}_t{i}", "", seq)
        with (
            patch("plastburstalign.plast_burst_align.ExtractAndCollect"),
            patch("plastburstalign.plast_burst_align.DataCleaning"),
        ):
            pba.execute()

        self.assertEqual(list(out_dir.glob("*.unalign.fasta")), [])
        manifest = json.loads((out_dir / "slurm_manifest.json").read_text())
        parts = manifest["partitions"]
        self.assertEqual({p["weight_kind"] for p in parts.values()}, {"work"})
        regions_by_partition = {k: set(v["regions"]) for k, v in parts.items()}
        self.assertIn({"big"}, regions_by_partition.values())
        self.assertIn({"small1", "small2", "small3"}, regions_by_partition.values())


class ManifestAdoptionTests(unittest.TestCase):
    """Workers and the finalizer run without -s/aligner options: the manifest rules."""

    def _manifest(self, **overrides: Any) -> ClusterManifest:
        values: dict[str, Any] = {
            "version": "1.0",
            "created_at": "2026-10-30T00:00:00Z",
            "select_mode": "SelectMode.IGS",
            "out_dir": "/does/not/matter",
            "total_regions": 1,
            "num_partitions": 1,
            "alignment_tool": "mafft",
            "alignment_tool_path": None,
            "alignment_command": None,
            "adaptive_heuristics": True,
            "partitions": {"0": {"regions": ["a"], "estimated_bytes": 1}},
        }
        values.update(overrides)
        return ClusterManifest(**values)

    def _worker(self, tmp: str, **kwargs: Any) -> PlastBurstAlign:
        params = UserParameters(out_dir=tmp, align_partition=0, **kwargs)
        return PlastBurstAlign(params)

    def test_mode_comes_from_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pba = self._worker(tmp)
            self.assertIs(pba.user_params["select_mode"], SelectMode.CDS)  # CLI default
            for text, expected in (
                ("SelectMode.IGS", SelectMode.IGS),
                ("rna", SelectMode.RNA),
                ("SelectMode.INT", SelectMode.INT),
            ):
                pba._apply_manifest_to_params(self._manifest(select_mode=text))
                self.assertIs(pba.user_params["select_mode"], expected)

    def test_aligner_settings_fall_back_to_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pba = self._worker(tmp)
            pba._apply_manifest_to_params(
                self._manifest(
                    alignment_tool="muscle",
                    alignment_command="cat %INPUT% > %OUTPUT%",
                    alignment_tool_path=sys.executable,
                    adaptive_heuristics=False,
                )
            )
            self.assertEqual(pba.user_params["alignment_tool"], "muscle")
            self.assertEqual(
                pba.user_params["alignment_command"], "cat %INPUT% > %OUTPUT%"
            )
            self.assertEqual(pba.user_params["alignment_tool_path"], sys.executable)
            self.assertFalse(pba.user_params["adaptive_heuristics"])

    def test_explicit_worker_options_win_and_missing_paths_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pba = self._worker(tmp, alignment_tool="clustalo")
            pba._apply_manifest_to_params(
                self._manifest(
                    alignment_tool="muscle", alignment_tool_path="/no/such/binary"
                )
            )
            self.assertEqual(pba.user_params["alignment_tool"], "clustalo")
            self.assertIsNone(pba.user_params["alignment_tool_path"])


def _synthetic_inputs(in_dir: Path, num_taxa: int = 5, seed: int = 7) -> None:
    """Write small GenBank files holding three CDS genes and three tRNAs each."""
    rng = random.Random(seed)
    sense = [
        a + b + c
        for a in "ACGT"
        for b in "ACGT"
        for c in "ACGT"
        if a + b + c not in ("TAA", "TAG", "TGA")
    ]
    cds_templates = {
        name: [rng.choice(sense) for _ in range(n)]
        for name, n in (("psbA", 55), ("rbcL", 45), ("atpB", 50))
    }
    trna_templates = {
        "trnH-GUG": "".join(rng.choice("ACGT") for _ in range(46)),
        "trnK-UUU": "".join(rng.choice("ACGT") for _ in range(61)),
        "trnL-UAA": "".join(rng.choice("ACGT") for _ in range(53)),
    }

    def spacer() -> str:
        return "".join(rng.choice("ACGT") for _ in range(rng.randint(8, 25)))

    in_dir.mkdir(parents=True, exist_ok=True)
    for t in range(num_taxa):
        pieces: list[str] = []
        features: list[SeqFeature] = []
        pos = 0

        def add(seq: str, kind: str, gene: str) -> None:
            nonlocal pos
            pieces.append(seq)
            features.append(
                SeqFeature(
                    FeatureLocation(pos, pos + len(seq), strand=1),
                    type=kind,
                    qualifiers={"gene": [gene]},
                )
            )
            pos += len(seq)

        for name, codons in cds_templates.items():
            body = list(codons)
            for _ in range(3):  # per-taxon substitutions
                body[rng.randrange(len(body))] = rng.choice(sense)
            if t % 3 == 1:  # in-frame deletion
                del body[rng.randrange(5, len(body) - 5)]
            if t % 3 == 2:  # in-frame insertion
                body.insert(rng.randrange(5, len(body) - 5), rng.choice(sense))
            pieces.append(spacer())
            pos += len(pieces[-1])
            add("ATG" + "".join(body) + "TAA", "gene", name)
            features.append(
                SeqFeature(
                    features[-1].location, type="CDS", qualifiers={"gene": [name]}
                )
            )
        for name, base in trna_templates.items():
            seq = list(base)
            for _ in range(2):
                seq[rng.randrange(len(seq))] = rng.choice("ACGT")
            if t % 2:
                del seq[rng.randrange(5, len(seq) - 5)]
            pieces.append(spacer())
            pos += len(pieces[-1])
            add("".join(seq), "tRNA", name)
        pieces.append(spacer())
        record = SeqRecord(
            Seq("".join(pieces)),
            id=f"TAXA{t}",
            name=f"TAXA{t}",
            description=f"synthetic genome {t}",
            annotations={"molecule_type": "DNA"},
        )
        record.features = features
        SeqIO.write(record, in_dir / f"TAXA{t}.gb", "genbank")


@unittest.skipUnless(HAS_MAFFT, "mafft not available")
class SlurmRealStoreEndToEndTests(unittest.TestCase):
    """export -> worker -> finalize in-process from a populated store (real MAFFT)."""

    def test_export_worker_finalize_without_unaligned_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            in_dir = root / "in"
            in_dir.mkdir()
            out_dir = root / "run"
            base: dict[str, Any] = {
                "in_dir": str(in_dir),
                "out_dir": str(out_dir),
                "select_mode": "igs",
            }
            rng = random.Random(3)
            loci: dict[str, dict[str, str]] = {}
            for region in ("alpha", "beta", "gamma", "delta"):
                template = "".join(rng.choice("ACGT") for _ in range(120))
                loci[region] = {}
                for t in range(4):
                    seq = list(template)
                    seq[rng.randrange(120)] = rng.choice("ACGT")
                    if t == 2:
                        del seq[40:46]
                    loci[region][f"taxon{t}"] = "".join(seq)

            exporter = PlastBurstAlign(
                UserParameters(**base, export_slurm=True, slurm_partitions=2)
            )
            store = exporter.plastid_data.store
            for region, taxa in loci.items():
                for i, (taxon, seq) in enumerate(taxa.items()):
                    store.add_record(
                        region,
                        taxon,
                        "nucl",
                        f"{region}_{taxon}",
                        "",
                        seq,
                        file_index=i,
                    )
            with (
                patch("plastburstalign.plast_burst_align.ExtractAndCollect"),
                patch("plastburstalign.plast_burst_align.DataCleaning"),
            ):
                exporter.execute()
            store.close()
            self.assertEqual(list(out_dir.glob("*.unalign.fasta")), [])

            manifest_path = str(out_dir / "slurm_manifest.json")
            manifest = json.loads(Path(manifest_path).read_text())
            self.assertEqual(manifest["num_partitions"], 2)
            for p_id in range(2):
                worker = PlastBurstAlign(
                    UserParameters(
                        **base,
                        align_partition=p_id,
                        manifest_path=manifest_path,
                        num_threads=2,
                    )
                )
                worker.execute()
                worker.plastid_data.store.close()
            # Workers wrote their results but no shared unaligned inputs.
            self.assertEqual(list(out_dir.glob("*.unalign.fasta")), [])
            for region in loci:
                aligned = out_dir / "fasta" / f"nucl_{region}.aligned.fasta"
                rows = gate.read_fasta(aligned)
                self.assertEqual(len(rows), 4)
                self.assertEqual(len({len(s) for _, s in rows}), 1)  # real alignment
                got = {t.split("_", 1)[1]: s.replace("-", "").upper() for t, s in rows}
                self.assertEqual(got, loci[region])

            finalizer = PlastBurstAlign(
                UserParameters(**base, finalize_slurm=True, manifest_path=manifest_path)
            )
            finalizer.execute()
            finalizer.plastid_data.store.close()
            concat = out_dir / "nucl_4concat.aligned.fasta"
            self.assertTrue(concat.exists())
            matrix = dict(gate.read_fasta(concat))
            self.assertEqual(set(matrix), {f"concat_taxon{t}" for t in range(4)})
            self.assertEqual(len({len(s) for s in matrix.values()}), 1)


@unittest.skipUnless(HAS_MAFFT, "mafft not available")
class SlurmCommandLineEndToEndTests(unittest.TestCase):
    """The Slurm path through the real CLI equals the monolithic run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.in_dir = cls.root / "in"
        _synthetic_inputs(cls.in_dir)
        cls.runs: dict[str, tuple[Path, Path]] = {}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    @staticmethod
    def _cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        proc = subprocess.run(
            [sys.executable, "-m", "plastburstalign", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"CLI failed ({proc.returncode}): {args}\n{proc.stderr[-2000:]}"
            )
        return proc

    def _run_both(self, mode: str) -> tuple[Path, Path]:
        common = ["-s", mode, "-l", "10", "-t", "3", "-n", "2"]
        mono = self.root / f"{mode}_mono"
        self._cli(["-i", str(self.in_dir), "-o", str(mono), "-c", *common], self.root)

        slurm = self.root / f"{mode}_slurm"
        self._cli(
            [
                "-i", str(self.in_dir), "-o", str(slurm), *common,
                "--export-slurm", "--slurm-partitions", "2",
            ],
            self.root,
        )  # fmt: skip
        self.assertEqual(
            list(slurm.glob("*.unalign.fasta")),
            [],
            "export must not depend on (or write) unaligned FASTA files",
        )
        manifest = json.loads((slurm / "slurm_manifest.json").read_text())
        self.assertEqual(manifest["num_partitions"], 2)
        # Exactly as in the generated submit_array.slurm / submit_finalize.slurm:
        # run from the output dir, no -i and no -s.
        for p_id in range(manifest["num_partitions"]):
            self._cli(
                [
                    "-o", ".", "--align-partition", str(p_id),
                    "--manifest", "slurm_manifest.json", "--numthreads", "1",
                ],
                slurm,
            )  # fmt: skip
        self._cli(
            ["-o", ".", "--finalize-slurm", "--manifest", "slurm_manifest.json",
             "--numthreads", "1"],
            slurm,
        )  # fmt: skip
        self.assertEqual(list(slurm.glob("*.unalign.fasta")), [])
        return mono, slurm

    def _assert_same_supermatrix(self, mono: Path, slurm: Path) -> int:
        mono_fa = sorted(mono.glob("nucl_*concat.aligned.fasta"))
        slurm_fa = sorted(slurm.glob("nucl_*concat.aligned.fasta"))
        self.assertEqual([p.name for p in mono_fa], [p.name for p in slurm_fa])
        self.assertEqual(len(mono_fa), 1)
        parts_name = mono_fa[0].name.replace(".aligned.fasta", ".partitions.txt")
        c_mono = gate.canonical_concat(mono_fa[0], mono / parts_name)
        c_slurm = gate.canonical_concat(slurm_fa[0], slurm / parts_name)
        self.assertEqual(c_mono["ntax"], 5)
        self.assertEqual(c_mono["partitions"], c_slurm["partitions"])
        self.assertEqual(c_mono["sha256"], c_slurm["sha256"])
        self.assertEqual(c_mono["partition_order"], c_slurm["partition_order"])
        self.assertEqual(c_mono["ordered_sha256"], c_slurm["ordered_sha256"])
        mono_loci = sorted(p.name for p in (mono / "fasta").glob("*.aligned.fasta"))
        slurm_loci = sorted(p.name for p in (slurm / "fasta").glob("*.aligned.fasta"))
        self.assertEqual(mono_loci, slurm_loci)
        for name in mono_loci:
            self.assertEqual(
                gate.canonical_alignment(gate.read_fasta(mono / "fasta" / name)),
                gate.canonical_alignment(gate.read_fasta(slurm / "fasta" / name)),
                name,
            )
        return len(mono_loci)

    def test_rna_slurm_equals_monolithic(self) -> None:
        # The worker CLI carries no `-s`; the manifest must restore mode `rna`
        # (the CLI default `cds` would look for protein loci that do not exist).
        mono, slurm = self._run_both("rna")
        self.assertEqual(self._assert_same_supermatrix(mono, slurm), 3)

    def test_cds_slurm_equals_monolithic(self) -> None:
        # Protein alignment + back-translation inside the workers.
        mono, slurm = self._run_both("cds")
        self.assertEqual(self._assert_same_supermatrix(mono, slurm), 3)
        for fasta in (slurm / "fasta").glob("*.aligned.fasta"):
            rows = gate.read_fasta(fasta)
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(len(s) % 3 == 0 for _, s in rows))


if __name__ == "__main__":
    unittest.main()
