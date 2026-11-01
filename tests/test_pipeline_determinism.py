"""Full-pipeline determinism: identical outputs across repeats, threads, hash seeds.

Skipped when MAFFT is not on ``PATH``. Uses a tiny synthetic multi-taxon set from
:func:`tests.synth.write_genbank_dataset` so wall time stays well under a minute.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from plastburstalign import PlastBurstAlign, UserParameters
from plastburstalign._simulate import simulate_locus
from plastburstalign.chunking import ChunkConfig, plan_chunks
from tests.synth import write_genbank_dataset

HAS_MAFFT = shutil.which("mafft") is not None
REPO_ROOT = Path(__file__).resolve().parents[1]

_PATH_PARAM_KEYS = frozenset(
    {"in_dir", "out_dir", "scratch_dir", "alignment_tool_path", "manifest_path"}
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _concat_hashes(out_dir: Path) -> dict[str, str]:
    files = sorted(out_dir.glob("nucl_*concat.aligned.fasta"))
    files += sorted(out_dir.glob("nucl_*concat.aligned.nexus"))
    files += sorted(out_dir.glob("nucl_*concat.partitions.*"))
    if not files:
        raise AssertionError(f"no concatenated outputs under {out_dir}")
    return {path.name: _sha256_file(path) for path in files}


def _stable_manifest(path: Path) -> dict[str, Any]:
    """Manifest fields that should match across identical logical runs.

    Drops timestamps (``run``), host/tuning/scratch/scheduler metrics that track
    free RAM and wall time, and absolute paths under ``params`` / ``supermatrix``.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    params = {
        key: value
        for key, value in (document.get("params") or {}).items()
        if key not in _PATH_PARAM_KEYS
    }
    versions = document.get("versions") or {}
    super_raw = document.get("supermatrix") or {}
    supermatrix = {key: value for key, value in super_raw.items() if key != "written"}
    # Writer sorts loci by (region, mode); keep list order for byte-stable compares.
    loci_list = list(document.get("loci") or [])
    return {
        "params": params,
        "loci": loci_list,
        "strategy_constants": document.get("strategy_constants"),
        "supermatrix": supermatrix,
        "versions": {
            "plastburstalign": versions.get("plastburstalign"),
            "aligner_version": versions.get("aligner_version"),
            "numpy": versions.get("numpy"),
            "biopython": versions.get("biopython"),
        },
    }


def _assert_loci_sorted(document: dict[str, Any]) -> None:
    loci = list(document.get("loci") or [])
    keys = [(str(e.get("region") or ""), str(e.get("mode") or "")) for e in loci]
    assert keys == sorted(keys), f"manifest loci not sorted by (region, mode): {keys}"


def _base_params(in_dir: Path, out_dir: Path, num_threads: int) -> UserParameters:
    return UserParameters(
        in_dir=str(in_dir),
        out_dir=str(out_dir),
        select_mode="rna",
        alignment_tool="mafft",
        num_threads=num_threads,
        concat=True,
        min_num_taxa=2,
        subdivide="off",
        seed=7,
        order="alpha",
        keep_unaligned=False,
        individual_nexus=False,
        partitions="raxml,nexus",
    )


def _run_pipeline(in_dir: Path, out_dir: Path, num_threads: int) -> Path:
    PlastBurstAlign(_base_params(in_dir, out_dir, num_threads)).execute()
    return out_dir


@unittest.skipUnless(HAS_MAFFT, "mafft not available")
class PipelineDeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._root = tempfile.TemporaryDirectory(prefix="pba_pipeline_det_")
        cls.in_dir = Path(cls._root.name) / "in"
        write_genbank_dataset(cls.in_dir, n_taxa=4, n_genes=8, seed=42)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._root.cleanup()

    def test_repeated_runs_identical_outputs_and_stable_manifest(self) -> None:
        out_a = Path(self._root.name) / "repeat_a"
        out_b = Path(self._root.name) / "repeat_b"
        _run_pipeline(self.in_dir, out_a, num_threads=1)
        _run_pipeline(self.in_dir, out_b, num_threads=1)
        self.assertEqual(_concat_hashes(out_a), _concat_hashes(out_b))
        for path in (out_a / "run_manifest.json", out_b / "run_manifest.json"):
            _assert_loci_sorted(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(
            _stable_manifest(out_a / "run_manifest.json"),
            _stable_manifest(out_b / "run_manifest.json"),
        )

    def test_thread_count_invariant_with_subdivide_off(self) -> None:
        out_1 = Path(self._root.name) / "threads_1"
        out_2 = Path(self._root.name) / "threads_2"
        _run_pipeline(self.in_dir, out_1, num_threads=1)
        _run_pipeline(self.in_dir, out_2, num_threads=2)
        self.assertEqual(_concat_hashes(out_1), _concat_hashes(out_2))
        # Logical alignment choices stay data-only even when the CPU budget changes;
        # sorted loci lists must match across thread counts.
        for path in (out_1 / "run_manifest.json", out_2 / "run_manifest.json"):
            _assert_loci_sorted(json.loads(path.read_text(encoding="utf-8")))
        loci_1 = _stable_manifest(out_1 / "run_manifest.json")["loci"]
        loci_2 = _stable_manifest(out_2 / "run_manifest.json")["loci"]
        self.assertEqual(loci_1, loci_2)

    def test_pythonhashseed_invariant_via_subprocess(self) -> None:
        script = r"""
import hashlib, json, sys
from pathlib import Path
from plastburstalign import PlastBurstAlign, UserParameters

in_dir, out_dir, threads = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
params = UserParameters(
    in_dir=str(in_dir),
    out_dir=str(out_dir),
    select_mode="rna",
    alignment_tool="mafft",
    num_threads=threads,
    concat=True,
    min_num_taxa=2,
    subdivide="off",
    seed=7,
    order="alpha",
    keep_unaligned=False,
    individual_nexus=False,
    partitions="raxml,nexus",
)
PlastBurstAlign(params).execute()
files = sorted(out_dir.glob("nucl_*concat.aligned.fasta"))
files += sorted(out_dir.glob("nucl_*concat.aligned.nexus"))
files += sorted(out_dir.glob("nucl_*concat.partitions.*"))
print(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
"""
        out_0 = Path(self._root.name) / "hashseed_0"
        out_1 = Path(self._root.name) / "hashseed_1"
        hashes: list[dict[str, str]] = []
        for seed, out_dir in (("0", out_0), ("1", out_1)):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            env["PATH"] = os.environ.get("PATH", "")
            completed = subprocess.run(
                [sys.executable, "-c", script, str(self.in_dir), str(out_dir), "1"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
                cwd=str(REPO_ROOT),
            )
            line = completed.stdout.strip().splitlines()[-1]
            hashes.append(json.loads(line))
        self.assertEqual(hashes[0], hashes[1])

    def test_chunk_plan_fingerprint_is_data_only(self) -> None:
        """Chunk boundaries are a pure function of the sequences (host-independent)."""
        records = simulate_locus(8, 6000, seed=99, divergence=0.01)
        cfg = ChunkConfig(min_locus_len=5000, target_chunk_len=2000)
        first = plan_chunks(records, "nucl", cfg, name="locus")
        second = plan_chunks(list(records), "nucl", cfg, name="locus")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertGreaterEqual(first.n_chunks, 2)


if __name__ == "__main__":
    unittest.main()
