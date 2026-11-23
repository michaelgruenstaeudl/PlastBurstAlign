"""
Distributed cluster execution and workload partitioning for PlastBurstAlign on HPC/Slurm.
"""

from __future__ import annotations

import datetime
import heapq
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from .logging_ops import logger as log


class ClusterError(Exception):
    """Base exception for cluster execution errors."""


class ClusterManifestError(ClusterError):
    """Raised when cluster manifest serialization or deserialization fails."""


class ClusterExecutionError(ClusterError):
    """Raised when cluster worker tasks fail or are incomplete during finalization."""


@dataclass
class PartitionInfo:
    """One worker's share of the loci.

    `estimated_bytes` is the cumulative partition weight in the unit given by
    `weight_kind`: input FASTA bytes (`"bytes"`, the file-size fallback) or the
    alignment work estimate N * L^alpha (`"work"`).
    """

    partition_id: int
    regions: list[str]
    estimated_bytes: int
    weight_kind: str = "bytes"


@dataclass
class ClusterManifest:
    version: str
    created_at: str
    select_mode: str
    out_dir: str
    total_regions: int
    num_partitions: int
    alignment_tool: str
    alignment_tool_path: str | None
    alignment_command: str | None
    adaptive_heuristics: bool
    partitions: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterManifest:
        try:
            return cls(
                version=data["version"],
                created_at=data["created_at"],
                select_mode=data["select_mode"],
                out_dir=data["out_dir"],
                total_regions=int(data["total_regions"]),
                num_partitions=int(data["num_partitions"]),
                alignment_tool=data.get("alignment_tool", "mafft"),
                alignment_tool_path=data.get("alignment_tool_path"),
                alignment_command=data.get("alignment_command"),
                adaptive_heuristics=data.get("adaptive_heuristics", True),
                partitions=data.get("partitions", {}),
            )
        except (KeyError, TypeError, ValueError) as err:
            raise ClusterManifestError(
                f"Invalid cluster manifest structure: {err}"
            ) from err


class ClusterPartitionEngine:
    """Partitions alignment tasks across cluster workers using Greedy LPT bin packing."""

    @staticmethod
    def estimate_alignment_work(
        count: int, max_len: int, *, alphabet: str = "nucl"
    ) -> int:
        """Alignment work estimate via :func:`strategy.estimate_work`.

        Returns 0 for empty loci so the caller can fall back to file sizes.
        """
        if count <= 0 or max_len <= 0:
            return 0
        from .strategy import estimate_work

        return int(round(estimate_work(count, max_len, alphabet=alphabet)))

    @classmethod
    def _weights_and_kind(
        cls,
        regions: list[str],
        out_dir: str,
        prefix: str,
        region_summary: dict[str, dict[str, int]] | None,
    ) -> tuple[dict[str, int], str]:
        alphabet = "prot" if prefix == "prot" else "nucl"
        if region_summary:
            work: dict[str, int] = {}
            for r in regions:
                entry = region_summary.get(r)
                if not isinstance(entry, dict):
                    break
                weight = cls.estimate_alignment_work(
                    int(entry.get("count", 0) or 0),
                    int(entry.get("max_len", 0) or 0),
                    alphabet=alphabet,
                )
                if weight <= 0:
                    break
                work[r] = weight
            else:
                return work, "work"

        weights: dict[str, int] = {}
        for r in regions:
            fn = os.path.join(out_dir, f"{prefix}_{r}.unalign.fasta")
            try:
                weights[r] = os.path.getsize(fn)
            except OSError:
                weights[r] = 0
        return weights, "bytes"

    @classmethod
    def compute_region_weights(
        cls,
        regions: list[str],
        out_dir: str,
        prefix: str = "nucl",
        region_summary: dict[str, dict[str, int]] | None = None,
    ) -> dict[str, int]:
        """Calculates a scheduling weight for every region.

        If `region_summary` (from `DiskBackedFeatureStore.get_region_summary`) has a
        usable entry (`count` > 0 and `max_len` > 0) for *every* region, the weights
        are alignment work estimates N * L^alpha and no unaligned FASTA files need
        to exist. Otherwise (no summary, or an incomplete one) the weights are the
        sizes of the existing `<prefix>_<region>.unalign.fasta` files, so that all
        weights of a run are always in the same unit.
        """
        return cls._weights_and_kind(regions, out_dir, prefix, region_summary)[0]

    @classmethod
    def partition_regions(
        cls,
        regions: list[str],
        out_dir: str,
        num_partitions: int,
        prefix: str = "nucl",
        region_summary: dict[str, dict[str, int]] | None = None,
    ) -> list[PartitionInfo]:
        """Distributes regions across partitions using Greedy LPT multi-bin allocation.

        Tasks are sorted descending by weight (with alphabetical tie-breaking).
        Each task is placed into the partition with the minimum cumulative weight.
        Weights come from `region_summary` (work estimate N * L^alpha) when it covers
        all regions, else from unaligned file sizes (see `compute_region_weights`).
        """
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive (got {num_partitions})")

        if not regions:
            return [
                PartitionInfo(partition_id=i, regions=[], estimated_bytes=0)
                for i in range(num_partitions)
            ]

        # Clamp partitions if regions < requested partitions
        actual_num_partitions = min(num_partitions, len(regions))

        weights, weight_kind = cls._weights_and_kind(
            regions, out_dir, prefix, region_summary
        )
        # Sort descending by weight, ascending by name for deterministic scheduling
        sorted_regions = sorted(regions, key=lambda k: (-weights.get(k, 0), k))

        # Min-heap storing tuples: (cumulative_weight, partition_index)
        heap: list[tuple[int, int]] = [(0, i) for i in range(actual_num_partitions)]
        heapq.heapify(heap)

        partition_assignments: dict[int, list[str]] = {
            i: [] for i in range(actual_num_partitions)
        }
        partition_weights: dict[int, int] = {i: 0 for i in range(actual_num_partitions)}

        for r in sorted_regions:
            w = weights.get(r, 0)
            cum_w, p_idx = heapq.heappop(heap)
            partition_assignments[p_idx].append(r)
            new_cum_w = cum_w + w
            partition_weights[p_idx] = new_cum_w
            heapq.heappush(heap, (new_cum_w, p_idx))

        return [
            PartitionInfo(
                partition_id=i,
                regions=partition_assignments[i],
                estimated_bytes=partition_weights[i],
                weight_kind=weight_kind,
            )
            for i in range(actual_num_partitions)
        ]

    @staticmethod
    def save_manifest(manifest: ClusterManifest, filepath: str) -> None:
        """Serializes cluster manifest atomically to disk."""
        tmp_path = f"{filepath}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest.to_dict(), f, indent=2)
            os.replace(tmp_path, filepath)
            log.info(f"Slurm manifest successfully written to: {filepath}")
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise ClusterManifestError(
                f"Failed to write manifest to {filepath}: {e}"
            ) from e

    @staticmethod
    def load_manifest(filepath: str) -> ClusterManifest:
        """Loads and parses a cluster manifest from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Slurm manifest not found at: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ClusterManifest.from_dict(data)
        except Exception as e:
            raise ClusterManifestError(
                f"Failed to load manifest from {filepath}: {e}"
            ) from e


class SlurmScriptGenerator:
    """Generates portable, robust Slurm batch scripts for array jobs and finalization."""

    @staticmethod
    def generate_scripts(
        out_dir: str,
        manifest_path: str,
        num_partitions: int,
        cpus_per_task: int = 8,
        mem_per_task: str = "16G",
        time_limit: str = "04:00:00",
        slurm_partition: str | None = None,
        finalize_cpus: int = 4,
        finalize_mem: str = "32G",
        finalize_time: str = "02:00:00",
    ) -> dict[str, str]:
        """Generates submit_array.slurm, submit_finalize.slurm, and run_pipeline.sh."""
        os.makedirs(out_dir, exist_ok=True)
        rel_manifest = os.path.relpath(manifest_path, out_dir)

        part_directive = f"#SBATCH -p {slurm_partition}\n" if slurm_partition else ""
        max_array_idx = max(0, num_partitions - 1)

        # 1. Worker Array Script
        array_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=pba_align
#SBATCH --array=0-{max_array_idx}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem_per_task}
#SBATCH --time={time_limit}
{part_directive}#SBATCH --output=slurm_logs/array_%A_%a.out
#SBATCH --error=slurm_logs/array_%A_%a.err

set -euo pipefail

# Ensure working directory is the run root
cd "{os.path.abspath(out_dir)}"
mkdir -p slurm_logs partitions

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting PlastBurstAlign partition ${{SLURM_ARRAY_TASK_ID}} on $(hostname)"

python3 -m plastburstalign \\
  -o . \\
  --align-partition "${{SLURM_ARRAY_TASK_ID}}" \\
  --manifest "{rel_manifest}" \\
  --numthreads "${{SLURM_CPUS_PER_TASK:-{cpus_per_task}}}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Partition ${{SLURM_ARRAY_TASK_ID}} finished successfully"
"""

        # 2. Finalize Script
        finalize_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=pba_finalize
#SBATCH --cpus-per-task={finalize_cpus}
#SBATCH --mem={finalize_mem}
#SBATCH --time={finalize_time}
{part_directive}#SBATCH --output=slurm_logs/finalize_%j.out
#SBATCH --error=slurm_logs/finalize_%j.err

set -euo pipefail

cd "{os.path.abspath(out_dir)}"
mkdir -p slurm_logs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting PlastBurstAlign finalization on $(hostname)"

python3 -m plastburstalign \\
  -o . \\
  --finalize-slurm \\
  --manifest "{rel_manifest}" \\
  --numthreads "${{SLURM_CPUS_PER_TASK:-{finalize_cpus}}}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Concatenation and pipeline finalization complete"
"""

        # 3. Master Pipeline Dispatcher
        pipeline_content = """#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p slurm_logs partitions

echo "Submitting worker array job..."
ARRAY_JOB_ID=$(sbatch --parsable submit_array.slurm)
echo "Submitted array job: ${ARRAY_JOB_ID}"

echo "Submitting finalization job with dependency afterok:${ARRAY_JOB_ID}..."
FINALIZE_JOB_ID=$(sbatch --parsable --dependency=afterok:"${ARRAY_JOB_ID}" submit_finalize.slurm)
echo "Submitted finalizer job: ${FINALIZE_JOB_ID}"

echo "Pipeline submission complete. Monitor with: squeue -j ${ARRAY_JOB_ID},${FINALIZE_JOB_ID}"
"""

        scripts = {
            os.path.join(out_dir, "submit_array.slurm"): array_content,
            os.path.join(out_dir, "submit_finalize.slurm"): finalize_content,
            os.path.join(out_dir, "run_pipeline.sh"): pipeline_content,
        }

        for path, content in scripts.items():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            if path.endswith(".sh"):
                os.chmod(path, 0o755)
            log.info(f"Generated Slurm execution file: {path}")

        return scripts


class PartitionReceipt:
    """Tracks and verifies completion of individual partition workers."""

    @staticmethod
    def get_receipt_path(out_dir: str, partition_id: int) -> str:
        return os.path.join(out_dir, "partitions", f"partition_{partition_id}.done")

    @classmethod
    def write_receipt(
        cls,
        out_dir: str,
        partition_id: int,
        regions: list[str],
        successful_regions: list[str],
    ) -> str:
        """Writes an atomic execution receipt for a completed partition."""
        receipt_path = cls.get_receipt_path(out_dir, partition_id)
        os.makedirs(os.path.dirname(receipt_path), exist_ok=True)
        tmp_path = f"{receipt_path}.tmp.{os.getpid()}"
        payload = {
            "partition_id": partition_id,
            "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "success",
            "total_regions": len(regions),
            "successful_regions": successful_regions,
            "failed_regions": [r for r in regions if r not in successful_regions],
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, receipt_path)
        log.info(
            f"Recorded completion receipt for partition {partition_id}: {receipt_path}"
        )
        return receipt_path

    @classmethod
    def verify_all_completed(
        cls,
        out_dir: str,
        num_partitions: int,
    ) -> tuple[bool, list[int], list[str]]:
        """Verifies that all partition receipts exist and collects all successful regions.

        Returns:
            (all_success, missing_partition_ids, all_successful_regions)
        """
        missing: list[int] = []
        all_successful_regions: list[str] = []

        for p_id in range(num_partitions):
            path = cls.get_receipt_path(out_dir, p_id)
            if not os.path.exists(path):
                missing.append(p_id)
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_successful_regions.extend(data.get("successful_regions", []))
            except Exception as e:
                log.warning(f"Corrupt receipt for partition {p_id} at {path}: {e}")
                missing.append(p_id)

        return (len(missing) == 0, missing, all_successful_regions)
