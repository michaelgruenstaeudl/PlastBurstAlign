"""
Main execution coordinator for PlastBurstAlign with cluster execution support.
"""

from __future__ import annotations

import datetime
import os
import time
from typing import Any

from .align_runtime import clear_stale_outputs
from .alignment_ops import AlignmentCoordination
from .cluster_ops import (
    ClusterExecutionError,
    ClusterManifest,
    ClusterPartitionEngine,
    PartitionReceipt,
    SlurmScriptGenerator,
)
from .extraction_ops import DataCleaning, ExtractAndCollect
from .logging_ops import logger as log
from .manifest import write_run_manifest
from .seqfeature_ops import PlastidData
from .user_parameters import SelectMode, UserParameters


class PlastBurstAlign:
    def __init__(self, user_params: dict[str, Any] | None = None):
        """Coordinates extraction of sequence annotations from GenBank flatfiles,
        the alignment of those sequences grouped by region name,
        and writing the concatenated MSA results to file in NEXUS and FASTA formats.

        Args:
            user_params: Specifications for how the execution should be performed.
        """
        if not user_params or not isinstance(user_params, dict):
            self.user_params: dict[str, Any] = UserParameters()
        else:
            self.user_params = user_params

        # Last AlignmentCoordination from a monolithic/finalize run (for
        # post-flight dashboard / ScalingReport). Cleared on each execute().
        self.last_aligncoord: AlignmentCoordination | None = None

        # In cluster worker or finalize mode, bypass raw GenBank file inspection
        is_worker_or_final = (
            self.user_params.get("align_partition") is not None
            or self.user_params.get("finalize_slurm") is True
        )
        self.plastid_data: PlastidData | None
        if not is_worker_or_final:
            # Fresh monolithic/export runs must not reuse a previous feature store
            # or leftover unaligned FASTA files.
            clear_stale_outputs(str(self.user_params.get("out_dir", ".")))
            self.plastid_data = PlastidData(self.user_params)
        else:
            self.plastid_data = None

    def _require_plastid_data(self) -> PlastidData:
        if self.plastid_data is None:
            raise ClusterExecutionError(
                "PlastidData is not initialized (cluster worker/finalizer runs "
                "create it from the manifest)"
            )
        return self.plastid_data

    def execute(self):
        """Dispatches execution based on user parameters."""
        self.last_aligncoord = None
        if self.user_params.get("align_partition") is not None:
            self._execute_worker()
        elif self.user_params.get("finalize_slurm"):
            self._execute_finalize()
        elif self.user_params.get("export_slurm"):
            self._execute_export()
        else:
            self._execute_monolithic()

    def _write_manifest(
        self,
        *,
        started_at: datetime.datetime,
        wall_seconds: float,
        aligncoord: AlignmentCoordination | None,
        ok: bool,
    ) -> None:
        aligner = getattr(aligncoord, "aligner", None) if aligncoord else None
        write_run_manifest(
            str(self.user_params.get("out_dir", ".")),
            started_at=started_at,
            wall_seconds=wall_seconds,
            params=self.user_params,
            plan=getattr(aligncoord, "_plan", None) if aligncoord else None,
            locus_records=list(getattr(aligncoord, "_locus_records", []) or [])
            if aligncoord
            else None,
            scheduler_report=getattr(aligncoord, "_scheduler_report", None)
            if aligncoord
            else None,
            scratch_stats=aligncoord.scratch_stats() if aligncoord else None,
            supermatrix=getattr(aligncoord, "_supermatrix_info", None)
            if aligncoord
            else None,
            aligner_version=getattr(aligner, "resolved_version", None),
            aligner_path=getattr(aligner, "resolved_path", None)
            or getattr(aligner, "exec_path", None),
            ok=ok,
        )

    def _execute_monolithic(self):
        """Standard single-node monolithic execution."""
        started = datetime.datetime.now(datetime.timezone.utc)
        t0 = time.monotonic()
        aligncoord: AlignmentCoordination | None = None
        ok = False
        try:
            plastid_data = self._require_plastid_data()
            extractor = ExtractAndCollect(plastid_data, self.user_params)
            extractor.extract()

            cleaner = DataCleaning(plastid_data, self.user_params)
            cleaner.clean()

            aligncoord = AlignmentCoordination(plastid_data, self.user_params)
            aligncoord.perform_MSAs()
            if self.user_params.get("concat"):
                aligncoord.concat_MSAs()
            ok = True
        finally:
            if aligncoord is not None:
                aligncoord.close_scratch()
                self.last_aligncoord = aligncoord
            self._write_manifest(
                started_at=started,
                wall_seconds=time.monotonic() - t0,
                aligncoord=aligncoord,
                ok=ok,
            )

    def _execute_export(self):
        """Extracts and cleans features, exports partition manifest and Slurm job scripts."""
        plastid_data = self._require_plastid_data()
        extractor = ExtractAndCollect(plastid_data, self.user_params)
        extractor.extract()

        cleaner = DataCleaning(plastid_data, self.user_params)
        cleaner.clean()

        out_dir: str = self.user_params["out_dir"]
        num_partitions = self.user_params.get("slurm_partitions", 10)
        mode: SelectMode = self.user_params["select_mode"]
        prefix = "prot" if mode.collects_proteins() else "nucl"
        regions = list(plastid_data.nucleotides.keys())

        log.info(
            f"Exporting Slurm cluster manifest for {len(regions)} regions into "
            f"{num_partitions} partitions..."
        )

        # Partition weights come from the store's region summary (work estimate
        # N * L^alpha of the alignment that will actually run: protein loci for CDS
        # mode), so export does not depend on any pre-existing unaligned FASTA.
        # Workers export their own unaligned FASTA from the store on demand.
        try:
            region_summary = plastid_data.store.get_region_summary(seq_type=prefix)
        except Exception as err:
            log.warning(f"Unable to read region summary for partitioning: {err}")
            region_summary = {}
        if not isinstance(region_summary, dict):
            region_summary = {}

        partitions = ClusterPartitionEngine.partition_regions(
            regions=regions,
            out_dir=out_dir,
            num_partitions=num_partitions,
            prefix=prefix,
            region_summary=region_summary,
        )

        manifest_partitions: dict[str, dict[str, Any]] = {}
        for p in partitions:
            manifest_partitions[str(p.partition_id)] = {
                "regions": p.regions,
                "estimated_bytes": p.estimated_bytes,
                "weight_kind": p.weight_kind,
            }

        manifest = ClusterManifest(
            version="1.0",
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            select_mode=str(mode),
            out_dir=os.path.abspath(out_dir),
            total_regions=len(regions),
            num_partitions=len(partitions),
            alignment_tool=self.user_params.get("alignment_tool", "mafft"),
            alignment_tool_path=self.user_params.get("alignment_tool_path"),
            alignment_command=self.user_params.get("alignment_command"),
            adaptive_heuristics=self.user_params.get("adaptive_heuristics", True),
            partitions=manifest_partitions,
        )

        manifest_path = os.path.join(out_dir, "slurm_manifest.json")
        ClusterPartitionEngine.save_manifest(manifest, manifest_path)

        SlurmScriptGenerator.generate_scripts(
            out_dir=out_dir,
            manifest_path=manifest_path,
            num_partitions=len(partitions),
            cpus_per_task=self.user_params.get("slurm_cpus", 8),
            mem_per_task=self.user_params.get("slurm_mem", "16G"),
            time_limit=self.user_params.get("slurm_time", "04:00:00"),
            slurm_partition=self.user_params.get("slurm_partition"),
        )
        log.info("Slurm cluster export complete.")

    def _apply_manifest_to_params(self, manifest: ClusterManifest) -> None:
        """Adopts what the export step recorded in the manifest.

        The generated Slurm scripts invoke workers and the finalizer without the
        selection mode or aligner options, so the manifest is the source of truth
        for the mode and falls back for the aligner settings (explicitly given
        non-default worker options win).
        """
        params = self.user_params
        mode = SelectMode.__members__.get(
            str(manifest.select_mode).split(".")[-1].strip().upper()
        )
        if mode is not None:
            params["select_mode"] = mode
        if manifest.alignment_tool and params.get("alignment_tool") in (None, "mafft"):
            params["alignment_tool"] = manifest.alignment_tool
        if manifest.alignment_command and not params.get("alignment_command"):
            params["alignment_command"] = manifest.alignment_command
        if (
            manifest.alignment_tool_path
            and not params.get("alignment_tool_path")
            and os.path.exists(manifest.alignment_tool_path)
        ):
            params["alignment_tool_path"] = manifest.alignment_tool_path
        if manifest.adaptive_heuristics is False:
            params["adaptive_heuristics"] = False

    def _execute_worker(self):
        """Worker task execution: aligns assigned partition regions."""
        out_dir: str = self.user_params["out_dir"]
        partition_id: int = self.user_params["align_partition"]
        manifest_path = self.user_params.get("manifest_path") or os.path.join(
            out_dir, "slurm_manifest.json"
        )

        manifest = ClusterPartitionEngine.load_manifest(manifest_path)
        self._apply_manifest_to_params(manifest)
        p_key = str(partition_id)
        if p_key not in manifest.partitions:
            raise ClusterExecutionError(
                f"Partition ID {partition_id} not found in manifest: {manifest_path}"
            )

        assigned_regions = manifest.partitions[p_key]["regions"]
        log.info(
            f"Worker running partition {partition_id} ({len(assigned_regions)} regions)..."
        )

        # Initialize minimal PlastidData connecting to the existing SQLite store
        self.plastid_data = PlastidData(self.user_params)
        aligncoord = AlignmentCoordination(self.plastid_data, self.user_params)
        try:
            successful_regions = aligncoord.align_partition_regions(assigned_regions)
        finally:
            aligncoord.close_scratch()

        # Record atomic receipt
        PartitionReceipt.write_receipt(
            out_dir=out_dir,
            partition_id=partition_id,
            regions=assigned_regions,
            successful_regions=successful_regions,
        )
        log.info(
            f"Partition {partition_id} complete. "
            f"Successfully aligned {len(successful_regions)}/{len(assigned_regions)} regions."
        )

    def _execute_finalize(self):
        """Finalizer task execution: validates receipts and concatenates supermatrices."""
        started = datetime.datetime.now(datetime.timezone.utc)
        t0 = time.monotonic()
        out_dir: str = self.user_params["out_dir"]
        manifest_path = self.user_params.get("manifest_path") or os.path.join(
            out_dir, "slurm_manifest.json"
        )

        manifest = ClusterPartitionEngine.load_manifest(manifest_path)
        self._apply_manifest_to_params(manifest)
        all_done, missing_ids, successful_regions = (
            PartitionReceipt.verify_all_completed(
                out_dir=out_dir,
                num_partitions=manifest.num_partitions,
            )
        )

        if not all_done:
            raise ClusterExecutionError(
                f"Cannot finalize: {len(missing_ids)} partitions incomplete: {missing_ids}"
            )

        log.info(
            f"All {manifest.num_partitions} partitions verified. "
            f"Total successful region alignments: {len(successful_regions)}"
        )

        self.plastid_data = PlastidData(self.user_params)
        aligncoord = AlignmentCoordination(self.plastid_data, self.user_params)
        ok = False
        try:
            aligncoord.collect_partition_nexus(successful_regions)
            aligncoord.concat_MSAs()
            ok = True
            log.info("Cluster finalization and concatenation completed successfully.")
        finally:
            aligncoord.close_scratch()
            self.last_aligncoord = aligncoord
            self._write_manifest(
                started_at=started,
                wall_seconds=time.monotonic() - t0,
                aligncoord=aligncoord,
                ok=ok,
            )
