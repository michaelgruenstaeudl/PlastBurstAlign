from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from plastburstalign.cluster_ops import (
    ClusterManifest,
    ClusterPartitionEngine,
    PartitionReceipt,
    SlurmScriptGenerator,
)


class TestClusterPartitionEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_unaligned_file(self, name: str, size_bytes: int):
        path = os.path.join(self.out_dir, f"nucl_{name}.unalign.fasta")
        with open(path, "wb") as f:
            f.write(b"N" * size_bytes)

    def test_partition_regions_greedy_lpt_distribution(self):
        self._create_unaligned_file("locus_huge", 100_000)
        self._create_unaligned_file("locus_large", 80_000)
        self._create_unaligned_file("locus_medium1", 40_000)
        self._create_unaligned_file("locus_medium2", 30_000)
        self._create_unaligned_file("locus_small1", 10_000)
        self._create_unaligned_file("locus_small2", 5_000)

        regions = [
            "locus_small2",
            "locus_medium1",
            "locus_huge",
            "locus_large",
            "locus_small1",
            "locus_medium2",
        ]

        partitions = ClusterPartitionEngine.partition_regions(
            regions=regions,
            out_dir=self.out_dir,
            num_partitions=2,
            prefix="nucl",
        )

        self.assertEqual(len(partitions), 2)
        all_assigned = partitions[0].regions + partitions[1].regions
        self.assertEqual(sorted(all_assigned), sorted(regions))

        p0 = next(p for p in partitions if p.partition_id == 0)
        p1 = next(p for p in partitions if p.partition_id == 1)
        self.assertEqual(p0.estimated_bytes, 135_000)
        self.assertEqual(p1.estimated_bytes, 130_000)

    def test_partition_regions_more_partitions_than_regions(self):
        self._create_unaligned_file("geneA", 100)
        self._create_unaligned_file("geneB", 200)

        partitions = ClusterPartitionEngine.partition_regions(
            regions=["geneA", "geneB"],
            out_dir=self.out_dir,
            num_partitions=10,
            prefix="nucl",
        )
        self.assertEqual(len(partitions), 2)

    def test_manifest_roundtrip(self):
        manifest_path = os.path.join(self.out_dir, "slurm_manifest.json")
        manifest = ClusterManifest(
            version="1.0",
            created_at="2026-10-27T00:00:00Z",
            select_mode="cds",
            out_dir=self.out_dir,
            total_regions=2,
            num_partitions=2,
            alignment_tool="mafft",
            alignment_tool_path=None,
            alignment_command=None,
            adaptive_heuristics=True,
            partitions={
                "0": {"regions": ["rbcL"], "estimated_bytes": 1000},
                "1": {"regions": ["matK"], "estimated_bytes": 1200},
            },
        )

        ClusterPartitionEngine.save_manifest(manifest, manifest_path)
        loaded = ClusterPartitionEngine.load_manifest(manifest_path)

        self.assertEqual(loaded.version, "1.0")
        self.assertEqual(loaded.total_regions, 2)
        self.assertEqual(loaded.partitions["0"]["regions"], ["rbcL"])
        self.assertEqual(loaded.partitions["1"]["regions"], ["matK"])


class TestSlurmScriptGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = self.temp_dir.name
        self.manifest_path = os.path.join(self.out_dir, "slurm_manifest.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_script_generation(self):
        scripts = SlurmScriptGenerator.generate_scripts(
            out_dir=self.out_dir,
            manifest_path=self.manifest_path,
            num_partitions=4,
            cpus_per_task=16,
            mem_per_task="32G",
            time_limit="08:00:00",
            slurm_partition="standard",
        )

        self.assertEqual(len(scripts), 3)

        array_path = os.path.join(self.out_dir, "submit_array.slurm")
        finalize_path = os.path.join(self.out_dir, "submit_finalize.slurm")
        run_path = os.path.join(self.out_dir, "run_pipeline.sh")

        self.assertTrue(os.path.exists(array_path))
        self.assertTrue(os.path.exists(finalize_path))
        self.assertTrue(os.path.exists(run_path))

        with open(array_path, "r", encoding="utf-8") as f:
            array_content = f.read()

        self.assertIn("#SBATCH --array=0-3", array_content)
        self.assertIn("#SBATCH --cpus-per-task=16", array_content)
        self.assertIn("#SBATCH --mem=32G", array_content)
        self.assertIn("#SBATCH -p standard", array_content)
        self.assertIn("--align-partition", array_content)

        with open(finalize_path, "r", encoding="utf-8") as f:
            finalize_content = f.read()

        self.assertIn("--finalize-slurm", finalize_content)


class TestPartitionReceipt(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_receipt_writing_and_verification(self):
        all_done, missing, _ = PartitionReceipt.verify_all_completed(
            self.out_dir, num_partitions=2
        )
        self.assertFalse(all_done)
        self.assertEqual(missing, [0, 1])

        PartitionReceipt.write_receipt(
            out_dir=self.out_dir,
            partition_id=0,
            regions=["atpA", "rbcL"],
            successful_regions=["atpA", "rbcL"],
        )

        all_done, missing, successful = PartitionReceipt.verify_all_completed(
            self.out_dir, num_partitions=2
        )
        self.assertFalse(all_done)
        self.assertEqual(missing, [1])
        self.assertEqual(successful, ["atpA", "rbcL"])

        PartitionReceipt.write_receipt(
            out_dir=self.out_dir,
            partition_id=1,
            regions=["matK"],
            successful_regions=["matK"],
        )

        all_done, missing, successful = PartitionReceipt.verify_all_completed(
            self.out_dir, num_partitions=2
        )
        self.assertTrue(all_done)
        self.assertEqual(missing, [])
        self.assertEqual(sorted(successful), ["atpA", "matK", "rbcL"])


class TestClusterPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = self.temp_dir.name
        self.fasta_dir = os.path.join(self.work_dir, "fasta")
        self.nexus_dir = os.path.join(self.work_dir, "nexus")
        os.makedirs(self.fasta_dir, exist_ok=True)
        os.makedirs(self.nexus_dir, exist_ok=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_mock_unaligned(self, region: str, seqs: dict[str, str]):
        fasta_path = os.path.join(self.work_dir, f"nucl_{region}.unalign.fasta")
        with open(fasta_path, "w", encoding="utf-8") as f:
            for header, seq in seqs.items():
                f.write(f">{header}\n{seq}\n")

    def test_full_cluster_pipeline_matches_expected_output(self):
        from plastburstalign.plast_burst_align import PlastBurstAlign
        from plastburstalign.user_parameters import UserParameters

        regions = {
            "atpA": {"taxon1": "ATGCCGTAA", "taxon2": "ATGCCGTAG"},
            "matK": {"taxon1": "ATGAAATAA", "taxon2": "ATGAAATAG"},
            "rbcL": {"taxon1": "ATGTTTTAA", "taxon2": "ATGTTTTAG"},
        }

        mock_plastid_data = MagicMock()
        mock_plastid_data.order = "alpha"
        mock_plastid_data.set_order_map.return_value = None
        mock_plastid_data.order_map = {"atpA": 0, "matK": 1, "rbcL": 2}
        mock_plastid_data.nucleotides = {r: None for r in regions}
        mock_plastid_data.store.get_region_summary.return_value = {}

        def mock_export_fasta(region: str, seq_type: str, path: str) -> None:
            seqs = regions[region]
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                for header, seq in seqs.items():
                    handle.write(f">{header}\n{seq}\n")

        mock_plastid_data.store.export_fasta.side_effect = mock_export_fasta

        # Stage 1: Export
        export_params = UserParameters(
            in_dir=self.work_dir,
            out_dir=self.work_dir,
            select_mode="igs",
            export_slurm=True,
            slurm_partitions=2,
            concat=True,
        )
        mock_plastid_data.mode = export_params.get("select_mode")

        with (
            patch("plastburstalign.plast_burst_align.ExtractAndCollect"),
            patch("plastburstalign.plast_burst_align.DataCleaning"),
            patch(
                "plastburstalign.plast_burst_align.PlastidData",
                return_value=mock_plastid_data,
            ),
        ):
            pba_export = PlastBurstAlign(export_params)
            pba_export.execute()

        # Fresh runs clear stale unaligned FASTA; recreate after export for workers
        # that still prefer shared out_dir copies when present.
        for r, seqs in regions.items():
            self._create_mock_unaligned(r, seqs)

        manifest_path = os.path.join(self.work_dir, "slurm_manifest.json")
        self.assertTrue(os.path.exists(manifest_path))
        manifest = ClusterPartitionEngine.load_manifest(manifest_path)
        self.assertEqual(manifest.num_partitions, 2)

        # Stage 2: Worker Tasks
        def mock_align(in_fn: str, out_fn: str, **kwargs):
            with (
                open(in_fn, "r", encoding="utf-8") as src,
                open(out_fn, "w", encoding="utf-8") as dst,
            ):
                dst.write(src.read())
            return True

        for p_id in range(manifest.num_partitions):
            worker_params = UserParameters(
                in_dir=self.work_dir,
                out_dir=self.work_dir,
                select_mode="igs",
                align_partition=p_id,
                manifest_path=manifest_path,
                num_threads=1,
            )
            with (
                patch(
                    "plastburstalign.plast_burst_align.PlastidData",
                    return_value=mock_plastid_data,
                ),
                patch("plastburstalign.alignment_ops.get_alignment_tool") as mock_tool,
            ):
                mock_tool.return_value.align.side_effect = mock_align
                pba_worker = PlastBurstAlign(worker_params)
                pba_worker.execute()

            receipt_path = os.path.join(
                self.work_dir, "partitions", f"partition_{p_id}.done"
            )
            self.assertTrue(os.path.exists(receipt_path))

        for r in regions:
            nexus_file = os.path.join(self.nexus_dir, f"nucl_{r}.aligned.nexus")
            self.assertTrue(os.path.exists(nexus_file), f"Missing {nexus_file}")

        # Stage 3: Finalizer Task
        finalize_params = UserParameters(
            in_dir=self.work_dir,
            out_dir=self.work_dir,
            select_mode="igs",
            finalize_slurm=True,
            manifest_path=manifest_path,
            concat=True,
        )

        with patch(
            "plastburstalign.plast_burst_align.PlastidData",
            return_value=mock_plastid_data,
        ):
            pba_finalize = PlastBurstAlign(finalize_params)
            pba_finalize.execute()

        concat_fasta = os.path.join(self.work_dir, "nucl_3concat.aligned.fasta")
        concat_nexus = os.path.join(self.work_dir, "nucl_3concat.aligned.nexus")

        self.assertTrue(os.path.exists(concat_fasta))
        self.assertTrue(os.path.exists(concat_nexus))
        self.assertGreater(os.path.getsize(concat_fasta), 0)
        self.assertGreater(os.path.getsize(concat_nexus), 0)


if __name__ == "__main__":
    unittest.main()
