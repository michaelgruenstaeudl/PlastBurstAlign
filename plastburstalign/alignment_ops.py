from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Data.CodonTable import ambiguous_generic_by_id
from Bio.Nexus.Nexus import Nexus as NexusAlignment
from Bio.Seq import MutableSeq, Seq
from Bio.SeqRecord import SeqRecord

from .alignment_tools import AlignmentTool, get_alignment_tool
from .helpers import bounded_submit_iterator, resolve_scratch_dir, split_list
from .logging_ops import logger as log
from .nexus_ops import (
    ConcatenationError,
    IncrementalNexusAccumulator,
    fast_write_fasta,
    fast_write_nexus,
    fast_write_nexus_partitions,
    fast_write_raxml_partitions,
)
from .seqfeature_ops import PlastidData
from .task_queue import CoreBudgetPool

# ===================================================================
# Module-level helper functions for multiprocessing
# ===================================================================


def fast_fasta_to_nexus(fasta_path: str, nexus_path: str) -> bool:
    """High-throughput FASTA-to-NEXUS converter bypassing SeqRecord object overhead."""
    taxa: list[tuple[str, str]] = []
    try:
        if not os.path.exists(fasta_path) or os.path.getsize(fasta_path) == 0:
            return False

        with open(fasta_path, "r", buffering=1024 * 1024) as f:
            curr_id = None
            chunks: list[str] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if curr_id:
                        taxa.append((curr_id, "".join(chunks)))
                        chunks.clear()
                    curr_id = line[1:].split()[0]
                else:
                    chunks.append(line)
            if curr_id:
                taxa.append((curr_id, "".join(chunks)))

        if not taxa:
            return False

        ntax = len(taxa)
        nchar = len(taxa[0][1])

        os.makedirs(os.path.dirname(os.path.abspath(nexus_path)), exist_ok=True)
        with open(nexus_path, "w", buffering=1024 * 1024) as f:
            f.write("#NEXUS\nbegin data;\n")
            f.write(f"\tdimensions ntax={ntax} nchar={nchar};\n")
            f.write("\tformat datatype=dna missing=? gap=-;\nmatrix\n")
            max_id_len = max(len(t[0]) for t in taxa) + 4
            for taxon_id, seq in taxa:
                f.write(f"{taxon_id.ljust(max_id_len)}{seq}\n")
            f.write(";\nend;\n")
        return True
    except Exception as e:
        log.warning(f"Failed to convert {fasta_path} to NEXUS: {e}")
        return False


class FastNexusMatrix:
    """Lightweight duck-typed container matching NexusAlignment interface for concatenation."""

    __slots__ = (
        "matrix",
        "taxlabels",
        "nchar",
        "datatype",
        "gap",
        "missing",
        "charsets",
        "taxsets",
        "charlabels",
        "statelabels",
        "charpartitions",
    )

    def __init__(self, matrix: dict[str, str], taxlabels: list[str], nchar: int):
        self.matrix = matrix
        self.taxlabels = taxlabels
        self.nchar = nchar
        self.datatype = "dna"
        self.gap = "-"
        self.missing = "?"
        self.charsets: dict[str, Any] = {}
        self.taxsets: dict[str, Any] = {}
        self.charlabels: list[str] | None = None
        self.statelabels: list[str] | None = None
        self.charpartitions: dict[str, Any] = {}


class AlignmentCoordination:
    def __init__(self, plastid_data: PlastidData, user_params: dict[str, Any]):
        """
        Coordinates the alignment of nucleotide or protein sequences.

        Args:
            plastid_data: Plastid data to be aligned.
            user_params: Specifications for how the alignment should be performed.
                These are `num_threads`, `out_dir`, `verbose`, and `concat`.
        """
        self.plastid_data = plastid_data
        self.user_params = user_params
        self.success_list: list[tuple[str, str]] = []
        self._unaligned_saved = False
        self._aligned_saved = False
        self._success_lock = threading.Lock()
        self._scratch_lock = threading.Lock()
        self._scratch_dir: str | None = None

        # Create the appropriate alignment tool
        tool_name = (user_params or {}).get("alignment_tool", "mafft") or "mafft"
        tool_path = user_params.get("alignment_tool_path") if user_params else None
        aligner = get_alignment_tool(tool_name, user_params, tool_path)

        if not aligner:
            log.error("Failed to initialize alignment tool, falling back to MAFFT")
            aligner = get_alignment_tool("mafft", user_params, None)
        if aligner is None:
            raise RuntimeError("No alignment tool could be initialized")
        self.aligner: AlignmentTool = aligner

    @property
    def _out_dir(self) -> str:
        """The output directory (a required user parameter)."""
        return str(self.user_params["out_dir"])

    @property
    def mode(self):
        """Returns the selection mode from PlastidData or UserParameters."""
        if (
            hasattr(self, "plastid_data")
            and self.plastid_data
            and hasattr(self.plastid_data, "mode")
            and self.plastid_data.mode
        ):
            return self.plastid_data.mode
        return self.user_params.get("select_mode")

    def _aligner_supports_param(self, param_name: str) -> bool:
        """Determines if the configured aligner accepts the given keyword parameter."""
        try:
            align_params = inspect.signature(self.aligner.align).parameters
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in align_params.values()
            )
            return param_name in align_params or has_var_keyword
        except (ValueError, TypeError):
            return False

    def save_unaligned(self):
        """Saves unaligned nucleotide and protein matrices in parallel."""
        log.info("saving individual regions as unaligned matrices in parallel")
        collect_proteins = self.plastid_data.mode.collects_proteins()
        out_dir = self._out_dir
        store = self.plastid_data.store

        def _export_single(feat_name: str) -> None:
            try:
                out_fn_nucl = os.path.join(out_dir, f"nucl_{feat_name}.unalign.fasta")
                store.export_fasta(feat_name, "nucl", out_fn_nucl)

                if collect_proteins:
                    out_fn_prot = os.path.join(
                        out_dir, f"prot_{feat_name}.unalign.fasta"
                    )
                    store.export_fasta(feat_name, "prot", out_fn_prot)
            except Exception as e:
                log.error(f"Error saving {feat_name}: {e}")

        regions = list(self.plastid_data.nucleotides.keys())
        raw_threads = self.user_params.get("num_threads", 4) or 4
        max_workers = min(int(raw_threads), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_export_single, regions))

        self._unaligned_saved = True

    def perform_MSA(self):
        """
        For each region in the plastid data, the sequences are aligned and saved to file as a nucleotide matrix.
        These are saved in FASTA format, located in the path specified by `user_params.out_dir`.

        In the nucleotide case, this is the direct process of aligning the unaligned matrix of each region
        using a third-party software tool. For the protein case, each unaligned matrix is aligned, and
        then back-translated to a nucleotide matrix before saving to file.
        """
        if not self.aligner or not self.aligner.exec_path:
            log.error(
                "unable to align regions; alignment tool is not properly configured"
            )
            return
        elif not self._unaligned_saved:
            self.save_unaligned()

        log.info("conducting the alignment of extracted sequences")
        if self.plastid_data.mode.collects_proteins():
            self._prot_MSA()
        else:
            self._nuc_MSA()
        self._aligned_saved = True

    # ------------------------------------------------------------------
    # Shared helpers for the monolithic and the Slurm-worker alignment paths
    # ------------------------------------------------------------------

    _ALIGNER_OPTIONAL_PARAMS = ("threads", "sequence_count", "max_len", "min_len")

    def _resolve_num_workers(self) -> int:
        raw_threads = self.user_params.get("num_threads", 1)
        try:
            num_workers = int(raw_threads)
        except (ValueError, TypeError):
            num_workers = os.cpu_count() or 1
        return max(1, num_workers)

    def _aligner_support(self) -> dict[str, bool]:
        """Which optional keyword arguments the aligner's `align` accepts."""
        return {
            name: self._aligner_supports_param(name)
            for name in self._ALIGNER_OPTIONAL_PARAMS
        }

    def _get_region_summary(self, seq_type: str) -> dict[str, dict[str, int]]:
        """Fetches the per-region summary (count, max_len, min_len) in one query."""
        try:
            summary = self.plastid_data.store.get_region_summary(seq_type=seq_type)
        except Exception:
            return {}
        return summary if isinstance(summary, dict) else {}

    @staticmethod
    def _build_align_kwargs(
        summary_entry: Mapping[str, int],
        leased_threads: int,
        support: Mapping[str, bool],
    ) -> dict[str, Any]:
        """Dynamically passes only the arguments the aligner accepts."""
        align_kwargs: dict[str, Any] = {}
        if support["threads"]:
            align_kwargs["threads"] = leased_threads
        seq_count = summary_entry.get("count")
        if support["sequence_count"] and seq_count is not None:
            align_kwargs["sequence_count"] = seq_count
            max_len = summary_entry.get("max_len")
            if support["max_len"] and max_len is not None:
                align_kwargs["max_len"] = max_len
            min_len = summary_entry.get("min_len")
            if support["min_len"] and min_len is not None:
                align_kwargs["min_len"] = min_len
        return align_kwargs

    def _get_worker_scratch(self) -> str:
        """Lazily creates a private scratch directory for exported unaligned FASTA."""
        with self._scratch_lock:
            if self._scratch_dir is None:
                base = resolve_scratch_dir(self.user_params.get("scratch_dir"))
                self._scratch_dir = tempfile.mkdtemp(prefix="pba_unaligned_", dir=base)
            return self._scratch_dir

    def _cleanup_worker_scratch(self) -> None:
        with self._scratch_lock:
            scratch, self._scratch_dir = self._scratch_dir, None
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    def _unaligned_fasta(self, region: str, seq_type: str) -> tuple[str, bool]:
        """Returns the unaligned FASTA of a region, exporting it if it is not on disk.

        An existing non-empty `<seq_type>_<region>.unalign.fasta` in `out_dir` is
        used as is. Otherwise the region is exported from the SQLite store into
        node-local scratch, so Slurm workers do not need any shared unaligned files.

        Returns:
            (path, is_temporary); temporary files should be removed after use.
        """
        shared = os.path.join(self._out_dir, f"{seq_type}_{region}.unalign.fasta")
        if os.path.exists(shared) and os.path.getsize(shared) > 0:
            return shared, False
        local = os.path.join(
            self._get_worker_scratch(), f"{seq_type}_{region}.unalign.fasta"
        )
        self.plastid_data.store.export_fasta(region, seq_type, local)
        return local, True

    @staticmethod
    def _remove_quietly(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    def _regions_to_align(self, target_regions: set[str] | None) -> list[str]:
        keys = list(self.plastid_data.nucleotides.keys())
        if target_regions is None:
            return keys
        return [k for k in keys if k in target_regions]

    def _nuc_MSA(self, target_regions: set[str] | None = None):
        """Aligns nucleotide regions (all of them, or only `target_regions`)."""
        # Pre-fetch region taxon counts in one O(1) bulk query
        region_summary = self._get_region_summary("nucl")
        num_workers = self._resolve_num_workers()

        log.info(
            "conducting multiple sequence alignments based on nucleotide sequence"
            f" data using {num_workers} threads"
        )

        # Cheap emptiness check: must not hydrate any SeqRecord
        if len(self.plastid_data.nucleotides) == 0:
            log.critical("No items in nucleotide main dictionary to process")
            raise RuntimeError("No items in nucleotide main dictionary to process")

        # Detect supported parameters on the aligner interface
        support = self._aligner_support()
        supports_threads = support["threads"]

        regions = self._regions_to_align(target_regions)
        total_tasks = len(regions)

        # Calculate water-filling schedule from region summary
        tasks_meta = {
            k: (
                region_summary.get(k, {}).get("count", 1),
                region_summary.get(k, {}).get("max_len", 1),
            )
            for k in regions
        }
        schedule = CoreBudgetPool.solve_water_filling_schedule(
            tasks_meta, total_cores=num_workers
        )

        budget_pool = CoreBudgetPool(
            total_cores=num_workers,
            total_tasks=total_tasks,
            max_cores_per_task=None if supports_threads else 1,
            schedule=schedule if supports_threads else None,
        )

        ### Inner Function - Start ###
        def single_nuc_MSA(k: str):
            log.debug(f" aligning {k}")
            # Define input and output names
            fasta_dir = os.path.join(self._out_dir, "fasta")
            os.makedirs(fasta_dir, exist_ok=True)
            out_fn_aligned_nucl = os.path.join(fasta_dir, f"nucl_{k}.aligned.fasta")

            out_fn_unalign_nucl, is_temp = self._unaligned_fasta(k, "nucl")
            file_size = (
                os.path.getsize(out_fn_unalign_nucl)
                if os.path.exists(out_fn_unalign_nucl)
                else 0
            )
            summary_entry = region_summary.get(k, {})

            try:
                try:
                    with budget_pool.lease(
                        file_size=file_size, task_id=k
                    ) as leased_threads:
                        align_kwargs = self._build_align_kwargs(
                            summary_entry, leased_threads, support
                        )

                        # Step 1. Align matrices via third-party alignment tool
                        self.aligner.align(
                            out_fn_unalign_nucl, out_fn_aligned_nucl, **align_kwargs
                        )
                finally:
                    budget_pool.task_completed()
            finally:
                if is_temp:
                    self._remove_quietly(out_fn_unalign_nucl)

            # Report whether the aligner produced output
            if (
                os.path.exists(out_fn_aligned_nucl)
                and os.path.getsize(out_fn_aligned_nucl) > 0
            ):
                log.debug(f"Alignment successful for `{k}` -> {out_fn_aligned_nucl}")
            else:
                log.warning(
                    f"Alignment for `{k}` produced no output: {out_fn_aligned_nucl}"
                )

            # Convert aligned nucleotide FASTA to NEXUS in-situ if enabled
            self._record_success(k, out_fn_aligned_nucl)

        ### Inner Function - End ###

        # Step 2. Use ThreadPoolExecutor to parallelize alignment with LPT ordering
        max_in_flight = num_workers * 2
        sorted_regions = [
            k
            for k in self._get_lpt_sorted_regions(prefix="nucl")
            if target_regions is None or k in target_regions
        ]
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for k, future in bounded_submit_iterator(
                    executor,
                    single_nuc_MSA,
                    sorted_regions,
                    max_in_flight=max_in_flight,
                ):
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"{k!r} generated an exception: {e}")
        finally:
            self._cleanup_worker_scratch()

    def _record_success(self, region: str, aligned_fasta: str) -> None:
        """Registers an aligned region, converting it to NEXUS in-situ if enabled."""
        individual_nexus = self.user_params.get("individual_nexus", True)
        if individual_nexus:
            nexus_dir = os.path.join(self._out_dir, "nexus")
            out_fn_aligned_nexus = os.path.join(
                nexus_dir, f"nucl_{region}.aligned.nexus"
            )
            if fast_fasta_to_nexus(aligned_fasta, out_fn_aligned_nexus):
                with self._success_lock:
                    self.success_list.append((region, out_fn_aligned_nexus))
        else:
            with self._success_lock:
                self.success_list.append((region, aligned_fasta))

    def _prot_MSA(self, target_regions: set[str] | None = None):
        """Aligns proteins and back-translates them (all regions, or `target_regions`)."""
        # Pre-fetch region taxon counts in one O(1) bulk query
        region_summary = self._get_region_summary("prot")
        num_workers = self._resolve_num_workers()

        log.info(
            " conducting multiple sequence alignments based on protein sequence"
            " data, followed by back-translation to nucleotides using"
            f" {num_workers} threads"
        )

        # Cheap emptiness check: must not hydrate any SeqRecord
        if len(self.plastid_data.nucleotides) == 0:
            log.critical("No items in nucleotide main dictionary to process")
            raise RuntimeError("No items in nucleotide main dictionary to process")

        # Detect supported parameters on the aligner interface
        support = self._aligner_support()
        supports_threads = support["threads"]

        regions = self._regions_to_align(target_regions)
        total_tasks = len(regions)

        # Calculate water-filling schedule from protein region summary
        tasks_meta = {
            k: (
                region_summary.get(k, {}).get("count", 1),
                region_summary.get(k, {}).get("max_len", 1),
            )
            for k in regions
        }
        schedule = CoreBudgetPool.solve_water_filling_schedule(
            tasks_meta, total_cores=num_workers
        )

        budget_pool = CoreBudgetPool(
            total_cores=num_workers,
            total_tasks=total_tasks,
            max_cores_per_task=None if supports_threads else 1,
            schedule=schedule if supports_threads else None,
        )

        ### Inner Function - Start ###
        def single_prot_MSA(k: str):
            log.debug(f" aligning {k}")
            # Define input and output names
            out_dir = self._out_dir
            out_fn_aligned_prot = os.path.join(out_dir, f"prot_{k}.aligned.fasta")
            fasta_dir = os.path.join(out_dir, "fasta")
            os.makedirs(fasta_dir, exist_ok=True)
            out_fn_aligned_nucl = os.path.join(fasta_dir, f"nucl_{k}.aligned.fasta")

            out_fn_unalign_prot, prot_is_temp = self._unaligned_fasta(k, "prot")
            file_size = (
                os.path.getsize(out_fn_unalign_prot)
                if os.path.exists(out_fn_unalign_prot)
                else 0
            )
            summary_entry = region_summary.get(k, {})

            try:
                try:
                    with budget_pool.lease(
                        file_size=file_size, task_id=k
                    ) as leased_threads:
                        align_kwargs = self._build_align_kwargs(
                            summary_entry, leased_threads, support
                        )

                        # Step 1. Align matrices based on their PROTEIN sequences via third-party alignment tool
                        self.aligner.align(
                            out_fn_unalign_prot, out_fn_aligned_prot, **align_kwargs
                        )
                finally:
                    budget_pool.task_completed()
            finally:
                if prot_is_temp:
                    self._remove_quietly(out_fn_unalign_prot)

            # Report whether the protein aligner produced output
            if (
                os.path.exists(out_fn_aligned_prot)
                and os.path.getsize(out_fn_aligned_prot) > 0
            ):
                log.debug(
                    f"Protein alignment successful for `{k}` -> {out_fn_aligned_prot}"
                )
            else:
                log.warning(
                    f"Protein alignment for `{k}` produced no output:"
                    f" {out_fn_aligned_prot}"
                )
                return

            # Step 2. Conduct actual back-translation from PROTEINS TO NUCLEOTIDES
            out_fn_unalign_nucl, nucl_is_temp = self._unaligned_fasta(k, "nucl")
            try:
                translator = BackTranslation(
                    k,
                    out_fn_aligned_prot,
                    out_fn_unalign_nucl,
                    out_fn_aligned_nucl,
                    11,
                )
                translator.backtranslate()
                if not (
                    os.path.exists(out_fn_aligned_nucl)
                    and os.path.getsize(out_fn_aligned_nucl) > 0
                ):
                    log.warning(
                        f"Back-translation produced no output for `{k}`: {out_fn_aligned_nucl}"
                    )
                    return
                log.debug(
                    f"Back-translation successful for `{k}` -> {out_fn_aligned_nucl}"
                )
            except Exception as err:
                log.warning(f"Unable to conduct back-translation of `{k}`: {err}")
                return
            finally:
                if nucl_is_temp:
                    self._remove_quietly(out_fn_unalign_nucl)

            # Step 3. Convert aligned nucleotide FASTA to NEXUS in-situ if enabled
            self._record_success(k, out_fn_aligned_nucl)

        ### Inner Function - End ###

        # Step 2. Use ThreadPoolExecutor to parallelize alignment and back-translation with LPT ordering
        max_in_flight = num_workers * 2
        sorted_regions = [
            k
            for k in self._get_lpt_sorted_regions(prefix="prot")
            if target_regions is None or k in target_regions
        ]
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for k, future in bounded_submit_iterator(
                    executor,
                    single_prot_MSA,
                    sorted_regions,
                    max_in_flight=max_in_flight,
                ):
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"{k} generated an exception: {e}")
        finally:
            self._cleanup_worker_scratch()

    def perform_MSAs(self):
        """Perform multiple sequence alignments and ensure all loci convert to NEXUS."""
        if not self._aligned_saved:
            self.perform_MSA()

        completed_keys = {name for name, _ in self.success_list}
        missing_loci = [
            k for k in self.plastid_data.nucleotides.keys() if k not in completed_keys
        ]
        if missing_loci and self.user_params.get("individual_nexus", True):
            log.info(
                f"converting {len(missing_loci)} remaining alignments to NEXUS format"
            )
            raw_threads = self.user_params.get("num_threads", 1) or 1
            num_workers = max(1, int(raw_threads))
            msa_lists = split_list(missing_loci, num_workers * 2)
            max_in_flight = num_workers * 2
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for _, future in bounded_submit_iterator(
                    executor, self.convert_MSAs, msa_lists, max_in_flight=max_in_flight
                ):
                    res = future.result()
                    if res:
                        with self._success_lock:
                            self.success_list.extend(res)

    def convert_MSAs(self, msa_list: list[str]) -> list[tuple[str, str]]:
        """Convert each region's aligned FASTA file to NEXUS format and collect successful conversions."""

        def _convert_FASTA_to_NEXUS(msa_name: str) -> str | None:
            fasta_dir = os.path.join(self._out_dir, "fasta")
            aligned_nucl_fasta = os.path.join(
                fasta_dir, f"nucl_{msa_name}.aligned.fasta"
            )
            nexus_dir = os.path.join(self._out_dir, "nexus")
            os.makedirs(nexus_dir, exist_ok=True)
            aligned_nucl_nexus = os.path.join(
                nexus_dir, f"nucl_{msa_name}.aligned.nexus"
            )
            try:
                AlignIO.convert(
                    aligned_nucl_fasta,
                    "fasta",
                    aligned_nucl_nexus,
                    "nexus",
                    molecule_type="DNA",
                )
                return aligned_nucl_nexus
            except Exception:
                log.warning(
                    f"Unable to convert alignment of `{msa_name}` from FASTA to NEXUS."
                )
                return None

        success_list = []
        for msa_name in msa_list:
            nexus_path = _convert_FASTA_to_NEXUS(msa_name)
            if nexus_path is not None:
                success_list.append((msa_name, nexus_path))
        return success_list

    def concat_MSAs(self):
        """Concatenate all successful alignments into a single alignment."""
        num_successful = len(self.success_list)
        if num_successful == 0:
            log.warning("No successful alignments to concatenate.")
            return

        def _concat_sync(alignm_concat: NexusAlignment):
            try:
                if alignm_concat.matrix is None:
                    raise ValueError("concatenated alignment has no data matrix")

                # 1. Fast linear FASTA export
                fast_write_fasta(
                    filename=out_fn_nucl_concat_fasta,
                    matrix=alignm_concat.matrix,
                    taxlabels=getattr(alignm_concat, "taxlabels", None),
                )

                # 2. Fast linear NEXUS export
                fast_write_nexus(
                    filename=out_fn_nucl_concat_nexus,
                    matrix=alignm_concat.matrix,
                    datatype=getattr(alignm_concat, "datatype", "dna"),
                    gap=getattr(alignm_concat, "gap", "-"),
                    missing=getattr(alignm_concat, "missing", "?"),
                    charpartitions=getattr(alignm_concat, "charpartitions", None),
                )

                # 3. RAxML-NG and IQ-TREE 2 standalone partition files
                charparts = getattr(alignm_concat, "charpartitions", None)
                if charparts:
                    fast_write_raxml_partitions(
                        filename=out_fn_nucl_concat_raxml_parts,
                        charpartitions=charparts,
                        datatype=getattr(alignm_concat, "datatype", "dna"),
                    )
                    fast_write_nexus_partitions(
                        filename=out_fn_nucl_concat_nexus_parts,
                        charpartitions=charparts,
                    )
            except Exception as e:
                log.error(f"Failure while exporting alignments: {e}")
                raise ConcatenationError(
                    f"Failure while writing concatenated alignments: {e}"
                ) from e

        fasta_dir = os.path.join(self._out_dir, "fasta")

        def _load_alignment_fast(msa_name: str, file_path: str) -> Any:
            prefix = f"{msa_name}_"
            r_prefix = f"_R_{msa_name}_"

            # 1. Resolve whether a valid aligned FASTA exists on disk
            fasta_path = None
            if file_path.endswith(".nexus"):
                candidate_fasta = os.path.join(
                    fasta_dir, f"nucl_{msa_name}.aligned.fasta"
                )
                if (
                    os.path.exists(candidate_fasta)
                    and os.path.getsize(candidate_fasta) > 0
                ):
                    fasta_path = candidate_fasta
            elif os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                fasta_path = file_path

            # 2. Fall back to BioPython parser if FASTA is unavailable
            if fasta_path is None:
                if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                    log.warning(
                        f"Alignment file missing or empty for `{msa_name}`: {file_path}"
                    )
                    return None
                try:
                    alignm_nexus = NexusAlignment(file_path)
                    source_matrix = alignm_nexus.matrix
                    if source_matrix is None:
                        raise ValueError("NEXUS file has no data matrix")
                    new_matrix = {}
                    new_taxlabels = []
                    for t in alignm_nexus.taxlabels:
                        clean_t = (
                            "concat_" + t[len(r_prefix) :]
                            if t.startswith(r_prefix)
                            else "concat_" + t[len(prefix) :]
                            if t.startswith(prefix)
                            else t
                        )
                        new_matrix[clean_t] = source_matrix[t]
                        new_taxlabels.append(clean_t)
                    alignm_nexus.matrix = new_matrix
                    alignm_nexus.taxlabels = new_taxlabels
                    return alignm_nexus
                except Exception as err:
                    log.warning(
                        f"BioPython failed to parse NEXUS for `{msa_name}` ({file_path}): {err}"
                    )
                    return None

            # 3. Fast linear FASTA parse
            matrix: dict[str, str] = {}
            taxlabels: list[str] = []
            try:
                with open(fasta_path, "r", buffering=1024 * 1024) as f:
                    curr_id = None
                    chunks: list[str] = []
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(">"):
                            if curr_id:
                                matrix[curr_id] = "".join(chunks)
                                taxlabels.append(curr_id)
                                chunks.clear()
                            raw_id = line[1:].split()[0]
                            if raw_id.startswith(r_prefix):
                                curr_id = "concat_" + raw_id[len(r_prefix) :]
                            elif raw_id.startswith(prefix):
                                curr_id = "concat_" + raw_id[len(prefix) :]
                            else:
                                curr_id = raw_id
                        else:
                            chunks.append(line)
                    if curr_id:
                        matrix[curr_id] = "".join(chunks)
                        taxlabels.append(curr_id)

                if not taxlabels:
                    return None
                nchar = len(matrix[taxlabels[0]])
                return FastNexusMatrix(matrix, taxlabels, nchar)
            except Exception as err:
                log.warning(
                    f"Fast FASTA parser failed for `{msa_name}` ({fasta_path}): {err}"
                )
                return None

        raw_threads = self.user_params.get("num_threads", 4) or 4
        num_workers = min(max(1, int(raw_threads)), 8)

        log.info(
            f"concatenating all successful alignments in `{self.plastid_data.order}` order"
        )
        self.plastid_data.set_order_map()
        self.success_list.sort(key=lambda t: self.plastid_data.order_map[t[0]])

        out_fn_nucl_concat_fasta = os.path.join(
            self._out_dir,
            f"nucl_{num_successful}concat.aligned.fasta",
        )
        out_fn_nucl_concat_nexus = os.path.join(
            self._out_dir,
            f"nucl_{num_successful}concat.aligned.nexus",
        )
        out_fn_nucl_concat_raxml_parts = os.path.join(
            self._out_dir,
            f"nucl_{num_successful}concat.partitions.txt",
        )
        out_fn_nucl_concat_nexus_parts = os.path.join(
            self._out_dir,
            f"nucl_{num_successful}concat.partitions.nex",
        )

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                matrices = list(
                    pool.map(
                        lambda pair: _load_alignment_fast(pair[0], pair[1]),
                        self.success_list,
                    )
                )

            accumulator = IncrementalNexusAccumulator()
            for (msa_name, _), single_matrix in zip(self.success_list, matrices):
                if single_matrix is not None and getattr(single_matrix, "nchar", 0) > 0:
                    accumulator.add(msa_name, single_matrix)
            alignm_concat = accumulator.finalize()
            if alignm_concat is None:
                raise ConcatenationError("no alignment could be loaded")
        except Exception as e:
            log.critical(f"Unable to concatenate alignments: {e}")
            raise ConcatenationError(f"Alignment concatenation failed: {e}") from e

        _concat_sync(alignm_concat)

    def _get_lpt_sorted_regions(self, prefix: str = "nucl") -> list[str]:
        """Sort region keys in descending order of computational work (LPT first).

        Uses theoretical work W = N * L^2 from the database summary when available,
        falling back to unaligned file size. Ties are broken deterministically by name.
        """
        out_dir = self.user_params.get("out_dir", "")
        keys = list(self.plastid_data.nucleotides.keys())

        try:
            summary = self.plastid_data.store.get_region_summary(seq_type=prefix)
        except Exception:
            summary = {}

        def task_weight(k: str) -> tuple[float, str]:
            # Priority 1: Calibrated complexity scaling from SQLite summary
            if (
                isinstance(summary, dict)
                and k in summary
                and isinstance(summary[k], dict)
            ):
                count = summary[k].get("count", 0) or 0
                max_len = summary[k].get("max_len", 0) or 0
                if count > 0 and max_len > 0:
                    exp = 2.4 if max_len >= 1800 else (2.1 if max_len >= 1200 else 2.0)
                    return (-float(count * (max_len**exp)), k)

            # Priority 2: Unaligned file size on disk
            path = os.path.join(out_dir, f"{prefix}_{k}.unalign.fasta")
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            return (-float(size), k)

        sorted_keys = sorted(keys, key=task_weight)
        if sorted_keys:
            log.debug(
                f"LPT scheduling active ({len(sorted_keys)} tasks). Heaviest: {sorted_keys[:3]}"
            )
        return sorted_keys

    def align_partition_regions(self, regions: list[str]) -> list[str]:
        """Executes alignment and conversion for a scoped subset of regions (worker task).

        The worker needs nothing but the SQLite feature store: unaligned FASTA files
        that are not already on disk are exported from the store into node-local
        scratch, and aligned outputs land in `out_dir/fasta` (and `out_dir/nexus`)
        exactly like in the monolithic path.

        Returns list of region names that aligned and converted successfully.
        """
        if not regions:
            log.warning("No regions assigned to this partition.")
            return []

        # Filter region keys to this partition
        target_set = set(regions)
        if self.mode and self.mode.collects_proteins():
            self._prot_MSA_subset(target_set)
        else:
            self._nuc_MSA_subset(target_set)

        # Convert aligned FASTAs that were not converted in-situ to NEXUS
        if self.user_params.get("individual_nexus", True):
            completed = {name for name, _ in self.success_list}
            missing = [r for r in regions if r not in completed]
            if missing:
                self.success_list.extend(self.convert_MSAs(missing))

        successful = {name for name, _ in self.success_list}
        return [r for r in regions if r in successful]

    def _nuc_MSA_subset(self, target_regions: set[str]):
        """Runs nucleotide MSA restricted to target_regions."""
        self._nuc_MSA(target_regions)

    def _prot_MSA_subset(self, target_regions: set[str]):
        """Runs protein MSA and back-translation restricted to target_regions."""
        self._prot_MSA(target_regions)

    def collect_partition_nexus(self, successful_regions: list[str]):
        """Collects aligned-file paths for successful regions into self.success_list.

        The individual NEXUS file is preferred; if it is absent (for example when the
        workers ran with individual NEXUS output disabled) the aligned FASTA is used.
        """
        out_dir = self._out_dir
        nexus_dir = os.path.join(out_dir, "nexus")
        fasta_dir = os.path.join(out_dir, "fasta")
        self.success_list = []
        for r in successful_regions:
            nexus_path = os.path.join(nexus_dir, f"nucl_{r}.aligned.nexus")
            fasta_path = os.path.join(fasta_dir, f"nucl_{r}.aligned.fasta")
            if os.path.exists(nexus_path) and os.path.getsize(nexus_path) > 0:
                self.success_list.append((r, nexus_path))
            elif os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0:
                self.success_list.append((r, fasta_path))
            else:
                log.warning(
                    f"Expected alignment file missing or empty for `{r}`: {nexus_path}"
                )


# -----------------------------------------------------------------#


class BackTranslation:
    def __init__(
        self,
        prot_name: str,
        prot_align_file: str,
        nuc_fasta_file: str,
        nuc_align_file: str,
        table: int = 0,
        align_format: str = "fasta",
        gap: str = "-",
    ):
        """
        Coordinate the back-translation of protein sequences to nucleotide sequences.

        Args:
            prot_name: Name of the protein being back-translated
            prot_align_file: Path to the file containing the aligned protein sequences.
            nuc_fasta_file: Path to the file containing the unaligned nucleotide sequences.
            nuc_align_file: Path to the output file for the back-translated nucleotide sequences.
            table: Genetic code table number (default is 0).
            align_format: Format of the alignment file (default is 'fasta').
            gap: String that designates a nucleotide gap (default is '-').
        """
        self.prot_name = prot_name
        self.align_format = align_format
        self.prot_align_file = prot_align_file
        self.nuc_fasta_file = nuc_fasta_file
        self.nuc_align_file = nuc_align_file
        self.table = table
        self.gap = gap

    def _evaluate_nuc(
        self, identifier: str, nuc: Seq | MutableSeq, prot: Seq | MutableSeq
    ) -> Seq | MutableSeq | None:
        """
        Returns nucleotide sequence if works (can remove trailing stop),
        or `None` if the translation does not match the protein.
        """
        if len(nuc) % 3:
            log.debug(
                f"Nucleotide sequence for {identifier} is length {len(nuc)} (not a multiple of three)"
            )

        p = str(prot).upper().replace("*", "X")
        t = str(nuc.translate(self.table)).upper().replace("*", "X")
        if (
            len(t) == len(p) + 1
            and str(nuc)[-3:].upper() in ambiguous_generic_by_id[self.table].stop_codons
        ):
            # Allow this...
            t = t[:-1]
            nuc = nuc[:-3]  # edit return value
        if len(t) != len(p):
            debug = (
                f"Inconsistent lengths for {identifier}, ungapped protein {len(p)}, "
                f"tripled {len(p) * 3} vs ungapped nucleotide {len(nuc)}."
            )
            if t.endswith(p):
                debug += (
                    f"\nThere are {len(t) - len(p)} extra nucleotides at the start."
                )
            elif t.startswith(p):
                debug += f"\nThere are {len(t) - len(p)} extra nucleotides at the end."
            elif p in t:
                debug += (
                    "\nHowever, protein sequence found within translated nucleotides."
                )
            elif p[1:] in t:
                debug += "\nHowever, ignoring first amino acid, protein sequence found within translated nucleotides."
            log.debug(debug)

        if t == p:
            return nuc
        elif p.startswith("M") and "M" + t[1:] == p:
            if (
                str(nuc[0:3]).upper()
                in ambiguous_generic_by_id[self.table].start_codons
            ):
                return nuc
            else:
                log.debug(
                    f"translation for {identifier} would match if {nuc[0:3].upper()} "
                    f"was a start codon (check correct table used)"
                )
                log.warning(f"translation check failed for {identifier}")

        else:
            m = "".join("." if x == y else "!" for (x, y) in zip(p, t))
            if len(prot) < 70:
                log.debug(
                    f"translation mismatch for {identifier} [0:{len(prot)}]\n"
                    f"protein:     {p}\n"
                    f"             {m}\n"
                    f"translation: {t}\n"
                )
            else:
                for offset in range(0, len(p), 60):
                    log.debug(
                        f"translation mismatch for {identifier} [{offset}:{offset + 60}]\n"
                        f"protein:     {p[offset : offset + 60]}\n"
                        f"             {m[offset : offset + 60]}\n"
                        f"translation: {t[offset : offset + 60]}\n"
                    )
            log.warning(f"translation check failed for {identifier}")
        return None

    def _backtrans_seq(
        self,
        identifier: str,
        aligned_protein_record: SeqRecord,
        unaligned_nucleotide_record: SeqRecord,
    ) -> SeqRecord | None:
        protein_seq = aligned_protein_record.seq
        nucleotide_seq = unaligned_nucleotide_record.seq
        if protein_seq is None or nucleotide_seq is None:
            log.warning(
                f"Backtranslation failed for {identifier} due to a missing sequence"
            )
            return None

        # Per https://biopython.org/docs/1.81/api/Bio.Seq.html,
        # `replace` is proper replacement for depreciated method `ungap`.
        ungapped_protein = Seq(str(protein_seq).replace(self.gap, ""))

        ungapped_nucleotide: Seq | MutableSeq | None = nucleotide_seq
        if self.table:
            ungapped_nucleotide = self._evaluate_nuc(
                identifier, nucleotide_seq, ungapped_protein
            )
        elif len(ungapped_protein) * 3 != len(nucleotide_seq):
            log.debug(
                f"Inconsistent lengths for {aligned_protein_record.id}, "
                f"ungapped protein {len(ungapped_protein)}, "
                f"tripled {len(ungapped_protein) * 3} vs "
                f"ungapped nucleotide {len(nucleotide_seq)}"
            )
            log.warning(
                f"Backtranslation failed for {identifier} due to ungapped length mismatch"
            )
        if ungapped_nucleotide is None:
            return None

        seq = []
        nuc = str(ungapped_nucleotide)
        gap_codon = self.gap * 3
        for amino_acid in protein_seq:
            if amino_acid == self.gap:
                seq.append(gap_codon)
            else:
                seq.append(nuc[:3])
                nuc = nuc[3:]
        if len(nuc) > 0:
            log.debug(
                f"Nucleotide sequence for {unaligned_nucleotide_record.id} "
                f"longer than protein {aligned_protein_record.id}"
            )
            log.warning(
                f"Backtranslation failed for {identifier} due to unaligned length mismatch"
            )
            return None

        aligned_nuc = unaligned_nucleotide_record[:]  # copy for most annotation
        aligned_nuc.letter_annotations = {}  # clear this
        aligned_nuc.seq = Seq(
            "".join(seq)
        )  # , alpha)  # Modification on 09-Sep-2022 by M. Gruenstaeudl
        if len(protein_seq) * 3 != len(aligned_nuc):
            log.debug(
                f"Nucleotide sequence for {aligned_nuc.id} is length {len(aligned_nuc)} "
                f"but protein sequence {protein_seq} is length {len(protein_seq)} "
                f"(3 * {len(aligned_nuc)} != {len(protein_seq)})"
            )
            log.warning(
                f"Backtranslation failed for {identifier} due to aligned length mismatch"
            )
            return None

        return aligned_nuc

    def _backtrans_seqs(
        self,
        protein_alignment: MultipleSeqAlignment,
        nucleotide_records: Mapping,
        key_function: Callable[[str], str] = lambda x: x,
    ):
        """
        Thread nucleotide sequences onto a protein alignment.
        """
        aligned = []
        for protein in protein_alignment:
            identifier = self._get_identifier(protein.id)
            try:
                nucleotide = nucleotide_records[key_function(protein.id)]
            except KeyError:
                raise ValueError(f"Could not find nucleotide sequence for {identifier}")
            sequence = self._backtrans_seq(identifier, protein, nucleotide)
            if sequence is not None:
                aligned.append(sequence)
        return MultipleSeqAlignment(aligned)

    def _get_identifier(self, rec_id: str) -> str:
        rec_name = rec_id.replace(self.prot_name + "_", "")
        identifier = f"protein '{self.prot_name}' in {rec_name}"
        return identifier

    def backtranslate(self):
        """
        Perform back-translation of a protein alignment to nucleotides.
        """
        # Step 1. Load the protein alignment
        prot_align = AlignIO.read(self.prot_align_file, self.align_format)
        # Step 2. Index the unaligned nucleotide sequences
        nuc_dict = SeqIO.index(self.nuc_fasta_file, "fasta")
        try:
            # Step 3. Perform back-translation
            nuc_align = self._backtrans_seqs(prot_align, nuc_dict)
        finally:
            nuc_dict.close()
        log.debug(
            f"  back-translation of `{self.prot_name}` complete, "
            f"{len(nuc_align)} sequences of length {nuc_align.get_alignment_length()}"
        )
        # Step 4. Write the back-translated nucleotide alignment to a file
        with open(self.nuc_align_file, "w") as output_handle:
            AlignIO.write(nuc_align, output_handle, self.align_format)
