from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Data.CodonTable import ambiguous_generic_by_id
from Bio.Seq import MutableSeq, Seq
from Bio.SeqRecord import SeqRecord

from .align_runtime import (
    attach_shared_scratch,
    parse_partitions_flag,
    pin_algorithm_args,
    resolve_tuning_plan,
    strategy_for_locus,
    trim_spec_from_params,
    unaligned_bytes_estimate,
    want_subdivide,
    water_fill_preferences,
    work_for_locus,
)
from .alignment_tools import MAFFT, AlignmentTool, get_alignment_tool
from .autotune import TuningPlan
from .chunking import ChunkConfig, plan_chunks_with_reason, splice_alignments
from .ephemeral_engine import ScratchEngine
from .events import ProgressSink
from .helpers import bounded_submit_iterator, resolve_scratch_dir, split_list
from .logging_ops import logger as log
from .nexus_ops import ConcatenationError
from .scheduler import FanOut, LocusScheduler, SchedTask, SchedulerReport, TaskContext
from .seqfeature_ops import PlastidData
from .strategy import AlignmentStrategy, estimate_work
from .supermatrix import LocusSource, SupermatrixBuilder, legacy_concat_rename

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
        self._plan: TuningPlan | None = None
        self._scheduler_report: SchedulerReport | None = None
        self._locus_records: list[dict[str, Any]] = []
        self._supermatrix_info: dict[str, Any] | None = None
        self.sink: ProgressSink | None = user_params.get("progress_sink")

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

        custom_scratch = user_params.get("scratch_dir") if user_params else None
        self._scratch = ScratchEngine(custom_scratch)
        attach_shared_scratch(self.aligner, self._scratch)

    @property
    def _out_dir(self) -> str:
        """The output directory (a required user parameter)."""
        return str(self.user_params["out_dir"])

    @property
    def keep_unaligned(self) -> bool:
        return bool(self.user_params.get("keep_unaligned", True))

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

    def scratch_stats(self) -> dict[str, Any] | None:
        """Scratch accounting for the run manifest (``None`` if engine closed)."""
        try:
            return self._scratch.stats().as_dict()
        except Exception:
            return None

    def close_scratch(self) -> None:
        """Release the shared scratch engine (idempotent)."""
        try:
            self._scratch.cleanup()
        except Exception as exc:
            log.debug(f"scratch cleanup: {exc}")

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
        if not self.keep_unaligned:
            self._unaligned_saved = True
            return
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

    _ALIGNER_OPTIONAL_PARAMS = (
        "threads",
        "sequence_count",
        "max_len",
        "min_len",
        "alphabet",
        "algorithm_args",
        "cpu_ids",
        "env",
        "pin",
    )

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

    def _ensure_plan(
        self, region_summary: Mapping[str, Mapping[str, int]], *, alphabet: str
    ) -> TuningPlan:
        if self._plan is not None:
            return self._plan
        plan = resolve_tuning_plan(self.user_params, region_summary, alphabet=alphabet)
        self._plan = plan
        self.user_params["num_threads"] = plan.cpu_budget
        for warning in plan.warnings:
            log.warning(warning)
        log.info(plan.explain())
        return plan

    def _build_align_kwargs(
        self,
        summary_entry: Mapping[str, int],
        leased_threads: int,
        support: Mapping[str, bool],
        *,
        alphabet: str | None = None,
        algorithm_args: Sequence[str] | None = None,
        cpu_ids: Sequence[int] | None = None,
        env: Mapping[str, str] | None = None,
        pin: bool = False,
    ) -> dict[str, Any]:
        """Dynamically passes only the arguments the aligner accepts."""
        align_kwargs: dict[str, Any] = {}
        if support.get("threads"):
            align_kwargs["threads"] = leased_threads
        seq_count = summary_entry.get("count")
        if support.get("sequence_count") and seq_count is not None:
            align_kwargs["sequence_count"] = seq_count
            max_len = summary_entry.get("max_len")
            if support.get("max_len") and max_len is not None:
                align_kwargs["max_len"] = max_len
            min_len = summary_entry.get("min_len")
            if support.get("min_len") and min_len is not None:
                align_kwargs["min_len"] = min_len
        if alphabet is not None and support.get("alphabet"):
            align_kwargs["alphabet"] = alphabet
        if algorithm_args is not None and support.get("algorithm_args"):
            align_kwargs["algorithm_args"] = list(algorithm_args)
        if cpu_ids is not None and support.get("cpu_ids"):
            align_kwargs["cpu_ids"] = list(cpu_ids)
        if env is not None and support.get("env"):
            align_kwargs["env"] = dict(env)
        if pin and support.get("pin"):
            align_kwargs["pin"] = True
        return align_kwargs

    def _get_worker_scratch(self) -> str:
        """Lazily creates a private scratch directory for exported unaligned FASTA."""
        with self._scratch_lock:
            if self._scratch_dir is None:
                try:
                    base = str(self._scratch.path)
                except Exception:
                    base = resolve_scratch_dir(self.user_params.get("scratch_dir"))
                self._scratch_dir = tempfile.mkdtemp(prefix="pba_unaligned_", dir=base)
            return self._scratch_dir

    def _cleanup_worker_scratch(self) -> None:
        with self._scratch_lock:
            scratch, self._scratch_dir = self._scratch_dir, None
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    def _unaligned_fasta(
        self, region: str, seq_type: str, *, scratch_dir: str | None = None
    ) -> tuple[str, bool]:
        """Returns the unaligned FASTA of a region, exporting it if needed.

        When ``keep_unaligned`` is True, writes/reuses ``out_dir`` files. When
        False, always exports into scratch and marks the path temporary.

        Returns:
            (path, is_temporary); temporary files should be removed after use.
        """
        shared = os.path.join(self._out_dir, f"{seq_type}_{region}.unalign.fasta")
        if os.path.exists(shared) and os.path.getsize(shared) > 0:
            return shared, False
        # Durable copies are created by ``save_unaligned`` (monolithic keep path).
        # Workers and ``keep_unaligned=False`` always export into scratch.
        if self.keep_unaligned and self._unaligned_saved:
            self.plastid_data.store.export_fasta(region, seq_type, shared)
            return shared, False
        base = scratch_dir or self._get_worker_scratch()
        local = os.path.join(base, f"{seq_type}_{region}.unalign.fasta")
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

    def _is_mafft(self) -> bool:
        if isinstance(self.aligner, MAFFT):
            return True
        name = str(self.user_params.get("alignment_tool", "mafft") or "mafft").lower()
        return name == "mafft"

    def _adjust_direction(self) -> bool:
        cfg = self.user_params.get("alignment_tool_config") or {}
        return bool(cfg.get("adjustdirection", False))

    def _record_locus(
        self,
        region: str,
        *,
        alphabet: str,
        strategy: AlignmentStrategy | None,
        chunked: bool = False,
        chunk_reason: str | None = None,
        ok: bool = True,
        error: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "region": region,
            "alphabet": alphabet,
            "ok": ok,
            "chunked": chunked,
        }
        mode = self.user_params.get("select_mode")
        if mode is not None:
            record["mode"] = str(getattr(mode, "value", mode))
        if strategy is not None:
            record["strategy"] = strategy.to_dict()
        if chunk_reason:
            record["chunk_reason"] = chunk_reason
        if error:
            record["error"] = error
        with self._success_lock:
            self._locus_records.append(record)

    def _run_scheduler(
        self, tasks: list[SchedTask], plan: TuningPlan
    ) -> SchedulerReport:
        scheduler = LocusScheduler(
            total_cores=plan.cpu_budget,
            topology=plan.topology,
            use_smt=plan.use_smt,
            memory_budget=plan.task_memory_envelope_bytes,
            scratch=self._scratch,
            sink=self.sink,
            pin=plan.pinning,
            max_workers=plan.workers,
        )
        report = scheduler.run(tasks, phase="align")
        self._scheduler_report = report
        return report

    def _align_fasta(
        self,
        input_file: str,
        output_file: str,
        *,
        summary_entry: Mapping[str, int],
        support: Mapping[str, bool],
        ctx: TaskContext,
        alphabet: str,
        algorithm_args: Sequence[str] | None = None,
    ) -> bool:
        align_kwargs = self._build_align_kwargs(
            summary_entry,
            ctx.threads,
            support,
            alphabet=alphabet,
            algorithm_args=algorithm_args,
            cpu_ids=ctx.cpu_ids,
            env=ctx.subprocess_env(),
            pin=ctx.pinning,
        )
        return bool(self.aligner.align(input_file, output_file, **align_kwargs))

    def _nuc_locus_task(
        self,
        region: str,
        *,
        region_summary: Mapping[str, Mapping[str, int]],
        support: Mapping[str, bool],
        preferred: int,
        plan: TuningPlan,
    ) -> SchedTask:
        summary_entry = dict(region_summary.get(region, {}))
        strategy = strategy_for_locus(summary_entry, alphabet="nucl")
        n_seq = int(summary_entry.get("count", 0) or 0)
        max_len = int(summary_entry.get("max_len", 0) or 0)
        work = work_for_locus(summary_entry, alphabet="nucl")
        bytes_est = unaligned_bytes_estimate(summary_entry)
        cap = max(1, int(plan.threads_per_task_cap))
        pref = max(1, min(int(preferred), cap))
        max_threads = cap if support.get("threads") else 1
        est_mem = plan.est_mem_bytes(
            max(1, n_seq),
            max(1, max_len),
            parttree=strategy.uses_parttree,
            alphabet="nucleotide",
            threads=pref,
        )
        subdivide_mode = str(
            self.user_params.get("subdivide", "auto") or "auto"
        ).lower()

        def run_locus(ctx: TaskContext) -> Any:
            fasta_dir = os.path.join(self._out_dir, "fasta")
            os.makedirs(fasta_dir, exist_ok=True)
            out_aligned = os.path.join(fasta_dir, f"nucl_{region}.aligned.fasta")
            scratch_dir = str(ctx.scratch.path) if ctx.scratch is not None else None
            unalign, is_temp = self._unaligned_fasta(
                region, "nucl", scratch_dir=scratch_dir
            )

            def _cleanup_unalign() -> None:
                if is_temp:
                    self._remove_quietly(unalign)

            try_chunks = want_subdivide(
                mode=subdivide_mode,
                alphabet="nucl",
                n_seq=n_seq,
                max_len=max_len,
                is_mafft=self._is_mafft(),
                adjust_direction=self._adjust_direction(),
                strategy=strategy,
            )
            if try_chunks:
                plan_chunks, reason = plan_chunks_with_reason(
                    unalign,
                    "nucl",
                    ChunkConfig(adjust_direction=self._adjust_direction()),
                    name=region,
                )
                if plan_chunks is not None:
                    pinned = pin_algorithm_args(strategy, n_seq, max_len)
                    chunk_dir = os.path.join(
                        self._get_worker_scratch(),
                        f"{region}_chunks",
                    )
                    os.makedirs(chunk_dir, exist_ok=True)
                    plan_chunks.write_chunk_fastas(chunk_dir)

                    def make_chunk(idx: int) -> SchedTask:
                        spec = plan_chunks.chunks[idx]

                        def chunk_fn(cctx: TaskContext) -> tuple[str, str | None]:
                            if not spec.needs_alignment:
                                return ("skip", None)
                            in_fa = str(plan_chunks.chunk_path(chunk_dir, idx))
                            out_fa = os.path.join(
                                chunk_dir, f"{region}.chunk{idx:02d}.aligned.fasta"
                            )
                            chunk_summary = {
                                "count": spec.n_seq,
                                "max_len": spec.max_len,
                                "min_len": spec.min_len,
                            }
                            try:
                                ok = self._align_fasta(
                                    in_fa,
                                    out_fa,
                                    summary_entry=chunk_summary,
                                    support=support,
                                    ctx=cctx,
                                    alphabet="nucl",
                                    algorithm_args=pinned,
                                )
                            except Exception as exc:
                                return ("fail", str(exc))
                            if not ok or not (
                                os.path.exists(out_fa) and os.path.getsize(out_fa) > 0
                            ):
                                return ("fail", f"chunk {idx} produced no output")
                            return ("ok", out_fa)

                        return SchedTask(
                            task_id=f"c{idx:02d}",
                            fn=chunk_fn,
                            work=max(1.0, work / max(1, plan_chunks.n_chunks)),
                            preferred_threads=1,
                            max_threads=max_threads,
                            n_seq=spec.n_seq,
                            max_len=spec.max_len,
                        )

                    def finalize(fctx: TaskContext, results: list[Any]) -> bool:
                        try:
                            paths: list[str | None] = []
                            failed = False
                            for idx, result in enumerate(results):
                                if result is None:
                                    failed = True
                                    paths.append(None)
                                    continue
                                status, path = result
                                if status == "fail":
                                    failed = True
                                paths.append(path if status == "ok" else None)
                                if (
                                    plan_chunks.chunks[idx].needs_alignment
                                    and status != "ok"
                                ):
                                    failed = True
                            if failed:
                                log.warning(
                                    f"chunked alignment of `{region}` failed; "
                                    "falling back to unchunked"
                                )
                                ok = self._align_fasta(
                                    unalign,
                                    out_aligned,
                                    summary_entry=summary_entry,
                                    support=support,
                                    ctx=fctx,
                                    alphabet="nucl",
                                )
                                self._record_locus(
                                    region,
                                    alphabet="nucl",
                                    strategy=strategy,
                                    chunked=False,
                                    chunk_reason="fallback_after_chunk_failure",
                                    ok=ok,
                                )
                                if ok:
                                    self._record_success(region, out_aligned)
                                return ok
                            splice_alignments(plan_chunks, paths, out_aligned)
                            self._record_locus(
                                region,
                                alphabet="nucl",
                                strategy=strategy,
                                chunked=True,
                                chunk_reason=reason,
                                ok=True,
                            )
                            self._record_success(region, out_aligned)
                            return True
                        finally:
                            _cleanup_unalign()
                            shutil.rmtree(chunk_dir, ignore_errors=True)

                    return FanOut(
                        [make_chunk(i) for i in range(plan_chunks.n_chunks)],
                        finalizer=finalize,
                    )
                chunk_reason = reason
            else:
                chunk_reason = "subdivide_skipped"

            try:
                ok = self._align_fasta(
                    unalign,
                    out_aligned,
                    summary_entry=summary_entry,
                    support=support,
                    ctx=ctx,
                    alphabet="nucl",
                )
                if (
                    ok
                    and os.path.exists(out_aligned)
                    and os.path.getsize(out_aligned) > 0
                ):
                    self._record_success(region, out_aligned)
                    self._record_locus(
                        region,
                        alphabet="nucl",
                        strategy=strategy,
                        chunked=False,
                        chunk_reason=chunk_reason,
                        ok=True,
                    )
                    return True
                self._record_locus(
                    region,
                    alphabet="nucl",
                    strategy=strategy,
                    ok=False,
                    error="no output",
                )
                log.warning(
                    f"Alignment for `{region}` produced no output: {out_aligned}"
                )
                return False
            finally:
                _cleanup_unalign()

        return SchedTask(
            task_id=region,
            fn=run_locus,
            work=work,
            est_mem_bytes=est_mem,
            preferred_threads=pref,
            max_threads=max_threads,
            scratch_bytes=max(1, 3 * bytes_est),
            n_seq=n_seq,
            max_len=max_len,
            meta={"strategy": strategy.to_dict()},
        )

    def _nuc_MSA(self, target_regions: set[str] | None = None):
        """Aligns nucleotide regions (all of them, or only `target_regions`)."""
        region_summary = self._get_region_summary("nucl")
        if len(self.plastid_data.nucleotides) == 0:
            log.critical("No items in nucleotide main dictionary to process")
            raise RuntimeError("No items in nucleotide main dictionary to process")

        plan = self._ensure_plan(region_summary, alphabet="nucl")
        support = self._aligner_support()
        regions = self._regions_to_align(target_regions)
        preferred = water_fill_preferences(region_summary, regions, plan.cpu_budget)
        log.info(
            "conducting multiple sequence alignments based on nucleotide sequence"
            f" data using {plan.cpu_budget} threads ({plan.workers} workers)"
        )
        tasks = [
            self._nuc_locus_task(
                k,
                region_summary=region_summary,
                support=support,
                preferred=preferred.get(k, 1),
                plan=plan,
            )
            for k in regions
        ]
        try:
            self._run_scheduler(tasks, plan)
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

    def _prot_locus_task(
        self,
        region: str,
        *,
        region_summary: Mapping[str, Mapping[str, int]],
        support: Mapping[str, bool],
        preferred: int,
        plan: TuningPlan,
    ) -> SchedTask:
        summary_entry = dict(region_summary.get(region, {}))
        strategy = strategy_for_locus(summary_entry, alphabet="prot")
        n_seq = int(summary_entry.get("count", 0) or 0)
        max_len = int(summary_entry.get("max_len", 0) or 0)
        work = work_for_locus(summary_entry, alphabet="prot")
        bytes_est = unaligned_bytes_estimate(summary_entry)
        cap = max(1, int(plan.threads_per_task_cap))
        pref = max(1, min(int(preferred), cap))
        max_threads = cap if support.get("threads") else 1
        est_mem = plan.est_mem_bytes(
            max(1, n_seq),
            max(1, max_len),
            parttree=strategy.uses_parttree,
            alphabet="protein",
            threads=pref,
        )
        subdivide_mode = str(
            self.user_params.get("subdivide", "auto") or "auto"
        ).lower()

        def run_locus(ctx: TaskContext) -> Any:
            out_dir = self._out_dir
            fasta_dir = os.path.join(out_dir, "fasta")
            os.makedirs(fasta_dir, exist_ok=True)
            out_aligned_nucl = os.path.join(fasta_dir, f"nucl_{region}.aligned.fasta")
            scratch_dir = str(ctx.scratch.path) if ctx.scratch is not None else None
            if self.keep_unaligned:
                out_aligned_prot = os.path.join(out_dir, f"prot_{region}.aligned.fasta")
                prot_is_temp = False
            else:
                base = scratch_dir or self._get_worker_scratch()
                out_aligned_prot = os.path.join(base, f"prot_{region}.aligned.fasta")
                prot_is_temp = True

            unalign_prot, prot_unalign_temp = self._unaligned_fasta(
                region, "prot", scratch_dir=scratch_dir
            )

            def _cleanup_prot_inputs() -> None:
                if prot_unalign_temp:
                    self._remove_quietly(unalign_prot)
                if prot_is_temp:
                    self._remove_quietly(out_aligned_prot)

            try_chunks = want_subdivide(
                mode=subdivide_mode,
                alphabet="prot",
                n_seq=n_seq,
                max_len=max_len,
                is_mafft=self._is_mafft(),
                adjust_direction=False,
                strategy=strategy,
            )
            chunk_reason = "subdivide_skipped"
            if try_chunks:
                plan_chunks, reason = plan_chunks_with_reason(
                    unalign_prot,
                    "prot",
                    ChunkConfig(),
                    name=region,
                )
                if plan_chunks is not None:
                    pinned = pin_algorithm_args(strategy, n_seq, max_len)
                    chunk_dir = os.path.join(
                        self._get_worker_scratch(),
                        f"{region}_prot_chunks",
                    )
                    os.makedirs(chunk_dir, exist_ok=True)
                    plan_chunks.write_chunk_fastas(chunk_dir)

                    def make_chunk(idx: int) -> SchedTask:
                        spec = plan_chunks.chunks[idx]

                        def chunk_fn(cctx: TaskContext) -> tuple[str, str | None]:
                            if not spec.needs_alignment:
                                return ("skip", None)
                            in_fa = str(plan_chunks.chunk_path(chunk_dir, idx))
                            out_fa = os.path.join(
                                chunk_dir, f"{region}.chunk{idx:02d}.aligned.fasta"
                            )
                            chunk_summary = {
                                "count": spec.n_seq,
                                "max_len": spec.max_len,
                                "min_len": spec.min_len,
                            }
                            try:
                                ok = self._align_fasta(
                                    in_fa,
                                    out_fa,
                                    summary_entry=chunk_summary,
                                    support=support,
                                    ctx=cctx,
                                    alphabet="prot",
                                    algorithm_args=pinned,
                                )
                            except Exception as exc:
                                return ("fail", str(exc))
                            if not ok or not (
                                os.path.exists(out_fa) and os.path.getsize(out_fa) > 0
                            ):
                                return ("fail", f"chunk {idx} produced no output")
                            return ("ok", out_fa)

                        return SchedTask(
                            task_id=f"c{idx:02d}",
                            fn=chunk_fn,
                            work=max(1.0, work / max(1, plan_chunks.n_chunks)),
                            preferred_threads=1,
                            max_threads=max_threads,
                            n_seq=spec.n_seq,
                            max_len=spec.max_len,
                        )

                    def finalize(fctx: TaskContext, results: list[Any]) -> bool:
                        try:
                            paths: list[str | None] = []
                            failed = False
                            for idx, result in enumerate(results):
                                if result is None:
                                    failed = True
                                    paths.append(None)
                                    continue
                                status, path = result
                                if status == "fail":
                                    failed = True
                                paths.append(path if status == "ok" else None)
                                if (
                                    plan_chunks.chunks[idx].needs_alignment
                                    and status != "ok"
                                ):
                                    failed = True
                            if failed:
                                log.warning(
                                    f"chunked protein alignment of `{region}` failed; "
                                    "falling back to unchunked"
                                )
                                ok = self._align_fasta(
                                    unalign_prot,
                                    out_aligned_prot,
                                    summary_entry=summary_entry,
                                    support=support,
                                    ctx=fctx,
                                    alphabet="prot",
                                )
                            else:
                                splice_alignments(plan_chunks, paths, out_aligned_prot)
                                ok = True
                            return self._finish_protein_locus(
                                region,
                                out_aligned_prot,
                                out_aligned_nucl,
                                scratch_dir=scratch_dir,
                                strategy=strategy,
                                chunked=not failed and ok,
                                chunk_reason=reason if not failed else "fallback",
                                ok_align=ok,
                            )
                        finally:
                            _cleanup_prot_inputs()

                    return FanOut(
                        [make_chunk(i) for i in range(plan_chunks.n_chunks)],
                        finalizer=finalize,
                    )
                chunk_reason = reason

            try:
                aligned_ok = self._align_fasta(
                    unalign_prot,
                    out_aligned_prot,
                    summary_entry=summary_entry,
                    support=support,
                    ctx=ctx,
                    alphabet="prot",
                )
                return self._finish_protein_locus(
                    region,
                    out_aligned_prot,
                    out_aligned_nucl,
                    scratch_dir=scratch_dir,
                    strategy=strategy,
                    chunked=False,
                    chunk_reason=chunk_reason,
                    ok_align=aligned_ok,
                )
            finally:
                _cleanup_prot_inputs()

        return SchedTask(
            task_id=region,
            fn=run_locus,
            work=work,
            est_mem_bytes=est_mem,
            preferred_threads=pref,
            max_threads=max_threads,
            scratch_bytes=max(1, 3 * bytes_est),
            n_seq=n_seq,
            max_len=max_len,
            meta={"strategy": strategy.to_dict()},
        )

    def _finish_protein_locus(
        self,
        region: str,
        aligned_prot: str,
        aligned_nucl: str,
        *,
        scratch_dir: str | None,
        strategy: AlignmentStrategy,
        chunked: bool,
        chunk_reason: str | None,
        ok_align: bool,
    ) -> bool:
        if not (
            ok_align
            and os.path.exists(aligned_prot)
            and os.path.getsize(aligned_prot) > 0
        ):
            log.warning(
                f"Protein alignment for `{region}` produced no output: {aligned_prot}"
            )
            self._record_locus(
                region,
                alphabet="prot",
                strategy=strategy,
                chunked=chunked,
                chunk_reason=chunk_reason,
                ok=False,
                error="no protein output",
            )
            return False
        unalign_nucl, nucl_temp = self._unaligned_fasta(
            region, "nucl", scratch_dir=scratch_dir
        )
        try:
            translator = BackTranslation(
                region, aligned_prot, unalign_nucl, aligned_nucl, 11
            )
            translator.backtranslate()
            if not (os.path.exists(aligned_nucl) and os.path.getsize(aligned_nucl) > 0):
                log.warning(
                    f"Back-translation produced no output for `{region}`: {aligned_nucl}"
                )
                self._record_locus(
                    region,
                    alphabet="prot",
                    strategy=strategy,
                    chunked=chunked,
                    ok=False,
                    error="backtranslate empty",
                )
                return False
            self._record_success(region, aligned_nucl)
            self._record_locus(
                region,
                alphabet="prot",
                strategy=strategy,
                chunked=chunked,
                chunk_reason=chunk_reason,
                ok=True,
            )
            return True
        except Exception as err:
            log.warning(f"Unable to conduct back-translation of `{region}`: {err}")
            self._record_locus(
                region,
                alphabet="prot",
                strategy=strategy,
                chunked=chunked,
                ok=False,
                error=str(err),
            )
            return False
        finally:
            if nucl_temp:
                self._remove_quietly(unalign_nucl)

    def _prot_MSA(self, target_regions: set[str] | None = None):
        """Aligns proteins and back-translates them (all regions, or `target_regions`)."""
        region_summary = self._get_region_summary("prot")
        if len(self.plastid_data.nucleotides) == 0:
            log.critical("No items in nucleotide main dictionary to process")
            raise RuntimeError("No items in nucleotide main dictionary to process")

        plan = self._ensure_plan(region_summary, alphabet="prot")
        support = self._aligner_support()
        regions = self._regions_to_align(target_regions)
        preferred = water_fill_preferences(region_summary, regions, plan.cpu_budget)
        log.info(
            " conducting multiple sequence alignments based on protein sequence"
            " data, followed by back-translation to nucleotides using"
            f" {plan.cpu_budget} threads ({plan.workers} workers)"
        )
        tasks = [
            self._prot_locus_task(
                k,
                region_summary=region_summary,
                support=support,
                preferred=preferred.get(k, 1),
                plan=plan,
            )
            for k in regions
        ]
        try:
            self._run_scheduler(tasks, plan)
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
        """Concatenate successful alignments via :class:`SupermatrixBuilder`."""
        num_successful = len(self.success_list)
        if num_successful == 0:
            log.warning("No successful alignments to concatenate.")
            return

        fasta_dir = os.path.join(self._out_dir, "fasta")
        log.info(
            f"concatenating all successful alignments in `{self.plastid_data.order}` order"
        )
        self.plastid_data.set_order_map()
        self.success_list.sort(key=lambda t: self.plastid_data.order_map[t[0]])

        mode = self.mode
        kind = str(getattr(mode, "value", mode) or "nucl").lower()
        is_cds = bool(mode and getattr(mode, "collects_proteins", lambda: False)())
        sources: list[LocusSource] = []
        for name, path in self.success_list:
            fasta_path = path
            if path.endswith(".nexus"):
                candidate = os.path.join(fasta_dir, f"nucl_{name}.aligned.fasta")
                if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                    fasta_path = candidate
            if not (os.path.exists(fasta_path) and os.path.getsize(fasta_path) > 0):
                log.warning(f"Alignment file missing or empty for `{name}`: {path}")
                continue
            sources.append(
                LocusSource(
                    name=name,
                    path=fasta_path,
                    kind=kind,
                    codon_aligned=is_cds,
                )
            )
        if not sources:
            raise ConcatenationError("no alignment could be loaded")

        raw_threads = self.user_params.get("num_threads", 4) or 4
        workers = min(max(1, int(raw_threads)), 8)
        trim = trim_spec_from_params(self.user_params)
        # Legacy byte parity when trimming is off: leave case as written by MAFFT.
        normalize = trim is not None
        try:
            builder = SupermatrixBuilder(
                sources,
                rename=legacy_concat_rename,
                trim=trim,
                normalize_case=normalize,
                max_workers=workers,
                scratch_dir=str(self._scratch.path),
                mem_budget_bytes=(
                    self._plan.task_memory_envelope_bytes if self._plan else None
                ),
            )
            matrix = builder.build()
            prefix = os.path.join(self._out_dir, f"nucl_{len(sources)}concat")
            partitions = parse_partitions_flag(self.user_params.get("partitions"))
            codon = bool(self.user_params.get("codon_partitions", False)) and is_cds
            written = matrix.write_all(
                prefix,
                phylip=bool(self.user_params.get("phylip", False)),
                partitions=partitions,
                codon_partitions=codon,
            )
            self._supermatrix_info = {
                "ntax": matrix.ntax,
                "nchar": matrix.nchar,
                "nloci": len(sources),
                "written": {k: str(v) for k, v in written.items()},
                "trim": trim,
                "partitions": list(partitions),
                "codon_partitions": codon,
            }
        except Exception as e:
            log.critical(f"Unable to concatenate alignments: {e}")
            raise ConcatenationError(f"Alignment concatenation failed: {e}") from e

    def _get_lpt_sorted_regions(self, prefix: str = "nucl") -> list[str]:
        """Sort region keys in descending order of computational work (LPT first).

        Uses :func:`estimate_work` from the database summary when available,
        falling back to unaligned file size. Ties break deterministically by name.
        """
        out_dir = self.user_params.get("out_dir", "")
        keys = list(self.plastid_data.nucleotides.keys())
        alphabet = "prot" if prefix == "prot" else "nucl"

        try:
            summary = self.plastid_data.store.get_region_summary(seq_type=prefix)
        except Exception:
            summary = {}

        def task_weight(k: str) -> tuple[float, str]:
            if (
                isinstance(summary, dict)
                and k in summary
                and isinstance(summary[k], dict)
            ):
                count = summary[k].get("count", 0) or 0
                max_len = summary[k].get("max_len", 0) or 0
                if count > 0 and max_len > 0:
                    return (-float(estimate_work(count, max_len, alphabet=alphabet)), k)

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
