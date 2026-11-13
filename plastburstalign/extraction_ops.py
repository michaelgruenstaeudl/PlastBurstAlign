import bisect
import os
from collections.abc import Generator, Iterator
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
from typing import Any, NamedTuple

from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from .extraction_report import ExtractionError, ExtractionReport
from .genbank_scanner import GenBankParseError, fast_read_genbank
from .helpers import ordered_bounded_map, split_list
from .logging_ops import Logger
from .logging_ops import logger as log

# Package imports
from .seqfeature_ops import (
    GeneFeature,
    IntergenicDict,
    IntergenicFeature,
    IntronFeature,
    PlastidData,
    PlastidDict,
    PlastidFeature,
    ProteinFeature,
)
from .task_queue import get_mp_context

# Flattened extraction results: region -> [(record id, description, sequence, file index)]
FlatRecords = dict[str, list[tuple[str, str, str, int]]]


def _mode_key(mode_str: object) -> str:
    """``SelectMode.CDS`` and ``cds`` both become ``cds``."""
    return str(mode_str).rsplit(".", 1)[-1].lower()


def _feature_label(feature: object) -> str:
    """Short, single-line description of a feature for the report."""
    if isinstance(feature, str):
        return feature
    try:
        ftype = getattr(feature, "type", "?")
        quals = getattr(feature, "qualifiers", {}) or {}
        gene_vals = quals.get("gene") if isinstance(quals, dict) else None
        gene = gene_vals[0] if gene_vals else ""
        loc = getattr(feature, "location", "")
        return f"{ftype} {gene} {loc}".strip()
    except Exception:
        return "feature"


def _has_regions(nuc: Any, prot: Any) -> bool:
    return bool(nuc) or bool(prot)


def _shutdown_pool(executor: ProcessPoolExecutor, *, broken: bool) -> None:
    """Stop a pool without waiting forever on a worker that already died.

    ``shutdown(wait=True)`` joins the pool's management thread. After a worker
    is killed that join can block (the thread is stuck terminating children),
    and the thread is not a daemon, so the interpreter then never exits.
    Broken pools are therefore cancelled, their children are killed, and the
    management thread is given a short grace period.
    """
    processes = list((getattr(executor, "_processes", None) or {}).values())
    thread = getattr(executor, "_executor_manager_thread", None)
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        log.debug("extraction pool shutdown failed", exc_info=True)
    if broken:
        for proc in processes:
            if not proc.is_alive():
                continue
            try:
                proc.kill()
            except Exception:
                pass
        for proc in processes:
            try:
                proc.join(timeout=0.5)
            except Exception:
                pass
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0 if broken else 120.0)


def _note_scan(report: ExtractionReport, file: str, record: SeqRecord) -> None:
    """Copy scanner skip/truncation notes onto the extraction report."""
    ann = record.annotations or {}
    raw = ann.get("pba_skipped_features") or 0
    try:
        n_skipped = int(raw)
    except (TypeError, ValueError):
        n_skipped = 0
    if n_skipped:
        details = str(ann.get("pba_skipped_feature_details") or "")
        lines = [ln for ln in details.splitlines() if ln]
        for line in lines:
            feature, sep, reason = line.partition(": ")
            if not sep:
                feature, reason = "feature", line
            report.add_feature_failure(file, "scan", feature, reason)
        report.add_unlisted_feature_failures("scan", n_skipped - len(lines))
        log.warning(
            "%s: skipped %d feature(s) that could not be parsed",
            os.path.basename(file),
            n_skipped,
        )
    extra = ann.get("pba_extra_records") or 0
    if extra:
        report.add_warning(file, f"{extra} additional record(s) were ignored")
    truncated = ann.get("pba_truncated_qualifiers") or 0
    if truncated:
        report.add_warning(
            file,
            f"{truncated} qualifier value(s) were truncated to the size limit",
        )


def _unpack_worker_result(
    res: Any,
) -> tuple[Any, Any, dict[str, str], ExtractionReport | None]:
    """Accept 2-tuples, 3-tuples, and report-bearing results.

    Existing callers unpack ``nuc, prot, inv = result``. A :class:`ChunkResult`
    still does that (``len`` is 3) and carries ``.report``. A plain 4-tuple whose
    last item is an :class:`ExtractionReport` is accepted too.
    """
    report = getattr(res, "report", None)
    n = len(res)
    if n == 2:
        nuc, prot = res
        inv: dict[str, str] = {}
    elif n == 3:
        nuc, prot, inv = res
    elif n >= 4:
        nuc, prot, inv = res[0], res[1], res[2]
        if report is None and isinstance(res[3], ExtractionReport):
            report = res[3]
    else:
        raise TypeError(
            f"extraction worker returned {n} value(s); expected 2, 3, or a"
            " report-bearing result"
        )
    return nuc, prot, inv, report


class ChunkResult(tuple):
    """``(nucleotides, proteins, inverse_name_map)`` plus an extraction report.

    ``nuc, prot, inv = result`` and ``len(result) == 3`` stay valid. ``report``
    is pickled across the process boundary with the three flattened dictionaries.
    """

    report: ExtractionReport

    def __new__(
        cls,
        nuc: FlatRecords | None,
        prot: FlatRecords | None,
        inv: dict[str, str],
        report: ExtractionReport,
    ) -> "ChunkResult":
        obj = tuple.__new__(cls, (nuc, prot, inv))
        obj.report = report
        return obj

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (ChunkResult, (self[0], self[1], self[2], self.report))


class ExtractAndCollect:
    def __init__(self, plastid_data: PlastidData, user_params: dict[str, Any]):
        """
        Coordinates the parsing of GenBank flatfiles, the extraction of the contained sequence annotations,
        and the collection of the resulting sequence records in a data structure.

        Args:
            plastid_data: Contains the locations of the files to be parsed,
                and where the extracted records will be stored.
            user_params: Specifications for how the extraction should be performed.
                These are `num_threads`, `out_dir`, `verbose`, `exclude_fullcds`,
                `include_trna`, and `include_rrna`.
        """
        self.plastid_data = plastid_data
        self.num_threads = user_params.get("num_threads")
        self.out_dir = user_params.get("out_dir")
        self.verbose = user_params.get("verbose")
        self.exclude_cds = user_params.get("exclude_fullcds")
        self.include_trna = user_params.get("include_trna", True)
        self.include_rrna = user_params.get("include_rrna", True)
        self.exclude_region = set(user_params.get("exclude_region") or [])
        min_len = user_params.get("min_seq_length")
        self.min_seq_length = min_len if min_len is not None else 3
        self._set_extract_fun()

    def _is_region_excluded(self, *names: str | None) -> bool:
        if not self.exclude_region:
            return False
        return any(name in self.exclude_region for name in names if name)

    def _set_extract_fun(self):
        if self.plastid_data.mode == "cds":
            self._extract_fun = self._extract_cds
        elif self.plastid_data.mode == "igs":
            self._extract_fun = self._extract_igs
        elif self.plastid_data.mode == "int":
            self._extract_fun = self._extract_int
        elif self.plastid_data.mode == "rna":
            self._extract_fun = self._extract_rna
        else:
            log.critical("Undefined extraction mode")
            raise ValueError

    def extract(self):
        """
        Parses all GenBank flatfiles of a given folder and extracts
        all sequence annotations of the type specified by the user for each flatfile.

        The result is independent of worker completion order: every record carries
        the position of its source file in `plastid_data.files` (its `file_index`),
        and the chunk results are committed to the store strictly in submission
        order through a bounded reorder window.

        A feature that cannot be read is skipped. A file that cannot be read is
        skipped. A worker that dies (or raises) takes only its chunk down; that
        chunk is re-run one file at a time and healthy chunks are kept. The run
        is fatal only when no file yields a region. What was skipped is recorded
        on ``plastid_data.extraction_report``.
        """
        log.info(
            f"parsing GenBank flatfiles and extracting their sequence annotations using "
            f"{self.num_threads} processes"
        )

        if not self.plastid_data.files:
            log.critical("No input files found for extraction")
            raise FileNotFoundError("No input files found for extraction")

        # Resolve mode primitives once upfront so they are available regardless of ordering
        mode_str = getattr(self.plastid_data.mode, "mode", str(self.plastid_data.mode))
        collects_proteins = self.plastid_data.mode.collects_proteins()
        report = self.plastid_data.extraction_report

        def commit(res) -> None:
            nuc_dict, prot_dict, inv_map, chunk_report = _unpack_worker_result(res)
            self.plastid_data.add_nucleotides(nuc_dict, inv_map=inv_map)
            self.plastid_data.add_proteins(prot_dict)
            if chunk_report is not None:
                report.merge(chunk_report)

        # Position of the first not-yet-processed file in the ordered file list
        next_file_index = 0

        # Step 0. Extract first genome in list for feature ordering
        if self.plastid_data.order == "seq":
            first_genome = self.plastid_data.files.pop(0)
            try:
                commit(
                    self._extract_recs_static(
                        [first_genome],
                        mode_str,
                        collects_proteins,
                        self.exclude_cds,
                        self.include_trna,
                        self.include_rrna,
                        self.min_seq_length,
                        self.verbose,
                        next_file_index,
                    )
                )
            except Exception as exc:
                report.files_total += 1
                report.add_file_failure(
                    first_genome,
                    "worker",
                    f"{type(exc).__name__}: {exc}",
                    next_file_index,
                )
                log.error(
                    " {!r} generated an exception: {}".format(
                        os.path.basename(first_genome), exc
                    )
                )
            next_file_index += 1

        # Step 1. Chunk remaining files for parallel workers
        # Target 25-40 genomes per chunk to balance IPC serialization and SQLite commit size
        total_files = len(self.plastid_data.files)
        target_chunk_size = max(
            15, min(40, total_files // (self.num_threads * 4) or 15)
        )
        num_chunks = max(
            self.num_threads * 2,
            (total_files + target_chunk_size - 1) // target_chunk_size,
        )
        file_lists = split_list(self.plastid_data.files, num_chunks)

        # Chunks are contiguous slices of the file list; remember where each starts
        chunk_args: list[tuple] = []
        for chunk in file_lists:
            if chunk:
                chunk_args.append(
                    (
                        chunk,
                        mode_str,
                        collects_proteins,
                        self.exclude_cds,
                        self.include_trna,
                        self.include_rrna,
                        self.min_seq_length,
                        self.verbose,
                        next_file_index,
                    )
                )
            next_file_index += len(chunk)

        # Step 2. Parallel extraction. A broken worker is isolated to its chunk;
        # that chunk is then re-run one file at a time. Commit order stays the
        # submission order, including across a pool restart.
        if chunk_args:
            self._extract_chunks(chunk_args, commit)

        if not report.ok:
            log.warning(report.summary())

        # Step 3. Stop execution if no nucleotides were extracted (O(1) DB query)
        if len(self.plastid_data.nucleotides) == 0:
            message = self._fatal_message()
            log.critical(message)
            raise ExtractionError(message, report)

    def _fatal_message(self) -> str:
        report = self.plastid_data.extraction_report
        mode = getattr(self.plastid_data.mode, "value", None) or str(
            self.plastid_data.mode
        )
        return (
            f"No items in main dictionary: {self.out_dir}\n"
            f"No usable records were extracted in mode {mode} from "
            f"{report.files_total} file(s) "
            f"({report.files_failed} failed, "
            f"{report.features_skipped} feature(s) skipped). "
            "Nothing was written because every file failed or contained no "
            f"features for mode {mode}. Check that the inputs are GenBank "
            "flatfiles and that they annotate the features this mode reads "
            "(CDS, gene, tRNA, rRNA).\n"
            f"{report.summary()}"
        )

    def _extract_chunks(self, chunk_args: list[tuple], commit) -> None:
        """Run chunks, restarting the pool after a worker crash.

        Chunks that already committed stay committed. The chunk whose worker
        died is re-run one file at a time; the chunks after it are submitted
        to a new pool.
        """
        pending = list(chunk_args)
        # One restart per chunk is enough; the guard stops a pool that can
        # never make progress from looping.
        for _ in range(len(chunk_args) + 1):
            if not pending:
                return
            done = self._run_chunk_pool(pending, commit)
            if done <= 0:
                self._isolate_chunk(
                    pending[0],
                    commit,
                    RuntimeError("worker pool failed before returning a result"),
                    use_subprocess=True,
                )
                pending = pending[1:]
                continue
            pending = pending[done:]
        for item in pending:
            self._isolate_chunk(
                item,
                commit,
                RuntimeError("extraction stopped making progress"),
                use_subprocess=False,
            )

    def _run_chunk_pool(self, chunk_args: list[tuple], commit) -> int:
        """Commit ``chunk_args`` in order. Returns how many leading chunks finished.

        On :class:`BrokenProcessPool` the chunk that raised is isolated and
        counted as finished; the caller retries everything after it on a new
        pool. A normal exception from the worker function isolates that chunk
        and the same pool keeps going.
        """
        ex = ProcessPoolExecutor(
            max_workers=self.num_threads, mp_context=get_mp_context()
        )
        finished = 0
        broken = False
        try:
            try:
                for item, fut in ordered_bounded_map(
                    ex,
                    self._worker_wrapper,
                    chunk_args,
                    window=max(1, self.num_threads * 2),
                ):
                    try:
                        res = fut.result()
                    except BrokenProcessPool as exc:
                        broken = True
                        self._isolate_chunk(item, commit, exc, use_subprocess=True)
                        finished += 1
                        break
                    except Exception as exc:
                        self._isolate_chunk(item, commit, exc, use_subprocess=False)
                    else:
                        commit(res)
                    finished += 1
            except BrokenProcessPool as exc:
                # The pool died while submitting or while the generator was
                # cleaning up. Isolate the chunk we had not yet finished.
                if not broken and finished < len(chunk_args):
                    self._isolate_chunk(
                        chunk_args[finished], commit, exc, use_subprocess=True
                    )
                    finished += 1
            except Exception as exc:
                # Generator cleanup after a crash is not another failed chunk.
                if broken:
                    log.debug(
                        "extraction pool cleanup failed after a worker crash: %s",
                        exc,
                    )
                elif finished < len(chunk_args):
                    self._isolate_chunk(
                        chunk_args[finished], commit, exc, use_subprocess=False
                    )
                    finished += 1
        finally:
            _shutdown_pool(ex, broken=broken)
        return finished

    def _isolate_chunk(
        self,
        args: tuple,
        commit,
        exc: BaseException,
        *,
        use_subprocess: bool,
    ) -> None:
        """Re-run one chunk's files one at a time. Healthy files are kept."""
        files = list(args[0])
        start = int(args[-1])
        report = self.plastid_data.extraction_report
        report.chunks_retried += 1
        if isinstance(exc, BrokenProcessPool):
            report.worker_crashes += 1
        first = os.path.basename(files[0]) if files else "?"
        log.warning(
            "extraction worker failed on a chunk of %d file(s) starting with %s"
            " (%s: %s); re-running those files one at a time",
            len(files),
            first,
            type(exc).__name__,
            exc,
        )
        for i, path in enumerate(files):
            one = ([path], *args[1:-1], start + i)
            try:
                if use_subprocess:
                    res = self._one_file_process(one)
                else:
                    res = self._extract_recs_static(*one)
            except BrokenProcessPool as file_exc:
                report.worker_crashes += 1
                report.files_total += 1
                report.add_file_failure(
                    path,
                    "worker",
                    f"worker process died: {file_exc}",
                    start + i,
                )
                log.error(
                    " {!r} generated an exception: {}".format(
                        os.path.basename(path), file_exc
                    )
                )
            except Exception as file_exc:
                report.files_total += 1
                report.add_file_failure(
                    path,
                    "worker",
                    f"{type(file_exc).__name__}: {file_exc}",
                    start + i,
                )
                log.error(
                    " {!r} generated an exception: {}".format(
                        os.path.basename(path), file_exc
                    )
                )
            else:
                commit(res)

    def _one_file_process(self, args: tuple) -> Any:
        """Run one file in its own process so a crash cannot kill the parent."""
        ex = ProcessPoolExecutor(max_workers=1, mp_context=get_mp_context())
        broken = False
        try:
            future = ex.submit(self._worker_wrapper, args)
            try:
                return future.result()
            except BrokenProcessPool:
                broken = True
                raise
        finally:
            _shutdown_pool(ex, broken=broken)

    @staticmethod
    def _extract_recs_static(
        files: list[str],
        mode_str: str,
        collects_proteins: bool,
        exclude_cds: list[str] | None,
        include_trna: bool,
        include_rrna: bool,
        min_seq_length: int,
        verbose: bool,
        start_index: int = 0,
    ) -> ChunkResult:
        """Static worker function isolated from sqlite connections.

        The return value unpacks as ``(nucleotides, proteins, inverse_name_map)``.
        It also carries an :class:`ExtractionReport` for this chunk.

        Args:
            files: GenBank files of this chunk, a contiguous slice of the ordered
                input file list.
            start_index: Position of `files[0]` in the ordered input file list; the
                i-th file's records are tagged with `file_index = start_index + i`
                (fourth tuple element) so the store can order rows deterministically.
        """

        class MockMode:
            def __init__(self, mode: str, collects: bool):
                self.mode = mode
                self._collects = collects

            def collects_proteins(self) -> bool:
                return self._collects

            def __str__(self):
                return self.mode

        class MockData:
            def __init__(self):
                self.mode = MockMode(mode_str, collects_proteins)

        mode_key = _mode_key(mode_str)
        report = ExtractionReport()

        def extract_rec(
            file: str,
            n_dict: PlastidDict,
            p_dict: PlastidDict | None,
            file_index: int,
        ) -> None:
            report.files_total += 1
            try:
                log.info(f" parsing {os.path.basename(file)}")
                record = fast_read_genbank(file)
            except GenBankParseError as exc:
                report.add_file_failure(file, exc.stage, exc.reason, file_index)
                log.error(
                    " {!r} generated an exception: {}".format(
                        os.path.basename(file), exc
                    )
                )
                return
            except Exception as exc:
                stage = "read" if isinstance(exc, OSError) else "parse"
                report.add_file_failure(
                    file, stage, f"{type(exc).__name__}: {exc}", file_index
                )
                log.error(
                    " {!r} generated an exception: {}".format(
                        os.path.basename(file), exc
                    )
                )
                return

            _note_scan(report, file, record)
            extractor = ExtractAndCollect.__new__(ExtractAndCollect)
            extractor.plastid_data = MockData()
            extractor.exclude_cds = exclude_cds or []
            extractor.exclude_region = []
            extractor.include_trna = include_trna
            extractor.include_rrna = include_rrna
            extractor.min_seq_length = min_seq_length
            extractor._report = report
            extractor._source_file = file
            try:
                if mode_key == "cds":
                    extractor._extract_cds(record, n_dict, p_dict)
                elif mode_key == "igs":
                    extractor._extract_igs(record, n_dict)
                elif mode_key == "int":
                    extractor._extract_int(record, n_dict)
                elif mode_key == "rna":
                    extractor._extract_rna(record, n_dict)
                else:
                    report.add_file_failure(
                        file,
                        "extract",
                        f"undefined extraction mode {mode_str!r}",
                        file_index,
                    )
                    return
            except Exception as exc:
                # Features already added to the dictionaries are kept.
                report.add_file_failure(
                    file, "extract", f"{type(exc).__name__}: {exc}", file_index
                )
                log.error(
                    " {!r} generated an exception: {}".format(
                        os.path.basename(file), exc
                    )
                )
                if _has_regions(n_dict, p_dict):
                    report.files_with_regions += 1
                return

            if _has_regions(n_dict, p_dict):
                report.files_ok += 1
                report.files_with_regions += 1
            else:
                report.files_ok += 1
                report.add_warning(
                    file, "parsed, but no regions were extracted for this mode"
                )

        Logger.reinitialize_logger(verbose)
        is_igs = mode_key == "igs"

        # Flatten records into lightweight primitive tuples to eliminate IPC pickling overhead
        def _flatten_into(
            flat: FlatRecords, pdict: PlastidDict | None, file_index: int
        ) -> None:
            if not pdict:
                return
            for k, recs in pdict.items():
                flat.setdefault(k, []).extend(
                    (
                        str(r.id) if r.id is not None else "",
                        r.description or "",
                        str(r.seq),
                        file_index,
                    )
                    for r in recs
                )

        flat_nuc: FlatRecords = {}
        flat_prot: FlatRecords = {}
        inv_map: dict[str, str] = {}

        # Every file is extracted into its own dictionaries so that each record can
        # be tagged with the position of the file it came from.
        for offset, f in enumerate(files):
            nuc_dict = IntergenicDict() if is_igs else PlastidDict()
            prot_dict = PlastidDict() if collects_proteins else None
            extract_rec(f, nuc_dict, prot_dict, start_index + offset)
            _flatten_into(flat_nuc, nuc_dict, start_index + offset)
            _flatten_into(flat_prot, prot_dict, start_index + offset)
            inv_map.update(getattr(nuc_dict, "feat_name_inv_map", {}))

        return ChunkResult(flat_nuc or None, flat_prot or None, inv_map, report)

    @staticmethod
    def _worker_wrapper(args_tuple):
        """Unpacks arguments from the iterator into the static worker."""
        return ExtractAndCollect._extract_recs_static(*args_tuple)

    def _source_of(self, rec: SeqRecord | None) -> str:
        src = getattr(self, "_source_file", None)
        if src:
            return str(src)
        if rec is not None and getattr(rec, "name", None):
            return str(rec.name)
        return ""

    def _skip_feature(
        self, rec: SeqRecord | None, stage: str, feature: object, exc: BaseException
    ) -> None:
        """Record one feature and keep going. The rest of the genome is extracted."""
        label = _feature_label(feature)
        reason = f"{type(exc).__name__}: {exc}"
        src = self._source_of(rec)
        log.warning("skipping %s [%s] in %s: %s", label, stage, src or "record", reason)
        report = getattr(self, "_report", None)
        if report is not None:
            report.add_feature_failure(src, stage, label, reason)

    def _note_rejected(
        self, rec: SeqRecord, stage: str, feature: SeqFeature, obj: Any
    ) -> None:
        """Record a feature whose sequence could not be read.

        Reading-frame skips store a plain string on the feature and stay quiet,
        matching the historical ``log_exception`` behaviour on valid input.
        """
        exc = getattr(obj, "_exception", None)
        if isinstance(exc, BaseException):
            self._skip_feature(rec, stage, feature, exc)

    def _drain_problems(
        self, rec: SeqRecord, features: "RecordFeatures", stage: str
    ) -> None:
        problems = getattr(features, "_problems", ())
        if not problems:
            return
        report = getattr(self, "_report", None)
        src = self._source_of(rec)
        for label, reason in problems:
            log.warning(
                "skipping %s [%s] in %s: %s", label, stage, src or "record", reason
            )
            if report is not None:
                report.add_feature_failure(src, stage, label, reason)

    def _iter_features(
        self, rec: SeqRecord, features: Iterator[SeqFeature], stage: str
    ) -> Iterator[SeqFeature]:
        """Yield features, stopping the generator without aborting the caller."""
        iterator = iter(features)
        while True:
            try:
                feature = next(iterator)
            except StopIteration:
                return
            except Exception as exc:
                self._skip_feature(rec, stage, "remaining features", exc)
                return
            yield feature

    def _extract_cds(
        self,
        rec: SeqRecord,
        nuc_dict: PlastidDict,
        protein_dict: PlastidDict | None = None,
    ):
        """
        Extracts all CDS (coding sequences = genes) from a given sequence record
        """
        record_features = RecordFeatures(rec, self.exclude_cds)
        for feature in self._iter_features(
            rec, record_features.cds_feats, "feature:cds"
        ):
            try:
                if len(feature.location) < self.min_seq_length:
                    continue
                feat_name = PlastidFeature.name_cleaner.get_safe_gene(feature)
                if self._is_region_excluded(feat_name):
                    continue

                # Step 1. Extract nucleotide sequence of each gene and filter against min_seq_length
                gene = GeneFeature(rec, feature)
                if (
                    gene.seq_obj is None
                    or len(gene.seq_obj) < self.min_seq_length
                    or self._is_region_excluded(gene.feat_name)
                ):
                    if gene.seq_obj is None:
                        self._note_rejected(rec, "feature:cds", feature, gene)
                    continue

                # Step 2. Translate nucleotide sequence to protein
                if protein_dict is not None:
                    protein = ProteinFeature(gene=gene)
                    if protein.seq_obj is None:
                        self._note_rejected(rec, "feature:cds", feature, protein)
                        continue
                    protein_dict.add_feature(protein)

                nuc_dict.add_feature(gene)
            except Exception as exc:
                self._skip_feature(rec, "feature:cds", feature, exc)
        self._drain_problems(rec, record_features, "feature:cds")

    def _extract_rna(self, rec: SeqRecord, nuc_dict: PlastidDict):
        """
        Extracts annotated tRNA and rRNA sequences without reading-frame checks or translation.
        """
        record_features = RecordFeatures(rec, self.exclude_cds)
        for feature in self._iter_features(
            rec, record_features.rna_feats, "feature:rna"
        ):
            try:
                if (feature.type == "tRNA" and self.include_trna) or (
                    feature.type == "rRNA" and self.include_rrna
                ):
                    if len(feature.location) < self.min_seq_length:
                        continue
                    feat_name = PlastidFeature.name_cleaner.get_safe_gene(feature)
                    if self._is_region_excluded(feat_name):
                        continue

                    feat = PlastidFeature(rec, feature)
                    if feat.seq_obj is None:
                        self._note_rejected(rec, "feature:rna", feature, feat)
                        nuc_dict.add_feature(feat)
                        continue
                    if len(
                        feat.seq_obj
                    ) < self.min_seq_length or self._is_region_excluded(feat.feat_name):
                        continue

                    nuc_dict.add_feature(feat)
            except Exception as exc:
                self._skip_feature(rec, "feature:rna", feature, exc)
        self._drain_problems(rec, record_features, "feature:rna")

    def _extract_igs(self, rec: SeqRecord, nuc_dict: PlastidDict):
        """
        Extracts all IGS (intergenic spacers) from a given sequence record
        """
        record_features = RecordFeatures(rec, self.exclude_cds)
        all_genes = self._iter_features(rec, record_features.igs_feats, "feature:igs")
        try:
            subsequent_feat = next(all_genes)
        except StopIteration:
            self._drain_problems(rec, record_features, "feature:igs")
            return
        except Exception as exc:
            self._skip_feature(rec, "feature:igs", "intergenic intervals", exc)
            self._drain_problems(rec, record_features, "feature:igs")
            return
        # Step 1. Loop through genes
        for feature in all_genes:
            current_feat = subsequent_feat
            subsequent_feat = feature
            try:
                # Early pre-checks before constructing IntergenicFeature
                curr_name = PlastidFeature.name_cleaner.get_safe_gene(current_feat)
                sub_name = PlastidFeature.name_cleaner.get_safe_gene(subsequent_feat)
                if curr_name and sub_name:
                    igs_name = f"{curr_name}_{sub_name}"
                    inv_name = f"{sub_name}_{curr_name}"
                    if self._is_region_excluded(igs_name, inv_name):
                        continue

                start_pos = int(current_feat.location.end)
                end_pos = int(subsequent_feat.location.start)
                if start_pos < end_pos and (end_pos - start_pos) < self.min_seq_length:
                    continue

                # Step 2. Make IGS SeqFeature
                igs = IntergenicFeature(rec, current_feat, subsequent_feat)
                if igs.seq_obj is None:
                    self._note_rejected(rec, "feature:igs", feature, igs)
                    nuc_dict.add_feature(igs)
                    continue
                if len(igs.seq_obj) < self.min_seq_length:
                    continue
                if self._is_region_excluded(
                    igs.feat_name, getattr(igs, "inv_feat_name", None)
                ):
                    continue

                # Step 3. Attach IGS to growing dictionary
                nuc_dict.add_feature(igs)
            except Exception as exc:
                self._skip_feature(rec, "feature:igs", feature, exc)
        self._drain_problems(rec, record_features, "feature:igs")

    def _extract_int(self, rec: SeqRecord, nuc_dict: PlastidDict):
        """
        Extracts all INT (introns) from a given sequence record
        """

        def _add_intron(feature_obj: SeqFeature, offset: int):
            exon_1 = feature_obj.location.parts[offset]
            exon_2 = feature_obj.location.parts[offset + 1]
            intron_len = (
                abs(int(exon_2.start) - int(exon_1.end))
                if exon_2.start >= exon_1.end
                else abs(int(exon_1.start) - int(exon_2.end))
            )
            if intron_len < self.min_seq_length:
                return

            gene_name = PlastidFeature.name_cleaner.get_safe_gene(feature_obj)
            intron_name = f"{gene_name}_intron{offset + 1}" if gene_name else None
            if self._is_region_excluded(intron_name):
                return

            intron = IntronFeature(rec, feature_obj, offset)
            if intron.seq_obj is None:
                self._note_rejected(rec, "feature:int", feature_obj, intron)
                nuc_dict.add_feature(intron)
                return
            if len(intron.seq_obj) < self.min_seq_length or self._is_region_excluded(
                intron.feat_name
            ):
                return
            nuc_dict.add_feature(intron)

        record_features = RecordFeatures(rec, self.exclude_cds)
        for feature in self._iter_features(
            rec, record_features.int_feats, "feature:int"
        ):
            try:
                n_parts = len(feature.location.parts)
            except Exception as exc:
                self._skip_feature(rec, "feature:int", feature, exc)
                continue
            # Step 1.a. If one intron in gene:
            if n_parts == 2:
                try:
                    _add_intron(feature, 0)
                except Exception as exc:
                    self._skip_feature(rec, "feature:int", feature, exc)
            # Step 1.b. If two introns in gene:
            elif n_parts == 3:
                try:
                    feature_copy = deepcopy(
                        feature
                    )  # Important b/c feature is overwritten in extract_internal_intron()
                except Exception as exc:
                    self._skip_feature(rec, "feature:int", feature, exc)
                    continue
                try:
                    _add_intron(feature, 0)
                except Exception as exc:
                    self._skip_feature(rec, "feature:int", feature, exc)
                try:
                    _add_intron(feature_copy, 1)
                except Exception as exc:
                    self._skip_feature(rec, "feature:int", feature, exc)
        self._drain_problems(rec, record_features, "feature:int")


class RecordFeatures:
    def __init__(self, record: SeqRecord, exclude_list: list[str] | None = None):
        """
        Returns generators for features in a `SeqRecord`.
        For all generators, ORFs and genes listed in `exclude_list` are filtered out.

        Args:
            record: A genome `SeqRecord`.
            exclude_list: A list of gene names to filter out.
        """
        self._features = record.features
        self.rec_name = record.name
        self.exclude_set = set(exclude_list) if exclude_list else set()
        self._problems: list[tuple[str, str]] = []

    def _keep(self, feature: SeqFeature, allowed: tuple[str, ...] | str) -> bool:
        try:
            if isinstance(allowed, str):
                type_ok = feature.type == allowed
            else:
                type_ok = feature.type in allowed
            return bool(type_ok and self._not_exclude(feature))
        except Exception as exc:
            self._problems.append(
                (_feature_label(feature), f"{type(exc).__name__}: {exc}")
            )
            return False

    @property
    def cds_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of CDS features in the record.
        """
        for f in self._features:
            if self._keep(f, "CDS"):
                yield f

    @property
    def rna_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of tRNA and rRNA features in the record.
        """
        for f in self._features:
            if self._keep(f, ("tRNA", "rRNA")):
                yield f

    @property
    def igs_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of IGS features in the record.
        """
        # Note: No need to include "if feature.type=='tRNA'", because all tRNAs are also annotated as genes
        features = [f for f in self._features if self._keep(f, "gene")]
        # Handle spliced exons. A bad location used to abort every gene of the
        # genome; the genes that can still be walked are yielded instead.
        handler = ExonSpliceHandler(features, self.rec_name)
        try:
            handler.resolve()
        except Exception as exc:
            self._problems.append(
                ("compound locations", f"{type(exc).__name__}: {exc}")
            )
        yield from features

    @property
    def int_feats(self) -> Generator[SeqFeature, None, None]:
        """
        A generator of intron-bearing CDS and tRNA features in the record.
        """
        for f in self._features:
            if self._keep(f, ("CDS", "tRNA")):
                yield f

    def _not_exclude(self, feature: SeqFeature) -> bool:
        try:
            gene = PlastidFeature.name_cleaner.get_safe_gene(feature)
            if not gene or not isinstance(gene, str):
                return False
            return gene not in self.exclude_set and "orf" not in gene
        except Exception as exc:
            self._problems.append(
                (_feature_label(feature), f"{type(exc).__name__}: {exc}")
            )
            return False


# -----------------------------------------------------------------#


class ExonSpliceHandler:
    def __init__(self, genes: list[SeqFeature], rec_name: str):
        """
        Coordinates the handling of spliced exons.
        Cis and trans-spliced genes are considered separately.

        Args:
            genes: List of gene features.
            rec_name: Name of the record that contains the genes.
        """
        self.feat_rearr = FeatureRearranger(genes, rec_name)

    def resolve(self):
        """
        The handling of compound locations begins with merging the locations of
        cis-spliced genes. During this process, compound location genes qualified as "trans_splicing"
        or containing rps12 are identified and removed from the genes list, as they will be handled next.

        In the second step, the genes identified as trans-spliced are split into multiple simple location
        features. Each of these splits results in a simple location feature for each contiguous location
        of the compound location. All of these simple location features are inserted (or merged) into their
        expected location in the gene list.
        """
        log.info(
            f"  resolving genes with compound locations in {self.feat_rearr.rec_name}"
        )
        log.info(f"   resolving cis-spliced genes in {self.feat_rearr.rec_name}")
        merger = ExonSpliceMerger(self.feat_rearr)
        merger.merge()
        log.info(f"   resolving trans-spliced genes in {self.feat_rearr.rec_name}")
        insertor = ExonSpliceInsertor(self.feat_rearr)
        insertor.insert()


class FeatureRearranger:
    class FeatureTuple(NamedTuple):
        """
        A tuple that contains a `SeqFeature` as well as other variables
        to assist with manipulating the feature within a features list.
        """

        feature: SeqFeature | None = None
        gene: str | None = None
        is_trans: bool = False
        is_compound: bool = False
        index: int = -1

    def __init__(self, features: list[SeqFeature], rec_name: str):
        """
        An interface to modify the features of a genome record.

        Args:
            features: List of features.
            rec_name: Name of the record that contains the features.
        """
        self._features = features
        self.rec_name = rec_name
        self._set_end_positions()
        self._compound_features: list[SeqFeature] = []
        self._compounds_removed = False
        self._is_sorted = False

    def _set_end_positions(self):
        self._end_positions = [f.location.end for f in self._features]

    def _size(self) -> int:
        return len(self._features)

    def _sort(self):
        if not self._is_sorted:
            self._features.sort(key=lambda gene: gene.location.end)
            self._set_end_positions()
            self._is_sorted = True

    def _get_insert_index(self, feature: SeqFeature) -> int:
        # feature list has to be sorted before being able to find index
        self._sort()
        return bisect.bisect_left(self._end_positions, feature.location.end)

    @staticmethod
    def _get_is_trans(feature: SeqFeature, gene: str) -> bool:
        trans_genes = ["rps12"]
        return bool(feature.qualifiers.get("trans_splicing")) or gene in trans_genes

    def _tuple_from_feature(
        self, feature: SeqFeature, index: int | None = None
    ) -> FeatureTuple:
        gene = PlastidFeature.name_cleaner.get_safe_gene(feature)
        is_trans = self._get_is_trans(feature, gene)
        is_compound = type(feature.location) is CompoundLocation
        index = self._get_insert_index(feature) if index is None else index
        return self.FeatureTuple(feature, gene, is_trans, is_compound, index)

    def _tuple_from_index(self, index: int) -> FeatureTuple:
        if index < 0 or index >= len(self._features):
            return self.FeatureTuple()
        else:
            feature = self._features[index]
            return self._tuple_from_feature(feature, index)

    def feature_tuples(
        self, reverse: bool = True
    ) -> Generator[FeatureTuple, None, None]:
        """
        A generator of feature tuples corresponding to the features list.
        By default, the features are returned in the reverse order that they appear in the file.

        Args:
            reverse: A boolean that indicates if the features should be returned in reverse order.

        Returns:
            A generator of feature tuples.
        """
        start = self._size() - 1 if reverse else 0
        end = -1 if reverse else self._size()
        increment = -1 if reverse else 1
        for index in range(start, end, increment):
            yield self._tuple_from_index(index)

    def _remove_compounds(self):
        # find the compound features and remove them from the gene list
        if not self._compounds_removed:
            self._compound_features.extend(
                f for f in self._features if type(f.location) is CompoundLocation
            )
            # in place: the caller iterates the very list that was passed in, so
            # rebinding `self._features` would silently drop every trans-spliced
            # gene that is re-inserted below
            self._features[:] = [
                f for f in self._features if f not in self._compound_features
            ]
            self._compounds_removed = True

    def simple_features(self) -> Generator[FeatureTuple, None, None]:
        """
        A generator of simple feature tuples created from compound location features.
        If there are simple features, the features list will be sorted by end
        location so that the expected insertion locations can be determined.

        Returns:
            A generator of simple feature tuples.
        """
        self._remove_compounds()
        for f in self._compound_features:
            for p in f.location.parts:
                yield self._tuple_from_feature(
                    SeqFeature(
                        location=p, type=f.type, id=f.id, qualifiers=f.qualifiers
                    )
                )

    def insert(self, feature: FeatureTuple):
        """
        Inserts the feature into the features list.

        Args:
            feature: A feature tuple.
        """
        self._features.insert(feature.index, feature.feature)
        self._end_positions.insert(feature.index, feature.feature.location.end)

    def remove(self, feature: FeatureTuple):
        """
        Removes the feature from the features list.
        Attempting to remove a feature that is not in the list will result in no change.

        Args:
            feature: A feature tuple.
        """
        current = self.get_current(feature)
        if feature == current:
            del self._features[feature.index]
            del self._end_positions[feature.index]
            if feature.is_compound:
                self._compound_features.append(feature.feature)

    def get_previous(self, feature: FeatureTuple) -> FeatureTuple:
        """
        Returns the feature that comes immediately before `feature`.

        Args:
            feature: A feature tuple.

        Returns:
            The previous feature.

        """
        return self._tuple_from_index(feature.index - 1)

    def get_current(self, feature: FeatureTuple) -> FeatureTuple:
        """
        Returns the feature at the current or desired location of `feature`.
        For example, if feature is in the list, what is returned will also be `feature`.
        If the feature is a candidate feature that is not in the list, what will
        be returned is the feature at the location that `feature` should be inserted based on location.

        Args:
            feature:

        Returns:
            The current feature.

        """
        return self._tuple_from_index(feature.index)

    def get_subsequent(self, feature: FeatureTuple) -> FeatureTuple:
        """
        Returns the feature that comes immediately after `feature`.

        Args:
            feature: A feature tuple.

        Returns:
            The subsequent feature.

        """
        return self._tuple_from_index(feature.index + 1)

    def log_position(self, current: FeatureTuple, merge: bool = True):
        """
        Logs the position of `current` within the features list with gene name and location information.

        Args:
            current: A feature tuple.
            merge: A boolean indicating that the associated operation was a merge.
        """
        previous = self.get_previous(current)
        previous_gene = "\t" if not previous.feature else previous.gene
        previous_location = (
            "\t\t\t\t\t" if not previous.feature else previous.feature.location
        )

        subsequent = self.get_subsequent(current)
        subsequent_gene = "\t" if not subsequent.feature else subsequent.gene
        subsequent_location = (
            "\t\t\t\t\t" if not subsequent.feature else subsequent.feature.location
        )

        message_sub = "Merging exons of" if merge else "Repositioning"
        message = f"{message_sub} {current.gene} within {self.rec_name}"
        log.debug(
            f"   {message}\n"
            "-----------------------------------------------------------\n"
            f"\t\t{previous_gene}\t\t\t\t{current.gene}\t\t\t\t{subsequent_gene}\n"
            f"\t{previous_location}\t{current.feature.location}\t{subsequent_location}\n"
            "-----------------------------------------------------------\n"
        )


class ExonSpliceMerger:
    def __init__(self, feat_rearr: FeatureRearranger):
        """
        Coordinates the handling of cis-spliced exons by merging.

        Args:
            feat_rearr: An interface for rearranging features.
        """
        self.feat_rearr = feat_rearr

    def merge(self):
        """
        Locations of compound cis-spliced genes are merged in-place as they occur within the source GenBank file
        (no repositioning of the annotations). Adjacent genes of the same name are assumed to be separately annotated
        cis-spliced genes and merged as well. During this process, compound location annotations qualified as
        "trans_splicing" or containing the gene rps12 (these are known to be trans but are not always qualified as such)
        are removed from the list of gene features, and retained in a separate list.
        """
        feat_tuples = self.feat_rearr.feature_tuples()
        # no initial subsequent feature
        subsequent = FeatureRearranger.FeatureTuple()
        for current in feat_tuples:
            if current.is_trans:
                self.feat_rearr.remove(current)
            elif current.is_compound:
                self._merge_cis_exons(current)
                # print new alignment
                self.feat_rearr.log_position(current)

            is_same_gene = current.gene == subsequent.gene
            if is_same_gene and not current.is_trans:
                self._merge_adj_exons(current, subsequent)
                # delete merged exon
                self.feat_rearr.remove(subsequent)
                # print new alignment
                self.feat_rearr.log_position(current)

            # update for next iteration
            subsequent = current

    @staticmethod
    def _merge_cis_exons(current: FeatureRearranger.FeatureTuple):
        loc_parts = current.feature.location.parts
        gene_start = min(p.start for p in loc_parts)
        gene_end = max(p.end for p in loc_parts)
        current.feature.location = FeatureLocation(gene_start, gene_end)

    @staticmethod
    def _merge_adj_exons(
        current: FeatureRearranger.FeatureTuple,
        subsequent: FeatureRearranger.FeatureTuple,
    ):
        gene_start = min(
            current.feature.location.start, subsequent.feature.location.start
        )
        gene_end = max(current.feature.location.end, subsequent.feature.location.end)
        current.feature.location = FeatureLocation(gene_start, gene_end)


class ExonSpliceInsertor:
    class TestsTuple(NamedTuple):
        is_same_previous: bool
        is_same_current: bool
        is_after_previous: bool
        is_before_current: bool
        not_overlap: bool
        not_same: bool

    def __init__(self, feat_rearr: FeatureRearranger):
        """
        Coordinates the handling of compound location features using insertion and merging.

        Args:
            feat_rearr: An interface for rearranging features.
        """
        self.feat_rearr = feat_rearr

    def insert(self):
        """
        The genes identified as having compound locations are split into simple location
        features. Each of these splits results in a simple location feature for each contiguous location
        of the compound location.

        We then examine each simple location and insert it into its expected location
        in the gene list. The expected location is determined by comparing the simple feature's end location
        with the end locations of the features in the gene list. If the expected location has no
        overlap with the proceeding and succeeding genes, and the feature is a different gene from those two,
        it is directly inserted into that expected location. Alternatively, if the expected location of the feature
        results in a flanking gene (strictly adjacent or overlapping) to be the same as the gene to be
        inserted they are merged. The merge can occur on the proceeding side, succeeding side, or both.
        When merging the gene location, the start location is minimized and the end location is maximized.
        """
        for insert in self.feat_rearr.simple_features():
            # set appropriate adjacent features
            previous = self.feat_rearr.get_previous(insert)
            current = self.feat_rearr.get_current(insert)

            # checks for how to handle the insert feature
            tests = self._get_adj_tests(previous, insert, current)

            # directly reposition if possible
            if tests.not_overlap and tests.not_same:
                self.feat_rearr.insert(insert)
                self.feat_rearr.log_position(insert, False)
            # attempt to resolve by merging
            else:
                self._try_merging(previous, insert, current, tests)

    def _get_adj_tests(
        self,
        previous: FeatureRearranger.FeatureTuple,
        insert: FeatureRearranger.FeatureTuple,
        current: FeatureRearranger.FeatureTuple,
    ) -> TestsTuple:
        is_same_previous = previous.gene == insert.gene
        is_same_current = current.gene == insert.gene
        is_after_previous = (
            not previous.feature
            or previous.feature.location.end < insert.feature.location.start
        )
        is_before_current = (
            not current.feature
            or insert.feature.location.end < current.feature.location.start
        )
        not_overlap = is_after_previous and is_before_current
        not_same = not is_same_previous and not is_same_current
        tests_tuple = self.TestsTuple(
            is_same_previous,
            is_same_current,
            is_after_previous,
            is_before_current,
            not_overlap,
            not_same,
        )
        return tests_tuple

    def _try_merging(
        self,
        previous: FeatureRearranger.FeatureTuple,
        insert: FeatureRearranger.FeatureTuple,
        current: FeatureRearranger.FeatureTuple,
        tests: TestsTuple,
    ):
        is_merged = False
        gene_start = insert.feature.location.start
        gene_end = insert.feature.location.end

        # if insert and current feature are the same gene, and insert starts before current,
        # remove current feature, and update ending location
        if (
            tests.is_same_current
            and insert.feature.location.start < current.feature.location.start
        ):
            gene_end = current.feature.location.end
            self.feat_rearr.remove(current)
            is_merged = True

        # if insert and previous feature are the same gene,
        # use the smaller start location, and remove previous feature
        if tests.is_same_previous:
            gene_start = min(gene_start, previous.feature.location.start)
            self.feat_rearr.remove(previous)
            # elements in list will shift to left, so update index
            # (`FeatureTuple` is an immutable NamedTuple: `insert.index -= 1` raised
            # AttributeError and discarded every gene of the genome)
            insert = insert._replace(index=insert.index - 1)
            is_merged = True

        # perform insertion if needed
        if is_merged:
            # updated feature to insert
            insert.feature.location = FeatureLocation(gene_start, gene_end)
            # insert
            self.feat_rearr.insert(insert)
            self.feat_rearr.log_position(insert)


# -----------------------------------------------------------------#


class DataCleaning:
    def __init__(self, plastid_data: PlastidData, user_params: dict[str, Any]):
        self.plastid_data = plastid_data
        self.min_num_taxa = user_params.get("min_num_taxa")
        self.min_seq_length = user_params.get("min_seq_length")
        self.exclude_region = user_params.get("exclude_region") or []

    def clean(self):
        log.info("cleaning extracted sequence annotations")
        if self.exclude_region:
            log.info("  removing excluded regions")
        if isinstance(self.min_num_taxa, float):
            log.info(
                f"  removing annotations that occur in fewer than {self.min_num_taxa:.2f} of taxa"
            )
        else:
            log.info(
                f"  removing annotations that occur in fewer than {self.min_num_taxa} taxa"
            )
        log.info(
            f"  removing annotations whose longest sequence is shorter than {self.min_seq_length} bp"
        )

        summary = self.plastid_data.store.get_region_summary("nucl")
        num_taxa = getattr(self.plastid_data, "num_taxa", None)

        for feat_name, stats in summary.items():
            if self.exclude_region and feat_name in self.exclude_region:
                log.info(f"    removing {feat_name} for being in exclusion list")
                self.plastid_data.remove_nuc(feat_name)
                continue

            if stats["max_len"] < self.min_seq_length:
                log.info(
                    f"    removing {feat_name} for not reaching the minimum sequence length defined"
                )
                self.plastid_data.remove_nuc(feat_name)
                continue

            count = stats["count"]
            if isinstance(self.min_num_taxa, float):
                if num_taxa is None or num_taxa <= 0:
                    log.critical(
                        "Total number of taxa is unavailable for frequency filtering"
                    )
                    raise AttributeError(
                        "Total number of taxa is unavailable for frequency filtering"
                    )
                if (count / num_taxa) < self.min_num_taxa:
                    log.info(
                        f"    removing {feat_name} for not reaching the minimum relative frequency of taxa defined"
                    )
                    self.plastid_data.remove_nuc(feat_name)
            else:
                if count < self.min_num_taxa:
                    log.info(
                        f"    removing {feat_name} for not reaching the minimum number of taxa defined"
                    )
                    self.plastid_data.remove_nuc(feat_name)
