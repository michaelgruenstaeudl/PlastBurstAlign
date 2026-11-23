"""Public library API: :func:`align`, :func:`autotune`, :func:`estimate`.

``pba.align(...)`` is the typed entry point for notebooks and pipelines. It
runs one or more selection modes, merges multi-mode loci into a single
supermatrix (partition names ``{mode}_{locus}``), and returns an
:class:`AlignmentResult` with writers and a path to ``run_manifest.json``.

Defaults that differ from the CLI (and preserve golden CLI behaviour):

* ``keep_unaligned=False`` here; CLI default remains ``True``.
* ``dashboard="auto"`` (null / logging / rich via :func:`~plastburstalign.ui.make_sink`).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from .align_runtime import parse_partitions_flag, trim_spec_from_params
from .autotune import TuningPlan
from .autotune import autotune as _autotune
from .estimate import PreflightEstimate, preflight_estimate
from .extraction_ops import DataCleaning, ExtractAndCollect
from .hardware import HostProfile
from .logging_ops import logger as log
from .manifest import MANIFEST_NAME
from .plast_burst_align import PlastBurstAlign
from .seqfeature_ops import PlastidData
from .supermatrix import (
    LocusSource,
    LocusStats,
    PartitionFileFormat,
    Supermatrix,
    SupermatrixBuilder,
    legacy_concat_rename,
)
from .ui import close_sink, make_sink
from .user_parameters import SelectMode, UserParameters

__all__ = [
    "AlignmentResult",
    "align",
    "autotune",
    "estimate",
    "parse_modes",
]

U8Array: TypeAlias = npt.NDArray[np.uint8]
DashboardMode: TypeAlias = Literal["auto", "rich", "plain", "off"] | bool
PathLike: TypeAlias = str | os.PathLike[str]

_VALID_MODES = frozenset(m.value for m in SelectMode)


def parse_modes(mode: str | Sequence[str]) -> tuple[str, ...]:
    """Split a mode string (``cds+igs+int`` / ``cds,igs``) into unique tokens."""
    if isinstance(mode, str):
        text = mode.strip().lower()
        if not text:
            raise ValueError("mode must be a non-empty string")
        parts = [p.strip() for p in text.replace(",", "+").split("+") if p.strip()]
    else:
        parts = [str(p).strip().lower() for p in mode if str(p).strip()]
    if not parts:
        raise ValueError("mode must list at least one of cds, igs, int, rna")
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if part not in _VALID_MODES:
            raise ValueError(
                f"unknown mode {part!r}; expected one of {sorted(_VALID_MODES)}"
            )
        if part not in seen:
            seen.add(part)
            out.append(part)
    return tuple(out)


def _dashboard_sink_mode(dashboard: DashboardMode) -> str:
    if isinstance(dashboard, bool):
        return "rich" if dashboard else "off"
    key = str(dashboard).strip().lower()
    if key in {
        "auto",
        "rich",
        "plain",
        "off",
        "none",
        "null",
        "quiet",
        "log",
        "logging",
    }:
        return "off" if key in {"none", "null", "quiet"} else key
    raise ValueError(
        f"dashboard must be auto/rich/plain/off or bool; got {dashboard!r}"
    )


def _resolve_out_dir(out_dir: PathLike | None) -> str:
    if out_dir is None:
        path = tempfile.mkdtemp(prefix="pba_align_")
        log.info(f"align() writing to temporary out_dir {path}")
        return path
    path = os.path.abspath(os.fspath(out_dir))
    os.makedirs(path, exist_ok=True)
    return path


_USER_PARAM_KEYS = frozenset(
    {
        "fileext",
        "exclude_fullcds",
        "exclude_region",
        "min_seq_length",
        "min_num_taxa",
        "verbose",
        "order",
        "alignment_tool",
        "alignment_tool_path",
        "alignment_tool_config",
        "alignment_command",
        "config_file",
        "include_trna",
        "include_rrna",
        "adaptive_heuristics",
        "individual_nexus",
        "scratch_dir",
        "min_occupancy",
        "partitions",
        "codon_partitions",
        "smt",
        "memory_limit",
        "seed",
        "phylip",
        "time_budget",
    }
)


def _build_params(
    *,
    in_dir: str,
    out_dir: str,
    select_mode: str,
    num_threads: int | str,
    keep_unaligned: bool,
    subdivide: str,
    trim: str | None,
    kwargs: Mapping[str, Any],
) -> UserParameters:
    known = {k: kwargs[k] for k in _USER_PARAM_KEYS if k in kwargs}
    unknown = sorted(set(kwargs) - _USER_PARAM_KEYS - {"progress_sink", "concat"})
    if unknown:
        raise TypeError(f"align()/estimate() got unexpected keyword(s): {unknown}")
    params = UserParameters(
        in_dir=in_dir,
        out_dir=out_dir,
        select_mode=select_mode,
        num_threads=num_threads,
        keep_unaligned=keep_unaligned,
        subdivide=subdivide,
        trim=trim,
        concat=True,
        **known,
    )
    sink = kwargs.get("progress_sink")
    if sink is not None:
        params["progress_sink"] = sink
    if kwargs.get("time_budget") is not None or params.get("time_budget") is not None:
        log.warning(
            "--time-budget / time_budget is opt-in and non-reproducible; "
            "host-adaptive scheduling is not fully wired yet (value recorded only)"
        )
    return params


def _region_key_for_rename(locus: str) -> str:
    """Map partition name to the FASTA id region key used by ``legacy_concat_rename``.

    Per-locus FASTA headers are ``{region}_{taxon}``. Multi-mode merge stores
    partition names as ``{mode}_{region}`` (e.g. ``cds_atpA``); stripping must
    use ``region``, otherwise each locus-prefixed id is treated as a distinct
    taxon and ``ntax`` inflates to roughly ``n_taxa * n_loci``.
    """
    for mode in _VALID_MODES:
        prefix = f"{mode}_"
        if locus.startswith(prefix) and len(locus) > len(prefix):
            return locus[len(prefix) :]
    return locus


def _api_concat_rename(locus: str, raw_id: str) -> str:
    """``legacy_concat_rename`` with multi-mode partition names."""
    return legacy_concat_rename(_region_key_for_rename(locus), raw_id)


def _collect_locus_sources(
    mode_dir: str, mode: str, *, prefix: bool
) -> list[LocusSource]:
    fasta_dir = os.path.join(mode_dir, "fasta")
    if not os.path.isdir(fasta_dir):
        return []
    sources: list[LocusSource] = []
    for name in sorted(os.listdir(fasta_dir)):
        if not (name.startswith("nucl_") and name.endswith(".aligned.fasta")):
            continue
        region = name[len("nucl_") : -len(".aligned.fasta")]
        path = os.path.join(fasta_dir, name)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            continue
        locus_name = f"{mode}_{region}" if prefix else region
        sources.append(
            LocusSource(
                name=locus_name,
                path=path,
                kind=mode,
                codon_aligned=(mode == "cds"),
            )
        )
    return sources


def _build_supermatrix(
    sources: Sequence[LocusSource],
    *,
    params: Mapping[str, Any],
    out_dir: str,
    write: bool,
) -> Supermatrix:
    if not sources:
        raise RuntimeError("no aligned loci available to build a supermatrix")
    trim = trim_spec_from_params(params)
    normalize = trim is not None
    workers = min(max(1, int(params.get("num_threads", 4) or 4)), 8)
    builder = SupermatrixBuilder(
        list(sources),
        rename=_api_concat_rename,
        trim=trim,
        normalize_case=normalize,
        max_workers=workers,
    )
    matrix = builder.build()
    if write:
        partitions = parse_partitions_flag(params.get("partitions"))
        codon = bool(params.get("codon_partitions", False)) and any(
            s.codon_aligned for s in sources
        )
        prefix = os.path.join(out_dir, f"nucl_{len(sources)}concat")
        matrix.write_all(
            prefix,
            phylip=bool(params.get("phylip", False)),
            partitions=partitions,
            codon_partitions=codon,
        )
    return matrix


@dataclass
class AlignmentResult:
    """Typed result of :func:`align`.

    Holds a :class:`~plastburstalign.supermatrix.Supermatrix` (read-only matrix
    view, per-locus stats) plus the path to ``run_manifest.json`` when written.
    """

    _matrix: Supermatrix
    out_dir: str
    modes: tuple[str, ...]
    manifest_path: str | None = None

    @property
    def ntax(self) -> int:
        return self._matrix.ntax

    @property
    def nchar(self) -> int:
        return self._matrix.nchar

    @property
    def informative_sites(self) -> int:
        return self._matrix.informative_sites

    @property
    def taxa(self) -> tuple[str, ...]:
        return self._matrix.taxa

    @property
    def matrix(self) -> U8Array:
        """Read-only ``(ntax, nchar)`` uint8 view of raw ASCII bytes."""
        return self._matrix.matrix

    @property
    def locus_report(self) -> tuple[LocusStats, ...]:
        """Per-locus statistics (kept columns)."""
        return self._matrix.stats

    @property
    def supermatrix(self) -> Supermatrix:
        return self._matrix

    def summary(self) -> str:
        modes = "+".join(self.modes)
        return f"[{modes}] {self._matrix.summary()}"

    def write_fasta(
        self,
        path: PathLike,
        *,
        partition_format: PartitionFileFormat | Literal["none"] | None = None,
    ) -> Path:
        dest = self._matrix.write_fasta(path)
        self._maybe_write_partitions(dest, partition_format)
        return dest

    def write_phylip(
        self,
        path: PathLike,
        *,
        relaxed: bool = True,
        partition_format: PartitionFileFormat | Literal["none"] | None = None,
    ) -> Path:
        dest = self._matrix.write_phylip(path, relaxed=relaxed)
        self._maybe_write_partitions(dest, partition_format)
        return dest

    def write_nexus(
        self,
        path: PathLike,
        *,
        partition_format: Literal["nexus", "none"] = "nexus",
        codon_partitions: bool = False,
    ) -> Path:
        return self._matrix.write_nexus(
            path,
            partition_format=partition_format,
            codon_partitions=codon_partitions,
        )

    def _maybe_write_partitions(
        self,
        alignment_path: Path,
        partition_format: PartitionFileFormat | Literal["none"] | None,
    ) -> None:
        if partition_format is None or partition_format == "none":
            return
        stem = alignment_path.with_suffix("")
        suffixes = {
            "raxml": ".partitions.txt",
            "iqtree": ".partitions.iqtree.txt",
            "nexus": ".partitions.nex",
            "partitionfinder": ".partitionfinder.cfg",
        }
        self._matrix.write_partitions(
            f"{stem}{suffixes[partition_format]}",
            partition_format,
            alignment_name=alignment_path.name,
        )


def align(
    input_dir: PathLike,
    mode: str = "cds+igs+int",
    *,
    trim: str | None = None,
    out_dir: PathLike | None = None,
    num_threads: int | str = "auto",
    keep_unaligned: bool = False,
    subdivide: str = "auto",
    dashboard: DashboardMode = "auto",
    **kwargs: Any,
) -> AlignmentResult:
    """Extract, align and concatenate plastome regions.

    Multi-mode values (``cds+igs+int``) run one pass per mode into
    ``out_dir/<mode>/`` and merge loci into one supermatrix with mode-prefixed
    partition names (e.g. ``cds_atpA``).

    Args:
        input_dir: Directory of GenBank flatfiles.
        mode: One mode or ``+``/``,``-joined modes.
        trim: Site-trim spec (``None`` / ``occupancy=...`` / ``gappyout`` / ...).
        out_dir: Output directory (a temp dir is created when omitted).
        num_threads: Positive int or ``"auto"``.
        keep_unaligned: Keep ``*.unalign.fasta`` (API default ``False``).
        subdivide: ``auto`` | ``off`` | ``force`` sub-locus chunking.
        dashboard: Progress sink mode (``auto``/``rich``/``plain``/``off`` or bool).
        **kwargs: Forwarded to :class:`~plastburstalign.user_parameters.UserParameters`
            (``fileext``, ``min_num_taxa``, ``alignment_tool``, ``partitions``, ...).

    Returns:
        :class:`AlignmentResult` for the (merged) supermatrix.
    """
    modes = parse_modes(mode)
    in_dir = os.path.abspath(os.fspath(input_dir))
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"input_dir does not exist: {in_dir}")
    root_out = _resolve_out_dir(out_dir)
    sink = kwargs.pop("progress_sink", None)
    own_sink = sink is None
    if own_sink:
        sink = make_sink(_dashboard_sink_mode(dashboard))

    sources: list[LocusSource] = []
    last_params: UserParameters | None = None
    multi = len(modes) > 1
    try:
        if sink is not None:
            kwargs = {**kwargs, "progress_sink": sink}
        for select_mode in modes:
            mode_out = os.path.join(root_out, select_mode) if multi else root_out
            os.makedirs(mode_out, exist_ok=True)
            params = _build_params(
                in_dir=in_dir,
                out_dir=mode_out,
                select_mode=select_mode,
                num_threads=num_threads,
                keep_unaligned=keep_unaligned,
                subdivide=subdivide,
                trim=trim,
                kwargs=kwargs,
            )
            last_params = params
            PlastBurstAlign(params).execute()
            sources.extend(_collect_locus_sources(mode_out, select_mode, prefix=multi))
        if last_params is None:
            raise RuntimeError("no modes executed")
        # Single-mode already wrote a concat via PlastBurstAlign; rebuild so the
        # result exposes an in-memory matrix (and rewrite for multi-mode).
        matrix = _build_supermatrix(
            sources, params=last_params, out_dir=root_out, write=multi
        )
        manifest: str | None
        if multi:
            from datetime import datetime, timezone

            from .manifest import write_run_manifest

            manifest = write_run_manifest(
                root_out,
                started_at=datetime.now(timezone.utc),
                wall_seconds=0.0,
                params={**dict(last_params), "select_mode": "+".join(modes)},
                supermatrix={
                    "ntax": matrix.ntax,
                    "nchar": matrix.nchar,
                    "nloci": len(sources),
                    "modes": list(modes),
                },
                ok=True,
                extra={"merged_modes": list(modes)},
            )
        else:
            candidate = os.path.join(root_out, MANIFEST_NAME)
            manifest = candidate if os.path.isfile(candidate) else None
        return AlignmentResult(
            _matrix=matrix,
            out_dir=root_out,
            modes=modes,
            manifest_path=manifest,
        )
    finally:
        if own_sink:
            close_sink(sink)


def autotune(
    host: HostProfile | None = None,
    *,
    num_threads: int | str = "auto",
    smt: bool | None = None,
    memory_limit: int | str | None = None,
    workload: Any = None,
    **kwargs: Any,
) -> TuningPlan:
    """Wrapper around :func:`plastburstalign.autotune.autotune`."""
    return _autotune(
        host,
        num_threads=num_threads,
        smt=smt,
        memory_limit=memory_limit,
        workload=workload,
        **kwargs,
    )


def estimate(
    input_dir: PathLike,
    mode: str = "cds",
    *,
    out_dir: PathLike | None = None,
    num_threads: int | str = "auto",
    keep_unaligned: bool = False,
    subdivide: str = "auto",
    trim: str | None = None,
    cleanup: bool = True,
    **kwargs: Any,
) -> PreflightEstimate:
    """Extract + clean, then return a :class:`~plastburstalign.estimate.PreflightEstimate`.

    Only a single selection mode is supported (same rule as
    :func:`~plastburstalign.estimate.preflight_estimate`).
    """
    modes = parse_modes(mode)
    if len(modes) != 1:
        raise ValueError(
            f"estimate() accepts one mode at a time; got {mode!r}. "
            "Call once per mode or use align() for multi-mode runs."
        )
    select_mode = modes[0]
    in_dir = os.path.abspath(os.fspath(input_dir))
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"input_dir does not exist: {in_dir}")

    own_tmp = out_dir is None
    root_out = _resolve_out_dir(out_dir)
    params = _build_params(
        in_dir=in_dir,
        out_dir=root_out,
        select_mode=select_mode,
        num_threads=num_threads,
        keep_unaligned=keep_unaligned,
        subdivide=subdivide,
        trim=trim,
        kwargs=kwargs,
    )
    from .align_runtime import clear_stale_outputs

    clear_stale_outputs(root_out)
    data = PlastidData(params)
    try:
        ExtractAndCollect(data, params).extract()
        DataCleaning(data, params).clean()
        prefix = "prot" if select_mode == "cds" else "nucl"
        try:
            summary = data.store.get_region_summary(seq_type=prefix)
        except Exception:
            summary = data.store.get_region_summary(seq_type="nucl")
        prot_summary = None
        if select_mode == "cds":
            try:
                prot_summary = data.store.get_region_summary(seq_type="prot")
            except Exception:
                prot_summary = None
        ntax = int(getattr(data, "num_taxa", 0) or 0) or None
        plan = _autotune(
            num_threads=params.get("num_threads_requested", num_threads),
            smt=params.get("smt"),
            memory_limit=params.get("memory_limit"),
        )
        want_sub = str(params.get("subdivide", "auto")).lower() != "off"
        return preflight_estimate(
            summary if isinstance(summary, dict) else {},
            mode=select_mode,
            ntax=ntax,
            plan=plan,
            subdivide=want_sub,
            prot_summary=prot_summary if isinstance(prot_summary, dict) else None,
            keep_unaligned=bool(params.get("keep_unaligned", False)),
        )
    finally:
        try:
            data.store.close()
        except Exception:
            pass
        if own_tmp and cleanup:
            import shutil

            shutil.rmtree(root_out, ignore_errors=True)
