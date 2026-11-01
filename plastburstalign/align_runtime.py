"""Shared runtime helpers for LocusScheduler-driven MSA (v3 wiring).

Keeps :class:`~plastburstalign.alignment_ops.AlignmentCoordination` readable
while owning the TuningPlan / scratch / chunk / pin decisions.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .alignment_tools import mafft_auto_explicit_args, mafft_auto_is_fft
from .autotune import TuningPlan, autotune
from .chunking import ChunkConfig
from .ephemeral_engine import ScratchEngine
from .logging_ops import logger as log
from .strategy import (
    AlignmentStrategy,
    AlignmentTier,
    determine_alignment_strategy,
    estimate_work,
)
from .task_queue import CoreBudgetPool


def attach_shared_scratch(aligner: Any, engine: ScratchEngine) -> None:
    """Point ``aligner.scratch_mgr`` at a shared :class:`ScratchEngine`."""
    mgr = getattr(aligner, "scratch_mgr", None)
    if mgr is None:
        return
    old = getattr(mgr, "engine", None)
    if old is not None and old is not engine:
        try:
            old.cleanup()
        except Exception as exc:
            log.debug(f"discarding per-aligner scratch failed: {exc}")
    mgr.engine = engine
    try:
        mgr.scratch_path = engine.path
        mgr._is_ramdisk = engine.is_ramdisk
        mgr._cleaned = False
    except Exception as exc:
        log.debug(f"could not rebind aligner scratch path: {exc}")


def resolve_tuning_plan(
    user_params: Mapping[str, Any],
    region_summary: Mapping[str, Mapping[str, int]],
    *,
    alphabet: str,
) -> TuningPlan:
    """Build a :class:`TuningPlan` and sync ``user_params['num_threads']``."""
    workload: list[tuple[Any, ...]] = []
    for entry in region_summary.values():
        if not isinstance(entry, Mapping):
            continue
        n = int(entry.get("count", 0) or 0)
        max_len = int(entry.get("max_len", 0) or 0)
        min_len = entry.get("min_len")
        if n <= 0 or max_len <= 0:
            continue
        strat = determine_alignment_strategy(
            n,
            max_len,
            int(min_len) if min_len is not None else None,
            alphabet=alphabet,
        )
        workload.append((n, max_len, strat.uses_parttree, alphabet))

    requested = user_params.get(
        "num_threads_requested", user_params.get("num_threads", "auto")
    )
    plan = autotune(
        num_threads=requested if requested is not None else "auto",
        smt=user_params.get("smt"),
        memory_limit=user_params.get("memory_limit"),
        workload=workload or None,
    )
    return plan


def water_fill_preferences(
    region_summary: Mapping[str, Mapping[str, int]],
    regions: Sequence[str],
    total_cores: int,
) -> dict[str, int]:
    """Preferred thread widths from the legacy water-filling schedule."""
    tasks_meta = {
        k: (
            int(region_summary.get(k, {}).get("count", 1) or 1),
            int(region_summary.get(k, {}).get("max_len", 1) or 1),
        )
        for k in regions
    }
    return CoreBudgetPool.solve_water_filling_schedule(
        tasks_meta, total_cores=max(1, int(total_cores))
    )


def strategy_for_locus(
    summary_entry: Mapping[str, int],
    *,
    alphabet: str,
    overrides: Mapping[str, Any] | None = None,
) -> AlignmentStrategy:
    n = int(summary_entry.get("count", 0) or 0)
    max_len = int(summary_entry.get("max_len", 0) or 0)
    min_raw = summary_entry.get("min_len")
    min_len = int(min_raw) if min_raw is not None else None
    return determine_alignment_strategy(
        n, max_len, min_len, alphabet=alphabet, overrides=overrides
    )


def pin_algorithm_args(
    strategy: AlignmentStrategy, n_seq: int, max_len: int
) -> tuple[str, ...]:
    """Parent-locus MAFFT flags safe to reuse on chunks (never ``--auto``)."""
    if strategy.tier is AlignmentTier.ACCURATE or "--auto" in strategy.mafft_args:
        return mafft_auto_explicit_args(n_seq, max_len)
    return tuple(str(a) for a in strategy.mafft_args)


def work_for_locus(summary_entry: Mapping[str, int], *, alphabet: str) -> float:
    n = int(summary_entry.get("count", 0) or 0)
    max_len = int(summary_entry.get("max_len", 0) or 0)
    if n <= 0 or max_len <= 0:
        return 1.0
    return float(estimate_work(n, max_len, alphabet=alphabet))


def unaligned_bytes_estimate(summary_entry: Mapping[str, int]) -> int:
    n = int(summary_entry.get("count", 0) or 0)
    max_len = int(summary_entry.get("max_len", 0) or 0)
    return max(0, n * max_len)


def want_subdivide(
    *,
    mode: str,
    alphabet: str,
    n_seq: int,
    max_len: int,
    is_mafft: bool,
    adjust_direction: bool,
    strategy: AlignmentStrategy,
    config: ChunkConfig | None = None,
) -> bool:
    """Whether to attempt anchor chunking for this locus."""
    if mode not in {"auto", "force"}:
        return False
    if adjust_direction or not is_mafft:
        return False
    cfg = config or ChunkConfig()
    min_len, _, _ = cfg.for_alphabet("prot" if alphabet == "prot" else "nucl")
    if n_seq < max(2, cfg.min_sequences):
        return False
    if mode == "force":
        return max_len >= 2  # still needs a valid anchor chain at plan time
    if max_len < min_len:
        return False
    # auto: only when full-locus --auto would stay in the FFT regime, so pinning
    # chunks does not silently upgrade them to L-INS-i.
    if strategy.tier is AlignmentTier.ACCURATE:
        return mafft_auto_is_fft(n_seq, max_len)
    return True


PartitionFmt = Literal["raxml", "iqtree", "nexus", "partitionfinder"]


def parse_partitions_flag(value: Any) -> tuple[PartitionFmt, ...]:
    """CLI ``--partitions`` -> formats for :meth:`Supermatrix.write_all`."""
    if value is None or value is True:
        return ("raxml", "nexus")
    if value is False or value in {"none", "off", ""}:
        return ()
    if isinstance(value, (list, tuple)):
        parts = [str(v).lower() for v in value]
    else:
        text = str(value).strip().lower()
        if text in {"none", "off"}:
            return ()
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    allowed: set[str] = {"raxml", "nexus", "iqtree", "partitionfinder"}
    out: list[PartitionFmt] = [p for p in parts if p in allowed]  # type: ignore[misc]
    return tuple(out) if out else ("raxml", "nexus")


def trim_spec_from_params(user_params: Mapping[str, Any]) -> str | None:
    """Build a TrimSpec string from ``trim`` / ``min_occupancy``."""
    trim = user_params.get("trim")
    min_occ = user_params.get("min_occupancy")
    if trim in (None, "", "none", False):
        if min_occ is None:
            return None
        return f"occupancy={float(min_occ)}"
    text = str(trim)
    if min_occ is not None and "occupancy=" not in text:
        text = (
            f"{text}+occupancy={float(min_occ)}"
            if text not in {"none", ""}
            else f"occupancy={float(min_occ)}"
        )
    return None if text in {"none", ""} else text


def clear_stale_outputs(out_dir: str) -> None:
    """Drop a previous run's feature store and unaligned FASTA (fresh runs only)."""
    for name in ("feature_store.db", "feature_store.db-wal", "feature_store.db-shm"):
        path = os.path.join(out_dir, name)
        try:
            os.remove(path)
        except OSError:
            pass
    try:
        for entry in os.listdir(out_dir):
            if entry.endswith(".unalign.fasta"):
                try:
                    os.remove(os.path.join(out_dir, entry))
                except OSError:
                    pass
    except OSError:
        pass
