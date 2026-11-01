#!/usr/bin/env python
"""Calibration of the alignment-strategy constants (``strategy_constants.py``).

Runs MAFFT's four strategy tiers on simulated divergent loci (random tree,
substitutions and indels; see ``plastburstalign/_simulate.py``) over an
``N x L`` grid, records wall time and peak RSS, and fits the constants of the
work model::

    W = N * L^alpha(L) + beta * N^2 * L          (per alphabet)

so that every tier boundary keeps a locus under ``--target-seconds`` (default
45 s) on ``--target-threads`` (default 8) cores. The fit PRINTS a ready-to-paste
``DEFAULT_CONSTANTS`` block for ``plastburstalign/strategy_constants.py``.

Run it on a *quiet* machine and serially (one MAFFT at a time); timings of a
busy host are useless. Nothing here is executed by the test suite except the
``--smoke`` mode (tiny, a few seconds).

Examples::

    python benchmarking/calibrate_strategy.py --smoke
    python benchmarking/calibrate_strategy.py --grid --ns 50,100,200,400 \\
        --lengths 500,1500,3000 --threads 1,8 --timeout 180 --out grid.json
    python benchmarking/calibrate_strategy.py --scenario monocot --threads 8
    python benchmarking/calibrate_strategy.py --fit grid.json monocot.json
    python benchmarking/calibrate_strategy.py --chunk-timing
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plastburstalign._simulate import simulate_locus, write_fasta  # noqa: E402
from plastburstalign.strategy import (  # noqa: E402
    TIER_MAFFT_ARGS,
    AlignmentTier,
    determine_alignment_strategy,
    estimate_work,
)
from plastburstalign.strategy_constants import (  # noqa: E402
    DEFAULT_CONSTANTS,
    AlphabetModel,
    StrategyConstants,
)

#: Named scenarios (loci that stalled in real runs).
SCENARIOS: dict[str, dict[str, Any]] = {
    "monocot": {
        "n": 700,
        "length": 3500,
        "alphabet": "nucl",
        "divergence": 0.35,
        "indel_per_sub": 0.08,
        "conserved_fraction": 0.2,
        "seed": 2026,
        "description": "synthetic Monocot IGS: ~700 divergent taxa, ~3.5 kb",
    },
}

SMOKE = {
    "ns": [8, 16],
    "lengths": [200, 400],
    "tiers": [AlignmentTier.ACCURATE, AlignmentTier.FAST_TREE],
    "threads": [1, 2],
    "timeout": 20.0,
}

TIER_BY_NAME = {t.name.lower(): t for t in AlignmentTier}


# --------------------------------------------------------------------------
# measuring
# --------------------------------------------------------------------------


def mafft_version() -> str:
    """``mafft --version`` (it prints to stderr)."""
    mafft = shutil.which("mafft")
    if mafft is None:
        return "not found"
    done = subprocess.run([mafft, "--version"], capture_output=True, text=True)
    return (done.stderr or done.stdout).strip()


def cpu_model() -> str:
    """CPU model name and logical core count."""
    name = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{name} ({os.cpu_count()} logical cpus)"


def _detect_gnu_time() -> str | None:
    for cand in ("gtime", "/usr/bin/time"):
        p = shutil.which(cand)
        if p and os.access(p, os.X_OK):
            try:
                proc = subprocess.run(
                    [p, "--version"], capture_output=True, text=True, timeout=2
                )
                if "GNU" in (proc.stdout + proc.stderr):
                    return p
            except Exception:
                pass
    return None


GNU_TIME = _detect_gnu_time()


def run_timed(
    cmd: Sequence[str], timeout: float, tmpdir: str
) -> tuple[float, int | None, int, bool]:
    """Run ``cmd`` once; return ``(seconds, returncode, peak_rss_kb, timed_out)``.

    The peak RSS comes from GNU ``time -f %M`` when available: it is the
    maximum over the command and its descendants (MAFFT's helper binaries
    behind the ``mafft`` shell wrapper) and, unlike ``wait4`` rusage taken in
    this (large) Python process, has no inherited-at-fork floor. Without GNU
    time the ``wait4`` rusage is used and values below this interpreter's own
    RSS (~50 MB) are an upper bound only. Timed-out runs report ``rss 0``.
    """
    env = dict(os.environ, TMPDIR=tmpdir)
    rss_file = os.path.join(tmpdir, "rss.txt")
    wrapped = list(cmd)
    if GNU_TIME is not None:
        wrapped = [GNU_TIME, "-f", "%M", "-o", rss_file, *cmd]
    start = time.perf_counter()
    proc = subprocess.Popen(
        wrapped,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    timed_out = False
    status = 0
    rusage = None
    while True:
        pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            break
        if time.perf_counter() - start > timeout:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # the whole tree (new session)
            except (PermissionError, ProcessLookupError):
                try:
                    proc.kill()  # restricted environments: at least the wrapper
                except (PermissionError, ProcessLookupError):
                    pass  # cannot signal at all: wait for the run to finish
            _, status, rusage = os.wait4(proc.pid, 0)
            break
        time.sleep(0.02)
    seconds = time.perf_counter() - start
    returncode = os.waitstatus_to_exitcode(status)
    rss = 0
    if not timed_out:
        try:
            rss = int(Path(rss_file).read_text().strip().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            rss = int(rusage.ru_maxrss) if rusage is not None else 0
    return seconds, (None if timed_out else returncode), rss, timed_out


def tier_command(tier: AlignmentTier, threads: int, path: str) -> list[str]:
    """``mafft <tier args> --thread T <file>``."""
    return ["mafft", *TIER_MAFFT_ARGS[tier], "--thread", str(threads), path]


def make_locus_file(
    directory: Path,
    n: int,
    length: int,
    *,
    alphabet: str,
    divergence: float,
    seed: int,
    indel_per_sub: float = 0.05,
    conserved_fraction: float = 0.25,
) -> Path:
    """Simulate a locus and write it as an unaligned FASTA."""
    records = simulate_locus(
        n,
        length,
        alphabet=alphabet,
        divergence=divergence,
        indel_per_sub=indel_per_sub,
        conserved_fraction=conserved_fraction,
        seed=seed,
    )
    stem = "prot" if alphabet == "prot" else "nucl"
    return write_fasta(directory / f"{stem}_n{n}_l{length}.fasta", records)


def measure_grid(
    ns: Iterable[int],
    lengths: Iterable[int],
    tiers: Sequence[AlignmentTier],
    threads: Sequence[int],
    *,
    alphabet: str = "nucl",
    divergence: float = 0.25,
    seed: int = 1,
    timeout: float = 120.0,
    scenario: dict[str, Any] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Time every ``(N, L, tier, threads)`` cell, one MAFFT at a time.

    Cells that are larger than one that already timed out for the same tier and
    thread count are skipped (recorded as ``skipped``).
    """
    results: list[dict[str, Any]] = []
    stalled: dict[tuple[str, int], tuple[int, int]] = {}
    version = mafft_version()
    scenario = scenario or {}
    with tempfile.TemporaryDirectory(prefix="pba_calibrate_") as tmp:
        root = Path(tmp)
        for n in sorted(ns):
            for length in sorted(lengths):
                path = make_locus_file(
                    root,
                    n,
                    length,
                    alphabet=alphabet,
                    divergence=scenario.get("divergence", divergence),
                    seed=scenario.get("seed", seed),
                    indel_per_sub=scenario.get("indel_per_sub", 0.05),
                    conserved_fraction=scenario.get("conserved_fraction", 0.25),
                )
                work = estimate_work(n, length, alphabet=alphabet)
                for tier in tiers:
                    for t in threads:
                        record: dict[str, Any] = {
                            "n": n,
                            "length": length,
                            "alphabet": alphabet,
                            "tier": tier.name,
                            "threads": t,
                            "work": work,
                            "mafft_version": version,
                            "divergence": scenario.get("divergence", divergence),
                            "seed": scenario.get("seed", seed),
                        }
                        key = (tier.name, t)
                        if (
                            key in stalled
                            and n >= stalled[key][0]
                            and length >= stalled[key][1]
                        ):
                            record.update(skipped=True, seconds=None, rss_kb=None)
                            results.append(record)
                            continue
                        tmpdir = tempfile.mkdtemp(dir=tmp)
                        seconds, rc, rss, timed_out = run_timed(
                            tier_command(tier, t, str(path)), timeout, tmpdir
                        )
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        record.update(
                            skipped=False,
                            seconds=seconds,
                            returncode=rc,
                            timed_out=timed_out,
                            rss_kb=rss,
                        )
                        if timed_out:
                            stalled[key] = (n, length)
                        results.append(record)
                        if verbose:
                            status = "TIMEOUT" if timed_out else f"{seconds:7.2f}s"
                            print(
                                f"N={n:6d} L={length:6d} {tier.name:14s} t={t:2d} "
                                f"{status} rss={rss / 1024:7.0f}MB W={work:.2e}",
                                flush=True,
                            )
                path.unlink(missing_ok=True)
    return results


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _usable(
    results: Iterable[dict[str, Any]], tier: str, alphabet: str, min_seconds: float
) -> list[dict[str, Any]]:
    return [
        r
        for r in results
        if r.get("tier") == tier
        and r.get("alphabet", "nucl") == alphabet
        and not r.get("skipped")
        and not r.get("timed_out")
        and r.get("returncode") == 0
        and (r.get("seconds") or 0.0) >= min_seconds
    ]


def _work(n: int, length: int, alphas: Sequence[float], beta: float) -> float:
    model = AlphabetModel(alphas=(alphas[0], alphas[1], alphas[2]), beta=beta)
    return n * float(length) ** model.alpha_for(length) + beta * n * n * length


def _log_spread(
    rows: Sequence[dict[str, Any]], alphas: Sequence[float], beta: float
) -> float:
    """Variance of ``log(seconds / W)``: zero when time is exactly ``kappa * W``."""
    values = [
        math.log(r["seconds"] / _work(r["n"], r["length"], alphas, beta)) for r in rows
    ]
    if len(values) < 2:
        return 0.0
    return statistics.pvariance(values)


def fit_model(
    rows: Sequence[dict[str, Any]],
    start: AlphabetModel,
    *,
    rounds: int = 3,
) -> tuple[AlphabetModel, int]:
    """Fit ``alphas`` and ``beta`` of one alphabet to timing rows (one tier).

    Coordinate descent over a grid; a regime without data keeps its exponent.
    Returns the model and the number of rows used.
    """
    alphas = list(start.alphas)
    beta = start.beta
    breaks = start.alpha_breaks
    regimes = [
        [r for r in rows if r["length"] < breaks[0]],
        [r for r in rows if breaks[0] <= r["length"] < breaks[1]],
        [r for r in rows if r["length"] >= breaks[1]],
    ]
    exponent_grid = [1.6 + 0.05 * i for i in range(29)]
    beta_grid = [10 ** (-2 + 0.25 * i) for i in range(17)]
    for _ in range(rounds):
        for idx, regime in enumerate(regimes):
            if len(regime) < 2:
                continue
            best = min(
                exponent_grid,
                key=lambda a: _log_spread(
                    regime, [*alphas[:idx], a, *alphas[idx + 1 :]], beta
                ),
            )
            alphas[idx] = best
        if len(rows) >= 3:
            beta = min(beta_grid, key=lambda b: _log_spread(rows, alphas, b))
    model = AlphabetModel(
        alpha_breaks=start.alpha_breaks,
        alphas=(round(alphas[0], 2), round(alphas[1], 2), round(alphas[2], 2)),
        scale=1.0,
        beta=round(beta, 4),
    )
    return model, len(rows)


def kappa(
    rows: Sequence[dict[str, Any]],
    model: AlphabetModel,
    threads: int,
    quantile: float = 0.75,
) -> float | None:
    """Conservative seconds-per-W of a tier at ``threads`` (upper quartile)."""
    ratios = sorted(
        r["seconds"] / _work(r["n"], r["length"], model.alphas, model.beta)
        for r in rows
        if r["threads"] == threads
    )
    if not ratios:
        return None
    index = min(len(ratios) - 1, int(math.ceil(quantile * len(ratios))) - 1)
    return ratios[max(0, index)]


def _round_sig(value: float, digits: int = 2) -> float:
    if value <= 0:
        return value
    exponent = math.floor(math.log10(value))
    factor = 10 ** (digits - 1 - exponent)
    return round(value * factor) / factor


def fit_constants(
    results: Sequence[dict[str, Any]],
    *,
    target_seconds: float = 45.0,
    target_threads: int = 8,
    min_seconds: float = 0.5,
    source: str = "benchmarking/calibrate_strategy.py",
    base: StrategyConstants = DEFAULT_CONSTANTS,
) -> tuple[StrategyConstants, list[str]]:
    """Fit new constants to measured timings.

    The accurate tier defines the work model (``alphas``, ``beta``). A tier
    ``i``'s kappa (seconds per unit of W at ``target_threads``) then gives the
    largest W it can take within ``target_seconds``; the boundaries are those
    limits (made non-decreasing: a tier that is not faster than its
    predecessor gets an empty band).

    Returns ``(constants, notes)``; ``notes`` lists data gaps and decisions.
    """
    notes: list[str] = []
    threads_available = sorted(
        {int(r["threads"]) for r in results if not r.get("skipped")}
    )
    if not threads_available:
        raise ValueError("no usable timing rows")
    threads = (
        target_threads
        if target_threads in threads_available
        else max(threads_available)
    )
    if threads != target_threads:
        notes.append(
            f"no {target_threads}-thread runs: used {threads} threads, so the "
            f"boundaries correspond to {threads} cores, not {target_threads}"
        )
    models: dict[str, AlphabetModel] = {}
    boundaries: dict[AlignmentTier, float] = {}
    for alphabet in ("nucl", "prot"):
        accurate = _usable(results, AlignmentTier.ACCURATE.name, alphabet, min_seconds)
        start = base.model_for(alphabet)
        if len(accurate) >= 3:
            models[alphabet], used = fit_model(accurate, start)
            notes.append(f"{alphabet}: model fitted on {used} ACCURATE runs")
        else:
            models[alphabet] = start
            notes.append(f"{alphabet}: only {len(accurate)} ACCURATE runs; model kept")
    nucl = models["nucl"]
    limits: dict[AlignmentTier, float | None] = {}
    for tier in AlignmentTier:
        rows = _usable(results, tier.name, "nucl", min_seconds)
        k = kappa(rows, nucl, threads)
        limits[tier] = None if not k else target_seconds / k
        if k:
            notes.append(f"{tier.name}: kappa = {k:.3e} s per W at {threads} threads")
        else:
            notes.append(f"{tier.name}: no usable runs at {threads} threads")

    fast = limits[AlignmentTier.ACCURATE] or base.work_fast_tree
    high_limit = limits[AlignmentTier.FAST_TREE]
    extreme_limit = limits[AlignmentTier.HIGH_SCALE]
    high = max(fast, high_limit or base.work_high_scale)
    extreme = max(high, extreme_limit or base.work_extreme_scale)
    for tier, limit, previous in (
        (AlignmentTier.FAST_TREE, high_limit, fast),
        (AlignmentTier.HIGH_SCALE, extreme_limit, high),
    ):
        if limit is not None and limit <= previous:
            notes.append(
                f"{tier.name} is not faster than the previous tier at the target: "
                "its band is empty"
            )
    boundaries[AlignmentTier.FAST_TREE] = fast
    boundaries[AlignmentTier.HIGH_SCALE] = high
    boundaries[AlignmentTier.EXTREME_SCALE] = extreme

    digest = hashlib.sha256(
        json.dumps(list(results), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    versions = sorted({str(r.get("mafft_version", "")) for r in results} - {""})
    constants = StrategyConstants(
        nucl=nucl,
        prot=models["prot"],
        work_fast_tree=_round_sig(boundaries[AlignmentTier.FAST_TREE]),
        work_high_scale=_round_sig(boundaries[AlignmentTier.HIGH_SCALE]),
        work_extreme_scale=_round_sig(boundaries[AlignmentTier.EXTREME_SCALE]),
        n_force_parttree=base.n_force_parttree,
        min_parttree_seq_len=base.min_parttree_seq_len,
        addfragments_max_n=base.addfragments_max_n,
        calibrated=True,
        source=f"{source}, results sha256:{digest}",
        mafft_version=", ".join(versions) or mafft_version(),
        cpu=cpu_model(),
        date=datetime.date.today().isoformat(),
        target_seconds=target_seconds,
        target_threads=threads,
    )
    notes.extend(_memory_notes(results, nucl))
    return constants, notes


def _memory_notes(results: Sequence[dict[str, Any]], model: AlphabetModel) -> list[str]:
    """Fit ``rss = a * N^2`` (ACCURATE tier) and suggest an N cap for 1.5 GB."""
    rows = [
        r
        for r in _usable(results, AlignmentTier.ACCURATE.name, "nucl", 0.0)
        if r["rss_kb"] and r["n"] >= 100
    ]
    if len(rows) < 2:
        return [
            "memory: too few ACCURATE runs with N >= 100 to suggest N_FORCE_PARTTREE"
        ]
    per_n2 = statistics.median(r["rss_kb"] * 1024.0 / (r["n"] ** 2) for r in rows)
    budget = 1.5 * 1024**3
    cap = int(math.sqrt(budget / per_n2) // 500 * 500)
    return [
        f"memory: ACCURATE peak RSS ~ {per_n2:.1f} B * N^2 (median); a 1.5 GB budget "
        f"suggests N_FORCE_PARTTREE ~ {cap} (kept at the fixed value in this block)"
    ]


def format_constants_block(constants: StrategyConstants) -> str:
    """Python source of a ``DEFAULT_CONSTANTS`` assignment."""

    def model(m: AlphabetModel) -> str:
        return (
            f"AlphabetModel(alpha_breaks={m.alpha_breaks!r}, alphas={m.alphas!r}, "
            f"scale={m.scale!r}, beta={m.beta!r})"
        )

    c = constants
    return "\n".join(
        [
            "DEFAULT_CONSTANTS = StrategyConstants(",
            f"    nucl={model(c.nucl)},",
            f"    prot={model(c.prot)},",
            f"    work_fast_tree={c.work_fast_tree:.3g},",
            f"    work_high_scale={c.work_high_scale:.3g},",
            f"    work_extreme_scale={c.work_extreme_scale:.3g},",
            f"    n_force_parttree={c.n_force_parttree},",
            f"    min_parttree_seq_len={c.min_parttree_seq_len},",
            f"    addfragments_max_n={c.addfragments_max_n},",
            f"    calibrated={c.calibrated},",
            f"    source={c.source!r},",
            f"    mafft_version={c.mafft_version!r},",
            f"    cpu={c.cpu!r},",
            f"    date={c.date!r},",
            f"    target_seconds={c.target_seconds!r},",
            f"    target_threads={c.target_threads!r},",
            ")",
        ]
    )


# --------------------------------------------------------------------------
# scenarios and chunk timing
# --------------------------------------------------------------------------


def run_scenario(
    name: str, tiers: Sequence[AlignmentTier], threads: Sequence[int], timeout: float
) -> list[dict[str, Any]]:
    """Run the tiers on a named scenario and report the strategy's choice."""
    spec = SCENARIOS[name]
    strategy = determine_alignment_strategy(
        spec["n"], spec["length"], spec["length"], alphabet=spec["alphabet"]
    )
    print(f"scenario {name}: {spec['description']}")
    print(f"strategy chooses {strategy.tier.name}: {strategy.reason}", flush=True)
    results = measure_grid(
        [spec["n"]],
        [spec["length"]],
        tiers,
        threads,
        alphabet=spec["alphabet"],
        timeout=timeout,
        scenario=spec,
    )
    for r in results:
        r["scenario"] = name
        r["strategy_choice"] = strategy.tier.name
    return results


def chunk_timing(
    n: int = 8, length: int = 8000, threads: int = 1
) -> list[dict[str, Any]]:
    """Chunked versus unchunked alignment of one long locus (MAFFT, 1 thread each)."""
    from plastburstalign.chunking import plan_chunks, splice_alignments

    records = simulate_locus(n, length, divergence=0.25, conserved_fraction=0.3, seed=4)
    out: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pba_chunk_timing_") as tmp:
        root = Path(tmp)
        full = write_fasta(root / "nucl_long.fasta", records)
        plan = plan_chunks(records, "nucl", name="nucl_long")
        if plan is None:
            print("no chunk plan for this locus")
            return out
        paths = plan.write_chunk_fastas(root / "chunks")
        modes: dict[str, list[str]] = {
            "--auto": ["--auto"],
            "--retree 2 --maxiterate 0": ["--retree", "2", "--maxiterate", "0"],
        }
        for label, algo in modes.items():
            cmd = ["mafft", *algo, "--thread", str(threads)]
            tmpdir = tempfile.mkdtemp(dir=tmp)
            full_s, *_ = run_timed([*cmd, str(full)], 600, tmpdir)
            chunk_s = [run_timed([*cmd, str(p)], 600, tmpdir)[0] for p in paths]
            row = {
                "algorithm": label,
                "n": n,
                "length": length,
                "threads": threads,
                "unchunked_seconds": full_s,
                "chunk_seconds": chunk_s,
                "chunks": plan.n_chunks,
                "chunked_serial_seconds": sum(chunk_s),
                "chunked_critical_path_seconds": max(chunk_s),
            }
            out.append(row)
            print(
                f"{label:26s} unchunked {full_s:6.2f}s | {plan.n_chunks} chunks "
                f"{', '.join(f'{s:.2f}' for s in chunk_s)} = {sum(chunk_s):.2f}s serial, "
                f"{max(chunk_s):.2f}s on {plan.n_chunks} cores",
                flush=True,
            )
        # splice sanity (invariants) with real MAFFT output of the last mode
        aligned: list[Path | None] = []
        for spec, path in zip(plan.chunks, paths):
            target = path.with_suffix(".aln")
            with open(target, "w") as handle:
                subprocess.run(
                    ["mafft", "--quiet", "--retree", "2", "--thread", "1", str(path)],
                    stdout=handle,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
            aligned.append(target)
        result = splice_alignments(plan, aligned, root / "spliced.aln")
        print(f"spliced width {result.width}; invariants checked by the unit tests")
    return out


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------


def _ints(text: str) -> list[int]:
    return [int(x) for x in re.split(r"[,\s]+", text.strip()) if x]


def _tiers(text: str) -> list[AlignmentTier]:
    names = [x.strip().lower() for x in text.split(",") if x.strip()]
    if names == ["all"]:
        return list(AlignmentTier)
    try:
        return [TIER_BY_NAME[n] for n in names]
    except KeyError as err:
        raise argparse.ArgumentTypeError(f"unknown tier {err}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="tiny run (< 10 s)")
    mode.add_argument(
        "--grid", action="store_true", help="N x L x tier x threads grid (default)"
    )
    mode.add_argument("--scenario", choices=sorted(SCENARIOS), help="named scenario")
    mode.add_argument("--fit", nargs="+", metavar="RESULTS.json", help="fit constants")
    mode.add_argument(
        "--chunk-timing", action="store_true", help="chunked vs unchunked"
    )
    parser.add_argument("--ns", type=_ints, default=[50, 100, 200, 400])
    parser.add_argument("--lengths", type=_ints, default=[500, 1500, 3000])
    parser.add_argument("--tiers", type=_tiers, default=list(AlignmentTier))
    parser.add_argument("--threads", type=_ints, default=[1, 8])
    parser.add_argument("--alphabet", choices=["nucl", "prot"], default="nucl")
    parser.add_argument("--divergence", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per run")
    parser.add_argument("--target-seconds", type=float, default=45.0)
    parser.add_argument("--target-threads", type=int, default=8)
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.5,
        help="--fit ignores runs shorter than this (start-up noise)",
    )
    parser.add_argument("--out", type=Path, help="write JSON results here")
    return parser


def _emit(results: list[dict[str, Any]], out: Path | None) -> None:
    payload = {
        "mafft_version": mafft_version(),
        "cpu": cpu_model(),
        "date": datetime.date.today().isoformat(),
        "results": results,
    }
    if out is not None:
        out.write_text(json.dumps(payload, indent=1, default=str))
        print(f"wrote {out}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if shutil.which("mafft") is None and not args.fit:
        print("mafft not found on PATH", file=sys.stderr)
        return 2

    if args.fit:
        rows: list[dict[str, Any]] = []
        for name in args.fit:
            data = json.loads(Path(name).read_text())
            rows.extend(data["results"] if isinstance(data, dict) else data)
        constants, notes = fit_constants(
            rows,
            target_seconds=args.target_seconds,
            target_threads=args.target_threads,
            min_seconds=args.min_seconds,
        )
        for note in notes:
            print(f"# {note}")
        print("# paste into plastburstalign/strategy_constants.py:")
        print(format_constants_block(constants))
        return 0

    if args.chunk_timing:
        _emit(chunk_timing(), args.out)
        return 0

    if args.scenario:
        results = run_scenario(args.scenario, args.tiers, args.threads, args.timeout)
        _emit(results, args.out)
        return 0

    if args.smoke:
        results = measure_grid(
            SMOKE["ns"],
            SMOKE["lengths"],
            SMOKE["tiers"],
            SMOKE["threads"],
            timeout=SMOKE["timeout"],
            verbose=True,
        )
        _emit(results, args.out)
        return 0

    results = measure_grid(
        args.ns,
        args.lengths,
        args.tiers,
        args.threads,
        alphabet=args.alphabet,
        divergence=args.divergence,
        seed=args.seed,
        timeout=args.timeout,
    )
    _emit(results, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
