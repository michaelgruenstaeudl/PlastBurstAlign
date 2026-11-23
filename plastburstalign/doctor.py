"""``plastburstalign --doctor``: environment diagnosis with concrete advice.

:func:`run_doctor` gathers the host profile, the tuning plan, the resolution
status of every supported aligner (path, version, source, rejected candidates)
and the scratch tiers, and turns problems into actionable recommendations
("MAFFT not found: conda install -c bioconda mafft; ..."). :func:`format_report`
renders the result as plain text; :func:`main` is the CLI entry point (the
integration layer wires it to ``plastburstalign --doctor``).

Exit codes of :func:`main`: ``0`` healthy (at least one aligner, possibly the
builtin fallback, is usable), ``1`` no usable aligner at all or ``--strict`` was
given and no *external* aligner was found.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

from . import aligner_resolver as resolver
from .autotune import (
    MIN_RAM_SCRATCH_FREE_BYTES,
    TuningPlan,
    autotune,
)
from .hardware import HostProfile, detect_host, format_bytes

__all__ = ["DoctorReport", "format_report", "main", "run_doctor"]

_DISPLAY = {"mafft": "MAFFT", "muscle": "MUSCLE", "clustalo": "Clustal Omega"}

_INSTALL_ADVICE = {
    "mafft": "conda install -c bioconda mafft  |  apt install mafft  |  brew install mafft",
    "muscle": "conda install -c bioconda muscle  |  apt install muscle  |  brew install muscle",
    "clustalo": (
        "conda install -c bioconda clustalo  |  apt install clustalo  |  "
        "brew install clustal-omega"
    ),
}

_DEPENDENCIES = ("biopython", "numpy", "psutil", "pyyaml", "coloredlogs", "rich")


@dataclass(frozen=True)
class DoctorReport:
    """Everything ``--doctor`` knows, structured for tests and ``--json``.

    Attributes:
        host: Detected host profile.
        plan: Tuning plan derived from it (``num_threads="auto"``).
        tools: One :class:`~plastburstalign.aligner_resolver.Resolution` per
            supported aligner (mafft, muscle, clustalo).
        builtin_available: Whether the pure-Python fallback can be used.
        builtin_note: Why/why not (import error, forced by environment, ...).
        forced_builtin: ``PBA_FORCE_BUILTIN_ALIGNER`` is set.
        dependencies: Installed versions of runtime dependencies (``None`` = missing).
        recommendations: Concrete, ordered advice.
        python: Interpreter description.
        package_version: PlastBurstAlign version.
    """

    host: HostProfile
    plan: TuningPlan
    tools: tuple[resolver.Resolution, ...]
    builtin_available: bool
    builtin_note: str
    forced_builtin: bool
    dependencies: dict[str, str | None]
    recommendations: tuple[str, ...]
    python: str
    package_version: str

    @property
    def external_found(self) -> bool:
        return any(r.tool is not None for r in self.tools)

    @property
    def ok(self) -> bool:
        """At least one aligner (external or builtin) is usable."""
        return self.external_found or self.builtin_available

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dictionary."""
        return dataclasses.asdict(self)


def _package_version() -> str:
    try:
        from ._meta import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _check_builtin() -> tuple[bool, str]:
    try:
        from .builtin_aligner import BuiltinAlignerConfig

        cfg = BuiltinAlignerConfig()
    except Exception as exc:
        return False, f"builtin fallback unavailable ({type(exc).__name__}: {exc})"
    return True, (
        f"pure-Python center-star aligner, lower accuracy; guards: up to "
        f"{cfg.max_sequences} sequences / {cfg.max_total_cells:.1e} DP cells per locus"
    )


def _recommendations(
    host: HostProfile,
    plan: TuningPlan,
    tools: Sequence[resolver.Resolution],
    builtin_available: bool,
    forced: bool,
    dependencies: Mapping[str, str | None],
) -> list[str]:
    recs: list[str] = []
    by_kind = {r.kind: r for r in tools}
    if forced:
        recs.append(
            f"{resolver.FORCE_BUILTIN_ENV} is set: external aligners are ignored and the "
            "builtin fallback (reduced accuracy, size-limited) will be used. Unset it to "
            "use MAFFT/MUSCLE/Clustal Omega."
        )
    else:
        mafft = by_kind["mafft"]
        if mafft.tool is None:
            fallback = (
                "the builtin fallback will be used with reduced accuracy"
                if builtin_available
                else "no fallback is available either"
            )
            recs.append(f"MAFFT not found: {_INSTALL_ADVICE['mafft']}; {fallback}.")
            for attempt in mafft.attempts:
                recs.append(
                    f"MAFFT candidate {attempt.path} ({attempt.source}) was rejected: "
                    f"{attempt.reason}."
                )
        for kind in ("muscle", "clustalo"):
            res = by_kind[kind]
            if res.tool is None:
                recs.append(
                    f"Optional: {_DISPLAY[kind]} not found ({_INSTALL_ADVICE[kind]}); only "
                    f"needed if you select it with --alignment-tool."
                )
                for attempt in res.attempts:
                    recs.append(
                        f"{_DISPLAY[kind]} candidate {attempt.path} ({attempt.source}) "
                        f"was rejected: {attempt.reason}."
                    )
        for res in tools:
            if res.tool is not None:
                recs.extend(f"{_DISPLAY[res.kind]}: {w}." for w in res.tool.warnings)
    if not builtin_available and not any(r.tool for r in tools):
        recs.append("No usable aligner at all: install MAFFT (see above).")

    topo = host.topology
    if topo.smt_width > 1 and not plan.use_smt:
        recs.append(
            f"SMT detected ({topo.logical_count} logical / {topo.physical_count} physical "
            "usable CPUs): using physical cores only by default (provisional policy); "
            "pass --smt to use the siblings too."
        )
    if host.cpu_limit_notes and len(host.cpu_limit_notes) > 1:
        recs.append("CPU limits in effect: " + "; ".join(host.cpu_limit_notes) + ".")
    if host.cgroup_version and host.memory.cgroup_limit_bytes is not None:
        recs.append(
            f"A cgroup memory limit of {format_bytes(host.memory.cgroup_limit_bytes)} applies; "
            f"the memory envelope is {format_bytes(plan.memory_envelope_bytes)}."
        )
    if host.memory.effective_bytes is None:
        recs.append(
            "Memory availability could not be determined; pass --memory-limit to set a budget."
        )
    elif plan.memory_envelope_bytes < 1024**3:
        recs.append(
            f"Only {format_bytes(plan.memory_envelope_bytes)} of memory is budgeted; large "
            "datasets will run with few workers (free memory or raise --memory-limit)."
        )
    ram = host.ram_tiers
    if ram and plan.scratch_tier is not None and not plan.scratch_tier.is_ram:
        best = max(ram, key=lambda t: t.free_bytes or 0)
        recs.append(
            f"RAM-disk {best.path} has only {format_bytes(best.free_bytes)} free "
            f"(< {format_bytes(MIN_RAM_SCRATCH_FREE_BYTES)} usable); scratch falls back to "
            f"{plan.scratch_tier.path}. Enlarge tmpfs or set PBA_SCRATCH to fast local disk."
        )
    if plan.scratch_tier is None:
        recs.append(
            "No writable scratch location detected; the system temp directory will be "
            "used. Set PBA_SCRATCH or $SLURM_TMPDIR to fast node-local storage."
        )
    elif plan.scratch_tier.is_network:
        recs.append(
            f"Scratch {plan.scratch_tier.path} is on a network filesystem "
            f"({plan.scratch_tier.fstype}); use node-local storage ($SLURM_TMPDIR, NVMe) "
            "for much faster I/O."
        )
    for name, version in dependencies.items():
        if version is None and name in {"biopython", "numpy", "psutil"}:
            recs.append(
                f"Required dependency {name} is not installed (pip install {name})."
            )
    if host.slurm_job:
        recs.append(
            "SLURM allocation detected: CPU/memory budgets follow the allocation; request "
            "node-local scratch ($SLURM_TMPDIR) for best performance."
        )
    recs.extend(f"Plan warning: {w}" for w in plan.warnings)
    recs.extend(f"Host probe: {w}" for w in host.warnings)
    return list(dict.fromkeys(recs))


def run_doctor(
    root: str = "/",
    env: Mapping[str, str] | None = None,
    *,
    host: HostProfile | None = None,
    num_threads: int | str = "auto",
    smt: bool | None = None,
    memory_limit: int | str | None = None,
) -> DoctorReport:
    """Diagnose the environment.

    Args:
        root: Filesystem root for ``/proc``/``/sys`` probing (fixture trees in tests).
        env: Environment mapping (``os.environ`` by default).
        host: A pre-built profile (skips detection; used by tests).
        num_threads, smt, memory_limit: Forwarded to :func:`autotune`.

    Never raises for host or tool problems; they become report content.
    """
    if host is None:
        host = detect_host(root, env, refresh=True)
    plan = autotune(host, num_threads=num_threads, smt=smt, memory_limit=memory_limit)
    environment: Mapping[str, str] | None = env
    tools = tuple(
        resolver.resolve_aligner_detailed(kind, env=environment)
        for kind in resolver.KINDS
    )
    builtin_ok, builtin_note = _check_builtin()
    forced = resolver.builtin_forced(environment)
    dependencies = _dependency_versions()
    recs = _recommendations(host, plan, tools, builtin_ok, forced, dependencies)
    return DoctorReport(
        host=host,
        plan=plan,
        tools=tools,
        builtin_available=builtin_ok,
        builtin_note=builtin_note,
        forced_builtin=forced,
        dependencies=dependencies,
        recommendations=tuple(recs),
        python=f"{platform.python_implementation()} {platform.python_version()} ({sys.executable})",
        package_version=_package_version(),
    )


def _tool_lines(res: resolver.Resolution) -> list[str]:
    name = _DISPLAY[res.kind]
    if res.forced_builtin:
        return [f"  {name:<14} skipped ({resolver.FORCE_BUILTIN_ENV} is set)"]
    if res.tool is not None:
        t = res.tool
        lines = [f"  {name:<14} OK  v{t.version}  {t.path}  [{t.source}]"]
        lines.extend(f"      warning: {w}" for w in t.warnings)
        return lines
    need = resolver.format_version(res.min_version) if res.min_version else "any"
    lines = [f"  {name:<14} NOT FOUND (needs >= {need})"]
    lines.extend(
        f"      rejected {a.path} [{a.source}]: {a.reason}" for a in res.attempts
    )
    return lines


def format_report(report: DoctorReport) -> str:
    """Render a :class:`DoctorReport` as plain text."""
    host, plan = report.host, report.plan
    out: list[str] = [
        f"PlastBurstAlign doctor - v{report.package_version}",
        f"Python: {report.python}",
        "",
        "== Host ==",
        host.summary(),
        "",
        "== Tuning plan (num_threads='auto') ==",
        f"  CPU budget: {plan.cpu_budget} thread(s), {plan.workers} concurrent task(s), "
        f"<= {plan.threads_per_task_cap} thread(s) per task, SMT {'on' if plan.use_smt else 'off'}",
        f"  Memory envelope: {format_bytes(plan.memory_envelope_bytes)}",
        f"  Scratch: {plan.scratch_path or 'system temp (none detected)'}"
        + (
            f" (fallback {plan.scratch_fallback_path})"
            if plan.scratch_fallback_path
            else ""
        ),
        f"  Pinning: {'recommended' if plan.pinning else 'not needed'}",
        "",
        "== Aligners ==",
    ]
    for res in report.tools:
        out.extend(_tool_lines(res))
    out.append(
        f"  {'Builtin':<14} {'available' if report.builtin_available else 'UNAVAILABLE'}"
        f" - {report.builtin_note}"
    )
    out.append("")
    out.append("== Scratch tiers ==")
    if host.storage:
        for tier in host.storage:
            rw = {True: "rw", False: "read-only", None: "?"}[tier.writable]
            out.append(
                f"  {tier.path:<28} {tier.kind:<8} {tier.device_class:<8} "
                f"{tier.fstype or '?':<8} free {format_bytes(tier.free_bytes):>10}  {rw}"
            )
    else:
        out.append("  (none detected)")
    out.append("")
    out.append("== Dependencies ==")
    for name, version in report.dependencies.items():
        out.append(f"  {name:<12} {version or 'not installed'}")
    out.append("")
    out.append("== Recommendations ==")
    if report.recommendations:
        out.extend(f"  - {r}" for r in report.recommendations)
    else:
        out.append("  - none: the environment looks healthy.")
    out.append("")
    out.append(
        "Status: "
        + ("OK" if report.ok else "PROBLEM - no usable aligner")
        + (
            ""
            if report.external_found
            else " (external aligner missing; builtin fallback only)"
        )
    )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """CLI entry point for ``plastburstalign --doctor`` (see module docstring)."""
    parser = argparse.ArgumentParser(
        prog="plastburstalign-doctor",
        description="Diagnose hardware, aligner binaries and scratch space.",
    )
    parser.add_argument(
        "--root", default="/", help="filesystem root to probe (default: /)"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if no external aligner is found (builtin fallback only)",
    )
    parser.add_argument("--num-threads", default="auto", help="'auto' or an integer")
    parser.add_argument("--smt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--memory-limit", default=None, help="e.g. 8G")
    args = parser.parse_args(list(argv) if argv is not None else None)
    stream = stdout if stdout is not None else sys.stdout
    try:
        report = run_doctor(
            root=args.root,
            num_threads=args.num_threads,
            smt=args.smt,
            memory_limit=args.memory_limit,
        )
    except ValueError as exc:  # bad --num-threads / --memory-limit
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(report.to_dict(), stream, indent=2, default=str)
        stream.write("\n")
    else:
        stream.write(format_report(report) + "\n")
    if not report.ok:
        return 1
    if args.strict and not report.external_found:
        return 1
    return 0
