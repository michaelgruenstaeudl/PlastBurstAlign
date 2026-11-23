"""Locate and *validate* external aligner executables (MAFFT, MUSCLE, Clustal Omega).

PlastBurstAlign ships **no bundled binaries**. Instead :func:`resolve_aligner`
walks a fixed search ladder and returns the first candidate that actually
works, falling through to the pure-Python
:class:`~plastburstalign.builtin_aligner.BuiltinAligner` when none does.

Search ladder (first valid candidate wins)::

    1. explicit_path                       (source "explicit")
    2. $PBA_MAFFT / $PBA_MUSCLE / $PBA_CLUSTALO   (source "env:PBA_MAFFT", ...)
    3. every executable on $PATH           (source "PATH")
    4. active virtualenv / conda prefix    ($VIRTUAL_ENV, $CONDA_PREFIX, sys.prefix)
    5. HPC module variables                (EBROOTMAFFT, MAFFT_HOME, ...)

Setting ``PBA_FORCE_BUILTIN_ALIGNER=1`` skips *all* external tools (used by the
CI "no aligner installed" leg and for reproducible fallback testing).

A candidate is only accepted after a **version probe** executed with a short
timeout: non-executable files, wrong-architecture / missing-library / crashing
binaries (``OSError``, signals, non-zero exit) and binaries that do not print a
parsable version are rejected, and the reason is recorded. Versions are parsed
into comparable integer tuples and checked against configurable minimums.

Why these minimum versions (``DEFAULT_MIN_VERSIONS``, all overridable)?

* **MAFFT >= 7.310** - PlastBurstAlign relies on the 7.3xx command-line
  surface: ``--thread N``, the ``--parttree --retree N --nofft`` family,
  ``--addfragments`` and ``--adjustdirection``. 7.310 (2017) is a conservative
  project-chosen floor inside the 7.3 series where these options are documented
  together; the package is developed and tested against 7.5xx. Older 6.x/7.0-7.2
  builds are rejected rather than risking silently different flag semantics.
* **MUSCLE >= 5.0** - the wrapper uses MUSCLE 5 syntax (``-super5``,
  ``-output``); MUSCLE 3.x has an incompatible command line.
* **Clustal Omega >= 1.2.0** - the wrapper uses ``--infile/--outfile/--outfmt``
  and ``--thread``, stable across the 1.2 series.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .logging_ops import logger as log

__all__ = [
    "DEFAULT_MIN_VERSIONS",
    "ENV_OVERRIDE",
    "FORCE_BUILTIN_ENV",
    "KINDS",
    "CandidateAttempt",
    "Resolution",
    "ResolvedTool",
    "builtin_forced",
    "clear_cache",
    "format_version",
    "normalize_kind",
    "parse_version",
    "probe_version",
    "resolve_aligner",
    "resolve_aligner_detailed",
]

KINDS: tuple[str, ...] = ("mafft", "muscle", "clustalo")

_ALIASES = {
    "mafft": "mafft",
    "muscle": "muscle",
    "muscle5": "muscle",
    "clustalo": "clustalo",
    "clustal": "clustalo",
    "clustal-omega": "clustalo",
    "clustalomega": "clustalo",
    "clustal_omega": "clustalo",
}

#: Minimum accepted version per tool (configurable per call via ``min_version``).
DEFAULT_MIN_VERSIONS: dict[str, tuple[int, ...]] = {
    "mafft": (7, 310),
    "muscle": (5, 0),
    "clustalo": (1, 2, 0),
}

#: Environment variable overriding the executable per tool.
ENV_OVERRIDE: dict[str, str] = {
    "mafft": "PBA_MAFFT",
    "muscle": "PBA_MUSCLE",
    "clustalo": "PBA_CLUSTALO",
}

#: Set to a truthy value to skip every external aligner.
FORCE_BUILTIN_ENV = "PBA_FORCE_BUILTIN_ALIGNER"

#: Seconds allowed for one version probe.
PROBE_TIMEOUT_SECONDS = 10.0

_EXECUTABLE_NAMES: dict[str, tuple[str, ...]] = {
    "mafft": ("mafft",),
    "muscle": ("muscle", "muscle5"),
    "clustalo": ("clustalo", "clustal-omega"),
}

#: Variables set by environment-module systems (Lmod/EasyBuild/Environment Modules).
#: ``EBROOT<NAME>`` is the EasyBuild install prefix; the others are common conventions.
_MODULE_ENV_VARS: dict[str, tuple[str, ...]] = {
    "mafft": ("EBROOTMAFFT", "MAFFT_HOME", "MAFFT_ROOT", "MAFFT_DIR", "MAFFT_BIN"),
    "muscle": (
        "EBROOTMUSCLE",
        "MUSCLE_HOME",
        "MUSCLE_ROOT",
        "MUSCLE_DIR",
        "MUSCLE_BIN",
    ),
    "clustalo": (
        "EBROOTCLUSTALMINOMEGA",
        "EBROOTCLUSTALO",
        "CLUSTALO_HOME",
        "CLUSTALO_ROOT",
        "CLUSTALO_DIR",
        "CLUSTALO_BIN",
    ),
}

_VERSION_FLAGS: dict[str, tuple[str, ...]] = {
    "mafft": ("--version",),
    "muscle": ("-version", "--version"),
    "clustalo": ("--version",),
}

# MAFFT prints "v7.526 (2024/Apr/26)" to STDERR; MUSCLE 5 "muscle 5.3.linux64 []";
# MUSCLE 3 "MUSCLE v3.8.31 by Robert C. Edgar"; Clustal Omega "1.2.4".
_VERSION_PATTERNS: dict[str, re.Pattern[str]] = {
    "mafft": re.compile(r"v(\d+(?:\.\d+)+)"),
    "muscle": re.compile(r"(?i)muscle\s+v?(\d+(?:\.\d+)+)"),
    "clustalo": re.compile(r"(?<![\w.])(\d+\.\d+(?:\.\d+)*)"),
}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ResolvedTool:
    """A validated external aligner.

    Attributes:
        kind: ``"mafft"``, ``"muscle"`` or ``"clustalo"``.
        path: Absolute path of the executable that passed validation.
        version: Version string as parsed (``"7.526"``).
        version_tuple: Comparable form (``(7, 526)``).
        source: Where it was found (``"explicit"``, ``"env:PBA_MAFFT"``,
            ``"PATH"``, ``"CONDA_PREFIX"``, ``"VIRTUAL_ENV"``, ``"sys.prefix"``,
            ``"module:EBROOTMAFFT"``, ...).
        warnings: Higher-priority candidates that were rejected, and why.
    """

    kind: str
    path: str
    version: str
    version_tuple: tuple[int, ...]
    source: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateAttempt:
    """One candidate examined by the resolver."""

    source: str
    path: str
    accepted: bool
    reason: str = ""
    version: str | None = None


@dataclass(frozen=True)
class Resolution:
    """Full outcome of a resolve call (used by ``--doctor``)."""

    kind: str
    tool: ResolvedTool | None
    attempts: tuple[CandidateAttempt, ...]
    min_version: tuple[int, ...]
    forced_builtin: bool = False


# --------------------------------------------------------------------------- #
# Versions
# --------------------------------------------------------------------------- #


def parse_version(text: str | Sequence[int]) -> tuple[int, ...]:
    """Parse ``"7.526"``/``"v1.2.4"`` (or pass through an int sequence) to a tuple.

    Raises:
        ValueError: If no dotted/plain numeric version is found.
    """
    if not isinstance(text, str):
        return tuple(int(x) for x in text)
    match = re.search(r"\d+(?:\.\d+)*", text)
    if not match:
        raise ValueError(f"no version number in {text!r}")
    return tuple(int(part) for part in match.group(0).split("."))


def format_version(version: Sequence[int]) -> str:
    """Inverse of :func:`parse_version` for display (``(7, 526)`` -> ``"7.526"``)."""
    return ".".join(str(int(v)) for v in version)


def normalize_kind(kind: str) -> str:
    """Map aliases (``"clustal"``, ``"clustal-omega"``) to the canonical kind.

    Raises:
        ValueError: For an unknown tool name.
    """
    key = kind.strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"unknown aligner kind {kind!r}; expected one of {KINDS}")
    return _ALIASES[key]


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #


def _probe_env(env: Mapping[str, str] | None) -> dict[str, str]:
    """Environment for the version probe.

    The caller's ``env`` entries override the live environment, except that the
    caller's ``PATH`` is *prepended* to (not substituted for) the live one, so
    wrapper scripts (MAFFT is a shell script) still find ``sed``/``uname``.
    """
    merged = dict(os.environ)
    if env is not None and env is not os.environ:
        merged.update({k: v for k, v in env.items() if k != "PATH"})
        parts = [p for p in (env.get("PATH"), os.environ.get("PATH")) if p]
        merged["PATH"] = os.pathsep.join(parts)
    return merged


def probe_version(
    kind: str,
    path: str,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> tuple[str | None, str]:
    """Run ``<path> <version flag>`` and parse the version.

    Returns:
        ``(version_string, "")`` on success, else ``(None, reason)``.
    """
    kind = normalize_kind(kind)
    pattern = _VERSION_PATTERNS[kind]
    last_reason = "no version output"
    for flag in _VERSION_FLAGS[kind]:
        try:
            proc = subprocess.run(
                [path, flag],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                timeout=timeout,
                env=_probe_env(env),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"version probe timed out after {timeout:g}s"
        except OSError as exc:  # wrong arch, bad interpreter, EACCES, ...
            return None, f"cannot execute: {exc.strerror or exc}"
        except Exception as exc:
            return None, f"version probe failed: {type(exc).__name__}: {exc}"
        output = f"{proc.stdout}\n{proc.stderr}"
        if proc.returncode < 0:
            return None, f"crashed with signal {-proc.returncode} during version probe"
        if proc.returncode != 0:
            first_line = output.strip().splitlines()[:1]
            hint = f" ({first_line[0][:80]})" if first_line else ""
            last_reason = f"`{flag}` exited with status {proc.returncode}{hint}"
            continue
        match = pattern.search(output)
        if match:
            return match.group(1), ""
        snippet = output.strip().replace("\n", " ")[:80]
        last_reason = f"no version number in output of `{flag}`: {snippet!r}"
    return None, last_reason


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def builtin_forced(env: Mapping[str, str] | None = None) -> bool:
    """``True`` if ``PBA_FORCE_BUILTIN_ALIGNER`` requests skipping external tools."""
    source = os.environ if env is None else env
    return _is_truthy(source.get(FORCE_BUILTIN_ENV))


def _executables_in(directory: str, kind: str) -> list[str]:
    found: list[str] = []
    for name in _EXECUTABLE_NAMES[kind]:
        hit = shutil.which(name, path=directory)
        if hit:
            found.append(hit)
    return found


def _expand_user_path(raw: str, kind: str, env: Mapping[str, str]) -> list[str]:
    """Turn an explicit/override value into candidate paths."""
    value = os.path.expanduser(raw.strip())
    if not value:
        return []
    if os.path.isdir(value):
        hits = _executables_in(value, kind) or _executables_in(
            os.path.join(value, "bin"), kind
        )
        return hits or [value]  # keep the directory so the rejection reason is reported
    if os.sep not in value and not (os.altsep and os.altsep in value):
        found = shutil.which(value, path=env.get("PATH", ""))
        return [found] if found else [value]
    return [value]


def _candidates(
    kind: str,
    explicit_path: str | None,
    env: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Ordered ``(source, path)`` candidates following the search ladder."""
    out: list[tuple[str, str]] = []
    if explicit_path:
        out.extend(
            ("explicit", p) for p in _expand_user_path(str(explicit_path), kind, env)
        )
    override_var = ENV_OVERRIDE[kind]
    override = env.get(override_var)
    if override:
        out.extend(
            (f"env:{override_var}", p) for p in _expand_user_path(override, kind, env)
        )
    for directory in (env.get("PATH") or "").split(os.pathsep):
        if directory:
            out.extend(("PATH", hit) for hit in _executables_in(directory, kind))
    prefixes: list[tuple[str, str | None]] = [
        ("VIRTUAL_ENV", env.get("VIRTUAL_ENV")),
        ("CONDA_PREFIX", env.get("CONDA_PREFIX")),
        ("sys.prefix", sys.prefix),
    ]
    for source, prefix in prefixes:
        if not prefix:
            continue
        for sub in ("bin", "Scripts", os.path.join("Library", "bin")):
            out.extend(
                (source, hit)
                for hit in _executables_in(os.path.join(prefix, sub), kind)
            )
    for var in _MODULE_ENV_VARS[kind]:
        root = env.get(var)
        if not root:
            continue
        for sub in ("bin", ""):
            directory = os.path.join(root, sub) if sub else root
            out.extend(
                (f"module:{var}", hit) for hit in _executables_in(directory, kind)
            )
    return out


# --------------------------------------------------------------------------- #
# Validation and resolution
# --------------------------------------------------------------------------- #


def _normalize_min(
    kind: str, min_version: str | Sequence[int] | None
) -> tuple[int, ...]:
    if min_version is None:
        return DEFAULT_MIN_VERSIONS[kind]
    if isinstance(min_version, str):
        text = min_version.strip()
        return parse_version(text) if text else ()
    return tuple(int(v) for v in min_version)


def _check_candidate(
    kind: str,
    source: str,
    path: str,
    min_version: tuple[int, ...],
    env: Mapping[str, str],
    timeout: float,
) -> tuple[CandidateAttempt, tuple[int, ...] | None]:
    if not os.path.exists(path):
        return CandidateAttempt(source, path, False, "does not exist"), None
    if os.path.isdir(path):
        return CandidateAttempt(
            source, path, False, "is a directory, not an executable"
        ), None
    if not os.access(path, os.X_OK):
        return CandidateAttempt(source, path, False, "not executable"), None
    version, reason = probe_version(kind, path, env=env, timeout=timeout)
    if version is None:
        return CandidateAttempt(source, path, False, reason), None
    try:
        parsed = parse_version(version)
    except ValueError:
        return CandidateAttempt(
            source, path, False, f"unparsable version {version!r}"
        ), None
    if min_version and parsed < min_version:
        need = format_version(min_version)
        return (
            CandidateAttempt(
                source,
                path,
                False,
                f"version {version} is older than the required {need}",
                version,
            ),
            None,
        )
    return CandidateAttempt(source, path, True, "", version), parsed


_CACHE: dict[tuple[object, ...], Resolution] = {}
_CACHE_LOCK = threading.Lock()


def clear_cache() -> None:
    """Forget memoized resolutions (e.g. after installing an aligner)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _env_signature(kind: str, env: Mapping[str, str]) -> tuple[str | None, ...]:
    keys = (
        "PATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        FORCE_BUILTIN_ENV,
        ENV_OVERRIDE[kind],
        *_MODULE_ENV_VARS[kind],
    )
    return tuple(env.get(k) for k in keys) + (sys.prefix,)


def resolve_aligner_detailed(
    kind: str,
    explicit_path: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    min_version: str | Sequence[int] | None = None,
    strict_explicit: bool = False,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> Resolution:
    """Like :func:`resolve_aligner` but returns every candidate examined.

    Results are memoized per ``(kind, explicit_path)`` *and* the inputs that
    influence them (minimum version, relevant environment variables, ``PATH``),
    so a changed environment never returns a stale answer. Call
    :func:`clear_cache` after installing a tool mid-session.
    """
    canonical = normalize_kind(kind)
    environment: Mapping[str, str] = os.environ if env is None else env
    minimum = _normalize_min(canonical, min_version)
    if builtin_forced(environment):
        return Resolution(canonical, None, (), minimum, forced_builtin=True)
    key = (
        canonical,
        explicit_path,
        minimum,
        strict_explicit,
        _env_signature(canonical, environment),
    )
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached

    attempts: list[CandidateAttempt] = []
    seen: set[str] = set()
    tool: ResolvedTool | None = None
    for source, path in _candidates(canonical, explicit_path, environment):
        try:
            identity = os.path.realpath(path)
        except OSError:
            identity = path
        if identity in seen:
            continue
        seen.add(identity)
        attempt, parsed = _check_candidate(
            canonical, source, path, minimum, environment, timeout
        )
        attempts.append(attempt)
        if attempt.accepted and parsed is not None and attempt.version is not None:
            warnings = tuple(
                f"{a.source} candidate {a.path} rejected: {a.reason}"
                for a in attempts[:-1]
            )
            tool = ResolvedTool(
                kind=canonical,
                path=os.path.abspath(path),
                version=attempt.version,
                version_tuple=parsed,
                source=source,
                warnings=warnings,
            )
            break
        if strict_explicit and source == "explicit":
            break  # an explicit path that fails is final
    if tool is None:
        log.debug(
            f"resolver: no valid {canonical} found ({len(attempts)} candidate(s) rejected)"
        )
    else:
        for w in tool.warnings:
            log.warning(f"{canonical}: {w}")
        log.debug(
            f"resolver: {canonical} {tool.version} at {tool.path} ({tool.source})"
        )
    result = Resolution(canonical, tool, tuple(attempts), minimum)
    with _CACHE_LOCK:
        _CACHE[key] = result
    return result


def resolve_aligner(
    kind: str,
    explicit_path: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    min_version: str | Sequence[int] | None = None,
    strict_explicit: bool = False,
) -> ResolvedTool | None:
    """Find a *working* aligner of ``kind`` or return ``None``.

    Args:
        kind: ``"mafft"``, ``"muscle"``, ``"clustalo"`` (aliases ``"clustal"``,
            ``"clustal-omega"`` accepted).
        explicit_path: User-supplied executable (or a directory containing it).
        env: Environment mapping (``os.environ`` by default). ``PATH``,
            ``VIRTUAL_ENV``, ``CONDA_PREFIX``, ``PBA_<TOOL>``, module variables
            and ``PBA_FORCE_BUILTIN_ALIGNER`` are read from it.
        min_version: Minimum acceptable version as ``"7.310"`` or ``(7, 310)``;
            ``None`` uses :data:`DEFAULT_MIN_VERSIONS`; ``()``/``""`` disables
            the check.
        strict_explicit: If true, a failing ``explicit_path`` yields ``None``
            instead of falling through the rest of the ladder.

    Returns:
        A :class:`ResolvedTool` (path, version, source, rejection warnings) or
        ``None`` - in which case callers fall back to ``BuiltinAligner``.
        Always ``None`` when ``PBA_FORCE_BUILTIN_ALIGNER`` is truthy.
    """
    return resolve_aligner_detailed(
        kind,
        explicit_path,
        env=env,
        min_version=min_version,
        strict_explicit=strict_explicit,
    ).tool
