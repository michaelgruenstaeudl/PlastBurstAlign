"""Run manifest (``out_dir/run_manifest.json``).

Written at the end of a monolithic run and of Slurm finalization. The writer
never raises: a manifest is diagnostic and must not fail an alignment that
already succeeded.

Timestamps live only in the ``run`` object. Everything else (parameters, host,
per-locus strategy, scheduler metrics) is reproducible data.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import os
import platform
from collections.abc import Mapping
from typing import Any

from .logging_ops import logger as log

MANIFEST_NAME = "run_manifest.json"


def _dist_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _jsonify(value: Any) -> Any:
    """Convert ``value`` into something ``json.dump`` can emit."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonify(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _params_view(params: Mapping[str, Any] | None) -> dict[str, Any]:
    if not params:
        return {}
    skip = {"num_threads_requested"}
    return {key: _jsonify(value) for key, value in params.items() if key not in skip}


def _versions(aligner_version: str | None, aligner_path: str | None) -> dict[str, Any]:
    from ._meta import __version__

    return {
        "plastburstalign": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _dist_version("numpy"),
        "biopython": _dist_version("biopython"),
        "aligner_version": aligner_version,
        "aligner_path": aligner_path,
    }


def _strategy_constants() -> dict[str, Any] | None:
    try:
        from .strategy_constants import DEFAULT_CONSTANTS

        return _jsonify(DEFAULT_CONSTANTS.to_dict())
    except Exception:
        return None


def _tuning_view(plan: Any) -> dict[str, Any] | None:
    if plan is None:
        return None
    try:
        data = plan.to_dict()
    except Exception:
        try:
            return {"explain": plan.explain()}
        except Exception:
            return None
    if isinstance(data, dict) and "host" in data:
        # The host profile is stored beside the plan; keep the plan readable.
        data = dict(data)
        data["host"] = {"summary": "see host"}
    return _jsonify(data)


def _host_view(plan: Any) -> dict[str, Any] | None:
    host = getattr(plan, "host", None) if plan is not None else None
    if host is None:
        return None
    try:
        return _jsonify(host.to_dict())
    except Exception:
        try:
            return {"summary": host.summary()}
        except Exception:
            return None


def _locus_sort_key(record: Mapping[str, Any]) -> tuple[str, str]:
    """Deterministic order for ``loci``: region name, then mode."""
    return (str(record.get("region") or ""), str(record.get("mode") or ""))


def write_run_manifest(
    out_dir: str,
    *,
    started_at: datetime.datetime,
    wall_seconds: float,
    params: Mapping[str, Any] | None = None,
    plan: Any = None,
    locus_records: list[dict[str, Any]] | None = None,
    scheduler_report: Any = None,
    scratch_stats: Mapping[str, Any] | None = None,
    supermatrix: Mapping[str, Any] | None = None,
    aligner_version: str | None = None,
    aligner_path: str | None = None,
    ok: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> str | None:
    """Write ``out_dir/run_manifest.json``. Returns the path, or ``None`` on failure.

    Never raises. ``loci`` are sorted by ``(region, mode)`` so raw JSON is stable
    across thread counts even when completion order differs.
    """
    try:
        finished = datetime.datetime.now(datetime.timezone.utc)
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=datetime.timezone.utc)
        scheduler: dict[str, Any] | None = None
        if scheduler_report is not None:
            try:
                scheduler = _jsonify(scheduler_report.to_dict())
                failed = getattr(scheduler_report, "failed", None)
                if failed:
                    scheduler["failed_units"] = [
                        _jsonify(record.to_dict()) for record in failed
                    ]
            except Exception as exc:
                scheduler = {"error": str(exc)}
        loci = sorted(list(locus_records or []), key=_locus_sort_key)
        document: dict[str, Any] = {
            "run": {
                "started_at": started_at.isoformat(),
                "finished_at": finished.isoformat(),
                "wall_seconds": round(max(0.0, float(wall_seconds)), 6),
                "ok": bool(ok),
            },
            "versions": _versions(aligner_version, aligner_path),
            "params": _params_view(params),
            "host": _host_view(plan),
            "tuning": _tuning_view(plan),
            "strategy_constants": _strategy_constants(),
            "loci": _jsonify(loci),
            "scheduler": scheduler,
            "scratch": _jsonify(dict(scratch_stats)) if scratch_stats else None,
            "supermatrix": _jsonify(dict(supermatrix)) if supermatrix else None,
        }
        if extra:
            for key, value in extra.items():
                if key == "run":
                    continue
                document[str(key)] = _jsonify(value)
        path = os.path.join(out_dir, MANIFEST_NAME)
        os.makedirs(out_dir, exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
        log.info(f"run manifest written to {path}")
        return path
    except Exception as exc:
        log.warning(f"could not write run manifest: {exc}")
        return None
