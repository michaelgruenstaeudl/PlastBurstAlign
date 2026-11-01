"""PlastBurstAlign package.

Public library surface (also available as ``import plastburstalign as pba``):

* :func:`align` / :class:`AlignmentResult` / :func:`estimate` / :func:`autotune`
* legacy :class:`PlastBurstAlign` / :class:`UserParameters`

``estimate`` and ``autotune`` are the API callables. The implementation modules
remain importable as ``plastburstalign.estimate`` / ``plastburstalign.autotune``
via :mod:`importlib` (``sys.modules``); prefer ``from plastburstalign.estimate
import preflight_estimate`` inside the package and tests.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._meta import __author__, __email__, __version__
from .alignment_ops import AlignmentCoordination
from .alignment_tools import MAFFT, MUSCLE, ClustalOmega
from .api import AlignmentResult, align
from .api import estimate as estimate
from .autotune import TuningPlan
from .autotune import autotune as autotune
from .extraction_ops import DataCleaning, ExtractAndCollect
from .hardware import HostProfile, detect_host
from .plast_burst_align import PlastBurstAlign
from .seqfeature_ops import PlastidData
from .user_parameters import UserParameters

# Ensure submodule objects stay reachable for ``importlib.import_module`` and
# for ``from plastburstalign.estimate import ...`` even though the package
# attributes ``estimate`` / ``autotune`` are the public API callables.
import_module("plastburstalign.estimate")
import_module("plastburstalign.autotune")

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "PlastBurstAlign",
    "UserParameters",
    "PlastidData",
    "ExtractAndCollect",
    "DataCleaning",
    "AlignmentCoordination",
    "MAFFT",
    "ClustalOmega",
    "MUSCLE",
    "align",
    "autotune",
    "estimate",
    "AlignmentResult",
    "TuningPlan",
    "HostProfile",
    "detect_host",
]


def __getattr__(name: str) -> Any:
    # Escape hatch: ``plastburstalign.estimate_module`` / ``autotune_module``
    # when a caller needs the submodule object after the API callable binding.
    if name == "estimate_module":
        return import_module("plastburstalign.estimate")
    if name == "autotune_module":
        return import_module("plastburstalign.autotune")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
