"""Package metadata shared by `__init__` and `logging_ops`.

Kept in its own module so that both can import it without depending on the order
in which `plastburstalign/__init__.py` executes its imports.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__author__ = "Michael Gruenstaeudl, PhD"
__email__ = "m_gruenstaeudl@fhsu.edu"

try:
    __version__: str = version("plastburstalign")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"
