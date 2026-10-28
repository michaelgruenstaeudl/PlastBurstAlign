"""Concurrency utilities and multiprocessing context handling."""

import multiprocessing
import os
import sys
from multiprocessing.context import BaseContext


def get_mp_context(force_context: str | None = None) -> BaseContext:
    """
    Returns an appropriate multiprocessing context based on platform and configuration.

    - Windows (win32): Defaults to 'spawn'.
    - macOS (darwin): Defaults to 'spawn' (due to crash risks with fork in macOS frameworks).
    - Linux/Unix: Defaults to 'fork', but supports 'spawn' via PBA_MP_CONTEXT environment variable.
    """
    if force_context:
        return multiprocessing.get_context(force_context)

    env_override = os.environ.get("PBA_MP_CONTEXT")
    if env_override:
        return multiprocessing.get_context(env_override)

    if sys.platform in ("win32", "darwin"):
        return multiprocessing.get_context("spawn")

    return multiprocessing.get_context("fork")
