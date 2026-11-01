from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import coloredlogs

# Package imports
from ._meta import __email__, __version__

if TYPE_CHECKING:  # ``rich`` is imported lazily, only when a live view is used
    from rich.console import Console

#: Attribute stamped on the ``RichHandler`` installed by :func:`rich_logging`
#: (value: pid of the installing process).
_RICH_PID_ATTR = "_pba_rich_pid"


def _discard_foreign_rich_handlers(target: logging.Logger) -> None:
    """Drop ``rich_logging`` handlers that were installed by *another* process.

    A forked worker inherits the parent's swapped-in ``RichHandler`` (which
    writes to the parent's live console). ``Logger.reinitialize_logger`` calls
    this first so the child ends up with a plain ``coloredlogs`` handler only.
    A no-op unless such a handler exists.
    """
    pid = os.getpid()
    for handler in list(target.handlers):
        owner = getattr(handler, _RICH_PID_ATTR, None)
        if owner is not None and owner != pid:
            target.removeHandler(handler)


class Logger:
    """
    A singleton class used to contain the `Logger` object used for all modules in the package.
    """

    _logger: tuple[int, Logger] | None = None
    logger: logging.Logger

    def __new__(cls, verbose: bool = False) -> Logger:
        """
        Creates a new `Logger` object for the class if one does not already exist,
        or if the current process is different from the process that originally created the `Logger`.
        Otherwise, the existing `Logger` is used.

        By default, the logging level is `logging.INFO`.

        Args:
            verbose: The level of logging: (False = `INFO`, True = `DEBUG`).
        """
        pid = os.getpid()
        current = cls._logger
        if current is None or current[0] != pid:
            instance = super().__new__(cls)
            cls._logger = (pid, instance)
            instance._setup_logger(verbose)
            return instance
        return current[1]

    @classmethod
    def _setup_logger(cls, verbose: bool = False):
        cls.logger = logging.getLogger(__name__)
        _discard_foreign_rich_handlers(cls.logger)
        pid = cls._logger[0] if cls._logger is not None else os.getpid()
        process_str = f" (process {pid}) " if verbose else " "
        log_format = f"%(asctime)s{process_str}[%(levelname)s] %(message)s"
        log_level = logging.DEBUG if verbose else logging.INFO
        coloredlogs.install(fmt=log_format, level=log_level, logger=cls.logger)

    @classmethod
    def reinitialize_logger(cls, verbose: bool = False):
        """
        Reinitialize the logger for child processes in a multiprocessing context.

        Args:
            verbose: The level of logging: (False = `INFO`, True = `DEBUG`).
        """
        cls._logger = None
        cls.__new__(cls, verbose)

    @classmethod
    def set_logger(cls, verbose: bool):
        """
        Mutator method for updating the `Logger` object.

        Args:
            verbose: The level of logging: (False = `INFO`, True = `DEBUG`).
        """
        cls._setup_logger(verbose)
        cls.logger.debug(f"{__package__} {__email__}|{__version__}")

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """
        Accessor method for retrieving the Logger object.

        Returns:
            The `Logger` object contained by the class.

        """

        return cls.logger


logger = Logger().get_logger()


def _is_console_stream_handler(handler: logging.Handler) -> bool:
    """True for plain terminal stream handlers (what ``coloredlogs`` installs)."""
    return isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, logging.FileHandler
    )


@contextmanager
def rich_logging(console: Console, verbose: bool = False) -> Iterator[logging.Logger]:
    """Route the package logger through ``rich`` while a live view is active.

    A ``rich.live.Live`` display and a plain stream handler writing to the same
    terminal corrupt each other. Inside this context the ``coloredlogs`` stream
    handler(s) of the package logger are replaced by one
    ``rich.logging.RichHandler(console=console)``, so log lines are printed
    *above* the live display. On exit (also on an exception) the replaced
    handlers and the logger level are restored.

    * Other handlers (file handlers, ``assertLogs``/pytest capture, ...) are left
      alone, including handlers attached while the context is active.
    * Re-entrant: contexts must be properly nested; an inner context swaps the
      outer ``RichHandler`` for its own and restores it afterwards.
    * If the logger was re-created while swapped (``Logger.set_logger``), a
      restored handler is skipped when a handler of the same class is already
      attached, so lines are not printed twice.
    * A forked child that calls ``Logger.reinitialize_logger`` gets a plain
      ``coloredlogs`` handler (see :func:`_discard_foreign_rich_handlers`).

    Args:
        console: The console that owns the live display.
        verbose: Raise the logger (and the rich handler) to ``DEBUG`` and prefix
            lines with the process id, like the ``verbose`` mode of ``Logger``.

    Yields:
        The package logger.
    """
    from rich.logging import RichHandler

    target = logging.getLogger(__name__)
    saved_level = target.level
    removed = [
        h
        for h in list(target.handlers)
        if _is_console_stream_handler(h) or hasattr(h, _RICH_PID_ATTR)
    ]
    handler = RichHandler(
        console=console,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
        log_time_format="[%H:%M:%S]",
    )
    setattr(handler, _RICH_PID_ATTR, os.getpid())
    if verbose:
        handler.setFormatter(logging.Formatter("(process %(process)d) %(message)s"))
        handler.setLevel(logging.DEBUG)
        target.setLevel(logging.DEBUG)
    for old in removed:
        target.removeHandler(old)
    target.addHandler(handler)
    try:
        yield target
    finally:
        target.removeHandler(handler)
        present = {type(h) for h in target.handlers}
        for old in removed:
            if old not in target.handlers and type(old) not in present:
                target.addHandler(old)
        target.setLevel(saved_level)
