"""silica.core.logger — stdlib logging scoped to the `silica.*` namespace.

Follows P-0 Strategy: stdlib `logging` plus a colored formatter, no structlog
or other third-party logging library. Configuration is scoped to the `silica`
logger subtree so silica does not hijack the root logger or affect loggers
owned by MLX, pydantic, or other dependencies.

Usage:

    from silica.core.logger import setup_logging, get_logger
    setup_logging(level="INFO")              # once at startup
    log = get_logger(__name__)
    log.info("engine ready")
"""

from __future__ import annotations

import logging
import os
import sys
from typing import IO

_SILICA_ROOT = "silica"

_ANSI_RESET = "\x1b[0m"
_ANSI_COLORS: dict[int, str] = {
    logging.DEBUG: "\x1b[2;37m",     # dim gray
    logging.INFO: "\x1b[36m",        # cyan
    logging.WARNING: "\x1b[33m",     # yellow
    logging.ERROR: "\x1b[31m",       # red
    logging.CRITICAL: "\x1b[1;35m",  # bold magenta
}


class _ColorFormatter(logging.Formatter):
    """Colorizes the level field when `use_color` is True.

    Output shape: `YYYY-MM-DD HH:MM:SS | LEVEL    | logger.name | message`
    """

    default_time_format = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_color: bool) -> None:
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        level_str = f"{record.levelname:<8}"
        if self._use_color:
            color = _ANSI_COLORS.get(record.levelno, "")
            level_str = f"{color}{level_str}{_ANSI_RESET}"
        timestamp = self.formatTime(record, self.default_time_format)
        return f"{timestamp} | {level_str} | {record.name} | {record.getMessage()}"


class _SilicaStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Marker subclass so `setup_logging` can replace its own handler idempotently."""


def setup_logging(
    level: str | int = "INFO",
    use_colors: bool | None = None,
    stream: IO[str] | None = None,
) -> None:
    """Install the silica colored handler on the `silica` logger namespace.

    Idempotent: removes any previously-installed silica handler before adding
    a new one. Does not touch the root logger or handlers owned by other
    libraries. `propagate` is disabled so silica messages do not bubble up to
    the root logger (which may be configured differently by the application).

    Color auto-detect precedence (when `use_colors=None`):
      1. Stream is a TTY.
      2. `NO_COLOR` environment variable is unset (no-color.org convention).
    """
    target_stream: IO[str] = stream if stream is not None else sys.stderr
    if use_colors is None:
        use_colors = (
            hasattr(target_stream, "isatty")
            and target_stream.isatty()
            and not os.environ.get("NO_COLOR")
        )

    root = logging.getLogger(_SILICA_ROOT)
    root.setLevel(level)
    for existing in list(root.handlers):
        if isinstance(existing, _SilicaStreamHandler):
            root.removeHandler(existing)

    handler = _SilicaStreamHandler(target_stream)
    handler.setFormatter(_ColorFormatter(use_color=use_colors))
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the silica namespace.

    Convention: pass `__name__` from each silica module. Names already under
    the silica namespace (`silica`, `silica.foo.bar`) pass through; external
    names (`foo.bar`) are prefixed to become `silica.foo.bar`.
    """
    if name == _SILICA_ROOT or name.startswith(_SILICA_ROOT + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_SILICA_ROOT}.{name}")
