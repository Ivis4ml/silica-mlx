"""silica.chat.cli — chat CLI assembly (Side track 2).

Module layout per ``docs/CHAT_CLI_OPENING.md`` §7:

- :mod:`silica.chat.cli.palette` — ANSI colour constants + capability
  detection (24-bit / 8-colour / NO_COLOR fallback).
- :mod:`silica.chat.cli.state` — :class:`ChatCliState` dataclass
  carrying live engine metrics, conversation log, configuration.
- :mod:`silica.chat.cli.toolbar` — pure formatter that renders the
  bottom-of-terminal status line from a :class:`ChatCliState`.

Later sub-units will add :mod:`silica.chat.cli.commands` (slash
command dispatch), :mod:`silica.chat.cli.formatter` (conversation
log + code-fence syntax highlight), :mod:`silica.chat.cli.config`
(``/config`` parser), and :mod:`silica.chat.cli.app` (the
prompt_toolkit application shell).

This package is structured so the data + formatting layers
(palette / state / toolbar / commands / config) are pure-Python
and unit-testable; only :mod:`silica.chat.cli.app` carries a
prompt_toolkit dependency, isolated for manual validation.
"""

from silica.chat.cli.palette import (
    Palette,
    detect_palette,
)
from silica.chat.cli.state import (
    ChatCliState,
    StreamState,
)
from silica.chat.cli.toolbar import render_toolbar

__all__ = [
    "ChatCliState",
    "Palette",
    "StreamState",
    "detect_palette",
    "render_toolbar",
]
