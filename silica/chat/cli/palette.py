"""silica.chat.cli.palette — ANSI colour palette and capability detection.

Encodes the colour scheme defined in ``docs/CHAT_CLI_OPENING.md``
§4.1. Three rendering modes:

- **24-bit** — full RGB ANSI escapes, used when the terminal
  declares ``COLORTERM=truecolor`` or ``COLORTERM=24bit``.
- **256-colour fallback** — 8-bit indexed escapes, used when the
  terminal only signals ``TERM`` (xterm-256color, screen-256color,
  etc.) without truecolor.
- **plain** — no escape codes, returned when ``NO_COLOR`` is set
  to any non-empty value, or when the output is not a TTY, or when
  ``TERM=dumb`` / ``TERM`` is unset.

The :func:`detect_palette` helper inspects the environment once at
construction time; runtime colour rendering is then a pure function
of the :class:`Palette` instance, so unit tests can drive any mode
deterministically by constructing the desired :class:`Palette`
directly without touching ``os.environ``.

Colour identity (per the design doc):

- **green** — user voice (``You ›`` prompt prefix, user-text echo,
  ``state=decode`` toolbar field).
- **orange** — silica brand (``silica ›`` reply prefix, ``MLX``
  badge, ``compr=`` codec callout).
- **cyan** — architectural-win signals (``tok/s``, ``prefix_hit=``).
- **magenta** — model reasoning (``state=thinking``).
- **yellow** — engine waiting on prompt (``state=prefill``).
- **dim grey** — secondary stats / idle / dim text.
- **red** — errors.

24-bit RGB anchors were picked for legibility on both light and
dark terminals; the 256-colour fallback maps each anchor to the
nearest indexed entry. The ``Palette`` class exposes both per-named
colour escape strings and a :meth:`Palette.colorize` helper.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal

# ---------------------------------------------------------------------------
# Mode + colour name enumerations
# ---------------------------------------------------------------------------


class PaletteMode(str, Enum):
    """Rendering mode for the palette.

    The string values are stable identifiers used in tests and
    debug output; pick them rather than relying on enum identity.
    """

    PLAIN = "plain"
    INDEXED_256 = "indexed-256"
    TRUECOLOR = "truecolor"


# Named colours. Each name is a stable key surfaced by
# :class:`Palette` and its serialised form.
ColorName = Literal[
    "green",
    "green_dim",
    "orange",
    "cyan",
    "magenta",
    "yellow",
    "red",
    "grey",
    "white",
    "default",
]


# ---------------------------------------------------------------------------
# Anchors — 24-bit RGB and 256-colour-indexed equivalents
# ---------------------------------------------------------------------------

# RGB anchors. Picked from a tested-on-Terminal.app + iTerm2 +
# Alacritty pass; values aim for readability against both common
# dark and light backgrounds.
_RGB: dict[ColorName, tuple[int, int, int]] = {
    "green": (102, 217, 102),       # bright leaf
    "green_dim": (76, 153, 76),     # darker green for echoed user text
    "orange": (255, 153, 51),       # silica brand
    "cyan": (102, 217, 217),        # signal cyan
    "magenta": (217, 102, 217),     # thinking
    "yellow": (255, 204, 0),        # waiting / prefill
    "red": (242, 95, 87),           # error
    "grey": (136, 136, 136),        # dim text / idle
    "white": (240, 240, 240),       # primary numeric
    "default": (0, 0, 0),           # placeholder; "default" maps
                                    # to the terminal default
                                    # foreground via ESC [39m, never
                                    # used as RGB.
}

# 256-colour-indexed equivalents (xterm 256-colour palette
# numbering). Picked by visual proximity to the RGB anchors above
# under a typical xterm-256color rendering.
_INDEXED_256: dict[ColorName, int] = {
    "green": 119,
    "green_dim": 71,
    "orange": 215,
    "cyan": 80,
    "magenta": 177,
    "yellow": 220,
    "red": 203,
    "grey": 244,
    "white": 255,
    "default": -1,  # sentinel — emit ESC [39m
}


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_DEFAULT_FG = "\x1b[39m"


@dataclass(frozen=True)
class Palette:
    """Rendering palette constructed once per chat session.

    The ``mode`` field controls which escape sequences come out of
    :meth:`fg` and :meth:`colorize`. Pure-Python; safe to construct
    directly from tests without touching the environment.
    """

    mode: PaletteMode

    @classmethod
    def plain(cls) -> Palette:
        """Construct a no-colour palette — useful for tests and
        ``NO_COLOR=1`` environments."""
        return cls(mode=PaletteMode.PLAIN)

    @classmethod
    def truecolor(cls) -> Palette:
        return cls(mode=PaletteMode.TRUECOLOR)

    @classmethod
    def indexed_256(cls) -> Palette:
        return cls(mode=PaletteMode.INDEXED_256)

    # -- escape generation -------------------------------------------------

    def fg(self, color: ColorName) -> str:
        """Return the ANSI escape sequence that switches the
        foreground to ``color`` (no reset). Returns ``""`` under
        plain mode."""
        if self.mode is PaletteMode.PLAIN:
            return ""
        if color == "default":
            return _DEFAULT_FG
        if self.mode is PaletteMode.TRUECOLOR:
            r, g, b = _RGB[color]
            return f"\x1b[38;2;{r};{g};{b}m"
        # INDEXED_256
        idx = _INDEXED_256[color]
        if idx < 0:
            return _DEFAULT_FG
        return f"\x1b[38;5;{idx}m"

    def colorize(
        self,
        text: str,
        color: ColorName,
        *,
        bold: bool = False,
        dim: bool = False,
    ) -> str:
        """Wrap ``text`` in foreground colour escapes.

        Under :attr:`PaletteMode.PLAIN` returns ``text`` unchanged
        (no escapes), so callers can use ``colorize`` unconditionally
        and let the palette decide what to emit.
        """
        if self.mode is PaletteMode.PLAIN:
            return text
        prefix_parts: list[str] = []
        if bold:
            prefix_parts.append(_BOLD)
        if dim:
            prefix_parts.append(_DIM)
        prefix_parts.append(self.fg(color))
        return "".join(prefix_parts) + text + _RESET

    def reset(self) -> str:
        """ANSI reset sequence; ``""`` under plain mode."""
        return "" if self.mode is PaletteMode.PLAIN else _RESET


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _is_no_color() -> bool:
    """``True`` if the standard ``NO_COLOR`` opt-out is set.

    Per https://no-color.org any non-empty value disables colour;
    presence-and-empty is treated as "not set" to match the spec.
    """
    val = os.environ.get("NO_COLOR")
    return val is not None and val != ""


def _is_truecolor(env: Mapping[str, str] | None = None) -> bool:
    """``True`` if the environment declares 24-bit colour support.

    Standard signals: ``COLORTERM=truecolor`` or ``COLORTERM=24bit``.
    """
    e: Mapping[str, str] = env if env is not None else os.environ
    val = e.get("COLORTERM", "").lower()
    return val in ("truecolor", "24bit")


def _supports_256(env: Mapping[str, str] | None = None) -> bool:
    """``True`` if the terminal declares at least 256-colour support.

    Conservative: requires ``TERM`` to contain ``256color``. Under
    an unset or ``dumb`` ``TERM`` we fall back to plain.
    """
    e: Mapping[str, str] = env if env is not None else os.environ
    term = e.get("TERM", "").lower()
    if not term or term == "dumb":
        return False
    return "256color" in term


def detect_palette(
    *,
    env: Mapping[str, str] | None = None,
    isatty: bool | None = None,
) -> Palette:
    """Pick the strongest supported palette under the current process.

    Decision order:

    1. ``NO_COLOR`` set with a non-empty value → :class:`PaletteMode.PLAIN`.
    2. Output is not a TTY (e.g. piped to a file) → :class:`PaletteMode.PLAIN`.
    3. ``COLORTERM`` declares truecolor → :class:`PaletteMode.TRUECOLOR`.
    4. ``TERM`` declares 256-colour support → :class:`PaletteMode.INDEXED_256`.
    5. Otherwise → :class:`PaletteMode.PLAIN`.

    Args:
        env: optional environment override (mainly for tests). When
            None the function reads ``os.environ``.
        isatty: optional TTY-state override. When None the function
            checks ``sys.stdout.isatty()``.
    """
    e: Mapping[str, str] = env if env is not None else os.environ
    val = e.get("NO_COLOR")
    if val is not None and val != "":
        return Palette.plain()
    on_tty = isatty if isatty is not None else _is_attached_tty()
    if not on_tty:
        return Palette.plain()
    if _is_truecolor(env=e):
        return Palette.truecolor()
    if _supports_256(env=e):
        return Palette.indexed_256()
    return Palette.plain()


def _is_attached_tty() -> bool:
    """``sys.stdout.isatty()`` with a defensive fallback.

    Some environments (notebooks, some CI runners) replace ``sys.stdout``
    with an object that lacks ``isatty``; treat that as "not a TTY"
    so we default to plain output rather than crashing.
    """
    isatty = getattr(sys.stdout, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except (OSError, ValueError):
        return False
