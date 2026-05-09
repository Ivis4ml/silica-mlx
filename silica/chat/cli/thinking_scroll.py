"""silica.chat.cli.thinking_scroll — fixed-height scroll window
for streaming ``<think>`` content in the chat REPL.

During a turn that produces a reasoning block, the renderer in
:mod:`silica.chat.cli.app` receives a sequence of
``EnterThinking`` → ``ThinkingChunk`` × N → ``ExitThinking``
events from :class:`silica.chat.cli.thinking_parser.ThinkingParser`.
The pre-v1.7.37 ``thinking="show"`` path wrote each chunk to
stdout in dimmed grey, which produced an unbounded "wall of text"
that scrolled the prompt off the top of the terminal whenever
Qwen3 / Qwen3.5 / Gemma 4 emitted a long ``Thinking Process: 1.
**Analyze the Request:** ...`` block.

This module replaces that path on TTY-capable terminals with an
in-place fixed-height window: a magenta header line
(``⠋ thinking...``) plus the most recent ``max_lines`` lines of
reasoning text in dimmed grey. On each :meth:`feed` the window
redraws using ANSI cursor controls (``\\x1b[NA`` cursor-up +
``\\x1b[J`` clear-from-cursor-to-end-of-screen); on :meth:`clear`
the entire region is erased so the renderer's post-collapse
``thought for Xs`` line lands cleanly.

Gating mirrors :func:`silica.chat.cli.live_toolbar.make_live_toolbar`:
the window degrades to a no-op (and the renderer falls back to the
v1.7.36 wall-of-text path) when any of the following hold:

* ``output_stream.isatty()`` is ``False`` — piping the chat to a
  file must not litter ANSI escapes into the captured stream.
* ``TERM=dumb`` (or unset) — legacy terminals that may not honour
  cursor-up sequences reliably.
* ``palette.mode is PaletteMode.PLAIN`` — the user opted out of
  colour (typically via ``NO_COLOR=1``); animated cursor dancing
  would surprise them.

CJK width handling. The user's primary workload is Chinese
prompts to Qwen3.5, and Chinese characters render as 2 cells
wide on every common terminal. Naive ``len(line)`` truncation
would under-count the on-screen footprint, the line would wrap,
the wrapped row would consume an additional terminal row, and the
next ``\\x1b[NA`` cursor-up would land in the wrong place — leaving
stale rows behind. To avoid pulling in a wcwidth library, this
module uses a coarse ord-range estimator (East-Asian-Wide + common
emoji ranges → 2 cells, else 1) that is correct enough for
Chinese / Japanese / Korean / fullwidth-punctuation / common emoji
to keep the cursor arithmetic aligned. Every rendered line is
truncated to ``terminal_cols - 1`` cells with an appended ``"…"``
when truncation occurs, so wrap-around is structurally impossible.

Lifecycle. The window is constructed once per chat turn, inside
the streaming loop in :mod:`silica.chat.cli.app`, not session-
scoped. Per-turn construction makes the buffer / rendered-row
counter trivially fresh on each new turn even if the previous
turn ended via ``KeyboardInterrupt`` or another exception path
that bypassed :meth:`clear`. There is no session-level
"poisoning" failure mode for the new turn's first redraw to
inherit.

Known limits. Mid-stream terminal resize is not handled (the
``\\x1b[NA`` cursor-up arithmetic is computed against the row
count at render time, so a shrink between two redraws may leave
fragments on screen for one frame). Acceptable trade-off for
v1.7.37; can be revisited if it shows up in real use.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections import deque
from typing import TextIO

from silica.chat.cli.palette import Palette, PaletteMode

DEFAULT_MAX_LINES = 6
"""Default scroll-window body height (lines, header excluded).

Six lines holds enough context that the user can follow the
reasoning ("the model is currently weighing X against Y") while
staying short enough to fit comfortably above the chat prompt
on an 80x24 terminal. Tunable per-instance via the ``max_lines``
constructor argument."""


# ---------------------------------------------------------------------------
# Cell-width estimator (no dependency on wcwidth / prompt_toolkit)
# ---------------------------------------------------------------------------


def _cell_width_of_char(ch: str) -> int:
    """Return ``2`` for East Asian Wide / CJK / common emoji code
    points, ``1`` otherwise.

    Coarse approximation. Correct enough to keep cursor-up
    arithmetic aligned for Chinese / Japanese / Korean / fullwidth-
    punctuation / common emoji without the cost of pulling in the
    wcwidth library. Intended for line-truncation only; not a
    full Unicode East-Asian-Width implementation.
    """
    cp = ord(ch)
    if (
        0x1100 <= cp <= 0x115F  # Hangul Jamo
        or 0x2E80 <= cp <= 0xA4CF  # CJK + Yi + Lisu (range covers Han/Hiragana/Katakana)
        or 0xAC00 <= cp <= 0xD7A3  # Hangul Syllables
        or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        or 0xFE30 <= cp <= 0xFE4F  # CJK Compatibility Forms
        or 0xFF00 <= cp <= 0xFF60  # Fullwidth Forms (ASCII range)
        or 0xFFE0 <= cp <= 0xFFE6  # Fullwidth Signs
        or 0x20000 <= cp <= 0x2FFFD  # CJK Ext B-F
        or 0x30000 <= cp <= 0x3FFFD  # CJK Ext G
        or 0x1F300 <= cp <= 0x1F64F  # Misc Symbols + Emoticons
        or 0x1F680 <= cp <= 0x1F6FF  # Transport + Map
        or 0x1F900 <= cp <= 0x1F9FF  # Supplemental Symbols + Pictographs
    ):
        return 2
    return 1


def _cell_width(text: str) -> int:
    """Sum :func:`_cell_width_of_char` over every char in ``text``."""
    return sum(_cell_width_of_char(c) for c in text)


def _truncate_to_cells(text: str, max_cells: int) -> str:
    """Return ``text`` truncated so its rendered width does not
    exceed ``max_cells`` cells. Appends ``"…"`` (1 cell) when
    truncation occurs. Returns ``""`` for ``max_cells <= 0``.

    Used to clip every rendered line to ``terminal_cols - 1``
    before write so the rightmost column never auto-wraps.
    """
    if max_cells <= 0:
        return ""
    if _cell_width(text) <= max_cells:
        return text
    out: list[str] = []
    used = 0
    for ch in text:
        cw = _cell_width_of_char(ch)
        # Reserve 1 cell for the trailing ellipsis.
        if used + cw > max_cells - 1:
            break
        out.append(ch)
        used += cw
    return "".join(out) + "…"


# ---------------------------------------------------------------------------
# Scroll window
# ---------------------------------------------------------------------------


class ThinkingScrollWindow:
    """Fixed-height ANSI scroll window for live ``<think>`` rendering.

    See module docstring for the lifecycle / fallback contract.

    Typical use, per turn::

        window = ThinkingScrollWindow(palette=palette)
        # On EnterThinking:
        if window.scroll_enabled:
            window.start()
        # On each ThinkingChunk:
        if window.scroll_enabled:
            window.feed(chunk_text)
        # On ExitThinking:
        if window.scroll_enabled:
            window.clear()

    The renderer keeps its existing wall-of-text fallback (a
    sequence of plain dimmed-grey writes via ``palette.colorize``)
    on the ``not window.scroll_enabled`` branch.
    """

    def __init__(
        self,
        *,
        palette: Palette,
        max_lines: int = DEFAULT_MAX_LINES,
        output_stream: TextIO | None = None,
        is_tty: bool | None = None,
        term: str | None = None,
        terminal_cols: int | None = None,
    ) -> None:
        self._palette = palette
        self._max_lines = max(1, int(max_lines))
        self._out: TextIO = (
            output_stream if output_stream is not None else sys.stdout
        )
        # TTY / TERM / palette gating — mirrors make_live_toolbar.
        if is_tty is None:
            isatty_fn = getattr(self._out, "isatty", None)
            is_tty_resolved = (
                bool(isatty_fn()) if callable(isatty_fn) else False
            )
        else:
            is_tty_resolved = is_tty
        if term is None:
            term = os.environ.get("TERM", "")
        term_dumb = term.strip().lower() in ("", "dumb")
        plain = palette.mode is PaletteMode.PLAIN
        self._scroll_enabled = (
            is_tty_resolved and not term_dumb and not plain
        )
        # Test seam: tests pass a fixed terminal_cols so they do
        # not depend on the host terminal size; production lookups
        # go through shutil.get_terminal_size() per redraw so a
        # mid-turn user resize is picked up at the next chunk.
        self._terminal_cols_override = terminal_cols
        # Completed lines and the in-progress line. The deque caps
        # at max_lines + 1 so the trailing-empty bookkeeping for a
        # ``\n``-terminated chunk does not push real content out of
        # view; the display logic in _display_lines clips the final
        # rendered set back to max_lines.
        self._completed: deque[str] = deque(maxlen=self._max_lines + 1)
        self._current: str = ""
        self._rendered_lines = 0

    @property
    def scroll_enabled(self) -> bool:
        """True iff the window will draw an in-place scroll. The
        renderer reads this flag to decide between the new path
        and the legacy wall-of-text fallback."""
        return self._scroll_enabled

    @property
    def max_lines(self) -> int:
        """Body height in lines (header excluded)."""
        return self._max_lines

    def _terminal_cols(self) -> int:
        if self._terminal_cols_override is not None:
            return max(8, self._terminal_cols_override)
        try:
            cols = shutil.get_terminal_size((80, 24)).columns
        except OSError:
            cols = 80
        return max(8, cols)

    def _display_lines(self) -> list[str]:
        """Return the body lines that should currently render,
        capped at :attr:`max_lines`. The in-progress line is
        included only when non-empty so a chunk that ended on
        ``\\n`` does not produce a stale blank row at the bottom."""
        completed = list(self._completed)
        if self._current != "":
            combined = completed + [self._current]
        else:
            combined = completed
        return combined[-self._max_lines:]

    def start(self) -> None:
        """Begin a new thinking window — render the magenta header
        immediately so the user sees the indicator even before the
        first ``ThinkingChunk`` arrives. No-op when scroll mode is
        disabled."""
        if not self._scroll_enabled:
            return
        self._completed = deque(maxlen=self._max_lines + 1)
        self._current = ""
        self._rendered_lines = 0
        self._redraw()

    def feed(self, text: str) -> None:
        """Append ``text`` to the body buffer (splitting on ``\\n``)
        and redraw. No-op when scroll mode is disabled."""
        if not self._scroll_enabled:
            return
        for ch in text:
            if ch == "\n":
                self._completed.append(self._current)
                self._current = ""
            else:
                self._current += ch
        self._redraw()

    def clear(self) -> None:
        """Erase the rendered region and reset the buffer. Cursor
        ends at column 0 of the line where the window started, so
        the renderer can immediately print a follow-up line. No-op
        when scroll mode is disabled."""
        if not self._scroll_enabled:
            return
        if self._rendered_lines:
            self._out.write(
                f"\r\x1b[{self._rendered_lines}A\x1b[J"
            )
            self._out.flush()
        self._completed = deque(maxlen=self._max_lines + 1)
        self._current = ""
        self._rendered_lines = 0

    def _redraw(self) -> None:
        # Erase any previously rendered region.
        if self._rendered_lines:
            self._out.write(
                f"\r\x1b[{self._rendered_lines}A\x1b[J"
            )
        cols = self._terminal_cols()
        # Reserve one cell of right margin so a width-exact line
        # cannot trigger Terminal.app's auto-wrap-on-final-column.
        max_text_cells = max(1, cols - 1)
        # Header.
        self._out.write(
            self._palette.colorize(
                _truncate_to_cells("⠋ thinking...", max_text_cells),
                "magenta",
                dim=True,
            )
        )
        self._out.write("\n")
        rendered = 1
        # Body.
        for line in self._display_lines():
            self._out.write(
                self._palette.colorize(
                    _truncate_to_cells(line, max_text_cells),
                    "grey",
                    dim=True,
                )
            )
            self._out.write("\n")
            rendered += 1
        self._rendered_lines = rendered
        self._out.flush()


__all__ = [
    "DEFAULT_MAX_LINES",
    "ThinkingScrollWindow",
]
