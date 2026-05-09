"""v1.7.37 unit tests for the chat-CLI thinking scroll window.

Pure-Python coverage of :mod:`silica.chat.cli.thinking_scroll`.
The helper writes ANSI escape sequences to a configurable
``output_stream``; tests use :class:`io.StringIO` so the assertion
surface is exact byte sequences without any real terminal.

Three test groups:

* The ord-range cell-width estimator (:func:`_cell_width`,
  :func:`_truncate_to_cells`) — correctness for ASCII, CJK,
  fullwidth punctuation, mixed text, and the truncation edge
  cases (zero / one cell, exact-fit, ellipsis).
* The TTY / TERM / palette gating (:attr:`scroll_enabled`) —
  every disable path must independently flip the flag, and the
  enabled path must require all three pass.
* The render side — :meth:`start` produces a header-only frame,
  :meth:`feed` appends and redraws with cursor-up + screen-clear
  escapes, :meth:`clear` erases the region, and the buffer cap
  drops the oldest body line when overflow happens.

CHAT-CLI v1.7.36 → v1.7.37 follow-up: the scroll window
replaces the v1.7.36 "wall of text" path for ``thinking="show"``
on TTY-capable terminals; non-TTY / ``TERM=dumb`` / ``NO_COLOR``
fall back to v1.7.36 wall-of-text behaviour, which is the
helper's no-op contract on the disabled branch.
"""

from __future__ import annotations

import io

import pytest

from silica.chat.cli.palette import Palette
from silica.chat.cli.thinking_scroll import (
    DEFAULT_MAX_LINES,
    ThinkingScrollWindow,
    _cell_width,
    _cell_width_of_char,
    _truncate_to_cells,
)


# ---------------------------------------------------------------------------
# Cell-width estimator
# ---------------------------------------------------------------------------


def test_cell_width_of_char_ascii_is_one() -> None:
    """ASCII letters / digits / punctuation render as 1 cell on
    every common terminal — the estimator's baseline branch."""
    for ch in "aZ09 .!?@":
        assert _cell_width_of_char(ch) == 1


def test_cell_width_of_char_cjk_is_two() -> None:
    """CJK Unified Ideographs (U+4E00..U+9FFF) render as 2 cells
    in every monospaced terminal."""
    for ch in "你好世界中文":
        assert _cell_width_of_char(ch) == 2


def test_cell_width_of_char_fullwidth_punctuation_is_two() -> None:
    """Fullwidth punctuation (U+FF00..U+FF60) — common in
    Chinese-language model output — renders as 2 cells."""
    assert _cell_width_of_char("，") == 2
    assert _cell_width_of_char("。") == 2


def test_cell_width_of_char_emoji_is_two() -> None:
    """Common emoji (U+1F300..U+1F64F etc.) render as 2 cells in
    most modern terminals."""
    assert _cell_width_of_char("😀") == 2
    assert _cell_width_of_char("🚀") == 2


def test_cell_width_empty_is_zero() -> None:
    assert _cell_width("") == 0


def test_cell_width_ascii_string() -> None:
    assert _cell_width("hello") == 5


def test_cell_width_cjk_string() -> None:
    assert _cell_width("你好") == 4


def test_cell_width_mixed_string() -> None:
    """Mixed Chinese + ASCII — the user's actual model output
    pattern (e.g. "直接走路去。50 米太近")."""
    assert _cell_width("直接走路去。50 米太近") == (
        2 * 5 + 2 + 1 + 1 + 1 + 2 + 2 + 2
    )


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncate_short_text_returned_verbatim() -> None:
    """When the rendered width already fits, no ellipsis is
    appended — the user keeps the whole line."""
    assert _truncate_to_cells("hello", 10) == "hello"


def test_truncate_exact_fit_returned_verbatim() -> None:
    """An exactly-fits line must NOT be truncated (the ellipsis
    reservation only kicks in when the line would otherwise
    overflow)."""
    assert _truncate_to_cells("hello", 5) == "hello"


def test_truncate_ascii_appends_ellipsis() -> None:
    """5 chars + ellipsis = 6 cells, fits inside max_cells=6."""
    assert _truncate_to_cells("hello world", 6) == "hello…"


def test_truncate_chinese_respects_two_cell_width() -> None:
    """``你好`` is 4 cells, ``…`` is 1 cell, total 5. ``你好世界``
    (8 cells) truncated to 5 must yield ``你好…`` — proves the
    estimator and truncation are wired up consistently."""
    assert _truncate_to_cells("你好世界", 5) == "你好…"


def test_truncate_zero_returns_empty() -> None:
    assert _truncate_to_cells("anything", 0) == ""


def test_truncate_negative_returns_empty() -> None:
    assert _truncate_to_cells("anything", -1) == ""


def test_truncate_one_cell_only_ellipsis() -> None:
    """At 1 cell the only valid output is the ellipsis itself."""
    assert _truncate_to_cells("hello", 1) == "…"


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


def _ansi_palette() -> Palette:
    return Palette.indexed_256()


def _plain_palette() -> Palette:
    return Palette.plain()


def test_scroll_disabled_when_not_a_tty() -> None:
    """Piping the chat to a file → scroll_enabled False → caller
    falls back to wall-of-text. Without this gate the captured
    output would contain raw ANSI escapes that look like garbage
    when the user later cats the file."""
    win = ThinkingScrollWindow(
        palette=_ansi_palette(),
        output_stream=io.StringIO(),
        is_tty=False,
        term="xterm-256color",
    )
    assert win.scroll_enabled is False


def test_scroll_disabled_when_term_is_dumb() -> None:
    """Legacy / unknown terminals may not honour cursor-up
    sequences; degrade safely."""
    win = ThinkingScrollWindow(
        palette=_ansi_palette(),
        output_stream=io.StringIO(),
        is_tty=True,
        term="dumb",
    )
    assert win.scroll_enabled is False


def test_scroll_disabled_when_term_is_empty() -> None:
    win = ThinkingScrollWindow(
        palette=_ansi_palette(),
        output_stream=io.StringIO(),
        is_tty=True,
        term="",
    )
    assert win.scroll_enabled is False


def test_scroll_disabled_when_palette_plain() -> None:
    """``NO_COLOR=1`` users get a plain palette; cursor dancing
    would surprise them. Same gating rule as live_toolbar."""
    win = ThinkingScrollWindow(
        palette=_plain_palette(),
        output_stream=io.StringIO(),
        is_tty=True,
        term="xterm-256color",
    )
    assert win.scroll_enabled is False


def test_scroll_enabled_in_normal_terminal() -> None:
    """All three gates pass → scroll is live."""
    win = ThinkingScrollWindow(
        palette=_ansi_palette(),
        output_stream=io.StringIO(),
        is_tty=True,
        term="xterm-256color",
    )
    assert win.scroll_enabled is True


# ---------------------------------------------------------------------------
# No-op behaviour in disabled mode
# ---------------------------------------------------------------------------


def test_disabled_window_is_silent() -> None:
    """When scroll is disabled, none of start/feed/clear may emit
    bytes — the renderer is the one that prints in that case
    (wall-of-text fallback)."""
    out = io.StringIO()
    win = ThinkingScrollWindow(
        palette=_ansi_palette(),
        output_stream=out,
        is_tty=False,
    )
    win.start()
    win.feed("anything\nmultiple\nlines")
    win.clear()
    assert out.getvalue() == ""


# ---------------------------------------------------------------------------
# Enabled-mode rendering
# ---------------------------------------------------------------------------


def _make_enabled(
    out: io.StringIO,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    terminal_cols: int = 80,
) -> ThinkingScrollWindow:
    return ThinkingScrollWindow(
        palette=_ansi_palette(),
        output_stream=out,
        is_tty=True,
        term="xterm-256color",
        max_lines=max_lines,
        terminal_cols=terminal_cols,
    )


def test_start_renders_only_header() -> None:
    """``start`` must place the magenta header on screen so the
    user sees thinking has begun even before the model emits a
    chunk; no body text yet."""
    out = io.StringIO()
    win = _make_enabled(out)
    win.start()
    s = out.getvalue()
    assert "⠋ thinking..." in s
    assert s.endswith("\n")
    # Exactly one rendered line so a future redraw moves cursor up 1.
    assert win._rendered_lines == 1


def test_feed_emits_cursor_up_and_screen_clear_then_redraws() -> None:
    """The redraw protocol is: ``\\r\\x1b[NA\\x1b[J`` to erase
    the previous frame, then write header + body. Tests verify
    both the escape and the new content."""
    out = io.StringIO()
    win = _make_enabled(out)
    win.start()
    out.truncate(0)
    out.seek(0)
    win.feed("first line of thinking")
    s = out.getvalue()
    # rendered_lines was 1 after start → cursor up by 1.
    assert "\x1b[1A" in s
    # Screen-clear from cursor.
    assert "\x1b[J" in s
    assert "first line of thinking" in s
    # Header re-rendered every redraw (it is owned by the window).
    assert "⠋ thinking..." in s
    # rendered_lines now 2 (header + 1 body line).
    assert win._rendered_lines == 2


def test_feed_handles_newline_separated_chunks() -> None:
    """Reasoning text typically streams with embedded ``\\n``
    between sentences. Each newline starts a new buffer entry."""
    out = io.StringIO()
    win = _make_enabled(out)
    win.start()
    out.truncate(0)
    out.seek(0)
    win.feed("first\nsecond\nthird")
    s = out.getvalue()
    assert "first" in s
    assert "second" in s
    assert "third" in s
    assert win._rendered_lines == 4  # header + 3 body lines


def test_feed_caps_body_at_max_lines() -> None:
    """When more lines arrive than ``max_lines``, the oldest body
    rows scroll off the top — that is the whole point of the
    fixed-height window."""
    out = io.StringIO()
    win = _make_enabled(out, max_lines=3)
    win.start()
    win.feed("a\nb\nc\nd\ne")
    s = out.getvalue()
    # Verify that the LAST redraw shows c, d, e — slice from the
    # final header occurrence onward.
    last_header = s.rfind("⠋ thinking...")
    final_frame = s[last_header:]
    assert "c" in final_frame
    assert "d" in final_frame
    assert "e" in final_frame
    assert "a" not in final_frame
    assert "b" not in final_frame
    assert win._rendered_lines == 4  # header + 3 body lines (cap)


def test_feed_skips_trailing_empty_line() -> None:
    """A chunk that ends with ``\\n`` must NOT leave a stale
    blank row at the bottom of the window — the trailing empty
    in-progress line is hidden until real content arrives on it."""
    out = io.StringIO()
    win = _make_enabled(out, max_lines=4)
    win.start()
    win.feed("a\nb\n")
    # 2 completed lines, current="" → display 2 lines.
    assert win._rendered_lines == 3  # header + 2 body
    out.truncate(0)
    out.seek(0)
    win.feed("c")
    # current="c" → display 3 lines.
    assert win._rendered_lines == 4


def test_clear_emits_screen_clear_and_resets_state() -> None:
    """``clear`` returns the cursor to the start of the window
    region and erases everything below it — leaving a blank
    surface for the renderer's "thought for Xs" follow-up."""
    out = io.StringIO()
    win = _make_enabled(out)
    win.start()
    win.feed("some line of thinking text")
    out.truncate(0)
    out.seek(0)
    win.clear()
    s = out.getvalue()
    # Cursor up by N + screen clear.
    assert "\x1b[" in s
    assert "A" in s
    assert "\x1b[J" in s
    assert win._rendered_lines == 0


def test_clear_after_start_only_emits_one_line_of_cursor_movement() -> None:
    """A ``clear`` immediately after ``start`` (no feed) erases
    the header alone — cursor up by 1, screen-clear."""
    out = io.StringIO()
    win = _make_enabled(out)
    win.start()
    out.truncate(0)
    out.seek(0)
    win.clear()
    s = out.getvalue()
    assert "\x1b[1A" in s
    assert "\x1b[J" in s


def test_clear_when_never_started_is_noop() -> None:
    """Defensive: ``clear`` before ``start`` (no rendered frame
    on screen) must NOT emit cursor-up sequences that would chew
    into prior output."""
    out = io.StringIO()
    win = _make_enabled(out)
    win.clear()
    assert out.getvalue() == ""


# ---------------------------------------------------------------------------
# CJK truncation in the rendered output
# ---------------------------------------------------------------------------


def test_chinese_line_truncated_to_terminal_width() -> None:
    """A wide Chinese line must be clipped before write so it
    does not wrap off the right edge — wrap would consume an
    extra terminal row that ``rendered_lines`` does not know
    about, so the next cursor-up arithmetic would be wrong."""
    out = io.StringIO()
    win = _make_enabled(out, terminal_cols=10)  # tight terminal
    win.start()
    out.truncate(0)
    out.seek(0)
    # 16 cells of Chinese — exceeds the 10-col terminal width.
    win.feed("你好世界你好世界")
    s = out.getvalue()
    # Ellipsis is the user-visible signal that truncation kicked in.
    assert "…" in s


def test_default_max_lines_is_six() -> None:
    """v1.7.37 height pin — six lines of body holds enough
    reasoning context for the user to follow without crowding
    the prompt off an 80x24 terminal."""
    assert DEFAULT_MAX_LINES == 6


def test_max_lines_clamped_to_at_least_one() -> None:
    """Constructor accepts any int; degenerate values clamp to 1
    so the buffer is always usable. (No ``ValueError`` — the
    chat-CLI shell would have to bubble it up to the user.)"""
    out = io.StringIO()
    win = ThinkingScrollWindow(
        palette=_ansi_palette(),
        output_stream=out,
        is_tty=True,
        term="xterm-256color",
        max_lines=0,
    )
    assert win.max_lines == 1


# ---------------------------------------------------------------------------
# Ascii rendering uses dim escapes (smoke test on the colour wiring)
# ---------------------------------------------------------------------------


def test_rendered_frame_contains_dim_escape() -> None:
    """The window writes through ``palette.colorize(..., dim=True)``
    so the body / header are dimmed — verify a ``\\x1b[2m`` (dim)
    escape appears at least once in the rendered frame."""
    out = io.StringIO()
    win = _make_enabled(out)
    win.start()
    win.feed("dimmed reasoning text")
    assert "\x1b[2m" in out.getvalue()
