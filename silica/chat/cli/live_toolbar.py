"""silica.chat.cli.live_toolbar — swappable bottom-toolbar backends.

CHAT-CLI-HARDENING-6 (F3). Pre-HARDENING-6 the bottom toolbar was
plumbed through ``PromptSession.bottom_toolbar`` which only refreshes
during the user-input phase; the generation phase wrote tokens
directly to stdout via ``sys.stdout.write`` and surfaced an inline
``⠋ prefilling/thinking...`` indicator. Symptom: ``tokens=N/max``,
``tok/s``, and ``state=prefill|thinking|decode`` were post-turn
snapshots, not live.

This module introduces a small backend abstraction so the chat-turn
flow can request live refreshes per token without committing to one
particular rendering strategy:

- :class:`LiveToolbar` — context-manager protocol with ``refresh``.
- :class:`NullLiveToolbar` — no-op backend; used when output is not
  a TTY, ``TERM=dumb``, palette is plain, or any other situation
  where ANSI escape sequences would corrupt the stream. The
  post-turn ``PromptSession.bottom_toolbar`` still surfaces the
  final values, so the user retains full visibility — just not
  live during generation.
- :class:`AnsiLiveToolbar` — sticky-bottom-line backend using ANSI
  cursor save (``\\x1b[s``) / restore (``\\x1b[u``). Reserves the
  line immediately below the streamed text for the toolbar, redraws
  on each ``refresh``, and clears the line on exit.

Path-A note (decision deferred). The original
``plans/CHAT_CLI_HARDENING.md`` v1 wording called for a full
prompt-toolkit ``Application`` layout that runs during generation
on a worker-thread engine driver. That is a richer-TUI rewrite
(scrollback pane, pause/resume, clickable status, live config) and
materially exceeds what F3 needs. The backend interface here keeps
that path open: a future ``ApplicationLiveToolbar`` is a drop-in
replacement at the construction seam.

Also exposes :class:`RollingTokRate` — a small, testable rolling
window that the chat-turn flow uses to drive live ``tok/s``
without touching the engine's metrics path.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections import deque
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Callable

from silica.chat.cli.palette import Palette, PaletteMode
from silica.chat.cli.state import ChatCliState
from silica.chat.cli.toolbar import render_toolbar


class RollingTokRate:
    """Sliding-window decode tok/s estimator.

    Records monotonic token timestamps in a deque sized to
    ``window``; ``rate()`` divides the elapsed wall-clock between
    the oldest and newest sample by the number of intervals
    (``len(samples) - 1``). Returns ``None`` until at least two
    samples are present and the elapsed window is positive — the
    chat-CLI surfaces ``None`` as the toolbar's ``tok/s=—``
    placeholder, matching the post-turn semantics for the same
    field.

    A rolling window (default 20 tokens) reflects the *current*
    decode rate rather than the cumulative-from-first-token rate.
    Cumulative rate is monotonic and lags the user's perception
    when prefill->decode->thinking transitions slow or speed up
    the stream; windowed rate updates within ~1 second of any
    rate change at typical decode speeds (20-100 tok/s).
    """

    def __init__(self, *, window: int = 20) -> None:
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self._window = window
        self._samples: deque[float] = deque(maxlen=window)

    def record(self, now: float) -> None:
        """Append one token's monotonic timestamp."""
        self._samples.append(now)

    def reset(self) -> None:
        """Drop all samples — call between turns."""
        self._samples.clear()

    def rate(self) -> float | None:
        """Tokens per second over the current window, or ``None``
        when the window has fewer than two samples / the elapsed
        time is non-positive."""
        if len(self._samples) < 2:
            return None
        elapsed = self._samples[-1] - self._samples[0]
        if elapsed <= 0.0:
            return None
        return (len(self._samples) - 1) / elapsed


class LiveToolbar(AbstractContextManager["LiveToolbar"], ABC):
    """Context-manager protocol for live-toolbar backends.

    Lifecycle:

    1. Caller enters the context at turn start
       (``with make_live_toolbar(...) as live:``).
    2. After each streamed reply chunk + on each phase transition
       the caller invokes ``live.refresh(state)``.
    3. ``__exit__`` runs on success / abort / exception and is
       responsible for restoring terminal state (clearing the
       reserved line; releasing scroll regions if any).

    Implementations must keep ``__exit__`` idempotent and safe to
    invoke from finally / except blocks even if ``refresh`` was
    never called.
    """

    def __enter__(self) -> "LiveToolbar":
        return self

    @abstractmethod
    def refresh(self, state: ChatCliState) -> None:
        """Re-render the toolbar from the current state snapshot."""

    @abstractmethod
    def clear(self) -> None:
        """Wipe the reserved line without ending the lifecycle.

        Used before printing out-of-band text that would otherwise
        overlap the toolbar (e.g. ``[generation aborted]`` on
        Ctrl-C). The backend remains active; subsequent
        ``refresh`` calls re-populate the same reserved line. The
        ``with``-block's ``__exit__`` still runs at scope exit and
        is responsible for the final teardown.
        """

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Restore terminal state. MUST be tolerant of ``refresh``
        never having been called (caller may have aborted before
        the first token)."""


class NullLiveToolbar(LiveToolbar):
    """No-op backend.

    Used when the output stream is not a TTY, the terminal does
    not support ANSI control sequences (``TERM=dumb``), the
    palette is plain (capability-detection forced no-colour), or
    the caller explicitly opted out for testing. The chat-turn
    flow still mutates ``ChatCliState`` live (``tokens_generated``,
    ``tok_per_sec``, ``stream_state``); the post-turn
    ``PromptSession.bottom_toolbar`` reflects those values on the
    next prompt iteration.
    """

    def refresh(self, state: ChatCliState) -> None:
        return None

    def clear(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class AnsiLiveToolbar(LiveToolbar):
    """Sticky-bottom-line backend.

    Strategy. The toolbar lives on the line directly below the
    streamed reply text. On entry the backend writes a placeholder
    line below the cursor and returns the cursor to its original
    position; subsequent ``refresh`` calls save the cursor with
    ``\\x1b[s``, advance one line with ``\\x1b[E``, clear that line
    with ``\\x1b[K``, write the new toolbar text, and restore the
    cursor with ``\\x1b[u``. Streamed tokens written between
    refreshes appear above the toolbar line; modern terminals
    (iTerm2, WezTerm, kitty, Alacritty, macOS Terminal.app) handle
    save/restore correctly across natural scroll.

    No scroll-region (DECSTBM, ``\\x1b[<top>;<bottom>r``) is set —
    scroll regions are persistent terminal state that survives a
    Python crash and can leave the user's shell in a confused
    state. Save/restore is per-call and self-cleaning.

    Exit cleanup: clears the reserved bottom line without moving
    the cursor; the caller's existing turn-end ``\\n`` then
    advances past the now-empty line as normal.
    """

    # Class-level constants make the ANSI sequences inspectable in
    # tests without scraping the file.
    SAVE_CURSOR = "\x1b[s"
    RESTORE_CURSOR = "\x1b[u"
    NEXT_LINE = "\x1b[E"
    CLEAR_LINE = "\x1b[K"

    def __init__(
        self,
        *,
        render: Callable[[ChatCliState], str],
        output_stream: Any = None,
    ) -> None:
        self._render = render
        self._out = output_stream if output_stream is not None else sys.stdout
        self._active = False

    def __enter__(self) -> "AnsiLiveToolbar":
        # Reserve the line below the cursor: save, advance, clear,
        # restore. Subsequent refreshes overwrite the same reserved
        # line. The cursor is back at its original position so the
        # caller can resume writing reply text without disturbance.
        self._out.write(
            self.SAVE_CURSOR
            + self.NEXT_LINE
            + self.CLEAR_LINE
            + self.RESTORE_CURSOR
        )
        self._out.flush()
        self._active = True
        return self

    def refresh(self, state: ChatCliState) -> None:
        if not self._active:
            return
        text = self._render(state)
        # Save cursor at end of streamed text → advance to reserved
        # line → clear it → write the toolbar → restore cursor.
        # ``text`` rendering is single-line by render_toolbar
        # contract; if a future palette quirks emits an embedded
        # newline the worst case is one extra visual line below
        # the toolbar that the next refresh will not clean up.
        self._out.write(
            self.SAVE_CURSOR
            + self.NEXT_LINE
            + self.CLEAR_LINE
            + text
            + self.RESTORE_CURSOR
        )
        self._out.flush()

    def clear(self) -> None:
        if not self._active:
            return
        # Same save → advance → clear → restore dance as
        # ``refresh`` but without the toolbar text — the reserved
        # line goes blank. The backend stays active, so the next
        # ``refresh`` call repopulates this same line.
        self._out.write(
            self.SAVE_CURSOR
            + self.NEXT_LINE
            + self.CLEAR_LINE
            + self.RESTORE_CURSOR
        )
        self._out.flush()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._active:
            return None
        # Clear the reserved line without moving the cursor; the
        # caller's turn-end ``\n`` (or the abort branch's ``\n``)
        # advances past the now-empty toolbar line on its own.
        self._out.write(
            self.SAVE_CURSOR
            + self.NEXT_LINE
            + self.CLEAR_LINE
            + self.RESTORE_CURSOR
        )
        self._out.flush()
        self._active = False
        return None


def make_live_toolbar(
    *,
    palette: Palette,
    output_stream: Any = None,
    is_tty: bool | None = None,
    term: str | None = None,
) -> LiveToolbar:
    """Construct the right backend for the runtime environment.

    Returns :class:`NullLiveToolbar` when:

    - ``output_stream`` is not a TTY (capability auto-detect via
      ``isatty()`` when ``is_tty`` is left ``None``); piping the
      chat to a file must not litter ANSI escapes into the
      captured stream.
    - ``term`` is ``"dumb"`` or empty; legacy terminals do not
      handle save/restore reliably.
    - ``palette.mode`` is :attr:`PaletteMode.PLAIN`; the user
      already opted out of colour, so live cursor dancing would
      surprise them.

    Returns :class:`AnsiLiveToolbar` otherwise. The render callable
    closes over ``palette`` so the bound backend can call
    :func:`render_toolbar` without re-resolving the palette per
    refresh.
    """
    out = output_stream if output_stream is not None else sys.stdout
    if is_tty is None:
        isatty_fn = getattr(out, "isatty", None)
        is_tty = bool(isatty_fn()) if callable(isatty_fn) else False
    if not is_tty:
        return NullLiveToolbar()
    if term is not None and term.strip().lower() in ("", "dumb"):
        return NullLiveToolbar()
    if palette.mode is PaletteMode.PLAIN:
        return NullLiveToolbar()

    def _render(state: ChatCliState) -> str:
        return render_toolbar(state, palette=palette)

    return AnsiLiveToolbar(
        render=_render,
        output_stream=out,
    )


__all__ = [
    "AnsiLiveToolbar",
    "LiveToolbar",
    "NullLiveToolbar",
    "RollingTokRate",
    "make_live_toolbar",
]
