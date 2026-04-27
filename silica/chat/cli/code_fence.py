"""silica.chat.cli.code_fence — Markdown code-fence stream parser.

Detects ``` ```language ... ``` ``` blocks in streamed assistant
output. Outside fences, text passes through as :class:`PlainText`
events. Inside a fence, content is buffered until the closing
fence arrives — at that point the parser emits a single
:class:`ExitFence` event carrying the full buffered code and the
language identifier, which the renderer feeds to pygments for
syntax highlighting.

Why buffer: pygments lexers need a complete code block to lex
correctly (token colours depend on full context like string
boundaries, comment regions, balanced brackets). Streaming
character-by-character through a lexer would either re-render the
entire block on every keystroke (expensive, visually flickery)
or produce wrong colours for the first half of the block.
Buffering and emitting once is the cleanest contract.

Fence syntax recognised:

- Opening: ``\\n```python\\n`` or ``\\n``` \\n`` (language optional).
  The fence must start at the beginning of a line — three
  backticks anywhere mid-line are not a fence (handles
  programming-prose like "use \\`\\`\\` to start a block").
- Closing: ``\\n``` \\n`` or ``\\n``` `` at end of stream.

Tag-fragmentation handling mirrors :mod:`silica.chat.cli.thinking_parser`:
a buffer holds back any trailing text that is a proper prefix of
the relevant fence marker so split-tag deltas like ``"\\n``"``
followed by ``"`python\\n"`` produce exactly one EnterFence event.

Out of scope: nested fences (Markdown disallows them), inline
backticks (those are bare code spans, not fences), language
auto-detection beyond the explicit identifier (the renderer can
guess via :func:`pygments.lexers.guess_lexer` if the fence omits
the language).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class FenceEvent:
    """Marker base class for code-fence parser events."""


@dataclass(frozen=True)
class PlainText(FenceEvent):
    """Text outside any code fence — emit as-is to the
    conversation log (with the assistant colour palette)."""

    text: str


@dataclass(frozen=True)
class EnterFence(FenceEvent):
    """Opening ``` ```language``` line observed. Renderer should
    show a "writing code..." indicator while content streams; the
    eventual ExitFence event carries the highlighted code."""

    language: str
    """The language identifier from the fence (``"python"``,
    ``"rust"``, ``""`` if omitted). Passed to pygments'
    :func:`get_lexer_by_name`; an empty string triggers the
    renderer's fallback path (guess by content or render
    monochrome)."""


@dataclass(frozen=True)
class ExitFence(FenceEvent):
    """Closing ``` ``` ``` line observed. ``code`` is the full
    buffered text between Enter and Exit, with leading and
    trailing newlines stripped to match what pygments expects."""

    code: str
    language: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _State(Enum):
    PLAIN = "plain"
    FENCE = "fence"


_FENCE_MARKER = "```"
_NEWLINE = "\n"


def _longest_suffix_prefix(buffer: str, tag: str) -> int:
    """Length of the longest proper prefix of ``tag`` that is a
    suffix of ``buffer``. Mirrors the same helper in
    :mod:`silica.chat.cli.thinking_parser` — kept module-local
    rather than shared because the chat-CLI's parser layer is
    intentionally a flat collection of focused components."""
    max_overlap = min(len(buffer), len(tag) - 1)
    for size in range(max_overlap, 0, -1):
        if buffer[-size:] == tag[:size]:
            return size
    return 0


class CodeFenceParser:
    """Stream parser collapsing Markdown code fences into
    :class:`EnterFence` / :class:`ExitFence` events plus a single
    :class:`PlainText` for everything outside.

    Single-use per turn — construct fresh on each chat turn so the
    state machine starts clean. Construction has no side effects;
    fence-position tracking lives in instance fields.
    """

    def __init__(self) -> None:
        self._state: _State = _State.PLAIN
        self._buffer: str = ""
        self._code_buffer: str = ""
        self._fence_lang: str = ""
        # Whether the previous emitted text ended with a newline —
        # fence markers must start at the beginning of a line. The
        # buffer's first feed always behaves as if at line start.
        self._at_line_start: bool = True

    @property
    def state(self) -> str:
        return self._state.value

    def feed(self, text: str) -> list[FenceEvent]:
        """Consume one streamed delta. Returns events in order.

        The buffer holds back any trailing text that could form
        the start of a fence (so a partial ``"\\n``"`` plus
        following ``"`"`` arrive as exactly one EnterFence, not as
        a spurious PlainText("\\n``") leak). Outside-fence text
        emits as PlainText chunks; inside-fence text accumulates
        in ``self._code_buffer`` and emits as ExitFence on close.
        """
        events: list[FenceEvent] = []
        self._buffer += text
        while True:
            if self._state is _State.PLAIN:
                idx, lang_end = self._find_open_fence()
                if idx >= 0 and lang_end >= 0:
                    if idx > 0:
                        events.append(
                            PlainText(text=self._buffer[:idx])
                        )
                    self._fence_lang = self._buffer[
                        idx + len(_FENCE_MARKER) : lang_end
                    ].strip()
                    # Consume up to and including the newline
                    # that terminates the opening line.
                    consume_end = lang_end + 1  # past the \n
                    self._buffer = self._buffer[consume_end:]
                    self._state = _State.FENCE
                    self._code_buffer = ""
                    self._at_line_start = True
                    events.append(
                        EnterFence(language=self._fence_lang)
                    )
                    continue
                # No complete open fence — emit safe plain text,
                # hold back any trailing partial-fence candidate.
                hold = self._held_back_for_open()
                emit = (
                    self._buffer[: len(self._buffer) - hold]
                    if hold > 0
                    else self._buffer
                )
                self._buffer = (
                    self._buffer[len(self._buffer) - hold :]
                    if hold > 0
                    else ""
                )
                if emit:
                    self._at_line_start = emit.endswith(_NEWLINE)
                    events.append(PlainText(text=emit))
                break
            # _State.FENCE
            close_idx = self._find_close_fence()
            if close_idx >= 0:
                # ``close_idx`` is the position of the newline that
                # terminates the last code line; the closing
                # ``` starts at close_idx + 1. By Markdown
                # convention, that terminating newline IS part of
                # the code content (every line's trailing newline
                # is intrinsic to the line). Include it.
                self._code_buffer += self._buffer[: close_idx + 1]
                # Skip past the closing fence (and its optional
                # trailing newline).
                after_marker = (
                    close_idx + 1 + len(_FENCE_MARKER)
                )
                if (
                    after_marker < len(self._buffer)
                    and self._buffer[after_marker] == _NEWLINE
                ):
                    after_marker += 1
                self._buffer = self._buffer[after_marker:]
                events.append(
                    ExitFence(
                        code=self._code_buffer,
                        language=self._fence_lang,
                    )
                )
                self._state = _State.PLAIN
                self._code_buffer = ""
                self._fence_lang = ""
                self._at_line_start = True
                continue
            # No close yet — accumulate safe portion to code
            # buffer, hold back trailing partial-close candidate.
            hold = self._held_back_for_close()
            emit = (
                self._buffer[: len(self._buffer) - hold]
                if hold > 0
                else self._buffer
            )
            self._buffer = (
                self._buffer[len(self._buffer) - hold :]
                if hold > 0
                else ""
            )
            if emit:
                self._code_buffer += emit
            break
        return events

    def finish(self) -> list[FenceEvent]:
        """Drain any text held back as a partial-fence candidate.

        Call once after the engine emits its terminal event. An
        unfinished fence at EOS becomes a final ExitFence with
        whatever code was buffered (so highlighting still runs
        on truncated blocks); a held-back partial-open candidate
        becomes a final PlainText.
        """
        events: list[FenceEvent] = []
        if not self._buffer and not self._code_buffer:
            return events
        if self._state is _State.PLAIN:
            if self._buffer:
                events.append(PlainText(text=self._buffer))
                self._buffer = ""
            return events
        # _State.FENCE — flush remaining text into code buffer
        # and emit ExitFence with whatever we have. Renderer can
        # still attempt highlighting on the partial block; if
        # pygments fails the fallback path emits monochrome.
        if self._buffer:
            self._code_buffer += self._buffer
            self._buffer = ""
        events.append(
            ExitFence(
                code=self._code_buffer, language=self._fence_lang
            )
        )
        self._code_buffer = ""
        self._fence_lang = ""
        self._state = _State.PLAIN
        return events

    # --- helpers --------------------------------------------------

    def _find_open_fence(self) -> tuple[int, int]:
        """Locate a complete opening fence (``\\n```lang\\n`` or
        ``\\n``` \\n``) in the buffer.

        Returns ``(fence_start, lang_end)`` where ``fence_start``
        is the position of the first backtick and ``lang_end`` is
        the position of the terminating newline. Returns
        ``(-1, -1)`` when no complete fence is present.

        Fence must start at the beginning of a line — either the
        absolute start of buffer (when the parser is at line
        start) or after a newline character.
        """
        search_start = 0
        while search_start < len(self._buffer):
            idx = self._buffer.find(_FENCE_MARKER, search_start)
            if idx < 0:
                return -1, -1
            # Validate line-start: idx == 0 with self._at_line_start,
            # or buffer[idx-1] == '\n'.
            at_line_start = (
                idx == 0
                and self._at_line_start
            ) or (idx > 0 and self._buffer[idx - 1] == _NEWLINE)
            if not at_line_start:
                search_start = idx + 1
                continue
            # Find the terminating newline of the opening line.
            line_end = self._buffer.find(
                _NEWLINE, idx + len(_FENCE_MARKER)
            )
            if line_end < 0:
                return -1, -1
            return idx, line_end
        return -1, -1

    def _find_close_fence(self) -> int:
        """Locate a closing ``\\n``` `` in the buffer.

        Returns the position of the newline that precedes the
        closing backticks, or -1 if no complete close is present.
        The closing must be at line-start: ``\\n``` `` followed
        by either another newline or end-of-buffer.
        """
        search_start = 0
        while True:
            # Find a candidate ``` at the start of a line. The
            # in-fence content always follows a newline, so the
            # leading newline is part of the buffer (we consumed
            # the opener's terminating newline already, so the
            # next code line starts here).
            idx = self._buffer.find(
                _NEWLINE + _FENCE_MARKER, search_start
            )
            if idx < 0:
                return -1
            after_marker = idx + 1 + len(_FENCE_MARKER)
            # The close is valid if it's followed by a newline or
            # the end of the buffer (the last fence in the stream).
            if (
                after_marker == len(self._buffer)
                or self._buffer[after_marker] == _NEWLINE
            ):
                return idx
            # Otherwise it's ```{nonterminator}, e.g. ```python
            # which would be an opening fence inside the code
            # block (which Markdown disallows but defensively
            # search past).
            search_start = idx + 1

    def _held_back_for_open(self) -> int:
        """How many characters at the buffer tail must be held
        back as a possible partial open fence.

        Held-back contents form an "incomplete opening line": a
        line that starts with one or more backticks (potentially
        the start of a fence marker) but has not yet been
        terminated by a newline. The next streamed chunk might
        complete the fence (``\\n``\\n``); holding back lets us
        emit nothing until the line is whole. The bare trailing
        newline ("plain text\\n") is NOT held back — empty
        lines never form fence openings, and the
        ``_at_line_start`` flag we set on emission already lets
        the next feed correctly detect fences at chunk
        boundaries.
        """
        # Find the start of the last (potentially-incomplete) line.
        last_nl = self._buffer.rfind(_NEWLINE)
        if last_nl == -1:
            # Whole buffer is one line. Only hold back when this
            # line starts with backticks AND we are at a line
            # start position (start of stream or after an emitted
            # newline).
            if not self._at_line_start:
                return 0
            line_start = 0
        else:
            line_start = last_nl + 1
        line = self._buffer[line_start:]
        if not line or not line.startswith("`"):
            return 0
        # The line starts with backticks and we have not seen a
        # terminating newline (otherwise ``_find_open_fence``
        # would have returned a match). The line might be a
        # partial fence opener — hold the entire tail-line back
        # so neither the backticks nor any pending language
        # identifier characters leak as plain text.
        return len(self._buffer) - line_start

    def _held_back_for_close(self) -> int:
        """How many characters at the buffer tail must be held
        back as a possible partial close fence (``\\n``` ``)."""
        return _longest_suffix_prefix(
            self._buffer, _NEWLINE + _FENCE_MARKER
        )


__all__ = [
    "CodeFenceParser",
    "EnterFence",
    "ExitFence",
    "FenceEvent",
    "PlainText",
]
