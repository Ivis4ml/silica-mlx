"""silica.chat.cli.thinking_parser — stream parser for ``<think>`` blocks.

Qwen3 / Qwen3.5 family models emit ``<think>...</think>`` blocks
before the actual reply when reasoning mode is on (the default for
those families). Streamed verbatim, the chain-of-thought floods the
conversation log with internal deliberation; modern chat apps
collapse it into a single-line indicator and stream the actual
reply afterwards. This module owns the state-machine parser the
chat CLI uses to do that.

Design contract per ``docs/CHAT_CLI_OPENING.md`` §3.2.1:

- Streaming state: ``IDLE`` (no opening tag seen) and ``THINKING``
  (between ``<think>`` and ``</think>``).
- ``feed(text)`` consumes one streamed delta and returns a list of
  :class:`ParserEvent` describing what the consumer should display.
  No I/O, no time, no colour — pure transformation.
- Tag fragmentation across stream chunks is handled by holding back
  any trailing buffer that is a proper prefix of the relevant tag.
  ``<th`` arriving in chunk N and ``ink>`` in chunk N+1 produces
  exactly one ``EnterThinking`` event at the chunk-N+1 boundary; no
  partial ``ReplyChunk("<th")`` leaks out beforehand.
- A ``finish()`` call drains any held-back text. Callers should
  invoke this after the engine's terminal event so trailing partial
  tags do not silently disappear (they would only appear if the
  model emitted a malformed ``<think`` followed by EOS without
  closing — defensive but cheap).

Out of scope: the elapsed-time field next to the thinking
indicator is the consumer's responsibility (the parser is pure).
The chat-CLI shell starts a wall-clock timer on
:class:`EnterThinking` and renders the elapsed seconds inline; the
parser stays free of timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class ParserEvent:
    """Marker base class for parser events. Concrete subclasses
    encode the decision a downstream renderer makes."""


@dataclass(frozen=True)
class EnterThinking(ParserEvent):
    """``<think>`` opening tag observed. Renderer should switch
    to the magenta ``thinking...`` indicator and stop streaming
    content into the reply buffer."""


@dataclass(frozen=True)
class ExitThinking(ParserEvent):
    """``</think>`` closing tag observed. Renderer should clear
    the indicator and switch to streaming reply tokens normally."""


@dataclass(frozen=True)
class ThinkingChunk(ParserEvent):
    """Text content captured between ``<think>`` and ``</think>``.

    The chat CLI accumulates these into ``ChatCliState.last_turn_thinking``
    so ``/expand`` can reprint the model's reasoning afterwards.
    Multiple chunks may arrive between one Enter and one Exit
    event when the streamed delta straddles a tag boundary."""

    text: str


@dataclass(frozen=True)
class ReplyChunk(ParserEvent):
    """Text content outside any thinking block. Renderer should
    write this to stdout as part of the assistant reply."""

    text: str


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class _State(Enum):
    IDLE = "idle"
    THINKING = "thinking"


_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def _longest_suffix_prefix(buffer: str, tag: str) -> int:
    """Return the length of the longest *proper* prefix of ``tag``
    that is a suffix of ``buffer``.

    Used to decide how much of the buffer's tail must be held back
    on the chance that the next streamed chunk completes a tag.
    Example: if ``tag = "<think>"`` and ``buffer`` ends in ``"<th"``,
    return 3 — those three characters might be the start of
    ``<think>`` and must not be emitted yet. If ``buffer`` ends in
    text that does not partially match any prefix of ``tag``,
    return 0.

    A whole-tag match (``buffer`` ends with the full tag) returns
    ``len(tag) - 1`` because the parser dispatches whole-tag
    matches via ``str.find`` first; this helper covers the
    "incomplete tag at the tail" case only.
    """
    max_overlap = min(len(buffer), len(tag) - 1)
    for size in range(max_overlap, 0, -1):
        if buffer[-size:] == tag[:size]:
            return size
    return 0


class ThinkingParser:
    """Stream parser collapsing ``<think>...</think>`` blocks into
    semantic events.

    Construction has no side effects. Call :meth:`feed` once per
    streamed delta and consume the returned events; call
    :meth:`finish` once after the terminal stream event so any
    text held back as a partial-tag candidate is flushed.

    The parser is single-use per turn — construct a new instance
    on each chat turn so the buffer is empty at the start.
    """

    def __init__(self) -> None:
        self._state: _State = _State.IDLE
        self._buffer: str = ""

    @property
    def state(self) -> str:
        """Current parser state as a stable string identifier:
        ``"idle"`` or ``"thinking"``. Useful for the chat CLI's
        toolbar ``state=`` field rendering and tests."""
        return self._state.value

    def feed(self, text: str) -> list[ParserEvent]:
        """Consume one streamed delta. Returns the events the
        renderer should react to, in order.

        The returned list may be empty (delta ended inside a
        partial tag candidate; nothing to do yet) or contain
        multiple events when the delta crosses a tag boundary.
        """
        events: list[ParserEvent] = []
        self._buffer += text
        while True:
            if self._state is _State.IDLE:
                idx = self._buffer.find(_OPEN_TAG)
                if idx >= 0:
                    if idx > 0:
                        events.append(
                            ReplyChunk(text=self._buffer[:idx])
                        )
                    self._buffer = self._buffer[idx + len(_OPEN_TAG):]
                    self._state = _State.THINKING
                    events.append(EnterThinking())
                    continue
                # No complete open tag — emit any safe portion of
                # the buffer and hold back the trailing partial-tag
                # candidate.
                hold = _longest_suffix_prefix(self._buffer, _OPEN_TAG)
                if hold == 0:
                    if self._buffer:
                        events.append(ReplyChunk(text=self._buffer))
                        self._buffer = ""
                    break
                emit = self._buffer[: len(self._buffer) - hold]
                self._buffer = self._buffer[len(self._buffer) - hold :]
                if emit:
                    events.append(ReplyChunk(text=emit))
                break
            # _State.THINKING
            idx = self._buffer.find(_CLOSE_TAG)
            if idx >= 0:
                if idx > 0:
                    events.append(
                        ThinkingChunk(text=self._buffer[:idx])
                    )
                self._buffer = self._buffer[idx + len(_CLOSE_TAG):]
                self._state = _State.IDLE
                events.append(ExitThinking())
                continue
            hold = _longest_suffix_prefix(self._buffer, _CLOSE_TAG)
            if hold == 0:
                if self._buffer:
                    events.append(ThinkingChunk(text=self._buffer))
                    self._buffer = ""
                break
            emit = self._buffer[: len(self._buffer) - hold]
            self._buffer = self._buffer[len(self._buffer) - hold :]
            if emit:
                events.append(ThinkingChunk(text=emit))
            break
        return events

    def finish(self) -> list[ParserEvent]:
        """Flush any text held back as a partial-tag candidate.

        Call once after the engine emits its terminal event. The
        parser does not auto-close an unfinished thinking block —
        a model that started ``<think>`` without closing it
        produces a final ``ThinkingChunk`` rather than a
        synthetic ``ExitThinking`` (the contract is "what the
        model emitted", not "what would have been valid").
        """
        events: list[ParserEvent] = []
        if not self._buffer:
            return events
        if self._state is _State.IDLE:
            events.append(ReplyChunk(text=self._buffer))
        else:
            events.append(ThinkingChunk(text=self._buffer))
        self._buffer = ""
        return events


__all__ = [
    "EnterThinking",
    "ExitThinking",
    "ParserEvent",
    "ReplyChunk",
    "ThinkingChunk",
    "ThinkingParser",
]
