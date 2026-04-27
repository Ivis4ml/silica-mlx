"""C-8 unit tests for the ``<think>``-stream parser.

Pure-Python coverage of ``silica.chat.cli.thinking_parser``. Verifies
the documented contract from ``docs/CHAT_CLI_OPENING.md`` §3.2.1
including the tag-fragmentation robustness that resolves Q-CHAT-4.
"""

from __future__ import annotations

import pytest

from silica.chat.cli.thinking_parser import (
    ParserEvent,
    ReplyChunk,
    ThinkingChunk,
    ThinkingParser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replies(events: list[ParserEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, ReplyChunk))


def _thoughts(events: list[ParserEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, ThinkingChunk))


def _kinds(events: list[ParserEvent]) -> list[str]:
    return [type(e).__name__ for e in events]


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_parser_starts_idle() -> None:
    p = ThinkingParser()
    assert p.state == "idle"


def test_empty_feed_returns_no_events() -> None:
    p = ThinkingParser()
    assert p.feed("") == []
    assert p.state == "idle"


# ---------------------------------------------------------------------------
# No tags — pure pass-through
# ---------------------------------------------------------------------------


def test_plain_text_emits_reply_chunk() -> None:
    p = ThinkingParser()
    events = p.feed("Hello, I am silica.")
    assert _kinds(events) == ["ReplyChunk"]
    assert _replies(events) == "Hello, I am silica."


def test_multiple_plain_chunks_concatenate() -> None:
    p = ThinkingParser()
    events_a = p.feed("Hello, ")
    events_b = p.feed("world.")
    assert _replies(events_a + events_b) == "Hello, world."
    assert p.state == "idle"


def test_text_with_unrelated_angle_brackets_passes_through() -> None:
    """`<` and `>` inside reply text without forming `<think>`
    must not trigger the parser. Common in code samples."""
    p = ThinkingParser()
    events = p.feed("Use <em>bold</em> for emphasis.")
    assert _replies(events) == "Use <em>bold</em> for emphasis."
    assert p.state == "idle"


# ---------------------------------------------------------------------------
# Single-chunk thinking block
# ---------------------------------------------------------------------------


def test_complete_thinking_block_in_one_chunk() -> None:
    p = ThinkingParser()
    events = p.feed(
        "<think>Reasoning step.</think>The actual reply."
    )
    assert _kinds(events) == [
        "EnterThinking",
        "ThinkingChunk",
        "ExitThinking",
        "ReplyChunk",
    ]
    assert _thoughts(events) == "Reasoning step."
    assert _replies(events) == "The actual reply."
    assert p.state == "idle"


def test_thinking_block_with_text_before_and_after() -> None:
    p = ThinkingParser()
    events = p.feed(
        "Pre-text <think>internal</think> post-text"
    )
    assert _replies(events) == "Pre-text  post-text"
    assert _thoughts(events) == "internal"
    assert _kinds(events) == [
        "ReplyChunk",
        "EnterThinking",
        "ThinkingChunk",
        "ExitThinking",
        "ReplyChunk",
    ]


# ---------------------------------------------------------------------------
# Multi-chunk thinking block
# ---------------------------------------------------------------------------


def test_thinking_block_split_across_chunks() -> None:
    p = ThinkingParser()
    out: list[ParserEvent] = []
    out += p.feed("<think>part one ")
    out += p.feed("part two</think>reply")
    assert _kinds(out) == [
        "EnterThinking",
        "ThinkingChunk",
        "ThinkingChunk",
        "ExitThinking",
        "ReplyChunk",
    ]
    assert _thoughts(out) == "part one part two"
    assert _replies(out) == "reply"


def test_open_tag_split_across_chunks() -> None:
    """Tag fragmentation: `<th` arrives in chunk N, `ink>` in
    chunk N+1. Parser must NOT emit `<th` as a ReplyChunk and
    must transition to thinking only on chunk N+1."""
    p = ThinkingParser()
    e1 = p.feed("hello <th")
    e2 = p.feed("ink>internal</think>reply")
    # Chunk 1: only the safe prefix "hello " (the "<th" is held).
    assert _kinds(e1) == ["ReplyChunk"]
    assert _replies(e1) == "hello "
    # Chunk 2: enter, internal text, exit, reply.
    assert _kinds(e2) == [
        "EnterThinking",
        "ThinkingChunk",
        "ExitThinking",
        "ReplyChunk",
    ]
    assert _thoughts(e2) == "internal"
    assert _replies(e2) == "reply"


def test_close_tag_split_across_chunks() -> None:
    p = ThinkingParser()
    out: list[ParserEvent] = []
    out += p.feed("<think>some thinking</")
    out += p.feed("think>reply text")
    assert _kinds(out) == [
        "EnterThinking",
        "ThinkingChunk",
        "ExitThinking",
        "ReplyChunk",
    ]
    assert _thoughts(out) == "some thinking"
    assert _replies(out) == "reply text"


@pytest.mark.parametrize(
    "split_at",
    [1, 2, 3, 4, 5, 6],  # split inside the literal "<think>" prefix
)
def test_open_tag_split_at_every_offset(split_at: int) -> None:
    """For each possible internal split of `<think>`, the parser
    must produce exactly one EnterThinking event regardless of
    which character boundary the chunk arrives on."""
    full = "prefix <think>inner</think>tail"
    cut = full.find("<think>") + split_at
    p = ThinkingParser()
    out = p.feed(full[:cut]) + p.feed(full[cut:])
    assert _replies(out) == "prefix tail"
    assert _thoughts(out) == "inner"
    assert _kinds(out).count("EnterThinking") == 1
    assert _kinds(out).count("ExitThinking") == 1


def test_per_character_feed_extreme_fragmentation() -> None:
    """Per-character feed exposes any fragmentation issue. Every
    character of a complete `<think>...</think>` block is fed
    individually; final content must still parse correctly."""
    full = "before <think>abc</think>after"
    p = ThinkingParser()
    out: list[ParserEvent] = []
    for ch in full:
        out += p.feed(ch)
    assert _replies(out) == "before after"
    assert _thoughts(out) == "abc"
    assert _kinds(out).count("EnterThinking") == 1
    assert _kinds(out).count("ExitThinking") == 1


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


def test_state_switches_to_thinking_on_open_tag() -> None:
    p = ThinkingParser()
    p.feed("<think>")
    assert p.state == "thinking"


def test_state_returns_to_idle_on_close_tag() -> None:
    p = ThinkingParser()
    p.feed("<think>x</think>")
    assert p.state == "idle"


def test_state_stays_thinking_until_close_tag() -> None:
    p = ThinkingParser()
    p.feed("<think>still going")
    assert p.state == "thinking"
    p.feed(" further")
    assert p.state == "thinking"
    p.feed("</think>done")
    assert p.state == "idle"


# ---------------------------------------------------------------------------
# False-alarm partial tags
# ---------------------------------------------------------------------------


def test_partial_open_tag_followed_by_unrelated_text() -> None:
    """If text starts what looks like `<think>` but then doesn't
    match (e.g. `<thank you>`), the parser must release the
    held-back text as a normal ReplyChunk on the next feed that
    breaks the partial-match assumption."""
    p = ThinkingParser()
    e1 = p.feed("Note: <th")
    # Before next feed, "<th" is held — buffer is partial-tag.
    assert _replies(e1) == "Note: "
    e2 = p.feed("ank you")
    # "<th" + "ank you" = "<thank you" — no longer matches "<think>".
    # Parser must release the full text including "<th".
    assert _replies(e2) == "<thank you"
    assert p.state == "idle"


def test_partial_close_tag_inside_thinking_followed_by_normal_text() -> None:
    """`</thi` then `nk` — second chunk does not complete the
    close tag. Parser must release `</think...` wait no, here the
    sequence is `</thi` + `nk` = `</think` (still incomplete; one
    char short of `</think>`). Parser must keep both chars held
    until a `>` arrives or some non-matching char arrives."""
    p = ThinkingParser()
    out: list[ParserEvent] = []
    out += p.feed("<think>reasoning </thi")
    out += p.feed("nk")  # `</think` — one char shy of `</think>`
    # Both feeds have produced nothing yet for the dangling tag.
    # Total released = "reasoning " only.
    assert _thoughts(out) == "reasoning "
    # Now break the partial: next char is whitespace, not `>`.
    out += p.feed(" but not closed</think>actual reply")
    assert _thoughts(out) == "reasoning </think but not closed"
    assert _replies(out) == "actual reply"


# ---------------------------------------------------------------------------
# finish()
# ---------------------------------------------------------------------------


def test_finish_flushes_held_partial_tag_in_idle() -> None:
    """If the model emits `<th` and then EOS without finishing
    the open tag, finish() must release the held buffer as a
    regular ReplyChunk so it is not silently dropped."""
    p = ThinkingParser()
    p.feed("partial <th")
    out = p.finish()
    assert _replies(out) == "<th"
    # Next feed should start clean (parser is single-use; this
    # behaviour is documented but worth pinning).


def test_finish_flushes_pending_thinking_text() -> None:
    """If the model emits `<think>` + content but never `</think>`
    (perhaps EOS truncated the reasoning), the in-flight thinking
    text is captured as ThinkingChunk events. The non-tag content
    is emitted during ``feed`` (because it does not look like a
    partial close-tag suffix); ``finish`` only releases anything
    held back as a partial-tag candidate. This test verifies
    that the model's reasoning ends up in the event stream as a
    ThinkingChunk so /expand has the content available, even if
    the close tag never arrives."""
    p = ThinkingParser()
    out_feed = p.feed("<think>reasoning was cut")
    out_finish = p.finish()
    out = out_feed + out_finish
    assert _thoughts(out) == "reasoning was cut"


def test_finish_flushes_held_partial_close_tag() -> None:
    """If the model emits `<think>some text</thi` and EOS,
    `</thi` is held back as a partial-close-tag candidate during
    feed; finish() releases it as the trailing ThinkingChunk."""
    p = ThinkingParser()
    out_feed = p.feed("<think>some text</thi")
    out_finish = p.finish()
    out = out_feed + out_finish
    # "some text" arrives during feed; "</thi" arrives at finish.
    assert _thoughts(out) == "some text</thi"


def test_finish_with_empty_buffer_returns_no_events() -> None:
    p = ThinkingParser()
    p.feed("clean reply text")
    out = p.finish()
    assert out == []


# ---------------------------------------------------------------------------
# Multiple thinking blocks
# ---------------------------------------------------------------------------


def test_multiple_thinking_blocks_in_one_stream() -> None:
    """A reply that emits two separate think blocks (rare but
    not impossible if model produces multi-step reasoning) must
    transition idle → thinking → idle → thinking → idle, with
    exactly two EnterThinking + two ExitThinking events."""
    p = ThinkingParser()
    out = p.feed(
        "<think>step1</think>partial reply <think>step2</think>final"
    )
    kinds = _kinds(out)
    assert kinds.count("EnterThinking") == 2
    assert kinds.count("ExitThinking") == 2
    assert _thoughts(out) == "step1step2"
    assert _replies(out) == "partial reply final"
