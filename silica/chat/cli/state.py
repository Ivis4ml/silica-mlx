"""silica.chat.cli.state — `ChatCliState` dataclass.

The single live data structure that the chat CLI's UI layer reads
from. The toolbar formatter and (later) the conversation log
formatter both consume :class:`ChatCliState`; the prompt_toolkit
event loop and the slash command dispatcher both write to it.
Keeping the data plain-Python and side-effect-free here means the
formatter unit tests can drive any state combination without
spinning up the engine or the UI.

Field semantics match ``docs/CHAT_CLI_OPENING.md`` §4 (toolbar
field list) and §3.2.1 (thinking-mode state machine).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class StreamState(str, Enum):
    """Streaming state machine driving the toolbar's ``state=`` field.

    Transitions (per §3.2.1):

    - ``idle`` → ``prefill`` when a turn starts and prefill begins.
    - ``prefill`` → ``thinking`` when the first streamed token after
      prefill belongs to a ``<think>...</think>`` block.
    - ``prefill`` → ``decode`` when prefill ends with no thinking
      block (e.g. thinking disabled or non-reasoning model).
    - ``thinking`` → ``decode`` (renamed ``replying`` in the design
      doc; same identity as decode) when ``</think>`` closes.
    - ``decode`` → ``idle`` on EOS / ``max_tokens`` / abort.
    - Any state → ``paused`` if the engine yields without producing
      a token (rare; reserved for future P-7 speculative wait).

    The string values are stable identifiers — pick them rather
    than relying on enum identity in tests.
    """

    IDLE = "idle"
    PREFILL = "prefill"
    THINKING = "thinking"
    DECODE = "decode"
    PAUSED = "paused"


@dataclass
class ChatCliState:
    """Live mutable state for the chat CLI.

    Constructed once per session; the prompt_toolkit event loop
    mutates ``stream_state``, ``tokens_generated``, ``tok_per_sec``,
    and the ``last_*`` fields as tokens arrive. Slash command
    handlers mutate the configuration fields. The toolbar formatter
    reads the snapshot and renders a string.

    All numeric fields are ``Optional`` to handle the legitimate
    "not yet measured" early-state (e.g. ``ttft`` is unknown until
    the first token of the first turn arrives).
    """

    # ---- session-stable fields (set at construction) -------------------
    model_name: str = "(unset)"
    """Basename of the loaded HF repo, e.g. ``"Qwen3-0.6B"``. Updated
    by ``/model``."""

    codec_id: str | None = None
    """Active KV codec id (e.g. ``"block_tq_b64_b4"``); ``None`` for
    fp16. Set at launch, surfaced as ``compr=`` in the toolbar."""

    # ---- live engine state ---------------------------------------------
    stream_state: StreamState = StreamState.IDLE
    """Current scheduler + parser stream-state machine value."""

    turn: int = 0
    """1-indexed turn counter incremented on each successful reply."""

    tokens_generated: int = 0
    """Tokens emitted so far in the current turn (resets at turn start)."""

    max_tokens: int = 1024
    """Per-turn ``max_tokens`` ceiling, surfaced as ``tokens=N/max``
    in the toolbar."""

    tok_per_sec: float | None = None
    """Live decode tok/s during the current turn; ``None`` between
    turns or before the first decode tick."""

    # ---- per-turn snapshots --------------------------------------------
    last_ttft_ms: float | None = None
    """TTFT of the most recent completed turn."""

    peak_memory_mb: float | None = None
    """Device peak memory in MB (``mx.get_peak_memory()`` / 1e6).
    Sticky high-water mark; never resets within a session."""

    kv_resident_mb: float | None = None
    """Bytes currently held in the prefix store, divided by 1e6."""

    kv_logical_mb: float | None = None
    """fp16-equivalent bytes for the same data; codec-relevant."""

    prefix_store_mb: float | None = None
    """Total prefix-cache footprint (== ``kv_resident_mb`` once codec
    overhead is included). Surface for the 200 MB hint trigger
    described in design doc §6.2."""

    # ---- prefix-cache hit metrics (Tier 2) -----------------------------
    prefix_hit_blocks: int | None = None
    """Block-aligned tokens reused from the cache on the most recent
    admit."""

    prefix_hit_max: int | None = None
    """Total prompt tokens that *could* have matched (the divisor of
    the ``prefix_hit=N/M`` field)."""

    # ---- cumulative session counters (Tier 1 — /showcase narrative) ---
    total_prefix_hit_tokens: int = 0
    """Sum of per-turn ``prefix_hit_tokens`` across the session.
    Reset to 0 by ``/reset`` and ``/load``; the narrative report
    surfaces ``reused N tokens of prefix`` from this field."""

    total_decode_tokens: int = 0
    """Sum of per-turn ``output_tokens`` across the session — the
    numerator of the session-average decode tok/s."""

    total_decode_seconds: float = 0.0
    """Sum of per-turn decode wall-clock seconds (output_tokens /
    decode_tok_s) — the denominator of session-average decode tok/s.
    Computed from ``TurnMetrics.decode_tok_s`` rather than measured
    directly so the report matches the toolbar's tok/s definition."""

    # ---- thinking-mode buffer ------------------------------------------
    last_turn_thinking: str = ""
    """Buffered ``<think>...</think>`` content from the previous turn.
    Reprinted by ``/expand``. Empty until the first reasoning turn
    completes."""

    thinking_started_at: float | None = None
    """Wall-clock seconds (``time.monotonic()``) at which the current
    turn entered ``StreamState.THINKING``; used by the toolbar to
    render ``thinking... 4.2s`` and by the log formatter to compute
    ``thought for 6.3s``."""

    # ---- configuration (mutated by /config) ----------------------------
    config: dict[str, str] = field(default_factory=dict)
    """Runtime overrides applied via ``/config key=value``. Schema
    is open-ended (validated at command-dispatch time, not here)
    so the chat CLI can add new keys without changing the dataclass."""

    # ---- helpers -------------------------------------------------------
    def compression_ratio(self) -> float | None:
        """Return ``kv_logical_mb / kv_resident_mb`` when both are
        present and ``kv_resident_mb > 0``; otherwise ``None``.

        The toolbar renders this as ``compr=Xx`` when present and
        ``compr=—`` when ``None`` (which includes the codec=fp16
        default case where logical == resident gives a 1.0× ratio
        that we deliberately do not render to avoid implying
        compression where there is none).
        """
        if self.kv_logical_mb is None or self.kv_resident_mb is None:
            return None
        if self.kv_resident_mb <= 0.0:
            return None
        ratio = self.kv_logical_mb / self.kv_resident_mb
        # Under fp16 (no codec) logical == resident == same physical
        # bytes; ratio is 1.0 and we surface as "—" to avoid
        # misleading the user. Under any real codec the ratio is > 1.
        if self.codec_id is None and abs(ratio - 1.0) < 0.01:
            return None
        return ratio

    def has_prefix_hit(self) -> bool:
        """Whether prefix-hit fields are populated and meaningful for
        rendering. Tier 2 — false until ``ChatSession`` integrates
        ``RadixPrefixCache`` (C-4)."""
        return (
            self.prefix_hit_blocks is not None
            and self.prefix_hit_max is not None
            and self.prefix_hit_max > 0
        )

    def session_avg_decode_tok_s(self) -> float | None:
        """Session-average decode throughput. ``None`` until the
        first completed turn populates the cumulative counters."""
        if self.total_decode_seconds <= 0.0 or self.total_decode_tokens <= 0:
            return None
        return self.total_decode_tokens / self.total_decode_seconds
