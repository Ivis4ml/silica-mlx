"""silica.chat.session — multi-turn chat loop over ``silica.engine.Engine``.

Scope: the minimum surface a human can open a REPL against and
see TTFT / throughput / KV / peak memory after every turn. The
session keeps OpenAI-style ``messages`` state, renders the prompt
via the tokenizer's ``apply_chat_template`` (with a Qwen-style
``<|im_start|>`` fallback when the tokenizer does not expose one),
and returns a fully-populated :class:`TurnMetrics` on every chat
call so callers do not have to read ``engine.metrics`` and MLX
peak memory separately.

What this is deliberately NOT:

  * A server. No HTTP / WebSocket / session-id routing — that is
    P-8.
  * A conversational agent. No tool calling, no structured
    output parsing, no retrieval. The module is a thin loop,
    not a framework.
  * A benchmark runner. :mod:`silica.bench` owns the
    scenarios / oracles / JSONL schema. The chat session emits
    per-turn metrics to its caller; a caller that wants to
    aggregate should own that aggregation.

Design constraints:

  * Reuses ``Engine.generate`` unchanged so any engine-level
    optimisation immediately benefits the chat path.
  * ``Engine.generate_batch`` is intentionally not wired in —
    single-request per session is what a user-facing REPL
    needs, and multi-user concurrent chat belongs to the P-8
    session layer.
  * Peak-memory probe is injectable. Tests provide a stub;
    on-device the defaults wrap ``mlx.core.{reset,get}_peak_memory``
    and silently no-op when MLX is missing (mirrors the runner's
    pattern).
  * Streaming is opt-in via a callback: ``stream_to`` receives
    each decoded text delta as tokens arrive. Deltas are
    computed by decoding the cumulative token buffer on every
    step and slicing off the prefix already emitted — simple,
    works with any BPE-style tokenizer, no separate streaming
    detokenizer needed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from silica.core.events import BatchEvent
from silica.core.sampling import SamplingParams
from silica.models.adapter import ModelAdapter


class _EngineLike(Protocol):
    """Narrow Engine surface the session actually consumes.

    The real :class:`silica.engine.Engine` trivially satisfies
    this; tests can pass any object that exposes the three
    attributes. Kept at Protocol level so importing this module
    does not pull ``silica.engine`` when only the session
    dataclasses are needed.

    ``generate_batch`` is only consumed when the session was
    constructed with a ``prefix_cache``; engines that do not
    support batched generation can still be used in single-request
    mode (the tests' minimal fakes work without it).
    """

    metrics: Any
    kv_manager: Any

    def generate(
        self, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[int]: ...

    def generate_batch(
        self,
        prompts: Any,
        params: SamplingParams | list[SamplingParams] | None = None,
        *,
        max_batch_size: int | None = None,
        prefix_cache: Any = None,
        length_spread_threshold: float = 2.0,
    ) -> Iterator[BatchEvent]: ...


class _PrefixCacheLike(Protocol):
    """Narrow ``RadixPrefixCache`` surface the session reads.

    Pulled out as a Protocol so the unit tests can inject a fake
    cache without depending on the real ``silica.kvcache.prefix``
    construction (which needs a store, codec, etc.). The session
    only needs to peek at the hit count for one turn's prompt;
    cache lifecycle (construction, replacement on ``/reset``)
    lives with the caller — see ``set_prefix_cache``."""

    block_size: int

    def peek(self, tokens: Any) -> Any: ...


@dataclass
class TurnMetrics:
    """One chat-turn outcome.

    Parallel in spirit to ``ScenarioResult`` but sized for a
    REPL: always populates the per-turn reply text plus the
    throughput / memory signals a human watches while iterating.

    ``finish_reason`` takes one of:

      * ``"stop_token"`` — last emitted token was in the
        tokenizer's EOS set (vLLM convention: the stop token is
        yielded before termination).
      * ``"max_tokens"`` — generation hit the configured cap.
      * ``"empty"`` — generator yielded nothing (empty prompt
        or immediate stop).
    """

    reply: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    raw_reply: str = ""
    """Full decoded reply text *before* any ``thinking_history``
    strip. Equal to :attr:`reply` when ``thinking_history=keep``
    or when the model produced no thinking content; differs when
    a ``<think>...</think>`` block was removed at finalise time.
    Default ``""`` keeps backward compatibility for tests /
    callers that construct ``TurnMetrics`` without the field
    (RP-1)."""
    ttft_ms: float | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    resident_mb: float | None = None
    peak_memory_mb: float | None = None
    logical_kv_bytes: int | None = None
    wall_s: float | None = None
    prefix_hit_blocks: int | None = None
    """Number of prefix-cache blocks reused for this turn's
    prompt. ``None`` when the session was constructed without a
    prefix cache (single-request path); ``0`` when the cache is
    present but the prompt did not match any cached prefix."""

    prefix_hit_tokens: int | None = None
    """Number of prompt tokens covered by ``prefix_hit_blocks``.
    Equal to ``prefix_hit_blocks * cache.block_size`` for an
    aligned hit. ``None`` mirrors the meaning of
    ``prefix_hit_blocks``."""

    prefix_store_resident_bytes: int | None = None
    """Cumulative bytes held by the prefix-cache store after this
    turn completes. Includes blocks inserted by every previous
    turn this session. ``None`` when no prefix cache is wired or
    when the store does not implement ``resident_bytes()``
    (PagedPrefixBlockStore today). Surfaced on the chat-CLI
    toolbar's ``kv=`` field — ``engine.kv_manager.budget()`` only
    reports the active per-row KV (which goes to zero between
    turns), so the prefix-store figure is the right "how much KV
    is in use right now" answer for the chat workload."""

    prefix_store_logical_bytes: int | None = None
    """fp16-equivalent bytes for the same data the prefix store
    holds. Equal to ``prefix_store_resident_bytes`` when no codec
    is bound (raw fp16 storage); larger by the codec's compression
    ratio when BlockTQ / RaBitQ is active. Surfaced on the
    toolbar's ``kv_log=`` field; combined with
    ``prefix_store_resident_bytes`` it drives the ``compr=`` ratio
    display."""


class ChatSession:
    """Stateful multi-turn chat loop.

    Construct once per ``(adapter, engine)`` pair; call
    :meth:`chat` repeatedly. The session retains message history
    until :meth:`reset` drops everything except the original
    system prompt.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        engine: _EngineLike,
        *,
        system_prompt: str | None = None,
        prefix_cache: _PrefixCacheLike | None = None,
        reset_peak_memory: Callable[[], None] | None = None,
        read_peak_memory_mb: Callable[[], float | None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        thinking_mode: bool | None = None,
        thinking_history: str = "strip",
        implicit_thinking_supported: bool = False,
    ) -> None:
        self._adapter = adapter
        self._engine = engine
        self._tokenizer = adapter.tokenizer()
        self._messages: list[dict[str, str]] = []
        if system_prompt is not None:
            self._messages.append(
                {"role": "system", "content": system_prompt}
            )
        self._eos_ids = tuple(
            sorted(getattr(self._tokenizer, "eos_token_ids", set()) or ())
        )
        self._reset_peak = reset_peak_memory or _mlx_reset_peak_memory
        self._read_peak_mb = read_peak_memory_mb or _mlx_peak_memory_mb
        self._clock = clock
        self._prefix_cache: _PrefixCacheLike | None = prefix_cache
        # CHAT-CLI-HARDENING-2 (F2): when non-None, ``_render_prompt``
        # forwards ``enable_thinking=self._thinking_mode`` to the
        # tokenizer's ``apply_chat_template``. ``None`` (the default)
        # omits the kwarg, preserving backward-compat for tokenizers /
        # templates that do not recognise it.
        self._thinking_mode: bool | None = thinking_mode
        # CHAT-CLI-RESPONSE-POLICY RP-1 (G1): controls whether the
        # ``<think>...</think>`` content is stripped from the
        # assistant message before it lands in ``_messages``. See
        # ``set_thinking_history`` for the value semantics.
        self._thinking_history: str = self._validate_thinking_history(
            thinking_history
        )
        # Whether the active model's chat template prepends
        # ``<think>\n`` to the prompt (Qwen3 family). Used by the
        # strip helper to decide whether implicit-leading
        # reasoning (no opening tag in the reply) should be
        # detected. Mirrors the chat-CLI's display-side parser
        # logic in ``_model_supports_implicit_thinking``.
        self._implicit_thinking_supported: bool = (
            implicit_thinking_supported
        )
        # When the previous turn ended at ``finish_reason=max_tokens``
        # under ``thinking_history=strip``, the assistant message
        # is left in raw form so RP-2 ``/continue`` can resume
        # mid-``<think>``. The next ``chat()`` call finalises the
        # raw form before appending its new user message; until
        # then the flag tells callers (and ``/continue``) that the
        # last assistant message is *not* the post-strip text.
        self._pending_finalize: bool = False
        # Snapshot of ``_should_strip_implicit_leading()`` taken at
        # the moment a turn entered deferred-finalise. The deferred
        # path consults THIS snapshot instead of the live value so
        # a ``/config thinking_mode=off`` flipped between the
        # truncated turn and the next user message cannot retroactively
        # change which strip shape applies to the orphaned raw text.
        # ``None`` means no defer is registered.
        self._pending_finalize_implicit_leading: bool | None = None
        # CHAT-CLI-RESPONSE-POLICY RP-2: continuation-side snapshot
        # of the truncation-time implicit-leading fact. Tracks the
        # SAME information as ``_pending_finalize_implicit_leading``
        # but with a different lifecycle and broader scope: this
        # field is set on ANY ``finish_reason="max_tokens"`` turn
        # regardless of ``thinking_history`` (strip OR keep), so
        # ``continue_last`` can rebuild the original generation
        # prompt's ``<think>\n`` prefix even under keep mode where
        # the finalise snapshot is unset. Lifecycle: set on
        # max_tokens (preserved across chained continuations);
        # cleared on natural completion, new user turn, ``reset``,
        # ``replace_messages``, ``pop_last_exchange``. ``None``
        # means no /continue is reachable.
        self._pending_continuation_implicit_leading: bool | None = None

    # --- observation -------------------------------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return a deep-ish copy of the current message history.

        Each call produces fresh dict objects so callers can
        snapshot the history before invoking mutator methods. RP-2
        ``/continue`` relies on this for the abort-rollback path —
        ``continue_last`` mutates ``self._messages[-1]["content"]``
        in place after generation completes; if the snapshot the
        shell holds shared dict references with the live list, an
        exception thrown after that mutation but before the
        function returned would leave the snapshot already
        corrupted and ``replace_messages(snapshot)`` would not
        restore the original assistant content.
        """
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages
        ]

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_ids

    @property
    def pending_continuation_implicit_leading(self) -> bool | None:
        """Truncation-time implicit-leading snapshot for ``/continue``.

        CHAT-CLI-RESPONSE-POLICY RP-2. Read by the chat-CLI shell
        to seed the streaming :class:`ThinkingParser` based on the
        FACT recorded at the truncated turn rather than the LIVE
        ``thinking_mode`` (which the user may have flipped between
        turns). Independent of ``thinking_history``: set on any
        ``finish_reason="max_tokens"`` regardless of strip / keep
        policy. ``None`` means no continuation is pending — fresh
        session, last turn completed naturally, or the prior
        truncated turn was dropped via reset / replace / pop.
        """
        return self._pending_continuation_implicit_leading

    @property
    def prefix_cache(self) -> _PrefixCacheLike | None:
        """Active prefix cache, or ``None`` if the session is
        single-request (``engine.generate``) only."""
        return self._prefix_cache

    # --- mutation ----------------------------------------------------

    def reset(self) -> None:
        """Drop every message except the original system prompt.

        Note: the prefix cache itself is **not** cleared here —
        ``RadixPrefixCache`` has no public ``clear()`` and replacing
        the instance is the caller's responsibility (the chat-CLI
        shell creates a fresh cache on ``/reset`` and assigns it
        via :meth:`set_prefix_cache`). Leaving the previous cache
        intact would leak prior-conversation tokens into the new
        conversation; the shell is expected to swap it out.

        RP-1 follow-up: clears the deferred-finalise pending flag.
        Any truncated assistant message is gone after the reset, so
        the next ``chat()`` must not try to strip a non-existent
        (or replaced) message.
        """
        self._messages = [
            m for m in self._messages if m["role"] == "system"
        ]
        self._pending_finalize = False
        self._pending_finalize_implicit_leading = None
        self._pending_continuation_implicit_leading = None

    def replace_messages(
        self, messages: list[dict[str, str]]
    ) -> None:
        """Replace the message history wholesale.

        Used by the chat-CLI's ``/load`` to restore a conversation
        from disk. The caller is responsible for swapping the
        prefix cache as well — the new history is unrelated to
        whatever was previously cached, so retaining the old cache
        would leak stale blocks into the restored session. This
        mirrors the contract documented on :meth:`reset`.

        RP-1 follow-up: clears the deferred-finalise pending flag.
        The prior session's truncated assistant (if any) is no
        longer the trailing message after a wholesale replace, so
        the strip-on-next-turn path has no valid target.
        """
        self._messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]
        self._pending_finalize = False
        self._pending_finalize_implicit_leading = None
        self._pending_continuation_implicit_leading = None

    def pop_last_exchange(self) -> str | None:
        """Drop the most recent ``(user, assistant)`` pair from history.

        Returns the popped user content, or ``None`` if the message
        log does not end with the canonical ``[..., user, assistant]``
        shape — fresh session, system-only history, or a turn that
        aborted mid-generation leaving only a user message. Caller
        pairs the pop with a fresh ``chat(returned_text)`` to
        regenerate the dropped turn.

        CHAT-CLI-HARDENING-4 (F4): the chat-CLI's ``/regenerate``
        command pre-HARDENING-4 only printed "(not wired yet)". This
        method is the session-side support for the wire-up; the
        shell pops, then re-issues ``chat()`` with the returned user
        prompt. The prefix cache is intentionally **not**
        invalidated — the same user text re-tokenises to the same
        prompt ids, so the cache's ``peek`` hits every block of the
        prior turn's prefill (Q-012 cross-call prefix reuse,
        resolved at v1.7.15).

        Strict shape: requires the last two messages to be exactly
        ``user`` then ``assistant``. A trailing user-only message
        (mid-generation abort) returns ``None`` rather than
        regenerating against a half-broken history.
        """
        if len(self._messages) < 2:
            return None
        if self._messages[-1]["role"] != "assistant":
            return None
        if self._messages[-2]["role"] != "user":
            return None
        self._messages.pop()  # assistant
        user_msg = self._messages.pop()
        # RP-1 follow-up: the popped assistant message is gone, so
        # any deferred-finalise flag that targeted it is stale. The
        # ``/regenerate`` flow then re-issues ``chat()`` with the
        # popped user text, which would otherwise see a stale flag
        # and try to strip a now-replaced message.
        # RP-2 follow-up: same applies to the continuation-side
        # snapshot — the popped assistant turn cannot be /continue'd.
        self._pending_finalize = False
        self._pending_finalize_implicit_leading = None
        self._pending_continuation_implicit_leading = None
        return user_msg["content"]

    @staticmethod
    def _validate_thinking_history(value: str) -> str:
        if value not in ("strip", "keep"):
            raise ValueError(
                f"thinking_history must be 'strip' or 'keep', got {value!r}"
            )
        return value

    def _resolve_finalise_implicit_leading(self) -> bool:
        """Pick the right ``implicit_leading`` value for a strip-finalise.

        CHAT-CLI-RESPONSE-POLICY RP-2 v3. Three-tier fallback:

        1. ``_pending_finalize_implicit_leading`` — set when the
           deferred-finalise path captured a strip-mode truncation
           snapshot. Most specific.
        2. ``_pending_continuation_implicit_leading`` — set on any
           ``finish_reason="max_tokens"`` turn (including keep
           mode). Covers the cross-policy case where a user
           truncated under keep then flipped to strip before
           ``/continue`` reached natural completion: the live
           ``thinking_mode`` no longer reflects the prompt-level
           fact, but the continuation snapshot does.
        3. ``_should_strip_implicit_leading()`` — degenerate
           fallback when no snapshot is registered (e.g.
           ``continue_last`` invoked on a naturally-completed
           turn).

        Order matters: the strip-finalise snapshot is the most
        specific (only set when a strip-mode chat() truncation
        registered finalise); the continuation snapshot is broader
        (any max_tokens registers it); live state is the last
        resort.
        """
        if self._pending_finalize_implicit_leading is not None:
            return self._pending_finalize_implicit_leading
        if self._pending_continuation_implicit_leading is not None:
            return self._pending_continuation_implicit_leading
        return self._should_strip_implicit_leading()

    def _should_strip_implicit_leading(self) -> bool:
        """Whether ``_strip_thinking_block`` should treat the
        decoded reply as starting *inside* an implicit thinking
        block.

        True iff the active model's template prepends
        ``<think>\\n`` to the assistant slot AND this turn was
        rendered with ``enable_thinking != False`` (the template
        default for Qwen3 is ``True`` when the kwarg is omitted,
        so ``_thinking_mode is None`` also counts as "thinking
        was on for this turn").
        """
        if not self._implicit_thinking_supported:
            return False
        return self._thinking_mode is not False

    def set_thinking_history(self, mode: str) -> None:
        """Replace the live session's ``thinking_history`` policy.

        CHAT-CLI-RESPONSE-POLICY RP-1. Controls whether the
        ``<think>...</think>`` content is stripped from the
        assistant message before it lands in :attr:`messages`.

        - ``"strip"`` (default): on natural turn completion
          (``finish_reason in {stop_token, done, eos}``) the
          stripped text is what gets stored. On
          ``finish_reason=max_tokens`` the raw text is stored
          temporarily so RP-2 ``/continue`` can resume an
          unfinished ``<think>`` block; the next ``chat()`` call
          finalises (strips) the raw form before appending the
          new user message.
        - ``"keep"``: the raw decoded reply is stored verbatim,
          regardless of ``finish_reason``. Right setting for
          ``/save``-then-archive workflows that want full
          reasoning traces preserved.

        The display side (``thinking=auto|show|hidden`` in the
        chat-CLI config) is independent — this knob only affects
        what lands in :attr:`messages` and what subsequent turns
        re-tokenise.
        """
        self._thinking_history = self._validate_thinking_history(mode)

    def set_implicit_thinking_supported(self, value: bool) -> None:
        """Set whether the active chat template prepends
        ``<think>\\n`` to the assistant generation slot.

        CHAT-CLI-RESPONSE-POLICY RP-1. Used by
        ``thinking_history=strip`` to decide whether the reply
        starts inside an implicit thinking block (Qwen3 family
        with ``enable_thinking=True``) and therefore needs the
        leading reasoning stripped up to the first ``</think>``.
        On ``/model`` swap the chat-CLI updates this via
        :meth:`set_implicit_thinking_supported` to match the new
        model's family.
        """
        self._implicit_thinking_supported = value

    def set_thinking_mode(self, mode: bool | None) -> None:
        """Replace the live session's ``enable_thinking`` propagation.

        CHAT-CLI-HARDENING-2 (F2): the chat-CLI's
        ``/config thinking_mode=on|off`` command pre-HARDENING-2
        flipped the parser-side ``start_in_thinking`` only —
        ``apply_chat_template`` was never told about the new mode, so
        the model continued emitting reasoning regardless of the
        user's preference. This method is the live-session update
        side of the fix; the next ``chat()`` call's
        ``_render_prompt`` forwards the new value to the template
        if non-``None``.

        Pass ``None`` to drop the kwarg from future template calls
        entirely (the tokenizer / model family default applies).
        Pass ``True`` / ``False`` to force the corresponding mode.
        """
        self._thinking_mode = mode

    def set_system_prompt(self, text: str | None) -> None:
        """Replace (or clear) the live session's system prompt.

        Mutates ``self._messages``: drops every existing
        ``role="system"`` entry, then prepends a fresh
        ``{"role": "system", "content": text}`` when ``text`` is
        non-empty. ``None`` and the empty string both clear.
        Non-system history (user / assistant turns) is preserved.

        CHAT-CLI-HARDENING-1 (F1): the chat-CLI's ``/system``
        command pre-HARDENING-1 wrote
        ``state.config["system_prompt"]`` only, never touching the
        live session. The user-facing help promised "for the rest
        of the session" but in-flight chat continued against the
        construction-time prompt. This method is the live-session
        update side of the fix; the shell pairs it with the
        existing config-side update so save / load fidelity is
        preserved.

        The prefix cache is **not** explicitly invalidated — the
        rendered prompt's leading tokens change with the new
        system content, so the next ``chat()`` call's prefix-cache
        ``peek`` mismatches the old radix tree at the root and
        the cache's mismatch path takes over naturally. Stale
        blocks remain in the store until the user explicitly
        ``/reset``s the conversation (which the shell pairs with
        a fresh prefix cache via :meth:`set_prefix_cache`).
        """
        non_system = [m for m in self._messages if m["role"] != "system"]
        if text:
            self._messages = [
                {"role": "system", "content": text},
                *non_system,
            ]
        else:
            self._messages = non_system

    def set_prefix_cache(
        self, prefix_cache: _PrefixCacheLike | None
    ) -> None:
        """Replace the active prefix cache. ``None`` disables
        prefix-cache routing (subsequent ``chat`` calls go through
        ``engine.generate`` instead of ``engine.generate_batch``).

        The chat-CLI shell calls this on ``/reset`` to swap in a
        freshly-constructed cache, ensuring no token leakage from
        the previous conversation."""
        self._prefix_cache = prefix_cache

    def chat(
        self,
        user_text: str,
        *,
        sampling_params: SamplingParams | None = None,
        stream_to: Callable[[str], None] | None = None,
    ) -> TurnMetrics:
        """Run one turn and return its :class:`TurnMetrics`.

        Appends the user message, renders the prompt via the
        chat template, drives ``engine.generate``, decodes and
        stores the assistant reply, then returns the per-turn
        metrics. If ``stream_to`` is provided, it is called with
        each incremental decoded delta as tokens arrive.
        ``sampling_params`` may override per-turn sampling; EOS
        stop ids default to the tokenizer's EOS set if the
        caller did not provide them explicitly.
        """
        # CHAT-CLI-RESPONSE-POLICY RP-1: finalise a deferred-strip
        # turn before the new user message lands. Under
        # ``thinking_history=strip`` a turn that ended at
        # ``finish_reason=max_tokens`` left the raw assistant text
        # in place so RP-2 ``/continue`` could resume mid-think;
        # once the user moves on with a fresh turn, the strip
        # applies (``/continue`` is no longer reachable). Skipped
        # when the live policy has flipped to ``keep`` between
        # turns (the user explicitly opted into preserving raw
        # text), or when no defer was registered. The
        # implicit-leading decision uses the snapshot captured at
        # truncation time so a mid-flight ``/config thinking_mode``
        # change cannot retroactively change which strip shape
        # applies to text the model already produced.
        if (
            self._pending_finalize
            and self._thinking_history == "strip"
            and self._messages
            and self._messages[-1]["role"] == "assistant"
        ):
            raw = self._messages[-1]["content"]
            self._messages[-1]["content"] = _strip_thinking_block(
                raw,
                implicit_leading=self._resolve_finalise_implicit_leading(),
            )
        self._pending_finalize = False
        self._pending_finalize_implicit_leading = None
        # RP-2: a new user turn closes the prior turn — ``/continue``
        # is no longer reachable for it, so the continuation
        # snapshot resets unconditionally before the new turn runs.
        # The fresh chat() result below re-seeds the snapshot if it
        # itself ends at ``max_tokens``.
        self._pending_continuation_implicit_leading = None

        self._messages.append({"role": "user", "content": user_text})
        prompt_text, prompt_ids = self._render_prompt()
        params = self._build_sampling_params(sampling_params)

        # Prefix-hit measurement before the turn runs. ``peek`` is
        # side-effect-free, so this number reflects the cache state
        # at the start of the turn — the right signal for the
        # toolbar's "how much of THIS turn's prompt was reused" field.
        prefix_hit_blocks: int | None = None
        prefix_hit_tokens: int | None = None
        if self._prefix_cache is not None:
            hit = self._prefix_cache.peek(prompt_ids)
            prefix_hit_blocks = len(getattr(hit, "block_ids", ()))
            prefix_hit_tokens = int(getattr(hit, "num_hit_tokens", 0))

        self._reset_peak()
        t_start = self._clock()
        (
            out_tokens,
            finish_reason_from_event,
            t_first_token,
        ) = self._run_generation(prompt_text, params, stream_to)
        wall_s = self._clock() - t_start
        peak_mb = self._read_peak_mb()

        # Strip a trailing stop token before decoding for the
        # stored reply text. mlx-lm / vLLM convention yields the
        # stop token before terminating; including it in the
        # message text would leave a literal ``<|im_end|>`` (or
        # equivalent) appended to every reply, polluting both the
        # rendered conversation log and any downstream chat
        # template that re-tokenises the assistant content for
        # the next turn's prompt.
        reply_tokens = out_tokens
        if reply_tokens and reply_tokens[-1] in self._eos_ids:
            reply_tokens = reply_tokens[:-1]
        # Trailing ``�`` chars indicate an incomplete multi-byte
        # UTF-8 sequence that EOS / max_tokens cut short before
        # the closing bytes arrived. Match the streaming
        # ``rstrip("�")`` so the stored reply equals what the
        # user saw on screen, and so the next turn's chat-template
        # tokenisation does not see a ghost replacement char.
        reply_text = self._tokenizer.decode(reply_tokens).rstrip("�")
        # Prefix-cache path surfaces finish_reason directly via the
        # batcher's terminal BatchEvent; single-request path infers
        # it from the token sequence.
        finish_reason = (
            finish_reason_from_event
            if finish_reason_from_event is not None
            else self._classify_finish(out_tokens, params)
        )
        # CHAT-CLI-RESPONSE-POLICY RP-1: gate the eager strip on
        # finish_reason. Natural completion (stop_token / done /
        # eos / empty) → strip immediately. Truncation
        # (max_tokens) → keep raw, set ``_pending_finalize`` so
        # the next ``chat()`` (or RP-2 ``/continue``) finalises
        # at the right moment. ``thinking_history=keep`` short-
        # circuits and stores raw verbatim regardless.
        raw_reply = reply_text
        if (
            self._thinking_history == "strip"
            and finish_reason != "max_tokens"
        ):
            stored_reply = _strip_thinking_block(
                raw_reply,
                implicit_leading=self._should_strip_implicit_leading(),
            )
            self._pending_finalize = False
            self._pending_finalize_implicit_leading = None
        elif (
            self._thinking_history == "strip"
            and finish_reason == "max_tokens"
        ):
            stored_reply = raw_reply
            self._pending_finalize = True
            # Snapshot the implicit-leading decision NOW so a later
            # ``/config thinking_mode`` flip cannot retroactively
            # alter the strip shape when the deferred finalise fires.
            self._pending_finalize_implicit_leading = (
                self._should_strip_implicit_leading()
            )
        else:
            # keep mode — verbatim regardless of finish_reason.
            stored_reply = raw_reply
            self._pending_finalize = False
            self._pending_finalize_implicit_leading = None
        # RP-2: continuation-side snapshot lifecycle — orthogonal
        # to the strip / keep finalise policy above. ANY truncated
        # turn registers the snapshot so ``/continue`` can resume
        # against the original generation prompt's
        # ``<think>\n`` boundary; natural completion leaves it
        # cleared (the top-of-method reset already handled that).
        if finish_reason == "max_tokens":
            self._pending_continuation_implicit_leading = (
                self._should_strip_implicit_leading()
            )
        self._messages.append(
            {"role": "assistant", "content": stored_reply}
        )

        # Engine.generate populates ttft / decode_tok_s / resident_mb
        # via _drive_one's metrics.set_metric calls. Engine.generate_batch
        # does not (the prefix-cache path currently goes through the
        # batched scheduler which does not own a per-request metrics
        # record). To keep TurnMetrics populated regardless of which
        # engine path drove the turn, ChatSession measures TTFT /
        # decode tok-s itself from the wall clock, and reads the KV
        # budget directly from the engine's kv_manager. The engine's
        # metrics snapshot remains the preferred source when present
        # (single-request path keeps the engine-level precision).
        snapshot = self._engine.metrics.snapshot()
        ttft_ms_computed: float | None = None
        decode_tok_s_computed: float | None = None
        if t_first_token is not None:
            ttft_ms_computed = (t_first_token - t_start) * 1000.0
            decode_elapsed = wall_s - (t_first_token - t_start)
            n_decoded = max(0, len(out_tokens) - 1)
            if decode_elapsed > 0 and n_decoded > 0:
                decode_tok_s_computed = n_decoded / decode_elapsed

        ttft_ms = (
            snapshot.ttft_ms
            if snapshot.ttft_ms is not None
            else ttft_ms_computed
        )
        decode_tok_s = (
            snapshot.decode_tok_s
            if snapshot.decode_tok_s is not None
            else decode_tok_s_computed
        )

        # KV budget fields: prefer engine snapshot, fall back to
        # reading the manager directly. Both code paths are cheap.
        resident_mb = snapshot.resident_mb
        logical_kv_bytes = snapshot.logical_kv_bytes
        if resident_mb is None or logical_kv_bytes is None:
            try:
                budget = self._engine.kv_manager.budget()
                if resident_mb is None:
                    resident_mb = budget.resident_bytes / 1e6
                if logical_kv_bytes is None:
                    logical_kv_bytes = int(budget.logical_bytes)
            except Exception:
                # Defensive: kv_manager.budget() failing should not
                # mask the rest of the turn metrics.
                pass

        # Prefix-store residency. ``RadixPrefixCache.stats()`` is the
        # public surface (HARDENING-3 / F6); it returns a frozen
        # ``PrefixCacheStats`` whose ``resident_bytes`` /
        # ``logical_bytes`` are populated via structural capability
        # checks against the underlying store. Backends that do not
        # implement the metric surface (PagedPrefixBlockStore) report
        # ``None``. A defensive ``hasattr`` guard on ``stats`` itself
        # tolerates the legacy / fake-cache shape from older tests
        # that predate this method.
        prefix_store_resident: int | None = None
        prefix_store_logical: int | None = None
        if self._prefix_cache is not None:
            stats_fn = getattr(self._prefix_cache, "stats", None)
            if callable(stats_fn):
                try:
                    pc_stats = stats_fn()
                except Exception:
                    pc_stats = None
                if pc_stats is not None:
                    prefix_store_resident = pc_stats.resident_bytes
                    prefix_store_logical = pc_stats.logical_bytes

        return TurnMetrics(
            reply=stored_reply,
            raw_reply=raw_reply,
            prompt_tokens=len(prompt_ids),
            output_tokens=len(out_tokens),
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            prefill_tok_s=snapshot.prefill_tok_s,
            decode_tok_s=decode_tok_s,
            resident_mb=resident_mb,
            peak_memory_mb=peak_mb,
            logical_kv_bytes=logical_kv_bytes,
            wall_s=wall_s,
            prefix_hit_blocks=prefix_hit_blocks,
            prefix_hit_tokens=prefix_hit_tokens,
            prefix_store_resident_bytes=prefix_store_resident,
            prefix_store_logical_bytes=prefix_store_logical,
        )

    def continue_last(
        self,
        *,
        sampling_params: SamplingParams | None = None,
        stream_to: Callable[[str], None] | None = None,
    ) -> TurnMetrics:
        """Resume the trailing assistant turn by appending generated
        tokens to the existing message — no new
        ``(user, assistant)`` pair is created.

        CHAT-CLI-RESPONSE-POLICY RP-2 (G2). Pairs with RP-1's
        deferred-finalise contract: a turn that ended at
        ``finish_reason="max_tokens"`` under
        ``thinking_history="strip"`` left the raw text on
        ``messages[-1]`` so this method can resume mid-``<think>``
        without losing the prefix.

        Renders the prompt with
        ``apply_chat_template(messages, continue_final_message=True)``
        — the tokeniser drops the trailing close tag from the
        assistant slot so the model resumes at the exact text
        boundary it left. Falls back to a Qwen-shaped manual block
        list when the tokeniser raises (older transformers, or a
        template that does not understand ``continue_final_message``).

        Finalisation policy mirrors :meth:`chat`:

        - ``thinking_history="strip"`` + natural completion
          (``finish_reason != "max_tokens"``) → strip the joined
          raw text using the truncation-time implicit-leading
          snapshot when present, falling back to the live decision.
          Pending state clears.
        - ``thinking_history="strip"`` + chained truncation
          (``max_tokens`` again) → keep the joined raw, preserve
          the snapshot.
        - ``thinking_history="keep"`` → append verbatim.

        Empty-continuation (``out_tokens == []``) follows the same
        finalisation branch as natural completion: under strip the
        existing prefix is finalised through the snapshot, so an
        immediate-EOS resume on a turn with a closed ``</think>``
        still strips the leading reasoning.

        Strictness: raises ``RuntimeError`` when ``messages[-1]`` is
        not an ``assistant`` turn. The "was the last turn actually
        truncated" check belongs to the caller (the chat-CLI shell
        guards on ``state.last_finish_reason``); a future code path
        bypassing the shell can still resume any open assistant
        message safely.

        RP-2 limitation: the streaming display side does not seed
        ``ThinkingParser.start_in_thinking`` from the prior assistant
        prefix here — the chat-CLI shell decides at call site
        whether to re-enter the thinking state by counting
        ``<think>`` / ``</think>`` against the existing message
        content.
        """
        if not self._messages or self._messages[-1]["role"] != "assistant":
            tail_role = (
                self._messages[-1]["role"] if self._messages else "(empty)"
            )
            raise RuntimeError(
                f"continue_last requires the last message to be an "
                f"assistant turn (got {tail_role!r})"
            )

        raw_prefix = self._messages[-1]["content"]
        prompt_text, prompt_ids = self._render_continuation_prompt()
        params = self._build_sampling_params(sampling_params)

        prefix_hit_blocks: int | None = None
        prefix_hit_tokens: int | None = None
        if self._prefix_cache is not None:
            hit = self._prefix_cache.peek(prompt_ids)
            prefix_hit_blocks = len(getattr(hit, "block_ids", ()))
            prefix_hit_tokens = int(getattr(hit, "num_hit_tokens", 0))

        self._reset_peak()
        t_start = self._clock()
        (
            out_tokens,
            finish_reason_from_event,
            t_first_token,
        ) = self._run_generation(prompt_text, params, stream_to)
        wall_s = self._clock() - t_start
        peak_mb = self._read_peak_mb()

        reply_tokens = out_tokens
        if reply_tokens and reply_tokens[-1] in self._eos_ids:
            reply_tokens = reply_tokens[:-1]
        continuation_text = self._tokenizer.decode(reply_tokens).rstrip("�")
        finish_reason = (
            finish_reason_from_event
            if finish_reason_from_event is not None
            else self._classify_finish(out_tokens, params)
        )

        raw_full = raw_prefix + continuation_text

        if (
            self._thinking_history == "strip"
            and finish_reason != "max_tokens"
        ):
            stored_reply = _strip_thinking_block(
                raw_full,
                implicit_leading=self._resolve_finalise_implicit_leading(),
            )
            self._pending_finalize = False
            self._pending_finalize_implicit_leading = None
        elif (
            self._thinking_history == "strip"
            and finish_reason == "max_tokens"
        ):
            stored_reply = raw_full
            self._pending_finalize = True
            # Carry the original truncation-time snapshot forward
            # so a chained continuation cannot retroactively change
            # the strip shape; only seed a fresh snapshot when no
            # chain was registered. The helper consults the
            # continuation-side snapshot before falling through to
            # a live decision, so a keep→strip mid-flight switch
            # still seeds the right value.
            if self._pending_finalize_implicit_leading is None:
                self._pending_finalize_implicit_leading = (
                    self._resolve_finalise_implicit_leading()
                )
        else:
            stored_reply = raw_full
            if finish_reason != "max_tokens":
                self._pending_finalize = False
                self._pending_finalize_implicit_leading = None

        # RP-2: continuation-side snapshot lifecycle in continue_last
        # — preserve across chained max_tokens, clear on natural
        # completion. Unlike the strip-finalise field above, this
        # one tracks the prompt-level truncation fact and is
        # independent of ``thinking_history``.
        if finish_reason == "max_tokens":
            if self._pending_continuation_implicit_leading is None:
                self._pending_continuation_implicit_leading = (
                    self._should_strip_implicit_leading()
                )
        else:
            self._pending_continuation_implicit_leading = None

        self._messages[-1]["content"] = stored_reply

        snapshot = self._engine.metrics.snapshot()
        ttft_ms_computed: float | None = None
        decode_tok_s_computed: float | None = None
        if t_first_token is not None:
            ttft_ms_computed = (t_first_token - t_start) * 1000.0
            decode_elapsed = wall_s - (t_first_token - t_start)
            n_decoded = max(0, len(out_tokens) - 1)
            if decode_elapsed > 0 and n_decoded > 0:
                decode_tok_s_computed = n_decoded / decode_elapsed

        ttft_ms = (
            snapshot.ttft_ms
            if snapshot.ttft_ms is not None
            else ttft_ms_computed
        )
        decode_tok_s = (
            snapshot.decode_tok_s
            if snapshot.decode_tok_s is not None
            else decode_tok_s_computed
        )

        resident_mb = snapshot.resident_mb
        logical_kv_bytes = snapshot.logical_kv_bytes
        if resident_mb is None or logical_kv_bytes is None:
            try:
                budget = self._engine.kv_manager.budget()
                if resident_mb is None:
                    resident_mb = budget.resident_bytes / 1e6
                if logical_kv_bytes is None:
                    logical_kv_bytes = int(budget.logical_bytes)
            except Exception:
                pass

        prefix_store_resident: int | None = None
        prefix_store_logical: int | None = None
        if self._prefix_cache is not None:
            stats_fn = getattr(self._prefix_cache, "stats", None)
            if callable(stats_fn):
                try:
                    pc_stats = stats_fn()
                except Exception:
                    pc_stats = None
                if pc_stats is not None:
                    prefix_store_resident = pc_stats.resident_bytes
                    prefix_store_logical = pc_stats.logical_bytes

        return TurnMetrics(
            reply=stored_reply,
            raw_reply=raw_full,
            prompt_tokens=len(prompt_ids),
            output_tokens=len(out_tokens),
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            prefill_tok_s=snapshot.prefill_tok_s,
            decode_tok_s=decode_tok_s,
            resident_mb=resident_mb,
            peak_memory_mb=peak_mb,
            logical_kv_bytes=logical_kv_bytes,
            wall_s=wall_s,
            prefix_hit_blocks=prefix_hit_blocks,
            prefix_hit_tokens=prefix_hit_tokens,
            prefix_store_resident_bytes=prefix_store_resident,
            prefix_store_logical_bytes=prefix_store_logical,
        )

    def _run_generation(
        self,
        prompt_text: str,
        params: SamplingParams,
        stream_to: Callable[[str], None] | None,
    ) -> tuple[list[int], str | None, float | None]:
        """Drive one turn's token stream.

        Routes through ``engine.generate_batch`` when a prefix
        cache is configured (so block-aligned KV from previous
        turns is reused), otherwise through ``engine.generate``
        (the original single-request path).

        Returns ``(out_tokens, finish_reason, t_first_token)``.
        ``finish_reason`` is the terminal ``BatchEvent``'s reason
        when the batched path produced one, ``None`` otherwise.
        ``t_first_token`` is the wall-clock at which the first
        decoded token was emitted (via the streaming callback or
        engine yield), or ``None`` if no token was produced — used
        by :meth:`chat` to compute TTFT independently of the
        engine's per-request metrics record (which the batched
        path does not populate).
        """
        out_tokens: list[int] = []
        printed_prefix = ""
        stop_set = set(self._eos_ids)
        t_first_token: float | None = None

        def _on_token(tok: int) -> None:
            nonlocal printed_prefix, t_first_token
            out_tokens.append(tok)
            if t_first_token is None:
                t_first_token = self._clock()
            if stream_to is None:
                return
            # Streaming output: skip the rendered text of stop
            # tokens. mlx-lm / vLLM convention yields the stop
            # token before terminating; without this guard the
            # decoded ``<|im_end|>`` (or equivalent) ends up on
            # screen as literal text, which has no value to the
            # user.
            if tok in stop_set:
                # Reset printed_prefix to the pre-stop-token text
                # so the assistant message text we feed downstream
                # does not include the stop bytes either. Strip
                # any trailing held-back replacement chars so
                # they are also dropped from the final reply.
                printed_prefix = self._tokenizer.decode(
                    out_tokens[:-1]
                ).rstrip("�")
                return
            current = self._tokenizer.decode(out_tokens)
            # UTF-8 boundary handling: hold back trailing
            # ``�`` (replacement character) chars. The
            # tokenizer's decode emits these whenever the
            # cumulative byte sequence ends mid-multi-byte-char
            # (commonly an emoji whose UTF-8 bytes are split
            # across two or three BPE tokens). The next token
            # carries the remaining bytes and the replacement
            # char vanishes — but if we streamed it as-is, a
            # permanent ``?`` glyph would already be on screen.
            # Holding back the trailing run lets the next decode
            # naturally produce the real character.
            safe = current.rstrip("�")
            if len(safe) <= len(printed_prefix):
                return
            delta = safe[len(printed_prefix):]
            stream_to(delta)
            printed_prefix = safe

        if self._prefix_cache is None:
            for tok in self._engine.generate(prompt_text, params):
                _on_token(tok)
            return out_tokens, None, t_first_token

        finish_reason: str | None = None
        # Drain the entire batched event stream. Breaking on the
        # ``done`` event would abort the generator before the
        # batcher runs its deferred reclaim step — and that
        # reclaim is what calls ``_extract_and_insert_prefix``
        # (silica/scheduler/batcher.py §reclaim_terminated, which
        # registers block-aligned K/V into the prefix cache for
        # subsequent turns to reuse). Aborting early leaves the
        # cache empty and turns prefix_hit=N/M permanently into
        # 0/N regardless of how long the conversation runs.
        # Token + done events are emitted in the same step(); the
        # next step() drains terminal rows. We must let the engine
        # keep iterating so its ``while has_work()`` loop reaches
        # that next step.
        for event in self._engine.generate_batch(
            [prompt_text],
            params,
            prefix_cache=self._prefix_cache,
        ):
            if event.kind == "token" and event.token_id is not None:
                _on_token(event.token_id)
            elif event.kind in ("done", "aborted"):
                # Capture the first terminal event's reason. The
                # generator continues to drain (no break) so the
                # batcher's reclaim runs. For B=1 only one
                # terminal event ever arrives.
                if finish_reason is None:
                    finish_reason = event.finish_reason
        return out_tokens, finish_reason, t_first_token

    # --- internals ---------------------------------------------------

    def _render_prompt(self) -> tuple[str, list[int]]:
        """Return ``(prompt_text, prompt_ids)`` from the chat template.

        Prefers the tokenizer's ``apply_chat_template`` (the HF /
        mlx-lm convention). Falls back to a Qwen-style
        ``<|im_start|>{role}\\n{content}<|im_end|>`` block list
        so the session still works against tokenizers that
        lack a template (e.g. a pure mlx-native adapter that has
        not been given one yet).
        """
        apply_template = getattr(
            self._tokenizer, "apply_chat_template", None
        )
        if callable(apply_template):
            try:
                # CHAT-CLI-HARDENING-2 (F2): forward
                # ``enable_thinking`` only when the session has an
                # explicit mode set. ``None`` (the default) omits the
                # kwarg so tokenizers without thinking-mode support
                # do not receive an unrecognised parameter and
                # families with their own default mode keep it.
                template_kwargs: dict[str, Any] = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                }
                if self._thinking_mode is not None:
                    template_kwargs["enable_thinking"] = self._thinking_mode
                prompt_ids = list(
                    apply_template(self._messages, **template_kwargs)
                )
                prompt_text = self._tokenizer.decode(prompt_ids)
                return prompt_text, prompt_ids
            except Exception:
                # Tokenizer advertised the method but refused this
                # messages shape (missing template, unsupported
                # role, etc.) — fall back to the manual block list.
                pass
        parts: list[str] = []
        for m in self._messages:
            parts.append(
                f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            )
        parts.append("<|im_start|>assistant\n")
        prompt_text = "".join(parts)
        prompt_ids = list(self._tokenizer.encode(prompt_text))
        return prompt_text, prompt_ids

    def _render_continuation_prompt(self) -> tuple[str, list[int]]:
        """Return ``(prompt_text, prompt_ids)`` for ``continue_last``.

        Prefers ``apply_chat_template(messages, continue_final_message=True)``
        so the tokeniser drops the trailing close tag from the
        assistant slot and the model resumes at the exact text
        boundary it left. Falls back to a Qwen-shaped manual block
        list (``<|im_start|>{role}\\n{content}<|im_end|>``) that
        omits the closing ``<|im_end|>`` from the trailing
        assistant message; this is the right shape for the Qwen3
        family RP-1 / RP-2 explicitly target.

        Implicit-leading restoration: when RP-2's continuation
        snapshot recorded
        ``_pending_continuation_implicit_leading=True``, the
        original generation prompt had a ``<think>\\n`` prefix
        prepended by the chat template (Qwen3 family with
        ``enable_thinking=True``). The model's reply text — stored
        on ``messages[-1].content`` — does NOT include that open
        tag because it came from the template, not from the model.
        Re-rendering the conversation through the template with
        ``continue_final_message=True`` would feed the raw prefix
        without the open tag, breaking the "same prompt at higher
        max_tokens" contract. We restore the synthetic
        ``<think>\\n`` prefix on a temporary copy of the message
        list (the in-memory ``self._messages`` is NOT mutated, so
        RP-1's raw-prefix preservation stays intact). Skipped when
        the prefix already starts with ``<think>`` — explicit opens
        need no synthetic restoration.

        Non-Qwen tokenisers that *also* lack ``continue_final_message``
        produce a prompt that may diverge from the model's
        expected format — a future RP-2 refinement can branch by
        tokeniser family, but the current scope leaves that for
        follow-up.
        """
        messages_for_render = self._messages
        if self._needs_implicit_thinking_restoration():
            tail = self._messages[-1]
            messages_for_render = [
                *self._messages[:-1],
                {
                    "role": tail["role"],
                    "content": "<think>\n" + tail["content"],
                },
            ]
        apply_template = getattr(
            self._tokenizer, "apply_chat_template", None
        )
        if callable(apply_template):
            try:
                template_kwargs: dict[str, Any] = {
                    "tokenize": True,
                    "continue_final_message": True,
                }
                if self._thinking_mode is not None:
                    template_kwargs["enable_thinking"] = self._thinking_mode
                prompt_ids = list(
                    apply_template(messages_for_render, **template_kwargs)
                )
                prompt_text = self._tokenizer.decode(prompt_ids)
                return prompt_text, prompt_ids
            except Exception:
                # Tokeniser refused continue_final_message — fall
                # through to the Qwen-shaped manual block list.
                pass
        parts: list[str] = []
        last_idx = len(messages_for_render) - 1
        for i, m in enumerate(messages_for_render):
            if i == last_idx and m["role"] == "assistant":
                parts.append(
                    f"<|im_start|>{m['role']}\n{m['content']}"
                )
            else:
                parts.append(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                )
        prompt_text = "".join(parts)
        prompt_ids = list(self._tokenizer.encode(prompt_text))
        return prompt_text, prompt_ids

    def _needs_implicit_thinking_restoration(self) -> bool:
        """Whether the continuation prompt should prepend a
        synthetic ``<think>\\n`` to the trailing assistant content.

        True iff RP-2's continuation-side snapshot
        (``_pending_continuation_implicit_leading is True``) AND
        the existing assistant content does not already begin with
        an explicit ``<think>``. Independent of
        ``thinking_history`` — keep mode also needs the
        restoration to give the model a prompt that matches what
        the original turn actually saw at its boundary.
        """
        if self._pending_continuation_implicit_leading is not True:
            return False
        if not self._messages or self._messages[-1]["role"] != "assistant":
            return False
        return not self._messages[-1]["content"].lstrip().startswith(
            "<think>"
        )

    def _build_sampling_params(
        self, override: SamplingParams | None
    ) -> SamplingParams:
        if override is None:
            return SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stop_token_ids=self._eos_ids,
            )
        # Override supplied but missing stop_token_ids — inject
        # the tokenizer's EOS so the model still terminates.
        # SamplingParams is a pydantic model, so model_copy is
        # the right immutable-update primitive.
        if not override.stop_token_ids and self._eos_ids:
            return override.model_copy(
                update={"stop_token_ids": self._eos_ids}
            )
        return override

    def _classify_finish(
        self, out_tokens: list[int], params: SamplingParams
    ) -> str:
        if not out_tokens:
            return "empty"
        if out_tokens[-1] in self._eos_ids:
            return "stop_token"
        if len(out_tokens) >= params.max_tokens:
            return "max_tokens"
        return "done"


def _strip_thinking_block(text: str, *, implicit_leading: bool) -> str:
    """Remove ``<think>...</think>`` content from a decoded reply.

    CHAT-CLI-RESPONSE-POLICY RP-1 (G1). Two shapes are supported,
    composed in this order:

    1. **Implicit-leading**. Qwen3 / Qwen3.5 chat templates with
       ``enable_thinking=True`` append ``<think>\\n`` to the prompt;
       the model's reply therefore starts *inside* a thinking block
       and the first ``</think>`` closes it. There is no opening
       tag in the reply itself. When ``implicit_leading=True``
       this helper drops everything from the start of ``text`` up
       to and including the first ``</think>`` (plus a single
       trailing newline if present, matching template convention).
       If no ``</think>`` is found, the text is returned unchanged
       — the model never closed its thought (e.g.
       ``finish_reason=max_tokens`` mid-think); preserving the
       text avoids guessing where reasoning ended.
    2. **Explicit pairs**. Any ``<think>...</think>`` spans that
       remain in the text are removed. An unclosed explicit
       ``<think>`` (no matching ``</think>``) drops the rest of
       the text — the same conservative choice as the
       implicit-leading branch.

    Pass-through cases:
    - ``implicit_leading=False`` and no explicit tags → text
      unchanged.
    - empty / whitespace-only text → returned unchanged.

    Pure function; no side effects. Used by ``ChatSession.chat``
    when ``thinking_history=strip`` finalises a turn.
    """
    if not text:
        return text

    if implicit_leading:
        close_idx = text.find("</think>")
        if close_idx >= 0:
            text = text[close_idx + len("</think>"):]
            # Template convention: a single ``\n`` follows the
            # closing tag. Strip it so the visible reply does not
            # carry a leading blank line.
            if text.startswith("\n"):
                text = text[1:]

    # Explicit ``<think>...</think>`` spans, zero or more.
    parts: list[str] = []
    pos = 0
    while True:
        open_idx = text.find("<think>", pos)
        if open_idx < 0:
            parts.append(text[pos:])
            break
        parts.append(text[pos:open_idx])
        close_idx = text.find("</think>", open_idx)
        if close_idx < 0:
            # Unclosed explicit tag — drop the remainder.
            break
        pos = close_idx + len("</think>")
        if pos < len(text) and text[pos] == "\n":
            pos += 1
    return "".join(parts)


def _mlx_reset_peak_memory() -> None:
    """Reset MLX peak-memory accounting. No-op if mlx unavailable."""
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:
        pass


def _mlx_peak_memory_mb() -> float | None:
    """Read MLX peak memory in MB, or None if mlx unavailable."""
    try:
        import mlx.core as mx

        return float(mx.get_peak_memory()) / 1e6
    except Exception:
        return None


__all__ = ["ChatSession", "TurnMetrics", "_strip_thinking_block"]
