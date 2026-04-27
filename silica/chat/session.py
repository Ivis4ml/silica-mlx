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

    # --- observation -------------------------------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return a shallow copy of the current message history."""
        return list(self._messages)

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_ids

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
        """
        self._messages = [
            m for m in self._messages if m["role"] == "system"
        ]

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
        self._messages.append(
            {"role": "assistant", "content": reply_text}
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

        # Prefix-store residency. The cache itself does not expose a
        # public byte-count getter, but the SyntheticPrefixBlockStore
        # backing it does (via the structural ``resident_bytes()``
        # method documented as P-5-A.2's ``MemoryBudgeter`` hook). A
        # ``hasattr`` guard treats backends that do not implement it
        # (PagedPrefixBlockStore) as "unknown" rather than zero.
        prefix_store_resident: int | None = None
        prefix_store_logical: int | None = None
        if self._prefix_cache is not None:
            store = getattr(self._prefix_cache, "_store", None)
            resident_fn = getattr(store, "resident_bytes", None)
            if callable(resident_fn):
                try:
                    prefix_store_resident = int(resident_fn())
                except Exception:
                    prefix_store_resident = None
            # Logical = fp16 equivalent. The pass-through path
            # (no codec) makes resident == logical. Under a codec,
            # the store exposes per-codec ``logical_bytes(num_tokens)``
            # but a clean public API for the cumulative figure
            # is not present today; for the chat-CLI's purpose we
            # approximate logical by walking detached blocks.
            num_blocks = 0
            try:
                num_blocks = len(getattr(store, "_detached", {}))
            except Exception:
                pass
            if (
                prefix_store_resident is not None
                and num_blocks > 0
            ):
                # Try the codec path first (each codec exposes
                # ``logical_bytes(num_tokens)``); fall back to
                # resident == logical for the no-codec path.
                k_codec = getattr(store, "_k_codec", None)
                v_codec = getattr(store, "_v_codec", None)
                num_layers = getattr(store, "_num_layers", None)
                bs = getattr(self._prefix_cache, "block_size", 0)
                if (
                    k_codec is not None
                    and v_codec is not None
                    and num_layers is not None
                    and bs > 0
                ):
                    tokens_per_block = bs
                    try:
                        per_block_logical = num_layers * (
                            k_codec.logical_bytes(tokens_per_block)
                            + v_codec.logical_bytes(tokens_per_block)
                        )
                        prefix_store_logical = (
                            num_blocks * per_block_logical
                        )
                    except Exception:
                        prefix_store_logical = (
                            prefix_store_resident
                        )
                else:
                    prefix_store_logical = prefix_store_resident
            elif prefix_store_resident is not None:
                prefix_store_logical = prefix_store_resident

        return TurnMetrics(
            reply=reply_text,
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
                prompt_ids = list(
                    apply_template(
                        self._messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
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


__all__ = ["ChatSession", "TurnMetrics"]
