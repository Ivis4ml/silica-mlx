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

from silica.core.sampling import SamplingParams
from silica.models.adapter import ModelAdapter


class _EngineLike(Protocol):
    """Narrow Engine surface the session actually consumes.

    The real :class:`silica.engine.Engine` trivially satisfies
    this; tests can pass any object that exposes the three
    attributes. Kept at Protocol level so importing this module
    does not pull ``silica.engine`` when only the session
    dataclasses are needed.
    """

    metrics: Any
    kv_manager: Any

    def generate(
        self, prompt: str, params: SamplingParams | None = None
    ) -> Iterator[int]: ...


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

    # --- observation -------------------------------------------------

    @property
    def messages(self) -> list[dict[str, str]]:
        """Return a shallow copy of the current message history."""
        return list(self._messages)

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_ids

    # --- mutation ----------------------------------------------------

    def reset(self) -> None:
        """Drop every message except the original system prompt."""
        self._messages = [
            m for m in self._messages if m["role"] == "system"
        ]

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

        self._reset_peak()
        t_start = self._clock()
        out_tokens: list[int] = []
        printed_prefix = ""
        for tok in self._engine.generate(prompt_text, params):
            out_tokens.append(tok)
            if stream_to is not None:
                current = self._tokenizer.decode(out_tokens)
                delta = current[len(printed_prefix):]
                if delta:
                    stream_to(delta)
                    printed_prefix = current
        wall_s = self._clock() - t_start
        peak_mb = self._read_peak_mb()

        reply_text = self._tokenizer.decode(out_tokens)
        finish_reason = self._classify_finish(out_tokens, params)
        self._messages.append(
            {"role": "assistant", "content": reply_text}
        )

        snapshot = self._engine.metrics.snapshot()
        return TurnMetrics(
            reply=reply_text,
            prompt_tokens=len(prompt_ids),
            output_tokens=len(out_tokens),
            finish_reason=finish_reason,
            ttft_ms=snapshot.ttft_ms,
            prefill_tok_s=snapshot.prefill_tok_s,
            decode_tok_s=snapshot.decode_tok_s,
            resident_mb=snapshot.resident_mb,
            peak_memory_mb=peak_mb,
            logical_kv_bytes=snapshot.logical_kv_bytes,
            wall_s=wall_s,
        )

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
