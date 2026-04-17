"""silica.engine — top-level generation orchestrator (P-1).

Engine glues the pieces together:
  - ``ModelAdapter`` (I-1) — tokenizer + prefill + decode_step.
  - ``KVManager`` (I-2) — reserve / free per-request KV (``SimpleKVCache`` in
    P-1, ``PagedKVCache`` in P-2; Engine is agnostic).
  - ``Sampler`` (P-0) — processor chain → sampled token from logits.

P-1 scope: one request at a time. Multi-request continuous batching is P-2
(``ContinuousBatcher`` + ``MemoryBudgeter``). ``Engine.generate`` returns an
iterator of token ids so callers (CLI, tests, bench harness) can consume
streaming output without any decode coupling.

Stop policy for P-1:
  - ``max_tokens`` is a hard upper bound on yielded tokens.
  - ``stop_token_ids`` tokens are yielded-then-stopped (vLLM / mlx-lm
    convention — the caller needs to see the stop token to know *why* we
    stopped).
  - String-sequence ``stop`` patterns are a P-2 concern (they require
    incremental decoding of generated tokens into a text buffer).
  - EOS handling: the caller is responsible for populating ``stop_token_ids``
    with the tokenizer's EOS id (or omitting it if ``ignore_eos``).
"""

from __future__ import annotations

from collections.abc import Iterator

import mlx.core as mx

from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.kvcache.manager import KVHandle, KVManager
from silica.models.adapter import ModelAdapter


class Engine:
    """Single-request generation orchestrator (P-1).

    Construct once per (adapter, kv_manager) pair; call ``generate`` any
    number of times. ``req_id`` is auto-assigned so callers don't manage it.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        kv_manager: KVManager,
        sampler: Sampler | None = None,
    ) -> None:
        self._adapter = adapter
        self._kv_manager = kv_manager
        self._sampler = sampler or Sampler()
        self._req_counter = 0

    def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
    ) -> Iterator[int]:
        """Yield generated token ids one at a time.

        Empty prompts yield nothing (mlx-lm's generate_step also requires a
        non-empty prompt / input_embeddings; we mirror that here).
        """
        effective = params or SamplingParams()
        tokenizer = self._adapter.tokenizer()
        prompt_ids: list[int] = list(tokenizer.encode(prompt))
        if not prompt_ids:
            return

        req_id = self._new_req_id()
        handle = KVHandle(req_id=req_id)
        self._kv_manager.reserve_for_prefill(req_id, prompt_ids)
        try:
            yield from self._drive(prompt_ids, handle, effective)
        finally:
            self._kv_manager.free(req_id)

    # --- private ---

    def _drive(
        self,
        prompt_ids: list[int],
        handle: KVHandle,
        params: SamplingParams,
    ) -> Iterator[int]:
        prompt_arr = mx.array(prompt_ids, dtype=mx.int32)
        logits, _ = self._adapter.prefill(prompt_arr, handle)
        history: list[int] = list(prompt_ids)

        token_scalar = self._sampler.sample(
            logits, mx.array(history, dtype=mx.int32), params
        )
        tok_int = int(token_scalar.item())
        yield tok_int
        history.append(tok_int)
        if tok_int in params.stop_token_ids:
            return

        n = 1
        while n < params.max_tokens:
            step_in = mx.array([tok_int], dtype=mx.int32)
            logits, _ = self._adapter.decode_step(step_in, handle)
            token_scalar = self._sampler.sample(
                logits, mx.array(history, dtype=mx.int32), params
            )
            tok_int = int(token_scalar.item())
            yield tok_int
            n += 1
            history.append(tok_int)
            if tok_int in params.stop_token_ids:
                return

    def _new_req_id(self) -> str:
        rid = f"req-{self._req_counter}"
        self._req_counter += 1
        return rid


__all__ = ["Engine"]
