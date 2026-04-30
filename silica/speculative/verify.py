"""silica.speculative.verify — D-021 step 5 sub-unit (a2) contract slice.

Free-function fallback for the speculative verify forward. The I-1
``ModelAdapter.decode_step_multi(tokens, kv_handle)`` Protocol method runs
one batched forward over ``T`` input tokens and returns logits at every
position — that is the speedup mechanism Track C.1 / C.4 / C.5 ride on
(P-6.0.5 Unit 7 cost curve).

Adapters that have not yet shipped a real ``decode_step_multi`` raise
``NotImplementedError`` from a placeholder body (so the runtime_checkable
``ModelAdapter`` Protocol membership stays satisfied — see
``silica/models/qwen3.py`` / ``qwen3_5.py`` / ``gemma4.py``). This module's
:func:`run_verify_forward` catches that raise and falls back to a
sequential ``decode_step`` loop, stacking per-position logits into the
``(T, V)`` shape the spec engine expects. The fallback is correct but
defeats verify amortisation — production paths land per-adapter
overrides in subsequent slices of (a2).

Fallback contract:

  - Input ``verify_input`` is a 1-D ``mx.array`` of shape ``(T,)``.
  - Output is ``(logits, state_delta)`` matching
    ``adapter.decode_step_multi`` — ``logits`` shape ``(T, V)``,
    ``state_delta`` reflecting the post-final-position adapter state
    (from the last ``decode_step`` call).
  - Each ``decode_step`` call appends one position to the underlying
    ``KVHandle``'s cache; that is the same effect a real
    ``decode_step_multi`` would have, just paid one position at a time.

Empty input is rejected loud (matches the OPENING contract that v0.1
``verify_k`` is at least 1).
"""

from __future__ import annotations

import mlx.core as mx

from silica.kvcache.manager import KVHandle
from silica.models.adapter import ModelAdapter, StateDelta


def run_verify_forward(
    adapter: ModelAdapter,
    verify_input: mx.array,
    kv_handle: KVHandle,
) -> tuple[mx.array, StateDelta]:
    """Run a verify forward over ``verify_input`` of shape ``(T,)``.

    Tries ``adapter.decode_step_multi`` first; on ``NotImplementedError``
    falls back to a sequential ``decode_step`` loop that produces the same
    ``(T, V)`` logits shape (correct but not amortised).
    """
    if verify_input.ndim != 1:
        raise ValueError(
            f"verify_input must be 1-D (T,), got shape "
            f"{tuple(verify_input.shape)}"
        )
    if int(verify_input.size) == 0:
        raise ValueError("verify_input must be non-empty")

    try:
        return adapter.decode_step_multi(verify_input, kv_handle)
    except NotImplementedError:
        return _decode_step_loop_fallback(adapter, verify_input, kv_handle)


def _decode_step_loop_fallback(
    adapter: ModelAdapter,
    verify_input: mx.array,
    kv_handle: KVHandle,
) -> tuple[mx.array, StateDelta]:
    """Sequential ``decode_step`` loop producing ``(T, V)`` stacked logits.

    The last ``StateDelta`` wins — earlier per-step deltas describe
    intermediate state which is folded into the underlying KV by the
    time the loop exits. This matches the contract that
    ``decode_step_multi``'s returned ``StateDelta`` reflects the
    post-final-position state.
    """
    T = int(verify_input.size)
    per_position_logits: list[mx.array] = []
    last_delta: StateDelta = StateDelta()
    for i in range(T):
        # mx.array slicing produces a ``(1,)`` array — matches the shape
        # ``decode_step`` accepts for a single-token forward.
        tok = verify_input[i : i + 1]
        logits, last_delta = adapter.decode_step(tok, kv_handle)
        per_position_logits.append(logits)
    return mx.stack(per_position_logits), last_delta


__all__ = ["run_verify_forward"]
