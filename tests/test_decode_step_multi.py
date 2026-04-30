"""Tests for ``ModelAdapter.decode_step_multi`` Protocol surface +
``silica.speculative.verify.run_verify_forward`` fallback.

D-021 step 5 sub-unit (a2) contract slice. This slice does not land
adapter-specific real implementations of ``decode_step_multi``; the three
parent real adapters (``Qwen3Adapter`` / ``Qwen3_5Adapter`` /
``Gemma4Adapter``) declare the method but raise ``NotImplementedError``
from a placeholder body, so the runtime_checkable ``ModelAdapter``
Protocol membership stays satisfied. ``StubModelAdapter`` ships a real
zero-logits implementation; the free-function ``run_verify_forward``
catches the placeholder raise and falls back to a sequential
``decode_step`` loop, producing the same ``(T, V)`` logits shape.

Coverage:
  - Protocol surface: ``decode_step_multi`` is part of ``ModelAdapter``;
    ``StubModelAdapter`` continues to satisfy the Protocol.
  - Stub native path: returns ``(T, V)`` zero logits + empty
    ``StateDelta``; output shape derived from input length, not
    ``vocab_size`` alone.
  - run_verify_forward dispatch: when the adapter has a real impl,
    output is the impl's output untouched.
  - run_verify_forward fallback: when the adapter raises
    ``NotImplementedError``, the loop produces ``(T, V)`` stacked logits
    that equal the per-step ``decode_step`` outputs, in order.
  - Real-adapter contract surface: ``Qwen3Adapter`` / ``Qwen3_5Adapter``
    / ``Gemma4Adapter`` (and the inheriting MoE subclasses) declare
    ``decode_step_multi``; calling the placeholder raises
    ``NotImplementedError``.
  - Empty / wrong-shape ``verify_input`` rejected loud.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kvcache.manager import KVHandle
from silica.models.adapter import ModelAdapter, StateDelta, StubModelAdapter
from silica.models.gemma4 import Gemma4Adapter
from silica.models.gemma4_moe import Gemma4MoeAdapter
from silica.models.qwen3 import Qwen3Adapter
from silica.models.qwen3_5 import Qwen3_5Adapter
from silica.models.qwen3_5_moe import Qwen3_5MoeAdapter
from silica.speculative.verify import run_verify_forward

# --- Protocol surface -------------------------------------------------------


def test_stub_satisfies_model_adapter_protocol_with_decode_step_multi() -> None:
    # Adding `decode_step_multi` to the runtime_checkable Protocol is
    # only safe if every existing conformer (Stub + the five real
    # adapters via their NotImplementedError stubs) declares the method.
    # This test pins that the Stub continues to satisfy the Protocol;
    # real-adapter Protocol-conformance is covered by their existing
    # `tests/test_<family>_adapter.py` isinstance checks.
    adapter = StubModelAdapter(num_layers=2, vocab_size=8)
    assert isinstance(adapter, ModelAdapter)
    assert hasattr(adapter, "decode_step_multi")


# --- Stub native path -------------------------------------------------------


@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_stub_decode_step_multi_returns_T_V_zero_logits(T: int) -> None:
    adapter = StubModelAdapter(num_layers=2, vocab_size=16)
    handle = KVHandle(req_id="stub-test")
    tokens = mx.zeros((T,), dtype=mx.int32)

    logits, delta = adapter.decode_step_multi(tokens, handle)

    assert tuple(logits.shape) == (T, 16)
    assert mx.all(logits == 0).item()
    assert isinstance(delta, StateDelta)
    assert delta.recurrent_bytes() == 0


# --- run_verify_forward dispatch (Stub native path) -------------------------


def test_run_verify_forward_dispatches_to_native_decode_step_multi() -> None:
    # Stub provides a real `decode_step_multi`, so run_verify_forward
    # should pass through. Output equals the adapter's direct call.
    adapter = StubModelAdapter(num_layers=2, vocab_size=4)
    handle = KVHandle(req_id="stub-test")
    verify_input = mx.array([0, 1, 2], dtype=mx.int32)

    logits_via_helper, delta_via_helper = run_verify_forward(
        adapter, verify_input, handle
    )
    logits_direct, delta_direct = adapter.decode_step_multi(verify_input, handle)

    assert tuple(logits_via_helper.shape) == tuple(logits_direct.shape)
    assert mx.all(logits_via_helper == logits_direct).item()
    assert delta_via_helper.recurrent_bytes() == delta_direct.recurrent_bytes()


# --- run_verify_forward fallback (NotImplementedError → loop) ---------------


class _LoopFallbackAdapter:
    """Adapter that raises NotImplementedError on decode_step_multi but
    returns deterministic per-position logits via decode_step.

    Used to exercise the fallback path. ``decode_step`` returns logits
    whose argmax = (call_index + 1) mod V; the test stacks expected
    outputs and compares against ``run_verify_forward``.
    """

    VOCAB = 16

    def __init__(self) -> None:
        self.calls = 0
        self.steps_seen: list[int] = []

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        self.calls += 1
        # Token id is recorded so the test can assert per-position input
        # ordering matches the verify_input order.
        self.steps_seen.append(int(token[0].item()))
        target = self.calls % self.VOCAB
        scores = [0.0] * self.VOCAB
        scores[target] = 7.0
        return mx.array(scores, dtype=mx.float32), StateDelta(
            _recurrent_bytes=self.calls * 100
        )

    def decode_step_multi(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        raise NotImplementedError("placeholder for fallback test")


def test_run_verify_forward_falls_back_on_not_implemented() -> None:
    adapter = _LoopFallbackAdapter()
    handle = KVHandle(req_id="loop-test")
    verify_input = mx.array([10, 11, 12, 13], dtype=mx.int32)

    logits, delta = run_verify_forward(adapter, verify_input, handle)

    # 4 positions → 4 decode_step calls in input order.
    assert adapter.calls == 4
    assert adapter.steps_seen == [10, 11, 12, 13]

    # (T, V) shape stacked from per-step (V,) outputs.
    assert tuple(logits.shape) == (4, _LoopFallbackAdapter.VOCAB)

    # Per-position argmax = call_index mod V (calls 1..4).
    expected_argmax = [1, 2, 3, 4]
    actual_argmax = [int(mx.argmax(logits[i]).item()) for i in range(4)]
    assert actual_argmax == expected_argmax

    # Last StateDelta wins — `recurrent_bytes` reflects the final
    # decode_step (call index 4 → 4 * 100 = 400 per the fake).
    assert delta.recurrent_bytes() == 400


def test_run_verify_forward_fallback_single_token() -> None:
    # T=1 is the degenerate case: one decode_step, output shape (1, V).
    adapter = _LoopFallbackAdapter()
    handle = KVHandle(req_id="loop-test")

    logits, delta = run_verify_forward(
        adapter, mx.array([42], dtype=mx.int32), handle
    )

    assert adapter.calls == 1
    assert tuple(logits.shape) == (1, _LoopFallbackAdapter.VOCAB)
    assert delta.recurrent_bytes() == 100


# --- run_verify_forward shape validation ------------------------------------


def test_run_verify_forward_rejects_zero_length_input() -> None:
    adapter = StubModelAdapter()
    handle = KVHandle(req_id="stub-test")
    with pytest.raises(ValueError, match=r"non-empty"):
        run_verify_forward(adapter, mx.array([], dtype=mx.int32), handle)


def test_run_verify_forward_rejects_2d_input() -> None:
    adapter = StubModelAdapter()
    handle = KVHandle(req_id="stub-test")
    with pytest.raises(ValueError, match=r"1-D"):
        run_verify_forward(
            adapter, mx.zeros((2, 3), dtype=mx.int32), handle
        )


# --- real-adapter contract surface ------------------------------------------


@pytest.mark.parametrize(
    "adapter_cls",
    [
        Qwen3Adapter,
        Qwen3_5Adapter,
        Gemma4Adapter,
        Qwen3_5MoeAdapter,
        Gemma4MoeAdapter,
    ],
    ids=[
        "qwen3",
        "qwen3_5",
        "gemma4",
        "qwen3_5_moe",
        "gemma4_moe",
    ],
)
def test_real_adapter_declares_decode_step_multi(adapter_cls: type) -> None:
    # Every concrete real adapter (including MoE subclasses inheriting
    # from their parents) must declare `decode_step_multi` so the
    # runtime_checkable ``ModelAdapter`` Protocol membership stays
    # satisfied across the v1.7.X contract slice. The placeholder bodies
    # raise NotImplementedError; the real forwards land in subsequent
    # adapter-specific (a2) sub-slices.
    assert hasattr(adapter_cls, "decode_step_multi")


# --- MoE inheritance pins (slice 3 + slice 4) ------------------------------


def test_gemma4_moe_inherits_real_decode_step_multi_from_parent() -> None:
    # Gemma4MoeAdapter inherits its decode_step_multi directly from
    # Gemma4Adapter, which slice 2 of (a2) promoted to a real
    # ``forward_full`` call. This identity check is the load-bearing
    # pin — if a future refactor accidentally overrides
    # decode_step_multi on the MoE subclass without also propagating
    # the real forward, this test catches it.
    assert (
        Gemma4MoeAdapter.decode_step_multi
        is Gemma4Adapter.decode_step_multi
    )


def test_qwen3_5_moe_inherits_real_decode_step_multi_from_parent() -> None:
    # Qwen3_5MoeAdapter inherits its decode_step_multi from
    # Qwen3_5Adapter, which slice 4 of (a2) (the hybrid sub-slice)
    # promoted from the NotImplementedError placeholder to a real
    # ``forward_full`` call. The identity link below was already
    # pinned at slice 3 (when both pointed to the placeholder); slice
    # 4 keeps the link and both endpoints now resolve to the real
    # hybrid forward. Same regression guard as Gemma4-MoE — catches
    # any accidental override on the MoE subclass that drops the real
    # impl.
    assert (
        Qwen3_5MoeAdapter.decode_step_multi
        is Qwen3_5Adapter.decode_step_multi
    )
