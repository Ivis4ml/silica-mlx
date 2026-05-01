"""Real-model tests for ``Qwen3_5MoeAdapter.decode_step_multi_with_capture``.

D-021 step 6 sub-unit (αβ.2). The MoE adapter inherits the entire
capture path from ``Qwen3_5Adapter`` because:

- mlx-lm's ``qwen3_5_moe.Model`` extends ``qwen3_5.Model`` directly;
  the outer ``__call__`` and inner ``Qwen3_5TextModel.__call__`` chain
  are identical between dense and MoE — only the per-layer MLP
  internals (``SparseMoeBlock``) differ. ``run_qwen3_5_forward_with_capture``
  in ``silica.models.hidden_capture`` walks ``inner.layers`` agnostic
  of MLP type, so the capture surface lifts unchanged.
- ``Qwen3_5MoeAdapter(Qwen3_5Adapter)`` inherits ``decode_step_multi``
  and ``decode_step_multi_with_capture`` verbatim.

These tests pin the inheritance: the same three contracts the dense
test pins on Qwen3.5-0.8B, re-pinned on the MoE checkpoint to confirm
the per-layer MLP type does not perturb the capture path's
correctness.

Dual-gated like ``test_decode_step_multi_real`` Gemma4-MoE: the MoE
checkpoint is ~20 GB. Required to run:

  1. ``mlx-community/Qwen3.5-35B-A3B-4bit`` cached locally; populate
     via ``scripts/probe_qwen3_5_moe_load.py`` or
     ``huggingface-cli download``.
  2. ``SILICA_REAL_QWEN3_5_MOE=1`` env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from silica.kvcache.manager import KVHandle
from silica.models.hidden_capture import HiddenCaptureAdapter
from silica.models.qwen3_5_moe import Qwen3_5MoeAdapter

REPO = "mlx-community/Qwen3.5-35B-A3B-4bit"

_QWEN3_5_MOE_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--mlx-community--Qwen3.5-35B-A3B-4bit"
)
_QWEN3_5_MOE_ENV = os.environ.get("SILICA_REAL_QWEN3_5_MOE") == "1"
_QWEN3_5_MOE_SKIP_REASON = (
    "Qwen3.5-35B-A3B-4bit MoE capture test is dual-gated. Required: "
    f"(1) checkpoint cached at {_QWEN3_5_MOE_CACHE} (run "
    "scripts/probe_qwen3_5_moe_load.py to populate, ~20 GB); "
    "(2) env var SILICA_REAL_QWEN3_5_MOE=1 to opt in."
)
_QWEN3_5_MOE_SKIP = (
    not _QWEN3_5_MOE_CACHE.exists()
    or not _QWEN3_5_MOE_ENV
    or bool(os.environ.get("SILICA_SKIP_MODEL_TESTS"))
)

T = 4
TOKEN_IDS = [101, 202, 303, 404][:T]


def _greedy_argmax_equal(a: mx.array, b: mx.array) -> None:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    for i in range(a.shape[0]):
        ai = int(mx.argmax(a[i]).item())
        bi = int(mx.argmax(b[i]).item())
        assert ai == bi, (
            f"position {i}: argmax diverges (a={ai}, b={bi})"
        )


def _per_element_close(
    a: mx.array, b: mx.array, *, rtol: float = 1e-4, atol: float = 1e-3
) -> None:
    diff = mx.max(mx.abs(a - b))
    bound = rtol * float(mx.max(mx.abs(b)).item()) + atol
    assert float(diff.item()) <= bound, (
        f"|a - b|_max={float(diff.item()):.4e} exceeds rtol*|b| + atol "
        f"= {bound:.4e}"
    )


@pytest.mark.skipif(_QWEN3_5_MOE_SKIP, reason=_QWEN3_5_MOE_SKIP_REASON)
def test_qwen3_5_moe_adapter_implements_hidden_capture_protocol() -> None:
    adapter, _ = Qwen3_5MoeAdapter.from_hf_repo(REPO)
    assert isinstance(adapter, HiddenCaptureAdapter), (
        "Qwen3_5MoeAdapter must implement HiddenCaptureAdapter via "
        "inheritance from Qwen3_5Adapter ((αβ.2) inheritance pin)."
    )


@pytest.mark.skipif(_QWEN3_5_MOE_SKIP, reason=_QWEN3_5_MOE_SKIP_REASON)
def test_moe_capture_disabled_matches_decode_step_multi() -> None:
    """Empty ``capture_layer_ids`` reproduces ``decode_step_multi``'s
    logits position-for-position on the MoE checkpoint. Same contract
    as the dense (αβ.1) test."""
    adapter_a, kv_a = Qwen3_5MoeAdapter.from_hf_repo(REPO)
    adapter_b, kv_b = Qwen3_5MoeAdapter.from_hf_repo(REPO)

    req_id = "moe-capture-eq-test"
    handle_a = KVHandle(req_id=req_id)
    handle_b = KVHandle(req_id=req_id)

    kv_a.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_b.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    tokens = mx.array(TOKEN_IDS, dtype=mx.int32)

    logits_baseline, _ = adapter_a.decode_step_multi(tokens, handle_a)
    logits_capture, captured, _ = adapter_b.decode_step_multi_with_capture(
        tokens, handle_b, frozenset()
    )

    assert captured == {}
    assert logits_capture.shape == logits_baseline.shape
    _greedy_argmax_equal(logits_capture, logits_baseline)
    _per_element_close(logits_capture, logits_baseline)


@pytest.mark.skipif(_QWEN3_5_MOE_SKIP, reason=_QWEN3_5_MOE_SKIP_REASON)
def test_moe_capture_enabled_returns_hidden_slices_with_correct_shape() -> None:
    """Non-empty capture on MoE: returned dict has the requested keys;
    each value has shape ``(1, T, hidden_dim)``. Logits invariant
    against capture-disabled."""
    adapter_a, kv_a = Qwen3_5MoeAdapter.from_hf_repo(REPO)
    adapter_b, kv_b = Qwen3_5MoeAdapter.from_hf_repo(REPO)

    req_id = "moe-capture-shape-test"
    handle_a = KVHandle(req_id=req_id)
    handle_b = KVHandle(req_id=req_id)

    kv_a.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_b.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    tokens = mx.array(TOKEN_IDS, dtype=mx.int32)

    num_layers = adapter_a.config.num_layers
    requested = frozenset({0, num_layers // 2, num_layers})

    logits_no_cap, _ = adapter_a.decode_step_multi(tokens, handle_a)
    logits_cap, captured, _ = adapter_b.decode_step_multi_with_capture(
        tokens, handle_b, requested
    )

    _greedy_argmax_equal(logits_cap, logits_no_cap)
    _per_element_close(logits_cap, logits_no_cap)

    assert set(captured.keys()) == set(requested)

    hidden_dim = adapter_a.config.hidden_size
    for layer_id in sorted(requested):
        h = captured[layer_id]
        assert h.shape == (1, T, hidden_dim), (
            f"layer {layer_id}: shape {h.shape} != (1, {T}, {hidden_dim})"
        )


# --- (αβ.3) prefill capture seed + cached-prefix regression on MoE ---------


@pytest.mark.skipif(_QWEN3_5_MOE_SKIP, reason=_QWEN3_5_MOE_SKIP_REASON)
def test_moe_prefill_with_capture_returns_full_prompt_hiddens() -> None:
    """MoE prefill capture: returned dict has the requested keys; each
    value has shape ``(1, prompt_len, hidden_dim)``. (αβ.3) inheritance
    pin: the dense prefill_with_capture lifts to the MoE adapter via
    Qwen3_5MoeAdapter(Qwen3_5Adapter)."""
    adapter, kv = Qwen3_5MoeAdapter.from_hf_repo(REPO)

    req_id = "moe-prefill-capture-shape-test"
    handle = KVHandle(req_id=req_id)
    kv.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    prompt_tokens = mx.array([101, 202, 303, 404, 505], dtype=mx.int32)
    prompt_len = int(prompt_tokens.size)

    num_layers = adapter.config.num_layers
    requested = frozenset({0, num_layers // 2, num_layers})

    _, captured, _ = adapter.prefill_with_capture(
        prompt_tokens, handle, requested
    )

    assert set(captured.keys()) == set(requested)
    hidden_dim = adapter.config.hidden_size
    for layer_id in sorted(requested):
        h = captured[layer_id]
        assert h.shape == (1, prompt_len, hidden_dim), (
            f"layer {layer_id}: shape {h.shape} != "
            f"(1, {prompt_len}, {hidden_dim})"
        )


@pytest.mark.skipif(_QWEN3_5_MOE_SKIP, reason=_QWEN3_5_MOE_SKIP_REASON)
def test_moe_capture_after_prefill_matches_decode_step_multi() -> None:
    """MoE cached-prefix regression: same as the dense (αβ.3) test on
    Qwen3.5-0.8B but on the 35B-A3B-4bit MoE checkpoint. Pins that
    capture-after-prompt-prefill produces bit-equivalent verify logits
    against the non-empty cache state, not just the empty-cache
    regime."""
    adapter_a, kv_a = Qwen3_5MoeAdapter.from_hf_repo(REPO)
    adapter_b, kv_b = Qwen3_5MoeAdapter.from_hf_repo(REPO)

    req_id = "moe-capture-after-prefill-test"
    handle_a = KVHandle(req_id=req_id)
    handle_b = KVHandle(req_id=req_id)
    kv_a.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_b.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    prompt_tokens = mx.array([101, 202, 303, 404, 505], dtype=mx.int32)
    adapter_a.prefill(prompt_tokens, handle_a)
    adapter_b.prefill(prompt_tokens, handle_b)

    verify_tokens = mx.array(TOKEN_IDS, dtype=mx.int32)
    logits_baseline, _ = adapter_a.decode_step_multi(verify_tokens, handle_a)

    num_layers = adapter_b.config.num_layers
    requested = frozenset({0, num_layers // 2, num_layers})
    logits_capture, captured, _ = adapter_b.decode_step_multi_with_capture(
        verify_tokens, handle_b, requested
    )

    assert logits_capture.shape == logits_baseline.shape
    _greedy_argmax_equal(logits_capture, logits_baseline)
    _per_element_close(logits_capture, logits_baseline)

    hidden_dim = adapter_b.config.hidden_size
    for layer_id in sorted(requested):
        h = captured[layer_id]
        assert h.shape == (1, T, hidden_dim), (
            f"layer {layer_id}: shape {h.shape} != (1, {T}, {hidden_dim})"
        )
