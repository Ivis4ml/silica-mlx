"""Real-model tests for ``Qwen3_5Adapter.decode_step_multi_with_capture``.

D-021 step 6 sub-unit (αβ.1). Pins three contracts on cached
``Qwen/Qwen3.5-0.8B``:

1. **Capture-disabled byte-exactness.** With ``capture_layer_ids =
   frozenset()`` the helper produces logits position-for-position
   identical to ``decode_step_multi``. The two paths reproduce
   ``Qwen3_5TextModel.__call__`` exactly (same embed → mask → layer
   iter → norm → projection), so an empty-capture invocation must be
   bit-equal in greedy argmax and within fp16 dispatch noise per
   element.
2. **Capture-enabled logits stability.** With a non-empty
   ``capture_layer_ids`` the returned logits agree with the
   capture-disabled call (greedy argmax exact, per-element bound).
   Capture is purely a side-effect that exposes intermediate
   hidden-state slices; it must not perturb the verify forward.
3. **Hidden-slice shape contract.** For each requested layer id, the
   captured hidden state has shape ``(1, T, hidden_dim)`` where T is
   the input window length. Convention: key ``0`` = embedding output;
   key ``i + 1`` = output of ``model.layers[i]`` (matching
   ``dflash_mlx.runtime.target_forward_with_hidden_states``).

The fixture is the cached Qwen3.5-0.8B hybrid (DeltaNet + GQA), the
same one ``test_decode_step_multi_real.py`` and ``test_spec_parity.py``
use. Skipped when the cache is absent or ``SILICA_SKIP_MODEL_TESTS=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from silica.kvcache.manager import KVHandle
from silica.models.hidden_capture import HiddenCaptureAdapter
from silica.models.qwen3_5 import Qwen3_5Adapter

REPO = "Qwen/Qwen3.5-0.8B"

_QWEN3_5_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3.5-0.8B"
)
_QWEN3_5_SKIP = not _QWEN3_5_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_QWEN3_5_SKIP_REASON = (
    f"Qwen3.5-0.8B not cached at {_QWEN3_5_CACHE}; pull via "
    "huggingface-cli or any test that loads the 0.8B fixture."
)

T = 4
TOKEN_IDS = [101, 202, 303, 404][:T]


def _greedy_argmax_equal(a: mx.array, b: mx.array) -> None:
    """Position-for-position argmax equality. The verify path reads
    only the argmax, so this is the load-bearing assertion."""
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
    """Tight per-element bound. Both paths reproduce the same
    forward-pass arithmetic, so the only acceptable drift is from MLX
    dispatch reductions, which is well below rtol=1e-4."""
    diff = mx.max(mx.abs(a - b))
    bound = rtol * float(mx.max(mx.abs(b)).item()) + atol
    assert float(diff.item()) <= bound, (
        f"|a - b|_max={float(diff.item()):.4e} exceeds rtol*|b| + atol "
        f"= {bound:.4e}"
    )


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_qwen3_5_adapter_implements_hidden_capture_protocol() -> None:
    adapter, _ = Qwen3_5Adapter.from_hf_repo(REPO)
    assert isinstance(adapter, HiddenCaptureAdapter), (
        "Qwen3_5Adapter must implement HiddenCaptureAdapter; (αβ.1) "
        "lands the first concrete adapter in the family."
    )


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_capture_disabled_matches_decode_step_multi() -> None:
    """Empty ``capture_layer_ids`` reproduces ``decode_step_multi``'s
    logits position-for-position. Both paths run the same embed →
    layers → norm → projection pipeline, so the result is byte-equal
    in argmax and tight per-element."""
    adapter_a, kv_a = Qwen3_5Adapter.from_hf_repo(REPO)
    adapter_b, kv_b = Qwen3_5Adapter.from_hf_repo(REPO)

    req_id = "capture-eq-test"
    handle_a = KVHandle(req_id=req_id)
    handle_b = KVHandle(req_id=req_id)

    kv_a.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_b.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    tokens = mx.array(TOKEN_IDS, dtype=mx.int32)

    logits_baseline, _ = adapter_a.decode_step_multi(tokens, handle_a)

    logits_capture, captured, _ = adapter_b.decode_step_multi_with_capture(
        tokens, handle_b, frozenset()
    )

    assert captured == {}, (
        "empty capture_layer_ids must return an empty dict; got "
        f"{sorted(captured.keys())}"
    )
    assert logits_capture.shape == logits_baseline.shape

    _greedy_argmax_equal(logits_capture, logits_baseline)
    _per_element_close(logits_capture, logits_baseline)


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_capture_enabled_returns_hidden_slices_with_correct_shape() -> None:
    """Non-empty capture: returned dict has the requested keys; each
    value has shape ``(1, T, hidden_dim)`` with the model's
    ``hidden_size``. Capture-disabled vs capture-enabled logits agree
    bit-for-bit in argmax and within tight tolerance per element."""
    adapter_a, kv_a = Qwen3_5Adapter.from_hf_repo(REPO)
    adapter_b, kv_b = Qwen3_5Adapter.from_hf_repo(REPO)

    req_id = "capture-shape-test"
    handle_a = KVHandle(req_id=req_id)
    handle_b = KVHandle(req_id=req_id)

    kv_a.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_b.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    tokens = mx.array(TOKEN_IDS, dtype=mx.int32)

    # Pick three layer ids spanning the model: embedding (0), an
    # early-mid layer, and the last layer. ``num_layers`` lives on
    # the adapter's config (silica's naming for ``num_hidden_layers``).
    num_layers = adapter_a.config.num_layers
    requested = frozenset({0, num_layers // 2, num_layers})

    logits_no_cap, _ = adapter_a.decode_step_multi(tokens, handle_a)
    logits_cap, captured, _ = adapter_b.decode_step_multi_with_capture(
        tokens, handle_b, requested
    )

    # Logit equivalence first — capture must not perturb the forward.
    _greedy_argmax_equal(logits_cap, logits_no_cap)
    _per_element_close(logits_cap, logits_no_cap)

    # Captured-set equality.
    assert set(captured.keys()) == set(requested)

    # Shape contract: each captured hidden has shape (1, T, hidden_dim).
    hidden_dim = adapter_a.config.hidden_size
    for layer_id in sorted(requested):
        h = captured[layer_id]
        assert h.shape == (1, T, hidden_dim), (
            f"layer {layer_id}: shape {h.shape} != (1, {T}, {hidden_dim})"
        )


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_capture_invalid_layer_id_is_silently_dropped() -> None:
    """Layer ids outside ``[0, num_hidden_layers]`` simply don't appear
    in the captured dict — the helper writes only when the running
    layer-output index is in the requested set. This keeps the
    drafter wrapper's call shape simple: the wrapper passes the
    drafter's ``target_layer_ids`` set directly without per-call
    bounds-check."""
    adapter, kv = Qwen3_5Adapter.from_hf_repo(REPO)

    req_id = "capture-invalid-test"
    handle = KVHandle(req_id=req_id)
    kv.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    tokens = mx.array(TOKEN_IDS, dtype=mx.int32)

    # 999 is far past num_hidden_layers; 0 is valid.
    requested = frozenset({0, 999})
    _, captured, _ = adapter.decode_step_multi_with_capture(
        tokens, handle, requested
    )

    assert set(captured.keys()) == {0}


# --- (αβ.3) prefill capture seed + cached-prefix regression -----------------


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_prefill_with_capture_empty_matches_prefill_last_pos_logits() -> None:
    """``prefill_with_capture(tokens, kv, frozenset())`` produces the
    same last-position logits as ``prefill(tokens, kv)`` and an empty
    captured dict. This is the cycle-1 seed precondition: the engine
    must be able to call ``prefill_with_capture`` instead of
    ``prefill`` without changing the next-token argmax."""
    adapter_a, kv_a = Qwen3_5Adapter.from_hf_repo(REPO)
    adapter_b, kv_b = Qwen3_5Adapter.from_hf_repo(REPO)

    req_id = "prefill-capture-empty-test"
    handle_a = KVHandle(req_id=req_id)
    handle_b = KVHandle(req_id=req_id)
    kv_a.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_b.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    prompt_tokens = mx.array([101, 202, 303, 404, 505], dtype=mx.int32)

    logits_a, _ = adapter_a.prefill(prompt_tokens, handle_a)
    logits_b, captured, _ = adapter_b.prefill_with_capture(
        prompt_tokens, handle_b, frozenset()
    )

    assert captured == {}
    # ``prefill`` returns ``(V,)`` last-position; both must agree.
    assert logits_a.shape == logits_b.shape
    a_top1 = int(mx.argmax(logits_a).item())
    b_top1 = int(mx.argmax(logits_b).item())
    assert a_top1 == b_top1, (
        f"prefill argmax mismatch: a={a_top1}, b={b_top1}"
    )
    diff = float(mx.max(mx.abs(logits_a - logits_b)).item())
    bound = 1e-4 * float(mx.max(mx.abs(logits_a)).item()) + 1e-3
    assert diff <= bound, f"|a - b|_max={diff:.4e} exceeds bound={bound:.4e}"


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_prefill_with_capture_returns_full_prompt_hiddens() -> None:
    """Non-empty capture during prefill returns hidden-state slices of
    shape ``(1, prompt_len, hidden_dim)`` — one captured frame per
    prompt position. This is the cycle-1 ``target_hidden`` source the
    (β) ``DFlashDrafter.prime(req_id, target_hidden)`` will consume."""
    adapter, kv = Qwen3_5Adapter.from_hf_repo(REPO)

    req_id = "prefill-capture-shape-test"
    handle = KVHandle(req_id=req_id)
    kv.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    prompt_tokens = mx.array([101, 202, 303, 404, 505], dtype=mx.int32)
    prompt_len = int(prompt_tokens.size)

    num_layers = adapter.config.num_layers
    requested = frozenset({0, num_layers // 2, num_layers})

    logits, captured, _ = adapter.prefill_with_capture(
        prompt_tokens, handle, requested
    )

    assert logits.shape == (adapter.config.vocab_size,) or (
        # mlx-lm padded vocab; assert the model actually produced a
        # 1-D logit row whose length matches the loop path.
        len(logits.shape) == 1 and logits.shape[0] >= adapter.config.vocab_size
    )
    assert set(captured.keys()) == set(requested)
    hidden_dim = adapter.config.hidden_size
    for layer_id in sorted(requested):
        h = captured[layer_id]
        assert h.shape == (1, prompt_len, hidden_dim), (
            f"layer {layer_id}: shape {h.shape} != "
            f"(1, {prompt_len}, {hidden_dim})"
        )


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_capture_after_prefill_matches_decode_step_multi_after_prefill() -> None:
    """The production spec-on cycle runs ``decode_step_multi`` *after*
    a prompt prefill, so the per-layer cache holds prompt KV and
    ``create_attention_mask`` / ``create_ssm_mask`` are computed
    against a non-empty offset. Pin that the capture path produces
    bit-equivalent logits in this regime, not just the empty-cache
    regime that ``test_capture_disabled_matches_decode_step_multi``
    pins. (αβ.3) regression."""
    adapter_a, kv_a = Qwen3_5Adapter.from_hf_repo(REPO)
    adapter_b, kv_b = Qwen3_5Adapter.from_hf_repo(REPO)

    req_id = "capture-after-prefill-test"
    handle_a = KVHandle(req_id=req_id)
    handle_b = KVHandle(req_id=req_id)
    kv_a.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]
    kv_b.reserve_for_prefill(req_id, [])  # type: ignore[arg-type]

    # Prefill the same prompt on both adapters so the per-layer cache
    # holds identical prompt KV state on each side.
    prompt_tokens = mx.array([101, 202, 303, 404, 505], dtype=mx.int32)
    adapter_a.prefill(prompt_tokens, handle_a)
    adapter_b.prefill(prompt_tokens, handle_b)

    # Now run a verify window with non-empty cache.
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

    # Hidden-shape contract holds with the prefilled cache state.
    hidden_dim = adapter_b.config.hidden_size
    for layer_id in sorted(requested):
        h = captured[layer_id]
        assert h.shape == (1, T, hidden_dim), (
            f"layer {layer_id}: shape {h.shape} != (1, {T}, {hidden_dim})"
        )
