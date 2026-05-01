"""D-021 step 6 (β) skeleton tests for DFlashDrafter + TargetHiddenConsumer.

Pins the (β) skeleton's contracts without depending on a real DFlash
drafter checkpoint:

- ``DFlashDrafter`` implements both ``DraftEngine`` and
  ``TargetHiddenConsumer`` Protocols.
- ``prime`` aggregates per-layer slices using upstream's convention
  (``[i + 1 for i in target_layer_ids]``) and stores
  ``(1, ctx_len, |L| * hidden_size)``.
- ``update_target_hidden`` applies the same aggregation, slices to
  ``1 + yielded_count`` on axis 1.
- ``free_target_hidden`` drops the per-request state.
- Missing layer ids in the captured dict raise a descriptive
  ``KeyError`` referencing the requested + present sets.
- ``propose`` raises ``NotImplementedError`` (γ / δ pending).
- ``commit`` is a no-op (consistent with the F-1 state machine).

The tests bypass ``DFlashDrafter.__init__`` (which would import
``dflash_mlx`` and load a real drafter checkpoint) and construct the
wrapper via ``object.__new__`` + manual attribute injection. This
matches the (β) skeleton's stated scope: skeleton ships *construction
+ side-channel* + state shape; the loader path lives at sub-unit (δ)
under a real ``dflash-mlx`` install.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import mlx.core as mx
import pytest

from silica.speculative.dflash_drafter import DFlashDrafter
from silica.speculative.engine import (
    DraftEngine,
    NoopDraftEngine,
    TargetHiddenConsumer,
)


# Each test that exercises the wrapper builds it via ``_build`` so the
# real ``__init__`` (which calls ``dflash_mlx.runtime.load_draft_bundle``)
# is bypassed. This is equivalent to a "constructor took the bundle from
# the real loader" — except in the skeleton there is no real loader yet.
def _build(
    *,
    target_layer_ids: tuple[int, ...] = (0, 7, 14),
    hidden_size: int = 8,
) -> tuple[DFlashDrafter, int]:
    drafter = object.__new__(DFlashDrafter)
    drafter._drafter_repo = "test/synthetic"
    drafter._target_adapter = None  # type: ignore[assignment]
    drafter._drafter_model = None
    drafter._drafter_meta = None
    drafter._target_layer_ids = target_layer_ids
    drafter._target_hidden = {}
    drafter._draft_caches = {}
    drafter._synthetic_emit = None
    drafter._cache_factory = None
    return drafter, hidden_size


def _captured_dict(
    *,
    layer_ids_plus_one: tuple[int, ...],
    ctx_len: int,
    hidden_size: int,
) -> dict[int, mx.array]:
    """Build a synthetic captured dict whose entries match the
    adapter-side capture-dict convention (key 0 = embedding output;
    key i + 1 = output of layer i). Each value is filled with a
    layer-id-keyed constant so an aggregation result can be inspected
    feature-wise."""
    dct: dict[int, mx.array] = {}
    for key in layer_ids_plus_one:
        # Fill with a layer-keyed constant so we can check that the
        # concat along axis=-1 places features in the expected order.
        dct[key] = mx.full(
            shape=(1, ctx_len, hidden_size),
            vals=float(key),
            dtype=mx.float32,
        )
    return dct


def test_dflash_drafter_implements_protocols() -> None:
    """``DFlashDrafter`` is both a ``DraftEngine`` and a
    ``TargetHiddenConsumer``. ``NoopDraftEngine`` is a ``DraftEngine``
    but **not** a ``TargetHiddenConsumer`` — the gate that lets the
    engine route the side channel only to drafters that need it."""
    drafter, _ = _build()
    assert isinstance(drafter, DraftEngine)
    assert isinstance(drafter, TargetHiddenConsumer)

    noop = NoopDraftEngine()
    assert isinstance(noop, DraftEngine)
    assert not isinstance(noop, TargetHiddenConsumer)


def test_capture_layer_ids_property_pins_plus_one_offset() -> None:
    """``capture_layer_ids`` must apply the upstream ``+1`` offset
    between drafter ``target_layer_ids`` and silica's adapter-side
    capture-dict keys. Mismatch surfaces as a ``KeyError`` inside
    ``prime``; pinning the property eliminates the ambiguity at the
    Protocol surface so engine ε can route the set without computing
    the offset itself."""
    target_layer_ids = (0, 7, 14)
    drafter, _ = _build(target_layer_ids=target_layer_ids)
    expected = frozenset({1, 8, 15})
    assert drafter.capture_layer_ids == expected
    # Read-only contract: the property does not allocate storage that
    # could drift from the construction-time list.
    assert drafter.capture_layer_ids == drafter.capture_layer_ids


def test_capture_layer_ids_matches_what_prime_consumes() -> None:
    """Round-trip pin: a captured dict whose keys are
    ``drafter.capture_layer_ids`` is the exact set ``prime`` accepts
    without raising."""
    target_layer_ids = (0, 7, 14)
    drafter, hidden_size = _build(
        target_layer_ids=target_layer_ids, hidden_size=4
    )
    captured = _captured_dict(
        layer_ids_plus_one=tuple(sorted(drafter.capture_layer_ids)),
        ctx_len=3,
        hidden_size=hidden_size,
    )
    drafter.prime("req-roundtrip", captured)
    assert drafter._target_hidden["req-roundtrip"].shape == (
        1,
        3,
        len(target_layer_ids) * hidden_size,
    )


def test_prime_aggregates_target_hidden_with_correct_shape() -> None:
    """``prime`` must concatenate ``[captured_dict[i + 1] for i in
    target_layer_ids]`` along axis=-1 and store the result.
    Shape contract: ``(1, ctx_len, |L| * hidden_size)``."""
    target_layer_ids = (0, 7, 14)
    drafter, hidden_size = _build(
        target_layer_ids=target_layer_ids, hidden_size=8
    )
    ctx_len = 5

    # Adapter-side capture set is {i + 1 for i in target_layer_ids}.
    captured = _captured_dict(
        layer_ids_plus_one=tuple(i + 1 for i in target_layer_ids),
        ctx_len=ctx_len,
        hidden_size=hidden_size,
    )

    drafter.prime("req-0", captured)

    th = drafter._target_hidden["req-0"]
    assert th.shape == (1, ctx_len, len(target_layer_ids) * hidden_size)

    # Verify per-layer ordering: the first hidden_size features come
    # from layer_id+1=1 (= layer 0 output), the next block from
    # layer_id+1=8, the last from layer_id+1=15. Each block is filled
    # with the layer-key constant.
    expected_keys = [i + 1 for i in target_layer_ids]
    for block_idx, key in enumerate(expected_keys):
        block = th[:, :, block_idx * hidden_size : (block_idx + 1) * hidden_size]
        # All entries in this slice should equal the layer key.
        block_min = float(mx.min(block).item())
        block_max = float(mx.max(block).item())
        assert block_min == float(key), (
            f"block {block_idx} (layer key {key}): min={block_min} "
            f"!= {float(key)}"
        )
        assert block_max == float(key), (
            f"block {block_idx} (layer key {key}): max={block_max} "
            f"!= {float(key)}"
        )


def test_update_target_hidden_slices_to_yielded_count() -> None:
    """After a verify forward of length k, ``update_target_hidden``
    aggregates the new dict and slices to ``1 + yielded_count``
    positions on axis=1. Tests both the shape and the value
    preservation along the slicing axis."""
    target_layer_ids = (0, 7)
    drafter, hidden_size = _build(
        target_layer_ids=target_layer_ids, hidden_size=4
    )

    # Cycle 1: prime with prompt-len ctx.
    prompt_dict = _captured_dict(
        layer_ids_plus_one=(1, 8),
        ctx_len=10,
        hidden_size=hidden_size,
    )
    drafter.prime("req-0", prompt_dict)

    # Cycle 2: verify forward over k=4 tokens, yielded_count=2 (so the
    # next cycle's target_hidden length is 3).
    verify_k = 4
    verify_dict = _captured_dict(
        layer_ids_plus_one=(1, 8),
        ctx_len=verify_k,
        hidden_size=hidden_size,
    )
    drafter.update_target_hidden("req-0", verify_dict, yielded_count=2)

    th = drafter._target_hidden["req-0"]
    assert th.shape == (1, 3, len(target_layer_ids) * hidden_size)


def test_update_target_hidden_rejects_overflow() -> None:
    """``yielded_count`` cannot exceed the capture window — the engine
    sets it from greedy_verify and capped at max_tokens / stop, so an
    overflow indicates wiring error. The wrapper raises rather than
    silently producing wrong shapes."""
    drafter, hidden_size = _build(target_layer_ids=(0,), hidden_size=4)
    drafter.prime(
        "req-0",
        _captured_dict(
            layer_ids_plus_one=(1,), ctx_len=5, hidden_size=hidden_size
        ),
    )

    verify_dict = _captured_dict(
        layer_ids_plus_one=(1,), ctx_len=4, hidden_size=hidden_size
    )
    with pytest.raises(ValueError, match="exceeds capture window"):
        drafter.update_target_hidden(
            "req-0", verify_dict, yielded_count=10
        )


def test_update_target_hidden_unknown_req_id_raises() -> None:
    drafter, hidden_size = _build(target_layer_ids=(0,), hidden_size=4)
    verify_dict = _captured_dict(
        layer_ids_plus_one=(1,), ctx_len=4, hidden_size=hidden_size
    )
    with pytest.raises(KeyError, match="not primed"):
        drafter.update_target_hidden(
            "missing-req", verify_dict, yielded_count=0
        )


def test_aggregate_target_hidden_descriptive_keyerror_on_missing_layer() -> None:
    """Missing layer ids in the captured dict raise a ``KeyError``
    that lists requested vs present, so an engine-wiring bug at the
    adapter / drafter layer-set boundary is diagnosed without a model
    load."""
    drafter, hidden_size = _build(
        target_layer_ids=(0, 7, 14), hidden_size=4
    )
    # Engine forwarded a dict missing the layer-id-7 + 1 = 8 entry.
    bad_dict = _captured_dict(
        layer_ids_plus_one=(1, 15), ctx_len=3, hidden_size=hidden_size
    )
    with pytest.raises(KeyError) as excinfo:
        drafter.prime("req-0", bad_dict)
    msg = str(excinfo.value)
    assert "8" in msg or "[1, 8, 15]" in msg
    assert "[1, 15]" in msg or "1, 15" in msg


def test_free_target_hidden_drops_state() -> None:
    drafter, hidden_size = _build(target_layer_ids=(0,), hidden_size=4)
    drafter.prime(
        "req-0",
        _captured_dict(
            layer_ids_plus_one=(1,), ctx_len=2, hidden_size=hidden_size
        ),
    )
    assert "req-0" in drafter._target_hidden
    drafter.free_target_hidden("req-0")
    assert "req-0" not in drafter._target_hidden
    assert "req-0" not in drafter._draft_caches
    # Idempotent — freeing an unknown req_id is a no-op.
    drafter.free_target_hidden("never-primed")


def _make_ctx(req_id: str = "req-0") -> Any:
    """Build a minimal RequestState for propose/commit signature tests."""
    from silica.core.request import Request, RequestState
    from silica.core.sampling import SamplingParams

    req = Request(
        prompt="",
        sampling_params=SamplingParams(),
        request_id=req_id,
    )
    return RequestState(request=req)


def test_real_mode_propose_requires_primed_request() -> None:
    """Real-mode ``propose`` (no synthetic emitter installed) routes
    through ``_propose_real`` per sub-unit (δ.1). The unprimed-request
    guard is the first check it hits — engine ε guarantees priming
    after ``prefill_with_capture``; unit tests that bypass priming
    surface the wiring violation here."""
    drafter, _ = _build()
    assert drafter._synthetic_emit is None
    ctx = _make_ctx()
    with pytest.raises(KeyError, match="not primed"):
        drafter.propose(ctx, k=4)


def test_commit_is_noop() -> None:
    """``commit`` is a no-op per the F-1 state machine — draft-side
    rollback is implicit (rejected drafts' noise keys/values were never
    appended to ``draft_caches``; ``update_target_hidden`` slices the
    new ``target_hidden`` to the committed length)."""
    drafter, _ = _build()
    ctx = _make_ctx()
    # Should not raise; should not perturb wrapper state.
    drafter.commit(ctx, accepted_len=2)
    assert drafter._target_hidden == {}
    assert drafter._draft_caches == {}


def test_read_target_layer_ids_loud_fails_on_missing_attr() -> None:
    """If the loaded drafter model lacks ``target_layer_ids``, the
    skeleton must raise ``RuntimeError`` referencing the upstream
    DFlashDraftModel.__init__ contract — silently returning ``()``
    would surface as a confusing ``mx.concatenate([], axis=-1)``
    crash inside ``prime``."""

    class _FakeBareModel:  # no target_layer_ids
        pass

    drafter = object.__new__(DFlashDrafter)
    drafter._drafter_repo = "test/synthetic"
    drafter._target_adapter = None  # type: ignore[assignment]
    drafter._drafter_model = _FakeBareModel()
    drafter._drafter_meta = None
    with pytest.raises(RuntimeError, match="target_layer_ids"):
        drafter._read_target_layer_ids()


def test_read_target_layer_ids_loud_fails_on_empty() -> None:
    """Empty ``target_layer_ids`` indicates a broken upstream config;
    upstream's ``build_target_layer_ids`` should always synthesise a
    non-empty list. Loud-fail rather than letting the empty list flow
    through ``mx.concatenate``."""

    class _FakeEmptyIds:
        target_layer_ids: tuple[int, ...] = ()

    drafter = object.__new__(DFlashDrafter)
    drafter._drafter_repo = "test/synthetic"
    drafter._target_adapter = None  # type: ignore[assignment]
    drafter._drafter_model = _FakeEmptyIds()
    drafter._drafter_meta = None
    with pytest.raises(RuntimeError, match="empty"):
        drafter._read_target_layer_ids()


def test_read_target_layer_ids_returns_tuple_from_model_instance() -> None:
    """Happy path: the field lives on the model instance, not on
    ``args``. The skeleton reads ``drafter_model.target_layer_ids``
    and materialises to a tuple."""

    class _FakeModel:
        target_layer_ids = [0, 7, 14]
        # An ``args`` attribute exists but does NOT carry
        # target_layer_ids — confirms the skeleton no longer reads
        # from args.target_layer_ids (the previous β bug).

        class args:
            target_layer_ids = [99, 99, 99]  # would be wrong if read

    drafter = object.__new__(DFlashDrafter)
    drafter._drafter_repo = "test/synthetic"
    drafter._target_adapter = None  # type: ignore[assignment]
    drafter._drafter_model = _FakeModel()
    drafter._drafter_meta = None
    assert drafter._read_target_layer_ids() == (0, 7, 14)


_HAS_DFLASH = importlib.util.find_spec("dflash_mlx") is not None


@pytest.mark.skipif(
    _HAS_DFLASH,
    reason="dflash-mlx is installed; this test pins the import-error path",
)
def test_init_raises_clear_import_error_when_dflash_missing() -> None:
    """Without ``dflash-mlx`` installed, ``DFlashDrafter()`` raises a
    descriptive ``ImportError`` mentioning the extras-marker install
    command."""
    with pytest.raises(ImportError, match="silica-mlx\\[dflash\\]"):
        DFlashDrafter(
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
            target_adapter=object(),  # type: ignore[arg-type]
        )
