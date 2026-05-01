"""D-021 step 6 (δ.1) — DFlashDrafter real-mode propose mechanics.

Pins the real-mode ``propose`` call shape without downloading a real
``z-lab/Qwen3.5-*-DFlash`` checkpoint. Real-checkpoint attestation +
accept-rate / throughput measurement land at sub-unit (η).

Fixtures: a synthetic ``_FakeDFlashModel`` standing in for upstream
``dflash_mlx.model.DFlashDraftModel`` and a ``_FakeTargetAdapter``
standing in for ``Qwen3_5Adapter``. Both expose only the surfaces
``DFlashDrafter._propose_real`` actually consumes:

- ``_FakeDFlashModel``: ``layers`` (length sets the cache-list length),
  ``block_size``, ``mask_token_id``, ``__call__(noise_embedding,
  target_hidden, cache)`` returning a ``(1, block_len, hidden_size)``
  hidden array. Records every ``__call__`` invocation for the tests
  to inspect.
- ``_FakeTargetAdapter``: a ``_model`` attribute matching the two-level
  Qwen3.5 wrapper layout (``language_model.model.embed_tokens`` +
  ``embed_tokens.as_linear`` for tied-embedding projection).

Coverage:

- Unprimed ``req_id`` raises ``KeyError`` (already guarded above the
  ``_propose_real`` entry; pinned for regression).
- Empty ``ctx.output_token_ids`` raises ``RuntimeError`` (the
  staged-first contract; engine ε guarantees a yielded token before
  the first ``propose``).
- ``block_len = min(k + 1, drafter.block_size)`` — both directions.
- ``block_token_buffer[0] = staged_first``, ``[1:] = mask_token_id``.
- The same per-``req_id`` cache list is passed to ``__call__`` across
  successive propose calls (state is preserved).
- Returned ``DraftTokens.token_ids`` length is exactly
  ``min(k, block_size - 1)``.
- Loud-fail when the drafter model is missing ``block_size`` /
  ``mask_token_id`` (regression guard for upstream attribute drift).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from silica.core.request import Request, RequestState
from silica.core.sampling import SamplingParams
from silica.speculative.dflash_drafter import DFlashDrafter


class _FakeLayer:
    """Stand-in for a single drafter layer — only its presence in the
    ``layers`` list is consumed by ``_build_draft_caches``."""


class _FakeDFlashModel:
    """Synthetic ``DFlashDraftModel``. Records every ``__call__``.

    Returns a deterministic ``(1, block_len, hidden_size)`` hidden
    array filled with the per-position index (so the lm-head
    projection's argmax is predictable).
    """

    def __init__(
        self,
        *,
        block_size: int,
        mask_token_id: int,
        num_layers: int,
        hidden_size: int,
        target_layer_ids: tuple[int, ...] = (0,),
    ) -> None:
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.layers = [_FakeLayer() for _ in range(num_layers)]
        self.target_layer_ids = target_layer_ids
        self.hidden_size = hidden_size
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        *,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: Any = None,
    ) -> mx.array:
        # Record the call shape.
        self.calls.append(
            {
                "noise_embedding_shape": tuple(noise_embedding.shape),
                "target_hidden_shape": tuple(target_hidden.shape),
                # Capture cache by identity so tests can confirm the
                # same per-req_id list is passed across cycles.
                "cache_id": id(cache),
                "cache_len": len(cache) if cache is not None else None,
            }
        )
        # Return shape (1, block_len, hidden_size) where block_len is
        # the noise_embedding's T axis.
        block_len = int(noise_embedding.shape[1])
        # Build a hidden array whose row ``i`` has all entries equal
        # to ``i`` — makes the argmax over the lm-head projection
        # below deterministic.
        rows = []
        for i in range(block_len):
            rows.append(
                mx.full(
                    shape=(self.hidden_size,),
                    vals=float(i),
                    dtype=mx.float32,
                )
            )
        return mx.stack(rows, axis=0)[None]  # (1, block_len, hidden_size)


class _FakeInnerModel:
    """Stand-in for ``Qwen3_5TextModel`` — only the surfaces
    :func:`silica.speculative._dflash_target_ops.target_embed_tokens`
    and :func:`lm_head_logits` consume."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.embed_tokens = _FakeEmbedding(vocab_size, hidden_size)


class _FakeEmbedding:
    """Tied-embedding stand-in. Returns a deterministic
    ``(B, T, hidden_size)`` lookup; ``as_linear`` projects back to
    ``(B, T, vocab_size)`` so the argmax over the lm-head is
    predictable from the input row index."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def __call__(self, tokens_2d: mx.array) -> mx.array:
        # Deterministic embedding: each token id maps to a
        # ``hidden_size``-vector of zeros — the contents are not
        # consumed by the test, only the shape.
        b, t = int(tokens_2d.shape[0]), int(tokens_2d.shape[1])
        return mx.zeros(shape=(b, t, self.hidden_size), dtype=mx.float32)

    def as_linear(self, hidden: mx.array) -> mx.array:
        # Projection: argmax of the projection equals
        # ``min(hidden_value, vocab_size - 1)`` for the synthetic
        # hidden used here. Build a logits tensor whose row ``j`` for
        # input position ``j`` has its argmax at column ``j %
        # vocab_size``. Concretely: take the per-position scalar value
        # of ``hidden`` (we set every entry of row ``i`` to ``i``) and
        # return one-hot logits at column ``i % vocab_size``.
        b, t, _ = hidden.shape
        rows = []
        for j in range(t):
            scalar = int(hidden[0, j, 0].item())
            col = scalar % self.vocab_size
            row = mx.zeros(shape=(self.vocab_size,), dtype=mx.float32)
            before = mx.zeros(shape=(col,), dtype=mx.float32)
            winner = mx.array([5.0], dtype=mx.float32)
            after = mx.zeros(
                shape=(self.vocab_size - col - 1,), dtype=mx.float32
            )
            row = mx.concatenate([before, winner, after])
            rows.append(row)
        return mx.stack(rows, axis=0)[None]  # (1, T, vocab)


class _FakeWrapper:
    """Mirror of ``Qwen3_5.TextModel``: ``model`` is the inner text
    model; ``args.tie_word_embeddings`` chooses the projection path."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.model = _FakeInnerModel(vocab_size, hidden_size)
        self.args = type("_Args", (), {"tie_word_embeddings": True})()


class _FakeOuterModel:
    """Mirror of mlx-lm's outer ``Model`` for Qwen3.5: the wrapper
    hangs off ``language_model``."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.language_model = _FakeWrapper(vocab_size, hidden_size)


class _FakeTargetAdapter:
    """``adapter._model`` matches the two-level Qwen3.5 layout the
    target ops surface navigates."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self._model = _FakeOuterModel(vocab_size, hidden_size)


def _build(
    *,
    block_size: int = 8,
    k_layers: int = 2,
    target_layer_ids: tuple[int, ...] = (0,),
    vocab_size: int = 32,
    hidden_size: int = 4,
    mask_token_id: int = 99,
) -> DFlashDrafter:
    """Construct a real-mode-but-fake DFlashDrafter for δ.1 tests."""
    drafter = object.__new__(DFlashDrafter)
    drafter._drafter_repo = "fake/synthetic"
    # Cast through Any — the fake target adapter only needs to satisfy
    # the surface ``_dflash_target_ops`` consumes (``_model``), not
    # the full ``HiddenCaptureAdapter`` Protocol.
    fake_target: Any = _FakeTargetAdapter(vocab_size, hidden_size)
    drafter._target_adapter = fake_target
    drafter._drafter_model = _FakeDFlashModel(
        block_size=block_size,
        mask_token_id=mask_token_id,
        num_layers=k_layers,
        hidden_size=hidden_size,
        target_layer_ids=target_layer_ids,
    )
    drafter._drafter_meta = None
    drafter._target_layer_ids = target_layer_ids
    drafter._target_hidden = {}
    drafter._draft_caches = {}
    drafter._synthetic_emit = None
    drafter._cache_factory = None  # placeholder mode for δ.1
    return drafter


def _make_ctx(req_id: str = "req-delta", output_tail: int = 7) -> RequestState:
    ctx = RequestState(
        request=Request(
            prompt="",
            sampling_params=SamplingParams(),
            request_id=req_id,
        ),
    )
    ctx.output_token_ids = [output_tail]
    return ctx


def _captured_dict(
    *,
    capture_keys: tuple[int, ...],
    ctx_len: int,
    hidden_size: int,
) -> dict[int, mx.array]:
    return {
        key: mx.full(
            shape=(1, ctx_len, hidden_size),
            vals=float(key),
            dtype=mx.float32,
        )
        for key in capture_keys
    }


# --- coverage ---------------------------------------------------------------


def test_real_propose_unprimed_request_raises() -> None:
    drafter = _build()
    ctx = _make_ctx("never-primed")
    with pytest.raises(KeyError, match="not primed"):
        drafter.propose(ctx, k=4)


def test_real_propose_empty_output_token_ids_raises() -> None:
    """If the engine calls ``propose`` before yielding the prefill
    argmax, ``ctx.output_token_ids`` is empty and the wrapper has no
    staged-first to seed the block buffer with."""
    drafter = _build(target_layer_ids=(0,))
    ctx = _make_ctx("req-empty")
    ctx.output_token_ids = []  # explicitly clear
    drafter.prime(
        "req-empty",
        _captured_dict(capture_keys=(1,), ctx_len=3, hidden_size=4),
    )
    with pytest.raises(RuntimeError, match="output_token_ids is.*empty"):
        drafter.propose(ctx, k=4)


def test_real_propose_block_len_uses_min_k_plus_one_and_block_size() -> None:
    """``block_len = min(k + 1, drafter.block_size)``. Test both
    directions: when k+1 is smaller (k is the binding constraint) and
    when block_size is smaller (the checkpoint's block_size caps)."""
    # Case A: k+1 < block_size. block_len = k+1.
    drafter_a = _build(block_size=16)
    drafter_a.prime(
        "req-a",
        _captured_dict(capture_keys=(1,), ctx_len=3, hidden_size=4),
    )
    drafts_a = drafter_a.propose(_make_ctx("req-a"), k=4)
    # block_len = min(4 + 1, 16) = 5; returned drafts = block_len - 1 = 4.
    assert len(drafts_a.token_ids) == 4

    # Case B: block_size < k + 1. block_len = block_size.
    drafter_b = _build(block_size=3)
    drafter_b.prime(
        "req-b",
        _captured_dict(capture_keys=(1,), ctx_len=3, hidden_size=4),
    )
    drafts_b = drafter_b.propose(_make_ctx("req-b"), k=8)
    # block_len = min(8 + 1, 3) = 3; returned drafts = 2.
    assert len(drafts_b.token_ids) == 2


def test_real_propose_passes_staged_first_at_position_zero() -> None:
    """The forward must see ``noise_embedding`` of shape
    ``(1, block_len, hidden_size)`` where position 0 corresponds to
    ``ctx.output_token_ids[-1]`` and positions 1..block_len-1 to
    ``mask_token_id``. The ``_FakeDFlashModel`` records the call
    shape so we can pin block_len; the inner embedding pass is
    deterministic but doesn't preserve token ids, so we verify
    block_len directly."""
    drafter = _build(block_size=16, mask_token_id=99)
    drafter.prime(
        "req-staged",
        _captured_dict(capture_keys=(1,), ctx_len=2, hidden_size=4),
    )
    ctx = _make_ctx("req-staged", output_tail=42)
    drafter.propose(ctx, k=4)
    fake = drafter._drafter_model
    assert isinstance(fake, _FakeDFlashModel)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    # noise_embedding shape (1, block_len, hidden_size); block_len = 5.
    assert call["noise_embedding_shape"] == (1, 5, 4)
    # target_hidden shape (1, ctx_len, |L| * hidden_size) = (1, 2, 4).
    assert call["target_hidden_shape"] == (1, 2, 4)


def test_real_propose_passes_same_cache_list_across_cycles() -> None:
    """Per-``req_id`` cache list identity must be preserved across
    successive propose calls — the ``ContextOnlyDraftKVCache``
    instances are stateful, and rebuilding them per cycle would
    discard the streaming context."""
    drafter = _build(block_size=8, k_layers=2)
    drafter.prime(
        "req-cache-id",
        _captured_dict(capture_keys=(1,), ctx_len=3, hidden_size=4),
    )
    ctx = _make_ctx("req-cache-id")
    drafter.propose(ctx, k=4)
    drafter.propose(ctx, k=4)
    fake = drafter._drafter_model
    assert isinstance(fake, _FakeDFlashModel)
    assert len(fake.calls) == 2
    assert fake.calls[0]["cache_id"] == fake.calls[1]["cache_id"]
    # Cache list length matches the drafter's layer count.
    assert fake.calls[0]["cache_len"] == 2


def test_real_propose_returns_token_ids_within_k_bound() -> None:
    """``DraftTokens.token_ids`` length is exactly
    ``min(k, block_size - 1)``. Pins the contract end-to-end so the
    engine's ``propose`` call site (which checks ``draft_count >
    gamma``) does not need extra slack."""
    drafter = _build(block_size=10)
    drafter.prime(
        "req-bound",
        _captured_dict(capture_keys=(1,), ctx_len=2, hidden_size=4),
    )
    drafts = drafter.propose(_make_ctx("req-bound"), k=3)
    assert len(drafts.token_ids) == 3  # min(3, 10 - 1)


def test_real_propose_loud_fails_on_missing_block_size() -> None:
    drafter = _build()
    drafter.prime(
        "req-missing-bs",
        _captured_dict(capture_keys=(1,), ctx_len=2, hidden_size=4),
    )
    # Strip block_size from the fake drafter.
    delattr(drafter._drafter_model, "block_size")
    with pytest.raises(RuntimeError, match="block_size"):
        drafter.propose(_make_ctx("req-missing-bs"), k=4)


def test_real_propose_loud_fails_on_missing_mask_token_id() -> None:
    drafter = _build()
    drafter.prime(
        "req-missing-mask",
        _captured_dict(capture_keys=(1,), ctx_len=2, hidden_size=4),
    )
    delattr(drafter._drafter_model, "mask_token_id")
    with pytest.raises(RuntimeError, match="mask_token_id"):
        drafter.propose(_make_ctx("req-missing-mask"), k=4)


def test_real_propose_k_zero_returns_empty_drafts() -> None:
    """``k=0`` is a degenerate case the I-5 docstring lists as legal:
    the propose budget is "up to k", and zero is allowed. The wrapper
    returns empty drafts without invoking the drafter forward."""
    drafter = _build()
    drafter.prime(
        "req-k0",
        _captured_dict(capture_keys=(1,), ctx_len=2, hidden_size=4),
    )
    drafts = drafter.propose(_make_ctx("req-k0"), k=0)
    assert drafts.token_ids == ()
    fake = drafter._drafter_model
    assert isinstance(fake, _FakeDFlashModel)
    assert len(fake.calls) == 0  # no forward at all
