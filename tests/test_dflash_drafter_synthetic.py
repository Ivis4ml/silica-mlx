"""D-021 step 6 (γ) — DFlashDrafter synthetic-emitter unit tests.

Pins the synthetic-mode ``propose`` path the (γ) seam ships:

- ``DFlashDrafter.for_synthetic(target_layer_ids, synthetic_emit)`` is
  the formal construction route — bypasses the real ``__init__`` (no
  ``dflash-mlx`` import or HF download) but produces a fully
  ``DraftEngine`` + ``TargetHiddenConsumer`` -conformant wrapper.
- Synthetic ``propose`` requires the request to be primed; raises
  ``KeyError`` otherwise. The engine ε will guarantee priming after
  ``prefill_with_capture``; tests must prime explicitly.
- The synthetic emitter receives ``(target_hidden, k)`` and must
  return ``len(token_ids) <= k`` — overflow is loud-fail rather
  than truncation.
- After ``update_target_hidden(req_id, captured, yielded_count)``,
  the next ``propose`` sees a stored ``target_hidden`` of length
  ``1 + yielded_count`` on axis 1, regardless of the previous
  block's length. This is the F-1 state-machine invariant.
- A small ``greedy_verify`` test on synthetic logits proves the
  oracle-replay design works at the verifier level: when the
  drafted tokens equal the target's argmax positions, the verifier
  accepts the full block.

(γ) does **not** integrate ``Engine.generate``. Engine wiring +
real-target cycle-1 byte-exact parity belong to sub-unit (ε).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.core.request import Request, RequestState
from silica.core.sampling import SamplingParams
from silica.speculative.dflash_drafter import DFlashDrafter, SyntheticEmit
from silica.speculative.engine import (
    DraftEngine,
    TargetHiddenConsumer,
)
from silica.speculative.verify import greedy_verify


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


def _make_ctx(req_id: str = "req-gamma") -> RequestState:
    return RequestState(
        request=Request(
            prompt="",
            sampling_params=SamplingParams(),
            request_id=req_id,
        ),
    )


# --- (γ) `for_synthetic` factory ------------------------------------------


def test_for_synthetic_produces_protocol_conformant_drafter() -> None:
    """``for_synthetic`` returns a wrapper that satisfies both
    ``DraftEngine`` and ``TargetHiddenConsumer`` — the same
    ``isinstance`` profile as a real-mode instance, so engine ε's
    routing works identically across modes."""

    def emit(target_hidden: mx.array, k: int) -> tuple[int, ...]:
        del target_hidden, k
        return ()

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0, 7, 14),
        synthetic_emit=emit,
    )
    assert isinstance(drafter, DraftEngine)
    assert isinstance(drafter, TargetHiddenConsumer)
    assert drafter.capture_layer_ids == frozenset({1, 8, 15})


def test_for_synthetic_rejects_empty_target_layer_ids() -> None:
    """Mirrors the real-path ``_read_target_layer_ids`` loud-fail."""

    def emit(target_hidden: mx.array, k: int) -> tuple[int, ...]:
        del target_hidden, k
        return ()

    with pytest.raises(ValueError, match="non-empty"):
        DFlashDrafter.for_synthetic(
            target_layer_ids=(),
            synthetic_emit=emit,
        )


# --- (γ) propose flow -----------------------------------------------------


def test_synthetic_propose_returns_emitter_output() -> None:
    """Happy path: prime, then ``propose(ctx, k)`` calls the emitter
    with the stored ``target_hidden`` and returns
    ``DraftTokens(token_ids=...)`` carrying the emitted ids verbatim."""
    target_layer_ids = (0, 7)
    hidden_size = 4
    captured_keys = tuple(i + 1 for i in target_layer_ids)
    expected_tokens = (101, 202, 303, 404)

    seen_target_hidden: list[mx.array] = []
    seen_k: list[int] = []

    def emit(target_hidden: mx.array, k: int) -> tuple[int, ...]:
        seen_target_hidden.append(target_hidden)
        seen_k.append(k)
        # Return the recorded oracle tokens; (γ)'s real test will pull
        # these from a spec-off run on Qwen3.5-0.8B at sub-unit (ε).
        return expected_tokens[:k]

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=target_layer_ids,
        synthetic_emit=emit,
    )
    drafter.prime(
        "req-0",
        _captured_dict(
            capture_keys=captured_keys, ctx_len=5, hidden_size=hidden_size
        ),
    )

    ctx = _make_ctx("req-0")
    drafts = drafter.propose(ctx, k=4)

    assert drafts.token_ids == expected_tokens
    assert len(seen_target_hidden) == 1
    # The emitter was handed the post-prime stored target_hidden of
    # shape (1, ctx_len, |L| * hidden_size).
    assert seen_target_hidden[0].shape == (
        1,
        5,
        len(target_layer_ids) * hidden_size,
    )
    assert seen_k == [4]


def test_synthetic_propose_raises_when_unprimed() -> None:
    """Engine ε guarantees priming; in tests the unprimed path must
    raise so wiring bugs surface immediately rather than silently
    proposing an empty block."""

    def emit(target_hidden: mx.array, k: int) -> tuple[int, ...]:
        del target_hidden, k
        return ()

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0,),
        synthetic_emit=emit,
    )
    ctx = _make_ctx("never-primed")
    with pytest.raises(KeyError, match="not primed"):
        drafter.propose(ctx, k=4)


def test_synthetic_propose_loud_fails_on_overflow() -> None:
    """The emitter promises ``len(token_ids) <= k``; any over-length
    return is a wiring bug that loud-fails rather than silently
    truncating."""

    def overflow_emit(target_hidden: mx.array, k: int) -> tuple[int, ...]:
        del target_hidden
        # Returns k + 2 tokens — wiring bug.
        return tuple(range(k + 2))

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0,),
        synthetic_emit=overflow_emit,
    )
    drafter.prime(
        "req-overflow",
        _captured_dict(capture_keys=(1,), ctx_len=2, hidden_size=4),
    )
    ctx = _make_ctx("req-overflow")
    with pytest.raises(ValueError, match="returned 6 tokens for k=4"):
        drafter.propose(ctx, k=4)


def test_synthetic_propose_accepts_short_returns() -> None:
    """Returning fewer than ``k`` tokens is legal — the engine
    interprets a short block as "drafter has no more proposal this
    cycle". Tested explicitly so the loud-fail branch above isn't
    accidentally widened to forbid it."""

    def short_emit(target_hidden: mx.array, k: int) -> tuple[int, ...]:
        del target_hidden, k
        return (101, 202)  # only 2 tokens

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0,),
        synthetic_emit=short_emit,
    )
    drafter.prime(
        "req-short",
        _captured_dict(capture_keys=(1,), ctx_len=2, hidden_size=4),
    )
    ctx = _make_ctx("req-short")
    drafts = drafter.propose(ctx, k=4)
    assert drafts.token_ids == (101, 202)


# --- (γ) update_target_hidden round-trip with propose ---------------------


def test_propose_after_update_sees_sliced_target_hidden() -> None:
    """F-1 state-machine invariant: after ``update_target_hidden(...,
    yielded_count)``, the next ``propose`` sees a ``target_hidden``
    whose axis-1 length is exactly ``1 + yielded_count``, regardless
    of how long the previous block's verify window was."""
    target_layer_ids = (0, 7)
    hidden_size = 4
    captured_keys = tuple(i + 1 for i in target_layer_ids)

    seen_shapes: list[tuple[int, ...]] = []

    def shape_recording_emit(
        target_hidden: mx.array, k: int
    ) -> tuple[int, ...]:
        seen_shapes.append(tuple(int(d) for d in target_hidden.shape))
        del k
        return ()

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=target_layer_ids,
        synthetic_emit=shape_recording_emit,
    )

    # Cycle 1: prime with prompt-len ctx = 6, propose, capture shape.
    drafter.prime(
        "req-0",
        _captured_dict(
            capture_keys=captured_keys, ctx_len=6, hidden_size=hidden_size
        ),
    )
    ctx = _make_ctx("req-0")
    drafter.propose(ctx, k=4)

    # Cycle 2: verify forward over k=4 tokens, yielded_count=3, propose
    # again — emitter must see shape (1, 4, |L| * hidden_size) where
    # 4 = 1 + yielded_count.
    drafter.update_target_hidden(
        "req-0",
        _captured_dict(
            capture_keys=captured_keys, ctx_len=4, hidden_size=hidden_size
        ),
        yielded_count=3,
    )
    drafter.propose(ctx, k=4)

    assert seen_shapes[0] == (1, 6, len(target_layer_ids) * hidden_size)
    assert seen_shapes[1] == (1, 4, len(target_layer_ids) * hidden_size)


# --- (γ) oracle-replay verifier-acceptance proof ---------------------------


def test_oracle_replay_design_lets_greedy_verify_accept_full_block() -> None:
    """End-to-end-shape proof at the verifier level: when the
    drafter emits exactly the target's argmax positions for the
    verify window, ``greedy_verify`` accepts the full block.

    This pins the (γ) oracle-replay design at silica's verifier
    surface without needing a real model. ``greedy_verify`` reads
    only the drafted tokens and the verify-forward logits; if the
    drafted tokens match the per-position argmax of the logits, all
    drafts are accepted. (ε) will exercise this end-to-end on cached
    Qwen3.5-0.8B by pre-recording the spec-off cycle-1 tokens.
    """
    vocab = 8
    k = 4

    # Pre-record an oracle: tokens that the target's argmax would
    # produce. Then construct synthetic logits whose argmax at each
    # position is the recorded token.
    oracle_tokens = (3, 5, 1, 7)
    logits_rows = []
    for tok in oracle_tokens:
        row = mx.full(shape=(vocab,), vals=-10.0, dtype=mx.float32)
        # Bump the chosen token's logit so it wins argmax.
        # ``mx.array`` does not support direct item assignment; build
        # via concatenation.
        before = mx.full(shape=(tok,), vals=-10.0, dtype=mx.float32)
        winner = mx.array([5.0], dtype=mx.float32)
        after = mx.full(shape=(vocab - tok - 1,), vals=-10.0, dtype=mx.float32)
        row = mx.concatenate([before, winner, after])
        logits_rows.append(row)
    verify_logits = mx.stack(logits_rows, axis=0)  # (k, vocab)

    drafted = mx.array(oracle_tokens, dtype=mx.uint32)

    # greedy_verify takes drafted (tuple/list of ints) and
    # verify_logits (k, vocab) and returns the count accepted. With
    # all positions matching, accept count == k.
    accepted = greedy_verify(oracle_tokens, verify_logits)
    assert accepted == k

    # Sanity: a single mismatched draft cuts acceptance early.
    bad_drafted = (3, 5, 99, 7)  # third token wrong
    accepted_bad = greedy_verify(bad_drafted, verify_logits)
    assert accepted_bad == 2

    # And the drafted mx.array variant is interface-compatible — kept
    # to confirm the typing surface (γ) tests will hand to the engine
    # at (ε).
    del drafted


# --- (γ) emitter type alias is exported -----------------------------------


def test_synthetic_emit_type_alias_is_exported() -> None:
    """The ``SyntheticEmit`` type alias is exported from the module so
    callers (and (ε) wiring) have a single source of truth for the
    callable shape."""
    assert SyntheticEmit is not None
