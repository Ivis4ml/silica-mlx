"""Tests for ``silica.speculative.draft_target.DraftTargetEngine``.

D-021 step 5 sub-unit (a). Synthetic-adapter unit tests: no real model load.
The fake adapter is scripted to return logits whose argmax is a deterministic
function of its KV depth, so the test can predict exactly which token ids the
draft will produce on each forward and assert against them.

Test coverage:
  - I-5 ``DraftEngine`` Protocol conformance.
  - Cycle-0 prefill path: draft KV grows from 0 to ``len(prompt + anchor)``
    plus γ; γ drafts produced with non-None logprobs.
  - Catch-up path: post-commit, propose feeds the new tail (the engine's
    bonus token) before drafting, restoring sync without re-prefilling.
  - Partial-accept commit: KV rolls back by ``γ - accepted_len``; cached
    tail logits are cleared so the next propose's catch-up runs.
  - Full-accept commit: no rollback fires; cached tail logits survive.
  - Empty draft (``k=0``) path: no forward issued, ``last_propose_count``
    stays 0, ``commit`` no-ops.
  - Defensive raises: ``accepted_len > γ``, draft running ahead of target.

The fake adapter is intentionally minimal — it does not implement
``snapshot_recurrent_state`` / ``restore_recurrent_state``, which models the
plain-KV draft case (Qwen3-0.6B). Hybrid-recurrent rollback joins at
sub-unit (e); sub-unit (f) covers the joined real-model path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.core.request import Request, RequestState
from silica.core.sampling import SamplingParams
from silica.kvcache.manager import BlockList, KVHandle, MemoryBudget, PrefixHit
from silica.models.adapter import StateDelta
from silica.speculative.draft_target import DraftTargetEngine
from silica.speculative.engine import DraftEngine, DraftTokens

# --- fake adapter + fake KV --------------------------------------------------


class _FakeKV:
    """Minimal SimpleKVCache-compatible stand-in for unit tests.

    Tracks owner + a logical KV depth that ``rollback`` shrinks. Does not
    model mlx-lm's heterogeneous per-layer cache list; the fake adapter
    below does not consult ``cache_list``.
    """

    block_size: int = 256

    def __init__(self) -> None:
        self._owner: str | None = None
        self.depth: int = 0
        self.rollback_calls: list[tuple[str, int]] = []

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        if self._owner is not None and self._owner != req_id:
            raise ValueError(
                f"single-request cache already owned by {self._owner!r}"
            )
        self._owner = req_id
        return BlockList()

    def append_slot(self, req_id: str, n: int) -> BlockList:
        return BlockList()

    def commit(self, req_id: str, n_accepted: int) -> None:
        return None

    def rollback(self, req_id: str, n_reject: int) -> None:
        if self._owner != req_id:
            raise ValueError("rollback owner mismatch")
        self.rollback_calls.append((req_id, n_reject))
        self.depth = max(0, self.depth - n_reject)

    def free(self, req_id: str) -> None:
        if self._owner == req_id:
            self._owner = None

    def get_computed_blocks(self, token_ids: Sequence[int]) -> PrefixHit:
        return PrefixHit()

    def available_blocks(self) -> int:
        return 0

    def budget(self) -> MemoryBudget:
        return MemoryBudget()


class _FakeAdapter:
    """Adapter that returns logits whose argmax is ``(kv_depth + 1) mod V``.

    Logits are derived from the fake KV's depth — not from a side counter
    on the adapter — so a ``rollback`` on the KV (which a correct
    ``DraftTargetEngine.reset`` must issue before ``free``) is visible to
    the next forward, just as it would be in a real adapter where the
    layers' state lives in the underlying KV cache list.
    """

    VOCAB = 64

    def __init__(self, fake_kv: _FakeKV) -> None:
        self._kv = fake_kv
        # Call log for assertions. Each entry is (op, n_tokens, depth_after).
        self.calls: list[tuple[str, int, int]] = []

    def _logits_for_depth(self, depth: int) -> mx.array:
        # One-hot at (depth + 1) % V — a deterministic, depth-keyed argmax.
        # Setting the rest to zero makes log_softmax well-defined and gives
        # a non-trivial logprob.
        target = (depth + 1) % self.VOCAB
        scores = [0.0] * self.VOCAB
        scores[target] = 5.0  # tall enough that argmax is unambiguous
        return mx.array(scores, dtype=mx.float32)

    # --- ModelAdapter surface (only the bits the draft engine consumes) ---

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        n = int(tokens.size)
        self._kv.depth += n
        self.calls.append(("prefill", n, self._kv.depth))
        return self._logits_for_depth(self._kv.depth), StateDelta()

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        n = int(token.size)
        self._kv.depth += n
        self.calls.append(("decode_step", n, self._kv.depth))
        return self._logits_for_depth(self._kv.depth), StateDelta()

    # The remaining ModelAdapter methods are not exercised by the draft
    # engine (it only uses prefill / decode_step). Stubs raise to surface
    # any accidental dependency.

    def kv_layout(self) -> Any:
        raise NotImplementedError("fake adapter")

    def attention_pattern(self) -> Any:
        raise NotImplementedError("fake adapter")

    def tokenizer(self) -> Any:
        raise NotImplementedError("fake adapter")

    def capabilities(self) -> Any:
        raise NotImplementedError("fake adapter")


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def kv() -> _FakeKV:
    return _FakeKV()


@pytest.fixture
def adapter(kv: _FakeKV) -> _FakeAdapter:
    return _FakeAdapter(kv)


@pytest.fixture
def engine(adapter: _FakeAdapter, kv: _FakeKV) -> DraftTargetEngine:
    return DraftTargetEngine(adapter, kv, req_id="test-draft")  # type: ignore[arg-type]


def _ctx_with(prompt: tuple[int, ...], output: tuple[int, ...]) -> RequestState:
    req = Request(
        prompt="",
        sampling_params=SamplingParams(),
        token_ids=prompt,
    )
    state = RequestState(request=req)
    state.output_token_ids = list(output)
    return state


# --- I-5 Protocol conformance -----------------------------------------------


def test_satisfies_draft_engine_protocol(engine: DraftTargetEngine) -> None:
    assert isinstance(engine, DraftEngine)


# --- cycle-0 prefill path ---------------------------------------------------


def test_propose_cycle_zero_prefills_and_drafts(
    engine: DraftTargetEngine, adapter: _FakeAdapter
) -> None:
    # Prompt of length 5, anchor already in output_token_ids → target
    # committed length 6 at propose entry.
    ctx = _ctx_with(prompt=(10, 11, 12, 13, 14), output=(99,))
    drafts = engine.propose(ctx, k=3)

    assert isinstance(drafts, DraftTokens)
    assert len(drafts.token_ids) == 3
    assert drafts.draft_logprobs is not None
    assert len(drafts.draft_logprobs) == 3

    # Cycle-0 issues exactly one prefill for the 6 committed tokens, plus
    # γ=3 single-token decode_step forwards.
    ops = [op for op, _, _ in adapter.calls]
    assert ops == ["prefill", "decode_step", "decode_step", "decode_step"]
    n_tokens = [n for _, n, _ in adapter.calls]
    assert n_tokens == [6, 1, 1, 1]

    # Draft KV ended at 6 + 3 = 9.
    assert engine.draft_kv_pos == 9
    assert engine.last_propose_count == 3


def test_propose_cycle_zero_argmax_chain_is_predictable(
    engine: DraftTargetEngine, adapter: _FakeAdapter
) -> None:
    # _FakeAdapter.argmax(depth) = (depth + 1) mod V.
    # Cycle-0: prefill→depth=6 → first draft = 7; decode_step(7)→depth=7
    # → second draft = 8; decode_step(8)→depth=8 → third draft = 9.
    ctx = _ctx_with(prompt=(10, 11, 12, 13, 14), output=(99,))
    drafts = engine.propose(ctx, k=3)
    assert drafts.token_ids == (7, 8, 9)


# --- empty draft path -------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, -7])
def test_propose_with_nonpositive_k_returns_empty_no_forward(
    engine: DraftTargetEngine, adapter: _FakeAdapter, k: int
) -> None:
    ctx = _ctx_with(prompt=(10, 11), output=(99,))
    drafts = engine.propose(ctx, k=k)
    assert drafts.token_ids == ()
    assert drafts.draft_logprobs is None
    assert adapter.calls == []
    assert engine.last_propose_count == 0


def test_commit_with_no_active_propose_is_noop(
    engine: DraftTargetEngine, kv: _FakeKV
) -> None:
    ctx = _ctx_with(prompt=(10,), output=(99,))
    # No propose has been called yet — commit must accept any accepted_len
    # without raising and without issuing a rollback call.
    engine.commit(ctx, accepted_len=0)
    engine.commit(ctx, accepted_len=4)
    assert kv.rollback_calls == []


# --- partial-accept commit + catch-up next cycle ---------------------------


def test_partial_accept_rolls_back_kv_and_clears_cached_logits(
    engine: DraftTargetEngine, adapter: _FakeAdapter, kv: _FakeKV
) -> None:
    ctx = _ctx_with(prompt=(10, 11), output=(99,))
    drafts = engine.propose(ctx, k=4)
    assert drafts.token_ids == (4, 5, 6, 7)  # depths 3,4,5,6 → +1
    assert engine.draft_kv_pos == 7  # 3 prefill + 4 drafts

    # Engine "accepts" 2 of 4 drafts, rejects 2. Bonus is something the
    # engine sampled from verify_logits[2] — append it to ctx.output.
    engine.commit(ctx, accepted_len=2)
    assert kv.rollback_calls == [("test-draft", 2)]  # γ - accepted_len = 2
    assert engine.draft_kv_pos == 5  # 7 - 2 rejected
    assert engine.last_propose_count == 0  # commit resets the counter

    # Engine commits accepted drafts + bonus into ctx.output_token_ids.
    # output is now (99, 4, 5, BONUS) — total target_committed = 2 + 4 = 6.
    ctx.output_token_ids.extend([4, 5, 234])  # bonus = 234

    adapter.calls.clear()
    drafts2 = engine.propose(ctx, k=3)
    # Catch-up branch: target_committed = 6, draft_kv_pos was 5, so feed 1
    # bonus token via decode_step before γ=3 drafts.
    ops = [op for op, _, _ in adapter.calls]
    assert ops == [
        "decode_step",  # catch-up: bonus
        "decode_step",  # draft 1
        "decode_step",  # draft 2
        "decode_step",  # draft 3
    ]
    assert len(drafts2.token_ids) == 3
    assert engine.draft_kv_pos == 9  # 5 + 1 catch-up + 3 drafts


# --- full-accept commit (no rollback) ---------------------------------------


def test_full_accept_does_not_rollback(
    engine: DraftTargetEngine, kv: _FakeKV
) -> None:
    ctx = _ctx_with(prompt=(10,), output=(99,))
    engine.propose(ctx, k=3)
    engine.commit(ctx, accepted_len=3)  # all γ accepted
    assert kv.rollback_calls == []
    assert engine.last_propose_count == 0


def test_full_accept_then_propose_consumes_only_bonus(
    engine: DraftTargetEngine, adapter: _FakeAdapter, kv: _FakeKV
) -> None:
    ctx = _ctx_with(prompt=(10,), output=(99,))
    engine.propose(ctx, k=3)  # full propose path
    engine.commit(ctx, accepted_len=3)  # full accept

    # Engine yields 4 tokens (3 drafts + 1 bonus). ctx.output gains those.
    drafts1_ids = (2, 3, 4)  # by the depth+1 rule starting at depth=2
    bonus = 5  # placeholder; actual bonus is the engine's call
    ctx.output_token_ids.extend(list(drafts1_ids) + [bonus])

    adapter.calls.clear()
    engine.propose(ctx, k=2)
    # draft_kv_pos was 5 (prefill 2 + 3 drafts). target_committed grew by 4
    # (3 accepted drafts + bonus) to 6 = prompt(1) + output(5). Catch-up
    # feeds 1 bonus token; no prefill.
    ops = [op for op, _, _ in adapter.calls]
    assert "prefill" not in ops
    # 1 catch-up + 2 drafts = 3 decode_steps.
    assert ops == ["decode_step"] * 3


# --- defensive raises -------------------------------------------------------


def test_propose_then_invalid_accepted_len_raises(
    engine: DraftTargetEngine,
) -> None:
    ctx = _ctx_with(prompt=(10,), output=(99,))
    engine.propose(ctx, k=3)
    with pytest.raises(ValueError, match=r"accepted_len=4"):
        engine.commit(ctx, accepted_len=4)
    with pytest.raises(ValueError, match=r"accepted_len=-1"):
        engine.commit(ctx, accepted_len=-1)


def test_propose_with_target_behind_raises(
    engine: DraftTargetEngine,
) -> None:
    # First cycle puts draft_kv_pos at 5 (1 prefill + 1 anchor + γ=3 drafts).
    ctx = _ctx_with(prompt=(10,), output=(99,))
    engine.propose(ctx, k=3)
    # Pretend the target somehow has fewer committed tokens than the draft
    # has consumed — should be caught loud, not silently mis-aligned.
    ctx.output_token_ids = []  # target_committed shrinks to len(prompt) = 1
    with pytest.raises(RuntimeError, match=r"draft has run ahead"):
        engine.propose(ctx, k=2)


def test_propose_without_any_committed_tokens_raises(
    engine: DraftTargetEngine,
) -> None:
    ctx = _ctx_with(prompt=(), output=())
    with pytest.raises(RuntimeError, match=r"empty target_committed"):
        engine.propose(ctx, k=2)


# --- reset lifecycle --------------------------------------------------------


def test_reset_releases_kv_and_trims_underlying_cache(
    engine: DraftTargetEngine, kv: _FakeKV
) -> None:
    # cycle-0: prefill (prompt 1 + anchor 1 = 2 tokens) + γ=2 drafts.
    # Underlying KV depth ends at 4; engine bookkeeping at 4.
    ctx = _ctx_with(prompt=(10,), output=(99,))
    engine.propose(ctx, k=2)
    assert kv._owner == "test-draft"
    assert engine.draft_kv_pos == 4
    assert kv.depth == 4
    rollback_calls_before_reset = list(kv.rollback_calls)

    engine.reset()

    # Owner released AND underlying cache fully trimmed — without the
    # rollback, the next request's cycle-0 prefill would write on top of
    # stale KV state and produce contaminated draft logits.
    assert kv._owner is None
    assert kv.depth == 0
    assert engine.draft_kv_pos == 0
    assert engine.last_propose_count == 0
    # Reset issued exactly one rollback for the full draft_kv_pos.
    new_rollback = kv.rollback_calls[len(rollback_calls_before_reset):]
    assert new_rollback == [("test-draft", 4)]


def test_reset_without_prior_propose_is_safe(
    engine: DraftTargetEngine, kv: _FakeKV
) -> None:
    # No propose has run; reset must not blow up and must not issue a
    # rollback (no ownership held, depth is 0).
    engine.reset()
    assert kv._owner is None
    assert kv.depth == 0
    assert kv.rollback_calls == []
    assert engine.draft_kv_pos == 0


def test_reset_then_new_request_does_not_carry_old_kv(
    engine: DraftTargetEngine, adapter: _FakeAdapter, kv: _FakeKV
) -> None:
    # First request: cycle-0 prefill at depth 0 → first draft argmax = 3
    # (prompt 1 + anchor 1 → depth 2 → +1 = 3).
    ctx1 = _ctx_with(prompt=(10,), output=(99,))
    drafts1 = engine.propose(ctx1, k=1)
    assert drafts1.token_ids == (3,)
    engine.reset()

    # New request, same engine. After reset, draft_kv_pos is 0 again and
    # the underlying cache was trimmed back to depth 0 — so the next
    # cycle-0 prefill must restart from depth 0, producing the same
    # depth-keyed argmax as the first request would. If reset failed to
    # trim the underlying cache, depth would still be 4 from the first
    # request and the new draft would be argmax (4 + 2 + 1) = 7, not 3.
    adapter.calls.clear()
    ctx2 = _ctx_with(prompt=(20,), output=(88,))
    drafts2 = engine.propose(ctx2, k=1)
    ops = [op for op, _, _ in adapter.calls]
    assert ops[0] == "prefill"  # cycle-0 path entered again
    # depth_after_prefill = 0 (post-reset) + 2 = 2, then draft = 3.
    assert drafts2.token_ids == (3,)
