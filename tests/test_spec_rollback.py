"""Tests for D-021 step 5 sub-unit (i) — three-rollback paths together.

The speculative engine has three rollback owners that must converge on
the same committed boundary at the end of every cycle:

  1. **Draft-side** — ``DraftEngine.commit(ctx, yielded_count)`` rolls
     the drafter's own KV back to ``yielded_count`` accepted drafts.
     Detailed coverage lives in ``tests/test_draft_target_engine.py``;
     here we only assert that the engine drives ``commit`` with the
     correct count.
  2. **Target-side KV** — ``KVManager.rollback(req_id, n_reject)`` shrinks
     the target's attention KV. Per-implementation correctness is in
     ``tests/test_paged_kvcache.py`` and ``tests/test_simple_kvcache.py``;
     here we assert the engine fires it with the right ``n_reject``.
  3. **Recurrent state** — ``SpecRecurrentRollbackAdapter.rollback_state``
     restores the pre-draft snapshot and the engine replays
     ``decode_step_multi`` over the committed prefix. Wiring landed in
     ``tests/test_engine_spec_decode.py`` for individual paths; here we
     line up all three patterns side by side.

Three synthetic patterns (verify_k = 4 → γ = 3 throughout):

  - **Pattern A** (full accept). All γ drafts match target argmax;
    engine yields γ + 1 = 4 tokens this cycle. No KV rollback. For a
    recurrent adapter, ``commit_state`` evicts the snapshot. Drafter
    sees ``commit(yielded_count=γ)``; its own state is already at the
    committed boundary.
  - **Pattern B** (partial accept, ``accepted_len = 1``). 1 draft
    matches, 2 reject. Engine yields 1 + 1 = 2 tokens. Non-recurrent:
    ``kv.rollback(γ - yielded_count=2)``. Recurrent: ``kv.rollback(γ + 1)``
    (drop the full verify forward), ``rollback_state(2)``, replay over
    ``[anchor, accepted_draft]``. Drafter sees ``commit(1)``.
  - **Pattern C** (full reject). 0 drafts match. Engine yields 1 token
    (the bonus from ``verify_logits[0]``). Non-recurrent:
    ``kv.rollback(γ)``. Recurrent: ``kv.rollback(γ + 1)``,
    ``rollback_state(γ)``, replay over ``[anchor]``. Drafter sees
    ``commit(0)``.

**Why no real-model rollback row here.** OPENING §3 (i) sketched a
cached-Qwen3-0.6B parity row driven by an always-reject drafter,
reasoning that full reject leaves only the anchor in target KV per
cycle and so byte equality with spec-off should hold. Empirically it
does not: the surviving anchor's K / V was written by the BATCHED
verify forward over ``[anchor, drafts...]`` rather than by
``decode_step([anchor])``, and mlx-lm's batched matmul reduction order
differs from the single-token path in fp16. The post-rollback KV
contents diverge by epsilon, drift compounds across cycles, and an
argmax flips around the same index 5 boundary that bounds the (f)
parity test. The byte-equal-across-many-cycles claim was overly
optimistic on this hardware regime; spec correctness over long real-
model runs is validated through the (h) bench scenarios (acceptance
rates, throughput, generated-text spot checks) rather than against a
sequential reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.core.request import RequestState
from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.kvcache.manager import (
    BlockList,
    KVHandle,
    MemoryBudget,
    NullKVManager,
    PrefixHit,
)
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    StateDelta,
    Tokenizer,
)
from silica.models.capabilities import capabilities_from_attention_pattern
from silica.speculative.engine import DraftTokens

# --- shared fakes ----------------------------------------------------------

VOCAB = 1024
VERIFY_K = 4
GAMMA = VERIFY_K - 1


class _ScriptedTokenizer:
    vocab_size: int = VOCAB

    def encode(self, text: str) -> list[int]:
        return [] if text == "" else [10, 11]

    def decode(self, token_ids: Sequence[int]) -> str:
        del token_ids
        return ""


class _PatternAdapter:
    """Non-recurrent target adapter scripted by per-position argmax
    targets. Does NOT define the four ``SpecRecurrentRollbackAdapter``
    helpers; ``isinstance(_, SpecRecurrentRollbackAdapter)`` therefore
    returns False and the engine takes the non-recurrent rollback
    branch. The recurrent variant is ``_RecurrentPatternAdapter``
    below, which adds the four helpers on the class so the Protocol
    membership check succeeds at the class level (overriding
    ``__getattribute__`` on the instance is insufficient because the
    Protocol's ``__instancecheck__`` walks the class methods).
    """

    def __init__(
        self,
        *,
        prefill_argmax: int,
        verify_logits: Sequence[Sequence[int]],
    ) -> None:
        self._prefill_q: list[int] = [prefill_argmax]
        self._verify_q: list[list[int]] = [list(row) for row in verify_logits]

        self.verify_inputs_seen: list[list[int]] = []

        self.config = ModelConfig(
            model_name="pattern",
            num_layers=1,
            hidden_size=4,
            vocab_size=VOCAB,
        )
        self._tokenizer = _ScriptedTokenizer()

    # --- ModelAdapter surface ---

    def build(self, weight_provider: Any) -> Any:
        del weight_provider
        return object()

    def kv_layout(self) -> KVLayout:
        return KVLayout(
            num_layers=1, n_kv_heads=1, head_dim=4, dtype=mx.float16
        )

    def attention_pattern(self) -> AttentionPattern:
        return AttentionPattern(per_layer=(AttentionKind.GLOBAL,))

    def capabilities(self) -> Any:
        return capabilities_from_attention_pattern(self.attention_pattern())

    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        target = self._prefill_q.pop(0)
        return _one_hot(target), StateDelta()

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        # Not exercised by the spec branch, but Protocol-required.
        return _one_hot(0), StateDelta()

    def decode_step_multi(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        flat = [int(tokens[i].item()) for i in range(int(tokens.size))]
        self.verify_inputs_seen.append(flat)
        T = int(tokens.size)
        targets = self._verify_q.pop(0) if self._verify_q else [0] * T
        assert len(targets) == T, (
            f"verify_logits row has length {len(targets)} but verify_input "
            f"has {T} tokens"
        )
        rows = [_one_hot(t) for t in targets]
        return mx.stack(rows), StateDelta()


class _RecurrentPatternAdapter(_PatternAdapter):
    """Recurrent variant. Adds the four ``SpecRecurrentRollbackAdapter``
    helpers on the class so the runtime-checkable Protocol membership
    check returns True; bookkeeping-only (no real recurrent state)."""

    def __init__(
        self,
        *,
        prefill_argmax: int,
        verify_logits: Sequence[Sequence[int]],
    ) -> None:
        super().__init__(
            prefill_argmax=prefill_argmax,
            verify_logits=verify_logits,
        )
        self.snapshot_calls: list[str] = []
        self.commit_state_calls: list[tuple[str, int]] = []
        self.rollback_state_calls: list[tuple[str, int]] = []
        self.free_state_calls: list[str] = []

    # --- SpecRecurrentRollbackAdapter Protocol ---

    def snapshot_pre_draft_state(self, req_id: str) -> Any:
        self.snapshot_calls.append(req_id)
        return object()

    def commit_state(self, req_id: str, n_accepted: int) -> None:
        self.commit_state_calls.append((req_id, int(n_accepted)))

    def rollback_state(self, req_id: str, n_reject: int) -> None:
        self.rollback_state_calls.append((req_id, int(n_reject)))

    def free_state(self, req_id: str) -> None:
        self.free_state_calls.append(req_id)


def _make_pattern_adapter(
    *,
    recurrent: bool,
    prefill_argmax: int,
    verify_logits: Sequence[Sequence[int]],
) -> _PatternAdapter:
    cls = _RecurrentPatternAdapter if recurrent else _PatternAdapter
    return cls(prefill_argmax=prefill_argmax, verify_logits=verify_logits)


class _PatternDraftEngine:
    """Drafter that emits scripted token sequences and records every
    propose / commit / reset call.

    Distinguished from production ``DraftTargetEngine``: no real model,
    no own KV. The third rollback path (drafter-side) is delegated to
    ``DraftTargetEngine`` and tested in ``test_draft_target_engine.py``;
    here we assert the engine drives ``commit`` with the right
    ``yielded_count`` (proxy for the third path's correctness).
    """

    def __init__(self, proposals: Sequence[Sequence[int]]) -> None:
        self._queue: list[list[int]] = [list(row) for row in proposals]
        self.commit_calls: list[int] = []
        self.reset_calls: list[str | None] = []

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        if not self._queue:
            return DraftTokens(token_ids=())
        ids = self._queue.pop(0)
        return DraftTokens(
            token_ids=tuple(ids),
            draft_logprobs=tuple([-0.5] * len(ids)) if ids else None,
        )

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        self.commit_calls.append(int(accepted_len))

    def reset(self, req_id: str | None = None) -> None:
        self.reset_calls.append(req_id)


class _TrackedKVManager(NullKVManager):
    def __init__(self) -> None:
        super().__init__()
        self.rollback_calls: list[tuple[str, int]] = []

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        return super().reserve_for_prefill(req_id, token_ids)

    def rollback(self, req_id: str, n_reject: int) -> None:
        self.rollback_calls.append((req_id, int(n_reject)))
        super().rollback(req_id, n_reject)

    def get_computed_blocks(self, token_ids: Sequence[int]) -> PrefixHit:
        return PrefixHit()

    def budget(self) -> MemoryBudget:
        return MemoryBudget()


def _one_hot(target: int) -> mx.array:
    scores = [0.0] * VOCAB
    scores[target] = 5.0
    return mx.array(scores, dtype=mx.float32)


def _greedy(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


# --- Pattern A: full accept ------------------------------------------------


@pytest.mark.parametrize("recurrent", [False, True])
def test_pattern_a_full_accept(recurrent: bool) -> None:
    """All γ drafts accepted; engine yields γ + 1 tokens; no KV rollback;
    on recurrent path, ``commit_state`` evicts the snapshot."""
    # γ = 3. anchor = 100, drafts = [200, 201, 202], target argmax matches.
    adapter = _make_pattern_adapter(
        recurrent=recurrent,
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 250]],
    )
    kv = _TrackedKVManager()
    drafter = _PatternDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=VERIFY_K)
    out = list(engine.generate("hi", _greedy(max_tokens=VERIFY_K + 1)))

    # Yielded: anchor + γ drafts + bonus = γ + 2 = 5 tokens.
    assert out == [100, 200, 201, 202, 250]
    # Pattern A invariant: no rollback on either path.
    assert kv.rollback_calls == []
    # Drafter sees commit with full γ.
    assert drafter.commit_calls == [GAMMA]
    # Single verify forward; no replay.
    assert len(adapter.verify_inputs_seen) == 1
    assert adapter.verify_inputs_seen[0] == [100, 200, 201, 202]

    if recurrent:
        assert isinstance(adapter, _RecurrentPatternAdapter)
        assert adapter.snapshot_calls == ["req-0"]
        assert adapter.commit_state_calls == [("req-0", GAMMA)]
        assert adapter.rollback_state_calls == []
        assert adapter.free_state_calls == ["req-0"]


# --- Pattern B: partial accept ---------------------------------------------


@pytest.mark.parametrize("recurrent", [False, True])
def test_pattern_b_partial_accept(recurrent: bool) -> None:
    """Partial accept (1 of 3 drafts). Engine yields 1 draft + 1 bonus.
    Non-recurrent: kv.rollback(γ - yielded = 2). Recurrent:
    kv.rollback(γ + 1 = 4) + rollback_state(2) + replay over
    [anchor, accepted_draft]."""
    # anchor = 100, drafts = [200, 201, 202]; verify argmax = [200, 999, 0, 0]
    # → drafts[0] accepts, drafts[1] rejects, bonus = 999.
    verify_rows: list[list[int]] = [[200, 999, 0, 0]]
    if recurrent:
        # Replay over [anchor, accepted_draft] = 2 tokens.
        verify_rows.append([0, 0])
    adapter = _make_pattern_adapter(
        recurrent=recurrent,
        prefill_argmax=100,
        verify_logits=verify_rows,
    )
    kv = _TrackedKVManager()
    drafter = _PatternDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=VERIFY_K)
    # max_tokens = 3 stops cleanly after the bonus (anchor + 1 draft + bonus).
    out = list(engine.generate("hi", _greedy(max_tokens=3)))

    assert out == [100, 200, 999]
    # Drafter commits yielded_count = 1.
    assert drafter.commit_calls == [1]

    if recurrent:
        # Recurrent path: full verify trim + rollback_state + replay.
        assert isinstance(adapter, _RecurrentPatternAdapter)
        assert kv.rollback_calls == [("req-0", GAMMA + 1)]
        assert adapter.snapshot_calls == ["req-0"]
        assert adapter.rollback_state_calls == [("req-0", GAMMA - 1)]
        assert adapter.commit_state_calls == []
        # Two decode_step_multi calls: verify + replay.
        assert len(adapter.verify_inputs_seen) == 2
        assert adapter.verify_inputs_seen[0] == [100, 200, 201, 202]
        assert adapter.verify_inputs_seen[1] == [100, 200]
        assert adapter.free_state_calls == ["req-0"]
    else:
        # Non-recurrent path: single rollback by un_committed = γ - 1 = 2.
        assert kv.rollback_calls == [("req-0", GAMMA - 1)]
        # No replay.
        assert len(adapter.verify_inputs_seen) == 1


# --- Pattern C: full reject ------------------------------------------------


@pytest.mark.parametrize("recurrent", [False, True])
def test_pattern_c_full_reject(recurrent: bool) -> None:
    """Full reject. Engine yields 1 bonus token (verify_logits[0] argmax).
    Non-recurrent: kv.rollback(γ). Recurrent: kv.rollback(γ + 1),
    rollback_state(γ), replay over [anchor]."""
    # drafts = [200, 201, 202]; verify argmax = [777, 0, 0, 0] → reject draft[0].
    verify_rows: list[list[int]] = [[777, 0, 0, 0]]
    if recurrent:
        # Replay over [anchor] only.
        verify_rows.append([0])
    adapter = _make_pattern_adapter(
        recurrent=recurrent,
        prefill_argmax=100,
        verify_logits=verify_rows,
    )
    kv = _TrackedKVManager()
    drafter = _PatternDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=VERIFY_K)
    out = list(engine.generate("hi", _greedy(max_tokens=2)))

    assert out == [100, 777]
    # Drafter commits yielded_count = 0.
    assert drafter.commit_calls == [0]

    if recurrent:
        assert isinstance(adapter, _RecurrentPatternAdapter)
        assert kv.rollback_calls == [("req-0", GAMMA + 1)]
        assert adapter.snapshot_calls == ["req-0"]
        assert adapter.rollback_state_calls == [("req-0", GAMMA)]
        assert adapter.commit_state_calls == []
        assert len(adapter.verify_inputs_seen) == 2
        assert adapter.verify_inputs_seen[0] == [100, 200, 201, 202]
        assert adapter.verify_inputs_seen[1] == [100]
        assert adapter.free_state_calls == ["req-0"]
    else:
        assert kv.rollback_calls == [("req-0", GAMMA)]
        assert len(adapter.verify_inputs_seen) == 1


