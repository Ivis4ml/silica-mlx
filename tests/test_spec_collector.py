"""Tests for D-021 step 5 sub-unit (g) — SpecMetricCollector emission.

Covers:
  - Collector unit semantics: accumulators, mean computations, edge
    cases (no proposes, self-spec flag, default parity status).
  - ``materialize`` output passes ``validate_speculative_metrics``.
  - ``Engine.generate`` emits to the collector across the three
    rollback patterns (full accept / partial / full reject) using
    the same scripted fixtures the (i) tests use.
  - Spec-off (NoopDraftEngine) leaves an attached collector untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.bench.spec_collector import SpecMetricCollector
from silica.bench.spec_metrics import (
    SPECULATIVE_METRIC_FIELDS,
    QualityParityStatus,
    validate_speculative_metrics,
)
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
from silica.speculative.engine import DraftTokens, NoopDraftEngine

VOCAB = 1024
VERIFY_K = 4
GAMMA = VERIFY_K - 1


# --- collector unit tests --------------------------------------------------


def test_collector_default_materialize_passes_validator() -> None:
    """Fresh collector materialises a valid (zero-everywhere) dict.
    ``quality_parity_status`` defaults to ``NOT_TESTED``."""
    collector = SpecMetricCollector()
    out = collector.materialize()
    assert set(out.keys()) == set(SPECULATIVE_METRIC_FIELDS)
    assert out["accept_rate"] == 0.0
    assert out["verify_cost_ms"] == 0.0
    assert out["draft_cost_ms"] == 0.0
    assert out["tokens_per_target_forward"] == 0.0
    assert out["rollback_count"] == 0
    assert out["tree_node_visits"] == 0
    assert out["quality_parity_status"] == QualityParityStatus.NOT_TESTED
    assert validate_speculative_metrics(out) == []


def test_collector_accumulates_propose_and_verify() -> None:
    collector = SpecMetricCollector()
    collector.record_propose(draft_count=3, elapsed_ms=2.0)
    collector.record_verify(accepted_len=2, yielded_count=2, elapsed_ms=10.0)
    collector.record_bonus()
    collector.record_propose(draft_count=3, elapsed_ms=4.0)
    collector.record_verify(accepted_len=3, yielded_count=3, elapsed_ms=12.0)
    collector.record_bonus()
    out = collector.materialize()
    # 5 / 6 drafts accepted.
    assert out["accept_rate"] == pytest.approx(5 / 6)
    # Mean verify cost = (10 + 12) / 2 = 11 ms.
    assert out["verify_cost_ms"] == pytest.approx(11.0)
    # Mean draft cost = (2 + 4) / 2 = 3 ms.
    assert out["draft_cost_ms"] == pytest.approx(3.0)
    # tokens_per_target_forward counts drafts + bonuses:
    # (2 yielded + 3 yielded + 2 bonuses) / 2 forwards = 7 / 2 = 3.5.
    assert out["tokens_per_target_forward"] == pytest.approx(3.5)
    assert validate_speculative_metrics(out) == []


def test_collector_bonus_only_no_drafts_accepted() -> None:
    """Full-reject scenario: 0 accepted drafts but bonus still emitted →
    tokens_per_target_forward = 1.0, not 0.0."""
    collector = SpecMetricCollector()
    collector.record_propose(draft_count=3, elapsed_ms=2.0)
    collector.record_verify(accepted_len=0, yielded_count=0, elapsed_ms=10.0)
    collector.record_bonus()
    out = collector.materialize()
    assert out["accept_rate"] == pytest.approx(0.0)
    assert out["tokens_per_target_forward"] == pytest.approx(1.0)


def test_collector_bonus_suppressed_no_emit() -> None:
    """When ``max_tokens`` / stop-token cut the cycle short the engine
    skips the bonus path; ``record_bonus`` is not called and the
    metric reflects only the yielded drafts."""
    collector = SpecMetricCollector()
    collector.record_propose(draft_count=3, elapsed_ms=2.0)
    collector.record_verify(accepted_len=2, yielded_count=2, elapsed_ms=10.0)
    # No record_bonus — bonus path was skipped (e.g. max_tokens cut).
    out = collector.materialize()
    assert out["tokens_per_target_forward"] == pytest.approx(2.0)


def test_collector_self_spec_zeros_draft_cost() -> None:
    collector = SpecMetricCollector()
    collector.set_self_spec()
    collector.record_propose(draft_count=3, elapsed_ms=1.0)
    collector.record_verify(accepted_len=3, yielded_count=3, elapsed_ms=8.0)
    out = collector.materialize()
    assert out["draft_cost_ms"] == 0.0
    # Verify cost still attributes normally.
    assert out["verify_cost_ms"] == pytest.approx(8.0)


def test_collector_rollback_count() -> None:
    collector = SpecMetricCollector()
    collector.record_rollback()
    collector.record_rollback()
    out = collector.materialize()
    assert out["rollback_count"] == 2


def test_collector_record_parity_overrides_default() -> None:
    collector = SpecMetricCollector()
    collector.record_parity(QualityParityStatus.PARITY)
    out = collector.materialize()
    assert out["quality_parity_status"] == QualityParityStatus.PARITY
    assert validate_speculative_metrics(out) == []


# --- engine emission tests --------------------------------------------------


class _ScriptedTokenizer:
    vocab_size: int = VOCAB

    def encode(self, text: str) -> list[int]:
        return [] if text == "" else [10, 11]

    def decode(self, token_ids: Sequence[int]) -> str:
        del token_ids
        return ""


def _one_hot(target: int) -> mx.array:
    scores = [0.0] * VOCAB
    scores[target] = 5.0
    return mx.array(scores, dtype=mx.float32)


class _ScriptedAdapter:
    """Plain (non-recurrent) target adapter scripted by per-call argmax."""

    def __init__(
        self,
        *,
        prefill_argmax: int,
        decode_argmaxes: Sequence[int] = (),
        verify_logits: Sequence[Sequence[int]] = (),
    ) -> None:
        self._prefill_q: list[int] = [prefill_argmax]
        self._decode_q: list[int] = list(decode_argmaxes)
        self._verify_q: list[list[int]] = [list(row) for row in verify_logits]
        self.config = ModelConfig(
            model_name="scripted",
            num_layers=1,
            hidden_size=4,
            vocab_size=VOCAB,
        )
        self._tokenizer = _ScriptedTokenizer()

    def build(self, weight_provider: Any) -> Any:
        del weight_provider
        return object()

    def kv_layout(self) -> KVLayout:
        return KVLayout(num_layers=1, n_kv_heads=1, head_dim=4, dtype=mx.float16)

    def attention_pattern(self) -> AttentionPattern:
        return AttentionPattern(per_layer=(AttentionKind.GLOBAL,))

    def capabilities(self) -> Any:
        return capabilities_from_attention_pattern(self.attention_pattern())

    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        return _one_hot(self._prefill_q.pop(0)), StateDelta()

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        target = self._decode_q.pop(0) if self._decode_q else 0
        return _one_hot(target), StateDelta()

    def decode_step_multi(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        T = int(tokens.size)
        targets = self._verify_q.pop(0) if self._verify_q else [0] * T
        rows = [_one_hot(t) for t in targets]
        return mx.stack(rows), StateDelta()


class _ScriptedDraftEngine:
    def __init__(self, proposals: Sequence[Sequence[int]] = ()) -> None:
        self._queue: list[list[int]] = [list(row) for row in proposals]

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        if not self._queue:
            return DraftTokens(token_ids=())
        ids = self._queue.pop(0)
        return DraftTokens(
            token_ids=tuple(ids),
            draft_logprobs=tuple([-0.5] * len(ids)) if ids else None,
        )

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        del accepted_len

    def reset(self, req_id: str | None = None) -> None:
        del req_id


class _NullKV(NullKVManager):
    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        return super().reserve_for_prefill(req_id, token_ids)

    def get_computed_blocks(self, token_ids: Sequence[int]) -> PrefixHit:
        return PrefixHit()

    def budget(self) -> MemoryBudget:
        return MemoryBudget()


def _greedy(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def test_engine_emits_for_full_accept_pattern() -> None:
    """Pattern A: γ drafts proposed, all accepted, no rollback."""
    adapter = _ScriptedAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 250]],
    )
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    collector = SpecMetricCollector()
    engine = Engine(
        adapter,
        _NullKV(),
        draft_engine=drafter,
        verify_k=VERIFY_K,
        spec_collector=collector,
    )
    out = list(engine.generate("hi", _greedy(max_tokens=VERIFY_K + 1)))
    assert out == [100, 200, 201, 202, 250]

    assert collector.proposed_drafts == GAMMA
    assert collector.accepted_drafts == GAMMA
    assert collector.yielded_drafts == GAMMA
    assert collector.bonus_tokens == 1
    assert collector.target_forward_count == 1
    assert collector.draft_forward_count == 1
    assert collector.rollback_count == 0

    out_md = collector.materialize()
    assert out_md["accept_rate"] == pytest.approx(1.0)
    # γ accepted drafts + 1 bonus per verify forward.
    assert out_md["tokens_per_target_forward"] == pytest.approx(
        float(GAMMA + 1)
    )
    assert out_md["rollback_count"] == 0
    assert validate_speculative_metrics(out_md) == []


def test_engine_emits_for_partial_accept_pattern() -> None:
    """Pattern B: 1 of γ drafts accepted; rollback fires once."""
    adapter = _ScriptedAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 999, 0, 0]],
    )
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    collector = SpecMetricCollector()
    engine = Engine(
        adapter,
        _NullKV(),
        draft_engine=drafter,
        verify_k=VERIFY_K,
        spec_collector=collector,
    )
    out = list(engine.generate("hi", _greedy(max_tokens=3)))
    assert out == [100, 200, 999]

    assert collector.proposed_drafts == GAMMA
    assert collector.accepted_drafts == 1
    assert collector.yielded_drafts == 1
    assert collector.bonus_tokens == 1
    assert collector.target_forward_count == 1
    assert collector.rollback_count == 1
    out_md = collector.materialize()
    # 1 yielded draft + 1 bonus = 2 tokens per verify forward.
    assert out_md["tokens_per_target_forward"] == pytest.approx(2.0)


def test_engine_emits_for_full_reject_pattern() -> None:
    """Pattern C: 0 drafts accepted; rollback fires once."""
    adapter = _ScriptedAdapter(
        prefill_argmax=100,
        verify_logits=[[777, 0, 0, 0]],
    )
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    collector = SpecMetricCollector()
    engine = Engine(
        adapter,
        _NullKV(),
        draft_engine=drafter,
        verify_k=VERIFY_K,
        spec_collector=collector,
    )
    out = list(engine.generate("hi", _greedy(max_tokens=2)))
    assert out == [100, 777]

    assert collector.proposed_drafts == GAMMA
    assert collector.accepted_drafts == 0
    assert collector.yielded_drafts == 0
    assert collector.bonus_tokens == 1
    assert collector.target_forward_count == 1
    assert collector.rollback_count == 1
    out_md = collector.materialize()
    assert out_md["accept_rate"] == 0.0
    # 0 drafts yielded + 1 bonus = 1.0, not 0.0 — the verify forward
    # still produced one emitted token despite full reject.
    assert out_md["tokens_per_target_forward"] == pytest.approx(1.0)


def test_engine_suppresses_bonus_record_when_max_tokens_cuts_cycle() -> None:
    """``max_tokens == prefill_yield + γ`` lands ``n`` exactly on the cap
    after the yield loop's natural exit. The post-loop guard sets
    ``stop_hit=True`` and the bonus path is skipped — collector must
    NOT increment ``bonus_tokens``."""
    adapter = _ScriptedAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 250]],
    )
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    collector = SpecMetricCollector()
    engine = Engine(
        adapter,
        _NullKV(),
        draft_engine=drafter,
        verify_k=VERIFY_K,
        spec_collector=collector,
    )
    # max_tokens = γ + 1 (4): cycle 1 yields 3 drafts, post-loop guard
    # sets stop_hit, bonus suppressed.
    out = list(engine.generate("hi", _greedy(max_tokens=GAMMA + 1)))
    assert out == [100, 200, 201, 202]
    assert collector.yielded_drafts == GAMMA
    assert collector.bonus_tokens == 0
    out_md = collector.materialize()
    # γ yielded drafts + 0 bonus = γ tokens per verify forward.
    assert out_md["tokens_per_target_forward"] == pytest.approx(float(GAMMA))


def test_engine_spec_off_does_not_emit() -> None:
    """Default ``NoopDraftEngine`` never enters the spec branch, so an
    attached collector stays at its zero-initialised state."""
    adapter = _ScriptedAdapter(
        prefill_argmax=5,
        decode_argmaxes=[6, 7],
    )
    collector = SpecMetricCollector()
    engine = Engine(
        adapter,
        _NullKV(),
        draft_engine=NoopDraftEngine(),
        spec_collector=collector,
    )
    out = list(engine.generate("hi", _greedy(max_tokens=3)))
    assert out == [5, 6, 7]
    assert collector.proposed_drafts == 0
    assert collector.target_forward_count == 0
    assert collector.draft_forward_count == 0
    assert collector.rollback_count == 0
    assert collector.materialize()["accept_rate"] == 0.0


def test_engine_accumulates_across_cycles() -> None:
    """Two cycles, full accept then partial; collector sums correctly."""
    # Cycle 1: γ=3 drafts all accepted; bonus = 250.
    # Cycle 2: anchor = 250; γ=3 drafts with 1 accepted (draft[1] mismatch).
    #   verify argmax = [300, 999, 0, 0] → accepted_len=1, bonus=999.
    adapter = _ScriptedAdapter(
        prefill_argmax=100,
        verify_logits=[
            [200, 201, 202, 250],   # cycle 1 (full accept)
            [300, 999, 0, 0],       # cycle 2 (partial accept)
        ],
    )
    drafter = _ScriptedDraftEngine(
        proposals=[[200, 201, 202], [300, 301, 302]]
    )
    collector = SpecMetricCollector()
    params = SamplingParams(
        temperature=0.0, max_tokens=10, stop_token_ids=(999,)
    )
    engine = Engine(
        adapter,
        _NullKV(),
        draft_engine=drafter,
        verify_k=VERIFY_K,
        spec_collector=collector,
    )
    out = list(engine.generate("hi", params))
    # Cycle 1: anchor 100, drafts 200/201/202, bonus 250.
    # Cycle 2: anchor 250, draft 300, bonus 999 (stop).
    assert out == [100, 200, 201, 202, 250, 300, 999]

    # Two propose cycles, two verify forwards.
    assert collector.proposed_drafts == 2 * GAMMA
    # 3 accepted in cycle 1 + 1 in cycle 2.
    assert collector.accepted_drafts == GAMMA + 1
    assert collector.yielded_drafts == GAMMA + 1
    # Both cycles emit a bonus (cycle 1 from full-accept, cycle 2 from
    # partial-reject; cycle 2's bonus = 999 hits stop_token after the
    # bonus path's record_bonus runs).
    assert collector.bonus_tokens == 2
    assert collector.target_forward_count == 2
    assert collector.draft_forward_count == 2
    # Only cycle 2 fires a rollback (un_committed > 0).
    assert collector.rollback_count == 1
    out_md = collector.materialize()
    assert out_md["accept_rate"] == pytest.approx((GAMMA + 1) / (2 * GAMMA))
    # (γ + 1 yielded + 2 bonuses) / 2 forwards.
    assert out_md["tokens_per_target_forward"] == pytest.approx(
        (GAMMA + 1 + 2) / 2
    )
    assert validate_speculative_metrics(out_md) == []
