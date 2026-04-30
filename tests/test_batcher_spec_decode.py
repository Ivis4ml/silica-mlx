"""ContinuousBatcher spec-on integration — D-021 step 5 sub-unit (c) slice 1.

Synthetic only — no real model, no cached fixtures. Pins the contract
locked in ``plans/P6_SPEC_FOUNDATION_C_ORIENTATION.md`` §5:

  - Constructor surface: ``draft_engine: DraftEngine | None`` +
    ``verify_k: int = 4``; ``verify_k < 1`` raises; spec-active gates
    to ``attention_kinds == {GLOBAL}`` (HYBRID_DELTANET / SLIDING
    rejected loud per [F-4] / O-6).
  - [F-1] / O-5 transient ctx: the drafter receives a fresh
    ``RequestState`` every cycle whose ``token_ids`` and
    ``output_token_ids`` mirror ``row.prompt_ids + row.generated``;
    ``row.state.output_token_ids`` itself is never mutated by the spec
    helper.
  - [F-3] right-trim: the verify forward writes ``verify_k`` positions
    per cycle, then ``BatchKVCache.prepare(right_padding) + finalize()``
    rolls ``(verify_k - 1) - yielded_count`` positions out per row.
  - [F-5] yielded_count rollback formula matches sub-unit (b) at
    ``silica/engine/__init__.py:283-305``: full-accept / partial /
    full-reject / fewer-than-γ / stop-token mid-yield / max_tokens
    mid-yield all key on yielded_count, not verifier accept_len.
  - Slice-1 scope: B=1 spec-on. Multi-row spec-active raises
    ``NotImplementedError("D-021 (c) slice 2b ...")``.

Spec-off byte-identical regression is covered by the existing
``tests/test_batcher.py`` fixtures running unchanged on this
revision (verified at slice-1 land time); this file does not
duplicate that coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.core.events import BatchEvent
from silica.core.request import RequestState
from silica.core.sampling import SamplingParams
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    StateDelta,
    Tokenizer,
)
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)
from silica.scheduler.batcher import ContinuousBatcher
from silica.speculative.engine import DraftTokens, NoopDraftEngine

# --- Synthetic adapter / model / drafter ------------------------------------


_VOCAB = 1024
_N_KV = 1
_HEAD_DIM = 4


class _SpecScriptedModel:
    """Per-call scripted model.

    Each script entry is a sequence of length ``T`` of per-position
    argmax targets; the model returns ``(B, T, V)`` one-hot logits at
    those targets and grows the (single-layer) BatchKVCache by ``T``
    positions of zeros so the cache's offset / _idx ledger advances
    realistically. ``B`` is always 1 in slice 1.

    The test fixture pre-populates the script with one entry per
    forward the batcher will issue: typically one prefill entry of
    ``T = len(prompt_ids)`` followed by one spec-verify entry of
    ``T = verify_k`` per decode step.
    """

    def __init__(
        self, script: Sequence[Sequence[int]] | list[list[int]]
    ) -> None:
        self.script: list[list[int]] = [list(s) for s in script]
        self.calls = 0
        self.recorded_inputs: list[list[int]] = []

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        self.calls += 1
        B, T = tokens.shape
        # B=1 in slice 1; record token sequence for verify-input asserts.
        flat: list[int] = []
        raw = tokens.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            inner = raw[0]
            assert isinstance(inner, list)
            for x in inner:
                assert isinstance(x, int | float)
                flat.append(int(x))
        self.recorded_inputs.append(flat)

        # Advance the (single) BatchKVCache so the offset ledger stays
        # honest. Slice-1 GLOBAL-only gate guarantees layer 0 is the
        # only layer present (n_layers=1 in tests).
        if cache is not None and len(cache) > 0 and cache[0] is not None:
            k = mx.zeros((B, _N_KV, T, _HEAD_DIM), dtype=mx.float16)
            v = mx.zeros((B, _N_KV, T, _HEAD_DIM), dtype=mx.float16)
            cache[0].update_and_fetch(k, v)

        if not self.script:
            raise AssertionError(
                f"_SpecScriptedModel script exhausted at call {self.calls}; "
                f"expected another entry of length T={T}"
            )
        targets = self.script.pop(0)
        if len(targets) != T:
            raise AssertionError(
                f"script entry has {len(targets)} targets but forward "
                f"received T={T}"
            )
        # Build (B=1, T, V) one-hot logits at per-position targets.
        logits = mx.zeros((B, T, _VOCAB), dtype=mx.float32)
        for t, tgt in enumerate(targets):
            logits[0, t, tgt] = 1.0
        return logits


class _SpecScriptedAdapter:
    """ModelAdapter satisfying the slice-1 GLOBAL-only constraint."""

    def __init__(
        self,
        script: Sequence[Sequence[int]],
        *,
        n_layers: int = 1,
        attention_pattern: AttentionPattern | None = None,
    ) -> None:
        self.config = ModelConfig(
            model_name="spec-scripted",
            num_layers=n_layers,
            hidden_size=16,
            vocab_size=_VOCAB,
        )
        self._n_layers = n_layers
        self._model = _SpecScriptedModel(script)
        self._pattern = attention_pattern or AttentionPattern(
            per_layer=tuple(
                AttentionKind.GLOBAL for _ in range(n_layers)
            )
        )
        self._kv_layout = KVLayout(
            num_layers=n_layers,
            n_kv_heads=_N_KV,
            head_dim=_HEAD_DIM,
            dtype=mx.float16,
        )

    def build(self, weight_provider: Any) -> _SpecScriptedModel:
        return self._model

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def attention_pattern(self) -> AttentionPattern:
        return self._pattern

    def capabilities(self) -> ModelCapabilities:
        return capabilities_from_attention_pattern(self._pattern)

    def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
        from mlx_lm.models.cache import BatchKVCache

        return [
            BatchKVCache(left_padding=left_padding)
            for _ in range(self._n_layers)
        ]

    class _Tokenizer:
        vocab_size: int = _VOCAB

        def encode(self, text: str) -> list[int]:
            del text
            return []

        def decode(self, token_ids: Sequence[int]) -> str:
            del token_ids
            return ""

    def tokenizer(self) -> Tokenizer:
        return _SpecScriptedAdapter._Tokenizer()

    def prefill(
        self, tokens: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:  # pragma: no cover
        raise NotImplementedError

    def decode_step(
        self, token: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:  # pragma: no cover
        raise NotImplementedError

    def decode_step_multi(
        self, tokens: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:  # pragma: no cover
        # Slice-1 batcher does not call ``adapter.decode_step_multi``;
        # it routes through ``forward_batched_full`` on the model the
        # adapter built. Method present so the runtime_checkable
        # ``ModelAdapter`` Protocol membership stays satisfied.
        raise NotImplementedError


class _ScriptedDraftEngine:
    """DraftEngine that returns scripted drafts and records every ctx.

    ``proposes`` is the list of draft-tuples to return per cycle (one
    list per ``propose`` call, in order). When the script runs out,
    further ``propose`` calls return empty drafts (degenerate path).
    ``recorded_ctx`` saves a snapshot per call so tests can pin the
    transient-ctx contract.
    """

    def __init__(
        self,
        proposes: Sequence[tuple[int, ...]],
    ) -> None:
        self.proposes: list[tuple[int, ...]] = list(proposes)
        self.recorded_ctx: list[
            tuple[tuple[int, ...], list[int], int, str]
        ] = []
        self.commits: list[int] = []
        # Snapshots ``ctx.output_token_ids`` at ``commit`` entry so a
        # test can pin that the batcher synced the per-cycle yielded
        # drafts into ctx before commit fired (slice-1 fix for the
        # commit-side stale-ctx finding).
        self.commit_output_ids: list[list[int]] = []
        self.reset_calls: int = 0

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        # Pin the transient ctx shape: token_ids tuple, the
        # output_token_ids list, the asked-for k, and the request_id.
        self.recorded_ctx.append(
            (
                tuple(ctx.request.token_ids),
                list(ctx.output_token_ids),
                k,
                ctx.request.request_id,
            )
        )
        if not self.proposes:
            return DraftTokens(token_ids=())
        token_ids = self.proposes.pop(0)
        return DraftTokens(token_ids=token_ids)

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        self.commits.append(accepted_len)
        self.commit_output_ids.append(list(ctx.output_token_ids))

    def reset(self) -> None:
        self.reset_calls += 1


def _greedy(
    *, max_tokens: int = 16, stop: Sequence[int] = ()
) -> SamplingParams:
    return SamplingParams(
        temperature=0.0, max_tokens=max_tokens, stop_token_ids=tuple(stop)
    )


def _drain_tokens(events: Sequence[BatchEvent]) -> list[int]:
    out: list[int] = []
    for e in events:
        if e.kind == "token":
            assert e.token_id is not None
            out.append(e.token_id)
    return out


def _drain_dones(events: Sequence[BatchEvent]) -> list[str]:
    return [
        e.finish_reason or ""
        for e in events
        if e.kind == "done"
    ]


# --- Constructor surface ----------------------------------------------------


def test_default_draft_engine_is_noop() -> None:
    """No ``draft_engine=`` kwarg ⇒ NoopDraftEngine ⇒ spec-off."""
    adapter = _SpecScriptedAdapter(script=[])
    batcher = ContinuousBatcher(adapter)
    assert isinstance(batcher._draft_engine, NoopDraftEngine)
    assert batcher._spec_active() is False


def test_explicit_noop_draft_engine_is_spec_off() -> None:
    """Passing ``NoopDraftEngine()`` explicitly ⇒ still spec-off."""
    adapter = _SpecScriptedAdapter(script=[])
    batcher = ContinuousBatcher(adapter, draft_engine=NoopDraftEngine())
    assert batcher._spec_active() is False


def test_real_draft_engine_is_spec_active() -> None:
    adapter = _SpecScriptedAdapter(script=[])
    drafter = _ScriptedDraftEngine(proposes=[])
    batcher = ContinuousBatcher(adapter, draft_engine=drafter)
    assert batcher._spec_active() is True


def test_verify_k_below_one_raises_at_construction() -> None:
    adapter = _SpecScriptedAdapter(script=[])
    drafter = _ScriptedDraftEngine(proposes=[])
    with pytest.raises(ValueError, match=r"verify_k must be >= 1"):
        ContinuousBatcher(adapter, draft_engine=drafter, verify_k=0)
    with pytest.raises(ValueError, match=r"verify_k must be >= 1"):
        ContinuousBatcher(adapter, draft_engine=drafter, verify_k=-3)


def test_global_only_gate_rejects_hybrid_deltanet_under_spec_active() -> None:
    """[F-4] / O-6 hard gate: HYBRID_DELTANET + spec-active raises."""
    pattern = AttentionPattern(
        per_layer=(AttentionKind.HYBRID_DELTANET,)
    )
    adapter = _SpecScriptedAdapter(
        script=[], n_layers=1, attention_pattern=pattern
    )
    drafter = _ScriptedDraftEngine(proposes=[])
    with pytest.raises(NotImplementedError, match=r"GLOBAL-only"):
        ContinuousBatcher(adapter, draft_engine=drafter)


def test_global_only_gate_rejects_sliding_under_spec_active() -> None:
    pattern = AttentionPattern(per_layer=(AttentionKind.SLIDING,))
    adapter = _SpecScriptedAdapter(
        script=[], n_layers=1, attention_pattern=pattern
    )
    drafter = _ScriptedDraftEngine(proposes=[])
    with pytest.raises(NotImplementedError, match=r"GLOBAL-only"):
        ContinuousBatcher(adapter, draft_engine=drafter)


def test_global_only_gate_admits_hybrid_deltanet_under_spec_off() -> None:
    """Spec-off batchers stay admissible across the full capability set."""
    pattern = AttentionPattern(
        per_layer=(AttentionKind.HYBRID_DELTANET,)
    )
    adapter = _SpecScriptedAdapter(
        script=[], n_layers=1, attention_pattern=pattern
    )
    # No raise — spec-off path uses _enforce_capability_gate's wider
    # admission set ({GLOBAL, HYBRID_DELTANET, SLIDING}).
    ContinuousBatcher(adapter)


# --- Slice-1 scope guard ----------------------------------------------------


def test_multi_row_spec_active_raises_with_slice_2b_marker() -> None:
    """B>1 spec-active raises ``NotImplementedError`` with slice-2b text."""
    # Prefill targets first sample for both rows (T=1 each, broadcast
    # via per-row script). Cohort prep batches to (B=2, T=1) prefill.
    adapter = _SpecScriptedAdapter(
        script=[[7, 7]],  # one prefill entry, B=2 rows expect targets
        n_layers=1,
    )
    drafter = _ScriptedDraftEngine(proposes=[(8, 9, 10)])
    batcher = ContinuousBatcher(
        adapter,
        draft_engine=drafter,
        verify_k=4,
        max_batch_size=2,
    )
    batcher.add_request(0, [1, 2], _greedy())
    batcher.add_request(1, [3, 4], _greedy())
    # The first step runs prefill; the second step would run spec
    # decode on B=2 rows — that is the call that must raise.
    batcher.step()  # prefill
    with pytest.raises(
        NotImplementedError, match=r"slice 2b — multi-row"
    ):
        batcher.step()


# --- Single-cycle contract pins ---------------------------------------------


def _run_two_step_batcher(
    *,
    prefill_target: int,
    drafts: tuple[int, ...],
    verify_targets: list[int],
    verify_k: int = 4,
    prompt_ids: Sequence[int] = (1, 2),
    max_tokens: int = 16,
    stop: Sequence[int] = (),
) -> tuple[
    ContinuousBatcher,
    _ScriptedDraftEngine,
    list[BatchEvent],
    list[BatchEvent],
]:
    """Build a B=1 batcher, run prefill + one spec decode step.

    Returns ``(batcher, drafter, prefill_events, decode_events)``.
    Caller asserts on the returned events / drafter recordings.
    ``verify_targets`` is the per-position argmax for the spec verify
    forward (length ``verify_k``).
    """
    script: list[list[int]] = [
        [prefill_target] * len(prompt_ids),  # prefill T = len(prompt)
        list(verify_targets),
    ]
    adapter = _SpecScriptedAdapter(script=script, n_layers=1)
    drafter = _ScriptedDraftEngine(proposes=[drafts])
    batcher = ContinuousBatcher(
        adapter, draft_engine=drafter, verify_k=verify_k
    )
    batcher.add_request(
        0, list(prompt_ids), _greedy(max_tokens=max_tokens, stop=stop)
    )
    prefill_events = batcher.step()
    decode_events = batcher.step()
    return batcher, drafter, prefill_events, decode_events


def test_full_accept_yields_all_drafts_plus_bonus_no_rollback() -> None:
    """All γ drafts accepted ⇒ k tokens yielded, right_padding == 0."""
    # γ = 3, k = 4. Drafts = (101, 102, 103). Verify targets:
    # slot 0 (anchor's pred) = 101 (matches drafts[0])
    # slot 1 = 102 (matches drafts[1])
    # slot 2 = 103 (matches drafts[2])
    # slot 3 = 200 (bonus prediction past last accepted draft)
    batcher, drafter, _prefill_events, decode_events = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[101, 102, 103, 200],
    )
    yielded = _drain_tokens(decode_events)
    assert yielded == [101, 102, 103, 200]  # 3 drafts + 1 bonus
    # right_padding = (k-1) - yielded_count = 3 - 3 = 0 ⇒ no roll.
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    # Cache offset: prefill T=2 + verify k=4 = 6 (no trim).
    assert int(layer.offset.tolist()[0]) == 6
    assert int(layer.left_padding.tolist()[0]) == 0
    # Drafter saw the right ctx and got committed accepted_len=3.
    assert drafter.commits == [3]


def test_partial_accept_rolls_back_rejected_drafts() -> None:
    """yielded < draft_count ⇒ right_padding = (k-1) - yielded."""
    # γ = 3, drafts = (101, 102, 103). Targets: [101, 999, 103, 200].
    # Verify rejects at slot 1 ⇒ accept_len = 1 ⇒ yielded = 1.
    # Bonus = verify_logits[1].argmax = 999 (the rejected slot).
    batcher, drafter, _prefill_events, decode_events = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[101, 999, 103, 200],
    )
    yielded = _drain_tokens(decode_events)
    assert yielded == [101, 999]  # 1 draft + 1 bonus
    # right_padding = 3 - 1 = 2.
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    # Cache offset: prefill T=2 + (verify k=4 - right_padding 2) = 4.
    assert int(layer.offset.tolist()[0]) == 4
    assert int(layer.left_padding.tolist()[0]) == 2
    assert drafter.commits == [1]


def test_full_reject_rolls_back_all_drafts() -> None:
    """yielded == 0 ⇒ right_padding = k - 1."""
    # First slot's argmax disagrees with drafts[0] ⇒ accept_len = 0.
    # Bonus = verify_logits[0].argmax = 999 (the anchor's prediction).
    batcher, drafter, _prefill_events, decode_events = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[999, 1, 2, 3],
    )
    yielded = _drain_tokens(decode_events)
    assert yielded == [999]  # bonus only
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    # right_padding = 3 - 0 = 3 ⇒ offset = 2 + (4 - 3) = 3.
    assert int(layer.offset.tolist()[0]) == 3
    assert int(layer.left_padding.tolist()[0]) == 3
    assert drafter.commits == [0]


def test_fewer_than_gamma_drafts_uses_actual_count_not_gamma() -> None:
    """Drafter returns < γ drafts ⇒ formula keys on draft_count.

    Mirrors the sub-unit (b) fix at silica/engine/__init__.py:283-305.
    γ = 3 but drafter returns 2 drafts. Pad slot at index 3 has
    target 9999 (irrelevant). bonus_idx = draft_count = 2 on full
    accept of the 2 drafts ⇒ bonus = verify_logits[2].argmax = 300.
    """
    batcher, drafter, _prefill_events, decode_events = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102),  # only 2 of γ=3
        verify_targets=[101, 102, 300, 9999],
    )
    yielded = _drain_tokens(decode_events)
    assert yielded == [101, 102, 300]
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    # right_padding = (k-1) - yielded = 3 - 2 = 1.
    assert int(layer.offset.tolist()[0]) == 5  # 2 + (4 - 1)
    assert int(layer.left_padding.tolist()[0]) == 1
    assert drafter.commits == [2]


def test_drafter_returning_more_than_gamma_raises() -> None:
    """Protocol violation: drafter returns > γ drafts ⇒ RuntimeError."""
    adapter = _SpecScriptedAdapter(
        script=[[42, 42], [0, 0, 0, 0]], n_layers=1
    )
    drafter = _ScriptedDraftEngine(
        proposes=[(1, 2, 3, 4)]  # γ=3 but 4 drafts returned
    )
    batcher = ContinuousBatcher(
        adapter, draft_engine=drafter, verify_k=4
    )
    batcher.add_request(0, [1, 2], _greedy())
    batcher.step()  # prefill
    with pytest.raises(
        RuntimeError, match=r"4 drafts.*at most γ = 3"
    ):
        batcher.step()


def test_stop_token_mid_yield_caps_at_yielded_count() -> None:
    """Stop-token among accepted drafts ⇒ yielded stops there.

    Drafts = (101, 102, 103); verify accepts all three; stop token = 102.
    yielded_count = 2 (101 + the stop 102). No bonus is sampled.
    right_padding = (k-1) - 2 = 1.
    """
    batcher, drafter, _, decode_events = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[101, 102, 103, 200],
        stop=(102,),
    )
    yielded = _drain_tokens(decode_events)
    dones = _drain_dones(decode_events)
    assert yielded == [101, 102]  # stop emitted then halt
    assert dones == ["stop_token"]
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    assert int(layer.offset.tolist()[0]) == 5  # 2 + (4 - 1)
    assert drafter.commits == [2]


def test_max_tokens_mid_yield_caps_at_budget() -> None:
    """max_tokens cuts the yield short ⇒ rollback uses yielded_count.

    Prompt T=2, max_tokens=2 means after 2 generated tokens the row
    must terminate. Prefill emits 1 token. Then the spec cycle is
    allowed exactly 1 more token before max_tokens fires.
    Drafts = (101, 102, 103); all would accept; yielded stops at 1.
    right_padding = (k-1) - 1 = 2.
    """
    batcher, drafter, _, decode_events = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[101, 102, 103, 200],
        max_tokens=2,
    )
    yielded = _drain_tokens(decode_events)
    dones = _drain_dones(decode_events)
    assert yielded == [101]  # one decode-step token, then max_tokens
    assert dones == ["max_tokens"]
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    assert int(layer.offset.tolist()[0]) == 4  # 2 + (4 - 2)
    # Drafter committed at yielded_count = 1 (NOT verifier accept_len 3).
    assert drafter.commits == [1]


# --- [F-1] / O-5 transient ctx pins -----------------------------------------


def test_cycle_zero_drafter_sees_prompt_plus_anchor() -> None:
    """Drafter at cycle 0 sees prompt_ids + [first_sampled_token].

    Without the [F-1] fix the drafter would see only ``prompt_ids``
    and its KV would land one position too low from the very first
    cycle.
    """
    batcher, drafter, _, _ = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[101, 102, 103, 200],
        prompt_ids=(7, 8, 9),
    )
    assert len(drafter.recorded_ctx) == 1
    token_ids, output_token_ids, asked_k, request_id = (
        drafter.recorded_ctx[0]
    )
    assert token_ids == (7, 8, 9)            # prompt_ids
    assert output_token_ids == [42]          # first sampled token (anchor)
    assert asked_k == 3                      # γ = verify_k - 1
    assert request_id == "req-0"             # row.req_id, F-2 future-ready


def test_row_state_output_token_ids_is_never_mutated() -> None:
    """[F-1] non-mutation pin: row.state.output_token_ids stays empty.

    The transient ctx mutation must not leak to ``row.state``. This
    is the load-bearing pin proving the slice-1 implementation does
    not introduce an implicit "spec path half-maintains output
    history" contract.
    """
    batcher, _, _, _ = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[101, 102, 103, 200],
    )
    row = batcher._rows[0]
    assert row.state.output_token_ids == []
    # ``row.generated`` is the single source of truth for emitted tokens.
    assert row.generated == [42, 101, 102, 103, 200]


def test_subsequent_cycle_drafter_ctx_includes_emitted_history() -> None:
    """Cycle N drafter sees prompt + everything emitted through cycle N-1.

    Two consecutive spec decode steps; assert the second cycle's ctx
    carries the first cycle's output (transient ctx is rebuilt from
    ``row.generated`` each step).
    """
    # Prefill T=2 emits one token (target 42). Two spec cycles follow:
    # cycle 0 yields 3 drafts + bonus (4 tokens), cycle 1 same.
    script: list[list[int]] = [
        [42, 42],                  # prefill
        [101, 102, 103, 200],      # spec verify cycle 0 (full accept)
        [201, 202, 203, 300],      # spec verify cycle 1 (full accept)
    ]
    adapter = _SpecScriptedAdapter(script=script, n_layers=1)
    drafter = _ScriptedDraftEngine(
        proposes=[(101, 102, 103), (201, 202, 203)]
    )
    batcher = ContinuousBatcher(adapter, draft_engine=drafter, verify_k=4)
    batcher.add_request(0, [1, 2], _greedy(max_tokens=16))
    batcher.step()  # prefill
    batcher.step()  # spec cycle 0
    batcher.step()  # spec cycle 1

    assert len(drafter.recorded_ctx) == 2
    # Cycle 0 ctx: prompt + anchor (the first prefill sample).
    _, out0, _, _ = drafter.recorded_ctx[0]
    assert out0 == [42]
    # Cycle 1 ctx: prompt + everything yielded by cycle 0.
    _, out1, _, _ = drafter.recorded_ctx[1]
    assert out1 == [42, 101, 102, 103, 200]


# --- Drafter declined / spec-active edge cases ------------------------------


def test_drafter_returning_zero_drafts_falls_through_to_single_token() -> None:
    """Drafter returns empty ⇒ verify forward pads all γ slots; yield 1.

    Under slice-1 the drafter's ``DraftTokens(token_ids=())`` triggers
    a degenerate spec cycle: verify_input is ``[anchor] + [pad]*γ``,
    accepted_len = 0, yielded_count = 0, right_padding = γ, bonus
    sampled from verify_logits[0]. Net effect equals a single-token
    decode at the cost of one wasted (1, k) verify forward.
    """
    script: list[list[int]] = [
        [42, 42],            # prefill
        [777, 0, 0, 0],      # spec verify: only slot 0 matters (bonus)
    ]
    adapter = _SpecScriptedAdapter(script=script, n_layers=1)
    drafter = _ScriptedDraftEngine(proposes=[()])
    batcher = ContinuousBatcher(adapter, draft_engine=drafter, verify_k=4)
    batcher.add_request(0, [1, 2], _greedy(max_tokens=16))
    batcher.step()
    decode_events = batcher.step()

    yielded = _drain_tokens(decode_events)
    assert yielded == [777]  # bonus only
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    # right_padding = (k-1) - yielded = 3 - 0 = 3.
    assert int(layer.offset.tolist()[0]) == 3  # 2 + (4 - 3)
    assert int(layer.left_padding.tolist()[0]) == 3
    assert drafter.commits == [0]


def test_verify_k_one_degenerates_to_single_token_decode() -> None:
    """verify_k=1 ⇒ γ=0 ⇒ verify_input = [anchor] only; no rollback."""
    script: list[list[int]] = [
        [42, 42],   # prefill
        [555],      # spec verify: T=1, target 555 (bonus / next token)
    ]
    adapter = _SpecScriptedAdapter(script=script, n_layers=1)
    drafter = _ScriptedDraftEngine(proposes=[()])
    batcher = ContinuousBatcher(adapter, draft_engine=drafter, verify_k=1)
    batcher.add_request(0, [1, 2], _greedy(max_tokens=16))
    batcher.step()
    decode_events = batcher.step()

    yielded = _drain_tokens(decode_events)
    assert yielded == [555]
    cache = batcher._batch_cache
    assert cache is not None
    layer = cache[0]
    # right_padding = (k-1) - 0 = 0 ⇒ no roll. Cache holds prefill
    # T=2 + verify T=1 = 3 positions.
    assert int(layer.offset.tolist()[0]) == 3
    assert int(layer.left_padding.tolist()[0]) == 0


# --- Round-3 review fixes ---------------------------------------------------


def test_commit_ctx_carries_yielded_drafts() -> None:
    """[P2] commit must see ctx.output_token_ids synced to row.generated.

    The drafter's ``commit(ctx, accepted_len)`` is part of the I-5
    Protocol surface; even though ``DraftTargetEngine.commit`` ignores
    ``ctx`` today, slice 2a will key bookkeeping on it. Without the
    sync the commit ctx would be the propose-time snapshot — stale
    by ``yielded_count`` tokens — and slice 2a's per-``req_id``
    drafter would lose ``yielded_count`` worth of committed history
    on every cycle.
    """
    batcher, drafter, _, _ = _run_two_step_batcher(
        prefill_target=42,
        drafts=(101, 102, 103),
        verify_targets=[101, 102, 103, 200],
    )
    del batcher
    # One commit on the spec cycle. Its ctx.output_token_ids must
    # reflect ``[anchor] + yielded drafts`` = ``[42, 101, 102, 103]``
    # (the bonus is sampled AFTER commit, so it is not yet in the row).
    assert drafter.commits == [3]
    assert drafter.commit_output_ids == [[42, 101, 102, 103]]


def test_terminal_reclaim_resets_draft_engine() -> None:
    """[P1] When a spec-active row terminates, drafter.reset() fires.

    Drives a row to DONE via stop_token, then advances ``step()`` once
    more so reclaim runs at the top of the next iteration. The
    drafter's ``reset_calls`` counter must increment, mirroring
    ``Engine._drive``'s ``finally`` cleanup.
    """
    # Prompt T=2 prefill (target 42), one decode cycle that yields
    # the stop token at slot 0 — drafts = (66, 67, 68); verify
    # accepts 66 which is the stop token; yield halts; row → DONE.
    script: list[list[int]] = [
        [42, 42],
        [66, 67, 68, 200],
    ]
    adapter = _SpecScriptedAdapter(script=script, n_layers=1)
    drafter = _ScriptedDraftEngine(proposes=[(66, 67, 68)])
    batcher = ContinuousBatcher(adapter, draft_engine=drafter, verify_k=4)
    batcher.add_request(0, [1, 2], _greedy(max_tokens=16, stop=(66,)))
    batcher.step()  # prefill
    decode_events = batcher.step()  # spec decode → DONE
    dones = _drain_dones(decode_events)
    assert dones == ["stop_token"]
    assert drafter.reset_calls == 0  # reset deferred to next step's reclaim
    # Next step triggers _reclaim_terminated which fires reset.
    batcher.step()
    assert drafter.reset_calls == 1


def test_terminal_reclaim_does_not_reset_under_spec_off() -> None:
    """Spec-off batchers must NOT call drafter.reset() on terminal reclaim.

    ``NoopDraftEngine`` does not define a ``reset`` method, but a
    third-party Protocol-only conformer might. The slice-1 reclaim
    code path is gated on ``self._spec_active()`` so spec-off
    batchers never dispatch the reset, regardless of what attributes
    the drafter happens to expose.
    """
    # Build a NoopDraftEngine + a synthetic adapter; run a row to
    # termination. We assert via not-raising rather than inspecting
    # state — NoopDraftEngine has no observable counter.
    script: list[list[int]] = [[55, 55], [66]]
    adapter = _SpecScriptedAdapter(script=script, n_layers=1)
    batcher = ContinuousBatcher(adapter)  # default Noop
    batcher.add_request(0, [1, 2], _greedy(max_tokens=1))
    batcher.step()  # prefill samples 55, hits max_tokens=1 → DONE
    batcher.step()  # reclaim runs; must not raise
    # No assertion target on NoopDraftEngine; the test is structural.


def test_global_only_gate_runs_before_adapter_build() -> None:
    """[P2] Spec-active gate fails fast — adapter.build is not called.

    Real-model adapters pay weight-loading cost (gigabytes; potentially
    OOM) inside ``build``. The slice-1 gate must reject HYBRID_DELTANET
    / SLIDING under spec-active BEFORE that cost is paid.
    """
    # Wrap the synthetic adapter to record ``build`` invocations.
    pattern = AttentionPattern(
        per_layer=(AttentionKind.HYBRID_DELTANET,)
    )

    class _BuildTrackingAdapter(_SpecScriptedAdapter):
        def __init__(self) -> None:
            super().__init__(
                script=[], n_layers=1, attention_pattern=pattern
            )
            self.build_calls: int = 0

        def build(self, weight_provider: Any) -> Any:
            self.build_calls += 1
            return super().build(weight_provider)

    adapter = _BuildTrackingAdapter()
    drafter = _ScriptedDraftEngine(proposes=[])
    with pytest.raises(NotImplementedError, match=r"GLOBAL-only"):
        ContinuousBatcher(adapter, draft_engine=drafter)
    assert adapter.build_calls == 0


def test_engine_generate_batch_threads_draft_engine_to_batcher() -> None:
    """[P1] ``Engine.generate_batch`` propagates draft_engine + verify_k.

    Without the pass-through the batched path silently degraded to
    spec-off regardless of the Engine-level config. This test
    constructs an Engine with a real (synthetic) drafter and drives
    one B=1 generate_batch call; the drafter's ``propose`` MUST be
    called at least once.
    """
    from silica.engine import Engine
    from silica.kvcache.manager import NullKVManager

    # Adapter + drafter setup. Prefill T = len(prompt) = 2 (target 42),
    # one spec verify cycle (anchor 42 + 3 drafts), full accept.
    # Note: ``_SpecScriptedAdapter._Tokenizer.encode`` returns []; we
    # override it on the test instance so a non-empty prompt produces
    # the 2-token prefill the script expects.
    script: list[list[int]] = [
        [42, 42],
        [101, 102, 103, 200],
    ]
    adapter = _SpecScriptedAdapter(script=script, n_layers=1)

    class _TwoTokenTokenizer:
        vocab_size: int = _VOCAB

        def encode(self, text: str) -> list[int]:
            del text
            return [10, 11]

        def decode(self, token_ids: Sequence[int]) -> str:
            del token_ids
            return ""

    # Patch the tokenizer factory so Engine.generate_batch's encode
    # path returns 2 ids matching the prefill script.
    adapter.tokenizer = lambda: _TwoTokenTokenizer()  # type: ignore[method-assign]
    drafter = _ScriptedDraftEngine(proposes=[(101, 102, 103)])

    # ``Engine.generate_batch`` does not use the kv_manager (it builds
    # its own ContinuousBatcher); a NullKVManager satisfies the
    # __init__ shape without paying a real allocator's cost.
    engine = Engine(
        adapter,
        NullKVManager(),
        draft_engine=drafter,
        verify_k=4,
    )
    # Drive generate_batch over a single B=1 prompt; max_tokens is
    # tight so the batcher exits after the first spec cycle.
    events = list(
        engine.generate_batch(
            ["hello"],
            params=_greedy(max_tokens=4),
        )
    )
    # Drafter was called exactly once (one spec cycle yields γ + 1
    # = 4 tokens which hits max_tokens=4).
    assert len(drafter.recorded_ctx) >= 1
    assert drafter.commits == [3]
    # Token events: prefill (42) + 3 accepted drafts (no bonus —
    # max_tokens=4 cuts the cycle just before the bonus sample).
    yielded_tokens = _drain_tokens(events)
    assert yielded_tokens == [42, 101, 102, 103]
