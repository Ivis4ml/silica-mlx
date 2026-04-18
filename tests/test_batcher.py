"""Tests for silica.scheduler.batcher.ContinuousBatcher (P-2 Units 16a / 16b).

B=1 scaffolding and fixed-cohort B>1 behavior are exercised with
scripted adapters so correctness is observable without any real-model
dependency. Capability-gate tests use minimal attention-pattern
fixtures; step-loop tests use scripted forwards that return peaked
logits from a pre-set queue of next-token targets.

Real-model (Qwen3-0.6B) B=1 oracle parity lives in
``tests/test_p2_preload_parity.py``; B>1 batched parity lives in
``tests/test_p2_batched_parity.py``. Both are skipped when the
checkpoint is not cached.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.core.events import BatchEvent
from silica.core.request import RequestStatus
from silica.core.sampling import SamplingParams
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    StateDelta,
)
from silica.scheduler.batcher import ContinuousBatcher

# --- scripted adapter ---------------------------------------------------------


_ScriptStep = int | Sequence[int]


class _ScriptedTokenizer:
    vocab_size = 32

    def encode(self, text: str) -> list[int]:
        return [] if text == "" else [1, 2, 3]

    def decode(self, ids: Any) -> str:
        return ""


class _ScriptedModel:
    """mlx-lm-shaped model.

    Returns peaked logits at positions driven by a per-instance
    ``script`` queue. Each script step can be a scalar target (broadcast
    to every row) or a sequence of per-row targets.
    """

    VOCAB = 32
    N_KV = 1
    HEAD_DIM = 4

    def __init__(self, script: Sequence[_ScriptStep]) -> None:
        self.script: list[_ScriptStep] = list(script)
        self.forward_calls = 0
        self.last_T: int | None = None

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        self.forward_calls += 1
        B, T = tokens.shape
        self.last_T = int(T)
        if cache is not None and cache[0] is not None:
            k = mx.zeros((B, self.N_KV, T, self.HEAD_DIM), dtype=mx.float16)
            v = mx.zeros((B, self.N_KV, T, self.HEAD_DIM), dtype=mx.float16)
            cache[0].update_and_fetch(k, v)
        target: _ScriptStep = self.script.pop(0) if self.script else 0
        return self._logits_for_target(target, B=B, T=T)

    def _logits_for_target(
        self, target: _ScriptStep, *, B: int, T: int
    ) -> mx.array:
        """Build ``(B, T, V)`` logits for scalar or per-row targets."""
        if isinstance(target, int):
            one_hot = mx.zeros((self.VOCAB,), dtype=mx.float32)
            one_hot[target] = 1.0
            # Broadcast to (B, T, V); the last-position slice reads ``target``.
            return mx.broadcast_to(
                one_hot.reshape(1, 1, self.VOCAB),
                (B, T, self.VOCAB),
            )
        targets = list(target)
        if len(targets) != B:
            raise AssertionError(
                f"per-row script step has {len(targets)} targets for B={B}"
            )
        rows: list[mx.array] = []
        for tok in targets:
            row_hot = mx.zeros((self.VOCAB,), dtype=mx.float32)
            row_hot[int(tok)] = 1.0
            rows.append(
                mx.broadcast_to(
                    row_hot.reshape(1, self.VOCAB),
                    (T, self.VOCAB),
                )
            )
        return mx.stack(rows, axis=0)


class _ScriptedAdapter:
    """Minimal ModelAdapter implementing the capability-gate-friendly shape."""

    def __init__(
        self,
        n_layers: int = 2,
        script: Sequence[_ScriptStep] = (),
        attention_pattern: AttentionPattern | None = None,
    ) -> None:
        self.config = ModelConfig(
            model_name="scripted",
            num_layers=n_layers,
            hidden_size=16,
            vocab_size=_ScriptedModel.VOCAB,
        )
        self._n_layers = n_layers
        self._model = _ScriptedModel(script=script)
        self._tokenizer = _ScriptedTokenizer()
        self._pattern = attention_pattern or AttentionPattern(
            per_layer=tuple(AttentionKind.GLOBAL for _ in range(n_layers))
        )
        self._kv_layout = KVLayout(
            num_layers=n_layers,
            n_kv_heads=_ScriptedModel.N_KV,
            head_dim=_ScriptedModel.HEAD_DIM,
            dtype=mx.float16,
        )

    # ModelAdapter Protocol surface
    def build(self, weight_provider: Any) -> Any:
        return self._model

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def attention_pattern(self) -> AttentionPattern:
        return self._pattern

    def tokenizer(self) -> _ScriptedTokenizer:
        return self._tokenizer

    def prefill(
        self, tokens: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:  # pragma: no cover - batcher uses model directly
        raise NotImplementedError

    def decode_step(
        self, token: mx.array, kv_handle: Any
    ) -> tuple[mx.array, StateDelta]:  # pragma: no cover
        raise NotImplementedError


def _greedy(max_tokens: int = 4, stop: Sequence[int] = ()) -> SamplingParams:
    return SamplingParams(
        temperature=0.0, max_tokens=max_tokens, stop_token_ids=tuple(stop)
    )


# --- capability gate ---------------------------------------------------------


@pytest.mark.parametrize(
    "bad_kind",
    [
        AttentionKind.HYBRID_DELTANET,
        AttentionKind.RECURRENT,
        AttentionKind.SLIDING,
        AttentionKind.HYBRID,
    ],
)
def test_capability_gate_rejects_non_global_patterns(
    bad_kind: AttentionKind,
) -> None:
    pattern = AttentionPattern(
        per_layer=(AttentionKind.GLOBAL, bad_kind, AttentionKind.GLOBAL)
    )
    adapter = _ScriptedAdapter(n_layers=3, attention_pattern=pattern)
    with pytest.raises(NotImplementedError, match="AttentionKind.GLOBAL only"):
        ContinuousBatcher(adapter)


def test_capability_gate_accepts_all_global() -> None:
    adapter = _ScriptedAdapter(n_layers=4)
    ContinuousBatcher(adapter)  # no raise


def test_capability_gate_error_names_phase_for_deltanet() -> None:
    pattern = AttentionPattern(per_layer=(AttentionKind.HYBRID_DELTANET,))
    adapter = _ScriptedAdapter(n_layers=1, attention_pattern=pattern)
    with pytest.raises(NotImplementedError, match="P-3"):
        ContinuousBatcher(adapter)


def test_capability_gate_error_names_phase_for_sliding() -> None:
    pattern = AttentionPattern(per_layer=(AttentionKind.SLIDING,))
    adapter = _ScriptedAdapter(n_layers=1, attention_pattern=pattern)
    with pytest.raises(NotImplementedError, match="Q-013"):
        ContinuousBatcher(adapter)


# --- admission / cohort rules -----------------------------------------------


def test_max_batch_size_below_1_raises() -> None:
    adapter = _ScriptedAdapter()
    with pytest.raises(ValueError, match=">= 1"):
        ContinuousBatcher(adapter, max_batch_size=0)


def test_add_request_at_capacity_raises_runtime() -> None:
    """Capacity exhaustion is a resource condition — RuntimeError, not
    NotImplementedError. 16c / 16d may grow the cohort dynamically, but
    that is orthogonal to static capacity."""
    adapter = _ScriptedAdapter(script=[1, 2])
    b = ContinuousBatcher(adapter, max_batch_size=1)
    b.add_request(0, [10, 11], _greedy())
    with pytest.raises(RuntimeError, match="capacity"):
        b.add_request(1, [20, 21], _greedy())


def test_add_request_rejects_empty_prompt() -> None:
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter)
    with pytest.raises(ValueError, match="non-empty"):
        b.add_request(0, [], _greedy())


def test_add_request_after_step_goes_to_waiting_queue() -> None:
    """Post-step add_request now enqueues for mid-run admission (16c.1
    step 3). The request lands in the waiting queue rather than
    self._rows until Phase 2 drains it on a later step."""
    adapter = _ScriptedAdapter(script=[(1,), (2,)])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [10, 11], _greedy(max_tokens=3))
    b.step()  # cohort prepared; row 0 admitted pre-step
    # Post-step admit: request goes into the queue, not into self._rows.
    b.add_request(1, [20, 21], _greedy(max_tokens=3))
    assert len(b._waiting_queue) == 1  # type: ignore[attr-defined]
    assert len(b._rows) == 1  # type: ignore[attr-defined]
    assert b._waiting_queue[0].req_index == 1  # type: ignore[attr-defined]


# --- step loop ---------------------------------------------------------------


def _drain(b: ContinuousBatcher) -> list[BatchEvent]:
    out: list[BatchEvent] = []
    while b.has_active():
        out.extend(b.step())
    return out


def test_single_step_emits_first_token_from_prefill() -> None:
    adapter = _ScriptedAdapter(script=[20, 21, 22])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1, 2, 3], _greedy(max_tokens=1))
    events = b.step()
    # One prefill forward produces one token; max_tokens=1 → done in same step.
    tokens = [e for e in events if e.kind == "token"]
    dones = [e for e in events if e.kind == "done"]
    assert len(tokens) == 1
    assert tokens[0].token_id == 20
    assert tokens[0].req_index == 0
    assert len(dones) == 1
    assert dones[0].finish_reason == "max_tokens"


def test_multi_step_emits_one_token_per_step() -> None:
    adapter = _ScriptedAdapter(script=[5, 9, 13])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1, 2], _greedy(max_tokens=3))
    events = _drain(b)
    tokens = [e.token_id for e in events if e.kind == "token"]
    assert tokens == [5, 9, 13]
    # Exactly one forward for prefill + two for decode = 3 total.
    assert adapter._model.forward_calls == 3


def test_stop_token_ends_in_same_step_with_done_event() -> None:
    adapter = _ScriptedAdapter(script=[7])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1], _greedy(max_tokens=16, stop=(7,)))
    events = b.step()
    assert [e.kind for e in events] == ["token", "done"]
    assert events[0].token_id == 7
    assert events[1].finish_reason == "stop_token"
    assert not b.has_active()


def test_stop_token_mid_decode_closes_request() -> None:
    adapter = _ScriptedAdapter(script=[1, 2, 29])  # 29 is stop (VOCAB=32)
    b = ContinuousBatcher(adapter)
    b.add_request(0, [10], _greedy(max_tokens=16, stop=(29,)))
    events = _drain(b)
    tokens = [e.token_id for e in events if e.kind == "token"]
    assert tokens == [1, 2, 29]
    dones = [e for e in events if e.kind == "done"]
    assert len(dones) == 1
    assert dones[0].finish_reason == "stop_token"


def test_max_tokens_caps_generation() -> None:
    adapter = _ScriptedAdapter(script=[1] * 20)
    b = ContinuousBatcher(adapter)
    b.add_request(0, [99], _greedy(max_tokens=5))
    events = _drain(b)
    tokens = [e.token_id for e in events if e.kind == "token"]
    assert len(tokens) == 5
    dones = [e for e in events if e.kind == "done"]
    assert dones[0].finish_reason == "max_tokens"


def test_req_index_preserved_in_events() -> None:
    """The req_index passed to add_request appears on every event."""
    adapter = _ScriptedAdapter(script=[1, 2])
    b = ContinuousBatcher(adapter)
    b.add_request(42, [7, 8], _greedy(max_tokens=2))
    events = _drain(b)
    assert all(e.req_index == 42 for e in events)


def test_has_active_false_after_done() -> None:
    adapter = _ScriptedAdapter(script=[1])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [7], _greedy(max_tokens=1))
    assert b.has_active()
    _drain(b)
    assert not b.has_active()


def test_has_active_false_with_no_requests() -> None:
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter)
    assert not b.has_active()


# --- state machine observability --------------------------------------------


def test_prefill_transitions_to_decode_when_not_terminal() -> None:
    adapter = _ScriptedAdapter(script=[11, 12, 13])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1, 2], _greedy(max_tokens=3))
    # First step: prefill + token 11, transition to DECODE.
    events1 = b.step()
    assert events1[0].kind == "token" and events1[0].token_id == 11
    row = b._rows[0]  # type: ignore[attr-defined]
    assert row.state.status == RequestStatus.DECODE
    # Second step: decode + token 12.
    events2 = b.step()
    assert events2[0].token_id == 12
    assert row.state.status == RequestStatus.DECODE


def test_prefill_short_circuits_to_done_on_first_stop_token() -> None:
    """First sampled token == stop → DONE without ever entering DECODE."""
    adapter = _ScriptedAdapter(script=[29])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1], _greedy(max_tokens=16, stop=(29,)))
    events = b.step()
    assert [e.kind for e in events] == ["token", "done"]
    row = b._rows[0]  # type: ignore[attr-defined]
    assert row.state.status == RequestStatus.DONE


# --- B>1 cohort (Unit 16b) -------------------------------------------------


def test_prefill_is_one_batched_forward_across_all_rows() -> None:
    """3 rows with varying prompt lengths must go through one prefill call,
    not B separate forwards — the point of Unit 16b."""
    adapter = _ScriptedAdapter(script=[1, 2, 3])
    b = ContinuousBatcher(adapter, max_batch_size=3)
    b.add_request(0, [10, 11, 12, 13, 14], _greedy(max_tokens=1))  # T=5
    b.add_request(1, [20, 21, 22], _greedy(max_tokens=1))           # T=3
    b.add_request(2, [30, 31, 32, 33, 34, 35, 36], _greedy(max_tokens=1))  # T=7
    b.step()
    # Exactly one model() invocation for the cohort's prefill.
    assert adapter._model.forward_calls == 1


def test_prefill_tokens_are_left_padded_to_T_max() -> None:
    """Shortest prompt gets the most left-padding; tensor shape is (B, T_max)."""
    adapter = _ScriptedAdapter(script=[1, 2, 3])
    b = ContinuousBatcher(adapter, max_batch_size=3)
    b.add_request(0, [10, 11, 12, 13, 14], _greedy(max_tokens=1))  # len 5
    b.add_request(1, [20, 21, 22], _greedy(max_tokens=1))          # len 3
    b.add_request(2, [30, 31, 32, 33, 34, 35, 36], _greedy(max_tokens=1))  # len 7
    b.step()
    # The scripted model records the last tokens it saw via last_T.
    assert adapter._model.last_T == 7  # T_max


def test_batch_cache_left_padding_matches_prompt_lengths() -> None:
    """BatchKVCache[left_padding][i] == T_max - len(prompt_i)."""
    adapter = _ScriptedAdapter(script=[1])
    b = ContinuousBatcher(adapter, max_batch_size=3)
    b.add_request(0, [10, 11, 12, 13, 14], _greedy(max_tokens=1))  # len 5
    b.add_request(1, [20, 21, 22], _greedy(max_tokens=1))          # len 3
    b.add_request(2, [30, 31, 32, 33, 34, 35, 36], _greedy(max_tokens=1))  # len 7
    b.step()
    cache = b._batch_cache[0]  # type: ignore[index]
    # tolist() returns Any by mlx stubs; narrowed via assert for mypy.
    left_padding_raw = cache.left_padding.tolist()
    assert isinstance(left_padding_raw, list)
    # T_max = 7; expected left_padding = [7-5, 7-3, 7-7] = [2, 4, 0].
    assert left_padding_raw == [2, 4, 0]


def test_decode_is_one_batched_forward_at_T1() -> None:
    """Decode step feeds (B, 1), regardless of any mid-cohort terminations."""
    adapter = _ScriptedAdapter(script=[1, 2, 3, 4])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [10, 11], _greedy(max_tokens=3))
    b.add_request(1, [20], _greedy(max_tokens=3))
    b.step()  # prefill, one forward
    calls_after_prefill = adapter._model.forward_calls
    b.step()  # decode, one forward for the whole batch
    assert adapter._model.forward_calls == calls_after_prefill + 1
    assert adapter._model.last_T == 1


def test_done_row_does_not_emit_more_tokens_or_append_to_generated() -> None:
    """A row that finishes early must stop emitting AND stop growing
    row.generated even while its slot stays in the batched forward."""
    # Script: prefill -> 11 (row_0 stops on this), decode -> 22.
    # Row 0 has stop=(11,). Row 1 has max_tokens=3, no stop.
    adapter = _ScriptedAdapter(script=[11, 22, 22])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1], _greedy(max_tokens=16, stop=(11,)))
    b.add_request(1, [2], _greedy(max_tokens=3))
    # Step 1: prefill → both rows sample 11. Row 0 sees stop → DONE.
    events_1 = b.step()
    row_0_tokens = [e for e in events_1 if e.req_index == 0 and e.kind == "token"]
    row_0_dones = [e for e in events_1 if e.req_index == 0 and e.kind == "done"]
    assert len(row_0_tokens) == 1 and row_0_tokens[0].token_id == 11
    assert len(row_0_dones) == 1
    row_0 = b._rows[0]  # type: ignore[attr-defined]
    assert len(row_0.generated) == 1  # frozen at 1

    # Step 2: decode. Row 0 is terminal, row 1 continues.
    events_2 = b.step()
    row_0_events = [e for e in events_2 if e.req_index == 0]
    row_1_events = [e for e in events_2 if e.req_index == 1]
    assert row_0_events == []  # DONE row emits nothing
    row_1_tokens = [e for e in row_1_events if e.kind == "token"]
    assert len(row_1_tokens) == 1

    # Row 0's generated STILL has only 1 token (placeholder feed didn't append).
    assert len(row_0.generated) == 1


def test_done_row_keeps_batch_axis_stable_until_all_terminal() -> None:
    """Terminal rows stay in the batch (placeholder feed); batch axis B stays
    constant until every row is terminal."""
    # Row 0 stops immediately, row 1 runs 3 tokens.
    adapter = _ScriptedAdapter(script=[15, 5, 5])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1], _greedy(max_tokens=16, stop=(15,)))
    b.add_request(1, [2], _greedy(max_tokens=3))
    _drain(b)
    # Both rows appear in every forward: 1 prefill + 2 decodes = 3 calls.
    assert adapter._model.forward_calls == 3


def test_b_gt_1_per_row_sampling_is_independent() -> None:
    """Each row samples its own row-i logits; req_index is preserved on events."""
    adapter = _ScriptedAdapter(script=[(6, 8, 10)])
    b = ContinuousBatcher(adapter, max_batch_size=3)
    b.add_request(10, [1, 2], _greedy(max_tokens=1))  # high req_index → 10
    b.add_request(20, [3, 4], _greedy(max_tokens=1))  # 20
    b.add_request(30, [5, 6], _greedy(max_tokens=1))  # 30
    events = b.step()
    tokens_by_req = {
        e.req_index: e.token_id
        for e in events
        if e.kind == "token" and e.token_id is not None
    }
    assert tokens_by_req == {10: 6, 20: 8, 30: 10}


# --- has_active / has_work predicate split (Unit 16c.1 step 1) -------------


def test_has_active_and_has_work_false_before_any_request() -> None:
    """Empty batcher: neither predicate is True."""
    b = ContinuousBatcher(_ScriptedAdapter())
    assert not b.has_active()
    assert not b.has_work()


def test_has_work_tracks_active_rows() -> None:
    """Adding a (pre-admit) row registers as both active and work."""
    adapter = _ScriptedAdapter(script=[1])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1, 2], _greedy(max_tokens=1))
    assert b.has_active()
    assert b.has_work()


def test_has_active_false_has_work_true_on_pending_reclaim() -> None:
    """After the last sample phase terminates the last row, has_active goes
    False but has_work stays True — the terminal row is still in
    self._rows awaiting deferred reclaim at the next step start."""
    adapter = _ScriptedAdapter(script=[5])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1, 2], _greedy(max_tokens=1))
    b.step()  # prefill → sample → max_tokens hit → DONE; row stays in self._rows
    assert not b.has_active()
    # Terminal rows still occupy self._rows; has_work must report True so
    # Engine.generate_batch's drain loop continues to the reclaim step
    # (which will be added in 16c.1 step 2).
    assert b.has_work()


def test_has_work_false_after_rows_cleared() -> None:
    """If terminal rows are drained manually, has_work turns False."""
    adapter = _ScriptedAdapter(script=[5])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1, 2], _greedy(max_tokens=1))
    b.step()
    # Simulate reclaim (16c.1 step 2 patch will make this automatic).
    b._rows.clear()  # type: ignore[attr-defined]
    assert not b.has_active()
    assert not b.has_work()


def test_has_work_true_when_waiting_queue_populated() -> None:
    """A pending admit in _waiting_queue counts as work even if no active
    rows exist yet. Forward-compat for 16c.1 step 3."""
    from silica.scheduler.batcher import _PendingAdmit

    b = ContinuousBatcher(_ScriptedAdapter())
    b._waiting_queue.append(  # type: ignore[attr-defined]
        _PendingAdmit(req_index=0, prompt_ids=(1,), params=_greedy())
    )
    assert not b.has_active()
    assert b.has_work()


# --- Reclaim phase (Unit 16c.1 step 2) -------------------------------------


def test_slot_table_populated_on_admission() -> None:
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter, max_batch_size=3)
    b.add_request(10, [1, 2], _greedy())
    b.add_request(20, [3, 4], _greedy())
    b.add_request(30, [5, 6], _greedy())
    assert b._slot_table == {10: 0, 20: 1, 30: 2}  # type: ignore[attr-defined]


def test_reclaim_drops_terminal_rows_and_rebuilds_slot_table() -> None:
    """One row finishes mid-cohort; the next step() reclaims it and
    slot_table re-indexes the survivors."""
    # step 1 prefill: B=3 (all three rows); row 10 hits max_tokens=1 → DONE.
    # step 2 decode: B=2 after reclaim (rows 20, 30 only).
    adapter = _ScriptedAdapter(script=[(7, 8, 9), (11, 12)])
    b = ContinuousBatcher(adapter, max_batch_size=3)
    b.add_request(10, [1, 2], _greedy(max_tokens=1))
    b.add_request(20, [3, 4], _greedy(max_tokens=3))
    b.add_request(30, [5, 6], _greedy(max_tokens=3))
    b.step()  # prefill
    # Row 10 terminal; still in self._rows pending reclaim.
    assert len(b._rows) == 3  # type: ignore[attr-defined]
    # Next step's reclaim phase fires first, then decode runs at B=2.
    b.step()
    assert len(b._rows) == 2  # type: ignore[attr-defined]
    # Survivors re-indexed 0..K-1 in kept order.
    assert b._slot_table == {20: 0, 30: 1}  # type: ignore[attr-defined]
    assert [r.req_index for r in b._rows] == [20, 30]  # type: ignore[attr-defined]


def test_reclaim_with_all_terminal_resets_batch_cache_to_none() -> None:
    """If every row terminates, reclaim must null the batch cache so a
    future admit installs a fresh one (not extend into stale rows).
    §4 Fix 2 from P2_UNIT_16C_PREP.md."""
    adapter = _ScriptedAdapter(script=[(7, 11)])  # both rows sample and finish
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(10, [1], _greedy(max_tokens=1))  # finishes immediately
    b.add_request(20, [2], _greedy(max_tokens=1))  # finishes immediately
    b.step()  # both DONE
    # Terminal rows still in self._rows; batch_cache still allocated.
    assert b._batch_cache is not None  # type: ignore[attr-defined]
    # Next step's reclaim fires with kept=[] → full reset.
    b.step()
    assert b._rows == []  # type: ignore[attr-defined]
    assert b._slot_table == {}  # type: ignore[attr-defined]
    assert b._batch_cache is None  # type: ignore[attr-defined]
    assert not b.has_work()


def test_engine_drain_uses_has_work_not_has_active() -> None:
    """After step 2, Engine's drain loop runs a trailing step() to reclaim
    the last terminal rows before exiting. Observable via total event
    count + final batcher state."""
    # Engine-level test lives in test_engine_generate_batch.py; here we
    # just verify that the batcher drains cleanly when called directly.
    adapter = _ScriptedAdapter(script=[5])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [1, 2], _greedy(max_tokens=1))

    step1 = b.step()
    assert len(step1) == 2  # token + done
    # Terminal row pending reclaim; has_work still True.
    assert not b.has_active()
    assert b.has_work()

    # One more step → reclaim empties self._rows.
    step2 = b.step()
    assert step2 == []  # reclaim runs, no forward to emit
    assert not b.has_work()


# --- Mid-run admit via extend (Unit 16c.1 step 3) --------------------------


def test_mid_run_add_request_enters_waiting_queue() -> None:
    """Post-prepare, add_request populates _waiting_queue (not self._rows)."""
    adapter = _ScriptedAdapter(script=[(1,)])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1], _greedy(max_tokens=3))
    b.step()
    b.add_request(1, [2, 3], _greedy(max_tokens=3))
    assert len(b._waiting_queue) == 1  # type: ignore[attr-defined]
    assert len(b._rows) == 1  # type: ignore[attr-defined]


def test_mid_run_admit_phase_runs_batched_prefill() -> None:
    """Phase 2 pops the waiting queue and runs its own batched prefill."""
    # step 1 prefill: B=1 (row 0); sample 5, DECODE.
    # admit of row 1 happens after step 1.
    # step 2 phase 2: prefill of row 1 (B=1 for the new admit only);
    #                 sample 9, emit; return from step (skip decode).
    # step 3 phase 3: decode for B=2 (old row 0 + new row 1). They
    #                 generate their next tokens.
    adapter = _ScriptedAdapter(script=[(5,), (9,), (10, 11)])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1, 2], _greedy(max_tokens=10))
    events1 = b.step()
    prefill_forward_count = adapter._model.forward_calls  # should be 1
    assert prefill_forward_count == 1
    tok_row0_step1 = [e for e in events1 if e.req_index == 0 and e.kind == "token"]
    assert tok_row0_step1[0].token_id == 5

    # Mid-run admit row 1.
    b.add_request(1, [3, 4, 5], _greedy(max_tokens=10))
    events2 = b.step()
    # Phase 2 ran its own prefill forward → forward_calls += 1.
    assert adapter._model.forward_calls == 2
    # Only newly-admitted row emitted this step; incumbent decode idled.
    tok_row1_step2 = [e for e in events2 if e.req_index == 1 and e.kind == "token"]
    tok_row0_step2 = [e for e in events2 if e.req_index == 0 and e.kind == "token"]
    assert len(tok_row1_step2) == 1
    assert tok_row1_step2[0].token_id == 9
    assert tok_row0_step2 == []  # row 0 idled this step

    # Next step: joint decode for B=2.
    events3 = b.step()
    assert adapter._model.forward_calls == 3
    tok_by_req = {
        e.req_index: e.token_id for e in events3 if e.kind == "token"
    }
    assert tok_by_req == {0: 10, 1: 11}


def test_mid_run_admit_extends_main_cache_preserving_incumbent_indices() -> None:
    """After extend, incumbent rows' indices are unchanged and new rows
    append at the tail (I-2 from prep doc)."""
    adapter = _ScriptedAdapter(script=[(5, 7), (9,), (11, 12, 13)])
    b = ContinuousBatcher(adapter, max_batch_size=3)
    b.add_request(0, [1], _greedy(max_tokens=10))
    b.add_request(2, [2], _greedy(max_tokens=10))
    b.step()  # prefill B=2 for rows 0, 2
    assert b._slot_table == {0: 0, 2: 1}  # type: ignore[attr-defined]

    b.add_request(5, [3], _greedy(max_tokens=10))
    b.step()  # admit row 5; prefill B=1 for just row 5; extend into main
    # Incumbent rows' indices preserved; new row appended at idx 2.
    assert b._slot_table == {0: 0, 2: 1, 5: 2}  # type: ignore[attr-defined]
    assert len(b._rows) == 3  # type: ignore[attr-defined]


def test_mid_run_admit_after_full_termination_replaces_main_cache() -> None:
    """Reclaim sets main cache to None when all rows terminate; next
    admit replaces main rather than extending into stale rows."""
    adapter = _ScriptedAdapter(script=[(7,), (13,)])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1], _greedy(max_tokens=1))  # finishes in step 1
    b.step()  # row 0 DONE
    # Queue a new request; on step 2, reclaim empties the cache and
    # admit installs a fresh one.
    b.add_request(1, [2, 3], _greedy(max_tokens=10))
    b.step()
    assert b._batch_cache is not None  # type: ignore[attr-defined]
    # New row is at index 0 (fresh slot).
    assert b._slot_table == {1: 0}  # type: ignore[attr-defined]
    assert len(b._rows) == 1  # type: ignore[attr-defined]


def test_pre_step_admit_respects_capacity() -> None:
    """Pre-step direct admission honors max_batch_size because those
    requests immediately occupy self._rows."""
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1], _greedy())  # pre-step, direct
    # Second pre-step direct admit fills capacity.
    b.add_request(1, [2], _greedy())
    # Third, also pre-step → capacity exceeded → RuntimeError.
    with pytest.raises(RuntimeError, match="capacity"):
        b.add_request(2, [3], _greedy())


def test_waiting_queue_is_unbounded_backlog() -> None:
    """Per prep doc §3 16c.1 acceptance: max_batch_size bounds ACTIVE
    rows, not queue length. Enqueuing beyond capacity is legal — the
    admit phase drains up to the available capacity per step."""
    adapter = _ScriptedAdapter(script=[(1,), (2,)])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1], _greedy(max_tokens=10))
    b.step()  # cohort prepared; active rows = 1
    # Queue three more beyond what capacity allows at any single step.
    b.add_request(1, [2], _greedy(max_tokens=10))
    b.add_request(2, [3], _greedy(max_tokens=10))
    b.add_request(3, [4], _greedy(max_tokens=10))
    assert len(b._waiting_queue) == 3  # type: ignore[attr-defined]
    # Active rows unchanged; all excess is in backlog.
    assert len(b._rows) == 1  # type: ignore[attr-defined]


def test_admit_drains_only_up_to_available_capacity() -> None:
    """Phase 2 admits ``max_batch_size - len(self._rows)`` at a time;
    rest stay in queue for later steps."""
    # max_batch_size=2; 1 row in flight (pre-step admit); queue 3 more
    # post-step. Next step: admit fills the one remaining slot; queue
    # still has 2. After the active row finishes + reclaims, another
    # admit fills its slot; etc.
    adapter = _ScriptedAdapter(
        script=[(5,), (7,), (8, 9), (10, 11), (12,), (13,)]
    )
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [1], _greedy(max_tokens=2))
    b.step()  # prefill row 0 (B=1); active=1
    b.add_request(1, [2], _greedy(max_tokens=2))
    b.add_request(2, [3], _greedy(max_tokens=2))
    # Queue now: [1, 2]; capacity remaining = 1.
    b.step()  # admits only row 1 (1 of 2 queued)
    assert len(b._rows) == 2  # type: ignore[attr-defined]
    assert len(b._waiting_queue) == 1  # type: ignore[attr-defined]
    assert b._waiting_queue[0].req_index == 2  # type: ignore[attr-defined]


def test_pending_reclaim_row_does_not_block_queue_admit() -> None:
    """max_batch_size=1; the single row finishes but hasn't been reclaimed
    yet; a new add_request must still succeed (queue is backlog, not
    capped by active rows). The next step reclaims the terminal and
    admits the queued request."""
    adapter = _ScriptedAdapter(script=[(5,), (9,)])
    b = ContinuousBatcher(adapter, max_batch_size=1)
    b.add_request(0, [1], _greedy(max_tokens=1))
    b.step()  # row 0 DONE; still in self._rows pending reclaim
    assert not b.has_active()
    assert len(b._rows) == 1  # type: ignore[attr-defined]

    # Queuing now must succeed regardless of active row count.
    b.add_request(1, [2], _greedy(max_tokens=1))
    assert len(b._waiting_queue) == 1  # type: ignore[attr-defined]

    # Next step: reclaim drops row 0 → capacity=1 → admit drains queue.
    b.step()
    assert len(b._rows) == 1  # type: ignore[attr-defined]
    assert b._rows[0].req_index == 1  # type: ignore[attr-defined]
    assert len(b._waiting_queue) == 0  # type: ignore[attr-defined]


# --- 16c.2 step 4 sub-commit 1: ctor wiring + S-6 no-op invariance ---

from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore


def _make_prefix_cache(block_size: int = 16) -> RadixPrefixCache:
    return RadixPrefixCache(
        block_size=block_size,
        store=SyntheticPrefixBlockStore(block_size=block_size),
    )


def test_ctor_without_prefix_cache_initialises_counters_to_zero() -> None:
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter)
    assert b.forward_prompt_tokens == 0
    assert b.prefix_hits == 0
    assert b._prefix_cache is None  # type: ignore[attr-defined]


def test_ctor_stores_prefix_cache_reference() -> None:
    """Cache ownership stays with the caller — batcher holds a reference,
    does not copy or deepcopy."""
    adapter = _ScriptedAdapter()
    pc = _make_prefix_cache()
    b = ContinuousBatcher(adapter, prefix_cache=pc)
    assert b._prefix_cache is pc  # type: ignore[attr-defined]


def test_s6_no_cache_path_matches_16c1_behaviour() -> None:
    """Invariant S-6: with prefix_cache=None, the batcher's observable
    event stream and counter trajectory must be bit-identical to the
    pre-step-4 behaviour. Sub-commit 1's counters are initialised but
    must stay at 0 on a run that never admits through a hit path."""
    adapter1 = _ScriptedAdapter(script=(5, 7))
    b1 = ContinuousBatcher(adapter1, max_batch_size=1)
    b1.add_request(0, [1, 2], _greedy(max_tokens=2))
    events1: list[BatchEvent] = []
    while b1.has_work():
        events1.extend(b1.step())

    adapter2 = _ScriptedAdapter(script=(5, 7))
    b2 = ContinuousBatcher(adapter2, max_batch_size=1, prefix_cache=None)
    b2.add_request(0, [1, 2], _greedy(max_tokens=2))
    events2: list[BatchEvent] = []
    while b2.has_work():
        events2.extend(b2.step())

    assert [
        (e.kind, e.req_index, e.token_id, e.finish_reason) for e in events1
    ] == [
        (e.kind, e.req_index, e.token_id, e.finish_reason) for e in events2
    ]
    # Counters untouched on the no-cache path.
    assert b1.forward_prompt_tokens == 0
    assert b1.prefix_hits == 0
    assert b2.forward_prompt_tokens == 0
    assert b2.prefix_hits == 0


def test_s6_unused_prefix_cache_does_not_mutate() -> None:
    """Passing a prefix_cache but never hitting its admission path (no
    code path uses it yet in sub-commit 1) leaves the cache untouched.
    Sub-commits 2-3 will start mutating it; this test will be rewritten
    then."""
    adapter = _ScriptedAdapter(script=(5,))
    pc = _make_prefix_cache()
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    assert pc.hits == 0
    assert pc.node_count() == 0
