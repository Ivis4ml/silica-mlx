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
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    StateDelta,
)
from silica.scheduler.batcher import ContinuousBatcher, _PendingAdmit
from silica.scheduler.budget import MemoryBudgeter

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
    # forward_prompt_tokens fires on prefill regardless of cache (S-5 is
    # the "how many tokens did we forward through prefill" counter,
    # not a "prefix-only" counter). prefix_hits stays 0 when no cache.
    assert b1.forward_prompt_tokens == b2.forward_prompt_tokens == 2
    assert b1.prefix_hits == 0
    assert b2.prefix_hits == 0


def test_s6_unused_prefix_cache_still_works_on_sub_block_prompt() -> None:
    """A prompt shorter than one block-aligned prefix cannot populate
    the cache even if prefix_cache is non-None — sub-commit 2's
    reclaim hook no-ops when aligned_tokens < block_size.
    """
    adapter = _ScriptedAdapter(script=(5,))
    # block_size=4 but prompt is only 2 tokens → no aligned block.
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(0, [1, 2], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    assert pc.hits == 0
    assert pc.node_count() == 0


# --- 16c.2 step 4 sub-commit 2: reclaim-path insert_detached ---


def _reclaim_adapter(script: Sequence[_ScriptStep] = ()) -> "_ScriptedAdapter":
    """Single-layer scripted adapter; the reclaim tests need every layer
    written by forward() (the scripted model only writes cache[0]), so
    n_layers=1 avoids the "layer 1 is None" artefact. Multi-layer
    correctness is covered by the real-model acceptance in sub-commit 4."""
    return _ScriptedAdapter(n_layers=1, script=script)


def test_reclaim_with_no_cache_does_not_mutate_prefix_state() -> None:
    """If prefix_cache is None (S-6) the reclaim hook never fires."""
    adapter = _reclaim_adapter(script=(5,))
    b = ContinuousBatcher(adapter, max_batch_size=1)
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    # No prefix_cache to assert on, but b should still be in a clean state.
    assert not b.has_work()
    assert b._batch_cache is None  # type: ignore[attr-defined]


def test_reclaim_extracts_and_inserts_detached_before_filter() -> None:
    """After a row terminates, its block-aligned prefix tokens must be
    in the radix tree with backing detached K/V in the store."""
    adapter = _reclaim_adapter(script=(5,))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    # prompt_ids = [1,2,3,4] (4 tokens = 1 block @ block_size=4)
    # generated = [5] → cache contents = prompt_ids + generated[:-1] = [1,2,3,4]
    # aligned_tokens = 4 → exactly one block retainable.
    assert pc.node_count() == 1
    peeked = pc.peek([1, 2, 3, 4])
    assert peeked.num_hit_tokens == 4
    (bid,) = peeked.block_ids
    assert pc._store.has_detached(bid)  # type: ignore[attr-defined]


def test_reclaim_excludes_unfed_last_generated_token() -> None:
    """S-3a: computed_ids = prompt_ids + generated[:-1]. The final
    sampled token was never fed through a forward; its K/V is NOT in
    cache. The radix tree must reflect that — only prompt_ids is
    aligned-retainable here."""
    adapter = _reclaim_adapter(script=(5,))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    # Prompt is exactly 1 block; generated will add 1 token (the sampled 5).
    # If the helper used prompt_ids + generated, the tree would contain
    # tokens [1,2,3,4,5] but only 4 of them in cache — corrupting future
    # hits on [1,2,3,4]. Correct behaviour: tree contains [1,2,3,4].
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    # Peek with exactly the prompt succeeds.
    assert pc.peek([1, 2, 3, 4]).num_hit_tokens == 4
    # Peek with prompt + sampled token should NOT find a second block
    # (the next chunk [5, *, *, *] is not in the tree).
    assert pc.peek([1, 2, 3, 4, 5, 6, 7, 8]).num_hit_tokens == 4


def test_reclaim_unaligned_terminal_prefix_drops_partial_block() -> None:
    """Aligned tokens < block_size → no insertion. A sub-block prefix
    is not retainable under option B."""
    adapter = _reclaim_adapter(script=(5,))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(0, [1, 2, 3], _greedy(max_tokens=1))  # only 3 tokens
    while b.has_work():
        b.step()
    assert pc.node_count() == 0


def test_reclaim_full_terminal_cohort_handles_all_None_main_cache() -> None:
    """All-terminal path: extract runs BEFORE batch_cache is dropped;
    the cache-None reset must not skip insertion."""
    adapter = _reclaim_adapter(script=(5,))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    # Drive to completion — after step 1 the only row is DONE; step 2's
    # reclaim phase is the all-terminal path.
    while b.has_work():
        b.step()
    # Main cache was dropped (all-terminal), but prefix cache DOES have
    # the detached K/V.
    assert b._batch_cache is None  # type: ignore[attr-defined]
    assert pc.node_count() == 1
    assert pc.peek([1, 2, 3, 4]).num_hit_tokens == 4


def test_reclaim_extract_respects_left_padding() -> None:
    """S-3b: axis-2 slice base = left_padding[row_idx]. Two rows with
    different prompt lengths → non-zero left_padding on the shorter
    row. The extracted K/V for the shorter row must come from the
    post-padding region, not the zero-filled prefix."""
    adapter = _reclaim_adapter(script=(5, 7))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=2, prefix_cache=pc)
    # Row 0: 4-token prompt (0 left-padding).
    # Row 1: 8-token prompt (forces row 0 to have left_padding=4).
    # Both max_tokens=1 so both terminate on step 1; step 2 reclaim
    # sees left_padding = [4, 0] on the incumbent cache.
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    b.add_request(1, [10, 20, 30, 40, 50, 60, 70, 80], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    # Row 0's 4 tokens form one aligned block.
    # Row 1's 8 tokens form two aligned blocks.
    # Total: 3 distinct radix nodes.
    assert pc.node_count() == 3
    assert pc.peek([1, 2, 3, 4]).num_hit_tokens == 4
    assert pc.peek([10, 20, 30, 40, 50, 60, 70, 80]).num_hit_tokens == 8


def test_reclaim_insert_detached_is_filter_safe() -> None:
    """Regression guard on Gate-0.75 probe B: after reclaim, the detached
    K/V must be bit-identical to what was in the source cache before
    filter. Drive this by terminating row 0 while row 1 survives, which
    forces the partial-filter path."""
    # Script interpretation:
    #   step 1 (prefill B=2): row 0 → 5 (terminates), row 1 → 7
    #   step 2 decode (B=1 after reclaim): row 1 → 11 (terminates)
    adapter = _reclaim_adapter(script=((5, 7), 11))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=2, prefix_cache=pc)
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))  # terminates step 1
    b.add_request(1, [10, 20, 30, 40], _greedy(max_tokens=2))  # continues
    while b.has_work():
        b.step()
    # Both rows' 4-token prompts aligned; both blocks registered.
    # Row 0 inserted at partial-filter time (step 2 reclaim, row 1 kept).
    # Row 1 inserted at all-terminal time (step 3 reclaim, row 1 DONE).
    assert pc.node_count() == 2
    # Peek returns hit for both prompts.
    assert pc.peek([1, 2, 3, 4]).num_hit_tokens == 4
    assert pc.peek([10, 20, 30, 40]).num_hit_tokens == 4


# --- 16c.2 step 4 sub-commit 3: admission-path hit split -------------------


def _admit_adapter(script: Sequence[_ScriptStep] = ()) -> "_ScriptedAdapter":
    """Single-layer scripted adapter for admission-path tests. Same rationale
    as _reclaim_adapter — the scripted model only writes cache[0], so
    n_layers=1 avoids the "layer 1 never written" artefact."""
    return _ScriptedAdapter(n_layers=1, script=script)


def test_admission_with_empty_cache_routes_all_to_miss_path() -> None:
    """Empty cache → peek returns no hits → every admission goes through
    the miss cohort path. Counter tracks all prompt tokens, prefix_hits
    stays 0."""
    adapter = _admit_adapter(script=(5,))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    assert b.prefix_hits == 0
    assert b.forward_prompt_tokens == 4


def test_single_hit_row_admission_skips_prefix_tokens_in_forward() -> None:
    """Core happy path. Request 0 pre-fills the cache (initial cohort,
    miss path). Request 1 (same prompt) admits MID-RUN via hit path
    and only forwards the suffix region.

    Pre-step add_request puts a row straight into the initial cohort,
    which goes through _prefill_phase — NOT _admit_waiting_requests.
    The hit path only fires for admissions routed through the
    waiting queue. Triggering it thus requires a step() between the
    two admits so _cohort_prepared flips to True and the second add
    is queued as backlog.
    """
    adapter = _admit_adapter(script=(5, 7))
    pc = _make_prefix_cache(block_size=4)
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)

    # Request 0: pre-step admit, initial-cohort prefill (miss path).
    b.add_request(0, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    b.step()  # seals cohort, runs B=1 prefill on 8 tokens → 5; req 0 DONE.
    assert b.forward_prompt_tokens == 8
    assert b.prefix_hits == 0

    # Request 1: mid-run admit. Step 2's reclaim populates the prefix
    # cache from req 0 (2 blocks at block_size=4), then admit phase
    # peeks req 1's prompt → 8 tokens hit, max_aligned=4, usable=4.
    # Suffix prefill forwards 4 tokens; req 1 DONE on that first token.
    b.add_request(1, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    events = []
    while b.has_work():
        events.extend(b.step())

    assert b.prefix_hits == 1
    assert b.forward_prompt_tokens == 8 + 4
    done_events = [e for e in events if e.kind == "done"]
    assert {e.req_index for e in done_events} == {1}


def test_hit_row_extends_main_cache_preserving_live_rows_I2() -> None:
    """Invariant I-2: extend appends to main, incumbent row's req_index
    stays at its original slot_table position while the hit row gets
    the next index. Drive this by admitting a long-running request 0,
    then mid-run admitting request 1 via hit path.
    """
    pc = _make_prefix_cache(block_size=4)
    # Bootstrap the cache.
    bootstrap = ContinuousBatcher(
        _admit_adapter(script=(5,)), max_batch_size=1, prefix_cache=pc
    )
    bootstrap.add_request(
        99, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1)
    )
    while bootstrap.has_work():
        bootstrap.step()

    # Request 0 is long-running (max_tokens=3) and uses a prompt that
    # does NOT share the cached prefix (miss path).
    # Request 1 shares the cached prefix and is admitted mid-run (hit).
    # Script interpretation:
    #   step 1 prefill B=1: target 20 (req 0 first token, miss)
    #   step 2 hit-admit for req 1 (suffix prefill B=1): target 30
    #          (req 1 terminates on its first sampled token)
    #   step 3 decode B=1 for req 0: target 21 (req 0 second token)
    #   step 4 decode B=1 for req 0: target 22 (terminates req 0)
    adapter = _admit_adapter(script=(20, 30, 21, 22))
    b = ContinuousBatcher(adapter, max_batch_size=2, prefix_cache=pc)
    b.add_request(0, [50, 51, 52, 53], _greedy(max_tokens=3))
    b.step()  # initial cohort prefill → req 0 emits token 20
    assert b._slot_table == {0: 0}  # type: ignore[attr-defined]

    b.add_request(1, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while b.has_work():
        b.step()

    # Hit fired for req 1. Req 0 terminated normally. Cache was
    # extended (not replaced) during req 1's admission because main
    # was non-None at that point.
    assert b.prefix_hits == 1


def test_forward_prompt_tokens_counts_prefill_not_decode() -> None:
    """Decode steps must not bump the counter — only prefill does."""
    adapter = _admit_adapter(script=(5, 7, 9))  # 3 steps: prefill + 2 decode
    b = ContinuousBatcher(adapter, max_batch_size=1)
    b.add_request(0, [1, 2, 3, 4, 5], _greedy(max_tokens=3))
    while b.has_work():
        b.step()
    # Prefill bumped by 5 (prompt length); decode steps bump nothing.
    assert b.forward_prompt_tokens == 5


def test_forward_prompt_tokens_reduces_on_prefix_hit() -> None:
    """Two staggered runs, same prompts; the second run (with cache)
    has strictly lower total forward_prompt_tokens because the second
    admit goes through the hit path."""
    # Run A: no prefix cache. Two requests staggered, both miss.
    adapter_a = _admit_adapter(script=(5, 7))
    b_a = ContinuousBatcher(adapter_a, max_batch_size=1)
    b_a.add_request(0, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    b_a.step()
    b_a.add_request(1, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while b_a.has_work():
        b_a.step()
    total_a = b_a.forward_prompt_tokens  # 8 + 8 = 16

    # Run B: with prefix cache. First request fills, second hits.
    adapter_b = _admit_adapter(script=(5, 7))
    pc = _make_prefix_cache(block_size=4)
    b_b = ContinuousBatcher(adapter_b, max_batch_size=1, prefix_cache=pc)
    b_b.add_request(0, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    b_b.step()
    b_b.add_request(1, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while b_b.has_work():
        b_b.step()
    total_b = b_b.forward_prompt_tokens  # 8 + 4 = 12

    assert total_b < total_a
    assert total_a == 16
    assert total_b == 12


def test_prefix_hits_counter_increments_per_hit_admission() -> None:
    """Each hit-path admission bumps prefix_hits by exactly 1. Drive
    two hits via staggered mid-run admits against a pre-populated cache.
    """
    pc = _make_prefix_cache(block_size=4)
    # Bootstrap: fill the cache via a separate batcher.
    bootstrap = ContinuousBatcher(
        _admit_adapter(script=(5,)), max_batch_size=1, prefix_cache=pc
    )
    bootstrap.add_request(99, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while bootstrap.has_work():
        bootstrap.step()

    # Real batcher: seal cohort with a short dummy so subsequent adds
    # route through the waiting queue (hit path). Dummy prompt length 1
    # — no aligned block, so bootstrap into the prefill path cleanly.
    adapter = _admit_adapter(script=(0, 7, 11))
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(100, [99], _greedy(max_tokens=1))  # dummy, sub-block
    b.step()  # seals cohort; dummy done
    b.add_request(0, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    b.add_request(1, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    assert b.prefix_hits == 2


def test_hit_admission_releases_hit_refs_on_forward_error() -> None:
    """S-2: if the suffix forward raises, retained hit refs MUST be
    released via the finally block. Otherwise subsequent evict_until
    skips the blocks forever."""
    pc = _make_prefix_cache(block_size=4)
    # Bootstrap: fill cache.
    bootstrap = ContinuousBatcher(
        _admit_adapter(script=(5,)), max_batch_size=1, prefix_cache=pc
    )
    bootstrap.add_request(99, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while bootstrap.has_work():
        bootstrap.step()
    hit_block_ids = pc.peek([1, 2, 3, 4, 5, 6, 7, 8]).block_ids
    assert len(hit_block_ids) > 0

    # Build a batcher whose model raises on forward. We trigger the
    # hit-admission path, expect an exception, and verify hit_refs is 0.
    class _ExplodingModel:
        def __call__(
            self, tokens: mx.array, cache: list[Any] | None = None
        ) -> mx.array:
            # Prove this is the hit path, not the miss cohort path:
            # usable_hit_tokens=4, so the model should see only the
            # 4-token suffix and a seeded cache whose logical offset is 4.
            assert tuple(tokens.shape) == (1, 4)
            assert cache is not None
            assert cache[0] is not None
            assert int(cache[0].offset[0].item()) == 4
            raise RuntimeError("boom")

    adapter = _admit_adapter(script=(0,))
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    # Seal the initial cohort with a sub-block dummy. This matters:
    # pre-step admits do NOT enter the prefix-hit path. The cached
    # prompt below must be added mid-run so it goes through
    # _admit_waiting_requests → _admit_single_hit_row.
    b.add_request(100, [50], _greedy(max_tokens=1))
    b.step()

    # Replace the BATCHER's model reference, not adapter._model. The
    # batcher snapshotted adapter.build() at construction time, so
    # mutating the adapter afterwards has no effect.
    b._model = _ExplodingModel()  # type: ignore[assignment]
    b.add_request(0, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    # Before admission, peek again (still side-effect-free).
    for bid in hit_block_ids:
        assert pc._store.hit_refs(bid) == 0  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="boom"):
        while b.has_work():
            b.step()
    # After the exception, every retained hit must be back to 0.
    for bid in hit_block_ids:
        assert pc._store.hit_refs(bid) == 0  # type: ignore[attr-defined]


def test_full_hit_reservation_keeps_one_block_of_suffix_prefill() -> None:
    """S-5 edge 1: when the prompt is fully covered by the cache, we
    still reserve one block of suffix prefill so first-token logits
    are generable."""
    pc = _make_prefix_cache(block_size=4)
    # Bootstrap: cache 8 tokens.
    bootstrap = ContinuousBatcher(
        _admit_adapter(script=(5,)), max_batch_size=1, prefix_cache=pc
    )
    bootstrap.add_request(99, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while bootstrap.has_work():
        bootstrap.step()

    # Seal cohort with sub-block dummy, then mid-run admit the real
    # request so it routes through the hit path.
    adapter = _admit_adapter(script=(0, 7))
    b = ContinuousBatcher(adapter, max_batch_size=1, prefix_cache=pc)
    b.add_request(100, [50], _greedy(max_tokens=1))
    b.step()  # dummy through
    # Now full-hit admission for 8-token prompt:
    # peek → 8 hit tokens, max_aligned = ((8-1)//4)*4 = 4, usable = 4.
    # Suffix = 4 tokens.
    b.add_request(0, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while b.has_work():
        b.step()
    assert b.prefix_hits == 1
    # Dummy's 1-token prefill + req 0's 4-token suffix prefill = 5.
    assert b.forward_prompt_tokens == 1 + 4


def test_mixed_hit_and_miss_rows_in_one_admit_call() -> None:
    """S-7: mixed hit/miss admissions in one step produce phase-grouped
    events (hits first, miss cohort after). This test asserts
    per-row token correctness only; cross-row interleave is NOT
    guaranteed by the contract.
    """
    pc = _make_prefix_cache(block_size=4)
    # Bootstrap: cache prompt [1,2,3,4,5,6,7,8].
    bootstrap = ContinuousBatcher(
        _admit_adapter(script=(5,)), max_batch_size=1, prefix_cache=pc
    )
    bootstrap.add_request(99, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    while bootstrap.has_work():
        bootstrap.step()

    # Script:
    #   step 1 prefill B=1 dummy: → 0 (terminates)
    #   step 2 hit admit for req 1 (B=1 suffix prefill, runs first per S-7): → 7
    #   step 2 miss admit for req 0 (B=1 full prefill): → 13
    # (hit + miss both fire in step 2's admit phase; admit returns both
    # sets of events and step() early-returns — no forward phase runs.)
    adapter = _admit_adapter(script=(0, 7, 13))
    b = ContinuousBatcher(adapter, max_batch_size=2, prefix_cache=pc)
    b.add_request(100, [50], _greedy(max_tokens=1))  # seal dummy
    b.step()

    b.add_request(0, [90, 91, 92, 93, 94, 95, 96, 97], _greedy(max_tokens=1))
    b.add_request(1, [1, 2, 3, 4, 5, 6, 7, 8], _greedy(max_tokens=1))
    events: list[BatchEvent] = []
    while b.has_work():
        events.extend(b.step())

    assert b.prefix_hits == 1
    # Per-row correctness: each real request gets exactly one token +
    # done event.
    tokens_by_req = {
        e.req_index: e.token_id for e in events if e.kind == "token"
    }
    done_by_req = {e.req_index for e in events if e.kind == "done"}
    assert 0 in tokens_by_req and 0 in done_by_req
    assert 1 in tokens_by_req and 1 in done_by_req


# --- 16d-2b: budgeter admit + reject plumbing --------------------------------
#
# Covers the batcher-side of 16d-2b: ContinuousBatcher.__init__ gains a
# ``budgeter`` kwarg, and ``_admit_waiting_requests`` is refactored into
# Phase A (decide + apply per-pending, pop-one-at-a-time) + Phase B
# (execute grouped by hit/miss). These tests pin the contract for
# AdmitDecision / RejectDecision only — AdmitAfterEvict /
# AdmitAfterPreempt raise NotImplementedError (16d-3 / 16d-4 install
# them) and are not exercised here.

def _budgeter_for_adapter(
    adapter: "_ScriptedAdapter",
    *,
    cap_bytes: int,
    weights_bytes: int = 0,
    block_size: int = 4,
) -> MemoryBudgeter:
    """Budgeter + standalone prefix cache for budgeter-only tests.

    Note the prefix cache used HERE (for the budgeter's evict-accounting)
    is separate from any ``prefix_cache`` kwarg passed to the batcher —
    these tests don't install a batcher-side cache, so every admission
    routes through the miss cohort path.
    """
    pc = RadixPrefixCache(
        block_size=block_size,
        store=SyntheticPrefixBlockStore(block_size=block_size),
    )
    return MemoryBudgeter.for_adapter(
        adapter,  # type: ignore[arg-type]
        prefix_cache=pc,
        weights_bytes=weights_bytes,
        cap_bytes=cap_bytes,
    )


def test_budgeter_none_matches_16c2_behaviour() -> None:
    """B-1: ``budgeter=None`` is bit-identical to not passing the kwarg.

    Every observable (event stream + all 16d counters) matches between
    an explicit-None and a default-None batcher. Passing the kwarg must
    not change the admission path.
    """
    adapter1 = _ScriptedAdapter(script=(5, 7))
    b1 = ContinuousBatcher(adapter1, max_batch_size=1, budgeter=None)
    b1.add_request(0, [1, 2], _greedy(max_tokens=2))
    events1: list[BatchEvent] = []
    while b1.has_work():
        events1.extend(b1.step())

    adapter2 = _ScriptedAdapter(script=(5, 7))
    b2 = ContinuousBatcher(adapter2, max_batch_size=1)
    b2.add_request(0, [1, 2], _greedy(max_tokens=2))
    events2: list[BatchEvent] = []
    while b2.has_work():
        events2.extend(b2.step())

    assert [
        (e.kind, e.req_index, e.token_id, e.finish_reason) for e in events1
    ] == [
        (e.kind, e.req_index, e.token_id, e.finish_reason) for e in events2
    ]
    assert b1.aborts == b2.aborts == 0
    assert b1.evictions == b2.evictions == 0
    assert b1.preempts == b2.preempts == 0


def test_admit_decision_proceeds_as_before() -> None:
    """AdmitDecision path is a pass-through: with a generously-sized
    budgeter, admissions produce a normal token + done stream and the
    reservation is released on terminal."""
    adapter = _admit_adapter(script=(5,))
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=100_000)
    b = ContinuousBatcher(adapter, max_batch_size=1, budgeter=budgeter)
    # Seal the empty initial cohort so the next add_request is queued
    # as backlog (mid-run admission → Phase A).
    b.step()
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=1))
    events: list[BatchEvent] = []
    while b.has_work():
        events.extend(b.step())
    # One token, one done — same as the no-budgeter path.
    assert [e.kind for e in events] == ["token", "done"]
    assert b.aborts == 0
    # Terminal release fired via _reclaim_terminated → reserved_bytes
    # returns to 0.
    assert budgeter.reserved_bytes() == 0


def test_admit_decision_commits_reservation_before_next_admit() -> None:
    """B-8 end-to-end — Phase A commits each admission's reservation
    before the next iteration's ``admit()`` call. Setup: three pendings
    where two fit together but the third cannot fit even after
    preempting the newest. Budgeter returns Reject for r3; if B-8 were
    violated (apply_admit deferred to after the loop), r3 would see
    full headroom and admit incorrectly, yielding 3 rows and aborts=0.

    bytes_per_token = 16 (n_layers=1 adapter). cap = 1600.
      - r1, r2: n_prompt=5 + max_tokens=45 = 50 tokens → worst=800.
      - r3:     n_prompt=5 + max_tokens=70 = 75 tokens → worst=1200.
    After r1, r2 apply: headroom = 0. r3.admit: fit 1200>0 no; evict
    n/a (no pc); preempt r2 freed=800, 1200 > 0+800 → Reject.
    """
    # Script: one per-row tuple consumed by the 2-row miss-cohort forward
    # that fires for the (r1, r2) accepted list.
    adapter = _admit_adapter(script=((5, 7),))
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=1600)
    b = ContinuousBatcher(adapter, max_batch_size=3, budgeter=budgeter)
    b.step()  # seal empty cohort

    b.add_request(1, [1, 2, 3, 4, 5], _greedy(max_tokens=45))
    b.add_request(2, [1, 2, 3, 4, 5], _greedy(max_tokens=45))
    b.add_request(3, [1, 2, 3, 4, 5], _greedy(max_tokens=70))

    events = b.step()

    # r1, r2 admitted; r3 rejected.
    assert b.aborts == 1
    assert len(b._rows) == 2  # type: ignore[attr-defined]
    assert budgeter.reserved_bytes() == 1600  # 800 + 800
    # Aborted event for r3 specifically.
    aborted = [e for e in events if e.kind == "aborted"]
    assert len(aborted) == 1
    assert aborted[0].req_index == 3
    assert aborted[0].finish_reason == "budget-exhausted"
    # r1 and r2 emitted tokens (Phase B ran the miss cohort forward).
    token_indices = {e.req_index for e in events if e.kind == "token"}
    assert token_indices == {1, 2}


def test_admit_body_raise_releases_reservation() -> None:
    """B-6'(c): if the Phase B miss-cohort forward raises AFTER
    apply_admit committed, the except handler must release the
    reservation — otherwise the budgeter leaks and subsequent
    admissions see phantom headroom pressure.
    """
    adapter = _admit_adapter(script=(0,))
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=100_000)
    b = ContinuousBatcher(adapter, max_batch_size=1, budgeter=budgeter)
    b.step()  # seal empty cohort

    class _Exploder:
        def __call__(
            self, tokens: mx.array, cache: list[Any] | None = None
        ) -> mx.array:
            raise RuntimeError("boom")

    # Replace BATCHER's model reference (not adapter._model); the
    # batcher snapshotted adapter.build() at ctor time.
    b._model = _Exploder()  # type: ignore[assignment]

    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=2))
    assert budgeter.reserved_bytes() == 0  # pre-condition
    with pytest.raises(RuntimeError, match="boom"):
        b.step()
    # Even though apply_admit committed inside Phase A, the except
    # block in Phase B's miss-cohort branch released it on the raise.
    assert budgeter.reserved_bytes() == 0


def test_reject_decision_emits_aborted_event() -> None:
    """A RejectDecision yields BatchEvent.aborted with
    finish_reason='budget-exhausted'."""
    adapter = _admit_adapter(script=(5,))
    # cap=1 byte; bpt=16 → any request rejects.
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=1)
    b = ContinuousBatcher(adapter, max_batch_size=1, budgeter=budgeter)
    b.step()  # seal empty cohort

    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=2))
    events = b.step()

    aborted = [e for e in events if e.kind == "aborted"]
    assert len(aborted) == 1
    assert aborted[0].req_index == 0
    assert aborted[0].finish_reason == "budget-exhausted"
    assert aborted[0].token_id is None


def test_reject_does_not_create_batch_row() -> None:
    """A rejected admission must not materialise a _BatchRow or extend
    the batched cache. After a reject-only step, _rows stays empty."""
    adapter = _admit_adapter(script=(5,))
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=1)
    b = ContinuousBatcher(adapter, max_batch_size=1, budgeter=budgeter)
    b.step()

    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=2))
    b.step()

    assert b._rows == []  # type: ignore[attr-defined]
    assert b._batch_cache is None  # type: ignore[attr-defined]


def test_reject_increments_aborts_counter() -> None:
    """Each RejectDecision bumps ``self.aborts`` by exactly one."""
    adapter = _admit_adapter(script=(5,))
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=1)
    b = ContinuousBatcher(adapter, max_batch_size=2, budgeter=budgeter)
    b.step()

    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=2))
    b.add_request(1, [1, 2, 3, 4], _greedy(max_tokens=2))
    b.step()

    assert b.aborts == 2


def test_reject_does_not_call_apply_admit() -> None:
    """B-6' / Q-4: a rejected admission never touches reserved_bytes —
    it was never ``apply_admit``'d, so there is no reservation to release.
    After a reject-only step, reserved_bytes stays at 0.
    """
    adapter = _admit_adapter(script=(5,))
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=1)
    b = ContinuousBatcher(adapter, max_batch_size=1, budgeter=budgeter)
    b.step()

    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=2))
    b.step()

    assert budgeter.reserved_bytes() == 0
    assert budgeter.active_requests() == []


def test_reject_event_does_not_block_later_queue_items() -> None:
    """Phase A's reject branch uses ``continue``, not ``break`` — a
    reject on one pending must NOT short-circuit the loop. Queue
    [huge, small] with cap sized to fit the small request: huge rejects,
    small admits, Phase B runs the miss cohort for small.
    """
    adapter = _admit_adapter(script=(5,))
    # cap = (4+2)*16 = 96: exactly the small request's worst-case bytes.
    budgeter = _budgeter_for_adapter(adapter, cap_bytes=96)
    b = ContinuousBatcher(adapter, max_batch_size=2, budgeter=budgeter)
    b.step()  # seal empty cohort

    # Huge request (worst = (4+100)*16 = 1664 >> 96 → reject).
    b.add_request(0, [1, 2, 3, 4], _greedy(max_tokens=100))
    # Small request fits exactly (worst = 96).
    b.add_request(1, [10, 20, 30, 40], _greedy(max_tokens=2))

    events = b.step()

    # Huge aborted, small admitted — same step, same event list.
    assert b.aborts == 1
    assert len(b._rows) == 1  # type: ignore[attr-defined]
    assert b._rows[0].req_index == 1  # type: ignore[attr-defined]

    aborted = [e for e in events if e.kind == "aborted"]
    assert len(aborted) == 1
    assert aborted[0].req_index == 0

    tokens = [e for e in events if e.kind == "token"]
    assert len(tokens) == 1
    assert tokens[0].req_index == 1
    # Small's reservation committed; huge's did not.
    assert budgeter.reserved_bytes() == 96


# --- 16d-3: evict path wiring ------------------------------------------------
#
# AdmitAfterEvictDecision's apply path. Happy path uses a real
# RadixPrefixCache pre-populated via a bootstrap batcher; the underrun
# tests use a spy subclass that returns ``n_blocks - 1`` from
# evict_until so the batcher sees an underrun without needing to
# manipulate live_hits / tree internals directly.


class _UnderrunPrefixCache(RadixPrefixCache):
    """Spy: ``evict_until(n)`` reports freeing ``n - 1`` blocks.

    Drives ``_apply_evict``'s underrun branch without coupling the test
    to RadixPrefixCache's internal eviction ordering. It intentionally
    does NOT mutate the tree: the invariant under review is that an
    underrun leaves batcher + prefix-cache state as it found them.
    """

    def evict_until(self, n_blocks: int) -> int:
        if n_blocks <= 0:
            return 0
        return n_blocks - 1


def _bootstrap_block_into(pc: RadixPrefixCache) -> None:
    """Run a one-request batcher through termination to install one
    aligned block into ``pc`` via the reclaim-side insert_detached
    path. Isolates evict-path tests from hand-rolled synthetic inserts.
    """
    bootstrap = ContinuousBatcher(
        _admit_adapter(script=(5,)), max_batch_size=1, prefix_cache=pc
    )
    bootstrap.add_request(99, [1, 2, 3, 4], _greedy(max_tokens=1))
    while bootstrap.has_work():
        bootstrap.step()
    assert pc.node_count() == 1


def test_evict_decision_calls_evict_until_and_admits() -> None:
    """AdmitAfterEvictDecision happy path: one evictable block in the
    tree; cap tight enough to force the evict branch; eviction succeeds
    → row admits, evictions bumped, leaf gone, reservation applied.

    bpt=16, block_size=4, block_bytes=64. cap=160, filler=96 →
    headroom=64. r0 worst=(4+2)*16=96 > 64 → shortfall=32, n_blocks=1.
    Evictable=1 → AdmitAfterEvict(n_blocks=1).
    """
    pc = _make_prefix_cache(block_size=4)
    _bootstrap_block_into(pc)

    adapter = _admit_adapter(script=(7,))
    budgeter = MemoryBudgeter.for_adapter(
        adapter, prefix_cache=pc, weights_bytes=0, cap_bytes=160
    )
    budgeter.apply_admit("filler", 96)
    b = ContinuousBatcher(
        adapter, max_batch_size=1, prefix_cache=pc, budgeter=budgeter
    )
    b.step()  # seal empty cohort

    b.add_request(0, [10, 20, 30, 40], _greedy(max_tokens=2))
    events = b.step()

    assert b.evictions == 1
    assert b.aborts == 0
    # Leaf evicted.
    assert pc.node_count() == 0
    # Row admitted via miss cohort (r0's prompt doesn't share the
    # bootstrapped prefix [1,2,3,4]).
    assert len(b._rows) == 1  # type: ignore[attr-defined]
    assert b._rows[0].req_index == 0  # type: ignore[attr-defined]
    # filler (96) + r0 (96) = 192.
    assert budgeter.reserved_bytes() == 192
    tokens = [e for e in events if e.kind == "token"]
    assert len(tokens) == 1
    assert tokens[0].req_index == 0


def test_evict_underrun_aborts_without_admit_or_reservation() -> None:
    """B-2: when ``evict_until`` frees fewer blocks than the decision
    asked for, the admission aborts cleanly — no row, no cache
    mutation, reservation untouched, ``evictions`` NOT bumped (only
    successful evicts count).
    """
    pc = _UnderrunPrefixCache(
        block_size=4, store=SyntheticPrefixBlockStore(block_size=4)
    )
    _bootstrap_block_into(pc)

    adapter = _admit_adapter(script=(7,))
    budgeter = MemoryBudgeter.for_adapter(
        adapter, prefix_cache=pc, weights_bytes=0, cap_bytes=160
    )
    budgeter.apply_admit("filler", 96)
    b = ContinuousBatcher(
        adapter, max_batch_size=1, prefix_cache=pc, budgeter=budgeter
    )
    b.step()

    b.add_request(0, [10, 20, 30, 40], _greedy(max_tokens=2))
    events = b.step()

    assert b.aborts == 1
    assert b.evictions == 0  # underrun does NOT count as an eviction
    aborted = [e for e in events if e.kind == "aborted"]
    assert len(aborted) == 1
    assert aborted[0].req_index == 0
    assert aborted[0].finish_reason == "budget-exhausted"
    # State uncorrupted: no row, no cache extension.
    assert b._rows == []  # type: ignore[attr-defined]
    assert b._batch_cache is None  # type: ignore[attr-defined]
    # Only filler's reservation remains; r0 was never apply_admit'd.
    assert budgeter.reserved_bytes() == 96
    # The spy reports an underrun without mutating the tree, so the
    # failed apply left the prefix cache exactly as it found it.
    assert pc.node_count() == 1


def test_evict_underrun_does_not_block_later_queue_items() -> None:
    """An evict underrun on one pending must NOT short-circuit Phase A's
    loop. Queue [r0 (triggers underrun-aborted), r1 (fits-as-is →
    normal admit)]. After the step: r0 aborted, r1 admitted.

    cap=256, filler=96 → headroom=160. r0 worst=(4+7)*16=176 → evict
    decision, spy makes it underrun. r1 worst=(4+2)*16=96 fits within
    the still-160 headroom.
    """
    pc = _UnderrunPrefixCache(
        block_size=4, store=SyntheticPrefixBlockStore(block_size=4)
    )
    _bootstrap_block_into(pc)

    adapter = _admit_adapter(script=(7,))
    budgeter = MemoryBudgeter.for_adapter(
        adapter, prefix_cache=pc, weights_bytes=0, cap_bytes=256
    )
    budgeter.apply_admit("filler", 96)
    b = ContinuousBatcher(
        adapter, max_batch_size=2, prefix_cache=pc, budgeter=budgeter
    )
    b.step()

    b.add_request(0, [10, 20, 30, 40], _greedy(max_tokens=7))  # evict underrun
    b.add_request(1, [50, 60, 70, 80], _greedy(max_tokens=2))  # fits as-is
    events = b.step()

    # r0 aborted, r1 admitted.
    assert b.aborts == 1
    assert b.evictions == 0
    assert len(b._rows) == 1  # type: ignore[attr-defined]
    assert b._rows[0].req_index == 1  # type: ignore[attr-defined]

    aborted = [e for e in events if e.kind == "aborted"]
    assert len(aborted) == 1
    assert aborted[0].req_index == 0

    tokens = [e for e in events if e.kind == "token"]
    assert len(tokens) == 1
    assert tokens[0].req_index == 1

    # filler (96) + r1 (96) = 192; r0 never reserved.
    assert budgeter.reserved_bytes() == 192


# --- 16d-4a: _PendingAdmit.is_replay + _find_row_by_req_id helper -----------
#
# Prep for 16d-4b/c/d preempt wiring. Pure data + read-only helper;
# no batcher behaviour changes land here. The explicit is_replay=True
# round-trip test specifically guards against a field-name typo at
# 16d-4c's appendleft(_PendingAdmit(..., is_replay=True)) site — a
# misspelling would silently fall through default=False and evade B-9.

def test_pending_admit_is_replay_defaults_to_false() -> None:
    """Default for ``is_replay`` is False — every existing construction
    site (add_request, _admit_*) continues to produce non-replay
    pendings without modification."""
    pending = _PendingAdmit(
        req_index=0,
        prompt_ids=(1, 2, 3),
        params=_greedy(max_tokens=4),
    )
    assert pending.is_replay is False


def test_pending_admit_explicit_is_replay_true_round_trips() -> None:
    """Explicit ``is_replay=True`` is preserved. Guards against a
    field-name typo at 16d-4c's re-enqueue site: if the field were
    renamed or misspelled, this test would catch it BEFORE a B-9
    regression could sneak in."""
    pending = _PendingAdmit(
        req_index=0,
        prompt_ids=(1, 2, 3),
        params=_greedy(max_tokens=4),
        is_replay=True,
    )
    assert pending.is_replay is True


def test_find_row_by_req_id_returns_row_idx_when_present() -> None:
    """Scan finds the row by its ``req_id`` string and returns its
    current index in ``self._rows``."""
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter, max_batch_size=3)
    # Pre-step add_request populates self._rows directly; row.req_id
    # format is ``f"req-{req_index}"``.
    b.add_request(0, [1, 2, 3], _greedy(max_tokens=2))
    b.add_request(5, [4, 5, 6], _greedy(max_tokens=2))
    b.add_request(7, [7, 8, 9], _greedy(max_tokens=2))

    assert b._find_row_by_req_id("req-0") == 0  # type: ignore[attr-defined]
    assert b._find_row_by_req_id("req-5") == 1  # type: ignore[attr-defined]
    assert b._find_row_by_req_id("req-7") == 2  # type: ignore[attr-defined]


def test_find_row_by_req_id_returns_none_when_missing() -> None:
    """Scan returns None for a req_id that no current row carries —
    covers both empty ``self._rows`` and populated-but-no-match."""
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter, max_batch_size=2)

    # Empty-rows case.
    assert b._find_row_by_req_id("req-0") is None  # type: ignore[attr-defined]

    # Populated-but-no-match case.
    b.add_request(0, [1, 2, 3], _greedy(max_tokens=2))
    assert b._find_row_by_req_id("req-99") is None  # type: ignore[attr-defined]
