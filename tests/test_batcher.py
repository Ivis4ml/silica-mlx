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


def test_add_request_after_step_raises_16c() -> None:
    """Cohort is sealed on the first step() call; further admits are 16c."""
    adapter = _ScriptedAdapter(script=[1, 2])
    b = ContinuousBatcher(adapter, max_batch_size=2)
    b.add_request(0, [10, 11], _greedy(max_tokens=3))
    b.step()  # seals cohort
    with pytest.raises(NotImplementedError, match="16c"):
        b.add_request(1, [20, 21], _greedy())


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
