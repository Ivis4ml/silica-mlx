"""Tests for silica.scheduler.batcher.ContinuousBatcher (P-2 Unit 16a).

Exercises the B=1 scaffolding with scripted adapters so correctness is
observable without any real-model dependency. Capability-gate tests use
minimal attention-pattern fixtures; step-loop tests use a scripted
forward that returns peaked logits on a pre-set queue of next tokens.

Real-model (Qwen3-0.6B) oracle parity lives in
``tests/test_p2_preload_parity.py`` (skipped when the checkpoint is
not cached).
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


class _ScriptedTokenizer:
    vocab_size = 32

    def encode(self, text: str) -> list[int]:
        return [] if text == "" else [1, 2, 3]

    def decode(self, ids: Any) -> str:
        return ""


class _ScriptedModel:
    """mlx-lm-shaped model. Returns peaked logits at positions driven by a
    per-instance ``script`` queue; pops one target id per forward call."""

    VOCAB = 32
    N_KV = 1
    HEAD_DIM = 4

    def __init__(self, script: Sequence[int]) -> None:
        self.script: list[int] = list(script)
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
        target = self.script.pop(0) if self.script else 0
        one_hot = mx.zeros((self.VOCAB,), dtype=mx.float32)
        one_hot[target] = 1.0
        # Broadcast to (B, T, V); the last-position slice will read ``target``.
        return mx.broadcast_to(
            one_hot.reshape(1, 1, self.VOCAB),
            (B, T, self.VOCAB),
        )


class _ScriptedAdapter:
    """Minimal ModelAdapter implementing the capability-gate-friendly shape."""

    def __init__(
        self,
        n_layers: int = 2,
        script: Sequence[int] = (),
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


# --- max_batch_size restriction ---------------------------------------------


def test_max_batch_size_greater_than_1_raises() -> None:
    adapter = _ScriptedAdapter()
    with pytest.raises(NotImplementedError, match="16b"):
        ContinuousBatcher(adapter, max_batch_size=2)


def test_add_request_beyond_capacity_raises() -> None:
    adapter = _ScriptedAdapter(script=[1, 2, 3, 4])
    b = ContinuousBatcher(adapter)
    b.add_request(0, [10, 11], _greedy())
    with pytest.raises(NotImplementedError, match="16b"):
        b.add_request(1, [20, 21], _greedy())


def test_add_request_rejects_empty_prompt() -> None:
    adapter = _ScriptedAdapter()
    b = ContinuousBatcher(adapter)
    with pytest.raises(ValueError, match="non-empty"):
        b.add_request(0, [], _greedy())


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
