"""Tests for silica.engine.Engine — P-1 single-request generation loop."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.core.profiler import MetricsRegistry
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
)
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)

# --- scripted fakes ---


class _ScriptedTokenizer:
    """Tokenizer that encodes any prompt to a fixed id list (length 2)."""

    vocab_size = 16

    def __init__(self, encode_to: Sequence[int] = (3, 4)) -> None:
        self._encoded = list(encode_to)

    def encode(self, text: str) -> list[int]:
        return [] if text == "" else list(self._encoded)

    def decode(self, ids: Iterable[int]) -> str:
        return ""


class _ScriptedAdapter:
    """Fake I-1 adapter returning logits peaked at a scripted next-token queue.

    Greedy sampling (temp=0) picks the peaked id, so the Engine's output
    matches the script byte-for-byte. Prefill / decode counters let tests
    assert the lifecycle shape.
    """

    VOCAB = 16

    def __init__(self, script: Sequence[int]) -> None:
        self.script: list[int] = list(script)
        self.prefill_calls = 0
        self.decode_calls = 0
        self.last_prefill_tokens: mx.array | None = None
        self.last_decode_token: mx.array | None = None
        self.config = ModelConfig(
            model_name="scripted",
            num_layers=1,
            hidden_size=4,
            vocab_size=self.VOCAB,
        )
        self._tokenizer = _ScriptedTokenizer()

    def build(self, weight_provider: Any) -> Any:  # pragma: no cover - unused
        del weight_provider
        return object()

    def kv_layout(self) -> KVLayout:
        return KVLayout(
            num_layers=1, n_kv_heads=1, head_dim=4, dtype=mx.float16
        )

    def attention_pattern(self) -> AttentionPattern:
        return AttentionPattern(per_layer=(AttentionKind.GLOBAL,))

    def capabilities(self) -> ModelCapabilities:
        return capabilities_from_attention_pattern(self.attention_pattern())

    def tokenizer(self) -> _ScriptedTokenizer:
        return self._tokenizer

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        self.prefill_calls += 1
        self.last_prefill_tokens = tokens
        return self._peaked_logits(), StateDelta()

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        self.decode_calls += 1
        self.last_decode_token = token
        return self._peaked_logits(), StateDelta()

    def _peaked_logits(self) -> mx.array:
        if not self.script:
            # Script exhausted; fall back to token 0.
            target = 0
        else:
            target = self.script.pop(0)
        # Build logits (V,) with target as the argmax.
        base = mx.zeros((self.VOCAB,), dtype=mx.float32)
        onehot = mx.zeros((self.VOCAB,), dtype=mx.float32)
        onehot[target] = 1.0
        return base + onehot


class _TrackedKVManager(NullKVManager):
    """NullKVManager subclass that records reserve / free calls."""

    def __init__(self) -> None:
        super().__init__()
        self.reserved: list[str] = []
        self.freed: list[str] = []

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        self.reserved.append(req_id)
        return super().reserve_for_prefill(req_id, token_ids)

    def free(self, req_id: str) -> None:
        self.freed.append(req_id)
        return super().free(req_id)


def _greedy() -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=16)


def _collect(it: Iterator[int]) -> list[int]:
    return list(it)


# --- happy path: scripted tokens come out verbatim ---


def test_generate_yields_scripted_tokens_under_greedy() -> None:
    adapter = _ScriptedAdapter(script=[5, 9, 2])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=3)
    out = _collect(engine.generate("hi", params))
    assert out == [5, 9, 2]


def test_generate_prefill_decode_counts() -> None:
    adapter = _ScriptedAdapter(script=[5])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=3)
    out = _collect(engine.generate("hi", params))
    # One prefill → 1 token. Two decodes → 2 tokens. Total 3 == max_tokens.
    assert len(out) == 3
    assert adapter.prefill_calls == 1
    assert adapter.decode_calls == 2


# --- max_tokens ---


def test_generate_caps_at_max_tokens() -> None:
    adapter = _ScriptedAdapter(script=[1] * 100)
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=5)
    out = _collect(engine.generate("hi", params))
    assert len(out) == 5
    assert all(t == 1 for t in out)


# --- stop_token_ids ---


def test_generate_stops_on_stop_token_and_yields_it() -> None:
    adapter = _ScriptedAdapter(script=[5, 7, 9, 12])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    params = SamplingParams(
        temperature=0.0, max_tokens=16, stop_token_ids=(9,)
    )
    out = _collect(engine.generate("hi", params))
    # Yields 5, 7, 9 — then stops. 9 is included (yielded-then-stop).
    assert out == [5, 7, 9]


def test_generate_stops_on_first_token_if_it_is_a_stop_token() -> None:
    adapter = _ScriptedAdapter(script=[9])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    params = SamplingParams(
        temperature=0.0, max_tokens=16, stop_token_ids=(9,)
    )
    out = _collect(engine.generate("hi", params))
    assert out == [9]
    # Only prefill ran; no decode needed.
    assert adapter.prefill_calls == 1
    assert adapter.decode_calls == 0


# --- empty prompt ---


def test_empty_prompt_yields_nothing_and_does_not_touch_kv() -> None:
    adapter = _ScriptedAdapter(script=[1, 2, 3])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    out = _collect(engine.generate("", _greedy()))
    assert out == []
    assert kv.reserved == []
    assert kv.freed == []
    assert adapter.prefill_calls == 0


# --- KV lifecycle ---


def test_kv_is_reserved_and_freed_on_normal_exit() -> None:
    adapter = _ScriptedAdapter(script=[1])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    _collect(engine.generate("hi", SamplingParams(temperature=0.0, max_tokens=1)))
    assert kv.reserved == ["req-0"]
    assert kv.freed == ["req-0"]


def test_kv_is_freed_even_if_adapter_raises() -> None:
    class _ExplodingAdapter(_ScriptedAdapter):
        def prefill(
            self, tokens: mx.array, kv_handle: KVHandle
        ) -> tuple[mx.array, StateDelta]:
            raise RuntimeError("boom")

    adapter = _ExplodingAdapter(script=[])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    with pytest.raises(RuntimeError, match="boom"):
        _collect(engine.generate("hi", _greedy()))
    assert kv.reserved == ["req-0"]
    assert kv.freed == ["req-0"]


def test_sequential_generate_uses_fresh_req_ids() -> None:
    adapter = _ScriptedAdapter(script=[1, 2])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    _collect(engine.generate("hi", SamplingParams(temperature=0.0, max_tokens=1)))
    _collect(engine.generate("hi", SamplingParams(temperature=0.0, max_tokens=1)))
    assert kv.reserved == ["req-0", "req-1"]
    assert kv.freed == ["req-0", "req-1"]


# --- shape-level sanity: adapter receives the right tensors ---


def test_prefill_receives_full_prompt_decode_receives_single_token() -> None:
    adapter = _ScriptedAdapter(script=[7, 8])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    params = SamplingParams(temperature=0.0, max_tokens=2)
    _collect(engine.generate("hi", params))
    assert adapter.last_prefill_tokens is not None
    assert adapter.last_prefill_tokens.shape == (2,)  # _ScriptedTokenizer → 2 ids
    assert adapter.last_decode_token is not None
    assert adapter.last_decode_token.shape == (1,)
    # The decode token fed in is the token yielded by prefill (7).
    assert int(adapter.last_decode_token[0].item()) == 7


# --- Protocol-level integration: budget / PrefixHit plumbing ---


def test_engine_does_not_require_prefix_hit_or_budget() -> None:
    """Engine never calls get_computed_blocks in P-1 (budget is queried at end)."""
    adapter = _ScriptedAdapter(script=[1])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    _collect(engine.generate("hi", SamplingParams(temperature=0.0, max_tokens=1)))
    # Sanity — budget() is always safe to call and returns zeros.
    assert kv.budget() == MemoryBudget(0, 0, 0)
    assert kv.get_computed_blocks([1, 2]) == PrefixHit()


# --- metrics (P-1 acceptance item #3) ---


def test_metrics_populated_after_single_token_generate() -> None:
    adapter = _ScriptedAdapter(script=[1])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    _collect(engine.generate("hi", SamplingParams(temperature=0.0, max_tokens=1)))
    snap = engine.metrics.snapshot()
    # TTFT / prefill_tok_s are always set when at least one token is yielded.
    assert snap.ttft_ms is not None and snap.ttft_ms > 0
    assert snap.prefill_tok_s is not None and snap.prefill_tok_s > 0
    # One-token output: no decode loop ran, decode_tok_s stays None.
    assert snap.decode_tok_s is None
    # resident_mb / logical_kv_bytes are recorded even for zero-budget
    # NullKVManager (both equal 0.0 / 0).
    assert snap.resident_mb == 0.0
    assert snap.logical_kv_bytes == 0


def test_metrics_include_decode_tok_s_when_multi_token() -> None:
    adapter = _ScriptedAdapter(script=[1, 2, 3, 4])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    _collect(engine.generate("hi", SamplingParams(temperature=0.0, max_tokens=4)))
    snap = engine.metrics.snapshot()
    assert snap.decode_tok_s is not None and snap.decode_tok_s > 0


def test_metrics_are_per_instance_not_global() -> None:
    """Passing two distinct MetricsRegistry instances isolates metrics."""
    m1 = MetricsRegistry()
    m2 = MetricsRegistry()
    e1 = Engine(_ScriptedAdapter(script=[5]), _TrackedKVManager(), metrics=m1)
    e2 = Engine(_ScriptedAdapter(script=[7]), _TrackedKVManager(), metrics=m2)
    _collect(e1.generate("hi", SamplingParams(temperature=0.0, max_tokens=1)))
    _collect(e2.generate("hi", SamplingParams(temperature=0.0, max_tokens=1)))
    assert e1.metrics is m1
    assert e2.metrics is m2
    # Each registry has its own non-zero ttft_ms.
    assert m1.snapshot().ttft_ms is not None
    assert m2.snapshot().ttft_ms is not None


def test_metrics_populated_even_when_stopped_by_stop_token() -> None:
    """Early stop via stop_token_ids still records tail metrics."""
    adapter = _ScriptedAdapter(script=[9])
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    params = SamplingParams(
        temperature=0.0, max_tokens=16, stop_token_ids=(9,)
    )
    _collect(engine.generate("hi", params))
    snap = engine.metrics.snapshot()
    assert snap.ttft_ms is not None
    assert snap.resident_mb == 0.0  # NullKVManager reports zero
