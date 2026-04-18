"""Tests for Engine.generate_batch — P-2 Unit 16a public API.

Uses the scripted adapter from ``tests/test_batcher.py`` indirectly via
a local copy (to keep test_batcher focused on its unit and this file
focused on the Engine wiring).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.core.events import BatchEvent
from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.kvcache.manager import NullKVManager
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    StateDelta,
)

# --- scripted fixtures (local copy — simpler than cross-file import) ---


class _Tokenizer:
    vocab_size = 32

    def __init__(self, encode_to: Sequence[int] = (1, 2, 3)) -> None:
        self._encoded = list(encode_to)

    def encode(self, text: str) -> list[int]:
        return [] if text == "" else list(self._encoded)

    def decode(self, ids: Any) -> str:
        return ""


class _Model:
    VOCAB = 32

    def __init__(self, script: Sequence[int]) -> None:
        self.script: list[int] = list(script)

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        if cache is not None and cache[0] is not None:
            k = mx.zeros((B, 1, T, 4), dtype=mx.float16)
            v = mx.zeros((B, 1, T, 4), dtype=mx.float16)
            cache[0].update_and_fetch(k, v)
        target = self.script.pop(0) if self.script else 0
        one_hot = mx.zeros((self.VOCAB,), dtype=mx.float32)
        one_hot[target] = 1.0
        return mx.broadcast_to(
            one_hot.reshape(1, 1, self.VOCAB),
            (B, T, self.VOCAB),
        )


class _Adapter:
    def __init__(self, script: Sequence[int] = (), n_layers: int = 1) -> None:
        self.config = ModelConfig(
            model_name="scripted",
            num_layers=n_layers,
            hidden_size=16,
            vocab_size=_Model.VOCAB,
        )
        self._model = _Model(script=script)
        self._tok = _Tokenizer()
        self._pattern = AttentionPattern(
            per_layer=tuple(AttentionKind.GLOBAL for _ in range(n_layers))
        )

    def build(self, weight_provider: Any) -> Any:
        return self._model

    def kv_layout(self) -> KVLayout:
        return KVLayout(
            num_layers=self.config.num_layers,
            n_kv_heads=1,
            head_dim=4,
            dtype=mx.float16,
        )

    def attention_pattern(self) -> AttentionPattern:
        return self._pattern

    def tokenizer(self) -> _Tokenizer:
        return self._tok

    def prefill(self, tokens: mx.array, kv_handle: Any) -> tuple[mx.array, StateDelta]:
        raise NotImplementedError  # pragma: no cover — Engine.generate_batch bypasses adapter.prefill

    def decode_step(self, token: mx.array, kv_handle: Any) -> tuple[mx.array, StateDelta]:
        raise NotImplementedError  # pragma: no cover


def _greedy(max_tokens: int = 3, stop: Sequence[int] = ()) -> SamplingParams:
    return SamplingParams(
        temperature=0.0, max_tokens=max_tokens, stop_token_ids=tuple(stop)
    )


def _make_engine(script: Sequence[int]) -> Engine:
    adapter = _Adapter(script=script)
    # KVManager is only used on the P-1 generate path; generate_batch
    # builds its own ContinuousBatcher with BatchKVCache internally, so
    # the kv_manager passed here is inert for these tests.
    return Engine(adapter, NullKVManager())


# --- happy path: single-prompt batch ---


def test_generate_batch_single_prompt_yields_token_and_done_events() -> None:
    engine = _make_engine(script=[5, 7])
    events = list(engine.generate_batch(["hi"], _greedy(max_tokens=2)))
    tokens = [e for e in events if e.kind == "token"]
    dones = [e for e in events if e.kind == "done"]
    assert [e.token_id for e in tokens] == [5, 7]
    assert len(dones) == 1
    assert dones[0].req_index == 0
    assert dones[0].finish_reason == "max_tokens"


def test_generate_batch_events_are_batchevent_instances() -> None:
    engine = _make_engine(script=[1])
    events = list(engine.generate_batch(["hi"], _greedy(max_tokens=1)))
    assert all(isinstance(e, BatchEvent) for e in events)


def test_generate_batch_with_default_params() -> None:
    engine = _make_engine(script=[4] * 300)
    events = list(engine.generate_batch(["hi"]))  # default SamplingParams
    tokens = [e for e in events if e.kind == "token"]
    # Default max_tokens = 256 per SamplingParams.
    assert len(tokens) == 256


# --- empty / trivial prompts ---


def test_empty_prompts_list_yields_nothing() -> None:
    engine = _make_engine(script=[])
    events = list(engine.generate_batch([], _greedy()))
    assert events == []


def test_all_empty_prompts_yield_nothing() -> None:
    engine = _make_engine(script=[])
    events = list(engine.generate_batch([""], _greedy()))
    assert events == []


# --- B>1 cohort ---


def test_multiple_prompts_produce_per_row_events() -> None:
    """Two non-empty prompts admit as a cohort; per-row events come through."""
    # This Engine-level fixture is intentionally coarse: each batched forward
    # pops ONE target and all rows in the batch see the SAME target that step.
    # With max_tokens=2, B=2 we get 2 forwards:
    # prefill pops target_a → both rows sample target_a;
    # decode pops target_b → both rows sample target_b.
    # Row-specific logits are covered in tests/test_batcher.py.
    engine = _make_engine(script=[5, 7])
    events = list(engine.generate_batch(["hi", "there"], _greedy(max_tokens=2)))
    row_0 = [e.token_id for e in events if e.kind == "token" and e.req_index == 0]
    row_1 = [e.token_id for e in events if e.kind == "token" and e.req_index == 1]
    assert row_0 == [5, 7]
    assert row_1 == [5, 7]


def test_one_empty_and_one_real_prompt_is_b1() -> None:
    """Empty prompts are skipped; ["", "hi"] runs as B=1 at req_index=1."""
    engine = _make_engine(script=[9])
    events = list(engine.generate_batch(["", "hi"], _greedy(max_tokens=1)))
    tokens = [e for e in events if e.kind == "token"]
    assert len(tokens) == 1
    # Empty prompt had req_index 0; "hi" had req_index 1. req_index is
    # preserved through events.
    assert tokens[0].req_index == 1


# --- params: SamplingParams | list[SamplingParams] ---


def test_params_as_single_samplingparams() -> None:
    engine = _make_engine(script=[2])
    events = list(engine.generate_batch(["hi"], _greedy(max_tokens=1)))
    assert events[0].token_id == 2


def test_params_as_homogeneous_list() -> None:
    engine = _make_engine(script=[3])
    p = _greedy(max_tokens=1)
    events = list(engine.generate_batch(["hi"], [p]))
    assert events[0].token_id == 3


def test_heterogeneous_params_list_raises() -> None:
    """Heterogeneous params is validated before admission (before B>1 check)."""
    engine = _make_engine(script=[])
    p1 = _greedy(max_tokens=1)
    p2 = _greedy(max_tokens=2)  # differs
    # Use two non-empty prompts; _resolve_batch_params validates heterogeneity
    # before admission, so we get NotImplementedError for the het case rather
    # than the B>1 case.
    with pytest.raises(NotImplementedError, match="Heterogeneous SamplingParams"):
        list(engine.generate_batch(["hi", "there"], [p1, p2]))


def test_params_list_length_mismatch_raises() -> None:
    engine = _make_engine(script=[])
    with pytest.raises(ValueError, match="length"):
        list(engine.generate_batch(["hi"], [_greedy(), _greedy()]))


def test_params_empty_list_is_allowed_when_no_prompts() -> None:
    engine = _make_engine(script=[])
    # Length matches; no prompts → no events, empty params list is fine.
    events = list(engine.generate_batch([], []))
    assert events == []


def test_params_wrong_type_raises() -> None:
    engine = _make_engine(script=[])
    with pytest.raises(TypeError, match="SamplingParams"):
        list(engine.generate_batch(["hi"], "not-params"))  # type: ignore[arg-type]


# --- Queue-bounded admission (Unit 16c.1 step 4) --------------------------


def test_max_batch_size_knob_rejects_below_one() -> None:
    engine = _make_engine(script=[])
    with pytest.raises(ValueError, match="must be >= 1"):
        list(engine.generate_batch(["hi"], _greedy(), max_batch_size=0))


def test_max_batch_size_default_admits_all() -> None:
    """No explicit max_batch_size → all non-empty prompts admit at step 0
    (16b default preserved)."""
    engine = _make_engine(script=[5])
    events = list(
        engine.generate_batch(
            ["hi", "there", "world"], _greedy(max_tokens=1)
        )
    )
    tokens = [e for e in events if e.kind == "token"]
    assert {e.req_index for e in tokens} == {0, 1, 2}


def test_queue_bounded_admission_drains_backlog() -> None:
    """4 prompts with max_batch_size=2: first 2 admit at step 0, next 2
    admit as the initial cohort finishes and reclaims free slots."""
    engine = _make_engine(script=[5] * 20)
    events = list(
        engine.generate_batch(
            ["p0", "p1", "p2", "p3"],
            _greedy(max_tokens=1),
            max_batch_size=2,
        )
    )
    # All 4 requests eventually emit their 1 token and terminate.
    tokens_by_req = {e.req_index: e.token_id for e in events if e.kind == "token"}
    dones_by_req = {e.req_index for e in events if e.kind == "done"}
    assert tokens_by_req == {0: 5, 1: 5, 2: 5, 3: 5}
    assert dones_by_req == {0, 1, 2, 3}


def test_queue_bounded_preserves_req_index_ordering() -> None:
    """Mid-run admits preserve original req_index (not reshuffled)."""
    engine = _make_engine(script=[5] * 20)
    events = list(
        engine.generate_batch(
            ["p0", "p1", "p2"],
            _greedy(max_tokens=1),
            max_batch_size=1,
        )
    )
    # First event per req_index ordering: 0, 1, 2 in queue order.
    first_token_req_indices = [
        e.req_index for e in events if e.kind == "token"
    ]
    assert first_token_req_indices == [0, 1, 2]


# --- prefix_cache kwarg (16c.2 step 4 sub-commit 1) ----------------------

from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore


def test_prefix_cache_kwarg_accepts_none_as_default() -> None:
    """Default behaviour is unchanged from 16c.1 — not passing
    prefix_cache at all must produce the same events as the existing
    cohort admission path."""
    engine = _make_engine(script=[5, 7])
    events_no_kwarg = list(
        engine.generate_batch(["hi"], _greedy(max_tokens=2))
    )
    engine2 = _make_engine(script=[5, 7])
    events_explicit_none = list(
        engine2.generate_batch(["hi"], _greedy(max_tokens=2), prefix_cache=None)
    )
    assert (
        [(e.kind, e.req_index, e.token_id) for e in events_no_kwarg]
        == [(e.kind, e.req_index, e.token_id) for e in events_explicit_none]
    )


def test_prefix_cache_kwarg_forwarded_to_batcher() -> None:
    """Pass a constructed cache through; the batcher must see the same
    instance (sub-commit 1 scope — no admission-path use yet, just the
    reference plumbing). We verify indirectly by patching the
    ContinuousBatcher ctor and capturing the kwarg."""
    store = SyntheticPrefixBlockStore(block_size=16)
    pc = RadixPrefixCache(block_size=16, store=store)

    captured: dict[str, RadixPrefixCache | None] = {"got": object()}  # type: ignore[dict-item]
    from silica import engine as eng_mod

    real_cls = eng_mod.ContinuousBatcher

    class _SpyBatcher(real_cls):  # type: ignore[valid-type, misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["got"] = kwargs.get("prefix_cache")  # type: ignore[assignment]
            super().__init__(*args, **kwargs)

    eng_mod.ContinuousBatcher = _SpyBatcher  # type: ignore[misc]
    try:
        engine = _make_engine(script=[5])
        list(
            engine.generate_batch(
                ["hi"], _greedy(max_tokens=1), prefix_cache=pc
            )
        )
    finally:
        eng_mod.ContinuousBatcher = real_cls  # type: ignore[misc]

    assert captured["got"] is pc


def test_prefix_cache_lifetime_spans_multiple_generate_batch_calls() -> None:
    """Caller-owned cache persists across calls. Sub-commit 1 is a
    no-op pass-through so the cache must remain untouched (hits
    counter stays 0 — future sub-commits will make it tick)."""
    store = SyntheticPrefixBlockStore(block_size=16)
    pc = RadixPrefixCache(block_size=16, store=store)

    engine = _make_engine(script=[5] * 10)
    list(engine.generate_batch(["hi"], _greedy(max_tokens=1), prefix_cache=pc))
    list(engine.generate_batch(["hi"], _greedy(max_tokens=1), prefix_cache=pc))
    list(engine.generate_batch(["hi"], _greedy(max_tokens=1), prefix_cache=pc))
    assert pc.hits == 0
    assert pc.node_count() == 0


# --- P-1 regression coverage lives in tests/test_engine.py; this file
#     focuses on the generate_batch (P-2 Unit 16a) surface exclusively. ---
