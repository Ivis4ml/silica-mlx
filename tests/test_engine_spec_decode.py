"""Tests for ``Engine`` speculative-decode wiring (D-021 step 5 sub-unit b).

Single-request engine main-loop integration. Covers:
  - Spec-off (default ``NoopDraftEngine``) byte-identical to the
    pre-spec single-token decode loop.
  - Full accept: γ drafts all match target argmax; engine yields γ + 1
    tokens, KV rollback not called, ``commit(γ)`` invoked.
  - Partial accept: j < γ drafts match; engine yields j drafts + 1
    bonus, ``rollback(γ - j)`` called, ``commit(j)`` invoked.
  - Full reject: 0 drafts match; engine yields 1 bonus, ``rollback(γ)``
    called, ``commit(0)`` invoked.
  - max_tokens cap mid-accept: yielded_count < accepted_len; KV
    rollback covers (γ - yielded_count); no bonus sampled.
  - Stop token mid-accept: same KV bookkeeping; no bonus.
  - Spec + temperature > 0 raises NotImplementedError.
  - Engine constructor rejects verify_k < 1.
  - Draft engine ``reset`` called at generate-cleanup.

Uses synthetic adapter + draft engine — no real model. Real-model
parity is sub-unit (f); this slice pins the wiring contract only.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import mlx.core as mx
import pytest

from silica.core.events import BatchEvent  # noqa: F401 — keeps import order stable
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
from silica.speculative.verify import greedy_verify

# --- shared fakes ----------------------------------------------------------


class _ScriptedTokenizer:
    """Tokenizer that returns a fixed token list for any non-empty prompt."""

    vocab_size: int = 1024

    def __init__(self, ids: Sequence[int] = (10, 11)) -> None:
        self._ids = list(ids)

    def encode(self, text: str) -> list[int]:
        return [] if text == "" else list(self._ids)

    def decode(self, token_ids: Sequence[int]) -> str:
        del token_ids
        return ""


class _ScriptedSpecAdapter:
    """Fake adapter that returns peaked logits scripted per call type.

    ``prefill`` consumes the next ``prefill_logits[i]``; ``decode_step``
    consumes the next ``decode_logits[i]``; ``decode_step_multi``
    consumes the next ``verify_logits[i]`` — a ``(T, V)`` array
    constructed from per-position argmax targets the test specifies.
    """

    VOCAB = 1024

    def __init__(
        self,
        *,
        prefill_argmax: int,
        decode_argmaxes: Sequence[int] = (),
        verify_logits: Sequence[Sequence[int]] = (),
    ) -> None:
        # ``verify_logits`` is a list of per-call argmax targets, one
        # per ``decode_step_multi`` invocation; each entry is a list of
        # length T (the verify_input size for that call).
        self._prefill_q: list[int] = [prefill_argmax]
        self._decode_q: list[int] = list(decode_argmaxes)
        self._verify_q: list[list[int]] = [list(row) for row in verify_logits]

        self.prefill_calls: int = 0
        self.decode_calls: int = 0
        self.decode_multi_calls: int = 0
        self.last_verify_input: mx.array | None = None

        self.config = ModelConfig(
            model_name="scripted-spec",
            num_layers=1,
            hidden_size=4,
            vocab_size=self.VOCAB,
        )
        self._tokenizer = _ScriptedTokenizer()

    # --- ModelAdapter surface (only what Engine consumes) ---

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
        self.prefill_calls += 1
        target = self._prefill_q.pop(0) if self._prefill_q else 0
        return self._one_hot_1d(target), StateDelta()

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        self.decode_calls += 1
        target = self._decode_q.pop(0) if self._decode_q else 0
        return self._one_hot_1d(target), StateDelta()

    def decode_step_multi(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        self.decode_multi_calls += 1
        self.last_verify_input = tokens
        targets = self._verify_q.pop(0) if self._verify_q else [0] * int(tokens.size)
        T = int(tokens.size)
        assert len(targets) == T, (
            f"verify_logits[{self.decode_multi_calls - 1}] has length "
            f"{len(targets)} but verify_input has {T} tokens"
        )
        rows = [self._one_hot_1d(t) for t in targets]
        return mx.stack(rows), StateDelta()

    def _one_hot_1d(self, target: int) -> mx.array:
        scores = [0.0] * self.VOCAB
        scores[target] = 5.0
        return mx.array(scores, dtype=mx.float32)


class _ScriptedDraftEngine:
    """Draft engine whose proposals come from a scripted queue.

    Each ``propose`` pops the next ``(token_ids, draft_logprobs)`` row
    off the queue. Empty rows return empty drafts (Noop-equivalent).
    Tracks every ``propose`` / ``commit`` / ``reset`` call so tests can
    assert the engine drove it correctly.
    """

    def __init__(
        self, proposals: Sequence[Sequence[int]] = ()
    ) -> None:
        self._queue: list[list[int]] = [list(row) for row in proposals]
        self.propose_calls: list[int] = []  # γ values
        self.commit_calls: list[int] = []  # accepted_len values
        self.reset_calls: int = 0
        self.reset_req_ids: list[str | None] = []

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        self.propose_calls.append(int(k))
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
        self.reset_calls += 1
        self.reset_req_ids.append(req_id)


class _TrackedKVManager(NullKVManager):
    """``NullKVManager`` plus rollback / reserve call recording."""

    def __init__(self) -> None:
        super().__init__()
        self.reserved: list[str] = []
        self.freed: list[str] = []
        self.rollback_calls: list[tuple[str, int]] = []

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList:
        self.reserved.append(req_id)
        return super().reserve_for_prefill(req_id, token_ids)

    def rollback(self, req_id: str, n_reject: int) -> None:
        self.rollback_calls.append((req_id, int(n_reject)))
        super().rollback(req_id, n_reject)

    def free(self, req_id: str) -> None:
        self.freed.append(req_id)
        super().free(req_id)

    def get_computed_blocks(self, token_ids: Sequence[int]) -> PrefixHit:
        return PrefixHit()

    def budget(self) -> MemoryBudget:
        return MemoryBudget()


def _collect(it: Iterator[int]) -> list[int]:
    return list(it)


def _greedy(max_tokens: int = 16) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


# --- greedy_verify helper unit tests ---------------------------------------


def test_greedy_verify_full_accept() -> None:
    drafts = (5, 6, 7)
    # verify_logits[i] argmax = drafts[i] for all i → full accept.
    rows = [
        mx.array([0.0] * 16, dtype=mx.float32).at[d].add(5.0)
        for d in drafts
    ]
    verify_logits = mx.stack(rows)
    assert greedy_verify(drafts, verify_logits) == 3


def test_greedy_verify_partial_accept() -> None:
    drafts = (5, 6, 7)
    # First 2 match, third diverges (target wants 9 instead of 7).
    targets = [5, 6, 9]
    rows = [
        mx.array([0.0] * 16, dtype=mx.float32).at[t].add(5.0)
        for t in targets
    ]
    verify_logits = mx.stack(rows)
    assert greedy_verify(drafts, verify_logits) == 2


def test_greedy_verify_full_reject() -> None:
    drafts = (5, 6, 7)
    targets = [9, 9, 9]
    rows = [
        mx.array([0.0] * 16, dtype=mx.float32).at[t].add(5.0)
        for t in targets
    ]
    verify_logits = mx.stack(rows)
    assert greedy_verify(drafts, verify_logits) == 0


# --- spec-off byte-identical -----------------------------------------------


def test_spec_off_default_is_noop_draft_engine() -> None:
    # Default ``draft_engine`` is NoopDraftEngine; verify_k = 4 default.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=5,
        decode_argmaxes=[6, 7],
    )
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv)
    out = _collect(engine.generate("hi", _greedy(max_tokens=3)))
    assert out == [5, 6, 7]
    # Spec branch never entered: no decode_step_multi, no rollback.
    assert adapter.decode_multi_calls == 0
    assert kv.rollback_calls == []


def test_spec_off_with_explicit_noop_matches_default() -> None:
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=1, decode_argmaxes=[2]
    )
    kv = _TrackedKVManager()
    engine = Engine(adapter, kv, draft_engine=NoopDraftEngine())
    out = _collect(engine.generate("hi", _greedy(max_tokens=2)))
    assert out == [1, 2]


# --- spec-on full accept ---------------------------------------------------


def test_spec_on_full_accept_yields_gamma_plus_one() -> None:
    # verify_k = 4 → γ = 3. Anchor = prefill argmax = 100. Drafts =
    # [200, 201, 202]. Verify input = [100, 200, 201, 202]. Adapter
    # argmax at each position = [200, 201, 202, 250] → all 3 drafts
    # accepted, bonus = 250.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 250]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=5)))
    # Yielded: anchor (100), 3 accepted drafts (200, 201, 202), bonus (250).
    assert out == [100, 200, 201, 202, 250]
    # Single decode_step_multi call; no decode_step (spec branch only).
    assert adapter.decode_multi_calls == 1
    assert adapter.decode_calls == 0
    # Full accept: no rollback.
    assert kv.rollback_calls == []
    # Draft engine: propose called once with γ = 3; commit called with
    # accepted_len = 3 (full accept; engine yielded all drafts).
    assert drafter.propose_calls == [3]
    assert drafter.commit_calls == [3]


# --- spec-on partial accept ------------------------------------------------


def test_spec_on_partial_accept_rolls_back_and_yields_bonus() -> None:
    # γ = 3. Drafts = [200, 201, 202]. Adapter argmax targets =
    # [200, 201, 999, ...] → first 2 match, draft[2] rejected. Bonus
    # sampled from verify_logits[2] (argmax 999).
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 999, 0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=4)))
    # Yielded: anchor (100), 2 accepted drafts (200, 201), bonus (999).
    assert out == [100, 200, 201, 999]
    # Rolled back 1 rejected draft (γ - yielded_count = 3 - 2 = 1).
    assert kv.rollback_calls == [("req-0", 1)]
    # commit(yielded_count = 2).
    assert drafter.commit_calls == [2]


# --- spec-on full reject ---------------------------------------------------


def test_spec_on_full_reject_yields_only_bonus() -> None:
    # γ = 3. Drafts = [200, 201, 202]. Adapter argmax = [777, ...] →
    # draft[0] rejected, no further drafts checked. Bonus from
    # verify_logits[0] = 777.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[777, 0, 0, 0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=2)))
    assert out == [100, 777]
    # Rolled back all γ = 3 drafts.
    assert kv.rollback_calls == [("req-0", 3)]
    # commit(0).
    assert drafter.commit_calls == [0]


# --- spec-on max_tokens / stop-token cap -----------------------------------


def test_spec_on_max_tokens_cap_mid_accept_skips_bonus() -> None:
    # γ = 3. Drafts = [200, 201, 202]; all would be accepted, but
    # max_tokens = 3 caps after yielding [anchor=100, draft0=200,
    # draft1=201]. yielded_count = 2. Rollback covers γ - 2 = 1.
    # No bonus sampled because we hit max_tokens mid-accept.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 250]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=3)))
    assert out == [100, 200, 201]
    # Rollback covers the un-yielded slots (the 3rd accepted draft 202
    # plus its position in the cache).
    assert kv.rollback_calls == [("req-0", 1)]
    # commit(yielded_count = 2), not commit(accepted_len = 3).
    assert drafter.commit_calls == [2]


def test_spec_on_stop_token_mid_accept_skips_bonus() -> None:
    # γ = 3. Drafts all match target argmax, but draft[1] = 13 is in
    # stop_token_ids. Engine yields [anchor=100, draft0=12, draft1=13]
    # then stops. yielded_count = 2; rollback γ - 2 = 1; commit(2).
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[12, 13, 14, 15]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[12, 13, 14]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    params = SamplingParams(
        temperature=0.0, max_tokens=8, stop_token_ids=(13,)
    )
    out = _collect(engine.generate("hi", params))
    assert out == [100, 12, 13]
    assert kv.rollback_calls == [("req-0", 1)]
    assert drafter.commit_calls == [2]


# --- fewer-than-γ proposals (Protocol allows "up to k") --------------------


def test_spec_on_fewer_than_gamma_full_accept_no_rollback() -> None:
    # γ = 3 (verify_k = 4) but the drafter legitimately returns only
    # 2 drafts. verify_input = [anchor=100, 200, 201] (3 tokens, NOT
    # 4). Both drafts argmax-match. Full accept of the actual returned
    # drafts → KV rollback must NOT fire (un_committed = 2 - 2 = 0);
    # bonus comes from verify_logits[verify_input.size - 1] = idx 2.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 250]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=4)))
    # Yielded: anchor (100), 2 drafts (200, 201), bonus (250).
    assert out == [100, 200, 201, 250]
    assert kv.rollback_calls == []
    # commit(yielded_count = 2). propose was called with γ = 3 but the
    # drafter returned only 2 — the engine respects that.
    assert drafter.propose_calls == [3]
    assert drafter.commit_calls == [2]
    # Verify forward consumed exactly draft_count + 1 = 3 tokens, not
    # verify_k = 4 — the verify input is sized to the actual drafts.
    assert adapter.last_verify_input is not None
    assert int(adapter.last_verify_input.size) == 3


def test_spec_on_fewer_than_gamma_partial_accept_rolls_back_only_actual() -> None:
    # γ = 3, drafter returns 2 drafts; verify rejects the second.
    # un_committed = draft_count - yielded_count = 2 - 1 = 1, NOT
    # γ - yielded_count = 2.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 999, 0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=3)))
    assert out == [100, 200, 999]
    assert kv.rollback_calls == [("req-0", 1)]
    assert drafter.commit_calls == [1]


def test_spec_on_more_than_gamma_drafts_raises() -> None:
    # γ = 2 (verify_k = 3). A buggy drafter returns 3 drafts —
    # violates the I-5 propose contract ("up to k"). Engine must
    # refuse loud rather than silently corrupt KV state.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 203]],  # never reached
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=3)
    with pytest.raises(RuntimeError, match=r"propose contract"):
        list(engine.generate("hi", _greedy(max_tokens=4)))


# --- defensive raises + lifecycle ------------------------------------------


def test_engine_rejects_verify_k_below_one() -> None:
    adapter = _ScriptedSpecAdapter(prefill_argmax=0)
    kv = _TrackedKVManager()
    with pytest.raises(ValueError, match=r"verify_k must be >= 1"):
        Engine(adapter, kv, verify_k=0)


def test_spec_on_with_temperature_above_zero_raises() -> None:
    adapter = _ScriptedSpecAdapter(prefill_argmax=0)
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[1, 2, 3]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    params = SamplingParams(temperature=0.7, max_tokens=4)
    with pytest.raises(NotImplementedError, match=r"greedy-only"):
        list(engine.generate("hi", params))


def test_draft_engine_reset_called_on_finally() -> None:
    adapter = _ScriptedSpecAdapter(prefill_argmax=0)
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine()  # no proposals → spec branch never enters
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    list(engine.generate("hi", _greedy(max_tokens=1)))
    assert drafter.reset_calls == 1
    assert drafter.reset_req_ids == ["req-0"]
    # Run a second generate; reset must run again.
    adapter2 = _ScriptedSpecAdapter(prefill_argmax=0)
    engine2 = Engine(adapter2, _TrackedKVManager(), draft_engine=drafter)
    list(engine2.generate("hi", _greedy(max_tokens=1)))
    assert drafter.reset_calls == 2
    assert drafter.reset_req_ids == ["req-0", "req-0"]


# --- D-021 step 5 sub-unit (e) slice 2 — recurrent rollback wiring ---------


class _RecurrentScriptedSpecAdapter(_ScriptedSpecAdapter):
    """``_ScriptedSpecAdapter`` plus the four ``SpecRecurrentRollbackAdapter``
    helpers, all bookkeeping-only.

    Records every snapshot / commit_state / rollback_state / free_state
    invocation and every ``decode_step_multi`` input so tests can pin the
    recurrent path's call ordering and replay-slice contents.
    """

    def __init__(
        self,
        *,
        prefill_argmax: int,
        decode_argmaxes: Sequence[int] = (),
        verify_logits: Sequence[Sequence[int]] = (),
    ) -> None:
        super().__init__(
            prefill_argmax=prefill_argmax,
            decode_argmaxes=decode_argmaxes,
            verify_logits=verify_logits,
        )
        self.snapshot_calls: list[str] = []
        self.commit_state_calls: list[tuple[str, int]] = []
        self.rollback_state_calls: list[tuple[str, int]] = []
        self.free_state_calls: list[str] = []
        self.verify_inputs_seen: list[list[int]] = []

    def decode_step_multi(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        # mx.array.tolist() on a 1-D int32 array returns list[int]; the
        # type is inferred broadly so coerce explicitly for the type
        # checker.
        flat = [int(tokens[i].item()) for i in range(int(tokens.size))]
        self.verify_inputs_seen.append(flat)
        return super().decode_step_multi(tokens, kv_handle)

    # --- SpecRecurrentRollbackAdapter Protocol ---

    def snapshot_pre_draft_state(self, req_id: str) -> Any:
        # Nested-window guard mirrors Qwen3_5Adapter — defends the
        # invariant that commit_state / rollback_state must close the
        # window before the next snapshot.
        if req_id in self.snapshot_calls and (
            self._open_window(req_id)
        ):
            raise RuntimeError(
                f"nested pre-draft snapshot for {req_id!r}"
            )
        self.snapshot_calls.append(req_id)
        return object()

    def commit_state(self, req_id: str, n_accepted: int) -> None:
        self.commit_state_calls.append((req_id, int(n_accepted)))

    def rollback_state(self, req_id: str, n_reject: int) -> None:
        self.rollback_state_calls.append((req_id, int(n_reject)))

    def free_state(self, req_id: str) -> None:
        self.free_state_calls.append(req_id)

    def _open_window(self, req_id: str) -> bool:
        # Return True if the most recent snapshot for this req_id
        # has not yet been closed by commit_state / rollback_state.
        commits = sum(1 for r, _ in self.commit_state_calls if r == req_id)
        rollbacks = sum(
            1 for r, _ in self.rollback_state_calls if r == req_id
        )
        snapshots = sum(1 for r in self.snapshot_calls if r == req_id)
        return snapshots > commits + rollbacks


def test_recurrent_full_accept_calls_commit_state_no_replay() -> None:
    """Full accept: verify forward already at the committed boundary;
    snapshot is dropped via commit_state, no rollback / replay."""
    adapter = _RecurrentScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 250]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=5)))
    assert out == [100, 200, 201, 202, 250]

    assert adapter.snapshot_calls == ["req-0"]
    assert adapter.commit_state_calls == [("req-0", 3)]
    assert adapter.rollback_state_calls == []
    # Single decode_step_multi call (verify only — no replay).
    assert adapter.decode_multi_calls == 1
    assert kv.rollback_calls == []
    # free_state runs in the generate finally tail.
    assert adapter.free_state_calls == ["req-0"]


def test_recurrent_partial_accept_trims_full_verify_then_replays() -> None:
    """Partial accept: trim attention KV by ``draft_count + 1`` (drop the
    full verify), restore recurrent state, then replay
    ``decode_step_multi`` over ``verify_input[:1 + yielded_count]``. No
    second KV trim. F-3 / F-3a in the orientation."""
    # γ = 3. Drafts = [200, 201, 202]. argmax targets =
    # [200, 999, 0, 0] → accepted_len = 1 (draft[0] matches, draft[1]
    # rejected). yielded_count = 1, un_committed = 2. bonus = 999.
    adapter = _RecurrentScriptedSpecAdapter(
        prefill_argmax=100,
        # First call: verify forward over [100, 200, 201, 202].
        # Second call: replay over [100, 200] — values discarded but
        # the synthetic adapter still consumes a row from the queue.
        verify_logits=[[200, 999, 0, 0], [0, 0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=3)))
    assert out == [100, 200, 999]

    # Recurrent path call sequence.
    assert adapter.snapshot_calls == ["req-0"]
    assert adapter.rollback_state_calls == [("req-0", 2)]  # un_committed
    assert adapter.commit_state_calls == []
    # Single KV rollback by ``draft_count + 1 = 4`` (no second trim).
    assert kv.rollback_calls == [("req-0", 4)]
    # Two decode_step_multi calls: verify + replay.
    assert adapter.decode_multi_calls == 2
    # Replay slice = [anchor, accepted_draft] = [100, 200].
    assert adapter.verify_inputs_seen == [
        [100, 200, 201, 202],
        [100, 200],
    ]
    assert adapter.free_state_calls == ["req-0"]


def test_recurrent_full_reject_replays_anchor_only() -> None:
    """Full reject (yielded_count == 0): replay slice is just the
    anchor token (length 1)."""
    adapter = _RecurrentScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[777, 0, 0, 0], [0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=2)))
    assert out == [100, 777]

    assert adapter.snapshot_calls == ["req-0"]
    assert adapter.rollback_state_calls == [("req-0", 3)]
    # KV trim by draft_count + 1 = 4; replay over [anchor] only.
    assert kv.rollback_calls == [("req-0", 4)]
    assert adapter.verify_inputs_seen == [
        [100, 200, 201, 202],
        [100],
    ]


def test_recurrent_max_tokens_cut_replays_over_yielded_count() -> None:
    """When ``max_tokens`` cuts the yield short of the verifier's
    accept, replay slice keys on ``yielded_count``, not
    ``accepted_len``. Mirrors the existing KV rollback contract."""
    # γ = 3, all drafts would accept. max_tokens = 3 caps at
    # [anchor=100, draft0=200, draft1=201] → yielded_count = 2.
    # accepted_len = 3 (verifier accepted all), un_committed = 1.
    adapter = _RecurrentScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 202, 250], [0, 0, 0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=3)))
    assert out == [100, 200, 201]

    # un_committed = draft_count - yielded_count = 3 - 2 = 1.
    assert adapter.rollback_state_calls == [("req-0", 1)]
    # KV trim by draft_count + 1 = 4 (whole verify), no second trim.
    assert kv.rollback_calls == [("req-0", 4)]
    # Replay slice is verify_input[:1 + yielded_count] = [100, 200, 201].
    assert adapter.verify_inputs_seen == [
        [100, 200, 201, 202],
        [100, 200, 201],
    ]


def test_recurrent_stop_hit_mid_yield_still_runs_recurrent_rollback() -> None:
    """F-6: terminal cycles (stop_hit on a yielded draft) take the same
    recurrent rollback path so the end-of-cycle invariant
    ``recurrent state at L + 1 + yielded_count`` holds uniformly."""
    # γ = 3. Drafts = [200, 99, 202]. argmax targets =
    # [200, 99, 202, 250] → all 3 would accept, but draft[1] = 99 is
    # in stop_token_ids → engine yields anchor + draft0 + draft1 then
    # stops. yielded_count = 2, un_committed = 1. No bonus sampled.
    adapter = _RecurrentScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 99, 202, 250], [0, 0, 0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 99, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    params = SamplingParams(
        temperature=0.0, max_tokens=8, stop_token_ids=(99,)
    )
    out = _collect(engine.generate("hi", params))
    assert out == [100, 200, 99]

    # Recurrent rollback runs even though we are about to terminate.
    assert adapter.rollback_state_calls == [("req-0", 1)]
    assert kv.rollback_calls == [("req-0", 4)]
    assert adapter.commit_state_calls == []
    assert adapter.verify_inputs_seen == [
        [100, 200, 99, 202],
        [100, 200, 99],
    ]
    # free_state still runs in the finally tail.
    assert adapter.free_state_calls == ["req-0"]


def test_recurrent_free_state_called_when_spec_branch_never_enters() -> None:
    """The cleanup tail calls ``free_state`` regardless of whether the
    spec branch entered the cycle. Idempotent under
    ``_pre_draft_snapshots.pop(..., None)``."""
    adapter = _RecurrentScriptedSpecAdapter(
        prefill_argmax=42,
        decode_argmaxes=[7],
    )
    kv = _TrackedKVManager()
    # Drafter returns no proposals — engine takes the single-token
    # decode_step path, never calls snapshot / rollback_state.
    drafter = _ScriptedDraftEngine()
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=2)))
    assert out == [42, 7]

    assert adapter.snapshot_calls == []
    assert adapter.commit_state_calls == []
    assert adapter.rollback_state_calls == []
    # Cleanup still fires.
    assert adapter.free_state_calls == ["req-0"]


def test_recurrent_two_cycles_full_accept_then_partial_does_not_nest() -> None:
    """Multi-cycle sanity: full-accept cycle drops snapshot via
    commit_state, so the next cycle's snapshot does not raise the
    nested-window guard."""
    # Cycle 1: full accept (γ=2 drafts). Bonus sampled.
    # Cycle 2: partial accept (1 of 2 accepted). Stop-hit on bonus.
    adapter = _RecurrentScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[
            [200, 201, 250],         # cycle 1 verify (full accept)
            [300, 999, 0],           # cycle 2 verify (1 accepted, reject draft[1]=301; bonus=999)
            [0, 0],                  # cycle 2 replay [250, 300]
        ],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201], [300, 301]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=3)
    params = SamplingParams(
        temperature=0.0, max_tokens=10, stop_token_ids=(999,)
    )
    out = _collect(engine.generate("hi", params))
    # Cycle 1: anchor 100, drafts 200, 201, bonus 250 (next cycle anchor).
    # Cycle 2: drafts 300 (accepted), bonus 999 (stop).
    assert out == [100, 200, 201, 250, 300, 999]

    assert adapter.snapshot_calls == ["req-0", "req-0"]
    assert adapter.commit_state_calls == [("req-0", 2)]    # cycle 1
    assert adapter.rollback_state_calls == [("req-0", 1)]  # cycle 2
    # Cycle 1 KV: no rollback. Cycle 2 KV: rollback by draft_count + 1 = 3.
    assert kv.rollback_calls == [("req-0", 3)]
    assert adapter.free_state_calls == ["req-0"]


def test_non_recurrent_adapter_path_unchanged() -> None:
    """Regression: non-recurrent adapters take the existing path.
    Single KV rollback by ``un_committed``; no recurrent helper calls
    invoked (and the engine does not raise even though those helpers
    don't exist on the adapter)."""
    # Same scenario as test_spec_on_partial_accept_rolls_back_and_yields_bonus.
    adapter = _ScriptedSpecAdapter(
        prefill_argmax=100,
        verify_logits=[[200, 201, 999, 0]],
    )
    kv = _TrackedKVManager()
    drafter = _ScriptedDraftEngine(proposals=[[200, 201, 202]])
    engine = Engine(adapter, kv, draft_engine=drafter, verify_k=4)
    out = _collect(engine.generate("hi", _greedy(max_tokens=4)))
    assert out == [100, 200, 201, 999]
    # Existing single rollback by un_committed = 1 (NOT draft_count + 1 = 4).
    assert kv.rollback_calls == [("req-0", 1)]
    # Single decode_step_multi (no replay).
    assert adapter.decode_multi_calls == 1
