"""Tests for silica.scheduler.budget — MemoryBudgeter (P-2 Unit #15).

Pure policy unit. Decisions are exercised against a real PagedKVCache +
RadixPrefixCache pair so that eviction-budget reasoning is tested against
the same refcount machinery the scheduler will actually use, but the
budgeter itself never mutates those objects (decide / apply separation).
"""

from __future__ import annotations

import pytest

from silica.kvcache.paged import PagedKVCache
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import PagedPrefixBlockStore
from silica.scheduler.budget import (
    AdmitAfterEvictDecision,
    AdmitAfterPreemptDecision,
    AdmitDecision,
    MemoryBudgeter,
    RejectDecision,
)

BLOCK_SIZE = 4


def _kv(*, num_blocks: int = 32) -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        max_batch_size=8,
        n_kv_heads=1,
        head_dim=4,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        dtype_bytes=2,
    )


def _bytes_per_token_from_kv(kv: PagedKVCache) -> int:
    """Derive the per-token KV bytes that matches the kv's internal math."""
    b = kv.budget()
    assert b.resident_bytes == 0
    # Reserve one block to expose bytes_per_block via the budget delta.
    kv.reserve_for_prefill("__probe__", [0] * BLOCK_SIZE)
    per_block = kv.budget().resident_bytes
    kv.free("__probe__")
    return per_block // BLOCK_SIZE


def _make(
    *,
    num_blocks: int = 32,
    cap_bytes: int = 10_000,
    weights_bytes: int = 0,
) -> tuple[MemoryBudgeter, PagedKVCache, RadixPrefixCache]:
    kv = _kv(num_blocks=num_blocks)
    pc = RadixPrefixCache(
        block_size=kv.block_size, store=PagedPrefixBlockStore(kv)
    )
    bpt = _bytes_per_token_from_kv(kv)
    b = MemoryBudgeter(
        prefix_cache=pc,
        weights_bytes=weights_bytes,
        bytes_per_token=bpt,
        block_size=kv.block_size,
        cap_bytes=cap_bytes,
    )
    return b, kv, pc


# --- constructor validation ---


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("weights_bytes", -1),
        ("bytes_per_token", 0),
        ("bytes_per_token", -1),
        ("cap_bytes", 0),
        ("cap_bytes", -1),
    ],
)
def test_constructor_validates_inputs(field: str, bad_value: int) -> None:
    kv = _kv()
    pc = RadixPrefixCache(
        block_size=kv.block_size, store=PagedPrefixBlockStore(kv)
    )
    kwargs: dict[str, object] = {
        "prefix_cache": pc,
        "weights_bytes": 0,
        "bytes_per_token": 8,
        "block_size": kv.block_size,
        "cap_bytes": 1000,
    }
    kwargs[field] = bad_value
    with pytest.raises(ValueError, match="must be"):
        MemoryBudgeter(**kwargs)  # type: ignore[arg-type]


def test_constructor_validates_block_size() -> None:
    kv = _kv()
    pc = RadixPrefixCache(
        block_size=kv.block_size, store=PagedPrefixBlockStore(kv)
    )
    with pytest.raises(ValueError, match="block_size must be"):
        MemoryBudgeter(
            prefix_cache=pc,
            weights_bytes=0,
            bytes_per_token=8,
            block_size=0,
            cap_bytes=1000,
        )


# --- for_adapter factory (16d-1) ---


def test_for_adapter_derives_bytes_per_token_from_layout() -> None:
    """Factory computes ``bytes_per_token`` as ``2 * num_layers *
    n_kv_heads * head_dim * dtype.size`` and pulls ``block_size``
    from the prefix cache so callers never re-derive either."""
    import mlx.core as mx

    from silica.models.adapter import (
        AttentionKind,
        AttentionPattern,
        KVLayout,
        ModelConfig,
    )

    class _StubAdapter:
        config = ModelConfig(
            model_name="stub",
            num_layers=3,
            hidden_size=16,
            vocab_size=8,
        )
        _layout = KVLayout(
            num_layers=3,
            n_kv_heads=2,
            head_dim=4,
            dtype=mx.float16,
        )
        _pattern = AttentionPattern(
            per_layer=(AttentionKind.GLOBAL,) * 3
        )

        def kv_layout(self) -> KVLayout:
            return self._layout

        def attention_pattern(self) -> AttentionPattern:
            return self._pattern

    kv = _kv()
    pc = RadixPrefixCache(
        block_size=kv.block_size, store=PagedPrefixBlockStore(kv)
    )
    b = MemoryBudgeter.for_adapter(
        _StubAdapter(),  # type: ignore[arg-type]
        prefix_cache=pc,
        weights_bytes=100,
        cap_bytes=10_000,
    )
    # 2 * 3 layers * 2 heads * 4 head_dim * 2 (fp16) = 96
    assert b.bytes_per_token == 96
    # Block size came from the prefix cache, not the adapter.
    assert b._block_size == kv.block_size  # type: ignore[attr-defined]
    # weights + cap pass through unchanged.
    assert b.weights_bytes == 100
    assert b.cap_bytes == 10_000


# --- worst_case_bytes ---


def test_worst_case_bytes_is_sum_times_bytes_per_token() -> None:
    b, _, _ = _make()
    bpt = b.bytes_per_token
    assert b.worst_case_bytes(10, 20) == 30 * bpt


def test_worst_case_bytes_validates_inputs() -> None:
    b, _, _ = _make()
    with pytest.raises(ValueError, match="must be >= 0"):
        b.worst_case_bytes(-1, 0)
    with pytest.raises(ValueError, match="must be >= 0"):
        b.worst_case_bytes(0, -1)


# --- headroom / reserved accounting ---


def test_initial_headroom_is_cap_minus_weights() -> None:
    b, _, _ = _make(cap_bytes=5000, weights_bytes=1000)
    assert b.headroom_bytes() == 4000


def test_apply_admit_records_reservation() -> None:
    b, _, _ = _make()
    b.apply_admit("r", 100)
    assert b.reserved_bytes() == 100
    assert b.active_requests() == ["r"]


def test_release_drops_reservation() -> None:
    b, _, _ = _make()
    b.apply_admit("r", 100)
    b.release("r")
    assert b.reserved_bytes() == 0
    assert b.active_requests() == []


def test_release_unknown_is_idempotent() -> None:
    b, _, _ = _make()
    b.release("never-admitted")
    assert b.reserved_bytes() == 0


def test_apply_admit_negative_delta_raises() -> None:
    b, _, _ = _make()
    with pytest.raises(ValueError, match="must be >= 0"):
        b.apply_admit("r", -5)


def test_apply_admit_same_req_twice_raises() -> None:
    b, _, _ = _make()
    b.apply_admit("r", 10)
    with pytest.raises(ValueError, match="already has an active reservation"):
        b.apply_admit("r", 10)


# --- (1) admit fits as-is ---


def test_admit_returns_admit_when_fits() -> None:
    b, _, _ = _make(cap_bytes=10_000)
    decision = b.admit("r", n_prompt=4, max_tokens=4)
    assert isinstance(decision, AdmitDecision)
    assert decision.reserved_delta == b.worst_case_bytes(4, 4)


def test_admit_raises_if_req_id_already_reserved() -> None:
    b, _, _ = _make()
    b.apply_admit("dup", 10)
    with pytest.raises(ValueError, match="already admitted"):
        b.admit("dup", 4, 4)


# --- (4) reject when nothing fits ---


def test_admit_returns_reject_when_no_room_and_no_recovery() -> None:
    # Tiny cap, no active reqs to preempt, empty prefix cache.
    b, _, _ = _make(cap_bytes=4, weights_bytes=0)
    # Any non-trivial request needs more than 4 bytes.
    decision = b.admit("r", n_prompt=100, max_tokens=100)
    assert isinstance(decision, RejectDecision)
    assert decision.reason == "budget-exhausted"


# --- (3) preempt newest DECODE ---


def test_admit_returns_preempt_when_only_option() -> None:
    """Cap sized so preempting the newest DECODE releases just enough."""
    b, _, _ = _make(cap_bytes=10_000)
    per_req_bytes = b.worst_case_bytes(8, 8)
    # Cap fits two of these requests exactly; we'll fill it with two and
    # then try to admit a third.
    tight = MemoryBudgeter(
        prefix_cache=b._pc,  # type: ignore[attr-defined]
        weights_bytes=0,
        bytes_per_token=b.bytes_per_token,
        block_size=b._block_size,  # type: ignore[attr-defined]
        cap_bytes=per_req_bytes * 2,
    )
    tight.apply_admit("old", per_req_bytes)
    tight.apply_admit("new", per_req_bytes)
    decision = tight.admit("incoming", n_prompt=8, max_tokens=8)
    assert isinstance(decision, AdmitAfterPreemptDecision)
    assert decision.preempt_req_id == "new"  # newest (last in list)


def test_admit_preempt_still_reject_if_freed_bytes_insufficient() -> None:
    # Cap so small that even freeing the only active reservation is not enough.
    b, _, _ = _make(cap_bytes=10)
    b.apply_admit("held", 5)
    decision = b.admit("huge", n_prompt=1_000, max_tokens=1_000)
    assert isinstance(decision, RejectDecision)


# --- (2) evict unpinned prefix blocks ---


def test_admit_returns_evict_when_prefix_cache_has_slack() -> None:
    b, kv, pc = _make(cap_bytes=10_000, num_blocks=32)
    # Install a prefix-cache source block (no live hit).
    blist = kv.reserve_for_prefill("prefix-src", [0, 1, 2, 3])
    pc.insert([0, 1, 2, 3], list(blist.block_ids))
    kv.free("prefix-src")
    # Eat most of the headroom via reservations, leaving a shortfall that
    # one prefix block can cover.
    block_bytes = b.bytes_per_token * BLOCK_SIZE
    bpt = b.bytes_per_token
    # Fill until remaining headroom < new request.
    new_worst = b.worst_case_bytes(8, 8)  # 16 * bpt
    # We want initial headroom so that new_worst exceeds it by <= 1 block.
    initial_headroom = b.headroom_bytes()
    filler = initial_headroom - new_worst + (block_bytes // 2)
    assert filler > 0
    # Reserve that many bytes under an unrelated req id.
    b.apply_admit("filler", filler)

    decision = b.admit("incoming", n_prompt=8, max_tokens=8)
    assert isinstance(decision, AdmitAfterEvictDecision)
    assert decision.n_blocks >= 1
    # Computed need must match (shortfall / block_bytes rounded up).
    shortfall = new_worst - b.headroom_bytes()
    expected = (shortfall + block_bytes - 1) // block_bytes
    assert decision.n_blocks == expected
    # Sanity: expected blocks are available from the prefix cache.
    assert decision.reserved_delta == 16 * bpt


def test_admit_prefers_evict_over_preempt_when_both_possible() -> None:
    """Priority check: if eviction suffices, don't preempt."""
    b, kv, pc = _make(cap_bytes=10_000, num_blocks=32)
    # Install evictable prefix block.
    blist = kv.reserve_for_prefill("src", [0, 1, 2, 3])
    pc.insert([0, 1, 2, 3], list(blist.block_ids))
    kv.free("src")
    # Fill headroom leaving a small shortfall.
    new_worst = b.worst_case_bytes(4, 4)
    block_bytes = b.bytes_per_token * BLOCK_SIZE
    filler = b.headroom_bytes() - new_worst + (block_bytes // 2)
    b.apply_admit("filler", filler)
    b.apply_admit("also-preemptable", 0)  # newer active req

    decision = b.admit("incoming", n_prompt=4, max_tokens=4)
    assert isinstance(decision, AdmitAfterEvictDecision)


def test_admit_skips_evict_when_prefix_cache_empty_falls_to_preempt() -> None:
    b, _, _ = _make(cap_bytes=10_000)
    per_req = b.worst_case_bytes(8, 8)
    tight = MemoryBudgeter(
        prefix_cache=b._pc,  # type: ignore[attr-defined]
        weights_bytes=0,
        bytes_per_token=b.bytes_per_token,
        block_size=b._block_size,  # type: ignore[attr-defined]
        cap_bytes=per_req,
    )
    tight.apply_admit("newest", per_req)
    decision = tight.admit("incoming", n_prompt=8, max_tokens=8)
    assert isinstance(decision, AdmitAfterPreemptDecision)
    assert decision.preempt_req_id == "newest"


# --- FIFO order semantics of active_requests ---


def test_active_requests_preserves_admission_order() -> None:
    b, _, _ = _make()
    for rid in ["a", "b", "c", "d"]:
        b.apply_admit(rid, 100)
    assert b.active_requests() == ["a", "b", "c", "d"]
    b.release("b")
    assert b.active_requests() == ["a", "c", "d"]


def test_preempt_victim_is_newest_after_release() -> None:
    """release() must remove the admit from the FIFO position, not just mark it."""
    b, _, _ = _make(cap_bytes=10_000)
    per_req = b.worst_case_bytes(4, 4)
    tight = MemoryBudgeter(
        prefix_cache=b._pc,  # type: ignore[attr-defined]
        weights_bytes=0,
        bytes_per_token=b.bytes_per_token,
        block_size=b._block_size,  # type: ignore[attr-defined]
        cap_bytes=per_req * 2,
    )
    tight.apply_admit("old", per_req)
    tight.apply_admit("middle", per_req)
    tight.apply_admit("newest", 0)  # degenerate 0-byte reservation
    tight.release("newest")
    # After release, "middle" is now FIFO-newest. admit triggers preempt.
    decision = tight.admit("incoming", n_prompt=4, max_tokens=4)
    assert isinstance(decision, AdmitAfterPreemptDecision)
    assert decision.preempt_req_id == "middle"


# --- integration sanity: apply_admit then release lines up ---


def test_full_admit_apply_release_cycle() -> None:
    b, _, _ = _make(cap_bytes=10_000)
    decision = b.admit("r", n_prompt=4, max_tokens=4)
    assert isinstance(decision, AdmitDecision)
    b.apply_admit("r", decision.reserved_delta)
    assert b.reserved_bytes() == decision.reserved_delta
    b.release("r")
    assert b.reserved_bytes() == 0
    # Now re-admit the same req id — should be allowed.
    decision2 = b.admit("r", n_prompt=4, max_tokens=4)
    assert isinstance(decision2, AdmitDecision)
