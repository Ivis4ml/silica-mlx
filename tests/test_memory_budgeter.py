"""Tests for silica.scheduler.budget — MemoryBudgeter (P-2 Unit #15).

Pure policy unit. Decisions are exercised against a real PagedKVCache +
RadixPrefixCache pair so that eviction-budget reasoning is tested against
the same refcount machinery the scheduler will actually use, but the
budgeter itself never mutates those objects (decide / apply separation).
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kvcache.paged import PagedKVCache
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import PagedPrefixBlockStore, SyntheticPrefixBlockStore
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


def test_for_adapter_prefers_kv_layout_bytes_per_token_total_when_set() -> None:
    """P-3-D4: when ``KVLayout.bytes_per_token_total`` is populated
    (heterogeneous-shape adapter, e.g. Gemma4's sliding + full layer
    mix), ``for_adapter`` uses the field verbatim and ignores the
    ``num_layers × n_kv_heads × head_dim`` product. This is the whole
    point of the field: adapter-authored per-kind sums override the
    naive pre-D4 formula."""
    import mlx.core as mx

    from silica.models.adapter import (
        AttentionKind,
        AttentionPattern,
        KVLayout,
        ModelConfig,
    )

    class _HeterogeneousAdapter:
        config = ModelConfig(
            model_name="hetero",
            num_layers=3,
            hidden_size=16,
            vocab_size=8,
        )
        _layout = KVLayout(
            num_layers=3,
            n_kv_heads=2,
            head_dim=4,
            dtype=mx.float16,
            # Naive formula would yield 2*3*2*4*2 = 96; the field
            # deliberately disagrees to prove it is load-bearing.
            bytes_per_token_total=50,
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
        _HeterogeneousAdapter(),  # type: ignore[arg-type]
        prefix_cache=pc,
        weights_bytes=0,
        cap_bytes=10_000,
    )
    assert b.bytes_per_token == 50


def test_for_adapter_falls_back_to_layout_formula_when_total_is_none() -> (
    None
):
    """Regression guard: adapters that leave
    ``bytes_per_token_total=None`` (the default — plain Qwen3 and
    Qwen3.5 dense do this, since their per-layer KV shapes are
    homogeneous) get the pre-D4 derivation unchanged."""
    import mlx.core as mx

    from silica.models.adapter import (
        AttentionKind,
        AttentionPattern,
        KVLayout,
        ModelConfig,
    )

    class _HomogeneousAdapter:
        config = ModelConfig(
            model_name="homo",
            num_layers=3,
            hidden_size=16,
            vocab_size=8,
        )
        _layout = KVLayout(
            num_layers=3,
            n_kv_heads=2,
            head_dim=4,
            dtype=mx.float16,
            # Explicitly default — belt and braces.
            bytes_per_token_total=None,
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
        _HomogeneousAdapter(),  # type: ignore[arg-type]
        prefix_cache=pc,
        weights_bytes=0,
        cap_bytes=10_000,
    )
    # Naive formula: 2 * 3 layers * 2 heads * 4 head_dim * 2 bytes = 96.
    assert b.bytes_per_token == 96


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


# --- 16d-2a: admit() purity + apply_admit() reservation-timing (B-8) ---
#
# B-8 (prep doc §5) says: within one batcher admit loop, every admission's
# ``apply_admit`` must commit to ``reserved_bytes`` BEFORE the next iteration's
# ``admit()`` runs. At the budgeter layer this decomposes into two primitive
# properties — (a) ``admit()`` is pure (no state mutation), and (b) a prior
# ``apply_admit()`` call IS visible to the next ``admit()``'s headroom math.
# If either fails, a batch of small pendings would each see the same initial
# headroom and systematically over-admit. These tests pin those primitives.


def test_admit_does_not_mutate_budgeter_state() -> None:
    """``admit()`` is pure: calling it any number of times leaves
    ``reserved_bytes`` / ``active_requests`` unchanged, and the decision
    is reproducible for identical inputs."""
    b, _, _ = _make(cap_bytes=10_000)
    before_reserved = b.reserved_bytes()
    before_active = b.active_requests()

    d1 = b.admit("r1", n_prompt=4, max_tokens=4)
    d2 = b.admit("r2", n_prompt=4, max_tokens=4)
    d3 = b.admit("r1", n_prompt=4, max_tokens=4)

    assert b.reserved_bytes() == before_reserved
    assert b.active_requests() == before_active
    assert isinstance(d1, AdmitDecision)
    assert isinstance(d2, AdmitDecision)
    assert isinstance(d3, AdmitDecision)
    # Same inputs → identical decision shape (dataclasses are value-equal).
    assert d1 == d2 == d3


def test_admit_same_req_id_twice_without_apply_admit_is_allowed() -> None:
    """Duplicate-req_id detection hinges on ``_reserved_per_req``, not on
    any in-flight admission-intent tracking. Two back-to-back ``admit()``
    calls for the same req_id without an intervening ``apply_admit`` must
    NOT raise — this lets a batcher re-evaluate a pending after evict /
    preempt without fabricating a fresh req_id."""
    b, _, _ = _make(cap_bytes=10_000)
    b.admit("r1", n_prompt=4, max_tokens=4)
    # No apply_admit between calls.
    b.admit("r1", n_prompt=4, max_tokens=4)  # must not raise


def test_apply_admit_updates_headroom_visible_to_next_admit() -> None:
    """B-8 positive: once ``apply_admit`` commits a reservation, the next
    ``admit()`` call sees the reduced headroom and its decision reflects
    that. Sizing: cap fits exactly one worst-case request; after
    apply_admit on the first, a second ``admit()`` cannot fit as-is and
    falls through to preempt."""
    b, _, _ = _make(cap_bytes=10_000)
    per_req = b.worst_case_bytes(4, 4)
    tight = MemoryBudgeter(
        prefix_cache=b._pc,  # type: ignore[attr-defined]
        weights_bytes=0,
        bytes_per_token=b.bytes_per_token,
        block_size=b._block_size,  # type: ignore[attr-defined]
        cap_bytes=per_req,
    )

    d1 = tight.admit("r1", n_prompt=4, max_tokens=4)
    assert isinstance(d1, AdmitDecision)
    tight.apply_admit("r1", d1.reserved_delta)
    assert tight.headroom_bytes() == 0

    # Next admit sees zero headroom → fit-as-is fails, preempt picks r1.
    d2 = tight.admit("r2", n_prompt=4, max_tokens=4)
    assert isinstance(d2, AdmitAfterPreemptDecision)
    assert d2.preempt_req_id == "r1"


def test_admit_without_apply_admit_sees_unchanged_headroom() -> None:
    """B-8 negative — the bug the batcher MUST prevent by interleaving
    ``apply_admit``. If the caller runs ``admit`` multiple times without
    committing, every decision sees the original headroom and every call
    returns ``AdmitDecision``. Same setup as the positive test, but no
    intervening ``apply_admit``; both admissions "fit as-is" even though
    their sum exceeds the cap. The budgeter is correct to report this —
    ``admit()`` is a pure decision over the current committed state —
    and the batcher is responsible for calling ``apply_admit`` between
    iterations."""
    b, _, _ = _make(cap_bytes=10_000)
    per_req = b.worst_case_bytes(4, 4)
    tight = MemoryBudgeter(
        prefix_cache=b._pc,  # type: ignore[attr-defined]
        weights_bytes=0,
        bytes_per_token=b.bytes_per_token,
        block_size=b._block_size,  # type: ignore[attr-defined]
        cap_bytes=per_req,
    )

    d1 = tight.admit("r1", n_prompt=4, max_tokens=4)
    d2 = tight.admit("r2", n_prompt=4, max_tokens=4)
    d3 = tight.admit("r3", n_prompt=4, max_tokens=4)

    assert isinstance(d1, AdmitDecision)
    assert isinstance(d2, AdmitDecision)
    assert isinstance(d3, AdmitDecision)
    # No reservation was committed — headroom never moved.
    assert tight.reserved_bytes() == 0
    assert tight.headroom_bytes() == per_req


# =============================================================================
# P-5-A.2 — account_prefix_residency three-mode accounting (§4.7)
# =============================================================================


def _synthetic_pc(
    block_size: int = 16,
) -> tuple[RadixPrefixCache, SyntheticPrefixBlockStore]:
    """Build a RadixPrefixCache over a bare SyntheticPrefixBlockStore
    (pass-through — no codec installed). Returns ``(pc, store)`` —
    mypy cannot narrow ``pc.store: PrefixBlockStore`` back to the
    concrete synthetic type, so returning the store avoids casts at
    every call site that reads ``resident_bytes`` /
    ``resident_bytes_per_block``.
    """
    store = SyntheticPrefixBlockStore(block_size=block_size)
    pc = RadixPrefixCache(block_size=block_size, store=store)
    return pc, store


def _synthetic_pc_with_codec(
    block_size: int = 16, *, codec: object
) -> tuple[RadixPrefixCache, SyntheticPrefixBlockStore]:
    store = SyntheticPrefixBlockStore(
        block_size=block_size, codec=codec  # type: ignore[arg-type]
    )
    pc = RadixPrefixCache(block_size=block_size, store=store)
    return pc, store


def _register_one_detached_block(
    pc: RadixPrefixCache, *, n_kv_heads: int = 2, head_dim: int = 64
) -> int:
    """Allocate + retain + register one detached block on the store; returns
    the block id. Shape matches what the codec expects:
    ``(1, n_kv_heads, block_size, head_dim)``."""
    store = pc.store
    bid = store.allocate_id()
    store.retain_source(bid)
    # Two layers' worth of per-layer (K, V) tuples.
    n_layers = 2
    layer_kv: list[tuple[mx.array, mx.array]] = []
    for _ in range(n_layers):
        shape = (1, n_kv_heads, store.block_size, head_dim)
        k = mx.zeros(shape, dtype=mx.float16)
        v = mx.zeros(shape, dtype=mx.float16)
        layer_kv.append((k, v))
    store.register_detached(bid, layer_kv)
    return int(bid)


# --- Mode (A): account_prefix_residency=False (P-4.5 regression path) ---


def test_mode_a_headroom_is_cap_minus_weights_minus_reserved_byte_for_byte() -> None:
    """§4.7 mode (A): ``account_prefix_residency=False`` keeps headroom
    at the P-4.5 formula regardless of store state. Regression-lock
    for the P-4.5 close sweep."""
    pc, store = _synthetic_pc(block_size=16)
    budgeter = MemoryBudgeter(
        prefix_cache=pc,
        weights_bytes=1000,
        bytes_per_token=8,
        block_size=16,
        cap_bytes=5000,
        account_prefix_residency=False,
    )
    # Register a detached block to prove residency is NOT charged.
    _register_one_detached_block(pc, n_kv_heads=2, head_dim=64)
    # Headroom should still equal cap - weights - reserved = 4000.
    assert budgeter.headroom_bytes() == 4000


def test_mode_a_evict_bytes_per_block_is_fp16_baseline() -> None:
    """Mode (A) evict-shortfall uses the fp16 ``bytes_per_token ×
    block_size`` formula regardless of any codec residency
    information the store might expose."""
    pc, store = _synthetic_pc(block_size=16)
    budgeter = MemoryBudgeter(
        prefix_cache=pc,
        weights_bytes=0,
        bytes_per_token=8,
        block_size=16,
        cap_bytes=10_000,
        account_prefix_residency=False,
    )
    assert budgeter._evict_bytes_per_block() == 8 * 16  # type: ignore[attr-defined]


# --- Mode (B): account_prefix_residency=True, IdentityCodec ---


def test_mode_b_charges_fp16_prefix_residency() -> None:
    """§4.7 mode (B): with ``account_prefix_residency=True`` + a
    synthetic store (pass-through counts raw tensor bytes), headroom
    subtracts honest fp16 per-payload bytes."""
    pc, store = _synthetic_pc(block_size=16)
    budgeter = MemoryBudgeter(
        prefix_cache=pc,
        weights_bytes=0,
        bytes_per_token=8,
        block_size=16,
        cap_bytes=1_000_000,
        account_prefix_residency=True,
    )
    # Empty store → no residency yet.
    assert budgeter.headroom_bytes() == 1_000_000

    # Register one block; headroom drops by the exact per-payload byte sum.
    _register_one_detached_block(pc, n_kv_heads=2, head_dim=64)
    prefix = store.resident_bytes()
    assert prefix > 0
    assert budgeter.headroom_bytes() == 1_000_000 - prefix


def test_mode_b_identity_codec_honest_fp16_residency() -> None:
    """Under ``codec=IdentityCodec(...)`` shorthand, the store wraps
    tensors in ``RawFp16Payload`` whose ``resident_bytes`` equals
    ``.nbytes`` of the wrapped fp16 tensor. Mode (B) charges these
    honestly — the count matches the pass-through path."""
    from silica.kvcache.codec import IdentityCodec

    n_kv_heads, head_dim, block_size = 2, 64, 16
    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    pc, store = _synthetic_pc_with_codec(block_size=block_size, codec=ic)
    budgeter = MemoryBudgeter(
        prefix_cache=pc,
        weights_bytes=0,
        bytes_per_token=8,
        block_size=block_size,
        cap_bytes=1_000_000,
        account_prefix_residency=True,
    )
    _register_one_detached_block(pc, n_kv_heads=n_kv_heads, head_dim=head_dim)
    # Identity payload bytes == raw .nbytes; per-block = 2 layers × 2
    # sides × n_kv_heads × block_size × head_dim × 2 (fp16 bytes).
    expected_per_block = 2 * 2 * n_kv_heads * block_size * head_dim * 2
    assert store.resident_bytes() == expected_per_block
    assert budgeter.headroom_bytes() == 1_000_000 - expected_per_block


# --- Mode (C): account_prefix_residency=True, BlockTQ compressed ---


def test_mode_c_blocktq_residency_is_compressed() -> None:
    """§4.7 mode (C): BlockTQ codec compresses K/V; store residency
    and thus headroom subtraction reflect compressed bytes, producing
    larger headroom than mode (B) at the same cap."""
    from silica.vq import BlockTurboQuantMSE

    n_kv_heads, head_dim, block_size = 2, 64, 16

    block_tq = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=32,
        num_bits=4,
    )
    pc_compressed, store_compressed = _synthetic_pc_with_codec(
        block_size=block_size, codec=block_tq
    )
    budgeter_c = MemoryBudgeter(
        prefix_cache=pc_compressed,
        weights_bytes=0,
        bytes_per_token=8,
        block_size=block_size,
        cap_bytes=1_000_000,
        account_prefix_residency=True,
    )
    _register_one_detached_block(pc_compressed, n_kv_heads=n_kv_heads, head_dim=head_dim)
    compressed_bytes = store_compressed.resident_bytes()

    # Mode (B) baseline for comparison.
    from silica.kvcache.codec import IdentityCodec

    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    pc_ident, store_ident = _synthetic_pc_with_codec(block_size=block_size, codec=ic)
    budgeter_b = MemoryBudgeter(
        prefix_cache=pc_ident,
        weights_bytes=0,
        bytes_per_token=8,
        block_size=block_size,
        cap_bytes=1_000_000,
        account_prefix_residency=True,
    )
    _register_one_detached_block(pc_ident, n_kv_heads=n_kv_heads, head_dim=head_dim)
    identity_bytes = store_ident.resident_bytes()

    # BlockTQ residency < IdentityCodec residency on the same block.
    assert compressed_bytes < identity_bytes
    # Headroom under mode (C) > headroom under mode (B) at same cap.
    assert budgeter_c.headroom_bytes() > budgeter_b.headroom_bytes()


def test_mode_c_evict_bytes_per_block_is_compressed() -> None:
    """Mode (C) ``_evict_bytes_per_block`` reads from
    ``store.resident_bytes_per_block()`` — honest compressed bytes,
    smaller than the fp16 baseline.

    Register one block first so ``num_layers`` is learned and the
    store returns a concrete per-block figure (not ``None``). Also
    anchor an orthogonal check: the returned value equals
    ``store.resident_bytes() // num_blocks`` — bytes freed per block
    matches the honest eviction unit.
    """
    from silica.vq import BlockTurboQuantMSE

    n_kv_heads, head_dim, block_size = 2, 64, 16
    n_layers = 2
    block_tq = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=32,
        num_bits=4,
    )
    pc, store = _synthetic_pc_with_codec(block_size=block_size, codec=block_tq)
    _register_one_detached_block(pc, n_kv_heads=n_kv_heads, head_dim=head_dim)
    # bytes_per_token set to a fp16-equivalent so the fallback would be
    # the fp16 baseline; mode (C) should read smaller compressed number.
    bpt_fp16 = 2 * n_kv_heads * head_dim * 2  # K + V × n_kv_heads × head_dim × 2 bytes
    budgeter = MemoryBudgeter(
        prefix_cache=pc,
        weights_bytes=0,
        bytes_per_token=bpt_fp16,
        block_size=block_size,
        cap_bytes=1_000_000,
        account_prefix_residency=True,
    )
    fp16_baseline = bpt_fp16 * block_size * n_layers
    evict_bytes = budgeter._evict_bytes_per_block()  # type: ignore[attr-defined]
    compressed_per_block = store.resident_bytes_per_block()
    assert compressed_per_block is not None
    assert evict_bytes == compressed_per_block
    # Orthogonal: honest per-eviction-unit bytes == total / num_blocks.
    num_blocks = len(store.live_block_ids())
    assert compressed_per_block == store.resident_bytes() // num_blocks
    assert evict_bytes < fp16_baseline


# --- Capability fallback: paged store lacks resident_bytes ---


def test_paged_store_falls_back_to_mode_a_like() -> None:
    """``PagedPrefixBlockStore`` does not implement ``resident_bytes``;
    the ``hasattr`` capability check falls back to zero residency, so
    mode (B) with a paged-backed cache behaves like mode (A) on the
    headroom math. Regression-lock for the pre-P-5 budgeter tests
    above which use this code path."""
    b, _, _ = _make(cap_bytes=5000, weights_bytes=1000)
    # Default account_prefix_residency=True (from for_adapter default),
    # but paged store lacks the capability → fallback to 0.
    # This test uses the module's _make() which returns a budgeter
    # with the paged-store path.
    assert b._account_prefix_residency is True  # type: ignore[attr-defined]
    assert b.headroom_bytes() == 4000  # cap - weights, no residency


# --- D-003: reservation math stays fp16 regardless of mode ---


def test_worst_case_bytes_is_fp16_under_all_modes() -> None:
    """D-003: reservation (``worst_case_bytes``) is always fp16
    worst-case, regardless of codec or ``account_prefix_residency``.
    Active-KV is fp16-scratched per mlx-lm SDPA's contract."""
    from silica.kvcache.codec import IdentityCodec
    from silica.vq import BlockTurboQuantMSE

    n_kv_heads, head_dim, block_size = 2, 64, 16

    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    btq = BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=32,
        num_bits=4,
    )

    pc_a, _ = _synthetic_pc(block_size=block_size)
    pc_b, _ = _synthetic_pc_with_codec(block_size=block_size, codec=ic)
    pc_c, _ = _synthetic_pc_with_codec(block_size=block_size, codec=btq)
    configs = [
        ("mode_a_passthrough", pc_a, False),
        ("mode_b_identity", pc_b, True),
        ("mode_c_blocktq", pc_c, True),
    ]
    for label, pc, flag in configs:
        budgeter = MemoryBudgeter(
            prefix_cache=pc,
            weights_bytes=0,
            bytes_per_token=8,
            block_size=block_size,
            cap_bytes=10_000,
            account_prefix_residency=flag,
        )
        # n_prompt + max_tokens = 10, bpt = 8 → worst = 80, fp16-baseline.
        assert budgeter.worst_case_bytes(5, 5) == 80, f"{label}: worst_case_bytes changed"


# --- resident_bytes_per_block on the store ---


def test_synthetic_store_resident_bytes_per_block_is_none_passthrough() -> None:
    """Pass-through store (no codec) returns ``None`` — caller falls
    back to its own fp16 formula."""
    pc, store = _synthetic_pc(block_size=16)
    assert store.resident_bytes_per_block() is None


def test_synthetic_store_resident_bytes_per_block_is_none_before_any_register() -> None:
    """Even with codecs bound, the method returns ``None`` before any
    ``register_detached`` call — ``num_layers`` has not been learned
    yet. Safe because no evictable blocks exist at that point; the
    budgeter falls back to the fp16 baseline which is never
    consulted in the absence of evictable blocks."""
    from silica.kvcache.codec import IdentityCodec

    n_kv_heads, head_dim, block_size = 2, 64, 16
    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    _pc, store = _synthetic_pc_with_codec(block_size=block_size, codec=ic)
    assert store.resident_bytes_per_block() is None


def test_synthetic_store_resident_bytes_per_block_includes_all_layers() -> None:
    """Regression lock for P-5-A.2 review finding H-1: the eviction
    unit is one radix ``block_id`` (covers all layers), not one
    layer's K+V. ``resident_bytes_per_block`` must therefore return
    ``num_layers × (k_codec.resident_bytes(1) + v_codec.resident_bytes(1))``.

    Invariant: when one block is registered,
    ``resident_bytes_per_block() == resident_bytes() // len(live_block_ids())``.
    Under the pre-fix implementation this was False — ``resident_bytes``
    summed across layers while ``resident_bytes_per_block`` returned
    single-layer bytes, off by a factor of num_layers.
    """
    from silica.kvcache.codec import IdentityCodec

    n_kv_heads, head_dim, block_size = 2, 64, 16
    n_layers = 2
    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    pc, store = _synthetic_pc_with_codec(block_size=block_size, codec=ic)
    _register_one_detached_block(pc, n_kv_heads=n_kv_heads, head_dim=head_dim)

    per_block = store.resident_bytes_per_block()
    assert per_block is not None
    total = store.resident_bytes()
    num_blocks = len(store.live_block_ids())
    assert num_blocks == 1
    # Honest eviction-unit arithmetic — the whole radix block id frees
    # these many bytes when evict_until(1) runs.
    assert per_block == total // num_blocks

    # Also pin the value to the expected formula for this shape:
    # num_layers × 2 sides × (block_size × n_kv_heads × head_dim × 2 bytes).
    expected = n_layers * 2 * (block_size * n_kv_heads * head_dim * 2)
    assert per_block == expected


def test_synthetic_store_register_detached_rejects_heterogeneous_num_layers() -> None:
    """P-5-A scope is homogeneous-shape models (opening §2.6); the
    store enforces constant ``num_layers`` across all
    ``register_detached`` calls. Heterogeneous support is a P-5-A
    follow-up, not a silent wrong-answer path."""
    from silica.kvcache.codec import IdentityCodec

    n_kv_heads, head_dim, block_size = 2, 64, 16
    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    pc, store = _synthetic_pc_with_codec(block_size=block_size, codec=ic)
    _register_one_detached_block(pc, n_kv_heads=n_kv_heads, head_dim=head_dim)

    # Second block registered with a different num_layers — must raise.
    bid2 = store.allocate_id()
    store.retain_source(bid2)
    layer_kv_three = [
        (
            mx.zeros((1, n_kv_heads, block_size, head_dim), dtype=mx.float16),
            mx.zeros((1, n_kv_heads, block_size, head_dim), dtype=mx.float16),
        )
        for _ in range(3)  # three layers instead of two
    ]
    with pytest.raises(ValueError, match="layers"):
        store.register_detached(bid2, layer_kv_three)


# --- End-to-end admission with evict-one-block (H-1 regression) ----------


def _insert_one_detached_radix_block(
    pc: RadixPrefixCache,
    *,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    tokens: list[int] | None = None,
) -> int:
    """Insert one block into the RadixPrefixCache via the detached path
    so it shows up as an evictable leaf in
    ``_count_evictable_prefix_blocks``. Returns the radix block_id.

    Distinct from ``_register_one_detached_block`` which only touches
    the store; the radix-path variant is needed for admission tests
    that go through ``MemoryBudgeter.admit``'s evict branch.
    """
    if tokens is None:
        tokens = list(range(pc.block_size))
    shape = (1, n_kv_heads, pc.block_size, head_dim)
    per_layer_block = [
        (
            mx.zeros(shape, dtype=mx.float16),
            mx.zeros(shape, dtype=mx.float16),
        )
        for _ in range(n_layers)
    ]
    # insert_detached takes token-blocks at the outer level, layers inner.
    ids = pc.insert_detached(tokens, [per_layer_block])
    assert len(ids) == 1
    return int(ids[0])


def test_mode_c_admit_evicts_one_block_when_shortfall_between_single_and_all_layer_bytes() -> None:
    """Regression lock for P-5-A.2 review H-1: ``admit`` must
    recognise that evicting one radix block_id frees
    ``num_layers × per-layer-bytes`` — not just one layer's worth.

    Construction: install 1 evictable block whose real eviction value
    is ``B_total = num_layers × (k_codec.resident_bytes(1) +
    v_codec.resident_bytes(1))``. Set cap / weights / reservation so
    shortfall lands between ``B_total // num_layers`` (the wrong per-
    layer value) and ``B_total`` (the correct all-layer value). The
    correct behaviour is ``AdmitAfterEvictDecision(n_blocks=1)``; the
    pre-fix implementation returned ``RejectDecision`` because
    ``blocks_needed = ceil(shortfall / (B_total // num_layers))`` >
    1 > ``evictable_blocks == 1``.
    """
    from silica.kvcache.codec import IdentityCodec

    n_kv_heads, head_dim, block_size = 2, 64, 16
    n_layers = 2
    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    pc, store = _synthetic_pc_with_codec(block_size=block_size, codec=ic)

    # Insert 1 evictable radix block (2 layers).
    _insert_one_detached_radix_block(
        pc, n_kv_heads=n_kv_heads, head_dim=head_dim, n_layers=n_layers
    )
    assert len(store.live_block_ids()) == 1
    b_total = store.resident_bytes_per_block()
    assert b_total is not None
    per_layer = b_total // n_layers
    # shortfall must land strictly between per_layer and b_total so
    # the bug (division by per_layer) requires n_blocks = 2 but the
    # fix (division by b_total) requires n_blocks = 1.
    shortfall_target = per_layer + 1
    assert per_layer < shortfall_target < b_total

    # Build a budgeter with cap sized to produce exactly shortfall_target
    # on a small incoming request. bytes_per_token deliberately chosen
    # so the fp16 baseline formula doesn't accidentally mask the bug
    # (we want the codec path to be load-bearing).
    bpt = 4  # tiny; shortfall is dominated by cap tuning, not bpt
    new_worst = 8 * bpt  # n_prompt=4, max_tokens=4 → 32 bytes
    # Pre-reserve enough to make headroom = new_worst - shortfall_target.
    # Under mode (C) with 1 block registered, prefix residency = b_total.
    cap_bytes = 100_000
    weights_bytes = 0
    budgeter = MemoryBudgeter(
        prefix_cache=pc,
        weights_bytes=weights_bytes,
        bytes_per_token=bpt,
        block_size=block_size,
        cap_bytes=cap_bytes,
        account_prefix_residency=True,
    )
    # Headroom at construction = cap - weights - reserved - prefix_resident.
    # Goal: add a filler reservation so new_worst exceeds headroom by
    # exactly shortfall_target bytes.
    current_headroom = budgeter.headroom_bytes()
    filler = current_headroom - new_worst + shortfall_target
    assert filler > 0
    budgeter.apply_admit("filler", filler)
    # Sanity: shortfall now equals our target.
    actual_shortfall = new_worst - budgeter.headroom_bytes()
    assert actual_shortfall == shortfall_target

    decision = budgeter.admit("incoming", n_prompt=4, max_tokens=4)
    assert isinstance(decision, AdmitAfterEvictDecision), (
        f"expected AdmitAfterEvictDecision(n_blocks=1), got {type(decision).__name__}. "
        f"shortfall={actual_shortfall}, b_total={b_total}, per_layer={per_layer}. "
        f"Pre-fix ``resident_bytes_per_block`` returned per_layer → "
        f"blocks_needed = ceil({actual_shortfall} / {per_layer}) > 1 → reject."
    )
    assert decision.n_blocks == 1, (
        f"expected n_blocks=1 (one evicted radix block frees all "
        f"layers), got {decision.n_blocks}"
    )


def test_synthetic_store_resident_bytes_per_block_under_codec() -> None:
    """Store with codec returns the per-block sum of K+V codec
    ``resident_bytes(1)`` × num_layers — bytes freed by evicting one
    radix block_id."""
    from silica.kvcache.codec import IdentityCodec

    n_kv_heads, head_dim, block_size = 2, 64, 16
    n_layers = 2
    ic = IdentityCodec(
        block_size=block_size, n_kv_heads=n_kv_heads, head_dim=head_dim
    )
    pc, store = _synthetic_pc_with_codec(block_size=block_size, codec=ic)
    # Register one block so num_layers is known.
    _register_one_detached_block(pc, n_kv_heads=n_kv_heads, head_dim=head_dim)
    per_block = store.resident_bytes_per_block()
    # num_layers × 2 sides × block_size × n_kv_heads × head_dim × 2 bytes.
    expected = n_layers * 2 * (block_size * n_kv_heads * head_dim * 2)
    assert per_block == expected
