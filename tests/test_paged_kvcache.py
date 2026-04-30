"""Tests for silica.kvcache.paged — PagedKVCache (P-2 Unit #13).

Pure bookkeeping unit; no mlx-lm / MLX coupling. Tests exercise the slot +
page + refcount state machines through the I-2 KVManager Protocol plus the
extension methods (``slot_of``, ``mark_active``, ``incref``, ``decref``).
"""

from __future__ import annotations

import pytest

from silica.kvcache.manager import BlockList, KVManager, MemoryBudget, PrefixHit
from silica.kvcache.paged import PagedKVCache, RowState


def _kv(
    *,
    num_layers: int = 2,
    max_batch_size: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 8,
    num_blocks: int = 16,
    block_size: int = 4,  # small for test math
    dtype_bytes: int = 2,
) -> PagedKVCache:
    return PagedKVCache(
        num_layers=num_layers,
        max_batch_size=max_batch_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype_bytes=dtype_bytes,
    )


# --- I-2 Protocol shape ---


def test_satisfies_kv_manager_protocol() -> None:
    assert isinstance(_kv(), KVManager)


def test_exposes_block_size_attribute() -> None:
    kv = _kv(block_size=16)
    assert kv.block_size == 16


# --- constructor validation ---


@pytest.mark.parametrize("field", ["num_layers", "max_batch_size", "num_blocks"])
def test_constructor_rejects_non_positive_sizes(field: str) -> None:
    kwargs = {
        "num_layers": 2,
        "max_batch_size": 4,
        "n_kv_heads": 1,
        "head_dim": 4,
        "num_blocks": 8,
    }
    kwargs[field] = 0
    with pytest.raises(ValueError, match="must be > 0"):
        PagedKVCache(**kwargs)  # type: ignore[arg-type]


# --- initial state ---


def test_initial_all_slots_free_and_all_blocks_free() -> None:
    kv = _kv(max_batch_size=3, num_blocks=5)
    assert kv.row_states() == [RowState.FREE] * 3
    assert kv.available_blocks() == 5
    assert kv.budget() == MemoryBudget(0, 0, 0)


def test_slot_of_raises_for_unknown_request() -> None:
    with pytest.raises(KeyError):
        _kv().slot_of("ghost")


def test_num_tokens_raises_for_unknown_request() -> None:
    with pytest.raises(KeyError):
        _kv().num_tokens("ghost")


# --- reserve_for_prefill ---


def test_reserve_allocates_slot_and_blocks() -> None:
    kv = _kv(block_size=4, num_blocks=8)
    out = kv.reserve_for_prefill("req-0", list(range(10)))  # 10 tokens → 3 blocks
    assert isinstance(out, BlockList)
    assert len(out.block_ids) == 3
    assert kv.slot_of("req-0") == 0
    assert kv.row_states()[0] == RowState.RESERVED
    assert kv.available_blocks() == 8 - 3
    assert kv.num_tokens("req-0") == 10


def test_reserve_rounds_up_blocks() -> None:
    """Exactly-aligned and one-over-aligned both round correctly."""
    kv = _kv(block_size=4, num_blocks=8)
    out = kv.reserve_for_prefill("req-exact", [0, 1, 2, 3])  # 4 tokens → 1 block
    assert len(out.block_ids) == 1
    kv.free("req-exact")
    out = kv.reserve_for_prefill("req-overflow", [0, 1, 2, 3, 4])  # 5 → 2 blocks
    assert len(out.block_ids) == 2


def test_reserve_assigns_lowest_free_slot() -> None:
    kv = _kv(max_batch_size=3, num_blocks=16)
    kv.reserve_for_prefill("a", [0])
    kv.reserve_for_prefill("b", [0])
    assert kv.slot_of("a") == 0
    assert kv.slot_of("b") == 1


def test_reserve_duplicate_req_id_raises() -> None:
    kv = _kv()
    kv.reserve_for_prefill("dup", [0])
    with pytest.raises(ValueError, match="already reserved"):
        kv.reserve_for_prefill("dup", [1])


def test_reserve_fails_when_no_slots_left() -> None:
    kv = _kv(max_batch_size=2, num_blocks=16)
    kv.reserve_for_prefill("a", [0])
    kv.reserve_for_prefill("b", [0])
    with pytest.raises(RuntimeError, match="no free slots"):
        kv.reserve_for_prefill("c", [0])


def test_reserve_fails_when_not_enough_blocks_and_does_not_consume_slot() -> None:
    kv = _kv(max_batch_size=2, num_blocks=2, block_size=4)
    with pytest.raises(RuntimeError, match="not enough free blocks"):
        kv.reserve_for_prefill("too-big", list(range(20)))  # needs 5 blocks
    # Slot should not have been consumed.
    assert kv.row_states() == [RowState.FREE, RowState.FREE]
    # Blocks should not have been touched either.
    assert kv.available_blocks() == 2


# --- mark_active (row lifecycle) ---


def test_mark_active_transitions_row_state() -> None:
    kv = _kv()
    kv.reserve_for_prefill("r", [0])
    assert kv.row_states()[kv.slot_of("r")] == RowState.RESERVED
    kv.mark_active("r")
    assert kv.row_states()[kv.slot_of("r")] == RowState.ACTIVE


def test_mark_active_on_unknown_raises() -> None:
    with pytest.raises(KeyError, match="not reserved"):
        _kv().mark_active("ghost")


# --- append_slot (decode growth) ---


def test_append_slot_no_new_blocks_when_last_has_room() -> None:
    kv = _kv(block_size=4, num_blocks=8)
    kv.reserve_for_prefill("r", [0])  # 1 token, 1 block (3 slots left in block)
    out = kv.append_slot("r", 2)  # fits
    assert out.block_ids == ()
    assert kv.num_tokens("r") == 3
    assert kv.available_blocks() == 7


def test_append_slot_allocates_when_block_overflows() -> None:
    kv = _kv(block_size=4, num_blocks=8)
    kv.reserve_for_prefill("r", list(range(4)))  # 4 tokens → 1 block full
    out = kv.append_slot("r", 1)  # needs a 2nd block
    assert len(out.block_ids) == 1
    assert kv.num_tokens("r") == 5
    assert kv.available_blocks() == 6


def test_append_slot_on_unknown_raises() -> None:
    with pytest.raises(KeyError, match="not reserved"):
        _kv().append_slot("ghost", 1)


def test_append_slot_fails_loud_on_budget_exhaustion() -> None:
    kv = _kv(block_size=4, num_blocks=2)
    kv.reserve_for_prefill("r", list(range(8)))  # 2 blocks consumed
    with pytest.raises(RuntimeError, match="cannot allocate"):
        kv.append_slot("r", 100)  # would need more blocks


# --- free ---


def test_free_returns_slot_and_blocks() -> None:
    kv = _kv(max_batch_size=2, num_blocks=8, block_size=4)
    kv.reserve_for_prefill("r", list(range(10)))  # 3 blocks
    assert kv.available_blocks() == 5
    kv.free("r")
    assert kv.row_states() == [RowState.FREE, RowState.FREE]
    assert kv.available_blocks() == 8


def test_free_is_idempotent_on_unknown() -> None:
    kv = _kv()
    kv.free("never-existed")  # no raise
    assert kv.available_blocks() == kv._num_blocks  # type: ignore[attr-defined]


def test_free_preserves_blocks_pinned_by_prefix_cache() -> None:
    """Option B retention: prefix cache incref before free keeps the block."""
    kv = _kv(num_blocks=4, block_size=4)
    blist = kv.reserve_for_prefill("r", list(range(8)))  # 2 blocks
    pinned = blist.block_ids[0]
    kv.incref(pinned)  # prefix cache claims this as a copy source

    kv.free("r")
    # Pinned block is retained (refcount 1 after free's decref); the
    # unpinned block returned to the pool (refcount 0).
    assert pinned not in kv._free_blocks  # type: ignore[attr-defined]
    assert kv.available_blocks() == 3

    # Releasing the prefix-cache refcount returns the block.
    kv.decref(pinned)
    assert kv.available_blocks() == 4


def test_free_allows_new_requests_to_reuse_slot() -> None:
    kv = _kv(max_batch_size=2, num_blocks=4)
    kv.reserve_for_prefill("a", [0])
    kv.free("a")
    # New request takes the lowest FREE slot.
    kv.reserve_for_prefill("b", [0])
    assert kv.slot_of("b") == 0


# --- incref / decref standalone ---


def test_incref_on_unowned_block_raises() -> None:
    """incref requires the block to be held by some request already."""
    kv = _kv()
    with pytest.raises(KeyError, match="not currently held"):
        kv.incref(0)


def test_decref_on_unowned_block_raises() -> None:
    kv = _kv()
    with pytest.raises(KeyError, match="no outstanding refs"):
        kv.decref(0)


def test_decref_returns_block_to_pool_when_refcount_hits_zero() -> None:
    kv = _kv(num_blocks=4, block_size=4)
    blist = kv.reserve_for_prefill("r", list(range(4)))
    block = blist.block_ids[0]
    kv.incref(block)  # refcount 2
    assert block not in kv._free_blocks  # type: ignore[attr-defined]
    kv.decref(block)  # 1
    kv.decref(block)  # 0 → returned
    assert block in kv._free_blocks  # type: ignore[attr-defined]


# --- commit (forward-compat no-op) ---


def test_commit_validates_request() -> None:
    kv = _kv()
    kv.reserve_for_prefill("r", [0])
    kv.commit("r", 1)  # no raise
    with pytest.raises(KeyError, match="not reserved"):
        kv.commit("ghost", 1)


# --- rollback (D-021 step 5 sub-unit (d) — concrete bookkeeping) ---


def test_rollback_validates_request() -> None:
    kv = _kv()
    kv.reserve_for_prefill("r", [0])
    kv.rollback("r", 1)  # no raise; shrinks num_tokens to 0
    with pytest.raises(KeyError, match="not reserved"):
        kv.rollback("ghost", 1)


def test_rollback_no_op_for_non_positive_n_reject() -> None:
    kv = _kv(num_blocks=4, block_size=4)
    kv.reserve_for_prefill("r", list(range(8)))  # 2 blocks, 8 tokens
    free_before = kv.available_blocks()

    kv.rollback("r", 0)
    assert kv.num_tokens("r") == 8
    assert kv.available_blocks() == free_before

    kv.rollback("r", -3)
    assert kv.num_tokens("r") == 8
    assert kv.available_blocks() == free_before


def test_rollback_within_block_keeps_blocks() -> None:
    """Shrinking inside the last block: count drops, page table unchanged."""
    kv = _kv(num_blocks=8, block_size=4)
    blist = kv.reserve_for_prefill("r", list(range(10)))  # 3 blocks (10 in 4+4+2)
    assert len(blist) == 3
    free_before = kv.available_blocks()

    kv.rollback("r", 1)  # 9 tokens still need 3 blocks
    assert kv.num_tokens("r") == 9
    assert kv.available_blocks() == free_before
    # block ids unchanged
    assert kv._page_table["r"] == list(blist.block_ids)  # type: ignore[attr-defined]


def test_rollback_at_block_boundary_releases_trailing_block() -> None:
    """Rolling back to an exact block boundary releases the now-empty block."""
    kv = _kv(num_blocks=8, block_size=4)
    blist = kv.reserve_for_prefill("r", list(range(10)))  # 3 blocks
    free_before = kv.available_blocks()
    last_block = blist.block_ids[-1]

    kv.rollback("r", 2)  # 8 tokens fit in 2 blocks
    assert kv.num_tokens("r") == 8
    assert kv.available_blocks() == free_before + 1
    assert last_block in kv._free_blocks  # type: ignore[attr-defined]
    assert kv._page_table["r"] == list(blist.block_ids[:-1])  # type: ignore[attr-defined]


def test_rollback_releases_multiple_trailing_blocks() -> None:
    kv = _kv(num_blocks=8, block_size=4)
    blist = kv.reserve_for_prefill("r", list(range(12)))  # 3 blocks, full
    free_before = kv.available_blocks()
    released = list(blist.block_ids[1:])

    kv.rollback("r", 8)  # back to 4 tokens => 1 block
    assert kv.num_tokens("r") == 4
    assert kv.available_blocks() == free_before + 2
    for b in released:
        assert b in kv._free_blocks  # type: ignore[attr-defined]
    assert kv._page_table["r"] == [blist.block_ids[0]]  # type: ignore[attr-defined]


def test_rollback_to_zero_releases_all_blocks() -> None:
    kv = _kv(num_blocks=8, block_size=4)
    kv.reserve_for_prefill("r", list(range(8)))  # 2 blocks
    kv.rollback("r", 8)  # full drain (mathematically valid; engine will not do this)
    assert kv.num_tokens("r") == 0
    assert kv.available_blocks() == 8
    assert kv._page_table["r"] == []  # type: ignore[attr-defined]
    # Slot is still RESERVED — rollback only touches blocks, not the slot.
    assert kv.row_states()[kv.slot_of("r")] == RowState.RESERVED


def test_rollback_over_cap_raises_loud() -> None:
    kv = _kv(num_blocks=8, block_size=4)
    kv.reserve_for_prefill("r", list(range(4)))
    with pytest.raises(ValueError, match="cannot rollback 5 tokens"):
        kv.rollback("r", 5)
    # State unchanged on raise.
    assert kv.num_tokens("r") == 4
    assert len(kv._page_table["r"]) == 1  # type: ignore[attr-defined]


def test_rollback_preserves_blocks_pinned_by_prefix_cache() -> None:
    """A trailing block held by both the request and a prefix-cache pin
    survives rollback's decref (refcount drops to 1, not 0)."""
    kv = _kv(num_blocks=8, block_size=4)
    blist = kv.reserve_for_prefill("r", list(range(12)))  # 3 blocks
    pinned = blist.block_ids[-1]
    kv.incref(pinned)  # prefix cache references the trailing block

    kv.rollback("r", 8)  # pops trailing 2 blocks
    # Pinned block stays out of the free pool until the prefix cache decrefs.
    assert pinned not in kv._free_blocks  # type: ignore[attr-defined]
    # The other popped block (refcount 0) returned to the pool.
    assert blist.block_ids[1] in kv._free_blocks  # type: ignore[attr-defined]

    kv.decref(pinned)
    assert pinned in kv._free_blocks  # type: ignore[attr-defined]


def test_rollback_then_append_reuses_freed_blocks() -> None:
    """After rollback releases a block, append_slot can claim it again."""
    kv = _kv(num_blocks=4, block_size=4)
    kv.reserve_for_prefill("r", list(range(8)))  # 2 blocks
    kv.rollback("r", 4)  # back to 1 block
    free_after_rollback = kv.available_blocks()

    appended = kv.append_slot("r", 4)  # need 1 more block
    assert len(appended) == 1
    assert kv.available_blocks() == free_after_rollback - 1
    assert kv.num_tokens("r") == 8


def test_rollback_then_append_within_existing_block_no_realloc() -> None:
    """Rolling back inside the last block, then re-extending into the same
    block, must not allocate a new block."""
    kv = _kv(num_blocks=4, block_size=4)
    kv.reserve_for_prefill("r", list(range(6)))  # 2 blocks (4+2)
    free_before = kv.available_blocks()

    kv.rollback("r", 1)  # still 2 blocks, 5 tokens
    appended = kv.append_slot("r", 1)  # back to 6 tokens, still 2 blocks
    assert len(appended) == 0
    assert kv.available_blocks() == free_before
    assert kv.num_tokens("r") == 6


# --- get_computed_blocks (until Unit #14 plugs in) ---


def test_get_computed_blocks_returns_empty_hit() -> None:
    hit = _kv().get_computed_blocks([0, 1, 2])
    assert isinstance(hit, PrefixHit)
    assert hit.num_hit_tokens == 0
    assert hit.block_ids == ()


# --- budget ---


def test_budget_aggregates_bytes_per_claimed_block() -> None:
    """2 layers × 2 heads × 8 head_dim × 4 block × 2 bytes × (K+V) = 512 / block."""
    kv = _kv(
        num_layers=2,
        n_kv_heads=2,
        head_dim=8,
        num_blocks=4,
        block_size=4,
        dtype_bytes=2,
    )
    bytes_per_block = 2 * 2 * 2 * 8 * 4 * 2  # K+V × layers × heads × dim × block × bytes
    assert bytes_per_block == 512
    kv.reserve_for_prefill("r", list(range(5)))  # 2 blocks
    b = kv.budget()
    assert b.resident_bytes == 2 * bytes_per_block
    assert b.logical_bytes == b.resident_bytes
    assert b.headroom_bytes == 0
