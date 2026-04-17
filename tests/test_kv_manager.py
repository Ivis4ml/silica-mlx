"""Tests for silica.kvcache.manager — I-2 contract + NullKVManager semantics."""

from __future__ import annotations

import pytest

from silica.kvcache.manager import (
    BlockList,
    KVHandle,
    KVManager,
    MemoryBudget,
    NullKVManager,
    PrefixHit,
)


@pytest.fixture
def manager() -> NullKVManager:
    return NullKVManager(block_size=16)


# --- I-2 Protocol shape ---


def test_null_manager_satisfies_kv_manager_protocol(manager: NullKVManager) -> None:
    assert isinstance(manager, KVManager)


def test_null_manager_exposes_block_size(manager: NullKVManager) -> None:
    assert manager.block_size == 16


# --- reservation / decoding lifecycle (no-op null semantics) ---


def test_reserve_for_prefill_returns_empty_block_list(manager: NullKVManager) -> None:
    out = manager.reserve_for_prefill("req-1", [1, 2, 3, 4])
    assert isinstance(out, BlockList)
    assert out.block_ids == ()
    assert len(out) == 0


def test_append_slot_returns_empty_block_list(manager: NullKVManager) -> None:
    out = manager.append_slot("req-1", 4)
    assert isinstance(out, BlockList)
    assert out.block_ids == ()


def test_commit_rollback_free_do_not_raise(manager: NullKVManager) -> None:
    manager.commit("req-1", 2)
    manager.rollback("req-1", 1)
    manager.free("req-1")


def test_get_computed_blocks_returns_empty_hit(manager: NullKVManager) -> None:
    hit = manager.get_computed_blocks([1, 2, 3])
    assert isinstance(hit, PrefixHit)
    assert hit.block_ids == ()
    assert hit.num_hit_tokens == 0


def test_available_blocks_is_zero(manager: NullKVManager) -> None:
    assert manager.available_blocks() == 0


# --- budget snapshot ---


def test_budget_returns_zero_snapshot(manager: NullKVManager) -> None:
    b = manager.budget()
    assert isinstance(b, MemoryBudget)
    assert b.logical_bytes == 0
    assert b.resident_bytes == 0
    assert b.headroom_bytes == 0


# --- auxiliary type sanity ---


def test_kv_handle_is_frozen() -> None:
    h = KVHandle(req_id="abc")
    assert h.req_id == "abc"
    with pytest.raises(Exception):
        h.req_id = "xyz"  # type: ignore[misc]


def test_block_list_len() -> None:
    assert len(BlockList()) == 0
    assert len(BlockList(block_ids=(0, 1, 2))) == 3


def test_prefix_hit_defaults() -> None:
    hit = PrefixHit()
    assert hit.block_ids == ()
    assert hit.num_hit_tokens == 0


def test_memory_budget_defaults() -> None:
    b = MemoryBudget()
    assert b.logical_bytes == 0
    assert b.resident_bytes == 0
    assert b.headroom_bytes == 0
