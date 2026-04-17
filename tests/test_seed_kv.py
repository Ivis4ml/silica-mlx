"""Shape + semantic contract tests for silica.scheduler.seed_kv.

Step 3.5 of Unit 16c.2: pin the ``build_seeded_batch_kv`` contract
before step 4 wires it into the admission flow. Behavioural equivalence
to ``update_and_fetch``-accumulated caches is already guaranteed by
Gate-0.75 probe A; these tests verify the helper's own plumbing
(layer dispatch, shape validation, concatenation along the sequence
axis, state setter output).
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache

from silica.scheduler.seed_kv import build_seeded_batch_kv

N_KV_HEADS = 2
HEAD_DIM = 8
BLOCK_SIZE = 4


def _kv_slice(
    value: float,
    *,
    n_kv_heads: int = N_KV_HEADS,
    block_size: int = BLOCK_SIZE,
    head_dim: int = HEAD_DIM,
) -> tuple[mx.array, mx.array]:
    shape = (1, n_kv_heads, block_size, head_dim)
    k = mx.full(shape, value, dtype=mx.float16)
    v = mx.full(shape, value + 0.5, dtype=mx.float16)
    return k, v


def _uniform_detached(
    num_blocks: int, num_layers: int, *, seed_base: float = 0.0
) -> list[list[tuple[mx.array, mx.array]]]:
    """``detached[b][l]`` populated with distinguishable per-cell values."""
    return [
        [_kv_slice(seed_base + b * 100.0 + layer) for layer in range(num_layers)]
        for b in range(num_blocks)
    ]


def _cache_tensors(cache: BatchKVCache) -> tuple[mx.array, mx.array]:
    keys = cache.keys
    values = cache.values
    if keys is None or values is None:
        raise AssertionError("cache tensors are not initialised")
    return keys, values


def _arrays_equal(a: mx.array, b: mx.array) -> bool:
    if tuple(a.shape) != tuple(b.shape):
        return False
    eq = a == b
    if isinstance(eq, bool):
        return eq
    return bool(mx.all(eq).item())


# --- input rejection / shape contract ---


def test_rejects_empty_detached_blocks() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_seeded_batch_kv([], num_layers=1)


def test_rejects_non_positive_num_layers() -> None:
    with pytest.raises(ValueError, match="num_layers must be > 0"):
        build_seeded_batch_kv(_uniform_detached(1, 1), num_layers=0)
    with pytest.raises(ValueError, match="num_layers must be > 0"):
        build_seeded_batch_kv(_uniform_detached(1, 1), num_layers=-1)


def test_rejects_inner_list_layer_count_mismatch() -> None:
    # detached has 2 layers but num_layers=3 requested.
    with pytest.raises(ValueError, match="block 0 supplies 2 layers"):
        build_seeded_batch_kv(_uniform_detached(1, 2), num_layers=3)


def test_rejects_mid_list_layer_count_mismatch() -> None:
    # First block has 2 layers, second block has 1 — inconsistent.
    blocks = _uniform_detached(1, 2) + [[_kv_slice(99.0)]]
    with pytest.raises(ValueError, match="block 1 supplies 1 layers"):
        build_seeded_batch_kv(blocks, num_layers=2)


def test_rejects_wrong_row_axis() -> None:
    shape = (2, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)  # B=2 — wrong
    k = mx.zeros(shape, dtype=mx.float16)
    v = mx.zeros(shape, dtype=mx.float16)
    with pytest.raises(ValueError, match="B=1"):
        build_seeded_batch_kv([[(k, v)]], num_layers=1)


def test_rejects_non_4d_keys() -> None:
    # 3-D K — missing the head axis, typical shape bug.
    shape = (1, BLOCK_SIZE, HEAD_DIM)
    k = mx.zeros(shape, dtype=mx.float16)
    v = mx.zeros(shape, dtype=mx.float16)
    with pytest.raises(ValueError, match="K must be 4-D"):
        build_seeded_batch_kv([[(k, v)]], num_layers=1)


def test_rejects_k_v_shape_mismatch_at_cell() -> None:
    """Within one cell, K and V shapes must agree."""
    k = mx.zeros((1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=mx.float16)
    v = mx.zeros((1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM + 1), dtype=mx.float16)
    with pytest.raises(ValueError, match="K/V shape mismatch"):
        build_seeded_batch_kv([[(k, v)]], num_layers=1)


def test_rejects_k_v_dtype_mismatch_at_cell() -> None:
    k = mx.zeros((1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=mx.float16)
    v = mx.zeros((1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=mx.float32)
    with pytest.raises(ValueError, match="K/V dtype mismatch"):
        build_seeded_batch_kv([[(k, v)]], num_layers=1)


def test_rejects_zero_token_block() -> None:
    k = mx.zeros((1, N_KV_HEADS, 0, HEAD_DIM), dtype=mx.float16)
    v = mx.zeros((1, N_KV_HEADS, 0, HEAD_DIM), dtype=mx.float16)
    with pytest.raises(ValueError, match="block size must be > 0"):
        build_seeded_batch_kv([[(k, v)]], num_layers=1)


def test_rejects_block_shape_drift() -> None:
    """Block 0 sets the shape; block 1 may not disagree."""
    block0 = [_kv_slice(0.0)]
    block1 = [_kv_slice(1.0, block_size=BLOCK_SIZE + 2)]
    with pytest.raises(ValueError, match="shape mismatch at block 1 layer 0"):
        build_seeded_batch_kv([block0, block1], num_layers=1)


def test_rejects_cross_layer_block_size_drift() -> None:
    """Layers may have different H/D, but the token block size is shared."""
    layer0 = _kv_slice(0.0, block_size=BLOCK_SIZE)
    layer1 = _kv_slice(1.0, block_size=BLOCK_SIZE + 1)
    with pytest.raises(ValueError, match="block size mismatch"):
        build_seeded_batch_kv([[layer0, layer1]], num_layers=2)


def test_accepts_layer_specific_head_shapes() -> None:
    """Each layer gets its own BatchKVCache, so H/D need not be global."""
    layer0 = _kv_slice(0.0, n_kv_heads=2, head_dim=8)
    layer1 = _kv_slice(1.0, n_kv_heads=3, head_dim=12)
    caches = build_seeded_batch_kv([[layer0, layer1]], num_layers=2)
    keys0, _ = _cache_tensors(caches[0])
    keys1, _ = _cache_tensors(caches[1])
    assert tuple(keys0.shape) == (1, 2, BLOCK_SIZE, 8)
    assert tuple(keys1.shape) == (1, 3, BLOCK_SIZE, 12)


# --- output shape / state ---


def test_single_block_single_layer_output_shape() -> None:
    caches = build_seeded_batch_kv(_uniform_detached(1, 1), num_layers=1)
    assert len(caches) == 1
    cache = caches[0]
    keys, values = _cache_tensors(cache)
    assert cache._idx == BLOCK_SIZE
    assert tuple(keys.shape) == (1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    assert tuple(values.shape) == (1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
    assert cache.left_padding.tolist() == [0]
    assert cache.offset.tolist() == [BLOCK_SIZE]


def test_multiple_blocks_concatenate_along_seq_axis() -> None:
    caches = build_seeded_batch_kv(_uniform_detached(3, 1), num_layers=1)
    cache = caches[0]
    keys, _ = _cache_tensors(cache)
    total_seq = 3 * BLOCK_SIZE
    assert cache._idx == total_seq
    assert tuple(keys.shape) == (1, N_KV_HEADS, total_seq, HEAD_DIM)
    assert cache.offset.tolist() == [total_seq]


def test_multiple_layers_produce_separate_caches() -> None:
    caches = build_seeded_batch_kv(_uniform_detached(2, 4), num_layers=4)
    assert len(caches) == 4
    # Each layer's cache is independent — distinct Python objects, same shape.
    assert len({id(c) for c in caches}) == 4
    for c in caches:
        assert c._idx == 2 * BLOCK_SIZE


def test_layer_state_arrays_are_independent() -> None:
    caches = build_seeded_batch_kv(_uniform_detached(1, 2), num_layers=2)
    suffix_k = mx.full((1, N_KV_HEADS, 1, HEAD_DIM), 42.0, dtype=mx.float16)
    suffix_v = mx.full((1, N_KV_HEADS, 1, HEAD_DIM), 42.5, dtype=mx.float16)
    caches[0].update_and_fetch(suffix_k, suffix_v)
    mx.eval(caches[0].offset, caches[1].offset, caches[0].left_padding, caches[1].left_padding)
    assert caches[0].offset.tolist() == [BLOCK_SIZE + 1]
    assert caches[1].offset.tolist() == [BLOCK_SIZE]
    assert caches[0].left_padding.tolist() == [0]
    assert caches[1].left_padding.tolist() == [0]


# --- value preservation ---


def test_concatenation_preserves_per_block_values_in_order() -> None:
    """Block 0 occupies seq [0:block_size), block 1 occupies [block_size:2*block_size), ..."""
    detached = _uniform_detached(2, 1, seed_base=0.0)
    # Known values: block 0 layer 0 K = fill(0.0 + 0 + 0) = 0.0
    #               block 1 layer 0 K = fill(0.0 + 100 + 0) = 100.0
    caches = build_seeded_batch_kv(detached, num_layers=1)
    cache = caches[0]
    keys, _ = _cache_tensors(cache)
    mx.eval(keys)
    # Every element of the first block should be 0.0 (converted to fp16).
    first_block_k = keys[..., :BLOCK_SIZE, :]
    second_block_k = keys[..., BLOCK_SIZE : 2 * BLOCK_SIZE, :]
    mx.eval(first_block_k, second_block_k)
    assert float(first_block_k.flatten()[0].item()) == 0.0
    assert float(second_block_k.flatten()[0].item()) == 100.0


def test_per_layer_isolation() -> None:
    """Layer 0 of a cache must not carry layer 1's values."""
    detached = _uniform_detached(1, 2, seed_base=0.0)
    # layer 0 K = 0.0, layer 1 K = 1.0.
    caches = build_seeded_batch_kv(detached, num_layers=2)
    keys0, _ = _cache_tensors(caches[0])
    keys1, _ = _cache_tensors(caches[1])
    mx.eval(keys0, keys1)
    assert float(keys0.flatten()[0].item()) == 0.0
    assert float(keys1.flatten()[0].item()) == 1.0


# --- integration: seeded cache behaves like a uaf-driven cache ---


def test_seeded_cache_extend_matches_gate_0_75_probe_A2() -> None:
    """Regression guard on probe A.2: extending a seeded row_cache into
    a main cache produces the same state as extending a uaf-driven
    row_cache. The probe lives in scripts/probe_16c2_seed_and_detach.py;
    this test pins the invariant inside the regression suite so step 4
    cannot silently regress it.
    """
    # Main: B=2, pre-filled via uaf with tokens at positions 0..4.
    main_lp = [0, 1]
    k_main = mx.full(
        (2, N_KV_HEADS, 5, HEAD_DIM), 7.0, dtype=mx.float16
    )
    v_main = mx.full(
        (2, N_KV_HEADS, 5, HEAD_DIM), 7.5, dtype=mx.float16
    )
    main_ref = BatchKVCache(left_padding=main_lp)
    main_ref.update_and_fetch(k_main, v_main)
    main_cand = BatchKVCache(left_padding=main_lp)
    main_cand.update_and_fetch(k_main, v_main)

    # Row via uaf (reference).
    row_T = BLOCK_SIZE  # 4 tokens = 1 block
    k_row = mx.full(
        (1, N_KV_HEADS, row_T, HEAD_DIM), 9.0, dtype=mx.float16
    )
    v_row = mx.full(
        (1, N_KV_HEADS, row_T, HEAD_DIM), 9.5, dtype=mx.float16
    )
    row_ref = BatchKVCache(left_padding=[0])
    row_ref.update_and_fetch(k_row, v_row)

    # Row via the helper (candidate).
    detached = [[(k_row, v_row)]]
    (row_cand,) = build_seeded_batch_kv(detached, num_layers=1)

    main_ref.extend(row_ref)
    main_cand.extend(row_cand)
    mx.eval(main_ref.keys, main_ref.values, main_cand.keys, main_cand.values)
    main_ref_keys, main_ref_values = _cache_tensors(main_ref)
    main_cand_keys, main_cand_values = _cache_tensors(main_cand)

    assert tuple(main_ref_keys.shape) == tuple(main_cand_keys.shape)
    assert _arrays_equal(main_ref_keys, main_cand_keys)
    assert _arrays_equal(main_ref_values, main_cand_values)
    assert main_ref._idx == main_cand._idx


def test_seeded_cache_accepts_update_and_fetch_after() -> None:
    """A seeded cache must accept a subsequent uaf without corruption —
    this is the suffix-prefill case from §4 step 4 of the prep doc.
    """
    caches = build_seeded_batch_kv(_uniform_detached(1, 1), num_layers=1)
    cache = caches[0]
    # Simulate a 1-token suffix prefill.
    suffix_k = mx.full((1, N_KV_HEADS, 1, HEAD_DIM), 42.0, dtype=mx.float16)
    suffix_v = mx.full((1, N_KV_HEADS, 1, HEAD_DIM), 42.5, dtype=mx.float16)
    cache.update_and_fetch(suffix_k, suffix_v)
    mx.eval(cache.keys, cache.values)
    keys, _ = _cache_tensors(cache)
    # _idx advances by 1.
    assert cache._idx == BLOCK_SIZE + 1
    # Last slot holds the suffix K (not the seeded block's K).
    last_k = keys[..., BLOCK_SIZE, :]
    mx.eval(last_k)
    assert float(last_k.flatten()[0].item()) == 42.0
