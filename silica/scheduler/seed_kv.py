"""silica.scheduler.seed_kv — build a seeded BatchKVCache row from prefix hit.

Unit 16c.2 step 3.5: isolated helper extracted from step 4's admission
flow so its shape contract can be pinned (tested) before the batcher
surgery lands. Step 4's ``_admit_waiting_requests`` will call this
after ``RadixPrefixCache.lookup`` returns a non-empty hit, to
materialise the per-row cache that will extend into the main batched
cache.

**What this helper does (and does not).**

It assembles a ``list[BatchKVCache]`` — one per transformer layer — each
with ``B = 1`` row already populated by the caller's detached K/V
slices along the sequence axis. Gate-0.75 probe A verified that the
``cache.state`` setter route is equivalent to driving
``update_and_fetch`` on an empty cache, so we use the setter.

This helper does NOT:
  - extend into the main batch cache; that is the caller's job (§4
    step 6 of docs/P2_UNIT_16C_2_PREP.md).
  - release hits in the radix cache; the caller owns the
    ``radix.release`` call once the copy is complete (§4 step 5).
  - prefill suffix tokens; the caller drives the model forward pass
    separately (§4 step 4).

Keeping the helper scoped to cache construction means each of those
concerns is a single-responsibility step in the admission path.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache


def build_seeded_batch_kv(
    detached_blocks: Sequence[Sequence[tuple[mx.array, mx.array]]],
    *,
    num_layers: int,
) -> list[BatchKVCache]:
    """Construct one ``BatchKVCache(B=1)`` per layer, pre-populated.

    Args:
        detached_blocks: indexed ``[block_idx][layer_idx]`` → ``(K, V)``.
            Each K/V shape must be ``(1, n_kv_heads, block_size,
            head_dim)``. The token block_size is shared across layers;
            ``n_kv_heads`` / ``head_dim`` may vary by layer, but each
            layer's shape and dtype must be stable across blocks.
        num_layers: total number of transformer layers; every inner
            list must have exactly this length.

    Returns:
        ``list[BatchKVCache]`` of length ``num_layers``. Each cache has:
          - keys / values concatenated along the sequence axis,
            shape ``(1, n_kv_heads, total_seq, head_dim)``
          - ``_idx == total_seq == num_blocks * block_size``
          - ``left_padding == mx.array([0])``
          - ``offset == mx.array([total_seq])``

    Raises:
        ValueError on empty input, shape disagreement, or layer-count
        mismatch. Loud-fail on contract violations because silent
        shape coercion in this function would manifest as divergent
        decode output one step later — hard to trace back.
    """
    if not detached_blocks:
        raise ValueError("detached_blocks must be non-empty")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {num_layers}")

    num_blocks = len(detached_blocks)

    # Derive one reference shape/dtype per layer from block 0, then
    # enforce it for that same layer across later blocks. Layers are
    # intentionally allowed to differ from each other; the helper returns
    # one BatchKVCache per layer, so there is no need to impose a global
    # head-count / head-dim shape across all layers.
    first_layer_list = detached_blocks[0]
    if len(first_layer_list) != num_layers:
        raise ValueError(
            f"block 0 supplies {len(first_layer_list)} layers, "
            f"expected {num_layers}"
        )

    expected_shapes: list[tuple[int, ...]] = []
    expected_dtypes: list[object] = []
    block_size: int | None = None
    for layer_idx, (k, v) in enumerate(first_layer_list):
        shape = tuple(k.shape)
        if len(shape) != 4:
            raise ValueError(
                f"K must be 4-D (B, H, T, D); got shape {shape}"
            )
        if shape[0] != 1:
            raise ValueError(
                f"expected row axis B=1 (one request per seeded cache); "
                f"got B={shape[0]}"
            )
        if shape[2] <= 0:
            raise ValueError(
                f"block size must be > 0; got shape {shape} at layer {layer_idx}"
            )
        if block_size is None:
            block_size = shape[2]
        elif shape[2] != block_size:
            raise ValueError(
                f"block size mismatch at block 0 layer {layer_idx}: "
                f"got {shape[2]}, expected {block_size}"
            )
        if tuple(v.shape) != shape:
            raise ValueError(
                f"K/V shape mismatch at block 0 layer {layer_idx}: "
                f"K={shape}, V={tuple(v.shape)}"
            )
        if v.dtype != k.dtype:
            raise ValueError(
                f"K/V dtype mismatch at block 0 layer {layer_idx}: "
                f"K={k.dtype}, V={v.dtype}"
            )
        expected_shapes.append(shape)
        expected_dtypes.append(k.dtype)

    # Validate every cell.
    for block_idx, per_layer in enumerate(detached_blocks):
        if len(per_layer) != num_layers:
            raise ValueError(
                f"block {block_idx} supplies {len(per_layer)} layers, "
                f"expected {num_layers}"
            )
        for layer_idx, (k, v) in enumerate(per_layer):
            expected_shape = expected_shapes[layer_idx]
            expected_dtype = expected_dtypes[layer_idx]
            if tuple(k.shape) != expected_shape:
                raise ValueError(
                    f"shape mismatch at block {block_idx} layer "
                    f"{layer_idx}: K={tuple(k.shape)}, "
                    f"expected {expected_shape}"
                )
            if tuple(v.shape) != expected_shape:
                raise ValueError(
                    f"K/V shape mismatch at block {block_idx} layer "
                    f"{layer_idx}: K={tuple(k.shape)}, V={tuple(v.shape)}"
                )
            if k.dtype != expected_dtype:
                raise ValueError(
                    f"dtype mismatch at block {block_idx} layer "
                    f"{layer_idx}: K={k.dtype}, expected {expected_dtype}"
                )
            if v.dtype != expected_dtype:
                raise ValueError(
                    f"K/V dtype mismatch at block {block_idx} layer "
                    f"{layer_idx}: K={k.dtype}, V={v.dtype}"
                )

    assert block_size is not None
    total_seq = num_blocks * block_size

    # Per-layer: concatenate incoming block K/V slices along the
    # sequence axis, then install via state setter. Gate-0.75 probe
    # A.1 proved this is equivalent to update_and_fetch for all
    # downstream consumers (decode + extend).
    caches: list[BatchKVCache] = []
    for layer_idx in range(num_layers):
        k_parts = [detached_blocks[b][layer_idx][0] for b in range(num_blocks)]
        v_parts = [detached_blocks[b][layer_idx][1] for b in range(num_blocks)]
        keys = mx.concatenate(k_parts, axis=2)
        values = mx.concatenate(v_parts, axis=2)
        cache = BatchKVCache(left_padding=[0])
        cache.state = (keys, values, mx.array([total_seq]), mx.array([0]))
        caches.append(cache)
    return caches


__all__ = ["build_seeded_batch_kv"]
