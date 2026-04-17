"""Tests for silica.kvcache.codec — I-3 contract + IdentityCodec semantics.

Covers:
  - I-3 Protocol shape (@runtime_checkable sanity + public attribute presence).
  - IdentityCodec pass-through semantics: encode / decode roundtrip preserves
    values (including identity when possible).
  - Byte accounting scales linearly with num_tokens / num_blocks.
  - D-012 resident_bytes idempotency across repeated calls.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kvcache.codec import CodedBlock, IdentityCodec, KVCodec

# Per-layer shape for all tests in this module
BLOCK_SIZE = 16
N_KV_HEADS = 4
HEAD_DIM = 64
K_DTYPE = mx.float16
V_DTYPE = mx.float16


@pytest.fixture
def codec() -> IdentityCodec:
    return IdentityCodec(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        k_dtype=K_DTYPE,
        v_dtype=V_DTYPE,
    )


@pytest.fixture
def random_block() -> tuple[mx.array, mx.array]:
    k = mx.random.normal(shape=(BLOCK_SIZE, N_KV_HEADS, HEAD_DIM)).astype(K_DTYPE)
    v = mx.random.normal(shape=(BLOCK_SIZE, N_KV_HEADS, HEAD_DIM)).astype(V_DTYPE)
    return k, v


# --- I-3 Protocol shape ---


def test_identity_codec_satisfies_kvcodec_protocol(codec: IdentityCodec) -> None:
    # @runtime_checkable verifies the methods exist; asserts a baseline contract.
    assert isinstance(codec, KVCodec)


def test_identity_codec_exposes_required_attributes(codec: IdentityCodec) -> None:
    # Attributes checked separately since @runtime_checkable doesn't verify them.
    assert codec.block_size == BLOCK_SIZE
    assert codec.k_dtype == K_DTYPE
    assert codec.v_dtype == V_DTYPE


# --- encode / decode roundtrip ---


def test_encode_decode_preserves_values(
    codec: IdentityCodec, random_block: tuple[mx.array, mx.array]
) -> None:
    k, v = random_block
    block = codec.encode_block(k, v)
    k_out, v_out = codec.decode_block(block)
    # Identity codec stores tensors directly — identity check is the strongest form.
    assert k_out is k
    assert v_out is v


def test_coded_block_carries_resident_bytes(
    codec: IdentityCodec, random_block: tuple[mx.array, mx.array]
) -> None:
    k, v = random_block
    block = codec.encode_block(k, v)
    expected = k.nbytes + v.nbytes
    assert block.resident_bytes == expected


# --- byte accounting ---


def test_logical_bytes_scales_linearly(codec: IdentityCodec) -> None:
    per_token = (
        N_KV_HEADS * HEAD_DIM * K_DTYPE.size + N_KV_HEADS * HEAD_DIM * V_DTYPE.size
    )
    assert codec.logical_bytes(0) == 0
    assert codec.logical_bytes(1) == per_token
    assert codec.logical_bytes(BLOCK_SIZE) == BLOCK_SIZE * per_token
    assert codec.logical_bytes(100) == 100 * per_token


def test_resident_bytes_scales_linearly(codec: IdentityCodec) -> None:
    per_block = (
        BLOCK_SIZE
        * (N_KV_HEADS * HEAD_DIM * K_DTYPE.size + N_KV_HEADS * HEAD_DIM * V_DTYPE.size)
    )
    assert codec.resident_bytes(0) == 0
    assert codec.resident_bytes(1) == per_block
    assert codec.resident_bytes(10) == 10 * per_block


def test_identity_logical_equals_resident_at_block_boundary(codec: IdentityCodec) -> None:
    """Pass-through has zero compression: logical(n * block_size) == resident(n)."""
    for n_blocks in (1, 4, 17):
        assert codec.logical_bytes(n_blocks * BLOCK_SIZE) == codec.resident_bytes(n_blocks)


# --- D-012 idempotency ---


def test_resident_bytes_is_idempotent(codec: IdentityCodec) -> None:
    """D-012: resident_bytes must return the same value across repeated calls
    outside a modifying operation."""
    first = codec.resident_bytes(5)
    for _ in range(5):
        assert codec.resident_bytes(5) == first


def test_logical_bytes_is_idempotent(codec: IdentityCodec) -> None:
    first = codec.logical_bytes(123)
    for _ in range(5):
        assert codec.logical_bytes(123) == first


# --- CodedBlock basics ---


def test_coded_block_is_constructible_without_codec(
    random_block: tuple[mx.array, mx.array],
) -> None:
    k, v = random_block
    block = CodedBlock(k=k, v=v, resident_bytes=42)
    assert block.k is k
    assert block.v is v
    assert block.resident_bytes == 42
