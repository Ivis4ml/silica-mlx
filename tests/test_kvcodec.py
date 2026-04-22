"""Tests for silica.kvcache.codec — I-3 contract + IdentityCodec semantics.

Side-level ``VectorCodec[P]`` API (P-5-A.0.4 migration).

Covers:
  - I-3 Protocol shape (``@runtime_checkable`` sanity + public attribute
    presence).
  - ``IdentityCodec`` pass-through semantics: ``encode_tensor`` /
    ``decode_tensor`` round-trip preserves values (identity-by-reference,
    no defensive copy).
  - Byte accounting scales linearly with ``num_tokens`` / ``num_blocks``.
  - ``IdentityCodec.resident_bytes`` / ``logical_bytes`` are **per side**
    (K or V alone), not a K+V sum.
  - D-012 idempotency of ``resident_bytes`` across repeated calls.

Payload-level tests (``RawFp16Payload`` / ``BlockTQPayload`` /
``RaBitQPayload`` construction + D-012 honesty) live in
``tests/test_vector_codec.py``; the two files together cover the full
side-level codec surface.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from silica.kvcache.codec import IdentityCodec, RawFp16Payload, VectorCodec

# Per-layer-per-side shape for all tests in this module.
BLOCK_SIZE = 16
N_KV_HEADS = 4
HEAD_DIM = 64
DTYPE = mx.float16


@pytest.fixture
def codec() -> IdentityCodec:
    return IdentityCodec(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
    )


@pytest.fixture
def random_tensor() -> mx.array:
    """One side (K or V) of one block — shape (1, n_kv_heads, block_size, head_dim)."""
    return mx.random.normal(shape=(1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)).astype(DTYPE)


# --- I-3 Protocol shape ---


def test_identity_codec_satisfies_vector_codec_protocol(codec: IdentityCodec) -> None:
    # ``@runtime_checkable`` verifies the methods exist.
    assert isinstance(codec, VectorCodec)


def test_identity_codec_exposes_required_attributes(codec: IdentityCodec) -> None:
    # Attributes are checked separately since ``@runtime_checkable`` does
    # not verify class-attribute presence.
    assert codec.block_size == BLOCK_SIZE
    assert codec.dtype == DTYPE


# --- encode / decode round-trip ---


def test_encode_decode_preserves_tensor_reference(
    codec: IdentityCodec, random_tensor: mx.array
) -> None:
    """Identity codec stores the tensor by reference — no copy."""
    payload = codec.encode_tensor(random_tensor)
    assert payload.t is random_tensor
    out = codec.decode_tensor(payload)
    assert out is random_tensor


def test_encode_produces_raw_fp16_payload(
    codec: IdentityCodec, random_tensor: mx.array
) -> None:
    payload = codec.encode_tensor(random_tensor)
    assert isinstance(payload, RawFp16Payload)


def test_payload_resident_bytes_equals_tensor_nbytes(
    codec: IdentityCodec, random_tensor: mx.array
) -> None:
    """D-012: declared resident_bytes on the payload matches actual
    ``.nbytes`` of the wrapped tensor."""
    payload = codec.encode_tensor(random_tensor)
    assert payload.resident_bytes == random_tensor.nbytes


# --- byte accounting (per-side) ---


def test_logical_bytes_scales_linearly_per_side(codec: IdentityCodec) -> None:
    """``logical_bytes`` is one-side fp16: ``n_kv_heads × head_dim × dtype.size``
    per token. Caller sums two sides when a K+V total is needed."""
    per_token_one_side = N_KV_HEADS * HEAD_DIM * DTYPE.size
    assert codec.logical_bytes(0) == 0
    assert codec.logical_bytes(1) == per_token_one_side
    assert codec.logical_bytes(BLOCK_SIZE) == BLOCK_SIZE * per_token_one_side
    assert codec.logical_bytes(100) == 100 * per_token_one_side


def test_resident_bytes_scales_linearly_per_side(codec: IdentityCodec) -> None:
    """``resident_bytes(num_blocks)`` is one-side fp16 bytes for num_blocks
    blocks of this side (K alone or V alone)."""
    per_block_one_side = BLOCK_SIZE * N_KV_HEADS * HEAD_DIM * DTYPE.size
    assert codec.resident_bytes(0) == 0
    assert codec.resident_bytes(1) == per_block_one_side
    assert codec.resident_bytes(10) == 10 * per_block_one_side


def test_identity_logical_equals_resident_at_block_boundary(
    codec: IdentityCodec,
) -> None:
    """Pass-through has zero compression: ``logical(n × block_size) ==
    resident(n)``. Both are one-side counts under the side-level API."""
    for n_blocks in (1, 4, 17):
        assert codec.logical_bytes(n_blocks * BLOCK_SIZE) == codec.resident_bytes(
            n_blocks
        )


def test_resident_bytes_matches_payload_sum_over_blocks(
    codec: IdentityCodec, random_tensor: mx.array
) -> None:
    """Summing ``RawFp16Payload.resident_bytes`` across N encode calls
    equals ``codec.resident_bytes(N)`` for this one side."""
    n = 5
    sum_payload_bytes = 0
    for _ in range(n):
        t = mx.random.normal(shape=(1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)).astype(DTYPE)
        sum_payload_bytes += codec.encode_tensor(t).resident_bytes
    assert sum_payload_bytes == codec.resident_bytes(n)


# --- D-012 idempotency ---


def test_resident_bytes_is_idempotent(codec: IdentityCodec) -> None:
    """D-012: ``resident_bytes`` must return the same value across repeated
    calls outside a modifying operation."""
    first = codec.resident_bytes(5)
    for _ in range(5):
        assert codec.resident_bytes(5) == first


def test_logical_bytes_is_idempotent(codec: IdentityCodec) -> None:
    first = codec.logical_bytes(123)
    for _ in range(5):
        assert codec.logical_bytes(123) == first


# --- dtype handling ---


def test_identity_codec_honors_custom_dtype() -> None:
    """``IdentityCodec`` can be constructed with a non-default dtype; byte
    accounting reflects the chosen dtype's itemsize."""
    codec_bf16 = IdentityCodec(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=mx.bfloat16,
    )
    assert codec_bf16.dtype == mx.bfloat16
    per_token = N_KV_HEADS * HEAD_DIM * mx.bfloat16.size
    assert codec_bf16.logical_bytes(1) == per_token
