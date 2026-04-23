"""P-5-B.1a unit tests for silica.vq.rabitq.RaBitQ1Bit.

Tests the side-level VectorCodec[RaBitQPayload] contract without a full
model. Each test constructs a fresh codec instance.

Coverage:
- VectorCodec interface: encode returns RaBitQPayload; decode shape + dtype
- Payload field shapes, dtypes, D-012 resident_bytes honesty
- Zero-vector edge case: norm_o = 0, ip_coeff = 0, all-ones packed bits,
  decode → zero tensor (also exercises the sign tie-break 0 → +1)
- ip_coeff formula check: ``sum(signs/sqrt(d) * Y)`` reproduced in NumPy
  against the same Haar rotation
- ip_coeff / norm_o non-negativity
- Round-trip MSE sanity band (catches algorithmic bugs before B.1c parity)
- Centroid field locked: shape, dtype, value, fit() no-op
- Constructor guards: num_bits not 1 (including 0, negative, >1), head_dim
  not multiple of 8, fp32 dtype
- Encode shape guards: wrong batch dim, head count, block size, head dim,
  wrong input dtype
- Decode shape guards: malformed packed_indices / norm_o / ip_coeff shapes
- logical_bytes / resident_bytes formula; resident_bytes(1) == payload value
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from silica.kvcache.codec import RaBitQPayload
from silica.vq import RaBitQ1Bit
from silica.vq._calibration import haar_rotation

# Standard codec parameters shared across most tests.
_BLOCK_SIZE = 16
_N_KV_HEADS = 4
_HEAD_DIM = 64
_SEED = 42
_N_VECTORS = _N_KV_HEADS * _BLOCK_SIZE  # 64


def _make_codec(
    *,
    block_size: int = _BLOCK_SIZE,
    n_kv_heads: int = _N_KV_HEADS,
    head_dim: int = _HEAD_DIM,
    seed: int = _SEED,
    dtype: mx.Dtype = mx.float16,
) -> RaBitQ1Bit:
    return RaBitQ1Bit(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        seed=seed,
        dtype=dtype,
    )


def _make_random_input(
    block_size: int = _BLOCK_SIZE,
    n_kv_heads: int = _N_KV_HEADS,
    head_dim: int = _HEAD_DIM,
    dtype: mx.Dtype = mx.float16,
    seed: int = 0,
) -> mx.array:
    """Deterministic standard-normal input of shape (1, n_kv_heads, block_size, head_dim)."""
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((1, n_kv_heads, block_size, head_dim)).astype(np.float32)
    return mx.array(x_np).astype(dtype)


def _all_true(arr: mx.array) -> bool:
    """Reduce a bool mx.array to Python bool; satisfies mypy's `array | bool`
    narrowing from operator results."""
    return bool(mx.all(arr).item())


def _equal_arrays(a: mx.array, b: mx.array) -> bool:
    """Element-wise equality reduced to bool (mypy-friendly)."""
    return bool(mx.array_equal(a, b).item())


class TestRaBitQ1BitInterface:
    """VectorCodec contract: encode returns RaBitQPayload; decode shape and dtype."""

    def test_encode_returns_rabitq_payload(self) -> None:
        codec = _make_codec()
        x = _make_random_input()
        payload = codec.encode_tensor(x)
        assert isinstance(payload, RaBitQPayload)

    def test_decode_shape(self) -> None:
        codec = _make_codec()
        x = _make_random_input()
        x_hat = codec.decode_tensor(codec.encode_tensor(x))
        assert tuple(x_hat.shape) == (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM)

    def test_decode_dtype_fp16(self) -> None:
        codec = _make_codec(dtype=mx.float16)
        x = _make_random_input(dtype=mx.float16)
        x_hat = codec.decode_tensor(codec.encode_tensor(x))
        assert x_hat.dtype == mx.float16

    def test_decode_dtype_bf16(self) -> None:
        codec = _make_codec(dtype=mx.bfloat16)
        x = _make_random_input(dtype=mx.bfloat16)
        x_hat = codec.decode_tensor(codec.encode_tensor(x))
        assert x_hat.dtype == mx.bfloat16


class TestRaBitQ1BitPayload:
    """RaBitQPayload field shapes, dtypes, and D-012 resident_bytes honesty."""

    def test_packed_indices_dtype(self) -> None:
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        assert payload.packed_indices.dtype == mx.uint8

    def test_packed_indices_shape(self) -> None:
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        assert tuple(payload.packed_indices.shape) == (_N_VECTORS, _HEAD_DIM // 8)

    def test_norm_o_dtype(self) -> None:
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        assert payload.norm_o.dtype == mx.float16

    def test_norm_o_shape(self) -> None:
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        assert tuple(payload.norm_o.shape) == (_N_VECTORS,)

    def test_ip_coeff_dtype(self) -> None:
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        assert payload.ip_coeff.dtype == mx.float16

    def test_ip_coeff_shape(self) -> None:
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        assert tuple(payload.ip_coeff.shape) == (_N_VECTORS,)

    def test_resident_bytes_honesty(self) -> None:
        """D-012: payload.resident_bytes == sum of array .nbytes."""
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        expected = (
            int(payload.packed_indices.nbytes)
            + int(payload.norm_o.nbytes)
            + int(payload.ip_coeff.nbytes)
        )
        assert payload.resident_bytes == expected


class TestRaBitQ1BitCentroid:
    """Centroid invariant: zero-valued fp32 (d,) field, fit() must not change it."""

    def test_centroid_shape_and_dtype(self) -> None:
        codec = _make_codec()
        assert tuple(codec._centroid.shape) == (_HEAD_DIM,)
        assert codec._centroid.dtype == mx.float32

    def test_centroid_is_zero(self) -> None:
        codec = _make_codec()
        zero = mx.zeros((_HEAD_DIM,), dtype=mx.float32)
        assert _equal_arrays(codec._centroid, zero)

    def test_fit_does_not_change_centroid(self) -> None:
        """fit() must not update centroid (§5.3 pins it at zero for P-5-B)."""
        codec = _make_codec()
        rng = np.random.default_rng(99)
        fake_corpus = mx.array(rng.standard_normal((32, _HEAD_DIM)).astype(np.float32))
        codec.fit(fake_corpus)
        zero = mx.zeros((_HEAD_DIM,), dtype=mx.float32)
        assert _equal_arrays(codec._centroid, zero)

    def test_fit_does_not_change_encode_output(self) -> None:
        """Full-encode round-trip: packed / norm / ip_coeff identical after fit()."""
        codec = _make_codec()
        x = _make_random_input()
        before = codec.encode_tensor(x)
        rng = np.random.default_rng(99)
        fake_corpus = mx.array(rng.standard_normal((32, _HEAD_DIM)).astype(np.float32))
        codec.fit(fake_corpus)
        after = codec.encode_tensor(x)
        assert _equal_arrays(before.packed_indices, after.packed_indices)
        assert _equal_arrays(before.norm_o, after.norm_o)
        assert _equal_arrays(before.ip_coeff, after.ip_coeff)


class TestRaBitQ1BitEdgeCases:
    """Zero vector, sign tie-break, ip_coeff formula, MSE sanity."""

    def test_zero_vector_decode_is_zero(self) -> None:
        codec = _make_codec()
        x_zero = mx.zeros((1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16)
        x_hat = codec.decode_tensor(codec.encode_tensor(x_zero))
        assert _equal_arrays(x_hat, mx.zeros_like(x_hat))

    def test_zero_vector_norm_o_is_zero(self) -> None:
        codec = _make_codec()
        x_zero = mx.zeros((1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16)
        payload = codec.encode_tensor(x_zero)
        assert _equal_arrays(payload.norm_o, mx.zeros_like(payload.norm_o))

    def test_zero_vector_ip_coeff_is_zero(self) -> None:
        """Zero vector → ip_coeff = 0 (batch-path semantics, not 0.7979)."""
        codec = _make_codec()
        x_zero = mx.zeros((1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16)
        payload = codec.encode_tensor(x_zero)
        assert _equal_arrays(payload.ip_coeff, mx.zeros_like(payload.ip_coeff))

    def test_zero_vector_packed_bits_all_ones(self) -> None:
        """Sign tie-break 0 → +1 round-trips through pack_sub_byte as 0xFF
        per byte (all bits = 1). Locks the tie-break direction against a
        future accidental flip to `0 → -1`."""
        codec = _make_codec()
        x_zero = mx.zeros((1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16)
        payload = codec.encode_tensor(x_zero)
        all_ones = mx.full(payload.packed_indices.shape, vals=0xFF, dtype=mx.uint8)
        assert _equal_arrays(payload.packed_indices, all_ones)

    def test_ip_coeff_formula(self) -> None:
        """ip_coeff == sum(signs * Y, axis=1) / sqrt(d), where
        signs = where(Y == 0, +1, sign(Y)). Reproduces the MLX computation
        in fp64 NumPy against the same Haar rotation, then compares fp16."""
        codec = _make_codec()
        x = _make_random_input(seed=42)
        payload = codec.encode_tensor(x)

        rotation = np.asarray(haar_rotation(_HEAD_DIM, _SEED), dtype=np.float64)
        x_np = np.asarray(x, dtype=np.float32).astype(np.float64)
        flat = x_np.reshape(_N_VECTORS, _HEAD_DIM)
        # Centroid = 0 per P-5-B §5.3; left in the formula as an identity step
        # so the NumPy reference mirrors the MLX encode path 1:1.
        centroid = np.zeros(_HEAD_DIM, dtype=np.float64)
        o = flat - centroid
        norms = np.linalg.norm(o, axis=1, keepdims=True)
        safe = np.maximum(norms, 1e-30)
        o_bar = o / safe
        Y = o_bar @ rotation.T
        signs = np.where(Y == 0.0, 1.0, np.sign(Y))
        expected_ip = np.sum(signs * Y, axis=1) / np.sqrt(_HEAD_DIM)

        got_ip = np.asarray(payload.ip_coeff).astype(np.float64)
        # fp16 rounding of ip_coeff values clustered near √(2/π) ≈ 0.798:
        # rtol 5e-3 covers the ~3-digit fp16 precision; atol handles values
        # near zero from pathological inputs (none here, but defensive).
        np.testing.assert_allclose(got_ip, expected_ip, rtol=5e-3, atol=5e-4)

    def test_norm_o_nonnegative(self) -> None:
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        min_norm = float(mx.min(payload.norm_o).item())
        assert min_norm >= 0.0

    def test_ip_coeff_nonnegative(self) -> None:
        """ip_coeff = sum|y_i|/sqrt(d) ≥ 0 in fp32; fp16 rounding allows tiny dip."""
        codec = _make_codec()
        payload = codec.encode_tensor(_make_random_input())
        min_ip = float(mx.min(payload.ip_coeff).item())
        assert min_ip >= -1e-3

    def test_roundtrip_mse_reasonable(self) -> None:
        """Decoded output lands in the expected MSE band for 1-bit RaBitQ on R^64.

        Analytical expectation for random-normal inputs:
        ``E[MSE/coord] ≈ 2(1 - √(2/π)) ≈ 0.404``.
        Range [0.2, 0.7] rejects both near-zero output (trivial decode,
        MSE ≈ mean(x²) ≈ 1.0) and garbage decode (MSE > 5).
        """
        codec = _make_codec()
        x = _make_random_input(seed=1234)
        x_fp32 = x.astype(mx.float32)
        x_hat_fp32 = codec.decode_tensor(codec.encode_tensor(x)).astype(mx.float32)
        mse = float(mx.mean((x_hat_fp32 - x_fp32) ** 2).item())
        assert 0.2 <= mse <= 0.7, (
            f"MSE={mse:.4f} outside expected range [0.2, 0.7] for 1-bit RaBitQ on R^64"
        )


class TestRaBitQ1BitConstructorGuards:
    """Constructor validation: reject invalid arguments early."""

    @pytest.mark.parametrize("bad_bits", [-1, 0, 2, 3, 4, 8])
    def test_rejects_non_one_num_bits(self, bad_bits: int) -> None:
        """num_bits in any non-1 value (incl. 0, negatives, >1) must raise."""
        with pytest.raises(ValueError, match="1-bit-only"):
            RaBitQ1Bit(block_size=16, n_kv_heads=4, head_dim=64, num_bits=bad_bits)

    def test_rejects_head_dim_not_multiple_of_8(self) -> None:
        with pytest.raises(ValueError, match="multiple of 8"):
            RaBitQ1Bit(block_size=16, n_kv_heads=4, head_dim=63)

    def test_rejects_zero_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            RaBitQ1Bit(block_size=0, n_kv_heads=4, head_dim=64)

    def test_rejects_zero_n_kv_heads(self) -> None:
        with pytest.raises(ValueError, match="n_kv_heads"):
            RaBitQ1Bit(block_size=16, n_kv_heads=0, head_dim=64)

    def test_rejects_zero_head_dim(self) -> None:
        with pytest.raises(ValueError, match="head_dim"):
            RaBitQ1Bit(block_size=16, n_kv_heads=4, head_dim=0)

    def test_rejects_fp32_dtype(self) -> None:
        with pytest.raises(ValueError, match="D-003"):
            RaBitQ1Bit(block_size=16, n_kv_heads=4, head_dim=64, dtype=mx.float32)


class TestRaBitQ1BitEncodeGuards:
    """encode_tensor shape / dtype validation."""

    def test_rejects_wrong_batch_dim(self) -> None:
        codec = _make_codec()
        x = mx.zeros((2, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16)
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_n_kv_heads(self) -> None:
        codec = _make_codec()
        x = mx.zeros((1, _N_KV_HEADS + 1, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16)
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_block_size(self) -> None:
        codec = _make_codec()
        x = mx.zeros((1, _N_KV_HEADS, _BLOCK_SIZE + 1, _HEAD_DIM), dtype=mx.float16)
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_head_dim(self) -> None:
        codec = _make_codec()
        x = mx.zeros((1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM * 2), dtype=mx.float16)
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_input_dtype(self) -> None:
        codec = _make_codec(dtype=mx.float16)
        x = mx.zeros((1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.bfloat16)
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)


class TestRaBitQ1BitDecodeGuards:
    """decode_tensor payload shape validation (protects against wrong codec config)."""

    def _make_payload(
        self, n_vectors: int, packed_width: int, meta_len: int
    ) -> RaBitQPayload:
        packed = mx.zeros((n_vectors, packed_width), dtype=mx.uint8)
        norm = mx.zeros((meta_len,), dtype=mx.float16)
        ip = mx.zeros((meta_len,), dtype=mx.float16)
        resident = int(packed.nbytes + norm.nbytes + ip.nbytes)
        return RaBitQPayload(
            resident_bytes=resident,
            packed_indices=packed,
            norm_o=norm,
            ip_coeff=ip,
        )

    def test_rejects_wrong_packed_n_vectors(self) -> None:
        codec = _make_codec()
        bad = self._make_payload(
            n_vectors=_N_VECTORS + _N_KV_HEADS,  # different block_size
            packed_width=_HEAD_DIM // 8,
            meta_len=_N_VECTORS + _N_KV_HEADS,
        )
        with pytest.raises(ValueError, match="packed_indices shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_packed_width(self) -> None:
        codec = _make_codec()
        bad = self._make_payload(
            n_vectors=_N_VECTORS,
            packed_width=_HEAD_DIM // 8 + 1,  # wrong head_dim implied
            meta_len=_N_VECTORS,
        )
        with pytest.raises(ValueError, match="packed_indices shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_norm_o_shape(self) -> None:
        codec = _make_codec()
        bad = self._make_payload(
            n_vectors=_N_VECTORS,
            packed_width=_HEAD_DIM // 8,
            meta_len=_N_VECTORS + 1,  # norm / ip mismatched against packed
        )
        with pytest.raises(ValueError, match="norm_o shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_ip_coeff_shape(self) -> None:
        codec = _make_codec()
        packed = mx.zeros((_N_VECTORS, _HEAD_DIM // 8), dtype=mx.uint8)
        norm = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        ip = mx.zeros((_N_VECTORS + 2,), dtype=mx.float16)
        resident = int(packed.nbytes + norm.nbytes + ip.nbytes)
        bad = RaBitQPayload(
            resident_bytes=resident,
            packed_indices=packed,
            norm_o=norm,
            ip_coeff=ip,
        )
        with pytest.raises(ValueError, match="ip_coeff shape"):
            codec.decode_tensor(bad)


class TestRaBitQ1BitByteAccounting:
    """logical_bytes and resident_bytes formula; consistency with payload."""

    def test_logical_bytes(self) -> None:
        codec = _make_codec()
        # fp16 = 2 bytes per coordinate
        assert codec.logical_bytes(100) == 100 * _N_KV_HEADS * _HEAD_DIM * 2

    def test_resident_bytes_formula(self) -> None:
        codec = _make_codec()
        n_vectors_per_block = _N_KV_HEADS * _BLOCK_SIZE
        bytes_per_vector = _HEAD_DIM // 8 + 4  # packed signs + fp16 norm_o + fp16 ip_coeff
        assert codec.resident_bytes(5) == 5 * n_vectors_per_block * bytes_per_vector

    def test_resident_bytes_matches_payload(self) -> None:
        """codec.resident_bytes(1) == payload.resident_bytes for one block."""
        codec = _make_codec()
        x = _make_random_input()
        payload = codec.encode_tensor(x)
        assert codec.resident_bytes(1) == payload.resident_bytes
