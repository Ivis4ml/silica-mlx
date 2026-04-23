"""P-5-B.2a unit tests for silica.vq.rabitq.ExtRaBitQ.

Companion to test_rabitq_1bit.py. Covers the multi-bit extension:
integer-grid codebook, per-vector fp16 dequantization scale, and
half-to-even rounding at midpoints.

Coverage:
- VectorCodec interface: encode returns ExtRaBitQPayload; decode shape
  + dtype across num_bits in {2, 3, 4}.
- Payload field shapes, dtypes, D-012 resident_bytes honesty (four
  mx.array fields — packed_indices + norm_o + ip_coeff + scale).
- Zero-vector batch semantics: norm_o = 0, ip_coeff = 0, packed
  indices all = 2^(B-1), decode -> zero tensor. Pins the exact
  half-to-even rounding at scaled=0.
- mx.round semantics: banker's rounding (half-to-even) at exact 0.5
  and 1.5 midpoints. Regression-locks the assumption ExtRaBitQ's
  encode depends on for parity with vqbench's np.round.
- Round-trip MSE band per num_bits + strict monotone improvement.
- ip_coeff non-negativity; norm_o non-negativity.
- Centroid invariant locked on fit() call.
- Constructor guards: num_bits not in {2, 3, 4} (including 1, 0,
  negative, 5+), head_dim % 8 != 0, fp32 dtype, zero shape dims.
- Encode shape / dtype guards (batch dim, n_kv_heads, block size,
  head_dim, wrong dtype).
- Decode shape guards (packed, norm_o, ip_coeff, scale each).
- logical_bytes / resident_bytes formula; codec.resident_bytes(1) ==
  payload.resident_bytes.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from silica.kvcache.codec import ExtRaBitQPayload, RaBitQPayload
from silica.vq import ExtRaBitQ

_BLOCK_SIZE = 16
_N_KV_HEADS = 4
_HEAD_DIM = 64
_SEED = 42
_N_VECTORS = _N_KV_HEADS * _BLOCK_SIZE  # 64


def _make_codec(
    *,
    num_bits: int,
    block_size: int = _BLOCK_SIZE,
    n_kv_heads: int = _N_KV_HEADS,
    head_dim: int = _HEAD_DIM,
    seed: int = _SEED,
    dtype: mx.Dtype = mx.float16,
) -> ExtRaBitQ:
    return ExtRaBitQ(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_bits=num_bits,
        seed=seed,
        dtype=dtype,
    )


def _make_random_input(
    *,
    block_size: int = _BLOCK_SIZE,
    n_kv_heads: int = _N_KV_HEADS,
    head_dim: int = _HEAD_DIM,
    dtype: mx.Dtype = mx.float16,
    seed: int = 0,
) -> mx.array:
    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal(
        (1, n_kv_heads, block_size, head_dim)
    ).astype(np.float32)
    return mx.array(x_np).astype(dtype)


def _equal_arrays(a: mx.array, b: mx.array) -> bool:
    return bool(mx.array_equal(a, b).item())


# =============================================================================
# mx.round semantics — regression-lock half-to-even before depending on it
# =============================================================================


class TestMLXRoundSemantics:
    """ExtRaBitQ's encode depends on ``mx.round`` being half-to-even
    (banker's rounding). Pinning this one assumption separately so a
    future MLX release that silently flipped to half-up would fire
    here immediately rather than cause quiet parity divergence from
    vqbench's ``np.round`` (which is half-to-even)."""

    def test_round_0_5_to_zero(self) -> None:
        assert float(mx.round(mx.array(0.5)).item()) == 0.0

    def test_round_1_5_to_two(self) -> None:
        assert float(mx.round(mx.array(1.5)).item()) == 2.0

    def test_round_negative_0_5_to_zero(self) -> None:
        assert float(mx.round(mx.array(-0.5)).item()) == 0.0

    def test_round_negative_1_5_to_negative_two(self) -> None:
        assert float(mx.round(mx.array(-1.5)).item()) == -2.0


# =============================================================================
# Interface — encode / decode surface
# =============================================================================


class TestExtRaBitQInterface:
    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_encode_returns_ext_rabitq_payload(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        payload = codec.encode_tensor(_make_random_input())
        assert isinstance(payload, ExtRaBitQPayload)
        # Subclass: ExtRaBitQPayload is also a RaBitQPayload.
        assert isinstance(payload, RaBitQPayload)

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_decode_shape(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        x = _make_random_input()
        x_hat = codec.decode_tensor(codec.encode_tensor(x))
        assert tuple(x_hat.shape) == (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM)

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_decode_dtype_fp16(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits, dtype=mx.float16)
        x = _make_random_input(dtype=mx.float16)
        x_hat = codec.decode_tensor(codec.encode_tensor(x))
        assert x_hat.dtype == mx.float16

    def test_decode_dtype_bf16(self) -> None:
        codec = _make_codec(num_bits=4, dtype=mx.bfloat16)
        x = _make_random_input(dtype=mx.bfloat16)
        x_hat = codec.decode_tensor(codec.encode_tensor(x))
        assert x_hat.dtype == mx.bfloat16


# =============================================================================
# Payload schema + D-012 resident_bytes honesty
# =============================================================================


class TestExtRaBitQPayload:
    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_packed_indices_shape_and_dtype(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        payload = codec.encode_tensor(_make_random_input())
        expected_width = num_bits * _HEAD_DIM // 8
        assert tuple(payload.packed_indices.shape) == (_N_VECTORS, expected_width)
        assert payload.packed_indices.dtype == mx.uint8

    def test_norm_o_shape_and_dtype(self) -> None:
        codec = _make_codec(num_bits=4)
        payload = codec.encode_tensor(_make_random_input())
        assert tuple(payload.norm_o.shape) == (_N_VECTORS,)
        assert payload.norm_o.dtype == mx.float16

    def test_ip_coeff_shape_and_dtype(self) -> None:
        codec = _make_codec(num_bits=4)
        payload = codec.encode_tensor(_make_random_input())
        assert tuple(payload.ip_coeff.shape) == (_N_VECTORS,)
        assert payload.ip_coeff.dtype == mx.float16

    def test_scale_shape_and_dtype(self) -> None:
        codec = _make_codec(num_bits=4)
        payload = codec.encode_tensor(_make_random_input())
        assert tuple(payload.scale.shape) == (_N_VECTORS,)
        assert payload.scale.dtype == mx.float16

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_resident_bytes_honesty(self, num_bits: int) -> None:
        """D-012: payload.resident_bytes == sum of all four mx.array
        field .nbytes. Catches the case where ExtRaBitQPayload's
        __post_init__ missed the scale field or the codec's manual
        total arithmetic drifted from the field widths."""
        codec = _make_codec(num_bits=num_bits)
        payload = codec.encode_tensor(_make_random_input())
        expected = (
            int(payload.packed_indices.nbytes)
            + int(payload.norm_o.nbytes)
            + int(payload.ip_coeff.nbytes)
            + int(payload.scale.nbytes)
        )
        assert payload.resident_bytes == expected

    def test_scale_is_constant_across_vectors_in_v01(self) -> None:
        """In v0.1 the per-vector scale is constant (same quant_scale
        derived from (num_bits, head_dim) applied to every vector).
        Pins that invariant; a future data-driven-scale variant would
        delete or adjust this test rather than silently relax the
        schema contract."""
        codec = _make_codec(num_bits=4)
        payload = codec.encode_tensor(_make_random_input())
        first = float(payload.scale[0].item())
        scale_fp64 = np.asarray(payload.scale).astype(np.float64)
        assert np.allclose(scale_fp64, first, rtol=0.0, atol=0.0)


# =============================================================================
# Centroid invariant (mirrors RaBitQ1Bit's equivalent class)
# =============================================================================


class TestExtRaBitQCentroid:
    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_centroid_shape_and_dtype(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        assert tuple(codec._centroid.shape) == (_HEAD_DIM,)
        assert codec._centroid.dtype == mx.float32

    def test_centroid_is_zero(self) -> None:
        codec = _make_codec(num_bits=4)
        zero = mx.zeros((_HEAD_DIM,), dtype=mx.float32)
        assert _equal_arrays(codec._centroid, zero)

    def test_fit_does_not_change_encode_output(self) -> None:
        """fit() is a no-op per §5.3; encode output must be
        bit-identical across packed_indices / norm_o / ip_coeff /
        scale after a fit(fake_corpus) call."""
        codec = _make_codec(num_bits=4)
        x = _make_random_input()
        before = codec.encode_tensor(x)
        rng = np.random.default_rng(99)
        fake_corpus = mx.array(
            rng.standard_normal((32, _HEAD_DIM)).astype(np.float32)
        )
        codec.fit(fake_corpus)
        after = codec.encode_tensor(x)
        assert _equal_arrays(before.packed_indices, after.packed_indices)
        assert _equal_arrays(before.norm_o, after.norm_o)
        assert _equal_arrays(before.ip_coeff, after.ip_coeff)
        assert _equal_arrays(before.scale, after.scale)


# =============================================================================
# Zero-vector semantics + edge cases
# =============================================================================


class TestExtRaBitQEdgeCases:
    @pytest.mark.parametrize("num_bits,expected_index", [(2, 2), (3, 4), (4, 8)])
    def test_zero_vector_packed_indices_are_2_to_B_minus_1(
        self, num_bits: int, expected_index: int
    ) -> None:
        """Zero vector -> Y=0 -> scaled=0 -> half-to-even round of
        (0 - 1) / 2 = -0.5 -> 0 -> rounded = +1 for all coords ->
        index = (1 + (2^B - 1)) / 2 = 2^(B - 1). Locks the exact
        codebook-index value per B (paper Eq. 3 + half-to-even
        rounding convention)."""
        codec = _make_codec(num_bits=num_bits)
        x_zero = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16
        )
        payload = codec.encode_tensor(x_zero)
        from silica.vq.core.packing import unpack_sub_byte

        indices = unpack_sub_byte(
            payload.packed_indices, num_bits=num_bits, d=_HEAD_DIM
        )
        # All coords should carry the same centre-of-codebook index.
        expected = mx.full(
            tuple(indices.shape), vals=expected_index, dtype=mx.uint8
        )
        assert _equal_arrays(indices, expected)

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_zero_vector_norm_o_is_zero(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        x_zero = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16
        )
        payload = codec.encode_tensor(x_zero)
        assert _equal_arrays(
            payload.norm_o, mx.zeros_like(payload.norm_o)
        )

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_zero_vector_ip_coeff_is_zero(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        x_zero = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16
        )
        payload = codec.encode_tensor(x_zero)
        assert _equal_arrays(
            payload.ip_coeff, mx.zeros_like(payload.ip_coeff)
        )

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_zero_vector_scale_is_constant(self, num_bits: int) -> None:
        """Zero-vector scale is the same constant ``scale_inv =
        3 / ((2^B - 1) · sqrt(d))`` as for any other vector; the zero
        edge case does not special-case the stored scale."""
        codec = _make_codec(num_bits=num_bits)
        x_zero = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16
        )
        payload = codec.encode_tensor(x_zero)
        expected_scale_inv = 3.0 / (
            (2 ** num_bits - 1) * math.sqrt(_HEAD_DIM)
        )
        scale_fp64 = np.asarray(payload.scale).astype(np.float64)
        np.testing.assert_allclose(
            scale_fp64, expected_scale_inv, rtol=5e-3, atol=5e-4
        )

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_zero_vector_decode_is_zero(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        x_zero = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16
        )
        x_hat = codec.decode_tensor(codec.encode_tensor(x_zero))
        assert _equal_arrays(x_hat, mx.zeros_like(x_hat))

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_norm_o_nonnegative(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        payload = codec.encode_tensor(_make_random_input())
        assert float(mx.min(payload.norm_o).item()) >= 0.0

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_ip_coeff_nonnegative(self, num_bits: int) -> None:
        """ip_coeff >= 0 always (the odd-integer codebook entries
        share signs with sign(Y), so <x_bar, Y> = sum |Y_i| · weight_i
        with non-negative weights). fp16 storage allows a tiny dip."""
        codec = _make_codec(num_bits=num_bits)
        payload = codec.encode_tensor(_make_random_input())
        assert float(mx.min(payload.ip_coeff).item()) >= -1e-3


# =============================================================================
# Round-trip MSE — B=4 < B=3 < B=2 (monotone), and absolute bounds
# =============================================================================


class TestExtRaBitQRoundTripMSE:
    """Round-trip MSE must decrease monotonically with num_bits on a
    fixed input. Not claimed as a cross-all-inputs mathematical
    theorem — the seed is fixed and the absolute bounds are generous
    but constraining enough to catch an algorithmic bug (e.g. wrong
    rotation direction, wrong codebook, missing re-normalization in
    decode)."""

    @staticmethod
    def _mse_at(num_bits: int, seed: int = 1234) -> float:
        codec = _make_codec(num_bits=num_bits)
        x = _make_random_input(seed=seed)
        x_fp32 = x.astype(mx.float32)
        x_hat_fp32 = codec.decode_tensor(codec.encode_tensor(x)).astype(
            mx.float32
        )
        return float(mx.mean((x_hat_fp32 - x_fp32) ** 2).item())

    def test_mse_decreases_with_num_bits(self) -> None:
        mse_b2 = self._mse_at(2)
        mse_b3 = self._mse_at(3)
        mse_b4 = self._mse_at(4)
        assert mse_b4 < mse_b3 < mse_b2, (
            f"MSE not monotone: B=2→{mse_b2:.4f}, B=3→{mse_b3:.4f}, "
            f"B=4→{mse_b4:.4f}"
        )

    def test_mse_b2_in_generous_band(self) -> None:
        # vqbench reference at matched input: ~0.263; use loose bound.
        assert 0.05 < self._mse_at(2) < 0.8

    def test_mse_b3_in_generous_band(self) -> None:
        # vqbench reference at matched input: ~0.059; use loose bound.
        assert 0.01 < self._mse_at(3) < 0.3

    def test_mse_b4_in_generous_band(self) -> None:
        # vqbench reference at matched input: ~0.013; use loose bound.
        assert 0.001 < self._mse_at(4) < 0.1


# =============================================================================
# Constructor guards
# =============================================================================


class TestExtRaBitQConstructorGuards:
    @pytest.mark.parametrize("bad_bits", [-1, 0, 1, 5, 8, 16])
    def test_rejects_num_bits_outside_range(self, bad_bits: int) -> None:
        with pytest.raises(ValueError, match="num_bits in"):
            ExtRaBitQ(
                block_size=16, n_kv_heads=4, head_dim=64, num_bits=bad_bits
            )

    def test_rejects_head_dim_not_multiple_of_8(self) -> None:
        with pytest.raises(ValueError, match="multiple of 8"):
            ExtRaBitQ(block_size=16, n_kv_heads=4, head_dim=63, num_bits=4)

    def test_rejects_zero_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            ExtRaBitQ(block_size=0, n_kv_heads=4, head_dim=64, num_bits=4)

    def test_rejects_zero_n_kv_heads(self) -> None:
        with pytest.raises(ValueError, match="n_kv_heads"):
            ExtRaBitQ(block_size=16, n_kv_heads=0, head_dim=64, num_bits=4)

    def test_rejects_zero_head_dim(self) -> None:
        with pytest.raises(ValueError, match="head_dim"):
            ExtRaBitQ(block_size=16, n_kv_heads=4, head_dim=0, num_bits=4)

    def test_rejects_fp32_dtype(self) -> None:
        with pytest.raises(ValueError, match="D-003"):
            ExtRaBitQ(
                block_size=16,
                n_kv_heads=4,
                head_dim=64,
                num_bits=4,
                dtype=mx.float32,
            )


# =============================================================================
# Encode shape / dtype guards
# =============================================================================


class TestExtRaBitQEncodeGuards:
    def test_rejects_wrong_batch_dim(self) -> None:
        codec = _make_codec(num_bits=4)
        x = mx.zeros(
            (2, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16
        )
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_n_kv_heads(self) -> None:
        codec = _make_codec(num_bits=4)
        x = mx.zeros(
            (1, _N_KV_HEADS + 1, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.float16
        )
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_block_size(self) -> None:
        codec = _make_codec(num_bits=4)
        x = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE + 1, _HEAD_DIM), dtype=mx.float16
        )
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_head_dim(self) -> None:
        codec = _make_codec(num_bits=4)
        x = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM * 2), dtype=mx.float16
        )
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)

    def test_rejects_wrong_input_dtype(self) -> None:
        codec = _make_codec(num_bits=4, dtype=mx.float16)
        x = mx.zeros(
            (1, _N_KV_HEADS, _BLOCK_SIZE, _HEAD_DIM), dtype=mx.bfloat16
        )
        with pytest.raises(ValueError, match="does not match"):
            codec.encode_tensor(x)


# =============================================================================
# Decode shape guards
# =============================================================================


class TestExtRaBitQDecodeGuards:
    def test_rejects_wrong_packed_shape(self) -> None:
        codec = _make_codec(num_bits=4)
        # Packed width wrong-for-num_bits: at num_bits=4 + d=64 the
        # codec expects (n, 32); we hand it (n, 33). The D-012
        # resident_bytes honesty check must pass here (payload is
        # well-formed internally) so the decode-time shape guard is
        # the one that fires.
        packed = mx.zeros((_N_VECTORS, _HEAD_DIM // 2 + 1), dtype=mx.uint8)
        norm = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        ip = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        scale = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        bad = ExtRaBitQPayload(
            resident_bytes=int(
                packed.nbytes + norm.nbytes + ip.nbytes + scale.nbytes
            ),
            packed_indices=packed,
            norm_o=norm,
            ip_coeff=ip,
            scale=scale,
        )
        with pytest.raises(ValueError, match="packed_indices shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_norm_o_shape(self) -> None:
        codec = _make_codec(num_bits=4)
        packed = mx.zeros(
            (_N_VECTORS, 4 * _HEAD_DIM // 8), dtype=mx.uint8
        )
        norm = mx.zeros((_N_VECTORS + 1,), dtype=mx.float16)
        ip = mx.zeros((_N_VECTORS + 1,), dtype=mx.float16)
        scale = mx.zeros((_N_VECTORS + 1,), dtype=mx.float16)
        bad = ExtRaBitQPayload(
            resident_bytes=int(
                packed.nbytes + norm.nbytes + ip.nbytes + scale.nbytes
            ),
            packed_indices=packed,
            norm_o=norm,
            ip_coeff=ip,
            scale=scale,
        )
        with pytest.raises(ValueError, match="norm_o shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_ip_coeff_shape(self) -> None:
        codec = _make_codec(num_bits=4)
        packed = mx.zeros(
            (_N_VECTORS, 4 * _HEAD_DIM // 8), dtype=mx.uint8
        )
        norm = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        ip = mx.zeros((_N_VECTORS + 2,), dtype=mx.float16)
        scale = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        bad = ExtRaBitQPayload(
            resident_bytes=int(
                packed.nbytes + norm.nbytes + ip.nbytes + scale.nbytes
            ),
            packed_indices=packed,
            norm_o=norm,
            ip_coeff=ip,
            scale=scale,
        )
        with pytest.raises(ValueError, match="ip_coeff shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_scale_shape(self) -> None:
        codec = _make_codec(num_bits=4)
        packed = mx.zeros(
            (_N_VECTORS, 4 * _HEAD_DIM // 8), dtype=mx.uint8
        )
        norm = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        ip = mx.zeros((_N_VECTORS,), dtype=mx.float16)
        scale = mx.zeros((_N_VECTORS + 3,), dtype=mx.float16)
        bad = ExtRaBitQPayload(
            resident_bytes=int(
                packed.nbytes + norm.nbytes + ip.nbytes + scale.nbytes
            ),
            packed_indices=packed,
            norm_o=norm,
            ip_coeff=ip,
            scale=scale,
        )
        with pytest.raises(ValueError, match="scale shape"):
            codec.decode_tensor(bad)


# =============================================================================
# Byte accounting
# =============================================================================


class TestExtRaBitQByteAccounting:
    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_logical_bytes(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        assert codec.logical_bytes(100) == 100 * _N_KV_HEADS * _HEAD_DIM * 2

    @pytest.mark.parametrize(
        "num_bits,expected_per_vec",
        [
            (2, 2 * 64 // 8 + 6),  # 16 packed + 6 metadata = 22
            (3, 3 * 64 // 8 + 6),  # 24 + 6 = 30
            (4, 4 * 64 // 8 + 6),  # 32 + 6 = 38
        ],
    )
    def test_resident_bytes_formula(
        self, num_bits: int, expected_per_vec: int
    ) -> None:
        codec = _make_codec(num_bits=num_bits)
        n_vec = _N_KV_HEADS * _BLOCK_SIZE
        assert codec.resident_bytes(5) == 5 * n_vec * expected_per_vec

    @pytest.mark.parametrize("num_bits", [2, 3, 4])
    def test_resident_bytes_matches_payload(self, num_bits: int) -> None:
        codec = _make_codec(num_bits=num_bits)
        payload = codec.encode_tensor(_make_random_input())
        assert codec.resident_bytes(1) == payload.resident_bytes
