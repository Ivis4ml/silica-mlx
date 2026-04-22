"""Tests for silica.vq.core.packing — sub-byte bit-plane pack / unpack.

Covers:
- Round-trip identity at every supported ``num_bits`` in ``{1, 2, 3, 4}``.
- Output shape matches ``num_bits * d // 8``.
- Leading-axis arity: 1-D, 2-D, 3-D, 4-D inputs all round-trip.
- dtype preserved (``mx.uint8`` throughout).
- Endianness: MSB-first within each byte (bit of value 0 lives in the
  byte's high bit).
- Bit-plane layout: planes are concatenated along the last axis in the
  order bit-0 ... bit-(num_bits - 1).
- Integration with ``BlockTQPayload`` / ``RaBitQPayload`` D-012 honesty.
- Error cases: bad ``num_bits``, non-uint8 input, ``d`` not a multiple of 8,
  mismatched ``packed`` last-axis size on unpack.

See ``docs/P5_OPENING.md`` §4.1 (payload schemas), §5 (codec catalogue),
and §4 (D-009 MLX-native constraint) for how this helper is consumed.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from silica.kvcache.codec import BlockTQPayload, RaBitQPayload
from silica.vq.core.packing import pack_sub_byte, unpack_sub_byte

# -----------------------------------------------------------------------------
# Round-trip identity at each bit width
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(64,), (10, 64), (2, 4, 64), (2, 3, 4, 64)])
def test_round_trip_identity(num_bits: int, shape: tuple[int, ...]) -> None:
    d = shape[-1]
    max_val = 2 ** num_bits
    x = mx.random.randint(0, max_val, shape=shape).astype(mx.uint8)
    packed = pack_sub_byte(x, num_bits=num_bits)
    unpacked = unpack_sub_byte(packed, num_bits=num_bits, d=d)
    assert unpacked.shape == x.shape
    assert unpacked.dtype == mx.uint8
    assert bool(mx.array_equal(unpacked, x).item())


# -----------------------------------------------------------------------------
# Output shape and dtype
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
@pytest.mark.parametrize("d", [8, 16, 64, 256])
def test_packed_shape_matches_formula(num_bits: int, d: int) -> None:
    x = mx.zeros((5, d), dtype=mx.uint8)
    packed = pack_sub_byte(x, num_bits=num_bits)
    expected = num_bits * d // 8
    assert packed.shape == (5, expected)
    # Also matches the ceil(d * num_bits / 8) form the payload schemas use
    # (equal when d % 8 == 0).
    assert expected == math.ceil(d * num_bits / 8)


def test_packed_dtype_is_uint8() -> None:
    x = mx.random.randint(0, 16, shape=(4, 32)).astype(mx.uint8)
    packed = pack_sub_byte(x, num_bits=4)
    assert packed.dtype == mx.uint8


def test_unpacked_dtype_is_uint8() -> None:
    x = mx.random.randint(0, 16, shape=(4, 32)).astype(mx.uint8)
    packed = pack_sub_byte(x, num_bits=4)
    unpacked = unpack_sub_byte(packed, num_bits=4, d=32)
    assert unpacked.dtype == mx.uint8


# -----------------------------------------------------------------------------
# MSB-first within byte + bit-plane layout across bytes
# -----------------------------------------------------------------------------


def test_msb_first_within_byte_num_bits_1() -> None:
    """Value 1 at position 0 lands in the high bit of byte 0."""
    x = mx.array([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.uint8)
    packed = pack_sub_byte(x, num_bits=1)
    assert packed.shape == (1, 1)
    assert int(packed[0, 0].item()) == 0b10000000  # 128


def test_msb_first_within_byte_trailing_bit() -> None:
    """Value 1 at position 7 lands in the low bit of byte 0."""
    x = mx.array([[0, 0, 0, 0, 0, 0, 0, 1]], dtype=mx.uint8)
    packed = pack_sub_byte(x, num_bits=1)
    assert int(packed[0, 0].item()) == 0b00000001  # 1


def test_bit_plane_layout_num_bits_4() -> None:
    """With a single value set to 2^k at position 0, only plane k is nonzero;
    that byte equals 0b10000000 (MSB-first within the plane)."""
    for k in range(4):
        x = mx.zeros((1, 8), dtype=mx.uint8)
        x_list = [0] * 8
        x_list[0] = 1 << k  # value with only bit-k set
        x = mx.array([x_list], dtype=mx.uint8)
        packed = pack_sub_byte(x, num_bits=4)
        assert packed.shape == (1, 4)  # 4 planes × 1 byte
        for plane_idx in range(4):
            if plane_idx == k:
                assert int(packed[0, plane_idx].item()) == 0b10000000, (
                    f"k={k} plane {plane_idx} should be 128, got "
                    f"{int(packed[0, plane_idx].item())}"
                )
            else:
                assert int(packed[0, plane_idx].item()) == 0, (
                    f"k={k} plane {plane_idx} should be 0, got "
                    f"{int(packed[0, plane_idx].item())}"
                )


def test_pack_ignores_high_bits_outside_num_bits() -> None:
    """Values beyond ``2 ** num_bits`` are reduced modulo via the low-bit
    mask. num_bits=3 → only low 3 bits count; 0b1000 ≡ 0."""
    x = mx.array([[0b1000, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.uint8)  # 8 → low 3 bits = 0
    packed = pack_sub_byte(x, num_bits=3)
    # All three planes should be zero since the low 3 bits of 8 are 000.
    for plane_idx in range(3):
        assert int(packed[0, plane_idx].item()) == 0


# -----------------------------------------------------------------------------
# Integration with payload honesty checks
# -----------------------------------------------------------------------------


def test_pack_output_passes_block_tq_payload_honesty() -> None:
    """A BlockTQPayload built from pack_sub_byte output should pass D-012
    honesty when resident_bytes equals packed.nbytes + scales.nbytes."""
    head_dim = 256
    num_bits = 4
    vq_block_size = 64
    n_vectors = 4 * 16  # n_kv_heads * kv_block_size
    n_vq_blocks = head_dim // vq_block_size

    indices = mx.random.randint(0, 2 ** num_bits, shape=(n_vectors, head_dim)).astype(
        mx.uint8
    )
    packed = pack_sub_byte(indices, num_bits=num_bits)
    assert packed.shape == (n_vectors, num_bits * head_dim // 8)

    scales = mx.ones((n_vectors, n_vq_blocks), dtype=mx.float16)
    total = int(packed.nbytes) + int(scales.nbytes)
    payload = BlockTQPayload(
        resident_bytes=total, packed_indices=packed, scales=scales
    )
    assert payload.resident_bytes == total


def test_pack_output_passes_rabitq_payload_honesty() -> None:
    """A RaBitQPayload (1-bit) built from pack_sub_byte output should pass
    D-012 honesty."""
    head_dim = 64
    num_bits = 1
    n_vectors = 4 * 16

    indices = mx.random.randint(0, 2, shape=(n_vectors, head_dim)).astype(mx.uint8)
    packed = pack_sub_byte(indices, num_bits=num_bits)
    assert packed.shape == (n_vectors, head_dim // 8)

    norm_o = mx.ones((n_vectors,), dtype=mx.float16)
    ip_coeff = mx.ones((n_vectors,), dtype=mx.float16)
    total = int(packed.nbytes) + int(norm_o.nbytes) + int(ip_coeff.nbytes)
    RaBitQPayload(
        resident_bytes=total,
        packed_indices=packed,
        norm_o=norm_o,
        ip_coeff=ip_coeff,
    )


# -----------------------------------------------------------------------------
# Error cases
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("num_bits", [0, 5, 6, 7, 8, 9, 16, -1])
def test_pack_rejects_unsupported_num_bits(num_bits: int) -> None:
    x = mx.zeros((1, 8), dtype=mx.uint8)
    with pytest.raises(ValueError, match="num_bits"):
        pack_sub_byte(x, num_bits=num_bits)


def test_pack_rejects_non_uint8_input() -> None:
    x = mx.zeros((1, 8), dtype=mx.int32)
    with pytest.raises(TypeError, match="uint8"):
        pack_sub_byte(x, num_bits=4)


@pytest.mark.parametrize("d", [1, 3, 7, 9, 15, 33])
def test_pack_rejects_d_not_multiple_of_8(d: int) -> None:
    x = mx.zeros((1, d), dtype=mx.uint8)
    with pytest.raises(ValueError, match="multiple of 8"):
        pack_sub_byte(x, num_bits=4)


@pytest.mark.parametrize("num_bits", [0, 5, 8, -1])
def test_unpack_rejects_unsupported_num_bits(num_bits: int) -> None:
    packed = mx.zeros((1, 4), dtype=mx.uint8)
    with pytest.raises(ValueError, match="num_bits"):
        unpack_sub_byte(packed, num_bits=num_bits, d=8)


def test_unpack_rejects_non_uint8_input() -> None:
    packed = mx.zeros((1, 4), dtype=mx.int32)
    with pytest.raises(TypeError, match="uint8"):
        unpack_sub_byte(packed, num_bits=4, d=8)


def test_unpack_rejects_d_not_multiple_of_8() -> None:
    packed = mx.zeros((1, 4), dtype=mx.uint8)
    with pytest.raises(ValueError, match="multiple of 8"):
        unpack_sub_byte(packed, num_bits=4, d=7)


def test_unpack_rejects_wrong_packed_last_axis() -> None:
    """packed last-axis size must equal num_bits * d // 8."""
    packed = mx.zeros((1, 10), dtype=mx.uint8)  # 10 != 4 * 64 / 8 = 32
    with pytest.raises(ValueError, match="last-axis size"):
        unpack_sub_byte(packed, num_bits=4, d=64)


# -----------------------------------------------------------------------------
# D-009 sanity: the module uses only mx.* primitives
# -----------------------------------------------------------------------------


def test_packing_module_has_no_numpy_import() -> None:
    """D-009: runtime hot-path modules must not import NumPy. Greps the
    module source for ``import numpy`` and ``from numpy`` variants."""
    from pathlib import Path

    import silica.vq.core.packing as pkg

    src = Path(pkg.__file__).read_text()
    assert "import numpy" not in src
    assert "from numpy" not in src
