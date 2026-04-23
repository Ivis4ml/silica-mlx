"""Tests for silica.vq.block_tq.BlockTurboQuantMSE + silica.vq.turboquant.TurboQuantMSE.

Covers:

- Construction-time validation (``head_dim % vq_block_size``,
  ``num_bits`` range, ``head_dim % 8`` for packing).
- ``VectorCodec`` Protocol conformance (side-level, P-5-A.0.4).
- ``BlockTQPayload`` honesty (D-012): packed indices + scales sizes
  match the payload's declared ``resident_bytes``.
- Round-trip shape + dtype preservation: input
  ``(1, n_kv_heads, kv_block_size, head_dim)`` fp16 round-trips to an
  identically-shaped fp16 tensor after encode → decode.
- **Scalar equivalence invariant** (docs/P5_OPENING.md §4.5 +
  vqbench's ``tests/test_block_quant.py::test_block_equals_scalar_when_B_equals_d``):
  ``BlockTurboQuantMSE(vq_block_size=head_dim)`` and
  ``TurboQuantMSE(...)`` with matching ``(head_dim, num_bits, seed)``
  produce bit-identical output on the same input tensor.
- Reconstruction error bounds on a synthetic gaussian tensor — rel MSE
  should be <5% at num_bits=4 and <10% at num_bits=3.
- ``norm_correction`` toggle changes the output (on by default).
- ``logical_bytes`` / ``resident_bytes`` per-side arithmetic matches
  hand-computed reference (D-012 + §4.3 of the opening doc).
- D-009 quarantine: ``silica.vq.block_tq`` and ``silica.vq.turboquant``
  do not import NumPy on the runtime hot path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from silica.kvcache.codec import BlockTQPayload, VectorCodec
from silica.vq import BlockTurboQuantMSE, TurboQuantMSE

# Per-test shape defaults — shared with vqbench's standard bench row
# (head_dim ∈ {64, 128, 256}; vq_block_size ∈ {32, 64}; num_bits ∈ {3, 4}).
BLOCK_SIZE = 16
N_KV_HEADS = 4
HEAD_DIM = 64
VQ_BLOCK_SIZE = 32
NUM_BITS = 4
DTYPE = mx.float16


def _input_tensor(seed: int = 0) -> mx.array:
    """One side (K or V) of one detached block — shape
    ``(1, n_kv_heads, kv_block_size, head_dim)`` fp16."""
    mx.random.seed(seed)
    return mx.random.normal(shape=(1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)).astype(DTYPE)


def _codec(**overrides: Any) -> BlockTurboQuantMSE:
    kw: dict[str, Any] = {
        "block_size": BLOCK_SIZE,
        "n_kv_heads": N_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vq_block_size": VQ_BLOCK_SIZE,
        "num_bits": NUM_BITS,
    }
    kw.update(overrides)
    return BlockTurboQuantMSE(**kw)


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_rejects_non_divisible_vq_block_size() -> None:
    with pytest.raises(ValueError, match="vq_block_size"):
        _codec(vq_block_size=30)  # 64 % 30 != 0


def test_rejects_unsupported_num_bits() -> None:
    for bad in (0, 5, 8, 16, -1):
        with pytest.raises(ValueError, match="num_bits"):
            _codec(num_bits=bad)


def test_rejects_head_dim_not_multiple_of_8() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        BlockTurboQuantMSE(
            block_size=BLOCK_SIZE,
            n_kv_heads=N_KV_HEADS,
            head_dim=7,
            vq_block_size=7,
            num_bits=NUM_BITS,
        )


def test_rejects_nonpositive_block_size() -> None:
    with pytest.raises(ValueError, match="block_size"):
        _codec(block_size=0)


def test_rejects_nonpositive_n_kv_heads() -> None:
    with pytest.raises(ValueError, match="n_kv_heads"):
        _codec(n_kv_heads=0)


def test_rejects_nonpositive_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        _codec(head_dim=0)


def test_rejects_fp32_dtype() -> None:
    """D-003: decode_tensor output is consumed by fp16 / bf16 SDPA;
    silently returning fp32 would double active-KV residency and
    break the budgeter's fp16 baseline arithmetic."""
    with pytest.raises(ValueError, match="dtype"):
        _codec(dtype=mx.float32)


def test_accepts_bfloat16_dtype() -> None:
    """bf16 is in the allowed set so Gemma-style K/V layouts can install
    BlockTQ without a dtype widening."""
    codec = _codec(dtype=mx.bfloat16)
    assert codec.dtype == mx.bfloat16


# ---------------------------------------------------------------------------
# encode_tensor input validation (shape + dtype)
# ---------------------------------------------------------------------------


def test_encode_tensor_rejects_wrong_n_kv_heads() -> None:
    """Input shape `(1, wrong_heads, block_size, head_dim)` with the
    same total element count as the codec's configured shape would
    silently quantize into a wrong layout without this check."""
    codec = _codec()  # n_kv_heads=4, block_size=16, head_dim=64
    # Same total elements (4 * 16 * 64 = 4096) but different (heads, block).
    x = mx.zeros((1, 8, 8, HEAD_DIM), dtype=DTYPE)
    with pytest.raises(ValueError, match="input shape"):
        codec.encode_tensor(x)


def test_encode_tensor_rejects_wrong_block_size() -> None:
    codec = _codec()
    # Same total elements but (block_size, head_dim) swapped arithmetically.
    x = mx.zeros((1, N_KV_HEADS, 32, 32), dtype=DTYPE)
    with pytest.raises(ValueError, match="input shape"):
        codec.encode_tensor(x)


def test_encode_tensor_rejects_wrong_head_dim() -> None:
    codec = _codec()
    x = mx.zeros((1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM // 2), dtype=DTYPE)
    with pytest.raises(ValueError, match="input shape"):
        codec.encode_tensor(x)


def test_encode_tensor_rejects_wrong_leading_axis() -> None:
    codec = _codec()
    x = mx.zeros((2, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=DTYPE)
    with pytest.raises(ValueError, match="input shape"):
        codec.encode_tensor(x)


def test_encode_tensor_rejects_wrong_dtype() -> None:
    codec = _codec(dtype=mx.float16)
    x = mx.zeros((1, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM), dtype=mx.bfloat16)
    with pytest.raises(ValueError, match="input dtype"):
        codec.encode_tensor(x)


# ---------------------------------------------------------------------------
# VectorCodec Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_vector_codec_protocol() -> None:
    codec = _codec()
    assert isinstance(codec, VectorCodec)


def test_exposes_required_attributes() -> None:
    codec = _codec()
    assert codec.block_size == BLOCK_SIZE
    assert codec.dtype == DTYPE


# ---------------------------------------------------------------------------
# Payload shape + honesty
# ---------------------------------------------------------------------------


def test_payload_is_block_tq_payload() -> None:
    codec = _codec()
    payload = codec.encode_tensor(_input_tensor())
    assert isinstance(payload, BlockTQPayload)


def test_packed_indices_shape_and_dtype() -> None:
    codec = _codec()
    payload = codec.encode_tensor(_input_tensor())
    n_vectors = 1 * N_KV_HEADS * BLOCK_SIZE  # (1, heads, block, head_dim) flattened
    packed_width = NUM_BITS * HEAD_DIM // 8
    assert payload.packed_indices.shape == (n_vectors, packed_width)
    assert payload.packed_indices.dtype == mx.uint8


def test_scales_shape_and_dtype() -> None:
    codec = _codec()
    payload = codec.encode_tensor(_input_tensor())
    n_vectors = 1 * N_KV_HEADS * BLOCK_SIZE
    num_vq_blocks = HEAD_DIM // VQ_BLOCK_SIZE
    assert payload.scales.shape == (n_vectors, num_vq_blocks)
    assert payload.scales.dtype == mx.float16


def test_payload_resident_bytes_is_honest() -> None:
    """D-012: ``BlockTQPayload.__post_init__`` already enforces that
    ``resident_bytes`` equals the sum of ``.nbytes`` across array
    fields. This test additionally pins the codec's arithmetic: the
    payload's declared bytes match the codec's own ``resident_bytes(1)``
    (one block)."""
    codec = _codec()
    payload = codec.encode_tensor(_input_tensor())
    expected_one_block = codec.resident_bytes(num_blocks=1)
    assert payload.resident_bytes == expected_one_block


# ---------------------------------------------------------------------------
# Round-trip shape + dtype
# ---------------------------------------------------------------------------


def test_decode_preserves_input_shape_and_dtype() -> None:
    codec = _codec()
    x = _input_tensor()
    out = codec.decode_tensor(codec.encode_tensor(x))
    assert out.shape == x.shape
    assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# Scalar equivalence invariant (B = head_dim)
# ---------------------------------------------------------------------------


def test_block_equals_scalar_when_b_equals_d() -> None:
    """``BlockTurboQuantMSE(vq_block_size=head_dim)`` reduces to scalar
    TurboQuantMSE (one block covers the whole vector). Output must be
    bit-identical to ``TurboQuantMSE(head_dim=d)`` under matching
    ``(num_bits, seed, norm_correction)``."""
    x = _input_tensor()

    block_scalar = BlockTurboQuantMSE(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        vq_block_size=HEAD_DIM,
        num_bits=NUM_BITS,
    )
    scalar = TurboQuantMSE(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_bits=NUM_BITS,
    )

    out_block = block_scalar.decode_tensor(block_scalar.encode_tensor(x))
    out_scalar = scalar.decode_tensor(scalar.encode_tensor(x))
    # Bit-identical — both paths call the same class body with the same args.
    assert bool(mx.array_equal(out_block, out_scalar).item())


def test_turboquant_mse_is_block_tq_instance() -> None:
    """``TurboQuantMSE`` is a factory returning a ``BlockTurboQuantMSE``
    instance; no separate class body."""
    scalar = TurboQuantMSE(
        block_size=BLOCK_SIZE,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_bits=NUM_BITS,
    )
    assert isinstance(scalar, BlockTurboQuantMSE)
    # vq_block_size defaulted to head_dim.
    assert scalar._vq_block_size == HEAD_DIM
    assert scalar._num_vq_blocks == 1


# ---------------------------------------------------------------------------
# Reconstruction error bounds
# ---------------------------------------------------------------------------


def _relative_mse(x: mx.array, x_hat: mx.array) -> float:
    err = x.astype(mx.float32) - x_hat.astype(mx.float32)
    mse = float(mx.mean(err * err).item())
    norm_sq = float(mx.mean(x.astype(mx.float32) ** 2).item())
    return mse / max(norm_sq, 1e-30)


def test_recon_error_num_bits_4_is_bounded() -> None:
    """4-bit BlockTQ with B=32 on unit-variance Gaussian: rel MSE should
    be well under 5%. Empirically lands around 0.8%."""
    codec = _codec(num_bits=4)
    x = _input_tensor()
    out = codec.decode_tensor(codec.encode_tensor(x))
    rel_mse = _relative_mse(x, out)
    assert rel_mse < 0.05, f"4-bit rel_mse {rel_mse:.4e} exceeds 5% bound"


def test_recon_error_num_bits_3_is_bounded() -> None:
    """3-bit BlockTQ: rel MSE should be under 10% on Gaussian input."""
    codec = _codec(num_bits=3)
    x = _input_tensor()
    out = codec.decode_tensor(codec.encode_tensor(x))
    rel_mse = _relative_mse(x, out)
    assert rel_mse < 0.10, f"3-bit rel_mse {rel_mse:.4e} exceeds 10% bound"


def test_recon_error_improves_with_more_bits() -> None:
    """Monotone in num_bits: 4-bit recon error < 3-bit < 2-bit."""
    x = _input_tensor()
    errs = []
    for bits in (2, 3, 4):
        codec = _codec(num_bits=bits)
        out = codec.decode_tensor(codec.encode_tensor(x))
        errs.append(_relative_mse(x, out))
    assert errs[0] > errs[1] > errs[2], (
        f"recon error should decrease monotonically with num_bits; got "
        f"bits=2: {errs[0]:.4e}, bits=3: {errs[1]:.4e}, bits=4: {errs[2]:.4e}"
    )


# ---------------------------------------------------------------------------
# norm_correction toggle
# ---------------------------------------------------------------------------


def test_norm_correction_changes_output() -> None:
    """``norm_correction=False`` produces a numerically different output
    from ``norm_correction=True`` — regression-locks that the flag is
    actually consulted in ``decode_tensor``."""
    x = _input_tensor()
    codec_on = _codec(norm_correction=True)
    codec_off = _codec(norm_correction=False)
    out_on = codec_on.decode_tensor(codec_on.encode_tensor(x))
    out_off = codec_off.decode_tensor(codec_off.encode_tensor(x))
    assert not bool(mx.array_equal(out_on, out_off).item())


# ---------------------------------------------------------------------------
# Byte accounting (per-side)
# ---------------------------------------------------------------------------


def test_logical_bytes_matches_fp16_baseline() -> None:
    codec = _codec()
    per_token = N_KV_HEADS * HEAD_DIM * DTYPE.size
    assert codec.logical_bytes(0) == 0
    assert codec.logical_bytes(1) == per_token
    assert codec.logical_bytes(100) == 100 * per_token


def test_resident_bytes_arithmetic_matches_formula() -> None:
    """``resident_bytes(num_blocks)`` = ``num_blocks × n_vectors_per_block
    × (packed_bytes + scale_bytes)`` per side."""
    codec = _codec()
    n_vectors_per_block = N_KV_HEADS * BLOCK_SIZE
    packed_bytes_per_vector = NUM_BITS * HEAD_DIM // 8
    num_vq_blocks = HEAD_DIM // VQ_BLOCK_SIZE
    scale_bytes_per_vector = num_vq_blocks * 2  # fp16
    bytes_per_block = n_vectors_per_block * (
        packed_bytes_per_vector + scale_bytes_per_vector
    )
    for n in (0, 1, 3, 17):
        assert codec.resident_bytes(n) == n * bytes_per_block


def test_compression_ratio_vs_fp16() -> None:
    """``logical_bytes(block_size × num_blocks)`` / ``resident_bytes(num_blocks)``
    gives the compression ratio; should be materially > 1 at 4-bit."""
    codec = _codec(num_bits=4, vq_block_size=64, head_dim=256)
    n_blocks = 10
    logical = codec.logical_bytes(BLOCK_SIZE * n_blocks)
    resident = codec.resident_bytes(n_blocks)
    ratio = logical / resident
    # 4-bit K with B=64 scale overhead — effective bits/val = 4 + 16/64 = 4.25
    # so fp16 (16) / 4.25 ≈ 3.76×.
    assert 3.5 < ratio < 4.0, (
        f"4-bit B=64 compression ratio {ratio:.2f} not in expected range "
        f"(3.5, 4.0); vqbench REPORT §3.1 pins 3.76×"
    )


# ---------------------------------------------------------------------------
# D-012 honesty via store integration (light smoke — full coverage in
# test_prefix_store.py)
# ---------------------------------------------------------------------------


def test_payload_honest_resident_bytes_via_payload_init() -> None:
    """Constructing a payload directly with a wrong resident_bytes raises;
    the codec's payload construction path must satisfy the honesty check."""
    codec = _codec()
    payload = codec.encode_tensor(_input_tensor())
    # Hand-compute the expected sum of array .nbytes.
    expected = int(payload.packed_indices.nbytes) + int(payload.scales.nbytes)
    assert payload.resident_bytes == expected


# ---------------------------------------------------------------------------
# decode_tensor payload shape validation — protects against a payload
# built for a different (head_dim, num_bits, vq_block_size) configuration
# silently reshaping into the wrong semantic layout on the codec's return.
# Mirrors TestRaBitQ1BitDecodeGuards / TestExtRaBitQDecodeGuards.
# ---------------------------------------------------------------------------


class TestBlockTQDecodeGuards:
    """decode_tensor rejects payloads whose array shapes disagree with the
    codec's (n_kv_heads, block_size, head_dim, num_bits, vq_block_size)
    configuration."""

    @staticmethod
    def _make_payload(
        *,
        packed_n_vectors: int,
        packed_width: int,
        scales_n_vectors: int,
        scales_num_vq_blocks: int,
    ) -> BlockTQPayload:
        packed = mx.zeros((packed_n_vectors, packed_width), dtype=mx.uint8)
        scales = mx.zeros(
            (scales_n_vectors, scales_num_vq_blocks), dtype=mx.float16
        )
        resident = int(packed.nbytes) + int(scales.nbytes)
        return BlockTQPayload(
            resident_bytes=resident,
            packed_indices=packed,
            scales=scales,
        )

    def test_rejects_wrong_packed_n_vectors(self) -> None:
        codec = _codec()
        n_vectors = N_KV_HEADS * BLOCK_SIZE
        num_vq_blocks = HEAD_DIM // VQ_BLOCK_SIZE
        bad = self._make_payload(
            # off-by-N_KV_HEADS: e.g. block_size mismatch on the source
            packed_n_vectors=n_vectors + N_KV_HEADS,
            packed_width=NUM_BITS * HEAD_DIM // 8,
            scales_n_vectors=n_vectors + N_KV_HEADS,
            scales_num_vq_blocks=num_vq_blocks,
        )
        with pytest.raises(ValueError, match="packed_indices shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_packed_width(self) -> None:
        codec = _codec()
        n_vectors = N_KV_HEADS * BLOCK_SIZE
        num_vq_blocks = HEAD_DIM // VQ_BLOCK_SIZE
        bad = self._make_payload(
            packed_n_vectors=n_vectors,
            # e.g. payload was packed under a different num_bits
            packed_width=(NUM_BITS + 1) * HEAD_DIM // 8,
            scales_n_vectors=n_vectors,
            scales_num_vq_blocks=num_vq_blocks,
        )
        with pytest.raises(ValueError, match="packed_indices shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_scales_n_vectors(self) -> None:
        codec = _codec()
        n_vectors = N_KV_HEADS * BLOCK_SIZE
        num_vq_blocks = HEAD_DIM // VQ_BLOCK_SIZE
        bad = self._make_payload(
            packed_n_vectors=n_vectors,
            packed_width=NUM_BITS * HEAD_DIM // 8,
            # scales length disagrees with packed_indices — mismatched
            # payload assembly
            scales_n_vectors=n_vectors + 1,
            scales_num_vq_blocks=num_vq_blocks,
        )
        with pytest.raises(ValueError, match="scales shape"):
            codec.decode_tensor(bad)

    def test_rejects_wrong_scales_num_vq_blocks(self) -> None:
        codec = _codec()
        n_vectors = N_KV_HEADS * BLOCK_SIZE
        num_vq_blocks = HEAD_DIM // VQ_BLOCK_SIZE
        bad = self._make_payload(
            packed_n_vectors=n_vectors,
            packed_width=NUM_BITS * HEAD_DIM // 8,
            scales_n_vectors=n_vectors,
            # e.g. payload was built under a different vq_block_size
            scales_num_vq_blocks=num_vq_blocks + 1,
        )
        with pytest.raises(ValueError, match="scales shape"):
            codec.decode_tensor(bad)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_output() -> None:
    """Two codec instances at the same ``(head_dim, seed)`` share the
    cached Haar rotation and produce identical outputs on the same
    input."""
    x = _input_tensor()
    c1 = _codec(seed=99)
    c2 = _codec(seed=99)
    out1 = c1.decode_tensor(c1.encode_tensor(x))
    out2 = c2.decode_tensor(c2.encode_tensor(x))
    assert bool(mx.array_equal(out1, out2).item())


def test_different_seeds_different_output() -> None:
    x = _input_tensor()
    c1 = _codec(seed=1)
    c2 = _codec(seed=2)
    out1 = c1.decode_tensor(c1.encode_tensor(x))
    out2 = c2.decode_tensor(c2.encode_tensor(x))
    assert not bool(mx.array_equal(out1, out2).item())


# ---------------------------------------------------------------------------
# D-009 quarantine — block_tq.py + turboquant.py have no NumPy on the
# runtime hot path
# ---------------------------------------------------------------------------


def test_block_tq_module_has_no_numpy_import() -> None:
    import silica.vq.block_tq as mod

    src = Path(mod.__file__).read_text()
    assert "import numpy" not in src, (
        "silica.vq.block_tq must not import NumPy on the hot path; "
        "calibration-time NumPy lives in silica.vq._calibration"
    )
    assert "from numpy" not in src


def test_turboquant_module_has_no_numpy_import() -> None:
    import silica.vq.turboquant as mod

    src = Path(mod.__file__).read_text()
    assert "import numpy" not in src
    assert "from numpy" not in src


# ---------------------------------------------------------------------------
# Sanity — encode + decode produce valid packed indices (unpack matches
# re-pack of uint8 indices in [0, 2^num_bits))
# ---------------------------------------------------------------------------


def test_packed_indices_values_in_codebook_range() -> None:
    """Every encoded index must be in ``[0, 2^num_bits)``. Rely on
    ``unpack_sub_byte`` + ``mx.max`` to bound the indices."""
    from silica.vq.core.packing import unpack_sub_byte

    codec = _codec()
    payload = codec.encode_tensor(_input_tensor())
    indices = unpack_sub_byte(
        payload.packed_indices, num_bits=NUM_BITS, d=HEAD_DIM
    )
    max_idx = int(mx.max(indices).item())
    codebook_size = 1 << NUM_BITS
    assert 0 <= max_idx < codebook_size, (
        f"encoded indices out of codebook range [0, {codebook_size}); "
        f"max seen = {max_idx}"
    )


# ---------------------------------------------------------------------------
# Shape variety — different (n_kv_heads, head_dim, vq_block_size)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_kv_heads,head_dim,vq_block_size,num_bits",
    [
        (2, 64, 32, 4),
        (4, 128, 64, 4),
        (8, 256, 64, 4),
        (4, 128, 32, 3),
        (4, 64, 64, 4),  # scalar case
    ],
)
def test_round_trip_shape_variety(
    n_kv_heads: int, head_dim: int, vq_block_size: int, num_bits: int
) -> None:
    codec = BlockTurboQuantMSE(
        block_size=BLOCK_SIZE,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=vq_block_size,
        num_bits=num_bits,
    )
    x = mx.random.normal(shape=(1, n_kv_heads, BLOCK_SIZE, head_dim)).astype(DTYPE)
    out = codec.decode_tensor(codec.encode_tensor(x))
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    # Reconstruction should be finite (no NaN / inf).
    assert bool(mx.all(mx.isfinite(out)).item())
