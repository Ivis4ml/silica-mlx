"""Tests for the P-5 side-level VectorCodec[P] Protocol and CodedPayload hierarchy.

Covers the scaffolding added in P-5-A.0 to ``silica.kvcache.codec``:

- ``CodedPayload._verify_resident_bytes`` enforces D-012 honesty at
  construction: the declared ``resident_bytes`` must equal the sum of
  ``.nbytes`` across every ``mx.array`` field on the concrete subclass.
- ``RawFp16Payload`` / ``BlockTQPayload`` / ``RaBitQPayload`` instantiate
  cleanly when ``resident_bytes`` is honest, and raise ``ValueError``
  otherwise.
- ``VectorCodec`` is ``@runtime_checkable`` and detects method presence
  across the four required members.

See ``docs/P5_OPENING.md`` §4.3 for the resident_bytes-honesty rule and
§4.1 for the VectorCodec[P] Protocol shape.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from silica.kvcache.codec import (
    BlockTQPayload,
    CodedPayload,
    RaBitQPayload,
    RawFp16Payload,
    VectorCodec,
)

# -----------------------------------------------------------------------------
# CodedPayload._verify_resident_bytes honesty
# -----------------------------------------------------------------------------


def test_raw_fp16_payload_accepts_honest_resident_bytes() -> None:
    t = mx.zeros(shape=(1, 4, 16, 64), dtype=mx.float16)
    payload = RawFp16Payload(resident_bytes=int(t.nbytes), t=t)
    assert payload.resident_bytes == t.nbytes
    assert payload.t is t


def test_raw_fp16_payload_rejects_mismatched_resident_bytes() -> None:
    t = mx.zeros(shape=(1, 4, 16, 64), dtype=mx.float16)
    with pytest.raises(ValueError, match="resident_bytes"):
        RawFp16Payload(resident_bytes=int(t.nbytes) + 1, t=t)


def test_block_tq_payload_accepts_honest_resident_bytes() -> None:
    head_dim = 64
    num_bits = 4
    vq_block_size = 32
    n_vectors = 4 * 16  # n_kv_heads * kv_block_size
    n_vq_blocks = head_dim // vq_block_size
    packed_width = math.ceil(head_dim * num_bits / 8)

    packed = mx.zeros(shape=(n_vectors, packed_width), dtype=mx.uint8)
    scales = mx.zeros(shape=(n_vectors, n_vq_blocks), dtype=mx.float16)
    total = int(packed.nbytes) + int(scales.nbytes)
    payload = BlockTQPayload(resident_bytes=total, packed_indices=packed, scales=scales)
    assert payload.resident_bytes == total
    assert payload.packed_indices is packed
    assert payload.scales is scales


def test_block_tq_payload_rejects_mismatched_resident_bytes() -> None:
    n_vectors = 64
    packed = mx.zeros(shape=(n_vectors, 32), dtype=mx.uint8)
    scales = mx.zeros(shape=(n_vectors, 2), dtype=mx.float16)
    honest = int(packed.nbytes) + int(scales.nbytes)
    with pytest.raises(ValueError, match="resident_bytes"):
        BlockTQPayload(
            resident_bytes=honest // 2,
            packed_indices=packed,
            scales=scales,
        )


def test_rabitq_payload_accepts_honest_resident_bytes() -> None:
    head_dim = 64
    num_bits = 1
    n_vectors = 4 * 16
    packed_width = math.ceil(head_dim * num_bits / 8)

    packed = mx.zeros(shape=(n_vectors, packed_width), dtype=mx.uint8)
    norm_o = mx.zeros(shape=(n_vectors,), dtype=mx.float16)
    ip_coeff = mx.zeros(shape=(n_vectors,), dtype=mx.float16)
    total = int(packed.nbytes) + int(norm_o.nbytes) + int(ip_coeff.nbytes)
    payload = RaBitQPayload(
        resident_bytes=total,
        packed_indices=packed,
        norm_o=norm_o,
        ip_coeff=ip_coeff,
    )
    assert payload.resident_bytes == total
    assert payload.packed_indices is packed
    assert payload.norm_o is norm_o
    assert payload.ip_coeff is ip_coeff


def test_rabitq_payload_rejects_mismatched_resident_bytes() -> None:
    packed = mx.zeros(shape=(64, 8), dtype=mx.uint8)
    norm_o = mx.zeros(shape=(64,), dtype=mx.float16)
    ip_coeff = mx.zeros(shape=(64,), dtype=mx.float16)
    honest = int(packed.nbytes) + int(norm_o.nbytes) + int(ip_coeff.nbytes)
    with pytest.raises(ValueError, match="resident_bytes"):
        RaBitQPayload(
            resident_bytes=honest + 7,
            packed_indices=packed,
            norm_o=norm_o,
            ip_coeff=ip_coeff,
        )


def test_rabitq_payload_multi_bit_width() -> None:
    """ExtRaBitQ uses the same payload class with a wider packed_indices
    field (``ceil(d × B / 8)``); verify the honesty check scales to the
    multi-bit case."""
    head_dim = 256
    num_bits = 4
    n_vectors = 64
    packed_width = math.ceil(head_dim * num_bits / 8)
    assert packed_width == 128

    packed = mx.zeros(shape=(n_vectors, packed_width), dtype=mx.uint8)
    norm_o = mx.zeros(shape=(n_vectors,), dtype=mx.float16)
    ip_coeff = mx.zeros(shape=(n_vectors,), dtype=mx.float16)
    total = int(packed.nbytes) + int(norm_o.nbytes) + int(ip_coeff.nbytes)
    RaBitQPayload(
        resident_bytes=total,
        packed_indices=packed,
        norm_o=norm_o,
        ip_coeff=ip_coeff,
    )


def test_verify_ignores_non_array_fields() -> None:
    """The verification sums only mx.array-valued fields; the int resident_bytes
    field itself must not be double-counted. This test anchors that invariant
    by checking that a payload whose resident_bytes matches the array .nbytes
    sum passes, independent of the base-class integer field."""
    t = mx.zeros(shape=(2, 2, 2, 2), dtype=mx.float16)
    # If the implementation accidentally counted the ``resident_bytes`` field
    # itself, the sum would be off by 28 (size of a Python int) and this
    # construction would raise. The test passes iff only mx.array fields count.
    payload = RawFp16Payload(resident_bytes=int(t.nbytes), t=t)
    assert payload.resident_bytes == t.nbytes


# -----------------------------------------------------------------------------
# VectorCodec Protocol shape
# -----------------------------------------------------------------------------


class _MinimalCodec:
    """Minimal class implementing the VectorCodec method set for shape tests."""

    block_size = 16
    dtype = mx.float16

    def encode_tensor(self, x: mx.array) -> RawFp16Payload:
        return RawFp16Payload(resident_bytes=int(x.nbytes), t=x)

    def decode_tensor(self, payload: RawFp16Payload) -> mx.array:
        return payload.t

    def logical_bytes(self, num_tokens: int) -> int:
        return num_tokens

    def resident_bytes(self, num_blocks: int) -> int:
        return num_blocks * self.block_size


class _IncompleteCodec:
    """Missing ``decode_tensor`` — should fail ``isinstance(VectorCodec)``."""

    block_size = 16
    dtype = mx.float16

    def encode_tensor(self, x: mx.array) -> RawFp16Payload:
        return RawFp16Payload(resident_bytes=int(x.nbytes), t=x)

    def logical_bytes(self, num_tokens: int) -> int:
        return num_tokens

    def resident_bytes(self, num_blocks: int) -> int:
        return num_blocks * self.block_size


def test_vector_codec_protocol_accepts_full_implementation() -> None:
    codec = _MinimalCodec()
    assert isinstance(codec, VectorCodec)


def test_vector_codec_protocol_rejects_incomplete_implementation() -> None:
    codec = _IncompleteCodec()
    assert not isinstance(codec, VectorCodec)


def test_vector_codec_protocol_is_generic_in_payload() -> None:
    """Parameterization is structural — isinstance does not check the payload
    type at runtime, but the Protocol should accept a generic bracket form
    without error (mirroring PEP 544 + PEP 695 shape)."""
    # Should not raise at class-body / module-import time.
    _: type = VectorCodec[RawFp16Payload]  # type: ignore[misc,type-arg]


# -----------------------------------------------------------------------------
# CodedPayload base — cannot be instantiated meaningfully but should not crash
# when constructed with resident_bytes only (no array fields to sum).
# -----------------------------------------------------------------------------


def test_coded_payload_base_with_no_array_fields_accepts_zero() -> None:
    """Base ``CodedPayload`` has no mx.array fields; honesty check requires
    ``resident_bytes == 0``. Concrete codecs always subclass; this test
    locks the base-class semantics in case anyone instantiates it directly."""
    payload = CodedPayload(resident_bytes=0)
    payload._verify_resident_bytes()  # no-op; asserts clean pass


def test_coded_payload_base_rejects_nonzero_when_no_array_fields() -> None:
    payload = CodedPayload(resident_bytes=100)
    with pytest.raises(ValueError, match="resident_bytes"):
        payload._verify_resident_bytes()
