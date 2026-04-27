"""silica.kvcache.codec — side-level VectorCodec[P] Protocol + IdentityCodec baseline.

I-3 interface definition (see ``plans/PLAN.md`` §6, updated at P-5-A.0.4).

A ``VectorCodec`` operates on one tensor at a time — either K or V from
one block of one layer; pair-level dispatch lives in
``SyntheticPrefixBlockStore``, which holds ``k_codec`` / ``v_codec``
separately and encodes / decodes independently per side. This is the
side-level resolution of Q-008 — the historical A / B / C options in
§10 are superseded by the side-level Protocol plus store-level
``k_codec`` / ``v_codec`` split shipped in P-5-A.0.

Payload honesty is enforced at construction: every concrete
``CodedPayload`` subclass calls ``_verify_resident_bytes`` from
``__post_init__`` to require ``resident_bytes`` equal the sum of
``.nbytes`` across its ``mx.array`` fields (D-012 canonical).

D-003: ``decode_tensor`` returns fp16; no compressed-domain attention.
D-009: codec runtime bodies are MLX-native (no NumPy / torch);
calibration helpers live under ``silica.vq._calibration``.
D-012: ``resident_bytes`` is the physical footprint owned by this
component in unified memory; excludes transient scratch / allocator
headroom; idempotent across repeated calls outside a modifying
operation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Generic, Protocol, TypeVar, runtime_checkable

import mlx.core as mx

# =============================================================================
# Payload hierarchy
# =============================================================================


@dataclass
class CodedPayload:
    """Base payload for one side (K or V) of one block under a side-level codec.

    Concrete subclasses add per-codec ``mx.array`` fields (packed indices,
    scales, signs, etc.). They must call ``_verify_resident_bytes`` from
    ``__post_init__`` so construction fails fast if the declared
    ``resident_bytes`` disagrees with actual unified-memory residency.
    """

    resident_bytes: int

    def _verify_resident_bytes(self) -> None:
        """D-012 honesty check.

        Raises ``ValueError`` if ``resident_bytes`` does not equal the sum of
        ``.nbytes`` across every ``mx.array``-valued field on this instance.
        """
        total = 0
        for f in fields(self):
            if f.name == "resident_bytes":
                continue
            val = getattr(self, f.name)
            if isinstance(val, mx.array):
                total += int(val.nbytes)
        if self.resident_bytes != total:
            raise ValueError(
                f"{type(self).__name__}.resident_bytes={self.resident_bytes} "
                f"must equal sum of mx.array field .nbytes ({total}) per D-012"
            )


@dataclass
class RawFp16Payload(CodedPayload):
    """Identity-codec payload — wraps the tensor unchanged.

    Shape of ``t`` matches whatever the caller passed to
    ``IdentityCodec.encode_tensor`` (typically
    ``(1, n_kv_heads, kv_block_size, head_dim)`` for one side of one
    detached block).
    """

    t: mx.array

    def __post_init__(self) -> None:
        self._verify_resident_bytes()


@dataclass
class BlockTQPayload(CodedPayload):
    """BlockTurboQuantMSE payload.

    Shapes for one side of one block (``n_vectors = n_kv_heads × kv_block_size``,
    per-vector dimension ``head_dim = d``, per-vq-block size ``B``, codebook
    width ``b`` bits):

    - ``packed_indices``: ``uint8``, ``(n_vectors, ceil(d × b / 8))`` —
      sub-byte-packed Lloyd-Max codebook indices.
    - ``scales``: ``fp16``, ``(n_vectors, d // B)`` — per-vq-block norms
      (one scale per contiguous length-``B`` span of the head-dim axis).
    """

    packed_indices: mx.array
    scales: mx.array

    def __post_init__(self) -> None:
        self._verify_resident_bytes()


@dataclass
class RaBitQPayload(CodedPayload):
    """RaBitQ 1-bit payload (P-5-B core shape).

    Shapes for one side of one block (``n_vectors = n_kv_heads × kv_block_size``,
    per-vector dimension ``head_dim = d``):

    - ``packed_indices``: ``uint8``, ``(n_vectors, d / 8)`` — sign bits
      packed via :func:`silica.vq.core.packing.pack_sub_byte` at
      ``num_bits=1``.
    - ``norm_o``: ``fp16``, ``(n_vectors,)`` — per-vector
      ``||x - centroid||_2``.
    - ``ip_coeff``: ``fp16``, ``(n_vectors,)`` — per-vector inner-product
      coefficient ``(x_bar · y)`` from the unbiased estimator.

    The centroid itself lives on the codec instance (pinned at zero in
    P-5-B per opening §5.3), not on the payload.

    ExtRaBitQ's multi-bit payload ships as :class:`ExtRaBitQPayload`, a
    subclass that adds a per-vector fp16 ``scale`` field; see §5.4.
    """

    packed_indices: mx.array
    norm_o: mx.array
    ip_coeff: mx.array

    def __post_init__(self) -> None:
        self._verify_resident_bytes()


@dataclass
class ExtRaBitQPayload(RaBitQPayload):
    """ExtRaBitQ multi-bit payload (P-5-B.2).

    Extends :class:`RaBitQPayload` with a per-vector fp16 ``scale``
    field. Shapes for one side of one block (``n_vectors = n_kv_heads ×
    kv_block_size``, per-vector dimension ``head_dim = d``, bits per
    coordinate ``num_bits = B ∈ {2, 3, 4}``):

    - ``packed_indices``: ``uint8``, ``(n_vectors, ceil(d × B / 8))`` —
      odd-integer codebook indices packed at ``num_bits=B`` via
      :func:`silica.vq.core.packing.pack_sub_byte`. Widths: B=2 → d/4,
      B=3 → 3d/8, B=4 → d/2.
    - ``norm_o``, ``ip_coeff``: inherited from ``RaBitQPayload``, both
      ``fp16 (n_vectors,)``.
    - ``scale``: ``fp16``, ``(n_vectors,)`` — per-vector
      **dequantization** scale (inverse of the quantization factor
      ``(2^B - 1) / (3 · σ)`` with ``σ = 1/sqrt(d)``). Decode multiplies
      integer codebook values by this scalar to recover the rotated-
      coordinate range. In v0.1 the value is constant across vectors at
      a given ``(num_bits, head_dim)`` and stored per-vector only for
      schema parity with future data-driven variants.

    The vqbench reference carries an ``offset`` field too. silica drops
    it in v0.1: offset is ``0.0`` in every vqbench batch / single-
    vector code path, and decode does not read it. See opening §5.4
    Amendment (P-5-B.2a). Adding it later is a deliberate schema
    extension, not a silent upgrade.

    ``_verify_resident_bytes`` inspects all :class:`mx.array` fields
    across the MRO (``packed_indices``, ``norm_o``, ``ip_coeff``,
    ``scale``), so the D-012 honesty check covers the extended schema
    without additional code — the explicit ``__post_init__`` here
    restates the call so the behaviour does not rely on dataclass
    inheritance subtleties.
    """

    scale: mx.array

    def __post_init__(self) -> None:
        self._verify_resident_bytes()


# =============================================================================
# VectorCodec[P] Protocol
# =============================================================================


P = TypeVar("P", bound=CodedPayload)


@runtime_checkable
class VectorCodec(Protocol, Generic[P]):
    """Side-level per-tensor encode / decode.

    A ``VectorCodec`` operates on one tensor (either K or V from one block
    of one layer). Pair-level dispatch happens in
    ``SyntheticPrefixBlockStore``: K and V codecs are held separately and
    encoded / decoded independently per detached block.

    ``@runtime_checkable`` verifies method presence only; attribute presence
    (``block_size``, ``dtype``) is asserted separately in contract tests.
    """

    block_size: int
    dtype: mx.Dtype

    def encode_tensor(self, x: mx.array) -> P:
        """Encode one fp16 tensor into a codec-specific payload."""
        ...

    def decode_tensor(self, payload: P) -> mx.array:
        """Reconstruct an fp16 tensor from a payload (D-003: output is fp16)."""
        ...

    def logical_bytes(self, num_tokens: int) -> int:
        """fp16-baseline-equivalent bytes for ``num_tokens`` tokens of this side."""
        ...

    def resident_bytes(self, num_blocks: int) -> int:
        """Physical bytes held for ``num_blocks`` blocks of this side (D-012)."""
        ...


# =============================================================================
# IdentityCodec — side-level baseline
# =============================================================================


class IdentityCodec:
    """Pass-through ``VectorCodec[RawFp16Payload]`` baseline.

    ``encoded == raw`` (no copy — ``RawFp16Payload.t is original_tensor``
    and ``decode_tensor(payload) is payload.t``). ``resident == logical``
    (fp16-equivalent). Fulfills the I-3 contract for P-0 stubs and for
    the bench-harness baseline column in P-5-C.

    Side-level: one instance represents **one side** (K or V) of the
    per-layer K/V pair. ``logical_bytes`` / ``resident_bytes`` report
    one-side bytes; the store sums across sides when reporting total
    resident prefix-cache residency. Callers that want symmetric K+V use
    the ``codec=`` shorthand on ``SyntheticPrefixBlockStore``
    (``k_codec = v_codec = IdentityCodec(...)``); asymmetric
    configurations pass ``k_codec`` and ``v_codec`` explicitly.

    Principle 9 stub-replacement pattern: this class stays in tree after
    P-5 as the baseline option on the ``--kv-codec`` switch.
    """

    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None:
        self.block_size = block_size
        self.dtype = dtype
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim

    def _bytes_per_token(self) -> int:
        """fp16-equivalent bytes per token for this one side."""
        return self._n_kv_heads * self._head_dim * self.dtype.size

    def encode_tensor(self, x: mx.array) -> RawFp16Payload:
        return RawFp16Payload(resident_bytes=int(x.nbytes), t=x)

    def decode_tensor(self, payload: RawFp16Payload) -> mx.array:
        return payload.t

    def logical_bytes(self, num_tokens: int) -> int:
        return num_tokens * self._bytes_per_token()

    def resident_bytes(self, num_blocks: int) -> int:
        return num_blocks * self.block_size * self._bytes_per_token()
