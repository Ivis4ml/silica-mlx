"""silica.bench.codec_registry — P-5 codec catalog (opening doc §6.1).

Registry of named ``VectorCodec`` configurations that the bench harness
(and future ``--kv-codec`` CLI flag) consumes by id. Each entry binds
a codec family + its algorithmic knobs (``num_bits``, ``vq_block_size``)
and exposes a factory that produces a ready-to-use codec given the
per-side shape params (``block_size``, ``n_kv_heads``, ``head_dim``,
``dtype``).

Registered entries (P-5-A.1b + P-5-B.1b scope):

- ``fp16``: IdentityCodec baseline. 16 bits/value; no compression.
- ``tq_mse_b3`` / ``tq_mse_b4``: Scalar TurboQuantMSE (aliased to
  ``BlockTurboQuantMSE(vq_block_size=head_dim)``). 3 / 4 nominal bits
  per value; actual is ``num_bits + 16 / head_dim``.
- ``block_tq_b32_b{3,4}`` / ``block_tq_b64_b{3,4}``: BlockTurboQuantMSE
  with ``vq_block_size`` ∈ {32, 64} and ``num_bits`` ∈ {3, 4}. Exact
  effective bits/value = ``num_bits + 16 / vq_block_size``.
- ``rabitq_b1``: RaBitQ1Bit (P-5-B.1b). 1-bit hypercube codec with
  per-vector fp16 ``norm_o`` + fp16 ``ip_coeff`` metadata. K-only
  (``v_supported=False``) — the estimator-native attention path the
  ``ip_coeff`` field feeds lives on K by construction; symmetric K+V
  installation via the ``kv_codec=`` shorthand is explicitly rejected
  at ``_maybe_build_prefix_cache`` time so a mis-wired scenario fails
  fast rather than silently quantizing V with a K-only codec.

Only ``block_tq_b64_b4`` is ``production_recommended`` — vqbench REPORT
§3.1 shows it is strictly lossless at ``std = 0.000%`` across three
seeds on Qwen3.5-4B WikiText-2, delivering 3.76× total-KV compression.
All other entries are bench-row / comparison configs.

P-5-B.2 adds ExtRaBitQ entries (``ext_rabitq_b{2,3,4}``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from silica.kvcache.codec import IdentityCodec, VectorCodec
from silica.vq import BlockTurboQuantMSE, RaBitQ1Bit, TurboQuantMSE

# Every factory takes the per-side shape via kwargs and returns a
# ready-to-use VectorCodec. The registry binds codec-specific knobs
# (num_bits, vq_block_size, seed, norm_correction) so the caller only
# supplies ``(block_size, n_kv_heads, head_dim, dtype)``.
CodecFactory = Callable[..., VectorCodec]


@dataclass(frozen=True)
class CodecSpec:
    """Registry entry for one named codec configuration.

    Attributes:
        id: unique string identifier (stable across versions; used by
            the ``--kv-codec`` CLI flag and bench JSONL rows).
        family: ``"fp16"`` | ``"tq_mse"`` | ``"block_tq"`` | ``"rabitq"``
            (P-5-B.2 will add ``"ext_rabitq"``).
        bits_per_value: nominal bits per coordinate. For BlockTQ this
            is exact (``num_bits + 16 / vq_block_size``, head-dim-
            independent). For scalar TQ it is ``float(num_bits)``; the
            true value is higher by ``16 / head_dim`` from the per-
            vector fp16 scale — use :meth:`effective_bits_per_value`
            when head_dim is known. For RaBitQ it is ``float(num_bits)``
            (nominal); the true value is higher by ``32 / head_dim``
            from two per-vector fp16 scalars (``norm_o`` + ``ip_coeff``).
        k_supported: whether this codec can serve as the K-side codec
            in ``SyntheticPrefixBlockStore(k_codec=..., v_codec=...)``.
        v_supported: whether this codec can serve as the V-side codec.
            ``rabitq_b1`` is K-only (``False``) because the estimator-
            native attention path the ``ip_coeff`` field feeds lives on
            K by construction. ``_maybe_build_prefix_cache`` rejects
            the symmetric ``kv_codec=`` shorthand for any spec where
            ``k_supported and v_supported`` is False so a K-only codec
            cannot silently land on V.
        requires_fit: whether the codec needs offline fitting on real
            activations. All P-5-A.1 and RaBitQ entries are ``False``
            (Haar + Lloyd-Max are analytical / Gaussian-based; RaBitQ
            centroid is pinned at zero per opening §5.3).
        payload_packed: whether the payload uses sub-byte packing
            (BlockTQPayload / RaBitQPayload pack indices via
            :mod:`silica.vq.core.packing`; RawFp16Payload does not).
        production_recommended: whether the opening doc (vqbench REPORT
            §3.1) pins this as a production config. Only
            ``block_tq_b64_b4`` qualifies in P-5-A.1 / P-5-B.
        factory: callable producing a ``VectorCodec`` instance given
            per-side shape params. Signature:
            ``factory(*, block_size, n_kv_heads, head_dim, dtype=fp16)
            -> VectorCodec``.
    """

    id: str
    family: str
    bits_per_value: float
    k_supported: bool
    v_supported: bool
    requires_fit: bool
    payload_packed: bool
    production_recommended: bool
    factory: CodecFactory

    def effective_bits_per_value(self, head_dim: int) -> float:
        """True effective bits per coordinate at the given ``head_dim``.

        For BlockTQ this equals the stored ``bits_per_value``
        (head-dim-independent). For scalar TQ this is
        ``num_bits + 16 / head_dim`` (the per-vector fp16 scale
        amortizes to ``16 / head_dim`` bits per coordinate). For RaBitQ
        this is ``num_bits + 32 / head_dim`` (two per-vector fp16
        scalars: ``norm_o`` and ``ip_coeff``, amortizing to
        ``32 / head_dim`` bits per coordinate). For fp16 baseline this
        is 16.

        Raises:
            ValueError: if ``head_dim`` is not a positive integer, or
                if ``family`` is not one of the known values. The
                positive-head_dim guard matters for ``tq_mse`` and
                ``rabitq`` families where ``head_dim`` appears in the
                denominator; ``head_dim = 0`` would otherwise divide-
                by-zero, and a negative head_dim would produce a
                meaningless negative bits/value result.
        """
        if head_dim <= 0:
            raise ValueError(
                f"head_dim must be a positive integer; got {head_dim}"
            )
        if self.family == "fp16":
            return 16.0
        if self.family == "tq_mse":
            return float(self.bits_per_value) + 16.0 / float(head_dim)
        if self.family == "block_tq":
            return float(self.bits_per_value)
        if self.family == "rabitq":
            return float(self.bits_per_value) + 32.0 / float(head_dim)
        raise ValueError(f"unknown codec family: {self.family!r}")


# =============================================================================
# Factory builders
# =============================================================================


def _identity_factory(
    *,
    block_size: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: mx.Dtype = mx.float16,
) -> VectorCodec:
    return IdentityCodec(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )


def _make_scalar_tq_factory(num_bits: int) -> CodecFactory:
    """Return a factory that builds scalar ``TurboQuantMSE`` with the
    given ``num_bits`` (``vq_block_size = head_dim`` special case)."""

    def factory(
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> VectorCodec:
        return TurboQuantMSE(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_bits=num_bits,
            dtype=dtype,
        )

    return factory


def _make_block_tq_factory(vq_block_size: int, num_bits: int) -> CodecFactory:
    """Return a factory that builds ``BlockTurboQuantMSE`` with the
    given ``(vq_block_size, num_bits)``."""

    def factory(
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> VectorCodec:
        return BlockTurboQuantMSE(
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            vq_block_size=vq_block_size,
            num_bits=num_bits,
            dtype=dtype,
        )

    return factory


def _rabitq_b1_factory(
    *,
    block_size: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: mx.Dtype = mx.float16,
) -> VectorCodec:
    """Produce a ``RaBitQ1Bit`` codec. Only one 1-bit variant exists in
    the catalog, so no maker closure is needed — a plain module-level
    function mirrors ``_identity_factory`` rather than the
    ``_make_*_factory`` pattern used for multi-variant families."""
    return RaBitQ1Bit(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_bits=1,
        dtype=dtype,
    )


# =============================================================================
# Registry
# =============================================================================


CODEC_REGISTRY: dict[str, CodecSpec] = {
    "fp16": CodecSpec(
        id="fp16",
        family="fp16",
        bits_per_value=16.0,
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=False,
        production_recommended=False,  # baseline, not a compression row
        factory=_identity_factory,
    ),
    "tq_mse_b3": CodecSpec(
        id="tq_mse_b3",
        family="tq_mse",
        bits_per_value=3.0,
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=True,
        # vqbench REPORT §3.2: 3-bit K+V is seed-dependent; Block B=32
        # b=3 has std >= mean. Scalar TQ b=3 is similar. Bench row only.
        production_recommended=False,
        factory=_make_scalar_tq_factory(num_bits=3),
    ),
    "tq_mse_b4": CodecSpec(
        id="tq_mse_b4",
        family="tq_mse",
        bits_per_value=4.0,
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=True,
        # vqbench REPORT §3.1: TQ-MSE 4-bit K+V is seed-dependent
        # (mean ΔPPL +0.262%, std 0.370%). Bench row, not production.
        production_recommended=False,
        factory=_make_scalar_tq_factory(num_bits=4),
    ),
    "block_tq_b32_b3": CodecSpec(
        id="block_tq_b32_b3",
        family="block_tq",
        bits_per_value=3.5,  # 3 + 16/32
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=True,
        # vqbench REPORT §3.2: best 3-bit entry (ΔPPL +1.324%), but
        # std (1.488%) > mean — still noise floor. Bench row.
        production_recommended=False,
        factory=_make_block_tq_factory(vq_block_size=32, num_bits=3),
    ),
    "block_tq_b32_b4": CodecSpec(
        id="block_tq_b32_b4",
        family="block_tq",
        bits_per_value=4.5,  # 4 + 16/32
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=True,
        production_recommended=False,
        factory=_make_block_tq_factory(vq_block_size=32, num_bits=4),
    ),
    "block_tq_b64_b3": CodecSpec(
        id="block_tq_b64_b3",
        family="block_tq",
        bits_per_value=3.25,  # 3 + 16/64
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=True,
        # 3-bit noise floor applies here too (REPORT §3.2).
        production_recommended=False,
        factory=_make_block_tq_factory(vq_block_size=64, num_bits=3),
    ),
    "block_tq_b64_b4": CodecSpec(
        id="block_tq_b64_b4",
        family="block_tq",
        bits_per_value=4.25,  # 4 + 16/64
        k_supported=True,
        v_supported=True,
        requires_fit=False,
        payload_packed=True,
        # vqbench REPORT §3.1: strictly lossless (std=0.000%) across
        # three seeds on Qwen3.5-4B WikiText-2, 3.76× total-KV
        # compression. The single production recommendation in P-5-A.1.
        production_recommended=True,
        factory=_make_block_tq_factory(vq_block_size=64, num_bits=4),
    ),
    "rabitq_b1": CodecSpec(
        id="rabitq_b1",
        family="rabitq",
        # Nominal 1 bit per coordinate. effective_bits_per_value(head_dim)
        # adds 32/head_dim for the fp16 norm_o + fp16 ip_coeff pair; for
        # head_dim=64 that is 1.5 bits/coord, for head_dim=128 1.25.
        bits_per_value=1.0,
        k_supported=True,
        # K-only per opening §5.3. The estimator-native attention path
        # the ``ip_coeff`` field feeds lives on K by construction;
        # installing RaBitQ1Bit on V would waste a per-vector fp16
        # metadata slot that never contributes to attention output.
        # ``_maybe_build_prefix_cache`` rejects the symmetric
        # kv_codec= shorthand for asymmetric specs so scenarios cannot
        # silently mis-configure.
        v_supported=False,
        # P-5-B pins centroid at zero (opening §5.3); no real-activation
        # fit is needed. The B.2 ExtRaBitQ entries inherit this.
        requires_fit=False,
        payload_packed=True,
        # Bench-row only; the hypercube 1-bit reconstruction MSE is
        # worse than BlockTQ at matching bit budget. Ships for
        # comparison tables, not for production.
        production_recommended=False,
        factory=_rabitq_b1_factory,
    ),
}


# =============================================================================
# Lookup helpers
# =============================================================================


def get_codec_spec(codec_id: str) -> CodecSpec:
    """Look up a codec spec by id.

    Raises:
        KeyError: if ``codec_id`` is not registered. The error message
            lists all available ids so callers can see valid choices.
    """
    try:
        return CODEC_REGISTRY[codec_id]
    except KeyError:
        available = ", ".join(sorted(CODEC_REGISTRY.keys()))
        raise KeyError(
            f"unknown codec id {codec_id!r}; available: {available}"
        ) from None


def list_codec_ids() -> list[str]:
    """Return the sorted list of registered codec ids. Order is stable
    across runs (alphabetical); used by ``--all-kv-codecs`` to iterate
    the catalog deterministically."""
    return sorted(CODEC_REGISTRY.keys())
