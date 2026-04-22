"""silica.vq.turboquant — Scalar TurboQuantMSE (alias for ``BlockTQ(B=d)``).

Per ``docs/P5_OPENING.md`` §4.5 + §5.1, the scalar TurboQuantMSE
algorithm from the TurboQuant paper (arXiv 2504.19874, Algorithm 1) is
the ``vq_block_size = head_dim`` special case of
``BlockTurboQuantMSE`` — one per-vector fp16 scale, block-local
codebook reduces to the scalar ``N(0, 1/head_dim)`` Lloyd-Max. Shipping
them as a single code path saves one implementation and keeps the
``B = head_dim`` invariant (regression-locked by
``tests/test_block_tq.py::test_block_equals_scalar_when_B_equals_d``)
a one-liner rather than a separate class body.

Callers that want the paper's scalar formulation use:

    codec = TurboQuantMSE(block_size=16, n_kv_heads=4, head_dim=128,
                          num_bits=4)

which is identical to:

    codec = BlockTurboQuantMSE(block_size=16, n_kv_heads=4,
                               head_dim=128, vq_block_size=128,
                               num_bits=4)
"""

from __future__ import annotations

import mlx.core as mx

from silica.vq.block_tq import BlockTurboQuantMSE


def TurboQuantMSE(
    *,
    block_size: int,
    n_kv_heads: int,
    head_dim: int,
    num_bits: int,
    seed: int = 42,
    norm_correction: bool = True,
    dtype: mx.Dtype = mx.float16,
) -> BlockTurboQuantMSE:
    """Scalar TurboQuantMSE — ``BlockTurboQuantMSE(vq_block_size=head_dim)``.

    Returns a ``BlockTurboQuantMSE`` instance; there is no separate
    ``TurboQuantMSE`` class body. Callers that type-check against
    ``BlockTurboQuantMSE`` (or ``VectorCodec[BlockTQPayload]``) treat
    the scalar path identically to the block path.
    """
    return BlockTurboQuantMSE(
        block_size=block_size,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        vq_block_size=head_dim,
        num_bits=num_bits,
        seed=seed,
        norm_correction=norm_correction,
        dtype=dtype,
    )
