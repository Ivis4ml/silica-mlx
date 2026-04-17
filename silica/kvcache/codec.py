"""silica.kvcache.codec — I-3 KVCodec Protocol and the IdentityCodec stub.

I-3 (PLAN.md §6) specifies a per-layer, per-block encode/decode contract plus
two byte-count reporters the scheduler uses for admission / budgeting
(Principle 8).

v0.1 intentionally excludes a compressed-domain attention fast path (D-003);
`decode_block` returns fp16-dtype K / V that downstream attention consumes as
if nothing was encoded.

D-012 canonical measurement of `resident_bytes`:
  - physical bytes currently owned by the component in unified memory
  - excludes transient scratch, allocator headroom, reclaimable regions
  - value must be idempotent across repeated calls outside a modifying op
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx


@dataclass
class CodedBlock:
    """Codec-produced encoded form of one block's K/V plus D-012 accounting.

    v0.1 (IdentityCodec): `k` / `v` hold the raw unencoded tensors. Future
    codecs (BlockTQ, RaBitQ — P-5) place the quantized payload here and will
    either extend this dataclass with per-codec auxiliary tensors or have
    `KVCodec` evolve to a Generic Protocol. That refactor is scoped to P-5.
    `resident_bytes` is the physical footprint of this one block per D-012.
    """

    k: mx.array
    v: mx.array
    resident_bytes: int


@runtime_checkable
class KVCodec(Protocol):
    """Per-layer encode / decode + byte-count reporters.

    See I-3 in PLAN.md. The codec is unaware of batch, scheduler, or model
    structure — it only sees one layer's K/V blocks.

    Note: `@runtime_checkable` verifies method presence only, not class
    attribute presence; the P-0 interface test asserts attributes separately.
    """

    block_size: int
    k_dtype: mx.Dtype
    v_dtype: mx.Dtype

    def encode_block(self, k: mx.array, v: mx.array) -> CodedBlock: ...

    def decode_block(self, block: CodedBlock) -> tuple[mx.array, mx.array]: ...

    def logical_bytes(self, num_tokens: int) -> int:
        """fp16-baseline-equivalent bytes for `num_tokens` tokens of this layer's K+V."""
        ...

    def resident_bytes(self, num_blocks: int) -> int:
        """Physical bytes held for `num_blocks` blocks, per D-012 definition."""
        ...


class IdentityCodec:
    """Pass-through codec: `encoded == raw`, `resident == logical`.

    Fulfills the I-3 contract for P-0, provides the fp16 baseline against
    which P-5 real codecs (BlockTQ / RaBitQ) are compared for compression
    ratio and quality delta. Principle 9 stub-replacement pattern: this class
    stays in tree after P-5 as the baseline option on the `--kv-codec` switch.
    """

    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        k_dtype: mx.Dtype = mx.float16,
        v_dtype: mx.Dtype = mx.float16,
    ) -> None:
        self.block_size = block_size
        self.k_dtype = k_dtype
        self.v_dtype = v_dtype
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim

    def _bytes_per_token(self) -> int:
        per_k = self._n_kv_heads * self._head_dim * self.k_dtype.size
        per_v = self._n_kv_heads * self._head_dim * self.v_dtype.size
        return per_k + per_v

    def encode_block(self, k: mx.array, v: mx.array) -> CodedBlock:
        return CodedBlock(k=k, v=v, resident_bytes=k.nbytes + v.nbytes)

    def decode_block(self, block: CodedBlock) -> tuple[mx.array, mx.array]:
        return block.k, block.v

    def logical_bytes(self, num_tokens: int) -> int:
        return num_tokens * self._bytes_per_token()

    def resident_bytes(self, num_blocks: int) -> int:
        return num_blocks * self.block_size * self._bytes_per_token()
