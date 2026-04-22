"""silica.kvcache — KV cache management (I-2 KVManager + I-3 KVCodec).

Codec exports carry two generations during the P-5-A.0 transition:

- Pre-P-5 pair-level: ``CodedBlock`` / ``KVCodec`` / ``IdentityCodec`` —
  still the types ``SyntheticPrefixBlockStore`` consumes today.
- P-5 side-level: ``VectorCodec[P]`` Protocol + ``CodedPayload`` hierarchy
  (``RawFp16Payload`` / ``BlockTQPayload`` / ``RaBitQPayload``) — the target
  shape; reached by the store-migration commit that retires the pair-level
  names and rewires the store's dispatch to run K and V codecs independently.
"""

from silica.kvcache.codec import (
    BlockTQPayload,
    CodedBlock,
    CodedPayload,
    IdentityCodec,
    KVCodec,
    RaBitQPayload,
    RawFp16Payload,
    VectorCodec,
)
from silica.kvcache.manager import (
    BlockList,
    KVHandle,
    KVManager,
    MemoryBudget,
    NullKVManager,
    PrefixHit,
)
from silica.kvcache.paged import PagedKVCache, RowState
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.simple import SimpleKVCache
from silica.kvcache.store import (
    PagedPrefixBlockStore,
    PrefixBlockStore,
    SyntheticPrefixBlockStore,
)

__all__ = [
    "BlockList",
    "BlockTQPayload",
    "CodedBlock",
    "CodedPayload",
    "IdentityCodec",
    "KVCodec",
    "KVHandle",
    "KVManager",
    "MemoryBudget",
    "NullKVManager",
    "PagedKVCache",
    "PagedPrefixBlockStore",
    "PrefixBlockStore",
    "PrefixHit",
    "RaBitQPayload",
    "RadixPrefixCache",
    "RawFp16Payload",
    "RowState",
    "SimpleKVCache",
    "SyntheticPrefixBlockStore",
    "VectorCodec",
]
