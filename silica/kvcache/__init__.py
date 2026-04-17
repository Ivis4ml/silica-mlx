"""silica.kvcache — KV cache management (I-2 KVManager + I-3 KVCodec)."""

from silica.kvcache.codec import CodedBlock, IdentityCodec, KVCodec
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
    "CodedBlock",
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
    "RadixPrefixCache",
    "RowState",
    "SimpleKVCache",
    "SyntheticPrefixBlockStore",
]
