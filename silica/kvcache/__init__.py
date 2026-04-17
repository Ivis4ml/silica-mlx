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
from silica.kvcache.simple import SimpleKVCache

__all__ = [
    "BlockList",
    "CodedBlock",
    "IdentityCodec",
    "KVCodec",
    "KVHandle",
    "KVManager",
    "MemoryBudget",
    "NullKVManager",
    "PrefixHit",
    "SimpleKVCache",
]
