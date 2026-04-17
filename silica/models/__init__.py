"""silica.models — model adapters (I-1, D-015 hybrid_deltanet, D-011 MoE)."""

from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelAdapter,
    ModelConfig,
    Module,
    StateDelta,
    StubModelAdapter,
    Tokenizer,
)
from silica.models.qwen3 import Qwen3Adapter

__all__ = [
    "AttentionKind",
    "AttentionPattern",
    "KVLayout",
    "ModelAdapter",
    "ModelConfig",
    "Module",
    "Qwen3Adapter",
    "StateDelta",
    "StubModelAdapter",
    "Tokenizer",
]
