"""silica.models — per-family I-1 adapters + dispatch factory.

One adapter class per model generation / family; each lives in its own
submodule so adding Qwen4 / DeepSeek / Kimi / GLM / MiniMax does not
regress existing families. See individual modules for per-family quirks
and ``silica.models.factory`` for repo-string → adapter dispatch.
"""

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
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)
from silica.models.factory import adapter_for_repo, supported_model_types
from silica.models.qwen3 import Qwen3Adapter
from silica.models.qwen3_5 import Qwen3_5Adapter

__all__ = [
    "AttentionKind",
    "AttentionPattern",
    "KVLayout",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelConfig",
    "Module",
    "Qwen3Adapter",
    "Qwen3_5Adapter",
    "StateDelta",
    "StubModelAdapter",
    "Tokenizer",
    "adapter_for_repo",
    "capabilities_from_attention_pattern",
    "supported_model_types",
]
