"""silica.models.adapter — I-1 ModelAdapter Protocol + auxiliary types + StubModelAdapter.

I-1 (PLAN.md §6) owns model structure, tokenizer, layer execution, attention
pattern, and execution semantics (prefill / decode). D-015 extends the contract
with hybrid recurrent layers (DeltaNet) — the Python signatures are unchanged,
but `AttentionPattern` gains `recurrent` / `hybrid_deltanet` enum values and
`StateDelta` becomes a read-only snapshot with a single public method.

D-016 (P-3 opening) adds ``capabilities() -> ModelCapabilities`` to I-1 as a
backwards-compatible extension: a typed summary view over the existing
``attention_pattern()`` plus an ``has_moe`` bit that ``AttentionPattern``
cannot express. ``AttentionPattern`` remains the authoritative per-layer
routing source; ``ModelCapabilities`` is a strictly coarser summary used by
scheduler-level gates (continuous batching today, MoE-aware budgeting and
the P-4 bench harness later).

Key constraints (from §6):
  1. `attention_pattern()` expresses per-layer routing (D-015). KV-attention
     layers dispatch to `KVManager`; recurrent layers dispatch to an
     adapter-owned state store keyed by req_id.
  2. KV mutation ownership belongs to `KVManager`. The adapter reads / writes
     KV via `kv_handle` only — it never holds block pointers, decides
     residency, or touches the prefix cache.
  3. `StateDelta` carries non-KV runtime state (sampling RNG, MoE router cache,
     position counter, sliding-window cursor, DeltaNet recurrent state
     ownership info). Counter-examples forbidden inside `state_delta`:
     KV blocks, cache residency mutations, prefix cache pinning.
  4. ``capabilities()`` returns a frozen ``ModelCapabilities`` summary
     derived from ``attention_pattern()`` (plus an ``has_moe`` bit the
     MoE adapters set). Scheduler-level gates must prefer it over
     re-walking ``AttentionPattern``.

P-0 stub (`StubModelAdapter`): minimal conforming implementation — config
carries the shape fields needed by the test harness, `build` is a no-op,
prefill / decode return zero logits + an empty `StateDelta`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import mlx.core as mx

from silica.kvcache.manager import KVHandle
from silica.weights.provider import WeightProvider

if TYPE_CHECKING:
    # Avoid a runtime cycle: capabilities.py imports AttentionKind /
    # AttentionPattern from this module. The Protocol signature uses
    # ``ModelCapabilities`` as a forward-referenced string annotation
    # via ``from __future__ import annotations``; StubModelAdapter's
    # ``capabilities()`` does a local import at call time.
    from silica.models.capabilities import ModelCapabilities

# `Module` and `Tokenizer` are deliberately permissive in v0.1.
# P-1 / P-3 adapters will carry concrete types (mlx-lm's `nn.Module`,
# mlx-lm's tokenizer wrapper); the Protocol is tightened at P-0 exit.
Module = Any


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer surface the engine relies on."""

    vocab_size: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


@dataclass
class ModelConfig:
    """Architecture-level model info consumed by Engine / Scheduler.

    v0.1 keeps the public fields minimal and exposes a free-form `extra` map
    for adapter-specific knobs (e.g. sliding-window size, DeltaNet hidden
    state shape). P-1 / P-3 adapters populate via their own factories.
    """

    model_name: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class KVLayout:
    """Per-layer KV shape — consumed by KVCodec and the KV allocator."""

    num_layers: int
    n_kv_heads: int
    head_dim: int
    dtype: mx.Dtype


class AttentionKind(str, Enum):
    """Per-layer attention classification (D-015).

    KV-attention variants (`GLOBAL` / `SLIDING` / `HYBRID`) dispatch to
    `KVManager`-owned blocks. Recurrent variants (`RECURRENT` pure-linear /
    `HYBRID_DELTANET` Qwen3.5's interleaved stack) dispatch to an
    adapter-owned per-request state store.
    """

    GLOBAL = "global"
    SLIDING = "sliding"
    HYBRID = "hybrid"
    RECURRENT = "recurrent"
    HYBRID_DELTANET = "hybrid_deltanet"


@dataclass(frozen=True)
class AttentionPattern:
    """Ordered attention kind per transformer layer.

    `len(per_layer) == KVLayout.num_layers`. The scheduler walks this tuple
    to decide, for each layer, whether to route KV through `KVManager` or to
    hand off to the adapter's recurrent-state store (D-015).
    """

    per_layer: tuple[AttentionKind, ...] = ()


@dataclass(frozen=True)
class StateDelta:
    """Read-only snapshot returned by `prefill` / `decode_step` (D-015).

    The only public method is `recurrent_bytes()` — used by the scheduler to
    include per-request recurrent state in its memory budget. `_payload` is
    adapter-specific and opaque to the engine (the adapter owns the per-
    request recurrent-state store keyed by `req_id` via `KVHandle`; the
    snapshot merely references an entry in that store).
    """

    _recurrent_bytes: int = 0
    _payload: Any = None

    def recurrent_bytes(self) -> int:
        return self._recurrent_bytes


@runtime_checkable
class ModelAdapter(Protocol):
    """Per-model bridge between the engine and the MLX-native model graph."""

    config: ModelConfig

    def build(self, weight_provider: WeightProvider) -> Module: ...

    def kv_layout(self) -> KVLayout: ...

    def attention_pattern(self) -> AttentionPattern: ...

    def tokenizer(self) -> Tokenizer: ...

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]: ...

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]: ...

    def capabilities(self) -> ModelCapabilities: ...


class _StubTokenizer:
    """Minimal tokenizer for the P-0 stub — roundtrips empty text only."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return []

    def decode(self, token_ids: Sequence[int]) -> str:
        return ""


class StubModelAdapter:
    """P-0 I-1 conformance stub — zero logits, empty state, no weights.

    The stub satisfies the Protocol shape (all methods callable, attributes
    present) so `tests/test_interfaces.py` can instantiate and exercise it.
    It is NOT a runnable model: `build` accepts any `WeightProvider`, returns
    a sentinel; `prefill` / `decode_step` return zero logits of `vocab_size`
    and an empty `StateDelta`.
    """

    def __init__(
        self,
        *,
        model_name: str = "stub",
        num_layers: int = 2,
        hidden_size: int = 32,
        vocab_size: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 16,
    ) -> None:
        self.config = ModelConfig(
            model_name=model_name,
            num_layers=num_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )
        self._kv_layout = KVLayout(
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            dtype=mx.float16,
        )
        self._pattern = AttentionPattern(
            per_layer=tuple(AttentionKind.GLOBAL for _ in range(num_layers))
        )
        self._tokenizer = _StubTokenizer(vocab_size=vocab_size)

    def build(self, weight_provider: WeightProvider) -> Module:
        return object()

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def attention_pattern(self) -> AttentionPattern:
        return self._pattern

    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        logits = mx.zeros((self.config.vocab_size,), dtype=mx.float16)
        return logits, StateDelta()

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        logits = mx.zeros((self.config.vocab_size,), dtype=mx.float16)
        return logits, StateDelta()

    def capabilities(self) -> ModelCapabilities:
        # Local import: capabilities.py imports AttentionKind / AttentionPattern
        # from this module, so pulling it at module load would cycle.
        from silica.models.capabilities import capabilities_from_attention_pattern

        return capabilities_from_attention_pattern(self._pattern)

    def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
        """Build a per-layer batched cache list (P-3-C3a).

        The stub has no real forward so the list is never consumed in
        anger, but it must match the scheduler's expectation: one
        ``BatchKVCache`` per layer with the shared ``left_padding``.
        Real adapters override to express family-specific shapes (e.g.
        ``Qwen3_5Adapter`` returns a hybrid ArraysCache / BatchKVCache
        list).
        """
        from mlx_lm.models.cache import BatchKVCache

        return [
            BatchKVCache(left_padding=left_padding)
            for _ in range(self.config.num_layers)
        ]
