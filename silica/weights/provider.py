"""silica.weights.provider — I-4 WeightProvider Protocol + shared data types.

I-4 (PLAN.md §6) is the residency / streaming abstraction for weights. D-011
extends the interface with three per-expert-granularity methods so MoE
adapters can load only active experts. Dense providers implement the per-expert
methods as `NotImplementedError` (not no-ops) so a MoE adapter wired to a
dense provider fails loudly — a model-registry configuration error, not a
silent runtime degradation.

Residency / prefetch semantics:
  - `get_layer(i)` blocks synchronously — returns a `LayerWeights` the adapter
    can use immediately.
  - `prefetch(indices)` is a hint; implementations are free to no-op.
  - `release(i)` signals "caller is done with this layer"; the provider may
    reclaim memory or keep it resident.
  - `resident_bytes()` is D-012-canonical — physical bytes currently held,
    excluding transient scratch and allocator headroom.

Streaming implementations (P-6) must exploit Apple unified memory
(Principle 2); they are not emulating a PCIe device.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import mlx.core as mx


@dataclass
class LayerWeights:
    """Weights for one transformer layer.

    v0.1 uses a free-form `tensors` dict so P-3 adapters can name their
    parameters without committing the Protocol to a specific layout (q_proj /
    k_proj / gate / up / down vary across Qwen3.5, Gemma4, and MoE variants).
    """

    tensors: dict[str, mx.array] = field(default_factory=dict)
    resident_bytes: int = 0


@dataclass
class ExpertWeights:
    """Weights for one MoE expert (typically a gate / up / down triple).

    Analogous to `LayerWeights` but scoped to a single expert within a MoE
    layer's FFN. D-011: MoE adapter FFN execution goes through `get_expert` to
    load only the active top-k experts; dense providers must not satisfy this
    call path.
    """

    tensors: dict[str, mx.array] = field(default_factory=dict)
    resident_bytes: int = 0


@runtime_checkable
class WeightProvider(Protocol):
    """Residency / streaming abstraction for weights (I-4, extended by D-011)."""

    def get_layer(self, layer_idx: int) -> LayerWeights: ...

    def prefetch(self, layer_indices: Sequence[int]) -> None: ...

    def release(self, layer_idx: int) -> None: ...

    def resident_bytes(self) -> int: ...

    # D-011 MoE-aware per-expert methods. Dense providers raise
    # NotImplementedError here; MoE providers (P-6 streaming + P-3 MoE
    # resident variant) implement them.
    def get_expert(self, layer_idx: int, expert_id: int) -> ExpertWeights: ...

    def prefetch_experts(
        self, layer_idx: int, expert_ids: Sequence[int]
    ) -> None: ...

    def release_expert(self, layer_idx: int, expert_id: int) -> None: ...
