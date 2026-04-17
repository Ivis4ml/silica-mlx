"""silica.weights.resident — ResidentWeightProvider (I-4 dense implementation).

P-0 stub: dict-backed storage; P-3 extends with real safetensors loading via
mlx-lm (D-004). Per D-011, the MoE per-expert call path raises
`NotImplementedError` — a MoE adapter paired with a dense provider must fail
loudly at the point of wiring, not silently later.
"""

from __future__ import annotations

from collections.abc import Sequence

from silica.weights.provider import ExpertWeights, LayerWeights

_DENSE_NO_EXPERT_MSG = "dense provider has no per-expert path"


class ResidentWeightProvider:
    """Dense, fully-resident weight provider.

    Every layer that matters is held in unified memory for the provider's
    lifetime. `prefetch` / `release` are no-ops (everything is already
    resident). `resident_bytes` sums the layers the provider currently owns.

    MoE paths raise `NotImplementedError` so that pairing a MoE adapter with
    this provider is a configuration error, not a silent runtime failure
    (D-011). A MoE-capable resident variant is a P-3 deliverable when the
    MoE adapters are introduced.
    """

    def __init__(self, layers: dict[int, LayerWeights] | None = None) -> None:
        self._layers: dict[int, LayerWeights] = dict(layers) if layers else {}

    def get_layer(self, layer_idx: int) -> LayerWeights:
        return self._layers[layer_idx]

    def prefetch(self, layer_indices: Sequence[int]) -> None:
        return None

    def release(self, layer_idx: int) -> None:
        return None

    def resident_bytes(self) -> int:
        return sum(layer.resident_bytes for layer in self._layers.values())

    def get_expert(self, layer_idx: int, expert_id: int) -> ExpertWeights:
        raise NotImplementedError(_DENSE_NO_EXPERT_MSG)

    def prefetch_experts(
        self, layer_idx: int, expert_ids: Sequence[int]
    ) -> None:
        raise NotImplementedError(_DENSE_NO_EXPERT_MSG)

    def release_expert(self, layer_idx: int, expert_id: int) -> None:
        raise NotImplementedError(_DENSE_NO_EXPERT_MSG)
