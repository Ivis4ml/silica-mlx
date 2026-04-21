"""silica.models.factory — dispatch an HF repo to the right family adapter.

When the caller **knows** the family (e.g. a probe that is specifically
about Qwen3.5) it should use the concrete class directly:
``Qwen3_5Adapter.from_hf_repo(repo)``. When the caller has only the repo
name (e.g. the CLI accepting ``--model``), ``adapter_for_repo`` dispatches
based on ``model.model_type`` — which is the source of truth mlx-lm
itself uses when picking its own model module.

Adding a new family is a two-step change, kept intentionally mechanical:

1. Write ``silica/models/<family>.py`` with a ``FooAdapter`` class that
   satisfies I-1 ``ModelAdapter``.
2. Register the mapping in ``_ADAPTERS`` below.

The registry is a small dict rather than a decorator system so the set
of supported families is greppable and reviewable.
"""

from __future__ import annotations

from typing import Any, Callable

from mlx_lm.utils import load as _mlx_lm_load

from silica.kvcache.simple import SimpleKVCache
from silica.models.adapter import ModelAdapter
from silica.models.gemma4 import Gemma4Adapter
from silica.models.gemma4_moe import Gemma4MoeAdapter
from silica.models.qwen3 import Qwen3Adapter
from silica.models.qwen3_5 import Qwen3_5Adapter
from silica.models.qwen3_5_moe import Qwen3_5MoeAdapter

_AdapterBuilder = Callable[[Any, Any, SimpleKVCache], ModelAdapter]


def _build_qwen3(
    model: Any, tokenizer: Any, kv: SimpleKVCache
) -> ModelAdapter:
    return Qwen3Adapter(model, tokenizer, kv_manager=kv)


def _build_qwen3_5(
    model: Any, tokenizer: Any, kv: SimpleKVCache
) -> ModelAdapter:
    return Qwen3_5Adapter(model, tokenizer, kv_manager=kv)


def _build_gemma4(
    model: Any, tokenizer: Any, kv: SimpleKVCache
) -> ModelAdapter:
    """Dispatch Gemma4 checkpoints between dense and MoE adapters.

    mlx-lm reports ``model_type="gemma4"`` for BOTH the dense
    Gemma4-31B path (``Gemma4Adapter``, P-3-D1) and the MoE
    Gemma4-26B-A4B path (``Gemma4MoeAdapter``, P-3-E1.2); keying the
    ``_ADAPTERS`` map on ``model_type`` alone is therefore ambiguous
    for the Gemma4 family. Instead of widening the map to a
    two-key tuple (no other family needs it today), we branch
    locally on ``args.text_config.enable_moe_block`` — the same
    field that controls mlx-lm's own
    ``gemma4_text.DecoderLayer`` dense-vs-MoE construction.
    """
    tc = Gemma4Adapter._text_config_dict(model)
    enable_moe = bool(tc.get("enable_moe_block", False))
    num_experts = int(tc.get("num_experts", 0) or 0)
    if enable_moe or num_experts > 0:
        return Gemma4MoeAdapter(model, tokenizer, kv_manager=kv)
    return Gemma4Adapter(model, tokenizer, kv_manager=kv)


def _build_qwen3_5_moe(
    model: Any, tokenizer: Any, kv: SimpleKVCache
) -> ModelAdapter:
    return Qwen3_5MoeAdapter(model, tokenizer, kv_manager=kv)


# mlx-lm's model_type → Silica adapter builder. The Gemma4-31B load
# probe (P-3-D0, commit 8718bcd) reports the outer model_type as
# ``"gemma4"`` for the multimodal-shell repo; that is the only
# alias D1 registers. The inner ``"gemma4_text"`` checkpoint stores
# its structural args directly on ``model.args`` (a ``ModelArgs``
# dataclass), not under ``args.text_config`` — ``Gemma4Adapter``
# currently only supports the outer wrapper's dict layout, so
# registering ``"gemma4_text"`` here would produce an adapter whose
# variant guard and layer_types reader silently see an empty config.
# Add the alias (plus a widened ``_text_config_dict``) only when a
# bare text-model checkpoint is a real target.
#
# P-3-E0 probe confirmed Qwen3.5-MoE reports a distinct
# model_type ``"qwen3_5_moe"`` (unlike Gemma4-MoE which collides
# with Gemma4-31B-dense on ``"gemma4"``), so a plain _ADAPTERS
# entry is sufficient; Gemma4-MoE dispatch via a local
# enable_moe_block branch inside _build_gemma4 lands in P-3-E1.2.
_ADAPTERS: dict[str, _AdapterBuilder] = {
    "qwen3": _build_qwen3,
    "qwen3_5": _build_qwen3_5,
    "qwen3_5_moe": _build_qwen3_5_moe,
    "gemma4": _build_gemma4,
}


def supported_model_types() -> tuple[str, ...]:
    """List of ``model_type`` strings the factory knows how to build for."""
    return tuple(sorted(_ADAPTERS.keys()))


def adapter_from_loaded_model(
    model: Any, tokenizer: Any
) -> tuple[ModelAdapter, SimpleKVCache]:
    """Build the matching family adapter for an already-loaded
    ``(model, tokenizer)`` pair.

    Skips ``mlx_lm.load`` — callers that have already paid the load
    cost (e.g. probe scripts capturing Phase 1 metadata before
    dispatching) can pass the result through this helper instead of
    re-calling ``adapter_for_repo`` which would load the checkpoint a
    second time. On large MoE checkpoints (16-20 GB quantized)
    double-loading may exhaust device memory and pollute probe
    findings; see ``scripts/probe_qwen3_5_moe_load.py`` / ``scripts/
    probe_gemma4_moe_load.py`` for the first callers.

    Returns ``(adapter, kv)`` exactly like ``adapter_for_repo``.
    Raises ``NotImplementedError`` with the same error text when the
    ``model_type`` is not registered.
    """
    kv = SimpleKVCache.from_model(model)
    model_type = getattr(model, "model_type", None)
    builder = _ADAPTERS.get(str(model_type)) if model_type is not None else None
    if builder is None:
        raise NotImplementedError(
            f"No Silica adapter registered for model_type={model_type!r}. "
            f"Supported families: {supported_model_types()}. "
            f"Add silica/models/<family>.py and register it in "
            f"silica.models.factory._ADAPTERS."
        )
    return builder(model, tokenizer, kv), kv


def adapter_for_repo(repo: str) -> tuple[ModelAdapter, SimpleKVCache]:
    """Load ``repo`` via mlx-lm and build the matching family adapter.

    Thin wrapper around ``adapter_from_loaded_model`` — loads the
    checkpoint once, then delegates to the dispatch helper. Returns
    ``(adapter, kv)`` so the Engine can drive both. Raises
    ``NotImplementedError`` with a greppable list of supported families
    when the checkpoint's ``model_type`` is not registered.
    """
    model, tokenizer = _mlx_lm_load(repo)  # type: ignore[misc]
    return adapter_from_loaded_model(model, tokenizer)
