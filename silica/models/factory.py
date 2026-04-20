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
from silica.models.qwen3 import Qwen3Adapter
from silica.models.qwen3_5 import Qwen3_5Adapter

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
    return Gemma4Adapter(model, tokenizer, kv_manager=kv)


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
_ADAPTERS: dict[str, _AdapterBuilder] = {
    "qwen3": _build_qwen3,
    "qwen3_5": _build_qwen3_5,
    "gemma4": _build_gemma4,
}


def supported_model_types() -> tuple[str, ...]:
    """List of ``model_type`` strings the factory knows how to build for."""
    return tuple(sorted(_ADAPTERS.keys()))


def adapter_for_repo(repo: str) -> tuple[ModelAdapter, SimpleKVCache]:
    """Load ``repo`` via mlx-lm and build the matching family adapter.

    Returns ``(adapter, kv)`` so the Engine can drive both. Raises
    ``NotImplementedError`` with a greppable list of supported families
    when the checkpoint's ``model_type`` is not registered.
    """
    model, tokenizer = _mlx_lm_load(repo)  # type: ignore[misc]
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
