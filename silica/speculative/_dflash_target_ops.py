"""DFlash-only private helpers — target-side ops the wrapper needs.

D-021 step 6 sub-unit (δ.1) — keep ``adapter._model`` access centralised
in this module rather than scattered across ``DFlashDrafter``. Two
operations the drafter forward needs:

- :func:`target_embed_tokens` — runs the target's input embedding on a
  ``(B, T)`` token-id batch. The drafter's noise-embedding input is
  ``embed_tokens(block_token_buffer)`` per upstream
  ``dflash_mlx.runtime.generate_dflash_once`` (line ~1450).
- :func:`lm_head_logits` — runs the target's language-model projection
  on a ``(B, T, hidden)`` array. Mirrors upstream's ``_lm_head_logits``
  helper, which dispatches between tied-embedding and a separate
  ``lm_head`` based on the model wrapper's ``args``.

These helpers are **DFlash-private** by design (underscore-prefixed
module). They duplicate two ~3-line wrappers from upstream
``dflash_mlx.runtime`` rather than promoting the surface to a public
``ModelAdapter`` method, because:

1. The DFlash drafter is the only consumer at sub-unit (δ.1);
   widening the public ``ModelAdapter`` Protocol with embed/lm-head
   accessors for a single drafter would over-fit the surface.
2. Other target-conditioned drafters (C.3 MTP head, C.6 self-spec,
   etc.) may need different target-side ops. When a second consumer
   lands, this module promotes to a public mixin; today there is no
   second consumer to design against.

Both helpers handle the two-level wrapper layout that αβ.1's
``run_qwen3_5_forward_with_capture`` already navigates: outer
``Model`` (with ``language_model`` attribute) vs outer ``TextModel``
(with ``model`` attribute). The inner ``Qwen3_5TextModel``'s
``embed_tokens`` is the load-bearing entry point.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _resolve_wrapper(target_model: Any) -> Any:
    """Return the wrapper that owns ``args`` and ``model`` (the inner
    text model). Mirrors
    ``dflash_mlx.runtime._target_text_wrapper`` and silica's
    ``run_qwen3_5_forward_with_capture`` resolution."""
    if hasattr(target_model, "model"):
        return target_model
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    raise AttributeError(
        f"Unsupported target model wrapper for DFlash target ops: "
        f"{type(target_model)!r} has neither 'model' nor 'language_model'."
    )


def target_embed_tokens(
    target_adapter: Any, tokens_2d: mx.array
) -> mx.array:
    """Run the target's input embedding on ``tokens_2d`` of shape
    ``(B, T)``. Returns ``(B, T, hidden_size)`` — the noise-embedding
    input the DFlash drafter consumes. Bypasses the adapter's standard
    forward path; this only runs the embedding lookup, no attention or
    cache mutation.

    The function reaches into ``target_adapter._model`` once; it is
    the only place in the DFlash wrapper that does so.
    """
    if tokens_2d.ndim != 2:
        raise ValueError(
            f"target_embed_tokens expects 2-D (B, T) tokens; got "
            f"shape {tuple(tokens_2d.shape)}"
        )
    model = target_adapter._model  # noqa: SLF001 — DFlash-private accessor
    inner = _resolve_wrapper(model).model
    embed: mx.array = inner.embed_tokens(tokens_2d)
    return embed


def lm_head_logits(
    target_adapter: Any, hidden: mx.array
) -> mx.array:
    """Project ``hidden`` of shape ``(B, T, hidden_size)`` to logits
    ``(B, T, vocab_size)`` via the target's language-model head.

    Mirrors ``dflash_mlx.runtime._lm_head_logits``: dispatches between
    tied-embedding (``inner.embed_tokens.as_linear``) and a separate
    ``lm_head`` linear based on the wrapper's ``args.tie_word_embeddings``.
    """
    model = target_adapter._model  # noqa: SLF001 — DFlash-private accessor
    wrapper = _resolve_wrapper(model)
    args = getattr(wrapper, "args", None)
    if getattr(args, "tie_word_embeddings", True):
        inner = wrapper.model
        out: mx.array = inner.embed_tokens.as_linear(hidden)
        return out
    out = wrapper.lm_head(hidden)
    return out


__all__ = [
    "lm_head_logits",
    "target_embed_tokens",
]
