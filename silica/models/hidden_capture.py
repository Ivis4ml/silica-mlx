"""D-021 step 6 (αβ) — target-hidden capture for target-conditioned drafters.

Public surface:

- :func:`run_qwen3_5_forward_with_capture` runs a single-request verify
  forward through an mlx-lm Qwen3.5 ``TextModel``, returning all-position
  logits AND a dict of selected-layer hidden states keyed at the
  ``bstnxbt/dflash-mlx`` convention: key ``0`` = embedding output (before
  any decoder layer), key ``i + 1`` (for ``i ∈ [0, num_hidden_layers)``)
  = the residual-stream hidden state immediately after
  ``model.layers[i]``. Empty ``capture_layer_ids`` returns ``{}`` and is
  byte-exact (modulo MLX dispatch) with the existing ``forward_full``
  path; the helper exists to support D-021 step 6 sub-unit (β)
  ``DFlashDrafter`` consuming target hidden states without modifying
  ``decode_step_multi``.
- :class:`HiddenCaptureAdapter` Protocol mixin declared on the adapter
  side — parallel to ``PreNormCaptureAdapter``. Adapters that do not
  ship hidden-state capture (Qwen3 / Gemma4 + their MoE variants in
  the C.4 spike) leave the method unimplemented and the
  ``isinstance(adapter, HiddenCaptureAdapter)`` check returns False.

The helper duplicates the layer-iteration logic from
``mlx_lm.models.qwen3_5.Qwen3_5TextModel.__call__`` (visible in the
installed mlx-lm package; see also `dflash_mlx.runtime`'s
``target_forward_with_hidden_states`` for the upstream-equivalent
pattern). Duplicating ~25 lines keeps the *capture-disabled* path on
``decode_step_multi`` byte-exact with v1.7.19 by construction — there
is no instrumentation in the model forward at all when capture is not
requested. The per-call cost is one extra ``embed_tokens`` evaluation
versus the in-place capture variant a future step could implement;
(αβ.1) opts for the simpler external loop.

For the MoE adapter (sub-unit (αβ.2)) the same pattern duplicates
``mlx_lm.models.qwen3_5_moe.Qwen3_5MoeTextModel.__call__``; that lands
in a separate slice once the dense path is stable.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from silica.kvcache.manager import KVHandle
from silica.models.adapter import StateDelta


def run_qwen3_5_forward_with_capture(
    model: Any,
    tokens: mx.array,
    cache_list: list[Any],
    capture_layer_ids: frozenset[int],
) -> tuple[mx.array, dict[int, mx.array]]:
    """Run a single-request Qwen3.5 verify forward with hidden-state capture.

    Args:
        model: mlx-lm ``Qwen3_5TextModel`` (the outer ``TextModel`` whose
            ``.model`` attribute is the inner ``Qwen3_5TextModel``; both
            shapes are accepted via duck-typed access to ``embed_tokens``
            / ``layers`` / ``norm`` / ``fa_idx`` / ``ssm_idx`` /
            ``embed_tokens.as_linear`` or ``lm_head``).
        tokens: 1-D ``(T,)`` token ids — single-request verify shape,
            matching ``decode_step_multi``'s contract.
        cache_list: per-layer Silica-owned cache list. Mutated in-place
            by mlx-lm's ``layer.__call__``.
        capture_layer_ids: which layer-output positions to capture.
            Convention: ``0`` = embedding output (no layer applied),
            ``i + 1`` for ``i ∈ [0, num_hidden_layers)`` = output of
            ``model.layers[i]``. Empty set returns an empty captured
            dict and is the cheapest way to invoke this helper without
            doing capture work.

    Returns:
        ``(logits_full, captured)`` where ``logits_full`` has shape
        ``(T, V)`` (post-norm + tied-embedding or ``lm_head``
        projection, matching ``forward_full``'s output) and
        ``captured`` is a ``dict[int, mx.array]`` whose keys are exactly
        the requested ``capture_layer_ids`` (only those that fall in
        valid range; the caller is responsible for asking for valid
        layer ids).

    Mirrors ``Qwen3_5TextModel.__call__`` from mlx-lm 0.31.x:

    - Embed tokens via ``inner.embed_tokens(tokens[None])``.
    - Build per-layer-type masks via ``create_attention_mask`` and
      ``create_ssm_mask`` against the appropriate cache rows.
    - Iterate ``inner.layers`` with the matching mask + cache row.
    - Apply final ``inner.norm`` then the tied-embedding or ``lm_head``
      projection.

    The ``(αβ.1)`` test pins logits byte-exactness against
    ``forward_full`` on cached ``Qwen/Qwen3.5-0.8B`` for an empty
    ``capture_layer_ids`` argument; the difference between the two
    paths in that case is one extra ``embed_tokens`` evaluation
    (versus ``forward_full``'s in-model path), which is allocation-
    deterministic on MLX.
    """
    if tokens.ndim != 1:
        raise ValueError(
            f"expected 1-D tokens (T,), got shape {tuple(tokens.shape)}"
        )
    if tokens.size == 0:
        raise ValueError("tokens must be non-empty")

    # mlx-lm 0.31.x's Qwen3.5 wraps the inner text model as either
    # ``model.model`` (when ``model`` is the outer ``TextModel`` directly)
    # or ``model.language_model.model`` (when ``model`` is the
    # multimodal-style outer ``Model`` whose ``.language_model`` is the
    # ``TextModel``). Mirror ``dflash_mlx.runtime._target_text_wrapper``:
    # the wrapper is whichever object has ``.model`` and ``.args``;
    # the inner Qwen3_5TextModel hangs off ``wrapper.model``.
    if hasattr(model, "model"):
        wrapper = model
    elif hasattr(model, "language_model"):
        wrapper = model.language_model
    else:
        raise AttributeError(
            f"Unsupported model wrapper: {type(model)!r} has neither "
            "'model' nor 'language_model' attribute"
        )
    inner = wrapper.model

    inputs_2d = tokens[None]  # (1, T)
    hidden_states: mx.array = inner.embed_tokens(inputs_2d)

    captured: dict[int, mx.array] = {}
    if 0 in capture_layer_ids:
        captured[0] = hidden_states

    fa_mask = create_attention_mask(hidden_states, cache_list[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache_list[inner.ssm_idx])

    for layer_index, (layer, layer_cache) in enumerate(
        zip(inner.layers, cache_list, strict=True)
    ):
        mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
        if (layer_index + 1) in capture_layer_ids:
            captured[layer_index + 1] = hidden_states

    normalised = inner.norm(hidden_states)

    # Tied-embedding projection vs lm_head — match mlx-lm's
    # ``Qwen3_5TextModel`` outer ``TextModel.__call__``. ``args``
    # lives on the wrapper (TextModel), not the inner Qwen3_5TextModel.
    if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
        logits_2d: mx.array = inner.embed_tokens.as_linear(normalised)
    else:
        logits_2d = wrapper.lm_head(normalised)

    # Drop the batch dim. ``forward_full`` returns ``(T, V)``.
    logits: mx.array = logits_2d[0]
    return logits, captured


@runtime_checkable
class HiddenCaptureAdapter(Protocol):
    """Mixin implemented by adapters that ship target-hidden capture.

    D-021 step 6 (αβ.1) lands this Protocol on ``Qwen3_5Adapter`` only;
    (αβ.2) extends it to ``Qwen3_5MoeAdapter``. Other families
    (``Qwen3Adapter``, ``Gemma4Adapter``, ``Gemma4MoeAdapter``) do not
    ship the capture surface — no upstream DFlash drafter targets them
    in ``dflash_mlx.generate.DRAFT_REGISTRY``, so the (β) wrapper's
    ``isinstance(adapter, HiddenCaptureAdapter)`` check is the gate
    that prevents silently-wrong fallback behaviour.

    The capture method takes a ``frozenset[int]`` of layer ids using
    the convention ``0`` = embedding output, ``i + 1`` = output of
    ``model.layers[i]``. The set choice is forwarded by the (β)
    wrapper from ``DFlashDraftModel.target_layer_ids`` (offset by +1
    to match ``dflash_mlx.runtime.target_forward_with_hidden_states``).
    Empty set returns an empty dict — the call shape is otherwise
    identical to ``decode_step_multi`` and exists so callers can route
    through one method when capture state is dynamically toggled per
    cycle.
    """

    def decode_step_multi_with_capture(
        self,
        tokens: mx.array,
        kv_handle: KVHandle,
        capture_layer_ids: frozenset[int],
    ) -> tuple[mx.array, dict[int, mx.array], StateDelta]:
        """Run a single-request verify forward and capture hidden states.

        Returns ``(logits, captured, state_delta)`` where ``logits`` has
        shape ``(T, V)`` matching ``decode_step_multi``, ``captured``
        is the layer-id-to-hidden-state dict described above, and
        ``state_delta`` is the same recurrent-bytes accounting payload
        ``decode_step_multi`` returns.
        """
        ...


__all__ = [
    "HiddenCaptureAdapter",
    "run_qwen3_5_forward_with_capture",
]
