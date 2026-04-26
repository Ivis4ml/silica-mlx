"""silica.models.qwen3_5 — I-1 adapter for the Qwen3.5 hybrid family.

Scope: Qwen3.5 models — 0.8B / 27B / 35B-A3B / ... — all of which share
an interleaved Gated DeltaNet + Gated Attention stack, MTP head, and a
multimodal lm_head built on top. mlx-lm ships them under
``mlx_lm.models.qwen3_5``; this adapter wraps that module for Silica's
I-1 Protocol.

For plain Qwen3 (0.6B / 4B / 7B etc., pure KV attention) use
``silica.models.qwen3.Qwen3Adapter`` instead. One adapter per model
generation keeps each family's quirks explicit; adding a new family
(Qwen4, DeepSeek, Kimi, GLM, MiniMax, …) is a new file, not a growing
conditional chain.

Qwen3.5-specific traits this adapter handles:

- `model.model_type == "qwen3_5"`.
- Per-layer `layer.is_linear` distinguishes DeltaNet from full attention;
  Silica maps it to ``AttentionKind.HYBRID_DELTANET`` vs ``GLOBAL``
  (D-015). The dispatch lines up 1:1 with mlx-lm's cache factory
  (``ArraysCache`` for linear, ``KVCache`` for full attention).
- `self_attn.num_key_value_heads` + `self_attn.head_dim` on the
  full-attention blocks (NOT `n_kv_heads` — that name is Qwen3-plain's).
- `model.args.text_config` is a **nested dict** carrying hidden_size /
  num_hidden_layers (flat Qwen3 exposes these on ``args`` directly).
- mlx-lm's ``Model.sanitize`` drops ``vision_tower`` / ``model.visual``
  at load; ``TextModel.sanitize`` drops ``mtp.*`` and shifts any
  ``RMSNorm.weight`` by +1.0 when MTP weights were present in the
  checkpoint (non-obvious load-correctness detail that D-004 explicitly
  relies on — one of the reasons Silica borrows mlx-lm's loader).
- P-1 runs single-request through ``SimpleKVCache`` (verified at M-2).
- P-2 batched path for hybrid DeltaNet is deferred to P-3 via an
  adapter-owned ``BatchRecurrentStateStore`` (see ``docs/P2_OPENING.md``
  scope statement). The P-2 ``ContinuousBatcher`` will refuse to accept
  this adapter at construction time.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.utils import load as _mlx_lm_load

from silica.kvcache.manager import KVHandle
from silica.kvcache.simple import SimpleKVCache
from silica.mlx.runner import forward
from silica.models.adapter import (
    AttentionKind,
    AttentionPattern,
    KVLayout,
    ModelConfig,
    Module,
    StateDelta,
    Tokenizer,
)
from silica.models.capabilities import (
    ModelCapabilities,
    capabilities_from_attention_pattern,
)
from silica.models.pre_norm_capture import (
    _PreNormCaptureBufferHolder,
    apply_k_norm_then_rope_to_block,
    install_pre_norm_capture_proxies,
)
from silica.models.recurrent import (
    RecurrentSnapshot,
    _RecurrentLayerEntry,
)
from silica.weights.provider import WeightProvider


class Qwen3_5Adapter:
    """I-1 ModelAdapter for the Qwen3.5 family (hybrid DeltaNet + full attn)."""

    config: ModelConfig

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        kv_manager: SimpleKVCache,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._kv_manager = kv_manager
        self.config = self._build_config(model, tokenizer)
        self._kv_layout = self._build_kv_layout(model)
        self._attention_pattern = self._build_attention_pattern(model)
        # P-5-F F.1: install pre-norm K capture proxies on full-
        # attention layers only. DeltaNet (``layer.is_linear``) layers
        # have no k_proj surface to wrap — they participate in the
        # recurrent-state path, not the prefix-cache K/V store.
        self._attn_layer_indices: list[int] = [
            i
            for i, layer in enumerate(model.layers)
            if not getattr(layer, "is_linear", False)
        ]
        self._capture_holder = _PreNormCaptureBufferHolder()
        install_pre_norm_capture_proxies(
            model,
            attn_layer_indices=self._attn_layer_indices,
            holder=self._capture_holder,
        )

    @classmethod
    def from_hf_repo(cls, repo: str) -> tuple[Qwen3_5Adapter, SimpleKVCache]:
        """Load ``repo`` via mlx-lm, build ``SimpleKVCache`` + adapter."""
        # mlx-lm's load() returns Union[2tuple, 3tuple]; the 2-tuple variant
        # applies with return_config omitted.
        model, tokenizer = _mlx_lm_load(repo)  # type: ignore[misc]
        kv = SimpleKVCache.from_model(model)
        return cls(model, tokenizer, kv_manager=kv), kv

    def build(self, weight_provider: WeightProvider) -> Module:
        """Return the already-loaded mlx-lm model (D-004: borrow the loader)."""
        del weight_provider
        return self._model

    def kv_layout(self) -> KVLayout:
        return self._kv_layout

    def attention_pattern(self) -> AttentionPattern:
        return self._attention_pattern

    def capabilities(self) -> ModelCapabilities:
        # Qwen3.5 is hybrid DeltaNet + GQA; has_recurrent_state is True
        # via ``HYBRID_DELTANET`` layers in the pattern. No MoE in the
        # dense variants; the MoE-A3B variant lives in a separate adapter
        # that will override ``has_moe=True`` when it lands in P-3.
        return capabilities_from_attention_pattern(self._attention_pattern)

    def make_batch_cache(self, left_padding: list[int]) -> list[Any]:
        """Build a hybrid per-layer batched cache list (P-3-C3a).

        DeltaNet layers (``layer.is_linear == True``) get
        ``ArraysCache(size=2)`` — the same shape mlx-lm's
        ``Model.make_cache`` returns for single-request, but with
        ``left_padding`` so ``make_mask`` can align the SSM mask across
        variable-length rows. Global attention layers get the usual
        ``BatchKVCache``.

        The gate in ``ContinuousBatcher._enforce_capability_gate`` still
        rejects ``HYBRID_DELTANET`` adapters as of C3a, so this method
        is reachable only once P-3-C3c lifts the gate. Writing it now
        gives C3b (wiring ``ArraysCache.filter`` / ``.extend`` into the
        batcher's reclaim and admit paths) a stable factory to call
        against.
        """
        from mlx_lm.models.cache import ArraysCache, BatchKVCache

        layers = self._model.layers
        caches: list[Any] = []
        for layer in layers:
            if getattr(layer, "is_linear", False):
                caches.append(
                    ArraysCache(size=2, left_padding=left_padding)
                )
            else:
                caches.append(BatchKVCache(left_padding=left_padding))
        return caches

    def tokenizer(self) -> Tokenizer:
        return self._tokenizer  # type: ignore[no-any-return]

    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        cache_list = self._kv_manager.cache_list(kv_handle.req_id)
        logits = forward(self._model, tokens, cache_list)
        return logits, StateDelta(
            _recurrent_bytes=self._recurrent_state_bytes(cache_list)
        )

    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]:
        cache_list = self._kv_manager.cache_list(kv_handle.req_id)
        logits = forward(self._model, token, cache_list)
        return logits, StateDelta(
            _recurrent_bytes=self._recurrent_state_bytes(cache_list)
        )

    # --- P-5-F F.1: PreNormCaptureAdapter implementation ---

    def install_pre_norm_capture(
        self, buffer: dict[int, mx.array] | None
    ) -> None:
        """Arm or disarm the K_pre capture buffer for the next forward.

        Disarmed by default (``buffer is None``); decode forwards
        through DeltaNet + GQA stack short-circuit on every full-
        attention layer's proxy. DeltaNet layers have no proxy
        installed and are unaffected either way.
        """
        self._capture_holder.buffer = buffer

    def apply_k_norm_then_rope(
        self,
        attn_layer_pos: int,
        k_pre_block: mx.array,
        *,
        offset: int,
    ) -> mx.array:
        """Reconstruct post-RoPE K on the absolute layer that
        ``attn_layer_pos`` indexes into ``_attn_layer_indices``.

        DeltaNet layers are skipped at install time; the layer this
        method targets is always a full-attention (GLOBAL) layer with
        ``self_attn.k_norm`` and ``self_attn.rope`` set up by mlx-lm's
        Qwen3-Next attention block.
        """
        layer_idx = self._attn_layer_indices[attn_layer_pos]
        attn = self._model.layers[layer_idx].self_attn
        return apply_k_norm_then_rope_to_block(
            k_pre_block,
            k_norm=attn.k_norm,
            rope_instance=attn.rope,
            offset=offset,
        )

    def _recurrent_state_bytes(self, cache_list: list[Any]) -> int:
        """Sum the bytes held by recurrent-state caches only (P-3-C2).

        Walks the (layer, cache) pairs and accumulates ``cache.nbytes``
        when ``layer.is_linear`` is true — those are the DeltaNet
        layers that back ``AttentionKind.HYBRID_DELTANET``. Global
        attention layers are skipped; their bytes belong to KV
        accounting via ``KVManager.budget().resident_bytes``, not to
        ``StateDelta.recurrent_bytes()`` (D-015 items 2 and 5).

        Defensive reads:
          - ``zip(..., strict=False)`` — adapter wraps an external
            cache structure (mlx-lm owns the list); shape mismatch in
            a test fake or probe should not crash the inference hot
            path.
          - ``int(getattr(cache, "nbytes", 0) or 0)`` — tolerates
            caches that do not expose ``nbytes`` (e.g. an in-flight
            None slot inside ``ArraysCache`` before first write).
        """
        total = 0
        for layer, cache in zip(self._model.layers, cache_list, strict=False):
            if getattr(layer, "is_linear", False):
                total += int(getattr(cache, "nbytes", 0) or 0)
        return total

    # --- D-015 adapter-local state lifecycle (P-3-C1) ---
    #
    # These four helpers are NOT part of I-1's frozen Python signatures
    # (see D-015: "lifecycle operations are adapter methods called by
    # the engine ... non-frozen helpers"). They are deliberately minimal
    # in the single-request / non-speculative path, which is what P-1
    # through P-3-B exercises. Batched admission (P-3-C3) and
    # speculative rollback (P-7) will extend the bodies; the signatures
    # are stabilised here so engine and scheduler wiring can be written
    # against them without further churn.

    def commit_state(self, req_id: str, n_accepted: int) -> None:
        """Mark ``n_accepted`` recurrent-state updates as committed.

        In mlx-lm's ``ArraysCache`` model the state is updated in place
        on every forward and there is no separate staging buffer —
        ``decode_step`` has already written the committed state by the
        time ``commit_state`` is called. So this is a no-op under the
        current forward contract. P-7 speculative decoding changes that
        (snapshot before draft, collapse on commit); at that point this
        body gains real behaviour.

        ``n_accepted`` is unused here but retained in the signature
        because D-015 pairs it with ``KVManager.commit(req_id,
        n_accepted)`` — the engine treats the KV and recurrent-state
        commit as a pair and passes the same count to both.
        """
        del req_id, n_accepted

    def rollback_state(self, req_id: str, n_reject: int) -> None:
        """Roll back the last ``n_reject`` recurrent-state steps.

        The current forward path updates ``cache[1]`` (recurrent state)
        in place via ``gated_delta_update``, so rolling back requires a
        pre-draft snapshot that P-1 through P-3 does not take. Raising
        here (rather than silently returning) is intentional: if a
        caller reaches this, they have wired in a draft source without
        the matching snapshot logic, and a silent no-op would corrupt
        decoding.

        Real rollback semantics land with P-7 (speculative decoding)
        alongside ``KVManager.rollback(req_id, n_reject)``.
        """
        del req_id
        raise NotImplementedError(
            f"Qwen3_5Adapter.rollback_state requires a pre-draft "
            f"snapshot of the recurrent state; that snapshot pathway "
            f"lands with P-7 speculative decoding. Got n_reject="
            f"{n_reject}."
        )

    def state_from_prefix(
        self, req_id: str, token_ids: list[int]
    ) -> StateDelta | None:
        """Return reusable recurrent state for a prefix — v0.1 returns None.

        D-015's v0.1 rule: reuse recurrent state only when the **full**
        KV prefix is reused. The caller has no way to ask for that
        specifically, so we conservatively return ``None`` for all
        prefix-reuse requests. DeltaNet's recurrent hidden state is a
        running accumulation over the entire sequence (see
        ``docs/P3_DELTANET_SURVEY.md`` §3.2) — unlike per-token K/V it
        cannot be sliced to a partial prefix, so "always None" is the
        only correct P-3 behaviour.

        Partial-prefix reuse is a v0.2 question (D-015 item 4).
        """
        del req_id, token_ids
        return None

    def free_state(self, req_id: str) -> None:
        """Release the adapter-local state for ``req_id``.

        Under ``SimpleKVCache`` the per-request cache list is owned by
        the KV manager and released by ``KVManager.free(req_id)`` — the
        ArraysCache slots (conv_state + recurrent state) are freed
        there. So the adapter has no separate tenant to release today,
        and this method is a no-op. When P-3-C3 introduces a batched
        recurrent-state store owned by the adapter, this helper is the
        hook where per-row state is evicted.
        """
        del req_id

    # --- P-3-C5.1: adapter-owned recurrent state snapshot / restore ---

    def snapshot_recurrent_state(
        self, cache_list: list[Any], row_idx: int
    ) -> RecurrentSnapshot:
        """Implements ``RecurrentStateAdapter.snapshot_recurrent_state``.

        Walks every DeltaNet layer (``layer.is_linear == True``) in
        ``cache_list``, slices each ``ArraysCache`` slot to leading
        ``(1, ...)`` at ``row_idx``, materialises a detached copy
        (R-C5-2: ``mx.eval`` then ``mx.array`` so subsequent in-place
        rebinds at ``mlx_lm/models/qwen3_5.py:164/166/197`` cannot
        mutate the snapshot), and packs the per-layer ``(conv_state,
        recurrent_state)`` pairs into a frozen ``RecurrentSnapshot``.

        GLOBAL (full-attention) layers are skipped — their K/V is
        owned by ``SyntheticPrefixBlockStore`` and is not part of the
        recurrent snapshot surface (per ``docs/P3_C5_OPENING.md`` §3.1).

        Lazy-allocation case (``cache.cache[i] is None`` because no
        forward has populated this slot yet on this row) is preserved
        by storing ``None`` in the entry; ``restore_recurrent_state``
        leaves the slot ``None`` so the next forward lazy-allocates
        normally rather than seeing a fp32-zero injection.
        """
        if row_idx < 0:
            raise ValueError(f"row_idx must be non-negative, got {row_idx}")
        if len(cache_list) != len(self._model.layers):
            raise ValueError(
                f"cache_list length {len(cache_list)} does not match "
                f"model layers {len(self._model.layers)}"
            )

        entries: list[_RecurrentLayerEntry] = []
        total_nbytes = 0
        for layer_idx, (layer, layer_cache) in enumerate(
            zip(self._model.layers, cache_list, strict=True)
        ):
            if not getattr(layer, "is_linear", False):
                continue
            inner = getattr(layer_cache, "cache", None)
            if inner is None:
                # Defensive — every DeltaNet layer's cache should be an
                # ArraysCache(size=2). If a future change uses a
                # different container, fail loudly rather than silently
                # producing an empty snapshot.
                raise TypeError(
                    f"layer {layer_idx} (DeltaNet) has no .cache attribute "
                    f"on its layer_cache; expected ArraysCache(size=2). "
                    f"Got {type(layer_cache).__name__}."
                )
            slot0 = inner[0]
            slot1 = inner[1]

            snap_conv = _detach_row(slot0, row_idx) if slot0 is not None else None
            snap_recur = (
                _detach_row(slot1, row_idx) if slot1 is not None else None
            )
            entries.append(
                _RecurrentLayerEntry(
                    layer_idx=layer_idx,
                    conv_state=snap_conv,
                    recurrent_state=snap_recur,
                )
            )
            if snap_conv is not None:
                total_nbytes += int(snap_conv.nbytes)
            if snap_recur is not None:
                total_nbytes += int(snap_recur.nbytes)

        return RecurrentSnapshot(entries=tuple(entries), nbytes=total_nbytes)

    def restore_recurrent_state(
        self,
        cache_list: list[Any],
        row_idx: int,
        snapshot: RecurrentSnapshot,
    ) -> None:
        """Implements ``RecurrentStateAdapter.restore_recurrent_state``.

        For each ``_RecurrentLayerEntry`` in ``snapshot.entries``,
        splice the captured per-row tensors back into the
        corresponding layer's ``ArraysCache`` slots at ``row_idx``.
        Other rows in the same batched cache are not modified.

        Restore is value-object semantics: the post-restore state at
        ``row_idx`` reflects what the snapshot captured. Cases per
        slot (delegated to :func:`_splice_row`):

        1. Snapshot entry is ``None`` and live slot is ``None``:
           remains ``None`` (both lazy). Next forward lazy-allocates
           as today.
        2. Snapshot entry is ``None`` and live slot is populated at
           ``B = 1``: live slot is wiped to ``None`` so the next
           forward lazy-allocates fresh state (the snapshot's
           "unallocated" capture is reproduced).
        3. Snapshot entry is ``None`` and live slot is populated at
           ``B > 1``: ``ValueError``. The lazy state is per-slot,
           not per-row; wiping one row of a populated multi-row slot
           is not expressible. C5.2 / C5.3 only take snapshots on
           caches that have already participated in a forward, so
           this case should not arise in production.
        4. Snapshot entry has data and live slot is ``None``:
           snapshot's ``(1, ...)`` tensor is assigned directly. Only
           ``row_idx = 0`` is valid here (no batch dim to index).
        5. Snapshot entry has data and live slot is populated:
           splice — replace exactly ``row_idx`` without touching
           other rows.

        ``row_idx`` is validated up front against the relevant range
        for each case; out-of-range values raise ``IndexError``.
        """
        if row_idx < 0:
            raise ValueError(f"row_idx must be non-negative, got {row_idx}")
        if len(cache_list) != len(self._model.layers):
            raise ValueError(
                f"cache_list length {len(cache_list)} does not match "
                f"model layers {len(self._model.layers)}"
            )

        for entry in snapshot.entries:
            layer_cache = cache_list[entry.layer_idx]
            inner = getattr(layer_cache, "cache", None)
            if inner is None:
                raise TypeError(
                    f"layer {entry.layer_idx} (DeltaNet) cache_list entry has "
                    f"no .cache attribute; expected ArraysCache(size=2). "
                    f"Got {type(layer_cache).__name__}."
                )
            inner[0] = _splice_row(inner[0], entry.conv_state, row_idx)
            inner[1] = _splice_row(inner[1], entry.recurrent_state, row_idx)

    # --- family-specific metadata extraction ---

    @staticmethod
    def _build_config(model: Any, tokenizer: Any) -> ModelConfig:
        text_config = Qwen3_5Adapter._text_config_dict(model)
        return ModelConfig(
            model_name=str(getattr(model, "model_type", "qwen3_5")),
            num_layers=len(model.layers),
            hidden_size=int(text_config.get("hidden_size", 0) or 0),
            vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 0),
            extra={"text_config_keys": sorted(text_config.keys())},
        )

    @staticmethod
    def _build_kv_layout(model: Any) -> KVLayout:
        """Extract KV shape from the first full-attention (non-linear) layer.

        Returns zeros for a pure-recurrent stack — P-2 paged KV of size 0
        is a no-op, which is the correct semantics for that case.

        ``dtype`` is read from a representative attention-projection
        weight (mirrors :func:`Qwen3Adapter._infer_attn_dtype`). Qwen3.5
        ships bf16; the runtime ``BatchKVCache`` stores K/V in the
        weight's dtype. Hardcoding ``mx.float16`` here would make
        codec construction (e.g. ``BlockTurboQuantMSE.encode_tensor``)
        reject the actual K/V dtype — codecs validate their input
        dtype against the codec dtype to keep numerical contracts
        with vqbench (silent up/down-cast would shift decoded output).
        """
        for layer in model.layers:
            if getattr(layer, "is_linear", False):
                continue
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            return KVLayout(
                num_layers=len(model.layers),
                n_kv_heads=int(getattr(sa, "num_key_value_heads", 0) or 0),
                head_dim=int(getattr(sa, "head_dim", 0) or 0),
                dtype=Qwen3_5Adapter._infer_attn_dtype(model),
            )
        return KVLayout(
            num_layers=len(model.layers),
            n_kv_heads=0,
            head_dim=0,
            dtype=Qwen3_5Adapter._infer_attn_dtype(model),
        )

    @staticmethod
    def _infer_attn_dtype(model: Any) -> mx.Dtype:
        """Read dtype from the first full-attention layer's projection
        weight. Mirrors :func:`Qwen3Adapter._infer_attn_dtype` but
        skips ``is_linear`` (DeltaNet) layers — those carry SSM-shape
        projections that don't represent the K/V dtype.
        """
        for layer in model.layers:
            if getattr(layer, "is_linear", False):
                continue
            sa = getattr(layer, "self_attn", None)
            if sa is None:
                continue
            for attr in ("k_proj", "q_proj", "v_proj", "o_proj"):
                proj = getattr(sa, attr, None)
                if proj is None:
                    continue
                w = getattr(proj, "weight", None)
                if w is not None and hasattr(w, "dtype"):
                    dtype: mx.Dtype = w.dtype
                    return dtype
        return mx.float16

    @staticmethod
    def _build_attention_pattern(model: Any) -> AttentionPattern:
        """``layer.is_linear`` → ``HYBRID_DELTANET``; else ``GLOBAL`` (D-015)."""
        kinds = tuple(
            AttentionKind.HYBRID_DELTANET
            if getattr(layer, "is_linear", False)
            else AttentionKind.GLOBAL
            for layer in model.layers
        )
        return AttentionPattern(per_layer=kinds)

    @staticmethod
    def _text_config_dict(model: Any) -> dict[str, Any]:
        """Return ``model.args.text_config`` as a dict, or ``{}`` if absent."""
        args = getattr(model, "args", None)
        if args is None:
            return {}
        raw = getattr(args, "text_config", None)
        if isinstance(raw, dict):
            return raw
        return {}


# --- P-3-C5.1 row-slicing helpers (module-level so the methods stay readable) ---


def _detach_row(tensor: mx.array, row_idx: int) -> mx.array:
    """Slice a single row out of a batched tensor and detach it from
    the live cache.

    R-C5-2 requirement: subsequent in-place rebinds at
    ``mlx_lm/models/qwen3_5.py:164/166/197`` must not be observable
    through the snapshot. ``mx.array(slice)`` constructs a new array
    distinct from the slice's underlying view; ``mx.eval`` forces
    materialisation so the snapshot does not hold a deferred-graph
    reference into the live cache.
    """
    if row_idx < 0 or row_idx >= int(tensor.shape[0]):
        raise IndexError(
            f"row_idx {row_idx} out of range for tensor with leading "
            f"dim {int(tensor.shape[0])}"
        )
    sliced = tensor[row_idx : row_idx + 1]
    detached = mx.array(sliced)
    mx.eval(detached)
    return detached


def _splice_row(
    live: mx.array | None, snap: mx.array | None, row_idx: int
) -> mx.array | None:
    """Splice ``snap`` (a ``(1, ...)`` snapshot tensor) into ``live``
    at ``row_idx``, restoring the value-object contract: the
    post-restore live state at ``row_idx`` reflects what the snapshot
    captured at its construction time.

    Cases:

    1. ``snap is None, live is None`` — both empty / lazy. Stays
       ``None``; next forward lazy-allocates as usual.
    2. ``snap is None, live is not None``:
       - ``B == 1`` — the snapshot recorded an unallocated slot for
         the only row. Restore that by wiping the live slot back to
         ``None``. The next forward lazy-allocates fresh state.
       - ``B > 1`` — raises ``ValueError``. The "lazy / unallocated"
         state is per-slot, not per-row, so wiping a single row of a
         populated multi-row slot is not expressible. C5.2 / C5.3
         take snapshots only after a forward has populated the cache,
         so an empty snapshot into a populated multi-row cache is a
         contract violation.
    3. ``snap is not None, live is None`` — only valid for
       ``row_idx == 0`` (no batch dim to index). The snapshot's
       ``(1, ...)`` tensor is assigned directly; the next forward
       sees a populated slot at B=1.
    4. ``snap is not None, live is not None`` — splice the snapshot
       at ``row_idx`` without touching other rows.

    All cases validate ``row_idx`` against the relevant range up
    front (``0`` for live-None / B=1; ``[0, B)`` for B>1).
    """
    if row_idx < 0:
        raise IndexError(f"row_idx must be non-negative, got {row_idx}")

    if snap is None:
        if live is None:
            return None
        B = int(live.shape[0])
        if B == 1:
            if row_idx != 0:
                raise IndexError(
                    f"row_idx {row_idx} invalid for live B=1 tensor "
                    f"(only row_idx=0 is meaningful)"
                )
            return None
        raise ValueError(
            f"cannot restore an empty (None) snapshot slot into a "
            f"populated multi-row cache (B={B}); the lazy / "
            f"unallocated state is per-slot, not per-row, so wiping "
            f"row {row_idx} alone is not expressible. Take and "
            f"restore snapshots on caches that have all participated "
            f"in at least one forward."
        )

    # snap is not None below.
    if live is None:
        if row_idx != 0:
            raise IndexError(
                f"row_idx {row_idx} invalid for live-None target "
                f"(only row_idx=0 is meaningful before lazy allocation)"
            )
        return snap

    B = int(live.shape[0])
    if row_idx >= B:
        raise IndexError(
            f"row_idx {row_idx} out of range for live tensor with "
            f"leading dim {B}"
        )
    if B == 1:
        return snap
    pieces: list[mx.array] = []
    if row_idx > 0:
        pieces.append(live[:row_idx])
    pieces.append(snap)
    if row_idx + 1 < B:
        pieces.append(live[row_idx + 1 :])
    spliced = mx.concatenate(pieces, axis=0)
    mx.eval(spliced)
    return spliced
