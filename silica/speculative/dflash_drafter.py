"""silica.speculative.dflash_drafter — D-021 step 6 (β) skeleton.

DFlash block-diffusion drafter wrapper. Implements
:class:`silica.speculative.engine.DraftEngine` and the optional
:class:`silica.speculative.engine.TargetHiddenConsumer` side-channel
mixin. The (β) skeleton lands the per-``req_id`` state shape and the
prime / update / free side-channel implementation; ``propose`` and
``commit`` raise ``NotImplementedError`` until sub-unit (γ) (synthetic
block-diffusion drafter for the cycle-1 byte-exact correctness gate)
and (δ) (real ``dflash_mlx.model.DFlashDraftModel`` forward).

Design notes:

- **Opt-in dependency.** ``dflash-mlx`` is shipped under the
  ``silica[dflash]`` extras marker. ``DFlashDrafter.__init__`` imports
  ``dflash_mlx`` lazily and raises a clear ``ImportError`` at
  construction time if the package is not installed; importing this
  module without instantiating the class does not pull ``dflash_mlx``
  into the runtime, so ``silica`` users who do not run
  ``--speculative dflash`` pay zero dep cost.
- **target_hidden shape.** Per-``req_id`` stored ``target_hidden`` has
  shape ``(1, ctx_len, |L| * hidden_size)`` where
  ``L = drafter.target_layer_ids`` is the (checkpoint-fixed) list of
  layer indices the upstream drafter consumes. The aggregation matches
  ``dflash_mlx.runtime.extract_context_feature_from_dict``:
  ``mx.concatenate([captured_dict[i + 1] for i in target_layer_ids],
  axis=-1)``. The "+1" maps from the upstream
  ``DFlashDraftModelArgs.target_layer_ids`` indexing
  (0 = embedding layer, i = output of layer i-1) onto silica's
  capture-dict convention (key 0 = embedding output, key i+1 = output
  of ``model.layers[i]``).
- **draft_caches.** Per-layer ``ContextOnlyDraftKVCache`` instances
  (one per drafter layer), with ``sink_size`` and ``window_size``
  read from the upstream ``runtime`` defaults. Mutated *inside*
  ``DFlashDraftModel.__call__`` via ``append_context``; (β) does not
  touch them post-construction.
- **TargetHiddenConsumer side channel.** ``DFlashDrafter`` exposes
  ``prime``, ``update_target_hidden``, ``free_target_hidden``. The
  silica engine integration (sub-unit (ε)) routes
  ``decode_step_multi_with_capture``'s captured dict through
  ``update_target_hidden`` after every verify forward. The
  ``DraftEngine.commit`` Protocol method stays a no-op: silica's
  draft-side rollback is implicit (rejected drafts were never
  written to the draft cache; the next ``propose`` re-runs the
  block-diffusion forward with the trimmed ``target_hidden``).

Sub-unit ordering this skeleton enables:

  - (γ) Synthetic drafter for cycle-1 byte-exact parity. ``propose``
    becomes a deterministic K-token block emitter; ``DFlashDrafter``
    accepts a ``synthetic_emit`` callable so the parity test can pin
    the engine wiring without depending on a real drafter checkpoint.
  - (δ) Real DFlash forward. ``propose`` calls
    ``DFlashDraftModel.__call__(noise_embedding=…, target_hidden=…,
    cache=draft_caches)``; the drafter checkpoint loaded via
    ``dflash_mlx.runtime.load_draft_bundle`` lives on
    ``self._drafter_model``.
  - (ε) Engine integration. ``silica.engine.Engine`` checks
    ``isinstance(drafter, TargetHiddenConsumer)`` after each verify
    and routes ``captured_dict`` + ``yielded_count``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import mlx.core as mx

from silica.core.request import RequestState
from silica.models.hidden_capture import HiddenCaptureAdapter
from silica.speculative._dflash_target_ops import (
    lm_head_logits,
    target_embed_tokens,
)
from silica.speculative.engine import DraftTokens

# Sub-unit (γ): the synthetic emitter consumed by ``propose`` in
# spec-off oracle replay tests. ``(target_hidden, k)`` -> token ids;
# returned tuple length must be ≤ ``k``. The (γ) tests pre-record
# spec-off cycle-1 tokens via ``Engine.generate(--speculative none)``
# and supply a closure that returns those tokens for the same prefix
# the verifier sees, so ``greedy_verify`` accepts the full block.
SyntheticEmit = Callable[[mx.array, int], tuple[int, ...]]


class DFlashDrafter:
    """DFlash block-diffusion drafter (D-021 step 6 (β) skeleton).

    Implements :class:`silica.speculative.engine.DraftEngine` (Protocol
    methods ``propose`` and ``commit``) and
    :class:`silica.speculative.engine.TargetHiddenConsumer` (side-channel
    methods ``prime``, ``update_target_hidden``, ``free_target_hidden``).

    Per-``req_id`` state:

    - ``self._target_hidden[req_id]`` — `(1, ctx_len, |L| *
      hidden_size)` mx.array, the drafter's conditioning input.
    - ``self._draft_caches[req_id]`` — list of
      ``ContextOnlyDraftKVCache`` (one per drafter layer).

    The (β) skeleton ``propose`` / ``commit`` raise ``NotImplementedError``
    until the (γ) synthetic and (δ) real-forward implementations land.
    """

    def __init__(
        self,
        drafter_repo: str,
        target_adapter: HiddenCaptureAdapter,
    ) -> None:
        """Construct a drafter against ``drafter_repo`` paired to ``target_adapter``.

        Args:
            drafter_repo: HF model id (e.g. ``z-lab/Qwen3.5-27B-DFlash``).
                The constructor calls ``dflash_mlx.runtime.load_draft_bundle``
                to load the drafter weights, then reads the
                ``target_layer_ids`` field from the loaded
                ``DFlashDraftModel`` instance (set by
                ``DFlashDraftModel.__init__`` from
                ``args.dflash_config["target_layer_ids"]`` or
                ``build_target_layer_ids(...)``) to fix the capture set
                at construction.
            target_adapter: an adapter implementing
                :class:`silica.models.hidden_capture.HiddenCaptureAdapter`
                (Qwen3.5 dense or MoE in (αβ.1) / (αβ.2)). The wrapper
                does not call into the adapter directly — sub-unit (ε)'s
                engine integration routes captured dicts to the wrapper.

        Raises:
            ImportError: if ``dflash-mlx`` is not installed. Install via
                ``pip install 'silica-mlx[dflash]'`` to opt in.
        """
        try:
            from dflash_mlx.runtime import (  # type: ignore[import-not-found]
                load_draft_bundle,
            )
        except ImportError as e:
            raise ImportError(
                "DFlashDrafter requires the 'dflash-mlx' package. "
                "Install via `pip install 'silica-mlx[dflash]'` to opt "
                "in to the C.4 DFlash spec path. "
                "(D-021 step 6 sub-unit (β) keeps dflash-mlx as an "
                "opt-in extras dep so default silica installs stay slim.)"
            ) from e

        self._drafter_repo = drafter_repo
        self._target_adapter = target_adapter
        # Per upstream verify (commit `cee5f68` reading of
        # `dflash_mlx.runtime.load_draft_bundle`) the bundle is
        # ``(model, meta_dict)`` where ``meta_dict`` carries
        # ``resolved_model_ref`` / ``config`` / ``quantize_draft``.
        bundle = load_draft_bundle(drafter_repo)
        if not (isinstance(bundle, tuple) and len(bundle) == 2):
            raise RuntimeError(
                "DFlashDrafter expected dflash_mlx.runtime.load_draft_bundle "
                f"to return (model, meta) tuple; got {type(bundle).__name__}. "
                "Upstream API may have changed — pin dflash-mlx at the "
                "version recorded in pyproject.toml's [dflash] extras."
            )
        self._drafter_model: Any = bundle[0]
        self._drafter_meta: Any = bundle[1]
        # ``target_layer_ids`` lives on the model instance, not on args
        # — upstream sets it in ``DFlashDraftModel.__init__`` from
        # ``args.dflash_config["target_layer_ids"]`` with a fallback.
        # Materialise to a tuple so it cannot be mutated post-init.
        self._target_layer_ids: tuple[int, ...] = self._read_target_layer_ids()

        # Per-req_id state; populated by ``prime`` and updated by
        # ``update_target_hidden``.
        self._target_hidden: dict[str, mx.array] = {}
        self._draft_caches: dict[str, list[Any]] = {}

        # Sub-unit (γ) synthetic seam — None on the real path; populated
        # by ``DFlashDrafter.for_synthetic`` for the cycle-1 oracle-replay
        # parity test.
        self._synthetic_emit: SyntheticEmit | None = None

        # Sub-unit (δ.1): per-cycle draft KV cache factory. The real
        # production path imports ``ContextOnlyDraftKVCache`` from
        # ``dflash_mlx.model`` (already loaded above via the deferred
        # import) and binds the upstream sink/window defaults that
        # ``runtime._resolve_draft_window`` resolves from env vars.
        # Tests with a fake drafter set ``self._cache_factory = None``
        # before ``prime`` runs — see ``_build_draft_caches`` for the
        # placeholder fallback.
        from dflash_mlx.model import (  # type: ignore[import-not-found]
            ContextOnlyDraftKVCache,
        )

        sink = int(os.environ.get("DFLASH_DRAFT_SINK_SIZE", "64"))
        window = int(os.environ.get("DFLASH_DRAFT_WINDOW_SIZE", "1024"))

        def _make_real_cache() -> Any:
            return ContextOnlyDraftKVCache(
                sink_size=sink, window_size=window
            )

        self._cache_factory: Any = _make_real_cache

    @classmethod
    def for_synthetic(
        cls,
        *,
        target_layer_ids: tuple[int, ...],
        synthetic_emit: SyntheticEmit,
    ) -> DFlashDrafter:
        """Construct a synthetic-mode drafter for sub-unit (γ) tests.

        Bypasses the real ``__init__`` (and therefore the
        ``dflash-mlx`` import + ``load_draft_bundle`` HF download) so
        the synthetic correctness gate runs against silica's own
        ``DFlashDrafter`` surface — the same wrapper the (δ) real
        forward will use — without depending on a checkpoint or the
        opt-in extras dep. The returned drafter:

        - Implements both ``DraftEngine`` and ``TargetHiddenConsumer``
          (same ``isinstance`` profile as a real-mode instance).
        - Has the same ``capture_layer_ids`` / ``prime`` /
          ``update_target_hidden`` / ``free_target_hidden`` shape
          contract as the real path.
        - Routes ``propose(ctx, k)`` to the supplied
          ``synthetic_emit(target_hidden, k)`` callable.

        Args:
            target_layer_ids: the (would-be-checkpoint-fixed) list of
                layer indices the drafter consumes. Drives both
                ``capture_layer_ids = {i + 1 for i in target_layer_ids}``
                and the per-layer concat in ``prime`` /
                ``update_target_hidden``.
            synthetic_emit: callable invoked by ``propose`` in synthetic
                mode. Signature ``(target_hidden, k) -> tuple[int, ...]``
                with returned length **≤ k** (overflow loud-fails).

        Not for production use. (ε) engine integration explicitly
        rejects synthetic-mode drafters in non-test contexts.
        """
        if not target_layer_ids:
            raise ValueError(
                "for_synthetic: target_layer_ids must be non-empty "
                "(matches the real-path loud-fail in _read_target_layer_ids)."
            )
        drafter = object.__new__(cls)
        drafter._drafter_repo = "<synthetic>"
        drafter._target_adapter = None  # type: ignore[assignment]
        drafter._drafter_model = None
        drafter._drafter_meta = None
        drafter._target_layer_ids = tuple(int(i) for i in target_layer_ids)
        drafter._target_hidden = {}
        drafter._draft_caches = {}
        drafter._synthetic_emit = synthetic_emit
        # No real drafter model loaded; the cache factory has nothing
        # to wire. ``_build_draft_caches`` returns ``[]`` in this mode.
        drafter._cache_factory = None
        return drafter

    # --- DraftEngine Protocol ----------------------------------------------

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        """Block-diffusion draft of up to ``k`` tokens.

        Two modes:

        - **Synthetic** (sub-unit (γ)): if the drafter was constructed
          via :meth:`for_synthetic`, ``propose`` reads the per-``req_id``
          stored ``target_hidden`` and forwards
          ``synthetic_emit(target_hidden, k)`` — the (γ) tests use this
          to replay a spec-off-recorded oracle so ``greedy_verify``
          accepts the full block at the verifier level.
        - **Real** (sub-unit (δ), pending): the synthetic emitter is
          ``None`` and ``propose`` raises ``NotImplementedError``.

        Synthetic-mode invariants:

        - ``ctx.request_id`` must already be primed via
          :meth:`prime` (the engine ε wiring will guarantee this; in
          unit tests the caller primes before the first ``propose``).
        - ``synthetic_emit`` must return ``len(token_ids) <= k``.
          Overflow is loud-fail rather than truncating, since silently
          accepting an over-length block would mask wiring bugs that
          (δ)'s real forward will not tolerate.
        """
        req_id = ctx.request_id
        if req_id not in self._target_hidden:
            raise KeyError(
                f"DFlashDrafter.propose: req_id {req_id!r} not primed. "
                "Engine ε must call prime() after prefill_with_capture "
                "before the first propose; unit tests must prime "
                "explicitly."
            )
        target_hidden = self._target_hidden[req_id]

        emit = self._synthetic_emit
        if emit is not None:
            token_ids = emit(target_hidden, int(k))
            token_tuple = tuple(int(t) for t in token_ids)
            if len(token_tuple) > k:
                raise ValueError(
                    f"DFlashDrafter.propose: synthetic_emit returned "
                    f"{len(token_tuple)} tokens for k={k}; expected at "
                    "most k. Loud-fail rather than silently truncate so "
                    "synthetic-emitter wiring bugs surface immediately."
                )
            return DraftTokens(token_ids=token_tuple)

        # Real-mode propose. (δ.1)
        return self._propose_real(ctx, req_id, target_hidden, int(k))

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        """No-op per the F-1 state machine (§4.1 of the OPENING).

        Draft-side "rollback" is implicit — rejected drafts' noise
        keys/values were never appended to ``draft_caches``, so
        slicing the new ``target_hidden`` to the committed length
        (which ``update_target_hidden`` does) is the only update
        needed. The engine still calls this method to keep the
        Protocol surface uniform across drafters.
        """
        return None

    # --- Real-mode propose internals (δ.1) --------------------------------

    def _propose_real(
        self,
        ctx: RequestState,
        req_id: str,
        target_hidden: mx.array,
        k: int,
    ) -> DraftTokens:
        """Run a real DFlash block-diffusion forward and return up to
        ``k`` drafted tokens.

        Mirrors the cycle body of upstream
        ``dflash_mlx.runtime.generate_dflash_once``:

        1. Pull the staged-first anchor from ``ctx.output_token_ids[-1]``
           — the engine just yielded that token at the end of the
           previous cycle (or as the prefill argmax on cycle 1). It is
           the input position-0 of the verify forward this cycle, and
           the input position-0 of the drafter forward.
        2. Build a length-``block_len`` token buffer where position 0
           is the staged-first and positions ``1..block_len-1`` are
           ``mask_token_id``. ``block_len = min(k + 1,
           drafter.block_size)`` — the +1 reserves position 0 for the
           anchor; positions 1..block_len-1 hold the drafted slots.
           Returned drafts thus number ``block_len - 1 ≤ k``.
        3. Embed via the target's input embedding, then call
           ``DFlashDraftModel(noise_embedding=..., target_hidden=...,
           cache=draft_caches)``. The drafter forward appends the
           target-hidden-derived context to the per-layer caches
           internally; positions 1..block_len-1 of the returned hidden
           are the drafter's per-position output.
        4. Project drafter hiddens at positions ``[1:block_len]``
           through the target's lm-head to logits, then greedy argmax
           to materialise drafted token ids.

        Loud-fails if ``ctx.output_token_ids`` is empty — that
        indicates the engine called ``propose`` before the prefill's
        argmax was yielded, a wiring contract violation.
        """
        if not ctx.output_token_ids:
            raise RuntimeError(
                "DFlashDrafter._propose_real: ctx.output_token_ids is "
                "empty; the engine must have yielded at least the "
                "prefill argmax (the staged-first anchor) before "
                "calling propose. Engine ε's _drive guarantees this."
            )
        staged_first = int(ctx.output_token_ids[-1])

        block_size = self._read_block_size()
        block_len = min(k + 1, block_size)
        if block_len < 2:
            # k=0 or block_size=1 — no draft slots, return empty.
            return DraftTokens(token_ids=())

        mask_token_id = self._read_mask_token_id()

        # Build (block_len,) buffer: [staged_first, mask, mask, ...].
        # ``mx.array`` does not support direct item assignment; build
        # via concatenation.
        anchor_arr = mx.array([staged_first], dtype=mx.uint32)
        mask_arr = mx.full(
            shape=(block_len - 1,),
            vals=int(mask_token_id),
            dtype=mx.uint32,
        )
        block_token_buffer = mx.concatenate([anchor_arr, mask_arr])

        # Embed via the target-side helper. Output: (1, block_len, hidden).
        noise_embedding = target_embed_tokens(
            self._target_adapter, block_token_buffer[None]
        )

        # Drafter forward — same call shape as upstream
        # ``DFlashDraftModel.__call__``. ``draft_hidden`` shape:
        # (1, block_len, drafter_hidden_size).
        draft_caches = self._draft_caches[req_id]
        draft_hidden = self._drafter_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_caches,
        )

        # Project positions 1..block_len-1 through target lm-head.
        # Position 0 is the anchor's prediction (= staged-first's own
        # next-token), but the engine already has the verify forward
        # for that — the drafter only contributes the drafted
        # positions.
        drafted_hidden = draft_hidden[:, 1:, :]
        logits = lm_head_logits(self._target_adapter, drafted_hidden)
        # Greedy argmax → draft tokens. ``mx.argmax`` returns shape
        # ``(1, block_len-1)``; ``.tolist()`` yields a nested list.
        drafted = mx.argmax(logits, axis=-1)
        py_drafts: Any = drafted[0].tolist()
        token_list: list[int] = [int(t) for t in py_drafts]
        return DraftTokens(token_ids=tuple(token_list))

    def _read_block_size(self) -> int:
        """Read ``DFlashDraftModel.block_size`` from the loaded model.

        Set by upstream ``DFlashDraftModel.__init__`` from
        ``args.dflash_config["block_size"]`` with a fallback. Loud-fail
        on missing attribute (mirrors ``_read_target_layer_ids``)."""
        block_size = getattr(self._drafter_model, "block_size", None)
        if block_size is None:
            raise RuntimeError(
                "DFlashDrafter could not read 'block_size' from the "
                f"loaded drafter model "
                f"{type(self._drafter_model).__name__}. Upstream sets "
                "this attribute on DFlashDraftModel; check that "
                "load_draft_bundle returned a DFlashDraftModel "
                "instance and that the dflash-mlx version pinned in "
                "pyproject.toml's [dflash] extras matches the runtime."
            )
        return int(block_size)

    def _read_mask_token_id(self) -> int:
        """Read ``DFlashDraftModel.mask_token_id`` from the loaded model.

        Used to fill the non-anchor slots of the noise-embedding input
        buffer. Loud-fail on missing attribute."""
        mask = getattr(self._drafter_model, "mask_token_id", None)
        if mask is None:
            raise RuntimeError(
                "DFlashDrafter could not read 'mask_token_id' from the "
                f"loaded drafter model "
                f"{type(self._drafter_model).__name__}. Upstream sets "
                "this attribute on DFlashDraftModel."
            )
        return int(mask)

    # --- TargetHiddenConsumer side channel ---------------------------------

    @property
    def capture_layer_ids(self) -> frozenset[int]:
        """Adapter-side capture set the engine should request.

        Returns ``frozenset(i + 1 for i in target_layer_ids)`` —
        the ``+1`` offset matches upstream
        ``dflash_mlx.runtime.target_forward_with_hidden_states``
        (``capture_layer_ids = {int(layer_id) + 1 for layer_id in
        draft_model.target_layer_ids}``) and silica's adapter-side
        convention (key 0 = embedding output, key ``i + 1`` = output
        of ``model.layers[i]``). The (ε) engine wiring forwards this
        directly to ``decode_step_multi_with_capture(...,
        capture_layer_ids)`` and ``prefill_with_capture(...,
        capture_layer_ids)`` so the layer set is sourced from the
        drafter checkpoint, not configured by the caller.
        """
        return frozenset(i + 1 for i in self._target_layer_ids)

    def prime(
        self, req_id: str, captured_dict: dict[int, mx.array]
    ) -> None:
        """Seed cycle-1 ``target_hidden`` for ``req_id`` from prefill capture.

        Aggregates per-layer slices at ``[i + 1 for i in
        self._target_layer_ids]`` from the captured dict, concatenates
        along ``axis=-1``, stores the result. Also initialises an
        empty ``draft_caches`` list for the request — populated lazily
        on the first ``propose`` (where ``DFlashDraftModel.__call__``
        appends context internally).
        """
        full = self._aggregate_target_hidden(captured_dict)
        self._target_hidden[req_id] = full
        self._draft_caches[req_id] = self._build_draft_caches()

    def update_target_hidden(
        self,
        req_id: str,
        captured_dict: dict[int, mx.array],
        yielded_count: int,
    ) -> None:
        """Advance cycle-N ``target_hidden`` after a verify forward.

        Engine wiring (sub-unit (ε)) calls this after each verify
        forward, with the captured dict from
        ``decode_step_multi_with_capture`` and the engine's
        ``yielded_count`` (= committed tokens this cycle). Aggregates
        the same way as ``prime``, then slices to ``1 + yielded_count``
        positions on ``axis=1`` so the next cycle's drafter sees only
        the committed prefix.
        """
        if req_id not in self._target_hidden:
            raise KeyError(
                f"DFlashDrafter.update_target_hidden: req_id {req_id!r} "
                "not primed. Engine must call prime() after "
                "prefill_with_capture before the first verify cycle."
            )
        full = self._aggregate_target_hidden(captured_dict)
        # Slice to the committed length on axis=1 (the ctx_len axis).
        commit_len = 1 + int(yielded_count)
        if commit_len > full.shape[1]:
            raise ValueError(
                f"yielded_count={yielded_count} exceeds capture window "
                f"(full.shape[1]={full.shape[1]}); engine wiring bug."
            )
        self._target_hidden[req_id] = full[:, :commit_len, :]

    def free_target_hidden(self, req_id: str) -> None:
        """Drop per-request state when the engine frees the request."""
        self._target_hidden.pop(req_id, None)
        self._draft_caches.pop(req_id, None)

    # --- Internal helpers --------------------------------------------------

    def _aggregate_target_hidden(
        self, captured_dict: dict[int, mx.array]
    ) -> mx.array:
        """Mirror upstream's ``extract_context_feature_from_dict``.

        Selects per-layer slices at ``[i + 1 for i in target_layer_ids]``
        from the captured dict, asserts each has the same
        ``(B, ctx_len, hidden_size)`` shape, and concatenates along
        ``axis=-1`` to produce ``(1, ctx_len, |L| * hidden_size)``.
        Raises a descriptive ``KeyError`` when the engine forwards a
        dict that is missing any of the requested layer ids — which
        means either the adapter's ``capture_layer_ids`` set was not
        derived from this drafter, or a downstream caller passed the
        wrong dict.
        """
        try:
            slices = [
                captured_dict[i + 1] for i in self._target_layer_ids
            ]
        except KeyError as e:
            requested = sorted(int(i) + 1 for i in self._target_layer_ids)
            present = sorted(int(k) for k in captured_dict.keys())
            raise KeyError(
                "DFlashDrafter._aggregate_target_hidden: captured_dict "
                f"missing layer id {e!r}. Drafter's target_layer_ids "
                f"(plus 1 offset) are {requested}; dict keys are "
                f"{present}. Ensure the adapter's "
                "capture_layer_ids set is "
                "frozenset(i + 1 for i in drafter.target_layer_ids)."
            ) from e
        return mx.concatenate(slices, axis=-1)

    def _read_target_layer_ids(self) -> tuple[int, ...]:
        """Pull ``target_layer_ids`` from the loaded drafter model.

        Upstream stores the field on the **model instance** in
        ``DFlashDraftModel.__init__``:

            target_layer_ids = list(
                (args.dflash_config or {}).get("target_layer_ids") or ()
            )
            self.target_layer_ids = target_layer_ids or build_target_layer_ids(
                args.num_target_layers, args.num_hidden_layers
            )

        — i.e. the field lives on the model, not on ``args``, and is
        always non-empty by upstream construction (``build_target_layer_ids``
        synthesises a list when no override is configured). Loud-fail
        if either invariant is violated, since silently returning an
        empty tuple here would surface as a confusing
        ``mx.concatenate([], axis=-1)`` failure deep inside ``prime``.
        """
        ids = getattr(self._drafter_model, "target_layer_ids", None)
        if ids is None:
            raise RuntimeError(
                "DFlashDrafter could not read 'target_layer_ids' from the "
                f"loaded drafter model {type(self._drafter_model).__name__}. "
                "Upstream DFlashDraftModel sets this attribute in __init__; "
                "check that load_draft_bundle returned a DFlashDraftModel "
                "instance and that the dflash-mlx version pinned in "
                "pyproject.toml's [dflash] extras matches the runtime."
            )
        try:
            tup = tuple(int(i) for i in ids)
        except TypeError as e:
            raise RuntimeError(
                "DFlashDrafter: 'target_layer_ids' on the drafter model "
                f"is not iterable (got {ids!r})."
            ) from e
        if not tup:
            raise RuntimeError(
                "DFlashDrafter: drafter model's 'target_layer_ids' is "
                "empty. Upstream's build_target_layer_ids should always "
                "produce a non-empty list; an empty list here means "
                "the checkpoint config or fallback path is broken."
            )
        return tup

    def _build_draft_caches(self) -> list[Any]:
        """Construct the per-layer draft cache list for one ``req_id``.

        Three modes:

        - **Synthetic (no fake drafter model)** — returns ``[]``. The
          synthetic emitter ignores the cache list, so an empty list
          is sufficient and avoids importing ``dflash_mlx`` indirectly.
        - **Fake drafter for δ.1 tests** — ``self._drafter_model`` is a
          test fake whose ``.layers`` attribute drives the cache-list
          length, but no ``_cache_factory`` is wired. Returns a list
          of ``None`` placeholders of the right length, which the fake
          drafter's ``__call__`` ignores.
        - **Real production (η)** — ``self._cache_factory`` is set in
          ``__init__`` to construct a fresh
          ``dflash_mlx.model.ContextOnlyDraftKVCache`` per drafter
          layer.
        """
        drafter = self._drafter_model
        if drafter is None or not hasattr(drafter, "layers"):
            return []
        factory = getattr(self, "_cache_factory", None)
        if factory is None:
            return [None] * len(drafter.layers)
        return [factory() for _ in drafter.layers]


__all__ = ["DFlashDrafter"]
