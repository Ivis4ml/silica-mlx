"""silica.speculative.engine — I-5 DraftEngine Protocol and the NoopDraftEngine stub.

I-5 (PLAN.md §6) describes draft-token proposal for speculative decoding. The
engine is unaware of the model's internals: it receives a `RequestState` and a
window size `k`, returns up to `k` draft tokens, and is told after verification
how many were accepted.

Principle 9 stub-replacement: `NoopDraftEngine` is the P-0 baseline
(propose -> empty, commit -> no-op); `DraftTargetEngine` landed in
D-021 step 5 sub-unit (a) at v1.7.19 (commit ``58d9fd9``). The
integration point — the decode loop calling `propose` / `commit` —
is fixed from P-0; spec-off remains the byte-equal default via
`NoopDraftEngine`, and spec-on is a constructor-time choice on
`silica.engine.Engine` (no conditional branches in the main loop).

D-021 step 6 sub-unit (β) adds :class:`TargetHiddenConsumer`, an
optional Protocol mixin for **target-conditioned** drafters (DFlash;
future C.3 MTP head, C.6 self-spec) that need the verify forward's
captured hidden states routed to them as a side channel. The
`DraftEngine` Protocol surface is unchanged: `propose(ctx, k)` and
`commit(ctx, accepted_len)` keep their signatures, so C.1's
`DraftTargetEngine` and `NoopDraftEngine` stay Protocol-conformant
unchanged. The engine checks `isinstance(drafter, TargetHiddenConsumer)`
before calling the side-channel methods; C.1 / Noop opt out by
simply not implementing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import mlx.core as mx

from silica.core.request import RequestState


@dataclass
class DraftTokens:
    """A speculative draft: up to `k` token ids + optional per-token log-probs.

    v0.1 stub (`NoopDraftEngine`) emits an empty `token_ids` and `None`
    logprobs. A real draft model (`DraftTargetEngine`, D-021 step 5
    sub-unit (a)) fills both — target verification needs the draft's
    logprobs to compute accept/reject probabilities. `draft_logprobs`,
    when present, has the same length as `token_ids`.
    """

    token_ids: tuple[int, ...]
    draft_logprobs: tuple[float, ...] | None = None


@runtime_checkable
class DraftEngine(Protocol):
    """Draft provider for speculative decoding.

    `propose(ctx, k)` — the scheduler calls this between decode steps when
    speculative decoding is enabled. Returns up to `k` draft tokens (fewer is
    allowed; zero is legal and signals "no proposal this step").

    `commit(ctx, accepted_len)` — called by the engine after target
    verification tells the draft how many of its proposed tokens survived.
    Implementations typically use this to advance internal state or evict
    unused draft cache. Rolling back the draft's own state on rejection is
    the draft engine's responsibility, not the scheduler's.
    """

    def propose(self, ctx: RequestState, k: int) -> DraftTokens: ...

    def commit(self, ctx: RequestState, accepted_len: int) -> None: ...


class NoopDraftEngine:
    """Draft disabled. `propose` returns empty, `commit` is a no-op.

    Installed as the default `DraftEngine` from P-0 so the decode loop can
    call `propose` / `commit` unconditionally — speculative decoding is
    toggled by swapping this for `DraftTargetEngine` (D-021 step 5
    sub-unit (a)), not by adding conditional branches in the engine
    main loop.
    """

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        return DraftTokens(token_ids=())

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        return None


@runtime_checkable
class TargetHiddenConsumer(Protocol):
    """Optional side-channel mixin for target-conditioned drafters.

    Implemented by drafters whose ``propose`` requires the *target*
    model's hidden states at specific layer indices as a conditioning
    input — the DFlash block-diffusion drafter is the load-bearing
    case (D-021 step 6); future C.3 MTP head and C.6 self-spec
    variants share the surface. The contract:

    - **`prime(req_id, captured_dict)`** is called once after
      ``adapter.prefill_with_capture(...)`` populates the captured
      dict, before the first ``propose``. Aggregates the per-layer
      slices the drafter needs and stores them as the cycle-1
      ``target_hidden`` for ``req_id``.
    - **`update_target_hidden(req_id, captured_dict, yielded_count)``**
      is called after each verify forward (downstream of
      ``decode_step_multi_with_capture``). The wrapper extracts the
      same per-layer slices from the new dict, slices to
      ``1 + yielded_count`` positions to match the engine's commit,
      and stores the result for the next cycle's ``propose``.
    - **`free_target_hidden(req_id)`** drops the per-request state
      when the engine frees a request.

    The captured dict the engine forwards has the convention
    ``key 0`` = embedding output, ``key i + 1`` = output of
    ``model.layers[i]`` (matching
    ``dflash_mlx.runtime.target_forward_with_hidden_states``). The
    wrapper internally maps the drafter's ``target_layer_ids`` (a
    checkpoint-fixed list) onto the dict via the +1 offset and
    concatenates along the last axis to produce the
    ``(1, ctx_len, |L| * hidden_size)`` array
    ``DFlashDraftModel.__call__`` consumes.

    Adapters / drafters that do not need this side channel simply
    skip implementing the Protocol — ``isinstance(drafter,
    TargetHiddenConsumer)`` returns False and the engine bypasses
    the calls entirely. C.1 ``DraftTargetEngine`` and
    ``NoopDraftEngine`` are deliberately not target-conditioned, so
    they do not implement this Protocol.
    """

    @property
    def capture_layer_ids(self) -> frozenset[int]:
        """Adapter-side capture-set the engine ε wiring should request.

        Returns the dict-key set the engine forwards to
        ``adapter.prefill_with_capture(...)`` and
        ``adapter.decode_step_multi_with_capture(...)``. The convention
        is the +1 offset between the upstream drafter's
        ``target_layer_ids`` (``i`` = layer ``i-1`` output, with 0 =
        embedding) and silica's adapter-side capture-dict keys
        (``key 0`` = embedding output, ``key i + 1`` = output of
        ``model.layers[i]``). Concretely:
        ``frozenset(i + 1 for i in self._target_layer_ids)`` for
        DFlash; future C.3 / C.6 wrappers compute their own offset
        from their architecture.

        Read-only after construction — the drafter's required layer
        set is fixed by its checkpoint.
        """
        ...

    def prime(
        self, req_id: str, captured_dict: dict[int, mx.array]
    ) -> None:
        """Seed cycle-1 ``target_hidden`` for ``req_id``."""
        ...

    def update_target_hidden(
        self,
        req_id: str,
        captured_dict: dict[int, mx.array],
        yielded_count: int,
    ) -> None:
        """Advance cycle-N ``target_hidden`` for ``req_id``."""
        ...

    def free_target_hidden(self, req_id: str) -> None:
        """Drop per-request state when the engine frees the request."""
        ...
