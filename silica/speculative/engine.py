"""silica.speculative.engine — I-5 DraftEngine Protocol and the NoopDraftEngine stub.

I-5 (PLAN.md §6) describes draft-token proposal for speculative decoding. The
engine is unaware of the model's internals: it receives a `RequestState` and a
window size `k`, returns up to `k` draft tokens, and is told after verification
how many were accepted.

Principle 9 stub-replacement: `NoopDraftEngine` is the P-0 baseline
(propose -> empty, commit -> no-op); `DraftTargetEngine` lands in P-7. The
integration point — the decode loop calling `propose` / `commit` — is fixed
from P-0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from silica.core.request import RequestState


@dataclass
class DraftTokens:
    """A speculative draft: up to `k` token ids + optional per-token log-probs.

    v0.1 stub (`NoopDraftEngine`) emits an empty `token_ids` and `None`
    logprobs. A real draft model (P-7 `DraftTargetEngine`) fills both — target
    verification needs the draft's logprobs to compute accept/reject
    probabilities. `draft_logprobs`, when present, has the same length as
    `token_ids`.
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
    toggled by swapping this for `DraftTargetEngine` (P-7), not by adding
    conditional branches in the engine main loop.
    """

    def propose(self, ctx: RequestState, k: int) -> DraftTokens:
        return DraftTokens(token_ids=())

    def commit(self, ctx: RequestState, accepted_len: int) -> None:
        return None
