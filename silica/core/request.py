"""Request and request lifecycle types for Silica engine.

``RequestState`` is a **pure FSM** (P-2 design, see ``docs/P2_OPENING.md``):
``transition`` validates the target against an allow-list and mutates
``status`` + ``_history`` + (on terminal) ``finish_reason``. It does **not**
release KV blocks, pin prefix-cache sources, zero batch rows, or touch any
external resource. Side-effects are driven by observers —
``ContinuousBatcher`` and ``PagedKVCache`` — reacting to the new status.

This separation is the design choice that makes P-7 speculative rollback
tractable: reversing a tentative transition needs only to flip ``status``
back; no resource churn has to be un-done. Coupling side-effects to the
state machine would turn every speculative reject into a distributed
transaction.

States (PLAN.md §7 P-2):

- ``WAITING`` — queued; not yet admitted (or re-queued after preemption).
- ``PREFILL`` — admitted; adapter is running prefill.
- ``DECODE`` — prefill complete; adapter is driving decode_step.
- ``PREEMPTED`` — admitted request evicted to honour memory budget; holds
  a ``state_delta_snapshot`` so re-admission can resume without losing
  recurrent state (D-015).
- ``DONE`` — completed normally (hit stop token or ``max_tokens``). Terminal.
- ``ABORTED`` — terminated by error / unrecoverable OOM / user cancel.
  Terminal.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from silica.core.sampling import SamplingParams


class InvalidTransition(ValueError):
    """Raised when ``RequestState.transition`` is given an illegal target."""


class RequestStatus(enum.Enum):
    """Request lifecycle states."""

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    PREEMPTED = "preempted"
    DONE = "done"
    ABORTED = "aborted"


# Allow-list for RequestState.transition. Any move not in this table raises
# InvalidTransition. Notable non-obvious edges:
#   - PREFILL → DONE is legal: first sampled token may be a stop token, in
#     which case the request terminates before entering DECODE (matches
#     P-1 Engine's yield-then-check-stop pattern).
#   - PREEMPTED → WAITING is the re-admission edge (after an earlier
#     preemption under budget pressure). PREEMPTED → ABORTED is the path
#     when re-admission cannot be satisfied.
#   - DONE and ABORTED are terminal — no outgoing transitions.
_ALLOWED_TRANSITIONS: dict[RequestStatus, frozenset[RequestStatus]] = {
    RequestStatus.WAITING: frozenset(
        {RequestStatus.PREFILL, RequestStatus.ABORTED}
    ),
    RequestStatus.PREFILL: frozenset(
        {
            RequestStatus.DECODE,
            RequestStatus.PREEMPTED,
            RequestStatus.DONE,
            RequestStatus.ABORTED,
        }
    ),
    RequestStatus.DECODE: frozenset(
        {
            RequestStatus.DONE,
            RequestStatus.PREEMPTED,
            RequestStatus.ABORTED,
        }
    ),
    RequestStatus.PREEMPTED: frozenset(
        {RequestStatus.WAITING, RequestStatus.ABORTED}
    ),
    RequestStatus.DONE: frozenset(),
    RequestStatus.ABORTED: frozenset(),
}

_TERMINAL: frozenset[RequestStatus] = frozenset(
    {RequestStatus.DONE, RequestStatus.ABORTED}
)


@dataclass(frozen=True)
class Request:
    """Immutable request input submitted to the engine.

    Attributes:
        prompt: Raw text prompt.
        sampling_params: Sampling configuration.
        request_id: Unique identifier (auto-generated if not provided).
        token_ids: Pre-tokenized input (populated by engine after
            tokenization; empty tuple for not-yet-tokenized requests).
    """

    prompt: str
    sampling_params: SamplingParams
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    token_ids: tuple[int, ...] = ()


@dataclass
class RequestState:
    """Mutable per-request state, owned by the scheduler.

    The ``transition`` method is the only legal way to change ``status``.
    Direct assignment is not blocked (keeping P-2 overhead minimal; can be
    tightened with a property setter later if drift becomes a problem).
    """

    request: Request
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    num_output_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None

    # Added for P-2 (P2_OPENING v2.1).
    prefix_hit_tokens: int = 0
    """Number of prompt tokens covered by a RadixPrefixCache hit on admit;
    used by the batcher to skip that portion of the prefill forward."""

    state_delta_snapshot: Any | None = None
    """Adapter-owned opaque payload — expected to be a
    ``silica.models.adapter.StateDelta`` (D-015) but typed ``Any`` here to
    avoid ``silica.core`` depending on ``silica.models``. Retained across
    PREEMPTED so re-admission can resume without losing recurrent state."""

    finish_reason: str | None = None
    """Set when entering a terminal state (DONE or ABORTED). Mirrors the
    ``reason`` argument of the terminal transition, for use in
    ``BatchEvent.finish_reason``."""

    _history: list[tuple[RequestStatus, str]] = field(default_factory=list)

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL

    @property
    def is_finished(self) -> bool:
        """Alias for ``is_terminal`` — preserves the pre-P-2 API."""
        return self.is_terminal

    @property
    def history(self) -> list[tuple[RequestStatus, str]]:
        """Chronological log of all transitions this request has undergone."""
        return list(self._history)

    def transition(
        self, to: RequestStatus, *, reason: str
    ) -> RequestStatus:
        """Validate and perform a status transition.

        Pure FSM — updates ``status``, appends to ``_history``, and sets
        ``finish_reason`` when entering a terminal state. Does not
        release KV blocks, pin prefix-cache sources, or mutate any
        external resource; observers act on status changes elsewhere.

        Args:
            to: Target status. Must be in the allow-list for the current
                status.
            reason: Short string recorded in ``_history`` and, for
                terminal transitions, in ``finish_reason``.

        Returns:
            The previous status (before the transition).

        Raises:
            InvalidTransition: if ``to`` is not in the allow-list.
        """
        allowed = _ALLOWED_TRANSITIONS[self.status]
        if to not in allowed:
            raise InvalidTransition(
                f"{self.request_id}: illegal transition "
                f"{self.status.value} → {to.value} (reason={reason!r})"
            )
        prev = self.status
        self.status = to
        self._history.append((to, reason))
        if to in _TERMINAL:
            self.finish_reason = reason
        return prev
