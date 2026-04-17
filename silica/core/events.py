"""silica.core.events — BatchEvent stream element for Engine.generate_batch.

P-2 Opening v2.1+ specifies that ``Engine.generate_batch`` yields an
``Iterator[BatchEvent]`` rather than a tuple stream, so terminal events
(``done`` / ``aborted``) are first-class values and future event kinds
(e.g. ``progress``, ``preempted`` if ever surfaced) can be added
backward-compatibly by widening the ``kind`` literal.

``BatchEvent`` lives under ``silica.core`` so both ``silica.scheduler``
(producer) and ``silica.engine`` (consumer + re-yielder) can import it
without creating a dependency cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EventKind = Literal["token", "done", "aborted"]


@dataclass(frozen=True)
class BatchEvent:
    """One event in the stream produced by ``Engine.generate_batch``.

    Attributes:
        req_index: Zero-based index of the request in the original
            ``prompts`` list passed to ``generate_batch``.
        kind: Event type.
            - ``"token"``: a generated token for the request. Many per
              request.
            - ``"done"``: request terminated normally. One per request;
              no further events for this index.
            - ``"aborted"``: request terminated abnormally (budget,
              error). One per request; no further events.
        token_id: Set when ``kind == "token"``; otherwise ``None``.
        finish_reason: Set when ``kind`` is terminal. Conventional
            values: ``"stop_token"``, ``"max_tokens"``,
            ``"budget-exhausted"``, ``"error"``.
    """

    req_index: int
    kind: EventKind
    token_id: int | None = None
    finish_reason: str | None = None

    @classmethod
    def token(cls, req_index: int, token_id: int) -> BatchEvent:
        return cls(req_index=req_index, kind="token", token_id=token_id)

    @classmethod
    def done(cls, req_index: int, reason: str) -> BatchEvent:
        return cls(req_index=req_index, kind="done", finish_reason=reason)

    @classmethod
    def aborted(cls, req_index: int, reason: str) -> BatchEvent:
        return cls(req_index=req_index, kind="aborted", finish_reason=reason)
