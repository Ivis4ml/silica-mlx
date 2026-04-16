"""Request and request lifecycle types for Silica engine.

RequestState tracks the lifecycle: WAITING → PREFILL → DECODE → DONE/ABORTED.
Request holds the immutable input; RequestState holds the mutable progress.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silica.core.sampling import SamplingParams


class RequestStatus(enum.Enum):
    """Request lifecycle states (P-2 state machine: WAITING → PREFILL → DECODE → DONE/ABORTED)."""

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    DONE = "done"
    ABORTED = "aborted"


@dataclass(frozen=True)
class Request:
    """Immutable request input submitted to the engine.

    Attributes:
        request_id: Unique identifier (auto-generated if not provided).
        prompt: Raw text prompt.
        token_ids: Pre-tokenized input (populated by engine after tokenization).
        sampling_params: Sampling configuration.
    """

    prompt: str
    sampling_params: SamplingParams
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    token_ids: tuple[int, ...] = ()


@dataclass
class RequestState:
    """Mutable state tracking a request's progress through the engine.

    Owned by the scheduler; updated as the request moves through its lifecycle.
    """

    request: Request
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    num_output_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.DONE, RequestStatus.ABORTED)
