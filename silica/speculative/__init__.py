"""silica.speculative — draft-target speculative decoding (I-5)."""

from silica.speculative.draft_target import DraftTargetEngine
from silica.speculative.engine import (
    DraftEngine,
    DraftTokens,
    NoopDraftEngine,
    TargetHiddenConsumer,
)
from silica.speculative.verify import greedy_verify, run_verify_forward

__all__ = [
    "DraftEngine",
    "DraftTargetEngine",
    "DraftTokens",
    "NoopDraftEngine",
    "TargetHiddenConsumer",
    "greedy_verify",
    "run_verify_forward",
]
