"""Tests for silica.speculative.engine — I-5 contract + NoopDraftEngine semantics."""

from __future__ import annotations

import pytest

from silica.core.request import Request, RequestState
from silica.core.sampling import SamplingParams
from silica.speculative.engine import DraftEngine, DraftTokens, NoopDraftEngine


@pytest.fixture
def ctx() -> RequestState:
    req = Request(prompt="hello", sampling_params=SamplingParams())
    return RequestState(request=req)


@pytest.fixture
def engine() -> NoopDraftEngine:
    return NoopDraftEngine()


# --- I-5 Protocol shape ---


def test_noop_satisfies_draft_engine_protocol(engine: NoopDraftEngine) -> None:
    assert isinstance(engine, DraftEngine)


# --- NoopDraftEngine semantics ---


def test_propose_returns_empty_draft(engine: NoopDraftEngine, ctx: RequestState) -> None:
    draft = engine.propose(ctx, k=4)
    assert isinstance(draft, DraftTokens)
    assert draft.token_ids == ()
    assert draft.draft_logprobs is None


def test_propose_is_insensitive_to_k(engine: NoopDraftEngine, ctx: RequestState) -> None:
    # NoopDraftEngine emits empty regardless of window size.
    for k in (0, 1, 4, 64):
        assert engine.propose(ctx, k=k).token_ids == ()


def test_commit_is_noop(engine: NoopDraftEngine, ctx: RequestState) -> None:
    # Must accept any accepted_len without raising (return type is None by contract).
    for accepted_len in (0, 1, 4, 100):
        engine.commit(ctx, accepted_len=accepted_len)


def test_commit_does_not_mutate_request_state(
    engine: NoopDraftEngine, ctx: RequestState
) -> None:
    before = (
        ctx.status,
        ctx.num_computed_tokens,
        ctx.num_output_tokens,
        tuple(ctx.output_token_ids),
    )
    engine.commit(ctx, accepted_len=3)
    after = (
        ctx.status,
        ctx.num_computed_tokens,
        ctx.num_output_tokens,
        tuple(ctx.output_token_ids),
    )
    assert before == after


# --- DraftTokens basics ---


def test_draft_tokens_with_logprobs() -> None:
    d = DraftTokens(token_ids=(1, 2, 3), draft_logprobs=(-0.5, -0.8, -1.2))
    assert d.token_ids == (1, 2, 3)
    assert d.draft_logprobs == (-0.5, -0.8, -1.2)


def test_draft_tokens_without_logprobs() -> None:
    d = DraftTokens(token_ids=(7, 11))
    assert d.draft_logprobs is None
