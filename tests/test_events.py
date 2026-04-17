"""Tests for silica.core.events — BatchEvent dataclass (P-2 Unit 16a)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from silica.core.events import BatchEvent


def test_token_factory_sets_kind_and_token_id() -> None:
    e = BatchEvent.token(req_index=3, token_id=42)
    assert e.req_index == 3
    assert e.kind == "token"
    assert e.token_id == 42
    assert e.finish_reason is None


def test_done_factory_sets_kind_and_reason() -> None:
    e = BatchEvent.done(req_index=0, reason="max_tokens")
    assert e.kind == "done"
    assert e.finish_reason == "max_tokens"
    assert e.token_id is None


def test_aborted_factory_sets_kind_and_reason() -> None:
    e = BatchEvent.aborted(req_index=7, reason="budget-exhausted")
    assert e.kind == "aborted"
    assert e.finish_reason == "budget-exhausted"
    assert e.token_id is None


def test_frozen() -> None:
    e = BatchEvent.token(req_index=0, token_id=1)
    with pytest.raises(FrozenInstanceError):
        e.req_index = 5  # type: ignore[misc]


def test_equality_by_fields() -> None:
    a = BatchEvent.token(req_index=1, token_id=10)
    b = BatchEvent.token(req_index=1, token_id=10)
    c = BatchEvent.token(req_index=1, token_id=11)
    assert a == b
    assert a != c


def test_direct_construction_with_all_fields() -> None:
    e = BatchEvent(
        req_index=2, kind="token", token_id=5, finish_reason=None
    )
    assert e.kind == "token"
