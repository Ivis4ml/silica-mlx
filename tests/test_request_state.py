"""Tests for silica.core.request — RequestState FSM (P-2 Unit #12)."""

from __future__ import annotations

import pytest

from silica.core.request import (
    InvalidTransition,
    Request,
    RequestState,
    RequestStatus,
)
from silica.core.sampling import SamplingParams


def _state(**overrides: object) -> RequestState:
    req = Request(prompt="hi", sampling_params=SamplingParams(), token_ids=(1, 2, 3))
    return RequestState(request=req, **overrides)  # type: ignore[arg-type]


# --- shape / defaults ---


def test_default_status_is_waiting() -> None:
    assert _state().status is RequestStatus.WAITING


def test_all_six_statuses_defined() -> None:
    values = {s.value for s in RequestStatus}
    assert values == {
        "waiting",
        "prefill",
        "decode",
        "preempted",
        "done",
        "aborted",
    }


def test_is_terminal_false_for_active_states() -> None:
    s = _state()
    assert not s.is_terminal
    s.transition(RequestStatus.PREFILL, reason="admit")
    assert not s.is_terminal
    s.transition(RequestStatus.DECODE, reason="prefill-done")
    assert not s.is_terminal


def test_is_finished_alias_preserves_preP2_api() -> None:
    s = _state()
    assert s.is_finished is False
    s.transition(RequestStatus.ABORTED, reason="cancel")
    assert s.is_finished is True


def test_request_id_property_delegates_to_request() -> None:
    req = Request(prompt="x", sampling_params=SamplingParams(), request_id="abc")
    s = RequestState(request=req)
    assert s.request_id == "abc"


# --- happy-path transitions ---


def test_happy_path_waiting_prefill_decode_done() -> None:
    s = _state()
    prev = s.transition(RequestStatus.PREFILL, reason="admit")
    assert prev is RequestStatus.WAITING
    assert s.status is RequestStatus.PREFILL

    prev = s.transition(RequestStatus.DECODE, reason="prefill-done")
    assert prev is RequestStatus.PREFILL

    prev = s.transition(RequestStatus.DONE, reason="max_tokens")
    assert prev is RequestStatus.DECODE
    assert s.is_terminal
    assert s.finish_reason == "max_tokens"


def test_prefill_may_short_circuit_to_done() -> None:
    """First sampled token is a stop token: request terminates before DECODE."""
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    s.transition(RequestStatus.DONE, reason="eos-on-first-token")
    assert s.is_terminal
    assert s.finish_reason == "eos-on-first-token"


def test_waiting_may_abort_directly() -> None:
    """User cancel before admission."""
    s = _state()
    s.transition(RequestStatus.ABORTED, reason="user-cancel")
    assert s.is_terminal
    assert s.finish_reason == "user-cancel"


# --- preempt / re-admit cycle ---


def test_preempt_from_prefill_then_readmit() -> None:
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    s.transition(RequestStatus.PREEMPTED, reason="budget-pressure")
    assert s.status.value == "preempted"
    s.transition(RequestStatus.WAITING, reason="re-admit")
    assert s.status.value == "waiting"
    # Can be admitted again.
    s.transition(RequestStatus.PREFILL, reason="admit-again")


def test_preempt_from_decode_then_readmit() -> None:
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    s.transition(RequestStatus.DECODE, reason="prefill-done")
    s.transition(RequestStatus.PREEMPTED, reason="budget-pressure")
    s.transition(RequestStatus.WAITING, reason="re-admit")
    s.transition(RequestStatus.PREFILL, reason="admit-again")
    s.transition(RequestStatus.DECODE, reason="prefill-done")
    assert s.status is RequestStatus.DECODE


def test_preempt_can_give_up_to_aborted() -> None:
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    s.transition(RequestStatus.PREEMPTED, reason="budget")
    s.transition(RequestStatus.ABORTED, reason="budget-unrecoverable")
    assert s.is_terminal
    assert s.finish_reason == "budget-unrecoverable"


# --- illegal transitions (enumerated) ---

# Every (from, to) pair NOT in the allow-list should raise. We enumerate
# them rather than sampling so the test fails loud on any allow-list
# widening or narrowing.
_STATUSES = list(RequestStatus)
_LEGAL = {
    (RequestStatus.WAITING, RequestStatus.PREFILL),
    (RequestStatus.WAITING, RequestStatus.ABORTED),
    (RequestStatus.PREFILL, RequestStatus.DECODE),
    (RequestStatus.PREFILL, RequestStatus.PREEMPTED),
    (RequestStatus.PREFILL, RequestStatus.DONE),
    (RequestStatus.PREFILL, RequestStatus.ABORTED),
    (RequestStatus.DECODE, RequestStatus.DONE),
    (RequestStatus.DECODE, RequestStatus.PREEMPTED),
    (RequestStatus.DECODE, RequestStatus.ABORTED),
    (RequestStatus.PREEMPTED, RequestStatus.WAITING),
    (RequestStatus.PREEMPTED, RequestStatus.ABORTED),
}
_ILLEGAL = [
    (f, t) for f in _STATUSES for t in _STATUSES if (f, t) not in _LEGAL
]


@pytest.mark.parametrize("from_status,to_status", _ILLEGAL)
def test_illegal_transitions_all_raise(
    from_status: RequestStatus, to_status: RequestStatus
) -> None:
    """Any (from, to) pair outside the allow-list raises InvalidTransition."""
    s = _state()
    # Force from_status without going through transition() — we are testing
    # the transition validator, not the route into from_status.
    s.status = from_status
    with pytest.raises(InvalidTransition):
        s.transition(to_status, reason="test-illegal")


def test_terminal_states_reject_all_transitions() -> None:
    for terminal in (RequestStatus.DONE, RequestStatus.ABORTED):
        for target in RequestStatus:
            s = _state()
            s.status = terminal
            with pytest.raises(InvalidTransition):
                s.transition(target, reason="from-terminal")


# --- history / finish_reason ---


def test_history_records_all_transitions_in_order() -> None:
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    s.transition(RequestStatus.DECODE, reason="prefill-done")
    s.transition(RequestStatus.DONE, reason="stop-token")
    assert s.history == [
        (RequestStatus.PREFILL, "admit"),
        (RequestStatus.DECODE, "prefill-done"),
        (RequestStatus.DONE, "stop-token"),
    ]


def test_history_property_returns_a_copy() -> None:
    """Mutating the returned history does not affect future transitions."""
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    h = s.history
    h.clear()
    assert len(s.history) == 1  # underlying log untouched


def test_finish_reason_only_set_on_terminal() -> None:
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    assert s.finish_reason is None
    s.transition(RequestStatus.DECODE, reason="prefill-done")
    assert s.finish_reason is None
    s.transition(RequestStatus.DONE, reason="max_tokens")
    assert s.finish_reason == "max_tokens"


# --- P-2 payload fields ---


def test_state_delta_snapshot_retained_across_preempt_readmit() -> None:
    """Scheduler stashes recurrent state here; it must survive re-admit."""
    s = _state()
    s.transition(RequestStatus.PREFILL, reason="admit")
    s.transition(RequestStatus.DECODE, reason="prefill-done")
    s.state_delta_snapshot = {"opaque": "payload"}
    s.transition(RequestStatus.PREEMPTED, reason="budget")
    assert s.state_delta_snapshot == {"opaque": "payload"}
    s.transition(RequestStatus.WAITING, reason="re-admit")
    assert s.state_delta_snapshot == {"opaque": "payload"}


def test_prefix_hit_tokens_defaults_zero() -> None:
    s = _state()
    assert s.prefix_hit_tokens == 0
    s.prefix_hit_tokens = 100
    assert s.prefix_hit_tokens == 100


# --- pure FSM: transition does not mutate unrelated fields ---


def test_transition_is_pure_on_unrelated_fields() -> None:
    s = _state()
    s.num_computed_tokens = 10
    s.output_token_ids = [7, 8, 9]
    arrival = s.arrival_time
    s.transition(RequestStatus.PREFILL, reason="admit")
    assert s.num_computed_tokens == 10
    assert s.output_token_ids == [7, 8, 9]
    assert s.arrival_time == arrival
