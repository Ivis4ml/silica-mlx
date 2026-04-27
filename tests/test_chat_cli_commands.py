"""C-2 unit tests for the chat-CLI slash-command dispatcher.

Pure-Python coverage of ``silica.chat.cli.commands``. Verifies the
recognition rule (`is_slash_command`), the per-command handler
behaviour, the request-flag plumbing on :class:`CommandResult`,
and the error-feedback path for unknown / malformed input.
"""

from __future__ import annotations

import pytest

from silica.chat.cli.commands import (
    COMMANDS,
    dispatch_command,
    is_slash_command,
)
from silica.chat.cli.state import ChatCliState

# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "/help",
        "/exit",
        "/config temperature=0.5",
        "/system You are concise.",
        "  /help",  # leading whitespace tolerated
        "/_internal",  # underscore-prefixed names allowed
    ],
)
def test_is_slash_command_recognises_commands(line: str) -> None:
    assert is_slash_command(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "what is mlx?",
        "/ ",  # bare slash not a command
        "/",
        "//double-slash starts a regular line",
        "    ",
        "",
        "12 / 4 = 3",  # slash inside text
    ],
)
def test_is_slash_command_rejects_non_commands(line: str) -> None:
    assert is_slash_command(line) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state() -> ChatCliState:
    return ChatCliState(model_name="Qwen3-0.6B")


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------


def test_help_lists_every_registered_command() -> None:
    res = dispatch_command("/help", _state())
    assert res.error is False
    assert res.quit is False
    rendered = "\n".join(res.feedback)
    for cmd in COMMANDS:
        assert f"/{cmd}" in rendered, (
            f"command /{cmd} missing from /help output"
        )


def test_help_includes_config_schema() -> None:
    res = dispatch_command("/help", _state())
    rendered = "\n".join(res.feedback)
    assert "config keys" in rendered
    assert "temperature" in rendered
    assert "max_tokens" in rendered


# ---------------------------------------------------------------------------
# /exit
# ---------------------------------------------------------------------------


def test_exit_sets_quit_flag() -> None:
    res = dispatch_command("/exit", _state())
    assert res.quit is True
    assert res.error is False


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------


def test_reset_sets_request_reset_flag() -> None:
    res = dispatch_command("/reset", _state())
    assert res.request_reset is True
    assert res.quit is False


# ---------------------------------------------------------------------------
# /system
# ---------------------------------------------------------------------------


def test_system_sets_prompt_in_config() -> None:
    state = _state()
    res = dispatch_command("/system You are a concise assistant.", state)
    assert res.error is False
    assert state.config["system_prompt"] == "You are a concise assistant."


def test_system_strips_surrounding_quotes() -> None:
    state = _state()
    dispatch_command('/system "You are concise."', state)
    assert state.config["system_prompt"] == "You are concise."
    state2 = _state()
    dispatch_command("/system 'single quoted'", state2)
    assert state2.config["system_prompt"] == "single quoted"


def test_system_empty_args_clears_prompt() -> None:
    state = _state()
    state.config["system_prompt"] = "earlier prompt"
    res = dispatch_command("/system", state)
    assert res.error is False
    assert state.config["system_prompt"] == ""


# ---------------------------------------------------------------------------
# /config
# ---------------------------------------------------------------------------


def test_config_no_args_lists_current_values() -> None:
    state = _state()
    state.config["temperature"] = 0.3
    res = dispatch_command("/config", state)
    assert res.error is False
    rendered = "\n".join(res.feedback)
    assert "temperature = 0.3" in rendered
    # schema listing shown too
    assert "schema:" in rendered


def test_config_assignment_applies_to_state() -> None:
    state = _state()
    res = dispatch_command("/config temperature=0.2", state)
    assert res.error is False
    assert state.config["temperature"] == pytest.approx(0.2)


def test_config_unknown_key_returns_error_feedback() -> None:
    state = _state()
    res = dispatch_command("/config nonsense=42", state)
    assert res.error is True
    assert "/config" in res.feedback[0]
    assert "nonsense" in res.feedback[0]


def test_config_validation_error_returned_as_feedback() -> None:
    state = _state()
    res = dispatch_command("/config temperature=99", state)
    assert res.error is True
    # State must not be mutated on validation failure.
    assert "temperature" not in state.config


def test_config_choice_value_applied() -> None:
    state = _state()
    res = dispatch_command("/config thinking=hidden", state)
    assert res.error is False
    assert state.config["thinking"] == "hidden"


# ---------------------------------------------------------------------------
# /regenerate
# ---------------------------------------------------------------------------


def test_regenerate_sets_request_flag() -> None:
    res = dispatch_command("/regenerate", _state())
    assert res.request_regenerate is True


# ---------------------------------------------------------------------------
# /save and /load
# ---------------------------------------------------------------------------


def test_save_carries_path_in_request() -> None:
    res = dispatch_command("/save chat.json", _state())
    assert res.request_session_save == "chat.json"
    assert res.error is False


def test_save_missing_path_reports_error() -> None:
    res = dispatch_command("/save", _state())
    assert res.error is True
    assert res.request_session_save is None


def test_load_carries_path_in_request() -> None:
    res = dispatch_command("/load /tmp/chat.json", _state())
    assert res.request_session_load == "/tmp/chat.json"


def test_load_missing_path_reports_error() -> None:
    res = dispatch_command("/load", _state())
    assert res.error is True
    assert res.request_session_load is None


# ---------------------------------------------------------------------------
# /model
# ---------------------------------------------------------------------------


def test_model_carries_repo_in_request() -> None:
    res = dispatch_command("/model Qwen/Qwen3-4B", _state())
    assert res.request_model_swap == "Qwen/Qwen3-4B"


def test_model_missing_repo_reports_error() -> None:
    res = dispatch_command("/model", _state())
    assert res.error is True
    assert res.request_model_swap is None


# ---------------------------------------------------------------------------
# /expand
# ---------------------------------------------------------------------------


def test_expand_no_thinking_buffered_returns_error() -> None:
    state = _state()
    assert state.last_turn_thinking == ""
    res = dispatch_command("/expand", state)
    assert res.error is True
    assert res.request_expand_thinking is False


def test_expand_with_thinking_sets_request_flag() -> None:
    state = _state()
    state.last_turn_thinking = "deliberating about this..."
    res = dispatch_command("/expand", state)
    assert res.error is False
    assert res.request_expand_thinking is True


# ---------------------------------------------------------------------------
# /showcase
# ---------------------------------------------------------------------------


def test_showcase_sets_request_flag() -> None:
    res = dispatch_command("/showcase", _state())
    assert res.request_showcase is True


# ---------------------------------------------------------------------------
# Unknown command
# ---------------------------------------------------------------------------


def test_unknown_command_returns_error_with_suggestions() -> None:
    res = dispatch_command("/notacommand", _state())
    assert res.error is True
    rendered = "\n".join(res.feedback)
    # Suggestion list must include real commands so the user can
    # recover without typing /help separately.
    assert "/help" in rendered
    assert "/config" in rendered


# ---------------------------------------------------------------------------
# Result purity — no mutations on read-only commands
# ---------------------------------------------------------------------------


def test_help_does_not_mutate_state() -> None:
    state = _state()
    snapshot = (
        dict(state.config),
        state.model_name,
        state.last_turn_thinking,
    )
    dispatch_command("/help", state)
    assert (
        dict(state.config),
        state.model_name,
        state.last_turn_thinking,
    ) == snapshot


def test_unknown_command_does_not_mutate_state() -> None:
    state = _state()
    snapshot = dict(state.config)
    dispatch_command("/totally_made_up", state)
    assert dict(state.config) == snapshot


# ---------------------------------------------------------------------------
# Dispatch-from-non-slash precondition
# ---------------------------------------------------------------------------


def test_dispatch_command_requires_slash_prefix() -> None:
    """``dispatch_command`` is documented to require a slash-
    prefixed line; calling it without is a programmer bug, not a
    user error. Document via assertion."""
    with pytest.raises(AssertionError):
        dispatch_command("hello", _state())


# ---------------------------------------------------------------------------
# Argument splitting — first whitespace boundary
# ---------------------------------------------------------------------------


def test_args_split_on_first_whitespace() -> None:
    """``/system`` followed by multi-word args must keep the args
    intact (no split into individual words). Same for ``/save``
    with a path containing spaces (treated as plain text path)."""
    state = _state()
    dispatch_command("/system the quick brown fox jumps", state)
    assert state.config["system_prompt"] == "the quick brown fox jumps"
