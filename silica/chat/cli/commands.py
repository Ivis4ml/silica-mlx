"""silica.chat.cli.commands — slash command dispatch.

Implements the ``/help``, ``/reset``, ``/exit``, ``/system``,
``/config``, ``/regenerate``, ``/save``, ``/load``, ``/model``,
``/expand``, and ``/showcase`` commands as plain functions
operating on :class:`ChatCliState`. The prompt_toolkit Application
shell (C-3) reads :func:`dispatch_command`'s :class:`CommandResult`
return and applies the requested side effects (printing feedback,
quitting the loop, asking the session to regenerate the last turn,
swapping models, etc.).

Side effects that the command handlers themselves cannot perform
(without an Engine / Session reference) are encoded as request
flags on :class:`CommandResult`. The shell drains the request
flags and dispatches them. This keeps the commands unit-testable
without any engine setup.

Command grammar:

- ``/help`` — list registered commands with their summaries.
- ``/exit`` — quit the REPL.
- ``/reset`` — clear the conversation log + invalidate prefix
  cache. Encoded as ``CommandResult.request_reset = True``.
- ``/system "..."`` — set the system prompt. The text after
  ``/system`` is treated as the new system prompt (with leading
  and trailing whitespace + optional surrounding quotes stripped).
- ``/config key=value`` — apply a runtime config override. With
  no argument, prints the current config + schema help.
- ``/regenerate`` — redo the previous turn with a fresh sample.
  Encoded as ``CommandResult.request_regenerate = True``.
- ``/save <path>`` — persist the conversation to a JSON file.
- ``/load <path>`` — load a conversation from a JSON file.
- ``/model <repo_id>`` — swap the active model. Heavy operation
  (full re-load); encoded as ``CommandResult.request_model_swap``
  for the shell to execute.
- ``/expand`` — reprint the previous turn's buffered thinking
  text (dimmed-italic).
- ``/showcase`` — print a one-paragraph narrative of the current
  session's silica-mlx-USP signals.

Unknown or malformed commands return a :class:`CommandResult`
with an error feedback line; they do not raise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from silica.chat.cli.config import (
    CONFIG_SCHEMA,
    ConfigError,
    parse_config_assignment,
    render_schema_help,
)
from silica.chat.cli.state import ChatCliState

# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """Outcome of a single slash-command dispatch.

    The command handler populates only the fields it cares about;
    the shell drains the flags after the dispatcher returns. All
    fields are optional so commands that just print feedback set
    only :attr:`feedback`.
    """

    feedback: list[str] = field(default_factory=list)
    """Lines to display to the user. Each entry is one logical
    line; the shell decides how to render (colour, prefix, etc.)."""

    error: bool = False
    """``True`` when :attr:`feedback` carries an error message; the
    shell renders these in red."""

    quit: bool = False
    """Shell exits the REPL after surfacing :attr:`feedback`."""

    request_reset: bool = False
    """Shell clears the conversation log + invalidates the
    prefix cache."""

    request_regenerate: bool = False
    """Shell redoes the previous turn with a fresh sample."""

    request_session_save: str | None = None
    """Path the shell should serialise the session to."""

    request_session_load: str | None = None
    """Path the shell should deserialise a session from."""

    request_model_swap: str | None = None
    """HF repo id for the shell to re-load."""

    request_expand_thinking: bool = False
    """Shell reprints :attr:`ChatCliState.last_turn_thinking` to
    the conversation log in dimmed-italic style."""

    request_showcase: bool = False
    """Shell renders the session-narrative report (uses the
    cumulative session metrics, not just the live state)."""


def _ok(*lines: str) -> CommandResult:
    return CommandResult(feedback=list(lines))


def _err(*lines: str) -> CommandResult:
    return CommandResult(feedback=list(lines), error=True)


# ---------------------------------------------------------------------------
# Command registry
# ---------------------------------------------------------------------------


CommandHandler = Callable[[ChatCliState, str], CommandResult]


@dataclass(frozen=True)
class Command:
    """Registry entry for one slash command."""

    name: str
    """Without the leading slash (``"help"``, ``"reset"``, ...)."""

    summary: str
    """One-line description shown by ``/help``."""

    handler: CommandHandler
    """Implementation. Takes ``(state, args_string)``; ``args_string``
    is what came after the command name on the REPL line, with the
    leading whitespace stripped."""

    args_help: str = ""
    """Optional usage string (e.g. ``"<repo_id>"``); rendered after
    the command name in ``/help``."""


# Concrete handlers ----------------------------------------------------------


def _cmd_help(state: ChatCliState, args: str) -> CommandResult:
    del state, args  # unused
    lines: list[str] = ["available commands:"]
    for cmd in COMMANDS.values():
        suffix = f" {cmd.args_help}" if cmd.args_help else ""
        lines.append(f"  /{cmd.name}{suffix}  — {cmd.summary}")
    lines.append("")
    lines.append("config keys (use /config key=value):")
    lines.extend(render_schema_help())
    return _ok(*lines)


def _cmd_exit(state: ChatCliState, args: str) -> CommandResult:
    del state, args
    return CommandResult(feedback=["bye."], quit=True)


def _cmd_reset(state: ChatCliState, args: str) -> CommandResult:
    del state, args
    return CommandResult(
        feedback=["session reset."], request_reset=True
    )


def _cmd_system(state: ChatCliState, args: str) -> CommandResult:
    text = args.strip()
    # Tolerate optional surrounding quotes so users can write
    # /system "You are concise." or /system You are concise.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1]
    state.config["system_prompt"] = text
    if text:
        return _ok(f"system prompt set ({len(text)} chars).")
    return _ok("system prompt cleared.")


def _cmd_config(state: ChatCliState, args: str) -> CommandResult:
    """Without args: print every schema entry plus the current
    overrides. With args: parse one assignment and apply it."""
    if not args.strip():
        lines: list[str] = ["current config (defaults shown unless overridden):"]
        for key in CONFIG_SCHEMA:
            current = state.config.get(key, CONFIG_SCHEMA[key].default)
            current_repr = "none" if current is None else repr(current)
            lines.append(f"  {key} = {current_repr}")
        lines.append("")
        lines.append("schema:")
        lines.extend(render_schema_help())
        return _ok(*lines)
    try:
        key, value = parse_config_assignment(args)
    except ConfigError as exc:
        return _err(f"/config: {exc}")
    state.config[key] = value
    rendered = "none" if value is None else repr(value)
    return _ok(f"config: {key} = {rendered}")


def _cmd_regenerate(state: ChatCliState, args: str) -> CommandResult:
    del state, args
    return CommandResult(
        feedback=["regenerating last turn..."],
        request_regenerate=True,
    )


def _cmd_save(state: ChatCliState, args: str) -> CommandResult:
    del state
    path = args.strip()
    if not path:
        return _err("/save: missing path. usage: /save <path>")
    return CommandResult(
        feedback=[f"saving session to {path}..."],
        request_session_save=path,
    )


def _cmd_load(state: ChatCliState, args: str) -> CommandResult:
    del state
    path = args.strip()
    if not path:
        return _err("/load: missing path. usage: /load <path>")
    return CommandResult(
        feedback=[f"loading session from {path}..."],
        request_session_load=path,
    )


def _cmd_model(state: ChatCliState, args: str) -> CommandResult:
    del state
    repo = args.strip()
    if not repo:
        return _err("/model: missing repo id. usage: /model <repo_id>")
    return CommandResult(
        feedback=[f"swapping model to {repo}..."],
        request_model_swap=repo,
    )


def _cmd_expand(state: ChatCliState, args: str) -> CommandResult:
    del args  # /expand <turn_idx> reserved; v0.1 always shows the latest
    if not state.last_turn_thinking:
        return _err(
            "/expand: no thinking buffered yet "
            "(model has not produced a <think> block this session)"
        )
    return CommandResult(
        feedback=[],  # actual rendering done by the shell
        request_expand_thinking=True,
    )


def _cmd_showcase(state: ChatCliState, args: str) -> CommandResult:
    del state, args
    return CommandResult(
        feedback=[],
        request_showcase=True,
    )


# Registry — declared after the handler functions so the dispatcher
# tests can introspect it without instantiating Command in test
# bodies. Order is the order /help prints them.
COMMANDS: dict[str, Command] = {
    "help": Command(
        name="help",
        summary="list commands and config keys",
        handler=_cmd_help,
    ),
    "exit": Command(
        name="exit",
        summary="quit the chat REPL",
        handler=_cmd_exit,
    ),
    "reset": Command(
        name="reset",
        summary="clear the conversation log and prefix cache",
        handler=_cmd_reset,
    ),
    "system": Command(
        name="system",
        summary="set the system prompt for the rest of the session",
        handler=_cmd_system,
        args_help='"text..."',
    ),
    "config": Command(
        name="config",
        summary="adjust runtime config (e.g. temperature, max_tokens)",
        handler=_cmd_config,
        args_help="[key=value]",
    ),
    "regenerate": Command(
        name="regenerate",
        summary="redo the previous turn with a fresh sample",
        handler=_cmd_regenerate,
    ),
    "save": Command(
        name="save",
        summary="persist the current conversation to a JSON file",
        handler=_cmd_save,
        args_help="<path>",
    ),
    "load": Command(
        name="load",
        summary="restore a conversation from a JSON file",
        handler=_cmd_load,
        args_help="<path>",
    ),
    "model": Command(
        name="model",
        summary="swap the active model (full re-load; heavy)",
        handler=_cmd_model,
        args_help="<repo_id>",
    ),
    "expand": Command(
        name="expand",
        summary="reprint the previous turn's collapsed thinking",
        handler=_cmd_expand,
    ),
    "showcase": Command(
        name="showcase",
        summary="print a one-paragraph narrative of the session",
        handler=_cmd_showcase,
    ),
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def is_slash_command(line: str) -> bool:
    """Return True if ``line`` is a slash command (after stripping
    leading whitespace). Plain chat messages return False; the
    shell routes those to the engine instead of the dispatcher.

    Note: a literal ``/`` followed by whitespace or nothing is not
    a command — the user might be starting a literal forward-slash
    message. The dispatcher requires at least one alphanumeric
    character after the slash to count.
    """
    s = line.lstrip()
    return len(s) >= 2 and s[0] == "/" and (s[1].isalpha() or s[1] == "_")


def dispatch_command(line: str, state: ChatCliState) -> CommandResult:
    """Parse ``line`` and dispatch to the registered handler.

    Caller should first check :func:`is_slash_command`. The
    function does not raise on unknown commands; it returns an
    error :class:`CommandResult` so the shell uniformly renders
    feedback regardless of dispatch outcome.

    Args:
        line: the raw REPL input. Leading whitespace and the
            initial ``/`` are stripped here; the rest is split
            into ``(command_name, args_string)`` on the first run
            of whitespace.
        state: the live :class:`ChatCliState`; passed through to
            the handler. The handler may mutate ``state.config``
            directly (e.g. ``/config key=value``); other state
            mutations are deferred to the shell via request flags
            on :class:`CommandResult`.
    """
    stripped = line.lstrip()
    assert stripped.startswith("/"), (
        "dispatch_command requires a slash-command line; "
        f"got {line!r} — call is_slash_command first"
    )
    body = stripped[1:]
    name, _, args = body.partition(" ")
    name = name.rstrip("\t\r\n")
    args = args.lstrip()
    cmd = COMMANDS.get(name)
    if cmd is None:
        suggestions = ", ".join(f"/{c}" for c in COMMANDS)
        return _err(
            f"unknown command /{name!r}. valid commands: {suggestions}"
        )
    return cmd.handler(state, args)


__all__ = [
    "COMMANDS",
    "Command",
    "CommandHandler",
    "CommandResult",
    "dispatch_command",
    "is_slash_command",
]
