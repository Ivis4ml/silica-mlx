"""silica.chat.cli.config — chat-CLI runtime configuration.

The config layer holds the values that ``/config key=value`` can
adjust mid-session: sampling defaults (``temperature``, ``top_p``,
``top_k``, ``max_tokens``), thinking-mode visibility / on-off,
the codec-hint threshold, the codec id (read-only mid-session in
v0.1; recorded for completeness), and a handful of UI knobs.

Schema is declarative (see :data:`CONFIG_SCHEMA`) so adding a new
``/config`` key is a one-line registration plus a parser entry —
the dispatcher in :mod:`silica.chat.cli.commands` reads the schema
to validate user input and produce error messages without any
key-by-key conditional logic.

Type coercion is conservative: every ``/config`` value arrives as
a string from the REPL line, and each entry's :attr:`Config.parse`
function is responsible for turning that string into the typed
value the rest of the chat code expects. Validation errors raise
:class:`ConfigError` with a one-line user-readable message; the
dispatcher catches these and surfaces them to the chat log as a
red error line.

Defaults match ``plans/CHAT_CLI_OPENING.md`` §6 with two v1.7.36
adjustments retained at v1.7.37: ``max_tokens`` raised from 1024
to **8192** so a single turn can comfortably hold reasoning + a
long reply (Qwen3 / Qwen3.5 / Gemma 4 reasoning models routinely
spend 1-2k tokens inside ``<think>`` before the visible reply
starts); ``thinking`` flipped from ``auto`` to **``show``** so
reasoning text streams inline as it is generated rather than
sitting behind a static spinner. The third v1.7.36 adjustment
— ``thinking_mode`` ``True → False`` — was **reverted at v1.7.37**:
the flip produced broken text-completion output on
Qwen3.5-Next-A3B (model emitted ``<|im_end|>`` literal text and
hallucinated additional turns) and a degraded version of the
same failure mode on Gemma 4 (mlx-community/gemma-4-31b-4bit).
The verbose-reasoning UX issue the v1.7.36 flip aimed to address
is best handled by the v1.7.38 fixed-height
:class:`silica.chat.cli.thinking_scroll.ThinkingScrollWindow`
(displays the last 6 lines of reasoning text in a rolling
window) rather than by suppressing reasoning entirely. Sampling
defaults are unchanged (temperature 0.7, top_p 0.9, top_k off);
``kv_codec`` defaults to None (fp16) per §6.1.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """User-facing config validation error.

    Exists as a distinct subclass so the command dispatcher can
    catch parser/validator failures and render them as a chat-log
    error line without conflating with internal :class:`ValueError`
    bugs raised elsewhere in the runtime.
    """


# ---------------------------------------------------------------------------
# Per-key parser / validator helpers
# ---------------------------------------------------------------------------


def _parse_float(key: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{key}: expected a float, got {raw!r}"
        ) from exc


def _parse_int(key: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{key}: expected an integer, got {raw!r}"
        ) from exc


def _parse_optional_int(key: str, raw: str) -> int | None:
    """Parse to int, treating ``"none"`` / ``"off"`` / ``""`` as
    "unset"."""
    lowered = raw.strip().lower()
    if lowered in ("none", "off", ""):
        return None
    return _parse_int(key, raw)


def _parse_bool(key: str, raw: str) -> bool:
    """Liberal bool parser. Accepts on/off, true/false, yes/no,
    1/0. Unmatched input raises :class:`ConfigError` so the user
    sees what was rejected."""
    lowered = raw.strip().lower()
    if lowered in ("on", "true", "yes", "1"):
        return True
    if lowered in ("off", "false", "no", "0"):
        return False
    raise ConfigError(
        f"{key}: expected on/off (or true/false, yes/no, 1/0); "
        f"got {raw!r}"
    )


def _parse_choice(
    key: str,
    raw: str,
    choices: tuple[str, ...],
) -> str:
    """Validate that ``raw`` matches one of ``choices`` (case-
    sensitive). Raises a :class:`ConfigError` listing the allowed
    values if not."""
    if raw not in choices:
        allowed = ", ".join(repr(c) for c in choices)
        raise ConfigError(
            f"{key}: expected one of {allowed}; got {raw!r}"
        )
    return raw


def _bounded_float(
    key: str,
    raw: str,
    *,
    lo: float,
    hi: float,
) -> float:
    val = _parse_float(key, raw)
    if not (lo <= val <= hi):
        raise ConfigError(
            f"{key}: expected float in [{lo}, {hi}]; got {val}"
        )
    return val


def _bounded_positive_int(
    key: str,
    raw: str,
    *,
    lo: int = 1,
    hi: int | None = None,
) -> int:
    val = _parse_int(key, raw)
    if val < lo:
        raise ConfigError(
            f"{key}: must be >= {lo}; got {val}"
        )
    if hi is not None and val > hi:
        raise ConfigError(
            f"{key}: must be <= {hi}; got {val}"
        )
    return val


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigEntry:
    """Schema definition for one ``/config`` key.

    The dispatcher reads this to validate user input and produce
    error messages and ``/help`` text. Adding a new ``/config``
    key is one new :class:`ConfigEntry` registered in
    :data:`CONFIG_SCHEMA`.
    """

    key: str
    """Public-facing key name (e.g. ``"temperature"``)."""

    default: Any
    """Built-in default. Used when the chat session starts and no
    user override has been applied."""

    parse: Callable[[str], Any]
    """String → typed value parser. Raises :class:`ConfigError`
    on malformed input."""

    summary: str
    """One-line description shown by ``/help`` and ``/config``."""

    valid_help: str = ""
    """Optional second line describing the valid range / choices."""


# Closed-form choice tuples surfaced both to the parser and to
# /help. Defining them as module constants keeps documentation and
# validation in lockstep.
THINKING_DISPLAY_CHOICES: tuple[str, ...] = ("auto", "show", "hidden")
THINKING_HISTORY_CHOICES: tuple[str, ...] = ("strip", "keep")
LIVE_TOOLBAR_CHOICES: tuple[str, ...] = ("on", "off")


CONFIG_SCHEMA: dict[str, ConfigEntry] = {
    "temperature": ConfigEntry(
        key="temperature",
        default=0.7,
        parse=lambda raw: _bounded_float(
            "temperature", raw, lo=0.0, hi=2.0
        ),
        summary="sampling temperature",
        valid_help="float in [0.0, 2.0]; 0.0 triggers greedy argmax",
    ),
    "top_p": ConfigEntry(
        key="top_p",
        default=0.9,
        parse=lambda raw: _bounded_float(
            "top_p", raw, lo=0.0, hi=1.0
        ),
        summary="nucleus sampling threshold",
        valid_help="float in [0.0, 1.0]; 1.0 disables nucleus filtering",
    ),
    "top_k": ConfigEntry(
        key="top_k",
        default=None,
        parse=lambda raw: _parse_optional_int("top_k", raw),
        summary="top-k filter cardinality",
        valid_help="positive int, or 'none'/'off' to disable",
    ),
    "max_tokens": ConfigEntry(
        key="max_tokens",
        default=8192,
        parse=lambda raw: _bounded_positive_int(
            "max_tokens", raw, lo=1, hi=32768
        ),
        summary="per-turn output token ceiling",
        valid_help="positive int up to 32768",
    ),
    "thinking": ConfigEntry(
        key="thinking",
        default="show",
        parse=lambda raw: _parse_choice(
            "thinking", raw, THINKING_DISPLAY_CHOICES
        ),
        summary="thinking-block visibility during stream",
        valid_help=(
            "show = stream inline dimmed-italic (default); "
            "auto = collapse during stream + /expand afterwards; "
            "hidden = silent (no spinner, no expand)"
        ),
    ),
    "thinking_mode": ConfigEntry(
        key="thinking_mode",
        default=True,
        parse=lambda raw: _parse_bool("thinking_mode", raw),
        summary="propagate enable_thinking through the chat template",
        valid_help=(
            "on enables reasoning (default — Qwen3 / Qwen3.5 / "
            "Gemma 4 RL-trained reasoning models behave best with "
            "thinking on; the v1.7.36 default flip to off produced "
            "broken text-completion output on Qwen3.5 and Gemma 4 "
            "and was reverted at v1.7.37); off disables for faster "
            "TTFT and shorter total tokens, opt-in only"
        ),
    ),
    "thinking_history": ConfigEntry(
        key="thinking_history",
        default="strip",
        parse=lambda raw: _parse_choice(
            "thinking_history", raw, THINKING_HISTORY_CHOICES
        ),
        summary="strip <think> from assistant message before next turn",
        valid_help=(
            "strip = drop reasoning from history so subsequent prompts "
            "stay short (default); keep = preserve raw decoded reply "
            "verbatim — useful for /save-then-archive workflows"
        ),
    ),
    "live_toolbar": ConfigEntry(
        key="live_toolbar",
        default="off",
        parse=lambda raw: _parse_choice(
            "live_toolbar", raw, LIVE_TOOLBAR_CHOICES
        ),
        summary="opt-in live bottom toolbar during generation",
        valid_help=(
            "off (default) = post-turn toolbar only via "
            "PromptSession.bottom_toolbar; on = AnsiLiveToolbar "
            "redraws per-token. Off by default because cursor "
            "save/restore is unreliable across terminal × prompt-"
            "toolkit interactions; SILICA_LIVE_TOOLBAR=1 env var "
            "is the higher-priority opt-in path. NO_COLOR=1 still "
            "forces off regardless."
        ),
    ),
    "kv_codec_hint_mb": ConfigEntry(
        key="kv_codec_hint_mb",
        default=200.0,
        parse=lambda raw: _bounded_float(
            "kv_codec_hint_mb", raw, lo=0.0, hi=1e6
        ),
        summary="prefix-store size threshold for the codec hint",
        valid_help=(
            "float in MB; when the prefix store exceeds this size "
            "under fp16 the toolbar suggests --kv-codec block_tq_b64_b4"
        ),
    ),
}


# ---------------------------------------------------------------------------
# Parsing API
# ---------------------------------------------------------------------------


def parse_config_assignment(line: str) -> tuple[str, Any]:
    """Parse a ``/config`` assignment of the form ``"key=value"``.

    The argument is what comes after ``/config`` on the REPL line
    (with leading and trailing whitespace stripped by the caller).
    Returns ``(key, typed_value)`` ready to merge into
    :attr:`ChatCliState.config`. Raises :class:`ConfigError` on:

    - missing ``=`` separator,
    - unknown key,
    - per-entry parser/validator failure.

    Assignment grammar is permissive on whitespace
    (``temperature = 0.5`` parses the same as ``temperature=0.5``)
    but strict on shape — a bare ``temperature`` with no ``=`` is
    rejected so the user sees the assignment requirement clearly.
    """
    if "=" not in line:
        raise ConfigError(
            "expected key=value (e.g. /config temperature=0.5)"
        )
    key, _, raw_value = line.partition("=")
    key = key.strip()
    raw_value = raw_value.strip()
    if not key:
        raise ConfigError("missing key on left-hand side of '='")
    entry = CONFIG_SCHEMA.get(key)
    if entry is None:
        allowed = ", ".join(sorted(CONFIG_SCHEMA))
        raise ConfigError(
            f"unknown config key {key!r}; valid keys: {allowed}"
        )
    try:
        value = entry.parse(raw_value)
    except ConfigError:
        raise
    except Exception as exc:  # defensive — parser bug, not user
        raise ConfigError(
            f"{key}: parser raised unexpectedly: {exc}"
        ) from exc
    return entry.key, value


def get_default(key: str) -> Any:
    """Return the schema default for ``key``. Raises
    :class:`KeyError` for unknown keys (configuration is not the
    place for silent fallbacks)."""
    return CONFIG_SCHEMA[key].default


def initial_config() -> dict[str, Any]:
    """Return a fresh dict mapping every schema key to its default
    value. Suitable for seeding :attr:`ChatCliState.config` at
    session start."""
    return {entry.key: entry.default for entry in CONFIG_SCHEMA.values()}


def render_schema_help() -> list[str]:
    """Return a sequence of ``/help``-style lines describing every
    config key, default, and valid range. The dispatcher prints
    these under ``/help`` and ``/config`` (no args)."""
    lines: list[str] = []
    for entry in CONFIG_SCHEMA.values():
        default_repr = (
            "none" if entry.default is None else repr(entry.default)
        )
        lines.append(
            f"  {entry.key} = {default_repr}  — {entry.summary}"
        )
        if entry.valid_help:
            lines.append(f"      {entry.valid_help}")
    return lines


__all__ = [
    "CONFIG_SCHEMA",
    "ConfigEntry",
    "ConfigError",
    "LIVE_TOOLBAR_CHOICES",
    "THINKING_DISPLAY_CHOICES",
    "THINKING_HISTORY_CHOICES",
    "get_default",
    "initial_config",
    "parse_config_assignment",
    "render_schema_help",
]
