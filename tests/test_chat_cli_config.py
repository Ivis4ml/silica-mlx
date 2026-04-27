"""C-2 unit tests for the chat-CLI ``/config`` parser + schema.

Pure-Python coverage of ``silica.chat.cli.config``. No engine, no
prompt_toolkit. Every parse path verified, including the error
cases the dispatcher surfaces back to the user.
"""

from __future__ import annotations

import pytest

from silica.chat.cli.config import (
    CONFIG_SCHEMA,
    ConfigError,
    get_default,
    initial_config,
    parse_config_assignment,
    render_schema_help,
)

# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------


def test_schema_has_documented_keys() -> None:
    """Every key the design doc §6 enumerates must be in the schema.
    A drift here means either the doc or the schema is out of date."""
    expected = {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "thinking",
        "thinking_mode",
        "kv_codec_hint_mb",
    }
    assert expected.issubset(set(CONFIG_SCHEMA.keys())), (
        f"missing schema keys: {expected - set(CONFIG_SCHEMA.keys())}"
    )


def test_initial_config_returns_a_fresh_dict() -> None:
    """Successive calls must return independent dicts so callers
    can mutate one without polluting another."""
    a = initial_config()
    b = initial_config()
    assert a == b
    a["temperature"] = 0.0
    assert b["temperature"] != 0.0


def test_initial_config_includes_every_schema_key() -> None:
    cfg = initial_config()
    assert set(cfg.keys()) == set(CONFIG_SCHEMA.keys())


def test_get_default_round_trips_for_every_schema_key() -> None:
    cfg = initial_config()
    for key in CONFIG_SCHEMA:
        assert get_default(key) == cfg[key]


def test_get_default_unknown_key_raises_keyerror() -> None:
    """Configuration is not the place for silent fallbacks; an
    unknown key is a programmer bug."""
    with pytest.raises(KeyError):
        get_default("not_a_real_key")


# ---------------------------------------------------------------------------
# Float bounds — temperature / top_p / kv_codec_hint_mb
# ---------------------------------------------------------------------------


def test_temperature_parses_in_range() -> None:
    key, val = parse_config_assignment("temperature=0.5")
    assert key == "temperature"
    assert val == pytest.approx(0.5)


def test_temperature_zero_allowed_for_greedy() -> None:
    """temperature=0.0 must be a valid value (greedy decoding)."""
    _key, val = parse_config_assignment("temperature=0.0")
    assert val == pytest.approx(0.0)


def test_temperature_above_2_rejected() -> None:
    with pytest.raises(ConfigError, match=r"temperature.*\[0\.0, 2\.0\]"):
        parse_config_assignment("temperature=2.5")


def test_temperature_negative_rejected() -> None:
    with pytest.raises(ConfigError, match=r"temperature"):
        parse_config_assignment("temperature=-0.1")


def test_temperature_non_numeric_rejected() -> None:
    with pytest.raises(ConfigError, match="expected a float"):
        parse_config_assignment("temperature=hot")


def test_top_p_in_range() -> None:
    _, val = parse_config_assignment("top_p=0.95")
    assert val == pytest.approx(0.95)


def test_top_p_out_of_range_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config_assignment("top_p=1.5")


# ---------------------------------------------------------------------------
# Optional int — top_k
# ---------------------------------------------------------------------------


def test_top_k_integer_value() -> None:
    _, val = parse_config_assignment("top_k=40")
    assert val == 40


def test_top_k_none_disables() -> None:
    """``top_k=none`` (or ``off`` or empty) must coerce to None
    so a power user can disable top-k filtering."""
    for raw in ("none", "off", "None", "OFF", ""):
        _, val = parse_config_assignment(f"top_k={raw}")
        assert val is None, f"expected None for top_k={raw!r}, got {val!r}"


def test_top_k_garbage_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_config_assignment("top_k=many")


# ---------------------------------------------------------------------------
# Bounded positive int — max_tokens
# ---------------------------------------------------------------------------


def test_max_tokens_positive() -> None:
    _, val = parse_config_assignment("max_tokens=2048")
    assert val == 2048


def test_max_tokens_zero_rejected() -> None:
    with pytest.raises(ConfigError, match=">= 1"):
        parse_config_assignment("max_tokens=0")


def test_max_tokens_above_ceiling_rejected() -> None:
    with pytest.raises(ConfigError, match="<= 32768"):
        parse_config_assignment("max_tokens=99999")


# ---------------------------------------------------------------------------
# Choice — thinking display
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("choice", ["auto", "show", "hidden"])
def test_thinking_display_choices_accepted(choice: str) -> None:
    _, val = parse_config_assignment(f"thinking={choice}")
    assert val == choice


def test_thinking_invalid_choice_rejected() -> None:
    with pytest.raises(ConfigError, match="expected one of"):
        parse_config_assignment("thinking=collapsed")


# ---------------------------------------------------------------------------
# Bool — thinking_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("on", True),
        ("off", False),
        ("true", True),
        ("false", False),
        ("yes", True),
        ("no", False),
        ("1", True),
        ("0", False),
        ("ON", True),  # case-insensitive
        ("Off", False),
    ],
)
def test_thinking_mode_bool_aliases(raw: str, expected: bool) -> None:
    _, val = parse_config_assignment(f"thinking_mode={raw}")
    assert val is expected


def test_thinking_mode_garbage_rejected() -> None:
    with pytest.raises(ConfigError, match="expected on/off"):
        parse_config_assignment("thinking_mode=maybe")


# ---------------------------------------------------------------------------
# Assignment grammar
# ---------------------------------------------------------------------------


def test_missing_equals_raises() -> None:
    with pytest.raises(ConfigError, match="key=value"):
        parse_config_assignment("temperature 0.5")


def test_empty_key_raises() -> None:
    with pytest.raises(ConfigError, match="missing key"):
        parse_config_assignment("=0.5")


def test_unknown_key_raises_with_valid_list() -> None:
    """Error must enumerate the valid keys so the user can recover
    without consulting the docs separately."""
    with pytest.raises(ConfigError, match="valid keys") as excinfo:
        parse_config_assignment("frobinate=42")
    # Spot-check that a real key is mentioned in the suggestion list.
    assert "temperature" in str(excinfo.value)


def test_whitespace_around_equals_tolerated() -> None:
    """``temperature = 0.5`` and ``temperature=0.5`` parse the same
    way — common typo path."""
    _, val_loose = parse_config_assignment("temperature = 0.5")
    _, val_tight = parse_config_assignment("temperature=0.5")
    assert val_loose == val_tight


# ---------------------------------------------------------------------------
# Help rendering
# ---------------------------------------------------------------------------


def test_render_schema_help_lists_every_key() -> None:
    lines = render_schema_help()
    rendered = "\n".join(lines)
    for key in CONFIG_SCHEMA:
        assert key in rendered, f"schema key {key!r} missing from /help"


def test_render_schema_help_includes_default_repr() -> None:
    lines = render_schema_help()
    rendered = "\n".join(lines)
    # Defaults shown verbatim; check a couple of representative
    # entries.
    assert "0.7" in rendered  # temperature
    assert "1024" in rendered  # max_tokens
    assert "'auto'" in rendered  # thinking
    assert "none" in rendered  # top_k default
