"""C-1 unit tests for the chat CLI palette / state / toolbar formatter.

Pure-Python coverage of ``silica.chat.cli.{palette,state,toolbar}``.
No prompt_toolkit, no Engine, no MLX — every test constructs
:class:`Palette` and :class:`ChatCliState` directly. Verifies the
documented field rendering rules in
``docs/CHAT_CLI_OPENING.md`` §4 and §6.2.
"""

from __future__ import annotations

import re

import pytest

from silica.chat.cli.palette import (
    Palette,
    PaletteMode,
    detect_palette,
)
from silica.chat.cli.state import ChatCliState, StreamState
from silica.chat.cli.toolbar import (
    render_codec_hint,
    render_showcase,
    render_toolbar,
)

# ---------------------------------------------------------------------------
# Palette detection
# ---------------------------------------------------------------------------


def test_palette_no_color_env_forces_plain() -> None:
    """``NO_COLOR=1`` always forces plain mode regardless of TTY /
    COLORTERM. Per the no-color.org spec any non-empty value
    counts as the opt-out."""
    p = detect_palette(env={"NO_COLOR": "1"}, isatty=True)
    assert p.mode is PaletteMode.PLAIN


def test_palette_no_color_empty_string_does_not_disable() -> None:
    """A literal empty string is treated as 'not set' per spec —
    the env var must be present *and* non-empty to opt out."""
    p = detect_palette(
        env={"NO_COLOR": "", "COLORTERM": "truecolor", "TERM": "xterm-256color"},
        isatty=True,
    )
    assert p.mode is PaletteMode.TRUECOLOR


def test_palette_not_a_tty_falls_back_to_plain() -> None:
    """Piping output to a file should produce no colour codes,
    even when COLORTERM declares truecolor — escapes would
    pollute the captured stream."""
    p = detect_palette(
        env={"COLORTERM": "truecolor", "TERM": "xterm-256color"},
        isatty=False,
    )
    assert p.mode is PaletteMode.PLAIN


def test_palette_truecolor_when_supported() -> None:
    p = detect_palette(
        env={"COLORTERM": "truecolor", "TERM": "xterm-256color"},
        isatty=True,
    )
    assert p.mode is PaletteMode.TRUECOLOR


def test_palette_indexed_256_when_only_term_signals_256color() -> None:
    p = detect_palette(env={"TERM": "xterm-256color"}, isatty=True)
    assert p.mode is PaletteMode.INDEXED_256


def test_palette_dumb_term_falls_back_to_plain() -> None:
    p = detect_palette(env={"TERM": "dumb"}, isatty=True)
    assert p.mode is PaletteMode.PLAIN


def test_palette_unset_term_falls_back_to_plain() -> None:
    p = detect_palette(env={}, isatty=True)
    assert p.mode is PaletteMode.PLAIN


# ---------------------------------------------------------------------------
# Palette colorize
# ---------------------------------------------------------------------------


def test_colorize_plain_mode_returns_text_unchanged() -> None:
    p = Palette.plain()
    assert p.colorize("hello", "green") == "hello"
    assert p.colorize("hello", "orange", bold=True) == "hello"
    assert p.fg("green") == ""
    assert p.reset() == ""


def test_colorize_truecolor_emits_24bit_escape() -> None:
    p = Palette.truecolor()
    out = p.colorize("hello", "green")
    # 24-bit foreground escape: ESC [38;2;R;G;Bm ... ESC [0m
    assert out.startswith("\x1b[38;2;")
    assert out.endswith("\x1b[0m")
    assert "hello" in out


def test_colorize_indexed_256_emits_8bit_escape() -> None:
    p = Palette.indexed_256()
    out = p.colorize("hello", "green")
    # 8-bit indexed foreground: ESC [38;5;Nm
    assert out.startswith("\x1b[38;5;")
    assert out.endswith("\x1b[0m")


def test_colorize_orange_brand_uses_brand_rgb() -> None:
    """The orange brand colour anchors the silica identity (MLX
    badge, codec compression callout). Lock the truecolor RGB
    triple so future palette tweaks register as a deliberate change
    rather than a silent drift."""
    p = Palette.truecolor()
    out = p.colorize("MLX", "orange", bold=True)
    assert "\x1b[38;2;255;153;51m" in out
    assert "\x1b[1m" in out  # bold prefix


def test_colorize_dim_modifier_emits_dim_escape() -> None:
    p = Palette.truecolor()
    out = p.colorize("idle", "grey", dim=True)
    assert "\x1b[2m" in out


# ---------------------------------------------------------------------------
# Toolbar rendering — plain mode (deterministic ASCII)
# ---------------------------------------------------------------------------


def _empty_state() -> ChatCliState:
    """Produce a state at session start: nothing measured yet, no
    codec, no prefix hit. Mirrors what the toolbar shows the moment
    the chat REPL opens, before any turn runs."""
    return ChatCliState(model_name="Qwen3-0.6B")


def test_toolbar_plain_mode_contains_all_static_fields() -> None:
    out = render_toolbar(_empty_state(), palette=Palette.plain())
    # Every field key must be present, in order.
    for key in (
        "state=idle",
        "model=Qwen3-0.6B",
        "MLX",
        "tok/s=",
        "tokens=0/1024",
        "ttft=",
        "peak=",
        "kv=",
        "kv_log=",
        "compr=",
        "prefix_hit=",
        "turn=",
    ):
        assert key in out, f"missing field {key!r} in toolbar: {out!r}"


def test_toolbar_plain_mode_has_no_ansi_escapes() -> None:
    out = render_toolbar(_empty_state(), palette=Palette.plain())
    assert "\x1b[" not in out


def test_toolbar_unknown_numerics_render_as_emdash() -> None:
    out = render_toolbar(_empty_state(), palette=Palette.plain())
    # Under empty state, every optional numeric field is "—".
    for key in ("tok/s=—", "ttft=—", "peak=—", "kv=—", "kv_log=—", "compr=—", "prefix_hit=—"):
        assert key in out, f"missing emdash field {key!r} in: {out!r}"


def test_toolbar_state_transitions_render_each_kind() -> None:
    """Confirm the four primary state values render as expected
    plain text. Colour assertions live in the truecolor tests
    below."""
    for sv in (
        StreamState.IDLE,
        StreamState.PREFILL,
        StreamState.THINKING,
        StreamState.DECODE,
    ):
        st = ChatCliState(model_name="Qwen3-0.6B", stream_state=sv)
        out = render_toolbar(st, palette=Palette.plain())
        assert f"state={sv.value}" in out, (
            f"state value {sv.value} not visible in: {out!r}"
        )


# ---------------------------------------------------------------------------
# Toolbar rendering — codec / compression
# ---------------------------------------------------------------------------


def test_compression_ratio_hidden_under_fp16_default() -> None:
    """Under codec=fp16 (None) with kv_logical == kv_resident, the
    ratio is 1.0×. The design choice (§6.1) is to surface this as
    "—" rather than "1.0x" so the toolbar does not imply
    compression where there is none."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id=None,
        kv_resident_mb=29.4,
        kv_logical_mb=29.4,
    )
    assert st.compression_ratio() is None
    out = render_toolbar(st, palette=Palette.plain())
    assert "compr=—" in out


def test_compression_ratio_shown_when_codec_active() -> None:
    """Under codec=block_tq_b64_b4 with the documented ~3.8x ratio,
    the toolbar surfaces ``compr=3.8x`` as the silica USP signal."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id="block_tq_b64_b4",
        kv_resident_mb=29.4,
        kv_logical_mb=111.7,  # 111.7 / 29.4 = 3.8
    )
    ratio = st.compression_ratio()
    assert ratio is not None
    assert abs(ratio - 3.8) < 0.05
    out = render_toolbar(st, palette=Palette.plain())
    assert "compr=3.8x" in out


def test_compression_ratio_handles_zero_resident_safely() -> None:
    """Pre-first-turn the prefix store may report zero resident
    bytes; division must not blow up."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id="block_tq_b64_b4",
        kv_resident_mb=0.0,
        kv_logical_mb=0.0,
    )
    assert st.compression_ratio() is None
    out = render_toolbar(st, palette=Palette.plain())
    assert "compr=—" in out


# ---------------------------------------------------------------------------
# Toolbar rendering — prefix hit (Tier 2 surface; default false)
# ---------------------------------------------------------------------------


def test_prefix_hit_emdash_when_unset() -> None:
    """Tier 2 (C-4) is not yet wired; until then prefix_hit is None
    on every state. Toolbar must still render as ``—`` so the
    layout stays stable."""
    st = ChatCliState(model_name="Qwen3-0.6B")
    assert st.has_prefix_hit() is False
    assert "prefix_hit=—" in render_toolbar(st, palette=Palette.plain())


def test_prefix_hit_visible_when_populated() -> None:
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        prefix_hit_blocks=128,
        prefix_hit_max=640,
    )
    assert st.has_prefix_hit() is True
    out = render_toolbar(st, palette=Palette.plain())
    assert "prefix_hit=128/640" in out


def test_prefix_hit_zero_max_treated_as_unset() -> None:
    """A zero divisor (``prefix_hit_max=0``) is meaningless; render
    as unset rather than producing ``0/0``."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        prefix_hit_blocks=0,
        prefix_hit_max=0,
    )
    assert st.has_prefix_hit() is False
    assert "prefix_hit=—" in render_toolbar(st, palette=Palette.plain())


# ---------------------------------------------------------------------------
# Toolbar rendering — peak memory unit selection
# ---------------------------------------------------------------------------


def test_peak_memory_under_1gb_renders_as_mb() -> None:
    st = ChatCliState(model_name="Qwen3-0.6B", peak_memory_mb=512.4)
    out = render_toolbar(st, palette=Palette.plain())
    assert "peak=512MB" in out


def test_peak_memory_at_or_above_1gb_renders_as_gb() -> None:
    st = ChatCliState(model_name="Qwen3-0.6B", peak_memory_mb=4096.0)
    out = render_toolbar(st, palette=Palette.plain())
    assert "peak=4.00GB" in out


# ---------------------------------------------------------------------------
# Toolbar rendering — colour mode integration
# ---------------------------------------------------------------------------


def test_truecolor_toolbar_contains_brand_orange_for_mlx_badge() -> None:
    out = render_toolbar(_empty_state(), palette=Palette.truecolor())
    # MLX badge uses the orange RGB anchor.
    assert "\x1b[38;2;255;153;51m" in out
    # And the badge itself is in the output.
    assert "MLX" in out


def test_truecolor_toolbar_state_decode_uses_green() -> None:
    st = ChatCliState(model_name="Qwen3-0.6B", stream_state=StreamState.DECODE)
    out = render_toolbar(st, palette=Palette.truecolor())
    # Decode renders in bright green (RGB 102, 217, 102 per palette).
    assert "\x1b[38;2;102;217;102m" in out


def test_truecolor_toolbar_state_thinking_uses_magenta() -> None:
    st = ChatCliState(model_name="Qwen3-0.6B", stream_state=StreamState.THINKING)
    out = render_toolbar(st, palette=Palette.truecolor())
    # Thinking renders in bright magenta (RGB 217, 102, 217).
    assert "\x1b[38;2;217;102;217m" in out


def test_truecolor_toolbar_resets_after_each_field() -> None:
    """Every coloured field must end with a reset so subsequent
    plain text in the prompt_toolkit Window does not inherit a
    stuck foreground colour."""
    out = render_toolbar(_empty_state(), palette=Palette.truecolor())
    # Count of reset escapes is at least the number of coloured
    # fields (states + MLX badge + tokens + turn at minimum).
    reset_count = out.count("\x1b[0m")
    assert reset_count >= 5, f"too few reset escapes ({reset_count}) in: {out!r}"


# ---------------------------------------------------------------------------
# Codec hint
# ---------------------------------------------------------------------------


def test_codec_hint_silent_when_codec_already_on() -> None:
    """Under an active codec, the hint must not fire — the user
    has already opted in."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id="block_tq_b64_b4",
        prefix_store_mb=512.0,
    )
    assert render_codec_hint(st) is None


def test_codec_hint_silent_below_threshold() -> None:
    """Under fp16 with a small prefix store, no hint."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id=None,
        prefix_store_mb=42.0,
    )
    assert render_codec_hint(st) is None
    # Custom threshold: hint stays silent if still below.
    assert render_codec_hint(st, threshold_mb=100.0) is None


def test_codec_hint_silent_when_size_unknown() -> None:
    """Pre-first-turn the prefix store size is None; no hint."""
    st = ChatCliState(model_name="Qwen3-0.6B", codec_id=None)
    assert render_codec_hint(st) is None


def test_codec_hint_fires_above_threshold() -> None:
    """Under fp16 with prefix store > 200 MB the hint fires."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id=None,
        prefix_store_mb=312.0,
    )
    hint = render_codec_hint(st, palette=Palette.plain())
    assert hint is not None
    assert "block_tq_b64_b4" in hint
    assert "312" in hint  # the actual prefix store size visible


def test_codec_hint_truecolor_uses_yellow() -> None:
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id=None,
        prefix_store_mb=312.0,
    )
    hint = render_codec_hint(st, palette=Palette.truecolor())
    assert hint is not None
    # Yellow RGB anchor (255, 204, 0); used for the hint per design
    # doc — the soft-warning category.
    assert "\x1b[38;2;255;204;0m" in hint


# ---------------------------------------------------------------------------
# Output stability — toolbar field order and width sanity
# ---------------------------------------------------------------------------


def test_toolbar_field_order_is_stable() -> None:
    """The toolbar field order is part of the user contract — pin
    it so a future refactor does not silently shuffle the columns
    a user is used to scanning."""
    out = render_toolbar(_empty_state(), palette=Palette.plain())
    indices = [
        out.index("state="),
        out.index("model="),
        out.index("MLX"),
        out.index("tok/s="),
        out.index("tokens="),
        out.index("ttft="),
        out.index("peak="),
        out.index("kv="),
        out.index("kv_log="),
        out.index("compr="),
        out.index("prefix_hit="),
        out.index("turn="),
    ]
    assert indices == sorted(indices), f"toolbar field order drifted: {out!r}"


def test_toolbar_renders_without_palette_argument() -> None:
    """Default behaviour (no palette argument) is plain, suitable
    for tests and for callers that haven't yet detected colour
    capability — produces no escape codes."""
    out = render_toolbar(_empty_state())
    assert "\x1b[" not in out


@pytest.mark.parametrize(
    "stream_state",
    [
        StreamState.IDLE,
        StreamState.PREFILL,
        StreamState.THINKING,
        StreamState.DECODE,
        StreamState.PAUSED,
    ],
)
def test_toolbar_renders_every_state_without_crashing(
    stream_state: StreamState,
) -> None:
    st = ChatCliState(model_name="Qwen3-0.6B", stream_state=stream_state)
    out_plain = render_toolbar(st, palette=Palette.plain())
    out_color = render_toolbar(st, palette=Palette.truecolor())
    out_indexed = render_toolbar(st, palette=Palette.indexed_256())
    # Every state must produce a non-empty rendering across all
    # palette modes.
    assert len(out_plain) > 0
    assert len(out_color) > 0
    assert len(out_indexed) > 0


# ---------------------------------------------------------------------------
# Mode-stripping invariant
# ---------------------------------------------------------------------------


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def test_truecolor_and_plain_produce_same_text_after_stripping_escapes() -> None:
    """Stripping ANSI escapes from the truecolor rendering must
    produce a string equal to the plain rendering. This pins the
    "colour is purely additive" contract: enabling colour cannot
    change the text content, only its presentation."""
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        stream_state=StreamState.DECODE,
        tok_per_sec=147.3,
        last_ttft_ms=22.1,
        peak_memory_mb=1234.5,
        kv_resident_mb=29.4,
        kv_logical_mb=111.7,
        codec_id="block_tq_b64_b4",
        prefix_hit_blocks=128,
        prefix_hit_max=640,
        turn=3,
        tokens_generated=42,
    )
    plain = render_toolbar(st, palette=Palette.plain())
    color = render_toolbar(st, palette=Palette.truecolor())
    stripped = _ANSI_ESCAPE.sub("", color)
    assert stripped == plain, (
        f"colour rendering changed text content:\n"
        f"  plain   : {plain!r}\n"
        f"  stripped: {stripped!r}"
    )



# ---------------------------------------------------------------------------
# render_showcase — /showcase narrative report
# ---------------------------------------------------------------------------


def test_showcase_pre_first_turn_uses_emdash_for_unknown_fields() -> None:
    """At session start (turn=0, no measurements), unknown fields
    render as ``—`` rather than ``0`` so the user can distinguish
    "not measured" from "measured zero"."""
    out = render_showcase(_empty_state(), palette=Palette.plain())
    assert "turns:" in out
    # turn=0 reads as "—" — there is no turn-zero, the counter
    # increments after the first reply lands.
    assert "turns:           —" in out
    assert "avg decode:      —" in out
    assert "last TTFT:       —" in out
    assert "peak memory:     —" in out
    assert "prefix reused:   —" in out


def test_showcase_renders_cumulative_signals_after_turns() -> None:
    st = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id=None,
        turn=3,
        last_ttft_ms=22.0,
        peak_memory_mb=2048.0,  # 2 GB
        kv_resident_mb=18.4,
        total_prefix_hit_tokens=512,
        total_decode_tokens=600,
        total_decode_seconds=4.0,  # 150 tok/s avg
    )
    out = render_showcase(st, palette=Palette.plain())
    assert "turns:           3" in out
    assert "model:           Qwen3-0.6B" in out
    assert "codec:           fp16" in out
    assert "prefix reused:   512 tokens" in out
    assert "avg decode:      150.0 tok/s" in out
    assert "last TTFT:       22 ms" in out
    assert "peak memory:     2.00 GB" in out
    assert "prefix store:    18.4 MB" in out


def test_showcase_codec_savings_only_when_codec_active() -> None:
    """The "codec savings" line is conditional on a real
    compression ratio. fp16 (no codec) omits the line; an active
    codec includes it with both saved-MB and ratio."""
    fp16_state = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id=None,
        turn=1,
        kv_resident_mb=10.0,
        kv_logical_mb=10.0,
    )
    out_fp16 = render_showcase(fp16_state, palette=Palette.plain())
    assert "codec savings" not in out_fp16

    codec_state = ChatCliState(
        model_name="Qwen3-0.6B",
        codec_id="block_tq_b64_b4",
        turn=1,
        kv_resident_mb=10.0,
        kv_logical_mb=38.0,  # 3.8x ratio
    )
    out_codec = render_showcase(codec_state, palette=Palette.plain())
    assert "codec savings:" in out_codec
    assert "3.8x" in out_codec
    assert "28.0 MB" in out_codec  # 38 - 10


def test_showcase_plain_mode_has_no_ansi_escapes() -> None:
    st = ChatCliState(model_name="Qwen3-0.6B", turn=1)
    out = render_showcase(st, palette=Palette.plain())
    assert "\x1b[" not in out
