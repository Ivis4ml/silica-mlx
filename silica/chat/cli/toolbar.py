"""silica.chat.cli.toolbar — bottom-status-bar formatter.

Pure function from a :class:`ChatCliState` snapshot to a string.
The string is the concatenation of toolbar fields in the order
defined in ``plans/CHAT_CLI_OPENING.md`` §4, with optional ANSI
colour escapes from a :class:`Palette`. No side effects, no I/O
— suitable for unit testing without the prompt_toolkit event loop.

Field rendering rules:

- Fields with a ``None`` numeric source render as ``"—"`` (em-dash)
  so the toolbar layout stays stable across the full state space.
- Numeric fields use a tight format (one decimal for tok/s, ms,
  MB; integer for token counts and prefix-hit blocks).
- The ``compr=`` field appears only when
  :meth:`ChatCliState.compression_ratio` is not ``None``; under
  fp16 default it surfaces as ``"—"`` (the design choice to make
  the codec capability discoverable without implying compression
  is happening).
- The ``prefix_hit=`` field appears only when
  :meth:`ChatCliState.has_prefix_hit` is true; otherwise it shows
  as ``"—"`` so the layout does not shift when prefix-cache reuse
  starts mid-session.
- The ``MLX`` badge is always present; it is the brand affordance
  per design doc §5.

The state-colour map is local to this module so any future palette
tweak (e.g. swapping cyan-1 for cyan-2 for the architectural-win
fields) lives in one place.
"""

from __future__ import annotations

from typing import Any

from silica.chat.cli.palette import ColorName, Palette
from silica.chat.cli.state import ChatCliState, StreamState

# State → colour lookup. Keys are the StreamState values; the
# colour is the foreground used to render the ``state=<value>``
# token. Misses are rendered in the default foreground so a future
# state addition does not crash the formatter.
_STATE_COLOURS: dict[StreamState, ColorName] = {
    StreamState.IDLE: "grey",
    StreamState.PREFILL: "yellow",
    StreamState.THINKING: "magenta",
    StreamState.DECODE: "green",
    StreamState.PAUSED: "grey",
}


# Em-dash placeholder used for unknown numeric fields. Single
# constant so tests can match against it.
_EMDASH = "—"


def _format_state(state: StreamState, palette: Palette) -> str:
    colour = _STATE_COLOURS.get(state, "default")
    label = state.value
    return f"state={palette.colorize(label, colour, bold=True)}"


def _format_model(name: str, palette: Palette) -> str:
    return f"model={palette.colorize(name, 'white')}"


def _format_mlx_badge(palette: Palette) -> str:
    badge = palette.colorize("MLX", "orange", bold=True)
    return badge


def _format_tok_per_sec(state: ChatCliState, palette: Palette) -> str:
    if state.tok_per_sec is None:
        return f"tok/s={_EMDASH}"
    value = palette.colorize(f"{state.tok_per_sec:.1f}", "cyan", bold=True)
    return f"tok/s={value}"


def _format_tokens(state: ChatCliState, palette: Palette) -> str:
    if state.max_tokens > 0:
        text = f"{state.tokens_generated}/{state.max_tokens}"
    else:
        text = f"{state.tokens_generated}"
    return f"tokens={palette.colorize(text, 'white')}"


def _format_ttft(state: ChatCliState, palette: Palette) -> str:
    if state.last_ttft_ms is None:
        return f"ttft={_EMDASH}"
    return f"ttft={palette.colorize(f'{state.last_ttft_ms:.1f}ms', 'cyan')}"


def _format_peak(state: ChatCliState, palette: Palette) -> str:
    if state.peak_memory_mb is None:
        return f"peak={_EMDASH}"
    mb = state.peak_memory_mb
    if mb >= 1024.0:
        text = f"{mb / 1024.0:.2f}GB"
    else:
        text = f"{mb:.0f}MB"
    return f"peak={palette.colorize(text, 'white', dim=True)}"


def _format_kv_resident(state: ChatCliState, palette: Palette) -> str:
    if state.kv_resident_mb is None:
        return f"kv={_EMDASH}"
    return f"kv={palette.colorize(f'{state.kv_resident_mb:.1f}MB', 'white', dim=True)}"


def _format_kv_logical(state: ChatCliState, palette: Palette) -> str:
    if state.kv_logical_mb is None:
        return f"kv_log={_EMDASH}"
    return (
        f"kv_log={palette.colorize(f'{state.kv_logical_mb:.1f}MB', 'white', dim=True)}"
    )


def _format_compr(state: ChatCliState, palette: Palette) -> str:
    ratio = state.compression_ratio()
    if ratio is None:
        return f"compr={_EMDASH}"
    return f"compr={palette.colorize(f'{ratio:.1f}x', 'orange', bold=True)}"


def _format_prefix_hit(state: ChatCliState, palette: Palette) -> str:
    if not state.has_prefix_hit():
        return f"prefix_hit={_EMDASH}"
    text = f"{state.prefix_hit_blocks}/{state.prefix_hit_max}"
    return f"prefix_hit={palette.colorize(text, 'cyan')}"


def _format_turn(state: ChatCliState, palette: Palette) -> str:
    return f"turn={palette.colorize(str(state.turn), 'white', dim=True)}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_toolbar(
    state: ChatCliState,
    *,
    palette: Palette | None = None,
) -> str:
    """Render the bottom toolbar line for ``state``.

    Returns a single-line string with embedded ANSI escapes when
    ``palette`` enables colour output, or a plain ASCII line when
    ``palette.mode is PaletteMode.PLAIN``. The trailing newline is
    *not* included — the prompt_toolkit Window inserts that itself.

    Field order and content match ``plans/CHAT_CLI_OPENING.md`` §4
    plus the brand ``MLX`` badge. The order is stable so users
    pinning their gaze to a column index always see the same field.

    Args:
        state: live chat-CLI state snapshot. Pure read; no
            mutation.
        palette: rendering palette. ``None`` defaults to a
            no-colour :class:`Palette` so callers that have not
            done capability detection (mostly tests) get a
            deterministic plain rendering.
    """
    p = palette if palette is not None else Palette.plain()

    fields: list[str] = [
        _format_state(state.stream_state, p),
        _format_model(state.model_name, p),
        _format_mlx_badge(p),
        _format_tok_per_sec(state, p),
        _format_tokens(state, p),
        _format_ttft(state, p),
        _format_peak(state, p),
        _format_kv_resident(state, p),
        _format_kv_logical(state, p),
        _format_compr(state, p),
        _format_prefix_hit(state, p),
        _format_turn(state, p),
    ]
    return "  ".join(fields)


def render_codec_hint(
    state: ChatCliState,
    *,
    palette: Palette | None = None,
    threshold_mb: float = 200.0,
) -> str | None:
    """Return the soft hint suggesting ``--kv-codec block_tq_b64_b4``
    when prefix store growth crosses ``threshold_mb`` under fp16.

    Returns ``None`` (do not render) when:

    - A codec is already active (``state.codec_id is not None``); or
    - Prefix store size is unknown (``state.prefix_store_mb is None``); or
    - Prefix store size is below the threshold.

    Otherwise returns a single-line string suitable for printing
    above or beside the toolbar. Per design doc §6.2 the hint is
    informative, not directive — it signals the codec capability
    is available and would help, without forcing it on.
    """
    if state.codec_id is not None:
        return None
    if state.prefix_store_mb is None or state.prefix_store_mb < threshold_mb:
        return None
    p = palette if palette is not None else Palette.plain()
    body = (
        f"prefix store now {state.prefix_store_mb:.0f}MB — "
        f"--kv-codec block_tq_b64_b4 would compress this ~3.8x"
    )
    return p.colorize(f"tip: {body}", "yellow", dim=True)


def render_showcase(
    state: ChatCliState,
    *,
    palette: Palette | None = None,
) -> str:
    """Render the ``/showcase`` session-narrative report.

    A multi-line block summarising the silica-mlx-USP signals
    accumulated over the session so far: turn count, prefix-cache
    reuse, peak device memory, average decode throughput, and (when
    a KV codec is active) compression-driven memory savings.

    Empty / pre-first-turn states are surfaced honestly — fields
    read ``—`` rather than ``0`` when they have not been measured
    yet, so the user can distinguish "no data" from "zero".
    """
    p = palette if palette is not None else Palette.plain()
    lines: list[str] = []
    title = p.colorize(
        "── silica session showcase ──",
        "cyan",
        bold=True,
    )
    lines.append(title)

    turns_str = str(state.turn) if state.turn > 0 else "—"
    lines.append(f"  turns:           {turns_str}")
    lines.append(f"  model:           {state.model_name}")
    lines.append(f"  codec:           {state.codec_id or 'fp16'}")

    if state.total_prefix_hit_tokens > 0:
        lines.append(
            f"  prefix reused:   {state.total_prefix_hit_tokens} tokens"
        )
    else:
        lines.append("  prefix reused:   —")

    avg = state.session_avg_decode_tok_s()
    if avg is not None:
        lines.append(f"  avg decode:      {avg:.1f} tok/s")
    else:
        lines.append("  avg decode:      —")

    if state.last_ttft_ms is not None:
        lines.append(f"  last TTFT:       {state.last_ttft_ms:.0f} ms")
    else:
        lines.append("  last TTFT:       —")

    if state.peak_memory_mb is not None:
        lines.append(
            f"  peak memory:     {state.peak_memory_mb / 1024.0:.2f} GB"
        )
    else:
        lines.append("  peak memory:     —")

    if state.kv_resident_mb is not None:
        lines.append(
            f"  prefix store:    {state.kv_resident_mb:.1f} MB"
        )
    else:
        lines.append("  prefix store:    —")

    ratio = state.compression_ratio()
    if (
        ratio is not None
        and state.kv_logical_mb is not None
        and state.kv_resident_mb is not None
    ):
        saved = state.kv_logical_mb - state.kv_resident_mb
        lines.append(
            f"  codec savings:   {saved:.1f} MB ({ratio:.1f}x vs fp16)"
        )

    return "\n".join(lines)


# Surface the public symbols cleanly so other CLI modules import
# from this file rather than reaching into private helpers.
__all__: list[str] = [
    "render_toolbar",
    "render_codec_hint",
    "render_showcase",
]


def _annotate_state_for_repr(state: ChatCliState) -> dict[str, Any]:
    """Debug helper — turn a state into a flat dict for ``print()``
    inspection. Used by manual smoke testing of the formatter, not
    by production code paths."""
    return {
        "stream_state": state.stream_state.value,
        "model_name": state.model_name,
        "codec_id": state.codec_id,
        "tok_per_sec": state.tok_per_sec,
        "tokens": f"{state.tokens_generated}/{state.max_tokens}",
        "ttft_ms": state.last_ttft_ms,
        "peak_memory_mb": state.peak_memory_mb,
        "kv_resident_mb": state.kv_resident_mb,
        "kv_logical_mb": state.kv_logical_mb,
        "compr": state.compression_ratio(),
        "prefix_store_mb": state.prefix_store_mb,
        "prefix_hit": (state.prefix_hit_blocks, state.prefix_hit_max),
        "turn": state.turn,
    }
