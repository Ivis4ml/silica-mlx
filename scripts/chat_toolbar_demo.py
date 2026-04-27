"""Manual visual demo for the C-1 chat CLI toolbar formatter.

Renders the bottom toolbar in a handful of canonical states so you
can see how it looks in your terminal before C-3 wires it into the
live prompt_toolkit Application. No engine, no MLX — just calls
``render_toolbar`` with hand-crafted ``ChatCliState`` snapshots.

Run::

    uv run python scripts/chat_toolbar_demo.py

The script auto-detects palette (truecolor / 256-colour / plain).
Force a specific mode with ``--mode``:

    uv run python scripts/chat_toolbar_demo.py --mode plain
    uv run python scripts/chat_toolbar_demo.py --mode truecolor
    uv run python scripts/chat_toolbar_demo.py --mode indexed-256

Or honour the standard ``NO_COLOR`` env var:

    NO_COLOR=1 uv run python scripts/chat_toolbar_demo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from silica.chat.cli.palette import (  # noqa: E402
    Palette,
    PaletteMode,
    detect_palette,
)
from silica.chat.cli.state import ChatCliState, StreamState  # noqa: E402
from silica.chat.cli.toolbar import render_codec_hint, render_toolbar  # noqa: E402


def _scenarios() -> list[tuple[str, ChatCliState]]:
    """Canonical state snapshots covering the typical toolbar
    appearances during a chat session."""
    return [
        (
            "1. fresh launch (idle, no codec, nothing measured yet)",
            ChatCliState(model_name="Qwen3-0.6B"),
        ),
        (
            "2. prefill — engine consuming the first user prompt",
            ChatCliState(
                model_name="Qwen3-0.6B",
                stream_state=StreamState.PREFILL,
                turn=1,
            ),
        ),
        (
            "3. thinking — model inside a <think>...</think> block",
            ChatCliState(
                model_name="Qwen3-0.6B",
                stream_state=StreamState.THINKING,
                turn=1,
                last_ttft_ms=22.1,
                peak_memory_mb=1240.5,
            ),
        ),
        (
            "4. decode — mid-stream, fp16 KV (default)",
            ChatCliState(
                model_name="Qwen3-0.6B",
                stream_state=StreamState.DECODE,
                turn=1,
                tokens_generated=42,
                max_tokens=1024,
                tok_per_sec=147.3,
                last_ttft_ms=22.1,
                peak_memory_mb=1240.5,
                kv_resident_mb=29.4,
                kv_logical_mb=29.4,  # codec=fp16 → no compression
            ),
        ),
        (
            "5. decode — codec block_tq_b64_b4 active (compr=3.8x in orange)",
            ChatCliState(
                model_name="Qwen3.5-4B",
                stream_state=StreamState.DECODE,
                turn=2,
                tokens_generated=128,
                max_tokens=1024,
                tok_per_sec=86.7,
                last_ttft_ms=51.4,
                peak_memory_mb=9355.1,
                kv_resident_mb=29.4,
                kv_logical_mb=111.7,  # 3.80x
                codec_id="block_tq_b64_b4",
            ),
        ),
        (
            "6. decode — Tier 2 preview, prefix_hit=128/640 visible",
            ChatCliState(
                model_name="Qwen3-0.6B",
                stream_state=StreamState.DECODE,
                turn=4,
                tokens_generated=212,
                max_tokens=1024,
                tok_per_sec=152.8,
                last_ttft_ms=8.3,  # ttft drops because prefix is cached
                peak_memory_mb=1456.2,
                kv_resident_mb=42.1,
                kv_logical_mb=42.1,
                prefix_hit_blocks=128,
                prefix_hit_max=640,
            ),
        ),
        (
            "7. idle — between turns, all stats from last turn visible",
            ChatCliState(
                model_name="Qwen3-0.6B",
                stream_state=StreamState.IDLE,
                turn=4,
                tokens_generated=0,
                max_tokens=1024,
                tok_per_sec=None,  # decoder stopped
                last_ttft_ms=8.3,
                peak_memory_mb=1456.2,
                kv_resident_mb=42.1,
                kv_logical_mb=42.1,
                prefix_hit_blocks=128,
                prefix_hit_max=640,
            ),
        ),
    ]


def _hint_demo() -> ChatCliState:
    """State that triggers the 200 MB prefix-store codec hint."""
    return ChatCliState(
        model_name="Qwen3-0.6B",
        stream_state=StreamState.IDLE,
        turn=15,
        last_ttft_ms=18.2,
        peak_memory_mb=2200.0,
        kv_resident_mb=312.0,
        kv_logical_mb=312.0,
        prefix_store_mb=312.0,  # crossed 200 MB threshold
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visual demo of the C-1 toolbar formatter."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "plain", "indexed-256", "truecolor"),
        default="auto",
        help="palette mode override (default: auto-detect)",
    )
    args = parser.parse_args(argv)

    if args.mode == "auto":
        palette = detect_palette()
    elif args.mode == "plain":
        palette = Palette.plain()
    elif args.mode == "indexed-256":
        palette = Palette.indexed_256()
    elif args.mode == "truecolor":
        palette = Palette.truecolor()
    else:
        # argparse already restricts choices, defensive only.
        palette = Palette.plain()

    print(f"Palette mode: {palette.mode.value}")
    print("(force a mode with --mode plain|indexed-256|truecolor)\n")

    for label, state in _scenarios():
        print(label)
        print("  " + render_toolbar(state, palette=palette))
        print()

    # Codec-hint demo — distinct from the toolbar line.
    hint_state = _hint_demo()
    print("Codec hint (rendered above the toolbar when fp16 prefix store crosses 200MB):")
    print("  " + render_toolbar(hint_state, palette=palette))
    hint = render_codec_hint(hint_state, palette=palette)
    if hint is not None:
        print("  " + hint)
    print()

    # Side-by-side plain vs current mode for the colour-purity invariant.
    if palette.mode is not PaletteMode.PLAIN:
        showcase_state = _scenarios()[4][1]  # codec-active row
        print("Colour purity check (plain vs current mode for the codec row):")
        print("  plain   : " + render_toolbar(showcase_state, palette=Palette.plain()))
        print("  current : " + render_toolbar(showcase_state, palette=palette))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
