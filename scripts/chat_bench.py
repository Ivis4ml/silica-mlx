"""scripts/chat_bench.py — non-interactive chat-REPL metric harness.

CHAT-CLI-HARDENING-7. Drives ``ChatSession`` against a real
model for ``--turns`` shared-context turns and reports per-turn
metrics + the Q-012 cross-call prefix-reuse verdict. The library
implementation lives in :mod:`silica.bench.chat_bench`; this
wrapper supplies the real adapter / engine factories and renders
the report.

Run::

    # Default: Qwen3-0.6B, 3 turns, fp16, text report on stdout
    python scripts/chat_bench.py

    # Explicit model + codec + machine-readable output
    python scripts/chat_bench.py \\
        --model Qwen/Qwen3.5-4B \\
        --kv-codec block_tq_b64_b4 \\
        --json

    # Supply custom prompts (one per turn)
    python scripts/chat_bench.py \\
        --turns 2 \\
        --prompt "Explain LLM inference latency tradeoffs." \\
        --prompt "Now apply that to Apple Silicon specifically."

Exit codes:
  * 0 — Q-012 verdict ``passed`` or ``n/a`` (single-turn run).
  * 1 — Q-012 verdict ``failed``.
  * 2 — argument parsing rejected the invocation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Ensure the repo root is importable when run as
# ``python scripts/chat_bench.py`` from any cwd — mirrors every
# other scripts/*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silica.bench.chat_bench import (  # noqa: E402
    render_json_report,
    render_text_report,
    run_chat_bench,
)

_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
_DEFAULT_TURNS = 3
_DEFAULT_MAX_TOKENS = 64


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="silica-chat-bench",
        description=(
            "Non-interactive chat-REPL metric harness. Runs a "
            "shared-context multi-turn conversation against a real "
            "model and reports per-turn TTFT, decode tok/s, prefix-"
            "cache hit, and the Q-012 cross-call reuse verdict."
        ),
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"HF repo id (default: {_DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--kv-codec",
        default=None,
        choices=("block_tq_b64_b4",),
        help=(
            "KV codec id; default fp16 (omit the flag). Pass "
            "'block_tq_b64_b4' to exercise the codec path through "
            "the harness."
        ),
    )
    p.add_argument(
        "--turns",
        type=int,
        default=_DEFAULT_TURNS,
        help=(
            f"Number of conversation turns (default: {_DEFAULT_TURNS}). "
            "Must be >= 1; the Q-012 verdict needs >= 2 to fire."
        ),
    )
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help=(
            "Per-turn user prompt. Repeat once per turn. When omitted, "
            "the harness uses a built-in deep-learning thread that "
            "naturally accumulates a long shared prefix across turns."
        ),
    )
    p.add_argument(
        "--system",
        default=None,
        help="System prompt (optional).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=_DEFAULT_MAX_TOKENS,
        help=(
            f"Per-turn max tokens (default: {_DEFAULT_MAX_TOKENS}). "
            "Larger values exercise more decode + grow the cached "
            "prefix faster across turns."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature (default: 0.0 = greedy, fully "
            "deterministic across runs)."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed when temperature > 0 (default: 42).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of the text table.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.turns < 1:
        sys.stderr.write("--turns must be >= 1\n")
        return 2
    user_prompts = args.prompt
    if user_prompts is not None and len(user_prompts) != args.turns:
        sys.stderr.write(
            f"--prompt was supplied {len(user_prompts)} times "
            f"but --turns is {args.turns}; counts must match\n"
        )
        return 2

    report = run_chat_bench(
        model=args.model,
        codec_id=args.kv_codec,
        n_turns=args.turns,
        user_prompts=user_prompts,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    if args.json:
        sys.stdout.write(render_json_report(report) + "\n")
    else:
        sys.stdout.write(render_text_report(report) + "\n")
    sys.stdout.flush()

    if report.q012_verdict == "failed":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
