"""scripts/chat.py — script alias for ``silica chat``.

Thin wrapper that delegates to :func:`silica.chat.cli.app.run_chat`,
exposing the same launch surface. Kept under ``scripts/`` for users
with muscle memory; the canonical entry point post-C-3 is
``silica chat`` (or, after C-7, the bare ``silica`` claude-style).

Run::

    python scripts/chat.py --model Qwen/Qwen3-0.6B
    python scripts/chat.py --model Qwen/Qwen3.5-4B --kv-codec block_tq_b64_b4
    python scripts/chat.py --model Qwen/Qwen3-0.6B --system "You are concise."

Sampling knobs (``temperature``, ``top_p``, ``top_k``,
``max_tokens``) are NOT CLI flags. Adjust them mid-session via
``/config key=value`` once the REPL is open. Run ``/help`` inside
the REPL for the full command list.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Ensure the repo root is importable when run as ``python scripts/chat.py``
# from any cwd — mirrors every other scripts/*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silica.chat.cli.app import run_chat  # noqa: E402

_DEFAULT_CHAT_MODEL = "Qwen/Qwen3-0.6B"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="silica-chat",
        description=(
            "Open the silica-mlx chat REPL with persistent bottom "
            "toolbar (tok/s, KV, compression, prefix-hit). Equivalent "
            "to `silica chat` once the package is installed."
        ),
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_CHAT_MODEL,
        help=(
            f"HF repo id (default: {_DEFAULT_CHAT_MODEL}). "
            "Sampling knobs are mid-session via /config, not CLI flags."
        ),
    )
    p.add_argument(
        "--system",
        default=None,
        help="system prompt (optional)",
    )
    p.add_argument(
        "--kv-codec",
        default=None,
        choices=(None, "block_tq_b64_b4"),
        help=(
            "KV codec id; default fp16. Use 'block_tq_b64_b4' for "
            "long-session / multi-doc workloads (3.8x KV compression)."
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_chat(args)


if __name__ == "__main__":
    sys.exit(main())
