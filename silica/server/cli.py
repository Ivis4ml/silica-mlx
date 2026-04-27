"""silica.server.cli — command-line driver.

Invocation:

    silica run --model Qwen/Qwen3.5-0.8B --prompt "The capital of France is"
    silica chat --model Qwen/Qwen3-0.6B
    silica chat --model Qwen/Qwen3.5-4B --kv-codec block_tq_b64_b4

Two subcommands:

- ``run`` — single-shot generation. P-1 acceptance surface: pulls
  EOS from the tokenizer's ``eos_token_ids`` automatically, prints
  the concatenated prompt + decoded generation to stdout, one-line
  metrics row to stderr.
- ``chat`` — interactive REPL via :mod:`silica.chat.cli.app` (Side
  track 2, C-3+). Persistent bottom toolbar surfacing the
  silica-mlx USP signals (tok/s, KV compression ratio, prefix-cache
  hit count). Slash commands (``/help``, ``/config key=value``,
  ``/system``, ``/reset``, ...) for mid-session control. Sampling
  knobs are intentionally not CLI flags here — the chat-app default
  is "open and use", power users override mid-session via
  ``/config``.

Bare ``silica`` (no subcommand) opens the chat REPL with all
defaults — claude-style. Implementation: :func:`main` pre-processes
``argv`` and prepends ``chat`` when the first positional token is
not a known subcommand (``run`` / ``chat``) and not a top-level
flag (``--help`` / ``-h`` / ``--version``). Both ``silica`` and
``silica chat`` therefore land on the same subparser, so
``--model`` / ``--system`` / ``--kv-codec`` work uniformly across
the two invocation forms.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.factory import adapter_for_repo

_DEFAULT_CHAT_MODEL = "Qwen/Qwen3-0.6B"


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="silica")
    sub = root.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="generate text from a prompt")
    run.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id (e.g. Qwen/Qwen3.5-0.8B)",
    )
    run.add_argument("--prompt", required=True, help="prompt text")
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 triggers greedy argmax; >0 activates the processor chain",
    )
    run.add_argument("--top-p", type=float, default=None)
    run.add_argument("--top-k", type=int, default=None)
    run.add_argument("--seed", type=int, default=None)

    chat = sub.add_parser(
        "chat",
        help="open the interactive chat REPL with bottom status bar",
    )
    chat.add_argument(
        "--model",
        default=_DEFAULT_CHAT_MODEL,
        help=(
            "HuggingFace repo id (default: "
            f"{_DEFAULT_CHAT_MODEL}). Sampling knobs (temperature, "
            "top_p, top_k, max_tokens) are NOT CLI flags here — "
            "set them mid-session via /config inside the REPL."
        ),
    )
    chat.add_argument(
        "--system",
        default=None,
        help="system prompt prepended to the conversation (optional)",
    )
    chat.add_argument(
        "--kv-codec",
        default=None,
        choices=(None, "block_tq_b64_b4"),
        help=(
            "KV codec id; default is fp16 (no compression). Use "
            "'block_tq_b64_b4' for 3.8x KV compression — best on "
            "long sessions / RAG / multi-doc workloads. See "
            "design doc §6 for when this is worth opting into."
        ),
    )
    return root


def _run(args: argparse.Namespace) -> int:
    adapter, kv = adapter_for_repo(args.model)
    engine = Engine(adapter, kv)
    tokenizer = adapter.tokenizer()
    eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))

    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop_token_ids=eos_ids,
        seed=args.seed,
    )

    generated: list[int] = list(engine.generate(args.prompt, params))
    print(args.prompt + tokenizer.decode(generated))
    _print_metrics(engine.metrics.snapshot())
    return 0


def _print_metrics(snapshot: Any) -> None:
    """Print engine metrics to stderr as a single line, skipping None fields."""
    parts: list[str] = []
    if snapshot.ttft_ms is not None:
        parts.append(f"ttft={snapshot.ttft_ms:.1f}ms")
    if snapshot.prefill_tok_s is not None:
        parts.append(f"prefill={snapshot.prefill_tok_s:.1f}tok/s")
    if snapshot.decode_tok_s is not None:
        parts.append(f"decode={snapshot.decode_tok_s:.1f}tok/s")
    if snapshot.resident_mb is not None:
        parts.append(f"resident={snapshot.resident_mb:.1f}MB")
    if parts:
        print("[metrics] " + " ".join(parts), file=sys.stderr)


_KNOWN_SUBCOMMANDS = frozenset({"run", "chat"})
_TOP_LEVEL_FLAGS = frozenset({"--help", "-h", "--version"})


def _preprocess_argv(argv: Sequence[str]) -> list[str]:
    """Prepend ``chat`` when ``argv`` does not lead with a known
    subcommand or a top-level flag.

    Bare ``silica`` (empty argv), ``silica --model X``, and
    ``silica --kv-codec foo`` all become ``silica chat ...``. The
    explicit ``silica chat ...`` and ``silica run ...`` paths are
    untouched. Top-level help / version flags pass through so
    ``silica --help`` lists the subcommands rather than running
    chat with ``--help`` swallowed by the chat subparser.
    """
    args = list(argv)
    if not args:
        return ["chat"]
    head = args[0]
    if head in _KNOWN_SUBCOMMANDS or head in _TOP_LEVEL_FLAGS:
        return args
    return ["chat", *args]


def main(argv: Sequence[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(_preprocess_argv(raw))
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "chat":
        from silica.chat.cli.app import run_chat

        return run_chat(args)
    parser.error(f"unknown command: {args.cmd!r}")
    return 2  # unreachable — parser.error raises SystemExit, kept for mypy


if __name__ == "__main__":
    sys.exit(main())
