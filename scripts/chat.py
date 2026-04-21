"""scripts/chat.py — minimal REPL chatbot over silica.engine.Engine.

Invocation:

    python scripts/chat.py --model Qwen/Qwen3-0.6B
    python scripts/chat.py --model Qwen/Qwen3-0.6B \
        --system "You are a concise assistant." \
        --temperature 0.7 --top-p 0.9 --max-tokens 256
    python scripts/chat.py --model Qwen/Qwen3-0.6B --no-stream

REPL commands:

  * ``/reset`` — drop all messages except the system prompt.
  * ``/stats`` — print cumulative session metrics.
  * ``/exit`` (or EOF / Ctrl-C) — quit.

Every turn prints the assistant reply followed by a single stderr
line with TTFT, prefill tok/s, decode tok/s, resident KV MB, peak
MB, logical KV MB, token count, wall seconds, and the finish
reason — so the chat loop doubles as an engine benchmark.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Ensure the repo root is importable when run as ``python scripts/chat.py``
# from any cwd — mirrors every other scripts/*.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from silica.chat.session import (  # noqa: E402
    ChatSession,
    TurnMetrics,
)
from silica.core.sampling import SamplingParams  # noqa: E402
from silica.engine import Engine  # noqa: E402
from silica.models.factory import adapter_for_repo  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="silica-chat",
        description=(
            "Minimal REPL chatbot over silica.engine.Engine. "
            "Loads one model, keeps a message history, streams the "
            "reply, prints per-turn TTFT / tok-s / KV / peak memory."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        help="HF repo id (e.g. Qwen/Qwen3-0.6B)",
    )
    p.add_argument(
        "--system",
        default=None,
        help=(
            "system prompt prepended to the message list (optional)"
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="sampling temperature (0.0 triggers greedy argmax)",
    )
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument(
        "--no-stream",
        action="store_true",
        help=(
            "disable token streaming; reply is printed only after "
            "generation completes"
        ),
    )
    return p


@dataclass
class _SessionTotals:
    """Cumulative session stats for the /stats command."""

    turns: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0
    ttft_ms_sum: float = 0.0
    ttft_ms_count: int = 0

    def record(self, m: TurnMetrics) -> None:
        self.turns += 1
        self.prompt_tokens += m.prompt_tokens
        self.output_tokens += m.output_tokens
        if m.wall_s is not None:
            self.wall_s += m.wall_s
        if m.ttft_ms is not None:
            self.ttft_ms_sum += m.ttft_ms
            self.ttft_ms_count += 1

    def render(self) -> str:
        avg_ttft = (
            f"{self.ttft_ms_sum / self.ttft_ms_count:.1f}ms"
            if self.ttft_ms_count
            else "n/a"
        )
        avg_decode = (
            f"{self.output_tokens / self.wall_s:.1f}tok/s"
            if self.wall_s
            else "n/a"
        )
        return (
            f"turns={self.turns} "
            f"prompt_tokens={self.prompt_tokens} "
            f"output_tokens={self.output_tokens} "
            f"wall={self.wall_s:.2f}s "
            f"avg_ttft={avg_ttft} "
            f"avg_decode={avg_decode}"
        )


def _print_turn_metrics(m: TurnMetrics) -> None:
    parts: list[str] = []
    if m.ttft_ms is not None:
        parts.append(f"ttft={m.ttft_ms:.1f}ms")
    if m.prefill_tok_s is not None:
        parts.append(f"prefill={m.prefill_tok_s:.1f}tok/s")
    if m.decode_tok_s is not None:
        parts.append(f"decode={m.decode_tok_s:.1f}tok/s")
    if m.resident_mb is not None:
        parts.append(f"resident_kv={m.resident_mb:.1f}MB")
    if m.peak_memory_mb is not None:
        parts.append(f"peak={m.peak_memory_mb:.1f}MB")
    if m.logical_kv_bytes is not None:
        parts.append(f"logical_kv={m.logical_kv_bytes / 1e6:.1f}MB")
    parts.append(f"prompt={m.prompt_tokens}")
    parts.append(f"out={m.output_tokens}")
    if m.wall_s is not None:
        parts.append(f"wall={m.wall_s:.2f}s")
    parts.append(f"finish={m.finish_reason}")
    print("[" + " ".join(parts) + "]", file=sys.stderr)


def _stream_stdout(delta: str) -> None:
    sys.stdout.write(delta)
    sys.stdout.flush()


def _read_user_input() -> str | None:
    try:
        line = input(">>> ")
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None
    return line


def _run_repl(session: ChatSession, args: argparse.Namespace) -> int:
    totals = _SessionTotals()
    stream_fn = None if args.no_stream else _stream_stdout
    print(
        "silica-chat ready. Commands: /reset  /stats  /exit  "
        "(or EOF / Ctrl-C)",
        file=sys.stderr,
    )
    while True:
        text = _read_user_input()
        if text is None:
            break
        user_text = text.strip()
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            session.reset()
            totals = _SessionTotals()
            print("(session reset)", file=sys.stderr)
            continue
        if user_text == "/stats":
            print("[session] " + totals.render(), file=sys.stderr)
            continue

        params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
        metrics = session.chat(
            user_text,
            sampling_params=params,
            stream_to=stream_fn,
        )
        if args.no_stream:
            sys.stdout.write(metrics.reply)
        sys.stdout.write("\n")
        sys.stdout.flush()
        _print_turn_metrics(metrics)
        totals.record(metrics)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"Loading {args.model} ...", file=sys.stderr)
    adapter, kv = adapter_for_repo(args.model)
    engine = Engine(adapter, kv)
    session = ChatSession(adapter, engine, system_prompt=args.system)
    return _run_repl(session, args)


if __name__ == "__main__":
    sys.exit(main())
