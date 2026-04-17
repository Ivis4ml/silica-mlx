"""silica.server.cli — P-1 command-line driver.

Invocation:

    python -m silica run --model Qwen/Qwen3.5-0.8B --prompt "The capital of France is"

Or, if the ``silica`` console script is installed (``pip install -e .``):

    silica run --model Qwen/Qwen3.5-0.8B --prompt "..."

Minimal for P-1 acceptance:
  - One subcommand (``run``).
  - EOS is pulled from the tokenizer's ``eos_token_ids`` and fed into
    ``SamplingParams.stop_token_ids`` automatically.
  - Output is the prompt concatenated with the decoded generation; no
    incremental streaming yet (clean incremental decoding needs the
    ``BPEStreamingDetokenizer`` plumbing which is a polish item).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.qwen3 import Qwen3Adapter


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
    return root


def _run(args: argparse.Namespace) -> int:
    adapter, kv = Qwen3Adapter.from_hf_repo(args.model)
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
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    parser.error(f"unknown command: {args.cmd!r}")
    return 2  # unreachable — parser.error raises SystemExit, kept for mypy


if __name__ == "__main__":
    sys.exit(main())
