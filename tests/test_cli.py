"""Tests for silica.server.cli — argument parsing only.

End-to-end generation is covered by the P-1 acceptance test (parity against
mlx-lm reference on a fixed prompt / seed); this file only asserts the CLI's
argument contract so future changes don't silently re-shuffle flags.
"""

from __future__ import annotations

import pytest

from silica.server.cli import build_parser


def test_run_parses_minimum_required_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--model", "Qwen/Qwen3.5-0.8B", "--prompt", "hi"]
    )
    assert args.cmd == "run"
    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.prompt == "hi"
    assert args.max_tokens == 64
    assert args.temperature == 0.0
    assert args.top_p is None
    assert args.top_k is None
    assert args.seed is None


def test_run_accepts_sampling_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--model",
            "m",
            "--prompt",
            "p",
            "--max-tokens",
            "10",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--top-k",
            "40",
            "--seed",
            "123",
        ]
    )
    assert args.max_tokens == 10
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.top_k == 40
    assert args.seed == 123


def test_missing_model_or_prompt_exits() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--prompt", "hi"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--model", "m"])


def test_unknown_subcommand_exits() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["doit"])
