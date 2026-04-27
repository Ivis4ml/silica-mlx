"""Tests for silica.server.cli — argument parsing only.

End-to-end generation is covered by the P-1 acceptance test (parity against
mlx-lm reference on a fixed prompt / seed); this file only asserts the CLI's
argument contract so future changes don't silently re-shuffle flags.
"""

from __future__ import annotations

import pytest

from silica.server.cli import _preprocess_argv, build_parser


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
        # An unknown token after preprocessing must not slip through
        # the parser. ``_preprocess_argv`` would prepend ``chat`` here
        # and the chat subparser would reject ``doit``; calling the
        # parser directly skips that prepend and is rejected at the
        # subparser-routing step.
        parser.parse_args(["doit"])


# ---------------------------------------------------------------------------
# Bare-``silica`` argv preprocessing (C-7)
# ---------------------------------------------------------------------------


def test_preprocess_empty_argv_routes_to_chat() -> None:
    assert _preprocess_argv([]) == ["chat"]


def test_preprocess_explicit_chat_passes_through() -> None:
    assert _preprocess_argv(["chat", "--model", "m"]) == [
        "chat",
        "--model",
        "m",
    ]


def test_preprocess_explicit_run_passes_through() -> None:
    assert _preprocess_argv(["run", "--model", "m", "--prompt", "p"]) == [
        "run",
        "--model",
        "m",
        "--prompt",
        "p",
    ]


def test_preprocess_bare_flag_prepends_chat() -> None:
    """``silica --model X`` becomes ``silica chat --model X``."""
    assert _preprocess_argv(["--model", "Qwen/Qwen3-4B"]) == [
        "chat",
        "--model",
        "Qwen/Qwen3-4B",
    ]


def test_preprocess_kv_codec_alone_prepends_chat() -> None:
    assert _preprocess_argv(["--kv-codec", "block_tq_b64_b4"]) == [
        "chat",
        "--kv-codec",
        "block_tq_b64_b4",
    ]


def test_preprocess_top_level_help_passes_through() -> None:
    """``silica --help`` should hit the *root* parser so it lists
    subcommands; do not prepend ``chat``."""
    assert _preprocess_argv(["--help"]) == ["--help"]
    assert _preprocess_argv(["-h"]) == ["-h"]


def test_preprocess_full_chat_argv_via_parser() -> None:
    """End-to-end: bare argv goes through preprocessing and parses
    cleanly on the chat subparser."""
    parser = build_parser()
    args = parser.parse_args(_preprocess_argv(["--model", "Qwen/Qwen3-4B"]))
    assert args.cmd == "chat"
    assert args.model == "Qwen/Qwen3-4B"


def test_preprocess_empty_argv_via_parser_uses_chat_default_model() -> None:
    parser = build_parser()
    args = parser.parse_args(_preprocess_argv([]))
    assert args.cmd == "chat"
    assert args.model == "Qwen/Qwen3-0.6B"
