"""Tests for the ``silica serve`` subcommand (P-8 sub-unit (a3)).

Covers argparse round-trip + the v0.1 single-process invariant
(:func:`silica.server.cli._validate_serve_args` rejecting ``--workers
> 1`` and ``--reload``) + a smoke test that ``_serve`` calls
:func:`silica.server.openai_api.configure` with the right config and
hands the *app object* (not the import-string form) to
``uvicorn.run`` so the in-process ``configure()`` state stays bound
to the running app.
"""

from __future__ import annotations

import pytest

# (a3) tests share the [serve]-extra dependency footprint with (a2);
# skip the whole module cleanly when those imports are missing rather
# than collecting an import error.
pytest.importorskip("fastapi", reason="P-8 [serve] extra not installed")
pytest.importorskip("uvicorn", reason="P-8 [serve] extra not installed")

import argparse  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from typing import Any  # noqa: E402

import uvicorn  # noqa: E402

from silica.server import cli, openai_api  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_module_config() -> Iterator[None]:
    """Wipe :data:`openai_api._config` between tests so a leak from one
    case does not flip the lifespan path of the next case."""
    openai_api._config = None
    yield
    openai_api._config = None


def test_serve_subparser_parses_flags() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--model", "Qwen/Qwen3.5-0.8B",
            "--host", "0.0.0.0",
            "--port", "9999",
            "--log-level", "debug",
        ]
    )

    assert args.cmd == "serve"
    assert args.model == "Qwen/Qwen3.5-0.8B"
    assert args.host == "0.0.0.0"
    assert args.port == 9999
    assert args.log_level == "debug"
    assert args.workers == 1
    assert args.reload is False


def test_serve_requires_model_flag() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve"])


def test_validate_serve_args_accepts_default_invariants() -> None:
    args = argparse.Namespace(workers=1, reload=False)
    assert cli._validate_serve_args(args) is None


def test_validate_serve_args_rejects_workers_above_one() -> None:
    args = argparse.Namespace(workers=2, reload=False)
    msg = cli._validate_serve_args(args)
    assert msg is not None
    assert "--workers=2" in msg
    assert "single-process" in msg


def test_validate_serve_args_rejects_reload() -> None:
    args = argparse.Namespace(workers=1, reload=True)
    msg = cli._validate_serve_args(args)
    assert msg is not None
    assert "--reload" in msg
    assert "configure()" in msg


def test_serve_returns_2_and_prints_error_on_workers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["serve", "--model", "stub", "--workers", "2"]
    )

    rc = cli._serve(args)

    assert rc == 2
    captured = capsys.readouterr()
    assert "silica serve: error:" in captured.err
    assert "--workers=2" in captured.err
    # Configure must NOT have run.
    assert openai_api._config is None


def test_serve_returns_2_and_prints_error_on_reload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        ["serve", "--model", "stub", "--reload"]
    )

    rc = cli._serve(args)

    assert rc == 2
    captured = capsys.readouterr()
    assert "--reload" in captured.err
    assert openai_api._config is None


def test_serve_smoke_configures_then_runs_uvicorn_with_app_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_serve`` calls ``openai_api.configure`` with the right config
    and hands the live app object (not the import-string form) to
    ``uvicorn.run``. This pin protects the (a3) review's hard
    requirement that the module-level ``configure()`` state stays
    bound to the running app."""

    captured: dict[str, Any] = {}

    def _fake_uvicorn_run(app: Any, **kwargs: Any) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", _fake_uvicorn_run)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--model", "Qwen/Qwen3.5-0.8B",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--log-level", "warning",
        ]
    )

    rc = cli._serve(args)

    assert rc == 0
    # configure() ran with the parsed model_repo and no test seam.
    assert openai_api._config is not None
    assert openai_api._config.model_repo == "Qwen/Qwen3.5-0.8B"
    assert openai_api._config.runtime_factory is None
    # uvicorn.run received the live app OBJECT (object identity check).
    # If the (a3) implementation regressed to the import-string form
    # "silica.server.openai_api:app", captured["app"] would be a str
    # and this identity assertion would fail.
    assert captured["app"] is openai_api.app
    # Host / port / log-level propagated.
    assert captured["kwargs"]["host"] == "127.0.0.1"
    assert captured["kwargs"]["port"] == 8001
    assert captured["kwargs"]["log_level"] == "warning"
