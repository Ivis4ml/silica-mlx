"""Tests for silica.core.logger — namespace scoping, idempotency, color handling."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest

from silica.core.logger import (
    _SILICA_ROOT,
    _ColorFormatter,
    _SilicaStreamHandler,
    get_logger,
    setup_logging,
)


@pytest.fixture(autouse=True)
def reset_silica_logger() -> Iterator[None]:
    """Save / restore the silica logger state so tests do not leak to each other."""
    root = logging.getLogger(_SILICA_ROOT)
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_propagate = root.propagate
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)
    root.propagate = saved_propagate


def _silica_handlers() -> list[_SilicaStreamHandler]:
    return [
        h for h in logging.getLogger(_SILICA_ROOT).handlers
        if isinstance(h, _SilicaStreamHandler)
    ]


def test_setup_logging_installs_one_handler() -> None:
    setup_logging(level="INFO", use_colors=False, stream=io.StringIO())
    assert len(_silica_handlers()) == 1
    assert logging.getLogger(_SILICA_ROOT).propagate is False


def test_setup_logging_is_idempotent() -> None:
    setup_logging(level="INFO", use_colors=False, stream=io.StringIO())
    setup_logging(level="DEBUG", use_colors=False, stream=io.StringIO())
    setup_logging(level="WARNING", use_colors=False, stream=io.StringIO())
    assert len(_silica_handlers()) == 1
    assert logging.getLogger(_SILICA_ROOT).level == logging.WARNING


def test_setup_logging_preserves_foreign_handlers() -> None:
    # A non-silica handler attached before setup must survive.
    foreign = logging.StreamHandler(io.StringIO())
    logging.getLogger(_SILICA_ROOT).addHandler(foreign)
    setup_logging(level="INFO", use_colors=False, stream=io.StringIO())
    assert foreign in logging.getLogger(_SILICA_ROOT).handlers


def test_get_logger_passes_silica_names_through() -> None:
    assert get_logger("silica").name == "silica"
    assert get_logger("silica.core.sampler").name == "silica.core.sampler"


def test_get_logger_prefixes_external_names() -> None:
    # __name__ in a silica module already starts with "silica.", but callers
    # passing anything else should get prefixed so stray modules still land
    # under the configured namespace.
    assert get_logger("foo.bar").name == "silica.foo.bar"


def _make_record(level: int = logging.WARNING, msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="silica.test",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=None,
        exc_info=None,
    )


def test_color_formatter_emits_ansi_when_enabled() -> None:
    out = _ColorFormatter(use_color=True).format(_make_record())
    assert "\x1b[" in out
    assert "WARNING" in out
    assert "hello" in out


def test_color_formatter_plain_when_disabled() -> None:
    out = _ColorFormatter(use_color=False).format(_make_record())
    assert "\x1b[" not in out
    assert "WARNING" in out
    assert "hello" in out


def test_end_to_end_log_record_reaches_stream() -> None:
    buf = io.StringIO()
    setup_logging(level="INFO", use_colors=False, stream=buf)
    get_logger("silica.test").info("engine ready")
    logged = buf.getvalue()
    assert "engine ready" in logged
    assert "INFO" in logged
    assert "silica.test" in logged


def test_no_color_env_disables_auto_color(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    stream = FakeTty()
    setup_logging(level="INFO", stream=stream)  # use_colors=None -> auto-detect
    get_logger("silica.test").info("colorless")
    assert "\x1b[" not in stream.getvalue()


def test_tty_stream_enables_auto_color(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    stream = FakeTty()
    setup_logging(level="INFO", stream=stream)
    get_logger("silica.test").warning("attention")
    assert "\x1b[" in stream.getvalue()
