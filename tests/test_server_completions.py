"""Tests for POST /v1/completions (P-8 sub-unit (e)).

Covers the non-streaming round-trip, the unsupported-field 501 set
(``stream=True``, non-string prompt, list shapes, ``stop``,
``logprobs``, ``echo``, ``best_of``, ``suffix``, ``presence_penalty``,
``frequency_penalty``, ``logit_bias``, ``n>1``, non-empty
``extension``), the sampling-bound 400 set (temperature, top_p,
max_tokens), the model-id 404 path, the readiness 503 path, and
the EOS-aware finish-reason + text-stripping invariants the route
inherits from :meth:`ChatSession.chat`.

The route runs MLX compute via
:func:`silica.server.routes.completions._drive_generate`, which
returns the raw output token ids + prompt-token count. Tests
monkeypatch that seam to inject deterministic id sequences. To
exercise the EOS strip + finish-reason classification path, the
tests also override the runtime tokenizer's ``decode`` and inject
``eos_token_ids`` so the route's strip + decode logic operates on
realistic inputs without driving a real model.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="P-8 [serve] extra not installed")
pytest.importorskip("httpx", reason="fastapi.testclient requires httpx")

from fastapi.testclient import TestClient  # noqa: E402

from silica.core.profiler import MetricsRegistry  # noqa: E402
from silica.kvcache.manager import NullKVManager  # noqa: E402
from silica.models.adapter import StubModelAdapter  # noqa: E402
from silica.server import openai_api  # noqa: E402
from silica.server.routes import completions as cc  # noqa: E402
from silica.server.runtime import Runtime  # noqa: E402


def _build_stub_runtime(
    *, model_repo: str = "stub/model", created_at: int = 1_700_000_000
) -> Runtime:
    return Runtime(
        StubModelAdapter(),
        NullKVManager(),
        model_repo=model_repo,
        metrics=MetricsRegistry(),
        created_at=created_at,
    )


@pytest.fixture(autouse=True)
def _isolate_module_state() -> Iterator[None]:
    openai_api._config = None
    yield
    openai_api._config = None


def _configure_runtime() -> Runtime:
    runtime = _build_stub_runtime()
    openai_api.configure(
        openai_api.ServerConfig(runtime_factory=lambda: runtime)
    )
    return runtime


def _install_drive_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_ids: list[int],
    prompt_tokens: int,
) -> dict[str, Any]:
    """Monkeypatch :func:`_drive_generate` to return a deterministic
    ``(output_ids, prompt_tokens)`` tuple. The route's strip + decode
    + classify path runs as written; tests use
    :func:`_install_decode_table` below to control the detokeniser
    when the response text matters."""
    captured: dict[str, Any] = {"calls": []}

    def _stub(
        runtime: Runtime, prompt: str, params: Any
    ) -> tuple[list[int], int]:
        captured["calls"].append({"prompt": prompt, "params": params})
        return list(output_ids), prompt_tokens

    monkeypatch.setattr(cc, "_drive_generate", _stub)
    return captured


def _install_decode_table(
    runtime: Runtime,
    *,
    decode_map: dict[tuple[int, ...], str] | None = None,
    eos_ids: set[int] | None = None,
) -> None:
    """Override the stub tokenizer's ``decode`` and ``eos_token_ids``
    so the route's EOS-strip + ``rstrip("�")`` + classify path
    operates on realistic inputs. The default ``StubModelAdapter``
    tokenizer decodes everything to the empty string, so without
    this seam the response text would always be empty regardless of
    the injected token ids."""
    tokenizer = runtime.adapter.tokenizer()
    if eos_ids is not None:
        tokenizer.eos_token_ids = set(eos_ids)  # type: ignore[attr-defined]
    if decode_map is not None:

        def _decode(token_ids: Any) -> str:
            return decode_map.get(tuple(int(t) for t in token_ids), "")

        tokenizer.decode = _decode  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# 200 round trip + Usage / finish-reason wiring.
# ---------------------------------------------------------------------------


def test_completions_round_trip_returns_strict_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _configure_runtime()
    _install_decode_table(
        runtime,
        decode_map={(101, 102): " Paris."},
    )
    captured = _install_drive_stub(
        monkeypatch, output_ids=[101, 102], prompt_tokens=8
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "The capital of France is",
                "max_tokens": 16,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "stub/model"
    assert body["choices"][0]["text"] == " Paris."
    assert body["choices"][0]["index"] == 0
    # 2 output tokens < max_tokens (16) and last token is not EOS =>
    # natural stop.
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["logprobs"] is None
    assert body["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
    }
    assert body["id"].startswith("cmpl-")

    # The stub captured the prompt + sampling params the route sent
    # through.
    assert len(captured["calls"]) == 1
    assert captured["calls"][0]["prompt"] == "The capital of France is"


def test_completions_finish_reason_length_when_at_max_tokens_without_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the engine yields exactly ``max_tokens`` non-EOS tokens,
    the response must report ``finish_reason='length'``."""
    runtime = _configure_runtime()
    seq = list(range(200, 216))  # 16 ids, none in the EOS set
    _install_decode_table(
        runtime, decode_map={tuple(seq): "x" * 16}, eos_ids={2}
    )
    _install_drive_stub(monkeypatch, output_ids=seq, prompt_tokens=4)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "go",
                "max_tokens": 16,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 16


def test_completions_default_max_tokens_finishes_stop_on_short_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``max_tokens`` is unset, the route falls back to silica's
    256 default; a short stub output therefore reports ``stop``."""
    runtime = _configure_runtime()
    _install_decode_table(runtime, decode_map={(7,): "hi"})
    _install_drive_stub(monkeypatch, output_ids=[7], prompt_tokens=1)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "go",
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 404 model-id mismatch.
# ---------------------------------------------------------------------------


def test_completions_404_when_model_id_does_not_match() -> None:
    _configure_runtime()

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "other/model",
                "prompt": "hi",
            },
        )

    assert response.status_code == 404
    detail = response.json()["error"]["message"]
    assert "other/model" in detail
    assert "stub/model" in detail


# ---------------------------------------------------------------------------
# 503 readiness path.
# ---------------------------------------------------------------------------


def test_completions_503_when_runtime_closed() -> None:
    runtime = _configure_runtime()

    with TestClient(openai_api.app) as client:
        runtime.close()
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "hi",
            },
        )

    assert response.status_code == 503
    assert response.json() == {"error": {"message": "engine not ready", "type": "server_error"}}


# ---------------------------------------------------------------------------
# 501 unsupported fields (parametrised — one assertion per field).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_field",
    [
        {"stream": True},
        {"stop": "###"},
        {"stop": ["a", "b"]},
        {"logprobs": 1},
        {"suffix": " end."},
        {"echo": True},
        {"best_of": 2},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
        {"logit_bias": {"42": -100}},
        {"n": 2},
        {"extension": {"thinking_mode": "off"}},
    ],
)
def test_completions_501_for_unsupported_fields(
    extra_field: dict[str, Any],
) -> None:
    """The schema parses unknown / unsupported OpenAI fields under
    ``extra='allow'``; the route is the rejection site."""
    _configure_runtime()
    body = {
        "model": "stub/model",
        "prompt": "hi",
        **extra_field,
    }

    with TestClient(openai_api.app) as client:
        response = client.post("/v1/completions", json=body)

    assert response.status_code == 501, response.text


@pytest.mark.parametrize(
    "non_str_prompt",
    [
        ["a", "b"],
        [1, 2, 3],
        [[1, 2], [3, 4]],
    ],
)
def test_completions_501_for_non_string_prompt(
    non_str_prompt: Any,
) -> None:
    """v0.1 only accepts a single string prompt. The schema parses
    the union but the route 501s on every other shape."""
    _configure_runtime()

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": non_str_prompt,
            },
        )

    assert response.status_code == 501


# ---------------------------------------------------------------------------
# 400 sampling-bound rejections.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("temperature", -0.1),
        ("temperature", 2.5),
        ("top_p", 0.0),
        ("top_p", 1.5),
        ("max_tokens", 0),
        ("max_tokens", -3),
    ],
)
def test_completions_400_for_out_of_range_sampling(
    field: str, value: Any
) -> None:
    _configure_runtime()
    body: dict[str, Any] = {
        "model": "stub/model",
        "prompt": "hi",
        field: value,
    }

    with TestClient(openai_api.app) as client:
        response = client.post("/v1/completions", json=body)

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# EOS handling: trailing-EOS strip + finish_reason classification.
# These pin the engine contract from silica/engine/__init__.py — the
# generator yields the stop token before terminating, and the route
# must (a) drop it before decoding the response text, (b) report
# finish_reason="stop" even when the EOS landed on the max_tokens-th
# yielded position. Mirrors ChatSession's _classify_finish + the
# decode strip at session.py:660-668.
# ---------------------------------------------------------------------------


def test_completions_strips_trailing_eos_from_response_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine.generate yields the stop token before terminating;
    the route must strip it before decoding so the response text
    does not surface the literal ``<|im_end|>``-style marker."""
    runtime = _configure_runtime()
    # Decode table: full sequence (including EOS=999) decodes to a
    # body with the marker; stripped sequence decodes to clean text.
    _install_decode_table(
        runtime,
        decode_map={
            (101, 102, 999): "Paris.<|im_end|>",
            (101, 102): "Paris.",
        },
        eos_ids={999},
    )
    _install_drive_stub(
        monkeypatch, output_ids=[101, 102, 999], prompt_tokens=4
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "go",
                "max_tokens": 16,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["text"] == "Paris."
    assert "<|im_end|>" not in body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] == "stop"
    # completion_tokens reflects the full engine output including
    # the trailing EOS — matches ChatSession.TurnMetrics.output_tokens
    # (silica/chat/session.py:799).
    assert body["usage"]["completion_tokens"] == 3


def test_completions_finish_reason_stop_when_eos_lands_on_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load-bearing: when the engine yields EOS as the final allowed
    token (output count == max_tokens), the response must report
    ``finish_reason='stop'`` (the model emitted a natural
    terminator), not ``'length'``. The earlier ``len(output_ids) >=
    max_tokens`` heuristic mis-classified this case and would also
    leak the EOS marker into ``choices[0].text``."""
    runtime = _configure_runtime()
    seq = list(range(200, 215)) + [999]  # 15 regular ids + EOS = 16
    _install_decode_table(
        runtime,
        decode_map={tuple(seq[:-1]): "x" * 15},
        eos_ids={999},
    )
    _install_drive_stub(monkeypatch, output_ids=seq, prompt_tokens=4)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "go",
                "max_tokens": 16,
            },
        )

    body = response.json()
    assert response.status_code == 200
    # Critical: EOS-last beats max_tokens cap.
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["text"] == "x" * 15
    assert body["usage"]["completion_tokens"] == 16


def test_completions_finish_reason_stop_for_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty engine output (e.g. immediate EOS on the prefill sample,
    or a truly empty prompt path) reports ``stop`` with empty text
    and zero completion tokens."""
    runtime = _configure_runtime()
    _install_decode_table(runtime, decode_map={(): ""}, eos_ids={999})
    _install_drive_stub(monkeypatch, output_ids=[], prompt_tokens=2)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "go",
                "max_tokens": 16,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["text"] == ""
    assert body["usage"]["completion_tokens"] == 0


def test_completions_strips_trailing_replacement_char_from_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trailing ``\\ufffd`` in the decoded text marks an incomplete
    multi-byte UTF-8 sequence cut short by EOS / max_tokens. The
    route ``rstrip("\\ufffd")``s after decode, mirroring
    ChatSession.chat (silica/chat/session.py:668)."""
    runtime = _configure_runtime()
    _install_decode_table(
        runtime, decode_map={(50, 51): "hello��"}, eos_ids={999}
    )
    _install_drive_stub(monkeypatch, output_ids=[50, 51], prompt_tokens=2)

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "go",
                "max_tokens": 16,
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["text"] == "hello"


# ---------------------------------------------------------------------------
# White-box pins on the helper functions.
# ---------------------------------------------------------------------------


def test_classify_finish_reason_eos_last_beats_max_tokens_cap() -> None:
    """White-box: ``_classify_finish_reason`` must return ``stop``
    when the last token is EOS, even when ``len(output_ids) ==
    max_tokens``. Pins the EOS-first ordering inside the helper."""
    eos = frozenset({999})
    # 16 tokens, last is EOS, max_tokens=16 — EOS wins.
    assert (
        cc._classify_finish_reason(
            list(range(200, 215)) + [999], eos_ids=eos, max_tokens=16
        )
        == "stop"
    )
    # 16 non-EOS tokens at the cap — length wins.
    assert (
        cc._classify_finish_reason(
            list(range(200, 216)), eos_ids=eos, max_tokens=16
        )
        == "length"
    )
    # Empty output — stop.
    assert cc._classify_finish_reason([], eos_ids=eos, max_tokens=16) == "stop"
    # Short output, no EOS, under cap — stop.
    assert (
        cc._classify_finish_reason([1, 2, 3], eos_ids=eos, max_tokens=16)
        == "stop"
    )


def test_strip_trailing_eos_drops_only_the_last_token_when_eos() -> None:
    """White-box: ``_strip_trailing_eos`` returns the input unchanged
    when the last token is not EOS, and drops exactly the trailing
    EOS otherwise. An EOS in the middle of the sequence is preserved
    (the engine never yields an interior EOS, but the helper's
    contract is "trailing only")."""
    eos = frozenset({999})
    assert cc._strip_trailing_eos([1, 2, 3], eos_ids=eos) == [1, 2, 3]
    assert cc._strip_trailing_eos([1, 2, 999], eos_ids=eos) == [1, 2]
    # Mid-sequence EOS — not stripped (last token is non-EOS).
    assert cc._strip_trailing_eos([999, 1, 2], eos_ids=eos) == [999, 1, 2]
    assert cc._strip_trailing_eos([], eos_ids=eos) == []
    # No EOS configured — never strips.
    assert cc._strip_trailing_eos([1, 2, 3], eos_ids=frozenset()) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Engine.generate is invoked with the eos_token_ids the tokenizer exposes.
# ---------------------------------------------------------------------------


def test_completions_wires_tokenizer_eos_into_sampling_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route must populate ``stop_token_ids`` from the tokenizer's
    ``eos_token_ids`` attribute (when present) so the engine halts on
    natural EOS."""
    runtime = _configure_runtime()
    _install_decode_table(
        runtime, decode_map={(11,): "ok"}, eos_ids={7, 9}
    )

    captured = _install_drive_stub(
        monkeypatch, output_ids=[11], prompt_tokens=1
    )

    with TestClient(openai_api.app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "stub/model",
                "prompt": "go",
            },
        )

    assert response.status_code == 200
    params = captured["calls"][0]["params"]
    assert params.stop_token_ids == (7, 9)
