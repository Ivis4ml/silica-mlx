"""C-7 unit tests for ``silica.chat.cli.persistence``.

Pure-Python coverage of the JSON session-file format. Verifies
round-trip serialisation, ``/load`` permissiveness on missing or
extra fields, and the error path for malformed files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from silica.chat.cli.persistence import (
    SCHEMA_VERSION,
    SessionFileError,
    load_session,
    save_session,
    serialize_session,
)


def _sample_messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# ---------------------------------------------------------------------------
# serialize_session
# ---------------------------------------------------------------------------


def test_serialize_session_includes_all_fields() -> None:
    payload = serialize_session(
        model="Qwen/Qwen3-0.6B",
        codec_id=None,
        messages=_sample_messages(),
        config={"temperature": "0.7"},
    )
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["model"] == "Qwen/Qwen3-0.6B"
    assert payload["codec_id"] is None
    assert payload["messages"] == _sample_messages()
    assert payload["config"] == {"temperature": "0.7"}
    assert isinstance(payload["saved_at"], str)
    assert payload["saved_at"].endswith("+00:00")


def test_serialize_session_codec_carried_through() -> None:
    payload = serialize_session(
        model="m",
        codec_id="block_tq_b64_b4",
        messages=[],
        config={},
    )
    assert payload["codec_id"] == "block_tq_b64_b4"


def test_serialize_session_does_not_mutate_input() -> None:
    msgs = _sample_messages()
    cfg = {"x": "1"}
    payload = serialize_session(
        model="m", codec_id=None, messages=msgs, config=cfg
    )
    payload["messages"].append({"role": "user", "content": "extra"})
    assert msgs == _sample_messages()
    assert cfg == {"x": "1"}


# ---------------------------------------------------------------------------
# save_session  /  load_session round trip
# ---------------------------------------------------------------------------


def test_round_trip_basic(tmp_path: Path) -> None:
    p = tmp_path / "chat.json"
    save_session(
        p,
        model="Qwen/Qwen3-0.6B",
        codec_id=None,
        messages=_sample_messages(),
        config={"temperature": "0.7", "max_tokens": "1024"},
    )
    loaded = load_session(p)
    assert loaded["model"] == "Qwen/Qwen3-0.6B"
    assert loaded["messages"] == _sample_messages()
    assert loaded["config"] == {
        "temperature": "0.7",
        "max_tokens": "1024",
    }


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "deep" / "chat.json"
    save_session(
        p, model="m", codec_id=None, messages=[], config={}
    )
    assert p.exists()


def test_save_returns_resolved_path(tmp_path: Path) -> None:
    p = tmp_path / "out.json"
    returned = save_session(
        p, model="m", codec_id=None, messages=[], config={}
    )
    assert returned == p


def test_save_writes_pretty_printed_json(tmp_path: Path) -> None:
    """The on-disk file should be human-inspectable (multi-line)."""
    p = tmp_path / "x.json"
    save_session(
        p,
        model="m",
        codec_id=None,
        messages=_sample_messages(),
        config={},
    )
    text = p.read_text()
    assert text.endswith("\n")
    assert "\n  " in text  # indent=2


def test_save_preserves_unicode(tmp_path: Path) -> None:
    """Non-ASCII content must survive the round trip without
    being escaped to ``\\uXXXX`` form (chat sessions routinely
    contain Chinese / emoji)."""
    p = tmp_path / "u.json"
    msgs = [{"role": "user", "content": "你好，世界。"}]
    save_session(p, model="m", codec_id=None, messages=msgs, config={})
    text = p.read_text()
    assert "你好" in text
    loaded = load_session(p)
    assert loaded["messages"] == msgs


# ---------------------------------------------------------------------------
# load_session — permissiveness
# ---------------------------------------------------------------------------


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(SessionFileError, match="file not found"):
        load_session(tmp_path / "absent.json")


def test_load_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not json")
    with pytest.raises(SessionFileError, match="invalid JSON"):
        load_session(p)


def test_load_non_object_top_level_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SessionFileError, match="JSON object"):
        load_session(p)


def test_load_messages_must_be_list(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"messages": "not-a-list"}))
    with pytest.raises(SessionFileError, match="messages.*must be a list"):
        load_session(p)


def test_load_message_must_have_role_and_content(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"messages": [{"role": "user"}]}))
    with pytest.raises(SessionFileError, match="role, content"):
        load_session(p)


def test_load_extra_fields_ignored(tmp_path: Path) -> None:
    """Forward compatibility: a file written by a later silica
    version with extra top-level keys still loads."""
    p = tmp_path / "future.json"
    p.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "model": "m",
                "messages": _sample_messages(),
                "config": {},
                "future_field": {"some": "thing"},
            }
        )
    )
    loaded = load_session(p)
    assert loaded["messages"] == _sample_messages()
    assert loaded["schema_version"] == 99


def test_load_missing_optional_fields_default_to_empty(
    tmp_path: Path,
) -> None:
    p = tmp_path / "minimal.json"
    p.write_text(json.dumps({"messages": []}))
    loaded = load_session(p)
    assert loaded["model"] == ""
    assert loaded["codec_id"] is None
    assert loaded["config"] == {}
    assert loaded["messages"] == []


def test_round_trip_via_filesystem_preserves_payload(
    tmp_path: Path,
) -> None:
    p = tmp_path / "rt.json"
    payload = serialize_session(
        model="Qwen/Qwen3-4B",
        codec_id="block_tq_b64_b4",
        messages=_sample_messages(),
        config={"temperature": "0.0"},
    )
    p.write_text(json.dumps(payload))
    loaded = load_session(p)
    assert loaded["model"] == payload["model"]
    assert loaded["codec_id"] == payload["codec_id"]
    assert loaded["messages"] == payload["messages"]
    assert loaded["config"] == payload["config"]
