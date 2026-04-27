"""silica.chat.cli.persistence — JSON serialisation for sessions.

Implements the file format behind ``/save <path>`` and
``/load <path>``. The on-disk schema is plain JSON, designed to
be human-inspectable: a curious user opens the file in ``cat`` and
reads the conversation back. No binary blobs, no engine state, no
KV.

Fields:

- ``schema_version`` — integer; bumped on incompatible changes.
- ``model`` — HF repo id the session was running against. ``/load``
  *records* it but does not auto-swap models; the user re-launches
  with the right ``--model`` (or runs ``/model <repo>`` first).
- ``codec_id`` — KV codec id, or ``null`` for fp16.
- ``messages`` — the chat session's message history, role+content
  pairs. The shape matches ``ChatSession.messages``.
- ``config`` — the runtime config snapshot from ``ChatCliState.config``.
- ``saved_at`` — ISO-8601 UTC timestamp the file was written.

``/load`` is permissive on the input: any extra fields are ignored
so a forward-compatible file from a later silica version still
restores the messages portion. Schema-version mismatches are
flagged but do not block restoration of the basic fields.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class SessionFileError(RuntimeError):
    """Raised when the session file cannot be read or its contents
    do not match the expected JSON shape (top-level dict, ``messages``
    list of role+content pairs).

    This is the single error type ``/save`` and ``/load`` surface to
    the chat shell — the shell renders it as a red feedback line.
    """


def serialize_session(
    *,
    model: str,
    codec_id: str | None,
    messages: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON-serialisable dict for a session snapshot.

    Pure function — no I/O. Caller writes the result with
    :func:`save_session` (or :func:`json.dumps` directly for tests).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model,
        "codec_id": codec_id,
        "messages": list(messages),
        "config": dict(config),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_session(
    path: str | Path,
    *,
    model: str,
    codec_id: str | None,
    messages: list[dict[str, str]],
    config: dict[str, Any],
) -> Path:
    """Write the session snapshot to ``path`` as pretty-printed JSON.

    Parent directories are created when missing — ``/save`` should
    just-work even on a fresh ``~/sessions/`` path. Returns the
    resolved :class:`Path` so the caller can echo the absolute
    location back to the user.
    """
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_session(
        model=model,
        codec_id=codec_id,
        messages=messages,
        config=config,
    )
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return p


def load_session(path: str | Path) -> dict[str, Any]:
    """Read a session snapshot back from disk.

    Returns the parsed dict with at least the ``messages`` /
    ``config`` / ``model`` / ``codec_id`` keys present. Missing
    keys default to empty values rather than raising — a half-
    written file is still partially usable.

    Raises :class:`SessionFileError` when the file does not exist,
    is unreadable, or is not a JSON object with a ``messages`` list.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise SessionFileError(f"file not found: {p}")
    try:
        text = p.read_text()
    except OSError as exc:
        raise SessionFileError(f"cannot read {p}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SessionFileError(f"invalid JSON in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise SessionFileError(
            f"expected JSON object at top level of {p}, got {type(data).__name__}"
        )
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        raise SessionFileError(
            f"`messages` must be a list in {p}, got {type(messages).__name__}"
        )
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise SessionFileError(
                f"messages[{i}] in {p} is not a {{role, content}} dict"
            )
    return {
        "schema_version": data.get("schema_version"),
        "model": data.get("model", ""),
        "codec_id": data.get("codec_id"),
        "messages": messages,
        "config": data.get("config") or {},
        "saved_at": data.get("saved_at"),
    }


__all__ = [
    "SCHEMA_VERSION",
    "SessionFileError",
    "load_session",
    "save_session",
    "serialize_session",
]
