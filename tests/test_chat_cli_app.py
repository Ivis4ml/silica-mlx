"""App-layer unit tests for ``silica.chat.cli.app``.

CHAT-CLI-HARDENING-8. Pure-Python coverage of the four helper
functions GPT-5.5 named:

- ``_build_prefix_cache`` — codec / store / cache construction.
- ``_sampling_params_from_state`` — config + EOS resolution.
- ``_resolve_thinking_mode`` — bool / non-bool / missing key.
- ``_apply_system_prompt_request`` — tri-state ``/system`` flow.

The two pre-existing helpers (``_build_prefix_cache``,
``_sampling_params_from_state``) are tested via injection; the
two extracted helpers (``_resolve_thinking_mode``,
``_apply_system_prompt_request``) are tested directly. No
prompt_toolkit, no engine, no MLX weights — every test
constructs the inputs directly so failures point at exactly
which helper drifted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import pytest

from silica.chat.cli.app import (
    DEFAULT_SYSTEM_PROMPT,
    _apply_rollback_snapshot,
    _apply_system_prompt_request,
    _assistant_ends_in_thinking,
    _build_prefix_cache,
    _capture_rollback_snapshot,
    _evaluate_continue_request,
    _print_truncation_marker,
    _resolve_initial_system_prompt,
    _resolve_live_toolbar_enabled,
    _resolve_thinking_history,
    _resolve_thinking_mode,
    _sampling_params_from_state,
)
from silica.chat.cli.commands import CommandResult
from silica.chat.cli.palette import Palette
from silica.chat.cli.state import ChatCliState

# ---------------------------------------------------------------------------
# _build_prefix_cache
# ---------------------------------------------------------------------------


@dataclass
class _FakeKVLayout:
    """Minimal KV layout — only the fields ``_build_prefix_cache``
    forwards to the codec factory."""

    num_layers: int = 24
    n_kv_heads: int = 4
    head_dim: int = 64
    dtype: mx.Dtype = mx.float16


@dataclass
class _FakeAdapter:
    layout: _FakeKVLayout = field(default_factory=_FakeKVLayout)

    def kv_layout(self) -> _FakeKVLayout:
        return self.layout


@dataclass
class _FakeStore:
    block_size: int
    codec: Any = None


@dataclass
class _FakeCache:
    block_size: int
    store: _FakeStore


@dataclass
class _FakeCodecSpec:
    factory: Any  # callable matching the spec.factory signature


def _spy_codec_factory(
    *,
    block_size: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: Any,
    seed: int,
) -> dict[str, Any]:
    """Records the kwargs the cache builder forwarded — used to
    lock the contract that the adapter's KV layout reaches the
    codec factory unchanged."""
    return {
        "block_size": block_size,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype,
        "seed": seed,
    }


def _make_get_codec_spec(spec: _FakeCodecSpec) -> Any:
    def _resolve(_codec_id: str) -> _FakeCodecSpec:
        return spec
    return _resolve


def test_build_prefix_cache_fp16_path_passes_codec_none() -> None:
    """``codec_id=None`` produces a store with ``codec=None``;
    no codec factory is invoked."""
    adapter = _FakeAdapter()
    spec = _FakeCodecSpec(factory=_spy_codec_factory)

    cache = _build_prefix_cache(
        adapter,
        codec_id=None,
        get_codec_spec=_make_get_codec_spec(spec),
        store_cls=_FakeStore,
        cache_cls=_FakeCache,
    )

    assert isinstance(cache, _FakeCache)
    assert cache.block_size == 4
    assert isinstance(cache.store, _FakeStore)
    assert cache.store.codec is None


def test_build_prefix_cache_codec_path_invokes_factory_with_layout() -> None:
    """Non-``None`` ``codec_id`` resolves the spec, calls its
    factory with the adapter's KV layout, and installs the
    returned codec on the store."""
    adapter = _FakeAdapter(
        layout=_FakeKVLayout(
            num_layers=12, n_kv_heads=2, head_dim=8, dtype=mx.float16
        )
    )
    factory_calls: list[dict[str, Any]] = []

    def _spy(**kwargs: Any) -> str:
        factory_calls.append(kwargs)
        return "fake-codec"

    spec = _FakeCodecSpec(factory=_spy)
    cache = _build_prefix_cache(
        adapter,
        codec_id="block_tq_b64_b4",
        get_codec_spec=_make_get_codec_spec(spec),
        store_cls=_FakeStore,
        cache_cls=_FakeCache,
    )
    assert len(factory_calls) == 1
    kwargs = factory_calls[0]
    # Block size matches the chat REPL's _PREFIX_CACHE_BLOCK_SIZE.
    assert kwargs["block_size"] == 4
    # Layout-derived fields are forwarded byte-equivalently.
    assert kwargs["n_kv_heads"] == 2
    assert kwargs["head_dim"] == 8
    assert kwargs["dtype"] is mx.float16
    # Seed is the literal value the chat REPL pins.
    assert kwargs["seed"] == 42
    # Codec is installed on the store.
    assert cache.store.codec == "fake-codec"


def test_build_prefix_cache_block_size_constant_matches_repl() -> None:
    """Regression-lock: chat REPL's _PREFIX_CACHE_BLOCK_SIZE is
    4. The chat-bench harness's _default_cache_builder mirrors
    this; if either drifts, prefix-cache hit rates between the
    REPL and the bench harness will diverge silently."""
    from silica.chat.cli.app import _PREFIX_CACHE_BLOCK_SIZE

    assert _PREFIX_CACHE_BLOCK_SIZE == 4


def test_build_prefix_cache_resolves_codec_spec_via_injection() -> None:
    """The ``get_codec_spec`` callable is the single seam through
    which the codec id resolves; the helper does not import the
    real registry."""
    seen_codec_ids: list[str] = []

    def _resolve(codec_id: str) -> _FakeCodecSpec:
        seen_codec_ids.append(codec_id)
        return _FakeCodecSpec(factory=lambda **_: "stub")

    _build_prefix_cache(
        _FakeAdapter(),
        codec_id="custom_codec",
        get_codec_spec=_resolve,
        store_cls=_FakeStore,
        cache_cls=_FakeCache,
    )
    assert seen_codec_ids == ["custom_codec"]


# ---------------------------------------------------------------------------
# _sampling_params_from_state
# ---------------------------------------------------------------------------


@dataclass
class _FakeTokenizer:
    eos_token_ids: set[int] = field(default_factory=set)


@dataclass
class _FakeAdapterWithTokenizer:
    _tok: _FakeTokenizer = field(default_factory=_FakeTokenizer)

    def tokenizer(self) -> _FakeTokenizer:
        return self._tok


def test_sampling_params_uses_schema_defaults_when_config_empty() -> None:
    """An empty ``state.config`` produces the schema defaults
    documented at the call site (temperature=0.7, top_p=0.9,
    top_k=None, max_tokens=1024)."""
    state = ChatCliState()
    state.config.clear()
    adapter = _FakeAdapterWithTokenizer()
    params = _sampling_params_from_state(state, adapter)
    assert params.temperature == 0.7
    assert params.top_p == 0.9
    assert params.top_k is None
    assert params.max_tokens == 1024
    assert params.stop_token_ids == ()


def test_sampling_params_forwards_config_overrides() -> None:
    """``/config key=value`` writes into ``state.config`` and the
    helper reads each override into the matching SamplingParams
    field (with type coercion via ``float`` / ``int``)."""
    state = ChatCliState()
    state.config["temperature"] = "0.3"
    state.config["top_p"] = "0.85"
    state.config["top_k"] = 50  # type: ignore[assignment]
    state.config["max_tokens"] = "512"
    adapter = _FakeAdapterWithTokenizer()

    params = _sampling_params_from_state(state, adapter)
    assert params.temperature == 0.3
    assert params.top_p == 0.85
    assert params.top_k == 50
    assert params.max_tokens == 512


def test_sampling_params_top_k_non_int_collapses_to_none() -> None:
    """``top_k`` only flows through when ``isinstance(raw, int)``;
    a string (legacy persisted-state shape) collapses to ``None``
    so the sampler runs unconstrained rather than crashing on a
    coercion failure."""
    state = ChatCliState()
    state.config["top_k"] = "50"  # string, not int
    adapter = _FakeAdapterWithTokenizer()
    params = _sampling_params_from_state(state, adapter)
    assert params.top_k is None


def test_sampling_params_extracts_eos_ids_from_tokenizer() -> None:
    """The adapter's tokenizer ``eos_token_ids`` set surfaces as
    a sorted tuple on ``stop_token_ids``."""
    adapter = _FakeAdapterWithTokenizer(
        _tok=_FakeTokenizer(eos_token_ids={151645, 151643, 99})
    )
    params = _sampling_params_from_state(ChatCliState(), adapter)
    assert params.stop_token_ids == (99, 151643, 151645)


def test_sampling_params_handles_tokenizer_without_eos_attr() -> None:
    """Defensive: a tokenizer that does not expose ``eos_token_ids``
    yields an empty stop-token tuple rather than raising
    ``AttributeError``."""

    class _NoEosTok:
        pass

    class _NoEosAdapter:
        def tokenizer(self) -> _NoEosTok:
            return _NoEosTok()

    params = _sampling_params_from_state(
        ChatCliState(), _NoEosAdapter()
    )
    assert params.stop_token_ids == ()


def test_sampling_params_handles_tokenizer_with_none_eos() -> None:
    """``eos_token_ids = None`` (vs missing attr) also collapses
    to an empty tuple."""

    @dataclass
    class _NoneEosTok:
        eos_token_ids: set[int] | None = None

    class _NoneEosAdapter:
        def tokenizer(self) -> _NoneEosTok:
            return _NoneEosTok()

    params = _sampling_params_from_state(
        ChatCliState(), _NoneEosAdapter()
    )
    assert params.stop_token_ids == ()


# ---------------------------------------------------------------------------
# _resolve_thinking_mode
# ---------------------------------------------------------------------------


# Note: ``ChatCliState.config`` is annotated ``dict[str, str]`` but
# the chat REPL writes ``bool`` / ``int`` / ``None`` into it at
# runtime (e.g. ``thinking_mode``, ``top_k``). The static type is
# narrower than the runtime contract — the ``# type: ignore[assignment]``
# pragmas below reflect that mismatch rather than a test-side
# sloppiness. A follow-up cleanup should widen ``ChatCliState.config``
# to ``dict[str, str | bool | int | None]`` and remove these pragmas.


def test_resolve_thinking_mode_returns_true_when_config_true() -> None:
    state = ChatCliState()
    state.config["thinking_mode"] = True  # type: ignore[assignment]
    assert _resolve_thinking_mode(state) is True


def test_resolve_thinking_mode_returns_false_when_config_false() -> None:
    state = ChatCliState()
    state.config["thinking_mode"] = False  # type: ignore[assignment]
    assert _resolve_thinking_mode(state) is False


def test_resolve_thinking_mode_returns_none_when_key_missing() -> None:
    """A fresh state with no thinking_mode override reports None
    so ChatSession omits the ``enable_thinking`` kwarg from the
    chat template (model family default applies)."""
    state = ChatCliState()
    state.config.pop("thinking_mode", None)
    assert _resolve_thinking_mode(state) is None


def test_resolve_thinking_mode_returns_none_when_value_is_none() -> None:
    state = ChatCliState()
    state.config["thinking_mode"] = None  # type: ignore[assignment]
    assert _resolve_thinking_mode(state) is None


def test_resolve_thinking_mode_returns_none_for_non_bool_string() -> None:
    """Defensive: a string value (legacy persisted-state shape)
    collapses to None rather than being coerced — the chat
    template kwarg must not silently flip semantics."""
    state = ChatCliState()
    state.config["thinking_mode"] = "true"  # type: ignore[assignment]
    assert _resolve_thinking_mode(state) is None


def test_resolve_thinking_mode_returns_none_for_int() -> None:
    """``isinstance(1, bool)`` is False on Python's bool/int
    hierarchy semantics — but ``isinstance(True, int)`` is True.
    The helper uses the bool check, so an int stays out."""
    state = ChatCliState()
    state.config["thinking_mode"] = 1  # type: ignore[assignment]
    assert _resolve_thinking_mode(state) is None


# ---------------------------------------------------------------------------
# _apply_system_prompt_request
# ---------------------------------------------------------------------------


class _FakeChatSession:
    """Records calls to ``set_system_prompt`` so the helper's
    propagation behaviour is observable without touching a real
    ChatSession."""

    def __init__(self) -> None:
        self.system_prompt_calls: list[str | None] = []

    def set_system_prompt(self, text: str | None) -> None:
        self.system_prompt_calls.append(text)


def test_apply_system_prompt_request_no_op_when_request_none() -> None:
    """``request_system_prompt=None`` means the dispatcher did
    not raise a /system request this turn — leave the live
    session untouched."""
    session = _FakeChatSession()
    result = CommandResult(feedback=[], request_system_prompt=None)
    _apply_system_prompt_request(result, session)
    assert session.system_prompt_calls == []


def test_apply_system_prompt_request_clears_on_empty_string() -> None:
    """``request_system_prompt=""`` (``/system`` with no args)
    clears the session's prompt — set_system_prompt receives
    None, not the empty string."""
    session = _FakeChatSession()
    result = CommandResult(feedback=[], request_system_prompt="")
    _apply_system_prompt_request(result, session)
    assert session.system_prompt_calls == [None]


def test_apply_system_prompt_request_replaces_on_non_empty() -> None:
    """``request_system_prompt="text..."`` replaces the live
    session's system prompt with that text."""
    session = _FakeChatSession()
    result = CommandResult(
        feedback=[], request_system_prompt="be terse"
    )
    _apply_system_prompt_request(result, session)
    assert session.system_prompt_calls == ["be terse"]


def test_apply_system_prompt_request_does_not_react_to_other_flags() -> None:
    """Other request_* flags on the same CommandResult must not
    trigger a system-prompt update — the helper inspects only
    request_system_prompt."""
    session = _FakeChatSession()
    result = CommandResult(
        feedback=[],
        request_reset=True,
        request_regenerate=True,
        request_session_save="x.json",
        request_model_swap="some/repo",
        request_system_prompt=None,  # explicit None
    )
    _apply_system_prompt_request(result, session)
    assert session.system_prompt_calls == []


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-1 — _resolve_thinking_history
# ---------------------------------------------------------------------------


def test_resolve_thinking_history_returns_strip_when_config_strip() -> None:
    state = ChatCliState()
    state.config["thinking_history"] = "strip"
    assert _resolve_thinking_history(state) == "strip"


def test_resolve_thinking_history_returns_keep_when_config_keep() -> None:
    state = ChatCliState()
    state.config["thinking_history"] = "keep"
    assert _resolve_thinking_history(state) == "keep"


def test_resolve_thinking_history_defaults_to_strip_when_missing() -> None:
    """A fresh state with no override falls back to ``strip`` —
    the safe-by-default behaviour (no thinking pollution into
    next-turn context)."""
    state = ChatCliState()
    state.config.pop("thinking_history", None)
    assert _resolve_thinking_history(state) == "strip"


def test_resolve_thinking_history_falls_back_on_invalid_value() -> None:
    """Defensive: a corrupted persisted-state value (``"true"``,
    a bool, an int) collapses to ``strip`` rather than erroring
    or silently producing ``keep`` semantics."""
    state = ChatCliState()
    state.config["thinking_history"] = "garbage"
    assert _resolve_thinking_history(state) == "strip"

    state2 = ChatCliState()
    state2.config["thinking_history"] = True  # type: ignore[assignment]
    assert _resolve_thinking_history(state2) == "strip"


# ---------------------------------------------------------------------------
# Live-toolbar opt-in — _resolve_live_toolbar_enabled
# ---------------------------------------------------------------------------


def test_resolve_live_toolbar_enabled_default_off() -> None:
    """Empty env + empty config → live toolbar off. The Ansi
    backend is opt-in; users with no explicit preference get
    ``NullLiveToolbar`` and the post-turn ``bottom_toolbar``."""
    state = ChatCliState()
    state.config.pop("live_toolbar", None)
    assert _resolve_live_toolbar_enabled(state, env={}) is False


@pytest.mark.parametrize("raw", ["1", "on", "true", "ON", "True", "TRUE"])
def test_resolve_live_toolbar_enabled_env_truthy_forces_on(raw: str) -> None:
    """``SILICA_LIVE_TOOLBAR=1|on|true`` (case-insensitive) opts
    in regardless of ``state.config``."""
    state = ChatCliState()
    state.config["live_toolbar"] = "off"
    assert (
        _resolve_live_toolbar_enabled(
            state, env={"SILICA_LIVE_TOOLBAR": raw}
        )
        is True
    )


@pytest.mark.parametrize("raw", ["0", "off", "false", "OFF", "False"])
def test_resolve_live_toolbar_enabled_env_falsy_forces_off(raw: str) -> None:
    """Explicit env opt-out overrides config opt-in. Lets a user
    disable for one shell session without editing config."""
    state = ChatCliState()
    state.config["live_toolbar"] = "on"
    assert (
        _resolve_live_toolbar_enabled(
            state, env={"SILICA_LIVE_TOOLBAR": raw}
        )
        is False
    )


def test_resolve_live_toolbar_enabled_empty_env_defers_to_config() -> None:
    """Empty / missing env var → consult ``state.config``."""
    state = ChatCliState()
    state.config["live_toolbar"] = "on"
    assert (
        _resolve_live_toolbar_enabled(state, env={"SILICA_LIVE_TOOLBAR": ""})
        is True
    )
    state.config["live_toolbar"] = "off"
    assert (
        _resolve_live_toolbar_enabled(state, env={"SILICA_LIVE_TOOLBAR": ""})
        is False
    )


def test_resolve_live_toolbar_enabled_unrecognised_env_defers_to_config() -> None:
    """An env value the resolver does not recognise (e.g.
    ``maybe``) is treated as "not set" — consult config."""
    state = ChatCliState()
    state.config["live_toolbar"] = "on"
    assert (
        _resolve_live_toolbar_enabled(
            state, env={"SILICA_LIVE_TOOLBAR": "maybe"}
        )
        is True
    )


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-2 — _assistant_ends_in_thinking
# ---------------------------------------------------------------------------


def test_assistant_ends_in_thinking_explicit_open_no_close() -> None:
    """An explicit ``<think>`` with no matching ``</think>`` ends
    inside the block. Implicit-leading does not need to be set
    when the open tag is in the text."""
    assert (
        _assistant_ends_in_thinking(
            "<think>partial reasoning",
            implicit_leading=False,
        )
        is True
    )


def test_assistant_ends_in_thinking_balanced_explicit_pair() -> None:
    """A complete ``<think>...</think>`` pair followed by visible
    text ends OUTSIDE the block."""
    assert (
        _assistant_ends_in_thinking(
            "<think>reason</think>visible body",
            implicit_leading=False,
        )
        is False
    )


def test_assistant_ends_in_thinking_implicit_leading_no_close() -> None:
    """Implicit-leading (Qwen3 family) — the model started inside
    a think block, never emitted ``</think>``. The leading open is
    not in the text but still counts."""
    assert (
        _assistant_ends_in_thinking(
            "leading reasoning content",
            implicit_leading=True,
        )
        is True
    )


def test_assistant_ends_in_thinking_implicit_leading_with_close() -> None:
    """Implicit-leading + a single ``</think>`` in the text → the
    model closed the implicit block and continued in visible
    territory."""
    assert (
        _assistant_ends_in_thinking(
            "reasoning\n</think>\nvisible answer",
            implicit_leading=True,
        )
        is False
    )


def test_assistant_ends_in_thinking_no_tags_at_all() -> None:
    """Without implicit-leading and no tags: not in thinking."""
    assert (
        _assistant_ends_in_thinking(
            "plain visible reply",
            implicit_leading=False,
        )
        is False
    )


def test_assistant_ends_in_thinking_nested_open_explicit() -> None:
    """Multiple opens / closes balance correctly. Two opens, one
    close → still inside."""
    assert (
        _assistant_ends_in_thinking(
            "<think>first</think>middle<think>second",
            implicit_leading=False,
        )
        is True
    )


def test_assistant_ends_in_thinking_implicit_leading_explicit_open_no_close() -> None:
    """Implicit + explicit open + no close → 2 opens, 0 closes;
    still inside."""
    assert (
        _assistant_ends_in_thinking(
            "leading\n</think>\ntext<think>nested",
            implicit_leading=True,
        )
        is True
    )


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-2 — _evaluate_continue_request
# ---------------------------------------------------------------------------


@dataclass
class _FakeChatSessionForContinue:
    """Minimal chat-session shape exposing the message log so the
    continue-request guard helper can be exercised without a
    real ``ChatSession``."""

    messages: list[dict[str, str]] = field(default_factory=list)


def test_evaluate_continue_request_no_prior_returns_warning() -> None:
    """An empty message log fails the no-prior-assistant guard."""
    session = _FakeChatSessionForContinue(messages=[])
    state = ChatCliState()
    state.last_finish_reason = "max_tokens"
    proceed, warning = _evaluate_continue_request(session, state)
    assert proceed is False
    assert warning is not None
    assert "no prior" in warning.lower()


def test_evaluate_continue_request_system_only_returns_warning() -> None:
    """System-only history fails the same guard — no assistant
    tail to extend."""
    session = _FakeChatSessionForContinue(
        messages=[{"role": "system", "content": "be terse"}]
    )
    state = ChatCliState()
    state.last_finish_reason = "max_tokens"
    proceed, warning = _evaluate_continue_request(session, state)
    assert proceed is False
    assert warning is not None
    assert "no prior" in warning.lower()


def test_evaluate_continue_request_user_only_tail_returns_warning() -> None:
    """A trailing user-only message (mid-generation abort shape)
    is not a valid continue target."""
    session = _FakeChatSessionForContinue(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "follow-up"},
        ]
    )
    state = ChatCliState()
    state.last_finish_reason = "max_tokens"
    proceed, warning = _evaluate_continue_request(session, state)
    assert proceed is False
    assert warning is not None
    assert "no prior" in warning.lower()


def test_evaluate_continue_request_not_truncated_returns_warning() -> None:
    """Last turn ended naturally (``stop_token`` / ``done`` /
    ``empty``) → guard fails closed."""
    session = _FakeChatSessionForContinue(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "complete reply"},
        ]
    )
    state = ChatCliState()
    state.last_finish_reason = "stop_token"
    proceed, warning = _evaluate_continue_request(session, state)
    assert proceed is False
    assert warning is not None
    assert "not truncated" in warning.lower()


def test_evaluate_continue_request_none_finish_reason_returns_warning() -> None:
    """A fresh session (no turns run, ``last_finish_reason``
    still ``None``) cannot be /continue'd even if a stray
    assistant message exists."""
    session = _FakeChatSessionForContinue(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "reply"},
        ]
    )
    state = ChatCliState()
    assert state.last_finish_reason is None
    proceed, warning = _evaluate_continue_request(session, state)
    assert proceed is False
    assert warning is not None


def test_evaluate_continue_request_truncated_assistant_proceeds() -> None:
    """Both guards pass: assistant tail + ``last_finish_reason ==
    "max_tokens"`` → proceed; warning is ``None``."""
    session = _FakeChatSessionForContinue(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>partial"},
        ]
    )
    state = ChatCliState()
    state.last_finish_reason = "max_tokens"
    proceed, warning = _evaluate_continue_request(session, state)
    assert proceed is True
    assert warning is None


# ---------------------------------------------------------------------------
# Last-finish-reason lifecycle on ChatCliState
# ---------------------------------------------------------------------------


def test_chat_cli_state_last_finish_reason_default_none() -> None:
    """Fresh state — no turn has run yet, so the field is ``None``.
    RP-2's continue guard relies on this default to refuse
    /continue on a never-used session."""
    state = ChatCliState()
    assert state.last_finish_reason is None


# ---------------------------------------------------------------------------
# RP-2 v3: parser-start composition uses the truncation-time snapshot
# ---------------------------------------------------------------------------


def test_continue_parser_start_uses_snapshot_when_mode_flipped_off() -> None:
    """Scenario: prior turn was generated under
    ``thinking_mode=on`` (Qwen implicit ``<think>`` prepended) and
    truncated mid-think. User flips
    ``/config thinking_mode=off`` before ``/continue``. The
    snapshot says implicit_leading=True; the live config says
    False. The shell must compose ``_assistant_ends_in_thinking``
    with the SNAPSHOT to keep the parser in thinking until the
    model emits ``</think>`` — otherwise reasoning leaks into
    the visible transcript."""
    raw_prefix = "leading reasoning"  # no </think> yet
    snapshot_implicit_leading = True
    live_implicit_thinking = False  # mode flipped off
    snap_or_live = (
        snapshot_implicit_leading
        if snapshot_implicit_leading is not None
        else live_implicit_thinking
    )
    parser_start = _assistant_ends_in_thinking(
        raw_prefix, implicit_leading=snap_or_live
    )
    assert parser_start is True


def test_continue_parser_start_uses_snapshot_when_mode_flipped_on() -> None:
    """Mirror scenario: prior turn ran with
    ``thinking_mode=off`` (no implicit ``<think>`` prepended) and
    truncated in visible reply. User flips
    ``/config thinking_mode=on`` before ``/continue``. Snapshot
    says implicit_leading=False; live says True. Composing with
    the snapshot keeps the parser OUT of thinking — otherwise the
    visible reply continuation gets hidden behind the magenta
    indicator."""
    raw_prefix = "visible reply prefix"
    snapshot_implicit_leading = False
    live_implicit_thinking = True  # mode flipped on
    snap_or_live = (
        snapshot_implicit_leading
        if snapshot_implicit_leading is not None
        else live_implicit_thinking
    )
    parser_start = _assistant_ends_in_thinking(
        raw_prefix, implicit_leading=snap_or_live
    )
    assert parser_start is False


def test_continue_parser_start_falls_back_to_live_when_snapshot_none() -> None:
    """Degenerate path: continue_last invoked on a turn that
    finished naturally (no snapshot). The shell falls back to the
    live ``implicit_thinking`` flag — which is the right answer
    because there is no truncation-time fact to honour."""
    raw_prefix = "visible reply"
    snapshot_implicit_leading = None
    live_implicit_thinking = True
    snap_or_live = (
        snapshot_implicit_leading
        if snapshot_implicit_leading is not None
        else live_implicit_thinking
    )
    parser_start = _assistant_ends_in_thinking(
        raw_prefix, implicit_leading=snap_or_live
    )
    # implicit_leading=True + content has no </think> → True.
    assert parser_start is True


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-3 — per-turn metric defaults + lifecycle
# ---------------------------------------------------------------------------


def test_chat_cli_state_per_turn_chars_default_zero() -> None:
    """Fresh state — RP-3's per-turn char counters and the
    cumulative continuation tally start at zero. ``/showcase``
    reads these to render the reasoning vs visible split."""
    state = ChatCliState()
    assert state.last_turn_reasoning_chars == 0
    assert state.last_turn_visible_chars == 0
    assert state.total_continuation_chunks == 0


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-3 — truncation marker helper
# ---------------------------------------------------------------------------


def test_truncation_marker_prints_on_max_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``finish_reason="max_tokens"`` renders the marker; the
    helper returns ``True`` so callers can branch without
    inspecting stdout."""
    printed = _print_truncation_marker("max_tokens", Palette.plain())
    assert printed is True
    out = capsys.readouterr().out
    assert "[truncated: /continue]" in out


@pytest.mark.parametrize(
    "reason",
    ["stop_token", "done", "eos", "empty", "aborted", None],
)
def test_truncation_marker_silent_on_other_reasons(
    reason: str | None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Any other finish reason — including ``None`` (the
    pre-first-turn / post-reset state) — must not print the
    marker. The helper returns ``False`` and stdout is empty."""
    printed = _print_truncation_marker(reason, Palette.plain())
    assert printed is False
    out = capsys.readouterr().out
    assert out == ""


def test_truncation_marker_does_not_pollute_message_log() -> None:
    """The marker is rendered via direct ``sys.stdout.write`` and
    must not be reachable through any path that feeds
    ``ChatSession.messages``. Static guarantee: the helper only
    accepts ``finish_reason`` and ``palette`` — there is no
    chat-session reference to write to. Sanity-checks the helper's
    signature so future refactors do not silently re-add a path
    through the streaming protocol."""
    import inspect

    sig = inspect.signature(_print_truncation_marker)
    assert set(sig.parameters) == {"finish_reason", "palette"}


# ---------------------------------------------------------------------------
# CHAT-CLI-RESPONSE-POLICY RP-2 / RP-3 — abort-rollback helper
# ---------------------------------------------------------------------------


class _RollbackFakeSession:
    """Minimal :class:`ChatSession` surrogate for the rollback
    tests. Records ``replace_messages`` calls and exposes the
    deep-copy ``messages`` property so ``_capture_rollback_snapshot``
    operates against the same surface the live session does."""

    def __init__(
        self, messages: list[dict[str, str]] | None = None
    ) -> None:
        self._messages: list[dict[str, str]] = (
            list(messages) if messages else []
        )

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._messages
        ]

    def replace_messages(
        self, messages: list[dict[str, str]]
    ) -> None:
        self._messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]


def _seeded_rollback_state() -> ChatCliState:
    """State with non-default values for every field
    ``_capture_rollback_snapshot`` reads — so a regression that
    misses a field shows up as a mismatch on restore."""
    state = ChatCliState()
    state.last_turn_thinking = "original-thinking"
    state.last_turn_reasoning_chars = 100
    state.last_turn_visible_chars = 50
    return state


def test_capture_rollback_snapshot_reads_messages_and_metrics() -> None:
    """The capture helper records the message log + the three
    per-turn metric fields. Pure read; no mutation of either
    surface."""
    session = _RollbackFakeSession(
        messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    state = _seeded_rollback_state()

    snapshot = _capture_rollback_snapshot(session, state)

    assert snapshot.messages == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    assert snapshot.thinking == "original-thinking"
    assert snapshot.reasoning_chars == 100
    assert snapshot.visible_chars == 50
    # Capture is non-mutating.
    assert state.last_turn_thinking == "original-thinking"


def test_capture_rollback_snapshot_messages_decoupled_from_session() -> None:
    """The snapshot's ``messages`` field must not share dict refs
    with the live session — otherwise an in-place write to
    ``self._messages[-1]['content']`` (e.g. ``continue_last``'s
    final assignment) would corrupt the snapshot before rollback
    runs. Tests RP-2 v4's deep-copy contract through the helper."""
    session = _RollbackFakeSession(
        messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "original"},
        ]
    )
    state = _seeded_rollback_state()
    snapshot = _capture_rollback_snapshot(session, state)
    # In-place mutation simulates continue_last's final write.
    session._messages[-1]["content"] = "(mid-flight partial)"
    # Snapshot stays at the pre-mutation content.
    assert snapshot.messages[-1]["content"] == "original"


def test_apply_rollback_snapshot_restores_messages() -> None:
    """``_apply_rollback_snapshot`` writes the captured message
    log back via ``replace_messages``. Mid-flight mutations are
    undone."""
    session = _RollbackFakeSession(
        messages=[
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "original"},
        ]
    )
    state = _seeded_rollback_state()
    snapshot = _capture_rollback_snapshot(session, state)
    # Simulate the chat-turn flow's mid-flight mutation.
    session.replace_messages(
        [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "(partial)"},
        ]
    )
    state.last_turn_thinking = "(stale)"
    state.last_turn_reasoning_chars = 999
    state.last_turn_visible_chars = 999
    # Apply rollback and verify restoration.
    _apply_rollback_snapshot(
        snapshot, chat_session=session, state=state
    )
    assert session.messages == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "original"},
    ]


def test_apply_rollback_snapshot_restores_thinking_and_chars() -> None:
    """The same rollback restores ``last_turn_thinking`` and the
    char counters — without this, an aborted /continue would leave
    half-streamed metrics in /showcase even though the message
    log was restored."""
    session = _RollbackFakeSession(
        messages=[{"role": "assistant", "content": "x"}]
    )
    state = _seeded_rollback_state()
    snapshot = _capture_rollback_snapshot(session, state)
    state.last_turn_thinking = "(half-streamed thinking)"
    state.last_turn_reasoning_chars = 1234
    state.last_turn_visible_chars = 567
    _apply_rollback_snapshot(
        snapshot, chat_session=session, state=state
    )
    assert state.last_turn_thinking == "original-thinking"
    assert state.last_turn_reasoning_chars == 100
    assert state.last_turn_visible_chars == 50


def test_apply_rollback_snapshot_idempotent_under_repeated_apply() -> None:
    """Applying the same snapshot twice yields the same final
    state as applying once — no compounding side effects. Useful
    if a /regenerate retry's rollback fires and a follow-up
    rollback runs against the same captured snapshot."""
    session = _RollbackFakeSession(
        messages=[{"role": "assistant", "content": "x"}]
    )
    state = _seeded_rollback_state()
    snapshot = _capture_rollback_snapshot(session, state)

    state.last_turn_thinking = "(once)"
    state.last_turn_reasoning_chars = 1
    _apply_rollback_snapshot(
        snapshot, chat_session=session, state=state
    )
    snapshot_after_first = (
        state.last_turn_thinking,
        state.last_turn_reasoning_chars,
        state.last_turn_visible_chars,
    )
    _apply_rollback_snapshot(
        snapshot, chat_session=session, state=state
    )
    assert (
        state.last_turn_thinking,
        state.last_turn_reasoning_chars,
        state.last_turn_visible_chars,
    ) == snapshot_after_first


# ---------------------------------------------------------------------------
# Default system prompt + _resolve_initial_system_prompt
# ---------------------------------------------------------------------------


def test_default_system_prompt_is_a_concise_string() -> None:
    """``DEFAULT_SYSTEM_PROMPT`` is a non-empty plain-text string
    of moderate length — concise enough to not bloat every turn's
    prompt, long enough to convey the directives. The string-shape
    asserts also pin the basic content (avoid an accidental empty
    or all-whitespace value)."""
    assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
    assert DEFAULT_SYSTEM_PROMPT.strip() == DEFAULT_SYSTEM_PROMPT
    assert 50 <= len(DEFAULT_SYSTEM_PROMPT) <= 400
    # Core directives the prompt commits to (case-insensitive
    # substring match — exact wording is allowed to evolve as long
    # as the spirit holds).
    lower = DEFAULT_SYSTEM_PROMPT.lower()
    assert "concise" in lower or "directly" in lower
    assert "preamble" in lower or "self-narration" in lower or "elaborate" in lower


def test_resolve_initial_system_prompt_none_returns_default() -> None:
    """No ``--system`` flag (``args.system is None``) routes to
    the bundled default — the silica chat REPL ships with a
    sensible system prompt out of the box rather than empty."""
    assert _resolve_initial_system_prompt(None) == DEFAULT_SYSTEM_PROMPT


def test_resolve_initial_system_prompt_empty_string_returns_none() -> None:
    """``--system ""`` is the explicit opt-out: the user wants no
    system message at all. The helper returns ``None`` so
    ``ChatSession`` skips the system entry."""
    assert _resolve_initial_system_prompt("") is None


def test_resolve_initial_system_prompt_explicit_text_passes_through() -> None:
    """Non-empty ``--system`` is the user's override; the helper
    returns it unchanged. The default does NOT augment a custom
    prompt — silica chat respects user intent verbatim."""
    custom = "You are a senior code reviewer. Be terse."
    assert _resolve_initial_system_prompt(custom) == custom


def test_resolve_initial_system_prompt_whitespace_only_passes_through() -> None:
    """A whitespace-only ``--system "   "`` is unusual but legal;
    ``ChatSession`` will create a system message with that content.
    We do NOT silently fall back to the default for whitespace
    input — the user opted in to something. The helper only
    triggers the default for the literal ``None`` (no flag)."""
    assert _resolve_initial_system_prompt("   ") == "   "


def test_rollback_full_continue_lifecycle_smoke() -> None:
    """End-to-end smoke for the chat-CLI shell pattern: capture
    snapshot before /continue, mutate state mid-flight to
    simulate streaming, then apply rollback on (simulated)
    KeyboardInterrupt. Verifies the four fields restore in
    lockstep — the failure mode user caught pre-commit was a
    half-streamed metric leaking through after the messages
    rolled back."""
    session = _RollbackFakeSession(
        messages=[
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "<think>partial"},
        ]
    )
    state = ChatCliState()
    state.last_turn_thinking = "<think>partial"
    state.last_turn_reasoning_chars = 14
    state.last_turn_visible_chars = 0
    state.last_finish_reason = "max_tokens"

    # Pre-/continue snapshot — what the slash branch captures.
    snapshot = _capture_rollback_snapshot(session, state)

    # Streaming mid-/continue mutates everything in-place.
    session.replace_messages(
        [
            {"role": "user", "content": "ask"},
            {
                "role": "assistant",
                "content": "<think>partial more reasoning",
            },
        ]
    )
    state.last_turn_thinking = "<think>partial more reasoning"
    state.last_turn_reasoning_chars = 35
    state.last_turn_visible_chars = 0

    # KeyboardInterrupt fires; abort branch runs the rollback.
    _apply_rollback_snapshot(
        snapshot, chat_session=session, state=state
    )

    # All four surfaces back to pre-/continue state.
    assert session.messages == [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": "<think>partial"},
    ]
    assert state.last_turn_thinking == "<think>partial"
    assert state.last_turn_reasoning_chars == 14
    assert state.last_turn_visible_chars == 0
    # last_finish_reason is NOT in the rollback snapshot — it is
    # set by metrics after the (failed) turn, so abort leaves it
    # at the pre-call value naturally.
    assert state.last_finish_reason == "max_tokens"
