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
    _apply_system_prompt_request,
    _build_prefix_cache,
    _resolve_live_toolbar_enabled,
    _resolve_thinking_history,
    _resolve_thinking_mode,
    _sampling_params_from_state,
)
from silica.chat.cli.commands import CommandResult
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
