"""Tests for silica.models.factory — adapter dispatch by model_type."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from mlx_lm.models.cache import KVCache

from silica.kvcache.simple import SimpleKVCache
from silica.models import factory
from silica.models.qwen3 import Qwen3Adapter
from silica.models.qwen3_5 import Qwen3_5Adapter


class _Tok:
    vocab_size = 16

    def encode(self, text: str) -> list[int]:
        return []

    def decode(self, ids: object) -> str:
        return ""


class _PlainModel:
    model_type = "qwen3"

    def __init__(self) -> None:
        class _Args:
            hidden_size = 64
            num_attention_heads = 4
            head_dim = 16

        self.args = _Args()

        class _SA:
            n_kv_heads = 2

        class _L:
            self_attn = _SA()

        self.layers = [_L() for _ in range(2)]

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


class _Qwen35Model:
    model_type = "qwen3_5"

    def __init__(self) -> None:
        class _Args:
            text_config = {"hidden_size": 64, "num_hidden_layers": 2}

        self.args = _Args()

        class _SA:
            num_key_value_heads = 2
            head_dim = 16

        class _L:
            is_linear = False
            self_attn = _SA()

        self.layers = [_L() for _ in range(2)]

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


class _UnknownModel:
    model_type = "totally_unknown_family"

    def __init__(self) -> None:
        self.args = None
        self.layers: list[object] = []

    def make_cache(self) -> list[KVCache]:
        return []


def test_supported_model_types_includes_qwen3_and_qwen3_5() -> None:
    supported = factory.supported_model_types()
    assert "qwen3" in supported
    assert "qwen3_5" in supported


def test_adapter_for_repo_dispatches_plain_qwen3() -> None:
    with patch(
        "silica.models.factory._mlx_lm_load",
        return_value=(_PlainModel(), _Tok()),
    ):
        adapter, kv = factory.adapter_for_repo("fake/plain")
    assert isinstance(adapter, Qwen3Adapter)
    assert isinstance(kv, SimpleKVCache)


def test_adapter_for_repo_dispatches_qwen3_5() -> None:
    with patch(
        "silica.models.factory._mlx_lm_load",
        return_value=(_Qwen35Model(), _Tok()),
    ):
        adapter, kv = factory.adapter_for_repo("fake/qwen3_5")
    assert isinstance(adapter, Qwen3_5Adapter)
    assert isinstance(kv, SimpleKVCache)


def test_adapter_for_repo_raises_with_supported_list_on_unknown() -> None:
    with patch(
        "silica.models.factory._mlx_lm_load",
        return_value=(_UnknownModel(), _Tok()),
    ):
        with pytest.raises(NotImplementedError, match="totally_unknown_family"):
            factory.adapter_for_repo("fake/unknown")


def test_unknown_error_message_names_registration_location() -> None:
    """The error must tell the user WHERE to register a new family."""
    with patch(
        "silica.models.factory._mlx_lm_load",
        return_value=(_UnknownModel(), _Tok()),
    ):
        with pytest.raises(NotImplementedError, match="silica.models.factory"):
            factory.adapter_for_repo("fake/unknown")


# --- adapter_from_loaded_model: skip-reload helper for probe scripts --------


def test_adapter_from_loaded_model_does_not_call_mlx_lm_load() -> None:
    """Probe scripts that already paid the ``mlx_lm.load`` cost in a
    Phase 1 must be able to dispatch without loading the checkpoint
    again. Using ``adapter_for_repo`` for Phase 2 would double-load a
    16-20 GB MoE checkpoint, which on a 48 GB M5 Pro can turn an
    expected variant-guard failure into an out-of-memory failure that
    masks the real probe finding.

    Pins the contract: this helper never touches ``_mlx_lm_load``."""
    model = _PlainModel()
    tokenizer = _Tok()
    with patch(
        "silica.models.factory._mlx_lm_load",
        side_effect=AssertionError(
            "adapter_from_loaded_model must not call mlx_lm.load"
        ),
    ):
        adapter, kv = factory.adapter_from_loaded_model(model, tokenizer)
    assert isinstance(adapter, Qwen3Adapter)
    assert isinstance(kv, SimpleKVCache)


def test_adapter_from_loaded_model_raises_same_error_as_adapter_for_repo() -> (
    None
):
    """The dispatch branch behind both helpers is shared; the error
    surface must match so callers (e.g. probe scripts) can rely on
    the same ``NotImplementedError`` text."""
    with pytest.raises(NotImplementedError, match="totally_unknown_family"):
        factory.adapter_from_loaded_model(_UnknownModel(), _Tok())
    with pytest.raises(NotImplementedError, match="silica.models.factory"):
        factory.adapter_from_loaded_model(_UnknownModel(), _Tok())


def test_adapter_for_repo_still_loads_via_mlx_lm() -> None:
    """The existing ``adapter_for_repo(repo)`` path is a thin wrapper
    over ``adapter_from_loaded_model`` that still calls
    ``_mlx_lm_load``. Callers with just the repo id (e.g. the CLI)
    continue to work unchanged."""
    load_mock = patch(
        "silica.models.factory._mlx_lm_load",
        return_value=(_PlainModel(), _Tok()),
    )
    with load_mock as mocked:
        adapter, _ = factory.adapter_for_repo("fake/plain")
    assert mocked.call_count == 1
    assert isinstance(adapter, Qwen3Adapter)
