"""Behaviour tests for ``silica.kernels.shadow_install``.

Verifies the env-flag-gated install / restore lifecycle for the two
shadow flags exposed at v1.7.23 — ``SILICA_USE_FA_DECODE_V10`` and
``SILICA_USE_BF16_DELTANET_STATE``. Tests are binding-level: they
inspect attribute identity on the patched modules, do not invoke any
Metal kernel, and do not load any real model. v10 numerical
correctness is the responsibility of a separate microbench test that
is intentionally not part of this commit (see v1.7.23 changelog).
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterator

import pytest

_FLAGS = ("SILICA_USE_FA_DECODE_V10", "SILICA_USE_BF16_DELTANET_STATE")


@pytest.fixture
def fresh_install_state() -> Iterator[None]:
    """Restore shadow_install state and clear flag env vars around each test.

    The fixture also restores any pre-existing values for the two flags
    after the test completes so that the suite remains environment-safe
    when run in isolation.
    """
    from silica.kernels import shadow_install

    shadow_install.restore()
    saved_env = {k: os.environ.get(k) for k in _FLAGS}
    for k in _FLAGS:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        shadow_install.restore()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _qn_module():  # type: ignore[no-untyped-def]
    pytest.importorskip("mlx_lm.models.qwen3_next")
    from mlx_lm.models import qwen3_next as qn

    return qn


def _gd_module():  # type: ignore[no-untyped-def]
    pytest.importorskip("mlx_lm.models.gated_delta")
    from mlx_lm.models import gated_delta as gd

    return gd


def test_default_off_install_is_no_patch(fresh_install_state: None) -> None:
    """With both env flags unset, ``install()`` reports both False and
    swaps no module attributes.
    """
    from silica.kernels import shadow_install

    qn = _qn_module()
    gd = _gd_module()
    original_attn_call = qn.Qwen3NextAttention.__call__
    original_gated_delta = gd.gated_delta_update
    original_qn_gated_delta = qn.gated_delta_update

    result = shadow_install.install(model=None)

    assert result == {"fa_decode_v10": False, "bf16_deltanet_state": False}
    assert qn.Qwen3NextAttention.__call__ is original_attn_call
    assert gd.gated_delta_update is original_gated_delta
    assert qn.gated_delta_update is original_qn_gated_delta


def test_v10_flag_patches_attn_call(fresh_install_state: None) -> None:
    """``SILICA_USE_FA_DECODE_V10=1`` alone patches
    ``Qwen3NextAttention.__call__`` and reports ``fa_decode_v10=True``;
    gated-delta references stay unchanged.
    """
    from silica.kernels import shadow_install

    qn = _qn_module()
    gd = _gd_module()
    original_attn_call = qn.Qwen3NextAttention.__call__
    original_gated_delta = gd.gated_delta_update

    os.environ["SILICA_USE_FA_DECODE_V10"] = "1"
    result = shadow_install.install(model=None)

    assert result == {"fa_decode_v10": True, "bf16_deltanet_state": False}
    assert qn.Qwen3NextAttention.__call__ is not original_attn_call
    assert gd.gated_delta_update is original_gated_delta


def test_bf16_state_flag_patches_gated_delta(fresh_install_state: None) -> None:
    """``SILICA_USE_BF16_DELTANET_STATE=1`` alone patches both
    ``gated_delta.gated_delta_update`` and the imported reference at
    ``qwen3_next.gated_delta_update``; ``Qwen3NextAttention.__call__``
    stays unchanged.
    """
    from silica.kernels import shadow_install

    qn = _qn_module()
    gd = _gd_module()
    original_attn_call = qn.Qwen3NextAttention.__call__
    original_gated_delta = gd.gated_delta_update
    original_qn_gated_delta = qn.gated_delta_update

    os.environ["SILICA_USE_BF16_DELTANET_STATE"] = "1"
    result = shadow_install.install(model=None)

    assert result == {"fa_decode_v10": False, "bf16_deltanet_state": True}
    assert qn.Qwen3NextAttention.__call__ is original_attn_call
    assert gd.gated_delta_update is not original_gated_delta
    assert qn.gated_delta_update is not original_qn_gated_delta
    # Both modules must point to the same patched callable so that the
    # bf16 path is reached via either attribute lookup.
    assert gd.gated_delta_update is qn.gated_delta_update


def test_restore_reverts_all_patches(fresh_install_state: None) -> None:
    """After ``install()`` with both flags, ``restore()`` returns every
    patched reference to its original.
    """
    from silica.kernels import shadow_install

    qn = _qn_module()
    gd = _gd_module()
    original_attn_call = qn.Qwen3NextAttention.__call__
    original_gated_delta = gd.gated_delta_update
    original_qn_gated_delta = qn.gated_delta_update

    os.environ["SILICA_USE_FA_DECODE_V10"] = "1"
    os.environ["SILICA_USE_BF16_DELTANET_STATE"] = "1"
    shadow_install.install(model=None)
    assert qn.Qwen3NextAttention.__call__ is not original_attn_call
    assert gd.gated_delta_update is not original_gated_delta

    shadow_install.restore()

    assert qn.Qwen3NextAttention.__call__ is original_attn_call
    assert gd.gated_delta_update is original_gated_delta
    assert qn.gated_delta_update is original_qn_gated_delta


def test_install_is_idempotent(fresh_install_state: None) -> None:
    """A second ``install()`` after the first does not re-patch."""
    from silica.kernels import shadow_install

    qn = _qn_module()

    os.environ["SILICA_USE_FA_DECODE_V10"] = "1"
    shadow_install.install(model=None)
    patched_attn_call = qn.Qwen3NextAttention.__call__

    second_result = shadow_install.install(model=None)

    # The second install short-circuits and reports the empty dict
    # without re-reading the environment or re-patching.
    assert second_result == {"fa_decode_v10": False, "bf16_deltanet_state": False}
    assert qn.Qwen3NextAttention.__call__ is patched_attn_call


def test_module_does_not_reference_retired_flags() -> None:
    """The slim shadow_install must not reference any retired flag or
    the fused-op kernels they patched on opus.
    """
    from silica.kernels import shadow_install

    src = inspect.getsource(shadow_install)
    forbidden = (
        "SILICA_USE_FUSED_GATED_OUTPUT",
        "SILICA_USE_FUSED_SILU_MUL",
        "SILICA_USE_FUSED_QK_NORM",
        "SILICA_USE_COMPILED_MLP",
        "fused_gated_output",
        "fused_silu_mul",
        "fused_qk_norm",
    )
    for token in forbidden:
        assert token not in src, (
            f"slim shadow_install must not reference {token!r}"
        )
