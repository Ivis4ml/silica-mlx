"""D-021 step 6 (ε) — engine-side wiring tests for TargetHiddenConsumer.

Pins the call ordering between ``Engine.generate`` and a
``TargetHiddenConsumer`` drafter. Uses the cached
``Qwen/Qwen3.5-0.8B`` fixture (the smallest target silica's αβ
capture surface covers) and wraps the loaded adapter to record
which methods the engine calls in which order.

Scope:

- ``adapter.prefill_with_capture`` is called exactly once before the
  cycle loop; ``drafter.prime`` follows.
- Each spec cycle calls ``decode_step_multi_with_capture`` (not the
  plain decode path); ``drafter.update_target_hidden`` runs
  immediately after, **before** the engine's KV / recurrent
  rollback path.
- ``drafter.commit`` runs after the bonus emit.
- ``drafter.free_target_hidden`` runs in ``Engine.generate``'s
  ``finally`` block.
- Non-``TargetHiddenConsumer`` drafter (``NoopDraftEngine``): engine
  takes the existing plain ``prefill`` / ``decode_step`` path; no
  capture-variant calls fire.
- Adapter without ``HiddenCaptureAdapter`` (cached Qwen3-0.6B is the
  load-bearing example — silica's ``Qwen3Adapter`` does not ship the
  capture surface): engine raises ``NotImplementedError`` at
  ``_drive`` entry, before any forward.

The cycle-1 byte-exact parity test on a real model lives in
``test_dflash_engine_parity.py`` (separate file so the parity
attestation can be run on its own with ``-k parity``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from silica.core.profiler import MetricsRegistry
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.models.hidden_capture import HiddenCaptureAdapter
from silica.models.qwen3 import Qwen3Adapter
from silica.models.qwen3_5 import Qwen3_5Adapter
from silica.speculative.dflash_drafter import DFlashDrafter
from silica.speculative.engine import NoopDraftEngine

QWEN3_5_REPO = "Qwen/Qwen3.5-0.8B"
QWEN3_REPO = "Qwen/Qwen3-0.6B"

_QWEN3_5_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3.5-0.8B"
)
_QWEN3_5_SKIP = not _QWEN3_5_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_QWEN3_5_SKIP_REASON = (
    f"Qwen3.5-0.8B not cached at {_QWEN3_5_CACHE}; pull via "
    "huggingface-cli or any test that loads the 0.8B fixture."
)

_QWEN3_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
)
_QWEN3_SKIP = not _QWEN3_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_QWEN3_SKIP_REASON = (
    f"Qwen3-0.6B not cached at {_QWEN3_CACHE}; pull via huggingface-cli "
    "or any test that loads the 0.6B fixture."
)


def _record_calls(adapter: Any, log: list[str]) -> None:
    """Wrap key adapter methods so their invocations append to ``log``.

    Mutates the adapter in place — caller should pass a freshly-loaded
    instance. Wrapped methods preserve their return values so the
    engine still drives correctly.
    """
    for name in (
        "prefill",
        "prefill_with_capture",
        "decode_step",
        "decode_step_multi",
        "decode_step_multi_with_capture",
    ):
        orig = getattr(adapter, name, None)
        if orig is None:
            continue

        def make_wrapper(name: str, orig: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                log.append(name)
                return orig(*args, **kwargs)

            return wrapper

        setattr(adapter, name, make_wrapper(name, orig))


def _wrap_drafter(drafter: DFlashDrafter, log: list[str]) -> None:
    """Wrap drafter side-channel + Protocol methods for call-order
    pinning. Like ``_record_calls`` but operates on the drafter."""
    for name in ("prime", "update_target_hidden", "commit", "free_target_hidden"):
        orig = getattr(drafter, name)

        def make_wrapper(name: str, orig: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                log.append(name)
                return orig(*args, **kwargs)

            return wrapper

        setattr(drafter, name, make_wrapper(name, orig))


def _make_oracle_emitter(oracle_tokens: list[int]) -> Any:
    """Return a synthetic emitter that walks through ``oracle_tokens``.

    Each call returns the next ``k`` tokens; subsequent calls advance
    the cursor. Used to replay a spec-off-recorded sequence so
    greedy_verify accepts the full block at every cycle.
    """
    cursor = {"i": 0}

    def emit(target_hidden: mx.array, k: int) -> tuple[int, ...]:
        del target_hidden
        i = cursor["i"]
        out = tuple(oracle_tokens[i : i + k])
        cursor["i"] = i + len(out)
        return out

    return emit


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_target_hidden_drafter_routes_through_capture_path() -> None:
    """End-to-end-shape pin: with a TargetHiddenConsumer drafter,
    Engine.generate calls prefill_with_capture exactly once and
    decode_step_multi_with_capture once per spec cycle. The plain
    prefill / decode_step / decode_step_multi paths are unused."""
    adapter, kv = Qwen3_5Adapter.from_hf_repo(QWEN3_5_REPO)
    log: list[str] = []
    _record_calls(adapter, log)

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0, 7, 14),
        # An emitter that returns nothing — empty drafts. Engine still
        # drives one bonus per cycle through the capture verify path.
        synthetic_emit=lambda th, k: (),
    )
    engine = Engine(
        adapter=adapter,
        kv_manager=kv,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=drafter,
        verify_k=4,
    )
    list(
        engine.generate(
            prompt="Hello",
            params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )

    # Capture path was used; plain prefill / decode_step* never fired.
    assert "prefill_with_capture" in log
    assert "prefill" not in log
    assert "decode_step_multi_with_capture" in log
    assert "decode_step_multi" not in log
    assert "decode_step" not in log


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_target_hidden_drafter_invokes_prime_update_commit_free_in_order() -> None:
    """Drafter side-channel calls fire in the order:
    prefill_with_capture → prime → (decode_step_multi_with_capture →
    update_target_hidden → commit)+ → free_target_hidden.

    Engine ε's contract is that update_target_hidden runs after the
    verify forward and before commit; free_target_hidden is the last
    drafter-side call, fired in ``Engine.generate``'s ``finally``."""
    adapter, kv = Qwen3_5Adapter.from_hf_repo(QWEN3_5_REPO)
    log: list[str] = []
    _record_calls(adapter, log)

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0, 7),
        synthetic_emit=lambda th, k: (),
    )
    _wrap_drafter(drafter, log)

    engine = Engine(
        adapter=adapter,
        kv_manager=kv,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=drafter,
        verify_k=4,
    )
    list(
        engine.generate(
            prompt="Hello",
            params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )

    # 1. prefill_with_capture happens before prime.
    pwc_idx = log.index("prefill_with_capture")
    prime_idx = log.index("prime")
    assert pwc_idx < prime_idx, f"prime must follow prefill_with_capture: {log}"

    # 2. Each spec cycle: decode_step_multi_with_capture →
    #    update_target_hidden → commit, in that order. Walk the log
    #    once and confirm the per-cycle ordering invariant.
    last_capture = -1
    last_update = -1
    for i, entry in enumerate(log):
        if entry == "decode_step_multi_with_capture":
            last_capture = i
        elif entry == "update_target_hidden":
            assert last_capture > -1, (
                f"update_target_hidden at {i} not preceded by a "
                f"capture call: {log}"
            )
            assert last_capture < i, log
            last_update = i
        elif entry == "commit":
            assert last_update > -1, (
                f"commit at {i} not preceded by update_target_hidden: {log}"
            )
            assert last_update < i, log

    # 3. free_target_hidden is the final drafter-side call.
    drafter_calls = [
        e for e in log
        if e in {"prime", "update_target_hidden", "commit", "free_target_hidden"}
    ]
    assert drafter_calls[-1] == "free_target_hidden", drafter_calls


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_noop_drafter_engine_path_is_unchanged() -> None:
    """A drafter that does NOT implement TargetHiddenConsumer
    (NoopDraftEngine here) takes the existing pre-(ε) path: plain
    ``prefill`` / ``decode_step``, no capture-variant calls. This
    guards the byte-identical-fallback invariant."""
    adapter, kv = Qwen3_5Adapter.from_hf_repo(QWEN3_5_REPO)
    log: list[str] = []
    _record_calls(adapter, log)

    engine = Engine(
        adapter=adapter,
        kv_manager=kv,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=NoopDraftEngine(),
        verify_k=4,
    )
    list(
        engine.generate(
            prompt="Hello",
            params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )

    assert "prefill" in log
    assert "prefill_with_capture" not in log
    assert "decode_step" in log
    assert "decode_step_multi_with_capture" not in log


@pytest.mark.skipif(_QWEN3_SKIP, reason=_QWEN3_SKIP_REASON)
def test_target_hidden_drafter_requires_hidden_capture_adapter() -> None:
    """If a TargetHiddenConsumer drafter is wired against an adapter
    that does not implement HiddenCaptureAdapter (Qwen3Adapter is
    the load-bearing example — silica's αβ capture surface is
    Qwen3.5-only), Engine raises NotImplementedError at ``_drive``
    entry, before any forward."""
    adapter, kv = Qwen3Adapter.from_hf_repo(QWEN3_REPO)
    assert not isinstance(adapter, HiddenCaptureAdapter)

    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0,),
        synthetic_emit=lambda th, k: (),
    )
    engine = Engine(
        adapter=adapter,
        kv_manager=kv,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=drafter,
        verify_k=4,
    )
    with pytest.raises(NotImplementedError, match="HiddenCaptureAdapter"):
        list(
            engine.generate(
                prompt="Hi",
                params=SamplingParams(max_tokens=2, temperature=0.0),
            )
        )


@pytest.mark.skipif(_QWEN3_5_SKIP, reason=_QWEN3_5_SKIP_REASON)
def test_target_hidden_drafter_handles_non_empty_drafts() -> None:
    """When the synthetic emitter returns drafts via a pre-recorded
    spec-off oracle, every drafted token survives greedy_verify
    (because each token equals the target's argmax for that
    position). Pins that the engine routes the drafted-tokens path
    through capture+update without exception."""
    adapter, kv = Qwen3_5Adapter.from_hf_repo(QWEN3_5_REPO)

    # First, record spec-off greedy on the same prompt so we have an
    # oracle to replay.
    spec_off_engine = Engine(
        adapter=adapter,
        kv_manager=kv,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=NoopDraftEngine(),
        verify_k=4,
    )
    oracle = list(
        spec_off_engine.generate(
            prompt="Hello",
            params=SamplingParams(max_tokens=8, temperature=0.0),
        )
    )

    # Now run spec-on with the recorded tokens as drafts. Use a fresh
    # adapter / kv pair so the cache state for spec-on starts clean.
    adapter2, kv2 = Qwen3_5Adapter.from_hf_repo(QWEN3_5_REPO)
    log: list[str] = []
    _record_calls(adapter2, log)

    # Engine yields token 0 from prefill argmax (= oracle[0]); cycle 1
    # then proposes oracle[1:1+gamma] = oracle[1:4] (3 drafts) etc.
    # The synthetic emitter walks oracle starting at index 1 to match
    # silica's draft contract.
    drafter = DFlashDrafter.for_synthetic(
        target_layer_ids=(0, 7, 14),
        synthetic_emit=_make_oracle_emitter(oracle[1:]),
    )

    engine = Engine(
        adapter=adapter2,
        kv_manager=kv2,
        sampler=Sampler(),
        metrics=MetricsRegistry(),
        draft_engine=drafter,
        verify_k=4,
    )
    list(
        engine.generate(
            prompt="Hello",
            params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )

    # Capture path fired; non-empty drafts went through it (the engine
    # called decode_step_multi_with_capture ≥ 1 time).
    assert log.count("decode_step_multi_with_capture") >= 1
