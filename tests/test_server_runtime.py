"""Tests for :mod:`silica.server.runtime` (P-8 sub-unit (a1)).

Pins the Runtime contract introduced at v1.7.32:

- Direct constructor accepts an :class:`I-1 ModelAdapter` + :class:`I-2
  KVManager` and builds an :class:`Engine` against them. No HF network
  load.
- Property accessors return the exact injected objects.
- ``engine_lock`` is an :class:`asyncio.Lock` and is constructible
  outside a running event loop (the FastAPI lifespan startup hook may
  build the runtime before the route layer drives the loop).
- ``close()`` flips ``closed`` from ``False`` to ``True`` once and is
  idempotent — calling it again is a silent no-op (the OPENING leaves
  post-close engine use unguarded; future hardening lands in (h)).
- A custom :class:`MetricsRegistry` injection round-trips into
  ``runtime.engine.metrics``.
"""

from __future__ import annotations

import asyncio

from silica.core.profiler import MetricsRegistry
from silica.engine import Engine
from silica.kvcache.manager import NullKVManager
from silica.models.adapter import StubModelAdapter
from silica.server.runtime import Runtime


def _build_runtime(*, metrics: MetricsRegistry | None = None) -> Runtime:
    return Runtime(
        StubModelAdapter(),
        NullKVManager(),
        model_repo="stub/model",
        metrics=metrics,
    )


def test_direct_constructor_builds_engine_against_injected_stubs() -> None:
    adapter = StubModelAdapter()
    kv = NullKVManager()
    runtime = Runtime(adapter, kv, model_repo="stub/model")

    assert runtime.adapter is adapter
    assert runtime.kv_manager is kv
    assert runtime.model_repo == "stub/model"
    assert isinstance(runtime.engine, Engine)


def test_engine_lock_is_asyncio_lock_constructed_outside_loop() -> None:
    runtime = _build_runtime()

    assert isinstance(runtime.engine_lock, asyncio.Lock)
    assert runtime.engine_lock.locked() is False


def test_metrics_registry_is_threaded_into_engine() -> None:
    metrics = MetricsRegistry()
    runtime = _build_runtime(metrics=metrics)

    assert runtime.engine.metrics is metrics


def test_close_flips_closed_and_is_idempotent() -> None:
    runtime = _build_runtime()
    assert runtime.closed is False

    runtime.close()
    assert runtime.closed is True

    runtime.close()
    assert runtime.closed is True
