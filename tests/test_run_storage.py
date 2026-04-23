"""Runner-level tests for :func:`silica.bench.runner._run_storage`.

Tests the shape-validation + workload-drive + store-read chain
without loading real weights. A fake engine's ``generate_batch``
mimics what ``ContinuousBatcher._extract_and_insert_prefix`` does
on row-0 termination: calls ``prefix_cache.insert_detached(...)``
with synthetic per-layer K/V that the configured codec happily
accepts. The store's ``resident_bytes()`` then reports an honest
per-codec residency, exactly the observable the STORAGE oracle
consumes.

Invariants pinned:

1. Shape-validation errors raise ``RuntimeError`` before the
   engine factory runs (handled at the outer ``BenchRunner.run``
   boundary — the helper itself raises synchronously so the
   caller decides whether to wrap).
2. Happy path: fp16 row with codec-backed store yields
   ``(collected, context)`` satisfying the oracle's structural
   contract.
3. Empty-store shape guard: a workload that drains events
   without registering any detached blocks raises
   ``RuntimeError``; the oracle never sees a zero-``live_blocks``
   row.
4. ``resident_bytes_per_block`` is a positive int under the
   codec-backed paths (IdentityCodec + TurboQuantMSE); the
   oracle's per-block positivity check is compatible with what
   the runner actually produces.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, cast

import mlx.core as mx
import pytest

from silica.bench.runner import _run_storage
from silica.bench.scenario import OracleKind, Scenario, Workload
from silica.models.adapter import KVLayout, ModelAdapter

_SHARED_PROMPT = "test prompt for shared prefix compression row"
_N_KV_HEADS = 2
_HEAD_DIM = 8
_N_LAYERS = 1
_BLOCK_SIZE = 16  # _maybe_build_prefix_cache hardcodes this


# =============================================================================
# Fake adapter + engine
# =============================================================================


class _FakeConfig:
    model_name = "stub-storage"
    num_layers = _N_LAYERS
    hidden_size = 64
    vocab_size = 256


class _FakeTokenizer:
    eos_token_ids: tuple[int, ...] = ()

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class _FakeAdapter:
    """Adapter fake exposing the kv_layout + tokenizer + config the
    runner consumes. ``_model`` / ``make_batch_cache`` are unused
    on the STORAGE path because ``_run_storage`` drives the engine
    (not the oracle) to populate the store."""

    config = _FakeConfig()

    def kv_layout(self) -> KVLayout:
        return KVLayout(
            num_layers=_N_LAYERS,
            n_kv_heads=_N_KV_HEADS,
            head_dim=_HEAD_DIM,
            dtype=mx.float16,
        )

    def tokenizer(self) -> Any:
        return _FakeTokenizer()


class _StorePopulatingFakeEngine:
    """Fake engine whose ``generate_batch`` inserts synthetic
    detached blocks into ``prefix_cache``, mimicking the
    scheduler's row-0-termination ``_extract_and_insert_prefix``
    path without needing a real batcher.

    The event stream is a minimal token / done pair per row so
    the collector does not hang; ``_run_storage`` drains it and
    reads the store afterwards.
    """

    def __init__(
        self,
        *,
        n_blocks: int = 2,
        simulate_hit: bool = True,
    ) -> None:
        from silica.core.profiler import MetricsRegistry

        self._n_blocks = n_blocks
        self._simulate_hit = simulate_hit
        self.metrics = MetricsRegistry()

    def generate_batch(
        self,
        prompts: Iterable[str],
        params: Any,
        *,
        max_batch_size: int = 1,
        prefix_cache: Any | None = None,
        length_spread_threshold: float = float("inf"),
    ) -> Iterator[Any]:
        from silica.core.events import BatchEvent

        if prefix_cache is not None and self._n_blocks > 0:
            block_size = prefix_cache.block_size
            tokens = list(range(block_size * self._n_blocks))
            detached: list[list[tuple[mx.array, mx.array]]] = []
            for _ in range(self._n_blocks):
                per_layer: list[tuple[mx.array, mx.array]] = []
                for _ in range(_N_LAYERS):
                    k = mx.zeros(
                        (1, _N_KV_HEADS, block_size, _HEAD_DIM),
                        dtype=mx.float16,
                    )
                    v = mx.zeros(
                        (1, _N_KV_HEADS, block_size, _HEAD_DIM),
                        dtype=mx.float16,
                    )
                    per_layer.append((k, v))
                detached.append(per_layer)
            prefix_cache.insert_detached(tokens, detached)
            if self._simulate_hit:
                prefix_cache.hits += 1

        yield BatchEvent(kind="token", req_index=0, token_id=1)
        yield BatchEvent(kind="done", req_index=0, finish_reason="max_tokens")
        yield BatchEvent(kind="token", req_index=1, token_id=2)
        yield BatchEvent(kind="done", req_index=1, finish_reason="max_tokens")


_TEST_REPO = "Qwen/Qwen3-0.6B"


def _make_scenario(
    *,
    kv_codec: str | None = "fp16",
    prompts: tuple[str, ...] = (_SHARED_PROMPT, _SHARED_PROMPT),
    max_batch_size: int = 1,
    prefix_cache: bool = True,
) -> Scenario:
    """Build a storage-oracle scenario. Uses ``Qwen/Qwen3-0.6B`` as
    the repo so the pre-load ``_check_gates`` HF-cache check can be
    neutralized via a ``monkeypatch`` on ``Path.home()`` pointing at
    a tmp dir that contains the expected ``models--Qwen--Qwen3-0.6B``
    directory (helper: :func:`_fake_hf_cache`). Same pattern
    ``test_bench_prefix_hit_decode_runner`` relies on."""
    return Scenario(
        id=f"test-compression-{kv_codec}",
        repo=_TEST_REPO,
        workload=Workload(
            name="compression-fake",
            prompts=prompts,
            max_tokens=4,
            max_batch_size=max_batch_size,
            prefix_cache=prefix_cache,
            kv_codec=kv_codec,
        ),
        oracle=OracleKind.STORAGE,
    )


@pytest.fixture
def _fake_hf_cache(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect ``Path.home()`` to ``tmp_path`` and create the
    HF-style cache directory for ``_TEST_REPO`` inside it, so
    ``_check_gates`` sees a present cache without this test
    depending on the real HF hub layout on the dev machine.
    """
    from pathlib import Path as _Path

    monkeypatch.setattr(
        _Path, "home", classmethod(lambda cls: tmp_path)
    )
    safe = _TEST_REPO.replace("/", "--")
    (
        tmp_path / ".cache" / "huggingface" / "hub" / f"models--{safe}"
    ).mkdir(parents=True)


def _call_run_storage(
    scenario: Scenario,
    engine: _StorePopulatingFakeEngine,
    adapter: _FakeAdapter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cast wrapper so the fake adapter/engine satisfy mypy at the
    call boundary. The runner helper needs only the subset of
    ``ModelAdapter`` / ``Engine`` that _FakeAdapter / _FakeEngine
    implement."""
    from silica.engine import Engine

    return _run_storage(
        scenario,
        cast(Engine, engine),
        cast(ModelAdapter, adapter),
    )


# =============================================================================
# Happy paths (fp16 + codec rows)
# =============================================================================


class TestHappyPath:
    def test_fp16_row_populates_collected_and_context(self) -> None:
        scenario = _make_scenario(kv_codec="fp16")
        engine = _StorePopulatingFakeEngine(n_blocks=3)
        adapter = _FakeAdapter()

        collected, context = _call_run_storage(
            scenario, engine, adapter
        )

        assert collected["live_blocks"] == 3
        assert collected["resident_bytes"] > 0
        assert collected["resident_bytes_per_block"] is not None
        assert collected["resident_bytes_per_block"] > 0
        assert collected["prefix_cache_hits"] == 1

        assert context["kv_codec"] == "fp16"
        assert context["block_size"] == _BLOCK_SIZE

    def test_resident_bytes_equals_blocks_times_per_block(self) -> None:
        """Under IdentityCodec all detached blocks have uniform
        size, so total == blocks × per-block. Spot-check the
        accounting so a store drift would surface here."""
        scenario = _make_scenario(kv_codec="fp16")
        engine = _StorePopulatingFakeEngine(n_blocks=4)
        adapter = _FakeAdapter()

        collected, _ = _call_run_storage(scenario, engine, adapter)
        assert collected["resident_bytes"] == (
            collected["live_blocks"]
            * collected["resident_bytes_per_block"]
        )

    def test_tq_mse_b4_row_returns_smaller_per_block_than_fp16(
        self,
    ) -> None:
        """TurboQuantMSE 4-bit packed payload should be strictly
        smaller per-block than IdentityCodec fp16. Not a
        compression-ratio gate (cross-row is a bench report
        concern) but a simple sanity check that the codec path
        actually reports different numbers."""
        adapter = _FakeAdapter()

        fp16_collected, _ = _call_run_storage(
            _make_scenario(kv_codec="fp16"),
            _StorePopulatingFakeEngine(n_blocks=2),
            adapter,
        )
        tq_collected, _ = _call_run_storage(
            _make_scenario(kv_codec="tq_mse_b4"),
            _StorePopulatingFakeEngine(n_blocks=2),
            adapter,
        )

        assert (
            tq_collected["resident_bytes_per_block"]
            < fp16_collected["resident_bytes_per_block"]
        ), (
            "TurboQuantMSE should pack K/V smaller than "
            "IdentityCodec fp16 at the same block shape"
        )


# =============================================================================
# Pre-load workload validation — mirrors the A.3b review M-2 pattern
# applied to DECODE_TOK_S_WITH_PREFIX_HIT. Authoring-error STORAGE
# scenarios must fail in ``_validate_workload_for_oracle`` BEFORE
# the engine factory loads weights. Load-bearing invariant:
# ``probe.calls == []`` — no ~600 MB Qwen3-0.6B load on a broken
# STORAGE row.
# =============================================================================


class _FactoryProbe:
    """engine_factory wrapper that records invocations so tests can
    assert it was (or was not) called."""

    def __init__(self) -> None:
        self.calls: list[Scenario] = []

    def __call__(self, scenario: Scenario) -> tuple[Any, Any]:
        self.calls.append(scenario)
        return _FakeAdapter(), _StorePopulatingFakeEngine()


def _make_runner(factory: _FactoryProbe) -> Any:
    from pathlib import Path

    from silica.bench.runner import BenchRunner

    return BenchRunner(
        engine_factory=factory,
        out_path=Path("/tmp/silica_storage_runner_test.jsonl"),
        clock=lambda: 0.0,
        reset_peak=lambda: None,
        read_peak_mb=lambda: 0.0,
    )


@pytest.mark.parametrize(
    "broken_workload_kwargs,expected_reason_substring",
    [
        (
            {"max_batch_size": 2},
            "max_batch_size=1",
        ),
        (
            {"prompts": ("one-prompt-only",)},
            "exactly 2 prompts",
        ),
        (
            {"prompts": ("different", "prompts-here")},
            "identical prompts",
        ),
        (
            {"prefix_cache": False, "kv_codec": None},
            "prefix_cache=True",
        ),
    ],
)
def test_invalid_workload_fails_before_engine_factory(
    broken_workload_kwargs: dict[str, Any],
    expected_reason_substring: str,
    _fake_hf_cache: None,
) -> None:
    """Authoring errors on the STORAGE oracle surface via
    ``_validate_workload_for_oracle`` and short-circuit
    ``_run_one`` before the adapter + engine get loaded. Mirrors
    ``test_bench_prefix_hit_decode_runner``'s M-2 regression."""
    base = {
        "prompts": (_SHARED_PROMPT, _SHARED_PROMPT),
        "max_batch_size": 1,
        "prefix_cache": True,
        "kv_codec": "fp16",
    }
    kwargs = {**base, **broken_workload_kwargs}
    scenario = _make_scenario(
        kv_codec=kwargs["kv_codec"],
        prompts=kwargs["prompts"],
        max_batch_size=kwargs["max_batch_size"],
        prefix_cache=kwargs["prefix_cache"],
    )

    probe = _FactoryProbe()
    runner = _make_runner(probe)
    result = runner._run_one(scenario)

    assert result.status == "failed"
    assert result.reason is not None
    assert expected_reason_substring in result.reason, (
        f"reason {result.reason!r} missing {expected_reason_substring!r}"
    )
    assert probe.calls == [], (
        "engine_factory should not be invoked when STORAGE workload "
        f"validation fails; got {len(probe.calls)} call(s)"
    )


# =============================================================================
# Empty-store guard — a runtime-only invariant (shape guards fire
# pre-load in _validate_workload_for_oracle, tested above).
# =============================================================================


class TestEmptyStoreGuard:
    def test_zero_blocks_raises_runtime_error(self) -> None:
        """If the workload drains its event stream without the
        scheduler ever calling ``_extract_and_insert_prefix``
        (scheduler regression, codec failure, etc.), the store is
        empty; ``_run_storage`` must raise loud so the oracle
        never sees a 0-block measurement.

        Simulated here by a fake engine configured with
        ``n_blocks=0``."""
        scenario = _make_scenario(kv_codec="fp16")
        engine = _StorePopulatingFakeEngine(n_blocks=0)
        adapter = _FakeAdapter()

        with pytest.raises(
            RuntimeError, match="zero detached blocks"
        ):
            _call_run_storage(scenario, engine, adapter)


# =============================================================================
# Event-stream validation (mirrors _collect_prefix_hit_decode)
# =============================================================================


class TestEventStreamValidation:
    """``_run_storage`` must reject aborted rows, unexpected
    ``req_index`` values, and rows that never emit ``done``.
    Skipping this check would let the store snapshot succeed
    against a half-completed workload and the STORAGE oracle
    would surface a residency number from a failed run as if it
    were valid."""

    def test_aborted_event_raises(self) -> None:
        from silica.core.events import BatchEvent

        class _AbortingEngine(_StorePopulatingFakeEngine):
            def generate_batch(
                self,
                prompts: Iterable[str],
                params: Any,
                *,
                max_batch_size: int = 1,
                prefix_cache: Any | None = None,
                length_spread_threshold: float = float("inf"),
            ) -> Iterator[Any]:
                # Populate the store (row 0 terminated cleanly) then
                # abort row 1 — without event validation, the store
                # snapshot would still be "plausible" and the oracle
                # would pass on half-completed data.
                if prefix_cache is not None:
                    block_size = prefix_cache.block_size
                    tokens = list(range(block_size))
                    per_layer = [
                        (
                            mx.zeros(
                                (1, _N_KV_HEADS, block_size, _HEAD_DIM),
                                dtype=mx.float16,
                            ),
                            mx.zeros(
                                (1, _N_KV_HEADS, block_size, _HEAD_DIM),
                                dtype=mx.float16,
                            ),
                        )
                    ]
                    prefix_cache.insert_detached(tokens, [per_layer])
                yield BatchEvent(
                    kind="token", req_index=0, token_id=1
                )
                yield BatchEvent(
                    kind="done", req_index=0, finish_reason="max_tokens"
                )
                yield BatchEvent(
                    kind="aborted",
                    req_index=1,
                    finish_reason="preempted",
                )

        scenario = _make_scenario(kv_codec="fp16")
        with pytest.raises(RuntimeError, match="storage_aborted"):
            _call_run_storage(scenario, _AbortingEngine(), _FakeAdapter())

    def test_unexpected_req_index_raises(self) -> None:
        from silica.core.events import BatchEvent

        class _UnknownRowEngine(_StorePopulatingFakeEngine):
            def generate_batch(
                self,
                prompts: Iterable[str],
                params: Any,
                *,
                max_batch_size: int = 1,
                prefix_cache: Any | None = None,
                length_spread_threshold: float = float("inf"),
            ) -> Iterator[Any]:
                yield BatchEvent(
                    kind="token", req_index=7, token_id=1
                )

        scenario = _make_scenario(kv_codec="fp16")
        with pytest.raises(
            RuntimeError, match="storage_unexpected_req_index"
        ):
            _call_run_storage(
                scenario, _UnknownRowEngine(), _FakeAdapter()
            )

    def test_row_never_completes_raises(self) -> None:
        from silica.core.events import BatchEvent

        class _Row1NeverDoneEngine(_StorePopulatingFakeEngine):
            def generate_batch(
                self,
                prompts: Iterable[str],
                params: Any,
                *,
                max_batch_size: int = 1,
                prefix_cache: Any | None = None,
                length_spread_threshold: float = float("inf"),
            ) -> Iterator[Any]:
                # Populate store so empty-store guard is bypassed;
                # the missing-done guard is what we're testing.
                if prefix_cache is not None:
                    block_size = prefix_cache.block_size
                    tokens = list(range(block_size))
                    per_layer = [
                        (
                            mx.zeros(
                                (1, _N_KV_HEADS, block_size, _HEAD_DIM),
                                dtype=mx.float16,
                            ),
                            mx.zeros(
                                (1, _N_KV_HEADS, block_size, _HEAD_DIM),
                                dtype=mx.float16,
                            ),
                        )
                    ]
                    prefix_cache.insert_detached(tokens, [per_layer])
                yield BatchEvent(
                    kind="token", req_index=0, token_id=1
                )
                yield BatchEvent(
                    kind="done", req_index=0, finish_reason="max_tokens"
                )
                yield BatchEvent(
                    kind="token", req_index=1, token_id=2
                )
                # Row 1 never emits done.

        scenario = _make_scenario(kv_codec="fp16")
        with pytest.raises(
            RuntimeError, match="storage_rows_never_completed"
        ):
            _call_run_storage(
                scenario, _Row1NeverDoneEngine(), _FakeAdapter()
            )
