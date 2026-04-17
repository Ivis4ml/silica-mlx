"""Tests for silica.core.profiler.

Covers:
  - UnifiedMetrics schema matches P-0 Acceptance floor fields.
  - time_block records elapsed + tags + exception-safe teardown.
  - MetricsRegistry routes unified vs extra metric names, snapshot isolation.
  - Global registry singleton behavior.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterator

import pytest

from silica.core.profiler import (
    MetricsRegistry,
    TimingRecord,
    UnifiedMetrics,
    registry,
    time_block,
)


@pytest.fixture(autouse=True)
def reset_global_registry() -> Iterator[None]:
    registry().reset()
    yield
    registry().reset()


# --- schema contract (P-0 Acceptance) ---


def test_unified_metrics_floor_fields_present() -> None:
    """P-0 Acceptance mandates these five fields; test pins the schema shape."""
    field_names = {f.name for f in dataclasses.fields(UnifiedMetrics)}
    required = {"ttft_ms", "prefill_tok_s", "decode_tok_s", "resident_mb", "logical_kv_bytes"}
    assert required.issubset(field_names)


def test_unified_metrics_defaults_are_none() -> None:
    """None distinguishes 'not yet measured' from 0.0 (a legitimate measurement)."""
    m = UnifiedMetrics()
    assert m.ttft_ms is None
    assert m.prefill_tok_s is None
    assert m.decode_tok_s is None
    assert m.resident_mb is None
    assert m.logical_kv_bytes is None
    assert m.extra == {}


# --- time_block ---


def test_time_block_records_elapsed() -> None:
    with time_block("probe") as t:
        time.sleep(0.01)
    assert t.elapsed_s is not None
    assert t.elapsed_s >= 0.01


def test_time_block_stores_tags() -> None:
    with time_block("prefill") as t:
        t.tags["num_tokens"] = 42
    assert t.tags["num_tokens"] == 42


def test_time_block_pushes_to_registry() -> None:
    with time_block("stage-a"):
        pass
    recs = registry().timings("stage-a")
    assert len(recs) == 1
    assert recs[0].elapsed_s is not None


def test_time_block_record_false_skips_registry() -> None:
    with time_block("local-only", record=False) as t:
        pass
    assert registry().timings("local-only") == []
    assert t.elapsed_s is not None  # still measured, just not stored globally


def test_time_block_sets_elapsed_even_on_exception() -> None:
    rec: TimingRecord | None = None
    with pytest.raises(RuntimeError):
        with time_block("boom", record=False) as t:
            rec = t
            raise RuntimeError("kaboom")
    assert rec is not None
    assert rec.elapsed_s is not None


# --- MetricsRegistry typed vs extra routing ---


def test_registry_routes_unified_names_to_typed_fields() -> None:
    r = MetricsRegistry()
    r.set_metric("ttft_ms", 12.3)
    r.set_metric("resident_mb", 100.0)
    r.set_metric("logical_kv_bytes", 1024)
    snap = r.snapshot()
    assert snap.ttft_ms == 12.3
    assert snap.resident_mb == 100.0
    assert snap.logical_kv_bytes == 1024
    assert snap.extra == {}


def test_registry_routes_unknown_names_to_extra() -> None:
    r = MetricsRegistry()
    r.set_metric("custom_counter", 7)
    r.set_metric("another_gauge", 3.14)
    snap = r.snapshot()
    assert snap.extra == {"custom_counter": 7, "another_gauge": 3.14}


def test_registry_get_metric_roundtrips() -> None:
    r = MetricsRegistry()
    r.set_metric("ttft_ms", 5.5)
    r.set_metric("custom", 99)
    assert r.get_metric("ttft_ms") == 5.5
    assert r.get_metric("custom") == 99
    assert r.get_metric("unset") is None


def test_registry_snapshot_is_detached() -> None:
    r = MetricsRegistry()
    r.set_metric("ttft_ms", 1.0)
    r.set_metric("custom", 2)
    snap = r.snapshot()
    snap.ttft_ms = 999.0
    snap.extra["custom"] = 999
    snap2 = r.snapshot()
    assert snap2.ttft_ms == 1.0
    assert snap2.extra["custom"] == 2


def test_registry_reset_clears_state() -> None:
    r = MetricsRegistry()
    r.set_metric("ttft_ms", 1.0)
    r.record_timing(TimingRecord(name="foo", started_at=0.0, elapsed_s=0.1))
    r.reset()
    assert r.snapshot() == UnifiedMetrics()
    assert r.timings() == []


# --- global accessor ---


def test_registry_is_process_global() -> None:
    r1 = registry()
    r2 = registry()
    assert r1 is r2


def test_global_registry_captures_time_block_output() -> None:
    with time_block("prefill"):
        pass
    snap_before = registry().snapshot()
    assert snap_before.prefill_tok_s is None  # no gauge yet
    registry().set_metric("prefill_tok_s", 1234.5)
    assert registry().snapshot().prefill_tok_s == 1234.5
