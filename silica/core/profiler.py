"""silica.core.profiler — timing + metrics collection under a MetricsRegistry.

P-0 Strategy:
  @contextmanager for timing blocks + MetricsRegistry + unified metrics schema.

P-0 Acceptance (floor set of fields):
  ttft_ms, prefill_tok_s, decode_tok_s, resident_mb, logical_kv_bytes.

D-012 canonical resident_bytes measurement:
  `resident_mb` is derived from sum(component.resident_bytes()) across KV /
  weight / other components, each implementing D-012's measurement contract.
  This module only stores the aggregated gauge; actual component accounting
  lives in `silica.kvcache` and `silica.weights`.

Q-005 (open):
  v0.1 uses a process-global MetricsRegistry (P-0 Strategy lean). The class
  itself is plain — instantiate locally if isolation is needed — so if Q-005
  later settles on per-Engine registries, only the global accessor changes.

Threading:
  v0.1 is single-process single-thread (PLAN §5.3). Add locking when the
  scheduler introduces tokenizer / detokenizer workers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

_UNIFIED_FIELDS: frozenset[str] = frozenset(
    {
        "ttft_ms",
        "prefill_tok_s",
        "decode_tok_s",
        "resident_mb",
        "logical_kv_bytes",
    }
)


@dataclass
class TimingRecord:
    """A single timing block: name, entry timestamp, elapsed, and free-form tags.

    `tags` is populated inside the `time_block` context so blocks can attach
    their own payload (e.g. `{"num_tokens": 128}`).
    """

    name: str
    started_at: float
    elapsed_s: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedMetrics:
    """Floor schema mandated by P-0 Acceptance plus a free-form `extra` dict.

    Every field defaults to None to distinguish "not yet measured" from a
    measured zero — bench output must preserve this distinction.
    """

    ttft_ms: float | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    resident_mb: float | None = None
    logical_kv_bytes: int | None = None
    extra: dict[str, float | int] = field(default_factory=dict)


class MetricsRegistry:
    """Collects TimingRecords and scalar metrics.

    Metric names in the unified schema go to the typed fields of
    `UnifiedMetrics`; unknown names fall into `UnifiedMetrics.extra`.
    """

    def __init__(self) -> None:
        self._timings: list[TimingRecord] = []
        self._metrics = UnifiedMetrics()

    def record_timing(self, record: TimingRecord) -> None:
        self._timings.append(record)

    def set_metric(self, name: str, value: float | int) -> None:
        if name in _UNIFIED_FIELDS:
            setattr(self._metrics, name, value)
        else:
            self._metrics.extra[name] = value

    def get_metric(self, name: str) -> float | int | None:
        if name in _UNIFIED_FIELDS:
            v: float | int | None = getattr(self._metrics, name)
            return v
        return self._metrics.extra.get(name)

    def timings(self, name: str | None = None) -> list[TimingRecord]:
        if name is None:
            return list(self._timings)
        return [t for t in self._timings if t.name == name]

    def snapshot(self) -> UnifiedMetrics:
        """Return a detached copy — mutating it does not affect the registry."""
        return replace(self._metrics, extra=dict(self._metrics.extra))

    def reset(self) -> None:
        self._timings.clear()
        self._metrics = UnifiedMetrics()


_global_registry = MetricsRegistry()


def registry() -> MetricsRegistry:
    """Return the process-global MetricsRegistry (Q-005 default — see module docs)."""
    return _global_registry


@contextmanager
def time_block(name: str, *, record: bool = True) -> Iterator[TimingRecord]:
    """Time a code block with `time.perf_counter`, optionally record to registry.

    Example:
        with time_block("prefill") as t:
            t.tags["num_tokens"] = 128
            ...  # work
        # t.elapsed_s is populated on exit, regardless of exception
    """
    started = time.perf_counter()
    rec = TimingRecord(name=name, started_at=started)
    try:
        yield rec
    finally:
        rec.elapsed_s = time.perf_counter() - started
        if record:
            _global_registry.record_timing(rec)
