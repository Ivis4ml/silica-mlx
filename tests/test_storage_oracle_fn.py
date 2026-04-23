"""Unit tests for :func:`silica.bench.oracles.storage_oracle`.

Pure-function validation — no runner, no engine, no adapter.
``storage_oracle`` is a shape checker on the ``(collected, context)``
pair the runner's ``_run_storage`` produces; these tests synthesize
both inputs directly.
"""

from __future__ import annotations

from typing import Any

from silica.bench.oracles import ORACLES, storage_oracle
from silica.bench.scenario import OracleKind, Scenario, Workload


def _make_scenario() -> Scenario:
    return Scenario(
        id="test-compression",
        repo="dummy/repo",
        workload=Workload(
            name="compression-fake",
            prompts=("a", "a"),
            max_tokens=4,
            max_batch_size=1,
            prefix_cache=True,
            kv_codec="fp16",
        ),
        oracle=OracleKind.STORAGE,
    )


def _valid_collected(
    *,
    resident_bytes: int = 1024,
    resident_bytes_per_block: int | None = 256,
    live_blocks: int = 4,
    prefix_cache_hits: int = 1,
) -> dict[str, Any]:
    return {
        "resident_bytes": resident_bytes,
        "resident_bytes_per_block": resident_bytes_per_block,
        "live_blocks": live_blocks,
        "prefix_cache_hits": prefix_cache_hits,
    }


def _valid_context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {"kv_codec": "fp16", "block_size": 16}
    base.update(overrides)
    return base


# =============================================================================
# Registry + accept
# =============================================================================


class TestRegistry:
    def test_storage_oracle_registered(self) -> None:
        assert OracleKind.STORAGE in ORACLES
        assert ORACLES[OracleKind.STORAGE] is storage_oracle


class TestAcceptsValidInput:
    def test_happy_path(self) -> None:
        ok, reason, metadata = storage_oracle(
            _make_scenario(), _valid_collected(), _valid_context()
        )
        assert ok is True
        assert reason is None
        assert metadata["resident_bytes"] == 1024
        assert metadata["resident_bytes_per_block"] == 256
        assert metadata["live_blocks"] == 4
        assert metadata["prefix_cache_hits"] == 1
        # Context fields forwarded into metadata.
        assert metadata["kv_codec"] == "fp16"
        assert metadata["block_size"] == 16

    def test_resident_bytes_per_block_none_is_accepted(self) -> None:
        """Pass-through stores (no codec) surface
        ``resident_bytes_per_block == None``. The oracle must accept
        it — not every caller pins an explicit codec, and the
        metadata should reflect "unknown per-block" rather than
        fail."""
        ok, _, metadata = storage_oracle(
            _make_scenario(),
            _valid_collected(resident_bytes_per_block=None),
            _valid_context(),
        )
        assert ok is True
        assert metadata["resident_bytes_per_block"] is None

    def test_resident_bytes_zero_with_live_blocks_accepted(self) -> None:
        """Zero resident_bytes with ``live_blocks >= 1`` is
        technically well-formed (empty codec payload). The shape
        guard is ``live_blocks >= 1``, not ``resident_bytes > 0``;
        compression-ratio sanity is a downstream concern."""
        ok, _, _ = storage_oracle(
            _make_scenario(),
            _valid_collected(resident_bytes=0),
            _valid_context(),
        )
        assert ok is True


# =============================================================================
# Rejection paths
# =============================================================================


class TestRejectsMalformedCollected:
    def test_rejects_non_dict_collected(self) -> None:
        ok, reason, _ = storage_oracle(
            _make_scenario(), 42, _valid_context()
        )
        assert ok is False
        assert reason == "storage_collected_shape_mismatch"

    def test_rejects_missing_resident_bytes(self) -> None:
        bad = _valid_collected()
        del bad["resident_bytes"]
        ok, reason, _ = storage_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert "resident_bytes" in reason

    def test_rejects_missing_live_blocks(self) -> None:
        bad = _valid_collected()
        del bad["live_blocks"]
        ok, reason, _ = storage_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert "live_blocks" in reason

    def test_rejects_missing_resident_bytes_per_block(self) -> None:
        """Field must be present (value ``None`` is fine)."""
        bad = _valid_collected()
        del bad["resident_bytes_per_block"]
        ok, reason, _ = storage_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert "resident_bytes_per_block" in reason

    def test_rejects_missing_prefix_cache_hits(self) -> None:
        bad = _valid_collected()
        del bad["prefix_cache_hits"]
        ok, reason, _ = storage_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert "prefix_cache_hits" in reason

    def test_rejects_non_castable_fields(self) -> None:
        bad = _valid_collected()
        bad["resident_bytes"] = "not-a-number"
        ok, reason, _ = storage_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert reason.startswith("storage_collected_field_not_castable")


class TestRejectsNumericViolations:
    def test_rejects_negative_resident_bytes(self) -> None:
        ok, reason, _ = storage_oracle(
            _make_scenario(),
            _valid_collected(resident_bytes=-1),
            _valid_context(),
        )
        assert ok is False
        assert reason == "storage_resident_bytes_negative"

    def test_rejects_zero_live_blocks(self) -> None:
        """Empty-store shape guard: a row that ran but registered
        no detached blocks means the workload did not exercise the
        codec hot path; fail loud."""
        ok, reason, _ = storage_oracle(
            _make_scenario(),
            _valid_collected(live_blocks=0),
            _valid_context(),
        )
        assert ok is False
        assert reason == "storage_live_blocks_below_one"

    def test_rejects_negative_live_blocks(self) -> None:
        ok, reason, _ = storage_oracle(
            _make_scenario(),
            _valid_collected(live_blocks=-1),
            _valid_context(),
        )
        assert ok is False
        assert reason == "storage_live_blocks_below_one"

    def test_rejects_negative_prefix_cache_hits(self) -> None:
        ok, reason, _ = storage_oracle(
            _make_scenario(),
            _valid_collected(prefix_cache_hits=-1),
            _valid_context(),
        )
        assert ok is False
        assert reason == "storage_prefix_cache_hits_negative"

    def test_rejects_nonpositive_resident_bytes_per_block(self) -> None:
        """``resident_bytes_per_block`` may be None (pass-through)
        or a positive int. Zero / negative indicates accounting
        drift — the store always tracks at least one layer of
        per-side bytes per block."""
        ok, reason, _ = storage_oracle(
            _make_scenario(),
            _valid_collected(resident_bytes_per_block=0),
            _valid_context(),
        )
        assert ok is False
        assert reason == "storage_resident_bytes_per_block_not_positive"

        ok, reason, _ = storage_oracle(
            _make_scenario(),
            _valid_collected(resident_bytes_per_block=-5),
            _valid_context(),
        )
        assert ok is False
        assert reason == "storage_resident_bytes_per_block_not_positive"


class TestContextForwarding:
    def test_all_context_fields_surfaced_into_metadata(self) -> None:
        ctx = _valid_context(extra_key="extra_val", kv_codec="block_tq_b64_b4")
        _, _, metadata = storage_oracle(
            _make_scenario(), _valid_collected(), ctx
        )
        assert metadata["kv_codec"] == "block_tq_b64_b4"
        assert metadata["extra_key"] == "extra_val"

    def test_non_dict_context_does_not_crash(self) -> None:
        ok, _, metadata = storage_oracle(
            _make_scenario(), _valid_collected(), "not-a-dict"
        )
        assert ok is True
        # Still carries collected fields.
        assert metadata["resident_bytes"] == 1024

    def test_collected_wins_over_context_on_same_key(self) -> None:
        """Collected's core numbers are load-bearing; a context dict
        with a conflicting key must not overwrite them. ``setdefault``
        semantics."""
        ctx = _valid_context(resident_bytes=-999)
        _, _, metadata = storage_oracle(
            _make_scenario(), _valid_collected(), ctx
        )
        assert metadata["resident_bytes"] == 1024
