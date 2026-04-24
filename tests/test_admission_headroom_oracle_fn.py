"""Unit tests for :func:`silica.bench.oracles.admission_headroom_oracle`.

Pure-function validation on the ``(collected, context)`` pair the
runner's ``_run_admission_headroom`` produces. Does not touch the
runner, budgeter, adapter, or prefix cache.

The oracle enforces two gates:
1. Structural — required fields, types, non-negative counts.
2. §7(c) hard comparison — ``n_block > n_fp16``.

These tests pin both.
"""

from __future__ import annotations

from typing import Any

from silica.bench.oracles import ORACLES, admission_headroom_oracle
from silica.bench.scenario import OracleKind, Scenario, Workload


def _make_scenario() -> Scenario:
    return Scenario(
        id="test-admission-headroom",
        repo="dummy/repo",
        workload=Workload(
            name="admission-headroom-fake",
            prompts=(),
            max_tokens=0,
            max_batch_size=1,
            prefix_cache=False,
            kv_codec=None,
        ),
        oracle=OracleKind.ADMISSION_HEADROOM,
    )


def _valid_collected(
    *,
    n_fp16: int = 3,
    n_block: int = 5,
    resident_bytes_fp16: int = 64 * 1024 * 1024,
    resident_bytes_block: int = 16 * 1024 * 1024,
    warmup_blocks: int = 32,
) -> dict[str, Any]:
    return {
        "n_fp16": n_fp16,
        "n_block": n_block,
        "resident_bytes_fp16": resident_bytes_fp16,
        "resident_bytes_block": resident_bytes_block,
        "warmup_blocks": warmup_blocks,
    }


def _valid_context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "cap_bytes": 128 * 1024 * 1024,
        "weights_bytes": 0,
        "warmup_ratio": 0.5,
        "n_prompt": 128,
        "max_tokens": 16,
        "fp16_codec": "fp16",
        "compressed_codec": "block_tq_b64_b4",
    }
    base.update(overrides)
    return base


# =============================================================================
# Registry + acceptance
# =============================================================================


class TestRegistry:
    def test_admission_headroom_registered(self) -> None:
        assert OracleKind.ADMISSION_HEADROOM in ORACLES
        assert (
            ORACLES[OracleKind.ADMISSION_HEADROOM]
            is admission_headroom_oracle
        )


class TestAcceptsValidInput:
    def test_happy_path_n_block_greater(self) -> None:
        ok, reason, metadata = admission_headroom_oracle(
            _make_scenario(), _valid_collected(), _valid_context()
        )
        assert ok is True
        assert reason is None
        assert metadata["n_fp16"] == 3
        assert metadata["n_block"] == 5
        assert metadata["n_delta"] == 2
        assert metadata["admit_ratio"] == 5 / 3
        assert metadata["residency_ratio"] == (
            16 * 1024 * 1024
        ) / (64 * 1024 * 1024)
        # Context fields forwarded into metadata.
        assert metadata["cap_bytes"] == 128 * 1024 * 1024
        assert metadata["warmup_ratio"] == 0.5

    def test_admit_ratio_inf_when_n_fp16_zero_but_n_block_positive(
        self,
    ) -> None:
        """Zero fp16 admissions is legal (cap too small for trial
        admissions under fp16); if compressed codec admits any,
        the oracle still passes the n_block > n_fp16 gate and
        reports an infinite ratio."""
        import math

        ok, _, metadata = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(n_fp16=0, n_block=2),
            _valid_context(),
        )
        assert ok is True
        assert metadata["admit_ratio"] == math.inf
        assert metadata["n_delta"] == 2


# =============================================================================
# Rejection paths — structural
# =============================================================================


class TestRejectsMalformedCollected:
    def test_rejects_non_dict_collected(self) -> None:
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(), 42, _valid_context()
        )
        assert ok is False
        assert reason == "admission_headroom_collected_shape_mismatch"

    def test_rejects_missing_n_fp16(self) -> None:
        bad = _valid_collected()
        del bad["n_fp16"]
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert "n_fp16" in reason

    def test_rejects_missing_n_block(self) -> None:
        bad = _valid_collected()
        del bad["n_block"]
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert "n_block" in reason

    def test_rejects_missing_warmup_blocks(self) -> None:
        bad = _valid_collected()
        del bad["warmup_blocks"]
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert "warmup_blocks" in reason

    def test_rejects_non_castable_fields(self) -> None:
        bad = _valid_collected()
        bad["n_fp16"] = "not-a-number"
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(), bad, _valid_context()
        )
        assert ok is False
        assert reason is not None
        assert reason.startswith(
            "admission_headroom_field_not_castable"
        )


# =============================================================================
# Rejection paths — numeric
# =============================================================================


class TestRejectsNumericViolations:
    def test_rejects_negative_n_fp16(self) -> None:
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(n_fp16=-1),
            _valid_context(),
        )
        assert ok is False
        assert reason == "admission_headroom_admission_count_negative"

    def test_rejects_negative_n_block(self) -> None:
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(n_block=-1, n_fp16=0),
            _valid_context(),
        )
        assert ok is False
        assert reason == "admission_headroom_admission_count_negative"

    def test_rejects_negative_resident_bytes_fp16(self) -> None:
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(resident_bytes_fp16=-1),
            _valid_context(),
        )
        assert ok is False
        assert reason == "admission_headroom_resident_bytes_negative"

    def test_rejects_zero_warmup_blocks(self) -> None:
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(warmup_blocks=0),
            _valid_context(),
        )
        assert ok is False
        assert reason == "admission_headroom_warmup_blocks_below_one"


# =============================================================================
# §7(c) hard gate: n_block > n_fp16
# =============================================================================


class TestN7cGate:
    def test_rejects_n_block_equal_n_fp16(self) -> None:
        """Equality would mean compression produced zero
        admission headroom benefit — either no compression
        (both sides IdentityCodec) or a budgeter regression
        where headroom_bytes() ignores store.resident_bytes()."""
        ok, reason, metadata = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(n_fp16=5, n_block=5),
            _valid_context(),
        )
        assert ok is False
        assert (
            reason
            == "admission_headroom_n_block_not_greater_than_n_fp16"
        )
        assert metadata["n_fp16"] == 5
        assert metadata["n_block"] == 5

    def test_rejects_n_block_below_n_fp16(self) -> None:
        ok, reason, _ = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(n_fp16=5, n_block=3),
            _valid_context(),
        )
        assert ok is False
        assert (
            reason
            == "admission_headroom_n_block_not_greater_than_n_fp16"
        )

    def test_accepts_n_block_strictly_greater(self) -> None:
        ok, _, metadata = admission_headroom_oracle(
            _make_scenario(),
            _valid_collected(n_fp16=2, n_block=3),
            _valid_context(),
        )
        assert ok is True
        assert metadata["n_delta"] == 1


# =============================================================================
# Context forwarding
# =============================================================================


class TestContextForwarding:
    def test_all_context_fields_surfaced(self) -> None:
        ctx = _valid_context(
            cap_bytes=64 * 1024 * 1024,
            fp16_codec="fp16",
            compressed_codec="ext_rabitq_b4",
        )
        _, _, metadata = admission_headroom_oracle(
            _make_scenario(), _valid_collected(), ctx
        )
        assert metadata["cap_bytes"] == 64 * 1024 * 1024
        assert metadata["compressed_codec"] == "ext_rabitq_b4"

    def test_non_dict_context_does_not_crash(self) -> None:
        ok, _, metadata = admission_headroom_oracle(
            _make_scenario(), _valid_collected(), "not-a-dict"
        )
        assert ok is True
        assert metadata["n_fp16"] == 3

    def test_collected_wins_over_context_on_same_key(self) -> None:
        ctx = _valid_context(n_fp16=-999)
        _, _, metadata = admission_headroom_oracle(
            _make_scenario(), _valid_collected(), ctx
        )
        assert metadata["n_fp16"] == 3
