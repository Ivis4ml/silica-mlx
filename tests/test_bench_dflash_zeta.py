"""D-021 step 6 sub-unit (ζ) — bench wiring tests for C.4 DFlash.

Pure code, no checkpoint download. Pins:

- ``--speculative`` parser accepts ``dflash`` and rejects unknown values.
- The two ``-c4-dflash`` scenario rows are registered in
  ``BUILTIN_SCENARIOS`` with the expected target/drafter repos, gates,
  and ``spec_config.kind == "dflash"``.
- ``--list`` enumerates them so users discover the rows without source
  inspection.
- ``BenchRunner._check_gates`` skips ``dflash`` rows when the drafter
  cache is missing, with a descriptive ``draft_cache_missing:`` reason.
- ``BenchRunner._check_gates`` skips ``dflash`` rows when the drafter
  env var is missing, with ``draft_env_var_not_set:`` reason.
- ``BenchRunner._check_gates`` runs the ``dflash`` row spec-off (no
  drafter checks) under ``--speculative draft_target`` because the
  ``spec_config.kind`` does not match.
- ``_default_engine_factory`` dispatches to ``DFlashDrafter`` when the
  scenario's ``kind == "dflash"`` and CLI mode matches; the test
  monkey-patches ``DFlashDrafter`` so no ``dflash-mlx`` import or HF
  download fires.

Real-checkpoint attestation belongs to (η).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from silica.bench.runner import BenchRunner, _check_gates
from silica.bench.scenario import OracleKind, SpecConfig
from silica.bench.scenarios import BUILTIN_SCENARIOS, get_scenario


def _load_bench_cli_module() -> Any:
    """Load ``scripts/bench.py`` as a fresh module so each test gets
    an unconfigured argparse namespace."""
    here = Path(__file__).resolve().parents[1]
    path = here / "scripts" / "bench.py"
    spec = importlib.util.spec_from_file_location("_bench_cli_zeta", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- argparse + --list ----------------------------------------------------


class TestSpeculativeDflashFlag:
    def test_dflash_choice_accepted(self) -> None:
        bench = _load_bench_cli_module()
        ns = bench.build_parser().parse_args(["--speculative", "dflash"])
        assert ns.speculative == "dflash"

    def test_choices_set_includes_three_values(self) -> None:
        bench = _load_bench_cli_module()
        # argparse stores choices on the action; pull from the parser.
        action = next(
            a
            for a in bench.build_parser()._actions
            if "--speculative" in (a.option_strings or [])
        )
        assert sorted(action.choices) == ["dflash", "draft_target", "none"]

    def test_unknown_value_rejected(self) -> None:
        bench = _load_bench_cli_module()
        with pytest.raises(SystemExit):
            bench.build_parser().parse_args(["--speculative", "auto"])

    def test_help_mentions_dflash(self) -> None:
        bench = _load_bench_cli_module()
        helptext = bench.build_parser().format_help()
        assert "dflash" in helptext


def test_cli_list_surfaces_c4_dflash_rows() -> None:
    """``python -m scripts.bench --list`` enumerates the two new rows
    so users find them by ID without grepping source."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.bench", "--list"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "qwen3.5-27b-warm-decode-c4-dflash" in result.stdout
    assert "qwen3.5-moe-35b-a3b-warm-decode-c4-dflash" in result.stdout


def test_builtin_scenarios_count_is_at_least_67() -> None:
    """Catalog grew from v1.7.19's 65 to 65 + 2 = 67 with the (ζ)
    additions; later sub-units may add more rows (D-021 step 7
    Track B added a 3-bit b1 row at v1.7.20+1 → 68). The (ζ)
    invariant is that ≥67 scenarios are registered, not that
    the count is exactly 67."""
    assert len(BUILTIN_SCENARIOS) >= 67


# --- scenario shape -------------------------------------------------------


def test_qwen3_5_27b_c4_dflash_row_shape() -> None:
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")
    assert sc.repo == "mlx-community/Qwen3.5-27B-4bit"
    assert sc.gate_env_var == "SILICA_REAL_QWEN3_5_27B"
    assert sc.oracle == OracleKind.WARM_DECODE
    cfg = sc.spec_config
    assert cfg is not None
    assert cfg.draft_repo == "z-lab/Qwen3.5-27B-DFlash"
    assert cfg.verify_k == 16
    assert cfg.draft_gate_env_var == "SILICA_BENCH_DFLASH_27B"
    assert cfg.kind == "dflash"


def test_qwen3_5_moe_c4_dflash_row_shape() -> None:
    sc = get_scenario("qwen3.5-moe-35b-a3b-warm-decode-c4-dflash")
    assert sc.repo == "mlx-community/Qwen3.5-35B-A3B-4bit"
    assert sc.gate_env_var == "SILICA_REAL_QWEN3_5_MOE"
    cfg = sc.spec_config
    assert cfg is not None
    assert cfg.draft_repo == "z-lab/Qwen3.5-35B-A3B-DFlash"
    assert cfg.verify_k == 16
    assert cfg.draft_gate_env_var == "SILICA_BENCH_DFLASH_35B_A3B"
    assert cfg.kind == "dflash"


def test_c4_dflash_rows_share_workload_with_b1_cousins() -> None:
    """C.4 dflash rows mirror the b1 baseline shape so the spec-on /
    spec-off ratio is comparable. Drift here would invalidate η's
    silica-integrated speedup denominator."""
    pairs = [
        (
            "qwen3.5-27b-warm-decode-c4-dflash",
            "qwen3.5-27b-warm-decode-b1",
        ),
        (
            "qwen3.5-moe-35b-a3b-warm-decode-c4-dflash",
            "qwen3.5-moe-35b-a3b-warm-decode-b1",
        ),
    ]
    for c4_id, b1_id in pairs:
        c4 = get_scenario(c4_id)
        b1 = get_scenario(b1_id)
        assert c4.repo == b1.repo
        assert c4.gate_env_var == b1.gate_env_var
        assert c4.workload.prompts == b1.workload.prompts
        assert c4.workload.max_tokens == b1.workload.max_tokens
        assert c4.workload.max_batch_size == b1.workload.max_batch_size


# --- _check_gates dispatch ------------------------------------------------


def _existing_cache_for(repo: str) -> bool:
    """Helper — tests that depend on a real cache hit skip when the
    target weights are not on disk. The C.4 rows reference the same
    27B / MoE caches the b1 cousins use."""
    flat = repo.replace("/", "--")
    return (
        Path.home() / ".cache" / "huggingface" / "hub" / f"models--{flat}"
    ).exists()


def test_check_gates_dflash_mode_skips_when_drafter_cache_missing() -> None:
    """With ``--speculative dflash`` and a C.4 row, the gate must
    short-circuit on the drafter HF cache before consulting any env
    var. The test uses Qwen3.5-27B-DFlash which is **not** cached
    locally (η's job to pull); the assertion is that the skip reason
    references the drafter cache path."""
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")
    if not _existing_cache_for(sc.repo):
        pytest.skip(f"target {sc.repo} not cached; gate ordering not exercised")
    cfg = sc.spec_config
    assert cfg is not None  # type-narrow
    drafter_cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{cfg.draft_repo.replace('/', '--')}"
    )
    if drafter_cache.exists():
        pytest.skip(
            f"drafter {cfg.draft_repo} unexpectedly cached at "
            f"{drafter_cache}; (η) territory — this test pins the "
            "missing-cache skip reason"
        )
    # Ensure target env is set so we'd reach the drafter check.
    with patch.dict(os.environ, {"SILICA_REAL_QWEN3_5_27B": "1"}):
        reason = _check_gates(sc, speculative_mode="dflash")
    assert reason is not None
    assert reason.startswith("draft_cache_missing:")
    assert "z-lab--Qwen3.5-27B-DFlash" in reason


def test_check_gates_dflash_mode_skips_when_drafter_env_missing(
    tmp_path: Path,
) -> None:
    """When the drafter cache is present but the drafter env var is
    not set to ``"1"``, the gate must surface ``draft_env_var_not_set``
    so CI logs explain the skip."""
    # Build a cache directory and patch hf_cache_path_for_repo so the
    # drafter cache check passes.
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")
    target_cache = tmp_path / "target"
    drafter_cache = tmp_path / "drafter"
    target_cache.mkdir()
    drafter_cache.mkdir()

    spec_cfg = sc.spec_config
    assert spec_cfg is not None  # type-narrow

    def fake_cache(repo: str) -> Path:
        if repo == sc.repo:
            return target_cache
        if repo == spec_cfg.draft_repo:
            return drafter_cache
        raise AssertionError(f"unexpected repo {repo}")

    env_overrides = {"SILICA_REAL_QWEN3_5_27B": "1"}
    if "SILICA_BENCH_DFLASH_27B" in os.environ:
        env_overrides["SILICA_BENCH_DFLASH_27B"] = ""  # unset the gate

    with patch(
        "silica.bench.runner.hf_cache_path_for_repo", side_effect=fake_cache
    ), patch.dict(os.environ, env_overrides, clear=False):
        # Make sure the drafter env is not set to "1" in the patched env.
        os.environ.pop("SILICA_BENCH_DFLASH_27B", None)
        reason = _check_gates(sc, speculative_mode="dflash")
    assert reason == "draft_env_var_not_set:SILICA_BENCH_DFLASH_27B"


def test_check_gates_dflash_row_skips_drafter_checks_under_draft_target_mode(
    tmp_path: Path,
) -> None:
    """A C.4 dflash row run under ``--speculative draft_target``
    falls to spec-off (kind mismatch); drafter cache / env are NOT
    consulted. Pinning so a future runner change doesn't accidentally
    trigger a 4-8 GB drafter HF download under the wrong CLI mode."""
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")
    target_cache = tmp_path / "target"
    target_cache.mkdir()

    def fake_cache(repo: str) -> Path:
        if repo == sc.repo:
            return target_cache
        # If the drafter cache lookup fires, the test fails — the
        # gate should bypass it under non-matching CLI mode.
        raise AssertionError(
            f"drafter cache lookup fired for {repo} under "
            "draft_target mode on a kind=='dflash' row; gate is "
            "incorrectly consulting the drafter"
        )

    with patch(
        "silica.bench.runner.hf_cache_path_for_repo", side_effect=fake_cache
    ), patch.dict(os.environ, {"SILICA_REAL_QWEN3_5_27B": "1"}):
        reason = _check_gates(sc, speculative_mode="draft_target")
    # Target gates pass; no drafter checks → None (runnable spec-off).
    assert reason is None


def test_check_gates_dflash_row_runs_spec_off_under_none_mode(
    tmp_path: Path,
) -> None:
    """Under default ``--speculative none`` the C.4 row runs as plain
    warm-decode against the target only; drafter checks are not
    consulted."""
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")
    target_cache = tmp_path / "target"
    target_cache.mkdir()

    def fake_cache(repo: str) -> Path:
        if repo == sc.repo:
            return target_cache
        raise AssertionError(f"unexpected drafter cache lookup for {repo}")

    with patch(
        "silica.bench.runner.hf_cache_path_for_repo", side_effect=fake_cache
    ), patch.dict(os.environ, {"SILICA_REAL_QWEN3_5_27B": "1"}):
        reason = _check_gates(sc, speculative_mode="none")
    assert reason is None


# --- _default_engine_factory dispatch -------------------------------------


def test_default_engine_factory_constructs_dflash_drafter() -> None:
    """Under ``--speculative dflash`` and a kind=='dflash' scenario,
    the runner constructs a ``DFlashDrafter`` (not
    ``DraftTargetEngine``). Monkey-patches ``DFlashDrafter`` so no
    real checkpoint loads — pinning the dispatch path, not the
    drafter behaviour (that lives in (δ.1) tests)."""
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")

    fake_drafter_class = MagicMock(name="FakeDFlashDrafter")
    fake_drafter_instance = MagicMock(name="fake_drafter_instance")
    fake_drafter_class.return_value = fake_drafter_instance

    fake_kv = MagicMock(name="kv")

    # The HiddenCaptureAdapter Protocol check uses isinstance against
    # a runtime_checkable Protocol. ``spec=HiddenCaptureAdapter`` on
    # the MagicMock guarantees isinstance returns True without
    # building a real adapter subclass.
    from silica.models.hidden_capture import HiddenCaptureAdapter

    fake_adapter: Any = MagicMock(spec=HiddenCaptureAdapter, name="adapter")

    runner = BenchRunner(speculative_mode="dflash")
    with patch(
        "silica.models.factory.adapter_for_repo",
        return_value=(fake_adapter, fake_kv),
    ), patch(
        "silica.speculative.dflash_drafter.DFlashDrafter",
        new=fake_drafter_class,
    ):
        adapter, engine = runner._default_engine_factory(sc)

    # The fake DFlashDrafter constructor was called with the
    # scenario's draft_repo and the loaded adapter.
    fake_drafter_class.assert_called_once()
    kwargs = fake_drafter_class.call_args.kwargs
    assert kwargs["drafter_repo"] == "z-lab/Qwen3.5-27B-DFlash"
    assert kwargs["target_adapter"] is fake_adapter

    # Engine sees the fake drafter as its draft_engine.
    assert engine._draft_engine is fake_drafter_instance
    assert engine._verify_k == 16


def test_default_engine_factory_dflash_row_under_none_mode_runs_spec_off() -> None:
    """``--speculative none`` + a kind=='dflash' scenario yields a
    spec-off engine — DFlashDrafter is not constructed at all (so no
    ``dflash-mlx`` import fires for users on the slim install)."""
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")

    fake_adapter = MagicMock(name="adapter")
    fake_kv = MagicMock(name="kv")

    runner = BenchRunner(speculative_mode="none")
    with patch(
        "silica.models.factory.adapter_for_repo",
        return_value=(fake_adapter, fake_kv),
    ), patch(
        "silica.speculative.dflash_drafter.DFlashDrafter",
        side_effect=AssertionError(
            "DFlashDrafter must not be constructed under --speculative none"
        ),
    ):
        adapter, engine = runner._default_engine_factory(sc)

    # Spec-off → engine's draft_engine is NoopDraftEngine (default).
    from silica.speculative.engine import NoopDraftEngine

    assert isinstance(engine._draft_engine, NoopDraftEngine)


def test_default_engine_factory_dflash_row_under_draft_target_runs_spec_off() -> None:
    """``--speculative draft_target`` + a kind=='dflash' scenario
    falls to spec-off (kind mismatch). Pin so the wrong drafter type
    is never wired up."""
    sc = get_scenario("qwen3.5-27b-warm-decode-c4-dflash")

    fake_adapter = MagicMock(name="adapter")
    fake_kv = MagicMock(name="kv")

    runner = BenchRunner(speculative_mode="draft_target")
    with patch(
        "silica.models.factory.adapter_for_repo",
        return_value=(fake_adapter, fake_kv),
    ), patch(
        "silica.speculative.draft_target.DraftTargetEngine.from_repo",
        side_effect=AssertionError(
            "DraftTargetEngine must not be constructed for a "
            "kind=='dflash' row under --speculative draft_target"
        ),
    ), patch(
        "silica.speculative.dflash_drafter.DFlashDrafter",
        side_effect=AssertionError(
            "DFlashDrafter must not be constructed under "
            "--speculative draft_target"
        ),
    ):
        adapter, engine = runner._default_engine_factory(sc)

    from silica.speculative.engine import NoopDraftEngine

    assert isinstance(engine._draft_engine, NoopDraftEngine)


# --- SpecConfig.kind validation ------------------------------------------


def test_spec_config_kind_default_preserves_pre_zeta_behaviour() -> None:
    """Pre-(ζ) ``SpecConfig`` rows did not pass ``kind``; their default
    must remain ``draft_target`` so the existing C.1 ``-spec-on``
    rows behave unchanged under ``--speculative draft_target``."""
    cfg = SpecConfig(draft_repo="Qwen/Qwen3.5-0.8B")
    assert cfg.kind == "draft_target"


def test_spec_config_kind_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        SpecConfig(
            draft_repo="z-lab/Qwen3.5-27B-DFlash", kind="invalid-kind"
        )


def test_existing_spec_on_rows_kind_is_draft_target() -> None:
    """Pin that the existing (h) rows still resolve to
    kind=='draft_target' — backwards-compat regression guard."""
    for sid in (
        "qwen3.5-27b-warm-decode-spec-on",
        "qwen3.5-moe-35b-a3b-warm-decode-spec-on",
    ):
        sc = get_scenario(sid)
        assert sc.spec_config is not None
        assert sc.spec_config.kind == "draft_target"
