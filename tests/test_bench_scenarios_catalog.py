"""Catalog-shape tests for ``silica.bench.BUILTIN_SCENARIOS``.

Split in two layers:

  * **Parametrized shape invariants** — iterate over every catalog
    entry and check the universal rules (non-empty id, repo shaped
    like ``owner/name``, ``scenario.id`` matches the dict key,
    ``oracle`` is an :class:`OracleKind`, ``gate_env_var`` is
    ``None | str``, workload constraints consistent with a
    single-request SMOKE row). Adding a fourth catalog entry in
    P-4.2b/c should not require touching this file — the new row
    just rides the existing parametrize sweep.

  * **Per-scenario lock-ins** — one deliberate test per row pins
    the fields that drift would silently break downstream
    (repo string, env-var name). These are load-bearing because
    ``get_scenario("qwen3.5-moe-smoke")`` is how the CLI looks up
    the scenario, and because the env-var names are contractual
    with the pytest-side real-model smokes (same env var gates
    both views of the same checkpoint).

Not tested here:

  * ``BenchRunner`` dispatch over catalog scenarios — already
    covered by ``tests/test_bench_runner.py`` with fakes.
  * CLI ``--list`` output — already covered by
    ``tests/test_bench_cli.py``.
"""

from __future__ import annotations

import pytest

from silica.bench import (
    BUILTIN_SCENARIOS,
    OracleKind,
    Scenario,
    Workload,
    get_scenario,
    list_scenario_ids,
)

# ---------- parametrized shape invariants ------------------------------


@pytest.mark.parametrize(
    "scenario_id,scenario",
    sorted(BUILTIN_SCENARIOS.items()),
    ids=lambda v: v if isinstance(v, str) else getattr(v, "id", repr(v)),
)
def test_catalog_entry_shape_invariants(
    scenario_id: str, scenario: Scenario
) -> None:
    """Every catalog entry obeys the universal scenario contract."""
    assert scenario_id == scenario.id, (
        f"catalog dict key {scenario_id!r} does not match "
        f"scenario.id {scenario.id!r} — one source of truth only"
    )
    assert scenario.id, "scenario id must be non-empty"
    assert isinstance(scenario.repo, str) and "/" in scenario.repo, (
        f"{scenario.id}: repo must be 'owner/name' shaped, got "
        f"{scenario.repo!r}"
    )
    assert isinstance(scenario.oracle, OracleKind)
    assert scenario.gate_env_var is None or isinstance(
        scenario.gate_env_var, str
    )
    assert isinstance(scenario.workload, Workload)
    wl = scenario.workload
    assert wl.max_tokens >= 1
    assert wl.max_batch_size >= 1
    assert len(wl.prompts) >= 1
    # If the row is single-request, the workload must carry exactly
    # one prompt — runner rejects len(prompts) != 1 for max_batch=1.
    if wl.max_batch_size == 1:
        assert len(wl.prompts) == 1, (
            f"{scenario.id}: single-request row must have exactly "
            f"one prompt, got {len(wl.prompts)}"
        )


def test_get_scenario_round_trips_every_entry() -> None:
    """``get_scenario`` returns the same object instance the dict holds."""
    for scenario_id, scenario in BUILTIN_SCENARIOS.items():
        assert get_scenario(scenario_id) is scenario


def test_list_scenario_ids_is_sorted_and_complete() -> None:
    ids = list_scenario_ids()
    assert ids == sorted(ids), "ids must come back sorted for stable CLI --list"
    assert set(ids) == set(BUILTIN_SCENARIOS), "list must cover the dict"


# ---------- per-scenario lock-ins --------------------------------------


def test_qwen3_0_6b_smoke_is_cache_only() -> None:
    """The P-4.1 seed row is cache-only (runs on any dev box with the
    P-2 parity test weights)."""
    scenario = get_scenario("qwen3-0.6b-smoke")
    assert scenario.repo == "Qwen/Qwen3-0.6B"
    assert scenario.gate_env_var is None
    assert scenario.oracle == OracleKind.SMOKE
    assert scenario.workload.max_tokens == 4
    assert scenario.workload.max_batch_size == 1


def test_qwen3_0_6b_b1_parity_is_cache_only() -> None:
    """The P-4.2b parity row reuses the cached 0.6B weights — no
    env-var gate — so B=1 scheduler correctness is exercised on any
    dev run that exercises the smoke row."""
    scenario = get_scenario("qwen3-0.6b-b1-parity")
    assert scenario.repo == "Qwen/Qwen3-0.6B"
    assert scenario.gate_env_var is None
    assert scenario.oracle == OracleKind.B1_PARITY_VS_SINGLE
    assert scenario.workload.max_tokens == 4
    assert scenario.workload.max_batch_size == 1
    assert scenario.workload.prompts == ("Hello",)
    # Same SamplingParams shape as the smoke row is intentional;
    # pin it so a drift on the smoke row does not silently widen
    # the parity claim (parity compares the specific params, not
    # an abstract "same model behaves the same").
    smoke = get_scenario("qwen3-0.6b-smoke")
    assert scenario.workload.temperature == smoke.workload.temperature
    assert scenario.workload.top_p == smoke.workload.top_p


def test_qwen3_5_moe_smoke_is_dual_gated() -> None:
    """Env var name must match the pytest-side
    tests/test_p3_qwen3_5_moe_smoke.py gate so the two views of the
    same checkpoint opt in together."""
    scenario = get_scenario("qwen3.5-moe-smoke")
    assert scenario.repo == "mlx-community/Qwen3.5-35B-A3B-4bit"
    assert scenario.gate_env_var == "SILICA_REAL_QWEN3_5_MOE"
    assert scenario.oracle == OracleKind.SMOKE
    assert scenario.workload.max_tokens == 4
    assert scenario.workload.max_batch_size == 1
    assert scenario.workload.prompts == ("Hello",)


def test_gemma4_moe_smoke_is_dual_gated() -> None:
    """Env var name must match the pytest-side
    tests/test_p3_gemma4_moe_smoke.py gate."""
    scenario = get_scenario("gemma4-moe-smoke")
    assert scenario.repo == "mlx-community/gemma-4-26b-a4b-4bit"
    assert scenario.gate_env_var == "SILICA_REAL_GEMMA4_MOE"
    assert scenario.oracle == OracleKind.SMOKE
    assert scenario.workload.max_tokens == 4
    assert scenario.workload.max_batch_size == 1
    assert scenario.workload.prompts == ("Hello",)


def test_moe_scenarios_share_workload_shape_but_not_object() -> None:
    """Advisor rejected DRY-ing the Workload constant across rows;
    pin that decision so a future refactor does not quietly share
    one Workload instance (which would prevent per-row tuning of
    max_tokens / prompts in P-4.3 without affecting siblings)."""
    q = get_scenario("qwen3.5-moe-smoke").workload
    g = get_scenario("gemma4-moe-smoke").workload
    # Same shape (the Workload dataclass is frozen, so equality is
    # by field value — two instances with identical fields compare
    # equal, but they remain distinct objects).
    assert q == g
    assert q is not g
