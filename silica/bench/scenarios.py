"""silica.bench.scenarios — P-4.1 built-in scenario catalog.

Modules that migrate an existing probe / smoke / parity into a
:class:`Scenario` register it here so ``scripts/bench.py
--scenario <id>`` can find it by name. The catalog is intentionally
a module-level ``dict`` rather than a registry class: scenario
definitions are static constants, and a dict lets readers see the
full roster at a glance.

P-4.1 seeds the catalog with exactly one row — the cheapest cached
smoke that still exercises every branch of ``BenchRunner``:

  * ``qwen3-0.6b-smoke`` — short-in / short-out on Qwen3-0.6B,
    ``max_tokens=4``, ``OracleKind.SMOKE``, cache-only gate (no
    env-var opt-in). The 0.6B-4bit checkpoint is already pulled by
    the P-2 batched parity tests, so anyone running the suite on a
    dev box can exercise the bench runner without a fresh 20 GB
    download.

P-4.2 adds the model-shaped migrations (D3.1 / E3 / 27B / 31B) and
the first workload-shaped rows (short-in/long-out, concurrent
shared-prefix, TTFT-under-concurrency). Each of those lands as a
new entry in this dict; runner code does not change.

The catalog stores repo strings, not pre-loaded adapters, so
instantiating scenarios does not touch HuggingFace or MLX. Adapter
loading happens only inside :class:`BenchRunner` on a scenario that
passes its gates.
"""

from __future__ import annotations

from silica.bench.scenario import OracleKind, Scenario, Workload

_QWEN3_0_6B_SMOKE = Scenario(
    id="qwen3-0.6b-smoke",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="short-in-short-out",
        prompts=("Hello",),
        max_tokens=4,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
    ),
    oracle=OracleKind.SMOKE,
    gate_env_var=None,
    description=(
        "Cheapest cached smoke: single-request Qwen3-0.6B, 4 decoded "
        "tokens from prompt 'Hello'. Cache-only gate so it runs on "
        "any dev box that has pulled the P-2 parity test weights "
        "(see tests/test_p2_batched_parity.py). Exercises the full "
        "runner path (load, generate, oracle, metrics, JSONL) "
        "without claiming correctness beyond 'did not crash and "
        "emitted valid token ids'."
    ),
)


BUILTIN_SCENARIOS: dict[str, Scenario] = {
    _QWEN3_0_6B_SMOKE.id: _QWEN3_0_6B_SMOKE,
}


def get_scenario(scenario_id: str) -> Scenario:
    """Look up a scenario by its id.

    Raises ``KeyError`` with the known-id list in the message so
    mistyped CLI args surface a useful error rather than a bare
    ``KeyError``.
    """
    try:
        return BUILTIN_SCENARIOS[scenario_id]
    except KeyError as exc:
        known = ", ".join(sorted(BUILTIN_SCENARIOS))
        raise KeyError(
            f"unknown scenario id {scenario_id!r}; known: {known}"
        ) from exc


def list_scenario_ids() -> list[str]:
    """Return scenario ids in stable order for CLI listing."""
    return sorted(BUILTIN_SCENARIOS)


__all__ = ["BUILTIN_SCENARIOS", "get_scenario", "list_scenario_ids"]
