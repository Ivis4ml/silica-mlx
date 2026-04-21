"""silica.bench.scenarios — built-in scenario catalog.

Modules that migrate an existing probe / smoke / parity into a
:class:`Scenario` register it here so ``scripts/bench.py
--scenario <id>`` can find it by name. The catalog is intentionally
a module-level ``dict`` rather than a registry class: scenario
definitions are static constants, and a dict lets readers see the
full roster at a glance.

Current catalog (P-4.1 + P-4.2a):

  * ``qwen3-0.6b-smoke`` (P-4.1) — short-in / short-out on
    Qwen3-0.6B, ``max_tokens=4``, ``OracleKind.SMOKE``, cache-only
    gate (no env-var opt-in). The 0.6B checkpoint is already pulled
    by the P-2 batched parity tests so anyone running the suite on
    a dev box can exercise the bench runner without a fresh
    download. Cheapest row; used by the CLI smoke test in
    ``tests/test_bench_cli.py``.
  * ``qwen3.5-moe-smoke`` (P-4.2a) — mirrors
    ``tests/test_p3_qwen3_5_moe_smoke.py``. Qwen3.5-35B-A3B-4bit
    MoE single-request on "Hello" with ``max_tokens=4``, SMOKE
    oracle, dual-gated (cache + ``SILICA_REAL_QWEN3_5_MOE=1``).
    ~20 GB on disk, ~30+ GB device memory for the forward.
  * ``gemma4-moe-smoke`` (P-4.2a) — mirrors
    ``tests/test_p3_gemma4_moe_smoke.py``. gemma-4-26b-a4b-4bit MoE
    single-request on "Hello" with ``max_tokens=4``, SMOKE oracle,
    dual-gated (cache + ``SILICA_REAL_GEMMA4_MOE=1``). ~16 GB on
    disk; always-on dense MLP + SwitchGLU experts forward is more
    expensive per token than dense Gemma4-31B.

The pytest-side real-model smokes remain in place: they pin the
adapter shape (``config.extra`` values, capability flags), which
the SMOKE oracle here does not check. The two views are
complementary — pytest guards the adapter contract, bench guards
the end-to-end harness on the same weights.

P-4.2b adds the B=1 parity oracle (``B1_PARITY_VS_SINGLE``); P-4.2c
adds the B>1 direct mlx-lm batched reference oracle
(``BGT1_DIRECT_BATCHED_REFERENCE``). Neither changes the catalog's
schema — they extend via :mod:`silica.bench.oracles`.

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


_QWEN3_5_MOE_SMOKE = Scenario(
    id="qwen3.5-moe-smoke",
    repo="mlx-community/Qwen3.5-35B-A3B-4bit",
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
    gate_env_var="SILICA_REAL_QWEN3_5_MOE",
    description=(
        "Qwen3.5-35B-A3B-4bit MoE single-request smoke. Mirrors the "
        "pytest-side tests/test_p3_qwen3_5_moe_smoke.py end-to-end "
        "claim ('does not crash on the real MoE weights') and adds "
        "the bench metrics + JSONL row. Dual-gated: the 4-bit "
        "checkpoint is ~20 GB on disk and the forward peaks at "
        "~30+ GB device memory, so a cache-only gate would risk "
        "surprise activations on dev boxes with other 0.6B / 4B "
        "weights already cached. No token parity claim — mlx-lm "
        "silently drops attn_output_gate on this family (E-open-5), "
        "so a hypothetical HF-parity attempt would fail on "
        "attention before reaching MoE."
    ),
)


_GEMMA4_MOE_SMOKE = Scenario(
    id="gemma4-moe-smoke",
    repo="mlx-community/gemma-4-26b-a4b-4bit",
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
    gate_env_var="SILICA_REAL_GEMMA4_MOE",
    description=(
        "gemma-4-26b-a4b-4bit MoE single-request smoke. Mirrors the "
        "pytest-side tests/test_p3_gemma4_moe_smoke.py ('does not "
        "crash through the always-on dense MLP + experts additive "
        "forward path on the real weights'). Dual-gated: the 4-bit "
        "checkpoint is ~16 GB on disk and the always-on dense MLP "
        "summed with the SwitchGLU experts branch makes per-token "
        "forward more expensive than dense Gemma4-31B."
    ),
)


BUILTIN_SCENARIOS: dict[str, Scenario] = {
    _QWEN3_0_6B_SMOKE.id: _QWEN3_0_6B_SMOKE,
    _QWEN3_5_MOE_SMOKE.id: _QWEN3_5_MOE_SMOKE,
    _GEMMA4_MOE_SMOKE.id: _GEMMA4_MOE_SMOKE,
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
