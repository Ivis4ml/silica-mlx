"""silica.bench.scenarios — built-in scenario catalog.

Modules that migrate an existing probe / smoke / parity into a
:class:`Scenario` register it here so ``scripts/bench.py
--scenario <id>`` can find it by name. The catalog is intentionally
a module-level ``dict`` rather than a registry class: scenario
definitions are static constants, and a dict lets readers see the
full roster at a glance.

Current catalog (P-4.1 + P-4.2a + P-4.2b + P-4.2c):

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
  * ``qwen3-0.6b-b1-parity`` (P-4.2b) — reuses the cached 0.6B
    weights from ``qwen3-0.6b-smoke``. Runs the single-request
    reference and a B=1 ``generate_batch`` on the same prompt
    with byte-identical ``SamplingParams``; ``B1_PARITY_VS_SINGLE``
    oracle asserts token-for-token equality. Cache-only gate so
    the scheduler's B=1 correctness rides every dev run.
  * ``qwen3-0.6b-bgt1-parity`` (P-4.2c) — same cached 0.6B
    weights; drives a B=2 ``generate_batch`` on two
    different-length prompts and compares per-row tokens against a
    direct mlx-lm batched forward driven with
    ``adapter.make_batch_cache(left_padding)``. Both sides use
    ``stop_token_ids=()`` so the reference (which runs
    unconditionally for ``max_tokens``) and Silica's stream stay
    length-aligned. Exercises the scheduler's B>1 glue, left
    padding, and row lifecycle on any dev box with the cached 0.6B.

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


_QWEN3_0_6B_B1_PARITY = Scenario(
    id="qwen3-0.6b-b1-parity",
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
    oracle=OracleKind.B1_PARITY_VS_SINGLE,
    gate_env_var=None,
    description=(
        "B=1 parity regression: for the same prompt, the B=1 batched "
        "path through Engine.generate_batch(..., max_batch_size=1) "
        "must emit the same token stream as the single-request "
        "Engine.generate path. Runner drives both executions with the "
        "shared _build_sampling_params helper so divergence cannot be "
        "blamed on drifted params. Reuses the cached Qwen/Qwen3-0.6B "
        "weights so the scheduler B=1 claim rides every dev run."
    ),
)


_QWEN3_0_6B_BGT1_PARITY = Scenario(
    id="qwen3-0.6b-bgt1-parity",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="bgt1-different-length-prompts",
        # Prompt lengths chosen so the tokenized lengths diverge on
        # the Qwen3 tokenizer — "Hello" is 1 token, "The capital of
        # Japan is" is 5 tokens, so make_batch_cache runs with
        # left_padding=[4, 0]. An earlier version used two "The
        # capital of X is" prompts where X was a single-token
        # country name; both tokenized to 5 tokens and left padding
        # was [0, 0], silently bypassing the branch this scenario
        # claims to exercise.
        prompts=(
            "Hello",
            "The capital of Japan is",
        ),
        max_tokens=8,
        max_batch_size=2,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
    ),
    oracle=OracleKind.BGT1_DIRECT_BATCHED_REFERENCE,
    gate_env_var=None,
    description=(
        "B>1 scheduler glue regression: Silica's generate_batch at "
        "max_batch_size=2 on two prompts whose tokenized lengths "
        "differ must emit the same per-row tokens as a direct mlx-lm "
        "batched forward driven with "
        "adapter.make_batch_cache(left_padding). Runner overrides "
        "stop_token_ids=() on both sides so the reference (which "
        "runs unconditionally for max_tokens) and Silica's stream "
        "stay length-aligned regardless of EOS. Prompts 'Hello' (1 "
        "token) and 'The capital of Japan is' (5 tokens) force "
        "left_padding=[4, 0] on the Qwen3 tokenizer, exercising the "
        "non-trivial left-pad branch of make_batch_cache. Cache-only "
        "gate."
    ),
)


BUILTIN_SCENARIOS: dict[str, Scenario] = {
    _QWEN3_0_6B_SMOKE.id: _QWEN3_0_6B_SMOKE,
    _QWEN3_0_6B_B1_PARITY.id: _QWEN3_0_6B_B1_PARITY,
    _QWEN3_0_6B_BGT1_PARITY.id: _QWEN3_0_6B_BGT1_PARITY,
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
