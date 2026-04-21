"""silica.bench.scenarios — built-in scenario catalog.

Modules that migrate an existing probe / smoke / parity into a
:class:`Scenario` register it here so ``scripts/bench.py
--scenario <id>`` can find it by name. The catalog is intentionally
a module-level ``dict`` rather than a registry class: scenario
definitions are static constants, and a dict lets readers see the
full roster at a glance.

Current catalog (P-4.1 + P-4.2a + P-4.2b + P-4.2c + P-4.2d-ii + P-4.2d-iii-a):

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
  * ``qwen3.5-0.8b-b1-parity`` (P-4.2d-ii) — cache-only; repo
    ``Qwen/Qwen3.5-0.8B``. Extends the cache-only B=1 parity claim
    to the hybrid DeltaNet / GLOBAL family so the bench keeps up
    with the P-3-C3c/d scheduler. Runner code is unchanged — the
    B1 oracle handles the hybrid path the same way.
  * ``qwen3.5-27b-smoke`` (P-4.2d-ii) — dual-gated on
    ``SILICA_REAL_QWEN3_5_27B`` (new env var; mirrors the
    ``SILICA_REAL_<FAMILY>`` convention used for the other big
    rows). No pytest-side equivalent existed — 27B had only the
    ``scripts/probe_qwen3_5_27b_load.py`` metadata probe before —
    so this row is the bench catalog's first end-to-end health
    check on the 27B checkpoint.
  * ``gemma4-31b-smoke`` (P-4.2d-ii) — dual-gated on
    ``SILICA_REAL_GEMMA4_31B``. Mirrors the existing
    ``tests/test_p3_gemma4_single_request_smoke.py`` end-to-end
    claim; pytest side stays in place because it also asserts
    adapter contract details beyond the bench SMOKE oracle.
  * ``gemma4-31b-b1-parity`` (P-4.2d-ii) — same dual gate;
    mirrors the B=1 half of
    ``tests/test_p3_gemma4_batched_parity.py``. Pins the sliding +
    global hybrid scheduler's B=1 claim on dev boxes with the
    18 GB checkpoint opted in.
  * ``gemma4-31b-bgt1-parity`` (P-4.2d-ii) — same dual gate;
    mirrors the B>1 direct-reference half of
    ``tests/test_p3_gemma4_batched_parity.py``. Uses the same
    different-tokenized-length prompt strategy as the 0.6B row,
    with a parametrized catalog test that re-asserts the
    ``left_padding`` invariant on every BGT1 scenario whose
    weights are on-device.

Workload-shaped rows (PLAN §P-4 names them by shape, not model):

  * ``qwen3-0.6b-short-in-long-out`` (P-4.2d-iii-a) — decode-
    dominated throughput shape. Single prompt "Hello" (1 token
    on Qwen3) and ``max_tokens=64``, so ~64 decode steps vs a
    one-token prefill. SMOKE oracle — the scenario's value is in
    the metrics it collects (decode tok/s on the reference
    cached 0.6B checkpoint), not in strict output validation.
  * ``qwen3-0.6b-long-in-short-out`` (P-4.2d-iii-a) — prefill-
    dominated throughput shape. Long synthetic prompt ("The quick
    brown fox …" repeated 30×, ~301 tokens on Qwen3) and
    ``max_tokens=4``, so prefill dominates the wall time. SMOKE
    oracle; the metrics signal how prefill tok/s scales.

P-4.2d-iii-b / -iii-c will land the concurrent-shared-prefix and
TTFT-under-concurrency rows, which require a runner extension to
drive SMOKE through ``Engine.generate_batch`` (multi-prompt B>1)
and to wire ``RadixPrefixCache``. Until then SMOKE only runs
single-request.

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


_QWEN3_5_0_8B_B1_PARITY = Scenario(
    id="qwen3.5-0.8b-b1-parity",
    repo="Qwen/Qwen3.5-0.8B",
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
        "Cache-only B=1 parity row for the Qwen3.5 hybrid "
        "DeltaNet / GLOBAL family (0.8B checkpoint). Mirrors the "
        "cache-only spirit of qwen3-0.6b-b1-parity but on the "
        "hybrid scheduler path wired in P-3-C3c/d. Pytest side "
        "still holds the stronger B>1 batched-vs-single-request "
        "parity claim (tests/test_p3_hybrid_batched_parity.py); "
        "this bench row covers the B=1 slice so regressions on "
        "the hybrid scheduler surface in the unified harness."
    ),
)


_QWEN3_5_27B_SMOKE = Scenario(
    id="qwen3.5-27b-smoke",
    repo="mlx-community/Qwen3.5-27B-4bit",
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
    gate_env_var="SILICA_REAL_QWEN3_5_27B",
    description=(
        "First end-to-end bench smoke for Qwen3.5-27B-4bit. Before "
        "P-4.2d-ii the 27B checkpoint only had the metadata probe "
        "scripts/probe_qwen3_5_27b_load.py — no pytest-side "
        "functional test exercised the forward path. This bench "
        "row fills the gap with a four-token greedy generation on "
        "prompt 'Hello'. Dual-gated: the 4-bit checkpoint is "
        "~16 GB on disk and peak device memory during the forward "
        "is ~30 GB on M5 Pro 48 GB, so opt-in via "
        "SILICA_REAL_QWEN3_5_27B=1 is mandatory."
    ),
)


_GEMMA4_31B_SMOKE = Scenario(
    id="gemma4-31b-smoke",
    repo="mlx-community/gemma-4-31b-4bit",
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
    gate_env_var="SILICA_REAL_GEMMA4_31B",
    description=(
        "Dense Gemma4-31B-4bit single-request smoke. Mirrors the "
        "end-to-end claim of tests/test_p3_gemma4_single_request_smoke.py "
        "('model loads + forward runs on the real weights'). "
        "Pytest side additionally pins adapter capability flags + "
        "KV layout details that the bench SMOKE oracle does not "
        "check. Dual-gated: the 4-bit checkpoint is ~18 GB on disk "
        "and the 50 sliding + 10 full layer forward is heavier than "
        "Qwen3.5-27B per token; opt-in via SILICA_REAL_GEMMA4_31B=1."
    ),
)


_GEMMA4_31B_B1_PARITY = Scenario(
    id="gemma4-31b-b1-parity",
    repo="mlx-community/gemma-4-31b-4bit",
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
    gate_env_var="SILICA_REAL_GEMMA4_31B",
    description=(
        "Gemma4-31B-4bit B=1 parity. Mirrors the B=1 half of "
        "tests/test_p3_gemma4_batched_parity.py: the B=1 batched "
        "path must emit the same token stream as the single-"
        "request Engine.generate path. The sliding-window "
        "BatchRotatingKVCache + full-attention BatchKVCache "
        "hybrid cache list produced by "
        "Gemma4Adapter.make_batch_cache is the specific thing "
        "being exercised at B=1. Dual-gated on "
        "SILICA_REAL_GEMMA4_31B because every parity run pays the "
        "cost of both a single-request and a batched forward."
    ),
)


_GEMMA4_31B_BGT1_PARITY = Scenario(
    id="gemma4-31b-bgt1-parity",
    repo="mlx-community/gemma-4-31b-4bit",
    workload=Workload(
        name="bgt1-different-length-prompts",
        # Same strategy as qwen3-0.6b-bgt1-parity: pick prompts
        # whose tokenized lengths differ on the target tokenizer
        # so left_padding exercises the non-trivial branch of
        # Gemma4Adapter.make_batch_cache. "Hello" vs "The capital
        # of Japan is" tokenize to different lengths on the
        # Gemma4 tokenizer as well — the catalog's parametrized
        # on-device invariant test verifies this at runtime gated
        # on cache presence.
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
    gate_env_var="SILICA_REAL_GEMMA4_31B",
    description=(
        "Gemma4-31B-4bit B>1 parity against a direct mlx-lm "
        "batched reference. Mirrors the B>1 half of "
        "tests/test_p3_gemma4_batched_parity.py — the exact-vs-"
        "single-request drift observed there (first mismatch at "
        "token 2 on fp16 SDPA 4-bit) is precisely why the oracle "
        "compares against the direct reference instead of against "
        "Engine.generate. Runner overrides stop_token_ids=() on "
        "both sides so the reference (max_tokens unconditional) "
        "and Silica's batched stream stay length-aligned. "
        "Different-tokenized-length prompts exercise the non-"
        "trivial left_padding branch of make_batch_cache. Dual-"
        "gated on SILICA_REAL_GEMMA4_31B."
    ),
)


# ---- Workload-shaped rows (P-4.2d-iii-a) ----------------------------
#
# PLAN §P-4 names four workload shapes: short-in/long-out,
# long-in/short-out, concurrent shared-prefix, TTFT-under-concurrency.
# The first two are single-request (B=1) and need no runner changes —
# they just exercise different prompt-length / max_tokens ratios
# against the existing SMOKE oracle. The other two require B>1
# SMOKE dispatch and land under P-4.2d-iii-b / -iii-c.
#
# Both rows below use the cached Qwen/Qwen3-0.6B weights so they ride
# every dev run without the cost of a dual-gated download.

# Thirty-sentence repeat deliberately constructed so the long prompt
# has enough structure that the tokenizer does not collapse it into
# a degenerate repeated-id run. Empirically 30 copies of this
# sentence tokenize to 301 tokens on the Qwen3 tokenizer — long
# enough for prefill to clearly dominate a max_tokens=4 decode.
_LONG_IN_PROMPT = "The quick brown fox jumps over the lazy dog. " * 30


_QWEN3_0_6B_SHORT_IN_LONG_OUT = Scenario(
    id="qwen3-0.6b-short-in-long-out",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="short-in-long-out",
        prompts=("Hello",),
        max_tokens=64,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
    ),
    oracle=OracleKind.SMOKE,
    gate_env_var=None,
    description=(
        "Decode-dominated throughput shape. One-token prompt "
        "'Hello' + max_tokens=64 means decode (64 steps) vastly "
        "exceeds prefill (1 token). SMOKE oracle — the point of "
        "the row is the decode_tok_s metric, not output "
        "correctness. Cache-only gate (reuses the qwen3-0.6b-smoke "
        "weights) so it rides every dev run."
    ),
)


_QWEN3_0_6B_LONG_IN_SHORT_OUT = Scenario(
    id="qwen3-0.6b-long-in-short-out",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="long-in-short-out",
        prompts=(_LONG_IN_PROMPT,),
        max_tokens=4,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
    ),
    oracle=OracleKind.SMOKE,
    gate_env_var=None,
    description=(
        "Prefill-dominated throughput shape. Long synthetic prompt "
        "('The quick brown fox jumps over the lazy dog. ' x 30 = "
        "~301 tokens on Qwen3) + max_tokens=4 means prefill "
        "dominates wall time. SMOKE oracle — the value is in the "
        "prefill_tok_s metric. Cache-only gate."
    ),
)


BUILTIN_SCENARIOS: dict[str, Scenario] = {
    _QWEN3_0_6B_SMOKE.id: _QWEN3_0_6B_SMOKE,
    _QWEN3_0_6B_B1_PARITY.id: _QWEN3_0_6B_B1_PARITY,
    _QWEN3_0_6B_BGT1_PARITY.id: _QWEN3_0_6B_BGT1_PARITY,
    _QWEN3_0_6B_SHORT_IN_LONG_OUT.id: _QWEN3_0_6B_SHORT_IN_LONG_OUT,
    _QWEN3_0_6B_LONG_IN_SHORT_OUT.id: _QWEN3_0_6B_LONG_IN_SHORT_OUT,
    _QWEN3_5_0_8B_B1_PARITY.id: _QWEN3_5_0_8B_B1_PARITY,
    _QWEN3_5_27B_SMOKE.id: _QWEN3_5_27B_SMOKE,
    _QWEN3_5_MOE_SMOKE.id: _QWEN3_5_MOE_SMOKE,
    _GEMMA4_31B_SMOKE.id: _GEMMA4_31B_SMOKE,
    _GEMMA4_31B_B1_PARITY.id: _GEMMA4_31B_B1_PARITY,
    _GEMMA4_31B_BGT1_PARITY.id: _GEMMA4_31B_BGT1_PARITY,
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
