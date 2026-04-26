"""silica.bench.scenarios — built-in scenario catalog.

Modules that migrate an existing probe / smoke / parity into a
:class:`Scenario` register it here so ``scripts/bench.py
--scenario <id>`` can find it by name. The catalog is intentionally
a module-level ``dict`` rather than a registry class: scenario
definitions are static constants, and a dict lets readers see the
full roster at a glance.

Current catalog (P-4.1 + P-4.2a + P-4.2b + P-4.2c + P-4.2d-ii + P-4.2d-iii-a + P-4.2d-iii-b + P-4.3):

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
  * ``qwen3-0.6b-concurrent-shared-prefix`` (P-4.2d-iii-b) —
    four prompts "The capital of {France, Germany, Italy, Spain}
    is" share a three-token prefix ``"The capital of "``. Runs at
    ``max_batch_size=4`` with ``prefix_cache=True``, so the
    scheduler's radix prefix cache reuses the shared block across
    rows. SMOKE oracle validates every row emits valid tokens.
  * ``qwen3-0.6b-ttft-under-concurrency`` (P-4.2d-iii-b) — one
    long (~301-token) prompt admitted alongside three short
    prompts ("A", "B", "C") at ``max_batch_size=4``,
    ``prefix_cache=False``, ``max_tokens=4``. Exercises the
    scheduler under the Q-010 shape PLAN §P-4 specifies. With
    Silica's current non-chunked prefill the short prompts' TTFT
    will be dominated by the long prompt's prefill; the metrics
    this row collects are the input to the Q-010 chunked-prefill
    promotion decision, not a correctness oracle (SMOKE remains
    the floor: all rows must emit valid tokens).
  * ``qwen3-0.6b-teacher-forced-argmax`` (P-4.3) — PLAN §P-3
    exit-criterion row. For a fixed prompt + target continuation,
    runs the silica adapter's ``prefill`` + ``decode_step`` loop
    with teacher-forced targets and compares the positional
    argmax sequence against a direct mlx-lm single-forward
    reference. Oracle config supplies ``target_continuation``
    (text tokenised by the adapter's tokenizer) and
    ``min_agreement_rate=0.98`` per PLAN; in practice both paths
    agree 100% on cached 0.6B because they drive the same
    mlx-lm model from identical state.

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

from pathlib import Path

from silica.bench.scenario import (
    OracleKind,
    Scenario,
    VqbenchXcheckSpec,
    Workload,
)
from silica.bench.vqbench_baseline import default_reproduce_script_path

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


# P-5-A.3c prefix-hit decode-speed bench rows. Both scenarios share
# the ``_PREFIX_HIT_DECODE_SHARED_PROMPT`` (long enough that Qwen3-0.6B
# tokenization produces ≥ 2 aligned radix blocks on block_size=16;
# identical prompts drive row 1 through the full-match hit path under
# max_batch_size=1). The two rows differ only in ``kv_codec``:
# ``"fp16"`` is the IdentityCodec baseline; the codec row installs the
# opening-doc §5.2 production recommendation (BlockTurboQuantMSE B=64
# 4-bit). The A.3c acceptance gate compares their ``decode_tok_s`` to
# assert BlockTQ ≥ 0.85× identity.
#
# **Model choice (2026-04-22 amendment):** opening §7(d) / §8 P-5-A.3
# named ``qwen3.5-0.8b-prefix-hit-decode`` as the row id. That target
# cannot run — Qwen3.5-0.8B is hybrid-DeltaNet (``has_recurrent_state
# =True``), and ``ContinuousBatcher`` refuses ``RadixPrefixCache`` on
# models with recurrent state because DeltaNet's running accumulation
# cannot be sliced into block-aligned prefix K/V (see
# ``docs/P3_DELTANET_SURVEY.md`` C-open-3). Qwen3-0.6B is the
# immediate alternative: plain dense GLOBAL attention, 28 layers,
# head_dim=128, no recurrent state. The BlockTQ-vs-identity ratio
# the §7(d) gate asserts is head-dim / layer-count independent in
# theory (both codecs pay the same per-block per-layer traversal
# cost), so the 0.85× threshold carries from 0.8B to 0.6B.
_PREFIX_HIT_DECODE_SHARED_PROMPT = (
    "The continuous-batching scheduler allocates fresh KV blocks at "
    "admission time, retains source references through the radix "
    "tree, and releases detached storage only once hit counts drop "
    "to zero. Prefix-cache hits on mid-run admission seed the "
    "BatchKVCache from detached per-layer payloads without re-running "
    "prefill, which is the exact path a compressed codec's "
    "decode_tensor exercises on every admitted row 1."
)


_QWEN3_0_6B_PREFIX_HIT_DECODE_FP16 = Scenario(
    id="qwen3-0.6b-prefix-hit-decode-fp16",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="prefix-hit-decode-fp16",
        prompts=(
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
        ),
        max_tokens=16,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="fp16",
    ),
    oracle=OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT,
    gate_env_var=None,
    description=(
        "P-5-A.3c baseline row. Two identical prompts at "
        "max_batch_size=1 prefix_cache=True: row 0 admits into the "
        "initial cohort and runs miss-path prefill + decode; row 1 "
        "enters the waiting queue and is admitted mid-run through "
        "_admit_single_hit_row. Under kv_codec='fp16' the store "
        "installs IdentityCodec, so row 1's seeded-admission + "
        "decode loop reconstructs fp16 K/V by reference — no "
        "compression, honest baseline. The oracle reports row 1's "
        "steady-state decode tok/s (inter-token interval, excluding "
        "first-token latency). Cache-only gate; the A.3c acceptance "
        "test compares this row's decode_tok_s against the paired "
        "block_tq_b64_b4 row to gate BlockTQ ≥ 0.85× identity. "
        "Opening §7(d) referred to Qwen3.5-0.8B as the target; that "
        "checkpoint is hybrid-DeltaNet and cannot host "
        "RadixPrefixCache — amended to Qwen3-0.6B (plain GLOBAL, "
        "no recurrent state)."
    ),
)


_QWEN3_0_6B_PREFIX_HIT_DECODE_BLOCK_TQ_B64_B4 = Scenario(
    id="qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="prefix-hit-decode-block-tq-b64-b4",
        prompts=(
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
        ),
        max_tokens=16,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT,
    gate_env_var=None,
    description=(
        "P-5-A.3c compression row. Identical workload shape to "
        "qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec="
        "'block_tq_b64_b4' — BlockTurboQuantMSE vq_block_size=64 "
        "num_bits=4, the opening §5.2 / vqbench REPORT §3.1 "
        "production recommendation (strictly lossless at std=0% "
        "across three seeds, 3.76× total-KV compression). Row 1's "
        "seeded-admission path exercises "
        "``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × "
        "num_layers × num_hit_blocks — the codec overhead this "
        "scenario measures. Cache-only gate."
    ),
)


_QWEN3_0_6B_PREFIX_HIT_DECODE_EXT_RABITQ_B4 = Scenario(
    id="qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="prefix-hit-decode-ext-rabitq-b4",
        prompts=(
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
        ),
        max_tokens=16,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b4",
    ),
    oracle=OracleKind.DECODE_TOK_S_WITH_PREFIX_HIT,
    gate_env_var=None,
    description=(
        "P-5-B.3 compression row — ExtRaBitQ arm of the prefix-hit "
        "decode-speed acceptance gate. Identical workload shape to "
        "qwen3-0.6b-prefix-hit-decode-fp16 but with kv_codec="
        "'ext_rabitq_b4' — ExtRaBitQ num_bits=4 (symmetric K+V, "
        "effective bits/coord = 4 + 48/head_dim = 4.375 at "
        "head_dim=128). Row 1's seeded-admission path exercises "
        "``k_codec.decode_tensor`` + ``v_codec.decode_tensor`` × "
        "num_layers × num_hit_blocks with ExtRaBitQ's integer-grid "
        "codebook lookup + per-vector scale multiply + re-"
        "normalization + inverse rotation; the codec overhead this "
        "scenario measures. The B.3 acceptance test compares this "
        "row's decode_tok_s against the paired fp16 baseline to gate "
        "ExtRaBitQ ≥ 0.85× identity (same threshold as BlockTQ). "
        "rabitq_b1 is deliberately excluded from the gate — it is "
        "K-only (``v_supported=False``) so the symmetric kv_codec= "
        "shorthand cannot install it, and its hypercube MSE is worse "
        "than BlockTQ at matching bit budget. Cache-only gate."
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


_QWEN3_0_6B_CONCURRENT_SHARED_PREFIX = Scenario(
    id="qwen3-0.6b-concurrent-shared-prefix",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="concurrent-shared-prefix",
        prompts=(
            "The capital of France is",
            "The capital of Germany is",
            "The capital of Italy is",
            "The capital of Spain is",
        ),
        max_tokens=8,
        max_batch_size=4,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
    ),
    oracle=OracleKind.SMOKE,
    gate_env_var=None,
    description=(
        "Four prompts share a three-token prefix 'The capital of' "
        "(tokenized as three ids on Qwen3). Runs at "
        "max_batch_size=4 with prefix_cache=True so the radix "
        "prefix cache reuses the shared block across rows. SMOKE "
        "oracle validates every row emits valid tokens; the "
        "JSONL row's per-row metadata makes the reader aware how "
        "many tokens each of the four rows produced. Cache-only "
        "gate."
    ),
)


_QWEN3_0_6B_TEACHER_FORCED_ARGMAX = Scenario(
    id="qwen3-0.6b-teacher-forced-argmax",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="teacher-forced-argmax",
        prompts=("The capital of France is",),
        # max_tokens is unused on the teacher-forced path — the
        # number of predictions is dictated by
        # oracle_config['target_continuation']. Pinned to 1 so the
        # Workload shape remains trivially single-request.
        max_tokens=1,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
    ),
    oracle=OracleKind.TEACHER_FORCED_ARGMAX,
    oracle_config={
        # Target continuation chosen to tokenize to ~40+ ids on
        # Qwen3 — far above PLAN §P-3's "first 100 teacher-forced
        # positions" is not required; the oracle is about
        # silica-vs-reference agreement rate, not total length.
        "target_continuation": (
            " Paris. The capital of Germany is Berlin. The capital "
            "of Italy is Rome. The capital of Spain is Madrid. The "
            "capital of Japan is Tokyo."
        ),
        "min_agreement_rate": 0.98,
    },
    gate_env_var=None,
    description=(
        "Teacher-forced next-token argmax parity (PLAN §P-3 exit "
        "criterion). Silica drives adapter.prefill + "
        "decode_step with teacher-forced target tokens; the "
        "reference is a single mlx-lm forward over prompt + "
        "target[:-1] with positional logits sliced to match. "
        "Oracle passes when silica's per-position argmax matches "
        "the reference at >= min_agreement_rate (0.98 per PLAN; "
        "100% expected on cached 0.6B because both paths drive "
        "the same mlx-lm model). Cache-only gate."
    ),
)


_QWEN3_0_6B_TTFT_UNDER_CONCURRENCY = Scenario(
    id="qwen3-0.6b-ttft-under-concurrency",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="ttft-under-concurrency",
        prompts=(
            _LONG_IN_PROMPT,
            "A",
            "B",
            "C",
        ),
        max_tokens=4,
        max_batch_size=4,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
    ),
    oracle=OracleKind.SMOKE,
    gate_env_var=None,
    description=(
        "One long (~301 tokens on Qwen3) prompt admitted alongside "
        "three one-character prompts at max_batch_size=4, "
        "prefix_cache=False, max_tokens=4. PLAN §P-4 specifies "
        "this shape to resolve Q-010 (chunked-prefill promotion): "
        "with Silica's current non-chunked prefill, the long "
        "prompt's prefill blocks the short prompts, inflating "
        "their TTFT. The metrics this row collects are the input "
        "to the Q-010 decision — the oracle itself is SMOKE "
        "because 'did not crash + all rows emitted valid tokens' "
        "is the correctness floor. Cache-only gate."
    ),
)


# =============================================================================
# P-5-C.2 step 3 — WikiText-2 PPL rows on Qwen3-0.6B.
#
# The four rows share the same tokenized WikiText-2 input (default
# ``~/.cache/silica/wikitext2-test.txt``, populate via
# ``scripts/prepare_wikitext2_cache.py``). The ``fp16`` row is the
# baseline — runs through ``teacher_forced_chunked_nll`` on the
# adapter's own ``BatchKVCache`` (no prefix cache, no codec). The
# three compression rows run through
# ``teacher_forced_chunked_nll_with_codec`` with a fresh
# ``RadixPrefixCache`` whose store carries the named codec; each
# chunk >= 1 decodes the prior prefix blocks (codec hot path fires)
# and encodes the newly-grown tail.
#
# Originally pinned to Qwen3-0.6B because the pre-C5.4
# ``ContinuousBatcher`` refused ``RadixPrefixCache`` on hybrid
# adapters. P-3-C5.4 removed that guard; P-3-C5-step2-W made the
# bench codec oracle hybrid-aware via heterogeneous cache assembly
# and recurrent snapshot/restore at chunk boundaries. Hybrid
# Qwen3.5 mirrors of these PPL rows live below at
# ``_QWEN3_5_0_8B_WIKITEXT_PPL_*`` — same workload knobs, different
# repo, no behaviour change on Qwen3-0.6B.
# =============================================================================

_WIKITEXT2_DEFAULT_PATH = str(
    Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
)

_WIKITEXT_PPL_ORACLE_CONFIG: dict[str, int | str] = {
    "wikitext_path": _WIKITEXT2_DEFAULT_PATH,
    # vqbench REPORT headline runs 512 tokens (two 256-token chunks).
    "chunk_size": 256,
    "max_tokens": 512,
    # At least one full chunk of scored tokens — a zero-scored-token
    # run is a measurement degenerate that the oracle should reject
    # loudly.
    "min_scored_tokens": 256,
}


_QWEN3_0_6B_WIKITEXT_PPL_FP16 = Scenario(
    id="qwen3-0.6b-wikitext-ppl-fp16",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-fp16",
        # PPL bypasses engine.generate_batch entirely — the runner's
        # ``_run_ppl`` reads wikitext_path + drives the adapter via
        # silica.bench.ppl_oracle. prompts / max_tokens carry no
        # signal for this oracle.
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
        kv_codec=None,
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-C.2 baseline PPL row. Teacher-forced streaming PPL on "
        "WikiText-2 raw test split (chunk_size=256, max_tokens=512 "
        "matching vqbench REPORT headline). Cache-agnostic fp16 "
        "baseline — drives the adapter's own BatchKVCache across "
        "chunks (shared cache, chunk-invariant). Cache-only gate "
        "plus WikiText cache file presence at "
        "~/.cache/silica/wikitext2-test.txt (populate once via "
        "scripts/prepare_wikitext2_cache.py)."
    ),
)


_QWEN3_0_6B_WIKITEXT_PPL_TQ_MSE_B4 = Scenario(
    id="qwen3-0.6b-wikitext-ppl-tq-mse-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-tq-mse-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="tq_mse_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-C.2 TurboQuantMSE 4-bit K+V PPL row. Same workload "
        "shape as qwen3-0.6b-wikitext-ppl-fp16 but routes every "
        "chunk's prior prefix through ``tq_mse_b4`` "
        "encode/decode. Emits a raw PPL number; ΔPPL vs the fp16 "
        "baseline is a downstream derivation (bench report / C.6 "
        "vqbench cross-check) — step 3a does not propagate the "
        "fp16 PPL between rows at runtime."
    ),
)


_QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4 = Scenario(
    id="qwen3-0.6b-wikitext-ppl-block-tq-b64-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-block-tq-b64-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-C.2 BlockTurboQuantMSE B=64 4-bit K+V PPL row. "
        "Production recommendation per vqbench REPORT §3.1 "
        "(strictly lossless at std=0% across three seeds on "
        "Qwen3.5-4B; 3.76x total-KV compression). silica mirrors "
        "the configuration on Qwen3-0.6B; the row emits a raw PPL "
        "number at step 3a. ΔPPL against the fp16 baseline is a "
        "downstream computation (bench report / C.6 vqbench "
        "cross-check), not wired into the per-row runner in 3a. "
        "C.6 step 1 wires this row through --vqbench-xcheck so a "
        "side-by-side vqbench PPL lands in metadata.vqbench_*."
    ),
    # P-5-C.6 step 1 live demo. Runner auto-appends --model /
    # --seed / --chunk / --max-tokens from execution context; the
    # spec only carries the fixed bits (method name, bit width,
    # BlockTQ-specific --block-size + --patch-v flags).
    vqbench_xcheck=VqbenchXcheckSpec(
        script_path=str(default_reproduce_script_path()),
        method="BlockTurboQuantMSE",
        bits=4,
        extra_args=("--block-size", "64", "--patch-v"),
    ),
)


# P-5-D.2a — vqbench-aligned BlockTQ arm. Same workload knobs as the
# post-RoPE row above, but oracle_config pins
# ``codec_quality_path="vqbench_aligned"`` so ``_run_ppl`` dispatches to
# the projection-patch oracle (pre-RoPE K+V compression, mirroring
# vqbench's ``methods.common.monkey_patch._QuantizedProj``). Same
# ``VqbenchXcheckSpec`` as the post-RoPE row so C.6 can tell the two
# observables apart in the JSONL — this row exists to bind the
# ``--vqbench-xcheck`` comparison against vqbench's own
# projection-patch semantic, i.e. "inject noise in the same pre-RoPE
# space vqbench does". The post-RoPE row's gap against vqbench is an
# honest measurement of the production store path's cost and does NOT
# have to close to zero. The P-5 (4-b) Acceptance rule — whether the
# vqbench-aligned arm gap is aggregated, what noise window is
# acceptable, whether the existing per-row epsilon gate
# (_compute_gap_fields / _VQBENCH_PCT_EPSILON) applies as-is —
# is owned by P-5-D.3, NOT by this scenario. D.2a's verification
# (docs/P5_D2_INVESTIGATION) recorded per-row |gap| ≈ 0.45–0.61 PPL
# against vqbench while mean-over-seeds agrees to ~0.15 PPL, so the
# per-row epsilon gate as currently coded will still fire the
# divergence warning on this row. That is expected and part of the
# D.3 input.
_QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_VQBENCH_ALIGNED = Scenario(
    id="qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-block-tq-b64-b4-vqbench-aligned",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        # ``prefix_cache=True`` is set only to satisfy the Workload
        # invariant (``kv_codec != None`` implies
        # ``prefix_cache=True``). The vqbench-aligned oracle path
        # does not consult ``_maybe_build_prefix_cache`` — the codec
        # fires inside the projection wrapper, not the store.
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config={
        **_WIKITEXT_PPL_ORACLE_CONFIG,
        "codec_quality_path": "vqbench_aligned",
    },
    gate_env_var=None,
    description=(
        "P-5-D.2a diagnostic PPL row. Identical workload shape to "
        "qwen3-0.6b-wikitext-ppl-block-tq-b64-b4 but runs the codec "
        "in vqbench's pre-RoPE projection-patch semantic instead of "
        "the production prefix-cache store. Exists because the D.2 "
        "investigation found the post-RoPE store path and the "
        "pre-RoPE projection-patch path give ΔPPL numbers that "
        "differ by ~20x at the same codec Frobenius; the C.6 "
        "vqbench cross-check binds against this row so the "
        "comparison is apples-to-apples with vqbench's own harness "
        "(both sides inject noise in pre-RoPE space). The post-RoPE "
        "store row is preserved as a separate observable of the "
        "production path's quality cost. See "
        "docs/P5_D2_INVESTIGATION/ for the probe scripts that "
        "established this split."
    ),
    vqbench_xcheck=VqbenchXcheckSpec(
        script_path=str(default_reproduce_script_path()),
        method="BlockTurboQuantMSE",
        bits=4,
        extra_args=("--block-size", "64", "--patch-v"),
    ),
)


_QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_PRE_NORM = Scenario(
    id="qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-norm",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-block-tq-b64-b4-pre-norm",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config={
        **_WIKITEXT_PPL_ORACLE_CONFIG,
        "codec_quality_path": "prefix_store_pre_norm",
    },
    gate_env_var=None,
    description=(
        "P-5-F (3b) production-store row — pre-k_norm store via "
        "the adapter PreNormCaptureAdapter Protocol "
        "(silica/models/pre_norm_capture.py). The adapter installs "
        "_PreNormCaptureProxy on every attention layer's k_proj at "
        "construction time; the oracle arms the proxy via "
        "adapter.install_pre_norm_capture(buffer), captures K_pre at "
        "projection output (the same space vqbench's _QuantizedProj "
        "injects in), persists pre-k_norm K in the prefix store, and "
        "calls adapter.apply_k_norm_then_rope on hit-path admit. "
        "F.0b' verified +0.015 PPL on the (4-b) reference scenario "
        "(better than D.2a's +0.51); persistent block-grained "
        "encoding mirrors what the F.2+ production store does, "
        "unlike the D.2a oracle which re-encodes per chunk. Proxy "
        "does NOT modify in-flight forward (returns k_proj(x) "
        "unchanged) — in-flight K stays clean; codec noise affects "
        "only prior chunks' K via the seeded-cache hit path. "
        "Matches the production prefix-cache deployment semantic."
    ),
)


_QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_PRE_ROPE = Scenario(
    id="qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-rope",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-block-tq-b64-b4-pre-rope",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config={
        **_WIKITEXT_PPL_ORACLE_CONFIG,
        "codec_quality_path": "prefix_store_pre_rope",
    },
    gate_env_var=None,
    description=(
        "P-5-F F.0b prototype row — same workload shape as the "
        "post-RoPE production row but ``codec_quality_path="
        "'prefix_store_pre_rope'`` selects the inverse-RoPE "
        "round-trip oracle (``teacher_forced_chunked_nll_with_codec_pre_rope``). "
        "The codec sees pre-RoPE K, mathematically equivalent to "
        "vqbench's ``_QuantizedProj`` injection space (RoPE is "
        "orthogonal). Persistent block-grained encoding mirrors "
        "what the F.1+ production store will do, unlike the D.2a "
        "row which re-encodes per chunk. F.0 (b) gate: ΔPPL "
        "should land inside the D.2a oracle envelope "
        "(+0.51 +/- 0.35 PPL) or at minimum <= 1.5 PPL "
        "representing >=13x reduction from the post-RoPE row's "
        "+20 PPL (P5_F_OPENING.md §6.1). 3-seed evaluation, same "
        "{42, 43, 44} the (4-b) gate uses."
    ),
)


_QWEN3_0_6B_WIKITEXT_PPL_EXT_RABITQ_B4 = Scenario(
    id="qwen3-0.6b-wikitext-ppl-ext-rabitq-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-C.2 ExtRaBitQ 4-bit K+V PPL row. RaBitQ family arm "
        "of the PPL bench; effective bits/coord = 4 + 48/head_dim "
        "(head_dim=128 -> 4.375). Emits a raw PPL number at step "
        "3a; ΔPPL derivation against fp16 is downstream. "
        "rabitq_b1 is deliberately excluded: K-only codec "
        "(``v_supported=False``) that cannot install via the "
        "symmetric ``kv_codec=`` shorthand, and its hypercube "
        "MSE is worse than BlockTQ at matching bit budget."
    ),
)


# =============================================================================
# P-3-C5-step2-D — WikiText-2 PPL rows on Qwen3.5-0.8B (hybrid).
#
# Mirror of the Qwen3-0.6B PPL block above. Same workload shape
# (chunk_size=256, max_tokens=512, wikitext-2 raw test split) so
# that running both rows on the same input gives a head-to-head
# comparison between pure-attention (Qwen3) and hybrid-DeltaNet
# (Qwen3.5) under the same K/V codec configuration. Pre-C5.4 this
# combination raised the recurrent-state ctor guard; pre-step2-W
# the codec oracle crashed on the heterogeneous cache shape. Both
# gates landed; these scenarios are the consumer.
#
# Skipped scenarios on the hybrid side (compared to the Qwen3-0.6B
# block):
#  - tq_mse_b4: pre-vqbench codec, lower priority for the hybrid
#    comparison.
#  - vqbench-aligned (P-5-D.2a): the pre-RoPE projection-patch
#    path lives in ``teacher_forced_chunked_nll_vqbench_aligned``,
#    which is a separate codepath from the production-store
#    oracle and has not been hybrid-adapted at this sub-unit. The
#    cross-method comparison vs vqbench's own Qwen3-0.6B numbers
#    therefore relies on the post-RoPE production-store rows
#    here.
#
# C.6 ``vqbench_xcheck`` is also dropped on the hybrid side: the
# referenced ``reproduce.py`` script in the vqbench repo predates
# Qwen3.5 hybrid models and may not run on this architecture.
# Adding the cross-check after confirming vqbench's hybrid support
# is a follow-up.
# =============================================================================


_QWEN3_5_0_8B_WIKITEXT_PPL_FP16 = Scenario(
    id="qwen3.5-0.8b-wikitext-ppl-fp16",
    repo="Qwen/Qwen3.5-0.8B",
    workload=Workload(
        name="wikitext-ppl-fp16",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
        kv_codec=None,
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2-D fp16 baseline PPL row on Qwen3.5-0.8B "
        "(hybrid-DeltaNet). Mirrors qwen3-0.6b-wikitext-ppl-fp16 "
        "with the same chunk_size=256 / max_tokens=512 workload "
        "knobs. Drives ``teacher_forced_chunked_nll`` against the "
        "adapter's own heterogeneous cache (``ArraysCache`` for "
        "DeltaNet layers + ``BatchKVCache`` for full-attention "
        "layers); recurrent state accumulates naturally across "
        "chunks. Cache-only gate plus WikiText cache file presence "
        "at ~/.cache/silica/wikitext2-test.txt."
    ),
)


_QWEN3_5_0_8B_WIKITEXT_PPL_BLOCK_TQ_B64_B4 = Scenario(
    id="qwen3.5-0.8b-wikitext-ppl-block-tq-b64-b4",
    repo="Qwen/Qwen3.5-0.8B",
    workload=Workload(
        name="wikitext-ppl-block-tq-b64-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2-D BlockTurboQuantMSE B=64 4-bit K+V PPL row "
        "on Qwen3.5-0.8B (hybrid-DeltaNet). Routes every chunk's "
        "prior prefix through the C5-step2-W hybrid-aware codec "
        "oracle: heterogeneous cache assembly at hit chunks plus "
        "recurrent snapshot/restore at chunk boundaries. The codec "
        "compresses K/V on full-attention layers only; DeltaNet "
        "recurrent state passes through unchanged (snapshot/restore "
        "preserves the fp16 trajectory across chunks). ΔPPL "
        "against the fp16 baseline is the production codec's "
        "quality cost on hybrid; cross-method comparison against "
        "vqbench's own Qwen3-0.6B numbers is the discriminator the "
        "user is after."
    ),
)


_QWEN3_5_0_8B_WIKITEXT_PPL_EXT_RABITQ_B4 = Scenario(
    id="qwen3.5-0.8b-wikitext-ppl-ext-rabitq-b4",
    repo="Qwen/Qwen3.5-0.8B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2-D ExtRaBitQ 4-bit K+V PPL row on "
        "Qwen3.5-0.8B (hybrid-DeltaNet). Same hybrid-aware codec "
        "oracle path as the BlockTQ row; effective bits/coord = "
        "4 + 48/head_dim (head_dim=256 → 4.1875)."
    ),
)


# =============================================================================
# P-3-C5-step2-E — WikiText-2 PPL rows on Qwen3.5-4B (hybrid).
#
# Same workload knobs as the Qwen3.5-0.8B block above; Qwen3.5-4B
# is the model vqbench REPORT §3.1 used to claim "strictly lossless
# at std=0% across three seeds" for block_tq_b64_b4. Mirroring on
# silica's post-RoPE production-store oracle gives the head-to-head
# comparison the user requested: vqbench's pre-RoPE-patch claim on
# Qwen3.5-4B vs silica's post-RoPE-store claim on the same model.
# =============================================================================


_QWEN3_5_4B_WIKITEXT_PPL_FP16 = Scenario(
    id="qwen3.5-4b-wikitext-ppl-fp16",
    repo="Qwen/Qwen3.5-4B",
    workload=Workload(
        name="wikitext-ppl-fp16",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
        kv_codec=None,
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2-E fp16 baseline PPL row on Qwen3.5-4B "
        "(hybrid-DeltaNet, the model vqbench REPORT used for its "
        "BlockTQ lossless claim). Same chunk_size=256 / "
        "max_tokens=512 wikitext-2 workload as the Qwen3.5-0.8B "
        "and Qwen3-0.6B fp16 rows so the codec rows below are "
        "directly comparable across model sizes."
    ),
)


_QWEN3_5_4B_WIKITEXT_PPL_BLOCK_TQ_B64_B4 = Scenario(
    id="qwen3.5-4b-wikitext-ppl-block-tq-b64-b4",
    repo="Qwen/Qwen3.5-4B",
    workload=Workload(
        name="wikitext-ppl-block-tq-b64-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2-E BlockTurboQuantMSE B=64 4-bit K+V PPL row "
        "on Qwen3.5-4B. Drives the C5-step2-W hybrid-aware codec "
        "oracle (heterogeneous cache + recurrent snapshot/restore "
        "across chunks) on the same model vqbench REPORT used for "
        "its lossless claim. Cross-method comparison: vqbench's "
        "pre-RoPE projection-patch path vs silica's post-RoPE "
        "production-store path on identical inputs."
    ),
)


_QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B2 = Scenario(
    id="qwen3.5-4b-wikitext-ppl-ext-rabitq-b2",
    repo="Qwen/Qwen3.5-4B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b2",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b2",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-F F.0a aggressive-codec ablation on Qwen3.5-4B "
        "(hybrid-DeltaNet, 4B class). ExtRaBitQ 2-bit K+V — most "
        "aggressive symmetric ext_rabitq variant currently "
        "registered. Goal: surface the post-RoPE failure surface "
        "on production-scale hybrid targets at low bit depth, "
        "feeding the P-5-F priority decision (does pre-RoPE store "
        "matter on 4B+ deployment, or only on small-model / "
        "ultra-aggressive settings?). Read alongside the b3 row "
        "as a bit-depth trend."
    ),
)


_QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B3 = Scenario(
    id="qwen3.5-4b-wikitext-ppl-ext-rabitq-b3",
    repo="Qwen/Qwen3.5-4B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b3",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b3",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-F F.0a intermediate-bit-depth row on Qwen3.5-4B. "
        "ExtRaBitQ 3-bit K+V; together with the b2 and b4 rows "
        "this gives a three-point bit-depth trend on the "
        "production-scale hybrid target."
    ),
)


_QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B4 = Scenario(
    id="qwen3.5-4b-wikitext-ppl-ext-rabitq-b4",
    repo="Qwen/Qwen3.5-4B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2-E ExtRaBitQ 4-bit K+V PPL row on "
        "Qwen3.5-4B. Same hybrid-aware codec oracle path as the "
        "BlockTQ row; effective bits/coord depends on Qwen3.5-4B's "
        "head_dim (read at runtime from the adapter's kv_layout)."
    ),
)


# =============================================================================
# P-3-C5-step2.5 — WikiText-2 PPL rows on Qwen3-4B (pure-attention
# control point).
#
# step2-D / step2-E surfaced a 3-orders-of-magnitude gap in K/V
# codec ΔPPL between pure-attention Qwen3-0.6B (+20 PPL) and
# hybrid Qwen3.5 (≈0 PPL). The gap conflates three variables:
# architecture (pure-attention vs hybrid-DeltaNet), head_dim (128
# vs 256), and model size (0.6B vs 0.8B/4B).
#
# Qwen3-4B is the cheapest control point that disambiguates size:
# pure-attention, head_dim=128, num_kv_heads=8 — identical to
# Qwen3-0.6B except scaled to 4B. Running the same three PPL rows
# gives a clean answer to "is the gap explained by model size?"
# Architecture vs head_dim disambiguation would require a hybrid
# head_dim=128 model (does not appear in the Qwen3.5 family) and
# is left as future work.
# =============================================================================


_QWEN3_4B_WIKITEXT_PPL_FP16 = Scenario(
    id="qwen3-4b-wikitext-ppl-fp16",
    repo="Qwen/Qwen3-4B",
    workload=Workload(
        name="wikitext-ppl-fp16",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
        kv_codec=None,
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2.5 fp16 baseline PPL row on Qwen3-4B "
        "(pure-attention, head_dim=128). Control point for the "
        "size dimension of the hybrid-vs-attention codec gap "
        "surfaced in step2-D / step2-E."
    ),
)


_QWEN3_4B_WIKITEXT_PPL_BLOCK_TQ_B64_B4 = Scenario(
    id="qwen3-4b-wikitext-ppl-block-tq-b64-b4",
    repo="Qwen/Qwen3-4B",
    workload=Workload(
        name="wikitext-ppl-block-tq-b64-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2.5 BlockTurboQuantMSE B=64 4-bit K+V PPL "
        "row on Qwen3-4B. Pure-attention control point at "
        "head_dim=128 same as Qwen3-0.6B, scaled to 4B size. "
        "ΔPPL against fp16 disambiguates whether the +20 PPL gap "
        "on Qwen3-0.6B was driven by model size."
    ),
)


_QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B2 = Scenario(
    id="qwen3-4b-wikitext-ppl-ext-rabitq-b2",
    repo="Qwen/Qwen3-4B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b2",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b2",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-F F.0a aggressive-codec ablation on Qwen3-4B "
        "(pure-attention, 4B class). ExtRaBitQ 2-bit K+V — most "
        "aggressive symmetric ext_rabitq variant currently "
        "registered. Pure-attention counterpart to the Qwen3.5-4B "
        "b2 row; together they ablate codec aggressiveness against "
        "architecture (attention vs hybrid) at the same 4B size."
    ),
)


_QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B3 = Scenario(
    id="qwen3-4b-wikitext-ppl-ext-rabitq-b3",
    repo="Qwen/Qwen3-4B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b3",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b3",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-5-F F.0a intermediate-bit-depth row on Qwen3-4B. "
        "ExtRaBitQ 3-bit K+V; together with the b2 and b4 rows "
        "this gives a three-point bit-depth trend on the "
        "production-scale pure-attention target."
    ),
)


_QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B4 = Scenario(
    id="qwen3-4b-wikitext-ppl-ext-rabitq-b4",
    repo="Qwen/Qwen3-4B",
    workload=Workload(
        name="wikitext-ppl-ext-rabitq-b4",
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b4",
    ),
    oracle=OracleKind.PPL,
    oracle_config=dict(_WIKITEXT_PPL_ORACLE_CONFIG),
    gate_env_var=None,
    description=(
        "P-3-C5-step2.5 ExtRaBitQ 4-bit K+V PPL row on Qwen3-4B. "
        "Same control-point intent as the BlockTQ row; provides "
        "a second codec data point for the size-dimension "
        "ablation."
    ),
)


# =============================================================================
# P-5-C.3 step 1 — memory-residency rows on Qwen3-0.6B.
#
# Same shared-prefix 2-prompt workload the A.3c / B.3 decode-speed
# rows use, so ``_extract_and_insert_prefix`` on row 0 termination and
# ``_admit_single_hit_row`` on row 1 admission both fire against the
# configured codec. Runner's ``_run_storage`` reads
# ``prefix_cache.store.resident_bytes()`` + live-block count + the
# per-block resident-bytes figure after the event stream drains.
#
# ``fp16`` uses ``kv_codec="fp16"`` (IdentityCodec baseline), NOT
# ``kv_codec=None``. The pass-through path (``codec=None``) stores raw
# ``mx.array`` references and reports ``resident_bytes_per_block() ==
# None``; the IdentityCodec path reports honest per-block fp16
# residency, giving compression rows a directly-comparable baseline
# at the same observable (D-012).
# =============================================================================


_QWEN3_0_6B_COMPRESSION_FP16 = Scenario(
    id="qwen3-0.6b-compression-fp16",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="compression-fp16",
        prompts=(
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
        ),
        max_tokens=16,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="fp16",
    ),
    oracle=OracleKind.STORAGE,
    gate_env_var=None,
    description=(
        "P-5-C.3 IdentityCodec baseline compression row. Same "
        "shared-prefix 2-prompt workload as the A.3c decode-speed "
        "rows; after the event stream drains, the runner reads "
        "prefix_cache.store.resident_bytes() and live-block "
        "counts. Under IdentityCodec the resident_bytes equals "
        "the uncompressed fp16 K/V sum, giving every subsequent "
        "compression row a directly-comparable baseline. "
        "Explicitly uses kv_codec='fp16' (not None) so the store "
        "is SyntheticPrefixBlockStore with IdentityCodec; the "
        "pass-through path would leave resident_bytes_per_block "
        "as None and break cross-row compression comparisons. "
        "Cache-only gate."
    ),
)


_QWEN3_0_6B_COMPRESSION_TQ_MSE_B4 = Scenario(
    id="qwen3-0.6b-compression-tq-mse-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="compression-tq-mse-b4",
        prompts=(
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
        ),
        max_tokens=16,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="tq_mse_b4",
    ),
    oracle=OracleKind.STORAGE,
    gate_env_var=None,
    description=(
        "P-5-C.3 TurboQuantMSE 4-bit K+V compression row. "
        "Scalar 4-bit quantization; effective bits/coord = 4 + "
        "2/head_dim (one fp16 scale per vector). Reports raw "
        "resident_bytes; cross-row compression ratio vs the fp16 "
        "row is a downstream derivation."
    ),
)


_QWEN3_0_6B_COMPRESSION_BLOCK_TQ_B64_B4 = Scenario(
    id="qwen3-0.6b-compression-block-tq-b64-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="compression-block-tq-b64-b4",
        prompts=(
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
        ),
        max_tokens=16,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="block_tq_b64_b4",
    ),
    oracle=OracleKind.STORAGE,
    gate_env_var=None,
    description=(
        "P-5-C.3 BlockTurboQuantMSE B=64 4-bit K+V compression "
        "row. Production recommendation per vqbench REPORT §3.1 "
        "(3.76x total-KV compression on Qwen3.5-4B). The "
        "resident_bytes number this row surfaces is the headline "
        "compression observable of P-5."
    ),
)


_QWEN3_0_6B_COMPRESSION_EXT_RABITQ_B4 = Scenario(
    id="qwen3-0.6b-compression-ext-rabitq-b4",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="compression-ext-rabitq-b4",
        prompts=(
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
            _PREFIX_HIT_DECODE_SHARED_PROMPT,
        ),
        max_tokens=16,
        max_batch_size=1,
        prefix_cache=True,
        temperature=0.0,
        top_p=1.0,
        kv_codec="ext_rabitq_b4",
    ),
    oracle=OracleKind.STORAGE,
    gate_env_var=None,
    description=(
        "P-5-C.3 ExtRaBitQ 4-bit K+V compression row. "
        "effective bits/coord = 4 + 48/head_dim (head_dim=128 -> "
        "4.375). Reports raw resident_bytes; compared against the "
        "fp16 baseline row at the bench report layer."
    ),
)


# =============================================================================
# P-5-C.3 step 2 — admission-headroom row on Qwen3-0.6B.
#
# Demonstrates opening §7(c): under ``account_prefix_residency=True``
# the compressed codec (mode C) frees enough prefix residency to
# strictly admit more concurrent trial requests than the IdentityCodec
# baseline (mode B). Workload is abstract — prompts=(), max_tokens=0
# at the workload level; all numeric knobs live in oracle_config.
#
# cap_bytes / n_prompt / max_tokens chosen so the fp16 baseline
# admits a small but non-zero number (~3-6) and the compressed
# arm admits strictly more. Leaving weights_bytes=0 isolates the
# signal on pure prefix residency (per user guidance: simplest
# first-version shape).
# =============================================================================


_QWEN3_0_6B_ADMISSION_HEADROOM_PREFIX_HEAVY = Scenario(
    id="qwen3-0.6b-admission-headroom-prefix-heavy",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="admission-headroom-prefix-heavy",
        # Oracle bypasses engine.generate_batch entirely; the
        # numeric workload knobs live in oracle_config below.
        prompts=(),
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=False,
        temperature=0.0,
        top_p=1.0,
        kv_codec=None,
    ),
    oracle=OracleKind.ADMISSION_HEADROOM,
    oracle_config={
        # Synthetic cap; small enough that Qwen3-0.6B's per-block
        # IdentityCodec residency (~1.8 MB under its layout) hits
        # the warmup target in a reasonable block count, but large
        # enough that subsequent trial admissions (per-request
        # worst_case_bytes ~= 115 KB × (n_prompt + max_tokens))
        # fit more than once.
        "cap_bytes": 128 * 1024 * 1024,  # 128 MB
        "weights_bytes": 0,
        "warmup_ratio": 0.5,
        "n_prompt": 128,
        "max_tokens": 16,
        "fp16_codec": "fp16",
        "compressed_codec": "block_tq_b64_b4",
    },
    gate_env_var=None,
    description=(
        "P-5-C.3 step 2 admission-headroom row demonstrating §4.7 "
        "mode (B) vs mode (C) and pinning §7(c) acceptance "
        "n_block > n_fp16. Runner warms a prefix cache under "
        "IdentityCodec until store.resident_bytes() >= cap_bytes * "
        "warmup_ratio, replays the identical block recipe under "
        "block_tq_b64_b4, then runs a consecutive-AdmitDecision "
        "count against each. Compressed residency frees more "
        "headroom, so the compressed arm strictly admits more "
        "trial requests than the fp16 baseline. Hard gate on the "
        "inequality is the oracle's acceptance check; metadata "
        "surfaces the admission delta and the residency ratio "
        "for the bench report. Cache-only gate (weight load via "
        "engine_factory is paid but the engine itself is "
        "discarded — v1 accepts this cost rather than inventing "
        "a metadata-only adapter loader)."
    ),
)


BUILTIN_SCENARIOS: dict[str, Scenario] = {
    _QWEN3_0_6B_SMOKE.id: _QWEN3_0_6B_SMOKE,
    _QWEN3_0_6B_B1_PARITY.id: _QWEN3_0_6B_B1_PARITY,
    _QWEN3_0_6B_BGT1_PARITY.id: _QWEN3_0_6B_BGT1_PARITY,
    _QWEN3_0_6B_SHORT_IN_LONG_OUT.id: _QWEN3_0_6B_SHORT_IN_LONG_OUT,
    _QWEN3_0_6B_LONG_IN_SHORT_OUT.id: _QWEN3_0_6B_LONG_IN_SHORT_OUT,
    _QWEN3_0_6B_CONCURRENT_SHARED_PREFIX.id: _QWEN3_0_6B_CONCURRENT_SHARED_PREFIX,
    _QWEN3_0_6B_TEACHER_FORCED_ARGMAX.id: _QWEN3_0_6B_TEACHER_FORCED_ARGMAX,
    _QWEN3_0_6B_TTFT_UNDER_CONCURRENCY.id: _QWEN3_0_6B_TTFT_UNDER_CONCURRENCY,
    _QWEN3_5_0_8B_B1_PARITY.id: _QWEN3_5_0_8B_B1_PARITY,
    _QWEN3_0_6B_PREFIX_HIT_DECODE_FP16.id: _QWEN3_0_6B_PREFIX_HIT_DECODE_FP16,
    _QWEN3_0_6B_PREFIX_HIT_DECODE_BLOCK_TQ_B64_B4.id: _QWEN3_0_6B_PREFIX_HIT_DECODE_BLOCK_TQ_B64_B4,
    _QWEN3_0_6B_PREFIX_HIT_DECODE_EXT_RABITQ_B4.id: _QWEN3_0_6B_PREFIX_HIT_DECODE_EXT_RABITQ_B4,
    _QWEN3_0_6B_WIKITEXT_PPL_FP16.id: _QWEN3_0_6B_WIKITEXT_PPL_FP16,
    _QWEN3_0_6B_WIKITEXT_PPL_TQ_MSE_B4.id: _QWEN3_0_6B_WIKITEXT_PPL_TQ_MSE_B4,
    _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4.id: _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4,
    _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_VQBENCH_ALIGNED.id: (
        _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_VQBENCH_ALIGNED
    ),
    _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_PRE_ROPE.id: (
        _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_PRE_ROPE
    ),
    _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_PRE_NORM.id: (
        _QWEN3_0_6B_WIKITEXT_PPL_BLOCK_TQ_B64_B4_PRE_NORM
    ),
    _QWEN3_0_6B_WIKITEXT_PPL_EXT_RABITQ_B4.id: _QWEN3_0_6B_WIKITEXT_PPL_EXT_RABITQ_B4,
    _QWEN3_5_0_8B_WIKITEXT_PPL_FP16.id: _QWEN3_5_0_8B_WIKITEXT_PPL_FP16,
    _QWEN3_5_0_8B_WIKITEXT_PPL_BLOCK_TQ_B64_B4.id: (
        _QWEN3_5_0_8B_WIKITEXT_PPL_BLOCK_TQ_B64_B4
    ),
    _QWEN3_5_0_8B_WIKITEXT_PPL_EXT_RABITQ_B4.id: _QWEN3_5_0_8B_WIKITEXT_PPL_EXT_RABITQ_B4,
    _QWEN3_5_4B_WIKITEXT_PPL_FP16.id: _QWEN3_5_4B_WIKITEXT_PPL_FP16,
    _QWEN3_5_4B_WIKITEXT_PPL_BLOCK_TQ_B64_B4.id: (
        _QWEN3_5_4B_WIKITEXT_PPL_BLOCK_TQ_B64_B4
    ),
    _QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B2.id: _QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B2,
    _QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B3.id: _QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B3,
    _QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B4.id: _QWEN3_5_4B_WIKITEXT_PPL_EXT_RABITQ_B4,
    _QWEN3_4B_WIKITEXT_PPL_FP16.id: _QWEN3_4B_WIKITEXT_PPL_FP16,
    _QWEN3_4B_WIKITEXT_PPL_BLOCK_TQ_B64_B4.id: (
        _QWEN3_4B_WIKITEXT_PPL_BLOCK_TQ_B64_B4
    ),
    _QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B2.id: _QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B2,
    _QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B3.id: _QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B3,
    _QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B4.id: _QWEN3_4B_WIKITEXT_PPL_EXT_RABITQ_B4,
    _QWEN3_0_6B_COMPRESSION_FP16.id: _QWEN3_0_6B_COMPRESSION_FP16,
    _QWEN3_0_6B_COMPRESSION_TQ_MSE_B4.id: _QWEN3_0_6B_COMPRESSION_TQ_MSE_B4,
    _QWEN3_0_6B_COMPRESSION_BLOCK_TQ_B64_B4.id: _QWEN3_0_6B_COMPRESSION_BLOCK_TQ_B64_B4,
    _QWEN3_0_6B_COMPRESSION_EXT_RABITQ_B4.id: _QWEN3_0_6B_COMPRESSION_EXT_RABITQ_B4,
    _QWEN3_0_6B_ADMISSION_HEADROOM_PREFIX_HEAVY.id: _QWEN3_0_6B_ADMISSION_HEADROOM_PREFIX_HEAVY,
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
