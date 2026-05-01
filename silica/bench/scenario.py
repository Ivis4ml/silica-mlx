"""silica.bench.scenario — P-4.0 scenario schema.

A ``Scenario`` describes a single bench row: a model + a workload +
an oracle + gates. The dataclass deliberately separates these three
axes so scenario composition ("model X × workload Y × oracle Z")
does not require adding new fields when a new workload or oracle
lands later. PLAN §P-4 lists workload-shaped scenarios
(short-in/long-out, concurrent shared-prefix, TTFT-under-concurrency);
the migrations of D3.1 / E3 / 27B / 31B are model-shaped. Both fit
the same schema because ``workload`` and ``oracle`` are separate
fields from ``repo``.

Dual-gate pattern inherited from existing tests: a scenario runs
only when its HF cache is present AND its ``gate_env_var`` (if set)
equals ``"1"``. Cache presence alone is the weak gate (short
scenarios like Qwen3-0.6B); env var is the strong gate (20 GB+ MoE
loads). Env var names mirror the existing
``SILICA_REAL_QWEN3_5_MOE`` / ``SILICA_REAL_GEMMA4_31B`` etc. so
test-suite and bench opt-in are identical.

Minimal by design: the schema captures only what P-4.0 / P-4.1 need.
Later phases (P-4.3 teacher-forced argmax, P-5 KV codec switching)
extend via optional fields, not by forking ``Scenario`` into
subclasses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OracleKind(str, Enum):
    """How a scenario decides pass / fail.

    - ``SMOKE`` — prompts run end-to-end, every row emits at least
      one token, all token ids lie in vocab, no aborts. Baseline
      oracle for "does not crash on the real weights".
    - ``B1_PARITY_VS_SINGLE`` — B=1 batched output equals
      ``Engine.generate`` single-request byte-for-byte. Hard gate on
      the scheduler's B=1 handling regardless of kernel drift.
    - ``BGT1_DIRECT_BATCHED_REFERENCE`` — B>1 Silica batched output
      matches a direct mlx-lm batched forward driven with the same
      ``adapter.make_batch_cache(left_padding)`` list. Structurally
      equal to ``tests/test_p3_gemma4_batched_parity.py``; captures
      scheduler glue correctness without claiming vs-solo parity.
    - ``TEACHER_FORCED_ARGMAX`` — position-by-position next-token
      argmax agreement on a fixed prefix vs a reference. PLAN §P-3
      exit criterion; lands as a P-4.3 oracle.

    Additional kinds land as new enum members; the schema does not
    fork.
    """

    SMOKE = "smoke"
    B1_PARITY_VS_SINGLE = "b1_parity_vs_single"
    BGT1_DIRECT_BATCHED_REFERENCE = "bgt1_direct_batched_reference"
    TEACHER_FORCED_ARGMAX = "teacher_forced_argmax"
    # P-5-A.3b. Runs a two-prompt ``max_batch_size=1
    # prefix_cache=True`` workload so row 1 enters the waiting queue
    # and is admitted mid-run through the prefix-hit path
    # (``_admit_single_hit_row`` → ``fetch_detached_blocks`` →
    # ``codec.decode_tensor`` × 2 × num_layers × num_hit_blocks).
    # Reports row 1's decode tok/s specifically — the metric the
    # opening §7(d) gate asserts: BlockTQ row-1 decode tok/s ≥
    # 0.85 × IdentityCodec row-1 decode tok/s. Scenario authoring
    # pins the workload shape; runner validates.
    DECODE_TOK_S_WITH_PREFIX_HIT = "decode_tok_s_with_prefix_hit"
    # P-5-C.2 step 3. Teacher-forced streaming PPL on a tokenized
    # WikiText-2 test split. Does **not** go through
    # ``engine.generate_batch``; the runner drives
    # :func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll`
    # (fp16 baseline) or
    # :func:`silica.bench.ppl_oracle.teacher_forced_chunked_nll_with_codec`
    # (codec-backed) depending on ``workload.kv_codec``.
    # ``oracle_config`` carries ``wikitext_path`` (local UTF-8 text
    # file), ``chunk_size`` (default 256), ``max_tokens`` (default
    # 512), and ``min_scored_tokens`` (floor on ``n_tokens``,
    # default 1). ``collected`` payload shape is
    # ``{"nll_sum": float, "n_tokens": int, "ppl": float}``.
    PPL = "ppl"
    # P-5-C.3 step 1. Memory-residency observable for the prefix
    # cache's store. Drives a shared-prefix 2-prompt workload
    # (same shape as ``DECODE_TOK_S_WITH_PREFIX_HIT``) so the
    # scheduler's ``_extract_and_insert_prefix`` / prefix-hit path
    # fires and populates the ``SyntheticPrefixBlockStore``; after
    # the workload completes the runner reads
    # ``prefix_cache.store.resident_bytes()`` plus
    # ``len(live_block_ids())`` / ``resident_bytes_per_block()`` /
    # ``prefix_cache.hits`` and hands them to the oracle. The
    # oracle is pure structural validation (all fields present,
    # correctly typed, ``resident_bytes >= 0``, ``live_blocks >=
    # 1``). Cross-codec compression-ratio comparison is a
    # downstream concern (bench report / C.6 vqbench cross-check),
    # not gated here. ``collected`` payload shape is
    # ``{"resident_bytes": int, "resident_bytes_per_block":
    # int | None, "live_blocks": int, "prefix_cache_hits": int}``.
    STORAGE = "storage"
    # P-5-C.3 step 2. Admission-headroom observable demonstrating
    # §4.7 mode (B) vs mode (C): a compressed codec's smaller
    # ``store.resident_bytes()`` translates into more headroom and
    # therefore more admitted requests under the same ``cap_bytes``.
    #
    # Runner bypasses ``engine.generate_batch`` entirely. Workload
    # is abstract (``prompts=()``, ``max_tokens=0``); all numeric
    # knobs live in ``oracle_config``:
    #   cap_bytes, weights_bytes, warmup_ratio,
    #   n_prompt, max_tokens, fp16_codec, compressed_codec
    #
    # Procedure (mirrors opening §7(c) verbatim):
    # 1. Build prefix_cache(fp16_codec), synthesize blocks of
    #    zero-filled K/V one at a time until
    #    ``store.resident_bytes() >= cap_bytes * warmup_ratio``.
    #    Record the recipe ``[(tokens, per_layer_kv), ...]``.
    # 2. Build prefix_cache(compressed_codec), replay the SAME
    #    recipe verbatim. The compressed codec stores the same
    #    logical content at smaller ``resident_bytes``.
    # 3. For each prefix_cache, build a MemoryBudgeter with
    #    ``account_prefix_residency=True``, then call
    #    ``budgeter.admit("trial-N", n_prompt, max_tokens)`` in a
    #    loop. Count *consecutive* ``AdmitDecision`` returns; stop
    #    at the first non-``AdmitDecision`` (Reject /
    #    AdmitAfterEvict / AdmitAfterPreempt) — those mix
    #    eviction / preemption policy into the signal that should
    #    be pure-headroom.
    #
    # Oracle enforces the hard gate ``n_block > n_fp16`` (§7(c)
    # acceptance) on top of the usual structural checks.
    # ``collected`` shape: ``{"n_fp16": int, "n_block": int,
    # "n_delta": int, "resident_bytes_fp16": int,
    # "resident_bytes_block": int, "warmup_blocks": int}``.
    ADMISSION_HEADROOM = "admission_headroom"
    # P-6.0 (the P-6 measurement gate). Sustained warm-start
    # decode_tok_s on a single long-running generation, with kernel-
    # compile + first-forward latency excluded by a two-stage warm-up
    # rule: discard at least ``warmup_min_steps`` decode steps, then
    # continue discarding until the rolling-``warmup_rolling_window``
    # inter-token interval std/mean falls below
    # ``warmup_rel_std_threshold`` (later wins). The remaining decodes
    # form the measurement window from which decode_tok_s_warm,
    # decode_tok_s_warm_per_row_mean, and decode_tok_s_warm_aggregate
    # are computed (aggregate = total measurement decodes across all
    # rows / aggregate measurement-window wall, comparable to
    # vllm-mlx's headline number).
    #
    # Workload contract (validated at scenario-author time by
    # ``_validate_workload_for_oracle`` and at run time by
    # ``_run_warm_decode``):
    #
    #   * ``max_batch_size`` >= 1.
    #   * ``len(prompts)`` == ``max_batch_size`` — one prompt per row.
    #     Identical-prompt B>1 workloads make per-row decode tok/s
    #     directly comparable; differing-length prompts are allowed
    #     but the per-row tok/s is then driven partly by prefill
    #     length, not steady-state decode.
    #   * ``max_tokens`` >= ``warmup_min_steps + measurement_steps_min
    #     + 1`` (default ``32 + 64 + 1 = 97``; production scenarios
    #     use 384 to leave plenty of headroom).
    #   * ``prefix_cache=False`` and ``kv_codec=None`` — codec / hit-
    #     path measurements are orthogonal levers handled by
    #     ``DECODE_TOK_S_WITH_PREFIX_HIT``; this oracle measures the
    #     codec-free steady-state decode hot path so its number is
    #     the reference for every later P-6 track ratio.
    #
    # ``oracle_config`` keys (all optional, with defaults):
    #
    #   * ``warmup_min_steps`` (int, default 32) — minimum decode
    #     steps to discard before the rolling-stability rule applies.
    #     32 covers the MLX kernel-compile cost on a 64-layer 27B
    #     first forward (already observed in the v1.7.x 27B load
    #     probe at ~2.4 s for a 1-token prompt).
    #   * ``warmup_rolling_window`` (int, default 16) — rolling-window
    #     length for the std/mean stability check.
    #   * ``warmup_rel_std_threshold`` (float, default 0.05) — exit
    #     warm-up when rolling-window std/mean drops below this.
    #   * ``measurement_steps_min`` (int, default 64) — minimum number
    #     of decode steps that must remain after the warm-up
    #     boundary. The oracle fails with a structured reason if a
    #     row's warm-up never stabilises within ``max_tokens``.
    #
    # ``collected`` shape: ``(tokens, token_ts_ms)`` per row, same
    # structure as ``DECODE_TOK_S_WITH_PREFIX_HIT``. ``tokens`` is
    # ``dict[int, list[int]]``; ``token_ts_ms`` is
    # ``dict[int, list[float]]`` keyed by row index.
    #
    # Oracle metadata (on success):
    #
    #   * ``decode_tok_s_warm_aggregate`` (float) — promoted to
    #     ``ScenarioResult.decode_tok_s`` for JSONL homogeneity.
    #   * ``decode_tok_s_warm_per_row_mean`` (float).
    #   * ``rows`` (list[dict]) — per-row ``decode_tok_s_warm``,
    #     ``warmup_steps_used``, ``measurement_steps``,
    #     ``decode_interval_ms_{mean,std,rel_std}``, ``cold_ttft_ms``.
    #   * ``warmup_*`` and ``measurement_steps_min`` echo back so the
    #     JSONL row is self-describing.
    #
    # Note on ``ttft_warm_ms``: warm TTFT (post-kernel-compile first
    # forward) is not measured by this oracle's single-pass workload —
    # the only TTFT it sees is the cold one (compile-included), which
    # is recorded as ``cold_ttft_ms`` in per-row metadata. The warm-
    # TTFT scenario shape (two consecutive prompts, second's TTFT
    # measured) lands as ``WARM_TTFT_PAIR`` below (P-6.0.5 sub-unit 6).
    WARM_DECODE = "warm_decode"
    # P-6.0.5 (D-021 step 3) sub-unit 6 — warm-TTFT pair. Two prompts
    # issued sequentially through the same ``Engine`` instance;
    # prompt 1 amortises one-time kernel compile / cache warmup,
    # prompt 2's TTFT is the warm number reported by the §6 TTFT
    # scenarios. The oracle returns one row per pair.
    #
    # Workload contract (validated at scenario-author time by
    # ``_validate_workload_for_oracle`` and at run time by
    # ``_run_warm_ttft_pair``):
    #
    #   * ``max_batch_size`` == 1 — sequential single-request issuance,
    #     not batched dispatch.
    #   * ``len(prompts)`` == 2 — prompt 1 and prompt 2 in author order.
    #   * ``prompts[0]`` != ``prompts[1]`` — gate row enforces distinct
    #     prompts so ``prefix_hit_tokens`` is structurally 0 and the
    #     warm TTFT is uncontaminated by prefix-cache reuse.
    #   * ``prefix_cache`` == False — gate row disables prefix cache;
    #     prefix-cache effect on warm TTFT is a follow-up
    #     ``-shared-prefix`` variant scope (deferred per
    #     ``plans/P6_0_5_OPENING.md`` §3.6 OQ-2).
    #   * ``kv_codec`` is None — codec-free baseline; codec-on
    #     warm-TTFT is an orthogonal lever.
    #   * ``max_tokens`` >= 1 — only the first token per prompt is
    #     consumed for the TTFT measurement; ``max_tokens`` above 1
    #     just lets the iterator drain naturally before the next
    #     prompt fires.
    #
    # ``oracle_config`` is unused on the gate row (room reserved for a
    # future ``-shared-prefix`` variant's prefix-hit knobs).
    #
    # ``collected`` shape: ``dict`` with keys ``prompt1_ttft_ms`` /
    # ``prompt2_ttft_ms`` / ``prompt1_tokens`` / ``prompt2_tokens``.
    # Wall-clock timing is captured by the runner via
    # ``time.perf_counter`` immediately before each
    # ``Engine.generate`` call and read at the first-token yield
    # (mirrors ``_collect_warm_decode_b1``); not via
    # ``engine.metrics`` because that counter is overwritten per
    # prompt and would lose prompt 1's TTFT.
    #
    # Oracle metadata (on success):
    #
    #   * ``warm_ttft_ms`` (float) — alias for ``prompt2_ttft_ms``;
    #     promoted to ``ScenarioResult.ttft_ms`` for JSONL homogeneity.
    #   * ``prompt1_ttft_ms`` (float) — cold + compile cost.
    #   * ``prompt2_ttft_ms`` (float) — warm number (= warm_ttft_ms).
    #   * ``compile_amortized_ms`` (float) — diagnostic
    #     ``prompt1_ttft_ms - prompt2_ttft_ms``; positive means kernel
    #     compile dominated cold TTFT.
    #   * ``prompt1_tokens`` / ``prompt2_tokens`` (int) — tokenised
    #     prompt lengths so prompt-length asymmetry is observable
    #     rather than silently absorbed.
    #   * ``prefix_hit_tokens`` (int) — radix-prefix-cache hit count
    #     on prompt 2 (always 0 on the gate row by
    #     ``prefix_cache=False``; room for a ``-shared-prefix``
    #     variant to populate).
    WARM_TTFT_PAIR = "warm_ttft_pair"


@dataclass(frozen=True)
class Workload:
    """Prompt set + decoding parameters. Separated from model because
    the same workload ("short-in/long-out 4 tokens") applies across
    every model family.

    ``kv_codec`` (P-5-A.3a) names the codec to install on the prefix
    cache's store by ``silica.bench.codec_registry`` id. ``None`` (the
    default) keeps the pre-P-5 pass-through behaviour (raw fp16
    tensors stored in ``_detached``). ``"fp16"`` installs an explicit
    ``IdentityCodec`` — the honest decode-speed baseline against
    which compressed codecs are compared (§7(d) pins the 0.85×
    IdentityCodec ratio, not 0.85× pass-through). Compressed codec
    ids (``"block_tq_b64_b4"``, etc.) install the corresponding
    ``VectorCodec`` on both K and V sides via the shorthand.

    ``kv_codec`` is only meaningful when ``prefix_cache=True``; the
    runner validates this at workload construction time and rejects
    unknown ids against the ``codec_registry`` catalogue.

    The ``--kv-codec`` CLI flag + multi-codec sweep that flips
    scenarios' codec at runtime is P-5-C scope; A.3 ships scenarios
    that pin a specific codec id at authoring time.
    """

    name: str
    prompts: tuple[str, ...]
    max_tokens: int
    max_batch_size: int = 1
    prefix_cache: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    kv_codec: str | None = None

    def __post_init__(self) -> None:
        if self.kv_codec is not None:
            if not self.prefix_cache:
                raise ValueError(
                    f"Workload {self.name!r}: kv_codec="
                    f"{self.kv_codec!r} requires prefix_cache=True; "
                    f"codecs install on the prefix cache's store, "
                    f"so they are meaningless when no prefix cache "
                    f"exists"
                )
            # Lazy import to keep silica.bench.scenario free of
            # silica.bench.codec_registry at module-import time —
            # codec_registry depends on silica.vq, which depends on
            # silica.kvcache.codec + mlx.core. scenario.py is
            # imported from many lightweight callers (scripts,
            # tests) that don't want to pay that cost.
            from silica.bench.codec_registry import CODEC_REGISTRY

            if self.kv_codec not in CODEC_REGISTRY:
                known = ", ".join(sorted(CODEC_REGISTRY))
                raise ValueError(
                    f"Workload {self.name!r}: unknown kv_codec id "
                    f"{self.kv_codec!r}; registered: {known}"
                )


@dataclass(frozen=True)
class VqbenchXcheckSpec:
    """Declarative spec for a ``--vqbench-xcheck`` cross-check arm.

    Present on ``Scenario.vqbench_xcheck`` when the scenario's
    baked codec arm has a corresponding vqbench reproduce-script
    configuration. Scenario authors fill the fixed/rare bits
    (``script_path``, ``method``, ``bits``, ``extra_args``); the
    BenchRunner auto-appends the common ones (``--model``,
    ``--seed``, ``--chunk``, ``--max-tokens``) from the live
    execution context so they cannot drift vs what silica
    actually ran.

    Attributes:
        script_path: Path (absolute or cwd-relative) to the vqbench
            reproduce script. ``run_vqbench_baseline`` resolves
            relative paths against ``cwd``; authors typically pass
            ``str(default_reproduce_script_path())`` so the spec
            carries the absolute location explicitly.
        method: Value for vqbench's ``--method`` flag, e.g.
            ``"BlockTurboQuantMSE"``. Must match the vqbench class
            name the reproduce script dispatches to; silica-side
            ``codec_id`` is deliberately NOT auto-mapped to this
            (silica naming diverges from vqbench naming, and
            silent mapping hides drift).
        bits: Value for vqbench's ``--bits`` flag, e.g. 4.
        extra_args: Tuple of additional argv passed verbatim after
            the auto-appended common flags. Use for
            scenario-specific switches like ``("--block-size",
            "64", "--patch-v")``; do not duplicate the auto-append
            flags here (runner does not deduplicate — vqbench may
            reject or take the last value unpredictably).
    """

    script_path: str
    method: str
    bits: int
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """One bench row.

    ``id`` is the human-readable key used by the CLI
    (``--scenario qwen3-0.6b-smoke``) and the JSONL report's row
    identifier. ``repo`` is the HF path; the cache directory is
    derived by :func:`hf_cache_path_for_repo`, so scenarios do not
    hard-code filesystem layout. ``gate_env_var=None`` means
    cache-presence is the only gate (cheap enough to run without
    opt-in); a string value is the name of the env var that must
    equal ``"1"`` on top of cache presence.

    ``oracle`` picks the pass/fail function from
    :class:`OracleKind`; ``oracle_config`` carries oracle-specific
    parameters without growing the top-level dataclass.

    ``vqbench_xcheck`` (P-5-C.6 step 1) declares a vqbench
    cross-check arm. Only valid on ``OracleKind.PPL`` scenarios —
    vqbench's reproduce scripts produce PPL numbers, so pairing
    with any other oracle would be a category error. The
    ``__post_init__`` guard enforces this at authoring time.

    Deliberately not carrying ``expected_adapter_class`` — bench is
    throughput/latency, not correctness verification; factory
    dispatch is pinned in ``tests/test_models_factory.py``. Adapter
    type surfacing (if useful) belongs in ``ScenarioResult.metadata``.
    """

    id: str
    repo: str
    workload: Workload
    oracle: OracleKind = OracleKind.SMOKE
    oracle_config: dict[str, Any] = field(default_factory=dict)
    gate_env_var: str | None = None
    description: str = ""
    vqbench_xcheck: VqbenchXcheckSpec | None = None
    # D-021 step 5 sub-unit (h): speculative-decoding configuration.
    # ``None`` (default) keeps the scenario spec-off. A non-None value
    # opts the scenario into spec mode, but the bench runner only
    # actually wires ``DraftTargetEngine`` when it is constructed with
    # ``speculative_mode="draft_target"``. See :class:`SpecConfig`.
    spec_config: SpecConfig | None = None

    def __post_init__(self) -> None:
        # vqbench_xcheck requires OracleKind.PPL: the cross-check
        # compares silica's PPL oracle against vqbench's reproduce
        # script (which itself reports PPL). Pairing it with a
        # parity / smoke / storage oracle would have nothing
        # meaningful to cross-check — silence would bury the
        # authoring error under a silent skip, so raise loudly.
        if (
            self.vqbench_xcheck is not None
            and self.oracle != OracleKind.PPL
        ):
            raise ValueError(
                f"Scenario {self.id!r}: vqbench_xcheck is only "
                f"meaningful on OracleKind.PPL scenarios; got "
                f"oracle={self.oracle.value!r}. vqbench reproduce "
                f"scripts report PPL, so other oracles have "
                f"nothing to cross-check against"
            )


@dataclass(frozen=True)
class SpecConfig:
    """D-021 step 5 sub-unit (h): speculative-decoding parameters for a
    bench scenario.

    A scenario carrying ``spec_config`` declares it WANTS to run under
    ``--speculative draft_target``; the bench runner reads
    ``draft_repo`` and ``verify_k`` to wire ``DraftTargetEngine`` +
    ``SpecMetricCollector`` into the engine. Scenarios without
    ``spec_config`` (the common case) run spec-off regardless of CLI
    flag, and scenarios with ``spec_config`` still run spec-off when
    the CLI passes ``--speculative none`` (the default; backwards-
    compatible). Frozen so a scenario cannot be mutated mid-run.

    Validates ``verify_k >= 1`` and ``draft_repo != ""`` at
    construction so a misconfiguration cannot silently disable spec
    or land at a degenerate forward shape.
    """

    draft_repo: str
    verify_k: int = 4

    def __post_init__(self) -> None:
        if not self.draft_repo:
            raise ValueError(
                "SpecConfig.draft_repo must be non-empty (e.g. "
                "'Qwen/Qwen3.5-0.8B'); got empty string"
            )
        if self.verify_k < 1:
            raise ValueError(
                f"SpecConfig.verify_k must be >= 1, got {self.verify_k}"
            )


@dataclass
class ScenarioResult:
    """Outcome of running one scenario.

    ``status`` takes one of ``"ok"`` / ``"skipped"`` / ``"failed"``.
    Skip / failure reasons land in ``reason``; successful runs
    populate the metric fields from the engine's own snapshot plus
    wall-clock timing. ``metadata`` is a free-form dict for
    scenario-specific extras (e.g. adapter class name on a smoke,
    token-list length on teacher-forced).
    """

    scenario_id: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str | None = None
    ttft_ms: float | None = None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    resident_mb: float | None = None
    peak_memory_mb: float | None = None
    total_tokens: int | None = None
    wall_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def hf_cache_path_for_repo(repo: str) -> Path:
    """Derive the HF hub cache directory for ``repo``.

    The HF hub uses ``models--<owner>--<name>`` with ``/`` replaced
    by ``--`` for on-disk layout. Mirrors the skip checks already
    used by the dual-gated smoke tests (e.g.
    ``tests/test_p3_gemma4_batched_smoke.py``).
    """
    safe = repo.replace("/", "--")
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe}"


# --- Oracle function signature -----------------------------------------------

# Oracle functions consume the engine output + expected-output
# helpers and return (ok, reason, metadata). Runner invokes these
# after running the workload; specifics per-kind are in
# silica.bench.oracles.
#
# The second argument is ``Any`` because the workload output shape
# depends on the oracle kind — ``list[int]`` for single-request
# SMOKE / B1_PARITY, ``dict[int, list[int]]`` for
# BGT1_DIRECT_BATCHED_REFERENCE (per-row streams), potentially a
# richer structure for later kinds (logits tensors for
# TEACHER_FORCED_ARGMAX). Oracles narrow the type locally via
# isinstance / explicit shape assertions.
OracleFn = Callable[
    [Scenario, Any, Any], tuple[bool, str | None, dict[str, Any]]
]

__all__ = [
    "OracleKind",
    "Workload",
    "Scenario",
    "ScenarioResult",
    "SpecConfig",
    "VqbenchXcheckSpec",
    "hf_cache_path_for_repo",
    "OracleFn",
]
