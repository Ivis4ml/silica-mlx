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
    "hf_cache_path_for_repo",
    "OracleFn",
]
