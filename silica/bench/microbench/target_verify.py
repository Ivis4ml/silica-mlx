"""silica.bench.microbench.target_verify — P-6.0.5 sub-unit 7.

Target-verification microbench. For each ``verify_k ∈ {1, 2, 4, 8}``
the harness times **only** the verify forward — a single multi-
position ``forward(model, candidate_arr, cache_list)`` call where
``cache_list`` already holds the primed prefix KV. The prefix
prefill cost is NOT folded into the timed segment; cache isolation
is ensured by rebuilding ``cache_list`` (via
``mlx_cache.make_prompt_cache(model)``) on every rep so successive
verify forwards do not share warm KV state.

The ``k=1`` row is the target single-token baseline. The marginal
verify-k cost is reported as
``verify_k_marginal_ms = forward_ms_p50[k] - forward_ms_p50[k=1]``.
The baseline is looked up by ``verify_k == 1`` rather than the
first row of the run, so reordering ``--verify-ks`` does not
silently corrupt the marginal column.

Why direct ``silica.mlx.runner.forward`` instead of ``Engine.generate``
---------------------------------------------------------------------

``Engine.generate(prompt, max_tokens=1)`` would time prefill +
first-decode over the full ``prefix + candidate`` sequence — that
is a "different-prompt-length warm TTFT" benchmark, not the
target-side verify-k cost curve Decision Gate 1 needs for
speculative ROI estimation. The opening doc (`plans/P6_0_5_OPENING.md`
§3.7) explicitly requires the prefix prefill cost be excluded from
the timed call. Calling ``silica.mlx.runner.forward`` directly with
a freshly built cache list gives us exact control over the timed
segment: prime the prefix with one untimed call, then time only the
second call which processes ``verify_k`` candidate tokens.

The microbench accesses ``adapter._model`` to reach the underlying
mlx-lm model object that ``forward`` consumes. This is private
attribute access by design — every silica adapter
(``qwen3.py`` / ``qwen3_5.py`` / ``qwen3_5_moe.py`` / ``gemma4.py``)
stores the model under that name, and the microbench is the one
caller that needs to bypass the engine's KV-management abstraction
for a clean cache-isolated measurement.

Output
------

Writes one JSONL row per ``verify_k`` plus a Markdown table under
``plans/P6_0_5_BASELINE/target_verify_microbench.{jsonl,md}`` by
default. The JSONL row carries:

  * ``verify_k`` (int) — candidate slice length (k=1 baseline).
  * ``candidate_token_count`` (int) — exact token count in the
    timed forward, equal to ``verify_k`` by construction
    (candidate ids are explicit ``[0] * k``, no tokenizer drift).
  * ``forward_ms_p50`` / ``forward_ms_p95`` (float) — verify-
    forward TTFT statistics over ``--timed-reps`` measurements
    after ``--warmup-reps`` discarded reps.
  * ``peak_memory_mb`` (float) — max device peak across timed reps.
  * ``verify_k_marginal_ms`` (float) — derived at write,
    ``forward_ms_p50 - forward_ms_p50[k=1]``.
  * ``kv_bytes_read_estimate`` / ``weight_bytes_read_estimate``
    (int) — analytic bytes-per-step from ``adapter.config`` /
    ``adapter.kv_layout`` (not measured).
  * ``prefix_token_count`` (int) — actual tokenised length of the
    prefix string under this checkpoint's tokenizer; surfaces
    drift from the ~128-token anchor so downstream readers know
    what bandwidth-bound base cost the verify forward sat on top
    of.
  * ``seed`` (int) — recorded for reproducibility (greedy at
    temperature=0 ignores it but the field stays in the schema).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models import cache as mlx_cache

from silica.mlx.runner import forward as _silica_forward
from silica.models.factory import adapter_for_repo

DEFAULT_REPO = "mlx-community/Qwen3.5-27B-4bit"
DEFAULT_VERIFY_KS: tuple[int, ...] = (1, 2, 4, 8)
DEFAULT_WARMUP_REPS = 2
DEFAULT_TIMED_REPS = 10
MAX_VERIFY_K = 64
DEFAULT_OUT_JSONL = Path(
    "plans/P6_0_5_BASELINE/target_verify_microbench.jsonl"
)
DEFAULT_OUT_MD = Path(
    "plans/P6_0_5_BASELINE/target_verify_microbench.md"
)


# Hand-calibrated to ~128 BPE tokens on the Qwen3 family tokenizer
# (Qwen3-0.6B encodes this to 113 tokens; Qwen3.5-27B-4bit's
# tokenizer drifts within ±15%). The actual encoded length is
# recorded into every JSONL row's ``prefix_token_count`` field so
# downstream readers see the bandwidth-bound base cost the verify
# forward sat on top of, not the design anchor.
_PREFIX_PROMPT = (
    "Memory bandwidth has emerged as the dominant constraint in "
    "single-stream autoregressive decoding for large language "
    "models on consumer hardware platforms. Each decode step must "
    "read the entire active parameter set from unified memory "
    "before any computation can begin. On Apple Silicon the "
    "available bandwidth caps the achievable tokens per second "
    "well below what arithmetic throughput would otherwise allow. "
    "As model parameter counts continue to grow, this bottleneck "
    "becomes more pronounced. Hardware vendors are responding "
    "with wider memory interfaces and dedicated neural acceleration "
    "units, while frameworks such as MLX try to extract every "
    "available cycle through aggressive kernel fusion and tensor "
    "reuse strategies."
)


@dataclass
class _MeasureResult:
    """Per-verify_k measurement bundle."""

    verify_k: int
    samples_ms: list[float]
    peak_memory_mb: float
    prefix_token_count: int
    candidate_token_count: int


@dataclass
class _RunConfig:
    """Resolved CLI arguments for one harness invocation."""

    repo: str
    verify_ks: tuple[int, ...]
    warmup_reps: int
    timed_reps: int
    out_jsonl: Path
    out_md: Path
    seed: int = 0


# ---------------------------------------------------------------------------
# Public entry: pure-function helpers that the tests pin
# ---------------------------------------------------------------------------


def make_candidate_token_ids(verify_k: int) -> list[int]:
    """Return ``[0] * verify_k`` — the exact candidate slice for the timed forward.

    Token id 0 is benign on every Qwen3 / Gemma4 tokenizer (typically
    ``<unk>`` or a control id); the actual content does not affect
    the verify-forward cost (the model still reads all weights and
    runs the same matmul shapes regardless of which input ids it
    receives). Returning a list of exact length ``verify_k`` makes
    the candidate slice deterministic and tokenizer-drift-free —
    contrast with a word-based suffix where ``" one two"`` may
    encode to 2 or 3 tokens depending on the tokenizer's
    leading-space merges.
    """
    if verify_k < 1 or verify_k > MAX_VERIFY_K:
        raise ValueError(
            f"verify_k must be in [1, {MAX_VERIFY_K}]; got {verify_k}"
        )
    return [0] * verify_k


def percentile(samples_ms: list[float], q: float) -> float:
    """Return the ``q``-th percentile of a non-empty sample list.

    Uses linear interpolation between the two nearest ranks so the
    small-N (default 10 reps) p95 is well-defined. ``q`` is in
    [0, 100].
    """
    if not samples_ms:
        raise ValueError("percentile requires a non-empty sample list")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100]; got {q}")
    sorted_samples = sorted(samples_ms)
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = (q / 100.0) * (len(sorted_samples) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = rank - lo
    return sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac


def estimate_weight_bytes_read(adapter: Any) -> int:
    """Analytic bytes-per-step weight footprint.

    Computed as ``num_params × 0.5`` for 4-bit checkpoints (Qwen3.5
    27B-4bit reports ~27e9 params × 0.5 = ~13.5 GB/step, the same
    figure ``plans/P6_OPENING.md`` §1.2 anchors the bandwidth
    ceiling on). The microbench reports this **as an analytic
    estimate**, not a measured number; if the checkpoint is not
    4-bit the figure is wrong by the bits-per-param ratio and the
    field's value should be interpreted with caution.
    """
    cfg = adapter.config
    num_params = getattr(cfg, "num_parameters", None)
    if isinstance(num_params, int) and num_params > 0:
        return num_params // 2
    nl = getattr(cfg, "num_layers", 0)
    h = getattr(cfg, "hidden_size", 0)
    return max(0, nl * h * h * 12 // 2)


def estimate_kv_bytes_read(
    adapter: Any, prefix_token_count: int
) -> int:
    """Analytic per-step KV bytes read at sequence length
    ``prefix_token_count``.

    ``2 × seqlen × num_layers × n_kv_heads × head_dim × dtype_bytes``
    — the factor of 2 accounts for K and V both being read on each
    forward. ``dtype_bytes`` is taken from
    ``adapter.kv_layout().dtype.size`` (typically fp16 → 2 bytes).
    Real KV bandwidth pressure is somewhat lower than this because
    GQA designs amortise reads across grouped query heads, but the
    formula matches the ``plans/P6_OPENING.md`` §1.2 anchor so the
    microbench number is comparable to the rest of the P-6 docs.
    """
    layout = adapter.kv_layout()
    dtype = layout.dtype
    dtype_bytes = getattr(dtype, "size", 2)
    return (
        2
        * int(prefix_token_count)
        * int(layout.num_layers)
        * int(layout.n_kv_heads)
        * int(layout.head_dim)
        * int(dtype_bytes)
    )


def measurement_to_jsonl_row(
    *,
    measurement: _MeasureResult,
    baseline_p50_ms: float,
    weight_bytes_read: int,
    seed: int,
    repo: str,
) -> dict[str, Any]:
    """Convert one ``_MeasureResult`` into the JSONL row schema."""
    p50 = percentile(measurement.samples_ms, 50.0)
    p95 = percentile(measurement.samples_ms, 95.0)
    return {
        "repo": repo,
        "verify_k": measurement.verify_k,
        "candidate_token_count": measurement.candidate_token_count,
        "forward_ms_p50": p50,
        "forward_ms_p95": p95,
        "peak_memory_mb": measurement.peak_memory_mb,
        "verify_k_marginal_ms": p50 - baseline_p50_ms,
        "kv_bytes_read_estimate": 0,  # filled in by caller (needs adapter)
        "weight_bytes_read_estimate": weight_bytes_read,
        "prefix_token_count": measurement.prefix_token_count,
        "seed": seed,
        "n_warmup_reps_discarded": 0,  # filled in by caller
        "n_timed_reps": len(measurement.samples_ms),
    }


def render_markdown_report(
    rows: list[dict[str, Any]], *, repo: str
) -> str:
    """Render the JSONL rows as a human-readable Markdown table."""
    lines: list[str] = []
    lines.append("# Target-Verify Microbench Report")
    lines.append("")
    lines.append(f"- **Repo**: `{repo}`")
    lines.append(f"- **Host**: `{platform.platform()}`")
    lines.append(
        f"- **Timestamp**: `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`"
    )
    lines.append("")
    lines.append(
        "P-6.0.5 sub-unit 7 (D-021 step 3). Each row times one "
        "``forward(model, candidate_arr, cache_list)`` call with "
        "the prefix KV pre-primed and untimed. Cache isolation: "
        "``cache_list`` rebuilt fresh per rep via "
        "``mlx_cache.make_prompt_cache(model)``."
    )
    lines.append("")
    lines.append(
        "| verify_k | cand_tokens | prefix_tokens | p50 (ms) | "
        "p95 (ms) | marginal (ms) | peak (MB) | kv bytes (est) |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for row in rows:
        lines.append(
            f"| {row['verify_k']} "
            f"| {row['candidate_token_count']} "
            f"| {row['prefix_token_count']} "
            f"| {row['forward_ms_p50']:.2f} "
            f"| {row['forward_ms_p95']:.2f} "
            f"| {row['verify_k_marginal_ms']:+.2f} "
            f"| {row['peak_memory_mb']:.1f} "
            f"| {row['kv_bytes_read_estimate']:_d} |"
        )
    lines.append("")
    lines.append(
        "``marginal = forward_ms_p50[k] - forward_ms_p50[k=1]``. "
        "A flat curve (marginal ≈ 0 across k) means the verify "
        "forward is bandwidth-bound on weights and the candidate "
        "slice adds negligible compute; a steep curve means the "
        "candidate slice is in the compute-bound regime and "
        "speculative-decoding ROI is sensitive to drafter "
        "acceptance rate."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Real-model entry: needs Qwen3.5-27B-4bit cached + env opt-in
# ---------------------------------------------------------------------------


def _measure_one_k_real(
    *,
    model: Any,
    prefix_token_ids: list[int],
    verify_k: int,
    warmup_reps: int,
    timed_reps: int,
) -> _MeasureResult:
    """Run warmup + timed verify forwards at one ``verify_k`` value.

    Each rep:

    1. Build a fresh ``cache_list`` via
       ``mlx_cache.make_prompt_cache(model)`` so the verify forward
       starts from cold KV state.
    2. Prime the prefix into the cache with one untimed
       ``forward(model, prefix_arr, cache_list)`` call; force
       evaluation via ``mx.eval(prefix_logits)`` so the prefix
       compute + KV writes complete before the timer starts.
    3. Time the verify forward:
       ``forward(model, candidate_arr, cache_list)`` followed by
       ``mx.eval(verify_logits)``. The candidate array has length
       ``verify_k`` (exact, by construction).
    """
    candidate_ids = make_candidate_token_ids(verify_k)
    prefix_arr = mx.array(prefix_token_ids, dtype=mx.int32)
    candidate_arr = mx.array(candidate_ids, dtype=mx.int32)

    samples_ms: list[float] = []
    peak_max_mb = 0.0
    for rep in range(warmup_reps + timed_reps):
        cache_list = mlx_cache.make_prompt_cache(model)

        # NOT timed: prime the prefix KV. ``mx.eval`` materialises
        # the prefix forward (including its KV writes) before the
        # timer starts, so the timed segment only measures the
        # verify forward over the candidate slice.
        prefix_logits = _silica_forward(model, prefix_arr, cache_list)
        mx.eval(prefix_logits)

        mx.reset_peak_memory()
        t_start = time.perf_counter()
        verify_logits = _silica_forward(model, candidate_arr, cache_list)
        mx.eval(verify_logits)
        t_end = time.perf_counter()
        peak_mb = mx.get_peak_memory() / 1e6

        if rep >= warmup_reps:
            samples_ms.append((t_end - t_start) * 1000.0)
            peak_max_mb = max(peak_max_mb, peak_mb)

        # Drop cache_list reference so the next rep's
        # ``make_prompt_cache`` allocates fresh KV storage rather
        # than reusing this one's buffers.
        del cache_list

    return _MeasureResult(
        verify_k=verify_k,
        samples_ms=samples_ms,
        peak_memory_mb=peak_max_mb,
        prefix_token_count=len(prefix_token_ids),
        candidate_token_count=verify_k,
    )


def run(config: _RunConfig) -> list[dict[str, Any]]:
    """Drive the full microbench against a real model.

    Loads the model once (via ``adapter_for_repo``), reaches into
    ``adapter._model`` for the underlying mlx-lm model object, runs
    warmup + timed verify forwards for every ``verify_k`` in
    ``config.verify_ks``, computes derived fields, and writes JSONL
    + Markdown artefacts. Returns the JSONL rows for downstream
    consumers (REPORT.md builders, debugging).

    Requires ``1`` to be one of the ``verify_ks`` because the
    marginal column is defined as ``p50[k] - p50[k=1]``. Without a
    k=1 measurement the baseline is undefined.
    """
    if 1 not in config.verify_ks:
        raise ValueError(
            "verify_ks must include 1 — the marginal column is "
            "defined relative to k=1 as the target single-token "
            "baseline; without it the marginal is undefined"
        )

    adapter, _ = adapter_for_repo(config.repo)
    model = getattr(adapter, "_model", None)
    if model is None:
        raise RuntimeError(
            f"adapter for {config.repo!r} does not expose ``_model``; "
            "the target-verify microbench reaches into the underlying "
            "mlx-lm model directly to bypass the engine's KV "
            "abstraction (cache isolation requires per-rep "
            "make_prompt_cache). Update the harness if a future "
            "adapter renames the attribute."
        )
    weight_bytes = estimate_weight_bytes_read(adapter)
    tokenizer = adapter.tokenizer()
    prefix_ids = list(tokenizer.encode(_PREFIX_PROMPT))

    measurements: dict[int, _MeasureResult] = {}
    for k in config.verify_ks:
        m = _measure_one_k_real(
            model=model,
            prefix_token_ids=prefix_ids,
            verify_k=k,
            warmup_reps=config.warmup_reps,
            timed_reps=config.timed_reps,
        )
        measurements[k] = m

    baseline_p50 = percentile(measurements[1].samples_ms, 50.0)
    rows: list[dict[str, Any]] = []
    for k in config.verify_ks:
        m = measurements[k]
        row = measurement_to_jsonl_row(
            measurement=m,
            baseline_p50_ms=baseline_p50,
            weight_bytes_read=weight_bytes,
            seed=config.seed,
            repo=config.repo,
        )
        row["kv_bytes_read_estimate"] = estimate_kv_bytes_read(
            adapter, m.prefix_token_count
        )
        row["n_warmup_reps_discarded"] = config.warmup_reps
        rows.append(row)

    write_artefacts(
        rows,
        out_jsonl=config.out_jsonl,
        out_md=config.out_md,
        repo=config.repo,
    )
    return rows


def write_artefacts(
    rows: list[dict[str, Any]],
    *,
    out_jsonl: Path,
    out_md: Path,
    repo: str,
) -> None:
    """Write JSONL + Markdown artefacts to disk.

    Both parent directories are created up front (``mkdir(parents=
    True, exist_ok=True)``) so a partial write cannot leave the
    JSONL on disk while the Markdown sibling fails because its
    directory does not exist. The mkdirs run before either write
    so the failure mode is "neither file written" rather than
    "JSONL written, Markdown missing".
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    out_md.write_text(render_markdown_report(rows, repo=repo))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_verify_ks(arg: str) -> tuple[int, ...]:
    """Parse a comma-separated list of ints; reject empty / non-positive / duplicates.

    Duplicate values (e.g. ``--verify-ks 1,1,2``) are rejected at
    parse time. ``run`` keys measurements by ``verify_k`` in a
    ``dict``, so a duplicate would silently overwrite the prior
    measurement and then the JSONL would emit two rows that look
    independent but actually share one underlying measurement.
    Failing fast at parse time makes the contract loud rather than
    silently corrupting the artefact.
    """
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--verify-ks may not be empty")
    out: list[int] = []
    seen: set[int] = set()
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--verify-ks: '{p}' is not an integer"
            ) from exc
        if v < 1 or v > MAX_VERIFY_K:
            raise argparse.ArgumentTypeError(
                f"--verify-ks: {v} outside [1, {MAX_VERIFY_K}]"
            )
        if v in seen:
            raise argparse.ArgumentTypeError(
                f"--verify-ks: {v} appears more than once; values must be unique"
            )
        seen.add(v)
        out.append(v)
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="silica-bench-microbench-target-verify",
        description=(
            "P-6.0.5 sub-unit 7: target-verify microbench on dense "
            "Qwen3.5-27B-4bit. Times only the verify forward over "
            "k candidate tokens with prefix KV pre-primed; cache "
            "isolation via per-rep ``mlx_cache.make_prompt_cache``. "
            "See plans/P6_0_5_OPENING.md §3.7 for the full contract."
        ),
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--verify-ks",
        type=_parse_verify_ks,
        default=DEFAULT_VERIFY_KS,
        help=(
            "Comma-separated list of verify_k values (default: "
            "1,2,4,8). Must include 1 — the marginal column is "
            "defined relative to k=1."
        ),
    )
    parser.add_argument(
        "--warmup-reps", type=int, default=DEFAULT_WARMUP_REPS
    )
    parser.add_argument(
        "--timed-reps", type=int, default=DEFAULT_TIMED_REPS
    )
    parser.add_argument(
        "--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.warmup_reps < 0:
        parser.error("--warmup-reps must be >= 0")
    if args.timed_reps < 1:
        parser.error("--timed-reps must be >= 1")

    config = _RunConfig(
        repo=args.repo,
        verify_ks=tuple(args.verify_ks),
        warmup_reps=args.warmup_reps,
        timed_reps=args.timed_reps,
        out_jsonl=args.out_jsonl,
        out_md=args.out_md,
        seed=args.seed,
    )
    rows = run(config)
    print(f"wrote {len(rows)} rows to {config.out_jsonl}")
    print(f"wrote markdown report to {config.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
