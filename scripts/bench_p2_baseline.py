"""P-2 baseline benchmark on Qwen3-0.6B (default) over 4 scenarios.

This is an ad-hoc baseline script, not the P-4 unified harness. Scope:

  S1  single-request, short-in long-out
  S2  single-request, long-in short-out
  S3  8-way continuous batching
  S4  shared-prefix cache on vs off (4 requests sharing a ~64-tok prefix)

Per scenario: 1 warmup run + N timed runs (default 3); median reported.
Per run: one JSONL record written to bench/results/. A markdown summary
is printed to stdout at the end.

Peak device memory is captured via ``mlx.core.{reset,get}_peak_memory``
around each run so S3/S4 (which do not route through ``Engine.metrics``)
still produce a comparable memory signal.

Usage:
  uv run python scripts/bench_p2_baseline.py
  uv run python scripts/bench_p2_baseline.py --repo Qwen/Qwen3-0.6B --runs 3
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402

from silica import Engine  # noqa: E402
from silica.core.sampler import Sampler  # noqa: E402
from silica.core.sampling import SamplingParams  # noqa: E402
from silica.kvcache.prefix import RadixPrefixCache  # noqa: E402
from silica.kvcache.simple import SimpleKVCache  # noqa: E402
from silica.kvcache.store import SyntheticPrefixBlockStore  # noqa: E402
from silica.models.factory import adapter_for_repo  # noqa: E402
from silica.scheduler.batcher import ContinuousBatcher  # noqa: E402

DEFAULT_REPO = "Qwen/Qwen3-0.6B"
DEFAULT_BLOCK_SIZE = 16

SHORT_PROMPT = "The capital of France is"

LONG_PROMPT_SEED = (
    "The following is a detailed historical passage. "
    "In the year 1789 a series of political events unfolded in Europe "
    "that would reshape the continent for the next two centuries. "
    "Philosophers of the age debated the meaning of liberty, the structure "
    "of legitimate government, and the rights of the individual. "
    "Economic transformation, driven by technological innovation, proceeded "
    "in parallel with political change. "
)

BATCH_PROMPTS_8 = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of Brazil is",
    "The capital of Canada is",
    "The capital of Australia is",
]

PREFIX_SHARED = (
    "You are a careful assistant. Answer each question in a single short "
    "sentence. Be precise and do not speculate beyond the evidence. "
    "Context: the following questions concern basic world knowledge.\n\n"
    "Question: "
)
PREFIX_SUFFIXES = [
    "What is the capital of France?",
    "What is two plus two?",
    "Name a primary color.",
    "What day comes after Monday?",
]


@dataclass
class RunRecord:
    scenario: str
    repo: str
    run_index: int
    warmup: bool
    wall_time_s: float
    peak_memory_mb: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSummary:
    scenario: str
    description: str
    runs: list[RunRecord]

    def median(self, field_path: str) -> float | None:
        raw = [_get_nested(r, field_path) for r in self.runs if not r.warmup]
        values: list[float] = [v for v in raw if v is not None]
        if not values:
            return None
        return float(statistics.median(values))


def _get_nested(rec: RunRecord, field_path: str) -> float | None:
    parts = field_path.split(".")
    obj: Any = rec
    for p in parts:
        if isinstance(obj, dict):
            obj = obj.get(p)
        else:
            obj = getattr(obj, p, None)
        if obj is None:
            return None
    return float(obj) if isinstance(obj, (int, float)) else None


def _long_prompt(tokenizer: Any, target_tokens: int) -> str:
    text = ""
    while len(tokenizer.encode(text)) < target_tokens:
        text += LONG_PROMPT_SEED
    return text


def _eos_ids(tokenizer: Any) -> tuple[int, ...]:
    return tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))


def _fresh_peak_memory_mb() -> float:
    mx.reset_peak_memory()
    return 0.0


def _read_peak_memory_mb() -> float:
    return mx.get_peak_memory() / 1e6


def _build_engine(repo: str) -> tuple[Engine, Any, Any]:
    adapter, kv = adapter_for_repo(repo)
    engine = Engine(adapter, kv)
    # Reach into the adapter to keep a reference to the underlying
    # mlx-lm model. We need this to rebuild fresh SimpleKVCache
    # instances between timed runs without reloading weights — see
    # ``_fresh_single_request_engine``. ``adapter.build(None)`` would
    # work too (Qwen3Adapter.build returns ``self._model`` and drops
    # its argument), but the attribute access is more honest about
    # what we're doing.
    model = adapter._model  # type: ignore[attr-defined]
    return engine, adapter, model


def _fresh_single_request_engine(adapter: Any, model: Any) -> Engine:
    """Rebuild a fresh SimpleKVCache + Engine for the next timed run.

    SimpleKVCache.free only nulls the owner; the underlying mlx-lm
    cache list keeps growing across generate() calls. Without this
    helper the per-run ``resident_mb`` / ``peak_memory_mb`` numbers
    accumulate KV state from earlier runs and are not comparable.

    The adapter's ``_kv_manager`` is swapped in place so ``prefill`` /
    ``decode_step`` pick up the fresh cache list. The mlx-lm model
    itself (weights) is reused to avoid paying a reload cost per run.
    """
    fresh_kv = SimpleKVCache.from_model(model)
    adapter._kv_manager = fresh_kv
    return Engine(adapter, fresh_kv)


def _run_single(
    engine: Engine,
    adapter: Any,
    prompt: str,
    params: SamplingParams,
) -> tuple[int, dict[str, Any]]:
    _fresh_peak_memory_mb()
    t0 = time.perf_counter()
    token_ids = list(engine.generate(prompt, params))
    wall = time.perf_counter() - t0
    snapshot = engine.metrics.snapshot()
    return len(token_ids), {
        "wall_time_s": wall,
        "peak_memory_mb": _read_peak_memory_mb(),
        "n_output_tokens": len(token_ids),
        "ttft_ms": snapshot.ttft_ms,
        "prefill_tok_s": snapshot.prefill_tok_s,
        "decode_tok_s": snapshot.decode_tok_s,
        "resident_mb": snapshot.resident_mb,
        "logical_kv_bytes": snapshot.logical_kv_bytes,
    }


def _scenario_single(
    *,
    label: str,
    description: str,
    adapter: Any,
    model: Any,
    prompt: str,
    params: SamplingParams,
    repo: str,
    warmup: int,
    runs: int,
) -> ScenarioSummary:
    records: list[RunRecord] = []
    for i in range(warmup + runs):
        is_warm = i < warmup
        # Release MLX allocator pools and rebuild a fresh KV/Engine
        # so this run's resident/peak memory is not polluted by the
        # previous run's accumulated cache.
        mx.clear_cache()
        engine = _fresh_single_request_engine(adapter, model)
        _, data = _run_single(engine, adapter, prompt, params)
        records.append(
            RunRecord(
                scenario=label,
                repo=repo,
                run_index=i,
                warmup=is_warm,
                wall_time_s=data["wall_time_s"],
                peak_memory_mb=data["peak_memory_mb"],
                extra={
                    k: v
                    for k, v in data.items()
                    if k not in ("wall_time_s", "peak_memory_mb")
                },
            )
        )
    return ScenarioSummary(scenario=label, description=description, runs=records)


def _run_generate_batch(
    engine: Engine,
    prompts: list[str],
    params: SamplingParams,
    *,
    max_batch_size: int,
    prefix_cache: RadixPrefixCache | None,
) -> tuple[int, dict[int, float]]:
    ttft_map: dict[int, float] = {}
    total_tokens = 0
    t0 = time.perf_counter()
    for event in engine.generate_batch(
        prompts,
        params,
        max_batch_size=max_batch_size,
        prefix_cache=prefix_cache,
    ):
        if event.kind == "token":
            if event.req_index not in ttft_map:
                ttft_map[event.req_index] = time.perf_counter() - t0
            total_tokens += 1
    return total_tokens, ttft_map


def _scenario_batch_8way(
    *,
    engine: Engine,
    params: SamplingParams,
    repo: str,
    warmup: int,
    runs: int,
) -> ScenarioSummary:
    records: list[RunRecord] = []
    label = "S3_batch_8way"
    desc = "8 concurrent prompts, max_tokens=64, no prefix cache"
    for i in range(warmup + runs):
        is_warm = i < warmup
        mx.clear_cache()
        _fresh_peak_memory_mb()
        t0 = time.perf_counter()
        total_tokens, ttft_map = _run_generate_batch(
            engine,
            BATCH_PROMPTS_8,
            params,
            max_batch_size=8,
            prefix_cache=None,
        )
        wall = time.perf_counter() - t0
        peak = _read_peak_memory_mb()
        aggregate_tps = total_tokens / wall if wall > 0 else 0.0
        ttft_median = (
            statistics.median(ttft_map.values()) * 1000.0 if ttft_map else None
        )
        records.append(
            RunRecord(
                scenario=label,
                repo=repo,
                run_index=i,
                warmup=is_warm,
                wall_time_s=wall,
                peak_memory_mb=peak,
                extra={
                    "n_requests": len(BATCH_PROMPTS_8),
                    "total_output_tokens": total_tokens,
                    "aggregate_tok_s": aggregate_tps,
                    "ttft_median_ms": ttft_median,
                },
            )
        )
    return ScenarioSummary(scenario=label, description=desc, runs=records)


def _run_shared_prefix_once(
    adapter: Any,
    sampler: Sampler,
    prompts_ids: list[list[int]],
    params: SamplingParams,
    *,
    prefix_cache: RadixPrefixCache | None,
    max_batch_size: int,
) -> tuple[int, float, int, int]:
    """Drive ContinuousBatcher directly so we can read its counters.

    Admission pattern mirrors Engine.generate_batch: the first
    ``max_batch_size`` prompts admit pre-step as the initial cohort
    (``_prepare_cohort`` path, no prefix-hit check — this is the batched-
    prefill invariant); the remainder enter ``_waiting_queue`` and the
    admit phase consults ``prefix_cache`` on each Phase A pop.

    Returns (total_tokens, wall_s, forward_prompt_tokens, prefix_hits).
    """
    batcher = ContinuousBatcher(
        adapter,
        sampler=sampler,
        max_batch_size=max_batch_size,
        prefix_cache=prefix_cache,
    )
    initial = prompts_ids[:max_batch_size]
    remainder = prompts_ids[max_batch_size:]
    for req_index, prompt_ids in enumerate(initial):
        batcher.add_request(req_index, prompt_ids, params)

    total_tokens = 0
    t0 = time.perf_counter()
    if remainder:
        for event in batcher.step():
            if event.kind == "token":
                total_tokens += 1
        base_index = len(initial)
        for offset, prompt_ids in enumerate(remainder):
            batcher.add_request(base_index + offset, prompt_ids, params)
    while batcher.has_work():
        for event in batcher.step():
            if event.kind == "token":
                total_tokens += 1
    wall = time.perf_counter() - t0
    return (
        total_tokens,
        wall,
        batcher.forward_prompt_tokens,
        batcher.prefix_hits,
    )


def _scenario_prefix_on_off(
    *,
    engine: Engine,
    adapter: Any,
    sampler: Sampler,
    params: SamplingParams,
    repo: str,
    warmup: int,
    runs: int,
    block_size: int,
) -> ScenarioSummary:
    tokenizer = adapter.tokenizer()
    prompts_ids = [
        list(tokenizer.encode(PREFIX_SHARED + suffix)) for suffix in PREFIX_SUFFIXES
    ]
    records: list[RunRecord] = []
    label = "S4_shared_prefix"
    desc = (
        f"4 prompts sharing ~{len(tokenizer.encode(PREFIX_SHARED))}-tok prefix, "
        "max_batch_size=2 (2 pre-step, 2 via waiting queue), on vs off"
    )
    for mode in ("off", "on"):
        # Persistent across warmup + timed runs for this mode: the
        # warmup run populates the cache via Phase A inserts, timed
        # runs then exercise hits. Without this the "on" branch is
        # effectively a fresh cache per run and never hits.
        prefix_cache: RadixPrefixCache | None = None
        if mode == "on":
            prefix_cache = RadixPrefixCache(
                block_size=block_size,
                store=SyntheticPrefixBlockStore(block_size=block_size),
            )
        for i in range(warmup + runs):
            is_warm = i < warmup
            mx.clear_cache()
            _fresh_peak_memory_mb()
            total_tokens, wall, fwd_tokens, hits = _run_shared_prefix_once(
                adapter,
                sampler,
                prompts_ids,
                params,
                prefix_cache=prefix_cache,
                max_batch_size=2,
            )
            peak = _read_peak_memory_mb()
            records.append(
                RunRecord(
                    scenario=f"{label}_{mode}",
                    repo=repo,
                    run_index=i,
                    warmup=is_warm,
                    wall_time_s=wall,
                    peak_memory_mb=peak,
                    extra={
                        "n_requests": len(prompts_ids),
                        "total_output_tokens": total_tokens,
                        "aggregate_tok_s": total_tokens / wall
                        if wall > 0
                        else 0.0,
                        "forward_prompt_tokens": fwd_tokens,
                        "prefix_hits": hits,
                        "mode": mode,
                    },
                )
            )
    return ScenarioSummary(scenario=label, description=desc, runs=records)


def _write_jsonl(path: Path, summaries: Iterable[ScenarioSummary]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for summary in summaries:
            for rec in summary.runs:
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def _fmt(v: float | None, spec: str = ".1f") -> str:
    return format(v, spec) if v is not None else "—"


def _render_markdown(
    summaries: list[ScenarioSummary],
    repo: str,
    runs: int,
    warmup: int,
) -> str:
    lines: list[str] = []
    lines.append(f"# P-2 Baseline — {repo}")
    lines.append("")
    lines.append(f"- Runs: {runs} timed + {warmup} warmup")
    lines.append(f"- Host: {platform.platform()}")
    lines.append(f"- Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    s1 = _find(summaries, "S1_single_short_long")
    s2 = _find(summaries, "S2_single_long_short")
    s3 = _find(summaries, "S3_batch_8way")

    if s1 or s2:
        lines.append("## Single-request scenarios")
        lines.append("")
        lines.append(
            "| Scenario | wall_time_s | ttft_ms | prefill_tok_s | "
            "decode_tok_s | resident_mb | peak_memory_mb |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for s in (s1, s2):
            if s is None:
                continue
            lines.append(
                "| {name} | {wall} | {ttft} | {pfl} | {dec} | {res} | {peak} |".format(
                    name=s.scenario,
                    wall=_fmt(s.median("wall_time_s"), ".3f"),
                    ttft=_fmt(s.median("extra.ttft_ms"), ".1f"),
                    pfl=_fmt(s.median("extra.prefill_tok_s"), ".1f"),
                    dec=_fmt(s.median("extra.decode_tok_s"), ".1f"),
                    res=_fmt(s.median("extra.resident_mb"), ".1f"),
                    peak=_fmt(s.median("peak_memory_mb"), ".1f"),
                )
            )
        lines.append("")

    if s3:
        lines.append("## Batched scenario")
        lines.append("")
        lines.append(
            "| Scenario | wall_time_s | aggregate_tok_s | ttft_median_ms | "
            "peak_memory_mb |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        lines.append(
            "| {name} | {wall} | {tps} | {ttft} | {peak} |".format(
                name=s3.scenario,
                wall=_fmt(s3.median("wall_time_s"), ".3f"),
                tps=_fmt(s3.median("extra.aggregate_tok_s"), ".1f"),
                ttft=_fmt(s3.median("extra.ttft_median_ms"), ".1f"),
                peak=_fmt(s3.median("peak_memory_mb"), ".1f"),
            )
        )
        lines.append("")

    s4_off = _find(summaries, "S4_shared_prefix_off")
    s4_on = _find(summaries, "S4_shared_prefix_on")
    if s4_off and s4_on:
        lines.append("## Shared-prefix scenario (S4)")
        lines.append("")
        lines.append(
            "| Mode | wall_time_s | aggregate_tok_s | forward_prompt_tokens | "
            "prefix_hits | peak_memory_mb |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for s in (s4_off, s4_on):
            lines.append(
                "| {name} | {wall} | {tps} | {fwd} | {hits} | {peak} |".format(
                    name=s.scenario,
                    wall=_fmt(s.median("wall_time_s"), ".3f"),
                    tps=_fmt(s.median("extra.aggregate_tok_s"), ".1f"),
                    fwd=_fmt(s.median("extra.forward_prompt_tokens"), ".0f"),
                    hits=_fmt(s.median("extra.prefix_hits"), ".0f"),
                    peak=_fmt(s.median("peak_memory_mb"), ".1f"),
                )
            )
        off_wall = s4_off.median("wall_time_s") or 0.0
        on_wall = s4_on.median("wall_time_s") or 0.0
        if off_wall > 0:
            delta_pct = (off_wall - on_wall) / off_wall * 100.0
            lines.append("")
            lines.append(
                f"Prefix cache reduces wall-time by **{delta_pct:.1f}%** "
                f"({off_wall:.3f}s → {on_wall:.3f}s)."
            )
        lines.append("")

    return "\n".join(lines)


def _find(
    summaries: list[ScenarioSummary], scenario: str
) -> ScenarioSummary | None:
    # Match either exact scenario label or scenario prefix (S4 splits on/off).
    exact = [s for s in summaries if s.scenario == scenario]
    if exact:
        return exact[0]
    # S4 uses run-level scenario labels; reconstruct summary from records.
    matches: list[RunRecord] = []
    for s in summaries:
        matches.extend(r for r in s.runs if r.scenario == scenario)
    if not matches:
        return None
    return ScenarioSummary(
        scenario=scenario, description=scenario, runs=matches
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--out-dir", default="bench/results")
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated scenario ids to skip (S1,S2,S3,S4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    engine, adapter, model = _build_engine(args.repo)
    tokenizer = adapter.tokenizer()
    eos_ids = _eos_ids(tokenizer)

    params_long_out = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop_token_ids=eos_ids,
    )
    params_short_out = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        stop_token_ids=eos_ids,
    )
    params_batch = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        stop_token_ids=eos_ids,
    )

    summaries: list[ScenarioSummary] = []

    if "S1" not in skip:
        summaries.append(
            _scenario_single(
                label="S1_single_short_long",
                description="single request, ~16-tok prompt, max_tokens=256",
                adapter=adapter,
                model=model,
                prompt=SHORT_PROMPT,
                params=params_long_out,
                repo=args.repo,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    if "S2" not in skip:
        long_prompt = _long_prompt(tokenizer, target_tokens=512)
        summaries.append(
            _scenario_single(
                label="S2_single_long_short",
                description="single request, ~512-tok prompt, max_tokens=32",
                adapter=adapter,
                model=model,
                prompt=long_prompt,
                params=params_short_out,
                repo=args.repo,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    if "S3" not in skip:
        summaries.append(
            _scenario_batch_8way(
                engine=engine,
                params=params_batch,
                repo=args.repo,
                warmup=args.warmup,
                runs=args.runs,
            )
        )

    if "S4" not in skip:
        summaries.append(
            _scenario_prefix_on_off(
                engine=engine,
                adapter=adapter,
                sampler=Sampler(),
                params=params_batch,
                repo=args.repo,
                warmup=args.warmup,
                runs=args.runs,
                block_size=args.block_size,
            )
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = out_dir / f"{stamp}_p2_baseline.jsonl"
    md_path = out_dir / f"{stamp}_p2_baseline.md"

    _write_jsonl(jsonl_path, summaries)
    markdown = _render_markdown(
        summaries, repo=args.repo, runs=args.runs, warmup=args.warmup
    )
    md_path.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print()
    print(f"Artifacts:\n  {jsonl_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
