"""Cycle 19 — extended top-b coverage probe for the spec-decode research loop.

Builds on `scripts/probe_c5_top_b_coverage.py` but extends b-values to
{1, 4, 8, 16, 32, 64, 128, 256, 512, 1000} and adds rank statistics
(mean, median, p25, p75, p95) to characterize the full rank distribution
of the target argmax in the drafter top-K.

Goal: understand whether tree-spec at b≥64 can push coverage above the
0.40 line that the user wants for >40% accept rate.

Usage:
    SILICA_REAL_QWEN3_5_27B=1 SILICA_REAL_QWEN3_5_0_8B_DRAFT=1 \
        uv run python -m scripts.probe_c5_extended_coverage \
            --out plans/P6_C5_DDTREE/extended_coverage_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Inherit defaults / helpers from the canonical probe.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.probe_c5_top_b_coverage import (
    DEFAULT_DRAFTER_REPO,
    DEFAULT_GATE_DRAFTER,
    DEFAULT_GATE_TARGET,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_SAMPLE_TEXTS,
    DEFAULT_SAMPLE_TEXT_CHARS,
    DEFAULT_TARGET_REPO,
    DEFAULT_WIKITEXT_PATH,
    _gate_check,
    _slice_sample_texts,
    compute_ranks,
    coverage_at,
    rank_histogram,
    tokenizer_attestation,
)

EXT_B_VALUES = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1000)
EXT_HISTOGRAM_BUCKETS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1000)


def rank_stats(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {}
    sorted_r = sorted(ranks)
    n = len(sorted_r)
    return {
        "n": n,
        "mean_rank": statistics.fmean(sorted_r),
        "median_rank": statistics.median(sorted_r),
        "p25_rank": sorted_r[max(0, int(0.25 * (n - 1)))],
        "p75_rank": sorted_r[min(n - 1, int(0.75 * (n - 1)))],
        "p95_rank": sorted_r[min(n - 1, int(0.95 * (n - 1)))],
        "p99_rank": sorted_r[min(n - 1, int(0.99 * (n - 1)))],
        "max_rank": sorted_r[-1],
        "min_rank": sorted_r[0],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_c5_extended_coverage")
    parser.add_argument("--target", default=DEFAULT_TARGET_REPO)
    parser.add_argument("--drafter", default=DEFAULT_DRAFTER_REPO)
    parser.add_argument("--target-gate-env", default=DEFAULT_GATE_TARGET)
    parser.add_argument("--drafter-gate-env", default=DEFAULT_GATE_DRAFTER)
    parser.add_argument("--corpus", default=str(DEFAULT_WIKITEXT_PATH))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--n-sample-texts", type=int, default=DEFAULT_N_SAMPLE_TEXTS)
    parser.add_argument("--sample-text-chars", type=int, default=DEFAULT_SAMPLE_TEXT_CHARS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    _gate_check(args.target_gate_env)
    _gate_check(args.drafter_gate_env)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load as _mlx_lm_load
    from silica.bench.wikitext import load_wikitext_text, tokenize_for_ppl
    from silica.mlx.runner import forward_full

    print(f"# extended coverage probe: target={args.target} drafter={args.drafter}")

    text = load_wikitext_text(args.corpus)
    sample_texts = _slice_sample_texts(text, args.n_sample_texts, args.sample_text_chars)

    t0 = time.perf_counter()
    target_model, target_tokenizer = _mlx_lm_load(args.target)  # type: ignore[misc]
    print(f"# target loaded in {time.perf_counter() - t0:.2f} s")

    t1 = time.perf_counter()
    drafter_model, drafter_tokenizer = _mlx_lm_load(args.drafter)  # type: ignore[misc]
    print(f"# drafter loaded in {time.perf_counter() - t1:.2f} s")

    attestation = tokenizer_attestation(target_tokenizer, drafter_tokenizer, sample_texts)
    print(f"# tokenizer alignment OK: vocab={attestation['vocab_size']}")

    tokens_2d = tokenize_for_ppl(target_tokenizer, text, max_tokens=args.max_tokens)
    tokens_1d = tokens_2d.reshape(-1)
    n_corpus_tokens = int(tokens_1d.shape[0])
    print(f"# corpus tokens: {n_corpus_tokens}")

    target_cache = make_prompt_cache(target_model)
    drafter_cache = make_prompt_cache(drafter_model)

    t2 = time.perf_counter()
    target_logits = forward_full(target_model, tokens_1d, target_cache)
    mx.eval(target_logits)
    print(f"# target forward_full: {time.perf_counter() - t2:.2f} s")

    t3 = time.perf_counter()
    drafter_logits = forward_full(drafter_model, tokens_1d, drafter_cache)
    mx.eval(drafter_logits)
    print(f"# drafter forward_full: {time.perf_counter() - t3:.2f} s")

    target_argmax = mx.argmax(target_logits, axis=-1)
    max_b = max(EXT_B_VALUES)
    # Sort drafter logits descending; take top max_b.
    sort_order_full = mx.argsort(-drafter_logits, axis=-1)
    drafter_top_ids = sort_order_full[:, :max_b]
    mx.eval(target_argmax, drafter_top_ids)

    target_argmax_py = [int(x) for x in target_argmax.tolist()]
    drafter_top_py = [[int(x) for x in row] for row in drafter_top_ids.tolist()]

    target_argmax_for_rank = target_argmax_py[:-1]
    drafter_top_for_rank = drafter_top_py[1:]

    ranks = compute_ranks(target_argmax_for_rank, drafter_top_for_rank, max_b=max_b)
    coverage = coverage_at(ranks, EXT_B_VALUES)
    histogram = rank_histogram(ranks, EXT_HISTOGRAM_BUCKETS)
    stats = rank_stats(ranks)

    elapsed_s = time.perf_counter() - t0

    row: dict[str, Any] = {
        "probe_id": "c5_extended_coverage",
        "target_repo": args.target,
        "drafter_repo": args.drafter,
        "n_positions_scored": len(ranks),
        "max_b_measured": int(max_b),
        "b_values": list(EXT_B_VALUES),
        "coverage_at": {str(k): float(v) for k, v in coverage.items()},
        "rank_histogram": histogram,
        "rank_stats": stats,
        "tokenizer_attestation": attestation,
        "elapsed_s": float(elapsed_s),
    }
    args.out.write_text(json.dumps(row) + "\n", encoding="utf-8")

    print()
    print("# coverage@b (extended):")
    for b in EXT_B_VALUES:
        c = coverage[b]
        bar = "#" * int(c * 50)
        print(f"  @{b:<5d}  {c:.4f}  {bar}")
    print()
    print("# rank statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:>14s}: {v:.2f}")
        else:
            print(f"  {k:>14s}: {v}")
    print()
    print(f"# wrote 1 row to {args.out}; elapsed {elapsed_s:.2f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
