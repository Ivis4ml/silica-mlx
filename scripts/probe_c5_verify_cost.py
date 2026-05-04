"""Cycle 20 — verify-cost microbench at k ∈ {1, 4, 8, 16, 32, 64, 128}.

Spec-decode tree-spec uses one target forward to verify k draft candidates.
If the target's forward cost per step is sub-linear in k (ie. constant
overhead amortizes), tree-spec at large b becomes viable. If it scales
linearly, the per-token cost stays constant and tree-spec doesn't beat
linear spec on throughput.

This probe just measures `target_model(x)` wall-clock as a function of
input token count k, with the rest of the spec machinery stripped away.

Usage:
    SILICA_REAL_QWEN3_5_27B=1 \
        uv run python -m scripts.probe_c5_verify_cost \
            --out plans/P6_C5_DDTREE/verify_cost_probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_TARGET = "mlx-community/Qwen3.5-27B-4bit"
DEFAULT_GATE = "SILICA_REAL_QWEN3_5_27B"
K_VALUES = (1, 4, 8, 16, 32, 64, 128)
N_PREFILL_TOKENS = 128  # warm cache to a representative T_kv
N_WARMUP = 5
N_ITER = 30


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_c5_verify_cost")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--target-gate-env", default=DEFAULT_GATE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if os.environ.get(args.target_gate_env) != "1":
        print(f"error: {args.target_gate_env}=1 required", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load as _mlx_lm_load

    print(f"# verify-cost probe: target={args.target}")
    t0 = time.perf_counter()
    target, _ = _mlx_lm_load(args.target)  # type: ignore[misc]
    print(f"# loaded in {time.perf_counter() - t0:.2f} s")

    # Build prefill cache.
    cache = make_prompt_cache(target)
    mx.random.seed(0)
    prefill = mx.random.randint(low=0, high=1000, shape=(1, N_PREFILL_TOKENS))
    _ = target(prefill, cache=cache)
    mx.eval(_)
    print(f"# prefilled {N_PREFILL_TOKENS} tokens; T_kv={N_PREFILL_TOKENS}")

    results: list[dict[str, Any]] = []
    for k in K_VALUES:
        # Fresh cache per k-value to keep T_kv constant across iterations.
        # This costs one prefill per k but makes the measurement clean.
        fresh_cache = make_prompt_cache(target)
        _ = target(prefill, cache=fresh_cache)
        mx.eval(_)

        x = mx.random.randint(low=0, high=1000, shape=(1, k))

        # Warmup (cache grows by k*N_WARMUP)
        for _ in range(N_WARMUP):
            out = target(x, cache=fresh_cache)
            mx.eval(out)

        # Measure (cache grows by k*N_ITER additional)
        samples = []
        for _ in range(N_ITER):
            t = time.perf_counter_ns()
            out = target(x, cache=fresh_cache)
            mx.eval(out)
            samples.append((time.perf_counter_ns() - t) / 1e6)

        p50 = percentile(samples, 0.5)
        p95 = percentile(samples, 0.95)
        results.append({
            "k": k,
            "p50_ms": p50,
            "p95_ms": p95,
            "mean_ms": statistics.fmean(samples),
            "ms_per_token": p50 / k,
            "n_iter": N_ITER,
        })
        print(f"  k={k:3d}  p50={p50:6.2f} ms  per-token={p50/k:5.2f} ms  vs k=1: {p50/results[0]['p50_ms']:.2f}x")

    elapsed_s = time.perf_counter() - t0
    row = {
        "probe_id": "c5_verify_cost",
        "target_repo": args.target,
        "n_prefill_tokens": N_PREFILL_TOKENS,
        "n_warmup_iter": N_WARMUP,
        "n_measure_iter": N_ITER,
        "results": results,
        "elapsed_s": elapsed_s,
    }
    args.out.write_text(json.dumps(row) + "\n", encoding="utf-8")

    print()
    print("# verify-cost slope analysis:")
    p50_k1 = results[0]["p50_ms"]
    for r in results:
        ratio = r["p50_ms"] / p50_k1
        ideal_lin = r["k"]
        sub_lin = ratio / ideal_lin
        print(f"  k={r['k']:<3d}  p50={r['p50_ms']:6.2f} ms  ratio_vs_k1={ratio:5.2f}x  k_factor={ideal_lin:<3d}  sub_linearity={sub_lin:.3f}")

    print(f"\n# wrote {args.out}; elapsed {elapsed_s:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
