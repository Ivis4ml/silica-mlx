"""Cycle 23 — batched verify-cost microbench at B={1,4,16,52,64} k={1,4,16,64}.

Cycle 20 measured verify cost at B=1 only. To project tree-spec
throughput at the cycle-14 v10+bf16 KEEP regime (B=52), we need the
verify cost at B=52 as a function of k.

Each cell is one target.forward(x[B, k]) call with warm cache T_kv=128.
Per-cell stats: p50 / p95 over 30 iters with 5 warmups.

Usage:
    SILICA_REAL_QWEN3_5_27B=1 SILICA_USE_BF16_DELTANET_STATE=1 \
        SILICA_USE_FA_DECODE_V10=1 \
        uv run python -m scripts.probe_c5_batched_verify_cost \
            --out plans/P6_C5_DDTREE/batched_verify_cost.jsonl
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
B_VALUES = (1, 4, 16, 52)
K_VALUES = (1, 4, 16, 64)
N_PREFILL_TOKENS = 128
N_WARMUP = 4
N_ITER = 12  # smaller (B=52 is expensive)


def percentile(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="probe_c5_batched_verify_cost")
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

    print(f"# batched verify-cost: target={args.target}")
    t0 = time.perf_counter()
    target, _tok = _mlx_lm_load(args.target)
    print(f"# loaded in {time.perf_counter() - t0:.2f} s")

    results = []
    for B in B_VALUES:
        # Fresh batched cache per B
        cache = make_prompt_cache(target)
        prefill = mx.random.randint(low=0, high=1000, shape=(B, N_PREFILL_TOKENS))
        _ = target(prefill, cache=cache); mx.eval(_)
        for k in K_VALUES:
            x = mx.random.randint(low=0, high=1000, shape=(B, k))
            # Warmup grows cache by k * N_WARMUP
            for _ in range(N_WARMUP):
                out = target(x, cache=cache); mx.eval(out)
            # Measure
            samples = []
            for _ in range(N_ITER):
                t = time.perf_counter_ns()
                out = target(x, cache=cache); mx.eval(out)
                samples.append((time.perf_counter_ns() - t) / 1e6)
            p50 = percentile(samples, 0.5)
            results.append({
                "B": B,
                "k": k,
                "p50_ms": p50,
                "p95_ms": percentile(samples, 0.95),
                "ms_per_token_per_row": p50 / (B * k),
                "n_iter": N_ITER,
            })
            print(f"  B={B:2d} k={k:3d}  p50={p50:7.2f} ms  per-token/row={p50/(B*k):6.3f} ms")

    elapsed_s = time.perf_counter() - t0
    row = {
        "probe_id": "c5_batched_verify_cost",
        "target_repo": args.target,
        "n_prefill_tokens": N_PREFILL_TOKENS,
        "n_warmup_iter": N_WARMUP,
        "n_measure_iter": N_ITER,
        "results": results,
        "elapsed_s": elapsed_s,
    }
    args.out.write_text(json.dumps(row) + "\n", encoding="utf-8")

    print()
    print("# Tree-spec projection at B=52, depth-2 balanced tree (b=8 per level, 64 leaves), p_match=0.40:")
    # Find B=52 k=64 row
    row52 = next((r for r in results if r["B"] == 52 and r["k"] == 64), None)
    row1 = next((r for r in results if r["B"] == 1 and r["k"] == 1), None)
    if row52 and row1:
        verify_ms = row52["p50_ms"]
        # drafter at B=52 estimated ~150ms × depth=2 = 300ms per cycle
        drafter_ms = 150 * 2
        cycle_ms = verify_ms + drafter_ms
        expected_tokens = 1 + 0.4 + 0.16  # 1.56
        per_row_throughput = expected_tokens / (cycle_ms / 1000)
        agg_throughput = per_row_throughput * 52
        print(f"  verify B=52 k=64: {verify_ms:.0f} ms")
        print(f"  drafter (est): 300 ms")
        print(f"  cycle: {cycle_ms:.0f} ms")
        print(f"  expected tokens/cycle: {expected_tokens}")
        print(f"  per-row tok/s: {per_row_throughput:.2f}")
        print(f"  aggregate tok/s at B=52: {agg_throughput:.0f}")

    print(f"\n# wrote {args.out}; elapsed {elapsed_s:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
