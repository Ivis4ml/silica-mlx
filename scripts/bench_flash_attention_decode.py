"""Microbench for silica.kernels.flash_attention_decode vs mlx SDPA.

Production shape (Qwen3.5-27B-4bit decode):
    Q: (B=48, H_q=24, T_q=1, D=256)  fp16
    K: (B=48, H_kv=4, T_kv, D=256)
    V: (B=48, H_kv=4, T_kv, D=256)
    gate: (B=48, H_q=24, T_q=1, D=256)

Sweep T_kv across 128, 256, 512, 1024 to see how amortisation changes.

Usage:
    uv run --extra bench python scripts/bench_flash_attention_decode.py
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from silica.kernels.flash_attention_decode import (
    flash_attention_decode,
    reference_flash_attention_decode,
)


def maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[k]


def time_call(fn, n_warmup: int = 8, n_iter: int = 64) -> tuple[float, float]:
    for _ in range(n_warmup):
        out = fn()
        mx.eval(out)
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        out = fn()
        mx.eval(out)
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return percentile(samples, 0.5), percentile(samples, 0.95)


def run_case(B: int, H_q: int, H_kv: int, T_kv: int, D: int, *, gated: bool, seed: int = 0) -> dict:
    mx.random.seed(seed)
    q = (mx.random.normal((B, H_q, 1, D)) * 0.1).astype(mx.float16)
    k = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
    v = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
    gate = (mx.random.normal((B, H_q, 1, D)) * 0.5).astype(mx.float16) if gated else None
    mx.eval(q, k, v)
    if gate is not None:
        mx.eval(gate)

    # Reference path: mlx SDPA + (optional) sigmoid * gate as un-fused ops.
    out_ref = reference_flash_attention_decode(q, k, v, gate=gate)
    out_si = flash_attention_decode(q, k, v, gate=gate)
    mx.eval(out_ref, out_si)
    err = maxabs(out_si, out_ref)

    p50_si, p95_si = time_call(lambda: flash_attention_decode(q, k, v, gate=gate))
    p50_ref, p95_ref = time_call(lambda: reference_flash_attention_decode(q, k, v, gate=gate))

    return {
        "B": B,
        "H_q": H_q,
        "H_kv": H_kv,
        "T_kv": T_kv,
        "D": D,
        "gated": gated,
        "err_maxabs": err,
        "p50_silica_ms": p50_si,
        "p95_silica_ms": p95_si,
        "p50_mlx_ms": p50_ref,
        "p95_mlx_ms": p95_ref,
        "ratio_p50": p50_si / p50_ref,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="prod", help="prod | sweep")
    args = ap.parse_args()

    if args.shapes == "prod":
        shapes = [(48, 24, 4, 128, 256), (48, 24, 4, 256, 256), (48, 24, 4, 512, 256)]
    else:
        shapes = [
            (4, 24, 4, 128, 256), (8, 24, 4, 128, 256), (16, 24, 4, 128, 256),
            (32, 24, 4, 128, 256), (48, 24, 4, 128, 256),
            (48, 24, 4, 256, 256), (48, 24, 4, 512, 256), (48, 24, 4, 1024, 256),
        ]

    print(f"{'shape':<28} {'gate':<5} {'err':<10} {'silica_p50':<12} {'mlx_p50':<12} {'ratio':<8}")
    print("-" * 80)
    for shape in shapes:
        for gated in (False, True):
            r = run_case(*shape, gated=gated)
            shape_str = f"B{r['B']}H{r['H_q']}/{r['H_kv']}T{r['T_kv']}D{r['D']}"
            print(
                f"{shape_str:<28} {str(r['gated']):<5} "
                f"{r['err_maxabs']:<10.2e} "
                f"{r['p50_silica_ms']:<12.4f} "
                f"{r['p50_mlx_ms']:<12.4f} "
                f"{r['ratio_p50']:<8.3f}"
            )


if __name__ == "__main__":
    main()
