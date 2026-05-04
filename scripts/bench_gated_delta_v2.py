"""Cycle 31 — microbench silica gated_delta_v2 vs mlx_lm gated_delta_kernel.

Production shape (Qwen3.5-27B-4bit DeltaNet):
    Hk=16, Hv=48, Dk=128, Dv=128, T=1 (decode)
    Sweep B in {4, 16, 52, 64}.

Reports correctness max-abs vs ops reference, p50 latency (ms), and
speedup ratio vs mlx kernel.
"""

from __future__ import annotations

import time

import mlx.core as mx
from mlx_lm.models.gated_delta import (
    gated_delta_kernel,
    gated_delta_ops,
)

from silica.kernels.gated_delta_v2 import gated_delta_kernel_v2


def maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def percentile(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))]


def time_fn(fn, n_warmup=4, n_iter=20):
    for _ in range(n_warmup):
        outs = fn()
        if isinstance(outs, tuple):
            mx.eval(*outs)
        else:
            mx.eval(outs)
    samples = []
    for _ in range(n_iter):
        t = time.perf_counter_ns()
        outs = fn()
        if isinstance(outs, tuple):
            mx.eval(*outs)
        else:
            mx.eval(outs)
        samples.append((time.perf_counter_ns() - t) / 1e6)
    return percentile(samples, 0.5), percentile(samples, 0.95)


def main() -> None:
    Hk, Hv, Dk, Dv = 16, 48, 128, 128
    print(f"{'B':<4} {'err_v2':<10} {'mlx_p50':<10} {'v2_p50':<10} {'ratio':<8} {'mlx_state_RW_GB':<14}")
    print("-" * 70)
    for B in (4, 16, 52, 64):
        T = 1
        mx.random.seed(0)
        q = (mx.random.normal((B, T, Hk, Dk)) * 0.1).astype(mx.bfloat16)
        k = (mx.random.normal((B, T, Hk, Dk)) * 0.1).astype(mx.bfloat16)
        v = (mx.random.normal((B, T, Hv, Dv)) * 0.1).astype(mx.bfloat16)
        # g: scalar gating (B, T, Hv) — most common production shape
        g = mx.random.uniform(low=0.5, high=1.0, shape=(B, T, Hv)).astype(mx.float32)
        beta = mx.random.uniform(low=0.5, high=1.0, shape=(B, T, Hv)).astype(mx.bfloat16)
        # State: bf16 (after cycle 12 fix)
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.bfloat16)
        mx.eval(q, k, v, g, beta, state)

        # Reference (ops, fp32 internal)
        y_ref, state_ref = gated_delta_ops(q, k, v, g, beta, state)
        mx.eval(y_ref, state_ref)

        # mlx kernel
        y_mlx, state_mlx = gated_delta_kernel(q, k, v, g, beta, state)
        mx.eval(y_mlx, state_mlx)
        err_mlx = maxabs(y_mlx, y_ref)

        # silica v2
        y_v2, state_v2 = gated_delta_kernel_v2(q, k, v, g, beta, state)
        mx.eval(y_v2, state_v2)
        err_v2 = maxabs(y_v2, y_ref)

        # Time both
        p50_mlx, _ = time_fn(lambda: gated_delta_kernel(q, k, v, g, beta, state))
        p50_v2, _ = time_fn(lambda: gated_delta_kernel_v2(q, k, v, g, beta, state))

        # State R/W bandwidth: state is (B, Hv, Dv, Dk) at sizeof(StT) bytes (bf16=2)
        state_size_bytes = B * Hv * Dv * Dk * 2  # bf16 = 2 bytes
        state_rw_gb = state_size_bytes * 2 / 1e9  # read + write

        ratio = p50_mlx / p50_v2
        print(f"{B:<4} {err_v2:<10.2e} {p50_mlx:<10.4f} {p50_v2:<10.4f} "
              f"{ratio:<8.3f} {state_rw_gb:<14.2f}")
        print(f"     mlx err = {err_mlx:.2e}")


if __name__ == "__main__":
    main()
