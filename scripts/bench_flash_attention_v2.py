"""Compare FA-decode v1 vs v2 vs mlx SDPA at the production shape sweep."""

from __future__ import annotations

import time

import mlx.core as mx

from silica.kernels.flash_attention_decode import (
    flash_attention_decode,
    reference_flash_attention_decode,
)
from silica.kernels.flash_attention_decode_v3 import flash_attention_decode_v3
from silica.kernels.flash_attention_decode_v4 import flash_attention_decode_v4
from silica.kernels.flash_attention_decode_v5 import flash_attention_decode_v5
from silica.kernels.flash_attention_decode_v6 import flash_attention_decode_v6


def maxabs(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[k]


def time_call(fn, n_warmup=8, n_iter=64):
    for _ in range(n_warmup):
        out = fn()
        mx.eval(out)
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        out = fn()
        mx.eval(out)
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return percentile(samples, 0.5), percentile(samples, 0.95)


def main() -> None:
    shapes = [
        (4, 24, 4, 128, 256),
        (16, 24, 4, 128, 256),
        (48, 24, 4, 128, 256),
        (48, 24, 4, 256, 256),
        (48, 24, 4, 512, 256),
        (48, 24, 4, 1024, 256),
    ]
    print(f"{'shape':<28} {'g':<5} {'e4':<8} {'e6':<8} {'v4':<7} {'v6':<7} {'mlx':<7} {'v4/m':<6} {'v6/m':<6}")
    print("-" * 95)
    for B, H_q, H_kv, T_kv, D in shapes:
        for gated in (False, True):
            mx.random.seed(0)
            q = (mx.random.normal((B, H_q, 1, D)) * 0.1).astype(mx.float16)
            k = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
            v = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
            gate = (mx.random.normal((B, H_q, 1, D)) * 0.5).astype(mx.float16) if gated else None
            mx.eval(q, k, v)
            if gate is not None: mx.eval(gate)

            out_ref = reference_flash_attention_decode(q, k, v, gate=gate)
            out_v4 = flash_attention_decode_v4(q, k, v, gate=gate)
            out_v6 = flash_attention_decode_v6(q, k, v, gate=gate)
            mx.eval(out_ref, out_v4, out_v6)

            err4 = maxabs(out_v4, out_ref)
            err6 = maxabs(out_v6, out_ref)

            p50_v4, _ = time_call(lambda: flash_attention_decode_v4(q, k, v, gate=gate))
            p50_v6, _ = time_call(lambda: flash_attention_decode_v6(q, k, v, gate=gate))
            p50_ref, _ = time_call(lambda: reference_flash_attention_decode(q, k, v, gate=gate))

            shape_str = f"B{B}H{H_q}/{H_kv}T{T_kv}D{D}"
            print(
                f"{shape_str:<28} {str(gated):<5} "
                f"{err4:<8.1e} {err6:<8.1e} "
                f"{p50_v4:<7.3f} {p50_v6:<7.3f} {p50_ref:<7.3f} "
                f"{p50_v4/p50_ref:<6.2f} {p50_v6/p50_ref:<6.2f}"
            )


if __name__ == "__main__":
    main()
