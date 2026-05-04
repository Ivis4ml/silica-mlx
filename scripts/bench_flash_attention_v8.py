"""Compare FA v6 vs v7 vs v8 vs mlx."""

from __future__ import annotations
import time
import mlx.core as mx
from silica.kernels.flash_attention_decode import reference_flash_attention_decode
from silica.kernels.flash_attention_decode_v6 import flash_attention_decode_v6
from silica.kernels.flash_attention_decode_v7 import flash_attention_decode_v7
from silica.kernels.flash_attention_decode_v8 import flash_attention_decode_v8


def maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def percentile(xs, p):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))]


def time_call(fn, n_warmup=8, n_iter=64):
    for _ in range(n_warmup):
        mx.eval(fn())
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter_ns()
        mx.eval(fn())
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return percentile(samples, 0.5), percentile(samples, 0.95)


def main():
    shapes = [
        (48, 24, 4, 128, 256),
        (48, 24, 4, 256, 256),
        (48, 24, 4, 512, 256),
        (48, 24, 4, 1024, 256),
    ]
    print(f"{'shape':<28} {'g':<5} {'e7':<8} {'e8':<8} {'v7':<8} {'v8':<8} {'mlx':<8} {'v7/m':<6} {'v8/m':<6}")
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
            out_v7 = flash_attention_decode_v7(q, k, v, gate=gate)
            out_v8 = flash_attention_decode_v8(q, k, v, gate=gate)
            mx.eval(out_ref, out_v7, out_v8)
            err7 = maxabs(out_v7, out_ref)
            err8 = maxabs(out_v8, out_ref)

            p50_v7, _ = time_call(lambda: flash_attention_decode_v7(q, k, v, gate=gate))
            p50_v8, _ = time_call(lambda: flash_attention_decode_v8(q, k, v, gate=gate))
            p50_ref, _ = time_call(lambda: reference_flash_attention_decode(q, k, v, gate=gate))

            shape_str = f"B{B}H{H_q}/{H_kv}T{T_kv}D{D}"
            print(f"{shape_str:<28} {str(gated):<5} {err7:<8.1e} {err8:<8.1e} "
                  f"{p50_v7:<8.4f} {p50_v8:<8.4f} {p50_ref:<8.4f} "
                  f"{p50_v7/p50_ref:<6.2f} {p50_v8/p50_ref:<6.2f}")


if __name__ == "__main__":
    main()
