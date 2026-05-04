"""Final ablation: v3 vs v6 vs v7 vs v8 vs mlx at production B=48 shape sweep."""

from __future__ import annotations
import time
import mlx.core as mx
from silica.kernels.flash_attention_decode import reference_flash_attention_decode
from silica.kernels.flash_attention_decode_v3 import flash_attention_decode_v3
from silica.kernels.flash_attention_decode_v6 import flash_attention_decode_v6
from silica.kernels.flash_attention_decode_v7 import flash_attention_decode_v7
from silica.kernels.flash_attention_decode_v8 import flash_attention_decode_v8


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
    print(f"{'shape':<25} {'g':<3} {'v3':<7} {'v6':<7} {'v7':<7} {'v8':<7} {'mlx':<7} {'v8/mlx':<7}")
    print("-" * 75)
    for T_kv in [128, 256, 512, 1024]:
        for gated in (False, True):
            B, H_q, H_kv, D = 48, 24, 4, 256
            mx.random.seed(0)
            q = (mx.random.normal((B, H_q, 1, D)) * 0.1).astype(mx.float16)
            k = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
            v = (mx.random.normal((B, H_kv, T_kv, D)) * 0.1).astype(mx.float16)
            gate = (mx.random.normal((B, H_q, 1, D)) * 0.5).astype(mx.float16) if gated else None
            mx.eval(q, k, v)
            if gate is not None: mx.eval(gate)

            p3, _ = time_call(lambda: flash_attention_decode_v3(q, k, v, gate=gate))
            p6, _ = time_call(lambda: flash_attention_decode_v6(q, k, v, gate=gate))
            p7, _ = time_call(lambda: flash_attention_decode_v7(q, k, v, gate=gate))
            p8, _ = time_call(lambda: flash_attention_decode_v8(q, k, v, gate=gate))
            pm, _ = time_call(lambda: reference_flash_attention_decode(q, k, v, gate=gate))

            shape_str = f"B{B}H{H_q}/{H_kv}T{T_kv}"
            print(f"{shape_str:<25} {str(gated)[0]:<3} "
                  f"{p3:<7.4f} {p6:<7.4f} {p7:<7.4f} {p8:<7.4f} {pm:<7.4f} {p8/pm:<7.3f}")


if __name__ == "__main__":
    main()
