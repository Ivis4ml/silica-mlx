"""Correctness + perf microbench for ``silica.kernels.fused_gated_output``.

Per AR.md Custom kernel authorisation: every custom kernel ships with
a correctness gate (max-abs / max-rel error vs the MLX reference on
≥3 production-shape input cases) and a perf microbench (p50 / p95
latency vs the reference at production shapes).

This microbench is the canonical pattern for any future kernel; new
kernels can copy this structure and substitute their own kernel +
reference. Pure synthetic input — no real-model load, no download.

Usage:
    uv run python -m silica.bench.microbench.fused_gated_output_microbench \\
        --out plans/P6_AUTORESEARCH/fused_gated_output_microbench.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx

from silica.kernels.fused_gated_output import (
    fused_gated_output,
    reference_gated_output,
)


# Production decode shapes for Qwen3.5-27B-4bit at the dense warm-decode
# B=4 hot path (full attention layers, after SDPA, before o_proj).
# Shape: (B, num_heads, T, head_dim).
PRODUCTION_SHAPES: tuple[tuple[int, int, int, int], ...] = (
    (4, 24, 1, 256),    # B=4 decode (primary)
    (1, 24, 1, 256),    # B=1 decode (B=1 baseline anchor)
    (4, 24, 4, 256),    # spec verify k=4 at B=4
    (4, 24, 8, 256),    # spec verify k=8 at B=4
)


def _make_random_pair(
    shape: tuple[int, ...], dtype: mx.Dtype, seed: int
) -> tuple[mx.array, mx.array]:
    """Build (x, g) of the requested shape/dtype with deterministic seed.

    g is sampled from N(0, 1) so sigmoid(g) covers the full output
    range; x is sampled uniformly in [-1, 1] mimicking post-SDPA
    activation magnitudes.
    """
    key = mx.random.key(seed)
    key1, key2 = mx.random.split(key, 2)
    x = mx.random.uniform(low=-1.0, high=1.0, shape=shape, key=key1).astype(dtype)
    g = mx.random.normal(shape=shape, key=key2).astype(dtype)
    return x, g


def _correctness_check(
    shape: tuple[int, ...], dtype: mx.Dtype, seed: int
) -> dict[str, Any]:
    """Compare fused kernel vs MLX reference on a single shape; return errors."""
    x, g = _make_random_pair(shape, dtype, seed)
    out_kernel = fused_gated_output(x, g)
    out_ref = reference_gated_output(x, g)
    mx.eval(out_kernel, out_ref)
    diff = out_kernel.astype(mx.float32) - out_ref.astype(mx.float32)
    abs_diff = mx.abs(diff)
    max_abs = float(mx.max(abs_diff).item())
    # Relative error: |diff| / max(|ref|, 1e-6)
    abs_ref = mx.abs(out_ref.astype(mx.float32))
    floor = mx.maximum(abs_ref, mx.array(1e-6, dtype=mx.float32))
    max_rel = float(mx.max(abs_diff / floor).item())
    mean_abs = float(mx.mean(abs_diff).item())
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "seed": seed,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "mean_abs_err": mean_abs,
        "output_norm": float(mx.linalg.norm(out_ref.astype(mx.float32)).item()),
    }


def _time_op(
    op_name: str,
    op_fn: Any,
    shape: tuple[int, ...],
    dtype: mx.Dtype,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    """Time a callable ``op_fn(x, g) -> result`` with warmup + iters."""
    x, g = _make_random_pair(shape, dtype, seed)

    # Warmup
    for _ in range(warmup):
        out = op_fn(x, g)
        mx.eval(out)

    # Measurement
    samples_ns: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = op_fn(x, g)
        mx.eval(out)
        samples_ns.append(time.perf_counter_ns() - t0)

    samples_ms = [s / 1e6 for s in samples_ns]
    samples_ms.sort()
    return {
        "op": op_name,
        "shape": list(shape),
        "dtype": str(dtype),
        "n": iters,
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": samples_ms[max(0, int(0.95 * len(samples_ms)) - 1)],
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
    }


def run(out_path: Path, warmup: int = 30, iters: int = 200) -> dict[str, Any]:
    """Run correctness + perf for both fp16 and bf16 across production shapes."""
    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()

    for dtype in (mx.float16, mx.bfloat16):
        for shape in PRODUCTION_SHAPES:
            # Correctness
            for seed in (0, 1, 2):
                row = _correctness_check(shape, dtype, seed)
                row.update(kind="correctness", ts=ts)
                rows.append(row)

            # Perf
            row_kernel = _time_op(
                "fused_gated_output", fused_gated_output, shape, dtype,
                warmup=warmup, iters=iters, seed=0,
            )
            row_kernel.update(kind="perf", ts=ts)
            rows.append(row_kernel)
            row_ref = _time_op(
                "reference_gated_output", reference_gated_output, shape, dtype,
                warmup=warmup, iters=iters, seed=0,
            )
            row_ref.update(kind="perf", ts=ts)
            rows.append(row_ref)
            speedup = row_ref["p50_ms"] / row_kernel["p50_ms"] if row_kernel["p50_ms"] > 0 else 0.0
            rows.append({
                "kind": "perf_summary",
                "ts": ts,
                "shape": list(shape),
                "dtype": str(dtype),
                "kernel_p50_ms": row_kernel["p50_ms"],
                "ref_p50_ms": row_ref["p50_ms"],
                "speedup_kernel_vs_ref": speedup,
                "kernel_p95_ms": row_kernel["p95_ms"],
                "ref_p95_ms": row_ref["p95_ms"],
            })

    # Top-level summary
    # Per-dtype correctness gate (fp16/bf16 carry their own ULP noise).
    # The kernel is mathematically identical to the reference; the
    # observed delta is fp16/bf16 representation noise from how
    # ``mx.sigmoid`` vs the inline ``1/(1+exp(-g))`` round per-element.
    GATE_BY_DTYPE = {
        "mlx.core.float32": {"max_abs": 1e-5, "max_rel": 1e-5},
        "mlx.core.float16": {"max_abs": 1e-2, "max_rel": 5e-2},
        "mlx.core.bfloat16": {"max_abs": 5e-2, "max_rel": 1e-1},
    }
    correctness_rows = [r for r in rows if r["kind"] == "correctness"]
    perf_summary_rows = [r for r in rows if r["kind"] == "perf_summary"]
    max_max_abs = max(r["max_abs_err"] for r in correctness_rows)
    max_max_rel = max(r["max_rel_err"] for r in correctness_rows)
    # Per-dtype gate evaluation
    gate_violations: list[dict[str, Any]] = []
    for r in correctness_rows:
        gate = GATE_BY_DTYPE.get(r["dtype"], {"max_abs": 1e-3, "max_rel": 1e-3})
        if r["max_abs_err"] >= gate["max_abs"] or r["max_rel_err"] >= gate["max_rel"]:
            gate_violations.append(
                {
                    "shape": r["shape"],
                    "dtype": r["dtype"],
                    "seed": r["seed"],
                    "max_abs_err": r["max_abs_err"],
                    "max_rel_err": r["max_rel_err"],
                    "gate_abs": gate["max_abs"],
                    "gate_rel": gate["max_rel"],
                }
            )
    median_speedup = statistics.median(
        r["speedup_kernel_vs_ref"] for r in perf_summary_rows
    )
    summary = {
        "kind": "summary",
        "ts": ts,
        "n_correctness_rows": len(correctness_rows),
        "n_perf_rows": len(perf_summary_rows),
        "max_max_abs_err": max_max_abs,
        "max_max_rel_err": max_max_rel,
        "median_speedup_kernel_vs_ref": median_speedup,
        "production_shapes": [list(s) for s in PRODUCTION_SHAPES],
        "warmup_iters": warmup,
        "perf_iters": iters,
        "correctness_gate_by_dtype": GATE_BY_DTYPE,
        "correctness_gate_violations": gate_violations,
        "correctness_gate_passed": len(gate_violations) == 0,
    }
    rows.append(summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"[fused_gated_output] wrote {len(rows)} rows to {out_path}")
    print(f"[fused_gated_output] max-abs error across all shapes/dtypes: {max_max_abs:.4e}")
    print(f"[fused_gated_output] max-rel error across all shapes/dtypes: {max_max_rel:.4e}")
    print(f"[fused_gated_output] correctness gate (per-dtype ULP-aware): "
          f"{'PASS' if summary['correctness_gate_passed'] else 'FAIL ' + str(len(gate_violations)) + ' violations'}")
    print(f"[fused_gated_output] median speedup kernel vs reference: "
          f"{median_speedup:.3f}×")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plans/P6_AUTORESEARCH/fused_gated_output_microbench.jsonl"),
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()
    run(args.out, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
