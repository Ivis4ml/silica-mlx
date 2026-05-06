"""γ.1 microbench: mx.compile on Qwen3.5 ``Qwen3NextMLP`` forward.

Probes whether ``mx.compile`` produces a per-call speedup of ≥ 1.07× on
the production MLP forward at decode shape (B=4, T_q=1) on the sonnet
stack (mlx 0.31.1). Cycle-17 (opus, earlier 0.31.x) measured 1.027×
synthetic / ~0.5% E2E and retired the lever as below noise floor. γ.1
re-measures with the v1.7.26 *bucket × reachable-scope × per-call-gain*
projection discipline applied ahead of time.

Why this microbench. The MLP layer-block bucket is 46.3% of step time
per α (linear.mlp 34.9% + full.mlp 11.4%, same ``Qwen3NextMLP`` class
in both layer kinds). Unlike β's ``self_attn`` post-cache target —
which excluded cache mutation and reached only ~3-4% step — γ's MLP
forward has **no cache, no offset-dependent rope, no Python control
flow**, so the entire forward is compile-reachable. A 1.07× per-call
gain on a 46.3% bucket projects to 3.03% E2E (just clearing β.4's 3%
gate); 1.10× projects to 4.20%. Per-call gains below 1.07× project
below the gate and γ closes.

Three timing arms per shape:

    1. Uncompiled — direct Python call.
    2. ``mx.compile(fn, shapeless=True)`` — the realistic decode path
       (input shape is stable per-step at T_q=1, but ``shapeless=True``
       remains the default for any future T_q variation).
    3. ``mx.compile(fn)`` — fixed-shape compile; reported as upper
       bound. At decode T_q=1 the shape is invariant so fixed and
       shapeless should converge.

Decision rule (γ.1 gate per ``plans/P6_SMALL_B_GAMMA_OPENING.md`` §3):

    PASS         — best per-call speedup ≥ 1.07× AND σ_ratio ≤ 0.03
                   on the same arm across ≥ 2 sessions. γ.2 opens.
    FAIL         — best per-call speedup < 1.07× or σ_ratio > 0.03 on
                   every arm that hits the speedup. γ closes with a
                   measurement-anchored negative. δ next.

Output schema (JSONL):

    {"kind": "shape_result", "b": int, "hidden_size": int,
     "intermediate_size": int, "dtype": str,
     "uncompiled_median_ms": float, "shapeless_compiled_median_ms": float,
     "fixed_compiled_median_ms": float, "speedup_shapeless": float,
     "speedup_fixed": float, "iters": int, "warmup": int, "ts": str}
    {"kind": "summary", ...}

Usage:

    SILICA_REAL_QWEN3_5_27B=1 \\
        uv run python -m silica.bench.microbench.compiled_mlp \\
            --repo mlx-community/Qwen3.5-27B-4bit --b 4 \\
            --warmup 3 --iters 20 \\
            --out plans/P6_SMALL_B/GAMMA/microbench/compiled_mlp_b4.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx


def find_full_attention_layer(model: Any) -> tuple[int, Any]:
    """Return ``(layer_idx, layer)`` for the first non-DeltaNet layer.

    The ``mlp`` attribute on a full-attention layer is the same
    ``Qwen3NextMLP`` class used in DeltaNet layers, so probing either
    layer kind exposes the production MLP forward.
    """
    for i, layer in enumerate(model.layers):
        if not getattr(layer, "is_linear", False):
            return i, layer
    raise RuntimeError("no full-attention layer found in model.layers")


def _read_mlp_shape(mlp: Any) -> tuple[int, int]:
    """Pull ``(hidden_size, intermediate_size)`` from the mlp module.

    Reads weight shape off ``gate_proj`` so the microbench matches the
    production layer without hardcoding the model config. Handles both
    ``QuantizedLinear`` (4-bit production target) and unquantized
    ``Linear``.

    For ``QuantizedLinear`` with ``bits`` packing into uint32 storage,
    weight shape is ``(out_features, in_features // pack_factor)``
    where ``pack_factor = 32 // bits``. So
    ``in_features = weight.shape[1] * pack_factor``,
    ``out_features = weight.shape[0]``.

    For ``Linear``, weight shape is ``(out_features, in_features)``
    directly.
    """
    gate = mlp.gate_proj
    if hasattr(gate, "in_features") and hasattr(gate, "out_features"):
        return int(gate.in_features), int(gate.out_features)
    weight = getattr(gate, "weight", None)
    if weight is None:
        raise RuntimeError(
            "mlp.gate_proj has no .weight attribute; cannot read shape"
        )
    out_features = int(weight.shape[0])
    bits = getattr(gate, "bits", None)
    if bits is None:
        # Unquantized Linear path.
        in_features = int(weight.shape[1])
    else:
        pack_factor = 32 // int(bits)
        in_features = int(weight.shape[1]) * pack_factor
    return in_features, out_features


def _synth_input(
    b: int, hidden_size: int, dtype: Any, seed: int
) -> mx.array:
    """Build a representative-shape MLP input at fixed seed."""
    x = mx.random.normal(
        (b, 1, hidden_size), key=mx.random.key(seed), dtype=mx.float32
    ).astype(dtype)
    mx.eval(x)
    return x


def _time_arm(
    fn: Callable[[mx.array], mx.array],
    input_x: mx.array,
    warmup: int,
    iters: int,
) -> list[float]:
    """Wall-time ``fn(input_x)`` over ``iters`` with ``warmup`` priming."""
    for _ in range(warmup):
        out = fn(input_x)
        mx.eval(out)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        out = fn(input_x)
        mx.eval(out)
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return times


def run_microbench(
    repo: str,
    b: int,
    warmup_iters: int,
    measurement_iters: int,
    dtype_str: str,
    out_path: Path,
) -> dict[str, Any]:
    """Run the three-arm MLP forward microbench; emit JSONL."""
    from silica.models.factory import adapter_for_repo

    print(f"[γ.1] loading adapter for {repo} (cache-warm)...")
    adapter, _ = adapter_for_repo(repo)
    model = adapter._model  # type: ignore[attr-defined]  # noqa: SLF001 — bench-side introspection.

    layer_idx, layer = find_full_attention_layer(model)
    mlp = layer.mlp
    print(f"[γ.1] using model.layers[{layer_idx}].mlp ({type(mlp).__name__})")

    hidden_size, intermediate_size = _read_mlp_shape(mlp)
    print(
        f"[γ.1] shape config: b={b} hidden_size={hidden_size} "
        f"intermediate_size={intermediate_size}"
    )

    dtype = mx.bfloat16 if dtype_str == "bfloat16" else mx.float16

    def fn_uncompiled(x: mx.array) -> mx.array:
        return mlp(x)

    fn_shapeless = mx.compile(fn_uncompiled, shapeless=True)
    fn_fixed = mx.compile(fn_uncompiled)

    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[γ.1] timing all three arms...")
    input_x = _synth_input(b, hidden_size, dtype, seed=0)

    u_times = _time_arm(fn_uncompiled, input_x, warmup_iters, measurement_iters)
    sl_times = _time_arm(fn_shapeless, input_x, warmup_iters, measurement_iters)
    fx_times = _time_arm(fn_fixed, input_x, warmup_iters, measurement_iters)

    u_med = statistics.median(u_times)
    sl_med = statistics.median(sl_times)
    fx_med = statistics.median(fx_times)
    sp_sl = u_med / sl_med if sl_med > 0 else float("nan")
    sp_fx = u_med / fx_med if fx_med > 0 else float("nan")

    row = {
        "kind": "shape_result",
        "b": b,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": dtype_str,
        "uncompiled_median_ms": u_med,
        "shapeless_compiled_median_ms": sl_med,
        "fixed_compiled_median_ms": fx_med,
        "uncompiled_min_ms": min(u_times),
        "shapeless_compiled_min_ms": min(sl_times),
        "fixed_compiled_min_ms": min(fx_times),
        "speedup_shapeless": sp_sl,
        "speedup_fixed": sp_fx,
        "iters": measurement_iters,
        "warmup": warmup_iters,
        "ts": ts,
    }
    rows.append(row)
    print(
        f"  uncompiled        = {u_med:7.3f} ms  (min {min(u_times):7.3f})"
    )
    print(
        f"  shapeless compile = {sl_med:7.3f} ms  ({sp_sl:5.3f}×)"
        f"  (min {min(sl_times):7.3f})"
    )
    print(
        f"  fixed-shape       = {fx_med:7.3f} ms  ({sp_fx:5.3f}×)"
        f"  (min {min(fx_times):7.3f})"
    )

    best = max(sp_sl, sp_fx)
    pass_local = best >= 1.07
    summary = {
        "kind": "summary",
        "repo": repo,
        "b": b,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "dtype": dtype_str,
        "iters": measurement_iters,
        "warmup": warmup_iters,
        "best_speedup": best,
        "best_arm": "shapeless" if sp_sl >= sp_fx else "fixed",
        "pass_local_1_07x": pass_local,
        "verdict_local": "PASS_LOCAL" if pass_local else "FAIL_LOCAL",
        "ts": ts,
    }
    rows.append(summary)

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print()
    print(
        f"[γ.1] best speedup: {best:.3f}× "
        f"({summary['best_arm']} arm)"
    )
    print(f"[γ.1] local verdict: {summary['verdict_local']}")
    print(f"[γ.1] wrote {len(rows)} rows to {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument("--b", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16"]
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not os.environ.get("SILICA_REAL_QWEN3_5_27B"):
        raise SystemExit(
            "SILICA_REAL_QWEN3_5_27B=1 required to load the production target."
        )

    run_microbench(
        repo=args.repo,
        b=args.b,
        warmup_iters=args.warmup,
        measurement_iters=args.iters,
        dtype_str=args.dtype,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
