"""D-021 step 7 sub-unit (B.1) — 3-bit Qwen3.5-27B loader smoke probe.

Verifies that ``NexVeridian/Qwen3.5-27B-3bit`` loads cleanly via
silica's adapter factory and that ``Engine.generate("Hello",
max_tokens=4)`` produces 4 non-empty tokens. Records peak resident
memory so the §6.1 B.1 acceptance gate (≤ 13 GB OR ≥ 20% reduction
vs the v1.7.14 4-bit anchor 15.34 GB) can be evaluated.

Mirrors the existing ``probe_qwen3_5_27b_load.py`` structure (the
4-bit cousin's smoke probe) so the two probes are directly
comparable. Single load, single forward, no benching machinery —
this is a smoke test, not a measurement.

Usage:

    SILICA_REAL_QWEN3_5_27B_3BIT=1 \\
        uv run python -m scripts.probe_qwen3_5_27b_3bit_load

The env-var gate mirrors the bench scenario gate (so a developer
who has not consciously cached the 11 GB checkpoint cannot
trigger a load by accident); the script asserts the gate is set
before importing mlx-lm.
"""

from __future__ import annotations

import os
import sys
import time

REPO = "NexVeridian/Qwen3.5-27B-3bit"
GATE_ENV = "SILICA_REAL_QWEN3_5_27B_3BIT"


def main() -> int:
    if os.environ.get(GATE_ENV) != "1":
        print(
            f"error: {GATE_ENV} not set to '1'. This probe loads an "
            "~11 GB checkpoint and ~12-13 GB peak memory; set the "
            "gate explicitly to run.",
            file=sys.stderr,
        )
        return 2

    # Import lazily so the gate check fires before any heavy MLX init.
    import mlx.core as mx

    from silica.core.profiler import MetricsRegistry
    from silica.core.sampling import SamplingParams
    from silica.engine import Engine
    from silica.models.factory import adapter_for_repo

    print(f"# B.1 loader smoke probe — repo={REPO}")
    print(f"# gate env: {GATE_ENV}={os.environ[GATE_ENV]}")

    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()

    t0 = time.perf_counter()
    adapter, kv = adapter_for_repo(REPO)
    load_s = time.perf_counter() - t0
    peak_after_load_b = (
        mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else 0
    )

    print(f"adapter loaded in {load_s:.2f} s")
    print(
        f"peak after load: {peak_after_load_b / 1e9:.2f} GB "
        f"({peak_after_load_b / 1024**3:.2f} GiB)"
    )
    print(
        f"adapter config: model_name={adapter.config.model_name!r}, "
        f"num_layers={adapter.config.num_layers}, "
        f"hidden_size={adapter.config.hidden_size}, "
        f"vocab_size={adapter.config.vocab_size}"
    )

    # Smoke: 4-token generate from a fresh engine.
    engine = Engine(adapter=adapter, kv_manager=kv, metrics=MetricsRegistry())
    t1 = time.perf_counter()
    tokens = list(
        engine.generate(
            prompt="Hello",
            params=SamplingParams(max_tokens=4, temperature=0.0),
        )
    )
    gen_s = time.perf_counter() - t1
    peak_after_gen_b = (
        mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else 0
    )

    print(f"generated {len(tokens)} tokens in {gen_s:.2f} s: {tokens}")
    print(
        f"peak after generate: {peak_after_gen_b / 1e9:.2f} GB "
        f"({peak_after_gen_b / 1024**3:.2f} GiB)"
    )

    # B.1 §6.1 gate evaluation.
    anchor_gb = 15.34  # v1.7.14 4-bit P-6.0 anchor (probe-corrected).
    measured_gb = peak_after_gen_b / 1024**3
    pass_abs = measured_gb <= 13.0
    reduction_pct = (1.0 - measured_gb / anchor_gb) * 100.0
    pass_rel = reduction_pct >= 20.0
    pass_either = pass_abs or pass_rel

    print()
    print("B.1 §6.1 acceptance gate:")
    print(
        f"  measured peak = {measured_gb:.2f} GiB; "
        f"anchor 4-bit = {anchor_gb:.2f} GiB"
    )
    print(
        f"  reduction vs anchor = {reduction_pct:.1f}% "
        f"(target ≥ 20% under relative form)"
    )
    print(
        f"  abs form (≤ 13.0 GiB): "
        f"{'PASS' if pass_abs else 'FAIL'}"
    )
    print(
        f"  rel form (≥ 20% reduction): "
        f"{'PASS' if pass_rel else 'FAIL'}"
    )
    print(f"  gate (either passes): {'PASS' if pass_either else 'FAIL'}")

    if len(tokens) != 4:
        print(
            f"error: expected 4 tokens, got {len(tokens)}",
            file=sys.stderr,
        )
        return 1
    if not pass_either:
        print(
            "error: §6.1 gate fails on both absolute and relative forms",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
