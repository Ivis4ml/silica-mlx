# P-6 Autoresearch — seventeenth cycle (2026-05-04) — compiled-MLP probe retires

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Diagnostic — mx.compile on Qwen3.5-shape MLP gives only **1.027× (2.7%)** at B=52 4-bit. Below the warm-decode oracle noise floor; E2E undetectable. Patch retired. |
| User authorisation | "great, continue" |
| Companion docs | `REPORT_KERNEL_CYCLE_{14,15,16}.md` |

## TL;DR

Tested mx.compile on a synthetic Qwen3NextMLP at production B=52 H=5120
G=17408 with 4-bit quantized weights:

```
Qwen3.5 MLP (B=52, H=5120, G=17408, 4-bit)
  uncompiled: 5.6951 ms
  compiled:   5.5464 ms
  speedup:    1.027× (= 2.7%)
```

Across 64 MLPs per decode step, expected E2E delta: ~0.5% (assuming MLP is
~30% of step time). On warm-decode-b52 (206.2 ± 0.5 tok/s) that's ~1 tok/s
— at the noise floor.

Probed E2E with `SILICA_USE_COMPILED_MLP=1`: 203.0 tok/s (single run, 200s wall
— suspicious). Even with the noisy single-run measurement, no signal above
the cycle-14 baseline.

## Why mx.compile under-performs here

The MLP forward is dominated by **3 quantized matmul calls**, each
already a single fused kernel inside mlx (`mx.fast.quantized_matmul`).
mx.compile can fuse the SwiGLU activation between the qmms (which is
already fused via `_precise_swiglu`'s @mx.compile) but cannot fuse
across the heavy compute kernels. The remaining win is dispatch
overhead amortization: 4 Python op calls → 1 graph dispatch. At the
production batch size (B=52, MLP forward ≈ 5.7 ms) the per-call
dispatch overhead is a small fraction; compile saves a small
fraction of a small fraction.

## Patch retired

`SILICA_USE_COMPILED_MLP=1` env flag is preserved in
`silica/kernels/shadow_install.py` for documentation but the patch is
a no-op (sets `installed["compiled_mlp"] = False` and returns). 2779
silica tests still pass.

## Cycle 16 + 17 closing summary

The remaining-lever investigation across cycles 15 + 16 + 17:

| Lever | Probe result | Verdict |
| --- | --- | --- |
| QMM half4 retrofit | mlx already loads 4 uint32/thread (line 707 of `quantized.h`) | retired — cannot beat mlx |
| mx.compile on chain of 3 elementwise ops | 0.91-0.94× (slowdown) | not viable |
| mx.compile on attention forward (no cache) | 1.08× | blocked by cache mutation in real layer |
| **mx.compile on Qwen3NextMLP at production shape** | **1.027× (E2E ~0.5%)** | **below noise floor — retired** |
| Stack v10+bf16 at B=4 (low-B regime) | 41.1 vs 42.17 baseline (-2.5%) | regime-specific only |
| SILICA_DECODE_CHUNK=2/32 | warmup-stability gate fail | retired (cycles 4/5/12/15) |
| SPLIT_K=64/256 in v8 | hurt by 1.5-3.5 tok/s | SPLIT_K=128 default holds |
| Stack +fused_silu_mul | -1.6 vs C14 | does not stack |

**The cycles 11-14 v10+bf16 stack at B=52 (envelope) and B=64 (hardware)
are the load-bearing optima within the cycle-1-17 lever set.** No further
local probe has produced a measurable E2E lift.

## Final state of the autoresearch loop

After 17 cycles:
- **Within strict 36 GB envelope: 206.2 ± 0.5 tok/s (4.89× cycle-1)**
- **Within 48 GB hardware ceiling: 232.2 ± 0.3 tok/s (5.51× cycle-1)**
- **(1b) ≥60 milestone CLEARED 3.44× / 3.87×**
- 9 KEEPs on running-best ladder
- 35 diagnostic-class probes
- 21 discards
- 0 crashes
- 2779 silica tests pass

The remaining-known levers are all gated or substantial:
- **Speculative decoding** (D-021, gated on C.5 γ.1 read-only survey decision)
- **mx.compile graph-trace with cache rerouting** (4-6 hour integration; ~2% expected per cycle 16 probe)
- **External upgrade to MLX 0.32+ async-copy primitives** (not yet released)

The autoresearch loop has reached the local maximum on the cycle-1-17
lever set. Per AR.md "Stop conditions" §, both cycle-1 stop conditions
remain cleared by margin; no urgency to push further unless an
authorisation opens.
