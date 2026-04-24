# P-5 §7(a-real) — real-activation BlockTQ xcheck on Qwen3.5-0.8B

**Gate (PLAN.md §7 P-5 Notes (a-real) + docs/P5_OPENING.md §7(a-real)):**
real-activation Frobenius cross-check of silica MLX
`BlockTurboQuantMSE` against the vqbench-transcribed NumPy reference
on real K / V activations extracted from a Qwen3.5-0.8B forward
pass. Design contract: `docs/P5_A_REAL_OPENING.md` §2.

**Run date:** 2026-04-24 (commit `1a3c578` base, v1.7.4).
**Test:** `tests/test_block_tq_real_activation_xcheck.py::TestARealFullSweep`.
**Raw JSONL:** `docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.jsonl`
(1 metadata row + 144 data rows).

## Command

```bash
uv run python -m pytest \
    tests/test_block_tq_real_activation_xcheck.py -q -s
```

Exit code: `0`. Skipped on machines without the Qwen3.5-0.8B HF
cache (single-gate `_hf_cache_has_repo("Qwen/Qwen3.5-0.8B")`; no
`VQBENCH_PYTHON_EXECUTABLE` gate — the reference is inline NumPy
per design §2.3).

## Captured activations

| field | value |
| --- | --- |
| model | `Qwen/Qwen3.5-0.8B` (bf16) |
| hidden layers | 24 |
| `full_attention_interval` | 4 |
| GLOBAL layer count | **6** |
| GLOBAL layer indices | `[3, 7, 11, 15, 19, 23]` |
| prompt tokens | 138 (inside the 96..160 drift-guard envelope) |
| `n_kv_heads` | 2 |
| `head_dim` | 256 |
| extraction space | **pre-RoPE** `k_proj` / `v_proj` output (vqbench-aligned space, per §2.2) |
| tensor dtype | `bfloat16` |

Capture mechanism: read-only wrappers temporarily replace
`self_attn.k_proj` / `self_attn.v_proj` on each GLOBAL layer for
the duration of one prefill pass, record the projection output,
and return it unchanged; original methods restored in `finally`.

## Sweep dimensions

- `(vq_block_size, num_bits) ∈ {(32, 3), (32, 4), (64, 3), (64, 4)}`
  (4 cells; all compatible with `head_dim = 256 >= max vq_block_size = 64`).
- `seed ∈ {42, 43, 44}` (per §2.8 — 3 Haar rotation draws).
- `side ∈ {K, V}` (per K/V schema guardrail — §2.1 pins layer subset,
  this gate closes both projections).
- 6 GLOBAL layers.

Total rows: `4 × 3 × 2 × 6 = 144` data rows, every one populated.

## Gate results

### Silica-vs-NumPy-reference per-row |gap|

Tolerance envelope (§2.5, reusing (a-algo)'s pinned bounds):

- `(B=64, num_bits=4)` production-recommended cell: `< 1e-3`.
- All other `(B, bits)` cells: `< 5e-3`.

All 144 rows pass. Per (B, bits) × side distribution:

| B | bits | side | min | p50 | p95 | **max** | tol |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3 | K | 6.04e-05 | 7.66e-05 | 9.04e-05 | **9.04e-05** | 5e-03 |
| 32 | 3 | V | 6.42e-05 | 8.10e-05 | 9.41e-05 | **9.41e-05** | 5e-03 |
| 32 | 4 | K | 6.50e-06 | 3.00e-05 | 8.76e-05 | **8.76e-05** | 5e-03 |
| 32 | 4 | V | 8.81e-06 | 2.58e-05 | 4.59e-05 | **4.59e-05** | 5e-03 |
| 64 | 3 | K | 7.14e-05 | 8.66e-05 | 1.15e-04 | **1.15e-04** | 5e-03 |
| 64 | 3 | V | 5.52e-05 | 8.80e-05 | 9.78e-05 | **9.78e-05** | 5e-03 |
| 64 | 4 | K | 3.47e-06 | 2.93e-05 | 4.97e-05 | **4.97e-05** | 1e-03 |
| 64 | 4 | V | 1.82e-05 | 3.06e-05 | 5.21e-05 | **5.21e-05** | 1e-03 |

- **Worst-case |gap| overall:** `1.15e-04` on `(B=64, bits=3, K)`
  at layer 15 — about **43× headroom** under the `5e-3` tolerance.
- **Worst-case production cell (`B=64, bits=4`) |gap|:** `5.21e-05`
  (V side) — about **19× headroom** under the tighter `1e-3`
  production tolerance.
- **K / V symmetry.** Worst-case K gap `1.15e-04`; worst-case V
  gap `9.78e-05`. K and V both land inside the same envelope; no
  per-side pathology.

Per-layer worst-case across all cells × seeds × sides
(`worst_abs_gap` column is `max abs(silica_frob - numpy_frob)`
over every `(vq_block_size, num_bits, seed, side)` row sharing
that `layer_idx`):

| layer | worst_abs_gap | K_worst | V_worst |
| ---: | ---: | ---: | ---: |
| 3 | 9.62e-05 | 9.62e-05 | 9.59e-05 |
| 7 | 9.18e-05 | 8.81e-05 | 9.18e-05 |
| 11 | 9.84e-05 | 9.84e-05 | 9.46e-05 |
| 15 | 1.15e-04 | 1.15e-04 | 8.95e-05 |
| 19 | 9.78e-05 | 8.68e-05 | 9.78e-05 |
| 23 | 9.41e-05 | 8.78e-05 | 9.41e-05 |

No outlier layer: the 6 GLOBAL layers all live in the same
`9e-5 … 1.2e-4` worst-case band.

### Baseline ratio gate — uniformly degenerate (§2.4 fallback engaged)

`IdentityCodec.encode_tensor → decode_tensor` produced
`baseline_frob = 0.0` on **144 / 144** rows. This matches the
§2.4 prediction: `IdentityCodec` wraps the input in
`RawFp16Payload`, which stores the `mx.array` directly; the round
trip is byte-exact. The `silica_frob <= 2 * baseline_frob` ratio
gate therefore degenerates on every row.

Per §2.4 fallback, the silica-vs-NumPy absolute gap (table above)
is the primary close gate when the baseline is degenerate. The
test asserts `n_baseline_degenerate == n_rows` so any future
change that breaks `IdentityCodec` dtype preservation surfaces
loudly rather than silently exiting the fallback path.

### Absolute BlockTQ reconstruction error on real K / V

For reference (not a gate — this is the codec's absolute Frobenius
at each config, comparable to the D.2 probe numbers on Qwen3-0.6B):

| B | bits | side | min | mean | max |
| ---: | ---: | :---: | ---: | ---: | ---: |
| 32 | 3 | K | 0.1741 | 0.1763 | 0.1781 |
| 32 | 3 | V | 0.1739 | 0.1763 | 0.1841 |
| 32 | 4 | K | 0.0903 | 0.0914 | 0.0925 |
| 32 | 4 | V | 0.0901 | 0.0919 | 0.0994 |
| 64 | 3 | K | 0.1789 | 0.1816 | 0.1848 |
| 64 | 3 | V | 0.1791 | 0.1822 | 0.1926 |
| 64 | 4 | K | 0.0928 | 0.0943 | 0.0962 |
| 64 | 4 | V | 0.0932 | 0.0948 | 0.1024 |

Production `(B=64, b=4)` lands at **~9.4% K and ~9.5% V mean
reconstruction Frobenius** on Qwen3.5-0.8B real activations — the
same order of magnitude as the D.2 investigation probes measured
on Qwen3-0.6B K (~9.3%), confirming the two models' full-attention
K / V distributions present the same compression cost to BlockTQ.

## Scope delimiters

- **(a-real) is an algorithmic-parity gate on pre-RoPE K / V.**
  It verifies silica's MLX-native BlockTQ produces the same
  per-row Frobenius as vqbench's NumPy reference on real
  activations. It does **not** measure PPL (that is (b-static),
  still pending) and does **not** measure post-RoPE or
  chunk-boundary-dependent production-path quality (that is
  P-5-F, still pending).
- **Qwen3.5-0.8B only.** Design §6 keeps a parametrisable `--repo`
  escape hatch for future Qwen3.5-4B / Qwen3.5-35B-A3B extensions;
  not exercised in this landing to keep the HF-cache gate
  footprint small (4B ≈ 8 GB, 35B ≈ 20 GB).
- **Tolerance not tightened.** Worst-case gaps (`≤ 1.15e-4`) sit
  well below the current `5e-3` / `1e-3` bounds; tightening is
  evidence-based and explicitly deferred per the §2.5 "measure
  first, pin later" rule. A future tightening revision can
  reference this JSONL for the new numeric bounds.

## Conclusion

Silica MLX `BlockTurboQuantMSE` on real Qwen3.5-0.8B pre-RoPE K / V
activations agrees with the vqbench-transcribed NumPy reference to
within **1.15e-4 worst-case relative Frobenius**, ~43× tighter
than the (a-algo) synthetic Gaussian envelope. K and V are
symmetric. All 6 GLOBAL layers land in the same `9e-5 … 1.2e-4`
band. Baseline is uniformly degenerate (`IdentityCodec` is
dtype-preserving), and the absolute BlockTQ error on real K / V
lands at the same ~9–10% band observed on Qwen3-0.6B in the D.2
probes.

**(a-real) is closed** at v1.7.5. `(b-static)` remains in
post-P-5 backlog (blocked on P-3-C5 recurrent-state snapshot or
on the monkey-patch measurement route).
