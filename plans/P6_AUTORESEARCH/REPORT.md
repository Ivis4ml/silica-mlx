# P-6 Autoresearch — first kernel cycle (2026-05-03)

| Field | Value |
| --- | --- |
| Date | 2026-05-03 |
| Branch | `opus` |
| Status | Diagnostic + first-kernel + honest-negative |
| Companion docs | `plans/P6_AUTORESEARCH_REORIENTATION.md` (orientation memo); `plans/P6_AUTORESEARCH_NOT_LIMIT_PROOF.md` (proof that 42.17 is not the limit); `plans/P6_AUTORESEARCH_LOG.tsv` (ledger) |

## Two new measurements in this cycle

### Measurement 1 — Per-step decode time decomposition on dense Qwen3.5-27B-4bit, B=4 (20 iters)

`plans/P6_AUTORESEARCH/decode_step_attribution_b4.jsonl`

| Quantity | Value |
| --- | --- |
| step_total median (instrumented) | 111.79 ms |
| step_total uninstrumented (= P-6.0.5 Unit 2) | ~95 ms |
| Instrumented overhead | 4.51 ms = 4.0% of step (per-layer `mx.eval` barriers) |
| Linear (DeltaNet) layers total | 82.96 ms = **74.2% of step** |
| Full attention layers total | 24.48 ms = **21.9% of step** |
| Per-DeltaNet-layer mean | 1.728 ms |
| Per-full-attn-layer mean | 1.530 ms |
| n_layers | 64 (48 linear + 16 full, verified vs config.json) |

**Reading:** the 3:1 layer count (48 DeltaNet : 16 full attention) drives the cost split. **Per-layer wall-clock is nearly equal across kinds.** This is a strong signal that:
- Each layer's cost is dominated by per-layer weight reads (MLP and projections) that are the same magnitude for both kinds.
- Layer-kind-specific kernel work (e.g., custom Gated SDPA for full-attn vs custom gated-delta-update-with-conv1d for DeltaNet) has roughly proportional headroom on a per-layer basis.
- The biggest absolute leverage lives on the 48 DeltaNet layers (74% of step time), but mlx-lm already ships a `mx.fast.metal_kernel`-backed `gated_delta_update`. Closing the conv1d-fusion gap (ZMLX prototype pattern) would be additive there.
- The 16 full-attention layers (22% of step time) carry the unfused gated SDPA gap the 2026-Q2 survey identified as the load-bearing kernel-fusion opportunity.

### Measurement 2 — Fused output-gate kernel correctness + perf microbench

`plans/P6_AUTORESEARCH/fused_gated_output_microbench.jsonl`

The kernel `silica.kernels.fused_gated_output(x, g)` fuses `x * sigmoid(g)` into a single `mx.fast.metal_kernel` dispatch — the first concrete custom MLX kernel in Silica's tree under the 2026-05-02 / 2026-05-03 kernel-write authorisation.

| Quantity | Value |
| --- | --- |
| Production shapes tested | 4: (4, 24, 1, 256), (1, 24, 1, 256), (4, 24, 4, 256), (4, 24, 8, 256) |
| Dtypes tested | fp16, bfloat16 |
| Seeds per (shape, dtype) | 3 |
| Total correctness rows | 24 |
| Max-abs error across all rows | 5.86e-3 (fp16; fp16 ULP noise) |
| Max-rel error across all rows | 1.39e-2 (fp16) |
| Per-dtype gate (fp16) | max_abs < 1e-2 AND max_rel < 5e-2 |
| Per-dtype gate (bf16) | max_abs < 5e-2 AND max_rel < 1e-1 |
| **Correctness gate** | **PASS** |
| Perf rows (kernel + reference per shape/dtype) | 16 |
| Median p50 speedup (kernel vs reference) | **1.006×** |

**Honest negative on speedup:** the kernel is numerically correct but does **not** measurably outperform the MLX reference `x * mx.sigmoid(g)`. Mechanism:
- Both implementations are memory-bound elementwise ops over the (4, 24, 1, 256) = ~25 KB output tensor.
- HBM traffic is identical: read x, read g, write out (3 round-trips of one tensor).
- The kernel saves one launch overhead (~10-20 μs) but at this op size that is a single-digit-microsecond signal swamped by per-iter jitter at p50.
- Fp16 ULP noise on the kernel's `1/(1+exp(-g))` vs `mx.sigmoid` accounts for the 5.86e-3 max-abs delta — pinned to within fp16 ULP, not a kernel bug.

**Lesson and path forward:** trivial fusion of two memory-bound elementwise ops cannot move T₄ because the HBM round-trips are identical. **Real kernel speedup requires fusing significantly more work into one dispatch** so that intermediates stay in registers / threadgroup memory rather than HBM. Three concrete next-kernel candidates that satisfy this:

1. **Fused FA-style gated SDPA.** One kernel does `Q @ K^T → softmax → @V → sigmoid_gate_apply` with the SDPA output staying in registers. Eliminates writing the SDPA output tensor (4 × 24 × 1 × 256 × 2 = 49 KB at decode shape; 16 layers × 4 KB/output = 64 KB/step of saved HBM traffic minimum). Bigger savings come from the FA-style tile-by-tile online-softmax reducing K/V reads. Engineering scope: substantial (FA kernel tuning + mlx-lm parity + gate-apply correctness) — appropriately sized as a multi-session work package. References per the 2026-Q2 survey: dflash-mlx JIT SDPA 2-pass + MFA Swift port (pattern only — no public Apple-Silicon implementation has the output-gate baked in).

2. **Fused output-gate-apply + o_proj quantised matmul.** One kernel does `out_gated = sigmoid(g) * sdpa_out; logits = out_gated @ W_o_proj_q4` keeping `out_gated` in threadgroup memory. Saves the (B, H, T, D) tensor write to HBM that step (1) above also targets, AND saves a kernel launch for o_proj. Engineering scope: medium (quantised matmul kernel + gate-apply epilogue + integration with mlx-lm's quantised projection path). References: dflash-mlx `verify_qmm` int4 simdgroup-MMA for M=16 (different M shape than B=4 decode's M=4, but the simdgroup-MMA pattern transfers).

3. **Fused gated-delta-update + conv1d** (DeltaNet path). The 2026-Q2 survey identified this as the largest still-unfused DeltaNet opportunity. mlx-lm ships `gated_delta_update` as a Metal kernel, but conv1d (kernel_dim=4) is a separate dispatch. Fusing conv1d into the recurrence kernel (per ZMLX v0.10.0 prototype) closes the gap on the 48 DeltaNet layers that own 74% of step time. Engineering scope: substantial (interacts with DeltaNet recurrent state semantics; mlx-lm reference kernel needs extension). References: ZMLX deltanet pattern + mlx-lm `gated_delta.py`.

Per the P6_AUTORESEARCH.md custom-kernel mandate, these candidates open under autonomous scope at microbench + correctness + shadow-mode level, with explicit user approval required for hot-path replacement. Items 1 and 3 are larger work; item 2 sits between.

## Aggregate position vs the 42.17 baseline

This cycle did NOT improve T₄. The kernel work delivered:
- A working correctness + perf microbench harness (`silica.bench.microbench.fused_gated_output_microbench`).
- A first concrete custom MLX kernel (`silica.kernels.fused_gated_output`) with a passing correctness gate but flat speedup.
- A measurement-anchored attribution of where the 95 ms step time goes (74% DeltaNet, 22% attention, 4% overhead).
- Validation that the per-step microbench harness (`silica.bench.microbench.decode_step_attribution`) is reusable for any future kernel candidate's end-to-end attribution.

Running best on `decode_tok_s` is **unchanged at 42.17 ± 0.21 tok/s**.

The cycle is a **diagnostic** in the P6_AUTORESEARCH.md status taxonomy:
- Decisively narrows where to spend the next kernel cycle (74% vs 22% leverage);
- Establishes that the simple end of fusion (two-op elementwise) is not enough;
- Leaves a working harness and a working trivial kernel as the foundation for the larger candidates.

## Files added in this cycle

- `silica/kernels/__init__.py` (new package)
- `silica/kernels/fused_gated_output.py` (kernel + reference)
- `silica/bench/microbench/decode_step_attribution.py` (per-step decomposition microbench)
- `silica/bench/microbench/fused_gated_output_microbench.py` (correctness + perf microbench pattern)
- `scripts/render_autoresearch_chart.py` (Karpathy-style progress chart from the TSV ledger)
- `pyproject.toml` `[project.optional-dependencies]` `bench` extra for matplotlib
- `plans/P6_AUTORESEARCH/decode_step_attribution_b4.jsonl` (real-model B=4, 20-iter measurement)
- `plans/P6_AUTORESEARCH/fused_gated_output_microbench.jsonl` (kernel correctness + perf bundle)
- `plans/P6_AUTORESEARCH_PROGRESS_DECODE_TOK_S.png` (chart)
- `plans/P6_AUTORESEARCH_NOT_LIMIT_PROOF.md` (formal proof T₄ is not the chip ceiling)
- `plans/P6_AUTORESEARCH/REPORT.md` (this file)

## Ledger rows appended this cycle

- `AR_DECODE_DECOMPOSITION_B4` — diagnostic; per-step decomposition; 74.2% DeltaNet / 21.9% full-attn.
- `AR_KERNEL_FUSED_GATED_OUTPUT` — diagnostic; first custom MLX kernel; correctness PASS, perf 1.006× (flat).

## Next-action options for the user

1. **Authorise FA-style fused gated SDPA kernel** for the 16 full-attention layers (item 1 above). Largest single-kernel opportunity on the attention side; multi-session work; targets ~22% of step time.
2. **Authorise fused output-gate + o_proj** kernel (item 2 above). Medium scope; targets the same 22% but with broader kernel-launch consolidation.
3. **Authorise fused gated-delta-update + conv1d** kernel for DeltaNet (item 3 above). Largest leverage zone (74% of step time); multi-session work; depends on extending mlx-lm's existing `gated_delta_update` kernel.
4. **Authorise the C.5 γ.1 read-only kernel survey** (still pending from 2026-05-02 orientation, item A5).
5. **Authorise the MTP path re-opening** via upstream re-conversion (item A7 in orientation memo) — would reopen ranked hypothesis #6.

The autonomous loop will continue with smaller probes (e.g., layer-internal decomposition microbench to attribute MLP vs attention/recurrence within a single layer) without these authorisations, but the bigger needle-movers require the user to pick.
