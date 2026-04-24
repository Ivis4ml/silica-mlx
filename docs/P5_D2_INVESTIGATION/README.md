# P-5-D.2 investigation notes

## Context

P-5-D.1 (commit `2b3868d`, 2026-04-24) propagated the execution seed
from the bench runner into every codec factory's Haar rotation so
`--seeds 42,43,44` produces genuinely distinct codec instances. With
that fixed, a new observable surfaced on the (4-b) Acceptance sweep:
`qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` under silica's prefix-cache
store routinely reported ΔPPL vs fp16 around 5–10 PPL on Qwen3-0.6B
WikiText-2, while vqbench's `reproduce_qwen35_4b_headline.py` at the
same configuration reported ΔPPL below 1 PPL. The two oracles should
have produced the same quality observable for the same codec
configuration; the factor-of-10+ gap was the D.2 investigation
subject.

## What the probes measured

Each probe script runs standalone under `/Users/xinyu/miniconda3/bin/python`
(for the probes that require `torch` / `transformers`) or the project
venv (for the MLX-only probes). Inputs are real Qwen3-0.6B K activations
on a fixed WikiText-2 prefix.

- `p5_d2_probe.py` — silica BlockTQ Frobenius on pre-RoPE vs
  post-RoPE K from the SAME forward pass. Also reports silica-MLX vs
  vqbench-NumPy BlockTQ Frobenius on the same input (algorithmic
  parity check orthogonal to the RoPE split). Output:
  - pre-RoPE K  : 0.093
  - post-RoPE K : 0.092
  - silica MLX vs NumPy ratio: 1.000 (bit-identical)
  - Gaussian-assumption coord std: 0.02–0.08 vs theoretical
    `(1/head_dim)**0.5 = 0.088` — real Qwen K has lower per-coord
    variance than Lloyd-Max assumes, so the codebook is slightly
    miscalibrated; this effect is present on BOTH the pre-RoPE and
    post-RoPE paths at the same Frobenius.

- `p5_d2_probe2.py` — shared-rotation (silica convention) vs
  per-head-rotation (vqbench convention) Frobenius. Output:
  - shared total Frobenius: 0.092
  - per-head total Frobenius: 0.094
  - ratio shared / per-head: 0.977 — well below the measurement
    floor of the ΔPPL signal being debugged.

- `p5_d2_probe3.py` — full silica prefix-cache round-trip Frobenius
  (`insert_detached` → `fetch_detached_blocks` →
  `build_seeded_batch_kv`). Output: bit-identical to direct codec
  `encode_tensor` / `decode_tensor` at bf16; silica's cache
  machinery is numerically neutral.

- `p5_d2_probe4.py` — vqbench NumPy BlockTQ on the same real K
  input silica-MLX BlockTQ was measured against. Output: identical
  Frobenius to silica's MLX port — silica's MLX BlockTQ is not a
  defective port of vqbench's algorithm.

## Root cause

Same codec, same 9% Frobenius reconstruction error, but the error
vectors live in different bases:

- **post-RoPE space (silica's prefix-cache store path, the C.2
  production routing).** Noise is injected after RoPE has been
  applied, so the attention's relative-position coupling sees the
  noise through the RoPE rotation.
- **pre-RoPE space (vqbench's `_QuantizedProj`).** Noise is injected
  before RoPE, so the attention sees noise rotated *by RoPE*
  alongside the signal. The two rotations align; the net effect on
  downstream attention is an order of magnitude smaller than the
  post-RoPE injection at the same Frobenius.

A teacher-forced streaming PPL run chunks the sequence; at every
chunk boundary the C.2 oracle rebuilds a fresh `BatchKVCache` from
the codec-backed prefix blocks, so the noise accumulates per chunk.
The resulting ΔPPL is dominated by the number of chunk boundaries
the sequence crosses — at `chunk_size=512` (one chunk, zero
boundaries) the silica C.2 path reports a PPL bit-identical to fp16.
At `chunk_size=256` (two chunks, one boundary) ΔPPL jumps to several
PPL. vqbench's pre-RoPE injection does not exhibit this chunking
dependence because the noise rides through RoPE with the signal.

## Resolution (P-5-D.2a)

The fix is an additive oracle path, not a replacement of the
production routing. See `silica/bench/ppl_oracle.py`:
`teacher_forced_chunked_nll_vqbench_aligned`.

- **`codec_quality_path="prefix_store_post_rope"`** (existing C.2
  path) — kept. Produces the honest ΔPPL of the production
  prefix-cache store path. This observable does not have to match
  vqbench; it measures something the pre-RoPE path does not (the
  chunk-boundary cost of noise injection in post-RoPE space).
- **`codec_quality_path="vqbench_aligned"`** (new D.2a path) —
  monkey-patches `attn.k_proj` / `attn.v_proj` on the mlx-lm
  attention modules, mirroring vqbench's `_QuantizedProj`. The
  codec fires before RoPE, in the same space vqbench's harness
  injects noise, so the C.6 `--vqbench-xcheck` cross-check can
  bind against this path for an apples-to-apples comparison
  with vqbench. The Acceptance (4-b) rule — whether the residual
  gap is aggregated across seeds, what noise window is accepted,
  whether the existing per-row ε_ppl gate
  (`_compute_gap_fields` / `_VQBENCH_PCT_EPSILON`) applies as
  written — is owned by P-5-D.3, not by D.2a. The D.2a
  contribution is the algorithmic gap closure; the gate's
  numeric interpretation is a separate revision.

The scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`
carries `codec_quality_path="vqbench_aligned"`. Both arms remain in
the catalog; the JSONL reader can tell them apart via the
`codec_quality_path` metadata field.

## D.2a verification — 3-seed data (2026-04-24)

`d2a_verification_3seeds.jsonl` is the bench run that produced the
numbers below. Command:

```
PYTHONPATH=vqbench /Users/xinyu/miniconda3/bin/python scripts/bench.py \
    --scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned \
    --vqbench-xcheck \
    --vqbench-python /Users/xinyu/miniconda3/bin/python \
    --seeds 42,43,44 \
    --out /tmp/p5_d2a_verify.jsonl
```

Per-seed comparison against vqbench's reproduce script at the same
arm (`--method BlockTurboQuantMSE --block-size 64 --bits 4 --patch-v`):

| seed | silica ΔPPL | vqbench ΔPPL | ΔPPL gap (signed) |
|-----:|------------:|-------------:|------------------:|
|   42 |      +0.881 |       +0.270 |            +0.611 |
|   43 |      +0.174 |       +0.785 |            -0.610 |
|   44 |      +0.477 |       +0.929 |            -0.452 |

Mean / std summary:

- silica mean ΔPPL: +0.511 ± 0.354
- vqbench mean ΔPPL: +0.661 ± 0.347
- per-row |gap| worst-case: 0.611
- mean-aggregated gap: -0.150 PPL (-0.76% of fp16 baseline mean)

Compared with the original `prefix_store_post_rope` arm where silica
measured ΔPPL ~5–10 PPL against vqbench's ~0.3–0.9 PPL (a 10×–30× raw
gap depending on seed), the vqbench-aligned path collapses the
algorithmic gap to within one unit of seed-level noise: silica and
vqbench now produce ΔPPL values drawn from overlapping ranges
(silica [+0.17, +0.88] vs vqbench [+0.27, +0.93]) with closely
matched mean and std.

## What remains open

- **Per-seed agreement is not algorithmic parity.** silica's
  `BlockTurboQuantMSE` builds one Haar rotation shared across all
  heads (seeded by the codec ctor's `seed`). vqbench's
  `quant_dequant_tensor` runs `quantizer_factory(D, seed=h)` per
  head (seeded by the per-head index), so a `--seed 42` run on
  vqbench actually samples 8 independent rotations (one per
  `n_kv_heads=8`). At the same outer seed silica and vqbench are
  drawing different rotations from the same Haar distribution —
  hence the per-row `vqbench_divergence_warning=true` even when
  the silica and vqbench ΔPPL distributions overlap in mean and
  std (see the table above). A future port (D.2a+, or a knob on
  `BlockTurboQuantMSE`) that adopts per-head rotation would
  tighten per-seed variance by roughly `sqrt(n_kv_heads) ≈ 2.8×`
  under 8 heads — not eliminate the per-seed gap, since both
  sides still draw from the same Haar distribution. Whether the
  variance reduction is worth the ctor-signature change is a
  separate decision.

- **P-5-D.3 acceptance close.** With D.2a landed, the (4-b)
  Acceptance gate that was written as "ε_ppl < 0.01 abs AND < 0.1%
  rel" on a single row needs a reinterpretation step before it can
  close — the relevant comparison is mean-over-seeds, not per-row,
  because per-row variance is dominated by rotation draws rather
  than algorithmic drift.

- **Pre-RoPE production routing (future unit — P-5-E / P-3-C
  follow-up).** The D.2a path is an oracle-only measurement; the
  production prefix-cache store still sees post-RoPE noise. A
  future unit may explore a pre-RoPE KV store (position-aware
  codec) to close the quality gap on the serving hot path. This is
  an architecture change, not a bench-report change, and it is
  deliberately kept out of P-5 scope.
