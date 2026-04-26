# P-5 Acceptance (b-static) close — Qwen3.5-4B BlockTurboQuantMSE B=64 4-bit K+V

**Date:** 2026-04-26  **Result:** PASS

## Summary

silica's `block_tq_b64_b4` codec on Qwen3.5-4B WikiText-2 (first 512
tokens, chunk = 256, 3 seeds {42, 43, 44}) measures mean ΔPPL =
**+0.0016 PPL** (std 0.0212 PPL) — within seed-level noise of
vqbench/REPORT.md §3.1's reported `+0.000% ± 0.000%` static baseline
on the same codec configuration. The two-part aggregated gate
(`|mean_gap| ≤ 2 × SEM_diff` AND `|mean_gap| < 1.0` PPL) inherited
from PLAN §7 Acceptance (4-b) passes with significant margin.

## Setup

- **Model:** `Qwen/Qwen3.5-4B` (hybrid DeltaNet + GQA, head_dim = 256,
  8 full-attention layers + 24 linear DeltaNet layers).
- **Workload:** WikiText-2 first 512 tokens, `chunk_size = 256`
  (matches vqbench/REPORT.md §2.2 exactly: "WikiText-2 first 512
  tokens, chunk = 256").
- **Seeds:** {42, 43, 44} — the same three seeds vqbench's
  `scripts/variance_qwen35_4b.py` uses.
- **Codec:** `BlockTurboQuantMSE`, `vq_block_size = 64`, `num_bits =
  4`, K + V symmetric. silica codec id `block_tq_b64_b4`.
- **Routing:** post-P-5-F default `codec_quality_path =
  "prefix_store_pre_norm"` — K is captured pre-k_norm via the
  `PreNormCaptureAdapter` Protocol, persisted in the prefix store,
  reconstructed via `apply_k_norm_then_rope` on hit-path admit.
  Slice-regime active (`RecurrentStateAdapter + prefix_cache != None`
  on the hybrid adapter).
- **Bench command:**
  ```
  uv run python scripts/bench.py \
      --scenario qwen3.5-4b-wikitext-ppl-fp16 \
      --scenario qwen3.5-4b-wikitext-ppl-block-tq-b64-b4 \
      --seeds 42,43,44 \
      --out  docs/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_3seeds.jsonl \
      --report-md docs/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_3seeds.md
  ```

## Per-seed numbers

silica fp16 baseline PPL (seed-independent — fp16 forward is
deterministic):

| Seed | fp16 PPL |
| --- | --- |
| 42 | 8.855979 |
| 43 | 8.855979 |
| 44 | 8.855979 |

silica `block_tq_b64_b4` PPL:

| Seed | codec PPL | ΔPPL | ΔPPL %  |
| --- | --- | --- | --- |
| 42 | 8.880764 | +0.024785 | +0.280% |
| 43 | 8.839237 | −0.016742 | −0.189% |
| 44 | 8.852605 | −0.003374 | −0.038% |

silica aggregated:

- mean ΔPPL = **+0.001556 PPL**
- std ΔPPL = 0.021198 PPL
- SEM(ΔPPL) = std / √3 = 0.012238 PPL

vqbench/REPORT.md §3.1 row "Block B=64 4-bit K+V" (3 seeds {42, 43,
44}, monkey-patch fallback on the 8 full-attention layers per
REPORT §2.2):

- mean ΔPPL_pct = **+0.000%**
- std ΔPPL_pct = 0.000%
- mean ΔPPL = 0.000 PPL (PPL_fp16 = 10.3866; 0% × 10.3866 = 0)

## Gate

(4-b)-style two-part aggregated gate per PLAN §7:

- **mean_gap** = silica.ΔPPL_mean − vqbench.ΔPPL_mean = +0.001556 − 0
  = **+0.001556 PPL**.
- **SEM_diff** = √(SEM(silica.ΔPPL)² + SEM(vqbench.ΔPPL)²) =
  √(0.012238² + 0²) = **0.012238 PPL** (vqbench's static row is
  reported with 0.000% std, so its sample SEM contribution is 0).
- **Gate (i):** `|mean_gap| ≤ 2 × SEM_diff` ⇔ `0.001556 ≤ 0.024477`
  → **PASS** (~16× headroom).
- **Gate (ii):** `|mean_gap| < 1.0 PPL` ⇔ `0.001556 < 1.0` → **PASS**
  (~640× headroom).

Both conditions pass with significant margin. silica's MLX-native
`BlockTurboQuantMSE B=64` 4-bit K+V on Qwen3.5-4B is statistically
indistinguishable from vqbench's reported lossless-at-measurement-
precision baseline.

## Why silica's absolute fp16 PPL differs from vqbench's 10.3866

silica's fp16 PPL = 8.856; vqbench's REPORT.md reports 10.3866. Both
use WikiText-2 first 512 tokens, chunk = 256. The difference is
expected:

- **Tokenisation:** silica uses the Qwen3.5 mlx-lm tokenizer; vqbench
  uses HuggingFace transformers' tokenizer with `add_special_tokens`
  and BOS placement that differ in subtle ways. The token count is
  identical at 512 tokens but the boundary handling (chunk-prefix
  bytes vs character bytes) is not bit-identical.
- **Precision:** silica MLX runs bf16; vqbench runs torch.float16 +
  MPS — different accumulation precision per matmul.
- **Harness:** silica's `teacher_forced_chunked_nll` with
  `_run_pre_norm_oracle_inner` vs vqbench's evaluation harness
  (`scripts/variance_qwen35_4b.py` calls `evaluate_perplexity`).

The (b-static) gate is on **ΔPPL** (relative to each respective fp16
baseline), not absolute PPL — this is the right comparison because
the absolute baseline is harness-dependent but the relative codec
overhead is the codec-quality observable. silica's relative
overhead matches vqbench's, which is what the gate measures.

## Conclusion

**(b-static) closes.** silica's `block_tq_b64_b4` codec on
Qwen3.5-4B reproduces vqbench/REPORT.md §3.1's lossless-at-
measurement-precision finding under independent measurement: at the
same workload (WikiText-2 first 512 tokens, chunk = 256, 3 seeds
{42, 43, 44}), silica's mean ΔPPL is statistically
indistinguishable from vqbench's reported `+0.000%`. The two-part
aggregated gate inherited from (4-b) passes with ~16× headroom on
the SEM gate and ~640× headroom on the absolute-PPL gate.

This evidence justifies flipping PLAN §7 P-5 Acceptance (b-static)
from "post-P-5 required follow-up; blocked on P-3-C cooperation
work" to "closed on the production hot path via P-5-F (3b)
projection-output capture path".

## Evidence files

- `qwen35_4b_b_static_3seeds.jsonl` — raw bench rows.
- `qwen35_4b_b_static_3seeds.md` — auto-generated bench report.
- `qwen35_4b_b_static_close.md` — this analysis.
