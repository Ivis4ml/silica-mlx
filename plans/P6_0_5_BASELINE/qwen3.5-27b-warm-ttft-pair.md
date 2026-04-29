# silica-mlx bench report

Generated: 2026-04-29T11:15:32

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-ttft-pair |  | 1 | 1 | 0 | 0 | 317.2 | 17.2 | 170.7 | 15693.4 | 3.836 | 2 |  |

## Scenario details

### `qwen3.5-27b-warm-ttft-pair` (seed=0)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `warm_ttft_pair`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=2`
- status: **ok**

**P-6.0.5 sub-unit 6 (D-021 step 3) — warm-TTFT pair on dense 27B.** Two prompts issued sequentially through the same Engine; prompt 1 amortises kernel-compile cost, prompt 2's TTFT is the warm number reported by the §6 TTFT scenarios. Two distinct paragraphs hand-calibrated to ~128 BPE tokens each (asserted within ±15% by the catalog tokenizer test); distinct content keeps ``prefix_hit_tokens`` structurally 0 on the gate row. Dual-gated on SILICA_REAL_QWEN3_5_27B — same checkpoint as the existing dense 27B warm-decode rows. See plans/P6_0_5_OPENING.md §3.6.

Metadata:

```
{
  "codec_id": null,
  "compile_amortized_ms": 617.2467910218984,
  "prefix_hit_tokens": 0,
  "prompt1_tokens": 112,
  "prompt1_ttft_ms": 934.4152079429477,
  "prompt2_tokens": 115,
  "prompt2_ttft_ms": 317.16841692104936,
  "seed": 0,
  "warm_ttft_ms": 317.16841692104936
}
```

## Interpretation (P-6.0.5 sub-unit 6 — feeds §6(3) TTFT-under-concurrency gate, dense arm)

This row was run **three times** (see ``qwen3.5-27b-warm-ttft-pair.jsonl``,
3 rows). The repeated runs deliver an unexpectedly clean
variance-decomposition picture rather than a single point:

| run | prompt1_ttft_ms (cold) | prompt2_ttft_ms (warm) | compile_amortized_ms |
| --- | ---: | ---: | ---: |
| 0 | 725.3 | 317.0 | 408.3 |
| 1 | 590.8 | 316.5 | 274.3 |
| 2 | 934.4 | 317.2 | 617.2 |
| **mean ± std** | **750.2 ± 143** | **316.9 ± 0.3** | **433.3 ± 144** |

The **warm second-prompt TTFT is reproducible to ±0.1%** across
runs (316.5 / 317.0 / 317.2 ms) — exactly the steady-state UX
number a chat-CLI session would see on every message after the
first. Cold first-prompt TTFT swings 591–934 ms (rel-std ≈19%),
and compile_amortized inherits that variance directly (33%
rel-std) since it is a difference of cold − warm. This is the
oracle-design intent made empirical: warm TTFT is the gate-bearing
number, compile cost is OS / MLX-cache-state dependent and is
**not** a stable metric to anchor an SLA on. Headline: **dense 27B
warm TTFT = 317 ms** for a 115-token prompt (≈360 tok/s prefill
rate); **first-call compile cost ≈ 433 ms one-time amortised**.
Prompt symmetry is identical across all three runs (112 / 115
tokens, asymmetry 2.6%, ``prefix_hit_tokens=0``). Decode-tail
tok/s 16.6–17.2 (matching the B=1 baseline 16.05) confirms the
runs sat in the same bandwidth-bound regime as the
``warm-decode-b1`` row, so the warm TTFT is a same-regime
measurement. **Decision Gate 1 reads this row as:** the §6(3)
"TTFT under concurrency" gate has its dense steady-state anchor
at **317 ms warm + ~430 ms one-time compile**, with the warm
component reproducible to sub-millisecond precision — Track D
concurrency work reads this as the B=1 baseline. The compile-cost
variance is the operator-visible signal that cold-first-call
benchmarks need ≥3 runs to land a reliable mean.
