# silica-mlx bench report

Generated: 2026-04-29T11:13:32

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-moe-35b-a3b-warm-ttft-pair |  | 1 | 1 | 0 | 0 | 169.3 | 80.8 | 69.6 | 19818.9 | 5.235 | 2 |  |

## Scenario details

### `qwen3.5-moe-35b-a3b-warm-ttft-pair` (seed=0)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `warm_ttft_pair`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=4`, `prompts=2`
- status: **ok**

**P-6.0.5 sub-unit 6 (D-021 step 3) — warm-TTFT pair on MoE 35B-A3B.** MoE counterpart to ``qwen3.5-27b-warm-ttft-pair``; identical prompt pair so the cross-family warm-TTFT delta is a clean architecture-only signal. Dual-gated on SILICA_REAL_QWEN3_5_MOE — same checkpoint as the existing MoE warm-decode rows. See plans/P6_0_5_OPENING.md §3.6.

Metadata:

```
{
  "codec_id": null,
  "compile_amortized_ms": 1361.2097490113229,
  "prefix_hit_tokens": 0,
  "prompt1_tokens": 112,
  "prompt1_ttft_ms": 1530.4641660768539,
  "prompt2_tokens": 115,
  "prompt2_ttft_ms": 169.25441706553102,
  "seed": 0,
  "warm_ttft_ms": 169.25441706553102
}
```

## Interpretation (P-6.0.5 sub-unit 6 — feeds §6(3) TTFT-under-concurrency gate, MoE arm)

Warm second-prompt TTFT on MoE 35B-A3B: **169 ms** for a
115-token prompt (≈680 tok/s prefill rate) — about half of the
dense 27B counterpart's 317 ms warm TTFT. The cross-family delta
is the MoE active-3B advantage made concrete: MoE prefill reads
only the active expert subset per layer, while dense 27B prefill
streams the full 13.5 GB weight file, so on a clean warm path
MoE's prefill latency is structurally lower regardless of decode
regime. Cold first-prompt TTFT was 1530 ms with a **1361 ms
compile-amortised delta** — **3.1× the dense 3-run mean compile
cost (433 ± 144 ms)**, because MoE's first call has to compile
per-expert kernel variants alongside the standard prefill /
decode kernels. The dense compile cost has high run-to-run
variance (~33% rel-std on 3 runs), so the 3.1× ratio is
order-of-magnitude rather than a precise multiplier; an MoE
multi-run pair would tighten the comparison but the qualitative
finding (MoE compile noticeably heavier than dense) is robust. That
one-time cost is paid on chat-CLI process boot only; subsequent
messages see the 169 ms warm number. Prompt symmetry checks pass
identically to the dense pair (112 / 115 tokens, asymmetry 2.6%,
``prefix_hit_tokens=0``), so the cross-family comparison is a
clean architecture-only signal modulo a 3-token tokenizer drift.
Decode-tail tok/s 80.8 sits 6% above the B=1 baseline 76.01,
within the run-to-run noise band already observed on the
``warm-decode-b1-4k`` row (Unit 5: +12% on the same checkpoint).
**Decision Gate 1 reads this row as:** §6(3) MoE steady-state
TTFT anchor at **169 ms warm + 1361 ms first-call**; the MoE
warm-TTFT advantage over dense (317 ms) is a load-bearing input
to the (2b) reframing — even on chat-CLI single-message latency,
MoE wins. The first-call compile cost is the only operator-
visible MoE downside in this regime, and it amortises after one
prompt.
