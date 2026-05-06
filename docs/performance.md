# Performance — P-6 performance phase closed at v1.7.28

```{image} _static/p6/decode-dense.png
:alt: Dense Qwen3.5-27B-4bit decode running-best across 35 cycles
:width: 100%
```

*Karpathy-style autoresearch ledger for dense Qwen3.5-27B-4bit
decode — each dot is a measurement, the ladder is the running
best. 38 experiments across 35 cycles, 9 KEEPs. Baseline 42.17
tok/s @ B=4 (cycle 1) → 232 tok/s @ B=64 (cycle 28).*

```{image} _static/p6/decode-moe.png
:alt: MoE Qwen3.5-35B-A3B-4bit decode running-best across cycles 34-35
:width: 100%
```

*MoE Qwen3.5-35B-A3B-4bit ledger — same lever stack (cycles 12 +
13) transferred via the shared `gated_delta` shadow patch.
11 experiments, 2 KEEPs. Baseline 188.5 tok/s @ B=4 → 791.8 tok/s
@ B=128 (cycle 35) — the largest absolute throughput observed
across the full effort.*

:::{admonition} Single-user reality
:class: warning

P-6 is a *server-throughput* phase. All gains route through
batch size: per-row throughput moves the *opposite* way —
10.5 tok/s/row at B=4, 3.92 tok/s/row at B=52, 3.62 tok/s/row at
B=64. **Single-user (B=1) latency on M5 Pro is bandwidth-capped
near 20 tok/s and unchanged by this phase.** D-022 then tested
the small-B single-user line and closed at v1.7.28: B=4 stayed at
10.29 tok/s/row, β/γ/δ all reached measurement-anchored negatives,
and the remaining Python-hygiene headroom projected ≤0.6% E2E.
:::

| Configuration | Aggregate | Per row | Frame |
| --- | ---: | ---: | --- |
| B=1 (single user) | ~20 tok/s | ~20 tok/s | bandwidth ceiling, derived |
| B=4 (cycle-1 baseline) | 42.17 tok/s | 10.54 tok/s | 52% bandwidth utilisation |
| B=52 (best within 36 GB) | 204 tok/s | 3.92 tok/s | strict envelope |
| B=64 (48 GB ceiling) | 232 tok/s | 3.62 tok/s | hardware cap |

The 35-cycle opus autoresearch loop pushed Qwen3.5-27B-4bit warm
decode 5.50× over the cycle-1 baseline on M5 Pro 48 GB **at the
server-aggregate level**. Two load-bearing levers — `cycle-10`
batched-aggregate axis-shift (B=4 → B=52) and `cycle-12` bf16
DeltaNet recurrent state (3.5 GB peak save opens B≥48 within the
36 GB envelope) — composed to clear all four P-6 acceptance gates
3.4-5.5× over baseline. **17 custom Metal kernel attempts closed
without a load-bearing E2E win**; the unlock came from data layout
(bf16 state) and operating-point selection (axis-shift). None of
these levers move B=1 single-user latency, which sits at the
chip's weights-only bandwidth ceiling on this stack. D-022 followed
that motivation and closed the small-B line with measurement-anchored
negatives: β had too little reachable scope, γ had too little per-call
gain, and δ found the apparent overhead bucket was mostly real compute.

## Acceptance gates — every server-throughput P-6 gate cleared

| Gate | Target | Cleared | Multiplier | Frame |
| --- | --- | --- | ---: | --- |
| (1a) Dense engineering | ≥40 tok/s | 204 ± 1 tok/s | **4.85×** | B=52 · 36 GB envelope |
| (1b) Dense stretch | ≥60 tok/s | 231.9 ± 0.3 tok/s | **3.87×** | B=64 · 48 GB hardware ceiling |
| (2a) MoE anchor | ≥100 tok/s | 120.93 tok/s (preserved) | — | MoE B=2 · v1.7.13 baseline |
| (2b) MoE stretch | ≥175 tok/s | 791.8 ± 5.2 tok/s | **4.52×** | MoE B=128 · 48 GB hardware ceiling |

<details>
<summary><strong>Headline numbers</strong> — full table with envelope vs hardware-ceiling framing</summary>

| Metric | Value | Frame |
| --- | ---: | --- |
| Dense 27B decode (envelope) | **204 ± 1 tok/s** | `mlx-community/Qwen3.5-27B-4bit` · B=52 · 36 GB · n=6 across 2 sessions |
| Dense 27B decode (hardware ceiling) | **231.9 ± 0.3 tok/s** | B=64 · 48 GB · n=3 |
| MoE 35B-A3B decode (envelope) | **464.1 ± 0.7 tok/s** | `mlx-community/Qwen3.5-35B-A3B-4bit` · B=64 · 33.8 GB · n=3 |
| MoE 35B-A3B decode (hardware ceiling) | **791.8 ± 5.2 tok/s** | B=128 · 47.96 GB · n=3 |
| Dense uplift from cycle-1 baseline | **5.50×** at B=64 / **4.85×** at B=52 | vs 42.17 tok/s @ B=4 |
| MoE uplift from cycle-1 baseline | **4.20×** at B=128 / **2.46×** at B=64 | vs 188.5 tok/s @ B=4 |

</details>

<details>
<summary><strong>Two load-bearing levers</strong> — running-best is composition, not a kernel</summary>

- **Cycle 10 — batched-aggregate axis-shift.** Re-reading the
  P6_AUTORESEARCH.md metric definition ("B is chosen to maximise aggregate")
  moved the operating point from B=4 → B=48 within the 36 GB
  envelope. Pure parameter selection; no kernel change. *4.60× on
  its own.*
- **Cycle 12 — bf16 DeltaNet recurrent state.** State shape
  `[B, Hv=48, Dv=128, Dk=128]` is 144 MB at fp32 per layer,
  72 MB at bf16. Across 48 DeltaNet layers the peak-memory save
  is ~3.5 GB — opens B≥48 within envelope and unlocks B=64 at
  hardware ceiling.
- **Cycle 13 — composition.** Cycle-12's peak save composed with
  cycle-10's B-axis lever produces the running-best line. The two
  levers are independent; together they dominate every later
  atomic probe.

Cycle 30 explained why kernel work did not pay back: at B=64 with
the v10+bf16 stack, DeltaNet owns 88% of step time, full-attn
12.5%, dispatch 0.3% — and mlx's existing `gated_delta` kernel
is already at HBM-bandwidth limit (cycle-31 silica
`gated_delta_v2` = 1.001× vs mlx). Source-string Metal kernels
in mlx 0.31.x do not pay back on dense 27B.

</details>

<details>
<summary><strong>Honest record</strong> — closures and a retraction</summary>

- **Spec-decode at production B — closed with measurement-anchored
  negative.** Cycle 23 measured the `B × k` verify-cost matrix:
  B=52 k=64 = **8105 ms** versus same-B plain-decode ~252 ms /
  step. Tree-spec at b=64 recomputes to ~10 tok/s aggregate, a
  20× regression vs plain decode. No B regime in {1, 4, 16, 52}
  where spec-decode beats plain on this stack. Track C settles:
  C.4 retired (η.1 = 0.482×), C.5 retired (cycle-23 closure),
  C.1 / C.2 / C.3 / C.6 deprioritised since (1b) no longer needs
  them.
- **Dense B-axis past 64 — closed at the architectural cliff.**
  Cycles 28-29 measured a 26% throughput drop at the
  B=64 → B=66 transition (40 GB peak boundary). Three
  allocator-hint probes
  (`mx.metal.set_cache_limit / set_memory_limit / set_wired_limit`)
  leave the cliff in place — architectural, not allocator policy.
- **Cycle 14's claimed v10 KEEP — retracted via codex review.**
  A codex cross-review on the `opus-codex` branch caught a
  14-cycle dtype-defect in `silica.kernels.shadow_install`
  (`queries.dtype == mx.float16` silently skipped the bf16
  production path). After the fix, an 8-rep reverify (cycles
  27 / 28) measured v10's E2E contribution at +0.5 tok/s @ B=52 /
  −1.7 tok/s @ B=64 — both within noise. The honest running-best
  is C10 axis-shift × C12 bf16 DeltaNet state composition alone;
  the retracted KEEP is published as part of the research record.

</details>

<details>
<summary><strong>Methodology</strong> — Karpathy-style ledger and variance discipline</summary>

Karpathy-style autoresearch ledger (one TSV row per measurement;
the main agent appends, sub-agents return findings). Variance
discipline: ≥3 reps per session, ≥2 sessions, combined σ check
before declaring a KEEP. Cycle 33's combined σ at B=52 across
2 sessions tightened to 0.83 tok/s on n=6 — the protocol
standard, not the exception. Toolchain pin: `mlx==0.31.1`,
`mlx-lm==0.31.2`, `mlx-metal==0.31.1`. Determinism gate:
`tests/test_p2_preload_parity.py` (3/3 pass).

</details>

<details>
<summary><strong>Reproducibility</strong> — three commands</summary>

```bash
# Dense 27B within 36 GB envelope (running-best 204 tok/s)
SILICA_USE_BF16_DELTANET_STATE=1 \
    SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b52 \
        --out /tmp/repro_b52_bf16.jsonl

# Dense 27B at 48 GB hardware ceiling (running-best 232 tok/s)
SILICA_USE_BF16_DELTANET_STATE=1 \
    SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b64 \
        --out /tmp/repro_b64_bf16.jsonl

# MoE 35B-A3B at 48 GB hardware ceiling (running-best 791.8 tok/s)
SILICA_USE_BF16_DELTANET_STATE=1 \
    SILICA_REAL_QWEN3_5_MOE=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-moe-35b-a3b-warm-decode-b128 \
        --out /tmp/repro_moe_b128.jsonl
```

n=3 reps recommended per scenario. Combined σ is typically
~1 tok/s on dense B=52 and ~5 tok/s on MoE B=128 in the same
environment with proper warm cache.

</details>

## Cycle deliverables — what each cycle produced

```{image} _static/p6/cycles.png
:alt: Per-cycle deliverables timeline across 35 cycles
:width: 100%
```

*Per-cycle deliverables: kept (running-best moved), discarded
(no improvement vs prior best), or correction (cycle 27
retraction). Five cycles produced lasting load-bearing changes:
C10 (axis-shift), C13 (composition KEEP), C27 (correction), C34
(MoE portability), C35 (MoE hardware ceiling).*

## D-022 small-B interactive QoE — closed at v1.7.28

The cleared P-6 gates were aggregate-throughput at high B, so D-022
tested whether per-step latency at small B (B ∈ {1, 2, 4, 8, 12})
had any load-bearing lever left on mlx 0.31.x. It did not. The line
closed with one diagnostic baseline and three measurement-anchored
negative conclusions:

| Sub-unit | Terminal state | Physical close reason | E2E vs gate |
| --- | --- | --- | --- |
| α | complete (v1.7.25) | sonnet baseline + bucket decomposition | diagnostic |
| β | NEGATIVE (v1.7.26) | compile-reachable scope too narrow (~3-4% step) | 0.25-0.41% / 3% |
| γ | NEGATIVE (v1.7.27) | per-call `mx.compile` gain too small (1.011×) | 0.51% / 3% |
| δ | NEGATIVE-on-audit (v1.7.28) | 3.6% overhead bucket is 70-90% real compute | ≤0.6% / 2% |
| ε | upstream waitlist | mlx 0.32+ async-copy re-open trigger | n/a |

Goal framing stayed interactive single-row latency / TTFT, **not**
throughput parity. Per-row throughput at B=4 (~10.5 tok/s/row from
the cycle-1 baseline, 10.29 tok/s/row in the sonnet refresh) already
exceeds per-row at B=52 (~3.92 tok/s/row); the throughput-parity frame
is structurally inverted.

See {doc}`plans-index` for the D-022 opening and closure artefacts.

## Read more

- `plans/P6_AUTORESEARCH_NOTES.md` — durable take-home companion.
- `plans/P6_AUTORESEARCH_FINAL_REPORT.md` — comprehensive 23-cycle
  final report.
- `P6_AUTORESEARCH.md` — autoresearch directive + addendums.
- `plans/P6_AUTORESEARCH_LOG.tsv` — raw 110-row Karpathy-style
  ledger.
- `plans/P6_AUTORESEARCH/` — per-cycle reports, JSONL artefacts,
  and progress charts.
- `plans/P6_SMALL_B_OPENING.md` and
  `plans/P6_SMALL_B/DELTA/PRE_PROJECTION.md` — D-022 small-B
  opening and closure audit.
