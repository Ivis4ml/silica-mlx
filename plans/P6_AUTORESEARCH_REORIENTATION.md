# Silica-MLX Autoresearch Reorientation + Hardware Limit Map + External Radar

| Field | Value |
| --- | --- |
| Date opened | 2026-05-02 |
| Branch | `opus` |
| Last commit at orientation | `309af8c` (D-021 step 8 beta measurement bundle — escalate per C.5 matrix) |
| Status | Orientation — doc only. No silica.* code change in this commit. No final-disposition PLAN edits. |
| Predecessor | D-021 step 8 (β.1 + β.2) escalate state at v1.7.22-pending |
| Audit chain | v1.7.20 C.4 retire (FAIL @ 0.482×) → v1.7.21 Track B retire (PPL FAIL +16.85% rel) → v1.7.21+ Track B follow-up survey closed empty → 309af8c C.5 β escalate; **(1b) ≥60 tok/s leg-i dead, leg-ii in escalate**. |
| Custom-kernel authorisation | Granted by user 2026-05-02 for autoresearch phase. PLAN.md §3.2 "no hand-rolled Metal kernels from scratch" relaxed. PLAN.md edit deferred until a kernel passes its end-to-end gate. |

---

## 0. TL;DR

This memo opens an autoresearch loop. Three findings shape the loop:

1. **The hardware-limit map's B=1 ceiling number in the prompt is stale.** P-6.0.5 corrected the dense Qwen3.5-27B-4bit weight footprint from 13.5 GB to **15.13 GB** (runtime-measured `mlx.utils.tree_flatten`). The B=1 weights-only ceiling is therefore **20.29 tok/s**, not 22.7 tok/s. All bandwidth-utilisation ratios in this memo use the corrected anchor; the B=4 aggregate ceiling (4× B=1 amortised) is **81.16 tok/s**, of which Silica currently realises 42.17 ≈ 52%.

2. **Qwen3.5-27B is a hybrid model, not pure dense — verified directly against the cached `config.json`.** 64 total decoder layers; the layer-type pattern is `linear_attention, linear_attention, linear_attention, full_attention` repeating (`full_attention_interval: 4`); **48 linear-attention (Gated DeltaNet) layers interleaved with 16 full Gated Attention layers**. Full-attention `head_dim = 256` (not 128 as the sub-agent surveys assumed), GQA ratio 24:4 = 6:1 KV-head sharing. **`attn_output_gate: true` — Qwen3.5 attention is Gated Attention with an output gate, not vanilla SDPA**, which materially changes the FlashAttention-style kernel candidacy: any custom kernel must implement `out = sigmoid(g) * SDPA(Q,K,V)` not just SDPA. `partial_rotary_factor: 0.25` (RoPE applied to only 25% of head_dim). The checkpoint is multimodal (333 vision-tower tensor keys, depth=27); per PLAN §3.2 D-014 the text-only path is in scope. The interleaved layer ordering (not "block of 48 then block of 16") means kernel work cannot batch-fuse "all attention layers" — they are dispatched between DeltaNet layers. Three downstream consequences: (a) custom-kernel candidates split, with the GatedAttention output-gate baked into the SDPA candidate; (b) the high rollback cost C.4 and C.5 β.1 measured is partly DeltaNet recurrent-state replay, which makes spec methods that assume cheap rollback (NodeNestor llama.cpp MTP, EAGLE-3 family) net-negative on this checkpoint; (c) Silica's existing `SpecRecurrentRollbackAdapter` + `Qwen3_5Adapter.snapshot_pre_draft_state` / `rollback_state` is precisely the infra that makes hybrid-aware spec rollback tractable.

3. **The verify-k microbench shows the dense decode is compute-bound at k≥4 / B≥4.** P-6.0.5 Unit 7 measured bandwidth utilisation dropping 83% (k=1) → 79% (k=2) → 55% (k=4) → 30% (k=8); Unit 2 measured B=1 → B=4 utilisation drop 79% → 52%. The two regimes cross at the same ~52-55% line, which means the 48% headroom on B=4 weights is **not pure bandwidth idle** — it is bounded by candidate-side compute (16 attention layers × 4 batch + 48 DeltaNet recurrent updates × 4 batch per step + per-step Python and small-op cost). Custom MLX kernels that reduce per-step compute (FlashAttention-style fused SDPA, fused RMSNorm+RoPE, fused gated-delta-update) directly attack this wall and are the single most measurement-anchored open lever today.

The recommended next probe is therefore **a microbench harness on the existing dense Qwen3.5-27B-4bit warm-decode B=4 path that decomposes per-step time into per-layer-kind components via per-layer `mx.eval` timing hooks** (full-attention layers, DeltaNet linear-attention layers, RMSNorm+RoPE small-op chains, logits projection, Python loop / scheduler overhead), on cached weights. The methodology uses **per-layer barriers, not layer-skipping**, because skipping DeltaNet layers in a hybrid model breaks the recurrent-state pipeline that subsequent layers depend on. No download, no commit, no destructive op, fits the autonomous-loop budget. The microbench either identifies a kernel-side gap that justifies opening a custom MLX-native kernel candidate (priority 3 in the P6_AUTORESEARCH.md hardware-limit ladder), or it shows the time is dominated by stock MLX ops at their achievable ceilings, in which case the bottleneck moves to the spec-composition leg (priority 2).

**The MTP probe path retires for the current production target** (`mlx-community/Qwen3.5-27B-4bit`): `safetensors.index.json` inspection (2026-05-02) found **zero tensor keys matching `mtp|nextn|multi_token|multistep`** in the 2180-key index, even though `config.json:mtp_num_hidden_layers = 1` says the architecture supports MTP. The production checkpoint ships without MTP weights. Re-opening the MTP path requires either downloading the upstream Qwen/Qwen3.5-27B BF16 (~52 GB) and re-converting, or downloading `trevon/Qwen3.5-27B-MLX-MTP` (29.1 GB) — both gated on explicit user authorisation. The QuantSpec and KnapSpec probes (priority 5) and the C.5 γ.1 read-only kernel survey (priority 1 of the existing escalate path) are queued behind the kernel microbench.

---

## 1. Current State

### 1.1 What is built (silica.* surface)

- `silica/engine/__init__.py` — `Engine.generate` (single-request) and `Engine.generate_batch` (multi-request) live; spec verify path with capture wired (D-021 step 5 closure at v1.7.19); `decode_step_multi(k)` available across production adapters.
- `silica/scheduler/batcher.py` — `ContinuousBatcher` with continuous batching, prefix cache pinning, admission ordering. **GLOBAL-only gate at lines 268-279 preserved**: multi-request hybrid batched-spec path (D-021 step 5 (c) slice 3) is deferred. B=1 single-request spec path is the only spec path exercised by current scenarios.
- `silica/kvcache/` — `SimpleKVCache` (per-request mlx-lm wrapper), `BatchKVCache` (multi-row continuous batching with right-padding primitive), `PagedKVCache` (Q-009 bookkeeping only, no Metal scatter/gather kernel; physical K/V remains contiguous), `RadixPrefixCache` (trie-based prefix dedup with `register_detached` / `fetch_detached` for codec-aware blocks).
- `silica/speculative/` — `DraftEngine` Protocol; `NoopDraftEngine` (default); `DraftTargetEngine` (C.1 baseline, single-request; per-`req_id` `_MultiKVCache`); `DFlashDrafter` (C.4 retired but code present, `TargetHiddenConsumer` Protocol mixin); spec-metrics schema `SPECULATIVE_METRIC_FIELDS` emitted into `ScenarioResult.metadata` via `SpecMetricCollector`.
- `silica/models/qwen3_5.py` — Hybrid Qwen3.5 adapter. `is_linear` flag distinguishes DeltaNet (recurrent) layers from full-attention layers. Pre-norm K capture proxies installed on full-attention layers only (P-5-F F.1). `snapshot_pre_draft_state` / `rollback_state` for recurrent rollback (D-021 step 5 (e)).
- `silica/bench/` — `BenchRunner`, scenario catalog (≥70 scenarios after D-021 step 7 B.2 PPL rows), oracles (warm-decode, PPL, codec-quality, target-verify-microbench), `--speculative {none,draft_target,dflash}` CLI flag, JSONL output.
- `silica/bench/microbench/target_verify.py` — verify-k microbench (Unit 7), measures `forward(model, candidate, cache_list)` cost in isolation with prefix primed; emits {forward_ms_p50/p95, peak_memory_mb, weight_bytes_read_estimate, ...}.
- `scripts/probe_c5_top_b_coverage.py` — read-only top-b coverage probe (β.2); two parallel teacher-forced forwards with rank comparison; tokenizer-alignment guard; emits coverage@b for b ∈ {1, 4, 8, 16, 32} and a per-position rank histogram. **Pattern for any future read-only probe.**
- `scripts/microbench_capture_hidden.py` — reused-adapter microbench harness; isolates `c_capture_hidden(k)` cost by running `decode_step_multi` vs `decode_step_multi_with_capture` over the same k. **Pattern for any future per-cycle cost-isolation microbench.**

Source map detail in the orientation transcript (Explore agent walk; section 8 "Kernel Candidacy Attribution Table" of the source map).

### 1.2 What is tested

- ≥2640 tests pass under `SILICA_SKIP_MODEL_TESTS=1` (post-B.2: 2647). Real-model tests are env-gated per checkpoint (target / drafter / 3-bit / DFlash / OptiQ / etc.) and quad-gated for spec rows.
- ruff clean on silica + tests + scripts; mypy clean on the silica package (~85 source files).
- 26 unit tests for `probe_c5_top_b_coverage.py` covering rank / coverage / tokenizer-alignment / JSONL schema; 19 ζ bench-wiring tests for the C.4 path; spec-rollback tests bind three rollback paths under synthetic patterns A/B/C.

### 1.3 What is measured (load-bearing rows)

Re-anchored hardware-limit map for dense Qwen3.5-27B-4bit on M5 Pro 48 GB (corrected against `plans/P6_0_5_BASELINE/REPORT.md` and downstream reports):

| Anchor | Value | Source | Note |
| --- | --- | --- | --- |
| Weight footprint (runtime, `tree_flatten`) | **15.13 GB** | P-6.0.5 REPORT § Weight-footprint reconciliation | corrects the v1.7.13 P-6.0 13.5 GB anchor and the AR-prompt's 13.5 GB number |
| B=1 weights-only bandwidth ceiling | **20.29 tok/s** = 307 / 15.13 | derived | the AR-prompt's "~22.7 tok/s" assumes the 13.5 GB anchor |
| B=1 measured | 16.05 tok/s @ 79.1% util | P-6.0 baseline | unchanged number, but % util now reads against the corrected anchor |
| B=2 measured | 31.22 tok/s @ 76.9% util | P-6.0.5 Unit 1 | 97% of ideal 2× linear |
| **B=4 measured (current best)** | **42.17 ± 0.21 tok/s @ 52.0% util** | P-6.0.5 Unit 2 (2-run) | 66% of ideal 4× linear; primary metric anchor |
| B=4 weights-only aggregate ceiling | 81.16 tok/s = 4 × 20.29 | derived | aggregate ceiling at 100% util |
| Run-to-run σ at B=4 | **0.21 tok/s** (0.5% rel-std) | P-6.0.5 Unit 2 | 3σ floor for keep on primary metric = ±0.63 tok/s |
| Within-run rel-std (B=4) | ~0.5% | P-6.0.5 per-row metadata | warm-decode oracle stability |
| Verify-k = 1 cost | 59.59 ms @ 82.7% util | P-6.0.5 Unit 7 | comparable regime to B=1 decode |
| Verify-k = 2 cost | 62.07 ms @ 79.4% util; +2.48 ms marginal | Unit 7 | bandwidth-bound regime |
| Verify-k = 4 cost | 89.01 ms @ 55.4% util; +9.81 ms/tok | Unit 7 | enters compute-bound regime |
| Verify-k = 8 cost | 162.96 ms @ 30.2% util; +14.77 ms/tok | Unit 7 | compute-bound |
| Verify-k zero-drafter ceiling at k=8 | **2.93×** | Unit 7 | linear-shape upper bound on spec-only speedup |
| Warm TTFT (B=1, ~115-token prompt) | 316.9 ± 0.3 ms (3-run) | P-6.0.5 Unit 6a | reproducible to ±0.1% |
| Cold TTFT incl. compile | 750 ms ± 143 ms (3-run) | Unit 6a | compile cost amortised after first run |
| RAM peak at B=4 | 17.10 GB | Unit 2 | 53% margin to §6(4) 36 GB envelope |

Secondary track (MoE Qwen3.5-35B-A3B-4bit):

| Anchor | Value | Source |
| --- | --- | --- |
| B=1 / B=2 / B=3 / B=4 aggregate | 76.01 / 120.93 / 163.50 / **188.50** tok/s | P-6.0 + P-6.0.5 |
| B=4 utilisation on 1.5 GB active-weight anchor | 92.1% | P-6.0.5 Unit 4 |
| 4K-context B=1 peak | 23.60 GB | Unit 5 |
| Warm TTFT | 169.3 ms (1 run) | Unit 6b |
| (2a) ≥100 tok/s cleared since v1.7.13 | yes (B=2 = 120.93) | P-6.0 |
| (2b) ≥175 tok/s aggregate at B≥3 | cleared (B=4 = 188.5; B=3 = 163.5 falls short) | Decision Gate 1 §3.4 |

**Run-to-run σ on the MoE row family is not characterised cross-run** (single-run Unit 4 measurement). Any future MoE keep needs at least one reproduction before it enters the running-best line.

Failed-track measurements (full postmortem in §2):

| Track | Disposition | Key numbers |
| --- | --- | --- |
| C.4 DFlash | retired v1.7.20 | spec-on 7.74 tok/s = **0.482× vs 16.05**; α=0.0881; draft 35.7 ms ≫ verify 2.45 ms; rollback 165/cycle; peak 19.04 GB |
| Track B native 3-bit | retired v1.7.21 | PPL 4-bit 6.91 / 3-bit 8.07; **ΔPPL_abs +1.16 (gate ≤ 0.5); ΔPPL_rel +16.85% (gate ≤ 5%)**; B.3 not run |
| Track B follow-up survey | closed empty 2026-05-01 | no activation-aware Qwen3.5-27B 3-bit MLX checkpoint exists |
| C.5 DDTree (β.1 / β.2) | escalate state per §9 matrix | β.1 spec-on 6.54 tok/s = **0.408×**; α=0.0908; coverage@4=0.143, @8=0.198, **@16=0.258**, @32=0.341; rollback 296/cycle |

### 1.4 What is unresolved (open questions before next-probe selection)

1. **Where does the 48% bandwidth headroom at B=4 go?** Is it (a) compute-bound on candidate-side ops (consistent with the verify-k regime transition at k=4), (b) per-step Python loop / small-op cost (sampling, broadcast, mlx async/sync), (c) DeltaNet recurrent update cost (48 layers × 4 batch per step), or (d) some mix? The microbench in §7 below decomposes this via per-layer `mx.eval` timing hooks.

2. **Is the verify-forward bottleneck at k=4 the same one as the warm-decode-B=4 bottleneck?** Both regimes hit ~52-55% util on the corrected anchor — strong same-cause hint, but verify-k holds B=1 fixed whereas decode-B holds k=1 fixed. A B=2 verify-k=2 cell would close the question. Easy microbench extension.

3. **What fraction of step time on dense 27B B=4 lives in the 16 full-attention layers vs the 48 DeltaNet layers?** Silica's Qwen3_5Adapter has the layer-iteration split available at silica/models/qwen3_5.py:100-104; per-layer `mx.eval` barrier instrumentation can attribute time by layer kind. This is the kernel-candidacy load-bearing question — without this attribution, custom-kernel proposals are blind picks.

4. ~~**Does `mlx-community/Qwen3.5-27B-4bit` preserve the MTP head?**~~ **RESOLVED 2026-05-02 — NO.** Inspection of `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-27B-4bit/snapshots/45797d2985a12c55e6473686e9ea91b95e959553/model.safetensors.index.json` returned **0 keys matching `mtp|nextn|multi_token|multistep`** out of 2180 total tensor keys (1840 in `language_model.model.layers`, 333 in `vision_tower`, 7 misc). `config.json:mtp_num_hidden_layers = 1` says the architecture supports MTP, but the production checkpoint ships without MTP weights. **The MTP probe path retires for this checkpoint without explicit user authorisation to re-convert from upstream.**

5. **Is the C.5 escalate disposition retire-or-survey-γ.1?** The C.5 REPORT recommends retire (1b) unless user authorises a γ.1 read-only survey of `humanrouter/ddtree-mlx` for sub-linear verify-cost evidence at T=32. This memo does not pre-empt that decision.

6. **MoE σ unmeasured.** Cross-run variance on the MoE B=4 row is unknown. A second run would let MoE wins enter the running-best line under the same 3σ rule applied to dense.

7. **Can the (c) slice 3 multi-request hybrid spec gate be lifted to compose B>1 with spec?** The spec foundation (D-021 step 5) is single-request; multi-row spec is structurally a separate orientation, deferred. Composing a +1.5× spec lever with the ~1.6× bandwidth-utilisation lever requires this gate lifted.

### 1.5 Verified architecture facts (config.json + safetensors-index inspection, 2026-05-02)

Direct evidence from the cached `mlx-community/Qwen3.5-27B-4bit` snapshot at `45797d2985a12c55e6473686e9ea91b95e959553`. These supersede sub-agent-derived claims where they differ.

| Field | Value | Note |
| --- | --- | --- |
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` | multimodal — text + vision tower |
| `model_type` | `qwen3_5` | |
| `text_config.num_hidden_layers` | 64 | confirmed (Track B B.1 also reported 64) |
| `text_config.layer_types` pattern | `[linear, linear, linear, full]` × 16 | full attention every 4th layer |
| Linear-attention (DeltaNet) layer count | **48** | counted from `layer_types` array |
| Full-attention layer count | **16** | counted from `layer_types` array |
| `text_config.full_attention_interval` | 4 | matches the layer pattern |
| `text_config.attn_output_gate` | **true** | Qwen3.5 attention is Gated Attention; output-gate sigmoid required in any custom SDPA kernel |
| `text_config.head_dim` | **256** | (NOT 128 — the sub-agent surveys' "head_dim=128" claim was for the linear-attention head only) |
| `text_config.num_attention_heads` | 24 | full-attention Q heads |
| `text_config.num_key_value_heads` | 4 | GQA ratio 24:4 = 6:1 KV-head sharing |
| `text_config.linear_num_key_heads` | 16 | DeltaNet K heads |
| `text_config.linear_num_value_heads` | 48 | DeltaNet V heads |
| `text_config.linear_key_head_dim` | 128 | DeltaNet head dim |
| `text_config.linear_value_head_dim` | 128 | DeltaNet head dim |
| `text_config.linear_conv_kernel_dim` | 4 | DeltaNet 1D conv kernel size |
| `text_config.hidden_size` | 5120 | |
| `text_config.intermediate_size` | 17408 | FFN dimension |
| `text_config.vocab_size` | 248320 | (the C.5 β.2 report's 248044 is the tokenizer-side vocab; 248320 is the embedding rows including reserved special tokens) |
| `text_config.partial_rotary_factor` | 0.25 | RoPE applied to only 25% of head_dim |
| `text_config.rope_parameters.rope_theta` | 10000000 | 10M, not the usual 10K |
| `text_config.rope_parameters.mrope_interleaved` | true | multi-modal RoPE for vision integration |
| `text_config.max_position_embeddings` | 262144 | 256K context support |
| `text_config.mtp_num_hidden_layers` | 1 | architecture supports MTP head |
| `text_config.mtp_use_dedicated_embeddings` | false | |
| `quantization.group_size` | 64 | confirms the ecosystem-survey-noted gap (no group_size<64 4-bit MLX checkpoint exists) |
| `quantization.bits` | 4 | |
| `quantization.mode` | affine | |
| Vision tower depth | 27 layers | |
| Tensor key count | 2180 total | 1840 `language_model.model.layers.*` + 333 `vision_tower.*` + 7 misc |
| **MTP / NEXTN / multi_token / multistep keys in safetensors index** | **0 / 2180** | resolves OQ-4 — MTP path retires for this checkpoint |

---

## 2. Failed-Path Postmortem

### 2.1 C.4 DFlash spike (retired v1.7.20)

**Headline:** measured 0.482× silica-integrated speedup on dense 27B-4bit, vs ≥1.8× engineering-continue floor.

**Three findings (source: `plans/P6_C4_DFLASH/REPORT.md` (η.1) + PLAN.md v1.7.20 changelog):**

1. **Drafter cost dominates verify cost by 15×.** `draft_cost_ms = 35.70` vs `verify_cost_ms = 2.45`. Two compounding causes: (a) the `z-lab/Qwen3.5-27B-DFlash` drafter is 2B parameters in BF16 (~3.4 GB on disk), and per-forward at this size on M5 Pro takes ~35 ms; (b) the verify forward against the 4-bit target reads 15.13 GB once and produces verify-k tokens, which at k=16 is 2.45 ms because the per-extra-token marginal at k=16 is small relative to the bandwidth-bound base. So the asymmetry is structural: a quantised target's verify forward at high k is nearly free; an FP drafter's forward is not.
2. **Accept rate collapsed to 0.0881 vs the §1 prediction band α ∈ [0.5, 0.7].** The drafter trains against the **full-precision** Qwen3.5-27B target; the 4-bit-quantised target's argmax distribution diverges materially from what the drafter learned. OQ-7's α-closure ("upstream `DRAFT_REGISTRY` maps the 4-bit MLX target ID, so pairing is supported") was necessary but not sufficient — the registry says the pairing **loads**, not that the accept rate is preserved.
3. **Rollbacks dominate decode time.** With α = 0.088 every cycle effectively rolls back; per-cycle overhead beyond propose+verify is ~80 ms (rollback + recurrent-state replay). Combined with the 35.7 ms drafter cost, each cycle yields 2.32 tokens for ~40 ms of work + ~80 ms of rollback/replay = ~19 tok/s peak per-cycle, which rollback variability flattens to 7.74 tok/s.

**What this reveals about the hybrid architecture:** the 80 ms rollback / replay cost is partly DeltaNet recurrent-state replay (48 layers per step, replayed over the committed prefix). On a pure dense model the rollback would be cheaper — but Qwen3.5-27B is not pure dense. Spec methods that ignore this (NodeNestor's MTP-via-llama.cpp port, EAGLE-3 family ports that assume cheap rollback) inherit the same trap: hybrid recurrent replay turns "mostly-reject" cycles into a cost surface that drowns the drafter savings.

**Don't-re-open conditions:**

- A new C.4-style drafter that does NOT change one of {drafter cost, accept-rate vs 4-bit target, rollback cost, verify-side kernel} fails the P6_AUTORESEARCH.md re-open gate by definition.
- `--quantize-draft` (4-bit-quantising the upstream DFlash drafter) is an open question that addresses (1) drafter cost and possibly (2) accept-rate-vs-4-bit-target. It is in scope for re-opening **only** with explicit user approval per P6_AUTORESEARCH.md.
- Porting upstream's `verify_qmm` int4 simdgroup-MMA Metal kernel + 2-pass JIT SDPA (the dflash-mlx kernel-reference patterns the kernel-ecosystem survey identified) addresses (4) verify-side kernel. Same gating: explicit user approval required.

### 2.2 Track B native 3-bit (retired v1.7.21)

**Headline:** 27% memory reduction, but +16.85% PPL drift breaches both `ΔPPL_abs ≤ 0.5` and `ΔPPL_rel ≤ 5%` bounds. B.3 speedup measurement not run.

**Source: `plans/P6_TRACK_B/REPORT.md` (B.2) + PLAN.md v1.7.21 changelog.**

The retire is unequivocal: the gate was declared up front as both-pass (ΔPPL_abs ≤ 0.5 AND ΔPPL_rel ≤ 5%), the candidate fails both, and the user-confirmed framing is "speedup numbers cannot rescue a candidate whose pre-declared quality bound is breached."

**Two interpretations the REPORT keeps live:**

1. The candidate-specific calibration is suboptimal: NexVeridian used vanilla `mlx_lm.convert -q --bits 3` at default `group_size=64` with no activation-aware step. A re-converted 3-bit checkpoint with tightened calibration could plausibly close part of the gap.
2. Pure weight-only Q3 is too aggressive at 27B: even well-calibrated weight-only Q3 typically pays 5-12% PPL on 7B-30B; 17% is high-end-of-distribution rather than an obvious calibration outlier.

**Follow-up survey closed empty 2026-05-01.** No matched-family activation-aware Qwen3.5-27B 3-bit MLX-native checkpoint exists on HF Hub (per `plans/P6_TRACK_B_FOLLOWUP_SURVEY.md` and re-confirmed by the 2026-05-02 HF survey in this orientation cycle). Re-look triggers: an MLX-native AWQ / GPTQ / AutoRound / DWQ / OptiQ Qwen3.5-27B 3-bit checkpoint with on-card PPL evidence; an mlx-lm 3-bit calibration step in the convert path.

**Don't-re-open conditions:** explicit user approval AND a hypothesis that changes one of {calibration recipe, activation-aware method, group size}. Re-quantising the ~52 GB FP weights in-house is not authorised and is not a default path.

### 2.3 C.5 DDTree (currently in escalate state, NOT yet retired)

**Headline:** β.1 spec-on warm-decode 6.54 tok/s = 0.408×; β.2 coverage profile spans 0.063 (b=1) → 0.341 (b=32). Decision matrix routes this to **escalate**, not retire — `coverage@8` and `@16` sit in [0.15, 0.30) and `@32` reaches 0.341.

**Source: `plans/P6_C5_DDTREE/REPORT.md` + `plans/P6_C5_DDTREE_OPENING.md` §9 decision matrix.**

The current REPORT engineering recommendation is "**retire (1b) ≥60 tok/s unless the user explicitly authorises a γ.1 read-only kernel survey targeting `b ≥ 32` tree-verify with sub-linear cost attestation.**" The recommendation rests on:

1. `coverage@b` at b ≤ 16 does not justify implementation work (gate window pre-declared at b ∈ {4, 8, 16}; nothing reaches 0.30).
2. β.1 cost-leg is materially worse than C.4 (0.408× vs 0.482×): swapping the 2B BF16 DFlash drafter for the 0.8B Qwen3.5 drafter saved 22% drafter cost but the lower `verify_k=4` and unchanged α≈0.09 floor drove `rollback_count` to 296 (1.8× C.4) and net speedup down. Tree shape on the same pairing inherits this.
3. The `coverage@32 = 0.341` signal is real but speculative: it requires `humanrouter/ddtree-mlx` to ship a no-torch MLX-native tree-attention kernel that scales sub-linearly to T=32, with verify-cost low enough to clear 3.74×. None of these is established in Silica-local data.

**This memo does not pre-empt the user's retire-vs-γ.1-survey decision.** The autoresearch loop adds new context to that decision: if the kernel microbench in §7 below identifies a measurable kernel-side gap on the existing dense decode path, the (1b) survival arithmetic shifts — bandwidth-side levers from custom kernels could narrow the spec headroom requirement, in which case C.5 needs less from the tree-shape leg to compose to 60 tok/s. Conversely, if the microbench shows stock MLX kernels already at ceiling, then C.5 carries (1b) alone and γ.1 becomes more attractive as the only remaining lever.

### 2.4 What this means for the loop

The pattern across all three failed paths: each was a single-lever play against the 2.93× verify-k cap or the 5% PPL cap, and each tripped on a hidden-cost surface (drafter cost dominating, calibration drift, hybrid rollback dominance) that the paper claim did not anchor. The retired/escalate states are correct disposals; **the autoresearch lesson is to compose levers rather than restart at a new single-lever pick, and to anchor every probe in a Silica-local microbench before any port lands**.

---

## 3. Bottleneck Model

This section decomposes the dense, MoE, and spec paths into the per-step costs Silica can measure, and refines the hardware-limit envelope against the latest data. **It does not defend the P6_AUTORESEARCH.md prompt's prior envelope numbers** — it rebuilds them from the corrected 15.13 GB anchor.

### 3.1 Dense Qwen3.5-27B-4bit decompose (B=4 warm-decode hot path)

The decode step at B=4, ctx≈128 (warm-decode shape), takes 95.0 ms wall time (measured: 1 / (42.17 / 4) = 0.0949 s). Decomposed bytes/step at 100% bandwidth utilisation = 95.0 ms × 307 GB/s = **29.16 GB/step**. Decomposed components (analytic, refined by §7 microbench):

| Component | Bytes (analytic) | Mechanism | Hot-path? | Note |
| --- | ---: | --- | --- | --- |
| Weight stream (amortised, B=4) | 15.13 GB | one weights-read produces 4 tokens | yes | matches v1.7.13's "B=1 weights = 15.13 GB amortised across the batch" framing |
| KV reads (16 full-attn × 4 rows, head_dim=256, GQA 24:4 = 6:1) | ~58 MB at ctx=128 (Unit 7 jsonl) × 4 ≈ 0.23 GB | bandwidth-bound at low ctx; grows with ctx; GQA already shrinks K/V by 6× | yes | small at ctx=128 but the 4 KV heads × 256 head_dim shape constrains FlashAttention kernel choices |
| DeltaNet recurrent updates (48 × 4 rows) | ?? | mlx-lm's `gated_delta_update` over per-layer recurrent state (16 K heads + 48 V heads, head_dim=128 each, conv_kernel_dim=4) | yes | **unmeasured locally**; this is the §1.4 OQ-3 gap and the largest layer-count contributor (48/64 = 75% of layers) |
| RMSNorm + RoPE small ops (per layer × per row) | ?? | `mx.norm` / `mx.fast.rope` chained; partial RoPE applied to 25% of head_dim per `partial_rotary_factor` | yes | currently NOT fused; the partial-rotary factor narrows the RoPE work per layer |
| Gated-attention output gate (per full-attn layer × per row) | small | `attn_output_gate: true` requires `out = sigmoid(g) * SDPA(...)` — extra small-op chain post-SDPA on the 16 attention layers | yes | not in stock SDPA kernels; any FlashAttention candidate must include this gate path |
| Logits projection | ~2 GB (amortised) | `lm_head` matmul (5120, 248320) per row | yes | weight-tied to embed? `tie_word_embeddings: false` per config — separate weight |
| Python loop / scheduler step | ?? | per-step Python in `Engine._drive` and `ContinuousBatcher.step` | yes | unmeasured locally |

The remaining ~14 GB headroom (29.16 GB/step at 100% util minus the ~15.4 GB analytic accountable) is **either** bandwidth-idle waiting on compute, **or** unaccounted bytes from the DeltaNet / KV / logits paths. The verify-k microbench's regime transition at k=4 (55% util at constant weight bytes) is strong evidence that compute-bound time, not bandwidth-idle, dominates the gap.

**Refined dense ceiling envelope (hypotheses to be tested, not defended):**

| Lever family | Mechanism | Best-case multiplier on 42.17 | Evidence |
| --- | --- | --- | --- |
| Bandwidth utilisation 52% → 80% via custom kernels | reduce candidate-side compute / fuse small ops to free bandwidth at B=4 | up to 1.54× (≈65 tok/s) | verify-k microbench shows the 30% gap at k=4 is compute-bound, custom kernels can target it |
| Spec amortisation at sustainable α | tokens/cycle > 1, e.g. k=4 with α=0.5 yields ~1.7 tokens/cycle bonus | up to ~1.5-1.8× over a bandwidth-improved baseline | bounded by verify-k 2.93× linear cap; real gain ≈ 1.5-1.8× per local microbench math |
| Bytes/step (weight quantisation) | 4-bit → activation-aware 3-bit | up to 1.3× (5% PPL bound permitting) | retired (Track B); contingent on a future quality-passing checkpoint |
| KV traffic reduction | paged-KV gather + KV-codec coupled at decode | up to ~10% at 4K context | partial; D-003 forbids compressed-domain attention in v0.1 |
| Scheduler overlap (B>1 spec) | composes spec lever with batched throughput | unknown | (c) slice 3 deferred |

**Composed envelope (subject to measurement, not a defence):**

- Single-lever upper-bound (kernel-side only, no spec): **65 tok/s** at 80% B=4 util.
- Two-lever composition (kernel + spec): **65 × 1.5 = 97 tok/s** at 80% util + 1.5× spec.
- The 60 tok/s milestone sits inside the envelope's mid-band.

The real question is not whether 60 is reachable in principle, but whether each lever's contribution is **independently measurable** — which is exactly the autoresearch loop's job. Without per-lever attribution, multiplicative composition of paper claims is the trap C.4 already fell into.

### 3.2 MoE Qwen3.5-35B-A3B-4bit

Different regime: weight-bound at higher B, climbs to 92% util at B=4. The (2a) ≥100 anchor (B=2 = 120.93) and (2b) ≥175 stretch (B=4 = 188.50) are both cleared. The remaining lever band (B=5/6 at 92%+ util) is in diminishing returns; gains here are real but small in absolute terms.

The relevant question for the autoresearch loop is **what kernel work portable across families** would lift both dense and MoE. Fused RMSNorm + RoPE applies to both. SDPA-style fused attention applies to both. DeltaNet kernel applies only to hybrid Qwen3.5 (both 27B dense and 35B-A3B MoE share the hybrid stack — verified at silica/models/qwen3_5.py:75 "hybrid DeltaNet + GQA").

### 3.3 Spec path bottleneck — what C.4 / C.5 evidence has nailed

| Component | Measured | Implication |
| --- | --- | --- |
| Drafter cost (2B BF16) | 35.7 ms | dominates by 15× — drafter quantisation or self-spec is required |
| Drafter cost (0.8B FP) | 27.6 ms | still dominates; size below 0.8B is unavailable in matched-family |
| Verify cost (k=4) | 2.09 ms; 9.81 ms/tok marginal | bandwidth-bound at low k, compute-bound at k≥4 |
| Verify cost (k=8) | 162.96 ms | linear-shape upper bound — 2.93× zero-drafter ceiling |
| Accept rate (4-bit target × FP drafter) | 0.088-0.091 | structural floor, not drafter-specific |
| Rollback cost (hybrid arch, α≈0.09) | ~80 ms/cycle | partly DeltaNet recurrent replay; cheap rollback assumes pure dense |
| Coverage@b on 0.8B drafter | 0.063 → 0.341 across b=1→32 | "directionally right but greedy-wrong" pattern — fat trees can in principle exploit |

**Three composable levers the spec path could exploit:**

1. **Self-spec / target-only methods** that eliminate the drafter cost — KnapSpec, LayerSkip, SWIFT. Capped at ~1.4-1.5× per published numbers (single-lever, no overlap).
2. **Bandwidth-shared self-drafter** — QuantSpec uses a hierarchical 4-bit quantised KV cache shared between target and self-drafter. Attacks the C.4 root cause (drafter KV bandwidth dominance) directly. Single-lever, claim 2.5×.
3. **In-model MTP head reuse** — the Qwen3.5-27B safetensors may already contain MTP keys per the Qwen card. NodeNestor's llama.cpp port reproduced 47.5% accept rate but suffered net-negative speedup due to recurrent-rollback overhead. Silica's `SpecRecurrentRollbackAdapter` is the natural place to test whether hybrid-aware rollback is cheaper.

The verify-k 2.93× cap holds for all three. Composing any of them with a bandwidth-side kernel lever is the only credible path past 2× over 42.17.

### 3.4 Weight / KV / memory

- Dense 27B-4bit weight footprint: 15.13 GB on-disk and runtime-resident (mlx-lm streams scales/zeros alongside packed weights).
- KV bytes at B=1 ctx=128: ~58 MB (per Unit 7 jsonl `kv bytes (est)`); scales with B and ctx.
- Peak resident at B=4 ctx=128: 17.10 GB; at B=4 ctx=4K: not measured; at B=1 ctx=4K MoE: 23.60 GB.
- §6(4) RAM gate (≤ 36 GB) is comfortably satisfied across all measured shapes. Memory is not the binding constraint on the dense path today.

### 3.5 Scheduler / batcher

- ContinuousBatcher GLOBAL-only gate at silica/scheduler/batcher.py:268-279 preserves correctness for hybrid + spec but precludes B>1 spec composition until (c) slice 3 lifts it.
- Prefix cache is functional and measured under the (4-b) regression gate; not currently a bottleneck on the warm-decode hot path (warm = no prefix cache benefit; cold tests live elsewhere).
- Admission ordering and chunked-prefill (D.1 in P-6 deliverables) is unimplemented — out of autoresearch loop scope unless §7 microbench surfaces a per-step Python overhead that justifies it.

---

## 4. External Research Radar

Sources: three sub-agent surveys (MLX kernel ecosystem; speculative methods; HF checkpoint inventory) executed 2026-05-02.

### 4.1 MLX / Apple-Silicon kernel ecosystem (highest-priority lever)

- **MLX core v0.31.x** has not landed FA-2 SDPA. Issue #2955 closed Dec 2025 without maintainer adoption. Decode-friendly perf items in v0.30.4-v0.30.5 are fast-vector-GQA and splitk gemm dispatch tuning. v0.31 perf work is CUDA-side. Silica should not wait for upstream FA-2.
- **`mx.fast.metal_kernel(name, input_names, output_names, source, ...)`** is the user-facing custom-kernel surface. JIT-compiled at runtime; framework auto-generates the function signature; thread position via `thread_position_in_grid.x`.
- **CRITICAL CONSTRAINT (vllm-metal RFC #188):** A custom MLX kernel registered via `mx.fast.metal_kernel` is NOT a true MLX primitive — it cannot return lazy arrays the way internal primitives can. Per-layer custom-kernel chains risk introducing a per-layer sync barrier. **Silica must aggregate multiple ops into one kernel rather than chaining single-op kernels.**
- **`Hmbown/ZMLX`** (v0.10.0, MIT, 2026-03-03): MLX-only kernel toolkit; fused RMSNorm / LayerNorm / Softmax replacements via `patch(model)`. Reports +7.5% on Qwen3.5-9B-4bit greedy decode on M4 Max. **Kernel-reference candidate**, not direct drop-in (% on 9B does not extrapolate cleanly to 27B).
- **`arozanov/turboquant-mlx`** + mlx-lm PR #1067 (Apache-2.0, open): fused QK-scoring kernel pattern (`prerot_fused_qk_scores`), fused quantize/dequant. Long-context KV compression target — not a B=4 short-decode lever, but the QK-fusion kernel pattern is a useful template.
- **`humanrouter/ddtree-mlx`** (MIT, 21 commits): MLX with custom Metal kernels for tree-aware GatedDelta recurrence. **Already on Silica's radar via C.5 escalate**. Reports M3 Ultra Qwen3.5-27B-4bit 42.3 vs 27.9 tok/s (~1.5× spec aggregate). The reported number is on M3 Ultra (not M5 Pro) and is spec-aggregate (not stock decode), so it does not establish that Silica's 42.17 is at or below the chip ceiling.
- **`bstnxbt/dflash-mlx`** + **`Aryagm/dflash-mlx`** (MIT): MLX-only; targeted custom kernels (`verify_qmm` int4 simdgroup-MMA for M=16, JIT SDPA 2-pass). M5 Max 64GB Qwen3.5-27B-4bit `79.02 tok/s @ 1024 ctx, 2.37× speedup, 90.04% acceptance` — but on **M5 Max** (different chip, more memory bandwidth) and **with their drafter**, which Silica has retired. **Kernel-reference candidate** — `verify_qmm` and JIT SDPA 2-pass are directly transferable patterns.
- **`philipturner/metal-flash-attention` (MFA)** — Swift + Metal FA-2 fwd+bwd. No MLX bindings. **Kernel-reference only.**
- **`vllm-project/vllm-metal` v0.2.0** (2026-05-02): Apache-2.0; unified paged varlen Metal kernel; RFC #188 documents the per-layer sync barrier issue. Torch-coupled at the framework boundary. **Kernel-reference + monitor**, not drop-in.
- **No public Apple-Silicon fused RMSNorm + RoPE kernel exists** — gap in ecosystem; Silica would have to roll its own via `mx.fast.metal_kernel`.

### 4.2 Speculative decoding (2024-Q4 to 2026-Q2)

- **EAGLE-3** (NeurIPS'25, 2503.01840): tree amortisation + multi-layer feature fusion. CUDA / Triton in upstream. **No Qwen3.5-27B drafter checkpoint exists**; only 9B / 35B-A3B targets exist (PyTorch-only). **Monitor.**
- **Speculative Streaming** (Apple, 2402.11131): no-aux-drafter; requires fine-tuning the target's MTP head. **No Qwen3.5-27B fine-tuned ckpt.** Conflicts with Silica's "no PPL trade" rule via target weights modification. **Monitor.**
- **SSD / Saguaro** (ICLR'26, 2603.03251): drafter and verify run in parallel on distinct accelerators. M5 Pro has Metal GPU + ANE; MLX has no NPU dispatch path today. **Monitor as design study.**
- **SSSD** (2411.05894): training-free n-gram lookup; claim ~2.9× *at the 2.93× cap* (consistent with our local ceiling). MLX-portable. **Probe candidate** — cheapest possible spec probe; useful as a sanity check on whether prompt-bound input-grounded workloads can hit the cap without a learned drafter.
- **QuantSpec** (ICML'25, 2502.10424): hierarchical 4-bit quantised KV cache shared between target and self-drafter; bit-sharing layout MLX-portable. Claim 2.5× at >90% accept; reduces KV memory ~1.3×. **Bandwidth-side composable; attacks C.4 root cause (drafter KV bandwidth dominance) directly. Probe candidate.**
- **MTP via Qwen3.5-27B native head** (per Qwen org card "trained with multi-steps"): head ships in the official safetensors. NodeNestor's llama.cpp port reproduced 47.5% accept rate but **net-negative speedup** (12.5 vs 17 tok/s) due to recurrent-state checkpointing overhead. **Probe candidate** — Silica's hybrid-aware rollback may resolve the NodeNestor failure mode.
- **FastMTP** (2509.18362, Tencent): shared-weights MTP head with curriculum training; CUDA. No Qwen3.5 ckpt. **Monitor** — improves MTP probe ceiling but only after MTP-direct probe succeeds.
- **KnapSpec** (2602.20217), **LayerSkip** (ACL'24), **SWIFT**, **DEL**, **CLaSp**, **SpecPV**: target-only self-spec early-exit / layer-skip. Claim 1.4-1.5×, training-free. KnapSpec's knapsack formulation matches Silica's hybrid-stack heterogeneous-layer-cost regime (DeltaNet linear-attn ≠ full-attn cost). **KnapSpec is a probe candidate** — lowest integration risk of the spec set.
- **Mirror-SD** (Apple, 2510.13161): cross-device overlap (GPU + NPU); +30% over EAGLE-3; the only legitimate >2.93× lever in the survey. No public code; no MLX. **Monitor as design study** until MLX exposes ANE dispatch.
- **Sequoia** / **OPT-Tree** / **Group Tree Optimization**: tree-shape DP. **Monitor** as input to a future DDTree retry.
- **DFlash / DFlash-MLX**: re-killed by C.4. Discard.
- **Medusa / Hydra**: superseded by EAGLE-3. Discard.
- **FlashSSD**: not a real method (search returns DFlash and SSD only). Discard.
- **FlashMoE**: fused MoE kernel, not a spec method. Discard for dense.

### 4.3 HF Hub checkpoint inventory (since 2026-05-01)

- **`mlx-community/Qwen3.5-27B-OptiQ-4bit`** (created 2026-04-24, 6 days ago, codelion): mixed-precision activation-sensitivity-aware 4-bit; 247 layers @ 8-bit + 343 layers @ 4-bit, target 4.5 BPW. 16.5 GB (slightly larger than current production 4-bit 16.1 GB). On-card eval: GSM8K 200×3-shot 87.5% vs uniform-4-bit 90.0% (-2.5pp). **Probe candidate for quality-floor recovery** if a future Track B re-attempt is authorised. MTP-head preservation unverified.
- **`trevon/Qwen3.5-27B-MLX-MTP`** (last commit 2026-05-01, ~13h before survey): affine 8-bit; 29.1 GB. **Confirms MLX-native MTP-preserving conversion is feasible.** Requires AirRunner mlx-lm fork branch `feat/mtp-native`. Too large for production; valuable as reference checkpoint. **The actionable artefact is the AirRunner fork**, not the 8-bit weights.
- **`mlx-community/Qwen3.5-27B-4bit-DWQ`** (2026-03-24, N8Programs): DWQ (data-free weighted quant, activation-aware variant). 15.2 GB. README empty; no eval evidence on card. **Monitor.**
- **No matched-family Qwen3.5 drafter smaller than 0.8B exists** (no 0.5B, no 0.3B).
- **No standalone Qwen3.5-27B EAGLE-3, Medusa, or Hydra MLX-native head exists.**
- **No new Qwen org 27B variant or drafter checkpoint added since 2025-12** beyond the Feb-2026 family release and recent SAE-Res sparse autoencoder probes (irrelevant).
- **No activation-aware Qwen3.5-27B 3-bit MLX checkpoint** — the Track B follow-up survey's empty close holds.

### 4.4 Reclassifications from prior project state

| Method | Prior classification | New classification | Why changed |
| --- | --- | --- | --- |
| C.4 DFlash | retired (1b) lever | retired AND C.4-style FP-drafter-vs-4-bit-target structural anti-pattern | hybrid recurrent rollback dominance is now part of the postmortem |
| Track B native 3-bit | retired pending re-look | retired; OptiQ-4bit emerges as quality-floor-recovery candidate at 4-bit (not 3-bit) | new HF checkpoint changes the available option set |
| MTP for Qwen3.5 | not on radar | **probe candidate** (head already in production weights; AirRunner fork shows MLX path) | Qwen org card + trevon precedent newly identified |
| QuantSpec | not on radar | **probe candidate** (bandwidth-side composable, attacks C.4 root cause) | newly identified as the cleanest fit for the C.4 failure mode |
| KnapSpec / self-spec | not on radar | **probe candidate** (training-free, lowest integration risk, hybrid-stack-aware) | knapsack formulation matches Silica's heterogeneous layer cost |
| ZMLX / turboquant-mlx kernel patterns | not on radar | **kernel-reference** | patterns inform a custom-kernel microbench |
| dflash-mlx kernel patterns (`verify_qmm`, JIT SDPA) | upstream stack out of scope | **kernel-reference** (selectively, under MLX-native re-impl) | C.4 retirement does not disqualify the kernel patterns |
| Mirror-SD | not on radar | **monitor as design study** | the only legitimate >2.93× lever; blocked on ANE-MLX bridge |

---

## 5. Intake Cards

One card per plausible method. Methods classified `monitor` get an abbreviated card; `discard` items live in §4 only.

### 5.1 In-model MTP head (Qwen3.5-27B native) — RETIRED for current production target

- **Name:** MTP via Qwen3.5-27B safetensors-resident multi-token head
- **Source links:** [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B); reference port [`NodeNestor/qwen3.5-27b-mtp-llamacpp`](https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp); MLX-side precedent [`trevon/Qwen3.5-27B-MLX-MTP`](https://huggingface.co/trevon/Qwen3.5-27B-MLX-MTP) using AirRunner mlx-lm `feat/mtp-native` fork
- **Status (2026-05-02):** **RETIRED for `mlx-community/Qwen3.5-27B-4bit`.** Direct inspection of the production target's `model.safetensors.index.json` (path: `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-27B-4bit/snapshots/45797d2985a12c55e6473686e9ea91b95e959553/model.safetensors.index.json`) found **0 keys matching `mtp|nextn|multi_token|multistep`** in the 2180-key index. `config.json:text_config.mtp_num_hidden_layers = 1` confirms the architecture supports an MTP head, but the production checkpoint ships without MTP weights. The MTP path can only be re-opened by re-converting from upstream — both options gated on explicit user authorisation:
  - **Option A:** Download upstream `Qwen/Qwen3.5-27B` BF16 (~52 GB), run `mlx_lm.convert -q --bits 4 --group-size 64 --keep-mtp` (or equivalent), produce a new MLX 4-bit checkpoint with MTP keys preserved. **Disk cost ~70 GB; conversion runtime ~30-60 minutes.**
  - **Option B:** Download `trevon/Qwen3.5-27B-MLX-MTP` (8-bit, 29.1 GB) and use as a reference; production speedups from this checkpoint are bounded by its memory cost (8-bit = ~25 GB resident, leaving little headroom on M5 Pro 48 GB at B>1).
  - **Option C:** Wait for upstream mlx-lm to merge the AirRunner `feat/mtp-native` fork's NEXTN execution path AND for an mlx-community MTP-preserving 4-bit conversion to land. **Monitor only.**
- **Date / freshness:** Qwen org card Feb-2026; trevon ckpt 2026-05-01; NodeNestor reproduction ongoing
- **Claimed speedup (if path were re-opened):** Qwen native MTP via SGLang `--speculative-algo NEXTN --speculative-num-steps 3` (CUDA path); NodeNestor llama.cpp ported 47.5% accept rate but net-negative wall clock (12.5 vs 17 tok/s) due to recurrent-state checkpointing overhead.
- **Mechanism:** in-model multi-token head predicts k future tokens per forward; verify path validates against the same model. No external drafter — drafter cost ≈ 0 by construction.
- **Runtime requirements:** mlx-lm runtime support for the NEXTN execution path (currently only AirRunner fork). Silica adapter wiring required.
- **Has Qwen3.5 / MoE checkpoint with MTP weights?** **Production target NO** (verified). `trevon/Qwen3.5-27B-MLX-MTP` YES (at 8-bit, too large).
- **Requires training / finetuning?** No.
- **Requires new model weights?** **YES** (the disposition that flipped). Conversion or alternative checkpoint required.
- **Recommendation:** **monitor**. Re-open only on user authorisation for Option A or B above. If re-opened: (i) prototype an MTP-aware `DraftEngine` impl that extracts the multi-token logits directly from the verify forward; (ii) measure tokens_per_target_forward and accept_rate; (iii) verify Silica's hybrid-aware rollback is cheaper than NodeNestor's llama.cpp path.
- **Kill criteria (if re-opened):** kill if drafter forward (1 MTP step inside verify) > 1.0 ms, OR accept@step1 < 0.40, OR rollback cost > drafter savings.

### 5.2 QuantSpec (hierarchical quantised KV self-drafter)

- **Name:** QuantSpec — hierarchical 4-bit quantised KV self-drafter
- **Source links:** [arXiv 2502.10424](https://arxiv.org/abs/2502.10424) (ICML'25)
- **Date / freshness:** Feb-2025
- **Claimed speedup:** ~2.5× with >90% accept rate on Llama-class targets; KV cache memory reduced ~1.3×.
- **Mechanism:** target uses high-bit KV; self-drafter uses lower-bit shared layout. Single-lever, single-device, no aux model.
- **Runtime requirements:** paper uses CUDA + flash-attn; the bit-sharing layout itself is bandwidth-only and MLX-portable. Re-impl required.
- **Hardware assumptions:** single device. No NPU / cross-device overlap.
- **Needs CUDA / Triton / torch / custom kernel?** Re-impl in MLX would need a custom KV layout primitive. The shared-layout itself is data-layout work; the verify path could re-use stock `mlx.fast.scaled_dot_product_attention` if the layout decode-on-fetch is wrapped at the cache surface.
- **Has MLX code?** No.
- **Has Qwen3.5 ckpt?** Not required (target-only self-drafter).
- **Requires training / finetuning?** No.
- **Requires new model weights?** No.
- **Exact / lossless or approximate?** Exact under SpecVerify.
- **Which Silica bottleneck it changes:** drafter cost (self-drafter at lower-bit = lower bandwidth); accept rate (shared layout → drafter argmax ≈ target argmax modulo quantisation noise).
- **Minimal local probe:** (i) measure stock-MLX 4-bit-KV-cache decode at B=1 on Qwen3.5-27B-4bit as baseline; (ii) prototype a 3-bit-or-lower self-drafter KV layout via `silica.kvcache` extension (no Metal kernel yet); (iii) measure accept rate on β.2-style teacher-forced corpus.
- **Kill criteria:** kill if drafter forward cost > 6 ms (vs the 35.7 ms C.4 result and 2.45 ms verify baseline at k=4), OR accept@k=8 < 0.45, OR PPL drift on the self-drafter's KV layout breaches the 5% relative bound.
- **Expected implementation cost:** medium — touches `silica.kvcache` layout, adds a `QuantSpecDraftEngine` impl. ~600-1000 LOC including tests.
- **Measurement cost:** probe-only run on cached weights; no download.
- **Quality / correctness risk:** medium — shared-layout quantisation could drift the self-drafter's logits vs target argmax in a way that doesn't surface until end-to-end PPL measurement. The 5% PPL gate inherited from Track B applies.
- **Composability:** strong with custom-kernel work on the shared-layout fetch (multiplicative).
- **Recommendation:** **probe** with explicit user authorisation for the kvcache extension. Behind the kernel microbench (§7) in priority — the kernel microbench may shift the picture by surfacing whether drafter forward cost would even matter at the new B=4 baseline.

### 5.3 KnapSpec (self-spec early-exit, layer-skip)

- **Name:** KnapSpec — knapsack-formulated layer-skip self-speculative
- **Source links:** [arXiv 2602.20217](https://arxiv.org/abs/2602.20217)
- **Date / freshness:** 2026-Q1
- **Claimed speedup:** up to 1.47× on Qwen3 (paper reports Qwen3-class targets); training-free.
- **Mechanism:** skip a subset of target layers as drafter; verify with full forward. Knapsack formulation handles heterogeneous per-layer cost.
- **Runtime requirements:** torch / CUDA in published code; layer-skip itself is implementable in MLX as a forward-mask.
- **Hardware assumptions:** single device.
- **Needs CUDA / Triton / torch / custom kernel?** No. Pure MLX forward-mask + verify.
- **Has MLX code?** No.
- **Has Qwen3.5 ckpt?** Not required (target-only).
- **Requires training / finetuning?** No.
- **Requires new model weights?** No.
- **Exact / lossless or approximate?** Exact under SpecVerify.
- **Which Silica bottleneck it changes:** drafter cost (layer-skip drafter is structurally cheaper than full forward); accept rate (depends on which layers are skipped).
- **Minimal local probe:** (i) instrument `Qwen3_5Adapter` forward with a layer-skip mask; (ii) measure accept rate on β.2-style corpus across a few skip schedules (skip the cheapest k DeltaNet layers; skip every other DeltaNet; skip the last few full-attn layers); (iii) attribute speedup against the verify-k baseline.
- **Kill criteria:** kill if best layer-skip schedule yields < 1.30× over k=8 zero-drafter ceiling on Qwen3.5-27B-4bit; or if accept rate falls below 0.40 across all tested schedules.
- **Expected implementation cost:** small — forward-mask injection + new `KnapSpecDraftEngine` impl. ~300-500 LOC.
- **Measurement cost:** probe-only on cached weights; no download.
- **Quality / correctness risk:** low for greedy parity; medium for sampling under arbitrary skip schedules.
- **Composability:** moderate — composes additively with kernel work, but layer-skip itself reduces the ceiling that kernel work can lift.
- **Recommendation:** **probe** as a self-spec floor candidate, after MTP probe disposition.

### 5.4 Custom MLX Metal kernels — fused RMSNorm+RoPE, fused gated-delta-update, FlashAttention-style SDPA

- **Name:** Custom MLX-native kernel candidates for the Qwen3.5-27B hot path
- **Source links:** [mlx.fast.metal_kernel docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html); reference patterns [Hmbown/ZMLX](https://github.com/Hmbown/ZMLX), [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx) (`prerot_fused_qk_scores`), [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx) (`verify_qmm`, JIT SDPA 2-pass), [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- **Date / freshness:** ZMLX v0.10.0 2026-03-03; turboquant PR #1067 open; dflash-mlx active; MFA stable
- **Claimed speedup:** ZMLX +7.5% on Qwen3.5-9B-4bit decode (M4 Max); turboquant 0.98× FP16 at 4.6× compression on Qwen2.5-32B (M4 Pro); dflash-mlx 79.02 tok/s on M5 Max Qwen3.5-27B-4bit (spec aggregate, not stock)
- **Mechanism:** fuse small ops (RMSNorm + RoPE; QK-norm + score; quantised matmul + bias) into a single Metal kernel dispatch to (i) reduce kernel-launch overhead, (ii) keep intermediates in registers / threadgroup memory rather than HBM, (iii) free bandwidth at high B for the weight stream.
- **Runtime requirements:** MLX `mx.fast.metal_kernel` API. JIT-compiled at runtime.
- **Hardware assumptions:** Apple Silicon Metal GPU; head-dim and KV-len shapes typical of Qwen3.5-27B (head_dim=128 for full-attn, MoE-style heads).
- **Needs CUDA / Triton / torch / custom kernel?** Custom Metal via mlx.fast.metal_kernel only.
- **Has MLX code?** Reference patterns exist; Silica re-impl required per P6_AUTORESEARCH.md ("kernels are not the default; they are an option opened by measurement, not by external paper claims").
- **Has Qwen3.5 / MoE checkpoint?** N/A — checkpoint-independent.
- **Requires training / finetuning?** No.
- **Requires new model weights?** No.
- **Exact / lossless or approximate?** Must be exact-or-tighter than the MLX reference within fp16 noise (per P6_AUTORESEARCH.md "Custom kernels must be exact-or-tighter").
- **Which Silica bottleneck it changes:** kernel cost per layer (priority 3 in P6_AUTORESEARCH.md hardware-limit ladder); bandwidth utilisation (priority 1).
- **Minimal local probe:** the **§7 microbench** decomposes per-step time by layer kind. The microbench is the gate to opening any specific kernel candidate.
- **Kill criteria per kernel:** if the microbench shows the targeted op is below 20% of step time, the kernel is not on the critical path and the candidate is not opened.
- **Expected implementation cost (per kernel, after microbench):** medium — 200-400 LOC kernel source + 50-100 LOC test harness + correctness gate per P6_AUTORESEARCH.md (max-abs / max-rel error vs MLX reference on ≥3 input shapes spanning the production decode profile).
- **Measurement cost:** the §7 microbench fits an autonomous-loop iteration; per-kernel microbenches stay under unit-test cost.
- **Quality / correctness risk:** low on fused-norm + RoPE (numerical equivalence is straightforward); medium on FlashAttention-style SDPA (online-softmax precision is the classical concern).
- **Composability:** kernels compose additively with each other and multiplicatively with bandwidth utilisation; they compose multiplicatively with spec amortisation if the kernel work raises the verify-forward util at the spec verify-k step.
- **Recommendation:** **microbench first** (§7), then per-kernel candidate opening behind explicit user authorisation. Per P6_AUTORESEARCH.md: microbench-only is autonomous; integration is not.

### 5.5 SSSD (training-free n-gram lookup spec)

- **Name:** SSSD — Simply-Scalable Speculative Decoding
- **Source links:** [arXiv 2411.05894](https://arxiv.org/abs/2411.05894)
- **Date / freshness:** 2024-Q4
- **Claimed speedup:** up to 2.9× vs autoregressive (consistent with the local 2.93× cap; honest)
- **Mechanism:** training-free n-gram lookup over the prompt; lightweight and hardware-aware scheduling.
- **Runtime requirements:** trivially MLX-portable (n-gram lookup is pure Python / mx).
- **Has MLX code?** No (re-impl trivial).
- **Has Qwen3.5 ckpt?** Not required.
- **Requires training?** No.
- **Recommendation:** **probe** as a sanity check on whether prompt-bound input-grounded workloads can hit the cap without a learned drafter. Lower priority than MTP / QuantSpec / KnapSpec because its accept rate depends heavily on prompt structure (input-grounded code completion or question-answering, not free-form).
- **Kill criteria:** kill if accept rate at k=4 < 0.55 on a representative prompt mix, OR end-to-end speedup < 1.3× against k=8 zero-drafter ceiling.

### 5.6 OptiQ-4bit (quality-floor recovery, not memory)

- **Name:** `mlx-community/Qwen3.5-27B-OptiQ-4bit`
- **Source links:** [HF card](https://huggingface.co/mlx-community/Qwen3.5-27B-OptiQ-4bit)
- **Date / freshness:** created 2026-04-24, last commit 2026-04-26 (codelion)
- **Claimed speedup:** none claimed; reduces ΔPPL gap vs uniform 4-bit baseline at slight memory cost.
- **Mechanism:** OptiQ — sensitivity-aware mixed-precision via KL-divergence calibration; layer-bit assignment (247 layers @ 8-bit + 343 layers @ 4-bit, target 4.5 BPW).
- **Runtime requirements:** mlx-lm (loads as standard quantised mlx model).
- **Hardware assumptions:** same as 4-bit production target; 16.5 GB resident (vs 16.1 GB current), 4-5% memory overhead.
- **Needs CUDA / Triton / torch / custom kernel?** No.
- **Has MLX code?** Yes (HF MLX-native).
- **Has Qwen3.5 ckpt?** Yes (this IS the ckpt).
- **Requires training / finetuning?** No.
- **Requires new model weights?** Yes — would replace the current production target.
- **Exact / lossless or approximate?** Approximate (different bit assignment changes argmax distribution slightly).
- **Which Silica bottleneck it changes:** quality floor (re-opens potential for a future Track B re-attempt at 3-bit if OptiQ lifts the floor enough that 3-bit drift fits the gate); not directly a tok/s lever on dense 27B.
- **Minimal local probe:** (i) `safetensors.index.json` inspection (does it preserve MTP keys?); (ii) WikiText-2 PPL on the same harness Track B B.2 used, on cached weights — but the checkpoint is **not yet downloaded**.
- **Kill criteria:** kill if PPL drift vs 4-bit production exceeds the 5% relative bound; kill if MTP keys absent and the chief use case (combined OptiQ + MTP) is unreachable.
- **Expected implementation cost:** zero (loads via existing factory).
- **Measurement cost:** ~10 min for a download (16.5 GB) + PPL row (the Track B B.2 fixture). **Requires explicit user authorisation for the download.**
- **Quality / correctness risk:** medium — the -2.5pp GSM8K reported on the model card is a single-task signal; PPL on WikiText-2 is the gate that retired Track B and is the binding metric here.
- **Composability:** weak (memory / quality-floor lever, not tok/s).
- **Recommendation:** **monitor** unless a future authorised re-attempt at quality-recovered 3-bit becomes the path forward. Not a current-cycle probe.

### 5.7 SSD / Saguaro — design study only

- **Name:** Speculative Speculative Decoding
- **Source links:** [arXiv 2603.03251](https://arxiv.org/abs/2603.03251)
- **Mechanism:** drafter and verify run in parallel on distinct accelerators; pre-emptive speculation.
- **Recommendation:** **monitor as design study**. M5 Pro has Metal GPU + ANE; MLX has no NPU dispatch path today. The algorithmic insight (predict verification outcomes pre-emptively) is composable with any drafter even single-device, but engineering depends on an ANE-MLX bridge that doesn't exist.

---

## 6. Ranked Hypotheses

Ranked by P6_AUTORESEARCH.md hardware-limit priority order, then by composability score. Each tagged with the lever family it touches.

| Rank | Hypothesis | Lever family | Expected speedup | Cost | Risk | Probe | Kill criteria |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | **Per-step time decomposition microbench** identifies a kernel-side gap (attention / DeltaNet / RMSNorm-RoPE / matmul / Python loop) on dense 27B B=4 | kernel fusion (priority 3) → opens priority 4 | 1.0× directly (diagnostic); enables 1.2-1.6× kernel work | low (autonomous-loop, no download) | low (read-only microbench; can't break runtime) | Add `silica/bench/microbench/decode_step_attribution.py`; instrument `Qwen3_5Adapter.decode_step_multi` with per-layer-kind timing; aggregate across N=20 iterations on cached weights; emit JSONL | If the largest single component is < 15% of step time, no kernel candidate justifies opening — bottleneck is "many small things" and the next probe is per-step Python overhead instead |
| 2 | **`safetensors.index.json` inspection** of `mlx-community/Qwen3.5-27B-4bit` confirms or refutes MTP key preservation | bytes-per-step / spec amortisation (priority 5) | 1.0× directly (diagnostic); enables MTP probe (~1.4-1.7× spec) | trivial (file-read only) | none | `python -c "import json; print(...)"` on the cached `safetensors.index.json` | Disposition is binary: keys present → MTP probe candidate opens (subject to user authorisation); keys absent → MTP path retires for this checkpoint, monitor mlx-lm upstream for AirRunner-fork merge |
| 3 | **C.5 γ.1 read-only kernel survey** on `humanrouter/ddtree-mlx` documents whether a no-torch tree-attention kernel exists with sub-linear verify cost at T=32 | spec amortisation (priority 1 in escalate path) | 1.0× directly (survey); enables/retires C.5 implementation | low (read-only docs / source survey, no clone) | low | Survey upstream README, license, kernel layout — NO download, NO clone, NO silica.* code | Per C.5 REPORT recommendation: this is the single user-decision the C.5 escalate state asks for. |
| 4 | **Verify-k microbench at B=2** decomposes whether the compute-bound regime at k=4 is the same one as B=4 dense decode | kernel / bandwidth diagnostic | 1.0× directly (diagnostic) | low (autonomous-loop, extends Unit 7 microbench) | none | Re-run `silica.bench.microbench.target_verify` at B=2, k ∈ {1, 2, 4, 8}; compare bandwidth-utilisation curve to B=1 from Unit 7 | If B=2 verify-k shows a different regime curve than B=1, the bottleneck attribution at B=4 needs further decomposition before any kernel candidate opens |
| 5 | **Per-step Python overhead microbench** on `Engine._drive` and `ContinuousBatcher.step` quantifies how much wall time is non-MLX work | scheduler overlap (priority 1 / 6) | up to 1.05-1.10× if Python is >5% of step | low (autonomous-loop, instrument-only) | none | Wrap `Engine._drive` inner loop with `time.perf_counter_ns` instrumentation gated by env var; aggregate per-step Python ms over N=20 iterations | If Python overhead is < 3% of step time, no scheduler-side optimisation is justified; if 5-10%, opens a `mx.compile`-fused sampler candidate (Track A in PLAN P-6 deliverables) |
| 6 | ~~MTP probe~~ — **RETIRED 2026-05-02** for current production target. MTP keys absent in `mlx-community/Qwen3.5-27B-4bit` safetensors index (verified). Re-open requires user authorisation for upstream-conversion (Option A) or trevon-MLX-MTP download (Option B); see §5.1. | spec amortisation (priority 5) | n/a | n/a | n/a | n/a | n/a |
| 7 | **QuantSpec self-drafter probe (gated)** — bandwidth-side attack on the drafter-cost root cause | bytes-per-step / spec (priority 1+2 composed) | 2.0-2.5× claimed | medium (touches `silica.kvcache` layout) | medium (PPL gate on shared layout) | New `QuantSpecDraftEngine` impl; teacher-forced accept-rate probe; warm-decode bench row | Kill if drafter forward > 6 ms, OR accept@k=8 < 0.45, OR PPL drift on shared layout > 5% |
| 8 | **KnapSpec self-spec probe (gated)** — training-free layer-skip; lowest integration risk | spec amortisation (priority 5) | 1.3-1.5× claimed | low-medium (forward-mask only) | low (greedy parity straightforward) | Forward-mask injection on `Qwen3_5Adapter`; β.2-style coverage probe across a few skip schedules; warm-decode bench row | Kill if best skip schedule < 1.30× over k=8 zero-drafter ceiling, OR accept rate < 0.40 across schedules |
| 9 | **`mx.compile`-fused sampler chain** (Track A.1 / A.2 in P-6 deliverables) — defer-and-batch sampler sync | scheduler overlap | 1.05-1.15× claimed | medium (touches `silica.core.sampler` + `silica.scheduler.batcher` + `silica.engine`) | low | If (5) shows >5% Python overhead, prototype `mx.compile`-wrapped sampler; measure on B=4 warm-decode | Kill if measured < 1.05× on B=4 |
| 10 | **Multi-request hybrid spec gate lift** ((c) slice 3) — composes spec with B>1 | scheduler overlap | unknown (depends on what spec-on-B=4 looks like) | high (separate orientation, multi-row dispatch over BatchKVCache) | medium | Out of autoresearch loop scope; surface only if (1)-(8) leave a measurement-anchored case for B>1 spec | n/a — phase deferred |

**Items 1, 3, 4, 5 fit the autonomous-loop budget. Item 2 has been resolved during this orientation (MTP keys absent — see §1.4 OQ-4 closure and §5.1).** Items 7, 8, 9, 10 require explicit user authorisation per P6_AUTORESEARCH.md (opening a large implementation track / shipping kernel into hot path / re-opening retired track / etc.). Item 6 is retired.

---

## 7. Recommended Next Probe

**Probe: per-step decode time decomposition microbench on dense Qwen3.5-27B-4bit warm-decode B=4.**

### 7.1 Why this probe

- It directly attacks the §1.4 OQ-1 + OQ-3 measurement gap: where does the 48% bandwidth headroom at B=4 actually go?
- It is the gate that opens or retires custom-kernel candidates per P6_AUTORESEARCH.md's "kernels are not the default; they are an option opened by measurement, not by external paper claims" rule.
- It composes with the C.5 escalate decision: kernel-side gains shift the (1b) survival arithmetic; current memo cannot pre-empt the user's retire-vs-γ.1 call without this data.
- It is the cheapest decisive probe in §6: read-only, no download, no commit, no destructive op, fits the autonomous-loop budget.
- It produces a comparable artefact (per-component-ms JSONL) that any future kernel candidate can be attributed against on the running-best progress chart.

### 7.2 Lever-family tag

Primary: **kernel fusion (priority 3 in P6_AUTORESEARCH.md hardware-limit ladder)**. Secondary: **bandwidth utilisation (priority 1)** — kernel work that frees compute at B=4 raises utilisation toward the 80%+ ceiling.

### 7.3 Hypothesis (predeclared)

The dense 27B-4bit B=4 warm-decode step at 95.0 ms is dominated by one or two of: (i) full-attention layer cost across the 16 attention layers × 4 batch = 64 attention ops; (ii) DeltaNet recurrent updates across 48 layers × 4 batch = 192 recurrent ops; (iii) RMSNorm + RoPE small-op chains across all 64 layers × per-position. The Python loop / scheduler step is < 5% of step time.

### 7.4 Pass / fail threshold (predeclared)

This is a **diagnostic** probe (per P6_AUTORESEARCH.md `experiment status` taxonomy), not an optimisation. The disposition decision tree:

- **Largest component ≥ 30% of step time AND below an achievable kernel ceiling** (e.g. attention layers at 30% of step with stock SDPA having a known 1.5× speedup headroom on M5 Pro) → opens a custom-kernel candidate. Status: `diagnostic` (with `keep` if a follow-up kernel proves the gap).
- **Largest component 15-30% of step time** → microbench that component at higher fidelity in a follow-up iteration.
- **Largest component < 15% of step time** → bottleneck is "many small things"; pivot to per-step Python overhead microbench (item 5 in §6) before proposing any kernel.
- **Probe crashes / instrumentation introduces > 5% wall overhead** → status `crash`; revise instrumentation strategy without affecting the 42.17 baseline.

### 7.5 Files / commands to touch

**Methodology — per-layer `mx.eval` timing barriers, NOT layer-skip-subtraction.** Skipping DeltaNet layers in a hybrid model breaks the recurrent-state pipeline subsequent layers depend on (Qwen3.5-27B's interleaved 3:1 linear:full pattern means a skipped DeltaNet layer at index `4i+0/1/2` corrupts the hidden state feeding the full-attention layer at `4i+3`); skipping full-attn layers similarly corrupts the residual stream feeding the next DeltaNet block. The "skip and time the difference" approach is therefore not safe on hybrid Qwen3.5. The microbench uses **per-layer barriers**: wrap each layer's forward with `mx.eval(out)` boundaries and record `time.perf_counter_ns()` at each boundary. Forward semantics are preserved; cost attribution is direct, not by subtraction.

**New file (autonomous-loop scope):**

- `silica/bench/microbench/decode_step_attribution.py` — per-layer-barrier instrumented forward. Pattern: reuse the `microbench_capture_hidden.py` reused-adapter shape (3 warmup iters + 20 measurement iters, median wall, materialisation forced via `mx.eval`). Per-iteration loop:
  1. Time the full step (uninstrumented baseline) — to compute instrumentation overhead.
  2. Time the per-layer split via an instrumented variant `decode_step_per_layer_timed(token, handle)` that walks `model.layers` directly (not via the model's top-level `__call__`), inserts `mx.eval(layer_out)` barriers between layers, and records `(layer_idx, layer_kind, ns_elapsed)` for each layer. Layer kind taken from `layer.is_linear` (DeltaNet) or its absence (full attention).
  3. Time `lm_head` projection in isolation by capturing the post-stack residual once and timing the `lm_head(residual)` step separately.
  4. Record per-step Python wall clock around `mx.eval` boundaries to measure scheduler / Python loop overhead.
  5. Optionally: a second variant that batches per-layer barriers into per-block (4-layer pattern) barriers, to amortise the per-barrier sync cost while still separating linear-block cost from full-attn-block cost.
- `tests/test_decode_step_attribution.py` — unit tests on a tiny synthetic Qwen3.5-style model fixture; verify (i) the per-layer barrier instrumentation produces a sum-of-component time within ≤5% of the uninstrumented baseline (instrumentation overhead bound), (ii) per-layer attribution sums to the total step time within fp16 noise, (iii) `is_linear` / full-attn split returns the expected layer counts (48 / 16 on a synthetic 64-layer config; 16 / 0 / etc. on smaller fixtures).

**No silica.* runtime code change.** The instrumented forward is a new bench-side microbench module that imports the production adapter; it does not modify production paths. The per-layer walk reuses `Qwen3_5Adapter._attn_layer_indices` (silica/models/qwen3_5.py:100-104) for the layer-kind filter without forking the production forward.

**Run command:**

```text
SILICA_REAL_QWEN3_5_27B=1 \
    uv run python -m silica.bench.microbench.decode_step_attribution \
        --repo mlx-community/Qwen3.5-27B-4bit --b 4 --warmup 3 --iters 20 \
        --out plans/P6_AUTORESEARCH/decode_step_attribution_b4.jsonl
```

### 7.6 Expected artefact

`plans/P6_AUTORESEARCH/decode_step_attribution_b4.jsonl` — one row per (component, iter) pair plus an aggregated row. Schema:

```jsonc
{
  "component": "full_attn" | "delta_net" | "rmsnorm_rope" | "lm_head" | "python_loop" | "step_total",
  "b": 4,
  "iter": 0,
  "wall_ms": 12.34,
  "layers": 16,                  // 16 for full_attn; 48 for delta_net; 64 for rmsnorm_rope; 1 otherwise
  "method": "subtraction" | "direct",
  "ts": "2026-05-02T..."
}
```

`plans/P6_AUTORESEARCH/decode_step_attribution_REPORT.md` — short summary (≤ 1 page) tying the JSONL into a per-component-percentage table, with a kernel-candidacy disposition per the §7.4 thresholds.

### 7.7 Run cost

Expected wall: ~5 minutes (3 warmup + 20 measurement iters at ~0.1 s/step × multiple component-isolation passes). Within the autonomous-loop "no long real-model benchmark over 10 minutes" rule. **No download** (mlx-community/Qwen3.5-27B-4bit is the cached production target).

### 7.8 What this probe does NOT do

- It does not run on MoE. Cross-family attribution is a follow-up after dense disposition lands.
- It does not write any custom kernel. Kernel work is gated behind explicit user authorisation per P6_AUTORESEARCH.md.
- It does not change PLAN.md. PLAN updates land only after a kernel candidate clears its end-to-end gate (per P6_AUTORESEARCH.md Custom kernel authorization).
- It does not exercise spec. Spec-side decomposition needs a separate microbench harness on `decode_step_multi(k=4)`.

---

## 8. Files / Commands to Touch (full inventory for the autoresearch loop)

This memo's deliverables, in order:

1. **This memo** at `plans/P6_AUTORESEARCH_REORIENTATION.md`.
2. **Experiment ledger** at `plans/P6_AUTORESEARCH_LOG.tsv` — header + initial 6 rows seeded from existing measurements (baseline / C.4 / Track B / C.5 β.1 / C.5 β.2 / verify-k microbench) so the running-best line starts at 42.17 with a populated history.
3. **Progress summary** at `plans/P6_AUTORESEARCH_PROGRESS.md` — short human-readable index over the TSV. **No .png chart in this commit** — matplotlib is not in `pyproject.toml` deps; per P6_AUTORESEARCH.md "do not add new plotting dependencies unless explicitly approved", the chart is queued for the first new keep-or-diagnostic experiment under explicit approval.

For the recommended next probe in §7:

4. **New microbench module** at `silica/bench/microbench/decode_step_attribution.py`.
5. **New tests** at `tests/test_decode_step_attribution.py`.
6. **New artefact directory** at `plans/P6_AUTORESEARCH/` — measurement JSONLs + per-probe REPORT.md files, paralleling the `plans/P6_C5_DDTREE/` shape.

For follow-up probes (§6 ranked items 2-5), the same `plans/P6_AUTORESEARCH/` directory holds the artefacts; new microbench modules under `silica/bench/microbench/` follow the same pattern.

---

## 9. Approval Requests

Per P6_AUTORESEARCH.md "You must ask before:" list, this memo lands without any of the gated actions. Explicit asks for the next iteration:

| # | Action | Required for | Authorisation status |
| --- | --- | --- | --- |
| A1 | **No download / no network call** in the next-probe (§7) — uses the cached `mlx-community/Qwen3.5-27B-4bit` target only. | Probe execution. | **Autonomous-loop scope; no ask needed.** |
| A2 | **No git commit** as part of this memo / ledger / probe. | Memo + ledger artefact landing. | **Per P6_AUTORESEARCH.md "creating any git commit (every commit needs explicit user approval, even when the toolchain is green)"; user must explicitly authorise the commit when ready.** |
| A3 | **Long real-model benchmark > 10 min** in any follow-up. | Future probe execution. | Not requested in this memo. |
| A4 | ~~`safetensors.index.json` inspection~~ | ~~Resolve OQ-4~~ | **DONE 2026-05-02 — RESOLVED NEGATIVE (MTP keys absent; see §1.4 OQ-4 closure and §5.1).** Ledger row `AR_MTP_KEY_INSPECTION`. |
| A5 | **C.5 γ.1 read-only upstream survey** on `humanrouter/ddtree-mlx` (item 3 in §6, the C.5 REPORT's standing user decision). | Resolve C.5 escalate state. | **Requires user decision per the C.5 REPORT engineering recommendation.** Not pre-empted by this memo. |
| A6 | **Opening a custom MLX-native kernel candidate** for hot-path integration (any of fused RMSNorm+RoPE, fused gated-delta-update, FlashAttention-style SDPA, paged-KV scatter). | Once §7 microbench identifies a measurable gap. | **Requires explicit user authorisation per P6_AUTORESEARCH.md "shipping a custom MLX Metal kernel into the Silica hot path".** |
| A7 | **MTP path re-opening** via Option A (re-convert upstream Qwen/Qwen3.5-27B BF16, ~52 GB download + conversion) or Option B (download trevon/Qwen3.5-27B-MLX-MTP, 8-bit 29.1 GB). | MTP probe is retired for the current production target (A4 resolved negative); re-opening requires a new checkpoint. | **Requires explicit user authorisation per P6_AUTORESEARCH.md "checkpoint downloads" + "opening a large implementation track".** Not requested. |
| A8 | **QuantSpec / KnapSpec probe implementation** (touches `silica.kvcache` or new self-spec drafter). | If MTP retires AND microbench leaves spec lever as the path. | **Requires explicit user authorisation.** |
| A9 | **OptiQ-4bit download (~16 GB) + PPL row** (§5.6). | Quality-floor recovery only — not a current-cycle probe. | **Requires explicit user authorisation.** Not requested. |
| A10 | **Re-opening C.4 DFlash via `--quantize-draft`** (the (1) drafter-cost finding's only open follow-up). | Only if a new-evidence hypothesis emerges. | **Requires explicit user approval per P6_AUTORESEARCH.md "re-opening any retired track".** |
| A11 | **PLAN.md final-disposition edit** (e.g. closing C.5 step 8, retiring (1b), updating §3.2 kernel non-goal once a kernel ships). | After the relevant decision is empirical. | **Per P6_AUTORESEARCH.md "committing final strategic decisions to PLAN" — explicit user authorisation required.** |

---

## 10. Progress Visualization Plan

### 10.1 Ledger

`plans/P6_AUTORESEARCH_LOG.tsv` — machine-readable experiment ledger. Columns per P6_AUTORESEARCH.md spec:

```
experiment_id  date  commit  track  hypothesis  metric_name  metric_value  baseline_value  relative_delta  direction  status  artifact_path  notes
```

Initial seeded rows (existing measurements, mapped onto the ledger schema):

- `BASELINE_27B_B4` — dense 27B B=4 = 42.17 tok/s, primary metric anchor (status: `diagnostic` because it is not an experiment but the baseline; running-best line starts here).
- `C4_DFLASH_RETIRE` — spec-on 7.74 tok/s = 0.482×, status `discard`.
- `TRACK_B_3BIT_RETIRE` — ΔPPL_rel +16.85%, status `discard` (PPL gate; not on tok/s axis).
- `C5_DDTREE_BETA1` — spec-on 6.54 tok/s = 0.408×, status `discard` on tok/s axis but `diagnostic` for the cost-leg evidence.
- `C5_DDTREE_BETA2_COVERAGE` — coverage@16 = 0.258, status `diagnostic` (escalate disposition; not on tok/s axis).
- `P605_VERIFY_K` — verify-k cap 2.93× at k=8 perfect, status `diagnostic` (microbench, not optimisation).

The TSV file is written by the main agent only (per P6_AUTORESEARCH.md "Ledger ownership"); sub-agents return findings, the main agent appends.

### 10.2 Charts

Per P6_AUTORESEARCH.md "do not add new plotting dependencies unless explicitly approved" and the local check showing matplotlib is not in `pyproject.toml`, **no .png chart in this commit**. The chart generation is queued for the first new keep-or-diagnostic experiment with explicit user approval to either install matplotlib or generate the chart out-of-band (e.g. via a separate venv).

When the chart is generated, the structure follows P6_AUTORESEARCH.md spec:

- `plans/P6_AUTORESEARCH_PROGRESS_DECODE_TOK_S.png` — primary metric (higher is better); discarded experiments as small gray points; kept improvements as green points; running best as a green step line. Title: "Silica-MLX dense 27B B=4 autoresearch — N experiments, K kept improvements".
- `plans/P6_AUTORESEARCH_PROGRESS_PPL.png` — quality metric (lower is better) for any compression / codec experiment.
- `plans/P6_AUTORESEARCH_PROGRESS_SPEC_ACCEPT_RATE.png` — spec accept rate (higher is better).
- `plans/P6_AUTORESEARCH_PROGRESS_MEMORY_MB.png` — memory peak (lower is better).

A breakthrough is labeled on the relevant chart. Diagnostic-only experiments are logged but not forced into the running-best curve unless they have a comparable metric.

### 10.3 Primary metric family for the next experiment cycle

`decode_tok_s` on the `qwen3.5-27b-warm-decode-*` row family. The §7 next-probe is diagnostic (per-component time attribution); its keep / discard / diagnostic disposition is recorded in the ledger but does not change the running-best line on `decode_tok_s` directly. A keep on the running-best line requires a new dense 27B warm-decode aggregate measurement that improves 42.17 by ≥3σ (= ≥0.63 tok/s = ≥42.80 tok/s) on ≥2 reproductions.

---

## 11. Custom-Kernel Candidacy Review

Per P6_AUTORESEARCH.md section 14 — "list any place where a microbench shows the existing MLX algorithm is on the dense 27B critical path AND clearly below an achievable kernel ceiling. If none, say so explicitly — do not invent candidates."

**As of this orientation, NO microbench has yet shown that an existing MLX algorithm is below an achievable kernel ceiling on the dense 27B critical path.** The verify-k microbench (P-6.0.5 Unit 7) shows the bandwidth-utilisation regime transition at k=4 but does not attribute it to a specific MLX op. The B=4 warm-decode at 52% utilisation is consistent with compute-bound regime but does not name the bottleneck op.

The §7 recommended probe is precisely the microbench that would close this gap. **No specific kernel candidate is proposed in this memo** — proposing one would invent it, which P6_AUTORESEARCH.md forbids.

The kernel ecosystem survey (§4.1) identified four reference patterns that would become candidates **if** §7 surfaces a measurable gap. Each pattern's candidacy is constrained by the verified architecture facts in §1.5:

1. **Fused RMSNorm + RoPE (with partial-rotary at 25%)** — if §7 attributes >15% of step time to small-op chains in the QK-norm + rotate path. Reference: ZMLX RMSNorm + turboquant `prerot_fused_qk_scores`. The kernel must respect `partial_rotary_factor: 0.25` (RoPE applied to the first 25% of head_dim only, not the full vector). Per the vllm-metal RFC #188 constraint, this should be one fused kernel rather than two chained ones.
2. **Fused gated-delta-update** for the 48 DeltaNet layers — if §7 attributes >25% of step time to DeltaNet recurrent updates AND mlx-lm's stock `gated_delta_update` measurably underperforms a fused alternative. The kernel must handle the layer's full state shape: 16 K heads + 48 V heads × head_dim=128, 1D conv with kernel_dim=4. Reference: ddtree-mlx kernel for tree-aware GatedDelta recurrence (although their target is tree-shape spec, the recurrence kernel itself transfers).
3. **FlashAttention-style fused Gated SDPA** for the 16 full-attention layers — if §7 attributes >20% of step time to full-attention layers AND M5 Pro Metal SDPA path measurably underperforms a fused alternative. **Critical constraint:** `attn_output_gate: true` per §1.5 — the kernel must implement `out = sigmoid(g) * SDPA(Q, K, V)` not just standard SDPA. Standard FlashAttention-2 implementations do NOT include the output gate; copying patterns from MFA / dflash-mlx wholesale would produce a numerically wrong result. Shape: 24 Q heads, 4 KV heads (GQA 6:1), head_dim=256, B=4. Reference: dflash-mlx JIT SDPA 2-pass + MFA Swift port (pattern only — output-gate must be added).
4. **Quantised matmul fast-path** for the `lm_head` projection — if §7 attributes >10% of step time to logits projection (5120, 248320 — `tie_word_embeddings: false` per §1.5 means lm_head is a separate weight, not embedding-tied). Reference: dflash-mlx `verify_qmm` int4 simdgroup-MMA for M=16 (M=4 for B=4 dense decode is a different kernel-shape regime than the spec-verify M=16 the dflash-mlx kernel was tuned for).

None of these is opened by this memo. The §7 probe gates each.

**Important constraint to carry forward:** vllm-metal RFC #188 documents that `mx.fast.metal_kernel`-registered kernels cannot return lazy arrays the way internal MLX primitives can; per-layer custom-kernel chains risk introducing a per-layer sync barrier. Any custom-kernel proposal must aggregate multiple ops into one kernel rather than chaining single-op kernels — which biases candidate selection toward fused-multi-op kernels (e.g. RMSNorm + RoPE + KV-write in one pass) and away from single-op replacements.

---

## 12. Cross-references

- `plans/PLAN.md` §1 status header (current at v1.7.21+pending C.5); §3.1-§3.2 (scope including the kernel non-goal now relaxed); §6 acceptance gates (1a / 1b / 2a / 2b); §7 D-021 step 4 / step 6 / step 7 / step 8 phase blocks; §13 changelog v1.7.20 / v1.7.21 entries.
- `plans/P6_OPENING.md` — original P-6 framing, five orthogonal tracks.
- `plans/P6_REVIEW_HANDOFF.md` — external-reviewer handoff with verification map.
- `plans/P6_0_5_BASELINE/REPORT.md` — corrected 15.13 GB weight anchor; §1 Q1-Q4 closures; verify-k 2.93× cap.
- `plans/P6_0_5_BASELINE/target_verify_microbench.md` — full Unit 7 microbench detail; weight-footprint reconciliation note.
- `plans/P6_0_DECISION_GATE_1_OPENING.md` — (1a) ≥40 unchanged / (1b) two-condition survival rule / (2b) ≥175 anchor.
- `plans/P6_SPEC_FOUNDATION_OPENING.md` §6.1 / §6.3 — D-021 step 5 closure (single-request spec live; (c) slice 3 deferred).
- `plans/P6_C4_DFLASH/REPORT.md` (η.1) — C.4 retire postmortem.
- `plans/P6_TRACK_B/REPORT.md` (B.2) — Track B PPL gate failure.
- `plans/P6_TRACK_B_FOLLOWUP_SURVEY.md` — empty follow-up survey close.
- `plans/P6_C5_DDTREE_OPENING.md` §9 — escalate decision matrix.
- `plans/P6_C5_DDTREE/REPORT.md` — β.1 + β.2 measurement bundle; engineering recommendation.

External anchors carried in §4:
- [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B); [`mlx-community/Qwen3.5-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit); [`mlx-community/Qwen3.5-27B-OptiQ-4bit`](https://huggingface.co/mlx-community/Qwen3.5-27B-OptiQ-4bit); [`trevon/Qwen3.5-27B-MLX-MTP`](https://huggingface.co/trevon/Qwen3.5-27B-MLX-MTP)
- [mlx.fast.metal_kernel docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html); [vllm-metal RFC #188](https://github.com/vllm-project/vllm-metal/issues/188); [mlx-lm PR #1067](https://github.com/ml-explore/mlx-lm/pull/1067)
- [Hmbown/ZMLX](https://github.com/Hmbown/ZMLX); [arozanov/turboquant-mlx](https://github.com/arozanov/turboquant-mlx); [bstnxbt/dflash-mlx](https://github.com/bstnxbt/dflash-mlx); [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx); [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention); [humanrouter/ddtree-mlx](https://github.com/humanrouter/ddtree-mlx); [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)
- [QuantSpec arXiv 2502.10424](https://arxiv.org/abs/2502.10424); [Apple Speculative Streaming 2402.11131](https://arxiv.org/abs/2402.11131); [Mirror-SD 2510.13161](https://arxiv.org/abs/2510.13161); [SSD/Saguaro 2603.03251](https://arxiv.org/abs/2603.03251); [SSSD 2411.05894](https://arxiv.org/abs/2411.05894); [EAGLE-3 2503.01840](https://arxiv.org/abs/2503.01840); [KnapSpec 2602.20217](https://arxiv.org/abs/2602.20217); [FastMTP 2509.18362](https://arxiv.org/abs/2509.18362)

---

## 13. Stop conditions (carried from P6_AUTORESEARCH.md, instantiated for this loop)

The autoresearch loop must surface one of these before declaring success:

1. **A reproduced ≥60 tok/s aggregate measurement** on the dense 27B `qwen3.5-27b-warm-decode-*` row family, on ≥2 runs with σ-bounded confidence (σ = 0.21 tok/s; 3σ = 0.63 tok/s).
2. **A reproduced new running-best ≥3σ above 42.17** that is also a measurement-anchored step on the hardware-limit ladder, with a clean attribution to which lever family delivered it (bandwidth utilisation / bytes-per-step / spec amortisation / scheduler overlap / kernel fusion).
3. **A measurement-anchored declaration** that the remaining open-lever set cannot multiplicatively reach 60, with each retired lever having an artifact and a postmortem.

"Continue forever" is not a stop condition. The §6 ranked hypotheses queue the levers; the §7 next-probe starts the empirical work.

---

## 14. Appendix — running-best frame initial state

| Frame | Value | Source |
| --- | --- | --- |
| Primary metric | `decode_tok_s` (higher is better) | P6_AUTORESEARCH.md mission statement |
| Primary row family | `qwen3.5-27b-warm-decode-*` | P6_AUTORESEARCH.md hardware-limit map |
| Running best | **42.17 tok/s** | P-6.0.5 Unit 2, B=4, 2-run mean |
| σ (run-to-run) | 0.21 tok/s | P-6.0.5 Unit 2, 2-run |
| 3σ floor for keep | 0.63 tok/s (a kept improvement must measure ≥ 42.80 tok/s on ≥2 reproductions) | derived |
| Stretch milestone | 60 tok/s ((1b)) | PLAN.md §6 (1b) |
| Hypothetical envelope (composed multi-lever) | 80-170 tok/s | P6_AUTORESEARCH.md prompt — to be refined by measurement |
| Refined envelope (this memo) | 60-100 tok/s mid-band, with kernel + spec composition | §3.1 of this memo |
| Hard cap on spec-only (linear k=8 perfect) | 2.93× × 16.05 ≈ 47 tok/s | P-6.0.5 Unit 7 |

---

End of memo. Awaiting user disposition on §9 approval requests; default next action under autonomous-loop rules is the §7 microbench probe.
