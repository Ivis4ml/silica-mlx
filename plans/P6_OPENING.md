# P-6 Opening — Performance Phase

| Field         | Value                                                                                                       |
| ------------- | ----------------------------------------------------------------------------------------------------------- |
| Phase         | P-6 (Performance Phase — re-scoped from "Weight Streaming" in PLAN.md v1.7.x; see proposed D-017 in §10)     |
| Status        | drafted; pending user review and PLAN.md edits                                                              |
| Last updated  | 2026-04-27                                                                                                  |
| Scope owner   | Xin Zhou                                                                                                    |
| Predecessors  | P-1 .. P-5 (all done; per PLAN.md v1.7.12 status header)                                                    |
| Successors    | P-7 (Speculative — promotion to T1 proposed below), P-8 (Mini-SGLang serving)                              |

This document is the opening for what was originally PLAN.md §P-6 ("Weight
Streaming"). The original P-6 is preserved as a sub-track of this phase
(Track C below); the phase as a whole is re-scoped from "weight streaming
alone" to **"the performance phase"** because the user's stated goals for
v0.1 launch — TTFT, decode tok/s, and RAM headroom — cannot be addressed by
weight streaming in isolation. The re-scoping is recorded as proposed
Decision D-017 in §10.

---

## 1. Targets and the Bandwidth Reality

### 1.1 User-stated goals

The user's brief on 2026-04-27 was: "make TTFT faster, tokens-per-second
faster, RAM smaller; we want to support the largest models on a 48 GB Mac
Pro; **at minimum, Qwen3.5-27B should reach 100+ tokens/sec**."

### 1.2 The bandwidth ceiling for dense Qwen3.5-27B-4bit on M5 Pro

A single autoregressive decode step must read every weight participating in
that step. For dense Qwen3.5-27B at MLX-native 4-bit quantization:

- Weight footprint per step ≈ 27B × 0.5 B/param = **~13.5 GB per token**
- M5 Pro unified-memory bandwidth = **307 GB/s** (verified against Apple's
  M5 Pro spec sheet, March 2026 launch; M4 Pro was 273 GB/s; M5 Max is
  higher but out of scope for the 48 GB target machine).
- Pure weight-bandwidth ceiling = 307 / 13.5 ≈ **22.7 tok/s**.

After accounting for KV reads, activation traffic, sampler, and per-step
Python overhead, a realistic upper bound is **~20 tok/s** on the dense
27B-4bit autoregressive path.

**Reconciliation with public benchmarks.** Two reported numbers exist for
27-32B-class dense 4-bit on M-series and they do not agree:

- **Qwen3.6-27B Q4_K_M @ 25.57 tok/s single-stream** (Simon Willison
  clock, flagship Mac) — sits within ~15% of the 22.7 tok/s ceiling and
  is consistent with the bandwidth math.
- **Qwen 32B 4-bit @ 42-50 tok/s on "M5 Pro 48 GB" @8K context**
  (third-party blog roundup) — exceeds the 19.2 tok/s ceiling for 32B
  by ~2×, which is **physically impossible at face value**. Plausible
  resolutions: (a) the source conflates M5 Pro and M5 Max (M5 Max
  bandwidth is roughly 1.5 × M5 Pro); (b) the headline number averages
  the much-faster prefill phase into the reported tok/s; (c) the
  benchmark uses GQA-aware partial weight reads that the naive
  "13.5 GB/step" formula over-counts. Without a reproducible run, this
  plan **discounts the 42-50 figure** and anchors to the 25.57 tok/s
  number, which both agrees with the bandwidth ceiling and is closest
  to silica's actual code path.
- **Conservatism caveat in the bandwidth formula:** GQA reduces some
  KV traffic but does not change the weight read per step; group
  quantization adds metadata (scales / zeros) that pushes the effective
  bytes/param above 0.5; layer-fused matmuls let SLC/L2 absorb a
  fraction of inter-layer reads. Net effect: the real ceiling is
  probably within ±20% of 22.7 tok/s, not 2× off. P-6.0 measures the
  real number; the dual-target reframing in §1.3 holds against either
  end of the ±20% band.

The only published >100 tok/s figure on Apple Silicon for 27B-class
checkpoints comes from MoE active-3B variants (Qwen3-30B-A3B-4bit:
127.7 tok/s on M4 Max via vllm-mlx) — bandwidth math agrees with this:
1.5 GB/step at 273 GB/s gives a 182 tok/s ceiling, well above the
reported 127.7.

### 1.3 What 100 tok/s actually requires on dense 27B

To reach 100 tok/s on dense Qwen3.5-27B from a ~20 tok/s autoregressive
floor, we need a multiplicative speedup of ~5×. The realistic levers:

| Lever                                | Multiplier          | Source                                        |
| ------------------------------------ | ------------------- | --------------------------------------------- |
| Speculative decoding (4-token draft) | 1.4 – 1.8×          | LM Studio reports 1.5-3× on similar shapes; dense acceptance is typically 50-70% |
| 3-bit weight option                  | 1.3×                | bytes/step 13.5 → 10.1 GB; ceiling 22.7 → 30.3 tok/s |
| Engine fusion (sampler, sync)        | 1.05 – 1.20×        | vllm-metal RFC #188 sync-barrier collapse + `mx.compile`-fused sampler |
| Apple ReDrafter (best-case spec)     | 2.0 – 2.3×          | Apple ML Research claim, MLX-native, but requires KD training |

Stacked: 1.6 × 1.3 × 1.15 = **2.4×** — about **48-55 tok/s on dense 27B**
under realistic conditions. Stretch (ReDrafter + 3-bit + fusion):
2.3 × 1.3 × 1.15 = **3.4×** ≈ **65-77 tok/s**.

**100 tok/s on dense Qwen3.5-27B on a 48 GB M5 Pro is not credibly
reachable** with the techniques available to silica-mlx in 2026. We
therefore propose a **dual-target reframing** that respects the user's
explicit framing (Qwen3.5-27B as the primary target):

- **Honest dense primary target:** **≥60 tok/s** on
  `mlx-community/Qwen3.5-27B-4bit` (and ≥55 tok/s on Gemma4-31B-4bit),
  warm-start B=1, 128-token prompt, 256-token generation. Achievable
  with everything stacked (Tracks A+B+C). This is the floor we commit
  to for v0.1 launch and the gate the phase actually exits on.
- **Stretch target validating the stack:** **≥100 tok/s** on
  `mlx-community/Qwen3.5-35B-A3B-4bit` (active 3B; bytes-per-step
  ≈ 1.5 GB; bandwidth ceiling ~200 tok/s; vllm-mlx already shows 127
  tok/s on M4 Max). The MoE shape is where the platform's
  active-parameter math earns the 100-tok/s figure outright; missing
  this gate is informational, missing the dense-60 gate is a phase
  failure.

The user has the final word on this re-target — see §11 for the
explicit decision request.

### 1.3a Why dense 27B is the primary target and MoE 100 tok/s is the stretch

(a) The user's brief named Qwen3.5-27B explicitly. Promoting MoE to the
primary slot and demoting the stated target to a fallback would
misrepresent priority. (b) Dense 27B is constrained by the chip's
bandwidth, not silica's engine — the dense gate measures whether the
platform extracts the chip's available throughput, which is the right
correctness signal. (c) The MoE 100-tok/s figure exists primarily to
validate that the optimization stack (sync collapse, sampler fusion,
chunked prefill, speculative, per-expert streaming) lands cleanly on
the hardest engine path silica supports. If the stack works on MoE,
all the dense levers are working too; treating MoE as the stretch
makes that validation visible. (d) Dense 27B at 100+ tok/s on 48 GB is
the kind of claim that requires M5 Max or speculative-with-very-good-
acceptance; v0.1 ships at 48 GB M5 Pro and silica should not commit to
a number the chip cannot deliver.

### 1.4 TTFT and RAM targets

- **TTFT (warm-start, B=1, 128-token prompt):** ≤ 200 ms on
  Qwen3.5-27B-4bit / ≤ 250 ms on Gemma4-31B-4bit / ≤ 80 ms on the MoE
  35B-A3B target. (Cold-start with kernel compile is excluded — that's
  the ~2.4 s figure observed in the v1.7.x 27B load probe and is a
  one-time cost.)
- **TTFT-under-concurrency:** the long-prompt request must not block
  short-prompt requests by more than 2× their solo TTFT — directly
  resolves Q-010 in PLAN.md (chunked prefill promotion).
- **Peak resident memory (Qwen3.5-27B-4bit, B=1, 4K context):** ≤ 36 GB,
  leaving 12 GB system headroom on a 48 GB machine. The v1.7.14
  P5.9 step 2(a) corrected probe reports ~15.3 GB peak on a 1-token
  forward (was ~30.5 GB at v1.6.1 due to probe double-load); P-6.0
  warm-decode B=1 at 384-token gen reports peak 15.4 GB, confirming
  sustained-decode peak matches the probe number. Real headroom at
  the 4K-context gate is therefore ~21 GB, not the ~5 GB previously
  assumed.
- **Concurrent request capacity (Qwen3.5-27B-4bit, mixed 512-token
  prompts):** ≥ 4 requests sustained, with admission-headroom-style
  evidence that BlockTQ KV codec actually translates into more admitted
  requests at 27B scale (currently only validated on 0.6B at v1.7.4).

---

## 2. Step 0 — Measurement Gate (Required Before Any Track Lands)

The current code state has **no warm-start sustained-decode measurement**
on the 27B / 31B / MoE targets. The v1.7.x P-5 acceptance sweep is 0.6B
cache-only; the 2026-04-19 27B load probe is a single-forward-with-
kernel-compile run, not a meaningful throughput baseline. Without a real
baseline, every optimization target above is hand-waving.

**P-6 sub-unit P-6.0 (must land first):**

- New scenarios in `silica.bench.scenarios.BUILTIN_SCENARIOS`:
  - `qwen3.5-27b-warm-decode-b1` — dual-gated on
    `SILICA_REAL_QWEN3_5_27B`. Workload: 128-token prompt, 384-token
    generation. **Warm-up rule: discard the first 32 decode steps OR
    until the rolling 16-step decode-tok/s standard deviation falls
    below 5% of the rolling mean, whichever comes later.** The 32-step
    floor matters because MLX's first-forward kernel-compile cost
    can dominate the first 30-50 forwards on a 64-layer 27B model
    (already observed in the v1.7.x load probe at ~2.4 s for a
    1-token prompt). Measurement window: the next 256 decodes after
    warm-up. Reports `decode_tok_s_warm`, `decode_tok_s_warm_std`,
    `warmup_steps_used`, `ttft_warm_ms`, `peak_mb`,
    `resident_mb_post_warmup`.
  - `qwen3.5-27b-warm-decode-b{2,4,8}` — same prompt structure, batched
    via `Engine.generate_batch`. Reports per-batch `decode_tok_s_aggregate`
    and `decode_tok_s_per_row`.
  - `gemma4-31b-warm-decode-b1` and `b{2,4}` — same, dual-gated on
    `SILICA_REAL_GEMMA4_31B`.
  - `qwen3.5-moe-35b-a3b-warm-decode-b1` and `b{2,4,8}` — dual-gated on
    `SILICA_REAL_QWEN3_5_MOE`.
- Bench runner change: extend `ScenarioResult.metadata` with the
  `warm_decode_*` fields; nothing else changes (this is purely additive).
- Acceptance: a single `python -m scripts.bench --scenario
  qwen3.5-27b-warm-decode-b1` run on a real 48 GB M5 Pro produces a
  number. The number itself does not have to meet any target; it just
  has to exist. Without it, nothing else in this phase has a reference
  point.

**P-6.0 is the gate**: every later sub-unit's success criterion is a
ratio against the P-6.0 baseline, not an absolute number. If P-6.0 ships
a 27B decode tok/s of 18, the speculative track's gate is "≥ 1.5 ×
baseline = 27 tok/s," not "≥ 100 tok/s." The 100 tok/s figure is the
phase-level success criterion, not a per-sub-unit gate.

---

## 3. Tracks — A through E

The phase decomposes into five orthogonal tracks. They are independent in
the sense that landing one does not require any other; they are stackable
in the sense that all of their wins multiply (the bandwidth analysis in
§1.3 assumes they all land).

Tracks are ranked by `(impact / risk-and-complexity)`. **Note (D-021,
v1.7.14):** the *execution order* committed at v1.7.14 is **not** the
ranked order below — the ranked order is "single-track ROI on a
homogeneous workload." Per D-021 the actual phase sequencing is
P5.9 hardening → P-6.0.5 measurement expansion → Decision Gate 1 →
spec foundation + C.1 → C.4 spike → B → C.5 / C.2 / C.3 → A → D / E.
Track A defers behind the speculative foundation because A's
+5-15% on bandwidth-bound dense is invisible without spec running
on top; A's +30-80% MoE leverage is real but lands on a target
that already cleared its baseline gate. A is documented below as
"general efficiency + MoE amplifier," not as the dense gate
cracker.

### Track A — Sync-Barrier Collapse

**Hypothesis (vllm-metal RFC #188):** silica's per-layer dispatch through
mlx-lm currently inserts `2 × n_layers` CPU↔GPU sync barriers per forward
because `silica/scheduler/batcher.py:1497`'s `snapshot_recurrent_state`
calls `mx.eval` per row per layer (see qwen3_5.py:384's `_detach_row`),
and `silica/engine/__init__.py:132,153` calls `int(token_scalar.item())`
on every decode token. On Qwen3.5-27B with 64 layers, this is up to 128
barriers per step plus a forced sync per token — one of the largest
non-physical losses in the current code path.

**Sub-units:**

- **A.1 Defer-and-batch sampler sync.** `Engine.generate` /
  `ContinuousBatcher` currently calls `.item()` per token (lines 132,
  153, 1546). Replace with a batched-categorical path that returns
  `mx.array(B,)` of token ids and only materializes scalars at the
  history-update / stop-check boundary. The history doesn't need to be
  an `mx.array` at all — it's only used by the repetition-penalty
  processor, which can be re-implemented as a stateless update on a
  small token-count vector. (See `silica/core/sampler.py:55-90` for the
  current processor chain.)
- **A.2 `mx.compile`-fused sampler.** Wrap the
  `temperature → repetition penalty → top-k → top-p → categorical`
  chain in a single `mx.compile`-decorated function returning sampled
  token ids. Top-k currently uses a full sort
  (`silica/core/sampler.py:82`); `mx.partition` is the right primitive
  and removes ~`O(V log V)` per row per step.
- **A.3 Lazy-graph snapshot capture.** `Qwen3_5Adapter.snapshot_recurrent_state`
  (`silica/models/qwen3_5.py:332-400`) currently calls `mx.eval` per
  layer per row inside `_detach_row` (line 384). Refactor to build a
  single per-batch graph that materializes once at the block-aligned
  boundary; let MLX's lazy scheduler fuse the per-layer detaches.
  Block-aligned snapshots already happen every `block_size=4` decode
  steps (cross-turn prefix reuse, v1.7.x), so the win compounds with
  any chat-style workload.

**Track A acceptance gates** (relative to P-6.0):

- A.1+A.2 ship together: decode_tok_s ≥ 1.10 × baseline on 0.6B and
  ≥ 1.05 × baseline on 27B.
- A.3 ships standalone: per-block snapshot wall ≤ 0.5 × baseline on
  Qwen3.5-0.8B + cross-turn prefix-reuse workload.

**Estimated impact (combined) — regime-dependent:**

- **Bandwidth-bound dense path (Qwen3.5-27B-4bit on M5 Pro): +5-15%.**
  When the per-step cost is dominated by reading 13.5 GB of weights at
  307 GB/s, the synchronization barriers and sampler overhead are
  largely *hidden* in the bandwidth wait — fixing them removes Python
  / dispatch overhead but cannot move the bandwidth ceiling. This is
  the main reason the §1.3 dense-target arithmetic uses 1.15 ×.
- **Compute-bound MoE active-3B path (Qwen3.5-35B-A3B-4bit): +30-80%.**
  Active weight read is ~1.5 GB per step, leaving the bandwidth pipe
  ~70-80% idle; sync-barrier overhead and sampler dispatch are now
  the dominant cost. This is the regime the vllm-metal RFC #188
  analysis applies to directly, and the figure underwrites the §1.3
  MoE stretch target. Reaching ≥100 tok/s on the MoE shape rests on
  Track A landing in the upper half of this band.
- **Snapshot-heavy hybrid path (Qwen3.5-0.8B / 27B with cross-turn
  prefix reuse and frequent block-aligned snapshots): +10-25%
  on per-block snapshot wall.** A.3 only.

**Complexity:** Small-Medium. All wins are pure Python / MLX graph work.
No new kernels, no new dependencies.

### Track B — 3-bit Weight Option

**Hypothesis:** Qwen3.5-class checkpoints survive 3-bit weight
quantization with a small PPL hit (precedent: `unsloth/Qwen3.6-27B-UD-MLX-3bit`
ships and is used in production on M-series). A 3-bit weight tier reduces
bytes-per-step from ~13.5 GB → ~10.1 GB on dense 27B, lifting the
bandwidth ceiling from 22.7 to 30.3 tok/s — pure free win if quality
holds.

**Sub-units:**

- **B.1 Loader path for 3-bit checkpoints.** `silica/weights/resident.py`
  is dict-backed and format-agnostic; the work is in
  `silica/mlx/runner.py` and the model factory (`silica/models/factory.py`)
  to surface 3-bit as a first-class quantization tier alongside the
  existing 4-bit / 8-bit. mlx-lm already supports 3-bit at the kernel
  level (`mlx_lm.convert -q --bits 3`); silica has to declare it
  acceptable and tell the bench harness about it.
- **B.2 Quality cross-check.** Add a bench oracle row
  `qwen3.5-27b-3bit-vs-4bit-ppl` running the existing PPL chunked-NLL
  oracle on WikiText-2 against both a 4-bit and 3-bit version of the
  same checkpoint. Acceptance: ΔPPL ≤ 0.5 absolute, or ΔPPL ≤ 5%
  relative — whichever is tighter. If it fails, flag the 3-bit path as
  "available with caveats" and do not promote to default.

**Track B acceptance gates:**

- B.1: 3-bit Qwen3.5-27B loads, runs `Engine.generate("Hello",
  max_tokens=4)` cleanly, peak RAM ≤ 12 GB (vs ~15.3 GB at 4-bit
  per the v1.7.14 corrected probe; the ~25% bytes/param reduction
  from 4-bit to 3-bit applies proportionally).
- B.2: ΔPPL gate above passes; the 3-bit row ships in the catalog.

**Estimated impact:** Decode tok/s +20-30% on dense 27B / 31B due to
bandwidth savings; peak RAM -20-25%.

**Complexity:** Small. mlx-lm does the kernel work; silica just plumbs
the format through.

### Track C — Speculative Decoding (P-7 promoted to T1)

**Hypothesis:** Speculative decoding is the only lever that can
realistically push dense 27B above ~30 tok/s on M5 Pro, because it
amortizes a single weight read across N accepted tokens. PLAN.md
currently has P-7 as T2 and Q-002 open; this phase resolves Q-002 in
the affirmative — speculative is required for the user's tok/s goal,
not optional.

**Sub-units:** (these are P-7 sub-units, executed under the P-6
performance umbrella; user-confirmed at 2026-04-27 that **all five
paths should be tested** rather than picked in advance — see §11 Q-B
resolution. The five form a comparison stack from cheapest-engine-
work to highest-claimed-speedup.)

- **C.1 Draft-target speculative on Qwen3-class shapes.** Simplest
  workable version: `Qwen/Qwen3-0.6B` or `Qwen3.5-0.8B` as the draft,
  Qwen3.5-27B-4bit as the target. mlx-lm ships a draft+target
  speculative path (`mlx_lm.generate` with `--draft-model`); silica
  reproduces the pattern at the engine level so it composes with
  paged KV, batching, and prefix cache. Pinning is at greedy parity:
  speculative-on must produce the exact same token sequence as
  speculative-off under fixed seed. Cheapest path to land; serves as
  the floor every other variant must exceed.
- **C.2 Apple ReDrafter as draft.** `apple/ml-recurrent-drafter`
  ships an MLX-native RNN draft + dynamic tree attention with a 2.3×
  Apple Silicon claim. Mid-tier engine work; the open question is
  the KD training pass cost (Q-015 in PLAN.md §10 — 2026-04-27
  resolution: KD is in scope under D-020 because the user opted to
  test all five paths rather than gate C.2 on C.1's acceptance rate).
- **C.3 Qwen3.5 MTP head as draft.** PLAN.md §P-1 D-014 disables MTP
  at decode; the MTP weights are still in the checkpoint. Re-enable
  the MTP head as a draft source once a target/MTP shape parity
  check is in place. Cheapest training story (no KD pass — the head
  is already trained), but most architecture-specific (only works
  on Qwen3.5 family checkpoints that ship MTP weights).
- **C.4 DFlash (block-diffusion drafter).** arxiv 2602.06036, z-lab
  Feb 2026. A lightweight block-diffusion model generates a K-token
  draft block in **one** forward pass (vs C.1's autoregressive draft
  pass per draft token). Reports 6× over autoregressive and 2.5×
  over EAGLE-3 on Qwen3-class targets. MLX port already exists at
  `bstnxbt/dflash-mlx`; vLLM nightly has DFlash support, SGLang
  integration in progress. Engine integration: silica's
  `silica.speculative.DraftEngine` interface needs a "block draft"
  variant where `propose()` returns K tokens at once and `commit()`
  validates them under the target's lossless verification rule;
  this is a new shape on the I-5 interface, but the existing Noop
  / DraftTarget shape stays valid. References: `dflash-mlx` README,
  vLLM `speculators.dflash` docs.
- **C.5 DDTree (block-diffusion draft tree).** arxiv 2604.12989,
  liranringel Apr 2026. Tree extension of DFlash — instead of a
  single drafted trajectory per round, builds a draft tree from
  the block diffusion drafter's per-position distributions, picks
  branches with a best-first heap under a fixed node budget, then
  verifies the whole tree in one target forward via an
  ancestor-only attention mask. Reports **8.2×** over autoregressive
  on Qwen3 — directly relevant to the dense 27B target. MLX port
  exists at `humanrouter/ddtree-mlx` with custom Metal kernels for
  hybrid model support (Qwen3.5 hybrid DeltaNet is in scope) and
  a published 1.5× over autoregressive on real M-series silicon
  (the gap from 8.2× to 1.5× is real porting overhead worth
  measuring honestly). C.5 reuses C.4's DFlash drafter — only the
  verification path changes from single-trajectory to tree.

**Track C acceptance gates:**

- **Cross-path correctness invariant:** for every variant C.1 .. C.5
  under greedy decoding, speculative-on must produce token sequences
  byte-equivalent to speculative-off under fixed seed. The
  correctness gate is the same across variants — speculative
  decoding is a lossless rewrite of the autoregressive forward, so
  any difference is a bug.
- **C.1 throughput:** decode_tok_s ≥ 1.4 × the P-6.0 baseline on
  the "long-in / short-out" bench scenario at temperature 0 on
  dense 27B. Cherry-picking is forbidden — measured on the same
  scenario set as the baseline.
- **C.2 / C.3 throughput:** ≥ 1.8 × the P-6.0 baseline.
- **C.4 throughput:** ≥ 2.0 × the P-6.0 baseline (the published
  DFlash claim is ~6× on GPU; we discount aggressively for MLX
  porting overhead and Apple-Silicon kernel maturity. If C.4 lands
  ≥ 3.0× the baseline, that's a strong outcome that anchors C.5's
  gate as well).
- **C.5 throughput:** ≥ 2.5 × the P-6.0 baseline (DDTree's MLX
  port reports ~1.5× over autoregressive on real silicon; we
  target ≥ 2.5× because silica's engine fusion + paged KV stacks
  on top of the kernel-level claim).
- **Comparative reporting:** the bench harness emits a single
  "speculative comparison" report row with all five variants'
  measured speedup, cold/warm TTFT impact, peak RAM impact, and
  acceptance rate (where applicable). The phase-exit decision uses
  the highest-performing variant that lands cleanly; the others
  remain in the catalog for users with different workloads.

**Estimated impact (per variant, vs the P-6.0 dense 27B baseline):**

| Variant | Decode tok/s | TTFT impact | Peak RAM | Engine work |
| ------- | ------------ | ----------- | -------- | ----------- |
| C.1 draft-target | 1.4 – 1.8× | +draft prefill (~5-10%) | +draft weights (~0.6 GB) | Medium |
| C.2 ReDrafter | 1.8 – 2.3× | small | +RNN draft (~0.3 GB) | Large (KD) |
| C.3 MTP head | 1.4 – 1.7× | small | ~0 (head already in ckpt) | Medium |
| C.4 DFlash | 2.0 – 4.0× | +block diffusion forward | +block-diff drafter | Medium-Large |
| C.5 DDTree | 2.5 – 5.0× | same as C.4 | same as C.4 + tree state | Large |

**Complexity:**

- C.1 medium (mlx-lm pattern is well understood; silica integration
  is the work).
- C.2 large because of the KD training pass (per Q-015 resolution
  under D-020).
- C.3 medium and architecture-specific.
- C.4 medium-large — the block-diffusion drafter forward and
  lossless verification rule both new; `bstnxbt/dflash-mlx` provides
  a reference for the kernel side, but the engine integration into
  silica's I-5 interface is fresh.
- C.5 large — depends on C.4 plus the tree-verification path. The
  ancestor-only attention mask requires an MLX kernel that
  `humanrouter/ddtree-mlx` provides; integrating it under silica's
  paged KV is the new work.

**Risk:**

- Acceptance rate on dense 27B-4bit is empirically 50-70% in similar
  systems; for chat-style outputs (high-entropy, code, multilingual)
  it can drop to 30-40%. The acceptance gates above must be measured
  on the representative scenario, not a best case.
- Block-diffusion drafters (C.4 / C.5) are recent (≤ 3 months at
  P-6 entry); MLX ports are even newer. R-P6-8 in §7 captures the
  upstream-stability risk.
- DDTree's MLX port specifically claims hybrid model support; this
  is the lever that makes C.5 work on Qwen3.5 hybrid DeltaNet at
  all. Failing that claim degrades C.5 to "Qwen3.5 dense layers
  only," which is a partial result, not a phase failure.

### Track D — TTFT Levers

**Hypothesis:** TTFT on long prompts is dominated by prefill kernel
launch latency and per-layer dispatch. Two levers compound:

**Sub-units:**

- **D.1 Sarathi-style chunked prefill + decode merging.** Currently the
  scheduler admits a full prompt as one forward (Q-010 in PLAN.md was
  measurement-gated and never resolved past 0.6B; the v1.7.x finding
  was "small models don't trigger the threshold"). At 27B prompt scale
  with mixed-length cohorts, the long-prompt request blocks short-prompt
  TTFT. Fix: chunk long prefills into N-token slices, merge with active
  decodes in the same forward, surface per-row first-token offset.
  The slice-prefill regime already exists in
  `silica/scheduler/batcher.py:_slice_prefill_with_capture`
  (P-3-C5.5 α-MVP); D.1 lifts it from "α-MVP, slice regime only" to
  "default for prompts ≥ 512 tokens, with decode merging." This is
  pure scheduler work.
- **D.2 mlx-mfa long-prefill kernel.** Drop-in replacement for
  `mx.fast.scaled_dot_product_attention` on the prefill code path
  (decode stays on the existing MLX SDPA — mlx-mfa's win is at long
  context, not short). Reports up to 3.85× over plain MLX SDPA at
  1024-token / B=8 in published benchmarks. Maintainer's last
  release is paused until M5 Max bring-up, so we treat this as
  measurement-gated: only land if the kernel runs cleanly on M5 Pro
  with macOS 26.x and produces the claimed speedup on our long-prompt
  scenario. (See PLAN.md Q-009 for the variable-length SDPA story —
  D.2 does not solve Q-009, it just makes the prefill kernel faster
  within the bucketed-padding regime silica already uses.)

**Track D acceptance gates:**

- D.1: TTFT-under-concurrency on the existing
  `qwen3-0.6b-ttft-under-concurrency` scenario improves by ≥ 30%; on
  27B equivalent (new scenario), the long-prompt request does not
  block short-prompt requests by more than 2× their solo TTFT.
- D.2: long-prefill TTFT on dense 27B / 31B at 4096-token prompt
  improves by ≥ 1.8 × the P-6.0 baseline. If the kernel does not load
  cleanly on M5 Pro, D.2 is dropped without affecting the rest of
  the phase.

**Estimated impact:** -30-60% on long-prompt TTFT; -50-70% on
short-prompt TTFT under concurrency.

**Complexity:** D.1 medium (scheduler work, well-understood from
sarathi-serve / vLLM). D.2 small but risky (third-party kernel
dependency).

### Track E — Weight Streaming and SSD-Tiered Prefix Cache

**Hypothesis (this is the original P-6 scope, preserved as a track):**
Even after the bandwidth analysis, weight streaming pays in two narrow
cases — (a) MoE per-expert residency at scale, where the active set is
much smaller than total parameters; and (b) chat-style sessions with
long-lived prefixes that exceed the in-RAM prefix cache budget.

**Sub-units:**

- **E.1 MoE per-expert streaming residency.** Original P-6 deliverable;
  applies to Qwen3.5-35B-A3B (256 experts × 8 active) and gemma-4-26B-A4B
  (128 experts × 8 active). Active expert set is ~1.5 GB; full expert
  set is ~12 GB. If a request's expert hit pattern is well-localized,
  resident bytes can drop by 8× without throughput regression.
  Acceptance per PLAN.md P-6 Acceptance: `StreamingWeightProvider.resident_bytes()`
  ≤ `active_experts × expert_size + non_FFN + 20% headroom`; decode
  tok/s ≥ 60% of the resident baseline (PLAN.md preserves this gate).
- **E.2 SSD-tiered prefix cache (oMLX pattern).** When the in-RAM
  radix prefix cache exceeds budget, evict cold nodes to a memory-mapped
  SSD blob keyed on radix node hash. On hit, mmap-restore is faster
  than re-prefilling. oMLX reports TTFT 30-90 s → 1-3 s on coding-agent
  workloads where the same long prefix recurs across sessions. Fits
  silica's chat REPL persistence (`silica/chat/cli/persistence.py`)
  and the v1.6.x prefix-store residency surface.
- **E.3 Dense layer-streaming (deferred).** Original P-6 layer-granular
  streaming for dense 27B / 31B is **deferred to v0.2** because the
  bandwidth analysis (§1.2) shows dense weights already saturate the
  pipe; layer streaming cannot make the per-step weight read faster
  than the bandwidth ceiling. Streaming dense weights from SSD to
  unified memory adds latency without removing the bandwidth wall.
  Re-evaluate only if a future Apple chip changes the ratio between
  flash bandwidth and unified-memory bandwidth.

**Track E acceptance gates:**

- E.1: PLAN.md P-6 §Acceptance MoE clauses (already specified there);
  preserved verbatim.
- E.2: chat-session restart with a 4K-token prior context yields TTFT
  ≤ 0.3 × the cold-start TTFT.
- E.3: dropped for v0.1; deferral recorded as proposed Decision D-018
  in §10.

**Estimated impact:** RAM -50-80% on MoE workloads (E.1); TTFT -90% on
hot-prefix sessions (E.2). Decode tok/s neutral by design.

**Complexity:** E.1 large (touches `silica.weights` and the model
adapter MoE forward); E.2 medium (new module, mostly orthogonal to the
hot path).

---

## 4. Cross-Track Dependencies and Execution Order

The order below is the v1.7.14 D-021 commitment (foundation-first).
The ranked-by-ROI order in §3 is the *single-track* impact ranking;
the *phase-execution* order is below.

```text
P-6.0 measurement gate    LANDED v1.7.13 (plans/P6_0_BASELINE/REPORT.md)
        |
        v
P5.9 hardening pass       D-021 step 2 — no new features:
        |                   load-bearing crack repair (probe double-load,
        |                   Q-012 prefix-cache initial-cohort, recurrent
        |                   rollback for spec, P-5 quality regression
        |                   gate, D-009 audit, full re-run)
        v
P-6.0.5 measurement       D-021 step 3 — 27B B=2/B=4, MoE B=3/B=4,
        |                   27B 4K-context peak, warm-TTFT scenario,
        |                   target-verification microbench
        v
Decision Gate 1           D-021 step 4 — fix (1a)/(1b) framing
        |                   from data; record re-confirm or re-target
        v
Spec foundation + C.1     D-021 step 5 — DraftEngine wiring,
        |                   greedy parity, metadata schema,
        |                   recurrent + KV rollback test
        v
C.4 DFlash spike          D-021 step 6 — minimal closed loop;
        |                   gate >=1.8x silica-integrated speedup;
        |                   >=2.5x justifies pursuing (1b) stretch
        v
Track B 3-bit             D-021 step 7 — loader + PPL oracle first,
        |                   quality gate, then runtime
        v
C.5 / C.2 / C.3           D-021 step 8 — selection driven by C.4
        |                   outcome and (1b) status
        v
Track A sync collapse     D-021 step 9 — repositioned as
        |                   "general efficiency + MoE amplifier"
        v
Track D / E               D-021 step 10 — D.1 chunked prefill +
                            decode merging; D.2 mlx-mfa
                            measurement-gated; E.1 MoE per-expert
                            streaming; E.2 SSD prefix cache
```

The diagram is sequential where dependent and parallel where not:
P5.9 hardening blocks everything (steps 3-10 reference its
deliverables); P-6.0.5 blocks Decision Gate 1; the spec foundation
blocks every C.x; B and A run independently of each other after
the foundation. Within Track A, sub-units land in numerical order
(A.1+A.2 ship as one PR, A.3 follows separately because it has its
own correctness contract on snapshot equivalence).

## 4a. Phase exit when?

The phase exits when **either** dual-target acceptance gates land
(§1.3 / §6) **or** the user accepts a re-targeted exit at lower numbers
based on Step 0 evidence. Concretely, P-6 is "done" when:

- P-6.0 has a published baseline number for at least the dense 27B and
  MoE 35B-A3B targets, AND
- At least three of the five tracks (A/B/C/D/E) have shipped their
  acceptance gates, AND
- The phase-level dual targets in §1.3 are met **or** PLAN.md records a
  user-confirmed re-target via a new Decision Log entry.

The intent is to avoid the trap where one track stalls (e.g. C.2
ReDrafter KD training pass blocks indefinitely) and the rest of the
phase cannot close.

---

## 5. What is Explicitly Not in This Phase

To keep the scope honest, the following are excluded:

- **EAGLE-2 / EAGLE-3 port to MLX.** No public MLX port; porting the
  training-time-test scheme is months of work. Defer to v0.2.
- **DFlash-MLX.** Experimental block-diffusion drafting; revisit after
  C.1 / C.2 land and we have a stable speculative baseline.
- **Custom Metal kernels.** PLAN.md non-goal §3.2 forbids hand-rolling
  kernels from scratch. mlx-mfa (existing kernel) is allowed; writing
  a new one is not.
- **NVFP4 / MXFP4 weights.** MLX's FP4 support is incomplete (Issue
  #2962 — UE4M3 vs E4M3 dynamic-range mismatch); 3-bit group-quant is
  the realistic floor for v0.1.
- **Multi-process / tokenizer split / detokenizer split.** PLAN.md
  §5.3 single-process model is preserved.
- **Compressed-domain attention fast path** (PLAN.md D-003): preserved.
  Track B (3-bit) is weight quantization, not KV codec; KV stays at
  fp16 on the active path.
- **Out-of-core mmap weight streaming for dense models.** Bandwidth
  analysis above; deferred to v0.2.
- **Dropping the user's primary 100-tok/s target without consent.**
  The dual-target reframing is **proposed** in §1.3 and §11; the
  actual decision is the user's.

---

## 6. Phase-Level Acceptance (revised at v1.7.14 per D-021)

The phase exits successfully when items 1a, 3, 4, 5, 6 are all true.
Items 1b and 2 are stretch validators — passing them is celebrated,
missing them requires a Decision Log entry but does not fail the
phase.

1. **(1a) Dense engineering gate — Qwen3.5-27B-4bit ≥40 tok/s
   (must pass):** sustained warm-start decode_tok_s on
   `mlx-community/Qwen3.5-27B-4bit`, B=1, 128-token prompt,
   384-token generation (warm-start rule per §2) ≥ **40 tok/s**,
   with the highest-performing landed Track C variant enabled and
   the B.1 3-bit option allowed but not required. Reachable from the
   16.05 tok/s baseline via Track A engine fusion 1.10-1.15× ×
   Track B 3-bit 1.30× × Track C.1 draft-target 1.40-1.80× → 32-50
   tok/s realistic envelope. **This is the gate the phase actually
   exits on.**

   **(1b) Dense stretch gate — Qwen3.5-27B-4bit ≥60 tok/s
   (stretch):** same workload, same enabled stack, but pinning the
   user's original v0.1 framing. **Reaching this requires Track C.4
   DFlash and/or C.5 DDTree to land ≥2.5× silica-integrated speedup
   over the C.1 baseline.** Decision Gate 1 (D-021 step 4) measures
   the C.4 spike and decides whether (1b) is in pursuit; if the
   measured C.4 silica-integrated speedup is ≤1.8× the C.1 baseline,
   (1b) is retired to a Decisions Log entry naming the empirical
   floor. The phase still exits on (1a) regardless.
2. **(2a) MoE anchor — Qwen3.5-35B-A3B-4bit ≥100 tok/s aggregate
   (already cleared at v1.7.13 baseline).** Sustained warm-start
   aggregate decode_tok_s on `mlx-community/Qwen3.5-35B-A3B-4bit`
   at B=2 was **120.93 tok/s** at the v1.7.13 P-6.0 baseline before
   any track work — see `plans/P6_0_BASELINE/qwen3.5-moe-35b-a3b-warm-decode-b2.jsonl`.
   The anchor is preserved as evidence that the optimization stack
   runs cleanly on the hardest engine path silica supports.

   **(2b) MoE stretch — ≥150 tok/s aggregate at B=2 OR ≥100 tok/s
   per-row at B=2.** Either form clears it; both demonstrate
   silica's MoE-batched throughput is competitive with the
   GPU-class numbers vllm-mlx publishes (127.7 tok/s on M4 Max
   single-row). Reachable via Track A sync collapse — the
   compute-bound MoE regime has 40%+ bandwidth slack at B=2
   (59.1% utilization in baseline) which is exactly where the
   +30-80% Track A leverage applies. Status: **stretch**.
   Missing it records a Decision Log entry; passing it
   demonstrates the engine's optimization stack lands its
   estimated leverage on a real Apple Silicon target.
3. **TTFT under concurrency:** on the new
   `qwen3.5-27b-ttft-under-concurrency-warm` scenario (1 long
   2048-token request + 3 short 64-token requests), the short
   requests' TTFT ≤ 2× their solo TTFT.
4. **RAM headroom:** Qwen3.5-27B-4bit B=1 4K-context peak ≤ 36 GB
   (12 GB system headroom on 48 GB).
5. **MoE per-expert streaming verified** (PLAN.md P-6 acceptance
   preserved): `StreamingWeightProvider.resident_bytes()` clause
   passes on Qwen3.5-35B-A3B at the original 24 GB budget.
6. **No quality regression:** the P-5 acceptance row
   `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` continues
   to pass the (4-b) two-part aggregated gate after every track
   lands. (Sanity gate: nothing in this phase is allowed to regress
   P-5's PPL evidence.)

If item 2 fails because the MoE bandwidth math holds but the engine
overhead is larger than estimated, the phase does **not** retroactively
re-target — instead it lands at "≥80 tok/s on MoE" and cites the
measured engine-overhead floor in a Decisions Log entry. Honest exit
beats moved goalposts.

---

## 7. Risks

- **R-P6-1: 100 tok/s on MoE not achievable on M5 Pro.** Public
  benchmark is 127 tok/s on M4 Max (not M4 Pro), and silica adds engine
  overhead vllm-mlx does not. Mitigation: lower phase-level target to
  ≥80 tok/s if Step 0 baselines come in below 50 tok/s on MoE.
- **R-P6-2: 3-bit Qwen3.5-27B has unacceptable PPL drift.** Mitigation:
  ship 3-bit as opt-in flag, default to 4-bit; only count B's win
  toward the dense-target gate if B.2 PPL gate passes.
- **R-P6-3: Speculative acceptance rate on dense 27B chat outputs is
  below 50%.** Mitigation: C.1 advances the engine-side wiring; the
  acceptance-rate question becomes a draft-quality discussion, not a
  silica-engine discussion. C.2 ReDrafter is the escape hatch.
- **R-P6-4: mlx-mfa kernel does not load on M5 Pro / macOS 26.x.**
  Mitigation: D.2 is measurement-gated; drop without phase impact.
- **R-P6-5: Sync-barrier collapse changes correctness.** Mitigation:
  A.1+A.2 ship behind a feature flag for a week with greedy-parity
  pinning; flip default only after at least one full P-5 acceptance
  re-run is clean.
- **R-P6-6: Step 0 measurement on 27B/31B reveals decode_tok_s far
  below the bandwidth ceiling.** This is **information**, not a risk —
  if the gap exists, the phase has even more headroom. The risk is
  failing to actually run Step 0 and continuing to plan against a
  notional baseline.
- **R-P6-7: Phase scope sprawl.** Five tracks is already a lot, and
  Track C now has five sub-units of its own (C.1 .. C.5 per the Q-B
  resolution). Mitigation: §4a phase-exit rules ("at least three of
  five tracks") let the phase close cleanly without forcing every
  sub-unit to ship; within Track C, phase-exit picks the
  highest-performing variant that lands cleanly rather than requiring
  all five.
- **R-P6-8: Upstream stability of recently-published draft methods
  (C.4 DFlash, C.5 DDTree).** Both are 2026-published with active
  upstream development. The MLX ports (`bstnxbt/dflash-mlx`,
  `humanrouter/ddtree-mlx`) are independent community work, not
  Apple-blessed. Risk: API churn, kernel incompatibility with
  macOS 26.x, or a paper retraction. Mitigation: C.1 / C.2 / C.3
  are landing in parallel and provide a known-good baseline; if
  C.4 / C.5 cannot be brought up cleanly, Track C still meets its
  acceptance gates via the older variants. The phase does not
  depend on any single C.* sub-unit landing.

---

## 8. Open Questions Created by This Plan

- **Q-013 (already in PLAN.md): SLIDING + prefix cache.** D-3 closed
  miss-only; the chunked-prefill work in D.1 may pull this forward
  because chunked prefill changes the admission contract anyway. Land
  D.1 with the existing `prefix_cache=None` constraint preserved;
  reopen Q-013 as a follow-up.
- **Q-014 (new — proposed): is the dense-target gate at 60 tok/s the
  right number, or should it be tied to "≥ 0.7 × the bandwidth
  ceiling" so it scales with hardware?** Defer to phase landing.
- **Q-015 (new — proposed): does ReDrafter's KD training pass count
  as v0.1 scope or v0.2?** PLAN.md §3.2 non-goals don't address
  draft-model training. If C.1 alone meets the dense gate, Q-015
  doesn't fire.
- **Q-016 (new — proposed): is the MoE 100-tok/s target the user's
  preferred reframing, or should silica push for dense 27B + M5 Max
  as the v0.1 hardware baseline instead?** This is §11.

---

## 9. Cross-References to Existing Plan Documents

- PLAN.md §7 P-6 (this phase, original "Weight Streaming" scope):
  re-scope proposed via D-017 in §10.
- PLAN.md §7 P-7 (Speculative): priority promotion to T1 proposed via
  D-019 in §10; C.1 / C.2 / C.3 sub-units above are P-7 deliverables
  pulled forward into the P-6 umbrella.
- PLAN.md §10 Q-009 (paged-attention kernel availability): not
  resolved by this phase. The vllm-metal RFC #188 finding (sync-
  barrier cost) is a separate axis from Q-009 (paged-attention
  primitive existence); Track A addresses sync-barrier overhead
  without committing on a paged primitive.
- PLAN.md §10 Q-010 (chunked prefill promotion): resolved by Track
  D.1 (promoted from "measurement-gated deferral" to "default
  scheduler behavior on prompts ≥ 512 tokens").
- plans/P5_OPENING.md: the codec swap-neutrality contract from
  P-5-A.0.4 must continue to hold under all P-6 changes.
- plans/P3_C5_OPENING.md: slice-prefill regime is the substrate that
  D.1 chunked prefill builds on.
- plans/P4_5_C_KVCODEC_OPENING.md: prefix-store-pre-norm path
  (P-5-F.3) is preserved; E.2 SSD tier reads from the same store.

---

## 10. Proposed PLAN.md Edits (do not apply yet)

These edits land **after** the user confirms direction. They are listed
here as a diff-shaped proposal so a reviewer can read them in one place.

**Decisions Log appendix (PLAN.md §9):**

- **D-017 — P-6 phase re-scoped to "Performance Phase"**
  - Date: 2026-04-27.
  - Status: proposed; pending user confirmation.
  - Decision: PLAN.md §7 P-6 is re-scoped from "Weight Streaming"
    alone to "the performance phase," with five orthogonal tracks (A
    sync-barrier collapse, B 3-bit weights, C speculative decoding
    pulled in from P-7, D TTFT levers, E weight streaming + SSD
    prefix tier preserving the original P-6 deliverables).
  - Rationale: the user's TTFT / decode-tok/s / RAM goals cannot be
    addressed by weight streaming in isolation, and the bandwidth
    analysis (`plans/P6_OPENING.md` §1.2) shows speculative decoding
    is required to credibly reach the 100-tok/s class on dense 27B.
    Bundling all five tracks under a single phase preserves PLAN.md's
    sequential phase numbering (no new P-9) and keeps the
    deliverable surface auditable.
  - Consequences: PLAN.md §7 P-7 priority promotes from T2 to T1
    (separate Decision D-019); P-6 deliverables list expands from
    one bullet (`StreamingWeightProvider`) to five tracks; phase-
    level acceptance moves from one milestone (M-7) to the dual-
    target form in `plans/P6_OPENING.md` §6.

- **D-018 — Dense layer-streaming deferred to v0.2; original 24 GB
  budget gate dropped**
  - Date: 2026-04-27.
  - Status: proposed.
  - Decision: original P-6 layer-granular streaming for dense 27B / 31B
    is dropped from v0.1 scope. Layer streaming cannot reduce the
    per-step weight read below the unified-memory bandwidth ceiling;
    SSD-to-RAM streaming adds latency without lifting the wall. MoE
    per-expert streaming (E.1) is preserved.
  - **Explicit consequence — original P-6 acceptance gate retired.**
    PLAN.md §7 P-6 currently lists "Under an artificial memory budget
    (e.g. 24 GB), Qwen3.5-27B int4 does not OOM (dense path)" as the
    dense streaming gate. **This gate is dropped.** v0.1 no longer
    validates that residency-relief mechanisms work on dense models;
    it commits instead to "27B-4bit fits within 48 GB unified memory
    with measured headroom," anchored on the corrected probe number
    (~15.3 GB peak per v1.7.14 P5.9 step 2(a) — supersedes the
    inflated ~30.5 GB v1.6.1 figure caused by probe double-load) and
    re-confirmed under P-6.0 with 4K-context decode. The validation
    we lose: independent evidence that a dense-streaming fallback
    exists if a future checkpoint pushes peak above 48 GB.
    Mitigations: Track B (3-bit) gives a ~25% bytes/param reduction
    lever before any streaming would be needed; the corrected
    ~15.3 GB peak measurement carries the dense-fit assertion;
    a future v0.2 dense-streaming track can re-validate if the gap
    reappears.
  - Rationale: bandwidth analysis in `plans/P6_OPENING.md` §1.2.
  - Consequences: PLAN.md M-7 milestone narrows from "dense + MoE
    streaming" to "MoE streaming + 27B/31B fit-at-48GB without
    streaming"; the dense-fit assertion is now the responsibility of
    Track B (3-bit) and the v1.7.14-corrected 27B-4bit ~15.3 GB peak
    measurement (P5.9 step 2(a); supersedes the inflated 30.5 GB
    v1.6.1 figure) rather than residency relief. PLAN.md §7 P-6 acceptance bullet
    (1) is retired explicitly via this Decision rather than silently
    by the re-scope.

- **D-019 — P-7 priority promoted from T2 to T1**
  - Date: 2026-04-27.
  - Status: proposed.
  - Decision: PLAN.md §8.1 priority tiers — P-7 (Speculative) moves
    from T2 to T1, joining P-5 / P-6 in the "make big models fit and
    run fast at 48 GB" bucket. P-8 (Mini-SGLang) remains T2.
  - Rationale: bandwidth analysis shows speculative is required, not
    optional, to reach the dense-27B target. Q-002 ("should P-8
    priority float up?") is unaffected; this is about P-7 not P-8.
  - Consequences: P-7 sub-units land under P-6 phase umbrella as
    Track C in `plans/P6_OPENING.md`. P-7 phase block in PLAN.md
    §7 stays at status "planned" but its T1 placement enables the
    pull-forward.

**Open Questions (PLAN.md §10):**

- Q-002 closure entry: "Q-002 resolved — P-8 stays at T2; the
  promotion that mattered for v0.1 launch was P-7's, recorded as
  D-019."
- Q-010 closure entry: "Q-010 resolved — chunked prefill promoted
  to default for prompts ≥ 512 tokens via P-6 Track D.1 (see
  `plans/P6_OPENING.md` §3 Track D)."
- Q-014, Q-015, Q-016: append per §8 above.

**Phase block edits (PLAN.md §7 P-6):**

- Status: "planned" → "in-progress (P-6.0 measurement gate first;
  see plans/P6_OPENING.md)"
- Goal: rewrite from "weight streaming relieves residency pressure"
  to "engineer the platform to a dense primary of ≥60 tok/s on
  Qwen3.5-27B-4bit and a MoE stretch validator of ≥100 tok/s on
  Qwen3.5-35B-A3B-4bit, with TTFT-under-concurrency fairness on 48 GB
  M5 Pro"
- Scope, Strategy, Deliverables, Acceptance, Notes: rewritten to
  match `plans/P6_OPENING.md`. The original "weight streaming" body
  becomes Track E with E.3 (dense layer streaming) deferred per
  D-018.

**Changelog (PLAN.md §13):**

- v1.7.13 (2026-04-27): P-6 re-scoped to performance phase per
  D-017 / D-018 / D-019; cross-references `plans/P6_OPENING.md`.
  Q-002, Q-010 closed; Q-014 / Q-015 / Q-016 opened.

**docs/plans-index.md:**

- New entry: "P-6 Performance Phase opening — `plans/P6_OPENING.md`"
  added under the "Phase openings" subsection.

---

## 11. Decisions — User Confirmations Recorded 2026-04-27

All three decisions resolved. The original three options for each Q
are preserved below for traceability; the **Resolution** block under
each one records the answer that anchors the phase forward.

**Q-A — Reframe the 100-tok/s target?** The bandwidth math (§1.2 / §1.3)
shows 100 tok/s on dense Qwen3.5-27B-4bit on M5 Pro 48 GB is not
credibly reachable. Three ways to keep the user's intent:

1. **Dual-target reframing:** dense primary ≥60 tok/s on
   `mlx-community/Qwen3.5-27B-4bit`; MoE stretch ≥100 tok/s on
   `mlx-community/Qwen3.5-35B-A3B-4bit`. Phase exits on the dense
   primary; the MoE stretch validates the optimization stack. Stays
   within the v0.1 hardware-target envelope (M5 Pro 48 GB) recorded
   in PLAN.md §3.3.
2. **Hardware-target reset:** keep "100 tok/s on dense 27B" as the
   gate but **change the v0.1 hardware target from M5 Pro 48 GB to
   M5 Max 64+ GB**. M5 Max bandwidth is roughly 1.5 × M5 Pro, putting
   the autoregressive ceiling around 35 tok/s and the stacked-with-
   speculative target around 80-100 tok/s. This is **not just a
   number swap** — it amends PLAN.md §3.3 (Target Hardware), the
   D-006 platform-positioning Decision, and the README's "M5 Pro
   48 GB" framing. Choosing this option needs to be confirmed
   deliberately because it changes how silica-mlx is positioned to
   users and Apple's pricing tiers.
3. **Maintain the original target** as written and accept the phase
   will likely exit at a re-targeted gate via Decision Log entry
   under §6 last paragraph. Lowest-risk to plan integrity, but
   commits to a number the bandwidth math says we cannot deliver —
   the gap surfaces as a phase-exit Decision rather than a phase-
   entry one.

**Resolution (2026-04-27): Option 1 — dual-target reframing accepted.**
Dense primary gate ≥60 tok/s on Qwen3.5-27B-4bit; MoE stretch
validator ≥100 tok/s on Qwen3.5-35B-A3B-4bit; v0.1 hardware target
remains M5 Pro 48 GB. PLAN.md §10 Q-016 closes via this resolution;
no edit to §3.3 / D-006 / README needed. The two judgment calls below
become governing observations rather than open issues:

- **MoE 100 tok/s rests on the least-validated code path.** Per-expert
  `get_expert` is intentionally stubbed under `ResidentWeightProvider`
  (PLAN.md §7 P-3 Deliverables); `tests/test_p3_qwen3_5_moe_batched_parity.py`
  is the most recently landed acceptance evidence. Track E.1 has to
  realize the stubbed path before the MoE 100-tok/s figure can be
  measured against streaming-on. If Track E.1 lags, the stretch may
  fail for reasons unrelated to optimization quality. Mitigation
  built in: §6 acceptance gate (2) is the stretch — failing it does
  not fail the phase, but does require a Decision Log entry stating
  why.
- **The dual-target reframing puts the engine's hardest path in the
  stretch slot, not the primary slot.** Deliberate choice to honor
  the user's explicit "Qwen3.5-27B" framing. The alternative (MoE
  primary, dense secondary) was considered and rejected because the
  user's brief named the dense target.

**Q-B — Speculative draft path priority.** Original framing: pick the
default attempt order, with the rest as fallbacks. C.1 (draft-target)
is cheapest; C.2 (Apple ReDrafter) is highest-throughput but adds a KD
training cost; C.3 (Qwen3.5 MTP head) is medium-cost.

**Resolution (2026-04-27): test all five paths, including two new
2026-published variants the user named explicitly.** The user's
direction was "speculative 都要测试，还有最近出来的 DFlash 和 DDTree" —
keep C.1 / C.2 / C.3 as separate measurable variants, **add C.4
DFlash** (block-diffusion drafter, arxiv 2602.06036, MLX port at
`bstnxbt/dflash-mlx`) and **C.5 DDTree** (DFlash + draft tree, arxiv
2604.12989, MLX port at `humanrouter/ddtree-mlx` with hybrid model
support) to the same Track C. Track C is now five sub-units, each
independently measurable; phase-exit picks the highest-performing
variant that lands cleanly without forcing the others to land. See
§3 Track C for the full sub-unit breakdown, the comparison-report
deliverable, and per-variant acceptance gates. Q-015 in PLAN.md §10
(ReDrafter KD as v0.1 scope) closes via this resolution: KD is in
scope because the user opted to measure C.2 alongside the others
rather than gate it on C.1's acceptance rate. Recorded as proposed
Decision **D-020** in §10 (PLAN.md edit batch).

**Q-C — Step 0 first, then plan?** Original framing: should the user
review P-6.0 baselines before Tracks A-E begin? Either order works —
the plan is structured so P-6.0 always runs first regardless.

**Resolution (2026-04-27): yes — pause for user review and target
re-confirmation between P-6.0 measurement landing and Tracks A-E
starting.** The user's direction was "先把 P-6.0 跑出真 baseline 再
回头确认目标." Concretely, this means:

- P-6.0.1 ✅ (survey, completed)
- P-6.0.2 ✅ (WARM_DECODE oracle, completed 2026-04-27)
- P-6.0.3 (scenario registration) and P-6.0.4 (tests + docs sync)
  proceed under auto mode without further user pause.
- After P-6.0.4 lands, the next step is the user (or anyone with
  M5 Pro 48 GB hardware + the gated checkpoints) running the new
  scenarios via `python -m scripts.bench --scenario
  qwen3.5-27b-warm-decode-b1` etc. and producing real numbers.
- A short follow-up document (`plans/P6_0_BASELINE.md` or appended
  to this opening) records the measured numbers and re-confirms or
  re-targets the dense-60 / MoE-100 gates before any Track A-E
  PR opens.
- If the measured baseline differs materially from the §1.3
  arithmetic (e.g. dense 27B comes in at 12 tok/s instead of ~20,
  or MoE 35B-A3B comes in at 30 tok/s instead of ~50), the phase
  pauses for an explicit re-target Decision Log entry before any
  Track work starts.

This resolution adds no proposed PLAN.md Decision entry — it is a
process commitment recorded in this opening doc, not a plan-level
amendment.

---

Once Tracks A-E begin, this section moves into a "Decisions Log"
sub-section under each track to record per-track close points.
Until P-6.0 landing the section stays in Q&A form for traceability.
