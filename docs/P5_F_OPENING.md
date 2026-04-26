# P-5-F — Pre-RoPE K/V store architecture: opening

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Version      | v1.0.0 (draft)                                                     |
| Last updated | 2026-04-26                                                         |
| Status       | opening doc — implementation not yet started                       |
| Maintainer   | Xin Zhou                                                           |
| Parent unit  | P-5 (KV-codec) post-P-5 follow-up                                  |
| Parent docs  | `docs/PLAN.md` §7 P-5 Notes ("Production `prefix_store_post_rope`  |
|              | prefix-cache quality cost — post-P-5 required follow-up");        |
|              | `docs/P5_D2_INVESTIGATION/README.md` §Root cause                   |
| Scope        | Move silica's production K/V codec injection from post-RoPE        |
|              | (current) to pre-RoPE (vqbench-aligned), so the                    |
|              | `prefix_store_post_rope` quality gap closes on serving paths       |
|              | that today route through `ContinuousBatcher` +                     |
|              | `RadixPrefixCache` + a non-identity codec. Exit criterion:         |
|              | production prefix-cache ΔPPL on the Qwen3-0.6B WikiText-2          |
|              | reference scenario (BlockTQ b64 b4) lands inside the D.2a          |
|              | vqbench-aligned oracle's `+0.51 ± 0.35 PPL` envelope, replacing    |
|              | the current `~+20 PPL` post-RoPE production-path cost.             |

## 1. Problem statement

### 1.1 What is broken on the current tree (v1.7.5 + post-C5.5)

silica's production K/V codec sees **post-RoPE** K/V tensors. The
codec hooks live in
`silica/kvcache/store.py:394-436` (`SyntheticPrefixBlockStore.register_detached`
/ `fetch_detached`), which receive K/V slices extracted from
`silica/scheduler/batcher.py:640-682` (`_extract_and_insert_prefix`)
out of the live `BatchKVCache`. mlx-lm's attention applies RoPE
**before** writing to that cache (e.g. `qwen3_next.py:147` —
`keys = self.rope(keys, offset=cache.offset)` then
`cache.update_and_fetch(keys, values)`), so by the time silica
reaches them they carry per-token rotation.

The vqbench investigation (P-5-D.2 / D.2a, `docs/P5_D2_INVESTIGATION/`)
nailed the root cause: at the same Frobenius reconstruction error,
**post-RoPE noise injection pays an extra chunk-boundary cost** that
the pre-RoPE injection does not. Concretely, on the
`qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` scenario:

| Path | ΔPPL (3-seed mean ± std) | Codec config |
| --- | --- | --- |
| `prefix_store_post_rope` (production today) | ~+5 to +20 PPL (chunk-dependent) | BlockTQ b64 b4 |
| `vqbench_aligned` oracle (D.2a) | +0.511 ± 0.354 PPL | BlockTQ b64 b4 |
| vqbench subprocess reference | +0.661 ± 0.347 PPL | BlockTQ b64 b4 |

D.2a closed the **algorithmic** gap between silica MLX and vqbench
NumPy when both inject in the same space (`+0.51` vs `+0.66` is
within one unit of seed-level noise per (4-b)). What it did **not**
close — and what P-5-F closes — is the production-path gap between
silica's `prefix_store_post_rope` arm and that aligned oracle.

### 1.2 ROI by model size — P-5-F's actual deployment value

The post-RoPE quality cost is sharply size-dependent. Cross-arch
PPL ablation on WikiText-2 (chunk_size=256, max_tokens=512,
block_size=16; `project_kv_codec_ppl_findings.md`):

| Model | Architecture | fp16 PPL | post-RoPE ΔPPL (BlockTQ b64 b4) |
| --- | --- | --- | --- |
| Qwen3-0.6B | pure-attention | 19.673 | **+20.83** |
| Qwen3-4B | pure-attention | 12.916 | +0.162 |
| Qwen3.5-0.8B | hybrid-DeltaNet | 15.586 | +0.008 |
| Qwen3.5-4B | hybrid-DeltaNet | 8.856 | -0.014 |

**P-5-F's primary value is on small (< 1B) targets and on
aggressive codec configurations (lower bit / smaller block size)
that have not yet been characterised on 4B+ targets.** The 4B+
post-RoPE numbers are essentially lossless at BlockTQ b64 b4, so
P-5-F's marginal benefit on the headline production targets
(Qwen3.5-27B / Gemma4-31B / 35B-A3B / 26B-A4B) at this codec
config is small. F.0 (§6.1) closes that uncertainty before
committing the full architecture change.

### 1.3 What this unit unblocks

- **`prefix_store_post_rope` quality close.** PLAN §7 P-5 Notes
  flagged it as a post-P-5 required follow-up.
- **Aggressive codec ROI on production targets.** With a pre-RoPE
  store, silica can deploy ext_rabitq_b2 / b1 codec configurations
  on Qwen3-0.6B class models without paying the post-RoPE
  chunk-boundary cost. F.0 measures this.
- **(b-static) Qwen3.5-4B vs `vqbench/REPORT.md` PPL baseline.**
  PLAN v1.7.2 deferred (b-static) as post-P-5 follow-up. P-3-C5
  lifted the recurrent-prefix-cache guard so Qwen3.5-4B can run
  through `RadixPrefixCache`; P-5-F's pre-RoPE store gives that
  path the same algorithmic injection space as vqbench's static
  PPL number.
- **Algorithmic convergence between bench oracle and production.**
  After P-5-F lands, `teacher_forced_chunked_nll_with_codec`
  (production-store oracle) and
  `teacher_forced_chunked_nll_vqbench_aligned` (D.2a research
  oracle) collapse onto the same algorithm — one persistent code
  path. The D.2a oracle becomes a redundant evidence strand and
  can be deprecated post-F.

## 2. Reference solution — vqbench `_QuantizedProj` + silica D.2a

### 2.1 vqbench's pre-RoPE injection (research harness)

`vqbench/vqbench/validation/monkey_patch.py::_QuantizedProj` is a
research-grade harness:

- **Where**: replaces `attn.k_proj` and (when `wrap_v=True`) `attn.v_proj`
  in-place, before mlx-lm/HF applies RoPE.
- **What**: on every forward, runs the original projection,
  reshapes to per-head `(B, n_kv_heads, L, head_dim)`, applies
  `quant_dequant_tensor` with **per-head Haar rotation** (`seed=h`
  per head per layer), reshapes back to flat. Output goes through
  RoPE downstream as if the projection had been noisy.
- **Why not production**: re-quantizes the **entire sequence** on
  every forward (no persistence); only works in teacher-forced /
  prefix-prefill mode; doesn't serialize codec state; doesn't
  integrate with continuous batching or prefix-cache reuse.

### 2.2 silica's D.2a vqbench-aligned oracle

`silica/bench/ppl_oracle.py:621-863::teacher_forced_chunked_nll_vqbench_aligned`
mirrors the same monkey-patch idea via `_WrappedProj`:

- Patches `attn.k_proj` / `attn.v_proj` per-layer at oracle entry,
  restores in `try/finally`.
- Uses **shared rotation** across all heads and layers (one seed
  per oracle call) — silica's deviation from vqbench, justified
  per (4-b) by the ~0.977× Frobenius ratio on real Qwen3.5 data
  (per-head vs shared contributes ~0.02× Frobenius, well inside
  the (4-b) gate envelope).
- Caches a codec instance per observed length L (`_get_codec(L)`)
  but the codec itself is **stateless across forwards** — the
  harness re-encodes the entire sequence on every call.

The algorithmic shape (where the codec sees K/V) is exactly what
P-5-F needs to bring to the production store. The persistence /
streaming gap is what makes it an oracle and not a store.

### 2.3 Algorithmic invariant

By construction, vqbench, the D.2a oracle, and any P-5-F
production store all inject the **same noise distribution
ε ∼ codec(K_pre)** in pre-RoPE projection space. Downstream
RoPE is a per-position **orthogonal rotation** that does not
amplify or attenuate the noise; the chunk-boundary cost the
post-RoPE store pays is from RoPE-coupled noise interacting
with subsequent positions' rotations during prefix-cache refills.
Removing that coupling — by injecting before RoPE rather than
after — removes the chunk-boundary cost.

## 3. Architectural options

### 3.1 Option A — Inverse-RoPE round-trip at the store seam (recommended)

**Where the change lives**: `silica/kvcache/store.py` (encode /
decode hooks) + a small adapter Protocol extension exposing the
RoPE handle.

**Algorithm**:

```
encode (post-RoPE K from BatchKVCache → store payload):
    K_pre = inverse_rope(K_post, offset=block_start)  # only on K
    payload = codec.encode_tensor(K_pre)
    # V is identical: V never has RoPE applied, so it goes
    # straight through without inverse — see §4 for proof

decode (store payload → seeded BatchKVCache, post-RoPE):
    K_pre_recon = codec.decode_tensor(payload)
    K_post_recon = forward_rope(K_pre_recon, offset=block_start)
```

**Why A is now clean (§5.1 numerical readiness):** `mx.fast.rope`
exposes inverse rotation directly via `freqs=-freqs` (or equivalently
`scale=-1.0`). On a representative `(B=1, H_kv=8, T=16, D=128)`
block with realistic projection-scale K (std ~1.0), the round-trip
Frobenius reconstruction ratio is **9.60e-8 in fp32** (fp noise
floor) and **1.36e-4 in fp16 native** — same Frobenius-norm units
as the codec's own ~2.0e-2 reconstruction ratio. The fp16
round-trip noise is **~150× smaller** than the codec's own noise
budget. fp16-via-fp32-cast brings the round-trip to fp32 levels at
near-zero cost (one cast each side) if a stricter path is wanted.

**Pros**:
- Blast radius confined to `kvcache/store.py` + adapter Protocol.
- Does not touch mlx-lm's `Attention.__call__` or `BatchKVCache`
  contract — live cache stays post-RoPE as mlx-lm expects.
- `prefix_store_post_rope` legacy path can be retained as a
  bench-only opt-in for regression comparison.
- Encode trigger time is unchanged from current contract
  (reclaim-time extract).

**Cons**:
- Adds one inverse + one forward RoPE per cached block at
  encode + decode time. Cost: small (block_size × n_kv_heads ×
  head_dim per block × number of attention layers) and it's a
  cold-path operation, not in the per-token decode loop.
- fp16 round-trip introduces ~2e-3 absolute noise on K_pre
  before codec; this is below codec-noise budget per §5.1 but
  must be empirically confirmed via F.0 PPL measurement.

### 3.2 Option B — Cache holds pre-RoPE K throughout (not v0.1)

**Where the change lives**: every adapter family's `Attention.__call__`
(or a silica-owned subclass) + cache layout semantics.

**Algorithm**: `BatchKVCache` stores pre-RoPE K. Every forward
reads cache, applies RoPE per-position to the **entire accumulated
K** before the attention dot product. silica's codec sees pre-RoPE
K cleanly; no inverse-RoPE math at any seam.

**Why not v0.1**:

- mlx-lm's `Attention.__call__` is built on the assumption that
  `cache.keys` is post-RoPE; reading cache and immediately doing
  `Q · K^T` works because rotation is already baked in. Changing
  the cache contract requires either (i) applying RoPE to the
  full accumulated K on every forward step (O(L) RoPE per step
  × decode steps — significant runtime cost), or (ii) forking
  mlx-lm's attention modules per family.
- (ii) breaks D-005 ("quantization rides on mlx-lm's existing
  4-bit / 8-bit path") and the MLX-native hard constraint
  (`feedback_mlx_native.md`) — silica is not in the business of
  re-implementing model forwards. Forking attention per family
  (Qwen3 / Qwen3.5 hybrid GLOBAL layers / Gemma4 SLIDING + GLOBAL /
  Qwen3.5-MoE / Gemma4-MoE) multiplies the maintenance surface.

B is algorithmically the most elegant — codec sees pre-RoPE
unconditionally, no inverse-RoPE math anywhere. The elegance cost
is owning attention forwards across five families. **Not a v0.1
candidate.**

### 3.3 Option C — Persistent projection-wrapper (fallback)

**Where the change lives**: a silica-owned `_QuantizedProjStore`
wrapping `attn.k_proj` / `attn.v_proj` per layer, backed by a
persistent prefix-cache.

**Algorithm**: on first encounter of a (block_idx, layer_idx)
tuple during prefill, encode the projection output and store;
on cache hit, fetch and decode back into the projection output
slot before RoPE. RadixPrefixCache provides persistence; the
wrapper provides the pre-RoPE injection point.

**Why fallback, not default**: this is the D.2a oracle's monkey-patch
pattern made persistent. It still wraps `k_proj` / `v_proj` at
runtime, which the oracle's own non-production designation
explicitly cited as too invasive for the production hot path.
Option C inherits that design wart and adds the new burden of
splitting the (B, L, H·D) projection output by token-block
boundaries to feed the prefix-cache. Encode-trigger-time
restructures (now at projection time during prefill, not at
reclaim time) — touches the slice-prefill path C5.5 just stabilised.

C remains a viable fallback if A's empirical measurement (F.0)
shows fp16 inverse-RoPE round-trip pushes ΔPPL outside the
D.2a `+0.51 ± 0.35 PPL` envelope and fp32-cast doesn't recover
it. F.0 result determines whether we ship A or escalate to C.

### 3.4 Decision

**A is the recommended default.** F.0 validation gates the
escalation to C (no escalation if A's empirical numbers land in
the D.2a envelope).

## 4. Per-family RoPE inventory

mlx-lm reaches RoPE through one common entry point —
`mlx_lm/models/rope_utils.py::initialize_rope` — which dispatches
to one of `nn.RoPE` (default / linear), `Llama3RoPE`, `YarnRoPE`,
`SuScaledRoPE`, `ProportionalRoPE`. **All five end at
`mx.fast.rope(..., freqs=freqs)` (or `base=base` on `nn.RoPE`).**
silica's `apply_rope_inverse_to_k` can therefore be a single
function in adapter code, dispatching on `getattr(rope_instance,
"_freqs", None)` and falling back to the `base` reconstruction
for the `nn.RoPE` path.

| Family | RoPE init site | RoPE class | RoPE on Q? | RoPE on K? | RoPE on V? |
| --- | --- | --- | --- | --- | --- |
| Qwen3 (0.6B / 4B / 27B) | `qwen3.py:51` | `initialize_rope(...)` → varies by config | yes | yes | **no** |
| Qwen3.5 GLOBAL layers | `qwen3_next.py:113` (used via `qwen3_5.py:18`) | `initialize_rope(...)` → varies | yes | yes | **no** |
| Qwen3.5 DeltaNet layers | n/a — recurrent, no K/V | n/a | n/a | n/a | n/a |
| Gemma4 SLIDING + GLOBAL | `gemma4_text.py:218` | `initialize_rope(...)` → varies | yes | yes | **no** |
| Qwen3.5-MoE | inherits Qwen3.5 via parent class | same | same | same | same |
| Gemma4-MoE | inherits Gemma4 via parent class | same | same | same | same |

**Critical simplification**: V is **not** rotated in any silica-supported
family. The pre-RoPE store change therefore applies inverse-RoPE
**only to K**; V is byte-identical between pre-RoPE and post-RoPE
spaces and goes straight through the codec round-trip. This halves
the per-block RoPE work (only one of K/V needs RoPE math) and
collapses the adapter Protocol surface to a single
`apply_rope_inverse_to_k(...)` method per family (or a base
implementation that all families inherit).

**Hybrid-family compatibility**: Qwen3.5 DeltaNet layers don't
participate in the K/V codec path at all (recurrent state, not K/V).
Gemma4 SLIDING and GLOBAL layers both use `BatchKVCache`-shaped
storage and both use RoPE — pre-RoPE store applies uniformly.
The existing `attn_layer_indices` filter in
`silica/scheduler/batcher.py:596` already skips non-attention
layers and is reused unchanged.

## 5. Numerical readiness

### 5.1 Inverse-RoPE round-trip error budget

Empirical measurement on a representative block
`(B=1, n_kv_heads=8, T=16, head_dim=128)` with realistic
projection-scale K (zero-mean Gaussian, std=1.0), `freqs`
reconstructed at `base=10000.0`. **All numbers in
Frobenius-reconstruction-ratio units (`||error||_F /
||original||_F`)** so the round-trip is comparable apples-to-apples
with the codec's own Frobenius reconstruction ratio.

| Path | dtype | Frobenius ratio (round-trip vs original) |
| --- | --- | --- |
| `mx.fast.rope` `freqs=+freqs` then `freqs=-freqs` | fp32 | **9.60e-08** (fp noise floor) |
| same | fp16 native | **1.36e-04** |
| same, with fp16 → fp32 → fast.rope → fp32 → fp16 cast | fp16 via fp32 | **9.60e-08** (fp32 path) |

Reference for what "OK" looks like:

- **BlockTQ b64 b4 codec Frobenius reconstruction ratio**: ~2.0e-2
  (per `docs/P5_D2_INVESTIGATION/` probe data). Same units as the
  table above.
- The fp16 native round-trip ratio of 1.36e-4 is **~150× smaller**
  than the codec's own noise budget. The round-trip should not
  measurably degrade codec PPL.
- fp16-via-fp32 path is available as a stricter route at
  effectively zero cost (one cast each side) if the fp16 native
  number turns out to be empirically problematic.

Per-element max-abs-diff numbers (for completeness — not the unit
the comparison is done in): fp32 4.77e-7, fp16 native 1.95e-3,
fp16 via fp32 0.00. Switching to Frobenius for the budget
comparison avoids the unit mismatch that an earlier draft of this
doc made (max abs vs Frobenius-norm-ratio).

F.0 measures the actual ΔPPL impact of the round-trip on the
Qwen3-0.6B + BlockTQ b64 b4 reference scenario. Pass criterion is
in §6.1 F.0 (b) (a two-part OR gate, not a single envelope).

### 5.2 Inverse via `freqs` negation, not `offset` negation

**`mx.fast.rope(y, ..., offset=-p)` is not a valid inverse.**
The fast kernel treats `offset` as an absolute token index, not
as a signed rotation parameter. A negative offset gives max diff
~6.65 on the same fp32 block — i.e. unrelated output. Inverse
rotation is reached by `freqs=-freqs` (or `scale=-1.0`); both
work because `R(-θ) = R(θ)^T` and the kernel computes
`cos(scale · offset · freqs[i])` / `sin(scale · offset · freqs[i])`
per dimension pair.

This is documented in `silica/kvcache/store.py` next to the
inverse call so future readers don't re-discover the wrong path.

## 6. Sub-unit decomposition

Each sub-unit is one commit (or a small commit pair where noted).
The user-stated cadence (`feedback_incremental_plan_execution.md`)
is one sub-unit at a time, pause for confirmation between commits.

### 6.1 F.0 — Data validation + minimum prototype

**Two independent measurements, one commit pair (probe + recording).**

**(a) Aggressive codec on 4B-class targets — P-5-F priority data.**
Add bench scenario rows for `qwen3-4b-wikitext-ppl-ext-rabitq-b1`
and `qwen3-4b-wikitext-ppl-ext-rabitq-b2` (and same on Qwen3.5-4B
hybrid if `prefix_store_post_rope` accepts hybrid post-C5). Run
through the existing `prefix_store_post_rope` arm; record ΔPPL.

  - **Pass criterion**: this is data gathering, not a gate. The
    ΔPPL determines whether P-5-F is critical-path on production
    targets (4B+ shows degradation under aggressive codec) or
    primarily a small-model concern (4B+ remains lossless even
    under b1).
  - **If 4B+ stays lossless under b1 / b2**: P-5-F's deployment
    priority drops to "small models + bench-quality close" and
    the architecture change still ships, but the urgency framing
    in PLAN updates downward.
  - **If 4B+ degrades**: confirms P-5-F is critical-path on
    production targets at aggressive codec.

**(b) Qwen3-0.6B + BlockTQ b64 b4 + Option A minimum prototype.**
Land an experimental-flag-gated path in `SyntheticPrefixBlockStore`
that does post-RoPE → inverse-RoPE → codec → forward-RoPE round-trip.
Feature-flagged off by default. Run
`qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` with the flag on, 3
seeds (the same `{42, 43, 44}` the (4-b) gate uses).

  - **Pass criterion (two-part OR gate)**: F.0 (b) passes if
    **either** of the following holds:
    - **(i) Tight gate**: 3-seed ΔPPL inside the D.2a oracle
      envelope `+0.51 ± 0.35 PPL`. This is the "algorithmic
      parity with D.2a" claim; passes when the production-store
      pre-RoPE injection produces noise statistically
      indistinguishable from the D.2a oracle's chunk-grained
      re-encoding pre-RoPE injection.
    - **(ii) Structural gate**: 3-seed mean ΔPPL ≤ 1.5 PPL,
      i.e. **at least ~13× reduction** from the
      `prefix_store_post_rope` arm's ~+20 PPL on the same
      scenario. Threshold rationale: D.2a's empirical floor on
      the same codec is +0.51 PPL, so a true pre-RoPE injection
      should land near +0.5–0.7; allowing up to 1.5 keeps ~3×
      slack against the floor while staying well under the
      "≥ 10× structural reduction" goal. Anything in 1.5–2.0
      would suggest a residual D1 / D2 issue worth diagnosing,
      not papering over.

    The OR is required because A and the D.2a oracle differ on
    two axes (block-grained persistent encoding vs chunk-grained
    re-encoding), so noise distributions are similar but not
    identical. The structural gate (ii) is the hard P-5-F
    closure target; (i) is the strong-form claim that A is
    genuinely D.2a-equivalent.

  - **If fails (neither (i) nor (ii) passes)**: diagnose the
    failure mode and route to the matching fallback. Three
    diagnostic categories, each with a distinct fallback:
    - **(D1) fp16 round-trip noise dominates**: ΔPPL with the
      pre-RoPE flag on is *higher than* fp16 native round-trip
      noise alone could explain (per §5.1, ~1/150× of codec
      noise). Diagnostic: re-run F.0 (b) with the
      fp16-via-fp32-cast round-trip path (0 added noise per
      §5.1). If this brings ΔPPL into envelope, the production
      path adopts fp32-cast.
    - **(D2) Codec calibration drift**: persistent block-grained
      encoding with one calibration round per block produces
      different codebooks than D.2a's chunk-grained
      re-calibration. Diagnostic: temporarily disable codec
      caching (re-calibrate per encode call) and re-run F.0 (b).
      If this brings ΔPPL into envelope, the production path
      adopts the same calibration policy. Codec-level fix, no
      architecture escalation.
    - **(D3) Encode trigger time matters**: encoding at reclaim
      (after live cache has accumulated full sequence) loses
      information that encoding at projection time (§3.3
      Option C) preserves. Diagnostic: only after (D1) and (D2)
      are ruled out — D3 is the "structural defect of Option A"
      diagnosis. **This is the only failure mode that justifies
      escalating to Option C.** Re-open the plan; F.1 onwards
      restart against C.

    The categorisation matters because Option C only solves D3.
    Escalating to C without diagnosis would burn implementation
    time on the wrong fallback if the failure is actually D1 or
    D2.

**Commit pair**:
1. F.0a: bench scenario additions for 4B aggressive codec; record
   measurements in `project_kv_codec_ppl_findings.md` (memory).
2. F.0b: experimental-flag-gated A prototype in store; run
   ΔPPL gate; record result in this opening doc and in PLAN
   §7 P-5 Notes empirical findings.

### 6.2 F.1 — Adapter Protocol: `apply_rope_inverse_to_k`

Add a new method to the adapter Protocol surface:

```python
def apply_rope_inverse_to_k(
    self,
    layer_idx: int,
    k: mx.array,
    *,
    offset: int | mx.array,
    inverse: bool = False,
) -> mx.array:
    """Apply RoPE (or its inverse) to K at the given absolute
    position offset. Reaches mlx-lm's `attn.rope` instance for
    this layer; uses `freqs=-freqs` for inverse. Behaviour for
    non-attention layers (Qwen3.5 DeltaNet) is undefined and
    must not be reached — caller filters via attn_layer_indices.
    """
```

Implement once in a base helper that introspects
`getattr(rope_instance, "_freqs", None)` (returns the `_freqs`
attribute for `Llama3RoPE` / `YarnRoPE` / `SuScaledRoPE` /
`ProportionalRoPE`) and falls back to `base ** (mx.arange(0, dims, 2) / dims)`
for `nn.RoPE`. Each adapter family's implementation is a 2-line
delegation to the helper.

**Actual in-use RoPE class subset** (from
`config.json::text_config.rope_scaling` on cached repos, 2026-04-25):

| Family | rope_scaling type | RoPE class via `initialize_rope` |
| --- | --- | --- |
| Qwen3-0.6B | `None`, theta=1e6 | `nn.RoPE` |
| Qwen3.5-0.8B / 4B / 35B-A3B | `'default'` (with mrope_section / partial_rotary, but mlx-lm 'default' branch returns `nn.RoPE`) | `nn.RoPE` |
| Gemma4 sliding_attention layers | `'default'` | `nn.RoPE` |
| Gemma4 full_attention layers | `'proportional'` | `ProportionalRoPE` |
| Qwen3.5-MoE / Gemma4-MoE | inherit dense parents | same |

**Two RoPE classes are actually in use across silica's targets:
`nn.RoPE` and `ProportionalRoPE`.** `Llama3RoPE` / `YarnRoPE` /
`SuScaledRoPE` would also work through the same `_freqs`
introspection path but no current silica adapter routes to them —
they are forward-compatibility coverage.

Unit tests against scripted adapters:
- **In-use coverage (load-bearing)**: forward-then-inverse identity
  on `nn.RoPE` (Frobenius ratio ≤ 1.5e-4 fp16, ≤ 1e-7 fp32) and
  `ProportionalRoPE` (same thresholds).
- **Forward-compatibility coverage (regression guard)**: the
  helper's `_freqs` introspection path covers all four scaling
  variants uniformly (`Llama3RoPE` / `YarnRoPE` / `SuScaledRoPE`
  / `ProportionalRoPE` all expose `_freqs`), so test coverage
  picks two representative classes rather than enumerating all
  four. **Pick `Llama3RoPE` and `YarnRoPE`** as the regression
  pair (covers both freq-scaling and yarn-scaling shapes);
  `SuScaledRoPE` is structurally similar to `Llama3RoPE` and is
  intentionally not separately tested per the "two representative
  classes" coverage policy. Documented in the F.1 test module
  docstring.
- inverse rotation matches a pure-numpy / fp32 reference for
  `nn.RoPE` and `ProportionalRoPE` (the in-use pair).
- non-attention layer indices raise `NotImplementedError` (matches
  the docstring contract).

**Commit**: one.

### 6.3 F.2 — Store: pre-RoPE round-trip behind a flag

Wire `SyntheticPrefixBlockStore.register_detached` /
`fetch_detached` to call the adapter's `apply_rope_inverse_to_k`
when constructed with a `pre_rope: bool = True` flag. Default
stays `False` (current post-RoPE behaviour) until F.3 flips it.

Position metadata: each `register_detached` call already carries
`(block_id, per_layer_kv)`; the absolute position is
`block_id * block_size`. No new metadata plumbing needed.
Numerical correctness via F.0 prototype — same algorithm, now
behind a typed Protocol method instead of an experimental flag.

**Commit**: one. Small (~150 lines) and entirely additive — the
default-off flag means existing behaviour is byte-identical.

### 6.4 F.3 — Production default flip + bench oracle convergence

Flip the default `pre_rope` flag in two places:

1. `SyntheticPrefixBlockStore` default → `True`.
2. `silica/bench/ppl_oracle.py::teacher_forced_chunked_nll_with_codec`
   default → uses pre-RoPE path. The D.2a oracle
   (`teacher_forced_chunked_nll_vqbench_aligned`) is **retained
   as a regression backstop**, not deprecated at F.3. Reasons:
   (i) the (4-b) gate's anchor row migration (§9.2 risk table)
   needs the original D.2a row reachable until the migrated gate
   accumulates multi-release stability data; (ii) D.2a remains
   the chunk-grained re-encoding oracle, structurally distinct
   from A's block-grained persistent encoding — preserving both
   strands gives a free divergence detector. Removal of D.2a is
   its own follow-up after the migrated (4-b) gate is observed
   stable across multiple releases.

PLAN.md updates:

- §7 P-5 Notes "Production `prefix_store_post_rope` prefix-cache
  quality cost" bullet flips from "post-P-5 required follow-up"
  to "closed at P-5-F".
- Status line removes "pre-RoPE production routing (P-5-F)" from
  backlog.

Tests:

- The (4-b) cross-check gate (mean_gap and 2·SEM_diff) now runs
  against the production path directly, not the D.2a oracle.
  **Anchor row migration**: PLAN §7 P-5 Acceptance (4-b) currently
  cites `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`
  (the D.2a row). After F.3 the gate's anchor row migrates to
  `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` (the production row,
  now pre-RoPE). silica-side and vqbench-side numbers should
  match to seed-level noise because they share the same pre-RoPE
  injection algorithm; if they don't, that's a F.2 wiring bug.
  See §9.2 Risks for the case where the migration breaks the gate
  thresholds.
- Existing `test_*_codec*.py` tests stay green by construction
  (they use small prompts where post-RoPE chunk-boundary cost
  was already small; pre-RoPE doesn't break them).

**Commit**: one.

### 6.5 F.4 — Legacy path retention + documentation sync

Keep `prefix_store_post_rope` reachable as a bench-only opt-in
(`--codec-quality-path post_rope` on the bench CLI), retained
indefinitely until the migrated (4-b) gate is observed stable
across multiple releases (consistent with §6.4 / §9.2 / §7.4).
Documentation:

- Module docstring on `SyntheticPrefixBlockStore` explaining the
  pre-RoPE default and the legacy post-RoPE option.
- README roadmap row (P-5-F closed).
- `docs/P5_OPENING.md` §7 P-5 Notes sync.
- Memory entry in `project_kv_codec_ppl_findings.md` augments the
  existing post-RoPE table with a parallel pre-RoPE table for
  cross-arch comparison.

Removal of the legacy path itself is its own follow-up unit, not
P-5-F scope.

**Commit**: one.

## 7. Open questions left for plan→implement

These are decisions deferred to the plan/implement boundary, not
data gaps. Each has a concrete tie-break path so they don't
re-open scope.

### 7.1 Slice-prefill (C5.5) interaction — encode trigger timing

**Question**: in the C5.5 batched slice-prefill path, the encode
hook fires at reclaim time as today. The pre-RoPE round-trip
happens against post-RoPE K extracted from `BatchKVCache` at that
time — same contract as the current post-RoPE path, just with two
extra RoPE applications.

**Tie-break**: stay with post-extract reverse-then-encode (Option
A's natural shape). If F.0 ΔPPL passes, this stays. If F.0 fails
and fp32-cast doesn't recover it, C-style "encode at projection
time" becomes the escape, but that's a re-plan, not a §7 detail.

### 7.2 fp16 native vs fp16-via-fp32 cast on the round-trip

**Question**: F.0 prototype uses fp16 native. If empirical ΔPPL
lands in envelope, fp16 native is the production path. If it
lands above envelope, F.0 retries with fp16-via-fp32-cast (one
extra cast each side, near-zero cost) before declaring A failed.

**Tie-break**: F.0's data drives this — no architectural decision
needed up front.

### 7.3 Per-head Haar rotation (vqbench) vs shared rotation (silica)

**Question**: vqbench rotates per-head (`seed=h`); silica's D.2a
oracle and current production codecs share rotation. (4-b) gate
already accepts the deviation. P-5-F inherits silica's shared-rotation
choice unchanged.

**Tie-break**: out of P-5-F scope. If a future codec needs
per-head rotation, that's a codec-level addition (`BlockTQ` /
`ExtRaBitQ` constructor option), independent of the pre-RoPE
store architecture this doc covers.

### 7.4 Legacy `prefix_store_post_rope` path lifecycle

**Question**: deprecation timeline.

**Tie-break**: F.4 retains it as a bench-only opt-in (`--codec-quality-path
post_rope`) indefinitely until stability data on the new
production path accumulates. D.2a oracle is also retained
(§6.4 / §6.5) until the migrated (4-b) gate is observed stable
across multiple releases. Removal of either is its own follow-up
unit, not P-5-F scope.

## 8. Acceptance criteria

P-5-F closes when **all** of the following hold:

1. F.0 (b) ΔPPL on `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`
   under the pre-RoPE prototype passes the F.0 (b) two-part OR
   gate (§6.1): **either** lands inside the D.2a oracle envelope
   `+0.51 ± 0.35 PPL` (tight gate (i)) **or** mean ΔPPL ≤ 2.0 PPL
   representing ≥ 10× reduction from the post-RoPE arm's ~+20 PPL
   (structural gate (ii)). 3-seed evaluation, same seeds
   `{42, 43, 44}` the (4-b) gate uses.
2. F.1 unit tests green on adapter `apply_rope_inverse_to_k` for
   each of the five RoPE classes (`nn.RoPE`, `Llama3RoPE`,
   `YarnRoPE`, `SuScaledRoPE`, `ProportionalRoPE`) — round-trip
   identity within fp16/fp32 noise floors.
3. F.3 production default flipped; the `prefix_store_post_rope`
   ΔPPL on the same scenario after the flip matches F.0 (b)'s
   measurement to within seed-level noise.
4. (4-b) gate continues to pass against the post-flip production
   path (same `mean_gap ≤ 2 · SEM_diff` and `|mean_gap| < 1.0
   PPL` thresholds, same evidence format).
5. PLAN.md / README sync; the "pre-RoPE production routing
   (P-5-F)" backlog item flips to closed.

Out of scope (tracked separately):

- (b-static) Qwen3.5-4B vs `vqbench/REPORT.md` static baseline —
  P-5-F unblocks it, but the actual measurement run is its own
  unit.
- Per-head Haar rotation — codec-level, independent.
- Legacy `prefix_store_post_rope` removal — separate follow-up.

## 9. Dependencies and risks

### 9.1 Dependencies

- **P-3-C5 (closed at C5.5)**: hybrid + `RadixPrefixCache`
  cooperation under slice-prefill is required for any benchmarking
  on Qwen3.5 targets. P-5-F does not introduce new requirements
  on P-3-C5; it consumes the closed surface.
- **mlx-lm `mx.fast.rope` `freqs=-freqs` inverse semantics**:
  empirically verified at fp32 max diff 4.77e-7 and fp16 max diff
  1.95e-3 (§5.1). If a future mlx-lm release breaks this, the
  fp32-cast path or a manual sin/cos implementation is a fallback;
  silica's code documents the dependency at the call site.

### 9.2 Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| F.0 (b) ΔPPL fails both tight (i) and structural (ii) gates despite §5.1 numerical readiness | medium | diagnose per §6.1 D1 / D2 / D3; only D3 escalates to Option C |
| Some adapter family's RoPE doesn't fit the `_freqs` / `base` shape | low | manual sin/cos fallback in adapter helper; current in-use classes are `nn.RoPE` (uses `base` reconstruction) and `ProportionalRoPE` (exposes `_freqs`); other classes are forward-compat coverage |
| Production runtime cost of one extra RoPE per cached block at encode + decode | low | cold-path operation; not in per-token decode loop; small block sizes (16) keep it negligible |
| (4-b) gate anchor row migration breaks gate thresholds | medium | (4-b) gate currently anchors on the D.2a row (`...-vqbench-aligned`). F.3 migrates the anchor to the production row. If the migrated gate's `mean_gap ≤ 2·SEM_diff` no longer passes despite same algorithm, the divergence is a F.2 wiring bug — diagnose, don't relax thresholds. **Hard constraint**: D.2a row is **retained, not removed** in F.4 (legacy bench-only opt-in) so the original (4-b) anchor remains reachable for one release as a regression backstop. Removal of D.2a is its own follow-up only after the migrated gate has stable production data |
| 4B aggressive codec ROI (F.0 (a)) shows P-5-F has low marginal value on production targets at b64 b4 | low | not a blocker — P-5-F still ships for small-model and aggressive-codec deployment; PLAN urgency framing adjusts downward |

## 10. Empirical findings (updated as sub-units land)

### 10.1 F.0a — Aggressive codec on 4B targets (2026-04-26, single-seed)

Goal per §6.1: surface whether post-RoPE production path fails on
4B+ targets at low bit depth (b2 / b3), so P-5-F's deployment
priority is informed by data rather than extrapolation from the
0.6B Qwen3 number.

Method: bench scenarios `qwen3-4b-wikitext-ppl-ext-rabitq-{b2,b3,b4}`
and `qwen3.5-4b-wikitext-ppl-ext-rabitq-{b2,b3,b4}` against fp16
baselines. WikiText-2 raw test split, chunk_size=256, max_tokens=512,
block_size=16, single seed=0 (multi-seed coverage moves to F.0b
where the ΔPPL gate fires).

Result table (post-RoPE production path):

| Model | Arch | fp16 PPL | ΔPPL b4 | ΔPPL b3 | ΔPPL b2 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B | pure-attn | 12.916 | +0.460 | +3.65 | **+62.73** |
| Qwen3.5-4B | hybrid | 8.856 | +0.026 | +0.043 | +0.67 |

Findings:

1. **Pure-attention 4B at b2 completely breaks** under post-RoPE.
   The +62.73 PPL is the same failure-mode signature as the
   Qwen3-0.6B + b4 +20.83 PPL number — chunk-boundary cost
   accumulates across prefix-cache refills until model output
   degenerates. P-5-F has clear deployment ROI on pure-attention
   targets at low bit depth.
2. **Hybrid 4B at b2 stays workable** (+0.67 PPL — already inside
   the D.2a oracle's empirical floor envelope of `+0.51 ± 0.35`).
   P-5-F's marginal benefit on hybrid 4B+ at b2 is small.
3. **Architecture × bit-depth interaction is the load-bearing
   axis.** At b4 the pure-attn vs hybrid tolerance gap is ~17x
   (0.460 vs 0.026); at b2 it widens to ~100x (62.73 vs 0.67).
   The hybrid DeltaNet's recurrent state effectively buffers the
   chunk-boundary cost that pure attention layers cannot absorb.

P-5-F priority recalibration: deployment urgency is highest on
pure-attention production targets aspiring to b3/b2 codec configs
(future Qwen3-27B-class deployments under aggressive memory
pressure). Hybrid targets at b3/b4 do not require P-5-F for
quality reasons; they may still benefit architecturally
(injection-space cleanliness, codec-oracle convergence) but the
quality urgency is on pure-attention.

### 10.2 F.0b — Option A inverse-RoPE round-trip prototype (2026-04-26, 3 seeds)

**Result: F.0 (b) gate FAILS on both (i) and (ii). Diagnosis surfaces a
plan-level finding: the inverse-RoPE round-trip alone does not
reach vqbench's injection space because of the `k_norm` (RMSNorm)
layer between `k_proj` and RoPE in mlx-lm's attention.**

Method: scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-rope`
runs `teacher_forced_chunked_nll_with_codec_pre_rope` —
inverse-RoPE on K at extract seam, forward-RoPE at decode seam,
V passes through unchanged. WikiText-2, chunk_size=256,
max_tokens=512, block_size=16, seeds {42, 43, 44}.

Result table (same fp16 baseline 19.6731 across all rows):

| Path | ΔPPL per seed (42, 43, 44) | mean ± std |
| --- | --- | --- |
| post-RoPE (production) | +4.93 / +5.25 / +6.11 | **+5.43 ± 0.61** |
| F.0b pre-RoPE (this) | +3.06 / +4.41 / +4.89 | **+4.12 ± 0.95** |
| D.2a vqbench-aligned | +0.88 / +0.17 / +0.48 | **+0.51 ± 0.35** |

F.0 (b) two-part OR gate (§6.1 (b)):

- **Tight gate (i)** — inside D.2a envelope `+0.51 ± 0.35 PPL`?
  F.0b mean +4.12 PPL — **FAIL** (8× outside envelope).
- **Structural gate (ii)** — mean ≤ 1.5 PPL representing ≥ 13×
  reduction from post-RoPE +5.43 PPL? F.0b reduction is 1.3× —
  **FAIL** (only marginal improvement).

#### Diagnosis — `k_norm` between `k_proj` and RoPE; verified empirically

D1 (fp16 round-trip noise) and D2 (codec calibration drift)
ruled out by sanity checks:

- **Identity codec via the same F.0b code path**: ΔPPL +0.044 PPL.
  The inverse-RoPE / forward-RoPE round-trip is fp16-clean; the
  +4.12 PPL is not from RoPE round-trip noise.
- **Codec output is space-invariant**: BlockTQ codec applied to
  the same K tensor in pre-k_norm vs post-k_norm spaces gives
  Frobenius reconstruction ratios that match within 2% (9.46e-2
  vs 9.24e-2 on Qwen3-0.6B layer 0, 256 tokens). The codec is
  not the issue; codec behaviour is independent of which
  pre-RoPE space the input lives in.

**Recon gap (not a "new failure category")**: mlx-lm's attention
forward (`qwen3_next.py:138-148`, mirrored in `qwen3.py` and
`gemma4_text.py`) inserts a `k_norm` (RMSNorm) between
`k_proj` and `rope`:

```python
keys = self.k_proj(x)
keys = self.k_norm(keys.reshape(...).transpose(...))  # RMSNorm
keys = self.rope(keys, offset=cache.offset)
cache.update_and_fetch(keys, values)
```

The §3.1 / §5.1 RoPE-orthogonality argument
(`R(-θ)·R(θ) = I`) is mathematically correct — but it accounts
only for the RoPE axis. It does not state where the codec sees K
in the unrotated space; the implicit assumption was "right after
`k_proj`", matching vqbench's `_QuantizedProj` injection point.
Reading 4 more lines of mlx-lm attention forward (the four lines
above) would have surfaced `k_norm` as the second intermediate
operation. The recon at task #184 focused on RoPE only and missed
this. **This is a recon gap to fix in future plans**, not a
re-categorisation of F.0's failure modes.

Three injection spaces exist on real mlx-lm; F.0b's inverse-RoPE
lands in the middle one, not vqbench's:

| Space | Codec sees | ΔPPL (Qwen3-0.6B + BlockTQ b64 b4) |
| --- | --- | --- |
| post-RoPE (current production) | `RoPE(k_norm(k_proj(x)))` | +5.43 PPL |
| F.0b post-k_norm pre-RoPE (this) | `k_norm(k_proj(x))` | +4.12 PPL |
| vqbench / D.2a pre-k_norm | `k_proj(x)` | +0.51 PPL |

**Mechanism — empirically verified, 2026-04-26**: codec
reconstruction quality is the same in both pre-norm and
post-norm spaces (Frobenius ratios match within 2%). The PPL gap
comes from how the noise propagates downstream:

| Path | Noise injection point | Frobenius noise ratio at the post-norm K mlx-lm reads | Effective amplification |
| --- | --- | --- | --- |
| D.2a | pre-norm: `codec(k_proj(x))` then `k_norm(...)` | 4.49e-2 | k_norm partially absorbs the noise (RMSNorm's `1/rms` scaling regularises perturbations on `K_pre + ε`) |
| F.0b | post-norm: `codec(k_norm(k_proj(x)))` directly | 9.24e-2 | No downstream regularisation; full codec error reaches RoPE |

Same codec on the same K, but D.2a's effective noise after the
full pre-norm → norm pipeline is **~2× smaller** than F.0b's at
the post-norm K mlx-lm's RoPE actually sees. The 2× absolute
noise ratio amplifies through the attention softmax to ~8× PPL
ratio — softmax is non-linear in K, so noise propagation is not
linear in PPL.

#### Path forward — three options, advisor decision pending

The three options below have updated blast-radius reading after
advisor pass on the verification data:

1. **Ship F.0b as the "post-k_norm pre-RoPE" variant of A.**
   Accept the 1.3× improvement over post-RoPE; the +4.12 PPL
   stays well above D.2a's +0.51. Cheap, no further work, but
   does NOT close (4-b) production-path gap. Useful only as a
   third data point alongside post-RoPE and D.2a.
2. **Extend A with k_norm inverse round-trip.** Originally
   framed as "store-layer + per-block metadata" cheap
   extension. Advisor correction: capturing the per-token
   `rms_x` scalar that k_norm consumes and discards requires a
   forward-time hook **inside attention**, the same hook surface
   as Option C. (2) is no longer "cheap" relative to (3) — they
   differ only in *what gets stored*, not in *where the hook
   lives*.
3. **Escalate to Option C (projection wrapper persistence).**
   Hook `attn.k_proj` directly so the codec sees `k_proj(x)` by
   construction — matches vqbench's injection point. Plan §3.3
   originally fallback; (2) and (3) now collapse to roughly the
   same blast radius, with (3) structurally simpler (replace one
   linear layer with a wrapper, not "leave linear, add second
   hook to capture rms, plus encode/decode at separate times").

F.0b code stays as a flag-gated bench scenario regardless of the
chosen path — the "post-k_norm pre-RoPE" measurement is itself
a useful data point for codec deployment decisions on smaller
models or different bit depths where the 1.3× improvement might
matter. Bench scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-rope`
remains in the catalog.

### 10.3 F.0b' — (3b) projection-output capture prototype (2026-04-26, 3 seeds)

**Result: F.0 (b) gate PASSES with significant margin. (3b) is
the chosen P-5-F architecture; plan §3 / §6 needs re-scope away
from Option A (inverse-RoPE round-trip) toward Option (3b)
(projection-output capture wrapper).**

Method: scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-norm`
runs `teacher_forced_chunked_nll_with_codec_pre_norm`. Wraps
each attention layer's `attn.k_proj` with `_PreNormCaptureProj`,
which **returns `k_proj(x)` unchanged** (in-flight forward sees
clean K) and side-effects a capture for later encode. At chunk
end, K_pre is split per block and stored. On hit-path admit:
decode K_pre → apply layer's `k_norm` → apply RoPE → seed cache.
V handled identically to F.0b (extracted from cache, encoded
through V codec). 3 seeds {42, 43, 44}.

Result table (same fp16 baseline 19.6731 across all rows):

| Path | ΔPPL per seed | mean ± std |
| --- | --- | --- |
| post-RoPE (production) | +4.93 / +5.25 / +6.11 | +5.43 ± 0.61 |
| F.0b post-k_norm pre-RoPE | +3.06 / +4.41 / +4.89 | +4.12 ± 0.95 |
| D.2a vqbench-aligned | +0.88 / +0.17 / +0.48 | +0.51 ± 0.35 |
| **(3b) pre-k_norm via capture** | **-0.07 / +0.01 / +0.10** | **+0.015 ± 0.085** |

**Sanity check**: IdentityCodec via the same `(3b)` code path gives
ΔPPL = **0.000000** exactly (clean fp16 reproduction), confirming
the K_pre → k_norm → RoPE → seed reconstruction matches mlx-lm's
attention forward bit-exactly. The +0.015 PPL on BlockTQ b64 b4
is real codec noise on prior chunks' K, not implementation drift.

**Why (3b) outperforms D.2a**: D.2a's `_QuantizedProj` wrapper
injects codec noise on **every** forward, including the current
chunk's K. (3b)'s wrapper only captures K_pre — in-flight forward
runs clean. Codec noise affects only PRIOR chunks' K via the
seeded cache hit path, mirroring the production prefix-cache
deployment semantic where current-chunk K is freshly computed
clean and only prior chunks' K is reconstructed from the store.
The result is **strictly less noisy than D.2a** and matches what
the F.1+ production store will do.

#### F.0 (b) gate verdict

- **Tight gate (i)** — inside D.2a envelope `+0.51 ± 0.35 PPL`?
  (3b) mean +0.015 PPL — **PASS** (well below the lower bound
  +0.16, but the gate's intent is "≤ D.2a + tolerance"; (3b)
  exceeds D.2a's quality so the gate condition is satisfied).
- **Structural gate (ii)** — mean ≤ 1.5 PPL representing ≥ 13×
  reduction from post-RoPE +5.43 PPL? (3b) reduction is
  **~360×** — **PASS** (far exceeds the 13× threshold).

Both gates pass. F.0 (b) closes; (3b) is verified empirically
and is the chosen architecture for P-5-F.

#### Plan re-scope implications

The plan's §3 / §6 / §7 / §8 / §9 sections were written assuming
Option A (inverse-RoPE round-trip at store seam) was the chosen
architecture. F.0b's failure + (3b)'s success means several
plan sections need re-write:

- **§3 Architectural options** — Option A (inverse-RoPE
  round-trip alone) is a measurement variant for the codec
  deployment ablation, not the production architecture; (3b)
  becomes the production architecture and was not enumerated in
  the original §3.
- **§4 Per-family RoPE inventory** — needs extension to per-family
  k_norm inventory (every supported family in mlx-lm uses RMSNorm
  on K with the same call convention `attn.k_norm(keys.reshape(...))`,
  so the inventory is uniform; the doc just needs to record this).
- **§6 Sub-unit decomposition** — F.1 (adapter Protocol method)
  changes from `apply_rope_inverse_to_k` to a hook that captures
  K_pre at projection time. F.2's store-flag structure changes
  from "inverse-RoPE round-trip" to "K_pre persistence + on-hit
  reconstruction". F.3's default flip changes target. F.4's
  legacy retention now retains both the post-RoPE arm AND the
  post-k_norm pre-RoPE arm (two regression backstops, not one).
- **§8 Acceptance** — F.0 (b) gate is met by (3b); §8 closure
  criteria stay similar (production path ΔPPL inside D.2a
  envelope OR ≤ 1.5 PPL) but the implementation that achieves
  it is (3b), not the original Option A.

The re-scope itself is a separate commit; **this F.0 commit
lands the prototypes + verification data without changing
the plan's §3-§9 forward**. F.1+ will incorporate (3b) and
update the plan in the same commit that lands F.1.

P-5-F's final architecture decision is **(3b)**: projection-output
capture wrapper + pre-k_norm K storage + on-hit
`decode → k_norm → RoPE → seed`.
