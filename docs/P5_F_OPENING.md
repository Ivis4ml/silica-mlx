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

> **Re-scoped 2026-04-26 (commit after `503da76`).** Original §3 had
> Option A as recommended default. F.0b's verification found a recon
> gap (mlx-lm's `k_norm` between `k_proj` and RoPE) that invalidates
> the RoPE-orthogonality argument for Option A as a stand-alone
> production path. F.0b''s (3b) variant (projection-output capture
> wrapper) was verified at +0.015 PPL — better than D.2a's +0.51 — and
> is the new recommended default. Sections §3.1-§3.4 below preserve
> the original options as written; §3.5 introduces (3b) as the chosen
> architecture; §3.6 records the updated decision. See §10.3 for the
> verification data.

### 3.1 Option A — Inverse-RoPE round-trip at the store seam (measurement variant only)

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

> **F.0b verification result (2026-04-26)**: Option A reaches
> "post-k_norm pre-RoPE" space, NOT vqbench's pre-k_norm space —
> mlx-lm's `k_norm` (RMSNorm) sits between `k_proj` and RoPE
> (`qwen3_next.py:138-148`). The inverse-RoPE round-trip undoes RoPE
> only; `k_norm` cannot be undone without per-token `rms_x` metadata
> the cache doesn't preserve. Empirical ΔPPL on Qwen3-0.6B + BlockTQ
> b64 b4: **+4.12 PPL** (3 seeds), 8× worse than D.2a's +0.51 because
> `k_norm` absorbs ~2× of the noise injected pre-norm but does not
> absorb noise injected post-norm. Option A **fails** the F.0 (b)
> gate. Demoted to a measurement variant of (3b) for codec deployment
> ablations on the post-k_norm space; not the production architecture.
> Bench scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-rope`
> remains in the catalog as a third data point.

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

### 3.3 Option C / (3a) — In-flight injecting projection-wrapper (rejected)

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

> **(3a) framing rejected 2026-04-26 in favour of (3b) — see §3.5.**
> Both (3a) and (3b) install a wrapper on `attn.k_proj`. The
> difference is what the wrapper does: (3a) injects codec noise on
> the in-flight forward (mirroring D.2a's `_QuantizedProj`); (3b)
> only captures K_pre and returns `k_proj(x)` unchanged. (3b) is
> strictly less noisy (current chunk's K stays clean — codec noise
> affects only PRIOR chunks via the seeded cache hit path) and
> matches the production prefix-cache deployment semantic exactly.
> Empirically, (3b) gives +0.015 PPL on the F.0 (b) reference scenario,
> vs D.2a's +0.51 PPL. (3a) was never prototyped because the
> in-flight injection is by construction at least as noisy as D.2a.

### 3.4 Decision (original — superseded by §3.6 on 2026-04-26)

**Superseded — current decision in §3.6.**

**A was the recommended default at first plan version (v1.0.0).**
F.0 validation found A reaches the wrong space (post-k_norm
pre-RoPE, not vqbench's pre-k_norm), so this decision is
superseded by §3.6 below.

### 3.5 Option (3b) — Capture-only projection wrapper (recommended)

**Where the change lives**: a thin proxy installed on every
attention layer's `attn.k_proj` at adapter build time. Adapter
exposes a method to enable / disable a per-request capture
buffer; scheduler manages the buffer lifecycle.

**Algorithm** (production hot path):

```
on prefill chunk forward (with capture enabled):
    K_post = mlx-lm forward as today  # in-flight forward UNCHANGED
    side-effect: capture_buffer[layer_idx] = k_proj(x_chunk)  # K_pre
on reclaim (request DONE):
    for each (block_idx, layer_idx) in capture_buffer:
        K_pre_block = capture_buffer[layer_idx][block_start:block_end]
        payload[layer_idx][block_idx] = codec.encode_tensor(K_pre_block)
    store(payload, V_extracted_from_cache)
on admit (next request, prefix hit):
    payload, V_blocks = store.fetch(block_ids)
    for each (block_idx, layer_idx):
        K_pre = codec.decode_tensor(payload[layer_idx][block_idx])
        K_post = adapter.apply_k_norm_then_rope(layer_idx, K_pre, offset=block_idx*block_size)
        seeded_cache[layer_idx][block_idx] = (K_post, V_blocks[...])
    forward through seeded cache
```

**Why (3b) over (3a)**: (3a) wraps `k_proj` to inject noise on
the in-flight forward as well, mirroring D.2a's `_QuantizedProj`.
(3b)'s wrapper is **capture-only** — `__call__` returns
`k_proj(x)` unchanged, only side-effecting a write to the
capture buffer. This means:

- In-flight attention sees clean K (no codec noise on the chunk's
  own forward).
- Codec noise enters the cache only via the **hit path** at admit
  time (decoded from store).
- This is exactly the production prefix-cache deployment
  semantic: a fresh chunk's K is always clean; only K reused
  from a prior request through the prefix cache carries codec
  noise.

**F.0b' verification (2026-04-26, §10.3)**: 3-seed ΔPPL = +0.015
PPL on Qwen3-0.6B + BlockTQ b64 b4 — well below D.2a's +0.51 PPL,
and ~360× reduction from the post-RoPE +5.43 PPL. IdentityCodec
sanity check gives ΔPPL = 0.000000 exactly. Both F.0 (b) gates
pass with significant margin.

**Pros**:
- In-flight forward is bit-identical to the unwrapped path when
  capture buffer is None (production decode path is uncharged).
- Codec noise lives in the same pre-k_norm space vqbench's
  `_QuantizedProj` injects in — algorithmically equivalent to
  vqbench's quality reference.
- Persistent block-grained encoding via `RadixPrefixCache`.
- V handling unchanged from the post-RoPE store: V is `v_proj(x)`
  in cache (no normalisation, no RoPE), extracted at reclaim,
  encoded through V codec, decoded at admit, seeded directly.
- Adapter Protocol surface is small: enable/disable capture +
  per-layer reconstruct method.

**Cons**:
- Adapter must permanently install the proxy on every attention
  layer — touches `adapter.build()` per family (Qwen3 / Qwen3.5 /
  Gemma4 + MoE variants). Build-time installation is one-time
  per adapter instance; not a per-request cost.
- Capture buffer adds `(B × chunk_len × n_kv_heads × head_dim ×
  fp16_bytes × num_attn_layers)` working memory during prefill.
  Worst case Qwen3-27B at chunk=256: 64 layers × 4 KV-heads ×
  256 head_dim × 256 tokens × 2 bytes = ~33 MB per request —
  small relative to the cache itself.
- Hit-path requires per-block `apply_k_norm_then_rope` math
  before seeding the cache. Cost is tiny per block (16 tokens ×
  RMSNorm + RoPE) but adds a step the post-RoPE path skipped.

### 3.6 Decision (current, 2026-04-26)

**(3b) is the recommended default.** F.0 (b) gate empirically
verified at +0.015 PPL (§10.3); §3.6 replaces §3.4. Plan §6 is
re-scoped around (3b) below; legacy text in the original §6.2-§6.5
is preserved as historical record.

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

### 4.1 Per-family `k_norm` inventory (added 2026-04-26 per F.0b finding)

mlx-lm's attention forward inserts an RMSNorm on K (and Q)
between projection and RoPE. F.0b's recon gap missed this; (3b)
relies on it for the hit-path reconstruction
(`decode → k_norm → RoPE → seed`). The call pattern is uniform
across silica's supported families:

```python
# qwen3_next.py:138-148 (used by Qwen3.5 / Qwen3.5-MoE)
keys = self.k_norm(keys.reshape(B, L, n_kv_heads, head_dim)).transpose(0, 2, 1, 3)

# qwen3.py and gemma4_text.py have the same k_norm().reshape().transpose() shape.
```

| Family | k_norm site | k_norm class | Acts on |
| --- | --- | --- | --- |
| Qwen3 (0.6B / 4B / 27B) | `qwen3.py` Attention | `nn.RMSNorm` | last axis (head_dim) |
| Qwen3.5 GLOBAL layers | `qwen3_next.py:Qwen3NextAttention` | `nn.RMSNorm` | last axis (head_dim) |
| Gemma4 SLIDING + GLOBAL | `gemma4_text.py` Attention | `nn.RMSNorm` | last axis (head_dim) |
| Qwen3.5-MoE / Gemma4-MoE | inherited from dense parent | same | same |

Implication for (3b): `adapter.apply_k_norm_then_rope(layer_idx,
k_pre, *, offset)` reads `model.layers[layer_idx].self_attn.k_norm`
and `.rope`, applies them in mlx-lm's order. The reconstruction is
deterministic given K_pre (k_norm consumes `rms(K_pre + ε) =
rms(K_pre + ε)` — for codec noise this equals
`rms(K_pre) * (1 + O(ε/||K_pre||))` to first order, so the
reconstruction is k_norm-faithful within fp noise).

**No k_norm on V**: V skips both `v_norm` and RoPE in mlx-lm's
attention. V is `v_proj(x)` directly written to cache. The (3b)
hit-path applies k_norm + RoPE to K only; V is decoded and
seeded directly.

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

> **§6.2-§6.5 superseded 2026-04-26 by §6.6+** (re-scoped around
> (3b) after F.0b verification). Original sub-units below preserve
> the v1.0.0 plan as historical record; current implementation
> follows §6.6+ instead.

### 6.2 F.1 — Adapter Protocol: `apply_rope_inverse_to_k` (superseded)

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

### 6.3 F.2 — Store: pre-RoPE round-trip behind a flag (superseded)

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

### 6.4 F.3 — Production default flip + bench oracle convergence (superseded)

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

### 6.5 F.4 — Legacy path retention + documentation sync (superseded)

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

### 6.6 F.1 (re-scoped) — Adapter Protocol for K_pre capture + reconstruction

(Replaces the original §6.2.) Adds two related methods to the
adapter Protocol surface:

```python
class PreNormCaptureAdapter(Protocol):
    """Adapter capability surface for the (3b) production prefix-cache."""

    def install_pre_norm_capture(self, buffer: dict[int, mx.array] | None) -> None:
        """Install or reset the K_pre capture buffer on every
        attention layer's k_proj proxy. When ``buffer`` is None,
        the proxy reverts to a no-op (returns k_proj(x) unchanged
        without writing). When non-None, the proxy writes
        ``buffer[attn_layer_pos] = k_proj(x_chunk)`` on every call.

        The proxy is installed permanently at adapter build time;
        this method only swaps the active buffer. Per-request
        buffer lifecycle is owned by the scheduler.
        """

    def apply_k_norm_then_rope(
        self,
        attn_layer_pos: int,
        k_pre_block: mx.array,
        *,
        offset: int,
    ) -> mx.array:
        """Reconstruct post-RoPE K from a pre-norm K block.

        ``attn_layer_pos`` is the index into the dense list of
        attention layers (the same indexing
        ``attn_layer_indices`` already produces in the scheduler /
        bench oracle). The method reads the layer's
        ``self_attn.k_norm`` and ``self_attn.rope`` and applies
        them in mlx-lm's order:

        ```
        k_post = rope(k_norm(k_pre_block.transpose(0,2,1,3)).transpose(0,2,1,3),
                      offset=offset)
        ```

        ``offset`` is the absolute token-position of the block's
        first token in the request's prefix.
        """
```

Implementation:

1. Adapter `build()` installs `_PreNormCaptureProxy` on every
   `self_attn.k_proj` (only attention layers; DeltaNet layers
   skipped via `attn_layer_indices`). The proxy holds a
   reference to the adapter's current capture buffer.
2. **Proxy `__call__` checks `buffer is None` first; when None,
   returns `k_proj(x)` immediately with no write — required to
   keep decode-step overhead at zero (decode forwards never need
   K_pre capture).** Capture is enabled only during chunked
   prefill forwards where the scheduler has set a non-None
   buffer via `install_pre_norm_capture`.
3. `install_pre_norm_capture` is a thin setter that updates the
   adapter's "current buffer" attribute the proxies all read.
   Setting `buffer=None` is the canonical "disable capture"
   call; setting a non-None buffer arms the next forward.
4. `apply_k_norm_then_rope` is the same logic the F.0b'
   `_apply_k_norm_then_rope_to_block` helper uses, lifted onto
   the adapter Protocol.

Unit tests:

- Capture proxy fires correctly during a real prefill forward;
  buffer contents match `attn.k_proj(x)` for each attention
  layer (Frobenius distance ≤ fp16 noise floor).
- `apply_k_norm_then_rope` round-trip: extract a known K_post
  from a fresh forward, capture K_pre via the proxy, run
  `apply_k_norm_then_rope(K_pre, offset=0)`, assert it matches
  K_post bit-exactly (the function is deterministic given inputs).
- IdentityCodec regression: roundtrip K_pre → encode → decode →
  apply_k_norm_then_rope must equal the original K_post bit-exactly.
- `install_pre_norm_capture(None)` returns proxy to no-op;
  forward through the adapter is bit-identical to the
  unwrapped reference.

In-use coverage spans `nn.RoPE` and `ProportionalRoPE` per the
§4 inventory; forward-compat coverage on `Llama3RoPE` /
`YarnRoPE` retained from the original §6.2 spec.

**Commit**: one (~300-500 lines including tests). Per-family
implementations (Qwen3 / Qwen3.5 / Gemma4 + MoE wrappers) are
2-line delegations to a base helper.

### 6.7 F.2 (re-scoped) — Store: pre-norm payload contract + scheduler integration

(Replaces the original §6.3.) Two sub-commits:

**F.2a — Store layer**: extend `SyntheticPrefixBlockStore` to
accept a `pre_norm: bool = False` flag at construction. When
True:

- `register_detached(block_id, per_layer_kv)` interprets the K
  side of `per_layer_kv` as pre-k_norm K (the wrapper-captured
  output of `k_proj`); V is unchanged from the post-RoPE path.
- `fetch_detached(block_id)` returns pre-norm K + V; the caller
  is responsible for `apply_k_norm_then_rope` before seeding
  the cache.

Default is False (post-RoPE legacy path stays).

**F.2b — Scheduler integration**: `ContinuousBatcher` /
`RadixPrefixCache` paths route K_pre to the store on
reclaim, and call `adapter.apply_k_norm_then_rope` on admit
hit. Specifically:

- Before each chunked prefill forward, `_prepare_cohort` calls
  `adapter.install_pre_norm_capture(buffer)` to enable capture.
- After forward, K_pre is read out of `buffer[attn_layer_pos]`,
  split per block, paired with V extracted from the live cache,
  and written to the store.
- Before each forward, `install_pre_norm_capture(None)` clears
  the buffer (decode steps don't need capture).
- `_admit_single_hit_row` (and the batched siblings) fetch
  pre-norm K + V, call `apply_k_norm_then_rope` per block per
  attention layer, build the seeded `BatchKVCache` with the
  reconstructed post-RoPE K.

The slice-prefill path (C5.5) integrates naturally: the wrapper
captures the chunk's K_pre at chunk-forward time; reclaim's
`_extract_and_insert_prefix` reads from the captured buffer
instead of slicing the live cache for K (V continues to come
from cache, unchanged).

**Gemma4 sliding-layer reconstruction (added 2026-04-26 after F.1
landed)**: F.1 installs the K_pre capture proxy on every Gemma4
attention layer (sliding and global) since both share the same
`Attention` class with `k_proj` / `k_norm` / `rope`. F.2's
scheduler integration must NOT then reconstruct sliding layers
through the standard `BatchKVCache` admit seam — sliding layers
use `BatchRotatingKVCache` and replacing a rotating cache with a
fresh `BatchKVCache` would break the window invariants. Two
viable fixes for F.2 to pick from:

1. Gemma4-specific filter on the admit side: skip sliding layers
   when assembling the seeded-cache replacement; sliding layers
   recompute K from the prompt on hit-chunk forward (same as the
   miss path today). Capture still runs on sliding layers but
   the captured K_pre is unused for hit-path admit.
2. Sliding-aware seeded-cache build: re-construct sliding layers
   into a `BatchRotatingKVCache` with the seeded K + V, mirroring
   the `make_batch_cache` factory's per-kind dispatch.

Option 1 is simpler and matches the current Gemma4 behaviour
under `prefix_cache=None` (sliding layers are not cached today).
Option 2 is the long-term answer when sliding-window prefix
cache becomes desirable. F.2 picks one and records the choice.

**Commit count**: two (F.2a store-level, F.2b scheduler
integration). F.2a is purely additive and small; F.2b touches
the hot path and needs careful testing.

### 6.8 F.3 (re-scoped) — Production default flip + bench oracle anchor migration

(Replaces the original §6.4.) Flip three defaults:

1. `RadixPrefixCache` constructor: default `pre_norm=True` on
   the store. Production prefix-cache uses (3b) by default.
2. `silica/bench/ppl_oracle.py::teacher_forced_chunked_nll_with_codec`
   now wraps `teacher_forced_chunked_nll_with_codec_pre_norm` —
   the F.0b' code path becomes the production-store oracle.
3. PLAN §7 P-5 Acceptance (4-b) anchor row migrates from
   `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`
   (D.2a) to `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`
   (production, now (3b)). Numbers should match D.2a's
   `+0.51 ± 0.35` envelope to seed-level noise; F.0b' showed
   even better (+0.015 PPL), so the gate continues to pass.

PLAN.md updates:

- §7 P-5 Notes "Production `prefix_store_post_rope` prefix-cache
  quality cost" bullet flips to "closed at P-5-F via (3b)".
- Status line removes "pre-RoPE production routing (P-5-F)" from
  backlog.
- Empirical findings add a 2026-04-26+ entry recording the F.0b'
  verification number and the (3b) close.

Tests:

- All existing `test_*_codec*.py` tests stay green (they use
  small prompts where post-RoPE chunk-boundary cost was already
  small; (3b) is strictly less noisy).
- New regression test: full ContinuousBatcher + RadixPrefixCache
  + BlockTQ codec end-to-end, verify token stream matches the
  fp16 reference within seed-level noise.
- (4-b) gate runs against the migrated anchor row.

**Commit**: one (default flips + PLAN sync + the new regression
test).

### 6.9 F.4 (re-scoped) — Legacy retention + three regression backstops

(Replaces the original §6.5.) Three legacy paths retained as
bench-only opt-ins, all available via `--codec-quality-path
<value>` on the bench CLI:

1. `prefix_store_post_rope` — the original production path (no
   pre-RoPE handling; ΔPPL +5.43 PPL). Quantity-cost observable
   for "what does the deployment look like without P-5-F".
2. `prefix_store_pre_rope` — F.0b's inverse-RoPE round-trip
   (post-k_norm pre-RoPE space; ΔPPL +4.12 PPL). Useful as a
   third data point for codec-deployment ablations on the
   intermediate space.
3. `vqbench_aligned` — D.2a oracle (chunk-grained re-encoding
   pre-k_norm; ΔPPL +0.51 PPL). Reference against vqbench's own
   harness; structurally distinct from (3b) (chunk-grained
   re-encode vs persistent block-grained).

`prefix_store_pre_norm` is the new default (3b). All four
columns can run side-by-side via `python -m scripts.bench
--scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4
--scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-rope
--scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-norm
--scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`.

**Reading order — what each arm answers**:

Production rows route through `prefix_store_pre_norm` (default
after F.3). The other three arms remain because each measures a
different question; consume them in this order when interpreting
codec ablations:

1. **`prefix_store_post_rope`** — the cost of NOT shipping P-5-F.
   Quantifies the chunk-boundary RoPE phase mismatch on the
   legacy production path. Read this first to bound P-5-F's
   deployment ROI on a given (model, codec) combination.
2. **`prefix_store_pre_rope`** — the intermediate post-k_norm
   pre-RoPE space (F.0b's failed prototype). Read this when
   characterising codec sensitivity to `k_norm` specifically: it
   isolates the RMSNorm-induced rescaling effect on K block
   statistics, since the codec sees post-norm activations but
   the chunk-boundary RoPE mismatch is removed.
3. **`vqbench_aligned`** — the chunk-grained re-encoding D.2a
   reference. Read this for cross-implementation parity against
   vqbench's harness; it is structurally distinct from (3b)
   (chunk-grained re-encode vs persistent block-grained store),
   so the (4-b) gate uses it as the seed envelope anchor.

The three arms share row-ID prefix
`qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`; suffixes
(`-pre-rope`, `-pre-norm`, `-vqbench-aligned`) distinguish the
oracle path. The unsuffixed row is the production default and
follows whatever `prefix_store_pre_norm` resolves to after F.3.

Documentation sync:

- Module docstring on `SyntheticPrefixBlockStore` explaining the
  three retention paths and (3b) as default.
- README roadmap row (P-5-F closed via (3b)).
- `docs/P5_OPENING.md` §7 P-5 Notes sync.
- `project_kv_codec_ppl_findings.md` augment with a parallel
  pre-norm column for the cross-arch table.

Removal of any legacy path is its own follow-up unit, not
P-5-F scope.

**Commit**: one (docs only).

## 7. Open questions left for plan→implement

> **§7.1 / §7.2 resolved by F.0b verification (2026-04-26).**
> §7.3 / §7.4 stand. New §7.5 / §7.6 added to track (3b)-specific
> open items.

### 7.1 Slice-prefill (C5.5) interaction — encode trigger timing (RESOLVED)

**Original question**: where the encode hook fires in the C5.5
batched slice-prefill path — reclaim time (Option A's "extract
from cache, inverse-RoPE, encode") vs projection time (Option C's
"wrap k_proj").

**Resolution**: (3b) fires the capture at **projection time**
(during prefill forward, via the wrapper) and the encode at
**reclaim time** (scheduler reads buffered K_pre, splits per
block, encodes). This is structurally what C5.5 already does for
recurrent snapshots (per-block capture during forward), so the
two mechanisms align — the wrapper's capture buffer is read at
the same reclaim seam C5.5's snapshot machinery uses.

### 7.2 fp16 native vs fp16-via-fp32 cast on the round-trip (RESOLVED — N/A under (3b))

**Original question**: whether the inverse-RoPE round-trip needs
fp32 cast to keep noise within the codec budget.

**Resolution**: not applicable under (3b). (3b) does not
inverse-RoPE; the wrapper captures K_pre directly from
`k_proj(x)` output. Hit-path applies `k_norm + RoPE` going forward
(no inverse). fp16 round-trip noise is now zero (verified by
F.0b' IdentityCodec sanity check: ΔPPL = 0.000000 exact).

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
(§6.9 / §9.2) until the migrated (4-b) gate is observed stable
across multiple releases. Removal of any legacy path is its own
follow-up unit, not P-5-F scope.

### 7.5 Capture buffer lifecycle under batched slice-prefill (NEW)

**Question**: under C5.5 B>1 slice-prefill, the wrapper writes
`buffer[attn_layer_pos] = k_proj(chunk)` where `chunk` has shape
`(B, L, n_kv_heads * head_dim)`. The scheduler must split the
buffer per row (`buffer[attn_pos][row_b, ...]`) when assembling
per-row blocks for the prefix store.

**Tie-break**: F.2b implements per-row split at the same seam
where the slice-prefill path already splits live cache per row.
Concretely, the per-row split happens inside
`_extract_and_insert_prefix_batched` in `silica/scheduler/batcher.py`
(post-C5.5), which already iterates `(row, block_idx)` to slice
live-cache K per row. The F.2b change replaces the
`cache.keys[row, ..., start:end]` lookup for K with
`buffer[attn_layer_pos][row, start:end, ...]` (V continues to
come from `cache.values[row, ..., start:end]`, unchanged). B=1
is the F.0b' verified case; B>1 will be tested as a regression
in F.2b once the substitution is wired.

### 7.6 Adapter Protocol shape — install lifecycle (NEW)

**Question**: should `install_pre_norm_capture(buffer)` be a
context manager (`with adapter.pre_norm_capture(buffer):`) or a
plain setter pair (`install`, then `install(None)` to restore)?

**Tie-break**: setter pair is simpler for the scheduler hot path
(no nesting of context managers around per-chunk forwards). F.1
ships the setter; if a future use case wants ergonomic
context-manager semantics, the wrapper is trivial. Setter is
the canonical Protocol method.

## 8. Acceptance criteria

P-5-F closes when **all** of the following hold:

1. **(closed 2026-04-26 via F.0b')** F.0 (b) ΔPPL on
   `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-pre-norm` passes the
   two-part OR gate from the original §6.1 (kept verbatim for
   audit trail): **either** lands inside the D.2a oracle envelope
   `+0.51 ± 0.35 PPL` **or** mean ΔPPL ≤ 1.5 PPL (≥ 13× reduction
   from post-RoPE). F.0b' result: **+0.015 PPL** mean over 3
   seeds — both gates pass. Evidence in §10.3.
2. F.1 unit tests green on the adapter Protocol surface (§6.6) —
   `install_pre_norm_capture` toggles between no-op and capture
   modes correctly; `apply_k_norm_then_rope` reconstructs
   post-RoPE K bit-exactly under IdentityCodec round-trip;
   forward-compat coverage on `Llama3RoPE` / `YarnRoPE`.
3. F.2b production hot path integration: `ContinuousBatcher` +
   `RadixPrefixCache` + non-identity codec end-to-end ΔPPL on
   the same reference scenario matches F.0b' measurement to
   seed-level noise. The bench oracle and the production hot
   path now exercise the same algorithmic path.
4. F.3 default flip lands; the migrated (4-b) gate continues to
   pass on the (3b)-anchor row (`qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`,
   now `prefix_store_pre_norm` default) with the same
   `mean_gap ≤ 2 · SEM_diff` and `|mean_gap| < 1.0 PPL`
   thresholds.
5. PLAN.md / README sync; the "pre-RoPE production routing
   (P-5-F)" backlog item flips to closed.

Out of scope (tracked separately):

- (b-static) Qwen3.5-4B vs `vqbench/REPORT.md` static baseline —
  P-5-F unblocks it, but the actual measurement run is its own
  unit.
- Per-head Haar rotation — codec-level, independent.
- Legacy path removal (post-RoPE / post-k_norm pre-RoPE / D.2a
  oracle) — separate follow-up. (3b) is the production default;
  the three legacy arms remain as bench-only opt-ins per §6.9.

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

| Risk | Severity | Mitigation / Status |
| --- | --- | --- |
| F.0 (b) ΔPPL fails the gate | resolved | (3b) closed F.0 (b) at +0.015 PPL (§10.3); both gates pass with significant margin. Original Option A risk no longer applies. |
| Adapter `install_pre_norm_capture` lifecycle bug — proxy installed but buffer never set, leaks captures across requests | medium | F.1 unit test asserts `install_pre_norm_capture(None)` returns proxy to no-op; F.2b adds a per-chunk-forward setter/clear pattern in `_prepare_cohort` so buffer is always reset before forward and never persists across requests. |
| Production runtime cost of capture buffer + per-block `apply_k_norm_then_rope` on hit | low | capture buffer is ~33 MB worst case per request (§3.5 cons); `apply_k_norm_then_rope` adds ~16-token RMSNorm + RoPE per block per attention layer, far below per-token decode cost; operations only fire on prefill (capture) and admit hits (reconstruct), not in the per-token decode loop. |
| (4-b) gate anchor row migration breaks gate thresholds | medium | (4-b) currently anchors on the D.2a row. F.3 migrates the anchor to the (3b) production row. Numbers should match D.2a's `+0.51 ± 0.35` envelope to seed-level noise; F.0b' showed even better (+0.015), so the gate continues to pass. **Hard constraint**: D.2a row is **retained, not removed** in F.4 (bench-only opt-in) so the original (4-b) anchor remains reachable for one release as a regression backstop. |
| Capture proxy interferes with mlx-lm's parameter registration | low | `_PreNormCaptureProj` is a thin callable wrapping the original `nn.Linear` k_proj; doesn't subclass `mlx.nn.Module`. F.0b' verified the wrapper installs cleanly without affecting parameter discovery (model.parameters() / sanitize()). F.1 adds a regression test. |
| Hybrid-DeltaNet adapters mis-route — DeltaNet layers don't have `k_proj` and shouldn't carry a capture proxy | low | `attn_layer_indices` already filters; F.1 install loop only touches indices in this set. F.0b' verified on Qwen3.5 hybrid (no DeltaNet layers in scope of test). |
| 4B aggressive codec ROI (F.0a) shows P-5-F has low marginal value on production targets at b64 b4 | low | not a blocker — P-5-F is critical-path on pure-attention 4B+ at b3/b2 (Qwen3-4B b2 = +62.7 PPL post-RoPE per §10.1), and remains useful on hybrid 4B+ for codec-deployment cleanliness even where the +0.67 PPL gap is small. |

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

#### Path forward — three options (resolved 2026-04-26 via §10.3 — option 3 escalated to (3b))

The three options below have updated blast-radius reading after
advisor pass on the verification data. Decision: option 3 below
was escalated to (3b) (projection-output capture wrapper); §10.3
records the verification of that variant.

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

(Re-scope applied to the doc on 2026-04-26 in the same commit as
this entry; the bullet list below records what was changed in §3
/ §4 / §6 / §8 — see the corresponding sections for the new
contents.)

The plan's §3 / §6 / §7 / §8 / §9 sections were originally
written assuming Option A (inverse-RoPE round-trip at store
seam) was the chosen architecture. F.0b's failure + (3b)'s
success required re-writes across:

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
