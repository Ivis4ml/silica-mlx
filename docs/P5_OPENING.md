# P-5 — VQ KV-cache compression: opening

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Version      | v1.0.0 (draft)                                                     |
| Last updated | 2026-04-21                                                         |
| Status       | opening doc — P-4.5 closed 2026-04-21 (v1.6.9), P-5 not yet started |
| Maintainer   | Xin Zhou                                                           |
| Scope        | pin the MLX-native VQ codec platform: a `VectorCodec[P]` interface |
|              | under which `vqbench`'s three main families (TurboQuantMSE,        |
|              | BlockTurboQuantMSE, RaBitQ / ExtRaBitQ) land as the first four     |
|              | inhabitants and future VQ methods add as a single file + registry  |
|              | entry. Pick the codec order, the interface extension shape, K/V    |
|              | split policy, offline / runtime boundary, and the reproducibility  |
|              | story (can silica bench one-flag reproduce every vqbench table);   |
|              | record three sub-units P-5-A / B / C that the implementation       |
|              | commits will land against. Headline user-visible win: unified-     |
|              | memory headroom for bigger models (Qwen3.5-27B / Gemma4-31B on     |
|              | 48 GB M5 Pro) via honest compression × decode-speed × residency.   |

## 0. TL;DR

vqbench (local reference checkout `vqbench/`, NumPy + PyTorch + HF transformers) has already done the empirical work: on Qwen3.5-4B WikiText-2 streaming PPL, **BlockTurboQuantMSE with `B=64` 4-bit K+V is strictly lossless across three seeds** (`std=0.000%`), delivering 3.76× total KV compression (matching `turboquant_plus turbo4`) and holding flat through 8192 tokens / 31 cache boundaries (vqbench REPORT §3.1 / §3.4). Scalar TQ-MSE and ExtRaBitQ at 4-bit K+V are seed-dependent; at 3-bit K+V Block B=32 is the best (+1.32%), ExtRaBitQ the worst (+3.73%); RaBitQ 1-bit gives 12.8× K compression at +13.3% PPL and is production-infeasible. This is the empirical ground truth P-5 builds on.

P-5 is a **native-capability-replacement phase** (Principle 9 stub → real): `IdentityCodec` is replaced by three real codecs, and the P-4.5-C `PrefixBlockStore(codec=...)` hook is carried forward unchanged. The target is "integrate vqbench's validated algorithms into an MLX-native runtime, and make reproducing vqbench's comparisons a one-flag operation inside silica so the same methods can be applied to real inference".

Key decisions pinned in this opening:

1. **Codec order** — value-first, not paper-first. Land `BlockTurboQuantMSE` first because vqbench REPORT §3.1 identifies `B=64 4-bit K+V` as the single production-recommended configuration (strictly lossless, 3.76× total KV compression). Scalar `TurboQuantMSE` is additionally the `B=head_dim` special case of Block's own code path (§4.3 / §5.1); shipping them in the same sub-unit saves one implementation without motivating the order. `RaBitQ1Bit` + `ExtRaBitQ` follow in P-5-B. PQ / OPQ stay out of the main line (vqbench PLAN §3).
2. **Interface — side-level codec, pair-level store.** `KVCodec` (the pair-level Protocol today) splits into a side-level `VectorCodec[P]` Protocol that operates on a single tensor (either K or V, never both), and the `SyntheticPrefixBlockStore` is where the K and V codecs are held separately and dispatched per side. A pair-level convenience wrapper `KVCodec = (VectorCodec, VectorCodec)` pair type alias exists for call sites that pin the same codec for both; it is documentation, not a separate Protocol. `CodedBlock` — which today is a pair `(k, v, resident_bytes)` — splits into per-side `CodedPayload` instances; each carries its own `resident_bytes` (D-012 canonical, must equal the sum of `.nbytes` of every `mx.array` field on the payload). `IdentityCodec` becomes a `VectorCodec[RawFp16Payload]` baseline. `BlockTurboQuantMSE` is a `VectorCodec[BlockTQPayload]`; RaBitQ a `VectorCodec[RaBitQPayload]`. This resolves the internal contradiction from the earlier draft where a pair-level `encode_block(k, v)` could not cleanly serve `k_codec != v_codec` split configurations. PLAN §7 P-5 Scope already foreshadows a refactor; this opening locks **side-level**.
3. **Granularity disambiguation** — kvcache-block-size (token-axis, default 16, radix-cache granularity) and vq-block-size (head_dim-axis, typically 32 or 64, BlockTQ algorithm's per-scale block) are **two orthogonal concepts**. A single `register_detached` call encodes `n_kv_heads × kvcache_block_size` vectors (each `R^head_dim`); each vector is internally split into `head_dim / vq_block_size` quantization sub-blocks. This doc uses `kv_block` vs `vq_block` everywhere; never "block" alone.
4. **K/V split codec** — `SyntheticPrefixBlockStore` extends to `SyntheticPrefixBlockStore(block_size, k_codec=..., v_codec=...)` (the single-`codec=` kwarg from P-4.5-C.1 becomes a shorthand for `k_codec=v_codec=codec`). This resolves PLAN §10 Q-008 in the same commit as P-5-A lands — per Q-008 the lean was Option A ("K and V get independent codecs"), now confirmed by vqbench's split-codec production recommendation (K on BlockTQ, V on BlockTQ for the 4-bit K+V config; other configs want V on identity for K-only comparisons).
5. **Offline / runtime boundary** — Haar rotation matrices and Lloyd-Max codebooks are **offline calibration** artefacts. They are computed once at codec construction via NumPy + stdlib `math` (the vqbench algorithms verbatim, except `scipy.stats.norm` is replaced with stdlib `math.erf` / `math.exp` — see §5.2) and serialized into **fp32** `mx.array` constants on the codec instance (P-5-A.1a landing: fp16 would lose precision on the Haar matmul and near the Lloyd-Max centroid boundaries). Runtime `encode_tensor` / `decode_tensor` see zero NumPy — all hot-path work is pure `mx.array`, with the output cast back to fp16 / bf16 at the D-003 boundary. This is not a D-009 violation; D-009 forbids runtime NumPy, not construction-time calibration.
6. **Reproducibility story — two-track** — silica grows its own MLX-native streaming PPL oracle (`silica.bench.ppl_oracle`) so silica's own bench harness can report `{ΔPPL, compression, decode tok/s}` columns in one `python -m scripts.bench --kv-codec <name>` invocation. The existing `silica/bench/vqbench_baseline.py` (P-4.4, already shipped) continues to serve as the **independent** cross-check column via subprocess. Same pattern as P-3-D3.1's "direct mlx-lm batched reference" vs "Silica single-request" dual reference.
7. **Acceptance adds a decode-speed regression gate on a prefix-hit workload** — PLAN §7 P-5 Acceptance currently specifies (a) per-block reconstruction error, (b) end-to-end PPL drift, and (c) admits more with same budget. This opening rewrites (b) as a PPL-delta cross-check (not absolute) and (c) as a prefix-residency headroom gate (not a blanket admission multiplier), and adds (d) `decode_tok_s ≥ 0.85 × IdentityCodec_baseline` on paired new rows `qwen3-0.6b-prefix-hit-decode-fp16` + `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (existing smoke / B=1 parity rows run `prefix_cache=None` so never exercise the codec; the new rows are `[p, p] max_batch_size=1 prefix_cache=True` and measure row 1's seeded-admission decode tok/s specifically). Target amended from Qwen3.5-0.8B to Qwen3-0.6B at A.3c landing — see §7(d) rationale. Q-007 moves from "deferred" to "partially resolved": decode overhead does not enter admission decisions in v0.1, but it does gate P-5 landing.

Three sub-units:

- **P-5-A** — side-level `VectorCodec[P]` Protocol refactor; `silica.vq.block_tq.BlockTurboQuantMSE` hot path (with Scalar as `B=head_dim` special case); store K/V split; offline calibration pipeline; budgeter prefix-residency accounting; prefix-hit decode-speed row. Split into A.0 (interface + packing + calibration), A.1 (BlockTQ hot path + Q-008 resolve), A.2 (budgeter headroom), A.3 (decode-speed bench row) — see §8. Acceptance: (a) recon-error, (c) headroom, (d) decode-speed, (f) no regression.
- **P-5-B** — `silica.vq.rabitq.RaBitQ1Bit` + `silica.vq.rabitq.ExtRaBitQ`. Bit-plane payload. Estimator-native attention deferred to v0.2 (Phase-10-style; out of P-5 scope).
- **P-5-C** — `silica.bench.ppl_oracle` (MLX-native streaming PPL on WikiText-2; new `forward_batched_full` runner entry point, see §6.2), `silica.bench.scenarios` gains `--kv-codec` parametrization, 3-seed variance runs, vqbench cross-check column via the P-4.4 subprocess path. Acceptance: (b) PPL-delta cross-check, (e) one-flag vqbench table reproduction, (f) regression. (c) is owned by P-5-A.2, (d) by P-5-A.3; neither lands in P-5-C.

This opening covers the three sub-units' design; the implementation commits land independently. The full-suite acceptance regression sweep closes P-5 the same way P-4.5-close did (v1.6.9, commit `5f1b1a5`).

---

## 1. Motivation

Two design intents, stated by the user and pinned in the PLAN header:

**Intent 1 — integrate VQ into an MLX-native runtime.** vqbench established that 4-bit K+V compression is production-feasible; it did not ship a runtime. vqbench's code runs NumPy + PyTorch on CPU/MPS, compresses the KV cache **after** the HF attention module ran, and its `VQBenchCache` is a `transformers.DynamicCache` subclass — the D-010 anti-pattern silica explicitly rejects (`docs/PLAN.md` D-010 Consequences). P-5 turns the empirical validation into running code on the same MLX path that serves the P-4.5-closed scheduler.

**Intent 2 — make VQ method comparison a first-class operation inside silica.** User wants `python -m scripts.bench --kv-codec block_tq_b64_b4 --scenario qwen3-0.6b-wikitext-ppl` (or similar) to produce the same ΔPPL / compression / decode tok/s columns that vqbench currently reports through bespoke scripts (`scripts/reproduce_qwen35_4b_headline.py`, `scripts/variance_qwen35_4b.py`, etc.). That requires:

- A codec switch at `silica.kvcache.store` (already shipped as of P-4.5-C.1).
- Three real codec families wired to that switch (P-5-A / P-5-B).
- An MLX-native PPL oracle so silica's bench runner can own the ΔPPL number (P-5-C).
- The existing `silica/bench/vqbench_baseline.py` subprocess continues to run vqbench's own harness for cross-validation; it does not replace silica's own reporting.

Combined effect: silica becomes both "runtime that can load a BlockTQ codec" and "benchmark that reproduces vqbench's comparison tables", with the codec identities and evaluation oracles shared between the two roles.

### 1.1 What the P-5 codec hot-path gap was, post-C.1

P-4.5-C closed the "zero runtime callers of `encode_block` / `decode_block`" gap identified at P-4 exit (pair-level API at that time). `SyntheticPrefixBlockStore(codec=IdentityCodec(...))` now routes encode on row-0 reclaim and decode on row-1 mid-run admission in the Qwen3-0.6B forward path. But the codec being reached is the **identity** codec — compression ratio 1×, no savings observable. P-5 replaces it with the first real codec that shows savings in `store.resident_bytes()` and lets `MemoryBudgeter` admit more requests under the same budget (PLAN Principle 8).

### 1.2 Empirical ground truth, from vqbench

vqbench REPORT §3.1 (Qwen3.5-4B, WikiText-2 first 512 tokens, chunk=256, 3 seeds) headline production config:

| Config                         | mean ΔPPL | std     | Total KV vs fp16 | Verdict                                      |
| ------------------------------ | --------: | ------: | ---------------: | -------------------------------------------- |
| **Block B=64 4-bit K+V**       | **0.000%** | **0.000%** |      **3.76×** | lossless, production recommendation          |
| TQ-MSE 4-bit K+V               |   +0.262% |  0.370% |            3.94× | seed-dependent                               |
| ExtRaBitQ 4-bit K+V            |   +0.262% |  0.371% |            3.56× | seed-dependent                               |
| Block B=32 3-bit K+V           |   +1.324% |  1.488% |            4.57× | std > mean (noise floor)                     |
| TQ-MSE 3-bit K+V               |   +0.524% |  0.371% |            5.22× | seed-dependent                               |
| ExtRaBitQ 3-bit K+V            |   +3.734% |  1.666% |            4.57× | real degradation                             |
| RaBitQ 1-bit K-only            |  +13.344% |  0.725% |            1.86× | low-variance but large mean; out of scope    |

**Production target for silica P-5 is Block B=64 4-bit K+V** — the only configuration that is simultaneously strictly lossless (std=0) AND delivers a real 3.76× total-KV compression. This is the codec P-5-A ships first.

### 1.3 PLAN commitments this opening answers against

- PLAN §4 Principle 8 — "savings must be observable by the scheduler (e.g. `KVCodec.logical_bytes` vs `resident_bytes`)". P-5 makes this observable **and** actionable via the budgeter switch planned in the P-4.5-C.0 opening §6.2.
- PLAN §4 Principle 9 — "native capabilities, swappable implementations". P-5 is the canonical stub-to-real replacement.
- PLAN §7 P-5 Strategy line 539 (as amended in v1.6.7) — `PrefixBlockStore(codec=...)` injection-based switching. P-5 uses exactly this seam.
- PLAN §7 P-5 Acceptance line 550 — "Switching the codec requires no change to the scheduler or model adapter". P-5's generic-Protocol refactor (§4 below) must preserve this — `ModelAdapter` and `ContinuousBatcher` stay untouched through P-5-A / P-5-B / P-5-C.

### 1.4 P-5 as an extensible MLX-native VQ platform

The three intents above combine into a stronger framing: P-5 is not "ship BlockTQ", it is **build the MLX-native VQ codec platform that BlockTQ is the first inhabitant of**. Three consequences:

**1.4.1 MLX-native codec hot path, not an MLX-wrapped reference.** vqbench's `block_turboquant_mse.py` is NumPy + PyTorch running on CPU / MPS; its `VQBenchCache` compresses K/V tensors **after** HF's attention module has materialized them in torch memory. Even if silica imported that code verbatim, the payload would still flow through `torch.Tensor → .numpy() → mx.array` on every `register_detached` and the reverse on every `fetch_detached_blocks`. D-009 forbids that. P-5 rewrites the encode / decode hot path in pure `mx.array` operations (`mx.matmul`, `mx.multiply`, `mx.linalg.norm`, `mx.argmin` — MLX lacks `mx.searchsorted`, so quantization uses a broadcast `argmin` against the Lloyd-Max centroids directly, equivalent to midpoint-boundary lookup; `mx.take` / fancy indexing for centroid lookup; bitwise ops for sub-byte packing) so the tensors that enter a codec came from MLX attention and the tensors that leave it go back into MLX attention without a torch round trip. The NumPy calibration code stays — but only for offline codec construction (Haar seeds, Lloyd-Max codebook). §2.2 expands this boundary; §4 and §5.2 pin the hot-path rules.

*Three-role separation — inline confirmation (2026-04-22).* The end goal is MLX-native. As P-5-A.1 landed, the boundary has three explicit roles:

- **Runtime / inference hot path — MLX only.** Every `silica.vq.*` module outside the `_calibration` quarantine: `silica.vq.block_tq.BlockTurboQuantMSE.encode_tensor` / `decode_tensor`, `silica.vq.core.packing.pack_sub_byte` / `unpack_sub_byte`, `silica.vq.turboquant.TurboQuantMSE` (an alias — no separate code path), the forthcoming `silica.vq.rabitq.RaBitQ1Bit` / `ExtRaBitQ`, and `silica.kvcache.store.SyntheticPrefixBlockStore`'s per-side codec dispatch. K/V enters a codec as `mx.array` from mlx-lm attention and leaves as `mx.array` fp16/bf16 back into mlx-lm SDPA — no torch round trip, no NumPy round trip. The `silica.vq.*` NumPy-import grep test (`tests/test_calibration.py::test_calibration_module_is_the_numpy_quarantine_exception`) enforces this by walking every module under the package.
- **Offline calibration — NumPy permitted.** Only `silica.vq._calibration` may import NumPy. `haar_rotation` + `lloyd_max_codebook` run once at codec `__init__`; their outputs become fp32 `mx.array` constants stored on the codec instance. Runtime never re-consults NumPy.
- **Tests / reference — NumPy / vqbench / scipy permitted.** `tests/test_block_tq_vqbench_xcheck.py::_numpy_block_tq_round_trip` is the algorithmic-parity judge used by both (a-algo) and the v1.7.5 (a-real) real-activation xcheck; `silica.bench.vqbench_baseline` is the independent PPL cross-check subprocess for Acceptance (4-b). Neither is on the inference path; both exist to prove the MLX translation is faithful.

Performance escalation inside the MLX boundary, in order: plain MLX Python (current P-5-A.1a shape) → `@mx.compile` on decode / unpack hot paths → `mx.fast.metal_kernel` for SIMD-friendly hotspots like `unpack_sub_byte`. Obj-C / C++ extensions stay v0.2-only, gated on genuine MLX primitive gaps (Q-009 / R-7 variable-length SDPA is the named case). See §2.8 for the full ladder. **vqbench / NumPy / scipy / torch are reference and cross-validation; `silica.vq.*` runtime is MLX-native. That separation does not shift.**

**1.4.2 Adding a new codec must be a bounded, mechanical change.** The target is: a future contributor adds a fifth VQ method (say, `ProductQuantizer` or a new hand-designed codec) by

1. implementing a `VectorCodec[NewPayload]` subclass under `silica/vq/<family>/<new_codec>.py`,
2. defining a `NewPayload(CodedPayload)` dataclass with bit-packed `mx.array` fields whose `.nbytes` sum equals the declared `resident_bytes`,
3. registering one `CodecSpec` entry in `silica.vq.registry` with a stable `id` string and `spec.factory`,
4. running `scripts/bench.py --kv-codec <new_id> --scenario qwen3-0.6b-wikitext-ppl` to see ΔPPL / compression / decode tok/s alongside every other codec.

No scheduler edits, no model-adapter edits, no store-interface edits — the extensibility surface is one file in `silica/vq/<family>/` plus one registry entry. §4.1 (Protocol), §6.1 (CodecSpec), and §8 (sub-unit structure) are designed backwards from this "five-minute new-codec story".

**1.4.3 Unified-memory headroom is the concrete user-visible win.** The M5 Pro 48 GB target pins a hard budget: after weights (Qwen3.5-27B ≈ 14 GB at 4-bit, Gemma4-31B ≈ 18 GB) and reserved overhead, KV cache has to fit in roughly 25-30 GB of unified memory for the scheduler to admit meaningful concurrency. Identity-codec fp16 K/V at Qwen3.5-4B scale already burns that budget fast; at Qwen3.5-27B / Gemma4-31B scale it is the dominant factor between "runs one request" and "runs a small batch". Every extra VQ method the platform supports is a lever on the same three knobs: **compression ratio** (how much KV fits — more concurrent requests and/or longer contexts), **decode tok/s** (how much of the compression ratio survives after reconstruction overhead — §7(d) gates this at ≥ 0.85× identity), **memory residency** (`resident_bytes` honesty — D-012). Better compression, faster decode, smaller memory residency — **and** a bigger model on the same hardware — are the same project, viewed through the platform's three metric columns. The vqbench cross-check (§6.5) is how silica proves each newly landed codec earns its place against the existing ones on all three axes.

This is why §8's sub-unit structure invests P-5-A.0 in the Protocol + payload + packing + calibration infrastructure (not in any specific codec) before P-5-A.1 ships BlockTQ.

---

## 2. Constraints

### 2.1 D-003 — no compressed-domain attention fast path in v0.1

`VectorCodec.decode_tensor` must return a shape-preserving fp16 `mx.array` matching the encoder input's shape (`(1, n_kv_heads, kv_block_size, head_dim)` for a detached block K or V side). The store reassembles K and V by running `k_codec.decode_tensor` and `v_codec.decode_tensor` independently and handing the fp16 pair to `build_seeded_batch_kv`. Attention kernels (mlx-lm SDPA) only ever see fp16. This means BlockTQ and RaBitQ **reconstruct** K / V into fp16 on fetch; there is no bit-level attention path in v0.1. vqbench's Phase 10 "RaBitQ as estimator-native attention" (vqbench/PLAN.md §5) is explicitly out of P-5 scope — it would require the variable-length SDPA kernel Q-009 / R-7 identify as missing from MLX today. (Pre-P-5, the equivalent pair-level `KVCodec.decode_block(payload) -> (K, V)` API is what `SyntheticPrefixBlockStore(codec=IdentityCodec(...))` uses today; P-5-A.0 retires the pair-level surface in favour of the side-level `decode_tensor`.)

### 2.2 D-009 — MLX-native runtime hot path

`silica.vq.*` runtime code must import only `mx` / `mlx.*` / `mlx_lm.*`. NumPy is **calibration-time only** (the NumPy quarantine `silica.vq._calibration` produces write-protected `np.ndarray` outputs; codec `__init__` uploads them as **fp32** `mx.array` constants — fp16 would lose precision on the Haar matmul and near Lloyd-Max centroid midpoints). scipy is not used at all — the standard-normal CDF / PDF needed inside Lloyd-Max come from stdlib `math.erf` + `math.exp`. This is enforced by `tests/test_calibration.py::test_calibration_module_is_the_numpy_quarantine_exception` which greps every `silica.vq.*` module (including package `__init__.py`) for `import numpy` / `from numpy` and asserts only `_calibration` has them.

### 2.3 Q-009 / R-7 — no MLX variable-length attention

Constrains the scope of the codec hook to `PrefixBlockStore` detached K/V only. Same boundary as P-4.5-C: active `BatchKVCache` is not codec-wrapped. BlockTQ / RaBitQ never see decode-time K / V growth; they see prefix-cache blocks that have been extracted at reclaim, are stable-shape `(1, n_kv_heads, kv_block_size, head_dim)`, and are consumed by `build_seeded_batch_kv` on hit.

### 2.4 Q-012 — initial cohort does not consult prefix cache (v0.1 design)

Carried forward from P-4.5-C. Relevant to the bench harness design — a single-call reproduction of vqbench's "compress + re-read" pattern needs the `[prompt, prompt] max_batch_size=1` shape (or a long-context workload that crosses multiple cache boundaries inside one `generate_batch` call). Opening doc §6.2 spells this out for the PPL oracle.

### 2.5 vqbench as algorithmic + numeric reference, not runtime

`vqbench/` is gitignored and carries its own NumPy / PyTorch / HF transformers environment. `vqbench/REPORT.md` + `vqbench/BlockTQ.md` are the algorithm walkthroughs; `vqbench/vqbench/methods/turboquant/block_mse.py`, `rabitq/rabitq_1bit.py`, `rabitq/rabitq_ext.py` are the reference implementations to translate. P-5 does **not** `pip install vqbench`; it rewrites each method's encode / decode loop on `mx.array`, cross-checks reconstruction numerics against in-test transcribed references, and cross-checks PPL numerics through the separate P-4.4 `silica.bench.vqbench_baseline` subprocess channel.

### 2.6 MoE-compatible interface

D-011 (v0.1 must run at least one MoE target) applies. The codec interface can not assume a single KV layout across layers — Gemma4's sliding 16×256 + full 4×512 mix needs per-layer codec instances or a shape-adaptive constructor. P-5-A ships homogeneous-shape only (Qwen3.5-0.8B / Qwen3.5-4B), aligned with the P-4.5-C.1 scope decision; heterogeneous-shape support is a P-5-A follow-up bullet, not a v0.2 punt.

### 2.7 P-4.5-C.1 interface inheritance

`SyntheticPrefixBlockStore(block_size, codec=None)` exists. `codec.block_size` precondition must match `kv_block_size` (= 16 by default). The codec's internal vq_block_size (default 64) is orthogonal — it lives inside the codec and never appears at the store interface. The `resident_bytes()` parallel observable already works on `CodedBlock.resident_bytes`; the generic-Protocol refactor preserves this attribute as part of the base `CodedPayload`. **See §4.4 for the `CodedBlock` → `CodedPayload` migration plan — P-5-A.0 retires the pair-level type; this bullet describes the pre-P-5 inheritance surface, not the target.**

### 2.8 MLX stack boundary and escalation ladder

P-5 v0.1 is a Python-only MLX effort. The runtime import boundary for `silica.vq.*` is:

- **Allowed on the hot path:** `mx.*`, `mlx.*`, `mlx_lm.*`, plus the two MLX-native performance escape hatches `mx.compile` (graph fusion decorator) and `mx.fast.metal_kernel` (Metal shader source registered from Python, dispatched through MLX, inputs and outputs are `mx.array`). `mx.fast.metal_kernel` counts as MLX-native under D-009 because it does not introduce a pybind11 / Objective-C / C++ build chain, does not produce a separate shared object silica has to load, and does not create a second `mx.array` type — the Metal source lives in a Python string and the dispatch goes through MLX's scheduler.
- **Disallowed on the hot path:** NumPy / scipy / PyTorch (D-009; calibration-only under `silica.vq._calibration` and codec `__init__` bodies is the one exception), and external C / C++ / Objective-C extensions. The MLX C++ `mlx::core::array` API + pybind11 extension path exists but is not reached in P-5.

Escalation ladder, driven by the §7(d) prefix-hit decode-speed gate at 0.85× IdentityCodec:

1. **Plain MLX Python first.** Ship `BlockTurboQuantMSE`, `RaBitQ1Bit`, `ExtRaBitQ`, `pack_sub_byte` / `unpack_sub_byte` as straightforward `mx.array` pipelines. Measure §7(d) on Qwen3-0.6B (target amended from Qwen3.5-0.8B at A.3c landing per §7(d) rationale — hybrid-DeltaNet `has_recurrent_state=True` incompat with `RadixPrefixCache`; the BlockTQ-vs-identity ratio is head-dim / layer-count independent so the escalation-ladder threshold carries over).
2. **If §7(d) fails: `@mx.compile` the decode + unpack hot paths.** Graph fusion typically recovers the last 10–20% on element-wise-heavy chains (codebook lookup + scale multiply + bit-unpack mask/shift). Remeasure.
3. **If still failing: write a targeted `mx.fast.metal_kernel`.** The most likely hotspot is `unpack_sub_byte` — the loop-over-`head_dim × num_bits / 8` pattern is exactly what SIMD was built for and is the hardest case for the MLX compiler to fuse across iterations. A Metal shader for the 4-bit unpack path is the expected ceiling of what v0.1 needs; the shader source stays inside `silica/vq/core/packing.py` as a Python string.
4. **Reserved for v0.2: Obj-C / C++ extension.** Only appropriate when (a) MLX lacks a primitive outright — the specific case is variable-length SDPA flagged by Q-009 / R-7, which gates vqbench Phase 10 "RaBitQ as estimator-native attention"; or (b) direct `MPSGraph` / Metal Performance Shaders interop is required. Neither applies to P-5. Pulling pybind11 / Obj-C into v0.1 would trade the main P-5 risk ("VQ semantics + budgeter accounting are correct") for a different risk ("native extension build / packaging / distribution works on macOS + iOS"); the swap is not worth it at this stage.

Boundary vs D-003 and D-009: D-003 forbids compressed-domain attention (bit-level attention kernels); an `mx.fast.metal_kernel` used only for pack / unpack / decode-reconstruct-to-fp16 does not cross D-003 because its output is still shape-preserving fp16 consumed by mlx-lm SDPA. D-009 forbids runtime NumPy / torch, not MLX-native custom kernels. Q-009 / R-7 variable-length SDPA is the compressed-domain-attention kernel D-003 defers; a P-5 Metal shader for `unpack_sub_byte` does not depend on or pre-empt that kernel.

Acceptance consequence: §7(d) must be measured on the ladder step that actually ships. The bench row records which step was taken (plain / compiled / metal_kernel) so later regressions can see whether the gate was held by a plain-Python baseline or by a Metal shader that has to keep working.

---

## 3. Scope

### 3.1 In scope

- `silica.vq.turboquant.TurboQuantMSE` — Algorithm 1 scalar quantization, MLX-native.
- `silica.vq.block_tq.BlockTurboQuantMSE` — per-vq-block-scale variant, MLX-native. `B ∈ {16, 20, 32, 40, 64}` parametrization follows vqbench REPORT §3. `B = head_dim` reduces to Scalar (tested invariant, mirrors vqbench).
- `silica.vq.rabitq.RaBitQ1Bit` — hypercube + unbiased IP estimator, MLX-native.
- `silica.vq.rabitq.ExtRaBitQ` — multi-bit via integer-grid codebook, MLX-native.
- `silica.kvcache.codec` side-level generic-Protocol refactor: new `VectorCodec[P]` plus `CodedPayload` base with per-codec payloads (`RawFp16Payload`, `BlockTQPayload`, `RaBitQPayload`). Retires the pre-P-5 pair-level `KVCodec` / `CodedBlock` shipped in P-4.5-C.1.
- `SyntheticPrefixBlockStore` K/V split: `k_codec` / `v_codec` kwargs.
- `silica.bench.ppl_oracle` — MLX-native WikiText-2 streaming PPL, chunk=256, same semantics as vqbench's `validation/streaming_ppl.py`.
- `silica.bench.scenarios` `--kv-codec` parametrization + codec-specific bench rows (`qwen3-0.6b-wikitext-ppl-{fp16,tq-mse-b4,block-tq-b64-b4,ext-rabitq-b4}`, `qwen3-0.6b-compression`, `qwen3-0.6b-prefix-hit-decode-{fp16,block-tq-b64-b4,ext-rabitq-b4}`). All P-5 rows target Qwen3-0.6B — a non-recurrent Qwen3 adapter that `ContinuousBatcher` accepts with `RadixPrefixCache`. The original draft put the PPL and compression rows on Qwen3.5-0.8B on the reasoning "the oracle does not need `RadixPrefixCache`", but P-5-C.2's codec-backed PPL arm does seed per-chunk caches from a prefix cache, so keeping every P-5 row on one tokenizer / one adapter shape removes a cross-row variability source. (Qwen3.5-0.8B is hybrid-DeltaNet (`has_recurrent_state=True`) and `ContinuousBatcher` refuses `RadixPrefixCache` on recurrent adapters — see `docs/P3_DELTANET_SURVEY.md` C-open-3 — so it cannot carry the C.2 codec-backed PPL arm regardless.)
- `MemoryBudgeter` prefix-residency accounting — three explicit modes per §4.7: (A) `account_prefix_residency=False` recovers the P-4.5 byte-for-byte headroom formula as a regression lock; (B) IdentityCodec store bound + flag on charges honest fp16 residency in `headroom_bytes()`; (C) compressed codec bound + flag on charges compressed residency. Evict-shortfall continues to use LRU count-based eviction in v0.1 (`AdmitAfterEvictDecision(n_blocks)` unchanged), but `bytes_freed_by_evicting` reads `resident_bytes_per_block` off the store rather than the fp16 constant. The P-4.5-C.0 opening §6.2 foreshadowed this transition; §4.7 is the full spec.

### 3.2 Explicitly out of scope

- **PQ / OPQ** — vqbench keeps them in `methods/pq/` for benchmark parity but does not recommend them for production. silica follows.
- **Compressed-domain attention / bit-plane estimator-native attention** — D-003 + Q-009 / R-7. vqbench's Phase 10 item is the v0.2 successor.
- **Per-head / per-layer rotation calibration** — vqbench observes each head's K distribution differs (`scripts/real_k_mse_qwen35.py`); per-layer calibration is a v0.2 codec enhancement, not a P-5 blocker. P-5 uses one `(head_dim, seed)` Haar rotation shared across all layers, matching vqbench's production path.
- **`PagedPrefixBlockStore` codec integration** — paged-attention kernel still requires D-003-excluded primitives. Keep the `NotImplementedError` stubs.
- **Active-K/V codec wrapping (Option A/C from P-4.5-C)** — same reason as v0.1.
- **Decode-overhead-driven admission** — Q-007 lean was "partial resolve at P-5"; this opening resolves it to "decode overhead is a release gate (§7 acceptance (d)) but not an admission signal in v0.1". Admission-signal work lands with v0.2 multi-codec multi-quality-setting support.
- **Structured output / logit-processor interaction** — Q-011 is orthogonal.

### 3.3 Deferred to P-5 follow-up, not blocking P-5 close

- **3-bit K+V** — vqbench REPORT §3.2 shows every 3-bit config has std ≥ mean (noise floor). P-5 lands 4-bit K+V first; 3-bit K+V becomes a bench row under the same codec class with a different `num_bits` parameter.
- **Long-context stability sweep** — vqbench REPORT §3.4 does 512 → 8192 tokens on 3-seed. P-5-C ships the single-length oracle; the length sweep is a bench-runner extension that operates on top of the codec.

---

## 4. Interface design

### 4.1 `VectorCodec[P]` — side-level, Generic

vqbench's `VectorQuantizer.quantize(x) -> QuantizedVector` is already side-level: one tensor in, one payload out. silica's pre-P-5 `KVCodec.encode_block(k, v)` was pair-level and could not cleanly route `k_codec != v_codec` configurations — if a caller passed `k_codec=BlockTQ, v_codec=IdentityCodec`, the pair-API implementation has to choose whose `encode_block` sees both tensors, and the type system no longer tracks which payload came from which side. P-5 therefore retires the pair-level Protocol and replaces it with a side-level `VectorCodec[P]`. The store is where pair-aware dispatch happens.

```python
# silica/kvcache/codec.py

from typing import Generic, Protocol, TypeVar

@dataclass
class CodedPayload:
    """Base payload. All codecs contribute resident_bytes for D-012
    accounting; resident_bytes MUST equal the sum of .nbytes across
    every mx.array field on the concrete payload (verified by a
    fairness test across all registered codecs)."""
    resident_bytes: int
    vector_shape: tuple[int, ...]   # original (1, n_kv_heads, kv_block_size, head_dim) before encode

@dataclass
class RawFp16Payload(CodedPayload):
    """IdentityCodec's payload: one fp16 tensor, by reference."""
    tensor: mx.array                # shape == vector_shape; K or V, not both

@dataclass
class BlockTQPayload(CodedPayload):
    """BlockTurboQuantMSE payload — bit-packed.

    packed_indices: uint8, shape (n_vectors, ceil(head_dim × num_bits / 8)).
                    Sub-byte bit-packed codes. For num_bits=4 head_dim=256
                    this is (n_vectors, 128), exactly 4 bits/coordinate.
    scales:         float16, shape (n_vectors, n_vq_blocks = head_dim / vq_block_size).
                    One fp16 scale per vq-block per vector.
    """
    packed_indices: mx.array
    scales: mx.array

@dataclass
class RaBitQPayload(CodedPayload):
    """RaBitQ family payload — bit-packed.

    packed_indices: uint8, shape (n_vectors, ceil(head_dim × num_bits / 8)).
                    For 1-bit RaBitQ, shape is (n_vectors, head_dim / 8).
                    For B-bit ExtRaBitQ, shape is (n_vectors, head_dim × B / 8).
    norm_o:         fp16, per-vector.
    ip_coeff:       fp16, per-vector.
    """
    packed_indices: mx.array
    norm_o: mx.array
    ip_coeff: mx.array
    # centroid is on the codec instance, not the payload (fit-once)

P = TypeVar("P", bound=CodedPayload)

class VectorCodec(Protocol, Generic[P]):
    """Side-level codec: operates on K OR V, not both.
    See §4.2 for how the store composes two VectorCodecs into K/V split."""
    block_size: int        # = kv_block_size, must match store.block_size
    dtype: mx.Dtype        # = mx.float16 for all v0.1 codecs (D-003)

    def encode_tensor(self, x: mx.array) -> P: ...
    def decode_tensor(self, payload: P) -> mx.array: ...
    def logical_bytes(self, num_tokens: int) -> int: ...
    def resident_bytes(self, num_blocks: int) -> int: ...
```

### 4.2 Store holds per-side codecs and per-side payloads

`SyntheticPrefixBlockStore` is the pair-aware layer. Its detached dict becomes (sketch — Unit 4 checklist `docs/P5_A_U4_STORE_MIGRATION.md` §5 refines this to a private `_DetachedLayer` frozen dataclass for readability; the behaviour is the same):

```python
self._detached: dict[int, tuple[tuple[CodedPayload, CodedPayload], ...]] = {}
#                                   ^K payload      ^V payload
#                              ^^^ one (K, V) pair per layer ^^^
```

and the constructor takes two side-level codecs:

```python
class SyntheticPrefixBlockStore:
    def __init__(
        self,
        *,
        block_size: int,
        k_codec: VectorCodec | None = None,
        v_codec: VectorCodec | None = None,
        codec: VectorCodec | None = None,     # shorthand: k_codec = v_codec = codec
    ) -> None:
        if codec is not None and (k_codec is not None or v_codec is not None):
            raise ValueError(...)
        if codec is not None:
            k_codec = v_codec = codec
        # codec=None (on either side) keeps the P-4.5-C.1 pass-through
        # path: raw tensors wrapped in RawFp16Payload without indirection.
        ...

    def register_detached(self, block_id, per_layer_kv):
        # per_layer_kv: Sequence[tuple[mx.array, mx.array]] — external shape unchanged
        self._detached[block_id] = tuple(
            (self._encode_k(k), self._encode_v(v)) for k, v in per_layer_kv
        )

    def fetch_detached(self, block_id):
        return tuple(
            (self._decode_k(kp), self._decode_v(vp))
            for (kp, vp) in self._detached[block_id]
        )
```

The external contract (`register_detached` takes `Sequence[tuple[mx.array, mx.array]]`; `fetch_detached` returns the same shape) is unchanged from P-4.5-C.1 — `build_seeded_batch_kv` does not need to know a codec exists, and `_admit_single_hit_row` in the batcher stays untouched. PLAN §7 P-5 Acceptance line 550 ("Switching the codec requires no change to the scheduler or model adapter") is honored by construction.

**Q-008 resolution (same commit as P-5-A lands):** K and V get independent codecs via `k_codec` / `v_codec`. vqbench's production data backs this (K-only configs use `value_q = IdentityCodec`; K+V configs use `value_q = BlockTurboQuantMSE`; future Phase-10-style configs would use `key_q = RaBitQ, value_q = BlockTurboQuantMSE` per vqbench PLAN §5).

### 4.3 `resident_bytes` must equal actual `mx.array` `.nbytes`

The D-012 canonical definition is "physical bytes currently owned by the component in unified memory". A payload that claims `resident_bytes = n_vectors × head_dim × num_bits / 8` while actually storing `indices` as int8 (1 byte per coordinate, not `num_bits / 8`) over-reports compression by a factor of `8 / num_bits`. At `num_bits = 4` that is a silent 2× over-report; at `num_bits = 3` a 2.7×. This breaks admission math (Acceptance (c)) and breaks the vqbench cross-check (Acceptance (b)) because the "savings" silica reports would not exist in unified memory.

P-5's rule: **`payload.resident_bytes` is computed as the sum of `.nbytes` over every `mx.array` field on the payload.** No hand-written bit-count arithmetic. `silica.vq.core.packing` ships the pack / unpack helpers so every sub-byte codec stores bit-packed `uint8` arrays natively. A per-codec fairness test asserts:

```python
# tests/test_vq_resident_bytes_honesty.py
for codec in registered_codecs:
    x = sample_vector(codec)
    payload = codec.encode_tensor(x)
    expected = sum(
        arr.nbytes for arr in payload_mx_array_fields(payload)
    )
    assert payload.resident_bytes == expected, (
        f"{codec.name}: payload.resident_bytes ({payload.resident_bytes}) "
        f"does not match actual .nbytes sum ({expected}) — codec is "
        f"mis-reporting compression ratio"
    )
```

BlockTQ storing `packed_indices: uint8` shape `(n_vectors, head_dim × num_bits / 8)` therefore gets honest `.nbytes` arithmetic; the `(num_bits + 16 / vq_block_size)` effective-bits/dim claim from vqbench BlockTQ.md §1 holds in unified memory, not only in theory.

### 4.4 Backward-compat alias for P-4.5-C.1 `CodedBlock`

P-4.5-C.1 introduced `silica.kvcache.codec.CodedBlock = (k, v, resident_bytes)` with `register_detached` returning it. P-5-A migrates this:

- `CodedBlock` is removed from the public API at P-5-A landing (not kept as a deprecated alias, because its pair shape is exactly what side-level routing fixes).
- Tests and call sites that referenced `CodedBlock` or `KVCodec` by name are updated in the same commit. The P-4.5-C.1 tree has two direct referents, both of which P-5-A.0 migrates to the side-level `RawFp16Payload` + `VectorCodec` names:
  - `tests/test_kvcodec.py` — the P-0 interface-conformance test (imports the Protocol name directly).
  - `tests/test_kvcodec_integration.py` — imports `CodedBlock`, `IdentityCodec`, `KVCodec` at module top (`tests/test_kvcodec_integration.py:79`) and constructs a `_CountingIdentityCodec` test double whose `encode_block` returns a `CodedBlock` (`tests/test_kvcodec_integration.py:149`). P-5-A.0 rewrites the test double as a `_CountingIdentityCodec(VectorCodec)` that implements `encode_tensor` / `decode_tensor` per side and whose store-level paired `(K, V)` encode is counted from the store's dispatch, not from a pair-level codec method. The two defensive `len(store._detached) == len(store.live_block_ids())` asserts at Section 2 and Section 4 continue to pass without change because the dict keys are block_ids, independent of payload structure.

### 4.5 Granularity — kvcache_block vs vq_block

Two orthogonal "block" concepts, consistently distinguished in code and prose:

| name                  | units        | default value | where it lives                          |
| --------------------- | ------------ | ------------: | --------------------------------------- |
| `kv_block_size`       | tokens       |            16 | `RadixPrefixCache.block_size`           |
| `vq_block_size`       | head_dim dim |            64 | codec-internal (`BlockTQPayload.scales`) |

Mapping flow for one `register_detached` call under `codec = BlockTQ(head_dim=256, vq_block_size=64, num_bits=4)`:

```
per_layer_kv[layer_idx] = (k, v)
  k shape = (1, n_kv_heads=4, kv_block_size=16, head_dim=256)
  → flatten to (n_kv_heads * kv_block_size, head_dim) = (64, 256) vectors
  → each vector ∈ R^256 is quantized BlockTurboQuantMSE:
    - Haar rotation (applied as mx.matmul with the pre-calibrated (256, 256) fp16 matrix)
    - reshape to (n_vq_blocks = 4, vq_block_size = 64)
    - per-vq-block norm → 4 fp16 scales
    - scalar Lloyd-Max quantize to 4-bit → 4 * 64 = 256 indices (int8)
  → payload.indices: shape (64, 256), int8
  → payload.scales:  shape (64, 4), fp16
  → payload.resident_bytes = 64 * (256 * 4 bits + 4 * 16 bits) / 8
                           = 64 * 136 bytes = 8704 bytes
```

`decode_tensor` reverses: codebook lookup per coordinate, re-multiply by per-vq-block scale, inverse Haar rotation, reshape back to `(1, n_kv_heads, kv_block_size, head_dim)`. Output is fp16 — D-003 compliant. (The store dispatches one `decode_tensor` call per side so the K and V payloads are reconstructed independently; the pre-P-5 `decode_block` pair API is retired at P-5-A.0.)

Invariant tested at P-5-A.1: on the boundary `vq_block_size = head_dim`, `BlockTurboQuantMSE` produces bit-identical output to `TurboQuantMSE` at the same bits / same seed, same as vqbench's `tests/test_block_quant.py::test_block_equals_scalar_when_B_equals_d`.

### 4.6 Offline calibration vs runtime

Codec construction runs offline (NumPy + stdlib ``math``):

```python
class BlockTurboQuantMSE:
    def __init__(self, *, head_dim, vq_block_size, num_bits, seed=42, norm_correction=True):
        # --- offline, NumPy + stdlib math (scipy not used; Lloyd-Max
        #     uses math.erf / math.exp for the standard-normal CDF / PDF),
        #     run once ---
        rotation_np = haar_rotation(head_dim, seed)                     # (head_dim, head_dim) float64
        centroids_np, _boundaries_np = lloyd_max_codebook(
            num_bits, sigma=1.0 / math.sqrt(vq_block_size),
        )
        # --- serialize as fp32 mx.array constants on the codec instance.
        #     fp32 matters: fp16 would erode Haar-matmul precision and
        #     shift values near Lloyd-Max centroid midpoints, flipping
        #     codebook indices relative to the vqbench reference.
        #     Memory cost for d=256: 256 KB rotation matrix — tiny vs
        #     the KV cache. Boundaries are not materialised as an mx.array
        #     because MLX lacks mx.searchsorted; quantization uses
        #     broadcast mx.argmin against centroids directly. ---
        self._rotation = mx.array(rotation_np, dtype=mx.float32)         # (head_dim, head_dim)
        self._centroids = mx.array(centroids_np, dtype=mx.float32)       # (2^num_bits,)
        # --- configuration ---
        self.block_size = <kv_block_size supplied separately by store>
        self.head_dim = head_dim
        self.vq_block_size = vq_block_size
        self.num_bits = num_bits

    def encode_tensor(self, x) -> BlockTQPayload:
        # --- runtime, 100% mx.array ---
        # ... validate shape + dtype against codec config,
        #     Haar rotate (fp32 matmul), split into vq_blocks, per-block
        #     norm, scalar quantize via mx.argmin((y - centroids)²)
        #     (broadcast-replaces mx.searchsorted which MLX lacks),
        #     pack_sub_byte to uint8
        ...
```

The `_haar_rotation_numpy` / `_lloyd_max_codebook_numpy` helpers live in `silica.vq._calibration` and are the **only** place NumPy is allowed to appear under `silica.vq.*`. A runtime-hot-path test greps every file for `^import numpy\b` / `^from numpy` / `np\.` and asserts they appear only inside `_calibration.py` or inside `__init__` bodies.

Construction is not free — `_lloyd_max_codebook_numpy` at `num_bits=4, sigma=1/sqrt(64)` runs ~200 iterations to 1e-12 tolerance, takes a few ms. That is acceptable: codec construction happens once per `Engine(...)`. The vqbench reference reuses the same Lloyd-Max codebook code verbatim.

**No on-disk serialization format in v0.1.** Every `Engine` re-calibrates on construction. A future v0.2 optimization caches `_calibration` output to disk keyed by `(head_dim, vq_block_size, num_bits, seed)`; deferred because cold-start cost is in the milliseconds.

### 4.7 Interface swap without scheduler / adapter changes

PLAN §7 P-5 Acceptance line 550: "Switching the codec requires no change to the scheduler or model adapter". The generic-Protocol refactor is designed to honour this:

- `ContinuousBatcher._extract_and_insert_prefix` hands per-layer `(K, V)` fp16 slices to `RadixPrefixCache.insert_detached`, which routes them to `store.register_detached`. No codec awareness.
- `build_seeded_batch_kv` receives per-layer `(K, V)` fp16 tuples from `store.fetch_detached` and concatenates. No codec awareness.
- `ModelAdapter` has no codec touch-point — the codec is installed on the `SyntheticPrefixBlockStore` inside the `RadixPrefixCache` the caller passes to `Engine.generate_batch`.
- `MemoryBudgeter.admit()` keeps its worst-case reservation formula untouched (reservation is still `(n_prompt + max_tokens) × bytes_per_token` — active-KV upper bound, D-003 constrained). The real gap P-5 closes is that today's budgeter does **not** charge prefix-cache residency in `headroom_bytes()` at all (`silica/scheduler/budget.py` line 237-243: `headroom = cap - weights - reserved`). Under IdentityCodec the omission is harmless because prefix bytes are small relative to active-KV worst-case; under a real codec the omission makes the compression win invisible to admission.

  The P-5 budgeter change is threefold, confined to `silica/scheduler/budget.py` + `silica/kvcache/store.py`, no `ContinuousBatcher` / `ModelAdapter` touch-points. It has three explicit accounting modes — reviewer called out that conflating "pass-through codec" with "no residency accounting" is wrong, because `IdentityCodec` wraps `RawFp16Payload` whose `.resident_bytes` correctly reports the fp16 bytes, not zero:

  | Mode | Construction | `store.resident_bytes()` | `headroom_bytes()` | Purpose |
  | ---- | ------------ | ------------------------ | ------------------ | ------- |
  | **(A) P-4.5 byte-for-byte** | `prefix_cache=None`, or `MemoryBudgeter(account_prefix_residency=False)` | — (not consulted) | `cap - weights - reserved` | Regression lock for the P-4.5 close sweep; acceptance (f). |
  | **(B) IdentityCodec baseline** | `SyntheticPrefixBlockStore(codec=IdentityCodec(...))` bound, flag on | `Σ raw fp16 payload .nbytes` (honest) | `cap - weights - reserved - store.resident_bytes()` | The `N_fp16` number in acceptance (c); the baseline row-1 decode tok/s in acceptance (d). |
  | **(C) BlockTQ / RaBitQ compressed** | `SyntheticPrefixBlockStore(k_codec=BlockTQ, v_codec=BlockTQ)` bound, flag on | `Σ compressed payload .nbytes` | same formula as (B), compressed store-bytes smaller | The `N_block` number in (c); the row-1 decode tok/s under test in (d). |

  Modes (B) and (C) must share one formula with honest per-mode residency — otherwise `N_block > N_fp16` has no arithmetic meaning (C would admit more "for free" because B is zero'd instead of charged). Mode (A) is a separate compatibility codepath selected by `account_prefix_residency=False` or by the absence of any `prefix_cache`, not by the pass-through store.

  The three implementation pieces:

  1. **Budgeter binds a store reference via a new accessor.** The current `MemoryBudgeter.__init__` already takes `prefix_cache: RadixPrefixCache`; P-5-A.0 adds a `RadixPrefixCache.store` property (so callers stop poking `_store`) and `MemoryBudgeter.__init__` gains `account_prefix_residency: bool = True` plus stores `self._store = prefix_cache.store if prefix_cache else None`. Mode (A) via `account_prefix_residency=False` is how the P-4.5 regression tests pin pre-P-5 numbers byte-for-byte without having to disable `RadixPrefixCache` entirely.
  2. **`headroom_bytes()` subtracts prefix residency in modes (B)/(C):** `headroom = cap - weights - reserved - (self._store.resident_bytes() if (self._account_prefix_residency and self._store is not None) else 0)`. This is the knob that makes codec savings visible to the admit path. Under IdentityCodec, prefix residency is the honest fp16 count — so the baseline row pays the fp16 admission tax. Under BlockTQ, residency is the compressed count — so headroom is larger → `new_worst ≤ headroom` fits for more incoming requests. **This is the direction of the compression gain**, not the evict-shortfall branch.
  3. **Evict-shortfall stays count / LRU in v0.1.** The existing decision is `AdmitAfterEvictDecision(n_blocks, reserved_delta)` (`silica/scheduler/budget.py:77`), applied via `RadixPrefixCache.evict_until(n_blocks)` (`silica/kvcache/prefix.py:289`) in strict LRU-leaf order. Under every P-5 v0.1 codec the resident-byte count per evictable block is constant (payload shape is a function of `(kv_block_size, head_dim, num_bits, vq_block_size, num_layers)`, all fixed at codec construction + first `register_detached`), so the count-based policy is already arithmetically honest — `shortfall_bytes / resident_bytes_per_block` gives the block count, where `resident_bytes_per_block` is the bytes freed by evicting one radix block_id (all layers, both sides). The store exposes this as a single `store.resident_bytes_per_block() -> int | None` method that returns `num_layers × (k_codec.resident_bytes(1) + v_codec.resident_bytes(1))` once `num_layers` has been learned from the first `register_detached` call, and `None` before that (caller falls back to the fp16 baseline which is never consumed in the absence of evictable blocks). **Why include `num_layers`**: one radix block_id covers all layers of one token-block; `RadixPrefixCache.evict_until(1)` drops all of them together. Returning a per-layer figure would under-count eviction bytes by `num_layers×` and over-reject admissions (regression-locked by `tests/test_memory_budgeter.py::test_mode_c_admit_evicts_one_block_when_shortfall_between_single_and_all_layer_bytes`). No per-id apply surface, no greedy-largest-first selection, no `block_ids` payload on the decision — all of those are v0.2 work (motivated by heterogeneous-shape Gemma4 or variable-size payloads, which P-5 does not ship). The single landed change inside the budgeter is `_evict_bytes_per_block()` reading from `store.resident_bytes_per_block()` rather than from `_kv_bytes_per_block()`'s fp16 constant.

     Rejected alternative, recorded for v0.2: promote `AdmitAfterEvictDecision` to carry an explicit `block_ids: tuple[int, ...]` list, add `RadixPrefixCache.evict_block_ids(ids)` + a per-ids `store.resident_bytes_for_block_ids(ids)` helper, and let the budgeter pick evictees greedily by payload size. That is mechanically straightforward but buys nothing in v0.1 where every evictable block is the same size; deferring it keeps the P-5-A.2 diff small.

  Acceptance (f) — the regression lock — pins mode (A) via `account_prefix_residency=False`, not via the pass-through IdentityCodec store. This resolves the earlier draft's contradiction where the opening simultaneously claimed `store.resident_bytes()` was zero under IdentityCodec (wrong — it is honest fp16 bytes) and that `N_block > N_fp16` held (which requires mode (B) to charge fp16 bytes).

---

## 5. Codec-by-codec details

### 5.1 `silica.vq.turboquant.TurboQuantMSE`

Algorithm: Zandieh et al., arXiv 2504.19874, Algorithm 1. vqbench reference: `vqbench/vqbench/methods/turboquant/mse.py`.

Input: `(K, V)` fp16 tensors, shape `(1, n_kv_heads, kv_block_size, head_dim)`.

Per-vector (flattened to `(n_kv_heads × kv_block_size, head_dim)`):
1. `x_norm = mx.linalg.norm(x, axis=-1, keepdims=True)` — one fp16 scalar per vector.
2. `x_hat = x / max(x_norm, 1e-30)`.
3. `y = x_hat @ rotation.T` — Haar rotate. Matmul against the cached `(head_dim, head_dim)` constant.
4. `indices = mx.argmin((y[..., None] - centroids) ** 2, axis=-1).astype(mx.uint8)` (int in `[0, 2^num_bits)`). MLX lacks `mx.searchsorted`; broadcast-argmin over the `(..., 2^num_bits)` distance tensor is semantically equivalent to boundary lookup for Lloyd-Max-optimal codebooks because the boundaries are exactly the midpoints between adjacent centroids. `num_bits ≤ 8` is enforced at codec construction; extending beyond 8 bits per coordinate would overflow uint8 and require a uint16 payload refactor, but the P-5 codec catalog caps at 4 bits so this is never triggered in v0.1.
5. `packed_indices = pack_sub_byte(indices, num_bits=b)` — `silica.vq.core.packing`; produces `uint8` array of shape `(n, ceil(d × b / 8))`.
6. Payload: `BlockTQPayload` with `n_vq_blocks = 1` (i.e. `vq_block_size = head_dim` — one global scale per vector). There is no separate `TQPayload` dataclass; Scalar TQ is represented as the degenerate `B = d` case of the same `BlockTQPayload` schema. `scales` shape becomes `(n_vectors, 1)` fp16, `packed_indices` shape `(n_vectors, ceil(head_dim × num_bits / 8))` uint8.

Decode:
1. `y_hat = centroids[indices]` — lookup.
2. Optional norm correction: `y_hat /= mx.linalg.norm(y_hat, axis=-1, keepdims=True)`. Matches `turboquant_plus` production setting; default `norm_correction=True`.
3. `y_hat *= norm[:, None]`.
4. `x_hat = y_hat @ rotation` — inverse rotate (`rotation.T.T = rotation` since Haar is orthogonal).
5. Reshape back to `(1, n_kv_heads, kv_block_size, head_dim)`.

`TurboQuantMSE(head_dim=d, num_bits=b)` is equivalent to `BlockTurboQuantMSE(head_dim=d, vq_block_size=d, num_bits=b)` — the pin mentioned in §4.5. P-5-A.1 uses Block's code path with `vq_block_size = head_dim` rather than shipping two separate implementations; Scalar is aliased via `silica.vq.turboquant.TurboQuantMSE`.

### 5.2 `silica.vq.block_tq.BlockTurboQuantMSE`

Algorithm: vqbench `methods/turboquant/block_mse.py` + `BlockTQ.md`. Per-vq-block fp16 scales absorb outlier channels that Scalar's global norm cannot — the single most impactful structural change over the paper's Algorithm 1.

Encode (batched over `n_vectors`):

```python
y = x @ rotation.T                                                  # Haar rotate (fp32 matmul)
y_blocks = y.reshape(n, n_vq_blocks, vq_block_size)                 # (n, d/B, B)
scales = mx.linalg.norm(y_blocks, axis=-1, keepdims=True)           # (n, d/B, 1)
y_normed = y_blocks / mx.maximum(scales, 1e-30)
# MLX lacks mx.searchsorted — broadcast-and-argmin replaces the
# boundary lookup. Equivalent under Lloyd-Max-optimal codebooks
# because boundaries are midpoints between adjacent centroids.
diff = y_normed[..., None] - centroids                              # (n, d/B, B, 2^b)
indices = mx.argmin(diff * diff, axis=-1).astype(mx.uint8)          # (n, d/B, B)
indices_flat = indices.reshape(n, head_dim)                         # flat-last
packed_indices = pack_sub_byte(indices_flat, num_bits=b)            # (n, ceil(d × b / 8)) uint8
```

Payload (`BlockTQPayload`, §4.1):

- `packed_indices`: `(n_vectors, ceil(head_dim × num_bits / 8))` `uint8`. Under default `head_dim=256, num_bits=4` this is exactly `(n_vectors, 128)`.
- `scales`: `(n_vectors, n_vq_blocks)` fp16. `n_vq_blocks = head_dim / vq_block_size`, default `256 / 64 = 4`.
- `resident_bytes = packed_indices.nbytes + scales.nbytes`. By §4.3's honesty rule this is the **only** way to compute it. The arithmetic simplifies to `n_vectors × (ceil(head_dim × num_bits / 8) + 2 × n_vq_blocks)` bytes, matching vqbench BlockTQ.md §1's `(num_bits + 16 / vq_block_size)` effective-bits-per-dim claim. Unit test: `payload.resident_bytes == packed_indices.nbytes + scales.nbytes`.

Decode:

```python
indices_flat = unpack_sub_byte(packed_indices, num_bits=b, output_dim=head_dim)
indices = indices_flat.reshape(n, n_vq_blocks, vq_block_size)
y_blocks = centroids[indices]                                       # (n, d/B, B)
if norm_correction:
    y_blocks = y_blocks / mx.maximum(mx.linalg.norm(y_blocks, axis=-1, keepdims=True), 1e-30)
y_blocks = y_blocks * scales[..., None]
y = y_blocks.reshape(n, head_dim)
x_hat = y @ rotation
```

**Block-level codebook** — codebook dimension argument is `vq_block_size`, not `head_dim`. The rotated-then-unit-normed coordinates inside a B-dim block are approximately `N(0, 1/B)`, not `N(0, 1/d)`. Same Lloyd-Max algorithm, different σ. vqbench `BlockTQ.md` §3-§4 derives this.

Default `vq_block_size = 64`, `num_bits = 4`. Other configurations surfaced as constructor kwargs, parametrized by bench rows.

### 5.3 `silica.vq.rabitq.RaBitQ1Bit`

Algorithm: Gao & Long, arXiv 2405.12497, Algorithm 1. vqbench reference: `methods/rabitq/rabitq_1bit.py`.

1. `fit(X)` once at codec construction — compute dataset centroid. In silica's streaming setting, centroid is either provided (calibration data) or zero (random-unit-vector assumption). P-5-B starts with zero centroid; `fit` is a no-op MLX-native wrapper that exists for interface parity with vqbench.
2. Encode:
```
o = x - centroid                                                    # (d,)
norm_o = mx.linalg.norm(o)                                          # scalar
o_hat = o / max(norm_o, 1e-30)                                      # unit-norm
y = rotation @ o_hat                                                # Haar rotate
signs = mx.sign(y).astype(mx.int8)                                  # (d,) in {-1, +1}
x_bar = signs / math.sqrt(d)                                        # unit hypercube vertex
ip_coeff = (x_bar * y).sum()                                        # scalar
```

Payload (`RaBitQPayload`, §4.1; `num_bits=1` here):

- `packed_indices`: `(n_vectors, d // 8)` `uint8`. For 1-bit, `packed_indices` holds the sign bits only (`1 bit → 1 entry`); multi-bit ExtRaBitQ reuses the same field with `ceil(d × B / 8)` width. Packing path: `((signs + 1) // 2).astype(mx.uint8)` maps `{-1, +1} → {0, 1}`, then `pack_sub_byte(..., num_bits=1)` folds 8 coordinates into each `uint8` via `mx.left_shift` + `mx.bitwise_or`. MLX exposes the primitives (`mx.bitwise_or`, `mx.bitwise_and`, `mx.left_shift`, `mx.right_shift`, `mx.bitwise_xor`, `mx.bitwise_invert`) — probed present on silica's current MLX version.
- `norm_o`: `(n_vectors,)` fp16.
- `ip_coeff`: `(n_vectors,)` fp16.
- `centroid` lives on the codec instance, not the payload (fit-once).
- `resident_bytes = packed_indices.nbytes + norm_o.nbytes + ip_coeff.nbytes` per §4.3. Arithmetic for 1-bit: `n_vectors × (d / 8 + 4)` bytes — same value the earlier draft claimed, but now computed from the actual array `.nbytes`, not from hand arithmetic.

Decode: unpack signs → `x_bar = signs / sqrt(d)` → inverse rotate → scale by `norm_o` → add centroid. Accuracy is deliberately low (hypercube vertices); the point of RaBitQ 1-bit in P-5 is **interface parity with vqbench**, not production use. vqbench REPORT §3.1 shows it costs +13.3% PPL; P-5-B lands it as a completeness / low-bit-point reference, gated behind its own bench row.

### 5.4 `silica.vq.rabitq.ExtRaBitQ`

Algorithm: Gao et al., arXiv 2409.09913, §3. Bit-plane decomposition.

Same preprocessing as 1-bit (centroid, normalize, rotate). Quantizer is an integer-grid codebook `{-(2^B - 1), ..., -1, 1, ..., 2^B - 1}` (odd integers centred at zero). Storage is B bits per coordinate plus the metadata header.

Payload (`ExtRaBitQPayload`, subclass of `RaBitQPayload` with an added per-vector `scale` field):

- `packed_indices`: `(n_vectors, ceil(d × B / 8))` `uint8`. At `d=256, B=4` this is `(n_vectors, 128)`; at `B=2` it is `(n_vectors, 64)`. Same `pack_sub_byte(indices, num_bits=B)` helper used by BlockTQ — one bit-packing implementation shared across all sub-byte codecs.
- `norm_o`, `ip_coeff`, `scale`: `(n_vectors,)` fp16 each. Total three fp16 per vector, = 6 bytes. `scale` is the per-vector **dequantization** scale (inverse of the quantization scale factor); decode multiplies integer codebook values by this to recover the rotated-coordinate range.
- `resident_bytes = packed_indices.nbytes + norm_o.nbytes + ip_coeff.nbytes + scale.nbytes` per §4.3. Computed from actual `.nbytes`, not hand arithmetic; a codec claiming `d × B / 8 + 6` while storing `int8` indices would fail the §4.3 fairness test before shipping.

**Amendment (P-5-B.2a):** the earlier draft of this section listed four fp16 fields (`norm_o`, `ip_coeff`, `scale`, `offset`). silica's implementation drops `offset` — vqbench's `offset` is `0.0` in every batch and single-vector code path, and its decode does not read the field. Re-adding it in a future variant that actually uses a non-zero affine shift would be an explicit schema extension; paying 2 bytes / vector for a permanent zero is rejected. Effective bits per coordinate is therefore `num_bits + 48 / head_dim` (3 × 16 = 48, not the 4 × 16 = 64 the pre-amendment draft implied).

vqbench REPORT §3.1 shows ExtRaBitQ at 4-bit K+V is seed-dependent (+0.262% ± 0.371%); at 3-bit K+V it is the worst of the three families (+3.73%); at 4-bit K-only it is lossless like the others. P-5-B ships it for bench parity, not as a production recommendation.

Centroid handling mirrors `RaBitQ1Bit` (§5.3 convention): the codec holds an explicit `_centroid = mx.zeros(head_dim, fp32)` field, encode subtracts and decode adds even though the value is a no-op, and `fit(X)` is a `pass`. Data-driven centroid fits are a v0.2 extension path, not a v0.1 behaviour; pinning centroid at zero in v0.1 matches vqbench's "no centroid subtraction" baseline row and keeps ExtRaBitQ's zero-vector semantics aligned with `RaBitQ1Bit`.

### 5.5 Method-reference map

| silica module                   | vqbench file                                                                 | Role                                        |
| ------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------- |
| `silica.vq.turboquant`          | `vqbench/vqbench/methods/turboquant/mse.py`                                  | Algorithm 1 scalar rewrite                  |
| `silica.vq.block_tq`            | `vqbench/vqbench/methods/turboquant/block_mse.py`                            | BlockTurboQuantMSE rewrite                  |
| `silica.vq.rabitq.rabitq_1bit`  | `vqbench/vqbench/methods/rabitq/rabitq_1bit.py`                              | 1-bit hypercube rewrite                     |
| `silica.vq.rabitq.rabitq_ext`   | `vqbench/vqbench/methods/rabitq/rabitq_ext.py`                               | Multi-bit bit-plane rewrite                 |
| `silica.vq._calibration`        | `vqbench/vqbench/core/rotation.py`, `methods/turboquant/codebook.py`         | Haar + Lloyd-Max; NumPy quarantine          |
| `silica.vq.core.packing`        | `vqbench/vqbench/core/packing.py`                                            | Bit-pack / unpack helpers (MLX-native)      |
| `silica.bench.ppl_oracle`       | `vqbench/vqbench/validation/streaming_ppl.py`                                | WikiText-2 streaming PPL, chunk=256         |
| `silica.bench.vqbench_baseline` | `vqbench/scripts/reproduce_qwen35_4b_headline.py` (delegate target)          | Independent cross-check subprocess         |

---

## 6. Reproducibility — bench harness extensions

### 6.1 `--kv-codec` CLI flag + `CodecSpec` registry

The registry is a dict of `CodecSpec` metadata, not bare factory lambdas. Every entry carries enough structured metadata that downstream tools (bench tables, admission logging, `--all-kv-codecs`) do not need per-codec special cases:

```python
# silica/bench/codec_registry.py

@dataclass(frozen=True)
class CodecSpec:
    id: str                              # registry key, e.g. "block_tq_b64_b4"
    family: str                          # "fp16" | "tq_mse" | "block_tq" | "rabitq" | "ext_rabitq"
    bits_per_value: float                # nominal bits/dim, e.g. 4.25 for Block B=64 b=4 (metadata-inclusive)
    k_supported: bool                    # may be used as k_codec
    v_supported: bool                    # may be used as v_codec
    requires_fit: bool                   # needs fit(X) calibration call before use (RaBitQ)
    payload_packed: bool                 # sub-byte bit-packed (False for fp16 baseline)
    production_recommended: bool         # vqbench-recommended production setting
    factory: Callable[..., VectorCodec]  # constructed with head_dim=... bound by BenchRunner

# CODEC_REGISTRY (public name as landed in P-5-A.1b). The CodecSpec
# dataclass also exposes ``effective_bits_per_value(head_dim)`` for the
# head-dim-dependent scalar TQ case.
CODEC_REGISTRY: dict[str, CodecSpec] = {
    "fp16": CodecSpec(
        id="fp16", family="fp16", bits_per_value=16.0,
        k_supported=True, v_supported=True,
        requires_fit=False, payload_packed=False,
        production_recommended=False,
        factory=_identity_factory,                                 # IdentityCodec
    ),
    "block_tq_b64_b4": CodecSpec(
        id="block_tq_b64_b4", family="block_tq", bits_per_value=4.25,
        k_supported=True, v_supported=True,
        requires_fit=False, payload_packed=True,
        production_recommended=True,                                # vqbench REPORT §3.1
        factory=lambda head_dim, **kw: BlockTurboQuantMSE(
            head_dim=head_dim, vq_block_size=64, num_bits=4),
    ),
    # ... tq_mse_b4, tq_mse_b3, block_tq_{b64,b32}_{b3,b4}, rabitq_1bit, ext_rabitq_{b2,b3,b4}
}
```

`scripts/bench.py --kv-codec block_tq_b64_b4 --scenario qwen3-0.6b-wikitext-ppl` threads the registry lookup through `BenchRunner.engine_factory`: the runner resolves `spec.factory(head_dim=adapter.kv_layout().head_dim)` once, checks `spec.k_supported` / `spec.v_supported` against the bench row's K-only vs K+V request, and installs the resulting `VectorCodec` on the `RadixPrefixCache.store` via `k_codec` / `v_codec`. For `--all-kv-codecs --k-only`, the runner iterates entries with `spec.k_supported=True` and constructs pairs `(codec, None)`.

Adding a new VQ method is one `CodecSpec` entry and one concrete `VectorCodec[P]` class — no bench-table code changes, no CLI flag plumbing, no per-codec conditionals in reporting. This is the "可以持续接入更多 VQ 方法的平台" user intent made operational at the registry layer.

### 6.2 `silica.bench.ppl_oracle` — MLX-native streaming PPL

vqbench's `validation/streaming_ppl.py` faithfully computes chunked PPL: for each chunk, current tokens use exact K/V, past chunks are served from compressed `VQBenchCache`. silica's MLX-native port uses the same semantics but **cannot go through `Engine.generate_batch`**: that entry point is a **sampling** API — it emits `BatchEvent(kind="token", token_id=...)` stream via `Sampler`, and never returns the positional logits tensor that per-token cross-entropy requires. Even `temperature=0` greedy returns the argmax token id, not the full vocab distribution. PPL computation needs teacher-forced positional logits for every token position in the chunk, not sampled tokens.

The P-5-C split:

- **P-5-C.1 landed** — a cache-agnostic fp16 baseline oracle plus the `forward_batched_full` runner entry point. No codec path; the oracle drives whatever `BatchKVCache` the adapter hands out and mutates it in place across chunks (the cache-sharing semantics is what makes PPL chunk-size invariant).
- **P-5-C.2 (next)** — a codec-backed oracle arm that seeds each chunk's cache from a `RadixPrefixCache` whose store carries a `VectorCodec`. This is the variant that actually exercises `encode_tensor` / `decode_tensor`. The C.1 oracle does **not** go through the codec; any PPL-delta claim against a VQ codec must route through the C.2 arm.

#### 6.2.1 P-5-C.1 entry point (landed)

```python
# silica/bench/ppl_oracle.py

def teacher_forced_chunked_nll(
    adapter: Any,
    token_ids: mx.array,            # (1, seq_len) token ids
    *,
    chunk_size: int = 256,
) -> tuple[float, int]:
    """Cumulative teacher-forced NLL + scored-token count.

    Allocates one ``BatchKVCache`` list via ``adapter.make_batch_cache([0])``
    and reuses it across every chunk — mlx-lm mutates the cache in place,
    so prior chunks' K/V remain available when the next chunk runs. This
    is what makes the returned NLL chunk-size invariant; the unit test
    regression-locks the same property vqbench's ``streaming_ppl`` module
    pins (``test_ppl_oracle.py::TestChunkInvariance``).

    Within each chunk the standard shift-by-1 CE
    (``logits[:, :-1, :]`` vs ``tokens[:, 1:]``) scores all-but-first
    positions. The chunk-boundary token is scored against the previous
    chunk's final logit (``prev_last_logit`` is materialized via
    ``mx.contiguous`` + ``mx.eval`` before the next forward mutates the
    shared cache).
    """


def perplexity_from_nll(nll_sum: float, n_tokens: int) -> float:
    """exp(nll_sum / n_tokens). n_tokens == 0 → inf (aggregation-friendly);
    n_tokens < 0 raises ValueError (caller-side accounting bug)."""
```

`adapter._model` / `adapter.make_batch_cache([0])` are the scheduler-convention attribute and method used by `silica.scheduler.batcher`; formalizing them as a Protocol is v0.2 scope. Token IDs come in as `(1, seq_len)` — streaming PPL is a per-sequence quantity; caller aggregates across sequences via NLL sum + token count before taking `exp`.

The sibling runner entry point `silica.mlx.runner.forward_batched_full(model, tokens, cache_list) -> (B, T, V)` landed with C.1; the existing last-position-only `forward_batched` is now a thin wrapper (`forward_batched_full(...)[:, -1, :]`) so sampling and teacher-forcing cannot drift. See `test_runner_forward_batched.py::test_forward_batched_is_last_position_of_full` for the bit-identical equality lock.

#### 6.2.2 P-5-C.2 entry point (planned)

The codec-backed variant seeds a `RadixPrefixCache` whose store owns a `VectorCodec`. Each chunk ≥ 1 fires `codec.decode_tensor` on the aligned prefix and `codec.encode_tensor` on the newly-grown tail, so the codec hot path is fully exercised. Target signature (subject to C.2 design finalization):

```python
def teacher_forced_chunked_nll_with_codec(
    adapter: Any,
    prefix_cache: RadixPrefixCache,   # store carries k_codec / v_codec
    token_ids: mx.array,
    *,
    chunk_size: int = 256,
) -> tuple[float, int]:
    ...
```

C.2 tests route through a counting codec / store stand-in that asserts both `encode_tensor` and `decode_tensor` are actually invoked — a quantized PPL row that silently equals fp16 because no encode/decode fired is the specific failure mode the counting harness is there to catch.

#### 6.2.3 Shared design notes

- **Chunk size 256**, matches vqbench.
- **Manual per-token CE with explicit boundary handling:** HF's `shift_labels` silently drops one token per chunk boundary and breaks chunk invariance; silica's port mirrors vqbench's explicit boundary loop. C.1 regression-locks this via `test_ppl_oracle.py` chunk invariance across `{8, 16, 32, 64, 128}` at `rel=1e-5, abs=1e-5` against a full-sequence NumPy-style reference, the same shape vqbench's `tests/test_streaming_ppl.py::test_baseline_chunk_invariance` does.
- **WikiText-2 first N tokens** — C.2 bench-row parameter; default matches vqbench headline (512 tokens ≈ 2 chunks). The C.1 oracle itself is text-free — token IDs only, so unit tests run without a tokenizer or dataset dependency.
- **Not through `Engine.generate_batch`** — neither C.1 nor C.2 routes through the sampler or scheduler. This deliberately avoids Q-012's initial-cohort-no-prefix-lookup limitation because the oracle drives admission manually per chunk (C.2) or runs straight on the adapter's cache (C.1).

C.2 output schema: `{ppl_fp16: float, ppl_quant: float, delta_ppl: float, delta_pct: float}` — same as `silica/bench/vqbench_baseline.py` so the two columns align in the bench table.

### 6.3 Variance / multi-seed runs

vqbench REPORT §3.1 uses `run_seed ∈ {42, 43, 44}`. silica bench runner gains `--seed` / `--seeds` flags; `--seeds 42,43,44` runs 3-seed sweeps and reports `mean ± std` per cell. Output JSONL has one row per (scenario, codec, seed).

### 6.4 Cross-check via `silica/bench/vqbench_baseline.py`

Already shipped at P-4.4. Under P-5 it becomes the **independent** column: `scripts/bench.py --kv-codec block_tq_b64_b4 --vqbench-xcheck` runs both the silica-native PPL oracle AND a vqbench subprocess with matching config; result row shows both numbers side-by-side. The per-row thresholds `vqbench_epsilon = 0.01` / `_VQBENCH_PCT_EPSILON = 0.1` in `silica/bench/runner.py::_compute_gap_fields` emit the `vqbench_divergence_warning` boolean as a **diagnostic** metadata field on every row — they do not define P-5 acceptance. The P-5 §7(b) close criterion is the mean-over-seeds aggregated gate on the `-vqbench-aligned` row defined in §7(b); per-row warnings are expected under D.2a shared- vs per-head-rotation sampling and are not close blockers.

### 6.5 Bench catalog rows P-5 adds

| id                                 | codec                 | oracle       | purpose                                      |
| ---------------------------------- | --------------------- | ------------ | -------------------------------------------- |
| `qwen3-0.6b-wikitext-ppl-fp16`              | `fp16`            | PPL          | C.2 baseline; C.1 oracle drives the adapter's fp16 cache |
| `qwen3-0.6b-wikitext-ppl-tq-mse-b4`         | `tq_mse_b4`       | PPL          | Scalar 4-bit K+V                             |
| `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`   | `block_tq_b64_b4` | PPL          | Production recommendation                    |
| `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4`     | `ext_rabitq_b4`   | PPL          | RaBitQ family                                |
| `qwen3-0.6b-compression`                    | cross-codec       | STORAGE      | One row per codec, reports `resident_bytes`  |
| `qwen3-0.6b-prefix-hit-decode-fp16` | `fp16`                | DECODE_TOK_S_WITH_PREFIX_HIT | Acceptance gate (d) baseline; `[p, p] max_batch_size=1 prefix_cache=True`. Qwen3-0.6B rather than Qwen3.5-0.8B — the latter is hybrid-DeltaNet, `ContinuousBatcher` refuses `RadixPrefixCache` on `has_recurrent_state=True` adapters (docs/P3_DELTANET_SURVEY.md C-open-3). Same workload shape; the ratio gate is codec-relative so the target swap is neutral |
| `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` | `block_tq_b64_b4` | DECODE_TOK_S_WITH_PREFIX_HIT | P-5-A.3c acceptance gate (d) BlockTQ arm; same workload shape as the fp16 row |
| `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` | `ext_rabitq_b4` | DECODE_TOK_S_WITH_PREFIX_HIT | P-5-B.3 acceptance gate ExtRaBitQ arm; same workload shape, same 0.85× threshold as the BlockTQ arm. `rabitq_b1` is deliberately excluded (K-only, cannot install via symmetric `kv_codec=` shorthand) |

`qwen3.5-4B` rows follow once compute budget allows — vqbench headline is on 4B, not 0.8B. The `STORAGE` oracle is new; reports pure memory without requiring PPL compute; useful for `qwen3.5-27B` where PPL compute is expensive.

---

## 7. Acceptance

Six clauses. (a)–(c) carry forward from PLAN §7 P-5 Acceptance (with (b) and (c) rewritten in this opening — see below), (d) is new, (e) is the opening's "reproducibility is the point" anchor, (f) is the regression lock.

### (a) Per-block reconstruction error vs vqbench — split in two (2026-04-22)

Acceptance (a) splits into two halves that landed independently:

- **(a-algo) algorithmic parity half** — landed at P-5-A.1c in
  `tests/test_block_tq_vqbench_xcheck.py`. The MLX hot path
  is a self-contained unit that deserves its own parity gate on
  every developer machine, and the gate runs on synthetic
  Gaussian inputs against a vqbench-transcribed NumPy reference
  (`_numpy_block_tq_round_trip`).
- **(a-real) real-activation half** — closed at v1.7.5 via
  `tests/test_block_tq_real_activation_xcheck.py`, which reuses
  the (a-algo) NumPy reference on real Qwen3.5-0.8B pre-RoPE K /
  V tensors extracted from a prefill pass. The v1.5.1-era
  "vqbench venv subprocess + recon-specific driver script" design
  was superseded by the inline-NumPy idiom (memory: "vqbench is
  reference-only — transcribe inline"); the transcription's
  faithfulness was already pinned at (a-algo), so real-activation
  reuses the same reference without new subprocess infrastructure
  or a second skip gate. Full design contract and per-decision
  rationale: `docs/P5_A_REAL_OPENING.md`. References to a
  "P-5-C subprocess landing" or a "vqbench venv" below this
  paragraph describe the superseded design only — the current
  landed design is inline NumPy, single HF-cache gate.

**(a-algo) — P-5-A.1c — algorithmic parity vs in-test NumPy reference.**

Metric: per-block relative Frobenius error
`||silica_MLX_out - numpy_reference_out||_F / ||numpy_reference_out||_F`,
computed on a fixed synthetic Gaussian input. The NumPy reference
reimplements vqbench's `block_mse.py` encode/decode verbatim but
uses only `silica.vq._calibration.haar_rotation` +
`lloyd_max_codebook` as shared inputs (those helpers are verbatim
NumPy ports of vqbench's calibration, already paper-reference-pinned
at `tests/test_calibration.py::test_lloyd_max_matches_paper_reference`).
All hot-path arithmetic — matmul, block split, scale extraction,
quantize, centroid lookup, norm correction, inverse rotate — is
independent of silica's `BlockTurboQuantMSE`.

Gate: per-block relative Frobenius error `< 5e-3` on the
``{vq_block_size, num_bits} ∈ {32, 64} × {3, 4}`` grid; `< 1e-3` on
the production-recommended `B=64 b=4` row (regression lock). Observed
values measured 2026-04-22 on `(1, 4, 16, 128)` Gaussian at seed 42:
BlockTQ paths ≈ 2.1e-4, scalar (`B=d=64`) ≈ 1.2e-3. Tolerance headroom
catches reshape / rotation-direction / argmin-vs-searchsorted
tie-break drift while absorbing seed variation.

Test: `tests/test_block_tq_vqbench_xcheck.py` (landed P-5-A.1c).
Runs on every developer machine; no vqbench, scipy, HF cache, or
model-load dependency. This is the primary guardrail that the MLX
translation of vqbench's BlockTQ algorithm is numerically faithful.

**(a-real) — closed at v1.7.5 — real Qwen3.5-0.8B K / V activations vs inline NumPy reference.**

Closed by P-5 (a-real) at v1.7.5 via
`tests/test_block_tq_real_activation_xcheck.py`. Evidence:
`docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.md` + `.jsonl`.
Design contract: `docs/P5_A_REAL_OPENING.md`.

Metric: per-block relative Frobenius error on real K / V
activations extracted from a Qwen3.5-0.8B prefill pass (GLOBAL
layers only — `layer.is_linear == False` — pre-RoPE
`k_proj` / `v_proj` output, the same vqbench-aligned space that
Acceptance (4-b) and D.2a inject noise into). The reference
side runs the vqbench-transcribed NumPy implementation landed
at P-5-A.1c (`tests/test_block_tq_vqbench_xcheck.py::_numpy_block_tq_round_trip`),
called with the same shared calibration helpers silica's codec
uses (`silica.vq._calibration.{haar_rotation, lloyd_max_codebook}`).
No vqbench subprocess: the reference is inline NumPy per the
established xcheck idiom (memory: "vqbench is reference-only —
transcribe inline"), and the transcription's faithfulness was
pinned at (a-algo) to `5e-3` / `1e-3` on synthetic Gaussian. The
v1.5.1-era "vqbench venv subprocess + new recon-specific driver
script" design has been superseded — see `docs/P5_A_REAL_OPENING.md`
§2.3 for the rationale.

Gate: silica-vs-NumPy absolute Frobenius-ratio gap must satisfy
`|silica_frob - numpy_frob| < 1e-3` on the production-recommended
`(B=64, num_bits=4)` cell and `< 5e-3` on the other
`(B, bits) ∈ {32, 64} × {3, 4}` cells — the same envelope
(a-algo) pinned for synthetic Gaussian, reused here without
tightening because the transcription is identical. The v1.7.2
"`ε_recon < 2 × fp16 round-trip baseline`" ratio is kept as a
secondary gate; the `IdentityCodec` `encode_tensor`→`decode_tensor`
round-trip baseline is dtype-preserving (`RawFp16Payload` stores
the `mx.array` directly), so the ratio degenerates on every row
and the absolute-gap gate above carries the close weight. The
test asserts baseline degeneracy on every row so a future
`IdentityCodec` change that breaks dtype preservation surfaces
loudly.

Evidence (144 rows — 6 GLOBAL layers × 2 sides × 4 (B, bits) × 3 seeds):

- Worst-case `|silica - numpy|`: `1.15e-4` (layer 15, K, B=64 b=3)
  — ~43× headroom under `5e-3`.
- Worst-case production cell `(B=64, b=4)`: `5.21e-5` (V side) —
  ~19× headroom under `1e-3`.
- K / V symmetric (worst K `1.15e-4`, worst V `9.78e-5`); all 6
  GLOBAL layers land in the same `9e-5 … 1.2e-4` band.
- Baseline degenerate on 144 / 144 rows as predicted.

Gating: **single gate** — HF cache has Qwen3.5-0.8B. No
`VQBENCH_PYTHON_EXECUTABLE` gate (the reference is inline NumPy).
The prompt is a checked-in deterministic English passage inside
`tests/test_block_tq_real_activation_xcheck.py` (~138 tokens under
the Qwen3.5 tokenizer, envelope-guarded at 96..160 for
tokenizer-drift catch).

### (b) End-to-end PPL agreement on the vqbench-aligned oracle — mean-over-seeds cross-check

Compare **deltas**, not absolute PPL. silica's MLX fp16 path and vqbench's NumPy/PyTorch fp16 path produce slightly different baseline PPL numbers (fp16 roundoff, different accumulation order, MPS vs MLX kernels); that baseline drift can exceed `0.01` absolute PPL even with IdentityCodec. Comparing `(silica.PPL_quant - silica.PPL_fp16)` against `(vqbench.PPL_quant - vqbench.PPL_fp16)` cancels the cross-implementation baseline term and leaves only the codec-induced signal. Same pattern vqbench REPORT §3.3 uses to compare against `turboquant_plus`.

Bind target: the **`qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` bench row** (Qwen3-0.6B, `BlockTurboQuantMSE B=64` 4-bit K+V, `codec_quality_path="vqbench_aligned"`), not the `-prefix_store_post_rope` production-routing arm. The vqbench-aligned row runs silica's D.2a pre-RoPE projection-patch oracle (`teacher_forced_chunked_nll_vqbench_aligned` in `silica/bench/ppl_oracle.py`, landed at P-5-D.2a), which injects reconstruction noise in the **same space** vqbench's `_QuantizedProj` harness does (pre-RoPE projection space). The production `prefix_store_post_rope` arm injects noise in post-RoPE space and pays an additional chunk-boundary cost at the same Frobenius reconstruction error; (b) does not bind to that arm — see (b-postrope) below.

Gate (mean-over-seeds aggregated form, both conditions required):

```
| mean_gap |  <=  2 * SEM_diff        (agreement within sampling noise)
| mean_gap |  <   1.0 PPL             (sanity cap, absolute)

mean_gap  = mean_seeds( silica.ΔPPL_seed − vqbench.ΔPPL_seed )
SEM_diff  = sqrt( std(silica.ΔPPL_seeds)^2 / n + std(vqbench.ΔPPL_seeds)^2 / n )
n         = 3  (seeds {42, 43, 44})
std       = Bessel-corrected sample std, divisor (n-1)
```

This is the standard error of the difference of two independent sample means and is the right quantity for deciding whether the two sides' mean ΔPPL estimates are consistent at `n=3`. `ΔPPL_seed = PPL_quant − PPL_fp16` is measured inside each oracle on the same calibration set (same three seeds, same WikiText-2 prefix, same chunk size, same codec config). Verified via `PYTHONPATH=vqbench scripts/bench.py --scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned --seeds 42,43,44 --vqbench-xcheck --vqbench-python <vqbench-venv-python> --out <path.jsonl>`.

**Per-row diagnostics preserved, not gate.** The per-row thresholds `vqbench_epsilon = 0.01` / `_VQBENCH_PCT_EPSILON = 0.1` in `silica/bench/runner.py::_compute_gap_fields` are **unchanged** and continue to emit the `vqbench_divergence_warning` boolean as a metadata field on every row. Under the D.2a 3-seed data, per-row `vqbench_divergence_warning=true` still fires at worst-case `|gap| ≈ 0.61` PPL because silica's `BlockTurboQuantMSE` shares one Haar rotation across all heads (ctor-seeded) while vqbench's `quant_dequant_tensor` samples one rotation per head (seed=h), so at the same outer seed the two sides draw different rotations from the same Haar distribution. The per-row warning is an expected sampling artefact and does **not** block (4-b) close; it remains useful as a diagnostic for spotting genuine algorithmic regression.

**Evidence (2026-04-24, landed at commit `ed57be1`; raw `docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`):**

| metric                         | value                |
|--------------------------------|----------------------|
| silica mean ΔPPL               | `+0.511`             |
| silica std (Bessel, n−1)       | `0.354`              |
| vqbench mean ΔPPL              | `+0.661`             |
| vqbench std (Bessel, n−1)      | `0.347`              |
| `mean_gap` (silica − vqbench)  | `−0.150` PPL         |
| `SEM_diff` (independent, n=3)  | `0.286`              |
| `2 * SEM_diff`                 | `0.572`              |
| gate: `\|mean_gap\| ≤ 2·SEM_diff`| `0.150 ≤ 0.572` — **pass** |
| gate: `\|mean_gap\| < 1.0 PPL`   | `0.150 < 1.0` — **pass** |

Both gate conditions pass. (4-b) closes at PLAN v1.7.3.

### (b-postrope) Production `prefix_store_post_rope` prefix-cache quality cost — closed at P-5-F (v1.7.6)

At the same `BlockTurboQuantMSE B=64` 4-bit K+V codec config, the production-routing arm (`codec_quality_path="prefix_store_post_rope"`, scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`, the C.2 post-RoPE prefix-cache store path) measured a ΔPPL in the ~5–10 PPL range on Qwen3-0.6B WikiText-2 pre-P-5-F — chunk-boundary-dependent: at `chunk_size=512` with the sequence fitting in one chunk the gap collapsed toward fp16, at `chunk_size=256` it jumped to several PPL, and continued to grow with the number of prefix-cache refills. This was a real **production-path quality cost**, not an algorithmic error. D.2a probes (`docs/P5_D2_INVESTIGATION/p5_d2_probe*.py`) confirmed silica MLX BlockTQ and vqbench NumPy BlockTQ produce bit-identical Frobenius reconstruction on the same K input, and silica's prefix-cache round-trip was numerically neutral; the cost arose because silica's prefix-cache store injected noise in **post-RoPE** space while attention's relative-position coupling sees that noise through the RoPE rotation rather than alongside the signal (`docs/P5_D2_INVESTIGATION/README.md` §Root cause).

**P-5-F closes this gap (v1.7.6).** F.1-F.4 land a pre-norm K/V store via the (3b) projection-output capture path: `silica/models/pre_norm_capture.py` ships a runtime-checkable `PreNormCaptureAdapter` Protocol (`install_pre_norm_capture` arming + `apply_k_norm_then_rope` reconstruction) plus a per-family proxy on `attn.k_proj` installed at adapter construction; the proxy returns `k_proj(x)` unchanged and side-effects pre-k_norm K to a buffer. `SyntheticPrefixBlockStore` accepts a `pre_norm: bool = False` contract flag; `ContinuousBatcher` arms / disarms the buffer per prefill chunk forward, sources K from a per-row `_BatchRow.k_pre_per_block` dict in `_extract_and_insert_prefix`, and calls `apply_k_norm_then_rope` per block per attn-pos in `_admit_single_hit_row` before `build_seeded_batch_kv`. F.3 flips the production wikitext PPL default to `codec_quality_path="prefix_store_pre_norm"`. The (4-b) anchor row now measures **ΔPPL +0.012** on the production path (single seed; F.0b' 3-seed verification is +0.015), inside D.2a's `+0.51 ± 0.35 PPL` envelope. ~360× improvement over the legacy +20.83 PPL post-RoPE store. Three legacy comparison arms (`prefix_store_post_rope`, `prefix_store_pre_rope`, `vqbench_aligned`) retained as bench-only opt-ins per `docs/P5_F_OPENING.md` §6.9 reading order. The (4-b) `[x]` continues to record algorithmic parity between silica MLX and vqbench NumPy BlockTQ; the production-routing arm is now structurally aligned with that algorithmic parity.

### (b-static) End-to-end PPL vs `vqbench/REPORT.md` static baseline — closed at v1.7.7

silica's MLX-native `BlockTurboQuantMSE B=64` 4-bit K+V codec on Qwen3.5-4B reproduces vqbench/REPORT.md §3.1's lossless-at-measurement-precision finding under independent measurement: at the same workload (WikiText-2 first 512 tokens, `chunk_size=256`, 3 seeds `{42, 43, 44}`), silica's mean ΔPPL is statistically indistinguishable from vqbench's reported `+0.000% ± 0.000%`.

**Setup.** Bench command `scripts/bench.py --scenario qwen3.5-4b-wikitext-ppl-{fp16,block-tq-b64-b4} --seeds 42,43,44`. The `block_tq_b64_b4` row routes through `prefix_store_pre_norm` by default after P-5-F F.3 (commit `df2b3f4`); no flag override needed. Hybrid Qwen3.5-4B's slice-regime is active (`RecurrentStateAdapter + prefix_cache != None`); the slice-prefill helpers exercise F.2b's per-chunk K_pre capture (commit `cc249e7`).

**Per-seed silica numbers** (raw JSONL at `docs/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_3seeds.jsonl`):

| Seed | fp16 PPL | codec PPL | ΔPPL    | ΔPPL %  |
| --- | --- | --- | --- | --- |
| 42 | 8.855979 | 8.880764 | +0.024785 | +0.280% |
| 43 | 8.855979 | 8.839237 | −0.016742 | −0.189% |
| 44 | 8.855979 | 8.852605 | −0.003374 | −0.038% |

silica aggregated: mean ΔPPL = `+0.001556 PPL`, std = `0.021198`, SEM = `0.012238`.

vqbench/REPORT.md §3.1 row "Block B=64 4-bit K+V": mean ΔPPL_pct = `+0.000%`, std = `0.000%` (3 seeds {42, 43, 44}, monkey-patch fallback on the 8 full-attention layers per REPORT §2.2).

**Gate.** The (4-b)-style two-part aggregated gate inherited from PLAN §7:

- `mean_gap` = silica.ΔPPL_mean − vqbench.ΔPPL_mean = `+0.001556 PPL`.
- `SEM_diff` = √(SEM_silica² + SEM_vqbench²) = `0.012238 PPL` (vqbench's static row reports 0.000% std, contributes 0).
- Gate (i): `|mean_gap| ≤ 2 × SEM_diff` ⇔ `0.001556 ≤ 0.024477` → **PASS** (~16× headroom).
- Gate (ii): `|mean_gap| < 1.0 PPL` ⇔ `0.001556 < 1.0` → **PASS** (~640× headroom).

**Production hot-path measurement, not monkey-patch.** v1.7.5's earlier wording of (b-static) recorded "blocked on P-3-C recurrent + prefix-cache cooperation work, or on an alternate monkey-patch measurement path wrapping `k_proj` / `v_proj` on the 8 full-attention layers". That dependency was correct at v1.7.5 — `ContinuousBatcher` refused to pair `RadixPrefixCache` with recurrent adapters until P-3-C5.4 lifted the C3b guard, and pre-P-5-F the post-RoPE prefix-cache store paid a chunk-boundary quality cost on the production path. Both blockers are resolved at v1.7.6: P-5-F's (3b) projection-output capture path lands the production hot path inside D.2a's quality envelope on Qwen3.5-4B, so the monkey-patch fallback is no longer needed. (b-static) closes via the same routing the production deployment will use, not via a measurement-only patch.

**Why silica's absolute fp16 PPL differs from vqbench's.** silica's fp16 PPL = `8.856`; vqbench's REPORT.md reports `10.3866`. The difference traces to tokeniser (silica uses Qwen3.5 mlx-lm tokenizer; vqbench uses HF transformers tokenizer with different boundary-handling), precision (silica MLX bf16; vqbench torch.float16 + MPS — different matmul accumulation), and harness (silica's `teacher_forced_chunked_nll` vs vqbench's `evaluate_perplexity`). The (b-static) gate compares **ΔPPL** (relative to each respective fp16 baseline), which is the harness-independent codec-quality observable.

**Evidence files.** `docs/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_close.md` (full analysis); `qwen35_4b_b_static_3seeds.jsonl` (raw bench rows); `qwen35_4b_b_static_3seeds.md` (auto-generated bench report).

### (c) Prefix residency enters `headroom_bytes()`, admission benefits follow

Rewritten from PLAN §7 P-5 Acceptance's original "admits more requests under the same memory budget" wording. The correct mechanism (per §4.7) is:

- `MemoryBudgeter.headroom_bytes()` subtracts `store.resident_bytes()` in addition to `weights_bytes` and `reserved_bytes()`.
- Under a codec, `store.resident_bytes()` is smaller than it would be under IdentityCodec by roughly the codec's compression ratio on prefix-cache K/V.
- Smaller prefix residency → larger headroom → `new_worst ≤ headroom` fits step (1) of `admit()` for more incoming requests → more Admit decisions, fewer Reject / evict-path decisions.

Gate, verifiable on a new bench row `qwen3-0.6b-admission-headroom-prefix-heavy` (§6.5). Both rows run with `account_prefix_residency=True` (§4.7 mode (B) vs mode (C)) so the residency numbers are directly comparable; mode (A) is the separate regression lock, not a leg of this comparison:

1. Warmup the prefix cache via a shared-prefix cohort until `store.resident_bytes() ≥ 0.5 × cap_bytes` under `SyntheticPrefixBlockStore(codec=IdentityCodec(...))` (§4.7 mode (B); fp16 residency honestly charged). Record the count `N_fp16` of subsequent concurrent admissions that `MemoryBudgeter.admit()` returns `AdmitDecision` for before the first `RejectDecision`.
2. Repeat under `block_tq_b64_b4` (§4.7 mode (C)) with the same workload and `cap_bytes`. Record `N_block`.
3. Assert `N_block > N_fp16`, with the delta lower bound pinned empirically at P-5-A.2 implementation time. Theoretical loose upper bound: `N_block ≈ N_fp16 × (1 + compression_factor × prefix_fraction)`, where `compression_factor ≈ 3.76` and `prefix_fraction = store_bytes / cap_bytes ≈ 0.5`, so `N_block / N_fp16` could be up to ≈ 2.88×. (PLAN v1.7.4 arithmetic correction: v1.5.1 through v1.7.3 cited this loose upper bound as "≈ 2.4×", which is the `1 + (compression_factor − 1) × prefix_fraction` "bytes freed" form, not the form stated in this line; `1 + 3.76 × 0.5 = 2.88`. A physically tighter ceiling that accounts for `reserved_bytes` continuing to charge fp16 worst-case per admitted request is lower still — the `qwen3-0.6b-admission-headroom-prefix-heavy` row observes `admit_ratio ≈ 1.75` at these parameters, which fits the band between the "bytes freed" form and the loose upper bound.) The "strictly greater" gate is the hard floor.

Also assert the mode-(A) byte-for-byte regression: with `account_prefix_residency=False`, `MemoryBudgeter.headroom_bytes()` on the new bench row equals the pre-P-5 formula's output byte-for-byte on the same `(cap, weights, admissions)` inputs. This is a separate test from the mode-(B)-vs-(C) admission count gate above; acceptance (f) expands it into the broader regression sweep.

**What the "3×" extrapolation from vqbench REPORT §3.1 (3.76× total KV compression) does NOT mean:** it does not mean P-5 admission count grows 3×. The vqbench 3.76× is on prefix-cache residency, not on active-KV reservation, and the budgeter's `reserved_bytes` still charges fp16 worst-case for every admitted request (D-003 constrained, §3.2). The admission-count gain is bounded by how much of `cap_bytes` is prefix-cache vs active-reservation. On prefix-cache-heavy workloads the gain is real; on active-reservation-heavy workloads (long `max_tokens`, few shared prefixes) the gain is small. Document both regimes in the bench row so readers see the shape.

### (d) Decode-speed regression gate on a prefix-hit workload (new)

The decode-overhead gate must run on a workload that **actually hits the codec**. The existing `qwen3-0.6b-smoke` and `qwen3.5-0.8b-b1-parity` rows run `prefix_cache=None` — the codec path is never reached, so a gate on those rows measures nothing. Instead, P-5-A.3c adds two paired bench rows `qwen3-0.6b-prefix-hit-decode-fp16` (baseline) and `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` (codec arm). The opening draft named the gate target `qwen3.5-0.8b-prefix-hit-decode`, but Qwen3.5-0.8B is hybrid-DeltaNet (`has_recurrent_state=True`) and `ContinuousBatcher` refuses `RadixPrefixCache` on recurrent adapters (docs/P3_DELTANET_SURVEY.md C-open-3 — DeltaNet's running accumulation cannot be sliced into block-aligned prefix K/V). Plain Qwen3-0.6B (28 GLOBAL layers, head_dim=128) is the amended target; the BlockTQ-vs-identity ratio is head-dim / layer-count independent so the 0.85× threshold carries over:

- Workload: `generate_batch([prompt, prompt], max_batch_size=1, prefix_cache=shared_pc)` with the same `[p, p]` shape `tests/test_kvcodec_integration.py` uses for C.1 acceptance. Row 0 runs miss-path prefill + decodes `max_tokens` tokens; reclaim inserts prefix; row 1 enters the waiting queue, hits the prefix via `_admit_single_hit_row` → `fetch_detached_blocks` (which invokes `k_codec.decode_tensor` and `v_codec.decode_tensor` per hit block), seeds the BatchKVCache, and decodes `max_tokens` suffix tokens. Row 1's decode loop is where codec overhead manifests on the hot path (through the seeded-prefix K/V being decoded on admission, plus ongoing decode appending to a cache seeded from compressed storage).
- Oracle: `DECODE_TOK_S_WITH_PREFIX_HIT` — measures row 1's decode tok/s specifically (separate from row 0's miss-path decode tok/s, which is the same under any codec because the first admission never consults the codec). Report both as separate columns so readers can see the seeded-prefix admission overhead.

Gate: under any of the P-5 production codecs (`tq_mse_b4`, `block_tq_b64_b4`, `ext_rabitq_b4`), row 1's decode tok/s is at least **0.85× of the IdentityCodec baseline row-1 decode tok/s** on the same bench row. Measured as the median of 3 alternating `fp16 → BlockTQ` measurement pairs after a discarded warmup pair — alternation guards against monotonic machine-load drift being attributed to one codec, and the median absorbs a single outlier without requiring a larger sample. See `tests/test_prefix_hit_decode_speed_gate.py` module docstring for the full measurement-protocol rationale (this is the Q-010 "mean of 5 runs" lineage adapted for a codec-vs-codec comparison rather than a single absolute number).

Rationale: Q-007 flagged that decode overhead was not baked into admission in v0.1; this gate ensures the memory win is not erased through throughput loss. 0.85× is deliberate: MLX-native encode/decode should be much faster than vqbench's NumPy reference but slower than fp16 pass-through. The seeded-prefix admission path pays a one-time `(k_codec.decode_tensor + v_codec.decode_tensor) × n_layers × n_hit_blocks` cost at row 1 admission; the 15% headroom budget is sized to absorb that cost on the Qwen3-0.6B target (amended from Qwen3.5-0.8B — see target-amendment paragraph above). Tighter bars are v0.2 work.

### (e) One-command fp16-vs-codec report across the registry

`scripts/bench.py --all --all-kv-codecs --seeds 42,43,44 --out <jsonl> --report-md <md>` produces a single Markdown table (plus the matching JSONL) that covers every scenario row × every codec in `CODEC_REGISTRY` × three seeds in one invocation. Each row carries the runner's standard columns (`status`, `reason`, `ttft_ms`, `decode_tok_s`, `resident_mb`, `peak_mb`, `wall_s`, `tokens`) plus the oracle-specific metadata (per-row `ΔPPL` / `ΔPPL%` on PPL rows; per-row `n_fp16` / `n_block` / `admit_ratio` on admission-headroom rows; per-row decode tok/s ratio on prefix-hit decode rows). Acceptance (2) is "this command exists and produces a coherent report"; it is **not** "every codec arm is numerically cross-checked against vqbench".

**xcheck scope — narrow, not full-registry.** Numerical cross-check against vqbench is owned by Acceptance (4-b), which binds specifically to the `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` row (the only row whose `VqbenchXcheckSpec` is declared in `silica/bench/scenarios.py`, per the P-5-C.6 declarative spec contract). Other codec arms — `tq_mse_b4`, `ext_rabitq_b4`, `rabitq_b1` — do not have authored vqbench method / bits mappings and do not carry a `VqbenchXcheckSpec`; running `--vqbench-xcheck` against them would force the runner to infer a mapping it cannot validate. Acceptance (2) therefore does **not** require `--vqbench-xcheck` to be on globally. The `vqbench_cross` / `vqbench_gap` column in the report is **structurally present on every row** (the report schema is uniform) but **populated only when two conditions co-occur**: (i) the scenario declares a `VqbenchXcheckSpec` — currently the `-vqbench-aligned` row only — and (ii) `--vqbench-xcheck` is passed to the runner — without the flag, `BenchRunner.vqbench_xcheck_enabled` is `False` and the vqbench subprocess path is skipped even for rows with a declared spec. Populated xcheck numbers for the `-vqbench-aligned` row live in Acceptance (4-b)'s dedicated 3-seed verification run (`docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`), not in the `--all --all-kv-codecs` sweep that (2) verifies; (2) verifies the report / schema coverage, (4-b) verifies the numerical agreement.

**"Easily and fully reproduce the VQBench comparison" intent**, scoped to what this gate actually delivers: the one command produces the **full silica-side table** (fp16 + every codec × quality / memory / decode columns) in one shot, and Acceptance (4-b) provides the single authored vqbench-aligned row where silica's and vqbench's ΔPPL are directly comparable on matching code paths. Full-registry method/bits-equivalent vqbench cross-check would require authoring `VqbenchXcheckSpec`s for the remaining codec families; that is not a P-5 close gate item.

### (f) No regression

The P-4.5 close sweep (v1.6.9 Changelog) must continue to pass on the post-P-5 tree. `python -m pytest tests/ -q` green; `python -m scripts.bench --all` behaviour unchanged (codec rows are opt-in via `--kv-codec`, default remains fp16 for all existing rows).

---

## 8. Sub-unit breakdown

### P-5-A — `BlockTurboQuantMSE` + interface generalization

Split into three sub-sub-units so each ships with its own acceptance and commit scope.

#### P-5-A.0 — Interface + calibration + packing scaffolding

Scope:

- `silica.kvcache.codec` refactor: `VectorCodec[P]` Protocol, `CodedPayload` base, `RawFp16Payload`, `BlockTQPayload`, `RaBitQPayload` dataclasses. Retire the old pair-level `KVCodec`.
- `silica.kvcache.codec.IdentityCodec` migrated from pair-level to side-level (`encode_tensor(x) -> RawFp16Payload`, `decode_tensor(payload) -> mx.array`).
- `silica.kvcache.store.SyntheticPrefixBlockStore` — K/V split kwargs (`k_codec`, `v_codec`, `codec` shorthand). `_detached` type narrows to `dict[int, tuple[tuple[CodedPayload, CodedPayload], ...]]`. `resident_bytes()` method signature unchanged; internal math adjusts to sum per-side payloads.
- `silica.vq._calibration`: NumPy-quarantined Haar rotation + Lloyd-Max helpers. Test: no NumPy import escapes from this module into any other `silica.vq.*` module.
- `silica.vq.core.packing`: MLX-native `pack_sub_byte` / `unpack_sub_byte` helpers shared by all sub-byte codecs. Supports 1 / 2 / 3 / 4 bits per coordinate.

Acceptance:

- (f) regression — all 328 tests in the P-4.5 close sweep pass byte-for-byte with the new interface and IdentityCodec unchanged.
- New unit tests: packing round-trip at all bit widths; the §4.3 `resident_bytes` honesty test (`payload.resident_bytes == sum(arr.nbytes for arr in payload fields)`); the `tests/test_kvcodec_integration.py` Section 2 / Section 4 defensive invariants continue to pass.

Blocked by: nothing (P-4.5 closed).

#### P-5-A.1 — BlockTurboQuantMSE hot path + Q-008 resolution

Scope:

- `silica.vq.block_tq.BlockTurboQuantMSE` — MLX-native encode / decode. Haar rotation via fp32 matmul against `_calibration`-produced fp32 constant (fp16 rotation matrix would erode Lloyd-Max boundary precision); per-vq-block norm extraction; Lloyd-Max scalar quantize via broadcast `mx.argmin((y - centroids)²)` (MLX lacks `mx.searchsorted`); bit-pack via `pack_sub_byte`. Input shape + dtype validated at `encode_tensor`: strict match against the codec's configured `(1, n_kv_heads, block_size, head_dim)` and `dtype ∈ {fp16, bf16}`.
- `silica.vq.turboquant.TurboQuantMSE` — thin alias `TurboQuantMSE = lambda head_dim, num_bits, **kw: BlockTurboQuantMSE(head_dim=head_dim, vq_block_size=head_dim, num_bits=num_bits, **kw)`. No separate class body.
- `silica.bench.codec_registry` — `CodecSpec` dataclass + initial entries for `fp16`, `tq_mse_b4`, `tq_mse_b3`, `block_tq_{b64,b32}_{b3,b4}` (no RaBitQ yet; P-5-B adds those).
- PLAN updates: Q-008 in §10 flipped to `resolved 2026-XX-XX → Option A (independent K/V codecs via k_codec/v_codec kwargs, landed P-5-A.1)`.

Acceptance:

- (a-algo) algorithmic parity vs in-test NumPy reference — closed by `tests/test_block_tq_vqbench_xcheck.py` in P-5-A.1c, per §7(a) split. (a-real) real-activation Frobenius on Qwen3.5-0.8B pre-RoPE K / V — closed at v1.7.5 by `tests/test_block_tq_real_activation_xcheck.py` using the same in-test NumPy reference (inline, not a vqbench subprocess — the v1.5.1 subprocess design was superseded, see `docs/P5_A_REAL_OPENING.md` §2.3). Single HF-cache skip gate; 144 rows across 6 GLOBAL layers × K/V × 4 `(B, bits)` cells × 3 seeds. Worst `|silica_frob - numpy_frob| = 1.15e-4` (~43× headroom under `5e-3`). Evidence: `docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.{md,jsonl}`.
- New: `tests/test_block_tq.py::test_block_equals_scalar_when_B_equals_d` — shipping the Block invariant vqbench already pins.

Blocked by: P-5-A.0.

#### P-5-A.2 — Budgeter headroom accounting

Scope:

- `silica.scheduler.budget.MemoryBudgeter` — constructor gains `account_prefix_residency: bool = True`; the existing `prefix_cache: RadixPrefixCache | None` parameter is the source of the bound store via the new `RadixPrefixCache.store` property (added in P-5-A.0). Three modes as §4.7 table: (A) `account_prefix_residency=False` or `prefix_cache=None` → P-4.5 byte-for-byte; (B) IdentityCodec store bound → honest fp16 residency; (C) compressed codec bound → honest compressed residency.
- `MemoryBudgeter.headroom_bytes()` subtracts `store.resident_bytes()` when both the store is bound (via `pc.store`) AND `account_prefix_residency=True`. See §4.7 for the three-mode formula. Evict-shortfall uses actual per-block bytes via a new single-method `store.resident_bytes_per_block()` helper (returns `num_layers × (k_codec.resident_bytes(1) + v_codec.resident_bytes(1))` or `None` on the pass-through path); the id-based variant (`resident_bytes_for_block_ids(ids)`) is v0.2 scope and does not land here.
- `_count_evictable_prefix_blocks()` **retained** in v0.1 — the count-based LRU eviction policy is arithmetically honest under P-5 codecs (constant per-block residency). The id-list variant (`_evictable_prefix_block_ids()`) is the v0.2 scope that pairs with the rejected-alternative decision surface above.
- `Engine.__init__` (or wherever `MemoryBudgeter` is constructed) — no new wiring; the existing P-4.5-C.1 `prefix_cache=` path on `generate_batch` already hands the cache through, and `MemoryBudgeter.__init__` reads `.store` off it. A single-line change in the budgeter construction site.

Acceptance:

- (c) prefix-residency headroom gate on the new `qwen3-0.6b-admission-headroom-prefix-heavy` bench row (defined in §6.5 / §7 (c)). Comparison is mode (C) vs mode (B) — both paying honest residency — not mode (C) vs mode (A).
- Mode-(A) regression unit test (`tests/test_budgeter_mode_a_regression.py`): with `account_prefix_residency=False`, `budgeter.headroom_bytes()` equals `cap_bytes - weights_bytes - reserved_bytes()` byte-for-byte on identical `(cap, weights, admissions)` inputs — i.e. the pre-P-5 formula. Demonstrates that turning the flag off recovers the P-4.5 code path exactly.
- Mode-(B) baseline-honesty unit test (same file): with `account_prefix_residency=True` and `SyntheticPrefixBlockStore(codec=IdentityCodec(...))` bound + a detached fixture of known block count, `budgeter.headroom_bytes()` == `cap - weights - reserved - (n_blocks × fp16_bytes_per_block)`. This pins that IdentityCodec's mode (B) charges the real fp16 count rather than zero.
- (f) regression — the 328 P-4.5 close tests remain green, achieved by defaulting `account_prefix_residency=True` but rerunning the P-4.5 suite under `account_prefix_residency=False` (the P-4.5 tests never exercise prefix-cache residency admission bounds, so mode (A) is a no-op for them — the gate is that turning the flag on under IdentityCodec must not regress any existing test's admission verdict).

Blocked by: P-5-A.1 (needs at least BlockTQ present to demonstrate the headroom gain).

**PLAN + Changelog block landing in the A.2 commit:** Q-007 in §10 flipped to `partially resolved (P-5, decode-overhead is release gate (§7 (d)) not admission signal)`; §7 P-5 Acceptance text updated to the (b)/(c) framing from this opening; Changelog v1.7.0 entry covers the full P-5-A arc.

#### P-5-A.3 — Decode-speed gate bench row and acceptance

Scope:

- `silica.bench.scenarios` — two paired rows `qwen3-0.6b-prefix-hit-decode-fp16` + `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` with the `[p, p] max_batch_size=1 prefix_cache=True` workload described in §7(d). Target amended from Qwen3.5-0.8B to Qwen3-0.6B at A.3c landing (hybrid-DeltaNet recurrent-state incompat with `RadixPrefixCache`; §7(d) rationale). The BlockTQ-vs-identity ratio is head-dim / layer-count independent so the 0.85× threshold carries over.
- `DECODE_TOK_S_WITH_PREFIX_HIT` oracle in `silica.bench.oracles`.
- Qwen3 adapter `_infer_attn_dtype` — reads dtype from a representative attention-projection weight rather than hardcoding fp16. Loaded Qwen3 checkpoints ship bf16, and the pre-A.3c hardcoded `mx.float16` caused the codec path to reject legitimate bf16 K/V at `BlockTurboQuantMSE.encode_tensor` shape-and-dtype validation. Locked by a unit test that provides a fake bf16 projection weight and asserts `kv_layout().dtype == mx.bfloat16`.
- `conftest.py::_TIMING_GATES` generalized from the Q-010-only pair to a marker/env table; new `prefix_hit_decode_timing` marker + `SILICA_PREFIX_HIT_DECODE_TIMING` env var gate the new acceptance test out of the default sweep (same pattern as Q-010).
- New acceptance test `tests/test_prefix_hit_decode_speed_gate.py` — alternating-pair measurement, median ratio comparison; dual-gated on HF cache presence + explicit opt-in.

Acceptance:

- (d) decode-speed gate on the new paired bench rows — BlockTQ 0.85× IdentityCodec baseline.

Blocked by: P-5-A.1. Can land in parallel with A.2.

### P-5-B — `RaBitQ1Bit` + `ExtRaBitQ`

Scope (landed in sub-units B.1a-c + B.2a-c + B.3):

- `silica.vq.rabitq.rabitq_1bit.RaBitQ1Bit` (B.1a) + bit-pack sign storage; `RaBitQPayload` with three fp16 fields (`norm_o`, `ip_coeff`, packed sign bits). Centroid pinned at zero per §5.3; `fit()` is a no-op.
- `silica.vq.rabitq.rabitq_ext.ExtRaBitQ` (B.2a) + integer-grid codebook; `ExtRaBitQPayload(RaBitQPayload)` subclass adds per-vector fp16 `scale`. `offset` omitted — see §5.4 amendment. `mx.round` half-to-even rounding matches `np.round` and is regression-locked in tests.
- `tests/test_rabitq_1bit_parity.py` (B.1c) + `tests/test_rabitq_ext_parity.py` (B.2c) — NumPy reference transcribed inline (no vqbench import); fp16-rounded shared input. B.2c uses fp32 reference to match silica's MLX precision and bounds indices/decoded at fractional-mismatch / relative-Frobenius rather than bit-for-bit (boundary-precision fuzz across Metal vs CPU fp32 matmul).
- `silica.bench.codec_registry` gains `rabitq_b1` (B.1b, K-only `v_supported=False`) + `ext_rabitq_b{2,3,4}` (B.2b, symmetric). `effective_bits_per_value` gets `rabitq` (`+32/head_dim`) and `ext_rabitq` (`+48/head_dim`) branches.
- `silica.bench.scenarios` gains `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` (B.3) as the ExtRaBitQ arm of the §7(d) decode-speed gate.
- `tests/test_prefix_hit_decode_speed_gate.py` (B.3) extends the P-5-A.3c gate with a matched ExtRaBitQ arm at the same 0.85× threshold. `rabitq_b1` is deliberately not gated — K-only codec that cannot install via the symmetric `kv_codec=` shorthand, and its hypercube MSE is worse than BlockTQ at matching bit budget.
- Acceptance: (d) decode-speed gate landed for both BlockTQ and ExtRaBitQ arms.

Blocked by: P-5-A (landed).

### P-5-C — Reproducibility + variance + cross-check

Decomposed into six sub-units. C.1 has landed; C.2 is next. The original flat scope below was rewritten in the C.1 opening commit.

#### P-5-C.1 — Teacher-forced streaming PPL oracle (landed 2026-04-23)

Scope (shipped):

- `silica.mlx.runner.forward_batched_full(model, tokens, cache_list) -> (B, T, V)` — new entry point returning all-position logits. Existing `forward_batched` rewired to `forward_batched_full(...)[:, -1, :]` so sampling and teacher-forced paths cannot drift (regression-locked by `test_runner_forward_batched.py::test_forward_batched_is_last_position_of_full`).
- `silica.bench.ppl_oracle.teacher_forced_chunked_nll(adapter, token_ids, *, chunk_size) -> (nll_sum, n_tokens)` — cache-agnostic fp16 baseline oracle. One `BatchKVCache` allocated via `adapter.make_batch_cache([0])` and *shared across chunks* (mlx-lm mutates in place); chunk-boundary token scored via the previous chunk's last logit (materialized with `mx.contiguous` + `mx.eval` before the next forward mutates the cache); within-chunk scored via shift-by-1. `perplexity_from_nll(nll_sum, n_tokens) -> float` wraps to `exp(mean)` (`n==0 → inf` for aggregation, `n<0 → ValueError`).
- `tests/test_ppl_oracle.py` — 19 tests. Self-contained fake adapter/model/cache stack where logits depend on absolute cache-offset position, so chunk invariance is meaningful. Parametrized across chunks `{8, 16, 32, 64, 128}` at `rel=1e-5, abs=1e-5`.

Scope deferrals (out of C.1, in C.2):

- Codec-backed oracle arm. C.1 drives the adapter's bound cache (fp16 baseline); the codec path runs through a `RadixPrefixCache` whose store carries a `VectorCodec` and is driven by `teacher_forced_chunked_nll_with_codec` in C.2. Any ΔPPL vs VQ claim goes through the C.2 arm.
- WikiText-2 loader. The C.1 oracle is text-free (token IDs only) — the unit tests use a synthetic fake model so they run without tokenizer or dataset dependency.

Blocked by: P-5-A.1 (landed).

#### P-5-C.2 — Codec-backed PPL + WikiText bench rows (landed 2026-04-23)

Scope:

- `silica.bench.ppl_oracle.teacher_forced_chunked_nll_with_codec(adapter, prefix_cache, token_ids, *, chunk_size)` — second oracle entry point. Seeds each chunk's `BatchKVCache` from `prefix_cache.lookup` → `fetch_detached_blocks` → `build_seeded_batch_kv`; after each forward, extracts the newly-grown aligned-prefix blocks and `insert_detached` them so chunk i+1 can consume them through the codec. Tests assert `encode_tensor` / `decode_tensor` are invoked via a counting-codec / counting-store stand-in — a quantized PPL row silently equal to fp16 because the codec never fired is the specific failure mode the counter is there to catch.
- `silica.bench.oracles` gains `PPL` `OracleKind`.
- WikiText-2 loader: deterministic local-cache path, no live download at bench time; tiny fixture under `tests/fixtures/` for offline unit tests.
- Bench rows: `qwen3-0.6b-wikitext-ppl-{fp16, tq-mse-b4, block-tq-b64-b4, ext-rabitq-b4}` per §6.5 (four rows, all on Qwen3-0.6B for consistency with the decode-speed and compression rows; scenario-id hyphenation mirrors the prefix-hit decode rows).

Blocked by: P-5-C.1 (landed), P-5-A.1 (landed).

#### P-5-C.3 — STORAGE oracle + compression / admission-headroom rows (landed 2026-04-23)

Scope:

- `silica.bench.oracles` gains `STORAGE` `OracleKind` reading `prefix_cache.store.resident_bytes()`.
- Bench rows: `qwen3-0.6b-compression` (per-codec resident-bytes column) and `qwen3-0.6b-admission-headroom-prefix-heavy` (§4.7 admission-signal row demonstrating the `resident_bytes`-aware budgeter).

#### P-5-C.4 — Variance / multi-seed plumbing (landed 2026-04-23)

Scope:

- `BenchRunner` — per-(scenario, codec, seed) JSONL rows; `--seeds 42,43,44` CLI flag; markdown table `mean ± std` aggregation column.

#### P-5-C.5 — `--kv-codec` / `--all-kv-codecs` CLI (landed 2026-04-23)

Scope:

- `scripts/bench.py` — `--kv-codec <id>` (registry dispatch through `BenchRunner.engine_factory`), `--all-kv-codecs` (iterate every `CodecSpec`), `--k-only` selection. The registry layer already exists (P-5-A.1b); C.5 is the CLI surface only.

#### P-5-C.6 — vqbench subprocess cross-check (landed 2026-04-23)

Scope:

- `--vqbench-xcheck` flag on `scripts/bench.py` — invokes the P-4.4 `silica/bench/vqbench_baseline.py` subprocess path alongside the silica oracle and reports the ΔPPL-delta per §7 (b).

Acceptance: (b) PPL-delta cross-check vs vqbench; (e) full vqbench table reproducible in one flag; (f) no regression. Closed out at C.6 land.

Blocked by: P-5-A.1 (needs at least BlockTQ live to produce a non-fp16 column). C.2 depends on C.1; C.3-C.6 can interleave with P-5-B.

---

## 9. What P-5 does NOT do

- **No compressed-domain attention** — D-003. vqbench Phase 10 ("RaBitQ as estimator-native attention") is v0.2 follow-up, gated on an MLX variable-length SDPA kernel (Q-009 / R-7).
- **No PQ / OPQ codec** — vqbench keeps them for benchmark parity but doesn't recommend them for production. silica follows; can be added in v0.2 if a use case appears.
- **No per-head / per-layer calibration** — single `(head_dim, seed)` Haar rotation shared across all layers. vqbench observes per-head K distributions differ (`scripts/real_k_mse_qwen35.py`); per-layer calibration is a v0.2 codec enhancement.
- **No heterogeneous per-layer-shape codec dispatch** — Gemma4's sliding 16×256 + full 4×512 mix requires `list[tuple[VectorCodec, VectorCodec]]` support on the store (one K/V pair per `AttentionKind`, possibly per layer). P-5 is Qwen3-family (homogeneous) only — same boundary P-4.5-C.1 took. Heterogeneous dispatch is a v0.2 item gated on Gemma4 being in v0.1's production target set (D-011 currently lists it as a generality target, not a production target). Not a P-5 close gate; explicitly removed from P-5's sub-unit breakdown (§8) to keep scope bounded.
- **No 3-bit K+V production recommendation** — vqbench data shows 3-bit K+V is noisy (std > mean on Block B=32 and ExtRaBitQ). Ships as a bench row + codec, not a production setting.
- **No codec on-disk calibration cache** — re-calibrate per-construction. v0.2 optimization if cold-start becomes a problem.
- **No `PagedPrefixBlockStore` codec integration** — paged-attention kernel still blocked by D-003 / Q-009 / R-7.
- **No active-`BatchKVCache` codec wrapping** — same boundary as P-4.5-C.1.
- **No decode-overhead-driven admission** — Q-007. Decode overhead gates P-5 landing via acceptance (d) but does not enter admission decisions. That is a v0.2 multi-codec multi-quality-setting item.

---

## 10. References

- **vqbench** (local `vqbench/`, gitignored). NumPy + PyTorch + HF transformers. Algorithmic + numeric reference; not imported at runtime.
  - `vqbench/REPORT.md` — Qwen3.5-4B 3-seed headline table + long-context sweep. Production recommendation: Block B=64 4-bit K+V.
  - `vqbench/BlockTQ.md` — algorithm walkthrough for BlockTurboQuantMSE.
  - `vqbench/PLAN.md` — vqbench's own roadmap, including Phase 10 "RaBitQ as estimator-native attention" (v0.2 silica item).
  - `vqbench/vqbench/methods/turboquant/mse.py`, `block_mse.py`, `codebook.py` — Algorithm 1 + Block variant + Lloyd-Max.
  - `vqbench/vqbench/methods/rabitq/rabitq_1bit.py`, `rabitq_ext.py`, `estimator.py` — RaBitQ family.
  - `vqbench/vqbench/core/rotation.py`, `packing.py` — Haar rotation + bit packing.
  - `vqbench/vqbench/kv_cache/compressor.py` — K/V split pattern (`KVCacheCompressor(key_q, value_q)`).
  - `vqbench/vqbench/validation/streaming_ppl.py` — streaming PPL oracle (silica's MLX-native reference).
- **silica P-4.5-C closed seam** — `silica/kvcache/store.py::SyntheticPrefixBlockStore(codec=...)`; `silica/kvcache/codec.py::IdentityCodec`, `CodedBlock`, `KVCodec`; `docs/P4_5_C_KVCODEC_OPENING.md` §6.2 / §8.
- **silica bench infrastructure** — `silica/bench/runner.py` (`BenchRunner`), `silica/bench/scenarios.py` (scenario catalog), `silica/bench/vqbench_baseline.py` (P-4.4 subprocess cross-check, already shipped), `scripts/bench.py`.
- **Design principles** — PLAN Principle 8 (savings must be observable), Principle 9 (native capabilities, swappable implementations). D-003 (no compressed-domain attention v0.1). D-009 (MLX-native runtime). D-011 (MoE + Dense generality). D-012 (canonical `resident_bytes`).
- **Open Questions resolved by this opening** — Q-008 (K/V independent codecs → Option A via §4.2); Q-007 partially (decode overhead as release gate, not admission signal).
- **Open Questions carried to v0.2** — Q-007 admission-path resolution; vqbench Phase 10 estimator-native attention; heterogeneous per-layer-shape codec; per-head / per-layer calibration.
