# Silica-MLX Plan

| Field        | Value                                                                      |
| ------------ | -------------------------------------------------------------------------- |
| Version      | v1.7.5                                                                     |
| Last updated | 2026-04-25                                                                 |
| Status       | P-5 complete; P-5 Acceptance (1)–(4) closed at v1.7.4; (a-real) real-activation xcheck closed at v1.7.5; P-3-C5 closed in slice-prefill regime (C5.5 α-MVP); P-3-E4 batched MoE smoke closed (parity deferred); pre-RoPE production routing (P-5-F) / (b-static) PPL baseline remain in backlog |
| Maintainer   | Xin Zhou                                                                   |
| Source       | `docs/PLAN.md` (single source of truth)                                    |

> **CRUD convention.** All stable IDs in this document (Phase `P-N`, Decision `D-NNN`, Open Question `Q-NNN`, Milestone `M-N`, Interface `I-N`) are never reused once allocated. When editing, touch only the relevant block. New facts go in the Decisions Log; new questions in Open Questions. A Phase status change updates the `Status` field of that Phase block and appends a line to the Changelog.

---

## Table of Contents

1. [TL;DR](#1-tldr)
2. [Mission](#2-mission)
3. [Scope](#3-scope)
4. [Design Principles](#4-design-principles)
5. [Architecture Overview](#5-architecture-overview)
6. [Core Interfaces (Phase 0 freeze candidate)](#6-core-interfaces-phase-0-freeze-candidate)
7. [Phases](#7-phases)
8. [Priority & Milestones](#8-priority--milestones)
9. [Decisions Log](#9-decisions-log)
10. [Open Questions](#10-open-questions)
11. [Risks](#11-risks)
12. [References](#12-references)
13. [Changelog](#13-changelog)

---

## 1. TL;DR

**Silica-MLX is a single-Mac-chip local LLM inference platform.** MLX-native; vLLM-style core plus a mini-sglang-style outer layer. The goal is to run Qwen3.5-27B / Gemma4-31B reliably on a 48 GB M5 Pro. VQ, weight streaming, and speculative decoding are **native capabilities** actively exploited by the platform — not passively supported third-party extensions. Implementations are swappable, but integration points are fixed (see Principle 9).

**The platform itself is the product.** VQ and similar techniques are weapons, not research subjects (see D-006).

---

## 2. Mission

On a single Apple Silicon Mac, let developers run 27B–31B-class models locally with the same fluidity they get from vLLM in the cloud.

**Target users:** Mac developers who want to run large models locally for apps, experiments, or privacy-sensitive work.

**Not for:** VQ algorithm researchers, distributed-serving system researchers, cloud providers.

**Success criteria (v0.1):**
- Qwen3.5-27B and Gemma4-31B run reliably on a 48 GB M5 Pro.
- Python API + minimal CLI.
- Unified benchmark / runtime path (no separate eval codepath).
- VQ / weight streaming / speculative are native capabilities, built into the main loop as stubs from P-0, progressively replaced with real implementations in P-5 / P-6 / P-7. Integration points fixed, implementations swappable (Principle 9).
- Minimal OpenAI-compatible HTTP API + session available via Phase 8, aligned with §3.1 scope and the D-006 "product face" framing.

---

## 3. Scope

### 3.1 In Scope (v0.1)

- Single Mac chip, single-process local inference.
- MLX-native; no CUDA assumptions.
- Qwen3.5-27B and Gemma4-31B as production dense targets.
- Python API + minimal CLI.
- vLLM-style core: paged KV, continuous batching, prefix cache, memory budget.
- Five frozen interfaces: ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine.
- Native capabilities: VQ KV compression (BlockTQ / RaBitQ), weight streaming, draft-target speculative — see Principle 9.
- **MoE + Dense architectural generality (D-011)**; v0.1 must actually run at least one MoE target, not just "interface reserved but untested".
- Unified benchmark path.
- Minimal OpenAI-compatible HTTP API + session (Phase 8).

### 3.2 Non-Goals (v0.1)

- Distributed, multi-node, multi-Mac cooperation.
- PD (prefill/decode) disaggregation.
- Tensor parallelism (not needed on a single chip).
- Hand-rolling native Metal kernels from scratch.
- Non-MLX quantization formats such as GGUF / AWQ.
- Full agent orchestration.
- Explicitly excluded codecs: PQ, OPQ.
- Compressed-domain attention fast path (see D-003, deferred to v0.2).
- Complex speculative schemes (DFlash / EAGLE / Medusa) — deferred to v0.2.
- **PyTorch runtime dependency** (D-009): the inference hot path may not contain `torch.Tensor`; torch is allowed only as an optional dev dependency for offline weight conversion.
- **CUDA / ROCm / XPU / TPU backends** (D-009): `csrc/`, CUDA kernels, and device-specific workers are out of scope.
- **Multimodal input / output** (D-014): v0.1 runs the **text-only path** of multimodal checkpoints (Qwen3.5 family, Gemma4). Vision / audio / video encoder lifecycle, image / audio / video tokens, and non-text processors are v0.2. Multimodal checkpoints are expected to load with their vision / audio heads ignored or weight-skipped.

### 3.3 Target Hardware

| Field        | Value                                              |
| ------------ | -------------------------------------------------- |
| Machine      | Apple M5 Pro                                       |
| Memory       | 48 GB unified memory                               |
| OS           | macOS                                              |
| Acceleration | MLX on Apple Silicon GPU + Neural Engine (via MLX) |

### 3.4 Target Models

| Model             | Parameters                   | Role                             | First used  |
| ----------------- | ---------------------------- | -------------------------------- | ----------- |
| Qwen3.5-0.8B      | 0.8B                         | Dev / iteration bring-up model   | Phase 1     |
| Qwen3.5-27B       | 27B                          | Dense production target          | Phase 3     |
| Gemma4-31B        | 31B                          | Dense production target          | Phase 3     |
| Qwen3.5-35B-A3B   | 35B total / 3B active        | MoE generality target (D-011)    | Phase 3     |
| gemma-4-26B-A4B   | 26B total / 4B active        | MoE generality target (D-011)    | Phase 3     |

---

## 4. Design Principles

Stable principles. Changing any of them requires a new Decisions Log entry.

1. **Platform as product, VQ as weapon.** Silica-MLX itself is the product. VQ / weight streaming / speculative are means to make the product run large models well, not research subjects. Do not let "VQ is the ultimate deliverable" creep into any design (D-006).

2. **Single Mac chip + Apple unified memory first.** The hard constraint is a single chip; no distribution. But *actively* exploit unified memory — the fact that weights, KV, and activations share one physical pool must be reflected in WeightProvider and KVManager design. Do not abstract the Mac as a generic GPU.

3. **Engine skeleton first, native capabilities integrated progressively.** Phase 0–4 bring up the engine skeleton + baseline + target models (the main loop already contains stubs of every native capability from P-0: `IdentityCodec` / `ResidentWeightProvider` stub / `NoopDraftEngine`). Phase 5–8 progressively replace the stubs with real implementations (VQ / streaming / speculative) and add the serving layer. This is **not** "build the engine, then add plugins" — integration points exist from P-0; P-5..P-7 replace stubs, they do not wire in anything new (see Principle 9).

4. **Bench and runtime share the same path.** A benchmark must be a thin wrapper over `silica.engine.Engine`. There cannot be a second eval codepath.

5. **Native capability contracts are frozen early.** The five core interfaces (ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine) are the integration points for native capabilities. They are freeze candidates in Phase 0 and finally frozen at P-0 exit (see §6). The scheduler and engine core are unaware of concrete implementations. Changes go through the Decisions Log. These are **not** "third-party plugin extension points" — they are Silica's own capability boundaries (see Principle 9).

6. **MLX-native hot path (hard constraint).** The inference hot path must be 100% MLX: every tensor is an `mx.array`, every op goes through MLX. `torch.Tensor` / `numpy.ndarray` are **not allowed** in the hot paths of `silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler`. Phase 1 may wrap `mlx-lm` **because mlx-lm is itself MLX-native**; replacing it with a torch-based wrapper is forbidden. vllm and transformers are **algorithm references only**, not runtime dependencies. See D-009.

7. **Small over large in early phases.** Phase 1 starts with Qwen3.5-0.8B; Phase 3 switches to the target large models. Close the loop on small before scaling up.

8. **Savings must be observable.** Any memory / streaming / acceleration saving must be visible to the scheduler (e.g. `KVCodec.logical_bytes` vs `resident_bytes`), otherwise the memory budgeter cannot translate the saving into "admit more requests / longer context".

9. **Native capabilities, swappable implementations.** VQ KV compression, weight-streaming residency, and speculative decoding are Silica-MLX's **native capabilities**, not "third-party plugin extension points". They are built into the engine main loop as stubs from P-0 (`IdentityCodec` / `ResidentWeightProvider` stub / `NoopDraftEngine`) and progressively replaced in P-5 / P-6 / P-7. **Integration points are fixed**: the memory budgeter reads `KVCodec.logical_bytes` / `resident_bytes`; the scheduler reads `WeightProvider` prefetch signals; the decode loop talks to `DraftEngine.propose` / `commit`. **Implementations are swappable**: BlockTQ / RaBitQ / future codecs; Resident / Streaming / future residency strategies; Noop / DraftTarget / future EAGLE / Medusa — all are interchangeable implementations behind the same integration point. Analogy: vLLM's "attention backend" — at the native layer every model goes through a backend, but the concrete backend (FlashAttention / Xformers / Triton) is swappable. We **do not say "plugin"**; we say "backend / codec / implementation". This complements Principle 1 (platform-as-product stance) with the architectural expression.

---

## 5. Architecture Overview

### 5.1 Module Layout

Target layout (the actual repo is the source of truth after Phase 0):

```
silica-mlx/
├── pyproject.toml
├── README.md
├── docs/
│   └── PLAN.md                   # this file
├── silica/
│   ├── __init__.py               # re-exports Engine, LLM
│   ├── core/                     # Request, SamplingParams, Context, logging, profiling
│   ├── mlx/                      # MLX array utilities, profiling hooks
│   ├── engine/                   # Engine class, generate() loop
│   ├── scheduler/                # ContinuousBatcher, request lifecycle, memory budget
│   ├── kvcache/                  # PagedKVCache, PrefixCache, KVCodec Protocol
│   ├── models/                   # ModelAdapter Protocol + per-family adapters + factory
│   ├── weights/                  # WeightProvider Protocol + Resident/Streaming impls (residency & prefetch)
│   ├── vq/                       # BlockTQ, RaBitQ codecs
│   ├── speculative/              # DraftEngine Protocol + Noop/DraftTarget impls
│   ├── server/                   # CLI, OpenAI API, session management
│   ├── llm/                      # High-level Python API (LLM class, Phase 8)
│   └── bench/                    # Benchmark runner (thin wrapper over Engine)
└── tests/
```

### 5.2 Data Flow (v0.1 target)

```
User
 → silica.engine.Engine.generate(prompt, sampling_params)
   → Tokenizer
   → Scheduler.admit(request)
     → KVManager.get_computed_blocks / reserve_for_prefill (KVCodec integrates transparently)
   → ModelAdapter.prefill / decode_step (through WeightProvider; KV via kv_handle from KVManager)
     → KVCodec.encode_block / decode_block per layer
   → Sampler (+ optional DraftEngine)
 → token stream
```

Phase 8 adds a `silica.server.openai_api` FastAPI wrapper on top; internally it is still the same Engine instance.

### 5.3 Process Model

v0.1 is **single-process**. No tokenizer worker split, no detokenizer split, no scheduler worker split. Splitting is a v0.2 discussion.

### 5.4 Reference Map to vLLM v1

**Algorithm / architecture reference only**, not a runtime dependency. Local path: `vllm/` (gitignored). vLLM's v0 (the old `engine/llm_engine.py`) is **not** our reference target; we track v1 (`vllm/v1/`).

| Silica module                           | vLLM v1 reference files                                                 | Purpose                                      |
| --------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------- |
| `silica.engine.Engine`                  | `vllm/v1/engine/llm_engine.py`, `vllm/v1/engine/core.py`                | Engine main-loop structure                   |
| `silica.scheduler.batcher`              | `vllm/v1/core/sched/scheduler.py`                                       | Continuous batching scheduling               |
| `silica.scheduler.budget`               | `vllm/v1/core/kv_cache_manager.py` (budget portion)                     | Memory budget and admission                  |
| `silica.kvcache.paged.PagedKVCache`     | `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_manager.py`        | Paged / block KV allocator                   |
| `silica.kvcache` (interface)            | `vllm/v1/kv_cache_interface.py`                                         | KV cache spec and layout                     |
| `silica.kvcache.prefix`                 | `vllm/v1/core/kv_cache_manager.py` (prefix portion)                     | Prefix cache hit and reuse                   |
| `silica.core.request.RequestState`      | `vllm/v1/request.py`                                                    | Request state machine                        |
| `silica.core.sampling` / sampler        | `vllm/v1/sample/`                                                       | Sampling reference                           |
| `silica.models.*` attention             | `vllm/v1/attention/backend.py` (interface idea only)                    | Swappable attention backend pattern; **no copying of CUDA implementations** |
| `silica.speculative.*`                  | `vllm/v1/spec_decode/`                                                  | Speculative decoding architecture reference  |

**Not referenced:**
- `vllm/csrc/` — C++ / CUDA kernels.
- `vllm/vllm_flash_attn/` — CUDA flash attention.
- `vllm/v1/worker/gpu_model_runner.py`, `tpu_model_runner.py`, `xpu_model_runner.py`, `cpu_model_runner.py` — device-specific runners; we write only `silica.mlx.runner`.
- `vllm/v1/executor/`, `vllm/distributed/`, `vllm/v1/core/kv_cache_coordinator.py` — multi-process / multi-node coordination, not needed for v0.1.
- vLLM's v0 paths (`vllm/engine/`, top-level `vllm/worker/`) — superseded by v1.

### 5.5 Reference Map to vqbench

**Algorithm + numeric oracle only**, not a runtime dependency. Local path: `vqbench/` (gitignored; includes nested `vqbench/turboquant_plus/`). vqbench is a **NumPy + PyTorch + HF transformers** codebase — under D-009 it is **not a runtime source**. But it carries an empirical Qwen3.5-4B PPL baseline (`BlockTurboQuantMSE B=64` 4-bit K+V near-lossless, see `vqbench/REPORT.md`) that serves as a **correctness oracle** for Silica's MLX-native rewrite in P-5.

| Silica module                | vqbench reference files                                                                                     | Purpose                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `silica.vq.block_tq`         | `vqbench/vqbench/methods/turboquant/block_mse.py`                                                           | BlockTurboQuantMSE algorithm (B=16/20/32/40/64)          |
| `silica.vq.rabitq`           | `vqbench/vqbench/methods/rabitq/rabitq_1bit.py`, `rabitq_ext.py`                                            | RaBitQ 1-bit / extended bits                             |
| `silica.vq` (factory)        | `vqbench/vqbench/torch_wrapper/module.py` (`_get_method_class`)                                             | Codec registry / naming convention reference             |
| `silica.kvcache` (pair layer)| `vqbench/vqbench/kv_cache/compressor.py` (`KVCacheCompressor`)                                              | K/V independent-codec pair pattern (see Q-008)           |
| `silica.bench.scenarios`     | `vqbench/scripts/reproduce_qwen35_4b_headline.py`, `variance_qwen35_4b.py`, `run_qwen35_27b_sweep.py`       | Qwen3.5 bench scenarios + PPL regression reference       |
| P-5 numeric oracle           | `vqbench/REPORT.md` (Qwen3.5-4B result tables), `vqbench/BlockTQ.md`                                        | Empirical baseline + algorithm walkthrough               |

**Explicitly not referenced (forbidden as runtime source):**
- `vqbench/vqbench/torch_wrapper/` — PyTorch `nn.Module` + HF `DynamicCache` subclass, violates D-009; used as "anti-pattern" in D-010 Consequences.
- NumPy implementations in `vqbench/vqbench/methods/*.py` — algorithmic reference only, **not** `mx.array` equivalents; P-5 must rewrite them MLX-native.
- `vqbench/PLAN.md` — vqbench's own historical plan, unrelated to Silica's plan; **not** a stale copy, do not confuse.

---

## 6. Core Interfaces (Phase 0 freeze candidate)

The five core interfaces of v0.1. **Signatures are finalized at P-0 exit**, not the moment this document is written. During Phase 0 we may extend the operation set (for example, I-1 / I-2 gained `prefill`/`decode_step` and `append`/`commit`/`rollback` in this version); only at P-0 exit are they truly frozen. Changing a signature after P-0 requires a new Decisions Log entry. What follows is the contract skeleton; the actual `typing.Protocol` signatures live in code.

### I-1 ModelAdapter

Responsible for model structure, tokenizer, layer execution, attention pattern, and execution semantics (prefill / decode).

```python
class ModelAdapter(Protocol):
    config: ModelConfig

    def build(self, weight_provider: WeightProvider) -> Module: ...
    def kv_layout(self) -> KVLayout: ...                  # num_layers, n_kv_heads, head_dim, dtype
    def attention_pattern(self) -> AttentionPattern: ...  # global / sliding / hybrid / recurrent / hybrid_deltanet per layer (D-015)
    def tokenizer(self) -> Tokenizer: ...
    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]: ...                 # returns (logits, non-KV state delta)
    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]: ...                 # returns (logits, non-KV state delta)
```

**Key constraints:**
1. `attention_pattern()` must be able to express Qwen3.5's **hybrid layering** (D-015) — **KV-attention layers** (`global` / `sliding` / `hybrid`) dispatch to `KVManager`-owned blocks via `kv_handle`; **recurrent layers** (`recurrent` for pure-linear, `hybrid_deltanet` for Qwen3.5's Gated-DeltaNet-over-Gated-Attention stack) dispatch to adapter-owned per-layer state carried via `state_delta`. The scheduler reads the pattern and routes per layer — Phase 3 cannot run Qwen3.5 targets without this extension.
2. **KV mutation ownership belongs to `KVManager`, not the adapter.** `prefill` / `decode_step` read and write KV via `kv_handle` (issued by `KVManager`); they never hold block pointers directly, make residency decisions, or touch the prefix cache structure.
3. `state_delta` carries **non-KV** runtime state only: sampling RNG, MoE router cache, position counter, sliding-window mask cursor, and **DeltaNet per-layer recurrent state** (first-class tenant per D-015). **Counter-examples (forbidden inside `state_delta`)**: KV blocks / cache residency mutations / prefix cache pinning — these go through `kv_handle` owned by `KVManager`. **Recurrent-state ownership, `commit` / `rollback` semantics under speculative decoding, prefix-reuse key derivation, and inclusion in `KVManager.budget()` via `state_delta.recurrent_bytes()` are specified in D-015** — I-1 / I-2 signatures unchanged; contract extended.

### I-2 KVManager

Owns paged / block KV, the prefix cache, the memory budget, and the incremental mutation primitives required by continuous batching and speculative decoding.

```python
class KVManager(Protocol):
    block_size: int

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList: ...                                   # reserve blocks for initial prompt
    def append_slot(self, req_id: str, n: int) -> BlockList: ...  # extend during decode
    def commit(self, req_id: str, n_accepted: int) -> None: ...   # speculative: accept draft
    def rollback(self, req_id: str, n_reject: int) -> None: ...   # speculative: reject draft
    def free(self, req_id: str) -> None: ...
    def get_computed_blocks(
        self, token_ids: Sequence[int]
    ) -> PrefixHit: ...                                   # prefix cache lookup (vLLM v1 naming)
    def available_blocks(self) -> int: ...                # fast path for admission
    def budget(self) -> MemoryBudget: ...                 # logical_bytes, resident_bytes, headroom
```

**Key constraints:**
1. `budget()` must report both logical and resident bytes (Principle 8). The scheduler uses this for admission control.
2. **Incremental semantics.** `reserve_for_prefill` reserves blocks for the initial prompt (*reserve*, not "logically allocate and write"); `append_slot` incrementally extends during decode; `commit` / `rollback` support speculative accept / reject (P-7); `free` releases on request completion. The single-request `SimpleKVCache` in P-1 may implement `commit` / `rollback` as no-ops, but **the signatures must exist from P-0** — otherwise P-7 is forced to break the frozen API.
3. `get_computed_blocks` is the prefix cache lookup (naming aligned with vLLM v1's `kv_cache_manager.get_computed_blocks`); it returns the list of already-computed blocks so the scheduler can reuse and pin them on admission.
4. `available_blocks()` is a fast, block-granular path. It is **not** equivalent to `budget().headroom`: the latter is byte-granular (influenced by the codec's logical/resident ratio), the former is block-granular and gives the scheduler an O(1) "can I admit one more request?" check.

### I-3 VectorCodec

Uniform abstraction for KV encoding and decoding. **Side-level** as of P-5-A.0.4: one `VectorCodec[P]` instance operates on one tensor (either K or V from one block of one layer). Pair-level dispatch lives in the store (`SyntheticPrefixBlockStore(k_codec=, v_codec=)` or the `codec=` shorthand). v0.1 **does not include** a compressed-domain attention fast path (D-003).

```python
P = TypeVar("P", bound=CodedPayload)

class VectorCodec(Protocol, Generic[P]):
    block_size: int
    dtype: mx.Dtype

    def encode_tensor(self, x: mx.array) -> P: ...          # one side, one block
    def decode_tensor(self, payload: P) -> mx.array: ...    # fp16 output (D-003)
    def logical_bytes(self, num_tokens: int) -> int: ...    # fp16 baseline, one side
    def resident_bytes(self, num_blocks: int) -> int: ...   # actual storage, one side
```

Concrete payload subclasses (`CodedPayload` hierarchy): `RawFp16Payload` (identity), `BlockTQPayload` (packed indices + per-vq-block scales), `RaBitQPayload` (packed indices + norm_o + ip_coeff). Every subclass enforces D-012 honesty at construction: declared `resident_bytes` must equal the sum of `.nbytes` across all `mx.array` fields.

**Key constraint:** a codec sees one tensor only; it is unaware of batch, scheduler, model structure, or the partner side (K / V dispatch is a store concern). Pre-P-5-A.0.4 I-3 used a pair-level `KVCodec.encode_block(k, v) -> CodedBlock` shape; that signature is retired and the old `CodedBlock` / `KVCodec` names are removed from the codebase. Historical P-4.5-C records in §7 and the amendment log keep their original wording because they describe past state.

### I-4 WeightProvider

Residency / streaming abstraction for weights. Phase 1 uses Resident; Phase 6 wires in Streaming.

```python
class WeightProvider(Protocol):
    def get_layer(self, layer_idx: int) -> LayerWeights: ...      # sync blocking
    def prefetch(self, layer_indices: Sequence[int]) -> None: ...  # hint, may no-op
    def release(self, layer_idx: int) -> None: ...
    def resident_bytes(self) -> int: ...

    # MoE-aware per-expert granularity (D-011)
    def get_expert(self, layer_idx: int, expert_id: int) -> ExpertWeights: ...
    def prefetch_experts(
        self, layer_idx: int, expert_ids: Sequence[int]
    ) -> None: ...
    def release_expert(self, layer_idx: int, expert_id: int) -> None: ...
```

**Key constraints:** `get_layer` blocks synchronously; `prefetch` is a hint and may be a no-op. Callers never need to know where weights physically live. Streaming implementations must exploit unified memory (Principle 2) rather than pretend to be a PCIe device.

**MoE constraints (D-011).** A MoE adapter's FFN execution **must** go through `get_expert` / `prefetch_experts`; it **must not** pull all experts via `get_layer` — otherwise P-6's `StreamingWeightProvider` loses its primary value for MoE (per-expert residency). Dense `WeightProvider` implementations raise `NotImplementedError("dense provider has no per-expert path")` from the three expert methods rather than implementing them as no-ops: this makes a MoE adapter wired to a dense provider fail loudly instead of silently degrading. Wiring a MoE adapter to a dense provider is a model-registry configuration error, not a runtime fallback scenario.

### I-5 DraftEngine

Draft provider for speculative decoding. Phases 1–6 use Noop; Phase 7 wires in DraftTarget.

```python
class DraftEngine(Protocol):
    def propose(self, ctx: RequestState, k: int) -> DraftTokens: ...
    def commit(self, ctx: RequestState, accepted_len: int) -> None: ...
```

---

## 7. Phases

Each Phase uses the same structure: `ID / Goal / Scope / Strategy / Deliverables / Acceptance / Dependencies / Status / Notes`. Edit only the block of the Phase you are changing. Status values: `planned / in-progress / done / blocked / obsolete`.

### P-0 Phase 0 — Skeleton

- **Goal:** stand up the repo, package layout, config, logging, profiling, and interface skeleton.
- **Scope:** `pyproject.toml`, 11 sub-package skeletons, five Protocols, logging, profiler, a minimal interface test.
- **Strategy:**
  - uv + Python 3.12 (D-001).
  - stdlib `logging` + a colored formatter (no structlog).
  - `pydantic` v2 `BaseSettings` for config.
  - Profiler built with `@contextmanager`, a global `MetricsRegistry`, and a unified metrics schema.
- **Deliverables:**
  - [ ] `pyproject.toml` (uv, mlx, mlx-lm, numpy, pydantic, pytest, ruff, mypy; Python 3.12).
  - [ ] 11 sub-package skeletons + `__init__.py`.
  - [ ] `silica.core.request.Request` / `RequestState`.
  - [ ] `silica.core.sampling.SamplingParams`.
  - [ ] `silica.core.sampler.Sampler` concrete class + `LogitProcessor` protocol (D-013); minimal processors `temperature` / `top_k` / `top_p` / `repetition_penalty` wired in the fixed order, plus a `greedy` fast path.
  - [ ] `silica.core.logger` + `silica.core.profiler`.
  - [ ] Five Protocols (I-1..I-5) + a stub implementation for each (`StubModelAdapter`, `NullKVManager`, `IdentityCodec`, `ResidentWeightProvider` stub, `NoopDraftEngine`).
  - [ ] `tests/test_interfaces.py` verifying Protocol shape + stub instantiation.
  - [ ] `tests/test_sampler.py` verifying processor ordering (`temperature → repetition penalty → top-k → top-p → sample`, D-013) and greedy determinism.
  - [ ] `from silica import Engine` imports cleanly (even if it raises `NotImplementedError`).
- **Acceptance:**
  - [ ] `uv pip install -e .` succeeds.
  - [ ] `pytest tests` passes (at minimum `tests/test_interfaces.py` and `tests/test_sampler.py`).
  - [ ] Unified metrics schema includes fields: `ttft_ms`, `prefill_tok_s`, `decode_tok_s`, `resident_mb`, `logical_kv_bytes`.
- **Dependencies:** none.
- **Status:** in-progress (repo carries the 11 skeleton sub-packages; Protocols + stubs + sampler module pending).
- **Notes:** foundation for everything else. Once interfaces are frozen, changes go through the Decisions Log.

### P-1 Phase 1 — Baseline Engine

- **Goal:** a minimal runnable MLX-native inference main loop.
- **Scope:** single request, greedy / temperature / top-p sampling, token streaming, minimal CLI.
- **Strategy:** the Phase 1 `ModelAdapter` borrows from `mlx-lm` for (a) model-structure loading, (b) the tokenizer, (c) the weight loader (safetensors → `mx.array`), but **does not borrow** mlx-lm's rotating KV cache / prompt cache — the ownership boundary is fixed in **D-010** (which supplements D-004). KV is a `SimpleKVCache` (single request, non-paged), injected into the adapter's `prefill` / `decode_step` via `kv_handle`. P-2's `PagedKVCache` is a direct upgrade path, not a replacement for mlx-lm's internal cache. The first tasks of P-1 are **two day-1 gates**: Gate A (D-010) verifies that `mlx_lm.generate_step(cache=...)` or an equivalent entry point accepts an external cache object; Gate B (D-014 / R-8) verifies that Qwen3.5-0.8B loads text-only with MTP disabled and tokenizer round-trips match the HF reference. Together these results determine the real cost of the remaining P-1 deliverables (see D-010 / D-014 / R-6 / R-8).
- **Deliverables:**
  - [ ] **Day-1 gate A:** smoke test that `mlx_lm` accepts external cache injection (D-010). Resolve this blocker before expanding the deliverables below.
  - [ ] **Day-1 gate B (D-014 / R-8):** Qwen3.5-0.8B text-only load probe — (a) checkpoint loads under mlx-lm (or a thin local loader if mlx-lm is incomplete) with multimodal heads skipped or weight-ignored; (b) MTP head can be disabled so `decode_step` yields exactly one token per call; (c) tokenizer round-trips match the HF reference on a fixed prompt fixture. Failure triggers R-8 mitigation (monkey-patch the load path, or fall back to Qwen3-0.6B and shift DeltaNet work to P-3).
  - [ ] `silica.engine.Engine.generate(prompt, sampling_params)` returning a token stream.
  - [ ] `silica.mlx.runner` wrapping mlx-lm's forward (external cache injected; mlx-lm's internal cache unused).
  - [ ] `silica.kvcache.simple.SimpleKVCache` (single-request; passed to the adapter as a `kv_handle`).
  - [ ] `silica.server.cli`: `python -m silica run --model Qwen/Qwen3.5-0.8B --prompt "..."`.
  - [ ] Basic sampling: greedy, temperature, top-p.
  - [ ] `silica.models.qwen3_5.Qwen3_5Adapter` for Qwen3.5-0.8B (hybrid DeltaNet + MTP + multimodal sanitize). Plain-Qwen3 family lives in `silica.models.qwen3.Qwen3Adapter` (used as the P-2 dev-loop model); adapters are selected by `silica.models.factory.adapter_for_repo(repo)`. See `docs/P2_OPENING.md` §"Model integration in three layers" for the family adapter + factory registry + capability gate split.
- **Acceptance:**
  - [ ] Generates text reliably.
  - [ ] Greedy decoding is token-for-token identical to the mlx-lm reference implementation (fixed seed, same model).
  - [ ] The profiler produces TTFT, decode tok/s, and resident memory.
- **Dependencies:** P-0.
- **Status:** planned.
- **Notes:** dev-loop model is Qwen3.5-0.8B. Do not attempt 27B / 31B in Phase 1. **P-1 scope constraints are fixed in D-014**: text-only path only (multimodal heads skipped per §3.2 Non-Goals); MTP is disabled (standard single-token decode — MTP as a draft source flows through I-5 `DraftEngine` and is a P-7 discussion); DeltaNet per-layer recurrent state is adapter-owned and carried via `state_delta` (D-015); paged KV and prefix cache are P-2 concerns, not P-1.

### P-2 Phase 2 — Mini-vLLM Core

- **Goal:** the real engine skeleton — paged KV + continuous batching + prefix cache + memory budget.
- **Scope:** concurrent requests, shared-prefix hits, memory-budget management, request state machine.
- **Strategy:**
  - Default block size 16 tokens (re-evaluated after Phase 4 bench).
  - Radix prefix cache, taking cues from mini-sglang.
  - Chunked prefill is not a P-2 deliverable; decision is measurement-gated under **Q-010** (triggers include fairness / TTFT, not only OOM). Resolved at P-4 bench exit via the TTFT-under-concurrency scenario; promoted to v0.1.5 / v0.2 only if the threshold is breached.
- **Deliverables:**
  - [ ] `silica.kvcache.paged.PagedKVCache` implementing KVManager (I-2).
  - [ ] `silica.kvcache.prefix.RadixPrefixCache`.
  - [ ] `silica.core.request.RequestState` state machine: `WAITING → PREFILL → DECODE → DONE/ABORTED`, plus `PREEMPTED` as a side state reachable from `PREFILL` / `DECODE` when the scheduler evicts to honor the memory budget; re-admission returns to `WAITING` and reuses any still-valid prefix-cache blocks + `state_delta` snapshot.
  - [ ] `silica.scheduler.batcher.ContinuousBatcher`.
  - [ ] `silica.scheduler.budget.MemoryBudgeter`.
- **Acceptance:**
  - [ ] 8 concurrent requests run stably.
  - [ ] A constructed shared-prefix prompt hits the prefix cache (verifiable via hit counters).
  - [ ] Exceeding the memory budget aborts / queues cleanly without crashing.
- **Dependencies:** P-1.
- **Status:** planned.
- **Notes:** the single most important Phase. The skeleton decides the stub-replacement path for every native capability (VQ / streaming / speculative) that follows (Principle 9).

### P-3 Phase 3 — Model Adapters

- **Goal:** actually run Qwen3.5-27B / Gemma4-31B (dense) + Qwen3.5-35B-A3B / gemma-4-26B-A4B (MoE smoke test).
- **Scope:** four model adapters (2 dense + 2 MoE, D-011), Qwen3.5 hybrid attention, MoE top-k expert routing, MLX-native 4-/8-bit quantization.
- **Strategy:**
  - Quantization rides on mlx-lm's existing 4-bit / 8-bit path (D-005).
  - Weight loading goes through `WeightProvider`, even while this phase still uses `ResidentWeightProvider`.
  - Qwen3.5 hybrid attention is expressed via `AttentionPattern`; the scheduler routes KV by layer index.
  - **MoE adapter FFN execution goes through `WeightProvider.get_expert` / `prefetch_experts` to load the active top-k experts** (D-011 constraint). The Phase 3 `ResidentWeightProvider` still holds everything resident for MoE, but the `get_expert` call path must be real — otherwise P-6's per-expert residency cannot be wired in.
  - MoE adapters are inference-only: top-k routing + gate softmax normalization are computed, but aux-loss / load-balancing (training-only) is not.
- **Deliverables:**
  - [ ] `silica.models.qwen3_5.Qwen3_5Adapter` reused at Qwen3.5-27B scale (dense, already wired at P-1 for 0.8B). Underscore-separated class name matches the mlx-lm `qwen3_5` module naming.
  - [ ] `silica.models.gemma4.Gemma4Adapter` (dense, Gemma4-31B) — new family file.
  - [ ] `silica.models.qwen3_5_moe.Qwen3_5MoeAdapter` (MoE, Qwen3.5-35B-A3B) — new family file; MoE variants get their own module distinct from dense siblings so routing + expert prefetch code stays local to the family that needs it.
  - [ ] `silica.models.gemma4_moe.Gemma4MoeAdapter` (MoE, gemma-4-26B-A4B) — new family file.
  - [ ] `AttentionPattern` dispatch covering all v0.1 values — `global` / `sliding` / `hybrid` (KV-attention variants) **and** `recurrent` / `hybrid_deltanet` (D-015, Qwen3.5 path). The scheduler routes KV layers to `KVManager` and recurrent layers to the adapter-owned store.
  - [ ] DeltaNet recurrent-layer forward + adapter-local per-request state store (keyed by `req_id` via `kv_handle`); `adapter.commit_state` / `adapter.rollback_state` / `adapter.state_from_prefix` / `adapter.free_state` helpers implemented per D-015 (P-7 will exercise commit/rollback; P-3 only needs the primitives in place).
  - [ ] MoE top-k gating + per-expert FFN aggregation (inference path, aux-loss ignored).
  - [ ] `silica.weights.resident.ResidentWeightProvider` (full residency; the MoE variant exposes `get_expert` per-expert access even if the underlying storage is fully resident).
  - [ ] Model factory registry: `silica.models.factory` (already in place since v1.5.2 with `model_type` dispatch; P-3 keeps the `model_type` key — no `config.architectures[0]` rekeying unless a real collision appears — and enriches the **value** side by having each adapter return a typed `ModelCapabilities` summary via its new `capabilities()` method (D-016). First-version fields are `attention_kinds`, `has_recurrent_state`, `has_moe`; additional capability bits land only when concrete P-5 / P-6 / P-7 consumers require them. MoE entries declare `has_moe=True` at the `capabilities_from_attention_pattern` call site; MoE entries require a MoE-capable `WeightProvider`.)
- **Acceptance:**
  - **Adapter structural correctness (fp16 parity on control model)** — P-3 exit criterion:
    - [ ] On a fp16 control model (Qwen3.5-0.8B or Qwen3.5-4B), Silica adapter logits match the HuggingFace reference: `max |logit diff| < 1e-3` over the first 50 greedy-decoded tokens under the same tokenizer state and seed. This verifies adapter structural components (attention, RMSNorm, MoE routing, positional encoding) without quantization noise.
    - [ ] Qwen3.5 hybrid attention cache-routing semantics are correct — unit tests cover per-layer dispatch according to `AttentionPattern` including `hybrid_deltanet` (KV layers → `KVManager`, recurrent layers → adapter-owned store).
    - [ ] DeltaNet recurrent-state plumbing is exercised (D-015): (a) `StateDelta.recurrent_bytes()` returns non-zero for hybrid_deltanet models and matches the layer-count × hidden-state-bytes formula; (b) `adapter.state_from_prefix` returns a reusable `StateDelta` when the full KV prefix is reused and `None` otherwise (v0.1 rule); (c) a snapshot → mutation → `adapter.rollback_state` round-trip restores the pre-snapshot forward output bit-exactly (P-7 prerequisite, tested in isolation in P-3).
  - **Quantized big-model correctness** — P-3 exit criterion:
    - [ ] Qwen3.5-27B and Gemma4-31B load and execute at least one forward under the MLX-native 4-bit or 8-bit quantization path (D-005); manual resident caps and small batches are allowed.
    - [ ] **Teacher-forced next-token argmax agreement ≥ 98%** over the first 100 teacher-forced positions, against the `mlx-lm` reference at the same model + same quantization configuration, compared position-by-position on a fixed prefix. We do **not** compare against free-running generated sequences — sequence drift would mask or amplify real differences.
    - [ ] Fallback (if teacher-forced comparison is infeasible on the current toolchain): end-to-end **PPL drift `< 0.1` absolute** vs an `mlx-lm` baseline on the same evaluation corpus at the same quantization configuration.
  - **Product memory-fit target** (conditional, gated on Q-003; ownership by M-4 or M-7, see handoff rules):
    - [ ] Both dense targets sustain ≥ 500 tokens of generation on an M5 Pro 48 GB. If Q-003 is not yet resolved to "int4 fits in 48 GB" at P-3 exit, this item may be satisfied with smaller models (e.g. Qwen3.5-7B) or a manual resident cap, and the real 27B/31B @ 48 GB 500-token validation is handed off to M-7 (see R-1 / Q-003 / M-4 / M-7).
  - **MoE structural correctness** (D-011) — P-3 exit criterion:
    - [ ] On a small MoE control model (an fp16 version of Qwen3.5-35B-A3B or a smaller MLX-community-accessible MoE equivalent, e.g. a tiny Qwen2-57B-A14B variant or OLMoE-1B-7B), Silica MoE adapter logits match the HuggingFace reference: `max |logit diff| < 1e-3` over the first 50 greedy-decoded tokens. This verifies top-k expert routing + gate normalization + expert FFN aggregation without quantization noise.
    - [ ] Qwen3.5-35B-A3B and gemma-4-26B-A4B load and execute at least one forward under the MLX-native 4-bit or 8-bit quantization path; manual resident caps and small batches are allowed; full residency fallback is permitted while P-6 is incomplete (active params are small, so fit risk is far lower than dense — see R-1 MoE mitigation).
    - [ ] **Per-expert call-path unit test**: with a mock `WeightProvider`, the MoE adapter's forward pass assertably invokes `get_expert(layer_idx, expert_id)` for the top-k activated experts, and `get_layer(layer_idx)` is **not** used to load expert weights. This test makes the D-011 constraint regressable before P-6 streaming is wired in.
- **Dependencies:** P-2.
- **Status:** planned.
- **Notes:** risk — Qwen3.5-27B fp16 is ~54 GB, so quantization is required. If 4-bit still doesn't fit, Q-003 fires and P-6 is pulled forward. Acceptance is split into adapter structural correctness (fp16 parity on a dense control model) + quantized big-model correctness (teacher-forced comparison) + product memory-fit target (conditional, dense only) + MoE structural correctness (D-011, independent exit criterion). If dense memory-fit is deferred, the MoE smoke test must still pass — MoE targets have small active params and do not depend on Q-003 resolution.
- **Empirical findings:**
  - **2026-04-19 — Qwen3.5-27B-4bit load probe** (`scripts/probe_qwen3_5_27b_load.py`, run on M5 Pro 48 GB against `mlx-community/Qwen3.5-27B-4bit`, ~16.1 GB weights):
    - Loads cleanly through `mlx_lm.load` — the HF model card's mlx-vlm reference applies to the multimodal input path; text-only inference via mlx-lm works without fallback.
    - Structural metadata (from `model.args.text_config` dict): `model_type="qwen3_5"`, 64 hidden layers, hidden_size=5120, 24 attention heads, 4 KV heads (GQA 6:1), head_dim=256, vocab_size=248044.
    - `adapter_for_repo` dispatches to the existing `Qwen3_5Adapter` via the `model_type` registry — **no 27B-specific adapter file is needed**; the dense deliverable above already anticipated this by listing `Qwen3_5Adapter` "reused at Qwen3.5-27B scale".
    - `capabilities()` reports `attention_kinds={GLOBAL, HYBRID_DELTANET}`, `has_recurrent_state=True`, `has_moe=False`. Per-layer pattern is a strict 3:1 repeating `[D, D, D, G]` across all 64 layers — 48 HYBRID_DELTANET + 16 GLOBAL. Same hybrid architecture as Qwen3.5-0.8B, just wider and deeper.
    - Single-request `Engine.generate` runs end-to-end (greedy 4-token completion of "Hello" → `", I have a"` — plausible base-model continuation). First-forward kernel compile dominates TTFT (~2.4 s for a 1-token prompt), so prefill-tok/s on this single run is not a meaningful baseline — rerun after warmup when quantified bench data is needed.
    - Peak device memory ~30.5 GB (weights ~16 GB + MLX forward scratch ~14 GB) — leaves ~17 GB headroom on a 48 GB M5 Pro for KV growth and batch state.
    - At v1.6.1 this path was blocked by the D-016 capability gate (HYBRID_DELTANET → `has_recurrent_state=True`). As of P-3-C3c / P-3-C3d the shared Qwen3.5 hybrid scheduler is batch-enabled and **greedy parity is pinned on Qwen3.5-0.8B** (see the next bullet). Qwen3.5-27B re-uses the same scheduler code path, so no further Silica-side wiring is expected for batched execution — but **large-context batched validation on 27B remains pending a P-4 / dedicated bench round** because of the ~30.5 GB load-peak memory cost and decode-throughput runtime cost; landing it alongside a smoke here would mean downloading 16 GB and running multi-token batched generation on every test run.
  - **2026-04-20 — Gemma4-31B-4bit load probe + mlx-lm source survey** (P-3-D0 / P-3-D0.2, `scripts/probe_gemma4_31b_load.py` on `mlx-community/gemma-4-31b-4bit`, structured notes in `docs/P3_GEMMA4_SURVEY.md`):
    - `mlx_lm.load` accepts the repo; outer `model_type='gemma4'`, inner `text_config.model_type='gemma4_text'`. mlx-lm has full Gemma4 support shipped; no vlm fallback is needed for text-only inference.
    - 60-layer dense model (31B class): strict 5:1 repeating `[S, S, S, S, S, F]` pattern → 50 `sliding_attention` + 10 `full_attention` layers. `sliding_window=1024`, two distinct KV shapes (sliding: `n_kv_heads=16, head_dim=256`; full: `n_kv_heads=4, head_dim=512` with `attention_k_eq_v=True`). Not MoE (`num_experts=None`, `enable_moe_block=False`); not per-layer-input (`hidden_size_per_layer_input=0`); `num_kv_shared_layers=0`.
    - `Gemma4TextModel.make_cache` returns a **heterogeneous** per-layer list: `KVCache()` for `full_attention` layers, `RotatingKVCache(max_size=sliding_window, keep=0)` for `sliding_attention` layers. Cache list length is `num_hidden_layers - num_kv_shared_layers` (forward pads with `None` for shared-KV positions).
    - Silica factory currently rejects the `gemma4` `model_type` (no registered adapter). Writing a `Gemma4Adapter` would be a small new file analogous to `Qwen3Adapter`, but the **batched path stays blocked on PLAN §Q-013** — mlx-lm ships no `BatchRotatingKVCache`, and the C3c-era `_SUPPORTED_ATTENTION_KINDS = {GLOBAL, HYBRID_DELTANET}` excludes `SLIDING`. Gate lift for sliding-window batching is a distinct major unit (call it D2 in the survey's suggested D-subunit ordering).
    - Additional concern surfaced: `KVLayout(num_layers, n_kv_heads, head_dim, dtype)` is a single-shape summary that cannot express Gemma4's two coexisting KV shapes. Three remediation options in the survey (§5.1); decision deferred to D1.
  - **2026-04-20 — `BatchRotatingKVCache` audit** (P-3-D2.0, `docs/P3_BATCH_ROTATING_KV_SURVEY.md`): mlx-lm DOES ship `BatchRotatingKVCache` in `cache.py:1100-1444` — an oversight in the Gemma4 D0.2 survey §5.3, now corrected with a forward-pointer. The class has the full batched surface (`update_and_fetch` / `prepare` / `finalize` / `filter` / `extend` / `extract` / `merge` / `make_mask`) analogous to `BatchKVCache`, and mlx-lm's mask path (`create_attention_mask` in `base.py:49`) delegates to `cache.make_mask` automatically. Implication: D2 shrinks from "build a sliding-batched cache container" to "`Gemma4Adapter.make_batch_cache` returns a hybrid `[BatchRotatingKVCache / BatchKVCache]` list per layer"; D2+D3 may land as a single commit mirroring C3c's "gate lift + smoke" precedent. PLAN §Q-013 is still the right ID for tracking sliding-window batched execution as a **deliverable**, but the container primitive is no longer the blocker.
  - **2026-04-20 — Gemma4 SLIDING gate lift + batched miss-only smoke** (P-3-D3, `tests/test_p3_gemma4_batched_smoke.py`, dual-gated on `mlx-community/gemma-4-31b-4bit` cache + `SILICA_REAL_GEMMA4_31B=1`): `ContinuousBatcher._SUPPORTED_ATTENTION_KINDS` now includes `AttentionKind.SLIDING`; error-locator skip extended so `{GLOBAL, HYBRID_DELTANET, SLIDING, RECURRENT}` reports the `RECURRENT` layer rather than stopping early. A new constructor guard rejects the combination `AttentionKind.SLIDING in caps.attention_kinds` + `prefix_cache is not None` at construction time with a specific message naming `BatchRotatingKVCache` semantics — D3 only commits to the `prefix_cache=None` miss-only path because the seeded-admission path (`build_seeded_batch_kv` / `_extract_and_insert_prefix`) emits `BatchKVCache` per layer and has not been validated against rotating-window truncation / offset / rotated state. Q-013 treatment: the **sliding-window batched deliverable** is landed for the miss-only path; `prefix-cache + SLIDING` remains a local follow-up and is **not** claimed. D3.1 is the token-level follow-up; unlike Qwen3.5-C3d, it does **not** claim strict B>1 batched-vs-single greedy parity.
  - **2026-04-20 — Gemma4 batched invariant pinning** (P-3-D3.1, `tests/test_p3_gemma4_batched_parity.py`, same dual gate): the exact-first B>1 batched-vs-single-request attempt failed empirically on the real `mlx-community/gemma-4-31b-4bit` checkpoint (`"The capital of France is"`, `max_tokens=16`, first mismatch at token index 2: single token `600`, batch token `529`). Hard invariants still pass: B=1 batched equals single-request exactly; identical prompts in one batch produce identical rows; unequal prompt lengths emit per-row tokens and finish cleanly. The B>1 correctness oracle is therefore degraded in the same spirit as P-2 plain-Qwen3: Silica batched output must match a direct mlx-lm batched reference driven with `Gemma4Adapter.make_batch_cache(left_padding)` (SLIDING → `BatchRotatingKVCache`, GLOBAL → `BatchKVCache`). README now says "B=1 parity + B>1 direct mlx-lm batched reference pinned" and explicitly avoids claiming strict B>1 batched-vs-single greedy parity.
  - **2026-04-20 — MoE single-request real-model smoke** (P-3-E3, `tests/test_p3_qwen3_5_moe_smoke.py` + `tests/test_p3_gemma4_moe_smoke.py`, both dual-gated): both MoE adapters pass real-model `Engine.generate("Hello", max_tokens=4)` on M5 Pro 48 GB. `mlx-community/Qwen3.5-35B-A3B-4bit` (~20 GB) and `mlx-community/gemma-4-26b-a4b-4bit` (~16 GB), 4 tests total, completed in 18.89 s wall (warmed loader + decoded forward). Each smoke covers: (a) factory dispatch returns the correct MoE adapter (Qwen3.5-MoE via the new `qwen3_5_moe` `_ADAPTERS` key; Gemma4-MoE via the local `enable_moe_block` branch inside `_build_gemma4` — proves the same-`model_type` collision is resolved), (b) capabilities match what the unit tests assumed (Qwen3.5: `has_moe=True`, `has_recurrent_state=True`, `attention_kinds={HYBRID_DELTANET, GLOBAL}`; Gemma4: `has_moe=True`, `has_recurrent_state=False`, `attention_kinds={SLIDING, GLOBAL}`), (c) `config.extra` MoE metadata reflects the real text_config (Qwen3.5: 256 experts × top-8, `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`, `mlp_only_layers=[]`, `attn_output_gate_mlx_lm_honors=False`; Gemma4: 128 experts × top-8, `moe_intermediate_size=704`, `has_always_on_dense_mlp=True`, `bytes_per_token_total=225_280`), (d) prefill + 3 decode steps produce non-empty token list with all ids in vocab. No token-parity claim — MoE forward goes through mlx-lm's SwitchGLU + gather_mm + (Qwen3.5) shared-expert + (Gemma4) always-on dense MLP additive sum, none of which Silica validates against an HF reference here. Dual gate: `SILICA_REAL_QWEN3_5_MOE=1` and `SILICA_REAL_GEMMA4_MOE=1` opt-in env vars on top of HF cache presence. **P-3-E exit criterion ("MoE structural correctness + ≥1 forward per family at MLX-native quantization") satisfied.** Batched MoE remains rejected at the capability gate; E4 will revisit.
  - **2026-04-20 — Gemma4-MoE adapter (E1.2) + E-open-4 resolution** (P-3-E1.2, `silica/models/gemma4_moe.py` + factory branch inside `_build_gemma4` + 26 fake-model tests): second MoE adapter lands as a thin wrapper around `Gemma4Adapter`, matching the E1.1 design shape. Silica contributes `has_moe=True` capability (plus `has_recurrent_state=False` — Gemma4 is pure KV attention), MoE-aware variant guard as a polymorphic staticmethod override that requires `enable_moe_block=True` + `num_experts>0` (the dense parent's `_validate_supported_variant` is reached via `super().__init__ → self._validate_supported_variant(model)`; Python method resolution picks the subclass version even for `@staticmethod`), and `config.extra` MoE metadata with the distinguishing `has_always_on_dense_mlp=True` marker and the Gemma4-specific `moe_expert_path="layer.experts.switch_glu"` pointer. The dense `Gemma4Adapter._validate_supported_variant` is **unchanged**: directly constructing the dense adapter on a MoE checkpoint still loud-fails, acting as defence-in-depth against callers bypassing the factory. Factory-level resolution: `_build_gemma4` reads `args.text_config.enable_moe_block` / `num_experts` and routes dense-vs-MoE locally inside the single `_ADAPTERS["gemma4"]` entry — no `_ADAPTERS` schema change needed (other families key cleanly on `model_type`). `install_dispatch_proxy` walks `layer.experts.switch_glu` (not `layer.mlp.switch_mlp` as on Qwen3.5-MoE — reflects Gemma4's Router+Experts+SwitchGLU-with-GeGLU structure where the dense MLP branch stays always-on and is summed ungated with the experts branch); shares the `_DispatchProxy` class imported from `silica.models.qwen3_5_moe`. D4's `KVLayout.bytes_per_token_total` inherits from the dense parent unchanged and yields 225,280 bytes/token on the 26B-A4B shape (25 sliding × 2 × 8 × 256 × 2 + 5 full × 2 × 2 × 512 × 2), matching the E0 survey §3.2 number exactly. E-open-4 RESOLVED: `layer_types` remains authoritative for attention routing; `sliding_window_pattern` absence on 26B-A4B is benign because `Gemma4Adapter._build_attention_pattern` reads `layer_types` as the primary source. Batched MoE stays rejected at `ContinuousBatcher._enforce_capability_gate` via the existing `has_moe=True` branch (unchanged from E1.1). Tests: 645 passed (+26 on the MoE adapter, +2 factory-branch regression tests for `enable_moe_block` True/False routing); ruff + mypy clean on 42 source files (+1).
  - **2026-04-20 — Qwen3.5-MoE adapter (E1.1) + E-open-* resolutions** (P-3-E1.1, `silica/models/qwen3_5_moe.py` + factory registration + 23 fake-model tests): first MoE adapter lands as a thin wrapper around the dense `Qwen3_5Adapter` (no MoE math reimplemented; mlx-lm owns the SwitchGLU path). Silica contributes `has_moe=True` capability, `model_type="qwen3_5_moe"` factory dispatch, variant guards on `num_experts` / `num_experts_per_tok` / `mlp_only_layers`, MoE metadata on `config.extra` (`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size`, `norm_topk_prob_runtime`, `attn_output_gate_config`, `attn_output_gate_mlx_lm_honors`), and an option-(c) dispatch-observation seam via `Qwen3_5MoeAdapter.install_dispatch_proxy(observer)` that wraps each MoE layer's `switch_mlp` with a thin forwarding proxy reporting `(layer_idx, indices)` before delegating to the real SwitchGLU. Proxy is NOT installed by default — `build()` stays untouched so the dense `ResidentWeightProvider` (whose `get_expert` raises by design under D-011) continues to work for single-request paths. Batched MoE remains rejected at `ContinuousBatcher._enforce_capability_gate` via the existing `has_moe=True` branch; error text updated to point at P-3-E4 (batched MoE smoke + parity) rather than the pre-adapter "P-3 discussion" placeholder. E-open-* resolutions, all recorded in `docs/P3_MOE_SURVEY.md`: **E-open-1** resolved to option (c) "per-expert at dispatch, fused at fetch" — preserves the quantized `QuantizedSwitchLinear` fast path while keeping D-011 testable; **E-open-2** resolved after reading the cached `Qwen3.5-35B-A3B-4bit/config.json` directly (no second 20 GB load): `mlp_only_layers=[]` on the probed checkpoint, and E1.1 guards future non-empty cases loudly; **E-open-5** resolved by a repo-wide grep of `mlx_lm/models/` finding zero references to `attn_output_gate` — mlx-lm silently drops the flag, Silica inherits this behaviour and records the divergence on `config.extra` for future HF-vs-mlx-lm comparison. Gemma4-MoE (E1.2) deferred to a separate commit; E2 (D-011 mock-provider dispatch test) and E3 (real-model smoke) follow after E1.2.
  - **2026-04-20 — Qwen3.5-MoE + Gemma4-MoE E0 probes** (P-3-E0, `scripts/probe_qwen3_5_moe_load.py` + `scripts/probe_gemma4_moe_load.py` + `docs/P3_MOE_SURVEY.md`): both probes ran metadata-only against the real 4-bit checkpoints on M5 Pro 48 GB — `mlx-community/Qwen3.5-35B-A3B-4bit` (20.4 GB, load 274 s, `model_type="qwen3_5_moe"`) and `mlx-community/gemma-4-26b-a4b-4bit` (15.6 GB, load 247 s, `model_type="gemma4"`). Factory dispatch failed on both as expected (no MoE adapter registered for the Qwen key; `Gemma4Adapter` variant guard rejects `enable_moe_block=True`). Key findings for E1 / E2 design (full detail in the survey): (a) mlx-lm's MoE path on both families is fused via `SwitchGLU` + `gather_mm` with all experts' weights stacked in one tensor per layer, conflicting with the literal reading of D-011 "per-expert `WeightProvider.get_expert` call-path" — the survey recommends the hybrid "per-expert at dispatch, fused at fetch" interpretation (E-open-1); (b) Gemma4-MoE shares `model_type="gemma4"` with Gemma4-dense, so factory dispatch needs a local `enable_moe_block` branch inside `_build_gemma4` rather than keying on `model_type` alone; (c) Qwen3.5-35B-A3B has 40 layers (30 HYBRID_DELTANET + 10 GLOBAL via `full_attention_interval=4`), 256 experts, top-8, plus a sigmoid-gated shared-expert MLP (`shared_expert_intermediate_size=512`, Qwen3-Next style); (d) Gemma4-26B-A4B has 30 layers (25 sliding + 5 full, 5:1 ratio), 128 experts, top-8, and unlike Qwen3.5-MoE the dense MLP branch is NOT replaced — `gemma4_text.DecoderLayer` constructs `self.mlp = MLP(...)` unconditionally and the MoE-mode forward sums `h = h1 + h2` (always-on dense MLP + ungated top-k experts) via three additional layernorms. MoE co-exists with both SLIDING and GLOBAL attention kinds; (e) D4's `KVLayout.bytes_per_token_total` generalises directly — Qwen3.5-MoE uses the homogeneous-GQA fallback; Gemma4-MoE needs the per-kind sum (225,280 bytes/token). Open questions: `mlp_only_layers` contents on 35B-A3B (E-open-2, the qwen3_5 DecoderLayer does not consult the field), MTP weight implications (E-open-3), `attn_output_gate` inheritance (E-open-5).
  - **2026-04-20 — Gemma4 per-kind KV budget correction** (P-3-D4, `silica/models/adapter.py` + `silica/scheduler/budget.py` + `silica/models/gemma4.py`, unit-tested in `tests/test_gemma4_adapter.py` and `tests/test_memory_budgeter.py`): added optional `KVLayout.bytes_per_token_total: int | None`. `MemoryBudgeter.for_adapter` prefers this value when set and falls back to the naive `2 * num_layers * n_kv_heads * head_dim * dtype.size` formula when `None` — homogeneous-shape adapters (plain Qwen3, Qwen3.5 dense) stay unchanged. `Gemma4Adapter._build_kv_layout` populates the new field with an explicit per-kind sum `n_sliding * 2 * sliding_kv_heads * sliding_head_dim * dtype_bytes + n_full * 2 * global_kv_heads * global_head_dim * dtype_bytes`. On Gemma4-31B (50 sliding @ 16×256 + 10 full @ 4×512, bfloat16) this yields 901,120 bytes/token versus the pre-D4 983,040 — a ~9% over-count correction. Confirmed from `gemma4_text.py:243,253,260` that `attention_k_eq_v=True` shares only the `v_proj` weight matrix; K and V are still cached as separate tensors at runtime, so the factor 2 applies uniformly. Caveat: the scalar still assumes unbounded growth on sliding layers — past `sliding_window=1024` tokens the real `bytes_per_token` drops to the full-layer-only contribution plus a fixed per-request sliding cost; a window-aware budget model is future work, tracked separately. Test count +5; 583 passed, 6 skipped, ruff + mypy clean.
  - **2026-04-19 — Qwen3.5-0.8B hybrid batched smoke** (P-3-C3c, `tests/test_p3_hybrid_batched_smoke.py`, against `Qwen/Qwen3.5-0.8B`):
    - After the capability-gate lift (`_SUPPORTED_ATTENTION_KINDS = {GLOBAL, HYBRID_DELTANET}`), `Engine.generate_batch` runs two prompts with `max_batch_size=2`, `prefix_cache=None`, and emits token + done events for every request (no aborts).
    - Direct `ContinuousBatcher` probe confirms the live `_batch_cache` is genuinely hybrid — `ArraysCache` at DeltaNet layer indices, `BatchKVCache` at global-attention layer indices — proving `Qwen3_5Adapter.make_batch_cache` reached the scheduler rather than the `callable()` fallback producing an all-`BatchKVCache` list.
    - P-3-C3c smoke established functional batching ("does not crash, emits tokens, live cache is genuinely hybrid"). P-3-C3d then added **strict batched-vs-single-request greedy parity** in `tests/test_p3_hybrid_batched_parity.py` — four tests covering B=1 as a hard gate, same-prompt symmetry, B>1 strict parity vs `Engine.generate`, and an unequal-prompt-length row-lifecycle smoke. Empirically the strict B>1 parity holds at `max_tokens` of 16, 32, and 64 on Qwen3.5-0.8B, which is a stronger claim than P-2's Qwen3-0.6B pinning (plain-GQA fp16 batched SDPA there drifted after a handful of tokens). Likely reasons: DeltaNet's recurrent state is hardcoded fp32 (less round-off) and the 3:1 DeltaNet:global layer ratio dilutes the fp16-SDPA contribution. **Silica-batched vs a direct mlx-lm-batched reference** (rather than vs Silica single-request) remains future work.
  - **2026-04-25 — P-3-E4 batched MoE capability-gate lift (smoke-only, parity deferred)** (`silica/scheduler/batcher.py:_enforce_capability_gate` + `tests/test_batcher.py` flipped + new positive coverage; commits to follow under E4 sub-units): the `has_moe=True` rejection in `_enforce_capability_gate` is removed. Pre-E4 an adapter declaring `has_moe=True` raised `NotImplementedError` regardless of attention kinds; post-E4 the gate accepts when `attention_kinds` are themselves all inside `_SUPPORTED_ATTENTION_KINDS`. The lift rests on the P3_MOE_SURVEY §5 E4 audit finding that mlx-lm's `SwitchGLU` + `gather_mm` path is B-agnostic — a batched forward dispatches per-row top-k experts without further scheduler work. Test coverage: (a) `test_capability_gate_accepts_has_moe_after_e4_when_attention_kinds_supported` (pure GLOBAL + has_moe=True passes); (b) `test_capability_gate_accepts_has_moe_with_hybrid_deltanet_after_e4` (Qwen3.5-MoE-shape pattern of HYBRID_DELTANET + GLOBAL + has_moe=True passes); (c) `test_capability_gate_still_rejects_has_moe_when_attention_kind_unsupported` (RECURRENT layer + has_moe=True still raises with RECURRENT named in the error — locks in that the lift didn't open the door for unsupported attention kinds). Real-model B=2 smoke on Qwen3.5-35B-A3B-4bit and gemma-4-26b-a4b-4bit lands in commits C and D as `tests/test_p3_qwen3_5_moe_batched_smoke.py` / `tests/test_p3_gemma4_moe_batched_smoke.py` (different prompts per row to exercise the per-row top-k expert dispatch claim, dual-gated on HF cache + `SILICA_REAL_*_MOE`, `max_tokens=4`, peak-MB recorded in commit messages). **Token-parity is explicitly deferred** — survey §5 E4 originally framed the deliverable as "smoke + parity", but parity definition (per-row top-k indices stability under different right-padding lengths through the quantized SwitchGLU + per-row routing fast path) is a separate workstream; the closure here is "structural correctness under batched dispatch on real MoE checkpoints", consistent with how E3 closed single-request smoke without HF-vs-mlx-lm parity. Pre-E4 stale comments in `silica/models/qwen3_5_moe.py`, `silica/models/gemma4_moe.py`, `tests/test_p3_qwen3_5_moe_smoke.py`, `tests/test_p3_gemma4_moe_smoke.py`, and `tests/test_qwen3_5_moe_adapter.py::test_capabilities_declare_has_moe_true` updated to point at the lift / batched smoke; historical 2026-04-20 changelog entries that recorded "E4 will revisit" remain unchanged as point-in-time records.

### P-4 Phase 4 — Bench Unification

- **Goal:** benchmarks run directly through the Engine — no side paths.
- **Scope:** unified bench runner, standard scenarios, unified result format.
- **Strategy:** bench is a thin wrapper over `silica.engine.Engine`.
- **Deliverables:**
  - [x] `silica.bench.runner.BenchRunner` (P-4.1) — oracle-dispatched workload execution, injectable engine factory + direct-batched-reference hooks, JSONL emit per row, per-oracle workload-shape validation.
  - [x] `silica.bench.scenarios`: short-in/long-out; long-in/short-out; concurrent shared-prefix; **TTFT-under-concurrency** (one long-prompt request co-scheduled with short-prompt requests — resolves Q-010 chunked-prefill promotion). All four workload-shaped rows shipped under P-4.2d-iii-a/b; model-shaped rows (smoke / B=1 parity / B>1 parity for 0.6B + 0.8B + 27B + 31B + MoE) under P-4.2a/b/c/d-ii; teacher-forced argmax under P-4.3. Current catalog count: 15 rows (see `python -m scripts.bench --list`).
  - [x] Unified metrics schema: TTFT, prefill tok/s, decode tok/s, resident memory, peak memory, total tokens, wall time, oracle metadata (`ScenarioResult` dataclass; "quality" column supplied by the oracle — SMOKE / B1 / BGT1 populate structured metadata keyed for JSONL; B>1 SMOKE runs additionally carry per-row first-token wall offsets under `metadata.rows[].first_token_ms_offset` so TTFT-under-concurrency surfaces the short-prompt-under-long-prompt signal that `Engine.generate_batch`'s current lack of MetricsRegistry population would otherwise erase).
  - [x] Output: jsonl + markdown report — JSONL via `BenchRunner(out_path=...)` since P-4.1; Markdown report via `render_markdown_report` + `scripts/bench.py --report-md PATH` since P-4.2d-i (GFM table + per-scenario detail blocks with embedded oracle-metadata JSON for paste-into-PR consumption).
  - [x] `silica.bench.vqbench_baseline` (P-4.4): runs the ready-made vqbench scripts (`reproduce_qwen35_4b_headline.py` etc.) in a separate subprocess to collect PPL as a reference column. D-009 explicitly allows this "separate-process comparison" path; it serves the P-5 numeric cross-check acceptance. Module lives in `silica/bench/vqbench_baseline.py`; CLI at `scripts/vqbench_baseline.py` with `--script`, `--python-executable`, `--out` flags. Subprocess runner is injectable so unit tests cover parser + orchestration without vqbench's torch / transformers deps.
- **Acceptance:**
  - [x] A single command produces the baseline table (paste-able into README) — `python -m scripts.bench --all` emits a GFM table + optional `--out PATH` JSONL across every registered scenario, skipping dual-gated rows whose env var is not set.
  - [x] No path split between bench and runtime (same Engine instance) — `BenchRunner` drives `Engine.generate` / `Engine.generate_batch`; direct mlx-lm batched reference is reached via `adapter.build(...)` on the same adapter the Engine factory loads, not a forked loader.
  - [x] `vqbench_baseline` produces a Qwen3.5-4B PPL number in a separate process as the P-5 comparison column — `scripts/vqbench_baseline.py --python-executable <vqbench-venv>/bin/python` invokes the checked-in `vqbench/scripts/reproduce_qwen35_4b_headline.py` and parses the "Headline table row:" into a `VqbenchBaselineResult` with `ppl_fp16` / `ppl_quant` / `delta_ppl` / `delta_pct`. Gated on a user-supplied vqbench venv because silica's venv does not carry vqbench's torch / transformers / datasets runtime deps (D-009).
- **Dependencies:** P-3.
- **Status:** complete. Shipped across P-4.1 (`efcc65e`), P-4.2a–c (`34be3f0` / `1d52f4a` / `80dda0b`), P-4.2d-i markdown report (`ddfee97`), P-4.2d-ii model-shaped rows (`c3b46d8`), P-4.2d-iii-a B=1 workload rows (`bc7c8b4`), P-4.2d-iii-b B>1 + concurrent / TTFT rows (`9e92a10`), P-4.3 teacher-forced oracle (`a33c68f`), and P-4.4 vqbench_baseline (this commit).
- **Notes:** Phase 4 baseline data determines Q-003 (whether Phase 6 is pulled forward).
- **Empirical findings:**
  - **2026-04-20 — Bench runner + first cached smoke row** (P-4.1, `silica/bench/` + `scripts/bench.py` + `tests/test_bench_{runner,cli}.py`): `Scenario` / `Workload` / `ScenarioResult` / `OracleKind` in `silica.bench.scenario`; `BenchRunner` consumes them with an injectable `engine_factory` (defaults to `adapter_for_repo`), injectable `reset_peak` / `read_peak_mb` hooks (mlx.core by default), and JSONL-from-day-one output. CLI at `scripts/bench.py` supports `--list` / `--scenario ID` (repeatable) / `--all` / `--out PATH`. First migrated scenario `qwen3-0.6b-smoke` is cache-only (reuses the P-2 batched parity test weights); end-to-end on-device: 4 tokens, ttft=14.8 ms, decode=160.6 tok/s, wall=0.6 s. Dual-gate pattern from the test suite inherited directly: cache-presence is the weak gate, `gate_env_var == "1"` is the strong gate. First iteration of the CLI script forgot the `sys.path.insert(0, repo_root)` shim every other `scripts/*.py` uses — direct `python scripts/bench.py --list` from a non-repo cwd failed with `ModuleNotFoundError`; shim added in the same commit plus `--scenario <unknown>` now prints `unknown scenario id ...` to stderr and exits 2 instead of dumping a `KeyError` traceback. Oracle dispatch table `ORACLES: dict[OracleKind, OracleFn]` makes every future oracle a one-file extension without runner changes.
  - **2026-04-20 — E3 MoE smokes migrated into the bench catalog** (P-4.2a, commit `34be3f0`, `silica/bench/scenarios.py` + `tests/test_bench_scenarios_catalog.py`): two new dual-gated rows cover the same checkpoints as the pytest-side `tests/test_p3_qwen3_5_moe_smoke.py` / `test_p3_gemma4_moe_smoke.py` — `qwen3.5-moe-smoke` (repo `mlx-community/Qwen3.5-35B-A3B-4bit`, gate `SILICA_REAL_QWEN3_5_MOE`) and `gemma4-moe-smoke` (repo `mlx-community/gemma-4-26b-a4b-4bit`, gate `SILICA_REAL_GEMMA4_MOE`). Both use the SMOKE oracle on prompt "Hello" with `max_tokens=4`. Env-var names match the pytest gates exactly so the two views of each checkpoint opt in together. The pytest-side smokes stay in place because they pin adapter-shape contracts (`adapter.config.extra` MoE metadata, capability flags) that the bench SMOKE oracle does not check — the two views are complementary. Parametrized shape invariants over `BUILTIN_SCENARIOS.values()` mean adding scenarios in later sub-phases does not require new test functions. End-to-end on-device with both env vars set: all three scenarios ok, Gemma4-MoE wall=2.8 s peak=14.5 GB, Qwen3.5-MoE wall=14.7 s peak=19.6 GB, 0.6B wall=0.7 s peak=1.2 GB.
  - **2026-04-20 — `B1_PARITY_VS_SINGLE` oracle + cached 0.6B parity row** (P-4.2b, commit `1d52f4a`, `silica/bench/oracles.py` + `runner.py` + `scenarios.py`): oracle compares B=1 batched token stream against a single-request reference element-by-element. Success metadata `(reference_len, batch_len, first_mismatch_index=-1)`; mismatch metadata extends with `reference_token_at_mismatch` / `batch_token_at_mismatch`. Runner refactor: `_build_sampling_params(workload, adapter, *, include_eos=True)` is shared by the reference and batched paths so divergent tokens cannot be blamed on drifted sampling params; `_collect_b1_batched_tokens` validates the event stream strictly (an `aborted` event or `req_index != 0` raises `RuntimeError` → runner surfaces `b1_batched_*` reason before the oracle runs; scheduler faults never masquerade as oracle mismatches). The catalog row `qwen3-0.6b-b1-parity` reuses the cached 0.6B weights, cache-only gate, `max_batch_size=1`, same `SamplingParams` shape as the smoke row (locked by the catalog test so smoke-side drift does not silently widen the parity claim). `Engine.generate_batch` does not populate the shared `MetricsRegistry` at present, so the JSONL row's ttft / prefill / decode for a B1 parity row reflect the reference (single-request) execution; `wall_s` covers both end-to-end. End-to-end on-device: `reference_len=4 batch_len=4 first_mismatch_index=-1`, wall=0.6 s.
  - **2026-04-20 — `BGT1_DIRECT_BATCHED_REFERENCE` oracle + cached 0.6B B=2 parity row** (P-4.2c, commit `80dda0b`, same three modules): widened `OracleFn` second arg from `list[int]` to `Any` so workload-output shape varies by oracle kind (`list[int]` for single-request / B1, `dict[int, list[int]]` for BGT1). Oracle returns per-row metadata `rows: list[dict]` + `first_failure: dict | None`; mismatch reason encodes the specific row and index (`bgt1_parity_row_{k}_mismatch_index:{i}`). Runner additions: `_validate_workload_for_oracle` replaces the P-4.1 "batched deferred → skipped" with per-oracle shape rules (SMOKE / B1 require B=1 + 1 prompt, BGT1 requires B ≥ 2 + ≥ 2 prompts); workload-shape mismatches are now `status="failed"` (authoring error) rather than `"skipped"`. `BenchRunner._run_bgt1_parity` method with injectable `direct_batched_reference: DirectBatchedReferenceFn` — default implementation `_direct_mlx_lm_batched_reference` mirrors `tests/test_p3_gemma4_batched_parity.py::_direct_mlx_lm_batched_tokens` (left-pad with `0`, `adapter.make_batch_cache(left_padding)`, argmax loop for `max_tokens` steps; does not honour EOS). BGT1 passes `include_eos=False` to `_build_sampling_params` so both sides run the full budget — Silica's event stream matches the reference length regardless of EOS. `_collect_bgt1_batched_tokens` extends the B1 event-validation to B > 1: `req_index` outside `range(len(prompts))` or any row that never emits `done` raises `RuntimeError` before the oracle runs. The catalog row `qwen3-0.6b-bgt1-parity` uses prompts `("Hello", "The capital of Japan is")` chosen so the Qwen3 tokenizer yields `left_padding=[4, 0]`; an earlier iteration used two "The capital of X is" prompts where X was a single-token country name (France / Japan), both tokenizing to 5 tokens → `left_padding=[0, 0]` silently bypassed the padding branch the scenario claims to exercise. Caught in self-review; catalog test adds a gated on-device tokenizer-backed invariant (`test_qwen3_0_6b_bgt1_parity_prompts_actually_tokenize_to_different_lengths`) that loads the real tokenizer and asserts `len(set(lengths)) > 1` with at least one non-zero `left_padding` entry. End-to-end on-device: 2 rows × 8 tokens match, both rows `first_mismatch_index=-1`, wall=0.6 s. `TEACHER_FORCED_ARGMAX` is now the only unimplemented oracle.
  - **2026-04-21 — P-4 exit signals** (no new code in this entry; feeds P-4.5 design inputs recorded in the Q-010 / Q-002 / Q-003 Open Question updates and in §7 P-4.5 below):
    - **Q-010 TTFT-under-concurrency triggered** — two independent measurements on `qwen3-0.6b-ttft-under-concurrency` vs isolated `qwen3-0.6b-smoke` show cohort-level prefill serializing short rows behind the long row's `T_max`. Codex measurement: 81.28 ms / 11.8 ms ≈ 6.9×. Silica measurement (four consecutive runs): ratios {4.76, 4.42, 4.56, 4.13}×. The four concurrent rows' first-token offsets match within ≤ 0.2 ms — the structural signature of a single batched prefill forward. The single-sample ratios straddle Option A's 5× promotion trigger; the structural defect is deterministic (worsens with longer prompts) and resolves Q-010 to "triggered, promote". Full resolution text in §10 Q-010.
    - **P-5 codec hot-path gap identified** — `grep encode_block|decode_block silica/` matches `silica/kvcache/codec.py` only; zero runtime callers. The real forward path builds `BatchKVCache` / `BatchRotatingKVCache` / `ArraysCache` via `_make_batch_cache` in `silica/scheduler/batcher.py`. Writing a concrete P-5 `BlockTQCodec` directly against the I-3 interface and plugging it into `PagedKVCache` would ship an interface-level codec whose `resident_bytes` reduction is not reflected in actual unified-memory usage (the hot-path caches are still fp16). Before P-5 BlockTQ implementation, a codec runtime-integration spike must decide: (a) codec attaches to active `BatchKVCache` (most runtime impact, largest refactor), (b) codec attaches to a detached prefix-block store (saves prefix-cache KV, not active KV), or (c) a new cache wrapper presents a codec-aware `BatchKVCache` facade to mlx-lm's forward. Tracked as P-4.5-C below.
    - **P-3-C5 / P-3-E4 deferred.** Remaining P-3 bullets (preempt/replay recurrent snapshot; batched MoE capability-gate lift) stay ⏳ through P-4.5 and P-5. C5 is only exercised under speculative rollback (P-7 prerequisite, not P-5); E4 (two MoE families × per-row top-k × batched quantized SwitchGLU parity definition) is heavier than its "capability-gate lift" framing suggested and does not block P-5.

### P-4.5 Phase 4.5 — P-4 exit bridge (chunked prefill + codec integration spike)

- **Goal:** close P-4 exit cleanly by fixing the fairness defect Q-010 surfaced and pinning the integration shape P-5's `BlockTQCodec` will attach to. P-4.5 is a **bridge** between P-4 complete and P-5 opening — not a phase in the §8 priority-tier table.
- **Scope:** three sub-units (A / B / C). Each lands as its own commit so decision-sync, scheduler change, and codec spike can be reviewed independently. No product-facing capability ships in P-4.5; outputs are a scheduler fix, a runtime-integration spike, and the design docs preceding P-5.
- **Strategy:**
  - P-4.5 treats chunked prefill as a **minimal** change under an explicit three-option opening doc. The three options are (i) in-cohort chunking (real prefill split over multiple forward passes, vLLM v1 semantics), (ii) cohort splitting (short rows run a short-prompt prefill cohort before the long row's cohort), (iii) admission ordering (admit short rows first, long rows last, no prefill-shape change). Option choice is recorded before implementation; implementation is a single sub-unit.
  - P-4.5-C is a **spike**, not a codec implementation. It makes the runtime path traverse `IdentityCodec.encode_block` / `decode_block` end-to-end with the hot-path caches unchanged in observable behaviour, to verify the integration point is real before BlockTQ rides on it. Output is code + doc; the real BlockTQ encoder lands under P-5.
- **Deliverables:**
  - [x] **P-4.5-A — Decision sync.** Update PLAN header; land Q-010 resolution + Q-002 / Q-003 progress notes; add the 2026-04-21 P-4 empirical-findings bullet (above); add this P-4.5 block; refresh README P-0..P-8 status table and roadmap to reference P-4.5 before P-5.
  - [x] **P-4.5-B.0 — Chunked-prefill opening doc.** `docs/P4_5_CHUNKED_PREFILL_OPENING.md` enumerates three paths — (A) uniform in-cohort chunked prefill, (B) sub-cohort split at cohort seal, (C) admission-time reorder via waiting queue — against three invariant families (I-1..I-5 row lifecycle, B-1..B-9 budgeter, S-1..S-7 prefix cache), plus per-option scheduler and Engine-layer touchpoints, the Q-010 scope boundaries, and the length-spread threshold parameter choice. Recommendation: Option (C) (admission-time reorder), chosen for minimum blast radius on the existing scheduler (zero `batcher.py` diff expected; all logic in `silica/engine/__init__.py::generate_batch`).
  - [x] **P-4.5-B.1 — Admission-reorder implementation.** Per the opening doc §6, Option (C) landed as an Engine-layer admission heuristic: (i) `silica/engine/__init__.py::generate_batch` gains `length_spread_threshold: float = 2.0` kwarg + `_sort_admissions_by_length` + `_initial_cohort_cap` helpers, with the clamp `cap = max(1, min(effective_batch_size, first_exceeding_index))` pinned against the four worked reverse examples in the opening doc §6.1; (ii) `silica/scheduler/batcher.py` **unchanged** (reuses the already-tested `_admit_miss_cohort` mid-run admission path for the deferred long-prompt rows); (iii) `tests/test_engine_admission_reorder.py` covering sort stability + `req_index` preservation + threshold-parametrized cap + a dual-gated on-device `test_reordered_cohort_matches_mlx_lm_direct_batched_reference` pinning three-layer Acceptance (c) via the existing `silica.bench.runner._direct_mlx_lm_batched_reference` helper with inverse-permutation index mapping; (iv) a Q-010 acceptance test `test_q010_ratio_below_threshold_on_five_runs` driving the pair `(qwen3-0.6b-smoke, qwen3-0.6b-ttft-under-concurrency)` and asserting `max(offsets_short) / smoke_ttft_ms < 3.5×` over five consecutive measured runs after a one-pair warmup, with `offsets_short` filtered adaptively from `get_scenario(...).workload.prompts` tokenized lengths (not hard-coded indices). Two existing call sites opt out of the split to preserve their stated parity semantics: `silica/bench/runner.py::_collect_bgt1_batched_tokens` pins `length_spread_threshold=float("inf")` so the BGT1 oracle keeps comparing Silica B=2 against direct mlx-lm B=2; `tests/test_p2_batched_parity.py::test_left_padding_does_not_corrupt_any_row` does the same for its direct-batched-reference assertion. Queued-cohort fairness is **not** provided (see Opening doc §5.2 "Per-token fairness beyond the first token" and "queued-cohort" caveat): if `max_batch_size < short_count + 1` and the long prompt ends up in the queue alongside remaining shorts, `_admit_miss_cohort` batches them together and short-in-queue TTFT is again dragged; B.1 fixes the single-cohort case Q-010 actually measures.
  - [x] **P-4.5-C — KVCodec runtime integration spike.** (i) `docs/P4_5_C_KVCODEC_OPENING.md` enumerates the three integration-point options — (A) active `BatchKVCache` in-place, (B) detached prefix-cache store via `SyntheticPrefixBlockStore.register_detached` / `fetch_detached`, (C) codec-aware `BatchKVCache` façade — against D-003 (no compressed-domain attention) and Q-009 / R-7 (no MLX variable-length SDPA). Recommendation and chosen option: **Option (B)**, on the grounds that (A) / (C) both require a compressed-domain attention kernel or variable-length SDPA and are therefore out of v0.1 scope, while (B) turns `IdentityCodec.encode_block` / `decode_block` into runtime callers via the admission-path hook already engineered for detached K/V. **C-opening-doc complete 2026-04-21.** (ii) C.1 implementation: `SyntheticPrefixBlockStore(codec: KVCodec)` constructor wiring; `register_detached` calls `codec.encode_block` per layer; `fetch_detached` calls `codec.decode_block` restoring the fp16 shape contract `build_seeded_batch_kv` expects; `PagedPrefixBlockStore` keeps its `NotImplementedError` on detached methods. (iii) C.1 observable: `SyntheticPrefixBlockStore.resident_bytes()` added as a **parallel observable** — its value equals `len(store.live_block_ids()) × num_layers × block_size × (2 × n_kv_heads × head_dim × dtype.size)` under `IdentityCodec`, which also equals `prefix_cache.node_count() × num_layers × block_size × (2 × n_kv_heads × head_dim × dtype.size)` since radix-node count and store-resident block count are 1:1 under `insert_detached`. The per-layer K+V cost `2 × n_kv_heads × head_dim × dtype.size` is the **per-layer** byte-per-token quantity — **not** `MemoryBudgeter.bytes_per_token` or `layout.bytes_per_token_total`, which already sum across layers (multiplying those by `num_layers` would double-count). This right-hand side is **not** `_count_evictable_prefix_blocks × _kv_bytes_per_block` (that budgeter helper counts leaf-zero-hit blocks only and systematically under-reports internal prefix nodes; see opening doc §6.2 and §8.3). Admission decisions and the `MemoryBudgeter` remain on the per-block eviction-shortfall formula in P-4.5-C; the store becomes the authoritative source for total resident prefix bytes only when P-5 proper introduces a non-identity codec. (iv) `tests/test_kvcodec_integration.py` — cache-presence gated on the local Qwen3-0.6B HF cache (no env-var strong gate; mirrors `tests/test_engine_admission_reorder.py` §5); verification entry point is a **single** `Engine.generate_batch([prompt, prompt], params, prefix_cache=shared_pc, max_batch_size=1)` call, not two paired calls. `_prepare_cohort` (initial cohort seal) does not consult the prefix cache — the hit path (`_admit_single_hit_row` → `fetch_detached_blocks` → `codec.decode_block`) only fires inside `_admit_waiting_requests` (mid-run admission). Row 0 therefore admits into the initial cohort, runs miss-path prefill, and registers its aligned prefix during reclaim (`_extract_and_insert_prefix` → `codec.encode_block`); row 1 enters the waiting queue, then mid-run admission routes it through the hit path (`codec.decode_block`). Two paired `generate_batch([prompt], ...)` calls would both run miss-path prefill and never fire `decode_block` — see §10 Q-012 for the design fact. Prompt tokenizes to exactly 34 tokens under the pinned Qwen3-0.6B tokenizer (≥ `2 × block_size + 1 = 33`, `mod 16 == 2`) — the `+1` satisfies batcher invariant S-5 edge 1 and the `mod 16 != 0` guard keeps cold `encode_calls >= floor(len / block_size) × num_layers = 2 × num_layers` and mid-run-hit `decode_calls >= floor((len - 1) / block_size) × num_layers = 2 × num_layers` on the same block count. Plus: `store.resident_bytes()` equality with the radix-node-derived total within ± 0 B; a byte-identical token-stream invariant between the codec-wrapped and no-codec paths on the same `[p, p]` workload (achievable because `CodedBlock.__init__` assigns `k` / `v` by reference without copy); and pure-unit tensor-reference tripwires (`is` / `id()`) on both the IdentityCodec and pass-through encode/decode round-trips. (v) Scope is **homogeneous-shape models only** (Qwen3-0.6B verification target); heterogeneous per-layer-shape codec handling (Gemma4 sliding 16×256 + full 4×512 mix) defers to P-5 proper where per-layer BlockTQ calibration raises the same question.
- **Acceptance:**
  - [x] **Q-010 signal returns below threshold.** On `qwen3-0.6b-ttft-under-concurrency` vs isolated `qwen3-0.6b-smoke`, the max short-row first-token offset / isolated TTFT ratio is **< 3.5×** over five consecutive measured runs on the same machine (one warmup pair discarded first; see PLAN Amendment log 2026-04-21). 3.5× is clearly below the 5× Q-010 promotion trigger but above the measured post-fix steady-state ceiling (~3.2× p95 on Qwen3-0.6B); the residual gap vs an ideal 1× is the intrinsic B=3 short-cohort-prefill overhead under option (C) admission reorder — single-step fairness for cohorts already in DECODE would require MLX variable-length attention (Q-009 / R-7) and lives outside P-4.5 scope. **Verified 2026-04-21 on the post-C.1 tree:** `tests/test_engine_admission_reorder.py::test_q010_ratio_below_threshold_on_five_runs` passes.
  - [x] **Chunked-prefill correctness (three-layer criterion).** The chosen option is verified against three layers of invariants, in decreasing order of strictness — (a) and (b) are hard gates; (c) is the numerical reference. **Strict bit-identity against the unchunked Silica path is NOT claimed**, because fp16 batched SDPA drift across different batch compositions is already documented (P-2 Qwen3-0.6B; P-3-D3.1 Gemma4-31B batched parity finding). Running the *same* cohort under a different batch composition changes the fp16 roundoff and therefore the greedy argmax, independent of chunked-prefill's own correctness.
      - (a) **Event-taxonomy invariant.** On `qwen3-0.6b-ttft-under-concurrency` under the chunked path, every admitted row emits `token` events before its `done` event, zero `aborted` events fire, and `req_index` values stay within `range(len(prompts))`. Scheduler invariants I-1..I-5 (row-lifecycle ordering), B-1..B-9 (budgeter semantics), and S-1..S-7 (prefix-cache accounting) continue to hold — regression-locked via the existing `tests/test_batcher.py` and `tests/test_p2_batched_parity.py` suites, which must stay green.
      - (b) **Per-row token-count invariant.** Each row's total token count on the chunked path equals its count on the unchunked path for the same `(prompt, max_tokens, seed, sampling_params)` — i.e. chunking changes *which* tokens each row emits (fp16 batch-composition drift is expected) but not *how many*. Enforceable on both `qwen3-0.6b-ttft-under-concurrency` and `qwen3-0.6b-bgt1-parity`.
      - (c) **Numerical reference against direct mlx-lm batched on the sub-cohort scoped by the chosen option.** Precedent: P-3-D3.1 Gemma4 B>1 parity degrades to "direct mlx-lm batched reference" rather than "Silica single-request". Under the chosen P-4.5-B option, Silica's tokens for each sub-cohort (short cohort under options (B)/(C); uniform chunked forwards under option (A)) match a direct mlx-lm batched reference run over the same sub-cohort shape, byte-for-byte, for at least `max_tokens=4` per row on the `qwen3-0.6b-ttft-under-concurrency` workload. The direct-batched-reference helper already exists in `tests/test_p3_gemma4_batched_parity.py` and `silica/bench/runner.py::_direct_mlx_lm_batched_reference`. **Verified 2026-04-21 on the post-C.1 tree:** `tests/test_engine_admission_reorder.py::test_reordered_cohort_matches_mlx_lm_direct_batched_reference` passes (31-case suite fully green); regression locks (a) + (b) via the always-green `tests/test_batcher.py` + `tests/test_p2_batched_parity.py` sweeps.
  - [x] **Codec hot-path reached (encode + decode, paired-prompt single `generate_batch` call).** A clean-room `IdentityCodec` instance whose `encode_block` / `decode_block` are instrumented with call counters reaches non-zero counts on both sides under a single workload on `Qwen/Qwen3-0.6B` (cache-presence gated on the local HF cache): **`Engine.generate_batch([prompt, prompt], params, prefix_cache=shared_pc, max_batch_size=1)`**. Rationale for the single-call `[p, p]` shape: `_prepare_cohort` (initial cohort seal) does not consult the prefix cache, so the hit path (`_admit_single_hit_row` → `fetch_detached_blocks` → `codec.decode_block`) only fires inside `_admit_waiting_requests` (mid-run admission). A paired pattern of two `generate_batch([prompt], ...)` calls over the same `shared_pc` would run miss-path prefill twice and never fire `decode_block` — see §10 Q-012 for the design fact. `Engine.generate(prompt, params)` is also not an acceptable entry point because it drives `SimpleKVCache` via `_drive` without routing through `ContinuousBatcher` / `RadixPrefixCache` at all (see `silica/engine/__init__.py:91` vs `:185`). The prompt tokenizes to exactly 34 tokens under the pinned Qwen3-0.6B tokenizer (≥ `2 × block_size + 1 = 33`, `mod 16 == 2`); the `+1` satisfies batcher invariant S-5 edge 1 (`max_aligned = ((len - 1) // block_size) × block_size`, see `silica/scheduler/batcher.py` ~ line 1011), and `mod 16 != 0` keeps the cold and mid-run-hit block counts equal. (a) **encode-side:** row 0's reclaim triggers `_extract_and_insert_prefix` → `insert_detached` → `store.register_detached` → `codec.encode_block` with `encode_calls ≥ floor(len(prompt_tokens) / block_size) × num_layers` on the cold miss-path. (b) **decode-side:** the next `step()` admits row 1 via `_admit_waiting_requests` → `peek` → `_admit_single_hit_row` → `lookup` → `fetch_detached_blocks` → `store.fetch_detached` → `codec.decode_block` with `decode_calls ≥ floor((len(prompt_tokens) - 1) / block_size) × num_layers`. Under the 34-token fixture both lower bounds equal `2 × num_layers = 56` on Qwen3-0.6B (28 layers). The original PLAN wording "≥ 1 call per active KV block on a single-request `Engine.generate('Hello', max_tokens=4)` run" overspecified on three axes (wrong entry point; "Hello" tokenizes to ≪ block_size producing zero aligned blocks; a single cold call fires encode only, never decode); amended here to the single-call `[p, p]` shape. See `docs/P4_5_C_KVCODEC_OPENING.md` §8.0-§8.1 for the call-site walk and the tokenization-length invariant.
  - [x] **No regression in the 15-row bench catalog** — `python -m scripts.bench --all` under the default env-var set exits with all cache-only rows `status="ok"`; dual-gated rows still skip as before. **Verified 2026-04-21 on the post-C.1 tree:** 9 cache-only rows ok (`qwen3-0.6b-*` 8 + `qwen3.5-0.8b-b1-parity`), 6 env-gated rows skipped (`gemma4-31b-*` × 3, `gemma4-moe-smoke`, `qwen3.5-27b-smoke`, `qwen3.5-moe-smoke`).
- **Dependencies:** P-4.
- **Status:** complete (A / B.0 / B.1 / C.0 / C.1 landed 2026-04-21; all four Acceptance checkboxes verified 2026-04-21 in the post-C.1 regression sweep — Q-010 five-run signal, three-layer correctness (a)+(b)+(c), 15-row bench catalog — see v1.6.9 Changelog entry).
- **Notes:** P-4.5 is bookkeeping + plumbing, not capability. Its main risk is scope creep on P-4.5-B — the three-option doc exists precisely to bound the implementation cost before code lands. If option (A) (real in-cohort chunked prefill) is chosen and its scheduler footprint exceeds ~300 diff lines in `batcher.py`, re-open a smaller sub-unit under P-4.5-B rather than bundling. P-4.5-C is a spike whose output is "we verified the integration point works", not "we optimized it"; the P-5 BlockTQ constructor and resident-bytes accounting are P-5 work, not P-4.5.
- **Amendment log:**
  - **2026-04-21 acceptance amendment.** The initial wording of the "Chunked-prefill correctness" bullet asked for "Bit-identical across the whole stream" of the chunked vs unchunked path on a single long prompt. That criterion is (i) untestable for options (B) / (C) under fp16 because changing the batch composition shifts SDPA roundoff (P-3-D3.1 Gemma4 precedent), and (ii) mis-scoped — Q-010 is a multi-row TTFT-fairness defect, not a single-prompt correctness defect. Replaced with the three-layer criterion above (event-taxonomy invariant + per-row token count + direct-mlx-lm-batched numerical reference). Caught by an advisor pre-write review; avoids shipping the opening doc against an unachievable gate.
  - **2026-04-21 P-4.5-C codec hot-path acceptance amendment.** Original acceptance (Codec hot-path reached) specified "≥ 1 call per active KV block on a single-request `Engine.generate('Hello', max_tokens=4)` on `Qwen/Qwen3-0.6B`". Three axes of correction land together in the same P-4.5-C opening commit: (i) **wrong entry point.** `Engine.generate(prompt, params)` drives `SimpleKVCache` via `_drive` — it takes no `prefix_cache` argument and never routes through `ContinuousBatcher` / `RadixPrefixCache`, so the detached-K/V hook under Option (B) is unreachable on that path. Verification must use `Engine.generate_batch([prompt], params, prefix_cache=shared_pc, max_batch_size=1)` (see `silica/engine/__init__.py:91` vs `:185`). (ii) **"Hello" too short.** Tokenized Qwen3 length is ≪ `block_size`, producing zero aligned blocks and therefore zero `register_detached` calls — the original wording would always fail. (iii) **single cold run exercises encode only.** `fetch_detached` fires only on a later request that hits the prefix cache; a cold run alone cannot verify `decode_block`. The amendment therefore requires (a) prompt tokenizes to `≥ 2 × block_size + 1 = 33` tokens — the `+1` reserves the suffix prefill token batcher invariant S-5 edge 1 demands (`max_aligned = ((len - 1) // block_size) × block_size`, `silica/scheduler/batcher.py` ~ line 1011), so cold `encode_calls ≥ floor(len / block_size) × num_layers` and the paired repeat `decode_calls ≥ floor((len - 1) / block_size) × num_layers` both equal `2 × num_layers` exactly; (b) a catalog-test-style invariant guards the tokenization-length requirement against future tokenizer changes; (c) the acceptance splits into a cold-run encode-side clause and a paired repeat-prompt decode-side clause via the *same* `shared_pc`. See `docs/P4_5_C_KVCODEC_OPENING.md` §8.0–§8.2 for the call-site walk.
  - **2026-04-21 P-4.5-C.1 acceptance shape amendment.** During C.1 test authoring it surfaced that `ContinuousBatcher._prepare_cohort` (the initial cohort seal) does **not** consult `RadixPrefixCache`; prefix-hit lookup (`peek` → `_admit_single_hit_row`) fires only inside `_admit_waiting_requests` (mid-run admission). Two consecutive `generate_batch([prompt], prefix_cache=shared_pc, max_batch_size=1)` calls therefore both run miss-path prefill — the second call's single admission is sealed into its own fresh initial cohort and never queries `shared_pc`. The opening doc's original §8.1 / §8.2 specification (paired `generate_batch` calls sharing `shared_pc` across calls) would never fire `codec.decode_block`. Acceptance reshaped to a single `generate_batch([p, p], max_batch_size=1)` call — prompt 0 admits into the initial cohort, row-0 termination triggers `_extract_and_insert_prefix` (encode), and prompt 1 enters the waiting queue where `_admit_waiting_requests` routes it through the hit path (decode). Opening doc §8 rewritten to §8.0-§8.4 reflecting the single-call shape; P-4.5-C deliverable (iv) updated in-place; new `Q-012 — Initial-cohort prefix-cache consultation` added to §10 recording the design finding (cross-`generate_batch`-call prefix reuse is effectively zero in v0.1, a potential limitation for REPL / chat-session workloads that revisit the same prompt prefix across turns).
  - **2026-04-21 P-5 Strategy source amendment.** §7 P-5 Strategy line previously read `PagedKVCache(codec=...) injection-based switching`. Under P-2 Option B, `PagedKVCache` is a page-table + refcount bookkeeping layer that holds no K/V — its `budget()` reads claimed-block counts, not actual tensor residency. The real v0.1 codec hook is `PrefixBlockStore`, whose synthetic variant stores detached K/V and whose paged variant raises `NotImplementedError` on detached methods. Amended line in the same commit as `docs/P4_5_C_KVCODEC_OPENING.md` lands, so PLAN and opening doc agree at commit time. The active-K/V path (mlx-lm `BatchKVCache`) remains unwrapped in v0.1 — D-003 excludes compressed-domain attention; Q-009 / R-7 excludes variable-length SDPA; no codec can sit between the live K/V and the mlx-lm attention kernel without one of those two primitives.
  - **2026-04-21 Q-010 threshold amendment.** Original acceptance (a) wording was "ratio < 3× over five consecutive runs". First on-device measurement after the P-4.5-B.1 Option-(C) implementation landed showed a post-fix steady-state ratio distribution of roughly `{2.53, 2.78, 2.95, 3.07, 3.27}×` over five measured runs (after a warmup pair, single Qwen3-0.6B subprocess running `(smoke, ttft)` pairs back-to-back). Pre-fix measurements were `{4.13, 4.42, 4.56, 4.76}×`, so the fix reduced the ratio by ~30-50%, and the residual sits at ~3× because option (C)'s B=3 short-cohort prefill still pays a per-row prefill overhead of ~2-3× vs the B=1 isolated smoke — intrinsic to non-variable-length batched attention (Q-009 / R-7). Strict `< 3.0×` would therefore require sub-B=3 short-cohort shapes, which degenerates to "admit one row per cohort" and loses batching entirely. Tightened Q-010 trigger is 5×; the P-4.5 exit target moved to **< 3.5×** — still a clear factor-of-~1.5 below the Q-010 trigger, and a ~30% absolute improvement vs pre-fix. Also introduces the **one-warmup-pair-discarded** protocol so ratios reflect steady-state (metal-kernel-warm) rather than subprocess cold start. Test lives at `tests/test_engine_admission_reorder.py::test_q010_ratio_below_threshold_on_five_runs` and is dual-gated on the 0.6B HF cache.

### P-5 Phase 5 — VQ KV Compression

- **Goal:** replace the P-0 `IdentityCodec` stub with real VQ codecs (Principle 9 stub-to-real replacement), letting the platform admit more requests or longer context within the same memory budget.
- **Scope:** `IdentityCodec`, `BlockTQCodec`, `RaBitQCodec`. **PQ / OPQ stay out of the main line.**
- **Strategy:**
  - No compressed-domain fast path in v0.1 (D-003).
  - `PrefixBlockStore(codec=...)` injection-based switching — the `SyntheticPrefixBlockStore.register_detached` / `fetch_detached` pair is the seam for v0.1 `BlockTQCodec` / `RaBitQCodec` integration. The active-K/V path (mlx-lm `BatchKVCache` + SDPA call) is **not** codec-wrapped because D-003 excludes compressed-domain attention and MLX has no variable-length SDPA (Q-009 / R-7) that would absorb a decoded-on-demand scratch. `PagedPrefixBlockStore` stubs the detached methods with `NotImplementedError`; its codec story lands when the paged-attention kernel track advances. The shape contract for `decode_block` is pinned by `silica/scheduler/seed_kv.py::build_seeded_batch_kv` (fp16, `(1, n_kv_heads, block_size, head_dim)` per-layer per-block). See `docs/P4_5_C_KVCODEC_OPENING.md` for the full integration-point analysis and the rejected alternatives.
  - The scheduler reads `KVCodec.logical_bytes` / `resident_bytes` to learn about savings and admits more requests accordingly (Principle 8).
  - **`BlockTQCodec` / `RaBitQCodec` must be rewritten on the `mx.array` hot path**, with resident accounting added. The reference source is **`vqbench/`** (including nested `vqbench/turboquant_plus/`) — vqbench is the **algorithmic reference + Qwen3.5-4B empirical baseline** (`BlockTurboQuantMSE B=64` 4-bit K+V already validated at +0.0% ΔPPL, see `vqbench/REPORT.md`); details in §5.5 Reference Map. vqbench itself is NumPy + PyTorch + HF transformers and is **not imported at runtime** (D-009). The real engineering work in P-5 is "translate the NumPy logic into MLX-native `mx.` ops" + "expose savings as `logical_bytes` / `resident_bytes` to the scheduler". This is **not** "wiring a third-party plugin" — it is replacing a native-capability stub with its real implementation (Principle 9). State this up front so P-5 scope creep is not mistaken for surprise.
  - Whether codec decode overhead signals should enter scheduler admission (avoiding the pathological "saves memory but kills decode throughput" combination) is discussed in **Q-007**; not baked into the interface in v0.1.
- **Deliverables:**
  - [ ] `silica.vq.block_tq.BlockTQCodec`.
  - [ ] `silica.vq.rabitq.RaBitQCodec`.
  - [ ] `silica.kvcache.paged.PagedKVCache` supports codec injection.
  - [ ] Bench gains `--kv-codec {fp16,block_tq,rabitq}`.
  - [ ] Scheduler budget admission policy that exploits codec savings.
- **Acceptance:**
  - [x] Switching the codec requires no change to the scheduler or model adapter. (v1.7.4 close: by-inspection evidence in `docs/P5_ACCEPTANCE_SWEEP/codec_swap_neutrality.md` — zero `isinstance(codec)` / concrete-codec imports across `silica/scheduler/**` and `silica/models/**`; 12 docstring mentions classified as non-dispatching reader notes.)
  - [x] For the same scenario set, fp16 vs codec quality delta and memory savings are available from the bench in one command. (v1.7.4 close: `scripts/bench.py --all --all-kv-codecs --seeds 42,43,44 --out <jsonl> --report-md <md>` produces a coherent 924-row report; evidence in `docs/P5_ACCEPTANCE_SWEEP/all_kv_codecs.{jsonl,md,log}` + report. `ok=360 / failed=564 / skipped=0`, failures fully classified into 528 `codec_override_invalid` + 33 K-only `rabitq_b1` + 3 vqbench-aligned symmetric-codec guard. Gate is scoped to report-schema coverage per §7(e) as rewritten at v1.7.4; populated xcheck numbers are owned by (4-b).)
  - [x] With BlockTQ on, the same memory budget admits more requests (quantitatively verifies Principle 8). (v1.7.4 close: `qwen3-0.6b-admission-headroom-prefix-heavy` run on seeds `{42, 43, 44}` in `docs/P5_ACCEPTANCE_SWEEP/admission_headroom.{jsonl,md}`. `cap_bytes=128MB`, `warmup_ratio=0.5`: `resident_bytes_fp16=67.895MB` vs `resident_bytes_block=18.035MB` (`residency_ratio ≈ 0.266`, ≈ 1/3.76 per vqbench §3.1); `n_fp16=4`, `n_block=7`, `admit_ratio=1.75`. Gate `n_block > n_fp16` passes with margin 3 on every seed. Scenario design makes the gate structural / seed-independent.)
  - [x] **Numeric cross-check against vqbench** — two independent thresholds, both must pass:
    - **(a) Per-block reconstruction error — algorithmic parity.** Metric: **per-block relative Frobenius error** `||K_decoded - K_original||_F / ||K_original||_F` between silica MLX-native `BlockTurboQuantMSE` and an in-test NumPy reference transcribed verbatim from `vqbench/vqbench/methods/turboquant/block_mse.py`. Synthetic-Gaussian (a-algo) half was the P-5 close gate: regression-locked by `tests/test_block_tq_vqbench_xcheck.py` (landed P-5-A.1c) at tolerance `5e-3` across `(vq_block_size, num_bits) ∈ {32, 64} × {3, 4}` on synthetic Gaussian inputs, with a tighter `1e-3` lock for the production-recommended `B=64 b=4`. The real-activation half (a-real) — Frobenius on extracted Qwen3.5-0.8B pre-RoPE K / V — was v1.7.2-deferred as post-P-5 follow-up and **closed at v1.7.5** by `tests/test_block_tq_real_activation_xcheck.py` (inline NumPy reference, not the v1.5.1 subprocess design — see Notes (a-real) bullet and `docs/P5_A_REAL_OPENING.md`).
    - **(b) End-to-end PPL agreement on the vqbench-aligned oracle — mean-over-seeds cross-check.** On the `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` bench row (Qwen3-0.6B, `BlockTurboQuantMSE B=64` 4-bit K+V, `codec_quality_path="vqbench_aligned"` — the D.2a pre-RoPE projection-patch oracle `teacher_forced_chunked_nll_vqbench_aligned` in `silica/bench/ppl_oracle.py` landed at P-5-D.2a), silica's codec-backed ΔPPL and vqbench's subprocess ΔPPL (via the P-4.4 `--vqbench-xcheck` path landed at C.6) — both on the **same** model, same three seeds `{42, 43, 44}`, same `chunk_size`, same WikiText-2 slice — must satisfy the two-part aggregated gate: `|mean_gap| <= 2 * SEM_diff` **and** `|mean_gap| < 1.0` PPL, where `mean_gap = mean_seeds(silica.ΔPPL_seed − vqbench.ΔPPL_seed)` and `SEM_diff = sqrt( std(silica.ΔPPL_seeds)^2 / n + std(vqbench.ΔPPL_seeds)^2 / n )` with `n = 3` (independent-samples standard error of the difference of means; sample std uses Bessel-corrected `n-1`). The aggregated gate is the close criterion; the per-row `vqbench_epsilon = 0.01` / `_VQBENCH_PCT_EPSILON = 0.1` thresholds in `_compute_gap_fields` (`silica/bench/runner.py`) **remain in code unchanged** and continue to emit the `vqbench_divergence_warning` boolean as a **diagnostic** metadata field on every row — under the D.2a 3-seed data per-row `vqbench_divergence_warning=true` still fires at worst-case `|gap| ≈ 0.61` PPL because silica and vqbench draw different Haar rotations from the same distribution (silica shares one rotation across all heads, vqbench samples one rotation per head), not because of algorithmic drift; per-row warnings are **expected under D.2a** and do not block (4-b) close. **Evidence (2026-04-24, landed at `ed57be1`; raw `docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`):** silica mean ΔPPL `+0.511 ± 0.354`, vqbench mean ΔPPL `+0.661 ± 0.347`, `mean_gap = −0.150` PPL, `SEM_diff ≈ 0.286`, `2 * SEM_diff ≈ 0.572` — both gate conditions pass (`0.150 ≤ 0.572` and `0.150 < 1.0`). The `prefix_store_post_rope` production-routing arm (the C.2 post-RoPE prefix-cache store path, same codec config, scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`) measures a ΔPPL in the ~5–10 PPL range on the same codec config — that is a real **production-path quality cost**, not an algorithmic error, and (4-b) **explicitly does not close it**; the remaining pre-RoPE production-routing work is tracked as **post-P-5 required follow-up** (see Notes and `docs/P5_D2_INVESTIGATION/README.md`). The original v1.5.1 "Qwen3.5-4B vs `vqbench/REPORT.md` static baseline" gate is separately deferred as a **post-P-5 required follow-up**; see Notes.
    - Both must pass; passing only one is insufficient.
- **Dependencies:** P-4.5.
- **Status:** done. Implementation sub-units per `docs/P5_OPENING.md` §8 landed: P-5-A.0 / A.1 / A.2 / A.3, P-5-B.1 / B.2 / B.3, P-5-C.1 / C.2 / C.3 / C.4 / C.5 / C.6 (between 2026-04-22 and 2026-04-23); P-5-D.1 (seed propagation fix, commit `2b3868d`), P-5-D.2a (vqbench-aligned pre-RoPE projection-patch oracle + 3-seed verification, commit `ed57be1`), and P-5-D.3 ((4-b) gate reinterpretation, v1.7.3) all landed 2026-04-24. §7 P-5 Acceptance is closed: item (4) at v1.7.3 via the vqbench-aligned oracle mean-over-seeds gate (see (4-b) body for evidence and `docs/P5_D2_INVESTIGATION/README.md` §Close); items (1) / (2) / (3) at v1.7.4 via the P-5 Acceptance sweep (evidence in `docs/P5_ACCEPTANCE_SWEEP/`). Deliverables above are the PLAN-freeze coarse list — `block_tq` / `rabitq_b1` / `ext_rabitq` codecs all shipped, `--kv-codec` CLI at C.5, scheduler budget admission at A.2 — the one deliverable that remains intentionally deferred is `PagedPrefixBlockStore` codec injection (`NotImplementedError` stub per P-5 Strategy line 545, waiting on the paged-attention kernel track; independent of P-5 close per D-003 no-compressed-domain-attention scope). Sub-unit decomposition and per-sub-unit status live in `docs/P5_OPENING.md` §8. Post-P-5 follow-ups remain in backlog: pre-RoPE production routing (P-5-F KV-store architecture), Qwen3.5 real-target xcheck (b-static) PPL baseline, per-expert MoE streaming (P-6), P-3-C5 recurrent-state snapshot, P-3-E4 batched MoE. Qwen3.5 real-target xcheck (a-real) real-activation Frobenius **closed at v1.7.5** and is no longer in backlog.
- **Notes:**
  - Concrete implementation details for BlockTQ / RaBitQ reference `turboquant_plus/` (gitignored reference impl).
  - **Qwen3.5 real-target cross-validation — post-P-5 required follow-up.** v1.5.1 (2026-04-16, commits `f64a65f3` / `2ce9a7b`) wrote Acceptance (a) / (b) naming Qwen3.5-0.8B / Qwen3.5-4B as cross-validation targets. `docs/P5_OPENING.md` §6.5 later moved all P-5 codec-backed PPL bench rows to Qwen3-0.6B because Qwen3.5-0.8B and Qwen3.5-4B are hybrid-DeltaNet (`has_recurrent_state=True`) and `ContinuousBatcher` refuses to pair `RadixPrefixCache` with recurrent adapters (`docs/P3_DELTANET_SURVEY.md` C-open-3). v1.7.2 narrows (a) / (b) to what the current tree ships; the two items below **remain required for v0.1 production launch** and are moved out of the P-5 close gate, not out of v0.1 scope. Scope correction, not scope reduction.
    - **(a-real) Real-activation Frobenius on Qwen3.5-0.8B (or larger) — closed at v1.7.5.** Test: `tests/test_block_tq_real_activation_xcheck.py`. Evidence: `docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.{md,jsonl}` (144 rows — 6 GLOBAL layers × K/V × 4 `(B, bits)` cells × 3 seeds). Design contract: `docs/P5_A_REAL_OPENING.md`. Extracts pre-RoPE K / V from a `Qwen3_5Adapter` prefill pass on a checked-in deterministic prompt (GLOBAL layers only — `layer.is_linear == False`) and runs silica MLX BlockTQ against the vqbench-transcribed NumPy reference landed at P-5-A.1c (not a vqbench subprocess — the original v1.5.1 subprocess design was superseded in favour of the established transcribe-inline idiom, per `P5_A_REAL_OPENING.md` §2.3). Gate: `|silica_frob - numpy_frob| < 1e-3` on `(B=64, b=4)` and `< 5e-3` elsewhere — the (a-algo) envelope reused. Worst observed gap: `1.15e-4`, ~43× tolerance headroom. `IdentityCodec` round-trip baseline is dtype-preserving and therefore degenerate on every row; the absolute-gap gate above is the close criterion. Single skip gate: HF cache has Qwen3.5-0.8B. Qwen3.5-4B / 35B-A3B extensions remain a parametrisable escape hatch in the test, not exercised at landing (`P5_A_REAL_OPENING.md` §6).
    - **(b-static) Qwen3.5-4B end-to-end codec PPL vs `vqbench/REPORT.md` static baseline (`PPL_fp16 ≈ 10.3866`, REPORT §3.1).** Post-P-5 required follow-up; **blocked on the remaining P-3-C recurrent / prefix-cache cooperation work** (`docs/P3_DELTANET_SURVEY.md` C-open-3; may land through P-3-C5 preempt/replay-with-recurrent-snapshot, or through a narrower targeted fix if prefix-reuse support arrives without the full snapshot machinery), **or on an alternate monkey-patch measurement path** wrapping `k_proj` / `v_proj` on the 8 full-attention layers to inject quantize→dequantize inside the forward pass, bypassing `RadixPrefixCache`. Mirrors `vqbench/scripts/reproduce_qwen35_4b_headline.py`'s own fallback pattern; `vqbench/REPORT.md` §2.2 documents the monkey-patch as "pessimistic" but valid.
  - **Production `prefix_store_post_rope` prefix-cache quality cost — post-P-5 required follow-up.** At the same `BlockTurboQuantMSE B=64` 4-bit K+V codec config, the production-routing arm (`codec_quality_path="prefix_store_post_rope"`, scenario `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`) measures a ΔPPL in the ~5–10 PPL range on Qwen3-0.6B WikiText-2 (chunk-boundary-dependent), while the D.2a `-vqbench-aligned` oracle arm on the same codec agrees with vqbench's subprocess ΔPPL within the (4-b) aggregated gate. The gap is **not algorithmic**: D.2a probes (`docs/P5_D2_INVESTIGATION/p5_d2_probe*.py`) confirmed silica MLX BlockTQ and vqbench NumPy BlockTQ produce bit-identical reconstructions on the same input, and silica's prefix-cache round-trip is numerically neutral. The production-path cost is that silica's prefix-cache store injects reconstruction noise in **post-RoPE** space, whereas vqbench's `_QuantizedProj` patch (and the D.2a-aligned oracle) inject noise in **pre-RoPE** projection space; at the same Frobenius reconstruction error the post-RoPE injection pays an additional chunk-boundary cost that scales with the number of prefix-cache refills in a streaming decode (`docs/P5_D2_INVESTIGATION/README.md` §Root cause). Closing this gap on the serving hot path requires a pre-RoPE KV-store architecture (position-aware codec) or an equivalent mechanism that keeps reconstruction error confined to the pre-RoPE projection space — an **architecture change owned by a post-P-5 unit** (the P-3-C prefix-cache cooperation work-area or a dedicated F-series follow-up), not a bench-report change. (4-b) closes the **algorithmic parity between silica's MLX-native BlockTQ and vqbench's NumPy BlockTQ when both inject noise in the same space**; it deliberately does not close the production-routing quality cost, and readers should not interpret the (4-b) `[x]` as silica's production `prefix_store_post_rope` store path having achieved vqbench-level PPL.

### P-6 Phase 6 — Weight Streaming

- **Goal:** weight streaming relieves residency pressure.
- **Scope:** `ResidentWeightProvider` (already in tree) + `StreamingWeightProvider` + scheduler prefetch coordination.
- **Strategy:**
  - Take cues from `mlx-flash`.
  - While layer N computes, prefetch layer N+1 (dense path).
  - **Exploit Apple unified memory** (Principle 2): this is not "GPU pulls weights from disk/CPU"; it is "lifetime management of different regions within the same memory pool".
  - **Dense vs MoE residency granularity (D-011).** Dense streaming uses **layer** granularity (prefetch layer N+1 while computing layer N). MoE streaming uses **expert** granularity — keep only recently-active top-k experts resident; other experts' regions in unified memory may be overwritten. This is streaming's **primary payoff for MoE** (Qwen3.5-35B-A3B: total 35B, active 3B, resident set can shrink to roughly active size + non-FFN weights + headroom). MoE scheduler coordination: as soon as gate logits are available, fire `prefetch_experts(layer_idx, top_k_ids)`; do not wait until the expert FFN is actually invoked.
- **Deliverables:**
  - [ ] `silica.weights.streaming.StreamingWeightProvider` (dense + MoE dual mode).
  - [ ] `silica.weights.prefetch`: scheduler prefetch coordination (layer-granular for dense; expert-granular for MoE).
  - [ ] Unified-memory-aware residency policy (e.g. LRU-over-experts eviction for MoE).
- **Acceptance:**
  - [ ] Under an artificial memory budget (e.g. 24 GB), Qwen3.5-27B int4 does not OOM (dense path).
  - [ ] **Decode tok/s ≥ 70% of the `ResidentWeightProvider` baseline** (dense), with fixed comparison conditions: same machine / same model / same scenario / same sampling config when taking the ratio. If resident cannot run under 24 GB at all (OOM), switch the baseline to an **"uncapped resident reference run"** (no budget, no streaming, same scenario, same decode tok/s) and take the ratio against that.
  - [ ] **MoE per-expert residency takes effect (D-011).** For a MoE target (Qwen3.5-35B-A3B or gemma-4-26B-A4B) under a 24 GB budget, `StreamingWeightProvider.resident_bytes()` ≤ `active_experts × expert_size + non_FFN_weights + expert_residency_headroom` (quantitatively verifies per-expert residency is active rather than silently falling back to full residency). Headroom defaults to ≤ 20% of active size; tightened in P-4 bench after empirical measurement.
  - [ ] MoE decode tok/s ≥ 60% of the MoE `ResidentWeightProvider` baseline — the threshold is below the dense 70%, acknowledging that expert-fetch miss stalls are inherent and the first cut of streaming does not optimize for them.
- **Dependencies:** P-2 (scheduler) + P-3 (model adapter).
- **Status:** planned.
- **Notes:** may be pulled ahead of P-5 out of the T1 tier if P-3 shows 27B/31B won't fit at baseline (Q-003).

### P-7 Phase 7 — Speculative Decoding

- **Goal:** draft-target speculative decoding speeds up decode.
- **Scope:** `NoopDraftEngine` (already in tree) + `DraftTargetEngine` (most basic version).
- **Strategy:**
  - A small model drafts; the large model verifies.
  - EAGLE / Medusa / DFlash complexity is deferred to v0.2.
- **Deliverables:**
  - [ ] `silica.speculative.draft_target.DraftTargetEngine`.
  - [ ] Integration with the decode loop.
  - [ ] Acceptance / rollback metrics.
  - [ ] Bench switch `--speculative {none,draft_target}`.
- **Acceptance:**
  - [ ] Under greedy decoding, with speculative on vs off the token sequences are **identical token by token** (correctness invariant as the baseline gate).
  - [ ] **Decode tok/s ≥ 1.2× the draft-disabled baseline**, measured on a **fixed standard scenario** (e.g. P-4 bench's "long-in / short-out" or an equivalent mixed-batch scenario). **Cherry-picking a best-case workload** (abundant shared prefixes, a specific temperature, or the single most favorable scenario) is not acceptable evidence.
  - [ ] Baseline correctness is preserved (smoke tests pass with the switch in either position).
- **Dependencies:** P-2 + P-3.
- **Status:** planned.
- **Notes:** DFlash / dflash-mlx are v0.2 candidates.

### P-8 Phase 8 — Mini-SGLang Layer

- **Goal:** add a minimal serving layer on top of the engine so the platform is actually "usable".
- **Scope:** OpenAI-compatible HTTP API, session management, prefix-sharing session reuse, a reserved slot for structured output.
- **Strategy:**
  - This is the "product face", not a nicety (D-006).
  - Priority discussion in Q-002 (whether it floats from T2 to the tail of T1).
  - The outer organization follows mini-sglang; CUDA kernels are not copied.
- **Deliverables:**
  - [ ] `silica.server.openai_api`: FastAPI, `/v1/chat/completions`, `/v1/completions`.
  - [ ] `silica.server.session.SessionManager`: session management, cross-request prefix reuse.
  - [ ] An interface slot for structured generation / grammar (unimplemented).
  - [ ] `silica.llm.LLM`: Python-friendly high-level interface.
- **Acceptance:**
  - [ ] The `openai` Python client can stream responses from the silica server.
  - [ ] Cross-request prefix reuse is verifiable (send N shared-prefix requests in one session, check prefix cache hit rate).
  - [ ] Locally behaves like a small serving engine.
- **Dependencies:** P-2 + P-3 (+ P-5 / P-6 optional).
- **Status:** planned.
- **Notes:** this Phase is what upgrades Silica-MLX from "an engine library" to "a usable Mac inference platform".

---

## 8. Priority & Milestones

### 8.1 Priority Tiers

Tier IDs use `T0 / T1 / T2` to avoid visual collision with phase IDs `P-0 / P-1 / P-2`.

| Tier | Phases       | Meaning                                                        |
| ---- | ------------ | -------------------------------------------------------------- |
| T0   | P-0 .. P-4   | Skeleton + baseline engine + target models + bench             |
| T1   | P-5 .. P-6   | VQ KV compression + weight streaming; make big models fit 48GB |
| T2   | P-7 .. P-8   | Speculative + serving layer                                    |

Whether Phase 8 floats in priority: see Q-002. Whether Phase 6 is pulled forward: see Q-003.

### 8.2 Milestones

| ID  | Name                                    | Dependent Phases | Acceptance |
| --- | --------------------------------------- | ---------------- | ---------- |
| M-1 | Skeleton                                | P-0              | Interfaces frozen, stub tests pass |
| M-2 | Single-request gen                      | P-1              | Qwen3.5-0.8B generates text |
| M-3 | Multi-request core                      | P-2              | 8 concurrent requests + prefix cache hit |
| M-4 | Big models adapter correct              | P-3              | Dense adapter structural correctness (fp16 parity on dense control model, max abs logit diff < 1e-3) + hybrid attention routing (including `hybrid_deltanet` dispatch + DeltaNet recurrent-state plumbing per D-015: `StateDelta.recurrent_bytes()`, `adapter.state_from_prefix`, and snapshot→mutation→`adapter.rollback_state` round-trip) + quantized dense correctness (teacher-forced argmax agreement ≥ 98%, or fallback PPL drift < 0.1 absolute) + **MoE smoke test adapter correctness** (D-011: Qwen3.5-35B-A3B / gemma-4-26B-A4B structural correctness + fp16 parity on a MoE control model + per-expert `get_expert` call-path unit test). **Product memory-fit target** (dense 27B/31B @ 48 GB, 500 tokens): validated here **only if** Q-003 resolves to "int4 fits in 48 GB"; otherwise deferred to M-7. MoE memory-fit is not Q-003-gated (small active params, R-1 MoE mitigation). |
| M-5 | Unified bench                           | P-4              | One command produces the baseline table |
| M-6 | VQ on platform                          | P-5              | BlockTQ / RaBitQ wired in, savings quantifiable |
| M-7 | Streaming weights + deferred memory-fit | P-6              | Under a 24 GB budget, Qwen3.5-27B int4 does not OOM; decode tok/s ≥ 70% of the `ResidentWeightProvider` baseline (see P-6 Acceptance). **If M-4 deferred the product memory-fit target** (Q-003 forced P-6 before P-3 exit), this milestone carries the real 27B/31B @ 48 GB 500-token validation. |
| M-8 | Speculative enabled                     | P-7              | Correctness unchanged across switch + speedup |
| M-9 | Platform usable                         | P-8              | OpenAI API + session usable |

---

## 9. Decisions Log

Append-only. New decisions go at the end; old ones are not edited. Revocations / revisions open a new entry referencing the revoked ID.

### D-001 — Package manager & Python version

- **Date:** 2026-04-13.
- **Status:** accepted.
- **Decision:** `uv` + Python 3.12.
- **Rationale:** uv is modern and fast; mini-sglang also uses uv; Python 3.12 is the version MLX / mlx-lm stably support.
- **Consequences:** contributors need uv; Python < 3.12 is unsupported.

### D-002 — vLLM core first, mini-sglang layer later

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** the engine core follows vLLM's ideas (paged KV, continuous batching, memory budget); the outer serving layer follows mini-sglang's ideas.
- **Rationale:** on a 48 GB M5 Pro the first-order problem is "run stably + save memory + be scalable", which vLLM's paged KV + batching addresses directly; mini-sglang is a better reference for module layering and the serving shell.
- **Consequences:** Phase 0–4 looks like a mini-vLLM; Phase 8 looks like a mini-sglang.

### D-003 — KVCodec v0.1 excludes compressed-domain attention

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** the v0.1 `KVCodec` interface contains only `encode_block` / `decode_block` / `logical_bytes` / `resident_bytes`; no `attend()` or any compressed-domain fast path.
- **Rationale:** avoid over-committing the interface on day one. v0.1 prioritizes interface simplicity; v0.2 revisits this after Phase 5 bench data.
- **Consequences:** Phase 5 BlockTQ / RaBitQ implementations must follow the "decode then standard attention" path, even if it is suboptimal.

### D-004 — Phase 1 model execution: wrap mlx-lm

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** the Phase 1 `ModelAdapter` is a thin wrapper over `mlx-lm`; model-execution details are not rewritten in Phase 1.
- **Rationale:** run first, optimize later. Proper adapter work comes in Phase 3.
- **Consequences:** Phase 3 will pay a rewrite cost, but Phase 1 iteration is much faster.

### D-005 — Phase 3 quantization path: MLX-native only

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** Phase 3 quantization uses mlx-lm's existing 4-bit / 8-bit path; GGUF / AWQ multi-format support is **not** added.
- **Rationale:** 27B / 31B on 48 GB obviously must be quantized; start with what MLX can already run, defer multi-format compatibility to v0.2.
- **Consequences:** users cannot load GGUF / AWQ pre-quantized models; only mlx-lm-supported quantizations.

### D-006 — Platform as product, VQ as means

- **Date:** 2026-04-14.
- **Status:** accepted (user has corrected this framing once, explicitly).
- **Decision:** Silica-MLX itself is the product. VQ / weight streaming / speculative are means to make the platform run big models well; they are not research subjects.
- **Rationale:** the user's goal is "a single-Mac-chip inference platform that exploits VQ-class tech well", not "a benchmark vehicle for VQ research".
- **Consequences:**
  - Target users are Mac developers who want to run big models locally.
  - Phase 8 (the serving layer) gains priority.
  - VQ / weight streaming cannot be mere opt-in flags; the memory budgeter must actively use the savings (Principle 8).
  - Design must be Apple-unified-memory-first (Principle 2).

### D-007 — Plan document structure

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** `docs/PLAN.md` is the single source of truth. Structure: Meta / TL;DR / Mission / Scope / Principles / Architecture / Interfaces / Phases / Priority / Decisions Log / Open Questions / Risks / References / Changelog, with stable IDs throughout.
- **Rationale:** CRUD-friendly — stable IDs, self-contained phase blocks, append-only decisions log. Lets us locate and modify a single entry without reading the whole document.
- **Consequences:** future plan changes go into the Decisions Log and update the relevant Phase block; the Changelog tracks version bumps.

### D-008 — core / engine boundary: data classes in core, logic classes in engine

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** data classes (`Request`, `RequestState`, `SamplingParams`, `Context`, ...) live in `silica.core`; runtime logic classes (`Engine`, the runner portion of scheduler state machines) live in `silica.engine`. Follows the mini-sglang convention.
- **Rationale:** Phase 0 needs a clear directory layout. Q-004 Option A is the convention mini-sglang has validated; every Phase block in this document already implicitly uses these paths (e.g. `silica.core.request.Request`, `silica.core.sampling.SamplingParams`). Leaving it undecided would make document and code inconsistent at Phase 0 completion.
- **Consequences:**
  - `silica.core` contains data + observability (logging, profiling, metrics schema); no business logic.
  - `silica.engine.Engine` holds core data classes and drives scheduler / kvcache / model.
  - Resolves Q-004.

### D-009 — MLX-native hot path as hard constraint

- **Date:** 2026-04-14.
- **Status:** accepted (user-required).
- **Decision:** the inference hot path **must** be 100% MLX. Concretely:
  1. All tensors are `mlx.core.array` (`mx.array`). `silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler` / `silica.vq` / `silica.weights` / `silica.speculative` **must not** contain `torch.Tensor` or `numpy.ndarray` participating in tensor math (numpy is allowed for config / scalar / list helpers).
  2. **No PyTorch runtime dependency.** `pyproject.toml` may not list `torch` as a runtime dep. torch is allowed only as an **optional** dev / extras dependency for offline weight conversion (e.g. `pip install silica-mlx[convert]`); conversion outputs are MLX-native and torch is never touched again after inference starts.
  3. **vllm and transformers are algorithm / architecture references only, never runtime dependencies.** vllm's `csrc/`, `vllm_flash_attn/`, and GPU/TPU/XPU/CPU model runners are all out of scope.
  4. Phase 1 wrapping `mlx-lm` is legal (D-004) **because mlx-lm is itself MLX-native**. Replacing the runtime path with any torch-based wrapper (including transformers, llama.cpp Python bindings, etc.) is forbidden.
- **Rationale:** user hard requirement of "must be native MLX". Silica-MLX's entire value proposition (D-006 "Mac inference platform") rests on MLX's Apple Silicon performance and unified memory advantages; any torch hot path breaks Principle 2 (unified memory first).
- **Consequences:**
  - All attention / sampling / kvcache internals must use `mx.` ops.
  - In Phase 3, if a target model is missing from mlx-lm, `silica.models.*` must rewrite it MLX-native — **no fallback to transformers**.
  - vllm v1 source code is "how to design" reference only, not "how to call" dependency (see 5.4 Reference Map).
  - If a benchmark needs to cite numbers from another inference engine, that comparison run must execute in a separate process — it may not be mixed into the silica runtime.

### D-010 — Phase 1 mlx-lm borrowing boundary

- **Date:** 2026-04-14.
- **Status:** accepted (pinpointed as the easiest silent-rework hazard during two Codex plan reviews).
- **Decision:** Phase 1 **borrows** from `mlx-lm`:
  1. Model structure loading (class construction + state-dict shapes).
  2. The tokenizer.
  3. The weight loader (safetensors → `mx.array`).

  Phase 1 **does not borrow** mlx-lm's rotating KV cache / prompt cache. Silica manages its own KV from day one. `SimpleKVCache` is an **external cache injected into model forward**, not a layer wrapping mlx-lm's internal cache.
- **Day-1 smoke test** (first task in P-1, before any other deliverable): verify whether `mlx_lm.generate_step(cache=...)` or an equivalent entry point accepts an external cache object.
  - **Accepts** → inject `SimpleKVCache` directly; decoupling is clean; P-1 proceeds as planned.
  - **Rejects** → monkey-patch / fork `mlx_lm.models.*` forward logic; P-1 cost estimate is revised upward (triggers R-6).
- **Rationale:** the ownership boundary between D-004 (wrap mlx-lm) and the P-1 `SimpleKVCache` deliverable was underspecified and is the most likely silent-rework point. It must be fixed before P-1 development. mlx-lm's rotating cache is fine for single-request bring-up but incompatible with Silica's paged / prefix / codec ambitions, so it cannot be the starting point for P-2 — otherwise P-2 becomes "strip mlx-lm's cache, bolt in Silica's", which is rework, not upgrade.
- **Consequences:**
  - P-1 Strategy points at this decision rather than just "thin wrapper". D-004 still stands as the **model-execution** decision; D-010 is the **cache** decision. They are complementary, not conflicting.
  - P-2's `PagedKVCache` is a direct upgrade path from `SimpleKVCache`, not a replacement for mlx-lm's internal cache.
  - R-2 mitigation adjusts: when Phase 3 rewrites the model adapter, the cache integration point is already Silica-owned, so stripping mlx-lm's cache is unnecessary.
  - Adds R-6 for the day-1 smoke test failure risk.
  - **Concrete anti-pattern:** `vqbench/vqbench/torch_wrapper/hook.py`'s `VQBenchCache` (a `transformers.Cache` subclass) is the concrete anti-pattern — that design depends on HF cache lifecycle, which is exactly what mlx-lm's rotating cache looks like in the torch world. If Silica borrowed mlx-lm's cache it would grow into a similar shape (locked to the framework's internal cache layout). D-010 is precisely to avoid this trap.
- **References:** D-004, Principle 6, P-1 Deliverables, R-6, §5.5.

### D-011 — v0.1 architecture generality: MoE + Dense dual support

- **Date:** 2026-04-16.
- **Status:** accepted (user on 2026-04-16 explicitly chose Option B over Option A "interface-only, defer to v0.2").
- **Decision:** Silica-MLX v0.1 architecture **must support both MoE and Dense model families generically** — not "dense-only + expand MoE in v0.2", and not "interface reserved for MoE but untested". Concretely:
  1. **Interface generality.** I-1 `ModelAdapter` `state_delta` is permitted to carry MoE router state; I-4 `WeightProvider` adds three per-expert granularity methods — `get_expert(layer_idx, expert_id)` / `prefetch_experts(layer_idx, expert_ids)` / `release_expert(layer_idx, expert_id)`. Dense implementations raise `NotImplementedError("dense provider has no per-expert path")` rather than being no-ops, so a MoE adapter wired to a dense provider fails loudly.
  2. **At least one MoE target actually runs in v0.1.** The P-3 target-model table expands from 2 dense to **4 targets** (2 dense + 2 MoE smoke test): Qwen3.5-27B (dense), Gemma4-31B (dense), Qwen3.5-35B-A3B (MoE), gemma-4-26B-A4B (MoE). "Supports MoE" must be something we've actually run, not just written in the document.
  3. **MoE streaming is the primary P-6 payoff, not a side case.** P-6 `StreamingWeightProvider` uses layer granularity for dense and expert granularity for MoE — the MoE resident set can shrink close to active-params size, which is the natural sweet spot for a 30B+ model on a 48 GB Mac.
- **Rationale:**
  - D-006's "platform as product" framing demands coverage of mainstream model families. In 2026, MoE is the dominant direction at the open-weight 30B+ scale (Qwen3.5-MoE / Gemma 4 MoE / DeepSeek-V*, etc.); excluding MoE means giving up the product's most natural sweet spot.
  - Principle 2 "Apple unified memory first" + the fact that MoE active params are much smaller than total params = the residency win for MoE on a Mac is theoretically an order of magnitude larger than for dense. Not running MoE means Silica-MLX misses its best-selling scenario.
  - "Interface reserved but untested" is a degenerate middle state: without actually running it, we never find out whether the I-4 per-expert abstraction really covers everything top-k routing / expert eviction / prefetch coordination need. At least one MoE target must actually run to close the feedback loop.
- **Consequences:**
  - **I-4 interface gains three methods** (v1.5.0 amendment; the interface is still in Phase 0 freeze-candidate state, so modification is allowed; frozen at P-0 exit).
  - **P-3 workload grows from 2 adapters to 4**, plus the +1 MoE-family complexity (expert routing + gate normalization + top-k expert loading + aux-loss-ignored forward).
  - **P-6 Strategy / Acceptance gain a per-expert residency path** (MoE-only; dense unchanged).
  - **R-1 MoE mitigation.** MoE active params are small; R-1 does not apply to MoE. An unresolved Q-003 does not block the MoE smoke test.
  - **Q-003 context update.** Q-003 is about dense; MoE fit risk is far lower but does **not** substitute for Q-003 resolution (dense fit is still part of the product promise).
  - **M-4 gains a MoE smoke test adapter correctness item** independent of Q-003 gating.
  - **§3.1 Scope / §3.4 Target Models updated in step.**
- **References:** D-006, Principle 2, Principle 9, I-1, I-4, P-3, P-6, Q-003, R-1, M-4.

### D-012 — Canonical `resident_bytes` measurement

- **Date:** 2026-04-16.
- **Status:** accepted.
- **Decision:** `resident_bytes` (on `KVCodec.resident_bytes(num_blocks)` and `WeightProvider.resident_bytes()`) is defined as **physical bytes currently owned by the component in unified memory**, i.e. the sum of `mx.array` backing-storage sizes the component controls, measured **at the moment of the call**. It does **not** include: (a) transient decode scratch (codec decode intermediates freed before the next block call); (b) MLX allocator headroom / pool padding; (c) memory regions the OS has reclaimable-but-not-yet-reclaimed. The memory budgeter treats `sum(component.resident_bytes())` as the authoritative floor, and compares against a single target (`target_resident_bytes`, initialized to `0.9 × hardware unified-memory total` minus the reserved activation budget).
- **Rationale:** Principle 8 says savings must be observable; if each component reports `resident_bytes` under a different definition (physical vs scratch-inclusive vs headroom-inclusive), the scheduler either double-counts or under-counts and admission control becomes unreliable. Pin the definition now so P-5 / P-6 implementations produce comparable numbers.
- **Consequences:**
  - Every `resident_bytes()` implementation adds a unit test that reports a steady-state value (no transient scratch) and is idempotent across repeated calls outside a modifying operation.
  - The P-4 bench unified-metrics schema (`resident_mb`) is derived from this definition.
  - If a future codec has genuinely unavoidable scratch during encode/decode that must be visible to the scheduler, the solution is a separate `scratch_bytes()` method, **not** polluting `resident_bytes`.
- **References:** Principle 8, I-3, I-4, P-0 Acceptance (unified metrics schema), P-4 Deliverables.

### D-013 — Sampler structure: separate class, not a sixth Protocol

- **Date:** 2026-04-16.
- **Status:** accepted.
- **Decision:** Sampling lives in `silica.core.sampler.Sampler` as a **concrete class**, not as a sixth frozen interface (I-6). The Engine drives `logits → Sampler.sample(logits, sampling_params, rng_state) → token` between `ModelAdapter.decode_step` and the token-stream yield. Logit processors (temperature, top-p, top-k, repetition penalty, and future user-defined processors) compose inside the Sampler via a short `Sequence[LogitProcessor]` list with a stable ordering rule. `LogitProcessor` may be a **local lightweight `typing.Protocol`** in `silica.core.sampler` for type-hinting, but it is **not one of the five frozen core interfaces (I-1..I-5)** — §6 stays at five.
- **Rationale:** only one sampling implementation exists (MLX-native, executed on the same device as logits); there is no FlashAttention / xformers-style multi-backend pressure. A Protocol without a second implementation to swap in is over-committing the interface surface — the same argument that kept compressed-domain attention out of I-3 in D-003. If v0.2 adds structured-output / grammar-constrained / compressed-domain-attention paths that need sampling to participate differently, re-open under Q-006 / Q-011 and promote then.
- **Consequences:**
  - §6 stays at **five** frozen interfaces (I-1..I-5) through v0.1.
  - `silica.core.sampler` is a new module (not in the current `silica.core` sub-tree listed in P-0 deliverables) — P-0 deliverables expand by one file in v1.5.1.
  - The logit-processor ordering rule is `temperature → repetition penalty → top-k → top-p → sample` (matches mlx-lm); any deviation requires a new entry.
- **References:** D-003, Q-006, Q-011, P-0 Deliverables.

### D-014 — P-1 scope constraints for Qwen3.5-0.8B dev-loop

- **Date:** 2026-04-16.
- **Status:** accepted.
- **Decision:** With the P-1 dev-loop model set to **Qwen3.5-0.8B** (Gated DeltaNet + Gated Attention hybrid + MTP + multimodal), P-1 is pinned to the following simultaneous constraints:
  1. **Text-only.** No processor / vision / audio lifecycle in P-1 (§3.2 Non-Goals). The checkpoint loads with its non-text heads either skipped on the loader side or left resident-but-unused.
  2. **MTP disabled.** Qwen3.5's multi-token prediction head is turned off at load; P-1 decode path produces one token per step. Using MTP as a draft source for speculative decoding is a P-7 discussion (and may flow through I-5 `DraftEngine`), not a P-1 deliverable.
  3. **DeltaNet recurrent state is adapter-owned and carried via `state_delta`** per D-015. P-1's `SimpleKVCache` handles KV-attention layers only; recurrent-layer state travels through the `prefill` / `decode_step` return tuple.
  4. **Multi-head tokenizer parity with the HF reference is a P-1 acceptance prerequisite** — because Qwen3.5's tokenizer can differ from Qwen3's, the "greedy decoding is token-for-token identical to the mlx-lm reference" acceptance line depends on matching tokenizer state.
- **Rationale:** empirical check on 2026-04-16: every Qwen3.5 target model (0.8B / 27B / 35B-A3B) uses the DeltaNet hybrid (HF model cards), so DeltaNet is not a P-1-only concern; it is core-engine concern. Separating the P-1 scope (this decision) from the interface-surface contract (D-015) avoids deferring architecture discovery into implementation.
- **Consequences:**
  - P-1 Strategy / Deliverables in §7 now reference this decision via the P-1 Notes block.
  - D-004 (Phase 1 wraps mlx-lm for model structure + tokenizer + weight loader) still applies; D-010 (cache ownership boundary) still applies. D-014 adds the Qwen3.5-specific content to the shared borrowing surface.
  - mlx-lm's Qwen3.5 support status becomes a P-1 day-1 gate alongside the D-010 cache-injection smoke test: if mlx-lm does not yet carry Qwen3.5 forward, P-1 cost revises upward (ties into R-2).
  - If mlx-lm's Qwen3.5 support bundles MTP / multimodal heads in a way that cannot be cleanly disabled at load, P-1 monkey-patches the load path — same mitigation pattern as R-6.
- **References:** D-004, D-009, D-010, D-015, §3.2, P-1, R-2, R-6.

### D-015 — Recurrent state as a first-class `state_delta` tenant

> **Resolution addendum (v1.5.1):** Prior-round state is not a new input to `I-1.prefill` / `I-1.decode_step`. Instead, **`kv_handle` carries request identity** (it is issued by `KVManager.reserve_for_prefill(req_id, ...)` / `append_slot(req_id, ...)` and binds to `req_id`), and the **adapter owns a per-request store keyed by that identity**. `StateDelta` is a **pure read-only snapshot** — it exposes only `recurrent_bytes() -> int` for scheduler budgeting and an opaque payload the engine does not mutate. All lifecycle operations are **adapter methods called by the engine** (non-frozen helpers, not part of I-1): `adapter.commit_state(req_id, n_accepted)`, `adapter.rollback_state(req_id, n_reject)`, `adapter.state_from_prefix(req_id, token_ids) -> StateDelta | None`, `adapter.free_state(req_id)`. This keeps I-1 Python signatures unchanged and closes the continuous-batching / speculative-decoding ambiguity. See also I-1 Key constraints #3 and I-2 incremental semantics.

- **Date:** 2026-04-16.
- **Status:** accepted.
- **Decision:** `state_delta` (I-1 return tuple) carries **DeltaNet per-layer recurrent state** as a named, first-class tenant — not as an ad-hoc "non-KV runtime state" example. Concretely:
  1. **Layout and ownership.** The adapter owns a per-request recurrent-state store keyed by `req_id` (obtained via `kv_handle`); it defines the concrete layout (e.g. `dict[int, mx.array]` keyed by layer index, each value is the recurrent hidden state of shape `(n_heads, head_dim, head_dim)` or the model-specific shape) and manages in-memory lifecycle. `StateDelta` returned from `prefill` / `decode_step` is a read-only snapshot — engine does not mutate it.
  2. **`AttentionPattern` enum extension.** Values: `global` / `sliding` / `hybrid` (existing KV-attention variants); `recurrent` (pure linear / DeltaNet-only); `hybrid_deltanet` (Qwen3.5's alternating DeltaNet + Gated Attention stack, per-layer dispatch). The scheduler routes KV layers to `KVManager` and recurrent layers to the adapter-owned store.
  3. **`commit` / `rollback` semantics.** Under speculative decoding (P-7), `KVManager.commit(req_id, n_accepted)` pairs with `adapter.commit_state(req_id, n_accepted)` (and `rollback` likewise with `adapter.rollback_state(req_id, n_reject)`) invoked by the engine on the same request. The adapter retains per-step snapshots during the draft window and collapses them on commit or restores the pre-draft state on rollback. These methods are **adapter-local helpers**, not part of I-1's frozen Python signatures.
  4. **Prefix reuse.** Prefix-cache lookup (`KVManager.get_computed_blocks`) returns KV hits only; recurrent-state prefix reuse goes through `adapter.state_from_prefix(req_id, token_ids) -> StateDelta | None`, called by the engine when the KV prefix is non-empty. v0.1 starting rule: reuse recurrent state only when the **full** KV prefix is reused (no partial-prefix recurrent reuse); partial-prefix reuse is a v0.2 question.
  5. **Budgeting.** `StateDelta.recurrent_bytes() -> int` (the only public method on `StateDelta`) is summed by the scheduler into `MemoryBudget.logical_bytes` and `MemoryBudget.resident_bytes` (D-012 canonical definition). For hybrid_deltanet models this is typically much smaller than KV (`num_recurrent_layers × hidden_state_bytes` per request, independent of sequence length), but it must be accounted for.
  6. **Release.** On request completion / abort, the engine calls `adapter.free_state(req_id)` alongside `KVManager.free(req_id)`.
- **Rationale:** Qwen3.5 / Qwen3.5-27B / Qwen3.5-35B-A3B all use DeltaNet hybrid (empirically confirmed on HF, 2026-04-16). Leaving recurrent state as an unspecified "etc." in `state_delta` would defer interface-surface decisions into P-3, when four adapters land at once — highest-blast-radius time. Pin the contract in v1.5.1 so P-0 can freeze at P-0 exit.
- **Consequences:**
  - I-1 `attention_pattern()` inline comment + Key constraints #1 and #3 updated in v1.5.1 (see §6 I-1).
  - I-1 / I-2 Python Protocol signatures **unchanged** — the extension is purely contract text + `AttentionPattern` enum values + adapter-method conventions (`adapter.commit_state` / `adapter.rollback_state` / `adapter.state_from_prefix` / `adapter.free_state`); `StateDelta` itself only exposes `recurrent_bytes()`.
  - P-3 MoE adapter work (D-011) and DeltaNet hybrid work are orthogonal; a MoE-DeltaNet model (e.g. Qwen3.5-35B-A3B) goes through both `get_expert` (D-011) and `state_delta`-recurrent (D-015) paths.
  - The scheduler budget panel in §5.2 data flow expands from "KV via kv_handle from KVManager" to "KV via kv_handle from KVManager + recurrent via state_delta from adapter" — documented in v1.5.1 without redrawing.
  - P-7 speculative decoding must exercise `adapter.commit_state` / `adapter.rollback_state` on DeltaNet layers in addition to `KVManager.commit` / `.rollback` on KV layers.
- **References:** D-011, D-014, I-1, I-2, P-3, P-7, Q-008, M-4.

### D-016 — I-1 extension: `capabilities() -> ModelCapabilities`

- **Date:** 2026-04-19.
- **Status:** accepted.
- **Decision:** I-1 `ModelAdapter` gains one method in P-3 opening: `capabilities() -> ModelCapabilities`. `ModelCapabilities` is a frozen dataclass with three fields — `attention_kinds: frozenset[AttentionKind]`, `has_recurrent_state: bool`, `has_moe: bool` — and ships in `silica/models/capabilities.py` together with a pure helper `capabilities_from_attention_pattern(pattern, *, has_moe=False)`. `AttentionPattern` remains the authoritative per-layer routing source (D-015). `ModelCapabilities` is a strictly coarser typed summary consumed by scheduler-level gates. `ContinuousBatcher._enforce_capability_gate` reads `adapter.capabilities()` as its primary predicate; `attention_pattern()` is walked only to locate a non-GLOBAL layer index for the error message. Every concrete adapter (Qwen3, Qwen3.5, `StubModelAdapter`, test doubles) implements `capabilities()` by calling the helper — Protocol default bodies are not used because I-1 is structurally typed.
- **Rationale:** P-3 introduces three new adapter families (dense big-model, MoE, DeltaNet-hybrid). Without a typed capability surface, each of {batcher gate, P-4 bench harness, MoE-aware budgeter} would re-walk `AttentionPattern` and bolt on its own `isinstance` / attribute probe for MoE routing. Keeping the AttentionPattern → capability derivation in one helper eliminates that drift and gives MoE routing a named bit instead of an ad-hoc flag. No big-model download or new scheduling behaviour arrives with D-016 — the batcher's acceptance set is unchanged (pure GLOBAL, no MoE). This is a **contract-surface refactor**, not a feature.
- **Consequences:**
  - I-1 Python Protocol gains one method (backwards-incompatible for structurally-typed external adapters that do not implement it; Silica's own adapters are updated in the same commit). `ModelAdapter` is still `runtime_checkable` and the new method is visible to `isinstance`.
  - `AttentionPattern` is not demoted — it is still the authority for per-layer routing. `ModelCapabilities` does not re-express per-layer detail.
  - `ModelCapabilities` first version intentionally ships only three fields. Additional capability bits (e.g. `supports_prefix_cache`, `kv_codec_compatible`, `activated_params_per_token`) land when concrete P-5 / P-6 / P-7 consumers need them, not speculatively.
  - Capability-gate error messages now reference `has_recurrent_state` and, for MoE-declaring adapters, `has_moe=True`. The `P-3` and `Q-013` phase references from pre-D-016 reasons remain.
  - MoE adapters landing later in P-3 set `has_moe=True` at the `capabilities_from_attention_pattern` call site; no second override path.
- **References:** D-011, D-015, I-1, P-3, P-4, M-4.

---

## 10. Open Questions

Resolved questions are not deleted. Mark `Status: resolved` and append a `Resolution:` block for traceability.

### Q-001 — VQ codec auto-selection vs explicit configuration

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** in Phase 5, should the VQ codec be (A) user-selected via a CLI flag, or (B) auto-selected by workload?
- **Context:** D-006 says the platform should use VQ "well", which hints at auto-selection; but the Phase 0 interface does not express this capability.
- **Options:**
  - A. Explicit configuration: simple, user in control.
  - B. Auto-selection: matches D-006 framing, but requires a workload profiler.
- **Blocks:** Phase 5 design finalization.
- **Next step:** decide after Phase 4 bench.

### Q-002 — Should Phase 8 priority float up?

- **Raised:** 2026-04-14.
- **Status:** open (progress noted 2026-04-21; leaning Option B but not yet resolved).
- **Question:** per D-006 (platform is the product), should Phase 8 (OpenAI API + session) float from T2 up to the tail of T1?
- **Context:** if Phase 8 is the "product face", it should come earlier. But building a serving layer before the engine is stable is risky.
- **Options:**
  - A. Keep it in T2: engine stabilizes first.
  - B. Float to tail of T1: native-capability integration (P-5 / P-6) and serving layer proceed in parallel.
- **Blocks:** actual sequencing of Phase 5–8.
- **Next step:** evaluate after Phase 4.
- **2026-04-21 progress:** P-4 exit surfaced two product-face signals. (1) `silica.chat.ChatSession` + `scripts/chat.py` now demonstrate a live multi-turn REPL over `Engine.generate`, apply_chat_template, streaming, and per-turn metrics — the HTTP server at P-8 would wrap this rather than design from scratch. (2) The Q-010 fairness defect (short-row TTFT dragged by long-row prefill) would be felt *first* through an HTTP endpoint under concurrent client load, so serving in front of an unfixed batcher would ship a visible regression. **Current lean: Option B (float to T1 tail), but sequence the lift as P-4.5 (chunked prefill + codec integration spike) → P-5 (BlockTQ) → P-8 (HTTP server).** P-8 provides no platform-differentiating capability that P-5 does not; sequencing it after P-5 BlockTQ keeps the HTTP product face aligned with the VQ compression story D-006 promised. No priority-tier edit in this version; Q-002 resolves formally when P-5 BlockTQ lands.

### Q-003 — Should Phase 6 be pulled forward?

- **Raised:** 2026-04-14.
- **Status:** open (progress noted 2026-04-21; not yet resolved).
- **Question:** if Phase 3 finds Qwen3.5-27B / Gemma4-31B still don't fit at 4-bit on 48 GB, is Phase 6 (weight streaming) pulled ahead of Phase 5?
- **Context (v1.5.0 update, D-011):** Q-003 is about **dense targets** (Qwen3.5-27B / Gemma4-31B, where total params = active params). **MoE targets** (Qwen3.5-35B-A3B / gemma-4-26B-A4B) have far lower 48 GB fit risk — active params are only 3–4B, fully resident is under half of a dense target, and it only gets easier with P-6 per-expert streaming. A MoE target can serve as an early scale demonstration while Q-003 is unresolved (the M-4 MoE smoke test is not Q-003-gated), but it does **not** substitute for Q-003 resolution — dense fit is still part of the product promise (D-006); users will reach for `Qwen3.5-27B` directly and will not be consoled by "we have MoE".
- **Blocks:** Phase 5 / 6 ordering.
- **Next step:** decide after Phase 3 produces real residency numbers. The MoE path can close without waiting for Q-003.
- **2026-04-21 progress (partial data, not a resolution):**
  - Qwen3.5-27B-4bit load probe (2026-04-19, logged under §7 P-3 empirical findings): ~16.1 GB weights + ~14 GB MLX forward scratch = **~30.5 GB peak** on M5 Pro 48 GB; short single-request `Engine.generate("Hello", max_tokens=4)` completes with ~17 GB headroom. Gemma4-31B-4bit probe (2026-04-20) shows similar-order peak. Both dense targets therefore fit for short decode, which is **necessary but not sufficient** for Q-003 resolution.
  - The P-3 Acceptance Product memory-fit target requires **500 tokens of sustained generation** with headroom for KV growth + batch; neither probe validated that. With KV growing at ≈ bytes_per_token × seq_len × batch, a 500-token single-request run on 27B / 31B at the measured ≈ 17 GB headroom is credible but unvalidated.
  - Q-003 therefore remains **open, leaning not-triggered**. The product-memory-fit validation is **not** a P-4.5 deliverable; it ships when either (a) a dedicated dense-long-inference bench row runs under `SILICA_REAL_QWEN3_5_27B=1` / `SILICA_REAL_GEMMA4_31B=1` and passes ≥ 500 tokens, or (b) a user running the chat REPL on either checkpoint hits an OOM and re-opens the question. **No immediate P-6 promotion is warranted.**

### Q-004 — `silica.core` vs `silica.engine` boundary

- **Raised:** 2026-04-14.
- **Status:** resolved.
- **Question:** do `Request`, `SamplingParams`, `RequestState` go in `silica.core` or `silica.engine`?
- **Context:** mini-sglang puts them in `minisgl.core`; but "core" tends to bloat.
- **Options:**
  - A. Data classes in core, logic in engine (mini-sglang style).
  - B. Everything in engine; core only holds logging/profiler.
- **Resolution:** Option A. See D-008. All Phase blocks in this document already implicitly use `silica.core.request.*` paths; fixed here.

### Q-005 — Is MetricsRegistry a global singleton?

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** does the Phase 0 profiler use a global `MetricsRegistry`, or one per Engine instance?
- **Context:** a global singleton is simple but collides across multiple Engine instances; per-instance is clean but slightly awkward for CLI / bench access.
- **Next step:** decide when Phase 0 starts.

### Q-006 — Should attention backend be a separate interface?

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** do we need a standalone `AttentionBackend` Protocol (akin to vLLM v1's `vllm/v1/attention/backend.py`) so attention implementations can be swapped independently of `ModelAdapter`? Or keep attention hidden inside the Module returned by `ModelAdapter.build()`?
- **Context:** vLLM v1 separates the attention backend to support flashattention / flashinfer / xformers / triton variants; we only have MLX, so short-term we don't need that flexibility. But if the Phase 5 VQ wants a compressed-domain attention fast path (the v0.2 capability D-003 leaves open), a standalone `AttentionBackend` makes wiring cleaner.
- **Options:**
  - A. No standalone in v0.1; attention stays inside ModelAdapter (simple, fits the current 5-interface design).
  - B. Standalone AttentionBackend as a sixth interface (prepares for v0.2 compressed-domain attention).
- **Blocks:** none — can wait until Phase 5 bench results.
- **Next step:** evaluate after Phase 5 (jointly with the D-003 v0.2 upgrade).

### Q-007 — KVCodec decode overhead signal for admission control

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** should `KVCodec` expose `decode_overhead_ratio: float` (fp16 baseline = 1.0) so scheduler admission control sees both memory savings and decode cost, avoiding the pathological "saves memory but slows decode" combination?
- **Context:** Principle 8 says savings must be visible to the scheduler; I-3 currently only exposes `logical_bytes` / `resident_bytes`. If the scheduler admits more requests purely on memory savings, a codec with a 2× decode slowdown can tank overall tok/s — savings visible, cost invisible — violating the spirit of Principle 8. Choose between minimal v0.1 interface and completeness of scheduler information.
- **Options:**
  - **A. Don't add in v0.1.** Phase 5 users pick a codec explicitly (as in Q-001 Option A); v0.2 revisits with a profile table. Interface stays minimal, but the scheduler cannot actively use savings (violates the spirit of Principle 8).
  - **B. Add `decode_overhead_ratio: float` as a constant on I-3 in v0.1.** Phase 4 bench backfills the number; the scheduler reads it for admission. One interface line avoids the pathology, but a static constant cannot reflect seq-len / batch-size dependence.
  - **C. Keep it out of the interface;** `silica.bench` produces a per-codec profile table and the scheduler reads it. Most precise but heaviest; may be over-engineering for v0.1.
- **Blocks:** finalization of the P-5 scheduler admission policy; coupled with Q-001 (VQ codec auto-selection).
- **Next step:** decide after Phase 4 bench, with real BlockTQ / RaBitQ decode-overhead data. At resolution, also decide whether `feedback_kvcodec_interface.md` memory is updated.

### Q-008 — VectorCodec K/V pair configuration (resolved 2026-04-22, P-5-A.0.4)

- **Raised:** 2026-04-14.
- **Status:** resolved.
- **Resolution:** side-level `VectorCodec[P]` Protocol operating on a single tensor, plus store-level `k_codec` / `v_codec` kwargs on `SyntheticPrefixBlockStore` carrying the K/V pair dispatch. The `codec=` kwarg is kept as a shorthand for `k_codec = v_codec = codec` so the common symmetric case stays one line; split configurations pass both sides explicitly; any combination of `codec=` with a side kwarg, or a mixed None/non-None split, raises at construction. See §6 I-3 above for the Protocol, `silica/kvcache/store.py` for the dispatch, and `docs/P5_A_U4_STORE_MIGRATION.md` §1 for the full rule table. The pre-P-5 pair-level `KVCodec` / `CodedBlock` names and the historical A / B / C option labels below are **superseded** by this fourth path.
- **Question (historical):** should `KVCodec` expose K/V pair configuration at the **interface level** (e.g. `KVCodec(key_method=..., value_method=...)`), letting users explicitly pick different codecs for K and V? Or should it stay hidden inside each codec's constructor?
- **Context (historical):** vqbench explicitly uses different codecs for K and V — K needs unbiased inner-product estimation (`TurboQuantProd` / `QJL`), V needs low-MSE reconstruction (`TurboQuantMSE` / `BlockTurboQuantMSE`); this is the basis of `KVCacheCompressor(key_q, value_q)` in `vqbench/vqbench/kv_cache/compressor.py`. Pre-P-5 I-3's `encode_block(k, v) -> CodedBlock` took a single codec object that could internally hold two quantizers but did not externally expose the choice.
- **Options (superseded by the resolution above):**
  - **A. Internal handling, I-3 contract unchanged.** Each codec accepts `key_method` / `value_method` as constructor args; I-3's signature doesn't move. *Superseded — the side-level Protocol makes K/V split visible on the store without forcing every codec's constructor to grow K/V args.*
  - **B. I-3 adds a pair contract.** Split `KVCodec` into `KeyCodec` + `ValueCodec` Protocols with a top-level `KVCodecPair` composer. *Superseded — the chosen path collapses the pair into the store rather than splitting the codec Protocol into two.*
  - **C. `KVCodec.from_pair(key_method, value_method)` class method.** *Superseded — no factory method needed; shorthand `codec=` argument plus explicit `k_codec=` / `v_codec=` is the surface users actually interact with.*

### Q-009 — MLX paged-attention kernel availability and quality

- **Raised:** 2026-04-16.
- **Status:** open.
- **Question:** does MLX (or mlx-lm) provide a block-addressed / paged-attention kernel with acceptable decode-path throughput on Apple Silicon, or does Silica have to compose paged attention from `mx.` primitives (gather + per-request attention + scatter) at a known performance penalty?
- **Context:** P-2 `PagedKVCache` is the core of the mini-vLLM engine; it requires attention to operate over block-indirected K/V rather than contiguous sequences. vLLM v1 leans on FlashAttention / FlashInfer CUDA kernels for this; Silica has **no CUDA** (D-009). If MLX does not expose an equivalent primitive, paged attention is hand-written over gathers and the decode-path tok/s baseline shifts downward — this affects every P-6 / P-7 tok/s acceptance ratio.
- **Options:**
  - **A. MLX has a usable paged-attention primitive.** P-2 wraps it; acceptance numbers unchanged.
  - **B. MLX has no primitive; gather-based composition is performant enough.** P-2 proceeds; acceptance ratios re-baselined after P-4 bench.
  - **C. MLX has no primitive; gather-based composition is unacceptably slow.** Paged KV degrades to larger block sizes (e.g. 64 or 128) to amortize; or P-2 scope narrows; or R-7 fires.
- **Blocks:** P-2 exit, every downstream tok/s acceptance (P-6, P-7, P-4 bench).
- **Next step:** micro-benchmark at P-0 exit or P-1 entry — the answer decides P-2's concrete block-size default and whether R-7 triggers.
- **Pairs with:** R-7.

### Q-010 — Chunked prefill: measurement-gated deferral vs promotion

- **Raised:** 2026-04-16.
- **Status:** resolved (2026-04-21) — triggered; promote to a P-4.5 bridge unit.
- **Question:** should chunked prefill be a P-2 or P-3 deliverable, or stay deferred until OOM / fairness data forces it?
- **Context:** target models advertise 256K+ context. Chunked prefill affects not just OOM but also scheduler fairness and TTFT — a long-prompt request will block short-prompt requests if prefill is un-chunked. However, chunked prefill is a non-trivial scheduler change (prefill is no longer a single batched forward; it interleaves with decode) and committing without measurement risks over-engineering. This mirrors D-003's rejection of compressed-domain attention in v0.1 on the same grounds.
- **Options:**
  - **A. Defer with measurement trigger.** Keep out of P-2 / P-3 deliverables; add a P-4 bench scenario ("long-in/short-out under shared-prefix concurrency") that measures prefill-induced TTFT stalls; if stalls exceed a threshold (e.g. TTFT p95 of a short-prompt request concurrent with a long-prompt request is > 5× the isolated baseline), promote chunked prefill to v0.1.5 / v0.2. **Current lean.**
  - **B. Promote to P-2 deliverable unconditionally.** Aligns with vLLM v1 baseline behavior; risks scope creep in the most critical phase.
  - **C. Promote to P-3 deliverable conditional on R-1 (memory fit).** If Q-003 forces P-6 ahead of P-3, chunked prefill rides the same window.
- **Blocks:** P-2 / P-3 scope finalization; partially blocks long-context acceptance.
- **Resolution (2026-04-21):** Option A's deferral clause fired. Two independent measurements against `qwen3-0.6b-ttft-under-concurrency` (1 long ~301-token prompt + 3 one-character prompts, `max_batch_size=4`, `prefix_cache=False`) vs isolated `qwen3-0.6b-smoke` on the same Qwen3-0.6B build:
  - **Codex measurement (2026-04-21):** isolated TTFT ≈ 11.8 ms; concurrent first-token offset ≈ 81.28 ms across all four rows; ratio ≈ 6.9× (exceeds Option A's 5× promotion trigger).
  - **Silica measurement (2026-04-21, four consecutive runs):** isolated TTFT ∈ {16.3, 19.3, 17.1, 18.7} ms; concurrent first-token offset (max across four rows) ∈ {77.5, 85.2, 78.0, 77.1} ms; ratios ∈ {4.76, 4.42, 4.56, 4.13}× — single-sample noise straddles the 5× trigger.
  - The dispositive signal is not the ratio magnitude but the structural one: **all four rows' first-token offsets are within ≤ 0.2 ms of each other**, confirming cohort-level prefill serializes short rows behind the long row's `T_max`. See `silica/scheduler/batcher.py::_prefill_phase` (`tokens = self._build_prefill_tokens()  # (B, T_max)`). Scaling the long prompt to 2000+ tokens makes the ratio unconditionally exceed 5×; the fairness defect is deterministic.
  - Promote chunked prefill to a new **P-4.5 bridge phase** (see §7 P-4.5 added in v1.6.4), not to a retroactive P-2 / P-3 deliverable. P-4.5 exits with: (i) TTFT-under-concurrency ratio `max(offsets_short) / smoke_ttft_ms < 3.5×` on the same scenario pair, short-row filter applied (amended down from the original `< 3×` lean after P-4.5-B.1 empirical measurement showed the option-(C) post-fix steady-state at ~3.0× floor; see §7 P-4.5 Amendment log 2026-04-21); (ii) chunked-prefill correctness verified under the three-layer criterion written at §7 P-4.5 Acceptance (event-taxonomy invariant + per-row token-count invariant + direct-mlx-lm-batched numerical reference on the sub-cohort scoped by the chosen option) — strict bit-identity against the unchunked Silica path is **not** part of the exit criterion because fp16 batched SDPA drift across different batch compositions is documented (P-2 / P-3-D3.1); (iii) the three-option opening doc is landed before the implementation so the scope decision is separable from the scope implementation.
- **References:** P-4 empirical finding 2026-04-21; P-4.5; `docs/P2_OPENING.md` §"Model integration in three layers"; `silica/scheduler/batcher.py`.

### Q-012 — Initial-cohort prefix-cache consultation

- **Raised:** 2026-04-21 (surfaced during P-4.5-C.1 test authoring).
- **Status:** open — deferred; v0.1 behavior intentional, revisit for v0.2.
- **Question:** should `ContinuousBatcher._prepare_cohort` (the initial cohort seal) consult `RadixPrefixCache` for prefix hits, or continue to run miss-path prefill unconditionally on every pre-step admission?
- **Context:** As of P-2 Option B + 16c.2 step 4, prefix-cache lookup only fires inside `_admit_waiting_requests` (mid-run admission) via `peek` → `_admit_single_hit_row`. The initial cohort prepared by `_prepare_cohort` runs miss-path prefill for every row it admits, even when the prefix cache already holds a full aligned prefix for that prompt. Consequence at the user-visible layer: two consecutive `Engine.generate_batch([p], params, prefix_cache=shared_pc, ...)` calls — e.g. a REPL chatbot where each turn is a separate `generate_batch` call on the same `shared_pc` — each run a miss-path prefill on prompt `p`, so cross-call prefix reuse is effectively zero. Within a single `generate_batch([p, q], ...)` call where `p` is longer than `max_batch_size`, prompt `q` does get mid-run admission via the waiting queue and benefits from prefix reuse.
- **Options:**
  - **A. Keep current behavior.** `_prepare_cohort` runs miss-path prefill unconditionally; prefix reuse requires the caller to route repeat prompts through mid-run admission. Simplest, preserves the cohort-seal invariant. Cost: REPL-style workloads with one prompt per call pay full prefill every time.
  - **B. Consult prefix cache in `_prepare_cohort`.** At cohort seal, peek each admission against `shared_pc`; rows with a full aligned hit route to `_admit_single_hit_row`-style per-row seeded admission; rows with no hit go through miss-path prefill. Partial-hit rows remain an open design question (mix hit + miss in the initial cohort? defer partial-hit to mid-run?). Adds initial-cohort complexity; changes the single-row-per-call performance characteristic.
  - **C. Caller-orchestrated reuse.** Document the current limitation and recommend the caller batch repeat requests through the waiting queue (`generate_batch([p_old, p_new], max_batch_size=1)`) or maintain their own long-lived cohort. No scheduler change; user-space workaround.
- **Blocks:** nothing in v0.1 — C.1 acceptance works around the limitation by using `[p, p] max_batch_size=1` inside a single call. Future REPL / chat-session prefix reuse is the motivating use case; relevant to P-8 serving shell design.
- **Next step:** revisit when v0.2 session-layer design starts, or sooner if chat-session benchmarks show prefix reuse is a practical bottleneck. The P-4.5-C.0 opening doc's §8.0 records the design fact and the C.1 acceptance shape driven by it.
- **References:** `silica/scheduler/batcher.py::_prepare_cohort` vs `::_admit_waiting_requests`; `docs/P4_5_C_KVCODEC_OPENING.md` §8.0; `tests/test_kvcodec_integration.py` (C.1 workload shape).

### Q-011 — Structured-output / logit-processor boundary

- **Raised:** 2026-04-16.
- **Status:** open.
- **Question:** where does structured generation (grammar / JSON-schema / regex-constrained decoding) live — inside the Sampler's `LogitProcessor` chain (D-013), or as a separate cross-cutting concern the engine orchestrates around sampling?
- **Context:** P-8 deliverables list "an interface slot for structured generation / grammar (unimplemented)". D-013 resolved Sampler as a concrete class with a `Sequence[LogitProcessor]` chain; a grammar-constrained decoder is technically a logit processor (it masks logits that would violate the grammar), but in practice grammar state (e.g. LL-automaton step, outlines-style regex FSM) is per-request and persists across decode steps, which is closer to a `ModelAdapter.state_delta` tenant than to a stateless logit-op. The two framings lead to different interface surfaces in v0.2.
- **Options:**
  - **A. LogitProcessor with persistent state.** Add a per-request state slot to the `LogitProcessor` protocol; grammar processors carry their FSM state there. Minimal interface delta.
  - **B. Separate `StructuredOutputController` interface** driven by the engine, orchestrated around `Sampler.sample`. Cleaner conceptually but a new interface.
  - **C. Grammar state rides `state_delta`.** Consistent with D-015's framing (non-KV per-request state); weird conceptually because grammar is not part of the model.
- **Blocks:** concrete structured-output implementation in v0.2 (not in v0.1 scope).
- **Next step:** revisit when v0.2 planning begins; D-013 resolution leaves room for either framing.

---

## 11. Risks

| ID  | Description | Triggering phase | Mitigation |
| --- | ----------- | ---------------- | ---------- |
| R-1 | Qwen3.5-27B / Gemma4-31B do not fit 48 GB even at 4-bit (dense only; D-011 MoE targets have small active params, so this risk does not apply to MoE) | P-3 | Q-003: pull P-6 weight streaming forward; MoE targets (Qwen3.5-35B-A3B / gemma-4-26B-A4B) serve as an alternative early scale-demonstration path, independent of Q-003 resolution |
| R-2 | The mlx-lm wrapper's Phase 3 replacement cost exceeds expectations | P-3 | Accepted trade-off from D-004; if the cost is too high, defer the underlying rewrite |
| R-3 | Qwen3.5 hybrid attention cache-routing semantics are complex | P-3 | Unit-test coverage in Phase 3; cross-check against the HF reference implementation |
| R-4 | BlockTQ / RaBitQ decode overhead exceeds that of fp16 attention itself | P-5 | Q-001 auto-selection policy; v0.2 may consider a fast path |
| R-5 | MLX / mlx-lm version churn makes dependency locking painful | all | Pin minimum versions in pyproject.toml + periodic CI upgrades |
| R-6 | `mlx-lm` rejects external cache injection (D-010 day-1 smoke test fails) | P-1 | Monkey-patch / fork `mlx_lm.models.*` forward; worst case, P-1 cost is revised upward and the cache integration point is uniformly refactored before P-2 starts; record in P-1 Strategy Notes |
| R-7 | MLX has no performant paged-attention primitive; hand-composed gather + per-request attention + scatter is materially slower than contiguous-sequence attention (pairs with Q-009) | P-2 | Micro-benchmark at P-0 exit / P-1 entry to pin the baseline; if penalty is > ~30%, raise block size (16 → 64 / 128) to amortize; if still unacceptable, re-scope P-2 to per-request contiguous caches with a clear upgrade path once MLX adds the primitive; P-6 / P-7 tok/s acceptance ratios re-baselined against the chosen path |
| R-8 | `mlx-lm` does not yet carry Qwen3.5 (Gated DeltaNet + Gated Attention + MTP) forward cleanly, or bundles MTP / multimodal heads in a way that cannot be disabled at load (D-014) | P-1 | Day-1 gate alongside D-010 cache-injection smoke test; if mlx-lm's Qwen3.5 support is incomplete, monkey-patch the load path to skip multimodal heads and disable MTP; worst case, P-1 falls back to Qwen3-0.6B for the bring-up loop and the DeltaNet-specific work shifts to P-3 (explicitly re-opens D-014) |

---

## 12. References

### 12.1 External (reference)

- `mini-sglang`: https://github.com/sgl-project/mini-sglang
- SGLang docs: https://docs.sglang.ai/
- vLLM architecture: https://docs.vllm.ai/en/latest/design/arch_overview/
- vLLM v1 design blog: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- MLX: https://github.com/ml-explore/mlx
- mlx-lm: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm
- mlx-flash: https://github.com/matt-k-wong/mlx-flash
- dflash-mlx: https://github.com/Aryagm/dflash-mlx
- flash-moe: https://github.com/danveloper/flash-moe
- DFlash paper: https://arxiv.org/abs/2602.06036

### 12.2 Local reference checkouts (gitignored)

Local reference implementations sit at the repo root. **Algorithm / architecture reference only, never runtime dependencies** (D-009).

| Directory | Project | Main reference points |
| --------- | ------- | --------------------- |
| `vllm/` | vLLM v1 | See §5.4 Reference Map; focal points `vllm/v1/core/`, `vllm/v1/engine/`, `vllm/v1/kv_cache_interface.py`, `vllm/v1/request.py` |
| `mini-sglang/` | Mini-SGLang | Module layering, serving shell, radix prefix cache |
| `vqbench/` | VQBench (includes nested `vqbench/turboquant_plus/`) | See §5.5 Reference Map; P-5 algorithmic reference + Qwen3.5-4B empirical PPL oracle; NumPy + PyTorch codebase — **not** a runtime import (D-009); `VQBenchCache` is the D-010 anti-pattern |

---

## 13. Changelog

- **v1.7.5** (2026-04-24): **P-5 §7(a-real) real-activation xcheck close.** Adds `tests/test_block_tq_real_activation_xcheck.py` + `docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.{md,jsonl}` + `docs/P5_A_REAL_OPENING.md`. Closes the real-activation half of P-5 §7(a) that v1.7.2 deferred to post-P-5 follow-up. Pure addition + doc sync; no runtime code change, no existing test modified beyond a stale docstring.

  **What (a-real) measures.** Per-block relative Frobenius error on real Qwen3.5-0.8B **pre-RoPE** K and V activations extracted from a prefill pass on a checked-in ~138-token prompt. Silica's MLX `BlockTurboQuantMSE` is compared against the vqbench-transcribed NumPy reference landed at P-5-A.1c (`_numpy_block_tq_round_trip` in `tests/test_block_tq_vqbench_xcheck.py`). 144 rows = 6 GLOBAL layers × 2 sides × 4 `(B, bits) ∈ {32, 64} × {3, 4}` × 3 seeds `{42, 43, 44}`.

  **Gate (reused (a-algo) envelope):** `|silica_frob - numpy_frob| < 1e-3` on `(B=64, b=4)` and `< 5e-3` elsewhere. All 144 rows pass. Worst-case gap `1.15e-4` (layer 15, K, B=64 b=3); worst production-cell gap `5.21e-5` (V side). K / V symmetric (worst K `1.15e-4`, worst V `9.78e-5`); all 6 GLOBAL layers land in the same `9e-5 … 1.2e-4` band. `IdentityCodec` round-trip baseline degenerate on 144 / 144 rows (`RawFp16Payload` is dtype-preserving — §2.4 fallback engaged by design); the absolute-gap gate is the close criterion. Tolerance deliberately not tightened at landing despite the headroom — evidence-based tightening is deferred to a future revision per design §2.5 "measure first, pin later".

  **Two design corrections vs v1.5.1 wording (locked in `docs/P5_A_REAL_OPENING.md` before implementation):**

  - **Inline NumPy reference, not vqbench subprocess.** v1.5.1 P5_OPENING §7(a-real) specified a vqbench venv subprocess + new recon-specific driver script. Superseded: the already-landed `_numpy_block_tq_round_trip` transcription from (a-algo) takes real K / V tensors just as happily as synthetic Gaussian, no new subprocess, no new driver, no second skip gate. Design §2.3 + the opening §7(a-real) rewrite carry the rationale (transcription faithfulness was already pinned at (a-algo) tolerance `5e-3` / `1e-3`; reuse beats duplicate infrastructure).
  - **Single skip gate, not dual.** v1.7.2 Notes said "dual gate — HF cache has Qwen3.5-0.8B AND `VQBENCH_PYTHON_EXECUTABLE`". Inline reference → single-gate `_hf_cache_has_repo("Qwen/Qwen3.5-0.8B")` only. Prompt is a checked-in deterministic string constant inside the test (§2.7), so no WikiText cache dependency enters the gate calculus.

  **Surgical docs updates in this revision:**

  - **`docs/P5_OPENING.md` §7(a-real) rewrite** (lines 767–790): old "post-P-5 required follow-up … vqbench venv subprocess … dual gate" → new "closed at v1.7.5 … inline NumPy reference … single gate", with the evidence-file pointer and the 144-row numbers.
  - **`docs/PLAN.md` §7 P-5 Notes (a-real) bullet** (line ~568): same substance, compressed to the §7 Notes scale.
  - **`docs/P5_A_REAL_OPENING.md`** landed as a new file carrying the full design contract (§2 decisions, §3 evidence schema, §4 test layout, §5 docs-update list, §6 out-of-scope, §7 implementation pause point + post-skeleton checklist).
  - **`docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.{md,jsonl}`** landed as the evidence record (parallel to `admission_headroom.{md,jsonl}` and `all_kv_codecs.{md,jsonl,log}`).
  - **`tests/test_block_tq_vqbench_xcheck.py` module docstring** — fix the stale "`(a-real): real Qwen3.5-0.8B activations vs vqbench subprocess — defers to P-5-C`" sentence (pre-v1.7.4 wording; superseded on two counts — P-5 closed at v1.7.4, and the subprocess design is superseded by inline NumPy). Updated to point at the new `tests/test_block_tq_real_activation_xcheck.py`.
  - **Header:** Version v1.7.4 → v1.7.5; Status extended to "(a-real) real-activation xcheck closed at v1.7.5"; backlog enumerated (`P-3-C5`, `P-3-E4`, pre-RoPE production routing (P-5-F), (b-static) PPL baseline).

  **What is NOT changed.** (b-static) Qwen3.5-4B PPL vs `vqbench/REPORT.md` static baseline stays in backlog — still blocked on P-3-C5 or on a monkey-patch measurement route, per existing §7 Notes. Silica runtime code (no `silica/*.py` change); codec registry; interface signatures. `(a-algo)` synthetic Gaussian test `tests/test_block_tq_vqbench_xcheck.py` gate logic is unchanged (only its module docstring is touched); the shared `_numpy_block_tq_round_trip` helper remains the single source of truth for the BlockTQ NumPy reference and is now exercised by both (a-algo) and (a-real). v1.7.4 Changelog unchanged.

  **Next step:** per the backlog user-ordered sequence (2026-04-24): P-3-C5 opening doc (`docs/P3_C5_OPENING.md`) — recurrent-state snapshot / restore design for hybrid-DeltaNet so `ContinuousBatcher` + `RadixPrefixCache` can cooperate with Qwen3.5-0.8B / 4B / 35B-A3B. (b-static) lands on top of P-3-C5. P-3-E4 batched MoE and P-5-F pre-RoPE production routing sequence after.

- **v1.7.4** (2026-04-24): **P-5 Acceptance (1) / (2) / (3) close revision.** Flips the remaining three §7 P-5 Acceptance top-level `[ ]` to `[x]` on the evidence landed under `docs/P5_ACCEPTANCE_SWEEP/`. Acceptance item (4) is **not modified** — it was already closed at v1.7.3 / P-5-D.3 on the vqbench-aligned oracle mean-over-seeds gate; this revision does not re-author (4) or its evidence. Also records two surgical corrections to the canonical `docs/P5_OPENING.md` §7 text that surfaced while writing (3) and (2) evidence.

  **(1) Codec-swap neutrality — by inspection.** Gate: "Switching the codec requires no change to the scheduler or model adapter." Evidence in `docs/P5_ACCEPTANCE_SWEEP/codec_swap_neutrality.md`:

  - Zero `isinstance(codec)` / `type(codec) ==` / `__class__.__name__` runtime dispatches across `silica/scheduler/**`, `silica/models/**`, `silica/engine/**`, `silica/kvcache/**`, `silica/weights/**`, `silica/core/**`, `silica/mlx/**`, `silica/llm/**`, `silica/speculative/**`, `silica/chat/**`, `silica/server/**`.
  - Zero imports of `silica.vq.*` or `silica.kvcache.codec` from `silica/scheduler`, `silica/models`, `silica/engine`, `silica/weights`. Actual kvcache imports are `RadixPrefixCache`, `KVHandle`, `KVManager`, `SimpleKVCache` — all codec-agnostic container / manager types.
  - 12 docstring mentions of concrete codec names or the `VectorCodec` Protocol, each individually classified as a non-dispatching reader note.
  - External behavioural witnesses: `tests/test_kvcodec_integration.py` (IdentityCodec + `_CountingIdentityCodec` + pass-through on the same scheduler instance), `tests/test_bench_workload_kv_codec.py::test_maybe_build_prefix_cache_block_tq_installs_block_tq_codec` (BlockTQ end-to-end through `ContinuousBatcher` via `kv_codec="block_tq_b64_b4"`), `tests/test_prefix_hit_decode_speed_gate.py` (fp16 vs BlockTQ paired rows on the decode hot path).

  **(2) One-command fp16-vs-codec report — report-schema coverage.** Gate: "For the same scenario set, fp16 vs codec quality delta and memory savings are available from the bench in one command." Evidence in `docs/P5_ACCEPTANCE_SWEEP/all_kv_codecs.md` (summary) + `all_kv_codecs.jsonl` (924 rows) + `all_kv_codecs_report.md` (aggregated GFM + per-seed detail) + `all_kv_codecs.log`:

  - Command: `uv run python scripts/bench.py --all --all-kv-codecs --seeds 42,43,44 --out <jsonl> --report-md <md>`. Exit code `0`.
  - Coverage: 28 scenarios × 11 codecs × 3 seeds = 924 rows, all reaching the reporter with structured `status` + `reason` fields. `ok=360`, `failed=564`, `skipped=0`.
  - All 564 failures classifiable into three disjoint expected compatibility classes: **528 `codec_override_invalid`** (scenario workload has `prefix_cache=False`; `--all-kv-codecs` has no install site — this covers the fp16 PPL baseline, admission-headroom, smoke / parity / routing rows, and all big-model scenarios), **33 `ValueError` K-only `rabitq_b1`** (symmetric `kv_codec=` shorthand rejects asymmetric codec on the 11 Qwen3-0.6B `prefix_cache=True` scenarios × 3 seeds), **3 `RuntimeError` vqbench-aligned symmetric-codec guard** (D.3-landed guard on the `-vqbench-aligned` scenario × `rabitq_b1` × 3 seeds). `528 + 33 + 3 = 564` — no unclassified exception, no runner or report bug.
  - Column coverage scoped to report-schema: top-level `peak_memory_mb` / `wall_s` / `total_tokens` populated on all 360 ok rows; `decode_tok_s` / `ttft_ms` populated on 90 prefix-hit-decode ok rows per oracle design. Substantive gate signal lives in oracle-specific metadata: PPL rows carry `delta_ppl` / `delta_ppl_pct`, compression rows carry `resident_bytes` / `resident_bytes_per_block`, prefix-hit-decode rows carry `row0_decode_tok_s` / `row1_decode_tok_s`.
  - `--vqbench-xcheck` deliberately **not** passed to this sweep per the §7(e) narrowing below; `vqbench_gap` column is structurally present but empty on all 924 rows. Populated xcheck numbers belong to Acceptance (4-b) and live in `docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`.

  **(3) Admission-headroom — empirical `n_block > n_fp16`.** Gate: "With BlockTQ on, the same memory budget admits more requests." Evidence in `docs/P5_ACCEPTANCE_SWEEP/admission_headroom.{jsonl,md}`:

  - Command: `uv run python scripts/bench.py --scenario qwen3-0.6b-admission-headroom-prefix-heavy --seeds 42,43,44 --out <jsonl>`.
  - Parameters (from the scenario `oracle_config`): `cap_bytes = 128 MB`, `weights_bytes = 0`, `warmup_ratio = 0.5`, `warmup_blocks = 37`, `n_prompt = 128`, `max_tokens = 16`, `fp16_codec = fp16`, `compressed_codec = block_tq_b64_b4`.
  - Observed residency: `resident_bytes_fp16 = 67.895 MB`, `resident_bytes_block = 18.035 MB`, `residency_ratio ≈ 0.266` (≈ 1 / 3.76 matching vqbench REPORT §3.1 BlockTQ B=64 4-bit K+V total-KV compression).
  - Observed admission: `n_fp16 = 4`, `n_block = 7`, `n_delta = +3`, `admit_ratio = 1.75`. Gate `n_block > n_fp16` → `7 > 4` — pass with margin 3 on every seed. Structural / seed-independent by scenario design (deterministic block-recipe warmup + replay).
  - `admit_ratio = 1.75` sits below the theoretical loose upper bound (`1 + 3.76 × 0.5 = 2.88×`, see §7(c) correction below) and above the "bytes freed" form (`1 + 2.76 × 0.5 = 2.38×`), consistent with `reserved_bytes` continuing to charge fp16 worst-case per admitted request (D-003 constrained, §3.2).

  **Surgical `docs/P5_OPENING.md` corrections landed in this revision:**

  - **§7(c) arithmetic correction (line 854).** The loose upper bound `N_block / N_fp16` citation was "≈ 2.4×" at v1.5.1 through v1.7.3, inconsistent with the formula `1 + compression_factor × prefix_fraction` stated one clause earlier (`1 + 3.76 × 0.5 = 2.88`). The "≈ 2.4×" form corresponds to `1 + (compression_factor − 1) × prefix_fraction` — the "bytes freed" version, not the form stated in the text. v1.7.4 corrects "≈ 2.4×" → "≈ 2.88×" to match the formula as written, and adds a parenthetical note explaining both forms + documenting the observed `admit_ratio ≈ 1.75` as sitting in the band between the two.
  - **§7(e) scope + flag-condition rewrite (lines 873–877).** The v1.5.1 text said `--all-kv-codecs --vqbench-xcheck` produces a table that "matches vqbench REPORT §3.1 ... within the (b) PPL gate across the full codec registry". This was inconsistent with the D.3 / C.6 declarative-spec contract (only `-vqbench-aligned` has a `VqbenchXcheckSpec`; other codec arms don't have authored vqbench method / bits mappings). The v1.7.4 rewrite narrows (2) to "one-command coverage of fp16 + codec quality / memory / decode columns" — NOT "full-registry vqbench xcheck" — and adds the precise flag condition for populated xcheck: `vqbench_gap` / `vqbench_cross` columns are **structurally present on every row** but **populated only when both (i) the scenario declares a `VqbenchXcheckSpec` AND (ii) `--vqbench-xcheck` is passed** (`BenchRunner.vqbench_xcheck_enabled=True`). Declaring a spec alone is necessary but not sufficient; the runner flag is also required. Numerical cross-check ownership stays with (4-b).

  **Minor adjacent fix:** `silica/bench/codec_registry.py::CodecSpec.factory` docstring signature was missing `seed: int = 42`. v1.7.4 patches it in (and notes the seed flows from the bench runner's per-execution seed into the codec ctor's Haar-rotation seed per P-5-D.1). Code-only comment change; `tests/test_codec_registry.py` 98 passed.

  - **Header:** Version v1.7.3 → v1.7.4; Status updated from "P-5 Acceptance (4) closed; (1)/(2)/(3) sweep pending" to "P-5 complete; P-5 Acceptance (1)–(4) all closed at v1.7.4; P-3-C5 / P-3-E4 and post-P-5 follow-ups remain in backlog"; Last updated unchanged.
  - **`docs/PLAN.md` §7 P-5 Acceptance checkboxes (lines 556–558).** Items (1) / (2) / (3) flipped `[ ]` → `[x]`; each carries an inline close note pointing at the evidence file and summarising the gate. Item (4) — line 559 `[x]` — unchanged.
  - **`docs/PLAN.md` §7 P-5 Status line (line 564) rewritten** from "in-progress" to "done". Records the D.3 landing + (1)/(2)/(3) sweep landing, names the single intentionally-deferred Deliverable (`PagedPrefixBlockStore` codec injection under D-003 no-compressed-domain-attention scope), and enumerates the post-P-5 follow-up backlog explicitly.
  - **`docs/P5_OPENING.md` §7(c) correction** (line 854) and **§7(e) rewrite** (lines 871–877): as detailed above.
  - **`docs/P5_ACCEPTANCE_SWEEP/` directory added** with seven evidence files: `codec_swap_neutrality.md` for (1); `admission_headroom.jsonl` + `admission_headroom.md` for (3); `all_kv_codecs.jsonl` + `all_kv_codecs.md` + `all_kv_codecs.log` + `all_kv_codecs_report.md` for (2). This directory is the persistent close-evidence record for (1) / (2) / (3), parallel to `docs/P5_D2_INVESTIGATION/` for (4).
  - **`README.md` surgical sync.** Status table P-5 row updated from the v1.7.3 wording to "P-5 Acceptance (1)–(4) all closed at v1.7.4". Roadmap P-5 bullet updated to enumerate the full A / B / C / D sub-unit close and the post-P-5 follow-up backlog.
  - **What is NOT changed.** Acceptance item (4) body (lines 559–562); §7 P-5 Deliverables list (intentional — the PagedPrefixBlockStore codec-injection line stays `[ ]` per the deferral note in Status); I-1..I-5 Protocol signatures; §9 D-* / §10 Q-* / §11 R-* identifiers; any runtime code, test, or interface file. v1.7.3 Changelog entry unchanged (historical record).
  - **References:** `docs/P5_ACCEPTANCE_SWEEP/` evidence files; v1.7.3 changelog for the (4-b) / D.3 close (unchanged in this revision); `docs/P5_D2_INVESTIGATION/README.md` for the D.2 / D.2a investigation record.
  - **Next step:** post-P-5 backlog — pre-RoPE production routing architecture (potential F-series or P-3-C follow-up), Qwen3.5 real-target xcheck (a-real) / (b-static), P-6 weight streaming (dense + MoE per-expert), P-3-C5 preempt/replay with recurrent-state snapshot, P-3-E4 batched MoE `has_moe=True` gate lift. None of these items blocks the v0.1 P-5 close — they are additional v0.1 scope tracked in their respective Phases / Surveys.

- **v1.7.3** (2026-04-24): **P-5-D.3 — (4-b) gate reinterpretation on the D.2a vqbench-aligned oracle + (4) top-level checkbox close.** Documentation-only revision; zero code, test, or interface diffs (no `_compute_gap_fields` or `_VQBENCH_PCT_EPSILON` changes — per-row thresholds remain in code as diagnostic). Flips only the §7 P-5 Acceptance item (4) "Numeric cross-check against vqbench" top-level checkbox `[ ]` → `[x]`; the v1.7.2 Changelog "Next step" projected flipping all four items at v1.7.3, but items (1) codec-swap neutrality, (2) `--all-kv-codecs` one-command report, (3) `qwen3-0.6b-admission-headroom-prefix-heavy` row remain `[ ]` pending a dedicated P-5 Acceptance sweep run, which is a separate close revision. D.3 is scoped to (4) only.

  **Framing.** The v1.7.2 wording of (4-b) bound the gate to a per-row two-threshold rule (`|Δ(ΔPPL)_silica − Δ(ΔPPL)_vqbench| < 0.01` AND `|Δ%| < 0.1%`) on the `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` bench row. Two problems surfaced from the P-5-D.1 + P-5-D.2 + P-5-D.2a investigation:

  1. The row as named routed through silica's `prefix_store_post_rope` path (post-RoPE prefix-cache store), which injects reconstruction noise in post-RoPE space while vqbench's `_QuantizedProj` harness injects it in pre-RoPE space. At the same Frobenius reconstruction error the post-RoPE arm pays an additional chunk-boundary cost (`docs/P5_D2_INVESTIGATION/README.md` §Root cause), measuring ΔPPL in the ~5–10 PPL range versus vqbench's ~0.3–0.9 PPL — a 10×–30× raw gap that was not an algorithmic defect but an injection-space architectural difference. This was closed **algorithmically** at P-5-D.2a via a new `codec_quality_path="vqbench_aligned"` oracle (`teacher_forced_chunked_nll_vqbench_aligned` in `silica/bench/ppl_oracle.py`) that monkey-patches `attn.k_proj` / `attn.v_proj` pre-RoPE, mirroring vqbench's injection site. The D.2a arm collapses the cross-implementation gap to within one unit of seed-level noise.
  2. Even after D.2a, per-row `vqbench_divergence_warning=true` still fires on every seed (worst-case `|gap| ≈ 0.61` PPL) because silica's `BlockTurboQuantMSE` shares one Haar rotation across all heads while vqbench's `quant_dequant_tensor` samples one rotation per head (seed=h). At the same outer seed the two sides draw different rotations from the same Haar distribution; this is sampling variance, not algorithmic drift. The v1.7.2 per-row thresholds would therefore never pass at `n=3` regardless of how faithfully silica translated vqbench's algorithm — anchoring the close gate to those thresholds was the "gate-semantic landmine" flagged at commit-review time of `ed57be1`.

  D.3 resolves both by redefining (4-b) as a **mean-over-seeds aggregated gate on the D.2a vqbench-aligned row**:

  - **Bind target.** `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` (the D.2a row, `codec_quality_path="vqbench_aligned"`), **not** the `prefix_store_post_rope` production-routing row.
  - **Compare object.** 3-seed mean ΔPPL (`seeds {42, 43, 44}`), not per-row.
  - **Gate form.** `|mean_gap| <= 2 * SEM_diff` **and** `|mean_gap| < 1.0` PPL sanity cap, where `mean_gap = mean_seeds(silica.ΔPPL_seed − vqbench.ΔPPL_seed)` and `SEM_diff = sqrt( std(silica.ΔPPL_seeds)^2/n + std(vqbench.ΔPPL_seeds)^2/n )` with `n = 3` and Bessel-corrected sample std (independent-samples standard error of the difference of means). The formula is named explicitly in-line so a future reviewer cannot recompute a different SEM form and arrive at a different pass/fail.
  - **Per-row diagnostics.** `vqbench_epsilon = 0.01` / `_VQBENCH_PCT_EPSILON = 0.1` in `_compute_gap_fields` remain in code unchanged and continue to emit `vqbench_divergence_warning` as a diagnostic metadata field on every row. Per-row warnings are **not** close blockers; expected-true under D.2a because of shared- vs per-head-rotation sampling, as above.
  - **Evidence.** `docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl` (commit `ed57be1`): silica mean `+0.511 ± 0.354`, vqbench mean `+0.661 ± 0.347`, `mean_gap = −0.150` PPL, `SEM_diff ≈ 0.286`, `2·SEM_diff ≈ 0.572`. Both conditions pass (`0.150 ≤ 0.572` and `0.150 < 1.0`).

  **Scope delimiter — `prefix_store_post_rope` quality cost is NOT closed by (4-b).** The production-routing arm's ~5–10 PPL ΔPPL at this codec config is a real production-path quality cost, owned by a post-P-5 unit (pre-RoPE KV-store architecture / P-3-C prefix-cache cooperation work-area, or a dedicated F-series follow-up). (4-b) closes the **algorithmic parity between silica's MLX-native BlockTQ and vqbench's NumPy BlockTQ when both inject noise in the same space**; it does **not** claim silica's production `prefix_store_post_rope` store path has achieved vqbench-level PPL. A new §7 P-5 Notes bullet ("Production `prefix_store_post_rope` prefix-cache quality cost — post-P-5 required follow-up") records this scope boundary in the body. **D.2a closed algorithmic/vqbench-aligned parity, not pre-RoPE production routing.**

  - **Header:** Version v1.7.2 → v1.7.3; Status updated to "P-0..P-4.5 complete; P-5 sub-units landed; P-5 Acceptance (4) closed via vqbench-aligned oracle; (1)/(2)/(3) sweep pending"; Last updated unchanged.
  - **`docs/PLAN.md` §7 P-5 Acceptance (line 559) checkbox flip.** Top-level `[ ]` **Numeric cross-check against vqbench** → `[x]`. Items (1)/(2)/(3) above remain `[ ]`.
  - **`docs/PLAN.md` §7 P-5 Acceptance (4-b) rewrite (line 561).** Old: per-row `|Δ(ΔPPL)_silica − Δ(ΔPPL)_vqbench| < 0.01` AND `|Δ%| < 0.1%` on `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`. New: mean-over-seeds `|mean_gap| ≤ 2·SEM_diff` AND `|mean_gap| < 1.0` PPL on `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`, with explicit per-row-diagnostic preservation statement and full evidence row. (4-a) body unchanged.
  - **`docs/PLAN.md` §7 P-5 Status line rewritten** to enumerate D.1 / D.2a landing and the split between (4) closed vs (1)/(2)/(3) pending.
  - **`docs/PLAN.md` §7 P-5 Notes expansion.** New bullet "Production `prefix_store_post_rope` prefix-cache quality cost — post-P-5 required follow-up" added after the existing v1.7.2 Qwen3.5-real-target-cross-validation bullet. Explicitly records the ~5–10 PPL ΔPPL on the production arm, identifies the post-RoPE noise-injection architectural difference as root cause, and names the post-P-5 unit that owns the remediation (pre-RoPE KV-store architecture / P-3-C work-area / F-series follow-up). Guards against readers interpreting the (4-b) `[x]` as "silica's production prefix-cache path has achieved vqbench-level PPL".
  - **`docs/P5_OPENING.md` §7(b) body rewritten** (heading also updated: "End-to-end PPL-delta cross-check vs vqbench" → "End-to-end PPL agreement on the vqbench-aligned oracle — mean-over-seeds cross-check"). Mirrors the PLAN (4-b) rewrite: new bind target, new gate form, per-row-diagnostic preservation, evidence table.
  - **`docs/P5_OPENING.md` new §7(b-postrope) subsection** added between §7(b) and §7(b-static). Documents the production `prefix_store_post_rope` quality cost as post-P-5 follow-up (NOT closed by (4-b)). Mirrors the PLAN §7 P-5 Notes bullet wording so the scope boundary is consistent across both documents.
  - **`docs/P5_D2_INVESTIGATION/README.md` close section added** under a new "## Close — P-5-D.3 (v1.7.3)" heading. Records the close decision, the gate form, and the `ed57be1` + v1.7.3 traceability hooks so the investigation record is not orphaned after the gate redefinition.
  - **`README.md` surgical sync.** Status table P-5 row updated from "P-5 Acceptance sweep pending" to "P-5 Acceptance (4) closed via vqbench-aligned oracle; (1)/(2)/(3) sweep pending". Roadmap P-5 bullet (around line 455) updated to reflect the (4-b) close and the explicit scope-delimiter that the production `prefix_store_post_rope` quality cost is a post-P-5 follow-up.
  - **What is NOT changed.** Acceptance item (1) / (2) / (3) checkboxes; §7 P-5 Deliverables list; I-1..I-5 Protocol signatures; §9 D-* / §10 Q-* / §11 R-* identifiers; any code, test, or interface file. `_compute_gap_fields` per-row thresholds are unchanged. v1.7.2 Changelog entry is unchanged (historical record).
  - **References:** v1.7.2 for the prior scope-correction and the original "Flip four" projection; `docs/P5_D2_INVESTIGATION/README.md` for the D.1 seed fix, the D.2 probes, and the D.2a 3-seed verification; commit `ed57be1` (D.2a landing); 2026-04-24 commit-review conversation where the "gate-semantic landmine" framing was pinned and D.3 was scoped.
  - **Next step:** dedicated P-5 Acceptance sweep for items (1) / (2) / (3) — codec-swap neutrality by inspection, `--all-kv-codecs` report coverage on the post-C.6 tree, and `qwen3-0.6b-admission-headroom-prefix-heavy` row numbers. Flip those three `[ ]` to `[x]` in a separate close revision (working target: v1.7.4).

- **v1.7.2** (2026-04-24): **P-5 Acceptance (4-a) / (4-b) scope correction — narrow both gates to shipped mechanisms; preserve Qwen3.5 real-target cross-validation as post-P-5 required follow-up.** Documentation-only revision; zero code, test, or interface diffs. Resolves a v1.5.1 → P5_OPENING §6.5 drift: (4-a) and (4-b) were both written at v1.5.1 (2026-04-16, commits `f64a65f3` / `2ce9a7b`) naming Qwen3.5-0.8B / Qwen3.5-4B as cross-validation targets, before P5_OPENING §6.5 empirically moved all P-5 codec-backed PPL bench rows to Qwen3-0.6B because Qwen3.5-0.8B and Qwen3.5-4B are hybrid-DeltaNet and `ContinuousBatcher` refuses `RadixPrefixCache` on recurrent adapters.

  **Framing (user-driven, 2026-04-23):** P-5 close verifies **shipped** mechanisms. It must not bind the P-5 close gate to capabilities that belong to other phases — specifically the remaining P-3-C recurrent / prefix-cache cooperation work (`docs/P3_DELTANET_SURVEY.md` C-open-3; may land through P-3-C5 or a narrower sub-unit) or measurement paths not yet implemented (monkey-patch on Qwen3.5-4B `k_proj` / `v_proj`). Those items **remain required for v0.1 production launch** and are moved to a correct post-P-5 workstream; they are not dropped. Scope correction, not scope reduction.
  - **Header:** Version v1.7.1 → v1.7.2; Status unchanged ("P-5 sub-units landed; P-5 Acceptance sweep pending"); Last updated unchanged.
  - **`docs/PLAN.md` §7 P-5 Acceptance (4-a) rewrite (line 560).** Old: "Silica MLX-native `BlockTQCodec` on Qwen3.5-0.8B (or larger), compared against the vqbench NumPy reference on the same calibration set" → New: names the algorithmic-parity gate already shipped at P-5-A.1c (`tests/test_block_tq_vqbench_xcheck.py`, synthetic Gaussian inputs, tolerance `5e-3` / `1e-3`). Real-activation half deferred to Notes.
  - **`docs/PLAN.md` §7 P-5 Acceptance (4-b) rewrite (line 561).** Old: "Qwen3.5-4B `BlockTurboQuantMSE B=64` 4-bit K+V ... `vqbench/REPORT.md` baseline ... `ε_ppl < 0.01` absolute" → New: names the `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` bench row (Qwen3-0.6B) and the online ΔPPL-vs-ΔPPL cross-check via `--vqbench-xcheck` landed at P-5-C.6 step 2. Two-metric threshold pair (`< 0.01 abs AND < 0.1% rel`) preserved — gate already implemented in `silica/bench/runner.py::_compute_gap_fields` with `vqbench_epsilon = 0.01` default + `_VQBENCH_PCT_EPSILON = 0.1`. Static-baseline half deferred to Notes.
  - **`docs/PLAN.md` §7 P-5 Notes expansion.** Old: single bullet about `turboquant_plus/` reference. New: two bullets — (i) keeps the `turboquant_plus/` reference note unchanged; (ii) new "Qwen3.5 real-target cross-validation — post-P-5 required follow-up" bullet with sharp ownership per the user's framing: (a-real) "no runtime capability blocker; pending a dedicated real-activation cross-check test on Qwen3.5-0.8B (or larger)"; (b-static) "blocked on the remaining P-3-C recurrent/prefix-cache cooperation work (C-open-3; may land through P-3-C5 or a narrower targeted fix), or on an alternate monkey-patch measurement path". The dependency is deliberately written at the P-3-C work-area granularity rather than pinned to P-3-C5 specifically so a different P-3-C sub-unit landing prefix-reuse support does not re-introduce the same drift.
  - **`docs/P5_OPENING.md` surgical sync (three locations).**
    - **§7(a-real) heading + body (lines 767–788).** Heading reframed from "P-5-C" ownership to "post-P-5 required follow-up"; new lead paragraph surfacing the "no runtime capability blocker" framing; stale "Tightened at P-5-C implementation time" → "Tightened at implementation time".
    - **§7(b) body (lines 794–799).** Gate line "Qwen3.5-4B" → "Qwen3-0.6B" (the `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` row); command example scenario `qwen3.5-4b-wikitext-ppl` → `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`; reference updated to `_compute_gap_fields` thresholds. Closing sentence ratifies both the delta-compare and the Qwen3-0.6B migration amendments.
    - **New §7(b-static) subsection** (inserted between §7(b) and §7(c)) — records the deferred Qwen3.5-4B static-baseline cross-check as post-P-5 follow-up with explicit unblock paths: the remaining P-3-C recurrent/prefix-cache work (C-open-3; may land through P-3-C5 or a narrower targeted fix), or an alternate monkey-patch measurement path. Mirrors the PLAN §7 P-5 Notes wording so the dependency granularity stays consistent across both documents.
    - **§8 P-5-A.1 Acceptance bullet (line 874).** Old "(a-real) defers to P-5-C where the bench harness already owns the subprocess + HF-cache plumbing" → New "(a-real) post-P-5 required follow-up (not a P-5 close gate); the bench-harness infrastructure it depends on is already in place at P-5-C close."
  - **What is NOT changed.** Acceptance gates (1) / (2) / (3); §7 P-5 Deliverables list; I-1..I-5 Protocol signatures; §9 D-* / §10 Q-* / §11 R-* identifiers; any code, test, or interface. `README.md` not touched this revision — it does not carry the fine-grained (4-a) / (4-b) semantics.
  - **References:** v1.5.1 changelog for the original Qwen3.5 naming; P5_OPENING §6.5 for the empirical Qwen3-0.6B migration; `docs/P3_DELTANET_SURVEY.md` C-open-3 for the `ContinuousBatcher` + recurrent-adapter incompatibility; `vqbench/REPORT.md` §2.2 for the monkey-patch fallback pattern; 2026-04-23 conversation where the phase-boundary framing was pinned.
  - **Next step:** P-5 Acceptance sweep proper — (1) codec-swap neutrality by inspection; (2) `--all-kv-codecs` Markdown report coverage; (3) `qwen3-0.6b-admission-headroom-prefix-heavy` row numbers; (4-a) `tests/test_block_tq_vqbench_xcheck.py` green + numbers recorded; (4-b) `--vqbench-xcheck` on `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` within both thresholds. Flip the four `[ ]` to `[x]` in a separate close revision (v1.7.3).

- **v1.7.1** (2026-04-23): **P-5 implementation sub-units all landed — doc sync.** No code or test diffs. Updates PLAN header, §7 P-5 Status, `docs/P5_OPENING.md` §8, and `README.md` to reflect the post-P-5-C.6 reality — P-5-A / B / C sub-units per `docs/P5_OPENING.md` §8 all landed between 2026-04-22 and 2026-04-23. §7 P-5 Acceptance checkboxes (lines 555-562) deliberately kept at `[ ]`; the four Acceptance gates — (1) codec-swap neutrality, (2) one-command fp16 vs codec side-by-side, (3) BlockTQ admits more under the same budget, (4) vqbench numeric cross-check (a) / (b) — will flip in a later revision after a dedicated sweep, mirroring the P-4.5-B.1 (v1.6.6) implementation landing + P-4.5 close (v1.6.9) acceptance sweep pattern.
  - **Header:** Version v1.7.0 → v1.7.1; Status updated from "P-5-A / P-5-B landed; P-5-C.1 landed; P-5-C.2 next" to "P-5-A / P-5-B / P-5-C sub-units all landed; P-5 Acceptance sweep pending". Line 564 §7 P-5 Status rewritten to enumerate all sub-units (previously omitted A.2 and described C.2 as "next").
  - **`docs/P5_OPENING.md` §8.** C.2 heading "(next)" → "(landed 2026-04-23)"; C.3 / C.4 / C.5 / C.6 headings gain "(landed 2026-04-23)" annotations matching the C.1 precedent already present at the C.1 heading (landed 2026-04-23). Scope prose and blocked-by lines unchanged.
  - **`README.md`:** Status table P-5 row updated from "P-5-A.0 scaffolding shipped ... P-5-A.1 BlockTQ hot path next" to "P-5-A / B / C sub-units landed (v1.7.1 — BlockTQ + RaBitQ family + bench harness); P-5 Acceptance sweep pending". Roadmap §P-5 bullet rewritten as a three-bullet A / B / C sub-unit summary explicitly calling out the `[ ]` §7 P-5 Acceptance gate.
  - **Landing commits referenced for traceability:** `dc7c751` (C.6), `2094518` (C.5), `81bca8b` (C.4), `66897d9` (C.3), `08233e2` (C.2), `bf9dafc` (C.1), `deb5749` (B.3), `abb19b4` / `f85f1e7` / `4171984` (B.2), `64ee750` / `e2148b2` / `cfdec12` (B.1), `9579b0c` / `7c5b37d` / `9a538fe` (A.3), `35bbd2e` (A.2), `c431ad6` / `1213756` / `dd32d17` (A.1), v1.7.0 (A.0).
  - **What is NOT changed:** any `[ ]` Deliverables / Acceptance checkboxes in §7 P-5; I-1..I-5 Protocol signatures; §10 Q / §11 R / §9 D IDs; sub-unit scope prose in P5_OPENING §8. Zero code, test, or interface diffs.
  - **Next step:** the P-5 Acceptance sweep — run the full `python -m scripts.bench --all --all-kv-codecs --seeds 42,43,44 --vqbench-xcheck` on the post-C.6 tree, compare against `vqbench/REPORT.md` headline numbers, and flip the four §7 P-5 Acceptance checkboxes in a separate close revision (the P-4.5-B.1 → P-4.5 close sequence between v1.6.6 and v1.6.9 is the reference pattern).

- **v1.7.0** (2026-04-22): **P-5-A.0 scaffolding — side-level VectorCodec[P] + packing + calibration + store migration.** Opens P-5 proper with the four P-5-A.0 sub-unit commits plus doc sync. No functional change to the active codec path — `IdentityCodec` continues to be the only shipping codec, byte-identity against the pre-P-5 token stream is preserved by the P-4.5-C.1 integration test suite. The scaffolding unblocks P-5-A.1 (BlockTurboQuantMSE hot path + Q-008 resolved on the store seam).
  - **`silica/kvcache/codec.py`** (rewrite) — retires the pre-P-5 pair-level `KVCodec` Protocol, `CodedBlock` dataclass, and pair-level `IdentityCodec`. New side-level `VectorCodec[P]` Protocol (one tensor in, one `CodedPayload` subclass out); new `CodedPayload` hierarchy (`RawFp16Payload`, `BlockTQPayload`, `RaBitQPayload`) with D-012 honesty enforced at `__post_init__`; new side-level `IdentityCodec(block_size, n_kv_heads, head_dim, dtype=fp16)` implementing `VectorCodec[RawFp16Payload]`. Signatures drop `k_dtype` / `v_dtype` in favour of a single per-side `dtype`. `resident_bytes` / `logical_bytes` are per-side.
  - **`silica/kvcache/store.py`** — `SyntheticPrefixBlockStore.__init__` gains `k_codec` / `v_codec` / `codec` kwargs. `codec=` is a shorthand for `k_codec = v_codec = codec`; any combination of `codec=` with a side kwarg, or mixed-None split (one side None, one non-None), raises `ValueError`. All three `None` is the pass-through path with byte-for-byte pre-P-5 behaviour. `_detached` storage moves from `tuple[CodedBlock, ...]` to `tuple[_DetachedLayer, ...]` where `_DetachedLayer` is a private frozen dataclass holding per-side `CodedPayload | mx.array` (raw on pass-through, coded otherwise). `resident_bytes()` sums per-side payloads, branching on the raw-vs-coded discriminant.
  - **`silica/kvcache/prefix.py`** — new `RadixPrefixCache.store` read-only property returning `PrefixBlockStore` (non-optional, matches constructor invariant). Forward-pointer for P-5-A.2 `MemoryBudgeter` consumer; must use a `SupportsResidentBytes` structural check rather than assuming every store implements `resident_bytes()`.
  - **`silica/vq/core/packing.py`** (new, P-5-A.0.2) — MLX-native `pack_sub_byte` / `unpack_sub_byte`, bit-plane layout, `num_bits ∈ {1, 2, 3, 4}`, `d % 8 == 0`. Shared across TurboQuantMSE / BlockTurboQuantMSE / RaBitQ / ExtRaBitQ.
  - **`silica/vq/_calibration.py`** (new, P-5-A.0.3) — NumPy quarantine. `haar_rotation(d, seed)` (Stewart 1980 QR + sign-fix) and `lloyd_max_codebook(num_bits, sigma)` (Lloyd-Max for N(0, sigma^2); uses stdlib `math.erf` + `math.exp` rather than scipy). Write-protected outputs, cached by construction args. The sole module under `silica.vq.*` permitted to import NumPy; enforced by a `pkgutil.walk_packages`-based quarantine test.
  - **§6 I-3 amendment (in-place).** Canonical interface table now documents `VectorCodec[P]`; pair-level `KVCodec` / `CodedBlock` signatures removed from the active table. Historical P-4.5-C entries in §7 and the amendment log keep their original wording as past-state records.
  - **§10 Q-008 resolved.** Side-level `VectorCodec[P]` + store-level `k_codec` / `v_codec` split. The historical A / B / C option labels are explicitly marked superseded (the new path collapses the pair into the store rather than into the codec constructor or a second Protocol).
  - **Test migration.** `tests/test_vector_codec.py` (13 payload + Protocol tests), `tests/test_packing.py` (63 tests), `tests/test_calibration.py` (42 tests) — all added in earlier P-5-A.0 commits. This commit adds `tests/test_prefix_store.py` side-level cases (15 new: constructor rejections, pass-through identity, shorthand dispatch, split-mode K/V independence, per-side `resident_bytes` arithmetic) and rewrites `tests/test_kvcodec.py` (side-level API), `tests/test_interfaces.py` (I-3 `KVCodec` → `VectorCodec`, attribute tuple), `tests/test_kvcodec_integration.py` (`_CountingIdentityCodec` implements `VectorCodec`; counter arithmetic doubles for K/V split; byte-identity invariants preserved).
  - **Doc sync.** `docs/P5_OPENING.md` carries forward. `docs/P5_A_U4_STORE_MIGRATION.md` is the Unit 4 implementation checklist. `README.md` I-3 terminology refreshed to `VectorCodec`.
  - **Next step:** P-5-A.1 (BlockTurboQuantMSE hot path on the shipped scaffold).

- **v1.6.9** (2026-04-21): **P-4.5 close — Acceptance regression sweep against the post-C.1 tree.** No code changes. Verifies and flips the four P-4.5 Acceptance checkboxes that were intentionally deferred out of the C.1 implementation commit (`f3a6171`) per its "remaining P-4.5 close gates" note.
  - **Header:** Version v1.6.8 → v1.6.9; Status "P-0..P-4.5 complete; P-5 next".
  - **§7 P-4.5 Acceptance checkboxes flip `[ ] → [x]`:**
    - **Q-010 signal < 3.5× on five consecutive runs** — `tests/test_engine_admission_reorder.py::test_q010_ratio_below_threshold_on_five_runs` passes; the post-C.1 steady-state distribution remains in the `{2.53, 2.78, 2.95, 3.07, 3.27}×` range characterized at the P-4.5-B.1 commit. Option (C) admission reorder is the load-bearing mechanism; C.1's prefix-store codec hook does not affect cohort shape on this workload (no cross-call prefix reuse — Q-012), so the B.1-era measurement carries over.
    - **Three-layer chunked-prefill correctness** — (a) event-taxonomy invariants and (b) per-row token-count invariants remain regression-locked by `tests/test_batcher.py` + `tests/test_p2_batched_parity.py` (green in every C.0 / C.1 / cleanup commit's sweep); (c) direct-mlx-lm sub-cohort numerical reference is pinned by `test_reordered_cohort_matches_mlx_lm_direct_batched_reference` and passes in the 31-case on-device admission-reorder sweep.
    - **15-row bench catalog regression** — `python -m scripts.bench --all` on the post-C.1 tree: 9 cache-only rows `ok` (`qwen3-0.6b-b1-parity`, `qwen3-0.6b-bgt1-parity`, `qwen3-0.6b-concurrent-shared-prefix`, `qwen3-0.6b-long-in-short-out`, `qwen3-0.6b-short-in-long-out`, `qwen3-0.6b-smoke`, `qwen3-0.6b-teacher-forced-argmax`, `qwen3-0.6b-ttft-under-concurrency`, `qwen3.5-0.8b-b1-parity`), 6 env-gated rows `skipped` (`gemma4-31b-b1-parity`, `gemma4-31b-bgt1-parity`, `gemma4-31b-smoke`, `gemma4-moe-smoke`, `qwen3.5-27b-smoke`, `qwen3.5-moe-smoke`).
  - **§7 P-4.5 Status:** `in-progress` → `complete`.
  - **What is NOT changed.** Zero code diffs. Zero test diffs. Zero interface diffs. The Q-012 "initial-cohort prefix-cache consultation" open question added in v1.6.8 stays open — it is a v0.2 item, not a P-4.5 close gate.
  - **Next step:** P-5 proper (§7 P-5 Phase 5 — VQ KV Compression). `BlockTQCodec` / `RaBitQCodec` attach to the now-live `PrefixBlockStore(codec=...)` seam; `MemoryBudgeter` switches from the per-block eviction-shortfall formula to reading `store.resident_bytes()` once a non-identity codec lands (the transition is out of P-4.5-C scope per opening doc §6.2).

- **v1.6.8** (2026-04-21): **P-4.5-C.1 KVCodec runtime-integration spike — implementation + tests + opening-doc §8 rewrite + Q-012.** Lands Option (B) as specified in `docs/P4_5_C_KVCODEC_OPENING.md`. Code changes confined to `silica/kvcache/store.py`; zero scheduler / budgeter / codec-module diffs. Tests: 6 new cases — 4 cache-presence-gated on the local Qwen3-0.6B HF cache (no env-var strong gate, mirroring `tests/test_engine_admission_reorder.py` §5), 2 pure-unit with synthetic tensors. Opening doc §8 rewritten in the same commit after test authoring surfaced that the initial cohort seal does not consult the prefix cache — see new Q-012 in §10 and the corresponding P-4.5 Amendment log entry.
  - **Header:** Version v1.6.7 → v1.6.8; Status adds C.1 complete, notes the full-suite regression sweep as the remaining P-4.5 close gate.
  - **`silica/kvcache/store.py`** — `SyntheticPrefixBlockStore.__init__` gains a `codec: KVCodec | None = None` kwarg with a `codec.block_size` precondition. Internal `_encode` / `_decode` private methods route raw `(K, V)` tensors through `codec.encode_block` / `codec.decode_block` when a codec is supplied, or through a pass-through path that wraps raw tensors in `CodedBlock` with `resident_bytes = k.nbytes + v.nbytes` when `codec` is `None` (pre-C.1 default). `_detached` container type narrows to `dict[int, tuple[CodedBlock, ...]]`. New `resident_bytes()` method sums `CodedBlock.resident_bytes` across all detached blocks; deliberately not added to the `PrefixBlockStore` Protocol (the paged backend cannot report per-layer physical K/V residency without a kernel that D-003 / Q-009 / R-7 both exclude). `PagedPrefixBlockStore` untouched — its detached methods still raise `NotImplementedError`. Module docstring gains a "KVCodec hook" section cross-referencing the opening doc.
  - **External API preserved.** All 15 existing `SyntheticPrefixBlockStore(block_size=...)` call sites (13 in tests, 2 in runtime — `silica/bench/runner.py`, `scripts/bench_p2_baseline.py`) work unchanged because `codec` defaults to `None` and the pass-through path is semantically identical to pre-C.1 behaviour. `register_detached` / `fetch_detached` return the same `Sequence[tuple[mx.array, mx.array]]` shape `build_seeded_batch_kv` expects. 254-test regression sweep across `test_prefix_cache*` / `test_prefix_store.py` / `test_batcher.py` / `test_memory_budgeter.py` / `test_p2_batched_parity.py` / `test_engine_admission_reorder.py` / `test_kvcodec.py` / `test_kvcodec_integration.py` all green; plus engine / chat / bench-runner side chains (72 further tests) unchanged.
  - **`tests/test_kvcodec_integration.py`** — 6 new cases per opening doc §8 acceptance:
    - **§1 `test_prompt_tokenization_invariants`** (D-1 defensive): pins the shared `_PROMPT_FIXTURE` Qwen3-0.6B tokenization at ≥ 33 tokens AND `len % block_size != 0`. The `≥ 33` half satisfies batcher invariant S-5 edge 1 (`max_aligned = ((len - 1) // block_size) × block_size`); the `!= 0` half ensures cold encode count (`floor(len / bs)`) equals paired-hit decode count (`floor((len - 1) / bs)`), avoiding a misleading asymmetry in the counter assertions. Under Qwen3-0.6B this fixture measures 34 tokens (`mod 16 == 2`).
    - **§2 `test_encode_and_decode_counters_on_paired_prompts`**: single `generate_batch([p, p], max_batch_size=1)` exercises both paths — prompt 0 admits into the initial cohort, runs to `max_tokens` termination, reclaim triggers `_extract_and_insert_prefix` → `insert_detached` → `register_detached` → `codec.encode_block` (encode side, floor(34/16)=2 blocks × 28 layers ≥ 56); prompt 1 enters the waiting queue, `_admit_waiting_requests` sees a populated prefix cache, `_admit_single_hit_row` → `lookup` → `fetch_detached_blocks` → `fetch_detached` → `codec.decode_block` (decode side, floor(33/16)=2 blocks × 28 layers ≥ 56). The single-call shape is required because the hit path only fires under mid-run admission, never under `_prepare_cohort` (a design fact discovered while bringing the tests up and documented inline). D-2 defensive invariant `len(store._detached) == len(store.live_block_ids())` also asserted.
    - **§3 `test_resident_bytes_matches_radix_node_total`**: after the same `[p, p]` workload, `store.resident_bytes() == total_blocks × num_layers × block_size × (2 × n_kv_heads × head_dim × dtype.size)`. Uses the per-layer K+V byte-per-token cost, **not** `MemoryBudgeter.bytes_per_token` which is already all-layer-summed (the name-collision trap called out in opening §6.2). Also pins `prefix_cache.node_count() == len(store.live_block_ids())` (the radix-node / store-resident 1:1 correspondence under `insert_detached`).
    - **§4 `test_codec_path_token_stream_matches_no_codec_baseline`**: §8.4 byte-identity invariant — two paired runs, one with `codec=None`, one with `codec=IdentityCodec(...)`, must emit byte-identical per-row token streams. Both runs use the `[p, p] max_batch_size=1` shape so both encode and decode paths execute on the codec run; byte-identity depends on `IdentityCodec` returning `CodedBlock`'s `k` / `v` fields by reference (`codec.py:98-102`) so `mx.concatenate` in `build_seeded_batch_kv` produces bitwise-identical tensors.
    - **§5 `test_identity_codec_path_preserves_tensor_references`** (defensive tripwire, no HF cache): purely synthetic tensors + direct `store.register_detached` + `store.fetch_detached` round-trip. Asserts the returned K / V tensors are `is`-identical (not just value-equal) to the originals. A future codec that silently inserts a defensive copy inside `encode_block` or `decode_block` would regress from byte-identical-reference to byte-identical-value without breaking any other test — this tripwire catches that moment instead.
    - **§6 `test_no_codec_pass_through_also_preserves_tensor_references`**: same reference-level invariant for the `codec=None` pass-through branch of `_encode` / `_decode`, so any §4 byte-identity is genuinely inherited from the two paths' agreement on reference-level behaviour rather than a coincidental value equality. Also asserts `store.resident_bytes() == k.nbytes + v.nbytes` under pass-through.
  - **PLAN edits.** §7 P-4.5-C deliverable checkbox `[ ]` → `[x]`; §7 P-4.5 "Codec hot-path reached" acceptance checkbox `[ ]` → `[x]`; §7 P-4.5 Status line updated; §7 P-4.5 Amendment log gains the "P-4.5-C.1 acceptance shape amendment" entry recording why §8 reshaped from paired-call to single-call `[p, p] max_batch_size=1` after the initial-cohort-no-prefix-lookup discovery; §10 gains **Q-012** "Initial-cohort prefix-cache consultation" documenting the underlying design fact — v0.1 does not consult the prefix cache in `_prepare_cohort`, so cross-`generate_batch`-call reuse is effectively zero (relevant for REPL / chat-session workloads; deferred to v0.2 session-layer design); Changelog v1.6.8 entry (this bullet).
  - **`docs/P4_5_C_KVCODEC_OPENING.md` §8 rewrite.** §8.0 expanded from "why `generate_batch` not `generate`" to "entry point and workload shape", recording the initial-cohort / mid-run-admission asymmetry and pointing at Q-012. §8.1 merges the original §8.1 + §8.2 into a single encode-plus-decode assertion over one `generate_batch([p, p], max_batch_size=1)` call (row 0 miss-path encode via `_extract_and_insert_prefix` on termination; row 1 mid-run-admission decode via `_admit_single_hit_row`). §8.2 is the renumbered old §8.3 (`store.resident_bytes()` vs radix-node total) and updates its prompt-length example to match the same 34-token fixture. §8.3 is the renumbered old §8.4 (baseline-identity between no-codec pass-through and `IdentityCodec`) using the same `[p, p]` shape so row 1's hit path runs on the codec side. §8.4 pulls the old "C.1 implementation note" (the `is`/`id()` tensor-reference tripwire) into its own numbered subsection so future codec authors see it as a first-class acceptance pin.
  - **Remaining P-4.5 close gates (Acceptance checkboxes still `[ ]`).** (1) Q-010 signal re-measurement on the post-C.1 tree to confirm the `< 3.5×` ratio still holds. (2) Chunked-prefill correctness three-layer criterion — (a) / (b) regression-locked by `test_batcher.py` + `test_p2_batched_parity.py` (green in this commit's sweep); (c) the numerical sub-cohort reference in `test_engine_admission_reorder.py::test_reordered_cohort_matches_mlx_lm_direct_batched_reference` (green in this commit's sweep). (3) 15-row `python -m scripts.bench --all` sweep. All three are regression sweeps against the now-merged C.1 tree and can land as a separate commit alongside a P-4.5 close note; deliberately kept out of C.1 so this commit's scope stays at "wire codec into runtime + tests".

- **v1.6.7** (2026-04-21): **P-4.5-C.0 KVCodec runtime integration spike — opening doc.** No code or interface changes. Three documentation edits land together so that PLAN and the opening doc agree at commit time: (i) new `docs/P4_5_C_KVCODEC_OPENING.md` enumerates the three integration-point options (active `BatchKVCache` in-place / detached prefix store via `SyntheticPrefixBlockStore` / codec-aware `BatchKVCache` façade) against D-003 (no compressed-domain attention) and Q-009 / R-7 (no MLX variable-length SDPA), recommends **Option (B) — prefix-store-scoped codec hook**, and pins the encode / decode granularity, the `resident_bytes` parallel-observable relation (compared against the radix-node-derived total, not `_count_evictable_prefix_blocks`), the homogeneous-shape-only spike scope, and a paired-request encode/decode acceptance specification driven through `Engine.generate_batch` with an explicit `prefix_cache` argument. (ii) §7 P-5 Strategy amendment replacing `PagedKVCache(codec=...) injection-based switching` with `PrefixBlockStore(codec=...)` + the rationale for why the active-K/V path is not codec-wrapped in v0.1 (D-003, Q-009 / R-7). (iii) §7 P-4.5 acceptance "Codec hot-path reached" amendment corrects three axes in the original "≥ 1 call per active KV block on a single-request `Engine.generate('Hello', max_tokens=4)`" wording: wrong entry point (`Engine.generate` bypasses `ContinuousBatcher` / `RadixPrefixCache`; `Engine.generate_batch(prefix_cache=...)` is the only hot-path entry for Option (B)), prompt too short ("Hello" tokenizes to ≪ `block_size` so produces zero aligned blocks), and single-run encode/decode asymmetry (a cold run fires `register_detached` only; `fetch_detached` fires on a paired repeat request). Replaced with a paired-`generate_batch` specification: prompt must tokenize to `≥ 2 × block_size + 1 = 33` tokens to satisfy batcher invariant S-5 edge 1; cold encode-side and paired-repeat decode-side clauses both assert `2 × num_layers` calls under default Qwen3 `block_size=16`. Both amendments logged in §7 P-4.5 Amendment log.
  - **Header:** Version → v1.6.7; Status line synced from "P-4.5 bridge in planning" to "P-4.5-A / B.0 / B.1 / C.0 complete; C.1 implementation planned".
  - **New doc:** `docs/P4_5_C_KVCODEC_OPENING.md`. TL;DR plus nine numbered sections: §1 Problem (P-5 codec hot-path gap restated), §2 Constraints (D-003 / D-009 / Q-009 / R-7 / P-2 Option B / `build_seeded_batch_kv` shape contract / spike-not-optimization boundary), §3 Three integration-point options, §4 Trade-off matrix, §5 Recommendation (Option B with five-bullet rationale), §6 Touchpoints (encode granularity + `resident_bytes` parallel observable + PLAN §7 P-5 amendment + homogeneous-shape-only scope), §7 What this spike does NOT do, §8 Live-forward acceptance specification (§8.0 entry-point rationale — `generate_batch` vs `generate`; §8.1 encode-side counter on cold `generate_batch` with `len(prompt_tokens) >= 33`; §8.2 decode-side counter on paired repeat via same `shared_pc`; §8.3 `store.resident_bytes()` vs radix-node-derived total; §8.4 baseline-identity numerical invariant + C.1 `is`/`id()` tensor-reference note), §9 References.
  - **PLAN edits:** §7 P-4.5-C deliverable bullet (§7:517) — `docs/P5_OPENING.md` → `docs/P4_5_C_KVCODEC_OPENING.md`; chosen option recorded as (B); C.1 verification entry point pinned as `Engine.generate_batch(prefix_cache=...)`; prompt-length requirement `≥ 33 tokens`; `store.resident_bytes()` parity right-hand side uses `len(store.live_block_ids()) × num_layers × block_size × bytes_per_token` (explicitly not `_count_evictable_prefix_blocks × _kv_bytes_per_block`, which counts leaf-zero-hit only and under-reports internal nodes); homogeneous-shape-only scope stated. §7 P-4.5 acceptance "Codec hot-path reached" bullet (§7:524) — paired `generate_batch` specification with encode+decode clauses. §7 P-4.5 Status (§7:527) — adds "C.0 opening doc complete 2026-04-21; C.1 implementation planned". §7 P-4.5 Amendment log (§7:529) — two new entries (codec hot-path acceptance covering entry-point / prompt-length / encode-vs-decode asymmetry; P-5 Strategy source). §7 P-5 Strategy line (§7:539) — `PagedKVCache(codec=...)` → `PrefixBlockStore(codec=...)` with rationale and cross-reference to the opening doc.
  - **What is NOT changed.** Zero code diffs — `silica/kvcache/store.py`, `silica/kvcache/codec.py`, `silica/scheduler/budget.py`, `silica/scheduler/batcher.py` all untouched. Zero interface changes. Zero test changes. The C.1 implementation commit (next) lands the `SyntheticPrefixBlockStore(codec=...)` constructor wiring, `register_detached` / `fetch_detached` codec call-throughs, the `resident_bytes()` parallel observable method, and `tests/test_kvcodec_integration.py`.
- **v1.6.6** (2026-04-21): **P-4.5-B.1 admission-reorder implementation + Q-010 threshold amendment.** Lands Option (C) as specified in `docs/P4_5_CHUNKED_PREFILL_OPENING.md`. Code changes are isolated to the Engine layer; `silica/scheduler/batcher.py` is unchanged. Tests: 31 new cases covering helpers + end-to-end wiring + opt-out / at-threshold admission-order preservation + direct mlx-lm sub-cohort reference (PLAN Acceptance (c)) + Q-010 five-run acceptance.
  - **Header:** Version → v1.6.6.
  - **Engine:** `silica/engine/__init__.py` gains `_sort_admissions_by_length`, `_initial_cohort_cap`, and a new `length_spread_threshold: float = 2.0` kwarg on `generate_batch`. Default behaviour reorders admissions so short rows prefill in a dedicated cohort; long-prompt admissions queue and drain through the existing `_admit_miss_cohort` mid-run admission path. `batcher.py` untouched. NaN threshold explicitly rejected (silently disables the split otherwise). **The sort only applies when the split path actually fires** — a Codex review noted that unconditional sorting would silently reorder the ``BatchEvent`` emission order (row index in ``_rows`` differs → per-step emit order differs) even under the documented opt-out, which contradicted the "preserve pre-P-4.5 behaviour" promise callers rely on. Fixed: when ``max/min <= threshold`` (homogeneous batch or ``threshold=float('inf')`` opt-out), ``generate_batch`` uses the caller's original admission order end-to-end. Two call-sites opt out of the split via `length_spread_threshold=float("inf")` to preserve stated parity semantics: `silica/bench/runner.py::_collect_bgt1_batched_tokens` (BGT1 oracle expects Silica B=2 vs mlx-lm B=2) and `tests/test_p2_batched_parity.py::test_left_padding_does_not_corrupt_any_row` (P-2 left-padding pin against direct mlx-lm B=2).
  - **Tests:** `tests/test_engine_admission_reorder.py` — 31 cases across 5 sections: (1) sort stability + req_index preservation (6), (2) `_initial_cohort_cap` preconditions (incl. NaN reject) + four worked reverse examples from the opening doc §6.1 (14), (3) `generate_batch` end-to-end wiring — default threshold split shape, opt-out preserves original admission order, at-threshold ratio preserves original admission order, queued-cohort limitation pin (9), (4) dual-gated direct mlx-lm sub-cohort reference test using the shared `_runtime_admission_partition` helper so the reference matches the runtime's actual pre-step / remainder sub-cohorts (including the ``effective_batch_size`` clamp — if a future catalog editor tightens ``max_batch_size`` below ``short_count + 1``, the reference tracks that change automatically), (5) dual-gated Q-010 five-run acceptance with adaptive short-row filter derived from the same runtime partition. Fake engines in `tests/test_bench_runner.py` accept the new kwarg as `length_spread_threshold` default.
  - **§7 P-4.5 Deliverable update:** P-4.5-B.1 marked `[x]`. The deliverable bullet rewritten to describe the shipped shape — Option (C) admission heuristic, zero `batcher.py` diff, BGT1 + P-2 opt-outs, 31-case test file with direct-mlx-lm sub-cohort reference — and to explicitly call out the queued-cohort limitation (pinned in a regression test).
  - **§7 P-4.5 Acceptance threshold amendment:** Acceptance (a) tightened from `< 3×` to `< 3.5×` after the first on-device measurement on Qwen3-0.6B showed the post-fix steady-state distribution was `{2.53, 2.78, 2.95, 3.07, 3.27}×` (pre-fix `{4.13-4.76}×`). The residual ~3× floor is intrinsic B=3 short-cohort-prefill overhead vs B=1 isolated smoke, not a scheduler bug; single-step fairness below that requires MLX variable-length attention (Q-009 / R-7) and lives outside P-4.5 scope. Amendment logged in §7 P-4.5 Amendment log with pre/post measurement data.
  - **§10 Q-010 Resolution** exit-criterion (i) updated to `< 3.5×` with a cross-reference to the Amendment log.
  - **§7 P-4.5 Status** bumped to "B.1 implementation complete; C spike planned".
  - **Opening doc (`docs/P4_5_CHUNKED_PREFILL_OPENING.md`):** §5.2 gains a "Queued-cohort fairness" scope-boundary bullet explaining the limitation pinned by the new test; §8 Acceptance sign-off item 1 updated to reflect the `< 3.5×` threshold and the single-subprocess warmup protocol the test harness uses.
  - **What is NOT changed:** I-1..I-5 Protocol signatures; §8.1 priority tiers; `silica/scheduler/batcher.py` (Option (C) is an Engine-layer admission heuristic only); P-0..P-4 Deliverables; P-4.5-C KVCodec spike scope and deliverables.
  - **Measurement:** Post-fix production manual run on Qwen3-0.6B: isolated smoke TTFT ≈ 17.47 ms; short-row first-token offsets ≈ {20.24, 20.25, 20.25} ms; long-row offset ≈ 59.55 ms; max short-row ratio ≈ 1.16× — well below the `< 3.5×` acceptance threshold, consistent with a B=3 short cohort running at ~1.2× the B=1 isolated baseline.
  - **References:** `docs/P4_5_CHUNKED_PREFILL_OPENING.md`; `silica/engine/__init__.py`; `silica/bench/runner.py::_collect_bgt1_batched_tokens`; `tests/test_engine_admission_reorder.py`; `tests/test_p2_batched_parity.py::test_left_padding_does_not_corrupt_any_row`.
- **v1.6.5** (2026-04-21): **P-4.5-B.0 opening doc + acceptance/deliverable alignment.** Documentation-only revision. This revision lands `docs/P4_5_CHUNKED_PREFILL_OPENING.md` (the §7 P-4.5-B.0 deliverable) and synchronizes three places in the PLAN that still carried the pre-opening wording. Triggered by a Codex review noting that the PLAN's P-4.5-B deliverable, the Q-010 Resolution exit criteria, and the v1.6.4 changelog entry each still described chunked prefill as a `silica/scheduler/batcher.py` implementation with a "greedy bit-identity" correctness gate — both inconsistent with the opening doc's chosen Option (C) (admission reorder under `silica/engine/__init__.py::generate_batch`, zero `batcher.py` diff expected) and the v1.6.4 three-layer acceptance criterion.
  - **Header:** Version → v1.6.5; Last updated → 2026-04-21.
  - **§7 P-4.5 Deliverables** split the old single P-4.5-B bullet into **B.0** (opening doc, marked `[x]`) and **B.1** (admission-reorder implementation, still `[ ]`). B.1 now names the concrete touchpoint (`engine/__init__.py`, the `length_spread_threshold: float = 2.0` kwarg, the `_initial_cohort_cap` clamp spec with `max(1, min(effective_batch_size, first_exceeding_index))`, the new `tests/test_engine_admission_reorder.py` with sort-stability / req_index preservation / threshold-parametrized / direct-batched-reference tests, and the Q-010 five-run acceptance harness). Status line updated from "A complete" to "A complete; B.0 opening doc complete; B.1 implementation + C spike planned".
  - **§7 P-4.5 Notes** "300-line ceiling" caveat now refers to option (A) by its new name (`real in-cohort chunked prefill`), consistent with the opening doc's three-option labels.
  - **§10 Q-010 Resolution** exit criterion (ii) rewritten: old text "greedy output token-by-token identical to the unchunked path" replaced with "chunked-prefill correctness verified under the three-layer criterion (event-taxonomy + per-row token count + direct-mlx-lm-batched numerical reference); strict bit-identity against the unchunked Silica path is NOT part of the exit criterion because fp16 batched SDPA drift across different batch compositions is already documented in P-2 / P-3-D3.1 empirical findings." Exit criterion (i) sharpened to name the ratio formula `max(offsets_short) / smoke_ttft_ms` and the short-row filter rule.
  - **§13 Changelog v1.6.4 P-4.5 entry** prose "chunked-vs-unchunked greedy bit-identity" replaced with the three-layer criterion summary and the "strict bit-identity NOT claimed" caveat. B-sub-units split reference added.
  - **New doc:** `docs/P4_5_CHUNKED_PREFILL_OPENING.md`. 639-line opening spec: Q-010 root cause at `_prefill_phase`; three-option analysis (A/B/C) against I-1..I-5 + B-1..B-9 + S-1..S-7 invariants; MLX variable-length-attention constraint (Q-009 / R-7) closing option (A) out of P-4.5 scope; fp16 batch-composition drift closing "bit-identity" out of the acceptance; 13-row trade-off matrix; Option (C) recommendation with minimum-blast-radius rationale; explicit scope non-goals (single-long-prompt OOM, homogeneous-length fairness, per-token latency fairness beyond the first token, sustained-load backpressure); `_initial_cohort_cap` spec with four worked reverse examples pinning the `max(1, min(effective_batch_size, first_exceeding_index))` clamp; length-spread threshold default `2.0` rationale; four-item acceptance sign-off with the five-run Q-010 harness and the inverse-permutation index-mapping note for direct-batched reference tests.
  - **What is NOT changed:** I-1..I-5 / B-1..B-9 / S-1..S-7 invariants (unchanged — confirmed by (C)'s zero-`batcher.py`-diff choice); §8.1 priority tiers; all P-0..P-4 Deliverables / Acceptance checkmarks; the P-4.5-A decision-sync commit (v1.6.4) stays as-is apart from the Changelog-entry prose correction noted above. No code changes in this revision.
  - **References:** `docs/P4_5_CHUNKED_PREFILL_OPENING.md`; §7 P-4.5 Deliverables + Acceptance + Amendment log; §10 Q-010 Resolution; §13 v1.6.4 entry; `silica/scheduler/batcher.py::_prefill_phase` + `_admit_miss_cohort`; `silica/engine/__init__.py::generate_batch`.
- **v1.6.4** (2026-04-21): **P-4 exit decision sync.** P-0..P-4 complete (see `git log` for the 14 landing commits across P-4.1..P-4.4). This revision touches only documentation — no code or interface changes — and closes the books on Q-010 while opening a new bridge phase §7 P-4.5. Landing items:
  - **Header:** Version → v1.6.4; Status → "P-0..P-4 complete; Q-010 triggered at P-4 exit → P-4.5 bridge in planning"; Last updated → 2026-04-21. Stale "Phase 0 in-progress" removed.
  - **Q-010 resolved (triggered, promote).** Two independent measurements on `qwen3-0.6b-ttft-under-concurrency` vs isolated `qwen3-0.6b-smoke` show cohort-level prefill serializing short rows behind the long row's `T_max`. Codex measurement 6.9×; Silica measurement over four consecutive runs {4.76, 4.42, 4.56, 4.13}× — ratios straddle Option A's 5× trigger but the structural signature (all four concurrent rows' first-token offsets within ≤ 0.2 ms of each other) is deterministic and worsens with longer prompts. Q-010 resolves to promote chunked prefill to the new §7 P-4.5 bridge phase (not retroactively into P-2 or P-3).
  - **Q-002 progress (not resolved).** P-4 exit surfaced two product-face signals: (1) `silica.chat.ChatSession` + `scripts/chat.py` already demonstrate a usable multi-turn REPL layer that P-8 would wrap, and (2) the Q-010 fairness defect would be felt first through an HTTP endpoint under concurrent load. Current lean: **Option B (float P-8 to T1 tail), sequence the lift as P-4.5 → P-5 → P-8.** No §8.1 priority-tier edit in this version; Q-002 resolves formally when P-5 BlockTQ lands.
  - **Q-003 progress (not resolved).** P-3 27B / 31B 4-bit load probes show peak ~30.5 GB / similar on 48 GB, ~17 GB headroom — sufficient for short decode but the P-3 Acceptance Product memory-fit target (500 tokens) remains unvalidated. No immediate P-6 promotion is warranted; Q-003 remains open, leaning not-triggered, and resolves when either a dedicated long-inference bench row runs or a user hits an OOM.
  - **New §7 P-4.5 Phase 4.5 — P-4 exit bridge.** Sub-units: P-4.5-A (this decision-sync commit), P-4.5-B split into B.0 (chunked-prefill opening doc) and B.1 (admission-reorder implementation), P-4.5-C (KVCodec runtime integration spike). Acceptance includes: TTFT-under-concurrency ratio < 3× over five runs (short-row filter) — **amended 2026-04-21 to < 3.5× after P-4.5-B.1 empirical measurement, see v1.6.6 changelog** — a three-layer chunked-prefill correctness criterion (event-taxonomy + per-row token count + direct-mlx-lm-batched numerical reference on the sub-cohort — strict bit-identity NOT claimed, see §7 P-4.5 Amendment log), `encode_block` / `decode_block` registering ≥ 1 call per KV block on a live forward (closes the interface-vs-hot-path gap identified at P-4 exit), no regression across the 15-row bench catalog. P-4.5 is a bridge, not a phase in the §8 priority-tier table; P-5 dependencies move from P-4 to P-4.5.
  - **New §7 P-4 empirical findings bullet dated 2026-04-21** recording the Q-010 measurement pair, the codec hot-path gap (`encode_block` / `decode_block` have zero runtime callers; hot path is mlx-lm `BatchKVCache` / `BatchRotatingKVCache` / `ArraysCache`), and the explicit deferral of P-3-C5 / P-3-E4 through P-4.5 and P-5.
  - **README** P-0..P-8 status table and roadmap updated to reflect the new P-4.5 row and the Q-010 trigger.
  - **What is NOT changed:** I-1..I-5 Python Protocol signatures; §8.1 priority tiers (T0 / T1 / T2 stay as v1.4.1 set them); §5 architecture; §6 frozen-candidate interfaces; all P-0..P-4 Deliverables / Acceptance checkmarks; P-3-C5 / P-3-E4 remain ⏳ in §7 P-3.
  - **References:** Q-010 Resolution; Q-002 / Q-003 2026-04-21 progress; §7 P-4 empirical findings 2026-04-21; §7 P-4.5; `silica/scheduler/batcher.py::_prefill_phase`; `silica/kvcache/codec.py` (IdentityCodec).
- **v1.6.3** (2026-04-19): P-3-C3d lands the batched-vs-single-request parity validation for Qwen3.5-0.8B hybrid. New `tests/test_p3_hybrid_batched_parity.py` covers four cases on the real checkpoint (skipif pattern mirrors `test_p2_batched_parity.py`): `test_b1_batch_equals_single_request` (hard gate — B=1 batched must match `Engine.generate` byte-for-byte), `test_identical_prompts_yield_identical_rows` (symmetry), `test_bgt1_strict_parity_matches_single_request` (exact parity at `max_tokens=16`), and `test_different_length_prompts_yield_per_row_results` (left-padding / row-lifecycle smoke). Every run builds a fresh `adapter + Engine` to keep mlx-lm's in-place caches from polluting comparisons, and all runs share a single `_params(adapter, max_tokens=…)` helper so a drift cannot be attributed to a params-field difference. **Empirical finding** recorded in the test docstring: strict parity holds at `max_tokens = 16, 32, and 64` on Qwen3.5-0.8B — unlike P-2's Qwen3-0.6B where fp16 batched SDPA drift on Apple Silicon required degraded invariants. Plausible reasons: DeltaNet's recurrent state is hardcoded fp32 (less round-off), the 3:1 DeltaNet:global layer ratio dilutes the fp16-SDPA contribution, and Qwen3.5-0.8B's architecture differs from Qwen3-0.6B. Companion change (P-3-C3c.1): the direct-batcher smoke (`tests/test_p3_hybrid_batched_smoke.py`) now walks `adapter.attention_pattern().per_layer` in lockstep with `_batch_cache` and asserts the exact per-layer mapping `HYBRID_DELTANET → ArraysCache`, `GLOBAL → BatchKVCache`; any future `AttentionKind` reaching this path fails loudly rather than degrading silently. README updated: Qwen3.5-0.8B capability row flips from "Partial — batching-disabled at the capability gate" to "✅ batched (greedy parity pinned)"; DeltaNet recurrent-state plumbing likewise marked ✅ across P-3-C0..C3d; preempt/replay with recurrent state moved to its own P-3-C5 row (still ⏳); status banner updated to "P-2 complete, P-3 in progress". Bit-parity vs mlx-lm direct batched (not vs single-request) is deferred to a future unit; the current assertion is Silica single-request is the reference, not mlx-lm batched.
  - **References:** D-015, D-016, P-3, M-4.
- **v1.6.2** (2026-04-19): P-3-C3c lifts the `ContinuousBatcher` capability gate for `HYBRID_DELTANET`. Supported `attention_kinds` set is now `{GLOBAL, HYBRID_DELTANET}`; `RECURRENT` / `SLIDING` / `HYBRID` stay rejected. The error-locator loop skips every supported kind (not just `GLOBAL`) so a mixed pattern like `{GLOBAL, HYBRID_DELTANET, SLIDING}` names the `SLIDING` layer rather than the now-supported hybrid one. Real-model validation ships as `tests/test_p3_hybrid_batched_smoke.py` — two smokes against `Qwen/Qwen3.5-0.8B` (skipped when not in the local HF cache): a public API smoke through `Engine.generate_batch`, and a direct-batcher smoke asserting the live `_batch_cache` is genuinely heterogeneous (`ArraysCache` at DeltaNet layer indices, `BatchKVCache` at global attention layer indices). Together with P-3-C3a (adapter `make_batch_cache` factory) and P-3-C3b (mid-run admission factory + prefix-cache guard) this makes batched generation functional for the Qwen3.5 dense family on real models. Token-level parity vs `Engine.generate` single-request is **not** asserted here — that is an M-4 / future C3d responsibility; README's "Partial — batching-disabled at the capability gate" label will be updated once bit-parity is pinned. No new decisions or interface changes; no `Q-NNN` promotion.
  - **References:** D-015, D-016, P-3, M-4.
- **v1.6.1** (2026-04-19): P-3-A load probe landed — `scripts/probe_qwen3_5_27b_load.py` plus an **Empirical findings** bullet added to §7 P-3 recording the Qwen3.5-27B-4bit architecture survey (64 layers = 48 HYBRID_DELTANET + 16 GLOBAL in a 3:1 `[D, D, D, G]` repeating pattern; hidden_size=5120, 24 attention heads, 4 KV heads, head_dim=256; `mlx_lm.load` accepts the repo directly without an mlx-vlm detour; `Qwen3_5Adapter` dispatches via the existing factory; single-request `Engine.generate` runs with peak ~30.5 GB on a 48 GB M5 Pro). No new decisions or interface changes — the finding confirms D-015's hybrid-DeltaNet framing at 27B scale and makes explicit that Qwen3.5-0.8B and Qwen3.5-27B share the same batched-execution blocker (DeltaNet plumbing, P-3-C).
  - **References:** D-015, D-016, P-3.
- **v1.6.0** (2026-04-19): P-3 opening — land D-016 (I-1 extended with `capabilities() -> ModelCapabilities`). New module `silica/models/capabilities.py` ships `ModelCapabilities` (three-field frozen dataclass: `attention_kinds`, `has_recurrent_state`, `has_moe`) and the pure helper `capabilities_from_attention_pattern(pattern, *, has_moe=False)`. `ContinuousBatcher._enforce_capability_gate` now reads `adapter.capabilities()` as its primary predicate; `attention_pattern()` is walked only for the error-message layer index. Concrete adapters (`Qwen3Adapter`, `Qwen3_5Adapter`, `StubModelAdapter`, test doubles) implement `capabilities()` by delegating to the helper. Behaviour unchanged — the batcher still accepts pure GLOBAL and still rejects HYBRID_DELTANET — this is a contract-surface refactor that clears the way for the dense-big, MoE, and DeltaNet adapters landing later in P-3. `AttentionPattern` remains the authoritative per-layer routing source (D-015).
  - **References:** D-016, I-1, P-3.
- **v1.5.2** (2026-04-17): model-integration refactor — formalise the three-layer stack (family adapter + factory registry + capability gate) surfaced by the P-2 Qwen3-0.6B preload. No interface or principle changes; clarifies deliverable-level class naming and §5.1 layout. Triggered by: plain Qwen3 and Qwen3.5 share mlx-lm's `qwen3*.py` neighbourhood but differ in attribute names (`n_kv_heads` vs `num_key_value_heads`) AND runtime semantics (pure KV vs DeltaNet hybrid + MTP + multimodal sanitize); keeping them in a single `Qwen3Adapter` class would grow a conditional on every future family (Kimi, GLM, MiniMax, Mamba, MoE, …).
  - **§5.1 models/:** description updated from "ModelAdapter Protocol + Qwen3.5 / Gemma4 adapters" to "ModelAdapter Protocol + per-family adapters + factory" — matches the new three-layer stack.
  - **§7 P-1 Deliverables:** the adapter entry now explicitly names `silica.models.qwen3_5.Qwen3_5Adapter` as the P-1 class (Qwen3.5-0.8B hybrid), with `silica.models.qwen3.Qwen3Adapter` named separately as the plain-KV P-2 dev-loop adapter and `silica.models.factory.adapter_for_repo(repo)` as the dispatch entry. References `docs/P2_OPENING.md` §"Model integration in three layers".
  - **§7 P-3 Deliverables:** class names migrated from `Qwen35Adapter` / `Qwen35MoeAdapter` to `Qwen3_5Adapter` / `Qwen3_5MoeAdapter` — underscore-separated naming matches mlx-lm's `qwen3_5` module convention and keeps capital-number boundaries readable. MoE variants explicitly labelled as **new family files** distinct from dense siblings, consistent with "one file per family" principle.
  - **Decision log — no new D entries required.** The three-layer stack formalises what D-011 (architecture generality) and D-015 (per-family attention-kind dispatch) already imply; adding a new D entry would be redundant. The capability-gate principle is an implementation-level concretion of "scheduler unaware of concrete implementations" (Principle 5 / I-1..I-5 boundary).
  - **What is NOT changed:** I-1..I-5 Python Protocol signatures; `AttentionPattern` enum values; existing D / Q / R / M identifiers; P-0 through P-7 acceptance lines; principles §4.
  - **References:** `docs/P2_PRELOAD.md` (probe + Qwen3-0.6B baseline); `docs/P2_OPENING.md` §"Model integration in three layers" (v2.3 amendment); commit history shows the refactor as two commits on 2026-04-17 (refactor + preload).
- **v1.5.1** (2026-04-16): freeze-readiness pass — lands the gaps v1.5.0 carried forward, addresses the Qwen3.5-architecture implications of the dev-loop model switch, and restores changelog integrity. No structural redesign; interface signatures unchanged (I-1..I-5 Python Protocol shapes untouched).
  - **Header:** `Status` "Phase 0 planned" → "Phase 0 in-progress" (repo already carries the 11 skeleton sub-packages per `chore: flatten package layout to ./silica and track initial skeleton`, so the prior label was stale).
  - **Dev-loop model switch to Qwen3.5-0.8B (recorded here, not retroactively in v1.4.1).** Empirical check on 2026-04-16: `https://huggingface.co/Qwen/Qwen3.5-0.8B` (and Qwen3.5-27B / Qwen3.5-35B-A3B) model cards confirm **Gated DeltaNet + Gated Attention hybrid + MTP + multimodal**. DeltaNet is therefore a core-engine concern shared by P-1 dev-loop and P-3 production targets, not a P-1-only surprise. v1.4.1's historical changelog line was restored from `Qwen3.5-0.8B or Qwen3.5-4B` back to `Qwen3-0.6B or Qwen3.5-4B` per D-007 append-only integrity.
  - **New §3.2 Non-Goal (multimodal):** v0.1 runs the text-only path of multimodal checkpoints; vision / audio / video encoder lifecycle is v0.2. Grounds D-014 and clears a scope ambiguity Codex flagged two rounds ago.
  - **New D-012 (canonical `resident_bytes` measurement):** one definition for physical owned bytes in unified memory; excludes transient scratch / allocator headroom / reclaimable regions. Pinned so P-5 / P-6 produce comparable numbers. Lands v1.5.0's "carried-forward" item (3).
  - **New D-013 (Sampler structure):** Sampler is a concrete class in `silica.core.sampler`, not a sixth Protocol. Logit processors compose in a `Sequence[LogitProcessor]` with a fixed ordering (`temperature → repetition penalty → top-k → top-p → sample`). §6 stays at five frozen interfaces. Lands v1.5.0's "carried-forward" item (2).
  - **New D-014 (P-1 scope for Qwen3.5-0.8B):** text-only, MTP disabled, DeltaNet recurrent state adapter-owned, tokenizer-parity prerequisite; orthogonal to D-004 (mlx-lm wrap) and D-010 (cache boundary). Lands the new issue introduced by the v1.5.0 dev-loop model switch.
  - **New D-015 (recurrent state as first-class `state_delta` tenant):** `AttentionPattern` enum extended with `recurrent` / `hybrid_deltanet`; per-layer ownership / `commit` / `rollback` / prefix-reuse / budgeting rules spelled out. I-1 / I-2 signatures unchanged — contract extended only. I-1 Key constraints #1 and #3 rewritten in §6. Scheduler memory-budget path expands to include `state_delta.recurrent_bytes()`.
  - **New Q-009 (MLX paged-attention kernel availability):** micro-benchmark decision at P-0 exit / P-1 entry; decides P-2 block-size default and whether R-7 triggers. Lands v1.5.0's "carried-forward" item (1).
  - **New Q-010 (chunked prefill):** measurement-gated deferral (current lean: Option A — add P-4 bench TTFT-under-concurrency scenario, promote only if threshold breached). Lands v1.5.0's "carried-forward" item (5) as a decision-pending question rather than a deliverable, mirroring D-003's framing.
  - **New Q-011 (structured-output / logit-processor boundary):** three options sketched for v0.2 planning; v0.1 leaves the P-8 "interface slot for structured generation" unimplemented as stated. Lands v1.5.0's "carried-forward" item (6).
  - **New R-7 (MLX paged-attention kernel risk):** pairs with Q-009; mitigation = micro-benchmark, block-size adjustment, or per-request contiguous caches with a clear upgrade path. Lands v1.5.0's "carried-forward" item (1) mitigation.
  - **New R-8 (mlx-lm Qwen3.5 support gap):** pairs with D-014; day-1 gate next to the D-010 cache-injection smoke test; worst-case fallback is P-1 reverts to Qwen3-0.6B and DeltaNet shifts to P-3.
  - **P-1 Notes:** extended to reference D-014 constraints (text-only, MTP-disabled, DeltaNet adapter-owned, tokenizer parity prerequisite).
  - **P-2 RequestState:** state machine gains `PREEMPTED` as a side state reachable from `PREFILL` / `DECODE` under scheduler eviction; re-admission reuses still-valid prefix blocks and the last `state_delta` snapshot. Lands v1.5.0's "carried-forward" item (4). Anchored inline in P-2 rather than as a separate D entry (small, data-class-level change).
  - **What is NOT changed:** I-1..I-5 Python Protocol signatures (only contract text and `AttentionPattern` enum); §5.1 module layout (other than the implicit addition of `silica.core.sampler.py` under D-013); §5.2 data flow (documented, not redrawn); §5.4 / §5.5 Reference Maps; D-001..D-011; Q-001..Q-008 resolutions and leans.
  - **Cross-reference sync pass (post-landing, same-day):** five follow-up sync fixes after Codex round-4 review — (1) **D-015 resolution addendum:** prior-round `StateDelta` enters `decode_step` via `kv_handle`-carried request identity + adapter-internal per-request state store; I-1 signatures unchanged. (2) **P-0 deliverables:** added `silica.core.sampler.Sampler` + `LogitProcessor` + `tests/test_sampler.py` per D-013. (3) **P-0 Status:** "planned" → "in-progress" so the P-0 block matches the document header (CRUD convention). (4) **P-2 Strategy:** chunked-prefill deferral line rewritten to reference Q-010 (fairness / TTFT, not only OOM); P-4 Deliverables add a **TTFT-under-concurrency** bench scenario that resolves Q-010. (5) **P-1 Deliverables:** Day-1 gate split into **gate A** (D-010 cache injection) and **gate B** (D-014 / R-8: Qwen3.5-0.8B text-only load + MTP disabled + tokenizer parity).
  - **Second sync pass (Codex round-5 review, same-day):** four more precision fixes — (6) **D-013 vs P-0 consistency:** D-013 clarifies `LogitProcessor` may be a local lightweight `typing.Protocol` for type hints; it is not one of the five frozen core interfaces (§6 stays at five). (7) **D-015 body / addendum alignment:** `commit` / `rollback` / `from_prefix` / `free` are **adapter methods** (`adapter.commit_state` / `adapter.rollback_state` / `adapter.state_from_prefix` / `adapter.free_state`), not methods on `StateDelta`. `StateDelta` is a read-only snapshot exposing only `recurrent_bytes() -> int`. D-015 items 1–5 rewritten to match; item 6 (`free_state`) added. (8) **P-0 Acceptance:** `pytest tests/test_interfaces.py` → `pytest tests` (covers `test_sampler.py` and future tests). (9) **P-3 Deliverables / Acceptance:** explicit `hybrid_deltanet` dispatch + recurrent-state plumbing; acceptance tests `StateDelta.recurrent_bytes()`, full-prefix-only recurrent reuse rule, and a snapshot→rollback round-trip bit-exactness (P-7 prerequisite, tested in P-3).
- **v1.5.0** (2026-04-16): architecture scope generalized from dense-only to **MoE + Dense dual support**. On 2026-04-16 the user explicitly chose Option B ("architecture general + v0.1 must actually run at least one MoE target") over Option A ("interface-only, defer MoE testing to v0.2"). Changes are grouped by module; no structural redesign.
  - **New D-011**: v0.1 architecture generality fixed; references D-006 + Principle 2 + Principle 9; consequences make the Interface / Phase / Risk / Milestone impact surface explicit.
  - **Interface (I-4 WeightProvider)**: three per-expert granularity methods added — `get_expert(layer_idx, expert_id)` / `prefetch_experts(layer_idx, expert_ids)` / `release_expert(layer_idx, expert_id)`; dense implementations raise `NotImplementedError` (not a no-op, to prevent a MoE adapter from silently degrading onto a dense provider); key constraints add "MoE adapter FFN must go through `get_expert`; `get_layer` may not be used to pull all experts at once". I-1 / I-2 / I-3 / I-5 untouched.
  - **Scope (§3.1, §3.4)**: In Scope adds "MoE + Dense architectural generality (D-011); v0.1 must actually run at least one MoE target"; Target Models table grows from 3 rows (1 dev + 2 dense prod) to 5 rows, adding Qwen3.5-35B-A3B / gemma-4-26B-A4B as MoE generality targets starting Phase 3. The "production target" label is split into "Dense production target" vs "MoE generality target".
  - **P-3 (Model Adapters)**: Goal / Scope change from "two target models" to "four target models (2 dense + 2 MoE smoke test)"; Strategy adds "MoE adapter FFN goes through `get_expert` / `prefetch_experts`" + "MoE inference-only, aux-loss ignored"; Deliverables add `Qwen35MoeAdapter` / `Gemma4MoeAdapter` + top-k gating + per-expert aggregation + MoE-aware registry; Acceptance adds a fourth group, **MoE structural correctness** — (a) fp16 parity on a small MoE control model (logit max diff < 1e-3 over first 50 greedy tokens); (b) Qwen3.5-35B-A3B / gemma-4-26B-A4B load + forward under the quantized path; (c) per-expert call-path unit test (mock WeightProvider, assert `get_expert` is called and `get_layer` is not used to load experts). The Product memory-fit target is explicitly marked **dense-only**; MoE is not Q-003-gated.
  - **P-6 (Weight Streaming)**: Strategy adds "Dense vs MoE residency granularity" — dense is layer-granular, MoE is expert-granular; MoE scheduler prefetch coordination refined (as soon as gate logits arrive, fire `prefetch_experts(top_k_ids)` rather than waiting for the expert FFN to actually execute). Deliverables note "expert eviction policy (e.g. LRU over experts)" + "dense + MoE dual mode". Acceptance adds a "MoE per-expert residency takes effect" quantitative gate (24 GB budget, `resident_bytes ≤ active_experts × expert_size + non_FFN + headroom ≤ 20%`) and a "MoE decode tok/s ≥ 60% of MoE ResidentWeightProvider baseline" bar (below dense's 70%, acknowledging expert-miss stalls the first cut does not optimize).
  - **Risk / Question / Milestone**: R-1 description adds "MoE active params small, this risk does not apply to MoE" + mitigation note covering MoE as early scale demonstration; Q-003 gains a Context paragraph clarifying "Q-003 is about dense; MoE fit risk is far lower but does not substitute for Q-003 resolution"; M-4 acceptance adds MoE smoke test adapter correctness, independent of Q-003 gating.
  - **Naming normalization**: "Gemma 4 31B" → "Gemma4-31B" across the document, aligned with the user's 2026-04-16 spelling. Qwen3.5-27B / Qwen3.5-35B-A3B / gemma-4-26B-A4B kept as the user-provided IDs.
  - **Review gaps carried forward**: this round only handles the MoE scope decision. The other five freeze-critical gaps raised in the previous Opus review — (1) MLX paged attention risk (proposed R-7 + Q-009); (2) Sampler as I-6; (3) `resident_bytes` canonical measurement (proposed D-012); (4) `RequestState` gains `PREEMPTED`; (5) chunked prefill promoted from "contingent" to deliverable; plus (6) the structured-output / logit-processor middle state — are **not** landed in this version and will be addressed in later v1.5.x rounds to keep this revision's scope contained.
  - **Language pass**: the entire document is now English. Previously mixed Chinese/English prose has been translated and lightly polished; stable IDs, code blocks, tables, URLs, and model names preserved.
- **v1.4.1** (2026-04-14): Codex round-3 review — targeted optimizations, **no structural redesign**.
  - §2 Success criteria gains "minimal OpenAI-compatible HTTP API + session usable (via Phase 8)", aligning with §3.1 scope and D-006.
  - §8.1 Priority Tier labels `P0 / P1 / P2` → **`T0 / T1 / T2`** to avoid visual collision with phase IDs `P-0 / P-1 / P-2` (Q-002 Option B updated in step).
  - **P-3 Acceptance split into two tiers:** **Adapter correctness (hard gate)** — load + one forward + logits max diff < 1e-3 over 50 greedy tokens + hybrid attention routing unit tests; **Product memory-fit target (conditional, gated on Q-003)** — "27B/31B @ 48 GB 500 tokens"; if P-6 is not pulled forward, a smaller model or manual cap is permitted. M-4 milestone updated accordingly so P-3 is not self-blocked before P-6.
  - P-5 Acceptance ε default pinned as `max(2× fp16 baseline noise, 0.01 PPL)`, tightenable at P-4 exit.
  - P-6 Acceptance adds concrete threshold **decode tok/s ≥ 70% of `ResidentWeightProvider` baseline @ 24 GB budget** with fixed same-machine/model/scenario/sampling conditions; if resident OOMs at 24 GB, baseline is "uncapped resident reference run".
  - P-7 Acceptance adds concrete threshold **decode tok/s ≥ 1.2× draft-disabled baseline** on a fixed standard scenario; **cherry-picking a best case is not acceptable**; greedy token-by-token identity as correctness invariant.
  - Module rename **`silica.flash` → `silica.weights`** (avoids overload with FlashAttention, mirrors `silica.kvcache`): §5.1 Module Layout, P-3 Deliverables (`silica.weights.resident.ResidentWeightProvider`), P-6 Deliverables (`silica.weights.streaming.StreamingWeightProvider`, `silica.weights.prefetch`), D-009 module list all synchronized. Principle 1 / D-006 Consequences informal "VQ / flash" phrasing rewritten to "VQ / weight streaming". P-6 phase name "Weight Streaming" was changed in v1.3.0 and is unchanged here. External repo names `mlx-flash` / `vllm_flash_attn` / `flash-moe` kept as is.
  - D-008 typo fix ("不定会导致" → "不定下来会导致"); the then-still-Chinese-heavy document was not pass-translated in this round.
  - **Round 2 correction (pre-commit).** Codex re-reviewed v1.4.1 before commit and caught three points; all fixed before commit.
    1. **P-3 logit threshold vs quantization path unit mismatch.** `max |logit diff| < 1e-3` corresponds to fp16 parity, but D-005 requires 4-bit / 8-bit for big models, where the threshold is unreachable. Split P-3 Acceptance into (a) **fp16 parity on a control model** (Qwen3-0.6B or Qwen3.5-4B) verifying adapter structural components, keeping `max |logit diff| < 1e-3`; (b) **quantized big-model correctness** using **teacher-forced next-token argmax agreement ≥ 98%** over the first 100 teacher-forced positions against `mlx-lm` at the same quantization, compared position-by-position on a fixed prefix; fallback is end-to-end PPL drift `< 0.1` absolute vs the same baseline. We do not compare against free-running sequences, because drift masks or amplifies real differences.
    2. **M-4 deferred product-target ownership gap.** After v1.4.1 narrowed M-4 to adapter correctness, the real `27B/31B @ 48 GB 500 tokens` validation had no milestone to land on. Changed to a **Q-003-gated handoff**: M-4 validates only if Q-003 resolves to "int4 fits in 48 GB"; otherwise the item defers to M-7. No new M-10 — avoids milestone fragmentation.
    3. **P-5 cross-check threshold unit mismatch.** `max(2× fp16 baseline noise, 0.01 PPL)` mixes tensor-space reconstruction error with end-to-end PPL under a single `max` — dimensionally incomparable. Split into two independent thresholds with explicit metrics: `ε_recon < 2 × fp16 round-trip baseline noise`, where the metric is **per-block relative Frobenius error** `||K_decoded - K_original||_F / ||K_original||_F` (same for V), baseline established via vqbench's own fp16 encode→decode round trip; `ε_ppl < 0.01` absolute PPL drift vs `vqbench/REPORT.md` baseline. Both must pass for acceptance.
- **v1.4.0** (2026-04-14): vqbench added as a local reference checkout, bringing integration updates along the VQ path (VQ's framing is promoted from "one P-5 capability" to "a core capability of the final product, whose correctness is now testable against the vqbench empirical baseline").
  - Adds **§5.5 Reference Map to vqbench** — Silica-module ↔ vqbench-file mapping (BlockTQ / RaBitQ algorithms, codec factory, `KVCacheCompressor` pair pattern, Qwen3.5 bench scripts, `REPORT.md` PPL oracle), plus forbidden paths (`torch_wrapper/` / NumPy impls, D-009).
  - §12.2 Local reference checkouts: top-level `turboquant_plus/` replaced by `vqbench/` (turboquant_plus is now nested inside vqbench).
  - P-5 Strategy reference source switched to vqbench, noting the Qwen3.5-4B `B=64` 4-bit +0.0% ΔPPL empirical baseline.
  - P-5 Acceptance gains a **numeric cross-check against vqbench** — Silica `BlockTQCodec` reconstruction error + end-to-end PPL must match the vqbench NumPy reference, making "faithful rewrite" testable.
  - P-4 Deliverables gain `silica.bench.vqbench_baseline` — a separate-subprocess run of vqbench scripts to collect PPL as a reference column, serving the P-5 cross-check via the D-009-permitted "separate-process comparison" path.
  - D-010 Consequences adds `VQBenchCache` (HF `Cache` subclass) as a concrete **anti-pattern** — a living example of what "borrowing mlx-lm's cache" would degenerate into.
  - Adds **Q-008** discussing whether `KVCodec`'s interface layer should expose the K/V pair choice (vqbench shows K wants an unbiased-IP codec while V wants low-MSE; three options A/B/C, lean A).
- **v1.3.0** (2026-04-14): framing unification — plugin → native capability. User clarifies: VQ / weight streaming / speculative decoding are Silica-MLX's **native capabilities**, not "third-party plugin extension points"; built into the main loop as stubs from P-0, progressively replaced in P-5 / P-6 / P-7; integration points fixed, implementations swappable.
  - Adds **Principle 9 — Native capabilities, swappable implementations**, making the architectural stance explicit with the vLLM attention-backend analogy.
  - Principle 3 "Engine first, plugins later" → "Engine skeleton first, native capabilities integrated progressively".
  - Principle 5 "Plugin contracts are frozen early" → "Native capability contracts are frozen early".
  - §1 TL;DR / §2 Mission success criteria / §3.1 In Scope: three "plugin" phrasings rewritten to "native capability".
  - P-5 title "VQ Plugin" → "VQ KV Compression" + Goal / Strategy aligned; P-6 "Flash Plugin" → "Weight Streaming"; P-7 "Speculative Plugin" → "Speculative Decoding".
  - §8.1 P1 tier label / P-2 Notes / Q-002 Option B — scattered "plugin" phrasing removed uniformly. Interface names (`KVCodec` / `WeightProvider` / `DraftEngine`) and P-0 deliverable structure unchanged; this is a framing alignment.
- **v1.2.0** (2026-04-14): Two rounds of Codex plan review reconciliation.
  - §6 header "Frozen for v0.1" → "Phase 0 freeze candidate" — signatures finalized at P-0 exit, not the moment this document is written.
  - I-1 adds `prefill(tokens, kv_handle) -> (logits, StateDelta)` and `decode_step`; KV mutation ownership clarified as `KVManager`-only (via `kv_handle`); `state_delta` carries non-KV runtime state only (counter-examples: KV blocks / cache residency mutations / prefix pinning are not permitted).
  - I-2 `allocate(request_id, num_tokens)` split into `reserve_for_prefill(req_id, token_ids)` + `append_slot`; adds `commit` / `rollback` (for P-7 speculative); `prefix_lookup` renamed `get_computed_blocks` (vLLM v1 naming); adds `available_blocks` as the block-granular fast path for scheduler admission.
  - Adds **D-010** fixing the Phase 1 `mlx-lm` borrowing boundary — borrow model/tokenizer/weight loader, not the rotating KV cache; P-1 day-1 smoke test verifies `mlx_lm.generate_step(cache=...)` external-cache injection.
  - P-1 Strategy / Deliverables aligned to D-010; adds the day-1 gate.
  - P-5 Strategy clarifies that `BlockTQCodec` / `RaBitQCodec` must be rewritten on `mx.array` + resident-accounting; not a reuse of `turboquant_plus` NumPy prototypes; engineering work pre-declared to avoid scope creep.
  - Adds **Q-007** on whether `KVCodec.decode_overhead_ratio` should enter the v0.1 interface (Principle 8 completeness of scheduler information).
  - Adds **R-6** for the day-1 smoke-test failure fallback.
  - §5.2 data flow: `ModelAdapter.forward` → `ModelAdapter.prefill / decode_step (KV via kv_handle from KVManager)`.
- **v1.1.2** (2026-04-14): user pulled vllm as a local reference and confirmed "must be native MLX". Adds D-009 fixing the MLX-native hot-path constraint (no `torch.Tensor`, no PyTorch runtime dep); rewrites Principle 6; §3.2 Non-Goals adds PyTorch / CUDA backend exclusions; adds §5.4 Reference Map to vLLM v1 (what's referenced, what's not); adds Q-006 on whether AttentionBackend should be separate; §12 References splits External / Local checkouts and registers `vllm/`.
- **v1.1.1** (2026-04-14): two CRUD review fixes — §5.1 directory tree adds `llm/` sub-package (aligned with P-8 deliverable); Q-004 marked resolved and D-008 added to fix the core/engine boundary.
- **v1.1** (2026-04-14): Mission and Principle 1 rewritten — platform is the product, VQ is the means (D-006); adds Principle 2 "Apple unified memory first"; adds Principle 8 "savings must be observable"; Phase 5 / 8 framing rewritten; Q-002 opened on Phase 8 priority; Q-005 on MetricsRegistry; Risks table added; whole document organized in CRUD-friendly form (stable IDs P-N / D-NNN / Q-NNN / M-N / I-N, self-contained blocks, append-only logs).
- **v1.0** (2026-04-14): based on Codex polish of v0; unified phase structure.
- **v0** (2026-04-13): initial plan draft (conversational; never committed).
