# Silica-MLX Plan

| Field        | Value                                  |
| ------------ | -------------------------------------- |
| Version      | v1.1.2                                 |
| Last updated | 2026-04-14                             |
| Status       | Phase 0 planned                        |
| Maintainer   | xinyu                                  |
| Source       | `docs/PLAN.md` (single source of truth) |

> **CRUD 约定**: 本文档所有稳定 ID (Phase `P-N`, Decision `D-NNN`, Open Question `Q-NNN`, Milestone `M-N`, Interface `I-N`) 一经分配不再复用。编辑请只改对应 block；新事实走 Decisions Log; 新问题走 Open Questions; Phase 状态变化改对应 Phase block 的 `Status` 字段并在 Changelog 追加一行。

---

## 目录

1. [TL;DR](#1-tldr)
2. [Mission](#2-mission)
3. [Scope](#3-scope)
4. [Design Principles](#4-design-principles)
5. [Architecture Overview](#5-architecture-overview)
6. [Core Interfaces (Frozen for v0.1)](#6-core-interfaces-frozen-for-v01)
7. [Phases](#7-phases)
8. [Priority & Milestones](#8-priority--milestones)
9. [Decisions Log](#9-decisions-log)
10. [Open Questions](#10-open-questions)
11. [Risks](#11-risks)
12. [References](#12-references)
13. [Changelog](#13-changelog)

---

## 1. TL;DR

**Silica-MLX 是一个单 Mac 芯片的本地 LLM 推理平台**，MLX-native，vLLM-style core + mini-sglang-style 外层。目标是让 M5 Pro 48GB 能稳定跑 Qwen3.5-27B / Gemma 4 31B。VQ、weight streaming、speculative decoding 作为插件被平台主动利用（不是被动支持），用来突破内存和计算限制。

**平台本身是产品**。VQ 等技术是武器，不是研究对象。(见 D-006)

---

## 2. Mission

在单台 Apple Silicon Mac 上，让开发者能像在云上用 vLLM 一样流畅地本地跑 27B–31B 量级的大模型。

**目标用户**: Mac 上想本地跑大模型做 app、做实验、做隐私敏感工作的开发者。

**不是**: VQ 算法研究者、分布式 serving 系统研究者、云厂商。

**成功标准 (v0.1)**:
- 在 48GB M5 Pro 上稳定跑通 Qwen3.5-27B 和 Gemma 4 31B
- 提供 Python API + 最小 CLI
- 统一 benchmark / runtime 路径
- 插件架构允许 VQ、weight streaming、speculative 无侵入接入

---

## 3. Scope

### 3.1 In Scope (v0.1)

- 单 Mac 芯片、单进程本地推理
- MLX-native，不依赖 CUDA 假设
- Qwen3.5-27B、Gemma 4 31B 两个目标模型
- Python API + 最小 CLI
- vLLM-style core: paged KV, continuous batching, prefix cache, memory budget
- 五个冻结接口: ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine
- 插件: VQ (BlockTQ, RaBitQ), weight streaming, draft-target speculative
- 统一 benchmark 路径
- 最小 OpenAI-compatible HTTP API + session (Phase 8)

### 3.2 Non-Goals (v0.1)

- 分布式、多节点、多 Mac 协作
- PD (prefill/decode) 分离
- Tensor parallelism (单芯片不需要)
- 一开始就重写 native Metal kernel
- GGUF / AWQ 等非 MLX 量化格式
- 完整 agent orchestration
- PQ / OPQ 等被明确排除的 codec
- 压缩域直接 attention fast path (见 D-003，留 v0.2)
- DFlash / EAGLE / Medusa 等复杂 speculative 方案 (留 v0.2)
- **PyTorch 运行时依赖** (D-009): inference hot path 不允许出现 `torch.Tensor`; torch 只能作为 offline 权重转换的可选 dev 依赖
- **CUDA / ROCm / XPU / TPU 后端** (D-009): `csrc/`、CUDA kernels、device-specific worker 都不在范围内

### 3.3 Target Hardware

| 字段 | 值 |
| --- | --- |
| 机型 | Apple M5 Pro |
| 内存 | 48 GB unified memory |
| OS | macOS |
| 加速 | MLX on Apple Silicon GPU + Neural Engine (MLX-managed) |

### 3.4 Target Models

| 模型 | 参数量 | 用途 | 起用阶段 |
| --- | --- | --- | --- |
| Qwen3-0.6B | 0.6B | 开发/迭代起步模型 | Phase 1 |
| Qwen3.5-27B | 27B | 生产目标 | Phase 3 |
| Gemma 4 31B | 31B | 生产目标 | Phase 3 |

---

## 4. Design Principles

稳定原则；修改需进 Decisions Log 并开新条目。

1. **Platform as product, VQ as weapon.** Silica-MLX 本身是产品；VQ / flash / speculative 是让产品跑好大模型的手段，不是研究对象。不要把"VQ 是终极 deliverable"写进任何设计。(D-006)

2. **Single Mac chip + Apple unified memory first.** 硬约束是单芯片；不做分布式。但要主动利用 unified memory —— 权重/KV/activations 共享同一物理池的事实必须反映到 WeightProvider 和 KVManager 设计里，不能把 Mac 抽象成一个普通 GPU。

3. **Engine first, plugins later.** 先做 engine core 再做插件。Phase 0–4 是 engine 骨架，Phase 5–8 是插件与外层。

4. **Bench and runtime share the same path.** Benchmark 必须是 `silica.engine.Engine` 的薄封装。不能有第二条评测路径。

5. **Plugin contracts are frozen early.** 五个核心接口 (ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine) 在 Phase 0 冻结；scheduler / engine core 不感知具体实现。改动需走 Decisions Log。

6. **MLX-native hot path (hard constraint).** Inference hot path 必须 100% 走 MLX: 所有 tensor 都是 `mx.array`, 所有 ops 都走 MLX。**不允许** `torch.Tensor` / `numpy.ndarray` 出现在 `silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler` 的 hot path。Phase 1 可以 wrap `mlx-lm` **因为 mlx-lm 本身就是 MLX-native**，不能替换成 torch-based 的 wrapper。vllm / transformers 仅作为**算法参考**, 不作为运行时依赖。详见 D-009。

7. **Small over large in early phases.** Phase 1 起步用 Qwen3-0.6B，Phase 3 才切到目标大模型。先把闭环跑通再扩大规模。

8. **Savings must be observable.** 任何压缩/流式/加速优化带来的节省必须能被 scheduler 看见 (e.g. `KVCodec.logical_bytes` vs `resident_bytes`)，否则 memory budgeter 没法把节省转化为"多收请求/更长 context"。

---

## 5. Architecture Overview

### 5.1 Module Layout

目标布局 (Phase 0 落盘后以实际仓库为 source of truth):

```
silica-mlx/
├── pyproject.toml
├── README.md
├── docs/
│   └── PLAN.md                   # this file
├── python/silica/
│   ├── __init__.py               # re-exports Engine, LLM
│   ├── core/                     # Request, SamplingParams, Context, logging, profiling
│   ├── mlx/                      # MLX array utilities, profiling hooks
│   ├── engine/                   # Engine class, generate() loop
│   ├── scheduler/                # ContinuousBatcher, request lifecycle, memory budget
│   ├── kvcache/                  # PagedKVCache, PrefixCache, KVCodec Protocol
│   ├── models/                   # ModelAdapter Protocol + Qwen3.5 / Gemma4 adapters
│   ├── flash/                    # WeightProvider Protocol + Resident/Streaming impls
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
     → KVManager.prefix_lookup / allocate (KVCodec 透明接入)
   → ModelAdapter.forward (through WeightProvider)
     → KVCodec.encode_block / decode_block per layer
   → Sampler (+ optional DraftEngine)
 → token stream
```

Phase 8 在最上层加一个 `silica.server.openai_api` FastAPI wrapper，内部还是同一个 Engine 实例。

### 5.3 Process Model

v0.1 **单进程**。无 tokenizer worker 拆分，无 detokenizer 拆分，无 scheduler worker 拆分。未来如果要拆，作为 v0.2 讨论。

### 5.4 Reference Map to vLLM v1

**仅用作算法/架构参考**，不是运行时依赖。vllm 仓库本地路径: `vllm/` (gitignored)。vllm 的 v0 (老的 `engine/llm_engine.py`) **不是**我们的参考目标; 我们对标的是 v1 (`vllm/v1/`)。

| Silica 模块 | vLLM v1 参考文件 | 用途 |
| --- | --- | --- |
| `silica.engine.Engine` | `vllm/v1/engine/llm_engine.py`, `vllm/v1/engine/core.py` | 引擎主循环结构 |
| `silica.scheduler.batcher` | `vllm/v1/core/sched/scheduler.py` | continuous batching 调度 |
| `silica.scheduler.budget` | `vllm/v1/core/kv_cache_manager.py` (budget 部分) | 内存预算与 admission |
| `silica.kvcache.paged.PagedKVCache` | `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_manager.py` | paged/block KV 分配器 |
| `silica.kvcache` (接口) | `vllm/v1/kv_cache_interface.py` | KV cache spec 与 layout |
| `silica.kvcache.prefix` | `vllm/v1/core/kv_cache_manager.py` (prefix 部分) | prefix cache 命中与复用 |
| `silica.core.request.RequestState` | `vllm/v1/request.py` | 请求状态机 |
| `silica.core.sampling` / sampler | `vllm/v1/sample/` | 采样实现参考 |
| `silica.models.*` attention | `vllm/v1/attention/backend.py` (仅接口思路) | attention backend 可插拔模式; **不 copy CUDA 实现** |
| `silica.speculative.*` | `vllm/v1/spec_decode/` | speculative decoding 架构参考 |

**不参考**的部分:
- `vllm/csrc/` — C++/CUDA 内核
- `vllm/vllm_flash_attn/` — CUDA flash attention
- `vllm/v1/worker/gpu_model_runner.py`, `tpu_model_runner.py`, `xpu_model_runner.py`, `cpu_model_runner.py` — 设备特定 runner; 我们只写一个 `silica.mlx.runner`
- `vllm/v1/executor/`, `vllm/distributed/`, `vllm/v1/core/kv_cache_coordinator.py` — 多进程 / 多节点协调, v0.1 不需要
- vllm 的 v0 路径 (`vllm/engine/`, `vllm/worker/` 顶层) — 已被 v1 取代

---

## 6. Core Interfaces (Frozen for v0.1)

v0.1 冻结的五个接口。实现可以换，签名不能随便动。要动需要进 Decisions Log 加新条目。下面给的是契约骨架，实际 `typing.Protocol` 签名以代码为准。

### I-1 ModelAdapter

负责模型结构、tokenizer、layer execution、attention pattern。

```python
class ModelAdapter(Protocol):
    config: ModelConfig

    def build(self, weight_provider: WeightProvider) -> Module: ...
    def kv_layout(self) -> KVLayout: ...                  # num_layers, n_kv_heads, head_dim, dtype
    def attention_pattern(self) -> AttentionPattern: ...  # global / sliding / hybrid per layer
    def tokenizer(self) -> Tokenizer: ...
```

**关键约束**: `attention_pattern()` 必须能表达 Qwen3.5 的 hybrid attention —— 不同 layer 走不同 cache routing，否则 Phase 3 会被迫改 scheduler。

### I-2 KVManager

负责 paged/block KV、prefix cache、memory budget。

```python
class KVManager(Protocol):
    block_size: int

    def allocate(self, request_id: str, num_tokens: int) -> BlockList: ...
    def free(self, request_id: str) -> None: ...
    def prefix_lookup(self, token_ids: Sequence[int]) -> PrefixHit: ...
    def budget(self) -> MemoryBudget: ...  # logical_bytes, resident_bytes, headroom
```

**关键约束**: `budget()` 必须同时报告 logical 和 resident 字节 (Principle 8)。Scheduler 基于此做 admission control。

### I-3 KVCodec

KV 编解码的统一抽象。**v0.1 不包含**压缩域直接 attention fast path (D-003)。

```python
class KVCodec(Protocol):
    block_size: int
    k_dtype: mx.Dtype
    v_dtype: mx.Dtype

    def encode_block(self, k: mx.array, v: mx.array) -> CodedBlock: ...
    def decode_block(self, block: CodedBlock) -> tuple[mx.array, mx.array]: ...
    def logical_bytes(self, num_tokens: int) -> int: ...   # fp16 baseline equivalent
    def resident_bytes(self, num_blocks: int) -> int: ...  # actual storage
```

**关键约束**: codec 只看一个 layer 的 K/V，不感知 batch / scheduler / 模型结构。

### I-4 WeightProvider

权重驻留/流式抽象。Phase 1 用 Resident，Phase 6 接 Streaming。

```python
class WeightProvider(Protocol):
    def get_layer(self, layer_idx: int) -> LayerWeights: ...      # sync blocking
    def prefetch(self, layer_indices: Sequence[int]) -> None: ...  # hint, may no-op
    def release(self, layer_idx: int) -> None: ...
    def resident_bytes(self) -> int: ...
```

**关键约束**: `get_layer` 同步阻塞；`prefetch` 是提示，可以是 no-op。调用方不需要知道权重实际在哪。Streaming 实现要利用 unified memory (Principle 2)，而不是假装是 PCIe 设备。

### I-5 DraftEngine

speculative decoding 的 draft 提供方。Phase 1–6 用 Noop，Phase 7 接 DraftTarget。

```python
class DraftEngine(Protocol):
    def propose(self, ctx: RequestState, k: int) -> DraftTokens: ...
    def commit(self, ctx: RequestState, accepted_len: int) -> None: ...
```

---

## 7. Phases

每个 Phase 的结构统一: `ID / Goal / Scope / Strategy / Deliverables / Acceptance / Dependencies / Status / Notes`。编辑某个 Phase 时只动该 block。状态值: `planned / in-progress / done / blocked / obsolete`。

### P-0 Phase 0 — Skeleton

- **Goal**: 把 repo、包结构、配置、日志、profiling、接口骨架搭起来。
- **Scope**: `pyproject.toml`, 11 个子包骨架, 五个 Protocol, logging, profiler, 最小 interface test。
- **Strategy**:
  - uv + Python 3.12 (D-001)
  - stdlib `logging` + 彩色 formatter (不上 structlog)
  - `pydantic` v2 BaseSettings 做 config
  - profiler 用 `@contextmanager`, 全局 `MetricsRegistry`, 统一 metrics schema
- **Deliverables**:
  - [ ] `pyproject.toml` (uv, mlx, mlx-lm, numpy, pydantic, pytest, ruff, mypy; Python 3.12)
  - [ ] 11 个子包骨架 + `__init__.py`
  - [ ] `silica.core.request.Request` / `RequestState`
  - [ ] `silica.core.sampling.SamplingParams`
  - [ ] `silica.core.logger` + `silica.core.profiler`
  - [ ] 五个 Protocol (I-1..I-5) + 每个的 stub 实现 (`StubModelAdapter`, `NullKVManager`, `IdentityCodec`, `ResidentWeightProvider` stub, `NoopDraftEngine`)
  - [ ] `tests/test_interfaces.py` 验证 Protocol 形状 + stub 实例化
  - [ ] `from silica import Engine` 可 import (即使是 `NotImplementedError` stub)
- **Acceptance**:
  - [ ] `uv pip install -e .` 成功
  - [ ] `pytest tests/test_interfaces.py` 通过
  - [ ] 统一 metrics schema 有字段: `ttft_ms`, `prefill_tok_s`, `decode_tok_s`, `resident_mb`, `logical_kv_bytes`
- **Dependencies**: 无
- **Status**: planned
- **Notes**: 这是所有后续工作的基座；接口一旦冻结，改动需走 Decisions Log。

### P-1 Phase 1 — Baseline Engine

- **Goal**: 最小可跑的 MLX-native 推理主循环。
- **Scope**: 单请求, greedy/temperature/top-p sampling, token streaming, 最小 CLI。
- **Strategy**: Phase 1 的 `ModelAdapter` 是 **thin wrapper over `mlx-lm`**, 不重写模型执行细节 (D-004)。KV 先用 `SimpleKVCache` (单请求, 非 paged)。
- **Deliverables**:
  - [ ] `silica.engine.Engine.generate(prompt, sampling_params)` 返回 token 流
  - [ ] `silica.mlx.runner` 包装 mlx-lm 的 forward
  - [ ] `silica.kvcache.simple.SimpleKVCache` (单请求版)
  - [ ] `silica.server.cli`: `python -m silica run --model Qwen/Qwen3-0.6B --prompt "..."`
  - [ ] 基础采样: greedy, temperature, top-p
  - [ ] `silica.models.qwen3.Qwen3Adapter` (thin wrapper)
- **Acceptance**:
  - [ ] 能稳定生成文本
  - [ ] greedy 下和 mlx-lm 参考实现逐 token 一致 (固定 seed / 同一模型)
  - [ ] 能从 profiler 拿到 TTFT, decode tok/s, resident memory
- **Dependencies**: P-0
- **Status**: planned
- **Notes**: 开发机模型: Qwen3-0.6B。不要在 Phase 1 就上 27B/31B。

### P-2 Phase 2 — Mini-vLLM Core

- **Goal**: 真正的 engine 骨架 —— paged KV + continuous batching + prefix cache + memory budget。
- **Scope**: 多请求并发, 共享前缀命中, 内存预算管理, 请求状态机。
- **Strategy**:
  - Block size 默认 16 tokens (Phase 4 bench 后评估)
  - Radix prefix cache 参考 mini-sglang
  - chunked prefill 先不做, Phase 3 如果被 OOM 逼出来再加
- **Deliverables**:
  - [ ] `silica.kvcache.paged.PagedKVCache` 实现 KVManager (I-2)
  - [ ] `silica.kvcache.prefix.RadixPrefixCache`
  - [ ] `silica.core.request.RequestState` 状态机: `WAITING → PREFILL → DECODE → DONE/ABORTED`
  - [ ] `silica.scheduler.batcher.ContinuousBatcher`
  - [ ] `silica.scheduler.budget.MemoryBudgeter`
- **Acceptance**:
  - [ ] 同时跑 8 个 request 稳定
  - [ ] 构造共享前缀 prompt 能命中 prefix cache (prefix cache hit 计数可验证)
  - [ ] 内存超预算时 abort / 排队不崩
- **Dependencies**: P-1
- **Status**: planned
- **Notes**: 这是最重要的一个 Phase; 骨架决定后续所有插件的接入方式。

### P-3 Phase 3 — Model Adapters

- **Goal**: Qwen3.5-27B 和 Gemma 4 31B 真正跑通。
- **Scope**: 两个目标模型的 adapter, Qwen3.5 hybrid attention, MLX-native 4/8-bit 量化。
- **Strategy**:
  - 量化走 mlx-lm 现成的 4-bit / 8-bit 路径 (D-005)
  - 权重加载走 `WeightProvider`, 即使此阶段仍是 `ResidentWeightProvider`
  - Qwen3.5 的 hybrid attention 通过 `AttentionPattern` 表达, scheduler 按 layer idx 路由 KV
- **Deliverables**:
  - [ ] `silica.models.qwen3_5.Qwen35Adapter`
  - [ ] `silica.models.gemma4.Gemma4Adapter`
  - [ ] `AttentionPattern` 的 hybrid 实现
  - [ ] `silica.flash.resident.ResidentWeightProvider` (全量驻留)
  - [ ] 模型 registry: `silica.models.registry`
- **Acceptance**:
  - [ ] 两个目标模型能在 M5 Pro 48GB 上跑通至少 500 token
  - [ ] logits 与 HF 参考实现对照差异在数值噪声范围内 (前 N token greedy)
  - [ ] Qwen3.5 hybrid attention 的 cache routing 语义正确 (单元测试覆盖)
- **Dependencies**: P-2
- **Status**: planned
- **Notes**: 风险点 —— Qwen3.5-27B fp16 ~54 GB，必须量化。若 4-bit 仍放不下，Q-003 触发, P-6 提前。

### P-4 Phase 4 — Bench Unification

- **Goal**: benchmark 直接走 Engine, 不再有旁路。
- **Scope**: 统一 bench runner, 标准场景, 统一结果格式。
- **Strategy**: bench 是 `silica.engine.Engine` 的薄封装。
- **Deliverables**:
  - [ ] `silica.bench.runner.BenchRunner`
  - [ ] `silica.bench.scenarios`: 短 in/长 out; 长 in/短 out; 多并发共享前缀
  - [ ] 统一指标 schema: TTFT, prefill tok/s, decode tok/s, resident memory, logical KV bytes, quality
  - [ ] 输出: jsonl + markdown 报告
- **Acceptance**:
  - [ ] 单条命令跑出 baseline 表 (可贴 README)
  - [ ] bench 和 runtime 无路径分叉 (同一个 Engine 实例)
- **Dependencies**: P-3
- **Status**: planned
- **Notes**: Phase 4 产出的 baseline 数据决定 Q-003 (Phase 6 是否提前)。

### P-5 Phase 5 — VQ Plugin

- **Goal**: KV 压缩作为插件挂进去, 让平台在同样内存预算下能容更多请求或更长 context。
- **Scope**: `IdentityCodec`, `BlockTQCodec`, `RaBitQCodec`。**PQ / OPQ 不进主线**。
- **Strategy**:
  - v0.1 不做压缩域 fast path (D-003)
  - `PagedKVCache(codec=...)` 注入式切换
  - scheduler 通过 `KVCodec.logical_bytes` / `resident_bytes` 获得节省信息并据此多收请求 (Principle 8)
- **Deliverables**:
  - [ ] `silica.vq.block_tq.BlockTQCodec`
  - [ ] `silica.vq.rabitq.RaBitQCodec`
  - [ ] `silica.kvcache.paged.PagedKVCache` 支持 codec 注入
  - [ ] bench 加 `--kv-codec {fp16,block_tq,rabitq}` 开关
  - [ ] scheduler budget 利用 codec savings 的 admission 策略
- **Acceptance**:
  - [ ] 切 codec 不改 scheduler / model adapter
  - [ ] 同一组场景下 fp16 vs codec 的 quality 差距和内存节省可从 bench 一键拿到
  - [ ] 开 BlockTQ 后同样内存预算能收更多请求 (量化验证 Principle 8)
- **Dependencies**: P-4
- **Status**: planned
- **Notes**: BlockTQ / RaBitQ 的具体实现参考 `turboquant_plus/` (gitignored reference impl)。

### P-6 Phase 6 — Flash Plugin

- **Goal**: weight streaming 解决权重驻留压力。
- **Scope**: `ResidentWeightProvider` (已有) + `StreamingWeightProvider` + scheduler prefetch 协同。
- **Strategy**:
  - 参考 `mlx-flash` 思路
  - 在 layer N 计算时 prefetch layer N+1
  - **利用 Apple unified memory** (Principle 2): 不是"GPU 从 disk/CPU 拉权重", 而是"同一内存池内不同区域的生命周期管理"
- **Deliverables**:
  - [ ] `silica.flash.streaming.StreamingWeightProvider`
  - [ ] `silica.flash.prefetch`: scheduler 预取协同逻辑
  - [ ] unified memory aware 的 residency 策略
- **Acceptance**:
  - [ ] 人为限制内存预算 (如 24GB) 下 Qwen3.5-27B int4 不 OOM
  - [ ] 性能退化可量化可接受 (bench 对照 ResidentWeightProvider)
- **Dependencies**: P-2 (scheduler) + P-3 (model adapter)
- **Status**: planned
- **Notes**: 可能从 P1 tier 提前到 P-5 之前 —— 如果 P-3 发现 27B/31B 在 baseline 就放不下 (Q-003)。

### P-7 Phase 7 — Speculative Plugin

- **Goal**: draft-target speculative decoding 加速 decode。
- **Scope**: `NoopDraftEngine` (已有) + `DraftTargetEngine` (最基础版)。
- **Strategy**:
  - 用一个小模型做 draft, 大模型做 verify
  - 先不做 EAGLE / Medusa / DFlash 复杂方案 (推到 v0.2)
- **Deliverables**:
  - [ ] `silica.speculative.draft_target.DraftTargetEngine`
  - [ ] 与 decode loop 的集成
  - [ ] acceptance / rollback metrics
  - [ ] bench 开关 `--speculative {none,draft_target}`
- **Acceptance**:
  - [ ] greedy 下开/关 speculative 输出 token 序列一致
  - [ ] 有可测的 decode tok/s 提升
  - [ ] 不破坏 baseline correctness
- **Dependencies**: P-2 + P-3
- **Status**: planned
- **Notes**: DFlash / dflash-mlx 作为 v0.2 候选。

### P-8 Phase 8 — Mini-SGLang Layer

- **Goal**: engine 之上补最小 serving 层, 让平台真的"可用"。
- **Scope**: OpenAI-compatible HTTP API, session management, prefix-sharing session reuse, structured output 接口预留。
- **Strategy**:
  - 这是"产品脸面", 不是锦上添花 (D-006)
  - 优先级讨论见 Q-002 (是否从 P2 tier 上浮到 P1 末尾)
  - 外层组织参考 mini-sglang, 但不照搬 CUDA 内核
- **Deliverables**:
  - [ ] `silica.server.openai_api`: FastAPI, `/v1/chat/completions`, `/v1/completions`
  - [ ] `silica.server.session.SessionManager`: 会话管理, 跨请求复用 prefix
  - [ ] structured generation / grammar 的接口位 (不实现)
  - [ ] `silica.llm.LLM`: Python 友好的 high-level 接口
- **Acceptance**:
  - [ ] 用 `openai` Python 客户端对 silica server 发请求能拿到流式返回
  - [ ] 跨请求的 prefix 复用可验证 (同一 session 里发 N 个共享前缀请求, prefix cache hit 率)
  - [ ] 本地可以像一个小型 serving engine 一样工作
- **Dependencies**: P-2 + P-3 (+ P-5 / P-6 optional)
- **Status**: planned
- **Notes**: 本阶段决定 Silica-MLX 从"一个 engine library"升格为"一个可用的 Mac 推理平台"。

---

## 8. Priority & Milestones

### 8.1 Priority Tiers

| Tier | Phases | 含义 |
| --- | --- | --- |
| P0 | P-0 .. P-4 | 骨架 + baseline engine + 目标模型 + bench |
| P1 | P-5 .. P-6 | VQ + flash 插件; 使大模型在 48GB 真正跑好 |
| P2 | P-7 .. P-8 | speculative + serving 层 |

Phase 8 优先级是否上浮见 Q-002。Phase 6 是否提前见 Q-003。

### 8.2 Milestones

| ID  | 名称                    | 依赖 Phases | 验收 |
| --- | ----------------------- | ----------- | --- |
| M-1 | Skeleton                | P-0          | 接口冻结, stub test 通过 |
| M-2 | Single-request gen      | P-1          | Qwen3-0.6B 能生成文本 |
| M-3 | Multi-request core      | P-2          | 8 并发 + prefix cache 命中 |
| M-4 | Big models running      | P-3          | 27B / 31B 稳定 500 token |
| M-5 | Unified bench           | P-4          | 一条命令出 baseline 表 |
| M-6 | VQ on platform          | P-5          | BlockTQ / RaBitQ 接入, 节省可量化 |
| M-7 | Streaming weights       | P-6          | 小内存预算下 27B 不 OOM |
| M-8 | Speculative enabled     | P-7          | 开关 correctness 不变 + 提速 |
| M-9 | Platform usable         | P-8          | OpenAI API + session 可用 |

---

## 9. Decisions Log

Append-only。新决策往后加，不改老条目。撤销/修订需新开一条并在新条目里引用被撤销的 ID。

### D-001 — Package manager & Python version

- **Date**: 2026-04-13
- **Status**: accepted
- **Decision**: `uv` + Python 3.12
- **Rationale**: uv 现代快; mini-sglang 也用 uv; Python 3.12 是 MLX / mlx-lm 稳定支持的版本。
- **Consequences**: 贡献者需要 uv; Python < 3.12 不支持。

### D-002 — vLLM core first, mini-sglang layer later

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: engine 核心按 vLLM 思路 (paged KV, continuous batching, memory budget); 外层服务按 mini-sglang 思路。
- **Rationale**: M5 Pro 48GB 上首要问题是"能跑稳 + 省内存 + 可扩展", vLLM 的 paged KV + batching 正好解决这个; mini-sglang 更适合拿来参考模块分层和 serving 外层。
- **Consequences**: Phase 0–4 像 mini-vLLM; Phase 8 像 mini-sglang。

### D-003 — KVCodec v0.1 excludes compressed-domain attention

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: v0.1 的 `KVCodec` 接口只含 `encode_block / decode_block / logical_bytes / resident_bytes`; 不包含 `attend()` 或其他压缩域 fast path。
- **Rationale**: 避免一开始就把接口绑死; v0.1 优先接口简洁, v0.2 再根据 Phase 5 bench 结果决定是否加。
- **Consequences**: BlockTQ / RaBitQ 的 Phase 5 实现必须走"decode 再标准 attention"路径, 即使性能次优。

### D-004 — Phase 1 model execution: wrap mlx-lm

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: Phase 1 的 `ModelAdapter` 是对 `mlx-lm` 的 thin wrapper, 不在 Phase 1 重写模型执行细节。
- **Rationale**: 先跑通再优化; Phase 3 做真正的 adapter 时再下沉。
- **Consequences**: Phase 3 重写时会有改动成本, 但 Phase 1 迭代周期缩短。

### D-005 — Phase 3 quantization path: MLX-native only

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: Phase 3 量化走 mlx-lm 现成的 4-bit / 8-bit 路径; **不**扩 GGUF / AWQ 多格式兼容。
- **Rationale**: 48GB 上 27B / 31B 必然要量化; 先用 MLX 已经能跑的方案, 多格式兼容推到 v0.2。
- **Consequences**: 用户不能用 GGUF / AWQ 预量化模型; 只能用 mlx-lm 支持的量化。

### D-006 — Platform as product, VQ as means

- **Date**: 2026-04-14
- **Status**: accepted (用户明确纠正过一次)
- **Decision**: Silica-MLX 本身是产品; VQ / weight streaming / speculative 是让平台跑好大模型的手段, 不是研究对象。
- **Rationale**: 用户目标是"单 Mac 芯片的推理平台, 能很好利用 VQ 之类 tech"; 不是"VQ 研究的 benchmark 载体"。
- **Consequences**:
  - 目标用户是 Mac 上想本地跑大模型的开发者
  - Phase 8 (serving 层) 重要性上升
  - VQ / flash 不应只是 opt-in flag, memory budgeter 要主动利用节省 (Principle 8)
  - 设计要 Apple unified memory first (Principle 2)

### D-007 — Plan document structure

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: 用 `docs/PLAN.md` 作为单一 source of truth; 结构为 Meta / TL;DR / Mission / Scope / Principles / Architecture / Interfaces / Phases / Priority / Decisions Log / Open Questions / Risks / References / Changelog, 所有条目带稳定 ID。
- **Rationale**: CRUD-friendly —— stable IDs, self-contained phase blocks, append-only decisions log; 允许在不读全文的情况下定位和修改某一条。
- **Consequences**: 未来 plan 变更要进 Decisions Log 并更新对应 Phase block; Changelog 跟版本号。

### D-008 — core / engine 边界: 数据类放 core, 逻辑类放 engine

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: 数据类 (`Request`, `RequestState`, `SamplingParams`, `Context` 等) 放 `silica.core`; 运行时逻辑类 (`Engine`, scheduler 内部状态机的 runner 部分) 放 `silica.engine`。遵循 mini-sglang 风格。
- **Rationale**: Phase 0 需要建立明确的目录结构; Q-004 里 Option A 是 mini-sglang 验证过的风格; 本文档所有 Phase block 也已隐式使用此路径 (如 `silica.core.request.Request`, `silica.core.sampling.SamplingParams`), 不定会导致 Phase 0 落盘时文档和代码不一致。
- **Consequences**:
  - `silica.core` 是数据 + 观测 (logging, profiling, metrics schema), 不含业务逻辑
  - `silica.engine.Engine` 持有 core 的数据类, 驱动 scheduler / kvcache / model
  - 解决 Q-004

### D-009 — MLX-native hot path as hard constraint

- **Date**: 2026-04-14
- **Status**: accepted (用户明确要求)
- **Decision**: Inference hot path **必须** 100% 走 MLX。具体约束:
  1. 所有 tensor 都是 `mlx.core.array` (`mx.array`)。`silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler` / `silica.vq` / `silica.flash` / `silica.speculative` 内部 **不允许** 出现 `torch.Tensor` 或 `numpy.ndarray` 参与 tensor math (numpy 仅可用于配置/标量/列表辅助)。
  2. **不依赖 PyTorch 运行时**。`pyproject.toml` 不得把 `torch` 列为 runtime dep。torch 只能作为 offline 权重转换的**可选** dev/extras 依赖 (例如 `pip install silica-mlx[convert]`); 转换结果是 MLX-native 格式, inference 启动后不再碰 torch。
  3. **vllm 和 transformers 只作为算法/架构参考, 不作为运行时依赖**。vllm 的 `csrc/`, `vllm_flash_attn/`, GPU/TPU/XPU/CPU model runner 一律不在范围内。
  4. Phase 1 wrap `mlx-lm` 是合法的 (D-004), 因为 **mlx-lm 本身就是 MLX-native**。禁止替换成任何 torch-based wrapper (包括 transformers、llama.cpp Python 绑定等) 作为运行时路径。
- **Rationale**: 用户硬要求 "必须是原生 mlx"; Silica-MLX 的整个价值命题 (D-006 "Mac 推理平台") 依赖于 MLX 在 Apple Silicon 上的性能和 unified memory 优势; 任何 torch hot path 都会破坏 Principle 2 (unified memory first)。
- **Consequences**:
  - 所有 attention / sampling / kvcache 内部实现必须用 `mx.` ops
  - Phase 3 如果某个目标模型在 mlx-lm 缺失, 需要在 `silica.models.*` 里 MLX-native 重写, **不能回退到 transformers**
  - vllm v1 的源代码仅用作"怎么设计"参考, 不用作"怎么调用"依赖 (见 5.4 Reference Map)
  - Benchmark 里如果要对照其他推理引擎的数字, 那个对照跑必须在独立 process 里, 不能混入 silica 运行时

---

## 10. Open Questions

Resolved 的 question 不删, 改 `Status: resolved` 并在条目末尾加 `Resolution:` 段, 方便追溯。

### Q-001 — VQ codec 自动选择 vs 显式配置

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: Phase 5 的 VQ codec 应该 (A) 用户显式选择 via CLI flag, 还是 (B) 按 workload 自动选择?
- **Context**: D-006 说平台要"很好地"利用 VQ, 暗示自动选择; 但 Phase 0 接口不表达这种能力。
- **Options**:
  - A. 显式配置: 简单, 用户控制
  - B. 自动选择: 符合 D-006 framing, 但需要 workload profiler
- **Blocks**: Phase 5 设计定稿
- **Next step**: Phase 4 bench 跑完后再判断。

### Q-002 — Phase 8 优先级是否上浮

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: 根据 D-006 (平台是产品), Phase 8 (OpenAI API + session) 是否应该从 P2 tier 上浮到 P1 末尾?
- **Context**: 如果 Phase 8 是"产品脸面", 应该早做; 但 engine 没稳定前做 serving 层风险大。
- **Options**:
  - A. 保持 P2: engine 先稳定
  - B. 上浮到 P1 末尾: 插件和 serving 层并行
- **Blocks**: Phase 5–8 实际排期
- **Next step**: Phase 4 完成后评估。

### Q-003 — Phase 6 是否提前

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: 如果 Phase 3 发现 Qwen3.5-27B / Gemma 4 31B 在 4-bit 量化下仍放不下 48GB, Phase 6 (weight streaming) 是否提前到 Phase 5 之前?
- **Blocks**: Phase 5 / 6 顺序
- **Next step**: Phase 3 跑出真实驻留数字后决策。

### Q-004 — `silica.core` vs `silica.engine` 边界

- **Raised**: 2026-04-14
- **Status**: resolved
- **Question**: `Request`, `SamplingParams`, `RequestState` 放 `silica.core` 还是 `silica.engine`?
- **Context**: mini-sglang 放 `minisgl.core`; 但 "core" 容易膨胀。
- **Options**:
  - A. 数据类放 core, 逻辑类放 engine (mini-sglang 风格)
  - B. 全放 engine, core 只留 logging/profiler
- **Resolution**: 选 Option A。见 D-008。文档内所有 Phase block 已隐式使用 `silica.core.request.*` 路径, 此处固化。

### Q-005 — MetricsRegistry 是否全局单例

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: Phase 0 的 profiler 用全局 `MetricsRegistry` 还是每个 Engine 实例一个?
- **Context**: 全局单例简单但多 Engine 实例会冲突; per-instance 干净但 CLI/bench 访问略麻烦。
- **Next step**: Phase 0 动手时定。

### Q-006 — Attention backend 是否独立接口

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: 是否需要一个独立的 `AttentionBackend` Protocol (类似 vllm v1 的 `vllm/v1/attention/backend.py`), 让 attention 实现可以在 `ModelAdapter` 之外被替换? 还是继续把 attention 藏在 `ModelAdapter.build()` 返回的 Module 内部?
- **Context**: vllm v1 把 attention backend 独立出来是为了支持 flashattention / flashinfer / xformers / triton 多种 CUDA 实现; 我们只有 MLX 一条路径, 短期不需要这种灵活性。但如果 Phase 5 的 VQ 需要压缩域 attention fast path (D-003 留的 v0.2 能力), 独立 `AttentionBackend` 会让接入更干净。
- **Options**:
  - A. v0.1 不独立, attention 藏在 ModelAdapter 内部 (简单, 契合当前 5 接口设计)
  - B. v0.1 就独立 AttentionBackend 作为第 6 个接口 (为 v0.2 的压缩域 attention 做准备)
- **Blocks**: 无 —— 可延到 Phase 5 bench 出结果后再定
- **Next step**: Phase 5 完成后评估 (与 D-003 的 v0.2 升级一起决定)

---

## 11. Risks

| ID | 描述 | 触发阶段 | 缓解 |
| --- | --- | --- | --- |
| R-1 | Qwen3.5-27B / Gemma4-31B 在 4-bit 量化下仍放不下 48GB | P-3 | Q-003: 提前 P-6 weight streaming |
| R-2 | mlx-lm 的 wrapper 在 Phase 3 替换成本高于预期 | P-3 | D-004 接受的权衡; 若过高可延后下沉 |
| R-3 | Qwen3.5 hybrid attention cache routing 语义复杂 | P-3 | Phase 3 单元测试覆盖; 对照 HF 参考实现 |
| R-4 | BlockTQ / RaBitQ 的 decode 开销比 fp16 attention 本身大 | P-5 | Q-001 自动选择策略; v0.2 考虑 fast path |
| R-5 | MLX / mlx-lm 版本不稳定导致依赖锁麻烦 | all | pyproject.toml 锁最低版本 + CI 定期升级 |

---

## 12. References

### 12.1 External (参考)

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

本地放在 repo 根的参考实现，**仅作为算法/架构参考, 非运行时依赖** (D-009):

| 目录 | 项目 | 主要参考点 |
| --- | --- | --- |
| `vllm/` | vLLM v1 | 见 §5.4 Reference Map; 重点 `vllm/v1/core/`, `vllm/v1/engine/`, `vllm/v1/kv_cache_interface.py`, `vllm/v1/request.py` |
| `mini-sglang/` | Mini-SGLang | 模块分层、serving 外层、radix prefix cache 思路 |
| `turboquant_plus/` | TurboQuant+ | Phase 5 VQ codec (BlockTQ / RaBitQ) 实现参考 |

---

## 13. Changelog

- **v1.1.2** (2026-04-14): 用户拉下 vllm 作为本地 reference, 明确 "必须是原生 mlx" —— 新增 D-009 固化 MLX-native hot path 硬约束 (禁 torch.Tensor / 禁 PyTorch runtime dep); Principle 6 重写; 3.2 Non-Goals 加 PyTorch / CUDA 等后端排除项; 新增 §5.4 Reference Map to vLLM v1 (明确参考哪些 / 不参考哪些); 新增 Q-006 讨论是否独立 AttentionBackend; 12 References 拆分 External / Local checkouts, 登记 `vllm/` 本地路径。
- **v1.1.1** (2026-04-14): 修 CRUD review 两处 —— 5.1 目录树补 `llm/` 子包 (和 P-8 deliverable 对齐); Q-004 标 resolved 并新增 D-008 固化 core/engine 边界。
- **v1.1** (2026-04-14): 重写 Mission 与 Principle 1 —— 平台是产品, VQ 是手段 (D-006); 加 Principle 2 "Apple unified memory first"; 加 Principle 8 "savings must be observable"; Phase 5 / 8 的 framing 重写; 新开 Q-002 讨论 Phase 8 优先级, Q-005 讨论 MetricsRegistry; 新增 Risks 表; 结构全部按 CRUD-friendly 格式组织 (stable IDs P-N / D-NNN / Q-NNN / M-N / I-N, self-contained blocks, append-only logs)。
- **v1.0** (2026-04-14): 基于 Codex 对 v0 的润色; 添加 Phase 的统一结构。
- **v0** (2026-04-13): 初版 plan 草稿 (对话形式, 未落盘)。
