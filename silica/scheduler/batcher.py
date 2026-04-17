"""silica.scheduler.batcher — ContinuousBatcher step loop (Unit 16a).

P-2 Opening v2.3 fixes the shape this class will take over Units 16a-d:

  - **16a** (this file): B=1 scaffolding. Build the step-loop phases
    (admission / prefill / decode / sample / finalize), the BatchEvent
    stream, the capability gate, and the slot_table + BatchKVCache
    wiring. Output must be token-identical to P-1's ``Engine.generate``
    when driven with one request.
  - **16b**: B>1 mid-cohort admission (all rows admitted at step 0);
    left-padding across prompts of different lengths.
  - **16c**: mid-run admission via ``BatchKVCache.extend`` + prefix
    cache integration (``RadixPrefixCache.lookup`` + copy source K/V
    into the new row).
  - **16d**: budget-aware preemption via ``BatchKVCache.filter`` +
    ``RequestState.transition(PREEMPTED)`` + re-admission cycle; abort
    with ``BatchEvent.aborted`` on ``RejectDecision``.

16a wilfully restricts ``max_batch_size`` to 1. Adding a second request
raises ``NotImplementedError("B > 1 arrives in Unit 16b")`` by design —
the restriction is what lets 16a stand as a P-1 oracle-parity layer.

Capability gate (Layer 3 of the v2.3 three-layer stack): the batcher
rejects adapters whose ``attention_pattern`` contains any kind other
than ``AttentionKind.GLOBAL``. This means:
  - Plain Qwen3 family: accepted (all GLOBAL).
  - Qwen3.5 hybrid family: rejected (contains ``HYBRID_DELTANET``).
    Revisit in P-3 via ``BatchRecurrentStateStore``.
  - Sliding-window families: rejected until Q-013 probe.
  - Attention-free / Mamba: rejected until a dedicated scheduler lands.

Scope note: 16a uses ``BatchKVCache`` even at B=1 to keep the batched
path load-bearing from day one. Using ``SimpleKVCache`` at B=1 and
swapping later would hide shape bugs; building on the final cache
shape now surfaces them immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache

from silica.core.events import BatchEvent
from silica.core.request import Request, RequestState, RequestStatus
from silica.core.sampler import Sampler
from silica.core.sampling import SamplingParams
from silica.mlx.runner import forward_batched
from silica.models.adapter import AttentionKind, ModelAdapter
from silica.weights.provider import WeightProvider
from silica.weights.resident import ResidentWeightProvider


@dataclass
class _BatchRow:
    """Per-row state held by the batcher (internal).

    One row = one request. The row persists from admission through
    DECODE / DONE / ABORTED / PREEMPTED. In 16a there is at most one.
    """

    req_index: int
    req_id: str
    prompt_ids: list[int]
    params: SamplingParams
    state: RequestState
    generated: list[int] = field(default_factory=list)


class ContinuousBatcher:
    """P-2 scheduler step loop — Unit 16a B=1 scaffolding."""

    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        sampler: Sampler | None = None,
        weight_provider: WeightProvider | None = None,
        max_batch_size: int = 1,
    ) -> None:
        if max_batch_size != 1:
            raise NotImplementedError(
                f"ContinuousBatcher 16a supports max_batch_size=1 only, "
                f"got {max_batch_size}. B > 1 arrives in Unit 16b."
            )
        self._enforce_capability_gate(adapter)
        self._adapter = adapter
        self._sampler = sampler or Sampler()
        self._model = adapter.build(weight_provider or ResidentWeightProvider())
        self._max_batch_size = max_batch_size
        self._rows: list[_BatchRow] = []
        # BatchKVCache is allocated lazily on first admission (we need the
        # adapter's layer count — known at adapter construction — but we
        # defer the allocation so an unused batcher holds no KV memory).
        self._batch_cache: list[BatchKVCache] | None = None

    # --- public admission / stepping surface ---

    def add_request(
        self,
        req_index: int,
        prompt_ids: list[int],
        params: SamplingParams,
    ) -> None:
        """Enqueue a new request for the current batch.

        Raises ``NotImplementedError`` once the batch hits
        ``max_batch_size`` (16a: 1). P-2 Unit 16b relaxes this.
        """
        if len(self._rows) >= self._max_batch_size:
            raise NotImplementedError(
                f"ContinuousBatcher max_batch_size={self._max_batch_size} "
                f"reached; B > 1 arrives in Unit 16b."
            )
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        request = Request(
            prompt="",  # text not carried through the batcher
            sampling_params=params,
            request_id=f"req-{req_index}",
            token_ids=tuple(prompt_ids),
        )
        state = RequestState(request=request)
        row = _BatchRow(
            req_index=req_index,
            req_id=f"req-{req_index}",
            prompt_ids=list(prompt_ids),
            params=params,
            state=state,
        )
        self._rows.append(row)

    def has_active(self) -> bool:
        """True iff at least one row is not in a terminal state."""
        return any(not r.state.is_terminal for r in self._rows)

    def step(self) -> list[BatchEvent]:
        """Advance every non-terminal row by one scheduler iteration.

        Per-row behaviour:
          WAITING  → admit → PREFILL.
          PREFILL  → one batched forward over the prompt; sample the
                     first token; emit ``BatchEvent.token``; transition
                     to DECODE (or DONE if a stop token was hit first).
          DECODE   → one batched forward on the last generated token;
                     sample; emit token; transition DONE on stop /
                     max_tokens.
          terminal → skipped.

        A row at most progresses one state per step, so 16a-oracle
        parity with P-1's ``Engine.generate`` is obvious: each step
        yields exactly one token per active row.
        """
        events: list[BatchEvent] = []
        for row in self._rows:
            if row.state.is_terminal:
                continue
            if row.state.status == RequestStatus.WAITING:
                self._admit_row(row)
            if row.state.status == RequestStatus.PREFILL:
                events.extend(self._prefill_row(row))
            elif row.state.status == RequestStatus.DECODE:
                events.extend(self._decode_row(row))
        return events

    # --- phase methods (step() calls these in order) ---

    def _admit_row(self, row: _BatchRow) -> None:
        """WAITING → PREFILL; lazily allocate the batched cache."""
        row.state.transition(RequestStatus.PREFILL, reason="admit")
        if self._batch_cache is None:
            num_layers = self._adapter.config.num_layers
            self._batch_cache = [
                BatchKVCache(left_padding=[0]) for _ in range(num_layers)
            ]

    def _prefill_row(self, row: _BatchRow) -> list[BatchEvent]:
        """Run prefill forward, sample first token, transition out."""
        assert self._batch_cache is not None
        prompt_arr = mx.array(row.prompt_ids, dtype=mx.int32)
        tokens_2d = prompt_arr[None]  # (1, T)
        logits = forward_batched(
            self._model, tokens_2d, list(self._batch_cache)
        )  # (1, V)
        return self._sample_and_emit(row, logits, is_prefill=True)

    def _decode_row(self, row: _BatchRow) -> list[BatchEvent]:
        """Run decode forward on the last sampled token; sample next."""
        assert self._batch_cache is not None
        last_tok = row.generated[-1]
        step_in = mx.array([[last_tok]], dtype=mx.int32)  # (1, 1)
        logits = forward_batched(
            self._model, step_in, list(self._batch_cache)
        )  # (1, V)
        return self._sample_and_emit(row, logits, is_prefill=False)

    def _sample_and_emit(
        self,
        row: _BatchRow,
        batched_logits: mx.array,
        *,
        is_prefill: bool,
    ) -> list[BatchEvent]:
        """Shared sample + emit + state-transition tail for prefill and decode."""
        row_logits = batched_logits[0]  # (V,)
        history_ids = list(row.prompt_ids) + list(row.generated)
        history = mx.array(history_ids, dtype=mx.int32)
        token_scalar = self._sampler.sample(row_logits, history, row.params)
        tok = int(token_scalar.item())
        row.generated.append(tok)

        events: list[BatchEvent] = [BatchEvent.token(row.req_index, tok)]

        if tok in row.params.stop_token_ids:
            row.state.transition(RequestStatus.DONE, reason="stop_token")
            events.append(BatchEvent.done(row.req_index, "stop_token"))
        elif len(row.generated) >= row.params.max_tokens:
            row.state.transition(RequestStatus.DONE, reason="max_tokens")
            events.append(BatchEvent.done(row.req_index, "max_tokens"))
        elif is_prefill:
            row.state.transition(RequestStatus.DECODE, reason="prefill_done")
        # Plain DECODE continuation: no transition; next step runs decode.
        return events

    # --- capability gate ---

    @staticmethod
    def _enforce_capability_gate(adapter: ModelAdapter) -> None:
        """P-2 batcher only accepts adapters whose every layer is GLOBAL.

        Layer 3 of the v2.3 three-layer stack. The batcher does not
        test ``isinstance(adapter, ...)``; it asks the adapter what
        ``AttentionPattern`` it claims and decides from that alone.
        Mamba / DeltaNet / sliding-window families will each need their
        own scheduler (or an extended capability matrix) — this gate
        simply refuses to guess.
        """
        pattern = adapter.attention_pattern()
        for layer_idx, kind in enumerate(pattern.per_layer):
            if kind == AttentionKind.GLOBAL:
                continue
            reason = _unsupported_kind_reason(kind)
            raise NotImplementedError(
                f"ContinuousBatcher accepts AttentionKind.GLOBAL only in P-2; "
                f"adapter layer {layer_idx} is {kind.value!r} — {reason}"
            )


def _unsupported_kind_reason(kind: AttentionKind) -> str:
    """Short explanation used by the capability-gate error message."""
    if kind == AttentionKind.HYBRID_DELTANET:
        return (
            "hybrid DeltaNet batching arrives in P-3 via "
            "BatchRecurrentStateStore on the adapter side"
        )
    if kind == AttentionKind.RECURRENT:
        return (
            "pure-recurrent batching arrives in P-3 via "
            "BatchRecurrentStateStore"
        )
    if kind == AttentionKind.SLIDING:
        return (
            "sliding-window batching is gated on the P-2 Q-013 "
            "BatchRotatingKVCache probe"
        )
    if kind == AttentionKind.HYBRID:
        return "sliding/global hybrid needs the Q-013 probe first"
    return "unknown AttentionKind; no scheduler mapping defined"
