"""P-3-C5.3.3b — real-model byte-exact-vs-slice-oracle gate.

Drives the C5.3.3a wiring through Qwen3.5-0.8B and asserts that a
prefix-hit admission's post-restore recurrent state byte-equals an
oracle batcher that runs the same trajectory regime (slice-prefill)
but doesn't see the prefix cache. Same model, two independent
batchers + caches.

Acceptance per ``docs/P3_C5_3_DESIGN.md`` §4.4:

- **Subject**: ``prefix_cache=RadixPrefixCache(block_size=B)`` +
  ``_allow_recurrent_prefix_cache_for_c5_3_testing=True``. Slice
  regime active via the production predicate's
  ``prefix_cache is not None`` clause. Processes request 1 to seed
  the radix tree, then request 2 (mid-run admission) hits.
- **Oracle**: ``prefix_cache=None`` +
  ``_force_recurrent_slice_prefill_for_c5_3_oracle=B`` (same
  block_size, regime parity). Slice regime active via the test-only
  predicate clause. Processes request 2 (pre-step admission) via
  the slice-regime miss path.
- **Comparison**: snapshot LIVE caches via
  ``adapter.snapshot_recurrent_state(batcher._batch_cache, 0)``
  and assert ``mx.array_equal`` on every DeltaNet layer's
  ``conv_state`` and ``recurrent_state``. Tree-stored snapshots
  are NOT compared — re-snapshotting the live cache is what
  validates the post-restore + post-suffix-prefill state.
- **Token-stream sanity**: subject and oracle's first sampled
  token must match (greedy bf16, identical prompts, identical
  regime → identical argmax).

HF-cache-skip-gated on ``Qwen/Qwen3.5-0.8B``; mirrors
``tests/test_qwen3_5_recurrent_snapshot.py``'s skip pattern.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.qwen3_5 import Qwen3_5Adapter
from silica.scheduler.batcher import ContinuousBatcher

_REPO = "Qwen/Qwen3.5-0.8B"


def _hf_cache_has_repo(repo: str) -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    cache_dir = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}"
    return cache_dir.exists()


_SKIP = not _hf_cache_has_repo(_REPO)
_SKIP_REASON = (
    f"Qwen3.5-0.8B ({_REPO}) not present in the local HF cache. "
    f"Run qwen3.5-0.8b-b1-parity once to populate, or export HF_HOME."
)


# Long enough to tokenize to ≥ 33 tokens under the Qwen3.5
# tokenizer (test asserts this below; 33 = 2*16+1 ensures
# usable=32 → 2-block hit + 1-token suffix at block_size=16).
_PROMPT_TEXT = (
    "The recurrent associative memory accumulates across every "
    "processed token in this sequence. Each forward pass advances "
    "the cache state without resetting until the request completes. "
    "Slice-prefill regime captures snapshots at fixed block "
    "boundaries to support prefix-cache cooperation under hybrid "
    "DeltaNet attention."
)
BLOCK_SIZE = 16


@pytest.fixture(scope="module")
def adapter_and_model() -> tuple[Qwen3_5Adapter, Any]:
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    from silica.models.factory import adapter_for_repo

    adapter, _kv = adapter_for_repo(_REPO)
    assert isinstance(adapter, Qwen3_5Adapter)
    return adapter, adapter._model


def _make_subject(adapter: Qwen3_5Adapter) -> ContinuousBatcher:
    pc = RadixPrefixCache(
        block_size=BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=BLOCK_SIZE),
    )
    return ContinuousBatcher(
        adapter,
        prefix_cache=pc,
        _allow_recurrent_prefix_cache_for_c5_3_testing=True,
    )


def _make_oracle(adapter: Qwen3_5Adapter) -> ContinuousBatcher:
    return ContinuousBatcher(
        adapter,
        prefix_cache=None,
        _force_recurrent_slice_prefill_for_c5_3_oracle=BLOCK_SIZE,
    )


def _params(max_tokens: int) -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=max_tokens)


def _token_events(events: list[Any]) -> list[int]:
    return [e.token_id for e in events if e.kind == "token"]


def _assert_recurrent_byte_exact(
    adapter: Qwen3_5Adapter,
    subject_cache: list[Any],
    oracle_cache: list[Any],
) -> None:
    """Snapshot both live caches and assert per-layer byte equality
    on conv_state and recurrent_state.

    Failure mode reporting names the offending layer + slot so a
    real-model regression points at the specific divergence rather
    than just "cache mismatch".
    """
    subject_snap = adapter.snapshot_recurrent_state(subject_cache, 0)
    oracle_snap = adapter.snapshot_recurrent_state(oracle_cache, 0)
    assert len(subject_snap.entries) == len(oracle_snap.entries), (
        f"DeltaNet layer count mismatch: subject "
        f"{len(subject_snap.entries)} vs oracle "
        f"{len(oracle_snap.entries)}"
    )
    for s_entry, o_entry in zip(
        subject_snap.entries, oracle_snap.entries
    ):
        assert s_entry.layer_idx == o_entry.layer_idx, (
            f"layer index mismatch: subject {s_entry.layer_idx} vs "
            f"oracle {o_entry.layer_idx}"
        )
        # conv_state
        if s_entry.conv_state is None:
            assert o_entry.conv_state is None, (
                f"layer {s_entry.layer_idx}: subject conv_state None "
                f"but oracle non-None"
            )
        else:
            assert o_entry.conv_state is not None, (
                f"layer {s_entry.layer_idx}: subject conv_state "
                f"non-None but oracle None"
            )
            assert mx.array_equal(
                s_entry.conv_state, o_entry.conv_state
            ), (
                f"layer {s_entry.layer_idx}: conv_state byte-exact "
                f"mismatch — slice-regime trajectory diverged"
            )
        # recurrent_state
        if s_entry.recurrent_state is None:
            assert o_entry.recurrent_state is None, (
                f"layer {s_entry.layer_idx}: subject recurrent_state "
                f"None but oracle non-None"
            )
        else:
            assert o_entry.recurrent_state is not None, (
                f"layer {s_entry.layer_idx}: subject recurrent_state "
                f"non-None but oracle None"
            )
            assert mx.array_equal(
                s_entry.recurrent_state, o_entry.recurrent_state
            ), (
                f"layer {s_entry.layer_idx}: recurrent_state "
                f"byte-exact mismatch — slice-regime trajectory "
                f"diverged"
            )


def test_subject_oracle_byte_exact_at_recurrent_layers(
    adapter_and_model: tuple[Qwen3_5Adapter, Any],
) -> None:
    adapter, _model = adapter_and_model

    prompt = adapter.tokenizer().encode(_PROMPT_TEXT)
    assert len(prompt) >= 2 * BLOCK_SIZE + 1, (
        f"_PROMPT_TEXT tokenized to {len(prompt)} tokens; need "
        f">= {2 * BLOCK_SIZE + 1} for a 2-block hit + 1-token suffix"
    )

    subject = _make_subject(adapter)
    oracle = _make_oracle(adapter)

    # Both batchers must agree on slice-regime activation.
    assert subject._slice_prefill_active() is True
    assert oracle._slice_prefill_active() is True
    assert subject._effective_slice_block_size() == BLOCK_SIZE
    assert oracle._effective_slice_block_size() == BLOCK_SIZE

    # === Subject: process request 1 to seed the radix tree ===
    # max_tokens=2 → 3 steps (prefill, decode, reclaim).
    subject.add_request(0, prompt, _params(max_tokens=2))
    subject.step()  # prefill (slice helper, captures at 0, 1)
    subject.step()  # decode (max_tokens reached → DONE)
    subject.step()  # reclaim → extract → tree gets snapshots

    assert subject._prefix_cache is not None
    assert subject._prefix_cache.node_count() >= 2, (
        f"request 1's reclaim should have inserted at least 2 "
        f"radix nodes; got {subject._prefix_cache.node_count()}"
    )

    # === Subject: request 2 admits mid-run via the hit path ===
    subject.add_request(1, prompt, _params(max_tokens=1))
    subject_events = subject.step()  # admit via Phase-B + hit

    assert subject.prefix_hits == 1, (
        f"request 2 must hit the radix tree; "
        f"prefix_hits={subject.prefix_hits}"
    )
    # max_tokens=1 → admit's sample step transitions DONE; the row
    # is still in subject._rows for the snapshot below.
    assert subject._batch_cache is not None
    assert len(subject._rows) == 1

    # === Oracle: request 2 admits pre-step via slice-regime miss ===
    oracle.add_request(1, prompt, _params(max_tokens=1))
    oracle_events = oracle.step()  # prefill (slice helper)

    assert oracle._batch_cache is not None
    assert len(oracle._rows) == 1
    assert oracle.prefix_hits == 0, (
        "oracle has no prefix_cache; prefix_hits must stay 0"
    )

    # === Byte-exact recurrent slot comparison (the real gate) ===
    _assert_recurrent_byte_exact(
        adapter, subject._batch_cache, oracle._batch_cache
    )

    # === Token-stream sanity (greedy argmax, same prompt, same regime) ===
    subject_tokens = _token_events(subject_events)
    oracle_tokens = _token_events(oracle_events)
    assert len(subject_tokens) == 1, (
        f"subject emitted {len(subject_tokens)} tokens, expected 1"
    )
    assert len(oracle_tokens) == 1, (
        f"oracle emitted {len(oracle_tokens)} tokens, expected 1"
    )
    assert subject_tokens[0] == oracle_tokens[0], (
        f"first-token mismatch: subject {subject_tokens[0]} vs "
        f"oracle {oracle_tokens[0]} — slice-regime greedy argmax "
        f"should agree under byte-exact recurrent state"
    )
