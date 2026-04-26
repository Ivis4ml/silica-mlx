"""P-5-F slice-regime + pre_norm=True end-to-end on hybrid Qwen3.5-0.8B.

Companion to ``tests/test_p5_f_pre_norm_e2e.py`` — that file pins the
F.2b discriminator on Qwen3-0.6B (pure GQA attention, non-slice
prefill paths). This file extends the same IdentityCodec
bit-equivalence pattern to Qwen3.5-0.8B (hybrid DeltaNet + full
attention) so the slice-regime helpers exercised under
``RecurrentStateAdapter + prefix_cache is not None`` are validated
end-to-end:

- ``_slice_prefill_with_capture`` (B=1) — the per-chunk arm / forward /
  disarm / split lifecycle that F.2b added to the existing slice loop.
- ``_split_capture_into_row_kpre`` — per-row block-aligned slicing
  with hybrid attention-layer indices (DeltaNet layers excluded).
- ``_admit_single_hit_row`` slice-regime branch — capture during the
  suffix prefill goes through ``_slice_prefill_with_capture``, while
  the fetch side (``apply_k_norm_then_rope``) runs over the
  attention-layer subset.

Subject and oracle both run with the same hybrid Qwen3.5-0.8B adapter
shape; the only difference is the prefix-cache store's ``pre_norm``
flag and the codec on the codec side. Under IdentityCodec the two
paths must produce a bit-identical token stream.

HF-cache-skip-gated on ``Qwen/Qwen3.5-0.8B`` (mirrors the skip pattern
in ``tests/test_qwen3_5_recurrent_snapshot.py`` and
``tests/test_batcher_hit_admission_byte_exact_oracle.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from silica.core.sampling import SamplingParams
from silica.engine import Engine
from silica.kvcache.codec import IdentityCodec
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.qwen3_5 import Qwen3_5Adapter

_REPO = "Qwen/Qwen3.5-0.8B"
_QWEN35_CACHE = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / f"models--{_REPO.replace('/', '--')}"
)
_SKIP = not _QWEN35_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_SKIP_REASON = (
    f"Qwen3.5-0.8B not cached at {_QWEN35_CACHE}; populate the HF "
    "cache before running this test."
)

# Block size 16 matches every other prefix-cache test in the suite
# (default ``RadixPrefixCache`` block_size). The shared prefix needs
# to tokenize to >= 1 block under the Qwen3.5 tokenizer for prompt B
# to land on the F.2b prefix-hit admit path; the long sentence below
# clears that bar with margin under both Qwen3 and Qwen3.5
# tokenisers (the Qwen3-0.6B sibling test uses the same prefix and
# observed an aligned hit; Qwen3.5 shares the BPE).
_BLOCK_SIZE = 16
_SHARED_PREFIX = (
    "The recurrent associative memory accumulates across every "
    "processed token in this sequence, advancing the cache state "
    "without resetting until the request completes."
)
_PROMPT_A = _SHARED_PREFIX + " First request continues here."
_PROMPT_B = _SHARED_PREFIX + " Second request reuses the prefix."
_MAX_TOKENS = 8


def _collect_tokens(events: object) -> list[int]:
    return [e.token_id for e in events if e.kind == "token"]  # type: ignore[attr-defined]


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_pre_norm_path_matches_post_rope_path_on_hybrid_under_identity_codec() -> None:
    """End-to-end on Qwen3.5-0.8B: F.2b pre_norm=True + IdentityCodec
    is bit-equivalent to the legacy post-RoPE path under the slice-
    prefill regime active on hybrid adapters with a prefix cache.

    Each engine constructs its own adapter so the per-adapter
    K_pre capture proxy installs are isolated. Both engines have
    ``RecurrentStateAdapter + prefix_cache is not None``, so
    ``_slice_prefill_active()`` returns True and prefill /
    suffix-admit forwards route through
    ``_slice_prefill_with_capture``.
    """
    # --- Subject: pre-norm store + IdentityCodec ---
    subject_adapter, subject_kv = Qwen3_5Adapter.from_hf_repo(_REPO)
    subject_layout = subject_adapter.kv_layout()
    subject_store = SyntheticPrefixBlockStore(
        block_size=_BLOCK_SIZE,
        codec=IdentityCodec(
            block_size=_BLOCK_SIZE,
            n_kv_heads=subject_layout.n_kv_heads,
            head_dim=subject_layout.head_dim,
        ),
        pre_norm=True,
    )
    subject_pc = RadixPrefixCache(
        block_size=_BLOCK_SIZE, store=subject_store
    )
    subject_engine = Engine(subject_adapter, subject_kv)

    # --- Oracle: legacy post-RoPE store, no codec ---
    oracle_adapter, oracle_kv = Qwen3_5Adapter.from_hf_repo(_REPO)
    oracle_pc = RadixPrefixCache(
        block_size=_BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=_BLOCK_SIZE),
    )
    oracle_engine = Engine(oracle_adapter, oracle_kv)

    params = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)

    # Prompt A populates each engine's prefix cache via slice-regime
    # miss-prefill; prompt B hits the shared prefix and exercises the
    # F.2b apply_k_norm_then_rope reconstruction on the seeded
    # cache + the slice-regime suffix forward.
    subject_a = _collect_tokens(
        list(
            subject_engine.generate_batch(
                [_PROMPT_A], params, prefix_cache=subject_pc
            )
        )
    )
    subject_b = _collect_tokens(
        list(
            subject_engine.generate_batch(
                [_PROMPT_B], params, prefix_cache=subject_pc
            )
        )
    )
    oracle_a = _collect_tokens(
        list(
            oracle_engine.generate_batch(
                [_PROMPT_A], params, prefix_cache=oracle_pc
            )
        )
    )
    oracle_b = _collect_tokens(
        list(
            oracle_engine.generate_batch(
                [_PROMPT_B], params, prefix_cache=oracle_pc
            )
        )
    )

    assert subject_a == oracle_a, (
        "Hybrid Qwen3.5-0.8B prompt A token streams diverged between "
        f"F.2b pre-norm path and legacy post-RoPE path. Subject: "
        f"{subject_a}; oracle: {oracle_a}. Slice-regime "
        "_slice_prefill_with_capture's per-chunk arm/disarm + "
        "_split_capture_into_row_kpre is not bit-equivalent to the "
        "legacy slice path under IdentityCodec."
    )
    assert subject_b == oracle_b, (
        "Hybrid Qwen3.5-0.8B prompt B token streams (slice-regime "
        "prefix-hit admit) diverged between F.2b pre-norm path and "
        f"legacy post-RoPE path. Subject: {subject_b}; oracle: "
        f"{oracle_b}. The F.2b apply_k_norm_then_rope "
        "reconstruction on hybrid attention-layer indices, paired "
        "with slice-regime suffix capture, is not bit-equivalent "
        "to the legacy slice path under IdentityCodec."
    )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_pre_norm_hybrid_path_with_no_prefix_hit_runs_clean() -> None:
    """Sanity: a single Qwen3.5-0.8B request through the F.2b pre-norm
    + slice-regime path runs end-to-end without raising and produces
    a non-empty token stream. Validates the slice-regime miss-prefill
    capture branch (``_slice_prefill_with_capture`` per-chunk
    arm/disarm) does not corrupt the in-flight forward on a hybrid
    DeltaNet + GQA stack.
    """
    adapter, kv = Qwen3_5Adapter.from_hf_repo(_REPO)
    layout = adapter.kv_layout()
    store = SyntheticPrefixBlockStore(
        block_size=_BLOCK_SIZE,
        codec=IdentityCodec(
            block_size=_BLOCK_SIZE,
            n_kv_heads=layout.n_kv_heads,
            head_dim=layout.head_dim,
        ),
        pre_norm=True,
    )
    pc = RadixPrefixCache(block_size=_BLOCK_SIZE, store=store)
    engine = Engine(adapter, kv)

    params = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)
    tokens = _collect_tokens(
        list(engine.generate_batch([_PROMPT_A], params, prefix_cache=pc))
    )
    assert len(tokens) == _MAX_TOKENS, (
        f"expected {_MAX_TOKENS} tokens, got {len(tokens)}: {tokens}"
    )
