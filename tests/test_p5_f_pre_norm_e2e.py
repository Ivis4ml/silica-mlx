"""P-5-F F.2b end-to-end discriminator — production hot path bit-equivalence.

Runs the same prefix-hit admission scenario through two engines on
fresh Qwen3-0.6B instances:

- **Subject**: ``RadixPrefixCache(store=SyntheticPrefixBlockStore(
  pre_norm=True, codec=IdentityCodec(...)))`` — the F.2b production
  pre-norm path. K_pre is captured at every prefill forward by the
  ``PreNormCaptureAdapter`` proxy installed in the adapter, persisted
  in the prefix store as pre-k_norm K, then reconstructed via
  ``apply_k_norm_then_rope`` on hit-path admit before seeding the
  live cache.
- **Oracle**: ``RadixPrefixCache(store=SyntheticPrefixBlockStore())``
  — the legacy post-RoPE store (no codec, no pre-norm). K is sliced
  from the live cache and re-seeded directly.

Under IdentityCodec the two paths must produce a bit-identical token
stream: the subject's K_pre → store → IdentityCodec round-trip →
``apply_k_norm_then_rope`` reconstructs exactly the K_post the oracle
stored directly. Any divergence indicates a structural issue in the
F.2b capture / split / reconstruction wiring.

HF-cache-skip-gated on ``Qwen/Qwen3-0.6B`` (the same skip pattern
``tests/test_p2_preload_parity.py`` uses).
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
from silica.models.qwen3 import Qwen3Adapter

_REPO = "Qwen/Qwen3-0.6B"
_QWEN3_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-0.6B"
)
_SKIP = not _QWEN3_CACHE.exists() or bool(
    os.environ.get("SILICA_SKIP_MODEL_TESTS")
)
_SKIP_REASON = (
    "Qwen3-0.6B not cached at "
    f"{_QWEN3_CACHE}; run scripts/probe_p2_preload.py to populate it."
)

# The two prompts share a prefix that tokenizes to >= 1 block (16
# tokens at block_size=16) so the second prompt's admission reaches
# the F.2b hit-path apply_k_norm_then_rope branch.
_BLOCK_SIZE = 16
_SHARED_PREFIX = (
    "The recurrent associative memory accumulates across every "
    "processed token in this sequence."
)
_PROMPT_A = _SHARED_PREFIX + " First request continues here."
_PROMPT_B = _SHARED_PREFIX + " Second request reuses the prefix."
_MAX_TOKENS = 8


def _collect_tokens(events: object) -> list[int]:
    return [e.token_id for e in events if e.kind == "token"]  # type: ignore[attr-defined]


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_pre_norm_path_matches_post_rope_path_under_identity_codec() -> None:
    """End-to-end: F.2b pre_norm=True + IdentityCodec === legacy post-RoPE.

    Two adapter instances are required because each adapter installs
    its capture proxy at construction time; sharing a single model
    between engines would route both engines' K_proj through the same
    proxy and conflate their capture buffers.
    """
    # --- Subject: pre-norm store + IdentityCodec ---
    subject_adapter, subject_kv = Qwen3Adapter.from_hf_repo(_REPO)
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
    oracle_adapter, oracle_kv = Qwen3Adapter.from_hf_repo(_REPO)
    oracle_pc = RadixPrefixCache(
        block_size=_BLOCK_SIZE,
        store=SyntheticPrefixBlockStore(block_size=_BLOCK_SIZE),
    )
    oracle_engine = Engine(oracle_adapter, oracle_kv)

    params = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)

    # Prompt A populates each engine's prefix cache; prompt B should
    # hit the shared prefix (≥ 1 block) and exercise the F.2b
    # apply_k_norm_then_rope branch.
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
        "Prompt A token streams diverged between F.2b pre-norm path "
        f"and legacy post-RoPE path. Subject: {subject_a}; "
        f"oracle: {oracle_a}."
    )
    assert subject_b == oracle_b, (
        "Prompt B token streams (prefix-hit admit) diverged between "
        "F.2b pre-norm path and legacy post-RoPE path. Subject: "
        f"{subject_b}; oracle: {oracle_b}. The F.2b capture / "
        "reconstruction wiring is not bit-equivalent to the legacy "
        "post-RoPE path under IdentityCodec."
    )


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
def test_pre_norm_path_with_no_prefix_hit_runs_clean() -> None:
    """Sanity: a single request through the F.2b pre-norm path runs
    end-to-end without raising and produces a non-empty token stream.
    Validates the contiguous-prefill capture branch
    (``_prefill_phase`` non-slice + ``_admit_miss_cohort`` non-slice)
    does not corrupt the in-flight forward.
    """
    adapter, kv = Qwen3Adapter.from_hf_repo(_REPO)
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
