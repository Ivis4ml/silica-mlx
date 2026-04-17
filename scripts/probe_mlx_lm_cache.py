"""Day-1 Gate A probe — mlx-lm external cache injection smoke test (D-010).

Verifies that ``mlx_lm.generate.generate_step(prompt_cache=[...])`` routes a
caller-supplied cache through to the model forward pass and that each per-layer
entry's ``update_and_fetch(...)`` is invoked by mlx-lm during generation. If
both hold, Silica's P-1 ``SimpleKVCache`` can inject directly — no monkey-patch
— and risk R-6 does not trigger.

A toy one-layer model synthesizes fake K/V of the shape mlx-lm's ``KVCache``
expects, pushes them through ``cache[0].update_and_fetch(...)``, and returns
constant logits. The probe exercises the cache protocol, not generation
quality.

Run: ``python scripts/probe_mlx_lm_cache.py``
Exit codes: 0 on ACCEPTS, 1 on REJECTS.
"""

from __future__ import annotations

import sys
from typing import Any

import mlx.core as mx
import mlx_lm
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache


class _TracedKVCache(KVCache):
    """KVCache subclass that counts ``update_and_fetch`` invocations."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        self.calls += 1
        # mlx-lm stubs return Any; cache contract is (k, v) pair.
        return super().update_and_fetch(keys, values)  # type: ignore[no-any-return]


class _ToyModel:
    """Minimal duck-typed model: one layer, invokes cache, returns constant logits."""

    def __init__(
        self,
        *,
        vocab_size: int = 8,
        n_kv_heads: int = 1,
        head_dim: int = 4,
    ) -> None:
        self.vocab_size = vocab_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    def __call__(
        self, tokens: mx.array, cache: list[Any] | None = None
    ) -> mx.array:
        B, T = tokens.shape
        k = mx.zeros((B, self.n_kv_heads, T, self.head_dim), dtype=mx.float16)
        v = mx.zeros((B, self.n_kv_heads, T, self.head_dim), dtype=mx.float16)
        if cache is not None and cache[0] is not None:
            cache[0].update_and_fetch(k, v)
        return mx.zeros((B, T, self.vocab_size), dtype=mx.float16)


def main() -> int:
    print(f"Gate A probe — mlx-lm {mlx_lm.__version__}")
    print(
        "Claim: generate_step(prompt_cache=[...]) routes the external cache "
        "to the model forward pass."
    )

    traced = _TracedKVCache()
    model = _ToyModel()
    prompt = mx.array([1, 2, 3], dtype=mx.int32)

    tokens: list[int] = []
    # generate_step expects an nn.Module; _ToyModel is duck-typed — the probe
    # intentionally bypasses nn.Module to keep the fixture minimal.
    for i, (tok, _) in enumerate(
        generate_step(prompt, model, max_tokens=4, prompt_cache=[traced])  # type: ignore[arg-type]
    ):
        tokens.append(int(tok))
        if i >= 3:
            break

    print(f"  generated tokens: {tokens}")
    print(f"  TracedKVCache.update_and_fetch calls: {traced.calls}")
    print(f"  TracedKVCache.offset after generation: {traced.offset}")

    ok = traced.calls > 0 and traced.offset > 0
    if ok:
        print("RESULT: ACCEPTS — D-010 clean path. R-6 does not trigger.")
        return 0
    print("RESULT: REJECTS — external cache not honored. R-6 mitigation required.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
